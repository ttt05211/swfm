"""Frozen-weight inference candidates. Only <=t0 evidence enters adapters.

Neither adapter accepts future poses/semantics from a GT source. ``planned``
must be the externally predicted poses, not evaluation targets. These change
predictions and are NOT a lossless execution backend or a passed method.
"""
import math
import numpy as np
from scipy.spatial import cKDTree

from real_motion.geometry import _occupied_indices_to_xyz, warp_semantic_grid
from real_motion.source_evidence_audit import transform_points, associate_backwards

PROTOCOL = 'stc_frozen_causal_ground_geometry_screen_v1'
GROUND = (11, 12, 13, 14)
RULES = dict(ground_classes=list(GROUND), ground_votes=2, max_ground_shift_cells=1,
             plane_class=11, plane_radius_m=8., plane_neighbors=256,
             plane_min_unique_columns=32, plane_max_residual_m=.2,
             plane_min_inlier_fraction=.75, plane_max_slope=.15,
             max_pose_z_correction_m=.8, max_pose_tilt_correction_deg=3.,
             plane_max_points_per_history=3000)


def history_arrays(raw, grid):
    occ = np.asarray(raw['history_occ'])
    poses = np.asarray(raw['history_poses'], np.float64)
    if (occ.shape != (4, *grid.shape_hwd) or poses.shape != (4, 4, 4)
            or not np.isfinite(poses).all() or occ.dtype.kind not in 'ui'
            or occ.min() < 0 or occ.max() > 17
            or raw.get('future_gt_occ') is not None):
        raise ValueError('adapters require four finite history-only XYZ semantic grids')
    return occ, poses


def ground_columns(sem):
    """Only single-class contiguous ground slabs of 1..3 cells qualify."""
    mask = np.isin(sem, GROUND); count = mask.sum(2)
    low = np.argmax(mask, 2); high = sem.shape[2]-1-np.argmax(mask[:, :, ::-1], 2)
    label = np.take_along_axis(sem, low[:, :, None], 2)[:, :, 0]
    same = np.all(~mask | (sem == label[:, :, None]), 2)
    valid = (count > 0) & (count <= 3) & (high-low+1 == count) & same
    return label, low, high, valid


def stabilize_stc_ground(raw, grid):
    """Two other real histories must agree; never vote free or touch dynamics.

    Temporal neighbors may be later than a particular history slot, but every
    frame is one of the original four <=t0 observations. No artificial new
    occupancy or visibility is supplied to the network.
    """
    occ, poses = history_arrays(raw, grid); result = occ.copy()
    audit = dict(eligible_columns=0, supported_columns=0, shifted_columns=0,
                 relabeled_columns=0, protected_destination_columns=0, changed_voxels=0)
    for f in range(4):
        own_label, own_low, own_high, own_valid = ground_columns(occ[f])
        audit['eligible_columns'] += int(own_valid.sum())
        prior = [ground_columns(warp_semantic_grid(occ[j], np.linalg.inv(poses[f]) @ poses[j], grid))
                 for j in range(4) if j != f]
        labels = np.stack([p[0] for p in prior]); lows = np.stack([p[1] for p in prior])
        valids = np.stack([p[3] for p in prior])
        # Explicit deterministic majority among ground classes, never empty.
        votes = np.stack([np.sum(valids & (labels == c), 0) for c in GROUND])
        winner = np.argmax(votes, 0); consensus = np.asarray(GROUND, np.uint8)[winner]
        ids = valids & (labels == consensus[None])
        number = ids.sum(0); sentinel = occ.shape[3]+2
        ordered = np.sort(np.where(ids,lows,sentinel),axis=0)
        smallest = ordered[0]; largest = np.max(np.where(ids,lows,-sentinel),axis=0)
        z = np.where(number == 3,ordered[1],(smallest+largest)*.5)
        floor,ceil = np.floor(z).astype(np.int64),np.ceil(z).astype(np.int64)
        desired = np.where(np.abs(floor-own_low) <= np.abs(ceil-own_low),floor,ceil)
        shift = desired-own_low; low=desired; high=own_high+shift
        supported = (own_valid & (number >= 2) & (largest-smallest <= 1)
                     & (np.abs(shift) <= 1) & (low >= 0) & (high < occ.shape[3]))
        x,y = np.indices(own_low.shape); collision=np.zeros_like(supported)
        for offset in range(3):
            dest=occ[f,x,y,np.clip(low+offset,0,occ.shape[3]-1)]
            collision |= (low+offset <= high) & (dest != 17) & ~np.isin(dest,GROUND)
        audit['protected_destination_columns'] += int((supported & collision).sum())
        supported &= ~collision
        audit['supported_columns'] += int(supported.sum())
        audit['shifted_columns'] += int((supported & (shift != 0)).sum())
        audit['relabeled_columns'] += int((supported & (consensus != own_label)).sum())
        result[f][supported[:,:,None] & np.isin(occ[f],GROUND)] = 17
        for offset in range(3):
            use=supported & (low+offset <= high)
            result[f,x[use],y[use],low[use]+offset] = consensus[use]
    audit['changed_voxels'] = int(np.count_nonzero(result != occ))
    protected = ~np.isin(occ, (*GROUND, 17))
    if not np.array_equal(result[protected], occ[protected]):
        raise RuntimeError('causal ground adapter modified protected occupancy')
    adapted = {k:v for k,v in raw.items() if not k.startswith('_')}
    adapted['history_occ'] = result
    return adapted, audit


def ground_points(raw, grid):
    occ, poses = history_arrays(raw, grid); points = []
    for sem, pose in zip(occ, poses):
        idx = np.argwhere(sem == 11)
        if len(idx) > RULES['plane_max_points_per_history']:
            idx = idx[np.linspace(0, len(idx)-1, RULES['plane_max_points_per_history'], dtype=int)]
        points.append(transform_points(_occupied_indices_to_xyz(idx, grid), pose))
    p = np.concatenate(points)
    if not len(p): return p
    # Duplicate history observations cannot masquerade as spatial support.
    _, ids = np.unique(np.floor(p[:, :2]/.2).astype(np.int64), axis=0, return_index=True)
    return p[np.sort(ids)]


def fit_ground(points, tree, xy):
    if tree is None or len(points) < RULES['plane_min_unique_columns']:
        return None
    dist, ids = tree.query(xy, k=min(len(points), RULES['plane_neighbors']), workers=1)
    ids = np.asarray(ids)[np.asarray(dist) <= RULES['plane_radius_m']]
    if len(ids) < RULES['plane_min_unique_columns']: return None
    p = points[ids]; q = p[:, :2]-np.asarray(xy)
    # A one-sided road strip is not enough to extrapolate a road surface here.
    if (np.min(q, 0) > 0).any() or (np.max(q, 0) < 0).any(): return None
    a = np.column_stack((q, np.ones(len(q)))); keep = np.ones(len(q), bool)
    for _ in range(3):
        if keep.sum() < RULES['plane_min_unique_columns']: return None
        if np.linalg.eigvalsh(np.cov(q[keep].T)).min() < .25: return None
        coef = np.linalg.lstsq(a[keep], p[keep, 2], rcond=None)[0]
        keep = np.abs(a @ coef-p[:, 2]) <= RULES['plane_max_residual_m']
    if keep.sum() < RULES['plane_min_unique_columns'] or keep.mean() < RULES['plane_min_inlier_fraction']:
        return None
    if np.linalg.eigvalsh(np.cov(q[keep].T)).min() < .25:
        return None
    # Axis-wise bracketing alone does not rule out diagonal extrapolation.
    # The query must lie in the convex hull of the retained local evidence.
    angles = np.sort(np.arctan2(q[keep,1], q[keep,0]))
    if np.diff(np.r_[angles, angles[0]+2*math.pi]).max() > math.pi+1e-12:
        return None
    coef = np.linalg.lstsq(a[keep], p[keep, 2], rcond=None)[0]
    if (np.mean(np.abs(a@coef-p[:,2]) <= RULES['plane_max_residual_m'])
            < RULES['plane_min_inlier_fraction'] or np.linalg.norm(coef[:2]) > RULES['plane_max_slope']):
        return None
    n = np.array([-coef[0], -coef[1], 1.]); n /= np.linalg.norm(n)
    return float(coef[2]), n


def road_rotation(normal, yaw):
    x = np.array([math.cos(yaw), math.sin(yaw), 0.])
    x[2] = -np.dot(normal[:2], x[:2])/normal[2]; x /= np.linalg.norm(x)
    return np.column_stack((x, np.cross(normal, x), normal))


def compensate_planned_ground_pose(raw, planned, grid):
    """Preserve planner world XY/yaw; only causal ground height/tilt correction."""
    _, history_poses = history_arrays(raw, grid)
    planned = np.asarray(planned, np.float64)
    if planned.shape != (6, 4, 4) or not np.isfinite(planned).all():
        raise ValueError('six finite externally predicted poses required')
    points = ground_points(raw, grid); tree = cKDTree(points[:, :2]) if len(points) else None
    current = history_poses[-1]; base = fit_ground(points, tree, current[:2, 3])
    output = planned.copy(); audit = dict(supported=0, corrected=0, unsupported=0, rejected_bound=0,
                                        z_correction_abs_sum_m=0., tilt_correction_sum_deg=0.)
    if base is None:
        audit['unsupported'] = 6
        return list(output), audit
    yaw0 = math.atan2(current[1, 0], current[0, 0])
    calibration = road_rotation(base[1], yaw0).T @ current[:3, :3]
    height = current[2, 3]-base[0]
    for h, pose in enumerate(planned):
        local = fit_ground(points, tree, pose[:2, 3])
        if local is None:
            audit['unsupported'] += 1; continue
        audit['supported'] += 1
        yaw = math.atan2(pose[1, 0], pose[0, 0])
        rotation = road_rotation(local[1], yaw) @ calibration
        # Remove tiny heading changes introduced by body/road calibration.
        own_yaw = math.atan2(rotation[1, 0], rotation[0, 0]); d = yaw-own_yaw
        rz = np.array([[math.cos(d), -math.sin(d), 0.], [math.sin(d), math.cos(d), 0.], [0., 0., 1.]])
        rotation = rz @ rotation
        dz = local[0]+height-pose[2, 3]
        tilt = math.degrees(math.acos(float(np.clip(np.dot(rotation[:, 2], pose[:3, 2]), -1., 1.))))
        if abs(dz) > RULES['max_pose_z_correction_m'] or tilt > RULES['max_pose_tilt_correction_deg']:
            audit['rejected_bound'] += 1; continue
        if abs(dz) < 1e-10 and tilt < 1e-6: continue
        output[h, :3, :3] = rotation; output[h, 2, 3] += dz
        audit['corrected'] += 1; audit['z_correction_abs_sum_m'] += abs(float(dz))
        audit['tilt_correction_sum_deg'] += tilt
    return list(output), audit


def motion_jitter_audit(prep):
    """Read-only causal association diagnostic, not a velocity override."""
    frames = prep.raw.get('_waymo_frame_geometry')
    if frames is None: return dict(sources=len(prep.state['current']), available=False)
    state = prep.state
    links, audit = associate_backwards([list(f.components) for f in frames],
        state['current'], state['velocities'], dt=.5)
    speeds = []; jitter = []; stable = 0
    for i, indices in enumerate(links):
        observed = [(f, frames[f].components[j]['centroid_world']) for f,j in enumerate(indices) if j is not None]
        v = state['velocities'].get(i)
        if v is not None: speeds.append(float(np.linalg.norm(v[:2])))
        if len(observed) < 3: continue
        t = np.asarray([f*.5 for f,_ in observed]); xy = np.asarray([p[:2] for _,p in observed])
        a = np.column_stack((t-t[-1], np.ones(len(t))))
        fit = np.linalg.lstsq(a, xy, rcond=None)[0]
        jitter.append(float(np.sqrt(np.mean(np.sum((a@fit-xy)**2, 1)))))
        stable += 1
    return dict(sources=len(links), matched_t0_sources=len(speeds), tracks_with_3plus_frames=stable,
                speed_p90_mps=float(np.percentile(speeds,90)) if speeds else None,
                centroid_fit_rmse_p90_m=float(np.percentile(jitter,90)) if jitter else None,
                association=audit, available=True, velocities_modified=False)
