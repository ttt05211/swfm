"""One causal identity table; registration validity is separate from identity validity."""
from dataclasses import dataclass
import numpy as np
from real_motion.source_evidence_audit import associate_backwards, register_history_shape, transform_points
from tools.real_motion.causal_column_common import pose_motion
from tools.real_motion import benchmark_p0_f9_v18_runtime as runtime

ARMS = ('baseline', 'unified_two', 'unified_fit')


@dataclass
class SourceTracks:
    links: np.ndarray  # [source,4], frame-local component indices; -1 means absent
    centers: np.ndarray  # ORIGINAL world centroids, never ICP-aligned centroids
    valid: np.ndarray
    times: np.ndarray  # actual seconds relative to t0
    registrations: list
    audit: dict


def checked_times(times):
    t = np.asarray(times, np.float64)
    if t.shape != (4,) or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError('four finite strictly increasing historical timestamps required')
    return t-t[-1]


def track_velocity(track, mode):
    if mode not in ('two', 'fit'): raise ValueError('two or fit required')
    t = checked_times(track.times)
    if track.centers.shape != (*track.valid.shape, 3) or track.valid.shape != track.links.shape:
        raise ValueError('track shape differs')
    velocity = np.zeros((len(track.links), 3), np.float64)
    for i in range(len(track.links)):
        ids = np.flatnonzero(track.valid[i])
        if not len(ids) or ids[-1] != 3: raise ValueError('each source must be anchored at t0')
        if len(ids) < 2: continue
        if mode == 'two': ids = ids[-2:]
        xy = track.centers[i, ids, :2]
        if not np.isfinite(xy).all(): raise ValueError('nonfinite valid centroid')
        dt = t[ids]-t[ids].mean()
        # Only estimate the slope. The forecast still starts at observed p0.
        velocity[i, :2] = (dt[:, None]*(xy-xy.mean(0))).sum(0)/(dt@dt)
    return velocity


def tracks_from_frames(frames, times, seed_velocity, grid, poses, strong):
    """Keep existing backward association protections and nominal matching clock.

    The old two-frame velocity is only the initialization of that existing
    association algorithm; all downstream identities/velocities use its ONE table.
    No new shape cost, grouped classes, stationary threshold or gap rule is added.
    """
    times = checked_times(times); poses = np.asarray(poses, np.float64)
    if len(frames) != 4 or poses.shape != (4, 4, 4): raise ValueError('four frames/poses required')
    current = frames[-1]
    linked, audit = associate_backwards(frames, current, seed_velocity,
        dt=.5, speed_gate=float(strong.max_match_speed_mps))
    links = np.asarray([[(-1 if j is None else j) for j in row] for row in linked], np.int64).reshape(-1, 4)
    centers = np.zeros((len(current), 4, 3), np.float64); valid = links >= 0
    registrations = [[None]*4 for _ in current]
    origin = np.array([grid.x_min, grid.y_min, grid.z_min]); step = np.asarray(grid.voxel_size)
    for i, source in enumerate(current):
        ref = transform_points(origin+(np.asarray(source['voxel_indices'])+.5)*step, poses[-1])
        for f in np.flatnonzero(valid[i]):
            component = frames[f][links[i, f]]
            centers[i, f] = np.asarray(component['centroid_world'], np.float64)
            ijk = np.asarray(component['voxel_indices'], np.int64)
            if f == 3:
                registrations[i][f] = (np.eye(4), ijk)
                continue
            points = transform_points(origin+(ijk+.5)*step, poses[f])
            fitted = register_history_shape(points, ref, allow_yaw=int(source['class_id']) != 7)
            key = 'registration_accepted' if fitted.accepted else 'registration_rejected'
            audit[key] = audit.get(key, 0)+1
            if fitted.accepted:
                matrix = pose_motion(np.zeros(3), np.zeros(3), fitted.yaw_rad)
                matrix[:2, 3] = fitted.points[:, :2].mean(0)-points[:, :2].mean(0)@matrix[:2, :2].T
                registrations[i][f] = (matrix, ijk)
    return SourceTracks(links, centers, valid, times, registrations, audit)


def build_tracks(row, provider, times):
    raw = row[1]; state = raw['_column_causal_preparation']['prepared_state']
    poses = raw['history_poses']
    # Only pure component extraction is reused from THIS baseline window, not
    # cached links/registrations/KTA. Rebuild the older frames from raw histories.
    frames = [runtime.extract_instances_cropped_exact(raw['history_occ'][f], poses[f],
              grid=provider.pcfg.grid, cfg=provider.strong) for f in range(2)]
    frames += [state['previous'], state['current']]
    return tracks_from_frames(frames, times, state['velocities'], provider.pcfg.grid, poses, provider.strong)


def make_row(row, track, velocity, history_builder):
    record, original, ego, _ = row
    if velocity.shape != (len(track.links), 3): raise ValueError('velocity/source shape differs')
    raw = {k: original[k] for k in ('history_occ', 'history_observed', 'history_poses')}
    fixed = original['_column_causal_preparation']; state = dict(fixed['prepared_state'])
    state['velocities'] = {i: v.copy() for i, v in enumerate(velocity)}
    raw['_column_causal_preparation'] = dict(prepared_state=state,
        registrations=track.registrations, audit=track.audit)
    native_velocity = velocity @ np.asarray(state['current_pose'])[:3, :3]
    kta = (native_velocity[:, None, :2]*(np.arange(1, 7)[None, :, None]*.5)).astype(np.float32)
    record = dict(record, kta_displacement_xy_m=kta,
        anchors_xy_t0_m=(np.asarray(record['source_centroid_xy_t0_m'])[:, None]+kta).astype(np.float32))
    return record, raw, ego, history_builder.build(raw)


def motion_errors(ktas, track, displacement, valid):
    """Diagnostic GT is supplied AFTER forecasts; same label mask for all arms."""
    target = np.asarray(displacement, np.float64); mask = np.asarray(valid, bool)
    if target.shape != (len(track.links), 6, 2) or mask.shape != target.shape[:2]:
        raise ValueError('source displacement label shape differs')
    if not np.isfinite(target[mask]).all(): raise ValueError('nonfinite motion label')
    npoints = track.valid.sum(1); result = {}
    for arm, kta in ktas.items():
        error = np.linalg.norm(np.asarray(kta, np.float64)-target, axis=-1)
        result[arm] = {group: [error[:, h][mask[:, h] & chosen].tolist() for h in (1, 3, 5)]
            for group, chosen in (('all', np.ones(len(npoints), bool)), ('two_points', npoints == 2),
                                  ('three_plus', npoints >= 3), ('one_point', npoints == 1))}
    return result
