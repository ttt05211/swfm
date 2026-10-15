"""Pinned original last pair, older-only shared identities (ab8339b geometry).

No multiframe KTA fitting. Identity validity, registration validity and observed
coverage are distinct. This module is opt-in; old preparation is unchanged.
"""
from dataclasses import dataclass
import numpy as np
from real_motion.source_evidence_audit import transform_points, register_history_shape
from tools.real_motion.causal_column_common import pose_motion

PROTOCOL = 'shared_source_pinned_baseline_pair_older_extension_v1'


@dataclass
class SharedSourceTrack:
    links: np.ndarray
    centers: np.ndarray
    track_valid: np.ndarray
    times: np.ndarray
    pair_velocity: dict
    registrations: list
    registration_valid: np.ndarray
    audit: dict


def match_pair(previous, current, dt_s=.5, *, max_speed_mps=25.):
    """Exact Strong matching arithmetic/ties/order, returning selected edges too."""
    dt = max(float(dt_s), 1e-3)
    links = np.full(len(current), -1, np.int64); velocities = {}; by_class = {}
    for j, row in enumerate(current): by_class.setdefault(int(row['class_id']), []).append(j)
    for cls, cur_ids in by_class.items():
        prev_ids = [i for i, row in enumerate(previous) if int(row['class_id']) == cls]
        if not prev_ids: continue
        p = np.stack([np.asarray(previous[i]['centroid_world'], np.float64) for i in prev_ids])
        c = np.stack([np.asarray(current[j]['centroid_world'], np.float64) for j in cur_ids])
        distance = np.linalg.norm(c[:, None, :]-p[None, :, :], axis=-1)
        nearest_prev, nearest_cur = distance.argmin(1), distance.argmin(0)
        for local, actor in enumerate(cur_ids):
            j = int(nearest_prev[local])
            if int(nearest_cur[j]) != local or float(distance[local, j]) > float(max_speed_mps)*dt: continue
            v = np.asarray((c[local]-p[j])/dt, np.float64); v[2] = 0.
            links[actor] = prev_ids[j]; velocities[actor] = v
    return links, velocities


def build_track(frames, poses, times, grid, strong, *, dt_s=.5):
    """Earlier identities never reselect t-1, nor create a new KTA velocity."""
    poses, times = np.asarray(poses), np.asarray(times, np.float64)
    if (len(frames) != 4 or poses.shape != (4, 4, 4) or times.shape != (4,)
            or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)):
        raise ValueError('four finite increasing history times and poses required')
    if dt_s != .5: raise ValueError('this pinned baseline protocol retains nominal dt=.5')
    times = times-times[-1]; current = frames[-1]
    pair, velocity = match_pair(frames[-2], current, dt_s, max_speed_mps=strong.max_match_speed_mps)
    links = np.full((len(current), 4), -1, np.int64)
    links[:, 3] = np.arange(len(current)); links[:, 2] = pair
    centers = np.zeros((len(current), 4, 3), np.float64)
    last = np.full(len(current), 3); recent, speed = [], []
    for i, source in enumerate(current):
        centers[i, 3] = source['centroid_world']
        if pair[i] >= 0:
            centers[i, 2] = frames[2][pair[i]]['centroid_world']; last[i] = 2
        recent.append(centers[i, last[i]].copy()); speed.append(velocity.get(i, np.zeros(3)).copy())
    audit = dict(protocol=PROTOCOL, pinned_pairs=int((pair >= 0).sum()), older_matched=0,
                 older_ambiguous=0, older_unmatched=0)
    for frame in (1, 0):
        pairs = []
        for i, source in enumerate(current):
            elapsed = (last[i]-frame)*dt_s; predicted = recent[i]-speed[i]*elapsed; choices = []
            for j, component in enumerate(frames[frame]):
                if int(component['class_id']) != int(source['class_id']): continue
                p = np.asarray(component['centroid_world'])
                distance = float(np.linalg.norm(p[:2]-predicted[:2]))
                physical = float(np.linalg.norm(p[:2]-recent[i][:2]))
                if max(distance, physical) <= strong.max_match_speed_mps*elapsed: choices.append((distance, j))
            choices.sort()
            if not choices: audit['older_unmatched'] += 1
            elif len(choices) > 1 and choices[1][0]-choices[0][0] < .4: audit['older_ambiguous'] += 1
            else: pairs.append((choices[0][0], i, choices[0][1]))
        used = set()
        for _, i, j in sorted(pairs):
            if j in used: audit['older_unmatched'] += 1; continue
            used.add(j); p = np.asarray(frames[frame][j]['centroid_world'], np.float64)
            speed[i] = (recent[i]-p)/((last[i]-frame)*dt_s)
            recent[i] = p.copy(); last[i] = frame; links[i, frame] = j; centers[i, frame] = p
            audit['older_matched'] += 1
    origin = np.array([grid.x_min, grid.y_min, grid.z_min]); step = np.asarray(grid.voxel_size)
    registrations = [[None]*4 for _ in current]
    for i, source in enumerate(current):
        ref = transform_points(origin+(np.asarray(source['voxel_indices'])+.5)*step, poses[-1])
        for f in np.flatnonzero(links[i] >= 0):
            ijk = np.asarray(frames[f][links[i, f]]['voxel_indices'], np.int64)
            if f == 3: registrations[i][f] = (np.eye(4), ijk); continue
            points = transform_points(origin+(ijk+.5)*step, poses[f])
            fitted = register_history_shape(points, ref, allow_yaw=int(source['class_id']) != 7)
            key = 'registration_accepted' if fitted.accepted else 'registration_rejected'
            audit[key] = audit.get(key, 0)+1
            if fitted.accepted:
                m = pose_motion(np.zeros(3), np.zeros(3), fitted.yaw_rad)
                m[:2, 3] = fitted.points[:, :2].mean(0)-points[:, :2].mean(0)@m[:2, :2].T
                registrations[i][f] = (m, ijk)
    reg_valid = np.array([[r is not None for r in row] for row in registrations], bool).reshape(-1, 4)
    return SharedSourceTrack(links, centers, links >= 0, times, velocity, registrations, reg_valid, audit)


def from_raw(raw, provider, times, *, current=None):
    from tools.real_motion import benchmark_p0_f9_v18_runtime as runtime
    frames = [runtime.extract_instances_cropped_exact(raw['history_occ'][f], raw['history_poses'][f],
        grid=provider.pcfg.grid, cfg=provider.strong) for f in range(3)]
    frames.append(current if current is not None else runtime.extract_instances_cropped_exact(
        raw['history_occ'][-1], raw['history_poses'][-1], grid=provider.pcfg.grid, cfg=provider.strong))
    return frames, build_track(frames, raw['history_poses'], times, provider.pcfg.grid, provider.strong,
                               dt_s=float(provider.pcfg.frame_dt_s))


def kta_and_anchors(current, track, pose, *, dt_s=.5):
    """Preserve original per-source/per-horizon FP64 -> FP32 rounding order."""
    from real_motion.motion_transport import world_points_to_t0
    native = np.asarray([world_points_to_t0(np.asarray(c['centroid_world'])[None], pose)[0]
                         for c in current]).reshape(-1, 3)
    kta = np.zeros((len(current), 6, 2), np.float32); anchors = np.zeros_like(kta)
    for i in range(len(current)):
        velocity = np.asarray(track.pair_velocity.get(i, np.zeros(3))) @ np.asarray(pose)[:3, :3]
        for h in range(6):
            delta = velocity[:2]*((h+1)*dt_s)
            kta[i, h] = delta; anchors[i, h] = native[i, :2]+delta
    return native, kta, anchors


def check_cached_record(record, current, track, pose):
    native, kta, anchors = kta_and_anchors(current, track, pose)
    if not np.array_equal(np.asarray(record['source_class_id']), [c['class_id'] for c in current]):
        raise RuntimeError('cached source identities/order differ')
    for name, value in (('source_centroid_xy_t0_m', native[:, :2]),
                        ('kta_displacement_xy_m', kta), ('anchors_xy_t0_m', anchors)):
        if not np.allclose(np.asarray(record[name]), value, atol=2e-4, rtol=0):
            raise RuntimeError('cached original pair geometry differs: '+name)
    # Check only: never replace record fields by recomputation (FP32 upgrades).
