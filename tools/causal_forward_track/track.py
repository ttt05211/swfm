"""Four observed frames only; never synthesize a missing component or KTA pair."""
import numpy as np
from tools.source_scene_stage2.track import SourceIdentities

PROTOCOL = 'causal_forward_CV_gated_greedy_unknown_MNN_gap4_last_pair_KTA_v1'


def greedy(cost):
    """Global minimum, deterministic row/column tie order; inf means forbidden."""
    cost = np.array(cost, np.float64, copy=True)
    pairs = []
    while cost.size and np.isfinite(cost).any():
        i, j = np.unravel_index(np.argmin(cost), cost.shape)
        pairs.append((int(i), int(j)))
        cost[i, :] = np.inf
        cost[:, j] = np.inf
    return pairs


def associate(tracks, components, frame, times, *, has_velocity, speed_gate, ambiguity_margin):
    if not tracks or not components:
        return [], 0
    points = np.asarray([c['centroid_world'] for c in components], np.float64)
    old = np.asarray([t['centers'][t['last']] for t in tracks])
    elapsed = np.asarray([times[frame]-times[t['last']] for t in tracks])
    predicted = old.copy()
    if has_velocity:
        predicted += np.asarray([t['velocity'] for t in tracks])*elapsed[:, None]
    distance = np.linalg.norm(predicted[:, None, :2]-points[None, :, :2], axis=-1)
    physical = np.linalg.norm(old[:, None, :2]-points[None, :, :2], axis=-1)
    same = np.asarray([t['class_id'] for t in tracks])[:, None] == np.asarray([c['class_id'] for c in components])[None]
    # Deliberately reuse the existing speed envelope, NOT CenterPoint's detector
    # thresholds or a guessed multiplier. No lateral-motion prohibition.
    cost = np.where(same & (physical <= speed_gate*elapsed[:, None]) &
                    (distance <= speed_gate*elapsed[:, None]), distance, np.inf)
    # Same .4m ambiguity gap already used by the old older-history extension.
    # Reject BEFORE assignment; do not consume a component for an ambiguous row.
    ambiguous = np.zeros(len(tracks), bool)
    if len(components) > 1:
        ordered = np.sort(cost, axis=1)
        both = np.isfinite(ordered[:, 1])
        ambiguous[both] = ordered[both, 1]-ordered[both, 0] < ambiguity_margin
    cost[ambiguous] = np.inf
    if not has_velocity:
        original = cost.copy()
        for i, j in np.argwhere(np.isfinite(original)):
            if j != original[i].argmin() or i != original[:, j].argmin():
                cost[i, j] = np.inf
    return greedy(cost), int(ambiguous.sum())


def build(frames, poses, times, strong, *, dt_s=.5, ambiguity_margin=.4):
    times, poses = np.asarray(times, np.float64), np.asarray(poses, np.float64)
    if (len(frames) != 4 or times.shape != (4,) or poses.shape != (4, 4, 4) or dt_s != .5 or
            not np.isfinite(times).all() or not np.isfinite(poses).all() or np.any(np.diff(times) <= 0)):
        raise ValueError('four finite increasing historical times/poses; original nominal .5s KTA required')
    if not np.isfinite(strong.max_match_speed_mps) or strong.max_match_speed_mps <= 0 or ambiguity_margin < 0:
        raise ValueError('positive speed envelope and nonnegative ambiguity gap required')
    for components in frames:
        for c in components:
            point = np.asarray(c['centroid_world'])
            if point.shape != (3,) or not np.isfinite(point).all():
                raise ValueError('finite world-frame component centroid required')
    times = times-times[-1]
    tracks, at_t0 = [], {}
    audit = dict(protocol=PROTOCOL, matched_with_velocity=0, matched_without_velocity=0,
        gap_reconnections=0, ambiguous_rows=0, registration_computed=False,
        association_time='actual_history_timestamps', KTA_time='original_nominal_last_pair_0.5s',
        speed_gate_mps=float(strong.max_match_speed_mps), ambiguity_gap_m=float(ambiguity_margin),
        detector_class_gates_copied=False, geometry_distance='world_XY', shape_cost=False)

    def observe(track, index, frame):
        point = np.asarray(frames[frame][index]['centroid_world'], np.float64)
        last = track['last']
        if last >= 0:
            track['velocity'] = (point-track['centers'][last])/(times[frame]-times[last])
            track['velocity'][2] = 0.
            track['has_velocity'] = True
            if frame-last > 1: audit['gap_reconnections'] += 1
        track['links'][frame] = index
        track['centers'][frame] = point
        track['last'] = frame
        if frame == 3: at_t0[index] = track

    for frame, components in enumerate(frames):
        available = list(range(len(components)))
        # Known-velocity first; within each group continuous before lost. Lost
        # tracks remain eligible throughout this four-frame window, not forever.
        for known in (True, False):
            for continuous in (True, False):
                candidates = [t for t in tracks if t['last'] < frame and t['has_velocity'] == known and
                              (t['last'] == frame-1) == continuous]
                pairs, ambiguous = associate(candidates, [components[j] for j in available], frame, times,
                    has_velocity=known, speed_gate=strong.max_match_speed_mps, ambiguity_margin=ambiguity_margin)
                audit['ambiguous_rows'] += ambiguous
                used = set()
                for i, j in pairs:
                    index = available[j]
                    observe(candidates[i], index, frame)
                    used.add(index)
                    audit['matched_with_velocity' if known else 'matched_without_velocity'] += 1
                available = [j for j in available if j not in used]
        for index in available:
            track = dict(class_id=int(components[index]['class_id']), links=np.full(4, -1, np.int64),
                centers=np.zeros((4, 3), np.float64), last=-1, has_velocity=False, velocity=np.zeros(3))
            observe(track, index, frame)
            tracks.append(track)

    ordered = [at_t0[i] for i in range(len(frames[-1]))]
    links = np.asarray([t['links'] for t in ordered], np.int64).reshape(-1, 4)
    centers = np.asarray([t['centers'] for t in ordered], np.float64).reshape(-1, 4, 3)
    velocities = {}
    for i in range(len(ordered)):
        if links[i, 2] >= 0:
            v = (centers[i, 3]-centers[i, 2])/dt_s
            v[2] = 0.
            velocities[i] = v
    voxels = [[None if j < 0 else np.asarray(frames[f][j]['voxel_indices'], np.int64)
               for f, j in enumerate(row)] for row in links]
    audit.update(t0_sources=len(ordered), reliable_last_pairs=len(velocities),
        historical_only_tracks=sum(t['last'] < 3 for t in tracks))
    return SourceIdentities(links, centers, links >= 0, times, velocities, voxels, audit)
