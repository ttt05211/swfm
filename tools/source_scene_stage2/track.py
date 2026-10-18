"""Pinned identities ONLY. Same association as stage1, no registration call."""
from dataclasses import dataclass
import numpy as np
from real_motion.shared_source_track import match_pair

PROTOCOL = 'shared_pinned_identity_only_no_registration_v1'


@dataclass
class SourceIdentities:
    links: np.ndarray
    centers: np.ndarray
    track_valid: np.ndarray
    times: np.ndarray
    pair_velocity: dict
    voxels: list
    audit: dict


def build(frames, poses, times, strong, *, dt_s=.5):
    poses, times = np.asarray(poses), np.asarray(times, np.float64)
    if (len(frames) != 4 or poses.shape != (4, 4, 4) or times.shape != (4,)
            or not np.isfinite(times).all() or np.any(np.diff(times) <= 0) or dt_s != .5):
        raise ValueError('four increasing finite times; nominal .5s pair convention required')
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
                 older_ambiguous=0, older_unmatched=0, registration_computed=False)
    # Deliberately identical to stage1 older-only identity extension. No new
    # association criterion, no refit of pair velocity, no registration gate.
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
    voxels = [[None if j < 0 else np.asarray(frames[f][j]['voxel_indices'], np.int64)
               for f, j in enumerate(row)] for row in links]
    return SourceIdentities(links, centers, links >= 0, times, velocity, voxels, audit)


def from_raw(raw, provider, times, *, current=None):
    from tools.real_motion import benchmark_p0_f9_v18_runtime as runtime
    frames = [runtime.extract_instances_cropped_exact(raw['history_occ'][f], raw['history_poses'][f],
        grid=provider.pcfg.grid, cfg=provider.strong) for f in range(3)]
    frames.append(current if current is not None else runtime.extract_instances_cropped_exact(
        raw['history_occ'][-1], raw['history_poses'][-1], grid=provider.pcfg.grid, cfg=provider.strong))
    return frames, build(frames, raw['history_poses'], times, provider.strong, dt_s=provider.pcfg.frame_dt_s)
