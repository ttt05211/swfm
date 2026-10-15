"""One pair match and older-only extension. No multiframe velocity fitting."""
import numpy as np
import torch
from real_motion.source_evidence_audit import transform_points, register_history_shape
from real_motion.ego_navigation import ego_history_features
from real_motion.motion_transport import world_points_to_t0
from real_motion.runtime_fastpath import compose_component_replacements_fast_exact
from real_motion.strong_majority_execution import strong_majority_execution
from real_motion.metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS
from tools.real_motion import benchmark_p0_f9_v18_runtime as runtime
from tools.real_motion.causal_column_common import pose_motion
from tools.shared_history_experiments.refine.history import ResidualHistoryInputs
from tools.shared_history_experiments.model import encode_window
from tools.scene_ego_head.integration import forecast_conditioned
from tools.unified_source_tracks.geometry import SourceTracks, checked_times


def match_pair(previous, current, max_speed=25.):
    """Exact existing Strong mutual-nearest math, ALSO returns its chosen edges.

    Nominal dt=.5 and original XYZ matching distance preserved; Z velocity zero.
    No second nearest-neighbor pass is used to reconstruct the selected IDs.
    """
    links = np.full(len(current), -1, np.int64); velocity = {}
    by_class = {}
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
            if int(nearest_cur[j]) != local or float(distance[local, j]) > max_speed*.5: continue
            v = np.asarray((c[local]-p[j])/.5, np.float64); v[2] = 0.
            links[actor] = prev_ids[j]; velocity[actor] = v
    return links, velocity


def build_track(frames, poses, times, grid, strong):
    if len(frames) != 4: raise ValueError('exact four historical component frames required')
    poses = np.asarray(poses); times = checked_times(times); current = frames[-1]
    pair, velocities = match_pair(frames[-2], current, strong.max_match_speed_mps)
    links = np.full((len(current), 4), -1, np.int64); links[:, 3] = np.arange(len(current)); links[:, 2] = pair
    centers = np.zeros((len(current), 4, 3), np.float64)
    last = np.full(len(current), 3); recent = []; speed = []
    for i, source in enumerate(current):
        centers[i, 3] = source['centroid_world']
        if pair[i] >= 0:
            centers[i, 2] = frames[2][pair[i]]['centroid_world']; last[i] = 2
        recent.append(centers[i, last[i]].copy()); speed.append(velocities.get(i, np.zeros(3)).copy())
    audit = dict(pinned_pairs=int((pair >= 0).sum()), older_matched=0, older_ambiguous=0, older_unmatched=0)
    # Crucially: frame2 is NOT visited again. Only extend into frames1 and0.
    for frame in (1, 0):
        pairs = []
        for i, source in enumerate(current):
            elapsed = (last[i]-frame)*.5; predicted = recent[i]-speed[i]*elapsed; choices = []
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
            speed[i] = (recent[i]-p)/((last[i]-frame)*.5)
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
    return SourceTracks(links, centers, links >= 0, times, registrations, audit), velocities


def from_raw(raw, provider, current=None):
    poses = raw['history_poses']
    frames = [runtime.extract_instances_cropped_exact(raw['history_occ'][f], poses[f],
        grid=provider.pcfg.grid, cfg=provider.strong) for f in range(3)]
    frames.append(current if current is not None else runtime.extract_instances_cropped_exact(
        raw['history_occ'][-1], poses[-1], grid=provider.pcfg.grid, cfg=provider.strong))
    track, velocities = build_track(frames, poses, raw['_track_history_seconds'], provider.pcfg.grid, provider.strong)
    return frames, track, velocities


def kta_from_velocity(current, velocities, pose):
    v = np.asarray([velocities.get(i, np.zeros(3)) for i in range(len(current))]).reshape(-1, 3) @ np.asarray(pose)[:3, :3]
    return (v[:, None, :2]*(np.arange(1, 7)[None, :, None]*.5)).astype(np.float32)


class PinnedHistory(ResidualHistoryInputs):
    def build(self, raw):
        fixed = raw['_column_causal_preparation']; current = fixed['prepared_state']['current']
        _, track, velocity = from_raw(raw, self.provider, current)
        changed = dict(raw, _column_causal_preparation=dict(fixed, registrations=track.registrations))
        history = super().build(changed)
        history[0]['pinned_pair'] = dict(velocity=velocity, audit=track.audit,
            expected_kta=kta_from_velocity(current, velocity, raw['history_poses'][-1]))
        return history


def prefetch_adapter(base, arm):
    if arm != 'unified_pair': raise ValueError('only unified_pair new training')
    def batches(*args, **kwargs):
        iterator = base(*args, **kwargs)
        try:
            for rows in iterator:
                published = []
                for record, raw, ego, history in rows:
                    # Original cache may have the documented FP32 anchor-upgrade
                    # rounding. Preserve ORIGINAL bytes, never replace by recompute.
                    expected = history[0]['pinned_pair']['expected_kta']
                    if not np.allclose(np.asarray(record['kta_displacement_xy_m']), expected, atol=2e-4, rtol=0):
                        raise RuntimeError('original KTA is not the nominal last-pair rule')
                    fixed = raw['_column_causal_preparation']
                    changed = dict(raw, _column_causal_preparation=dict(fixed, registrations=history[0]['registrations']))
                    published.append((record, changed, ego, history))  # record/anchors/KTA EXACT unchanged
                yield published
        finally:
            close = getattr(iterator, 'close', None)
            if close: close()
    return batches


class PinnedPredictor:
    """Override preparation only: no old four-frame association before new track."""
    def __init__(self, body, pcfg, device, workers):
        from tools.final_semantic_evaluation.predictor import Predictor
        self.base = Predictor(body, pcfg, device, workers); self.provider = self.base.provider
        self.history = self.base.history
    def close(self): self.base.close()
    def dense(self, fw): return self.base.dense(fw)
    def prepare(self, record, raw, *, times):
        if set(raw) != {'history_occ', 'history_observed', 'history_poses'} or len(raw['history_occ']) != 4:
            raise ValueError('four historical fields ONLY')
        if str(record['history_tokens'][-1]) != str(record['t0_token']): raise ValueError('t0 history identity differs')
        raw = dict(raw, _track_history_seconds=np.asarray(times)); p = self.provider; pose = np.asarray(raw['history_poses'][-1])
        frames, track, velocity = from_raw(raw, p); current = frames[-1]
        points = runtime._precompute_source_world(current, pose, p.pcfg.grid)
        xyz = np.asarray([c['centroid_world'] for c in current]).reshape(-1, 3)
        native = np.asarray([world_points_to_t0(c[None], pose)[0] for c in xyz]).reshape(-1, 3)
        # Preserve the original predictor's per-source/per-horizon FP64->FP32
        # evaluation encoding, including anchor rounding, not center32+KTA32.
        kta = np.zeros((len(current), 6, 2), np.float32); anchors = np.zeros_like(kta)
        for i in range(len(current)):
            v = np.asarray(velocity.get(i, np.zeros(3))) @ pose[:3, :3]
            for h in range(6):
                delta = v[:2]*((h+1)*.5); kta[i, h] = delta; anchors[i, h] = native[i, :2]+delta
        rec = {**{k: record[k] for k in ('scene_name', 't0_token', 'future_tokens')},
            'history_tokens': tuple(record['history_tokens'][-4:]),
            'source_class_id': np.asarray([c['class_id'] for c in current], np.int64),
            'source_centroid_xy_t0_m': native[:, :2].astype(np.float32), 'kta_displacement_xy_m': kta,
            'anchors_xy_t0_m': anchors}
        state = dict(current=current, current_pose=pose, source_world_points=points,
            source_rel_xy=[v[:, :2]-xyz[i, None, :2] for i, v in enumerate(points)], source_z_t0=native[:, 2])
        raw['_column_causal_preparation'] = dict(prepared_state=state, registrations=track.registrations)
        raw['_pinned_velocity'] = velocity
        history = self.history.build(raw)
        return rec, raw, (ego_history_features(raw['history_poses'], times), None, None), history


def forward_pair(body, provider, row, poses):
    """Strong consumes the already selected pair velocity; never matches again."""
    from contextlib import nullcontext
    rec, raw, ego, history = row; device = provider.device
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        field, reads, times = encode_window(body, raw, history, ego[0], device)
        output = body.transport(field, reads, times, torch.as_tensor(rec['source_class_id'], device=device).long(),
            torch.as_tensor(rec['kta_displacement_xy_m'], device=device).float())
        lifted = body.columns.lift_history(field)
    def backgrounds(historical, state, predicted, p):
        majority = getattr(p, 'strong_majority', None)
        with strong_majority_execution(majority) if majority is not None else nullcontext():
            anchors, components = runtime._strong_all_horizons(historical['history_occ'][-1], state['current_pose'],
                predicted.detach().cpu().numpy(), state['current'], raw['_pinned_velocity'], state['source_world_points'],
                frame_dt_s=.5, grid=p.pcfg.grid, cfg=p.strong, runtime_device=device,
                majority_backend='native' if majority is not None else 'sparse_cuda')
        return [compose_component_replacements_fast_exact(a, c, [], dynamic_class_ids=DYNAMIC_CLASS_IDS,
            free_label=17, grid=p.pcfg.grid, precomputed_clear_flat_indices=runtime.baseline_clear_flat_indices(c, grid=p.pcfg.grid))
            for a, c in zip(anchors, components)]
    return forecast_conditioned(body, provider, row, dict(field=field, output=output, lifted=lifted), poses,
        background_builder=backgrounds)
