"""One history association, independent GT/Pred future geometry."""
from contextlib import nullcontext
from dataclasses import asdict
import numpy as np
import torch
from real_motion import shared_source_track as tracks
from real_motion.causal_geometry_cache import CausalGeometryCache
from real_motion.ego_navigation import ego_history_features, poses_from_se2
from real_motion.metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS
from real_motion.runtime_fastpath import compose_component_replacements_fast_exact
from real_motion.strong_majority_execution import strong_majority_execution
from tools.final_semantic_evaluation.predictor import Predictor
from tools.real_motion.run_p0_f9_shared_evidence_pilot import PilotProvider
from tools.real_motion import benchmark_p0_f9_v18_runtime as runtime
from tools.shared_history_experiments.model import encode_window
from tools.shared_history_experiments.refine.history import ResidualHistoryInputs
from tools.scene_ego_head.integration import forecast_conditioned
from tools.scene_ego_head.model import prepare_scene


class TrackProvider(PilotProvider):
    """Read old immutable components/GT backgrounds; NEVER rerun old association.

    New track/registrations live in a separate, versioned, causal cache. Loading
    old cached associations does not consume them or execute their builder.
    """
    def attach_tracks(self, root, *, rank, implementation):
        from real_motion.v21_source_induction import stable_json_fingerprint
        identity = stable_json_fingerprint(dict(protocol=tracks.PROTOCOL, grid=asdict(self.pcfg.grid),
            strong=asdict(self.strong), frame_dt_s=float(self.pcfg.frame_dt_s), implementation=implementation))
        # Separate writer roots: the existing cache implementation is process-local.
        self.track_cache = CausalGeometryCache(str(root)+f'/rank{rank}', identity,
            max_bytes=8*2**30, ram_bytes=128*2**20, reserve_bytes=2*2**30)
        from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
        self.track_writer_lease = evaluation_lock(self.track_cache.cache_root)
        self.track_writer_lease.__enter__()
        return dict(protocol=tracks.PROTOCOL, namespace=self.track_cache.namespace,
                    max_disk_gib_per_rank=8, storage_root=str(root), history_only=True,
                    old_registration_consumed=False, old_backgrounds='GT only; original last pair verified')

    def close_tracks(self):
        try: self.track_cache.close()
        finally: self.track_writer_lease.__exit__(None, None, None)

    def load_raw_columns(self, source, record, *, include_gt):
        if getattr(self, 'ccr_history_cache_mode', None) != 'require' or getattr(self, 'ccr_val_history_cache_mode', None) != 'require':
            raise RuntimeError('shared-track training reuses existing original caches READ ONLY')
        raw = super().load_raw_columns(source, record, include_gt=include_gt)
        fixed = raw['_column_causal_preparation']; current = fixed['prepared_state']['current']
        times = np.asarray([source.nusc.get('sample', str(t))['timestamp']/1e6
                            for t in record['history_tokens'][-4:]], np.float64)
        # The generic cache hash has a future_poses slot: use a FIXED marker,
        # never actual future poses. Track payload/identity are history-only.
        key_raw = {k: raw[k] for k in ('history_occ', 'history_observed', 'history_poses')}
        key_raw['future_poses'] = np.array([0], np.uint8)
        key = str(record['scene_name']), str(record['t0_token']), tuple(times)
        track, hit = self.track_cache.get_or_build(key, key_raw,
            lambda: tracks.from_raw(raw, self, times, current=current)[1])
        tracks.check_cached_record(record, current, track, raw['history_poses'][-1])
        # Copy containers; original cache rows / KTA / anchors remain immutable.
        return dict(raw, _shared_source_track=track, _shared_track_cache_hit=hit,
            _column_causal_preparation=dict(fixed, registrations=track.registrations,
                                           source_track=track, audit=track.audit))


def backgrounds(raw, state, poses, provider, track):
    majority = getattr(provider, 'strong_majority', None)
    predicted = poses.detach().cpu().numpy() if isinstance(poses, torch.Tensor) else np.asarray(poses)
    with strong_majority_execution(majority) if majority is not None else nullcontext():
        anchors, components = runtime._strong_all_horizons(raw['history_occ'][-1], state['current_pose'],
            predicted, state['current'], track.pair_velocity, state['source_world_points'],
            frame_dt_s=float(provider.pcfg.frame_dt_s), grid=provider.pcfg.grid, cfg=provider.strong,
            runtime_device=provider.device, majority_backend='native' if majority is not None else 'sparse_cuda')
    return [compose_component_replacements_fast_exact(a, c, [], dynamic_class_ids=DYNAMIC_CLASS_IDS,
        free_label=17, grid=provider.pcfg.grid,
        precomputed_clear_flat_indices=runtime.baseline_clear_flat_indices(c, grid=provider.pcfg.grid))
        for a, c in zip(anchors, components)]


def forward_conditioned(body, provider, row, poses, *, head=None, commands=None):
    rec, raw, ego, history = row
    track = raw['_shared_source_track']; device = provider.device
    tracks.check_cached_record(rec, history[0]['state']['current'], track, raw['history_poses'][-1])
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        field, reads, times = encode_window(body, raw, history, ego[0], device)
        output = body.transport(field, reads, times, torch.as_tensor(rec['source_class_id'], device=device).long(),
                                torch.as_tensor(rec['kta_displacement_xy_m'], device=device).float())
        lifted = body.columns.lift_history(field)
    predicted = None
    if poses is None:
        if head is None or commands is None: raise ValueError('internal Pred needs a head and declared commands')
        known = np.asarray(raw['history_observed'], bool) & (np.asarray(raw['history_occ']) < 18)
        scene = prepare_scene(field, observed=known, history_poses=raw['history_poses'], ego_history=ego[0],
                              grid=provider.pcfg.grid, pool_side=head.config.pool_side)
        predicted = head(torch.as_tensor(np.asarray(ego[0])[None], device=device),
                         torch.as_tensor(np.asarray(commands)[None], device=device), scene)
        poses = poses_from_se2(raw['history_poses'][-1], predicted['se2'][0].detach().cpu().numpy())
    fw = forecast_conditioned(body, provider, row, dict(field=field, output=output, lifted=lifted), poses,
        background_builder=lambda historical, state, dest, p: backgrounds(historical, state, dest, p, track))
    if predicted is not None: fw['ego'] = predicted
    return fw


class SharedTrackPredictor(Predictor):
    """Opt-in production interface. No old association and no runtime rebinding."""
    def prepare(self, record, raw, *, times=None):
        if set(raw) != {'history_occ', 'history_observed', 'history_poses'}:
            raise ValueError('historical fields ONLY')
        if len(raw['history_occ']) != 4 or str(record['history_tokens'][-1]) != str(record['t0_token']):
            raise ValueError('exact four histories ending at t0 required')
        frames, track = tracks.from_raw(raw, self.provider, times)
        current = frames[-1]; pose = np.asarray(raw['history_poses'][-1])
        points = runtime._precompute_source_world(current, pose, self.provider.pcfg.grid)
        native, kta, anchors = tracks.kta_and_anchors(current, track, pose)
        xyz = np.asarray([s['centroid_world'] for s in current]).reshape(-1, 3)
        rec = {**{k: record[k] for k in ('scene_name', 't0_token', 'future_tokens')},
            'history_tokens': tuple(record['history_tokens'][-4:]),
            'source_class_id': np.asarray([s['class_id'] for s in current], np.int64),
            'source_centroid_xy_t0_m': native[:, :2].astype(np.float32),
            'kta_displacement_xy_m': kta, 'anchors_xy_t0_m': anchors}
        state = dict(current_pose=pose, current=current, source_world_points=points,
            source_rel_xy=[v[:, :2]-xyz[i, None, :2] for i, v in enumerate(points)], source_z_t0=native[:, 2])
        value = dict(raw, _shared_source_track=track,
            _column_causal_preparation=dict(prepared_state=state, registrations=track.registrations,
                                           source_track=track, audit=track.audit))
        history = self.history.build(value)  # consumers retain registration_valid, not track_valid
        return rec, value, (ego_history_features(raw['history_poses'], times), None, None), history

    @torch.no_grad()
    def forward(self, row, *, poses=None, commands=None):
        return forward_conditioned(self.body, self.provider, row, poses, head=self.head, commands=commands)


def historical_row(provider, record, historical, ego_history, *, times=None):
    """Same adapter for real STC eval, without initializing another GPU model."""
    from types import SimpleNamespace
    adapter = SimpleNamespace(provider=provider, history=ResidualHistoryInputs(provider, ram_mib=0))
    # Actual timestamps are present in the existing ego-history representation.
    if times is None:
        times = np.asarray(ego_history)[:, -1].astype(np.float64)*1.5  # undo time normalization
    rec, raw, _, history = SharedTrackPredictor.prepare(adapter, record, historical, times=times)
    return rec, raw, (ego_history, None, None), history  # preserve original history feature bytes
