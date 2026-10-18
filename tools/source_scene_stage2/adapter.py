"""No-ICP cache/live paths; all future ego-conditioned geometry rebuilt live."""
from dataclasses import asdict
import numpy as np
import torch
from real_motion.causal_geometry_cache import CausalGeometryCache
from real_motion.ego_navigation import ego_history_features
from real_motion.shared_source_track import kta_and_anchors, check_cached_record
from tools.real_motion.run_p0_f9_shared_evidence_pilot import PilotProvider
from tools.shared_source_track.adapter import backgrounds
from tools.final_semantic_evaluation.predictor import Predictor
from tools.shared_history_experiments.model import encode_window
from tools.shared_history_experiments.refine.geometry import TransportPrior
from tools.joint_occ_experiments.soft_geometry import source_targets, effective_yaw
from tools.joint_occ_experiments.data import PURE_STATE
from . import track
from .history import DirectHistory
from .geometry import aligned_memory


class DirectProvider(PilotProvider):
    def attach_tracks(self, root, *, rank, implementation):
        from real_motion.v21_source_induction import stable_json_fingerprint
        from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
        identity = stable_json_fingerprint(dict(protocol=track.PROTOCOL, grid=asdict(self.pcfg.grid),
            strong=asdict(self.strong), frame_dt_s=self.pcfg.frame_dt_s, implementation=implementation))
        self.track_cache = CausalGeometryCache(str(root)+f'/rank{rank}', identity,
            max_bytes=8*2**30, ram_bytes=128*2**20, reserve_bytes=2*2**30)
        self.track_writer_lease = evaluation_lock(self.track_cache.cache_root); self.track_writer_lease.__enter__()
        return dict(protocol=track.PROTOCOL, root=str(root), history_only=True, registration_computed=False,
                    old_registration_consumed=False, original_two_frame_KTA=True)

    def close_tracks(self):
        try: self.track_cache.close()
        finally: self.track_writer_lease.__exit__(None, None, None)

    def load_raw_columns(self, source, record, *, include_gt):
        if self.ccr_history_cache_mode != 'require' or self.ccr_val_history_cache_mode != 'require':
            raise RuntimeError('old raw geometry caches must remain READ ONLY')
        if source is not self.ccr_history_cache_source and source is not self.ccr_val_history_cache_source:
            raise RuntimeError('source is not attached to a required read-only cache; no legacy builder fallback')
        raw = super().load_raw_columns(source, record, include_gt=include_gt)
        fixed = raw['_column_causal_preparation']; current = fixed['prepared_state']['current']
        times = np.asarray([source.nusc.get('sample', str(t))['timestamp']/1e6
                            for t in record['history_tokens'][-4:]], np.float64)
        key_raw = {k: raw[k] for k in ('history_occ', 'history_observed', 'history_poses')}
        key_raw['future_poses'] = np.array([0], np.uint8)
        identity, hit = self.track_cache.get_or_build((record['scene_name'], record['t0_token'], tuple(times)),
            key_raw, lambda: track.from_raw(raw, self, times, current=current)[1])
        check_cached_record(record, current, identity, raw['history_poses'][-1])
        # Old pickle may contain ICP payloads; NEVER expose them to consumers.
        state = {k: fixed['prepared_state'][k] for k in (*PURE_STATE, 'column_backgrounds')}
        return dict(raw, _shared_source_track=identity, _shared_track_cache_hit=hit,
            _column_causal_preparation=dict(prepared_state=state, source_track=identity))


class DirectPredictor(Predictor):
    def __init__(self, body, pcfg, device, workers=4):
        super().__init__(body, pcfg, device, workers)
        self.history = DirectHistory(self.provider, ram_mib=0)

    def prepare(self, record, raw, *, times=None):
        if set(raw) != {'history_occ', 'history_observed', 'history_poses'}:
            raise ValueError('four historical inputs ONLY')
        if len(raw['history_occ']) != 4 or str(record['history_tokens'][-1]) != str(record['t0_token']):
            raise ValueError('four histories ending at t0 required')
        from tools.real_motion import benchmark_p0_f9_v18_runtime as runtime
        frames, identity = track.from_raw(raw, self.provider, times)
        current = frames[-1]; pose = np.asarray(raw['history_poses'][-1])
        points = runtime._precompute_source_world(current, pose, self.provider.pcfg.grid)
        native, kta, anchors = kta_and_anchors(current, identity, pose)
        xyz = np.asarray([c['centroid_world'] for c in current]).reshape(-1, 3)
        rec = {**{k: record[k] for k in ('scene_name', 't0_token', 'future_tokens')},
            'history_tokens': tuple(record['history_tokens'][-4:]),
            'source_class_id': np.asarray([c['class_id'] for c in current], np.int64),
            'source_centroid_xy_t0_m': native[:, :2].astype(np.float32),
            'kta_displacement_xy_m': kta, 'anchors_xy_t0_m': anchors}
        state = dict(current_pose=pose, current=current, source_world_points=points,
            source_rel_xy=[v[:, :2]-xyz[i, None, :2] for i, v in enumerate(points)], source_z_t0=native[:, 2])
        value = dict(raw, _shared_source_track=identity, _column_causal_preparation=dict(prepared_state=state))
        return rec, value, (ego_history_features(raw['history_poses'], times), None, None), self.history.build(value)

    @torch.no_grad()
    def forward(self, row, *, poses): return forward_window(self.body, self.provider, row, poses=poses, fresh=True)


def forward_window(model, provider, row, *, poses=None, fresh=False):
    rec, raw, ego, history = row; state = history[0]['state']; device = provider.device
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        field, reads, times = encode_window(model, raw, history, ego[0], device)
        output = model.transport(field, reads, times, None, None)
        lifted = model.columns.lift_history(field)
    poses = torch.as_tensor(np.asarray(raw['future_poses'] if poses is None else poses), device=device).double()
    if poses.shape != (6, 4, 4) or not bool(torch.isfinite(poses).all()): raise ValueError('six finite ego conditions required')
    bg = (backgrounds(raw, state, poses, provider, raw['_shared_source_track']) if fresh else
          raw['_column_causal_preparation']['prepared_state']['column_backgrounds'])
    state = dict(state, column_backgrounds=bg)
    targets = source_targets(output, rec, state['current_pose'], state['source_z_t0'])
    yaws = effective_yaw(output, [int(s['class_id']) for s in state['current']])
    scene = TransportPrior(raw, state, provider.pcfg.grid, device); prior = scene.render(poses, targets, yaws)
    memory = aligned_memory(model.columns, lifted, output, history, raw, targets, yaws, poses, provider.pcfg.grid)
    logits = model.columns(memory, prior, range(6), scene.shape)
    return dict(field=field, output=output, lifted=lifted, state=state, poses=poses,
                targets=targets, yaws=yaws, prior=prior, memory=memory, logits=logits, shape=scene.shape)
