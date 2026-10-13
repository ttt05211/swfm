from types import SimpleNamespace
import numpy as np
import torch
from real_motion.local_st_world_model_v17 import config_from_mapping_v17
from real_motion.strong_w2det import StrongW2DetConfig
from real_motion.strong_majority_execution import ParallelNativeMajority
from real_motion.native_column_cpu import prepare_native
from real_motion.ego_navigation import ego_history_features
from real_motion.motion_transport import world_points_to_t0
from tools.shared_history_experiments.full_gt.model import GTBody
from tools.shared_history_experiments.refine.history import ResidualHistoryInputs
from tools.scene_ego_head.diagnostics.engine import external_forward
from tools.scene_ego_head.frozen_gt import engine as frozen
from tools.scene_ego_head.frozen_gt.run import validate_completed_source
from tools.scene_ego_head.model import SceneEgoHead, SceneEgoConfig
from tools.real_motion import benchmark_p0_f9_v18_runtime as runtime
from tools.real_motion.causal_column_common import causal_source_history
from tools.real_motion.surface_ego_ablation_common import tensor_state_fingerprint


def load_body(path, device):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    validate_completed_source(saved)
    config = saved['contract']['model']; shared, repair = config['shared_history'], config['repair']
    body = GTBody(config_from_mapping_v17(config['motion']), feature_width=shared['width'],
        z_bins=shared['z_bins'], memory_width=repair['memory_width'], width=repair['width'], chunk_size=repair['chunk_size'])
    body.columns.epsilon = repair['epsilon']
    body.load_state_dict(saved['models']['gt'], strict=True)
    return saved, body.to(device).eval().requires_grad_(False)


def load_head10(path, body, body_sha, device):
    v = torch.load(path, map_location='cpu', weights_only=False); c = v['contract']
    from tools.scene_ego_head.frozen_gt.training import PROTOCOL
    if (v.get('protocol') != PROTOCOL or v.get('epochs') != 10 or c['schedule']['epochs'] != 10 or not v.get('final_epoch_only')
            or c['stage']['source_sha256'] != body_sha
            or c['body_state_sha256'] != tensor_state_fingerprint(body.state_dict())
            or tensor_state_fingerprint(v['head']) != v['state_sha256']
            or v['updates'] != 10*((len(c['source_train_keys'])+c['schedule']['batch_size']-1)//c['schedule']['batch_size'])):
        raise RuntimeError('only original completed head10 bound to this GT epoch2 body is allowed')
    spec = c['head']
    head = SceneEgoHead(SceneEgoConfig(**{k: spec[k] for k in ('feature_width', 'width', 'heads', 'pool_side')}))
    head.load_state_dict(v['head'], strict=True)
    return head.to(device).eval().requires_grad_(False)


class Predictor:
    def __init__(self, body, pcfg, device, workers=4, head=None):
        prepare_native()  # explicit startup preflight; never silently fall back to another backend
        self.body, self.head = body, head
        self.provider = SimpleNamespace(joint=body, pcfg=pcfg, strong=StrongW2DetConfig(free_label=17),
            device=torch.device(device), workers=workers, strong_majority=ParallelNativeMajority(min(workers, 4)))
        self.history = ResidualHistoryInputs(self.provider, ram_mib=0)

    def close(self): self.provider.strong_majority.close()

    def prepare(self, record, raw, *, times=None):
        if any(k in raw for k in ('future_poses', 'future_observed', 'future_annotations', 'future_mask')):
            raise RuntimeError('predictor preparation accepts historical fields ONLY')
        if raw.get('future_gt_occ') is not None: raise RuntimeError('future occupancy supplied to predictor')
        raw = {k: raw[k] for k in ('history_occ', 'history_observed', 'history_poses')}
        if len(raw['history_occ']) != 4 or np.asarray(raw['history_observed']).shape != np.asarray(raw['history_occ']).shape:
            raise ValueError('exact four semantic/observation histories required')
        if (len(record['history_tokens']) < 4 or record['history_tokens'][-1] != record['t0_token']
                or len(record['future_tokens']) != 6):
            raise ValueError('historical identity and six destination keys required')
        p = self.provider; poses = np.asarray(raw['history_poses']); current_pose = poses[-1]
        current = runtime.extract_instances_cropped_exact(raw['history_occ'][-1], poses[-1], grid=p.pcfg.grid, cfg=p.strong)
        previous = runtime.extract_instances_cropped_exact(raw['history_occ'][-2], poses[-2], grid=p.pcfg.grid, cfg=p.strong)
        velocities = runtime.match_instances(previous, current, float(p.pcfg.frame_dt_s), max_speed_mps=p.strong.max_match_speed_mps)
        points = runtime._precompute_source_world(current, current_pose, p.pcfg.grid)
        world = np.asarray([c['centroid_world'] for c in current], np.float64).reshape(-1, 3)
        native = np.asarray([world_points_to_t0(c[None], current_pose)[0] for c in world]).reshape(-1, 3)
        kta = np.zeros((len(current), 6, 2), np.float32); anchors = np.zeros_like(kta)
        for i in range(len(current)):
            v = np.asarray(velocities.get(i, np.zeros(3))) @ current_pose[:3, :3]
            for h in range(6):
                delta = v[:2]*((h+1)*float(p.pcfg.frame_dt_s))
                kta[i, h] = delta; anchors[i, h] = native[i, :2]+delta
        state = dict(current_pose=current_pose, current=current, previous=previous, velocities=velocities,
            source_world_points=points, source_rel_xy=[x[:, :2]-np.asarray(c['centroid_world'])[None, :2] for x, c in zip(points, current)],
            source_z_t0=native[:, 2])
        registrations, _, _, audit = causal_source_history(raw['history_occ'], poses, state, p.pcfg.grid,
            p.strong, min(2, p.workers), previous_instances=previous)
        rec = {**{k: record[k] for k in ('scene_name', 't0_token', 'future_tokens')},
            'history_tokens': tuple(record['history_tokens'][-4:]),
            'source_class_id': np.asarray([c['class_id'] for c in current], np.int64),
            'source_centroid_xy_t0_m': native[:, :2].astype(np.float32), 'kta_displacement_xy_m': kta,
            'anchors_xy_t0_m': anchors}
        raw['_column_causal_preparation'] = dict(prepared_state=state, registrations=registrations, audit=audit)
        history = self.history.build(raw)
        return rec, raw, (ego_history_features(poses, times), None, None), history

    @torch.no_grad()
    def forward(self, row, *, poses=None, commands=None):
        if poses is not None:
            return external_forward(self.body, self.provider, row, poses)
        if self.head is None or commands is None: raise ValueError('internal Pred requires fixed head10 and declared commands')
        row = (row[0], row[1], (row[2][0], commands, None), row[3])
        return frozen.predict(self.body, self.head, self.provider, row)

    @staticmethod
    def dense(fw):
        return {n: v.argmax(1).reshape(6, *fw['shape']).byte().cpu().numpy()
                for n, v in (('transport', fw['prior']), ('semantic_refine', fw['logits']))}


@torch.no_grad()
def audit_gt_entrance(predictor, inputs, record):
    """Compare fresh history adapter with existing GT training forward, no score."""
    from .sources import window
    from tools.real_motion.joint_column_full_common import build_ccr_training_geometry
    from tools.shared_history_experiments.full_gt import engine as gt
    from tools.scene_ego_head.frozen_gt.engine import deterministic_entrance
    from tools.scene_ego_head.frozen_gt_diagnostics.audit import difference
    w = window(record); _, historical, times = inputs.history(w)
    poses = inputs.poses(w); p = predictor.provider
    with deterministic_entrance():
        ref_raw = {**historical, 'future_poses': poses, 'future_gt_occ': None}
        ref_raw['_column_causal_preparation'] = build_ccr_training_geometry(ref_raw, record, p.pcfg, p.strong, min(2, p.workers))
        hist = predictor.history.build(ref_raw)
        ref = gt.forward_window(predictor.body, p, (record, ref_raw, (ego_history_features(historical['history_poses'], times), None, None), hist))
        fresh = predictor.forward(predictor.prepare(record, historical, times=times), poses=poses)
        fields = {k: difference(ref[k], fresh[k]) for k in ('field', 'prior', 'memory', 'logits')}
        backgrounds = ref_raw['_column_causal_preparation']['prepared_state']['column_backgrounds']
        background_changed = sum(int(np.count_nonzero(np.asarray(a) != np.asarray(b)))
                                 for a, b in zip(backgrounds, fresh['state']['column_backgrounds']))
        changed = int((ref['logits'].argmax(1) != fresh['logits'].argmax(1)).sum())
        report = dict(fields=fields, semantic_changed_voxels=changed, background_changed_voxels=background_changed,
            pass_gate=background_changed == 0 and changed == 0 and all(v['pass_gate'] for v in fields.values()),
            deterministic_diagnostic_only=True, tolerances_unchanged=True)
    return report
