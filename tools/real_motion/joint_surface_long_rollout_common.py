"""GT-free Surface CCR blocks; no Local frontier decoder or future cache reads."""
from collections import defaultdict
import time

import numpy as np
import torch

from real_motion.surface_canonical_repair import SurfaceAtlas, augment_evidence, augment_projection
from real_motion.surface_projection_execution import SurfaceMapExecution
from real_motion.surface_ccr_execution import SurfaceExecution
from real_motion.canonical_repair_execution import CanonicalCpuExecution
from real_motion.canonical_causal_repair import build_canonical_evidence, map_canonical_evidence
from real_motion.causal_rollout_handoff import handoff_from_prepared
from tools.real_motion import ccr_screen_common as ccr
from tools.real_motion import joint_long_rollout_common as rollout
from tools.real_motion import benchmark_p0_f9_v18_runtime as runtime
from tools.real_motion.joint_column_common import JointColumnProvider
from tools.real_motion.joint_column_full_common import build_ccr_training_geometry
from tools.real_motion.causal_column_common import causal_source_history

PROTOCOL = 'p0_f9_surface_ccr_frozen_mean_causal_rollout_6s_v1'
PRIMARY_ROUTE = 'reconciled'
THRESHOLDS = (.5, None)


def require_history_only(raw):
    if raw.get('future_gt_occ') is not None:
        raise RuntimeError('Surface rollout prediction received future GT occupancy')
    if any(k in raw for k in ('future_observed', 'future_mask', 'future_annotations')):
        raise RuntimeError('Surface rollout prediction received future sensor/annotation inputs')
    if len(raw['history_occ']) != 4 or len(raw['history_poses']) != 4 or len(raw['history_observed']) != 4:
        raise RuntimeError('Surface rollout requires exactly four history frames')
    if len(raw['future_poses']) != 6:
        raise RuntimeError('Surface block requires six future ego poses')


def causal_geometry(raw, state, provider):
    """Use current predicted Strong state, INCLUDING a reconciled velocity.

    Do not rebuild Strong/motion from raw after a handoff, and do not compute
    obsolete Local future-memory/footprint/frontier branches.
    """
    registrations, _, _, audit = causal_source_history(
        raw['history_occ'], raw['history_poses'], state, provider.pcfg.grid,
        provider.strong, min(2, provider.workers), previous_instances=state.get('previous'))
    return dict(current=state['current'], registrations=registrations, audit=audit,
                footprints=None, memory=None, prepared_state=state)


class SurfaceRolloutProvider(JointColumnProvider):
    """Whitelist initial real histories; synthetic blocks never call the source.

    An optional verified VAL cache is used ONLY for first-block records already
    in the six-history cache. Early GenieDrive starts are rebuilt, never dropped.
    Known cache corruption/misses fail closed rather than silently falling back.
    """
    def load_raw_columns(self, source, record, *, include_gt):
        if include_gt or source is None:
            raise RuntimeError('real rollout inputs require a source and include_gt=False')
        from real_motion.prepared import _load_history_semantics_and_observation
        tokens = tuple(record['history_tokens'][-4:])
        future = tuple(record['future_tokens'])
        if len(tokens) != 4 or len(future) != 6:
            raise RuntimeError('first block requires four real histories and six future keys')
        hist, observed = _load_history_semantics_and_observation(
            source, record['scene_name'], tokens, self.pcfg.free_label, 1)
        raw = dict(history_occ=hist, history_observed=observed,
                   history_poses=[source.pose(t) for t in tokens],
                   future_poses=[source.pose(t) for t in future], future_gt_occ=None)
        require_history_only(raw)
        if 'features' in record:
            cache = getattr(self, 'rollout_val_cache', None)
            if cache is not None:
                raw['_column_causal_preparation'] = cache.require(
                    (str(record['scene_name']), str(record['t0_token'])), raw)
                raw['_ccr_history_cache_hit'] = True
            else:
                raw['_column_causal_preparation'] = build_ccr_training_geometry(
                    raw, record, self.pcfg, self.strong, 1)
        return raw

    def prepare_columns(self, source, record, *, include_gt, raw_window=None, outputs=None):
        if include_gt:
            raise RuntimeError('Surface rollout forbids supervised preparation')
        raw = self.load_raw_columns(source, record, include_gt=False) if raw_window is None else raw_window
        require_history_only(raw)
        if 'features' not in record:
            state = rollout.build_four_history_state(raw['history_occ'], raw['history_poses'],
                raw['future_poses'], self.pcfg, self.strong, self.device)
            record = {**state['rec'], **record}
            state['rec'] = record
            raw['_column_causal_preparation'] = causal_geometry(raw, state, self)
        causal = raw.get('_column_causal_preparation')
        # Slim VAL artifacts omit Strong tensors required by the first renderer
        # exactness gate. Rebuild once live, while retaining verified support.
        rebuild = (not getattr(self, 'columns_checked', False) and causal is not None
                   and 'anchors' not in causal.get('prepared_state', {}))
        if rebuild:
            raw['_column_causal_preparation'] = {k:v for k,v in causal.items() if k != 'prepared_state'}
        try:
            return super().prepare_columns(source, record, include_gt=False, raw_window=raw, outputs=outputs)
        finally:
            if rebuild: raw['_column_causal_preparation'] = causal


class SurfaceBlockExecution:
    """Fresh evidence/projection/source values each call; bounded verified graphs."""
    def __init__(self, provider, *, mode='native_parallel', workers=4, query_workers=4, graphs=True):
        self.provider = provider
        self.cpu = CanonicalCpuExecution(mode, workers)
        provider.ccr_execution = SurfaceMapExecution(self.cpu)
        provider.surface_query_workers = query_workers
        provider.ccr_augment_evidence = lambda e,p: augment_evidence(e, self.atlas(p,e))
        provider.ccr_augment_plan = lambda e,plan,p: augment_projection(
            e, plan, p.state['current_pose'], p.state['world_to_future'], provider.pcfg.grid)
        self.head = SurfaceExecution(provider.joint.columns, provider.device, graphs=graphs,
                                     capture_full_chunks_only=True)

    def atlas(self, prep, evidence):
        atlas = SurfaceAtlas(evidence.world, evidence.classes, evidence.presence,
                             evidence.actor, prep.state['current_pose'], self.provider.pcfg.grid)
        atlas.query_workers = self.provider.surface_query_workers
        return atlas

    @torch.no_grad()
    def predict(self, prep, *, reference=False):
        require_history_only(prep.raw)
        provider = self.provider; grid = provider.pcfg.grid
        stage = {}; tick = time.perf_counter()
        if reference:
            plain = build_canonical_evidence(prep, grid)
            atlas = self.atlas(prep, plain); atlas.query_workers = 1
            evidence = augment_evidence(plain, atlas)
            plan = map_canonical_evidence(evidence, prep, grid)
            plan = augment_projection(evidence, plan, prep.state['current_pose'], prep.state['world_to_future'], grid)
            stage['evidence_projection'] = time.perf_counter()-tick; tick = time.perf_counter()
            probability = ccr.probabilities(provider.joint.columns, evidence, plan, prep.outputs, provider.device)
        else:
            # Atlas and phase are invocation-local: no reuse across synthetic
            # history, changed owners, learned source poses, or checkpoints.
            evidence = ccr.build_inputs(provider, prep)
            plan = ccr.map_inputs(provider, evidence, prep)
            stage['evidence_projection'] = time.perf_counter()-tick; tick = time.perf_counter()
            probability = self.head(provider.joint.columns, evidence, plan, prep.outputs, provider.device)
        stage['head'] = time.perf_counter()-tick; tick = time.perf_counter()
        if not np.isfinite(probability).all() or np.any(probability[...,1] != 0):
            raise RuntimeError('nonfinite Surface probabilities or REMOVE unexpectedly enabled')
        dense = ccr.compose_canonical(prep.baseline, evidence, plan, probability[...,0],
                                      probability[...,1], thresholds=THRESHOLDS)
        if len(dense) != 6 or any(x.shape != tuple(grid.shape_hwd) for x in dense):
            raise RuntimeError('Surface rollout must produce all SIX dense frames')
        edits = defaultdict(int)
        for old, new in zip(prep.baseline, dense):
            edits['added'] += int(((old == 17)&(new != 17)).sum())
            edits['removed'] += int(((old != 17)&(new == 17)).sum())
            edits['changed'] += int((old != new).sum())
            if np.any(new[old != 17] != old[old != 17]):
                raise RuntimeError('Surface ADD-only compositor modified occupied voxels')
        edits['canonical_points'] = len(evidence)
        stage['composition'] = time.perf_counter()-tick
        return dense, dict(edits), stage, probability

    def verify(self, prep, predicted, probability):
        reference, _, _, scores = self.predict(prep, reference=True)
        if not np.array_equal(probability, scores):
            raise RuntimeError('Surface optimized/reference probability bytes differ')
        rollout.assert_dense_equal(reference, predicted)

    def close(self):
        self.head.close(); self.cpu.close()


def synthetic_preparation(predictions, first_raw, poses, window, provider, *, handoff=None, frames=None):
    if len(predictions) != 6 or len(poses) != 12:
        raise ValueError('six first predictions and twelve ego poses required')
    require_history_only(first_raw)
    raw = dict(history_occ=np.stack(predictions[-4:]).astype(np.uint8, copy=True),
        history_poses=list(poses[2:6]), future_poses=list(poses[6:]), future_gt_occ=None,
        history_observed=rollout.inherited_observation_masks(first_raw, poses[2:6], provider.pcfg.grid, provider.workers))
    state = rollout.build_four_history_state(raw['history_occ'], raw['history_poses'], raw['future_poses'],
        provider.pcfg, provider.strong, provider.device, motion_handoff=handoff, component_frames=frames)
    rec = state['rec']
    rec.update(scene_name=str(window.scene_name), t0_token=str(window.future_tokens[5]),
        history_tokens=tuple(window.future_tokens[2:6]), future_tokens=tuple(window.future_tokens[6:]),
        sample_id='surface-predicted-block2')
    raw['_column_causal_preparation'] = causal_geometry(raw, state, provider)
    # Explicit raw, no source, no persistent lookup: even a cache at this
    # future t0 must NEVER replace the predicted histories/handed-off motion.
    return provider.prepare_columns(None, rec, include_gt=False, raw_window=raw)


@torch.no_grad()
def verify_first_block(provider, record, prep, dense, probability, execution):
    state = rollout.build_four_history_state(prep.raw['history_occ'], prep.raw['history_poses'],
        prep.raw['future_poses'], provider.pcfg, provider.strong, provider.device)
    rollout.assert_four_inputs_equal(record, state['rec'])
    runtime._stage_gpu_inputs(state, provider.device)
    try:
        baseline = runtime._forecast_once(provider.joint.transport, state, provider.pcfg, provider.strong, provider.device)
    finally:
        runtime._release_gpu_inputs(state)
    rollout.assert_dense_equal(prep.baseline, baseline)
    execution.verify(prep, dense, probability)
