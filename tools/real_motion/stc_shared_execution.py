"""Exact STC execution: one invocation-local history bundle, two live ego routes.

Never reuse Strong, future transforms, owners, projection, probability or output.
Only frozen history-only motion inputs/outputs and canonical evidence are shared.
Independent original execution byte-checks each setting before speed measurement.
"""
from collections import defaultdict
from contextlib import nullcontext
from copy import deepcopy
import time

import numpy as np
import torch

from real_motion.stc_camera_protocol import PROTOCOL, SETTINGS
from real_motion.waymo_i2world import fingerprint
from real_motion.strong_majority_execution import ParallelNativeMajority, strong_majority_execution
from tools.real_motion.waymo_fast_execution_v2 import FastV2WaymoProvider, FastV2SurfaceExecution
from tools.real_motion.joint_column_common import JointColumnProvider
from tools.real_motion.joint_surface_long_rollout_common import require_history_only, verify_first_block, SurfaceBlockExecution
from tools.real_motion.eval_p0_f9_joint_surface_waymo import WaymoSurfaceProvider
from tools.real_motion import joint_long_rollout_common as rollout
from tools.real_motion.stc_camera_evaluation import restore

INPUT_KEYS = ('features', 'local_semantic_tube', 'frame_motion_features', 'target_source_mask_tube',
              'kta_displacement_xy_m', 'anchors_xy_t0_m', 'source_centroid_xy_t0_m', 'source_class_id')


class SharedHistoryProvider(FastV2WaymoProvider):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pair_key = None
        self.pair = self.plain_evidence = self.enriched_evidence = None
        self.reused_histories = 0

    def clear_pair(self):
        self.pair_key = self.pair = self.plain_evidence = self.enriched_evidence = None

    def prepare_columns(self, source, record, *, include_gt, raw_window=None, outputs=None):
        if include_gt or raw_window is None or outputs is not None or 'features' in record:
            raise RuntimeError('STC shared preparation requires raw four-history inference')
        if self.joint.training or any(p.requires_grad for p in self.joint.parameters()):
            raise RuntimeError('STC history sharing requires frozen eval weights')
        raw = raw_window; require_history_only(raw); tick = time.perf_counter()
        key = (record['scene_name'], record['t0_token'], tuple(record['history_tokens']),
               tuple(self.geometry.key(t, a, m, p) for t, a, m, p in zip(record['history_tokens'],
                     raw['history_occ'], raw['history_observed'], raw['history_poses'])))
        if key != self.pair_key:
            self.clear_pair(); self.pair_key = key
            prep = super().prepare_columns(source, record, include_gt=False, raw_window=raw)
            self.pair = prep
            self.fast_prepare_stages['shared_history'] = 0.
            return prep
        # A copied state dictionary; the first route's state/raw/results remain intact.
        previous = self.pair; state = dict(previous.state); future = list(raw['future_poses'])
        state['future_poses'] = future
        state['world_to_future'] = [np.linalg.inv(np.asarray(p, np.float64)) for p in future]
        legacy = rollout.legacy
        anchors, baseline = legacy._strong_all_horizons(raw['history_occ'][-1], state['current_pose'],
            future, state['current'], state['velocities'], state['source_world_points'],
            frame_dt_s=float(self.pcfg.frame_dt_s), grid=self.pcfg.grid, cfg=self.strong,
            runtime_device=self.device)
        state.update(anchors=anchors, baseline_by_hi=baseline,
            baseline_clear_by_hi=[legacy.baseline_clear_mask(v, grid=self.pcfg.grid) for v in baseline],
            baseline_clear_flat_by_hi=[legacy.baseline_clear_flat_indices(v, grid=self.pcfg.grid) for v in baseline],
            gpu=None)
        state['rec'] = {**previous.state['rec'], **record}
        raw['_waymo_frame_geometry'] = previous.raw['_waymo_frame_geometry']
        raw['_column_causal_preparation'] = dict(previous.raw['_column_causal_preparation'], prepared_state=state)
        # The frozen network is a function of history only, not future ego poses.
        # Never reuse the rendered baseline/layers: fresh Strong/SE(2)/owners below.
        prep = JointColumnProvider.prepare_columns(self, None, state['rec'], include_gt=False,
            raw_window=raw, outputs=dict(previous.outputs))
        self.reused_histories += 1
        self.fast_prepare_stages = dict(shared_history=1., live_strong_motion_layers=time.perf_counter()-tick)
        return prep


class SharedSurfaceExecution(FastV2SurfaceExecution):
    def __init__(self, provider, **kwargs):
        super().__init__(provider, **kwargs)
        delegate = provider.ccr_execution; enrich = provider.ccr_augment_evidence
        class PairMap:
            def build(self, prep, grid):
                if provider.plain_evidence is None:
                    provider.plain_evidence = delegate.build(prep, grid)
                return provider.plain_evidence
            def map(self, evidence, prep, grid):
                return delegate.map(evidence, prep, grid)
        def shared_enrich(evidence, prep):
            if provider.enriched_evidence is None:
                provider.enriched_evidence = enrich(evidence, prep)
            return provider.enriched_evidence
        provider.ccr_execution = PairMap()
        provider.ccr_augment_evidence = shared_enrich


class Predictor:
    def __init__(self, joint, pcfg, device, *, workers=4, graphs=True, parallel_majority=True,
                 shared=True, geometry_mib=512):
        self.shared = shared
        self.provider = (SharedHistoryProvider(joint, pcfg, device, workers, geometry_mib=geometry_mib)
                         if shared else WaymoSurfaceProvider(joint, pcfg, device, workers))
        cls = SharedSurfaceExecution if shared else SurfaceBlockExecution
        self.execution = cls(self.provider, workers=workers, query_workers=workers, graphs=graphs)
        self.majority = ParallelNativeMajority(min(4, workers)) if parallel_majority else None

    @torch.no_grad()
    def full(self, record, raw, *, verify=False):
        tick = time.perf_counter()
        with strong_majority_execution(self.majority) if self.majority else nullcontext():
            prep = self.provider.prepare_columns(None, record, include_gt=False, raw_window=raw)
        stages = dict(history_and_transport_prepare=time.perf_counter()-tick)
        dense, _, detail, probability = self.execution.predict(prep); stages.update(detail)
        if verify:
            verify_first_block(self.provider, prep.state['rec'], prep, dense, probability, self.execution)
        return prep, dense, probability, stages

    def __call__(self, record, raw, *, verify):
        _, dense, _, stages = self.full(record, raw, verify=verify)
        return dense, stages

    def close(self):
        self.execution.close()
        if self.majority is not None: self.majority.close()
        if self.shared:
            self.provider.clear_pair(); self.provider.close()


def bytes_equal(a, b, name):
    if isinstance(a, torch.Tensor): a = a.detach().cpu().numpy()
    if isinstance(b, torch.Tensor): b = b.detach().cpu().numpy()
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.dtype != b.dtype or a.tobytes() != b.tobytes():
        raise RuntimeError('STC shared original byte gate failed: '+name)


@torch.no_grad()
def check_and_time(source, windows, reference, shared, *, repeats=2, stop_event=None):
    """Full actual four-setting execution; no GT targets, no saved score updates."""
    if not windows or repeats < 1: raise ValueError('nonempty paired speed population required')
    def run(predictor, window, setting, verify=False):
        if stop_event is not None and stop_event.is_set(): raise InterruptedError('STC speed stopped')
        return predictor.full(*source.prediction_inputs(window, setting), verify=verify)
    for index, window in enumerate(windows):
        for setting in SETTINGS:
            a = run(reference, window, setting, verify=index == 0)
            b = run(shared, window, setting, verify=index == 0)
            for key in INPUT_KEYS: bytes_equal(a[0].state['rec'][key], b[0].state['rec'][key], setting+'.'+key)
            for i, (x, y) in enumerate(zip(a[0].baseline, b[0].baseline)): bytes_equal(x, y, setting+'.transport'+str(i))
            bytes_equal(a[2], b[2], setting+'.probability')
            for i, (x, y) in enumerate(zip(a[1], b[1])): bytes_equal(x, y, setting+'.dense'+str(i))
    elapsed = defaultdict(list)
    def sync():
        if shared.provider.device.type == 'cuda': torch.cuda.synchronize(shared.provider.device)
    for repeat in range(repeats):
        for name, predictor in ([('original', reference), ('shared', shared)] if repeat % 2 == 0 else
                                [('shared', shared), ('original', reference)]):
            if predictor.shared: predictor.provider.clear_pair()
            for setting in SETTINGS: run(predictor, windows[0], setting)
            sync(); tick = time.perf_counter()
            for window in windows:
                for setting in SETTINGS: run(predictor, window, setting)
            sync(); elapsed[name].append((time.perf_counter()-tick)/len(windows))
    means = {k: float(np.mean(v)) for k, v in elapsed.items()}
    return dict(windows=len(windows), settings=list(SETTINGS), repeats=repeats,
        seconds_per_four_setting_window=means, repeat_seconds=dict(elapsed),
        speedup=means['original']/means['shared'], motion_input_bytes_exact=True,
        transport_bytes_exact=True, probability_bytes_exact=True, six_dense_bytes_exact=True,
        future_targets_read=False, scientific_counts_committed=False,
        scope='same frozen model, same four-setting windows, AB/BA; NOT raw-camera FPS')


def migrate_state(old_contract, old_state, new_contract, *, shape):
    if old_contract.get('protocol') != PROTOCOL or new_contract.get('protocol') != PROTOCOL:
        raise RuntimeError('STC migration requires identical protocol')
    excluded = {'implementation', 'fast_execution', 'execution_migration'}
    science = lambda c: {k: v for k, v in c.items() if k not in excluded}
    if science(old_contract) != science(new_contract):
        raise RuntimeError('STC migration changes data/population/weights/config/thresholds/runtime')
    implementations = old_contract.get('implementation', {})
    if not implementations or any(new_contract.get('implementation', {}).get(k) != v for k, v in implementations.items()):
        raise RuntimeError('original STC code changed; cannot bridge old integer counts')
    value = deepcopy(restore(old_state, old_contract, int(np.prod(shape))))
    value['contract_fingerprint'] = fingerprint(new_contract)
    value['fingerprint'] = fingerprint(value)
    return value
