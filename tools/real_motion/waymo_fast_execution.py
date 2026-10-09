"""Isolated exact Waymo preparation and Surface execution, no model changes."""
from collections import defaultdict
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import torch

from real_motion.waymo_geometry_execution import (FrameGeometryCache, CachedSurfaceMapExecution,
    ChunkedSurfaceAtlas, augment_chunked)
from real_motion.canonical_causal_repair import build_canonical_evidence, map_canonical_evidence
from real_motion.surface_canonical_repair import SurfaceAtlas, augment_evidence, augment_projection
from real_motion.source_evidence_audit import associate_backwards, register_history_shape
from real_motion.waymo_i2world import fingerprint, file_sha256
from tools.real_motion.causal_column_common import pose_motion
from tools.real_motion.eval_p0_f9_joint_surface_waymo import WaymoSurfaceProvider
from tools.real_motion.joint_column_common import JointColumnProvider
from tools.real_motion.joint_surface_long_rollout_common import SurfaceBlockExecution, require_history_only
from tools.real_motion import joint_long_rollout_common as rollout, ccr_screen_common as ccr
from tools.real_motion.waymo_zero_shot_common import restore


def cached_causal_geometry(raw, state, frames):
    """Same matching/ICP as causal_source_history, no component re-extraction."""
    components = [list(f.components) for f in frames]
    points = [f.registration_points for f in frames]
    links, audit = associate_backwards(components, state['current'], state['velocities'], dt=.5)
    registrations = [[None]*4 for _ in state['current']]
    for i, comp in enumerate(state['current']):
        registrations[i][-1] = (np.eye(4), np.asarray(comp['voxel_indices'], np.int64))
        for f, j in enumerate(links[i][:-1]):
            if j is None: continue
            p = points[f][j]
            result = register_history_shape(p, state['source_world_points'][i], allow_yaw=int(comp['class_id']) != 7)
            key = 'registration_accepted' if result.accepted else 'registration_rejected'
            audit[key] = audit.get(key, 0)+1
            if result.accepted:
                r = pose_motion(np.zeros(3), np.zeros(3), result.yaw_rad)
                r[:2,3] = result.points[:,:2].mean(0)-p[:,:2].mean(0)@r[:2,:2].T
                registrations[i][f] = (r, np.asarray(components[f][j]['voxel_indices'], np.int64))
    return dict(current=state['current'], registrations=registrations, audit=audit,
                footprints=None, memory=None, prepared_state=state)


class FastWaymoSurfaceProvider(WaymoSurfaceProvider):
    def __init__(self, joint, pcfg, device, workers, *, geometry_mib=1024):
        super().__init__(joint, pcfg, device, workers)
        self.geometry = FrameGeometryCache(pcfg.grid, self.strong, ram_mib=geometry_mib)
        self.fast_prepare_stages = {}

    def prepare_columns(self, source, record, *, include_gt, raw_window=None, outputs=None):
        if include_gt or raw_window is None or 'features' in record:
            raise RuntimeError('fast Waymo requires actual FOUR raw histories, not prepared/future cache records')
        raw = raw_window; require_history_only(raw)
        tick = time.perf_counter()
        frames = raw.get('_waymo_frame_geometry')
        if frames is None: frames = self.geometry.window(record, raw)
        raw['_waymo_frame_geometry'] = frames
        timing = dict(frame_geometry_wait=time.perf_counter()-tick); tick = time.perf_counter()
        state = rollout.build_four_history_state(raw['history_occ'], raw['history_poses'], raw['future_poses'],
            self.pcfg, self.strong, self.device, component_frames=[[], [], *[list(f.components) for f in frames]])
        record = {**state['rec'], **record}; state['rec'] = record
        timing['state_tracks_tubes_strong'] = time.perf_counter()-tick; tick = time.perf_counter()
        raw['_column_causal_preparation'] = cached_causal_geometry(raw, state, frames)
        timing['live_history_registration'] = time.perf_counter()-tick; tick = time.perf_counter()
        # Bypass only the redundant raw-history reconstruction of SurfaceRolloutProvider.
        prep = JointColumnProvider.prepare_columns(self, None, record, include_gt=False,
                                                   raw_window=raw, outputs=outputs)
        timing['live_motion_and_layers'] = time.perf_counter()-tick
        self.fast_prepare_stages = timing
        return prep


class FastSurfaceBlockExecution(SurfaceBlockExecution):
    def __init__(self, provider, *, mode='native_parallel', workers=4, query_workers=4,
                 graphs=True, surface_chunk=4096):
        super().__init__(provider, mode=mode, workers=workers, query_workers=query_workers, graphs=graphs)
        self.surface_chunk = surface_chunk; self.detail = defaultdict(float)
        owner = self
        class TimedMap(CachedSurfaceMapExecution):
            def build(self, prep, grid):
                tick = time.perf_counter(); result = super().build(prep, grid)
                owner.detail['canonical_support'] += time.perf_counter()-tick
                return result
            def map(self, evidence, prep, grid):
                tick = time.perf_counter(); result = super().map(evidence, prep, grid)
                owner.detail['six_projection'] += time.perf_counter()-tick
                return result
        provider.ccr_execution = TimedMap(self.cpu)
        def enrich(evidence, prep):
            tick = time.perf_counter(); atlas = self.atlas(prep, evidence)
            self.detail['surface_atlas_build'] += time.perf_counter()-tick; tick = time.perf_counter()
            result = augment_chunked(evidence, atlas)
            self.detail['surface_neighbours_and_fit'] += time.perf_counter()-tick
            return result
        def phase(evidence, plan, prep):
            tick = time.perf_counter()
            result = augment_projection(evidence, plan, prep.state['current_pose'],
                                        prep.state['world_to_future'], provider.pcfg.grid)
            self.detail['live_surface_phase'] += time.perf_counter()-tick
            return result
        provider.ccr_augment_evidence, provider.ccr_augment_plan = enrich, phase

    def atlas(self, prep, evidence):
        atlas = ChunkedSurfaceAtlas(evidence.world, evidence.classes, evidence.presence,
                                   evidence.actor, prep.state['current_pose'], self.provider.pcfg.grid)
        atlas.chunk_rows, atlas.fit_pool = self.surface_chunk, self.cpu.pool
        return atlas

    @torch.no_grad()
    def predict(self, prep, *, reference=False):
        if reference: raise RuntimeError('use verify: reference must not reuse optimized atlas')
        self.detail.clear()
        dense, edits, timing, scores = super().predict(prep)
        timing.update({'detail.'+k:v for k,v in self.detail.items()})
        return dense, edits, timing, scores

    @torch.no_grad()
    def verify(self, prep, predicted, probability):
        # Independent old NumPy support/Atlas/projection/head: no fast callback.
        grid = self.provider.pcfg.grid
        plain = build_canonical_evidence(prep, grid)
        atlas = SurfaceAtlas(plain.world, plain.classes, plain.presence, plain.actor,
                             prep.state['current_pose'], grid)
        expected = augment_evidence(plain, atlas)
        actual = ccr.build_inputs(self.provider, prep)
        for field in ('features', 'labels', 'world', 'actor', 'classes', 'presence'):
            if not np.array_equal(getattr(actual, field), getattr(expected, field)):
                raise RuntimeError('fast Waymo canonical/surface bytes differ: '+field)
        plan = augment_projection(expected, map_canonical_evidence(expected, prep, grid),
                                  prep.state['current_pose'], prep.state['world_to_future'], grid)
        live = ccr.map_inputs(self.provider, actual, prep)
        for field in ('flat', 'base', 'fallback', 'legal', 'context'):
            if not np.array_equal(getattr(live, field), getattr(plan, field)):
                raise RuntimeError('fast Waymo live projection bytes differ: '+field)
        scores = ccr.probabilities(self.provider.joint.columns, expected, plan, prep.outputs, self.provider.device)
        if not np.array_equal(scores, probability): raise RuntimeError('fast Waymo probability bytes differ')
        reference = ccr.compose_canonical(prep.baseline, expected, plan, scores[...,0], scores[...,1], thresholds=(.5,None))
        rollout.assert_dense_equal(reference, predicted)


def migrate_state(old_contract, old_state, new_contract, *, shape):
    """Explicit execution-only bridge; reject ANY scientific/input/code change.

    Caller snapshots source files under the original output's kernel lease.
    New output owns its own contract and integer state. Never weaken ordinary
    --resume checks or silently reinterpret scores under a new method.
    """
    from real_motion.waymo_i2world_10hz import PROTOCOL
    if old_contract.get('protocol') != PROTOCOL or new_contract.get('protocol') != PROTOCOL:
        raise RuntimeError('only literal 10Hz -> same 10Hz execution migration is allowed')
    excluded = {'cpu_workers', 'implementation', 'fast_execution', 'execution_migration'}
    scientific = lambda c: {k:v for k,v in c.items() if k not in excluded}
    if scientific(old_contract) != scientific(new_contract):
        raise RuntimeError('10Hz migration changes population/weights/data/config/environment/semantics')
    old_implementation, new_implementation = old_contract.get('implementation',{}), new_contract.get('implementation',{})
    if not old_implementation or any(new_implementation.get(k) != v for k,v in old_implementation.items()):
        raise RuntimeError('original 10Hz implementation changed; cannot bridge saved counts')
    value = restore(old_state, old_contract, voxel_count=int(np.prod(shape)))
    value['contract_fingerprint'] = fingerprint(new_contract)
    value['fingerprint'] = fingerprint(value)
    return value


def read_continuation(directory):
    """Keep original result/state bytes intact; the kernel lock is not a result."""
    from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
    directory = Path(directory).resolve()
    with evaluation_lock(directory):
        contract = json.loads((directory/'contract.json').read_text(encoding='utf-8'))
        state = json.loads((directory/'state.json').read_text(encoding='utf-8'))
        receipt = dict(directory=str(directory), contract_sha256=file_sha256(directory/'contract.json'),
                       state_sha256=file_sha256(directory/'state.json'), completed_windows=state['completed_windows'])
    return contract, state, receipt


@torch.no_grad()
def paired_speed(source, windows, joint, pcfg, device, *, workers, graphs, parallel_majority,
                 surface_chunk, geometry_mib, repeats=2, stop_event=None):
    """Same ordered windows/weights, real SIX dense output, no targets/selection.

    Prewarm each arm outside timing; alternating whole-population passes retain
    realistic sliding-history cache hits. Report ratios, never assert L40S ETA.
    """
    from contextlib import nullcontext
    from real_motion.strong_majority_execution import ParallelNativeMajority, strong_majority_execution
    if not windows or repeats < 1: raise ValueError('nonempty paired speed population/repeats required')
    original = WaymoSurfaceProvider(joint, pcfg, device, workers)
    fast = FastWaymoSurfaceProvider(joint, pcfg, device, workers, geometry_mib=geometry_mib)
    executions = dict(original=SurfaceBlockExecution(original, workers=workers, query_workers=workers, graphs=graphs),
                      fast=FastSurfaceBlockExecution(fast, workers=workers, query_workers=workers,
                                                     graphs=graphs, surface_chunk=surface_chunk))
    providers = dict(original=original, fast=fast); durations = defaultdict(list); details = defaultdict(lambda:defaultdict(float))
    majority = ParallelNativeMajority(min(4,workers)) if parallel_majority else None
    def sync():
        if torch.device(device).type == 'cuda': torch.cuda.synchronize(device)
    def run(arm, window):
        record, raw = source.prediction_inputs(window)
        with strong_majority_execution(majority) if majority is not None else nullcontext():
            prep = providers[arm].prepare_columns(None, record, include_gt=False, raw_window=raw)
        dense, edits, stages, scores = executions[arm].predict(prep)
        return prep, dense, scores, stages
    try:
        # Per-window actual eager/native/graph bytes, outside speed timing.
        for window in windows:
            if stop_event is not None and stop_event.is_set(): raise InterruptedError('speed check stopped')
            a = run('original', window); b = run('fast', window)
            rollout.assert_four_inputs_equal(a[0].state['rec'], b[0].state['rec'])
            rollout.assert_dense_equal(a[0].baseline, b[0].baseline)
            rollout.assert_dense_equal(a[1], b[1])
            if not np.array_equal(a[2], b[2]): raise RuntimeError('paired fast Waymo probability bytes differ')
        for repeat in range(repeats):
            for arm in (('original','fast') if repeat%2 == 0 else ('fast','original')):
                # Never turn a short diagnostic into all-population geometry
                # cache hits. Real sequential 10Hz has ONE new frame/window.
                if arm == 'fast': fast.geometry.clear()
                # Warm ONLY first history set; subsequent windows miss on the
                # new frame, exactly as a fresh full-data sequential pass.
                run(arm, windows[0]); sync(); tick = time.perf_counter()
                for window in windows:
                    if stop_event is not None and stop_event.is_set(): raise InterruptedError('speed check stopped')
                    prep, dense, scores, stages = run(arm, window)
                    for k,v in stages.items(): details[arm][k] += v
                    del prep, dense, scores
                sync(); durations[arm].append((time.perf_counter()-tick)/len(windows))
        mean = {k:float(np.mean(v)) for k,v in durations.items()}
        return dict(windows=len(windows), repeats=repeats, seconds_per_window=mean,
            repeat_seconds_per_window=dict(durations), speedup=mean['original']/mean['fast'],
            stages_seconds_per_window={k:{n:v/(len(windows)*repeats) for n,v in d.items()} for k,d in details.items()},
            full_population_equality_checks=len(windows), probability_and_six_dense_bytes_exact=True,
            frame_geometry=fast.geometry.stats(), no_future_GT_reads=True, no_saved_scientific_updates=True,
            geometry_timing_policy='clear each timed pass; prewarm first history set ONLY; one new frame/window',
            scope='paired complete history/Strong/motion/CCR/SIX dense; excludes GT/metrics and warmup; NOT formal FPS')
    finally:
        for execution in executions.values(): execution.close()
        if majority is not None: majority.close()
