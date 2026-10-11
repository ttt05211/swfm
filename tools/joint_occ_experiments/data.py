"""Read-only legacy history reuse, with explicit pose-dependent invalidation."""
from collections import OrderedDict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from types import SimpleNamespace
from pathlib import Path
import hashlib
import json
import time
from threading import RLock
import numpy as np
import torch
from contextlib import nullcontext
from real_motion.causal_geometry_cache import CausalGeometryCache, PROTOCOL as CACHE_PROTOCOL, array_bytes
from real_motion.ccr_history_geometry import namespace as train_namespace, PROTOCOL as TRAIN_PROTOCOL
from real_motion.ccr_val_history_cache import namespace as val_namespace, PROTOCOL as VAL_PROTOCOL
from real_motion.canonical_causal_repair import materialize_compact_canonical, map_canonical_evidence
from real_motion.surface_canonical_repair import SurfaceAtlas, augment_evidence
from real_motion.ego_navigation import ego_history_features, relative_se2, navigation_commands
from tools.real_motion import benchmark_p0_f9_v18_runtime as runtime
from tools.real_motion.causal_column_common import render_column_layers
from real_motion.runtime_fastpath import compose_component_replacements_fast_exact
from real_motion.metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS

PURE_STATE = ('current_pose', 'current', 'source_world_points', 'source_rel_xy', 'source_z_t0')
DISCARDED = ('world_to_future', 'column_backgrounds', '_ccr_compact_conflicts',
             'footprints', 'memory', 'fixed_candidate_geometry')


def history_only_payload(causal):
    state = {k: causal['prepared_state'][k] for k in PURE_STATE}
    return dict(state=state, registrations=causal['registrations'],
                support=causal['_ccr_compact_support'], audit=causal.get('audit', {}))


def attach_caches(provider, args, root):
    report = dict(reused=list(PURE_STATE)+['registrations', 'compact canonical support'],
        rebuilt_for_pred=list(DISCARDED), learned_ego_bank_reused=False,
        old_signatures_unchanged=True, persistent_writes=False)
    for split, location, namespace, count in (
        ('train', args.ccr_history_cache, train_namespace, 20430),
        ('val', args.ccr_val_history_cache, val_namespace, 4369),
    ):
        identity = namespace(provider, args, root)
        digest = hashlib.sha256((CACHE_PROTOCOL+identity).encode()).hexdigest()
        directory = Path(location).resolve()/digest
        path = directory/'manifest.json'
        if not path.is_file(): raise RuntimeError(f'existing {split} history cache required: {path}; no rebuild')
        manifest = json.loads(path.read_text(encoding='utf-8'))
        if (manifest.get('complete') is not True or manifest.get('namespace') != digest
                or int(manifest.get('windows', -1)) != count
                or int(manifest.get('artifacts', -1)) != count
                or manifest.get('protocol') != (TRAIN_PROTOCOL if split == 'train' else VAL_PROTOCOL)
                or manifest.get('future_GT_cached') is not False
                or manifest.get('learned_outputs_cached') is not False
                or manifest.get('sampled_ids_cached') is not False
                or manifest.get('optimizer_or_rng_cached') is not False):
            raise RuntimeError(f'{split} cache manifest/provenance mismatch')
        cache = CausalGeometryCache(location, identity, max_bytes=0, ram_bytes=args.geometry_ram_mib*2**20,
                                    reserve_bytes=0, compression_level=6)
        setattr(provider, 'ccr_'+('history' if split == 'train' else 'val_history')+'_cache', cache)
        setattr(provider, 'ccr_'+('history' if split == 'train' else 'val_history')+'_cache_mode', 'require')
        report[split] = dict(directory=str(directory), namespace=digest, windows=count)
    return report


class HistoryInputs:
    """Bounded read-only CPU descriptors shared by BOTH arms, never live features."""
    def __init__(self, provider, *, ram_mib=512):
        self.provider = provider
        self.limit = ram_mib*2**20
        self.rows, self.bytes = OrderedDict(), 0
        self.hits = self.misses = 0
        self.lock, self.pending = RLock(), {}

    def get(self, record, raw):
        key = (record['scene_name'], record['t0_token'])
        with self.lock:
            if key in self.rows:
                self.hits += 1
                self.rows.move_to_end(key)
                return self.rows[key][0]
            future = self.pending.get(key)
            owner = future is None
            if owner:
                self.misses += 1
                future = Future(); self.pending[key] = future
            else: self.hits += 1
        if not owner: return future.result()
        try:
            value = self.build(raw)
            size = array_bytes(value)
            with self.lock:
                if size <= self.limit:
                    while self.rows and self.bytes+size > self.limit:
                        _, (_, old) = self.rows.popitem(last=False)
                        self.bytes -= old
                    self.rows[key] = value, size
                    self.bytes += size
                self.pending.pop(key)
                future.set_result(value)
            return value
        except BaseException as error:
            with self.lock:
                self.pending.pop(key, None)
                future.set_exception(error)
            raise

    def build(self, raw):
        causal = history_only_payload(raw['_column_causal_preparation'])
        # No future poses, occupancy or label-bearing record are passed to
        # materialization/plane fitting. Static conflict maps are NOT reused.
        historical = {k: raw[k] for k in ('history_occ', 'history_observed', 'history_poses')}
        prep = SimpleNamespace(raw=historical, state=causal['state'], registrations=causal['registrations'])
        support = causal['support']
        evidence = materialize_compact_canonical(support, prep, self.provider.pcfg.grid,
                                                np.arange(len(support), dtype=np.int64))
        atlas = SurfaceAtlas.from_compact(support, prep, self.provider.pcfg.grid)
        atlas.query_workers = 1
        evidence = augment_evidence(evidence, atlas)
        return causal, evidence

    def stats(self):
        with self.lock:
            return dict(hits=self.hits, misses=self.misses, ram_mib=self.bytes/2**20,
                        limit_mib=self.limit/2**20, pending=len(self.pending))


def prepare_projection(provider, record, raw, output, future_poses, history, *, use_gt_cache):
    causal, evidence = history
    state = {**causal['state'], 'rec': record, 'gpu': None}
    if len(state['current']) != len(record['features']):
        raise RuntimeError('history-cache/source-record actor count mismatch')
    if state['current']:
        expected = np.asarray(record['source_class_id'])
        if not np.array_equal(expected, [int(c['class_id']) for c in state['current']]):
            raise RuntimeError('history-cache/source-record class/order mismatch')
        world = np.array([c['centroid_world'] for c in state['current']], np.float64)
        t0 = (world-np.asarray(state['current_pose'])[:3, 3]) @ np.asarray(state['current_pose'])[:3, :3]
        if not np.allclose(t0[:, :2], np.asarray(record['source_centroid_xy_t0_m']), atol=2e-4, rtol=0):
            raise RuntimeError('history-cache/source-record centroid/order mismatch')
    poses = np.asarray(future_poses, np.float64)
    if poses.shape != (6, 4, 4) or not np.isfinite(poses).all():
        raise ValueError('six finite future transforms required')
    if use_gt_cache:
        if not np.array_equal(poses, np.asarray(raw['future_poses'])):
            raise RuntimeError('GT cached projection requested for different ego poses')
        state['column_backgrounds'] = raw['_column_causal_preparation']['prepared_state']['column_backgrounds']
    else:
        # All horizon-dependent Strong/KTA/CLEAR fields are rebuilt from the
        # predicted ego. Historical association and source point clouds survive.
        previous = runtime.extract_instances_cropped_exact(raw['history_occ'][-2], raw['history_poses'][-2],
                                                           grid=provider.pcfg.grid, cfg=provider.strong)
        velocities = runtime.match_instances(previous, state['current'], float(provider.pcfg.frame_dt_s),
                                              max_speed_mps=provider.strong.max_match_speed_mps)
        majority = getattr(provider, 'strong_majority', None)
        from real_motion.strong_majority_execution import strong_majority_execution
        with strong_majority_execution(majority) if majority is not None else nullcontext():
            anchors, components = runtime._strong_all_horizons(raw['history_occ'][-1], state['current_pose'], poses,
                state['current'], velocities, state['source_world_points'], frame_dt_s=float(provider.pcfg.frame_dt_s),
                grid=provider.pcfg.grid, cfg=provider.strong, runtime_device=provider.device,
                majority_backend='native' if majority is not None else 'sparse_cuda')
        state['column_backgrounds'] = [compose_component_replacements_fast_exact(a, c, [],
            dynamic_class_ids=DYNAMIC_CLASS_IDS, free_label=17, grid=provider.pcfg.grid,
            precomputed_clear_flat_indices=runtime.baseline_clear_flat_indices(c, grid=provider.pcfg.grid))
            for a, c in zip(anchors, components)]
    state['world_to_future'] = [np.linalg.inv(p) for p in poses]
    forward_raw = {k: raw[k] for k in ('history_occ', 'history_observed', 'history_poses')}
    forward_raw['future_poses'] = poses
    forward_raw['future_gt_occ'] = None
    diagnostic = {k: v.detach() for k, v in output.items() if isinstance(v, torch.Tensor)}
    baseline, owners, fallbacks, components, targets, yaws = render_column_layers(state, record, diagnostic, provider.pcfg.grid)
    prep = SimpleNamespace(raw=forward_raw, state=state, registrations=causal['registrations'],
        baseline=baseline, owners=owners, fallbacks=fallbacks, components=components,
        targets=targets, yaws=yaws, outputs=output)
    execution = getattr(provider, 'ccr_execution', None)
    plan = (execution.map(evidence, prep, provider.pcfg.grid) if execution is not None
            else map_canonical_evidence(evidence, prep, provider.pcfg.grid))
    return prep, evidence, plan


def ego_inputs(source, record, raw):
    # Future poses appear ONLY as labels; navigation commands are the declared
    # benchmark condition. No occupancy, mask, annotation or GT motion feature.
    tokens = tuple(record['history_tokens'])[-4:]
    times = np.array([source.nusc.get('sample', str(t))['timestamp']/1e6 for t in tokens])
    history = ego_history_features(raw['history_poses'], times)
    target = relative_se2(raw['history_poses'][-1], raw['future_poses'])
    commands = navigation_commands(source.nusc, tuple(record['future_tokens']))
    return history, commands, target


def prefetched_batches(provider, source, groups, descriptors, *, workers=4, depth=2):
    """Bounded ordered window futures; workers never touch CUDA, model or RNG."""
    if not 1 <= workers <= 8 or not 1 <= depth <= 4: raise ValueError('bounded prefetch required')
    iterator, pending = iter(groups), deque()
    def load(record):
        tick = time.perf_counter()
        raw = provider.load_raw_columns(source, record, include_gt=True)
        inputs = ego_inputs(source, record, raw)
        io = time.perf_counter()-tick
        tick = time.perf_counter()
        history = descriptors.get(record, raw)
        raw['_joint_worker_timing'] = dict(io_and_ego_metadata=io, history_descriptors=time.perf_counter()-tick)
        return record, raw, inputs, history
    with ThreadPoolExecutor(max_workers=workers) as pool:
        def submit():
            group = next(iterator, None)
            if group is None: return False
            pending.append([pool.submit(load, r) for r in group])
            return True
        for _ in range(depth):
            if not submit(): break
        while pending:
            futures = pending.popleft()
            tick = time.perf_counter()
            rows = [future.result() for future in futures]
            wait = time.perf_counter()-tick
            submit()
            # CPU-only descriptor fitting is also prefetched, not a serial
            # training-thread pause. Single-flight prevents duplicate fits.
            if rows: rows[0][1]['_joint_prepare_timing'] = dict(input_wait=wait, descriptors=0.)
            yield rows
