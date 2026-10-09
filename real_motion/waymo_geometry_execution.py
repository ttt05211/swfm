"""Exact, invocation-local Waymo execution; never cache learned/future state.

Only per-frame components and metric points have an LRU. Registrations,
canonical lattices, surface fits, Strong and six future projections stay live.
Separate module so the running 2Hz implementation fingerprint is unchanged.
"""
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
import hashlib
from threading import RLock
import time
from types import MappingProxyType

import numpy as np

from .canonical_causal_repair import CanonicalEvidence, FEATURE_DIM, STATIC, _entity, grid_arrays
from .rigid_transport import rigid_source_points_world
from .runtime_fastpath import extract_instances_cropped_exact
from .source_evidence_audit import transform_points
from .surface_canonical_repair import NEIGHBORS, RADIUS, SURFACE_DIM, SurfaceAtlas
from .surface_projection_execution import SurfaceMapExecution

PROTOCOL = 'waymo_exact_history_geometry_execution_v1'


def readonly(value):
    value = np.asarray(value)
    value.setflags(write=False)
    return value


@dataclass(frozen=True)
class FrameGeometry:
    components: tuple
    registration_points: tuple
    canonical_points: tuple
    static_indices: np.ndarray
    static_classes: np.ndarray
    static_world: np.ndarray

    @property
    def nbytes(self):
        arrays = [self.static_indices, self.static_classes, self.static_world,
                  *self.registration_points, *self.canonical_points]
        arrays += [a for c in self.components for a in c.values() if isinstance(a, np.ndarray)]
        return sum(a.nbytes for a in arrays)


class FrameGeometryCache:
    """Bounded RAM, content+pose keys, immutable values, no disk writes.

    An NPZ token alone is not a cache key. Grid/Strong configuration is fixed
    for this cache's lifetime; source validates NPZ size/mtime on every access.
    """
    def __init__(self, grid, strong, *, ram_mib=1024, max_entries=32):
        if not 0 <= ram_mib <= 4096 or not 1 <= max_entries <= 128:
            raise ValueError('bounded frame geometry RAM/entry budget required')
        self.grid, self.strong = grid, strong
        self.limit, self.max_entries = int(ram_mib*1024**2), max_entries
        self.entries = OrderedDict(); self.bytes = 0; self.lock = RLock()
        self.hits = self.misses = self.evictions = 0
        self.inflight = {}; self.build_seconds = 0.

    @staticmethod
    def key(token, occupancy, visibility, pose):
        h = hashlib.sha256(str(token).encode())
        for array in (occupancy, visibility, pose):
            a = np.ascontiguousarray(array)
            if a.dtype.hasobject: raise ValueError('object geometry cache input')
            h.update(str(a.dtype).encode()); h.update(str(a.shape).encode())
            if a.size: h.update(memoryview(a).cast('B'))
        return h.digest()

    def get(self, token, occupancy, visibility, pose):
        key = self.key(token, occupancy, visibility, pose)
        with self.lock:
            if key in self.entries:
                self.hits += 1; self.entries.move_to_end(key)
                return self.entries[key]
            owner = key not in self.inflight
            if owner:
                self.inflight[key] = Future(); self.misses += 1
            else: self.hits += 1
            ready = self.inflight[key]
        if not owner: return ready.result()  # NEVER hold the lock while waiting/building
        tick = time.perf_counter()
        try:
            value = self._build(occupancy, visibility, pose)
            with self.lock:
                self.build_seconds += time.perf_counter()-tick
                if value.nbytes <= self.limit:
                    while self.entries and (self.bytes+value.nbytes > self.limit
                                            or len(self.entries) >= self.max_entries):
                        _, old = self.entries.popitem(last=False)
                        self.bytes -= old.nbytes; self.evictions += 1
                    self.entries[key] = value; self.bytes += value.nbytes
                del self.inflight[key]; ready.set_result(value)
            return value
        except BaseException as error:
            with self.lock:
                self.inflight.pop(key, None); ready.set_exception(error)
            raise

    def _build(self, occupancy, visibility, pose):
        components = extract_instances_cropped_exact(occupancy, pose, grid=self.grid, cfg=self.strong)
        origin, step, _ = grid_arrays(self.grid)
        registration = tuple(readonly(rigid_source_points_world(c['voxel_indices'], pose, grid=self.grid))
                             for c in components)
        canonical = tuple(readonly(transform_points(origin+(c['voxel_indices']+.5)*step, pose))
                          for c in components)
        # SAME mixed-class transform batch/order as the canonical reference.
        occ, vis = np.asarray(occupancy), np.asarray(visibility, bool)
        idx = np.argwhere(vis & ((occ == 11) | (occ == 13)))
        labels = occ[tuple(idx.T)] if len(idx) else np.empty(0, np.uint8)
        world = transform_points(origin+(idx+.5)*step, pose)
        components = tuple(MappingProxyType({k: readonly(v) if isinstance(v, np.ndarray) else v
                                             for k, v in c.items()}) for c in components)
        return FrameGeometry(components, registration, canonical, readonly(idx), readonly(labels), readonly(world))

    def window(self, record, raw):
        if raw.get('future_gt_occ') is not None or len(raw['history_occ']) != 4:
            raise RuntimeError('frame geometry requires FOUR history-only inputs')
        return tuple(self.get(token, occ, seen, pose) for token, occ, seen, pose in zip(
            record['history_tokens'], raw['history_occ'], raw['history_observed'], raw['history_poses']))

    def stats(self):
        with self.lock:
            return dict(hits=self.hits, misses=self.misses, evictions=self.evictions,
                        entries=len(self.entries), RAM_mib=self.bytes/1024**2,
                        RAM_limit_mib=self.limit/1024**2, persistent_writes=0,
                        pending_frames=len(self.inflight), build_worker_seconds_sum=self.build_seconds,
                        cached='single-frame components/static points ONLY')

    def clear(self):
        with self.lock:
            self.entries.clear(); self.bytes = 0


class GeometryPrefetchSource:
    """One bounded next-history task, scheduled AFTER current SIX forecasts.

    Source I/O remains on the main thread. Only immutable frame geometry goes
    to the worker; no CUDA, registration, model features, targets or metrics.
    Future target access remains inside metric_targets after evaluator's gate.
    """
    def __init__(self, source, selected, cache, *, enabled=True):
        self.source, self.cache = source, cache
        self.next = {w.anchor: n for w, n in zip(selected, selected[1:])}
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='waymo-frame') if enabled else None
        self.pending = None; self.pending_anchor = None; self.raw_next = None
        self.wait_seconds = self.submit_seconds = 0.
        self.last_wait_seconds = self.last_submit_seconds = 0.

    def __getattr__(self, name): return getattr(self.source, name)

    def prediction_inputs(self, window):
        tick = time.perf_counter()
        if self.pending_anchor == window.anchor:
            record, raw = self.raw_next
            frames = self.pending.result()
            self.pending = self.pending_anchor = self.raw_next = None
        else:
            record, raw = self.source.prediction_inputs(window)
            frames = self.cache.window(record, raw)
        self.last_wait_seconds = time.perf_counter()-tick
        self.wait_seconds += self.last_wait_seconds
        raw['_waymo_frame_geometry'] = frames
        return record, raw

    def metric_targets(self, window):
        self.last_submit_seconds = 0.
        targets = self.source.metric_targets(window)  # called AFTER six forecast gate
        nxt = self.next.get(window.anchor)
        if self.pool is not None and nxt is not None:
            if self.pending is not None: raise RuntimeError('bounded prefetch task was not consumed')
            tick = time.perf_counter()
            record, raw = self.source.prediction_inputs(nxt)  # only next window's actual histories
            self.pending_anchor, self.raw_next = nxt.anchor, (record, raw)
            self.pending = self.pool.submit(self.cache.window, record, raw)
            self.last_submit_seconds = time.perf_counter()-tick
            self.submit_seconds += self.last_submit_seconds
        return targets

    def close(self):
        if self.pool is not None: self.pool.shutdown(wait=True, cancel_futures=True); self.pool = None
        self.pending = self.pending_anchor = self.raw_next = None


def build_cached_evidence(prepared, grid, *, kernels=None, executor=None):
    """Exact canonical reference ordering/arithmetic, reuse only frame primitives."""
    frames = prepared.raw.get('_waymo_frame_geometry')
    if frames is None or len(frames) != 4:
        raise RuntimeError('four authenticated historical geometry entries required')
    raw, state = prepared.raw, prepared.state
    origin, step, _ = grid_arrays(grid); inverse = np.linalg.inv(np.asarray(state['current_pose']))
    groups, actors, classes = [], [], []
    counts = dict(dense_entities=0, sparse_entities=0, max_lattice_cells=0)
    def entity(actor, cls, points, cells):
        options = dict(halo=True, max_lattice_cells=4_000_000, materialize_features=True, kernels=kernels)
        return (_entity(actor, cls, points, cells, prepared, grid, **options) if executor is None else
                executor.submit(_entity, actor, cls, points, cells, prepared, grid, **options))
    lookup = [{id(c['voxel_indices']): p for c, p in zip(frame.components, frame.canonical_points)}
              for frame in frames]
    for actor, comp in enumerate(state['current']):
        points, cells = [], []
        for f, reg in enumerate(prepared.registrations[actor]):
            if reg is None:
                points.append(np.empty((0, 3))); cells.append(np.empty((0, 3), np.int64)); continue
            ijk = np.asarray(reg[1], np.int64)
            world = lookup[f].get(id(ijk))
            if world is None: world = transform_points(origin+(ijk+.5)*step, raw['history_poses'][f])
            aligned = transform_points(world, reg[0])
            if f < 3: aligned = aligned[np.asarray(raw['history_observed'][f], bool)[tuple(ijk.T)]]
            cell = ijk.copy() if f == 3 else np.floor((transform_points(aligned, inverse)-origin)/step).astype(np.int64)
            points.append(aligned); cells.append(cell)
        groups.append(entity(actor, int(comp['class_id']), points, cells))
        actors.append(actor); classes.append(int(comp['class_id']))
    sp, sc = {11: [], 13: []}, {11: [], 13: []}
    for f, frame in enumerate(frames):
        ijk, world = frame.static_indices, frame.static_world
        cell = ijk if f == 3 else np.floor((transform_points(world, inverse)-origin)/step).astype(np.int64)
        for cls in (11, 13):
            take = frame.static_classes == cls
            sp[cls].append(world[take]); sc[cls].append(cell[take])
    for cls in (11, 13):
        groups.append(entity(STATIC, cls, sp[cls], sc[cls])); actors.append(STATIC); classes.append(cls)
    data, labels, worlds, presence, aa, cc = [], [], [], [], [], []
    for group, actor, cls in zip(groups, actors, classes):
        if executor is not None: group = group.result()
        if group is None: continue
        feature, lab, world, pres, dense, volume, _ = group
        data.append(feature); labels.append(lab); worlds.append(world); presence.append(pres)
        aa.append(np.full(len(world), actor, np.int32)); cc.append(np.full(len(world), cls, np.uint8))
        counts['dense_entities' if dense else 'sparse_entities'] += 1
        counts['max_lattice_cells'] = max(counts['max_lattice_cells'], volume)
    cat = lambda xs, shape, dtype: np.concatenate(xs) if xs else np.empty(shape, dtype)
    actor, cls, pres = cat(aa, (0,), np.int32), cat(cc, (0,), np.uint8), cat(presence, (0,4), bool)
    return CanonicalEvidence(cat(data, (0,FEATURE_DIM), np.float32), cat(labels, (0,4), np.uint8),
        actor, cls, cat(worlds, (0,3), np.float64), pres,
        {**counts, 'points':len(actor), 'dynamic_points':int((actor >= 0).sum()),
         'static_points':int((actor == STATIC).sum()), 'halo_points':int((~pres.any(1)).sum()),
         'future_GT_used':False, 'metric_observations_preserved':True,
         'support':'all t0 sources + visible registered source history + observed road/sidewalk + one face halo',
         'features_materialized':True})


class CachedSurfaceMapExecution(SurfaceMapExecution):
    def build(self, prepared, grid):
        return build_cached_evidence(prepared, grid, kernels=self.kernels, executor=self.pool)


def _describe_rows(atlas, cls, query):
    """Same row-local float64 expressions and reduction order as SurfaceAtlas."""
    distance, index = atlas.trees[cls].query(query*atlas.scale, k=NEIGHBORS,
                                          distance_upper_bound=RADIUS, workers=1)
    valid = np.isfinite(distance); safe = np.minimum(index, len(atlas.metric[cls])-1)
    delta = (atlas.metric[cls][safe]-query[:, None])/atlas.step
    delta = np.where(valid[..., None], delta, 0.)
    weight = np.where(valid, 1./(1.+np.where(valid, distance, 0.)**2), 0.)
    total = weight.sum(1).clip(1e-12)
    mean = (weight[..., None]*delta).sum(1)/total[:, None]
    centered = delta-mean[:, None]
    def cov(a, b): return (weight*centered[..., a]*centered[..., b]).sum(1)/total
    xx, yy, xy = cov(0,0)+1e-3, cov(1,1)+1e-3, cov(0,1)
    xz, yz = cov(0,2), cov(1,2)
    determinant = (xx*yy-xy*xy).clip(1e-9)
    gx, gy = (xz*yy-yz*xy)/determinant, (yz*xx-xz*xy)/determinant
    intercept = mean[:,2]-gx*mean[:,0]-gy*mean[:,1]
    error = delta[...,2]-(intercept[:,None]+gx[:,None]*delta[...,0]+gy[:,None]*delta[...,1])
    rms = np.sqrt((weight*error**2).sum(1)/total); count = valid.sum(1)
    nearest = np.where(count > 0, distance[:,0], RADIUS)/RADIUS
    other = atlas.trees[13 if cls == 11 else 11]; opposite = np.full(len(query), np.inf)
    if other is not None:
        opposite, _ = other.query(query*atlas.scale, k=1, distance_upper_bound=RADIUS, workers=1)
    opposite_seen = np.isfinite(opposite); opposite = np.where(opposite_seen, opposite/RADIUS, 1.)
    recent = (weight*atlas.recent[cls][safe]).sum(1)/total
    column = valid & (np.abs(delta[...,0]) < .55) & (np.abs(delta[...,1]) < .55)
    values = np.column_stack((count >= 3, -intercept, rms, np.sqrt(np.maximum(cov(2,2), 0.)),
        gx, gy, count/NEIGHBORS, nearest, opposite, opposite_seen,
        column.sum(1)/np.maximum(count,1), recent))
    return np.clip(values, -4., 4.).astype(np.float32)


class ChunkedSurfaceAtlas(SurfaceAtlas):
    """Bounded row chunks in the existing CPU pool, unchanged K/radius/fit.

    Transform each class's entire query batch BEFORE splitting: changing FP64
    GEMM/GEMV batch shape can change floor/boundary results. Only independent
    nearest-neighbour queries and row-local fits are parallelized.
    """
    def describe(self, evidence):
        chunk = int(getattr(self, 'chunk_rows', 4096))
        if not 256 <= chunk <= 65536: raise ValueError('bounded surface chunk rows required')
        result = np.zeros((len(evidence), SURFACE_DIM), np.float32)
        jobs = []
        for cls in (11,13):
            ids = np.flatnonzero((evidence.actor == STATIC) & (evidence.classes == cls))
            if not len(ids) or self.trees[cls] is None: continue
            query = transform_points(evidence.world[ids], self.inverse)
            jobs += [(cls, ids[b:b+chunk], query[b:b+chunk]) for b in range(0,len(ids),chunk)]
        def work(job): return job[1], _describe_rows(self, job[0], job[2])
        pool = getattr(self, 'fit_pool', None)
        for ids, values in (map(work, jobs) if pool is None else pool.map(work, jobs)):
            result[ids] = values
        if not np.isfinite(result).all(): raise RuntimeError('nonfinite surface descriptor')
        return result


def augment_chunked(evidence, atlas):
    if evidence.features.shape[1] != FEATURE_DIM: raise ValueError('unaugmented features required')
    return replace(evidence, features=np.concatenate((evidence.features, atlas.describe(evidence)), 1))
