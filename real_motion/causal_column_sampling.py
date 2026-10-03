"""Bounded, window-local exact inverse maps for overlapping historical patches.

These are uint8 labels/visibility, NOT learned dense feature volumes. Mapping
uses the same float64 world centres, matrix products and floor as the reference.
No future GT, model output score or supervision influences cache construction.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from .causal_column_completion import UNKNOWN
from .source_evidence_audit import transform_points
from .column_cpu_kernels import metric_indices_inplace


class ColumnHistoryIndex:
    """One prepared window's immutable history geometry, shared by six horizons.

    No GT or predicted pose is indexed. Bounded tight-range membership tables
    replace repeated isin setup; larger/dispersed sources use sorted searches.
    Never retain this index across prepare() calls or persist it to disk.
    """
    def __init__(self, prepared, grid, *, max_membership_mib=8, actors=None):
        self.inverse_history = [np.linalg.inv(t) for t in prepared.raw['history_poses']]
        self.inverse_registration = {}; self.members = {}; self.tables = {}
        self.table_bytes = 0; limit = int(max_membership_mib*2**20)
        wanted = None if actors is None else set(int(a) for a in actors if a >= 0)
        for actor, row in enumerate(prepared.registrations):
            if wanted is not None and actor not in wanted: continue
            inverses, members, tables = [], [], []
            for reg in row:
                if reg is None:
                    inverses.append(None); members.append(None); tables.append(None); continue
                inverses.append(np.linalg.inv(reg[0]))
                owned = np.unique(np.ravel_multi_index(reg[1].T, grid.shape_hwd))
                members.append(owned); table = None
                span = int(owned[-1]-owned[0]+1) if len(owned) else 0
                if span and self.table_bytes+span <= limit:
                    bits = np.zeros(span, bool); bits[owned-owned[0]] = True
                    table = (int(owned[0]), bits); self.table_bytes += span
                tables.append(table)
            self.inverse_registration[actor] = inverses; self.members[actor] = members; self.tables[actor] = tables

    def contains(self, actor, frame, flat):
        table = self.tables[actor][frame]
        if table is not None:
            start, bits = table; at = flat-start
            valid = (at >= 0)&(at < len(bits)); out = np.zeros(len(flat), bool)
            out[valid] = bits[at[valid]]; return out
        owned = self.members[actor][frame]
        if not len(owned): return np.zeros(len(flat), bool)
        at = np.searchsorted(owned, flat)
        valid = at < len(owned); out = np.zeros(len(flat), bool)
        out[valid] = owned[at[valid]] == flat[valid]; return out


class ColumnFeatureSampler:
    def __init__(self, prepared, h, plan, grid, config, motion_factory, *, workers=1, max_cache_mib=64, history_index=None):
        self.prepared, self.h, self.grid, self.config = prepared, h, grid, config
        self.maps = {}; self.windows = {}; self.cache_bytes = 0
        self.history = np.asarray(prepared.raw['history_occ'])
        self.t = len(self.history)
        if self.t not in (4, 6): raise ValueError('four or six historical observations required')
        self.observed = np.asarray(prepared.raw['history_observed'])
        self.origin = np.asarray((grid.x_min, grid.y_min, grid.z_min))
        self.step = np.asarray(grid.voxel_size)
        self.shape = np.asarray(grid.shape_hwd)
        self.p, self.z = config.patch, config.z_bins
        self.transforms = {}; self.members = {}
        self.optimized = getattr(prepared, 'cpu_pipeline_optimized', True)
        self.kernels_optimized = self.optimized and getattr(prepared, 'cpu_kernels_optimized', True)
        from .native_column_cpu import get_native
        self.native = get_native() if self.kernels_optimized else None
        from .native_column_cpu import bundle_enabled
        self.bundle = self.native is not None and getattr(prepared,'cpu_bundle_optimized',True) and bundle_enabled()
        self.index = ((history_index if history_index is not None else getattr(prepared, 'column_history_index', None))
                      if self.optimized else None)
        if self.optimized and self.index is None:
            self.index = ColumnHistoryIndex(prepared, grid, actors=np.unique(plan.actor))
        inverse_history = (self.index.inverse_history if self.index is not None else
                           [np.linalg.inv(t) for t in prepared.raw['history_poses']])
        future_pose = prepared.raw['future_poses'][h]
        if int(workers) <= 1:
            self._build(plan, inverse_history, future_pose, motion_factory, max_cache_mib, None)
        else:
            with ThreadPoolExecutor(max_workers=min(int(workers), 6)) as pool:
                self._build(plan, inverse_history, future_pose, motion_factory, max_cache_mib, pool)

    def _build(self, plan, inverse_history, future_pose, motion_factory, max_cache_mib, pool):
        grid = self.grid
        prepared, h = self.prepared, self.h
        groups = np.where(plan.actor < 0, -1, plan.actor)
        for actor in np.unique(groups):
            actor = int(actor)
            transforms, members = [], []
            inverse_motion = None
            if actor >= 0:
                inverse_motion = np.linalg.inv(motion_factory(prepared.state['current'][actor]['centroid_world'],
                    prepared.targets[h][actor], prepared.yaws[h][actor]))
            for f in range(self.t):
                transform = inverse_history[f]@future_pose
                owned = None
                if actor >= 0:
                    reg = prepared.registrations[actor][f]
                    if reg is None:
                        transforms.append(None); members.append(None); continue
                    # Preserve reference operation order; do not algebraically
                    # reassociate matrices or approximate with float32/GPU grids.
                    inverse_reg = self.index.inverse_registration[actor][f] if self.index is not None else np.linalg.inv(reg[0])
                    transform = inverse_history[f]@inverse_reg@inverse_motion@future_pose
                    owned = self.index.members[actor][f] if self.index is not None else np.unique(np.ravel_multi_index(reg[1].T, grid.shape_hwd))
                transforms.append(transform); members.append(owned)
            self.transforms[actor], self.members[actor] = transforms, members
            centres = plan.evidence_xy[groups == actor]
            if not len(centres): continue
            lo = centres.min(0)-self.p//2; hi = centres.max(0)+self.p//2+1
            extent = hi-lo; area = int(np.prod(extent)); bytes_needed = 2*self.t*area*self.z
            # TRAIN's few sampled columns and pathological/dispersed components
            # should remain sparse. Never truncate candidates to fit a cache.
            if (len(centres) < 32 or area*2 > len(centres)*self.p**2
                    or self.cache_bytes+bytes_needed > max_cache_mib*2**20):
                continue
            self.maps[actor] = self._map(actor, lo, extent, pool)
            _, labels, flags = self.maps[actor]
            # Views only: no 49x replicated dense cache. One gather copies just
            # the requested batch to its original N,6,P,P,Z ordering.
            self.windows[actor] = (np.lib.stride_tricks.sliding_window_view(labels, (self.p, self.p), axis=(1, 2)),
                                   np.lib.stride_tricks.sliding_window_view(flags, (self.p, self.p), axis=(1, 2)))
            self.cache_bytes += bytes_needed

    def _map(self, actor, lo, extent, pool):
        x = np.arange(lo[0], lo[0]+extent[0]); y = np.arange(lo[1], lo[1]+extent[1])
        idx = np.stack(np.meshgrid(x, y, np.arange(self.z), indexing='ij'), -1)
        xyz = (self.origin+(idx+.5)*self.step).reshape(-1, 3)
        shape = (*extent, self.z)
        if self.bundle and pool is None:
            # TRAIN has a window-level pool already. One bounded integer call
            # for all frames per chunk, not nested frame pools/calls/copies.
            labels=np.empty((self.t,len(xyz)),np.uint8); flags=np.empty_like(labels)
            for start in range(0,len(xyz),65536):
                stop=min(start+65536,len(xyz))
                values,bits=self._batch_gather(actor,xyz[start:stop],stop-start)
                labels[:,start:stop]=values[0]; flags[:,start:stop]=bits[0]
            return lo,labels.reshape((self.t,*shape)),flags.reshape((self.t,*shape))
        def frame(f):
            labels = np.full(len(xyz), UNKNOWN, np.uint8); flags = np.zeros(len(xyz), np.uint8)
            transform = self.transforms[actor][f]
            if transform is None: return labels.reshape(shape), flags.reshape(shape)
            membership = None
            if actor >= 0 and self.index is None:
                membership = np.zeros(int(np.prod(self.shape)), bool)
                membership[self.members[actor][f]] = True
            # Bound working memory for six concurrent frames, independent of
            # component/grid size. Mapping arrays themselves are uint8.
            for start in range(0, len(xyz), 65536):
                stop = min(start+65536, len(xyz))
                pts = transform_points(xyz[start:stop], transform)
                if self.kernels_optimized:
                    ijk, valid = metric_indices_inplace(pts, self.origin, self.step, self.shape, check_bounds=self.native is None)
                else:
                    ijk = np.floor((pts-self.origin)/self.step).astype(np.int64)
                    valid = ((ijk >= 0)&(ijk < self.shape)).all(1)
                if self.native is not None:
                    labels[start:stop], flags[start:stop] = self._native_gather(actor, f, ijk)
                    continue
                at = tuple(ijk[valid].T)
                labels[start:stop][valid] = self.history[f][at]
                bits = self.observed[f][at].astype(np.uint8)
                if actor >= 0 and self.index is not None:
                    bits |= self.index.contains(actor, f, np.ravel_multi_index(ijk[valid].T, tuple(self.shape))).astype(np.uint8)*2
                elif membership is not None:
                    bits |= membership[np.ravel_multi_index(ijk[valid].T, tuple(self.shape))].astype(np.uint8)*2
                flags[start:stop][valid] = bits
            return labels.reshape(shape), flags.reshape(shape)
        active = [f for f in range(self.t) if self.transforms[actor][f] is not None]
        # Single static map shared by generation + static refine, six frames
        # parallelized only here; no nested pools and no CUDA work in threads.
        frames = [None]*self.t
        mapped_frames = map(frame, active) if pool is None else pool.map(frame, active)
        for f, mapped in zip(active, mapped_frames): frames[f] = mapped
        for f in range(self.t):
            if frames[f] is None: frames[f] = (np.full(shape, UNKNOWN, np.uint8), np.zeros(shape, np.uint8))
        return lo, np.stack([v[0] for v in frames]), np.stack([v[1] for v in frames])

    def _native_gather(self, actor, frame, indices):
        owned = self.members[actor][frame] if actor >= 0 else None
        table = self.index.tables[actor][frame] if actor >= 0 and self.index is not None else None
        return self.native.gather(indices, self.history[frame], self.observed[frame], owned, table)

    def _batch_gather(self,actor,xyz,row_size):
        indices=[None]*self.t
        valid=np.array([m is not None for m in self.transforms[actor]],bool)
        for f,transform in enumerate(self.transforms[actor]):
            if transform is None: continue
            # Exact original float64 operations, independently for every frame.
            pts=transform_points(xyz,transform)
            indices[f],_=metric_indices_inplace(pts,self.origin,self.step,self.shape,check_bounds=False)
        members=self.members[actor] if actor >= 0 else [None]*self.t
        tables=self.index.tables[actor] if actor >= 0 and self.index is not None else [None]*self.t
        return self.native.gather_many(indices,self.history,self.observed,members,tables,valid,row_size)

    def _sparse(self, plan, actor):
        """Same reference arithmetic, reusing inverses for uncached/small actors."""
        # Several frontier queries attend the SAME nearest causal anchor. Warp
        # each distinct anchor only once, then restore exact original query order.
        # Class membership is applied after expansion (GEN/static may differ).
        xy = plan.evidence_xy
        if self.optimized and np.all((xy >= 0)&(xy < self.shape[:2])):
            keys = xy[:, 0].astype(np.int64)*int(self.shape[1])+xy[:, 1]
            _, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
            centres = xy[first]  # identical lexicographic XY order, fewer sorts
        else: centres, inverse = np.unique(xy, axis=0, return_inverse=True)
        n, p, z = len(centres), self.p, self.z
        batched=self.bundle and n*p*p*z*self.t*3*8 <= 32*2**20
        hist = None if batched else np.full((n, self.t, p, p, z), UNKNOWN, np.uint8)
        flags = None if batched else np.zeros_like(hist)
        if self.optimized:
            # Same per-coordinate float64 multiply/add as reference, without
            # allocating N*P*P*Z*3 integer indices and arithmetic temporaries.
            xyz = np.empty((n, p, p, z, 3), np.float64)
            offsets = np.arange(p)-p//2
            xyz[..., 0] = (self.origin[0]+(centres[:, 0, None]+offsets[None]+.5)*self.step[0])[:, :, None, None]
            xyz[..., 1] = (self.origin[1]+(centres[:, 1, None]+offsets[None]+.5)*self.step[1])[:, None, :, None]
            xyz[..., 2] = self.origin[2]+(np.arange(z)+.5)*self.step[2]
            xyz = xyz.reshape(-1, 3)
        else:
            offsets = np.stack(np.meshgrid(np.arange(p)-p//2, np.arange(p)-p//2, np.arange(z), indexing='ij'), -1)
            idx = offsets[None]+np.pad(centres, ((0, 0), (0, 1)))[:, None, None, None, :]
            xyz = (self.origin+(idx+.5)*self.step).reshape(-1, 3)
        if batched:
            hist,flags=self._batch_gather(actor,xyz,p*p*z)
            hist=hist.reshape(n,self.t,p,p,z); flags=flags.reshape(n,self.t,p,p,z)
            return self.native.expand(hist,flags,inverse.astype(np.int64,copy=False),plan.classes,actor < 0)
        for f, transform in enumerate(self.transforms[actor]):
            if transform is None: continue
            pts = transform_points(xyz, transform)
            if self.kernels_optimized:
                ijk, valid = metric_indices_inplace(pts, self.origin, self.step, self.shape, check_bounds=self.native is None)
            else:
                ijk = np.floor((pts-self.origin)/self.step).astype(np.int64)
                valid = ((ijk >= 0)&(ijk < self.shape)).all(1)
            if self.native is not None:
                labels, bits = self._native_gather(actor, f, ijk)
            else:
                at = tuple(ijk[valid].T)
                labels = np.full(len(ijk), UNKNOWN, np.uint8); bits = np.zeros(len(ijk), np.uint8)
                labels[valid] = self.history[f][at]; bits[valid] = self.observed[f][at].astype(np.uint8)
                if actor >= 0:
                    flat = np.ravel_multi_index(ijk[valid].T, tuple(self.shape))
                    owned = self.index.contains(actor, f, flat) if self.index is not None else np.isin(flat, self.members[actor][f])
                    bits[valid] |= owned.astype(np.uint8)*2
            hist[:, f] = labels.reshape(n, p, p, z); flags[:, f] = bits.reshape(n, p, p, z)
        if self.native is not None:
            return self.native.expand(hist, flags, inverse.astype(np.int64, copy=False), plan.classes, actor < 0)
        hist, flags = hist[inverse], flags[inverse]
        if actor < 0: flags |= (hist == plan.classes[:, None, None, None, None]).astype(np.uint8)*2
        return hist, flags

    def sample(self, plan, reference_sampler):
        n, p, z = len(plan), self.p, self.z
        hist = np.full((n, self.t, p, p, z), UNKNOWN, np.uint8); flags = np.zeros_like(hist)
        groups = np.where(plan.actor < 0, -1, plan.actor)
        for actor in np.unique(groups):
            take = np.flatnonzero(groups == actor); actor = int(actor)
            if actor not in self.maps:
                hist[take], flags[take] = self._sparse(plan.subset(take), actor); continue
            lo, labels, bits = self.maps[actor]
            starts = plan.evidence_xy[take]-lo-p//2
            lw, fw = self.windows[actor]
            if np.any(starts < 0) or np.any(starts >= np.asarray(lw.shape[1:3])):
                raise RuntimeError('patch cache does not cover supplied actor/anchor queries')
            xx, yy = starts.T
            hist[take] = lw[:, xx, yy].transpose(1, 0, 3, 4, 2)
            flags[take] = fw[:, xx, yy].transpose(1, 0, 3, 4, 2)
            if actor < 0:
                # Class-specific membership is applied AFTER the shared warp.
                flags[take] |= (hist[take] == plan.classes[take, None, None, None, None]).astype(np.uint8)*2
        return {'history': hist, 'flags': flags, 'base': plan.base, 'fallback': plan.fallback,
                'context': plan.context, 'kind': plan.kind, 'classes': plan.classes}
