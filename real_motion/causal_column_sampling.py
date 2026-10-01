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


class ColumnFeatureSampler:
    def __init__(self, prepared, h, plan, grid, config, motion_factory, *, workers=1, max_cache_mib=64):
        self.prepared, self.h, self.grid, self.config = prepared, h, grid, config
        self.maps = {}; self.windows = {}; self.cache_bytes = 0
        self.history = np.asarray(prepared.raw['history_occ'])
        self.observed = np.asarray(prepared.raw['history_observed'])
        self.origin = np.asarray((grid.x_min, grid.y_min, grid.z_min))
        self.step = np.asarray(grid.voxel_size)
        self.shape = np.asarray(grid.shape_hwd)
        self.p, self.z = config.patch, config.z_bins
        self.transforms = {}; self.members = {}
        inverse_history = [np.linalg.inv(t) for t in prepared.raw['history_poses']]
        future_pose = prepared.raw['future_poses'][h]
        with ThreadPoolExecutor(max_workers=max(1, min(int(workers), 6))) as pool:
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
            for f in range(6):
                transform = inverse_history[f]@future_pose
                owned = None
                if actor >= 0:
                    reg = prepared.registrations[actor][f]
                    if reg is None:
                        transforms.append(None); members.append(None); continue
                    # Preserve reference operation order; do not algebraically
                    # reassociate matrices or approximate with float32/GPU grids.
                    transform = inverse_history[f]@np.linalg.inv(reg[0])@inverse_motion@future_pose
                    owned = np.unique(np.ravel_multi_index(reg[1].T, grid.shape_hwd))
                transforms.append(transform); members.append(owned)
            self.transforms[actor], self.members[actor] = transforms, members
            centres = plan.evidence_xy[groups == actor]
            if not len(centres): continue
            lo = centres.min(0)-self.p//2; hi = centres.max(0)+self.p//2+1
            extent = hi-lo; area = int(np.prod(extent)); bytes_needed = 2*6*area*self.z
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
        def frame(f):
            labels = np.full(len(xyz), UNKNOWN, np.uint8); flags = np.zeros(len(xyz), np.uint8)
            transform = self.transforms[actor][f]
            if transform is None: return labels.reshape(shape), flags.reshape(shape)
            membership = None
            if actor >= 0:
                membership = np.zeros(int(np.prod(self.shape)), bool)
                membership[self.members[actor][f]] = True
            # Bound working memory for six concurrent frames, independent of
            # component/grid size. Mapping arrays themselves are uint8.
            for start in range(0, len(xyz), 65536):
                stop = min(start+65536, len(xyz))
                pts = transform_points(xyz[start:stop], transform)
                ijk = np.floor((pts-self.origin)/self.step).astype(np.int64)
                valid = ((ijk >= 0)&(ijk < self.shape)).all(1)
                at = tuple(ijk[valid].T)
                labels[start:stop][valid] = self.history[f][at]
                bits = self.observed[f][at].astype(np.uint8)
                if membership is not None:
                    bits |= membership[np.ravel_multi_index(ijk[valid].T, tuple(self.shape))].astype(np.uint8)*2
                flags[start:stop][valid] = bits
            return labels.reshape(shape), flags.reshape(shape)
        active = [f for f in range(6) if self.transforms[actor][f] is not None]
        # Single static map shared by generation + static refine, six frames
        # parallelized only here; no nested pools and no CUDA work in threads.
        frames = [None]*6
        for f, mapped in zip(active, pool.map(frame, active)): frames[f] = mapped
        for f in range(6):
            if frames[f] is None: frames[f] = (np.full(shape, UNKNOWN, np.uint8), np.zeros(shape, np.uint8))
        return lo, np.stack([v[0] for v in frames]), np.stack([v[1] for v in frames])

    def _sparse(self, plan, actor):
        """Same reference arithmetic, reusing inverses for uncached/small actors."""
        n, p, z = len(plan), self.p, self.z
        hist = np.full((n, 6, p, p, z), UNKNOWN, np.uint8); flags = np.zeros_like(hist)
        offsets = np.stack(np.meshgrid(np.arange(p)-p//2, np.arange(p)-p//2, np.arange(z), indexing='ij'), -1)
        idx = offsets[None]+np.pad(plan.evidence_xy, ((0, 0), (0, 1)))[:, None, None, None, :]
        xyz = (self.origin+(idx+.5)*self.step).reshape(-1, 3)
        inherited = np.repeat(plan.classes, p*p*z) if actor < 0 else None
        for f, transform in enumerate(self.transforms[actor]):
            if transform is None: continue
            ijk = np.floor((transform_points(xyz, transform)-self.origin)/self.step).astype(np.int64)
            valid = ((ijk >= 0)&(ijk < self.shape)).all(1); at = tuple(ijk[valid].T)
            labels = np.full(len(ijk), UNKNOWN, np.uint8); bits = np.zeros(len(ijk), np.uint8)
            labels[valid] = self.history[f][at]; bits[valid] = self.observed[f][at].astype(np.uint8)
            owned = (labels[valid] == inherited[valid]) if actor < 0 else np.isin(
                np.ravel_multi_index(ijk[valid].T, tuple(self.shape)), self.members[actor][f])
            bits[valid] |= owned.astype(np.uint8)*2
            hist[:, f] = labels.reshape(n, p, p, z); flags[:, f] = bits.reshape(n, p, p, z)
        return hist, flags

    def sample(self, plan, reference_sampler):
        n, p, z = len(plan), self.p, self.z
        hist = np.full((n, 6, p, p, z), UNKNOWN, np.uint8); flags = np.zeros_like(hist)
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
