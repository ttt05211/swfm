"""Optional byte-exact historical patch gather; never a learned GPU warp.

CPU computes the ORIGINAL float64 inverse matrices. CUDA batches coordinate
mapping and byte/membership gathers only. Near a floor boundary the WHOLE
original horizon is replayed on CPU (not a differently sized BLAS subset).
No model, RNG, future labels or persistent learned geometry live here.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np
import torch

from .causal_column_completion import UNKNOWN
from .causal_column_sampling import ColumnFeatureSampler, ColumnHistoryIndex


@dataclass
class PackedColumnWindow:
    prepared_identity: int
    plan_identities: tuple
    transforms: np.ndarray
    valid_frames: np.ndarray
    evidence_xy: np.ndarray
    classes: np.ndarray
    dynamic: np.ndarray
    slots: np.ndarray
    membership: np.ndarray
    sizes: tuple
    history_index: ColumnHistoryIndex


def pack_column_window(prepared, selected, grid, config, motion_factory, *, history_index=None):
    """Pure CPU setup suitable for workers; preserves matrix multiplication order."""
    actors = {int(a) for _, plan, _, _ in selected for a in np.unique(plan.actor) if a >= 0}
    index = history_index or ColumnHistoryIndex(prepared, grid, actors=actors)
    frames = len(index.inverse_history)
    if frames not in (4, 6): raise ValueError('four or six history frames required')
    if config.z_bins != grid.shape_hwd[2]: raise ValueError('column/grid Z mismatch')
    cells = int(np.prod(grid.shape_hwd))
    actor_slots = {a: i for i, a in enumerate(sorted(actors))}
    if (len(actor_slots)+1)*frames*cells >= np.iinfo(np.int64).max:
        raise ValueError('membership key exceeds int64')
    membership = []
    for actor, slot in actor_slots.items():
        for f, owned in enumerate(index.members[actor]):
            if owned is not None and len(owned): membership.append((slot*frames+f)*cells+owned)
    matrices = []; valid = []; xy = []; classes = []; dynamic = []; slots = []; sizes = []
    for h, plan, _, _ in selected:
        n = len(plan); sizes.append(n)
        mat = np.zeros((n, frames, 4, 4), np.float64); ok = np.zeros((n, frames), bool)
        groups = np.where(plan.actor < 0, -1, plan.actor)
        query_slots = np.zeros(n, np.int64)
        for a in np.unique(groups):
            a = int(a); take = groups == a
            inverse_motion = None
            if a >= 0:
                inverse_motion = np.linalg.inv(motion_factory(prepared.state['current'][a]['centroid_world'],
                    prepared.targets[h][a], prepared.yaws[h][a]))
                query_slots[take] = actor_slots[a]
            for f, inverse_history in enumerate(index.inverse_history):
                if a >= 0:
                    inverse_reg = index.inverse_registration[a][f]
                    if inverse_reg is None: continue
                    transform = inverse_history@inverse_reg@inverse_motion@prepared.raw['future_poses'][h]
                else: transform = inverse_history@prepared.raw['future_poses'][h]
                mat[take, f] = transform; ok[take, f] = True
        matrices.append(mat); valid.append(ok); xy.append(plan.evidence_xy)
        classes.append(plan.classes); dynamic.append(groups >= 0); slots.append(query_slots)
    def cat(items, shape, dtype):
        return np.ascontiguousarray(np.concatenate(items), dtype=dtype) if items else np.empty(shape, dtype)
    return PackedColumnWindow(id(prepared), tuple((h,id(plan)) for h,plan,_,_ in selected),
        cat(matrices, (0, frames, 4, 4), np.float64),
        cat(valid, (0, frames), bool), cat(xy, (0, 2), np.int64), cat(classes, (0,), np.uint8),
        cat(dynamic, (0,), bool), cat(slots, (0,), np.int64),
        cat(membership, (0,), np.int64), tuple(sizes), index)


class GpuColumnSampler:
    """Bounded main-thread CUDA work, optional CPU emulation ONLY for tests.

    Fail-closed CPU verification on the first three windows and every 128th.
    The boundary guard remains enabled on EVERY window. An OOM is scoped to
    this temporary sampler and falls back; other errors propagate before step.
    """
    def __init__(self, device, *, chunk_queries=32, verify_first=3, verify_every=128,
                 allow_cpu=False, max_working_mib=256):
        self.device = torch.device(device)
        if self.device.type != 'cuda' and not allow_cpu: raise ValueError('GPU sampler requires CUDA')
        if self.device.type == 'cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
        if chunk_queries < 1 or max_working_mib < 1 or verify_first < 1 or verify_every < 1:
            raise ValueError('positive chunk/budget and mandatory verification required')
        self.chunk_queries = int(chunk_queries); self.max_working_mib = float(max_working_mib)
        self.verify_first = int(verify_first); self.verify_every = int(verify_every); self.windows = 0

    @staticmethod
    def _cpu(prepared, row, grid, config, motion_factory, index):
        h, plan, labels, weight = row
        values = ColumnFeatureSampler(prepared, h, plan, grid, config, motion_factory,
            workers=1, history_index=index).sample(plan, None)
        return {**values, 'legal': plan.legal, 'target': labels, 'weight': weight}

    @torch.no_grad()
    def _gather(self, prepared, packed, grid, config):
        """FP64 elementary ops: no TF32, interpolation, FMA or ownership collapse."""
        device = self.device; frames = len(prepared.raw['history_occ'])
        shape = tuple(grid.shape_hwd); cells = int(np.prod(shape)); p, z = config.patch, config.z_bins
        history = torch.as_tensor(np.ascontiguousarray(prepared.raw['history_occ']), device=device)
        observed = torch.as_tensor(np.ascontiguousarray(prepared.raw['history_observed'], dtype=np.uint8), device=device)
        history = history.reshape(-1); observed = observed.reshape(-1)
        member = torch.as_tensor(packed.membership, device=device)
        matrices = torch.as_tensor(packed.transforms, device=device)
        valid_frames = torch.as_tensor(packed.valid_frames, device=device)
        centres = torch.as_tensor(packed.evidence_xy, device=device)
        dynamic = torch.as_tensor(packed.dynamic, device=device)
        slots = torch.as_tensor(packed.slots, device=device)
        classes = torch.as_tensor(packed.classes, device=device)
        origin = torch.tensor((grid.x_min, grid.y_min, grid.z_min), dtype=torch.float64, device=device)
        step = torch.tensor(grid.voxel_size, dtype=torch.float64, device=device)
        limits = torch.tensor(shape, device=device)
        frame = torch.arange(frames, device=device)
        dx, dy, zz = torch.meshgrid(torch.arange(p, device=device)-p//2,
            torch.arange(p, device=device)-p//2, torch.arange(z, device=device), indexing='ij')
        offset = torch.stack((dx, dy, zz), -1).reshape(1, -1, 3)
        outputs = []; bits = []; guards = []
        for start in range(0, len(centres), self.chunk_queries):
            stop = start+self.chunk_queries
            centre = torch.nn.functional.pad(centres[start:stop], (0, 1))
            xyz = origin+((centre[:, None]+offset).to(torch.float64)+.5)*step
            t = matrices[start:stop, :, :3]; points = xyz[:, None, :, :]
            # Keep each multiply/add separate. A conservative rounding bound
            # covers both this reduction and NumPy BLAS/FMA reduction order.
            terms = [points[..., a, None]*t[:, :, None, :, a] for a in range(3)]
            mapped = ((terms[0]+terms[1])+terms[2])+t[:, :, None, :, 3]
            uvw = (mapped-origin)/step
            magnitude = (sum(v.abs() for v in terms)+t[:, :, None, :, 3].abs()+origin.abs())/step.abs()
            error = 128*np.finfo(np.float64).eps*(magnitude+uvw.abs()+1)
            enabled = valid_frames[start:stop, :, None]
            uncertain = ((uvw-uvw.round()).abs() <= error).any(-1) | ~torch.isfinite(uvw).all(-1)
            guards.append((uncertain & enabled).any(dim=(1, 2)))
            indices = torch.floor(uvw).to(torch.int64)
            inside = ((indices >= 0)&(indices < limits)).all(-1)&enabled
            flat = (indices[..., 0]*shape[1]+indices[..., 1])*shape[2]+indices[..., 2]
            safe_flat = flat.clamp(0, cells-1)
            address = frame[None, :, None]*cells+safe_flat
            values = history[address]; flag = observed[address]
            owned = torch.zeros_like(inside)
            if len(member):
                key = (slots[start:stop, None, None]*frames+frame[None, :, None])*cells+safe_flat
                at = torch.searchsorted(member, key)
                owned = (at < len(member))&(member[at.clamp(max=len(member)-1)] == key)
            owned = torch.where(dynamic[start:stop, None, None], owned,
                                values == classes[start:stop, None, None])
            values = torch.where(inside, values, UNKNOWN)
            flag = torch.where(inside, flag | (owned.to(torch.uint8)*2), 0)
            outputs.append(values.reshape(-1, frames, p, p, z)); bits.append(flag.reshape(-1, frames, p, p, z))
        n = len(centres)
        empty = lambda: torch.empty((0, frames, p, p, z), dtype=torch.uint8, device=device)
        return (torch.cat(outputs) if n else empty(), torch.cat(bits) if n else empty(),
                torch.cat(guards).cpu().numpy() if n else np.zeros(0, bool))

    def sample(self, prepared, selected, grid, config, motion_factory, *, packed=None):
        started = time.perf_counter()
        if not selected:
            # Empty windows must not consume the first-three verification
            # quota or upload a history volume for no requested columns.
            return [], {'gpu_feature_windows':0, 'gpu_feature_horizons':0,
                'gpu_feature_fallback_horizons':0, 'gpu_feature_boundary_horizons':0,
                'gpu_feature_verified_horizons':0, 'gpu_feature_oom_windows':0,
                'gpu_feature_budget_windows':0, 'gpu_feature_estimated_mib':0.,
                'gpu_feature_host_seconds':time.perf_counter()-started}
        self.windows += 1
        packed = packed or pack_column_window(prepared, selected, grid, config, motion_factory)
        if packed.prepared_identity != id(prepared) or packed.plan_identities != tuple((h,id(plan)) for h,plan,_,_ in selected):
            raise ValueError('stale GPU window/pose packing')
        if packed.sizes != tuple(len(row[1]) for row in selected): raise ValueError('GPU packing/query order mismatch')
        frames = len(prepared.raw['history_occ']); volume = int(np.prod(grid.shape_hwd))
        # Conservative transient scratch + outputs + immutable inputs estimate.
        bytes_needed = (2*frames*volume+packed.membership.nbytes+packed.transforms.nbytes+
            2*len(packed.classes)*frames*config.patch**2*config.z_bins+
            min(self.chunk_queries, len(packed.classes))*frames*config.patch**2*config.z_bins*256)
        reason = None; hist = flags = guard = None
        if bytes_needed > self.max_working_mib*2**20: reason = 'budget'
        else:
            try: hist, flags, guard = self._gather(prepared, packed, grid, config)
            except torch.cuda.OutOfMemoryError: reason = 'oom'
        # OOM traceback/workspace leaves _gather before allocating CPU fallback.
        arrays = []; cursor = 0; fallback = boundary = verified = 0
        verify = self.windows <= self.verify_first or self.windows % self.verify_every == 0
        for row, n in zip(selected, packed.sizes):
            plan = row[1]
            needs_cpu = reason is not None or bool(guard[cursor:cursor+n].any())
            if needs_cpu:
                fallback += 1; boundary += int(reason is None)
                values = self._cpu(prepared, row, grid, config, motion_factory, packed.history_index)
                values['history'] = torch.as_tensor(values['history'], device=self.device)
                values['flags'] = torch.as_tensor(values['flags'], device=self.device)
            else:
                values = {'history': hist[cursor:cursor+n], 'flags': flags[cursor:cursor+n],
                    'base': plan.base, 'fallback': plan.fallback, 'context': plan.context,
                    'kind': plan.kind, 'classes': plan.classes, 'legal': plan.legal, 'target': row[2], 'weight': row[3]}
                if verify:
                    reference = self._cpu(prepared, row, grid, config, motion_factory, packed.history_index)
                    for key in ('history', 'flags'):
                        if not np.array_equal(values[key].cpu().numpy(), reference[key]):
                            raise RuntimeError('GPU column '+key+' byte exactness failed; refusing optimizer update')
                    verified += 1
            arrays.append(values); cursor += n
        stats = {'gpu_feature_windows': 1, 'gpu_feature_horizons': len(selected),
            'gpu_feature_fallback_horizons': fallback, 'gpu_feature_boundary_horizons': boundary,
            'gpu_feature_verified_horizons': verified, 'gpu_feature_oom_windows': int(reason == 'oom'),
            'gpu_feature_budget_windows': int(reason == 'budget'), 'gpu_feature_estimated_mib': bytes_needed/2**20,
            'gpu_feature_host_seconds': time.perf_counter()-started}
        return arrays, stats
