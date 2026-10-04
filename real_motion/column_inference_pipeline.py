"""Full-population streaming byte features. No GT, query cap or model cache."""
from contextlib import nullcontext
import numpy as np
import torch

from .column_gpu_sampling import GpuColumnSampler, PackedColumnWindow, pack_column_window
from .causal_column_sampling import ColumnFeatureSampler, ColumnHistoryIndex


class InferenceFeatures:
    def __init__(self, prepared, h, plan, grid, config, device, motion_factory, *,
                 backend='cpu', workers=1, history_index=None, gpu_sampler=None, verify=False):
        if backend not in ('cpu', 'gpu'): raise ValueError('invalid inference feature backend')
        self.prepared, self.h, self.plan, self.grid, self.config = prepared, h, plan, grid, config
        self.device, self.motion_factory, self.workers = torch.device(device), motion_factory, workers
        self.index = history_index or getattr(prepared,'column_history_index',None) or ColumnHistoryIndex(
            prepared, grid, actors=np.unique(plan.actor))
        self.gpu = gpu_sampler if backend == 'gpu' else None
        self.cpu = None
        self.template = self.template_actors = self.gpu_membership = None
        self.verify = bool(verify); self.verified = False
        self.audit = dict(gpu_chunks=0, cpu_chunks=0, boundary_chunks=0, budget_chunks=0,
                          oom_chunks=0, verified_chunks=0, unique_patch_queries=0, total_queries=0)
        if self.gpu is None: self._cpu()

    def _cpu(self):
        if self.cpu is None:
            # Build from the WHOLE original horizon. A boundary fallback must
            # preserve NumPy BLAS/map shapes, not build a differently sized ROI.
            self.cpu = ColumnFeatureSampler(self.prepared, self.h, self.plan, self.grid, self.config,
                self.motion_factory, workers=self.workers, history_index=self.index)
        return self.cpu

    def sample(self, small, reference_sampler):
        self.audit['total_queries'] += len(small)
        if self.gpu is None:
            self.audit['cpu_chunks'] += 1
            return self._cpu().sample(small, reference_sampler)
        # Static/frontier queries may read identical anchors. Only BYTE gathers
        # are deduplicated: every distinct query still runs the original network.
        actor = np.where(small.actor < 0, -1, small.actor)
        keys = np.column_stack((actor, small.evidence_xy, small.classes))
        _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
        unique = small.subset(first)
        self.audit['unique_patch_queries'] += len(unique)
        packed = self._pack(unique)
        frames = len(self.prepared.raw['history_occ']); cells = int(np.prod(self.grid.shape_hwd))
        estimate = (2*frames*cells+packed.membership.nbytes+packed.transforms.nbytes+
            2*(len(unique)+len(small))*frames*self.config.patch**2*self.config.z_bins+
            min(self.gpu.chunk_queries,len(unique))*frames*self.config.patch**2*self.config.z_bins*256)
        reason = 'budget' if estimate > self.gpu.max_working_mib*2**20 else None
        if reason is None:
            try:
                if self.gpu_membership is None:
                    self.gpu_membership = torch.as_tensor(packed.membership, device=self.device)
                packed.membership = self.gpu_membership
                history, flags, guard = self.gpu._gather(self.prepared, packed, self.grid, self.config)
                if guard.any(): reason = 'boundary'
            except torch.cuda.OutOfMemoryError: reason = 'oom'
        if reason is not None:
            self.audit[reason+'_chunks'] += 1; self.audit['cpu_chunks'] += 1
            if reason == 'oom':
                self.gpu_membership = None;self.gpu = None  # don't repeat an OOM for every remaining chunk
            return self._cpu().sample(small, reference_sampler)
        ids = torch.as_tensor(inverse, device=self.device)
        values = dict(history=history.index_select(0, ids), flags=flags.index_select(0, ids),
            base=small.base, fallback=small.fallback, context=small.context, kind=small.kind, classes=small.classes)
        if self.verify and not self.verified:
            reference = self._cpu().sample(small, reference_sampler)
            for key in ('history', 'flags'):
                if not np.array_equal(values[key].cpu().numpy(), reference[key]):
                    raise RuntimeError('inference GPU '+key+' byte exactness failed')
            self.verified = True; self.audit['verified_chunks'] += 1
        self.audit['gpu_chunks'] += 1
        return values

    def _pack(self, small):
        """One CURRENT horizon's actor transforms/membership, not per chunk.

        Same original CPU FP64 matrix products. The small table is local to
        this prediction call and discarded before the next learned update.
        """
        if self.template is None:
            actors = np.where(self.plan.actor < 0, -1, self.plan.actor)
            self.template_actors, first = np.unique(actors, return_index=True)
            representatives = self.plan.subset(first)
            self.template = pack_column_window(self.prepared, [(self.h,representatives,None,None)],
                self.grid,self.config,self.motion_factory,history_index=self.index)
        table = self.template
        actors = np.where(small.actor < 0, -1, small.actor)
        at = np.searchsorted(self.template_actors, actors)
        if (np.any(at >= len(self.template_actors)) or
                not np.array_equal(self.template_actors[at], actors)):
            raise ValueError('stream query actor outside original full population')
        return PackedColumnWindow(id(self.prepared), ((self.h,id(small)),), table.transforms[at],
            table.valid_frames[at], small.evidence_xy, small.classes, actors >= 0, table.slots[at],
            table.membership, (len(small),), self.index)

    @property
    def cache_bytes(self):
        return self.cpu.cache_bytes if self.cpu is not None else 0


def inference_gpu(device, backend, prepared, grid, config):
    device = torch.device(device)
    if backend == 'gpu' and device.type == 'cuda':
        # Larger BYTE gather tiles reduce launches; network batch256 is kept.
        # Conservative workspace guard/OOM fallback still applies unchanged.
        sampler = GpuColumnSampler(device, chunk_queries=128)
        return sampler, sampler.resident_window(prepared, grid, config)
    return None, nullcontext(False)
