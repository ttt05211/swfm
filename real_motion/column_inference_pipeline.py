"""Full-population streaming byte features. No GT, query cap or model cache."""
from contextlib import nullcontext
import numpy as np
import torch

from .column_gpu_sampling import GpuColumnSampler, PackedColumnWindow, pack_column_window
from .causal_column_sampling import ColumnFeatureSampler, ColumnHistoryIndex


def inference_tensors(arrays, legal, device, *, packed=False):
    """One host-byte upload instead of one synchronous copy per field.

    Original dtypes/shapes and network query batch are preserved. Already
    CUDA-resident byte features are not round-tripped. No FP computation.
    """
    values = dict(arrays, legal=legal)
    if not packed: return {k:torch.as_tensor(v,device=device) for k,v in values.items()}
    result = {}; pieces = []; descriptors = []; offset = 0
    for key,value in values.items():
        tensor = torch.as_tensor(value)
        if tensor.device.type != 'cpu' or tensor.numel() == 0:
            result[key] = tensor.to(device);continue
        tensor = tensor.contiguous()
        alignment = tensor.element_size();padding = (-offset)%alignment
        if padding:pieces.append(np.zeros(padding,np.uint8));offset+=padding
        raw = tensor.reshape(-1).view(torch.uint8).numpy()
        descriptors.append((key,offset,len(raw),tensor.dtype,tuple(tensor.shape)))
        pieces.append(raw);offset+=len(raw)
    if descriptors:
        block = torch.from_numpy(np.concatenate(pieces)).to(device)
        for key,start,size,dtype,shape in descriptors:
            result[key]=block[start:start+size].view(dtype).reshape(shape)
    return result


class ProbabilityReadback:
    """Bounded output-only buffering, not a different network batch/precision.

    CUDA chunks retain only final FP32 probabilities. One contiguous readback
    replaces many tiny synchronizations. No tensor arithmetic is changed.
    """
    def __init__(self, *, buffered=False, max_bytes=8*2**20):
        if type(max_bytes) is not int or max_bytes < 1: raise ValueError('positive readback byte budget required')
        self.buffered, self.limit = bool(buffered), max_bytes
        self.pending, self.parts, self.bytes = [], [], 0
        self.transfers = self.peak_bytes = 0

    def append(self, probability):
        size = probability.numel()*probability.element_size()
        if not self.buffered:
            self.parts.append(probability.cpu().numpy()); self.transfers += 1
            return
        if self.bytes+size > self.limit: self.flush()
        if size > self.limit:
            self.parts.append(probability.cpu().numpy()); self.transfers += 1
            return
        self.pending.append(probability); self.bytes += size
        self.peak_bytes = max(self.peak_bytes, self.bytes)

    def flush(self):
        if self.pending:
            # cat is a byte copy, not a reduction. Temporarily needs at most
            # another limit bytes. Concatenation never changes network shape.
            value = self.pending[0] if len(self.pending) == 1 else torch.cat(self.pending)
            self.parts.append(value.cpu().numpy()); self.transfers += 1
            self.pending.clear(); self.bytes = 0

    def result(self, empty_shape):
        self.flush()
        return np.concatenate(self.parts) if self.parts else np.empty(empty_shape, np.float32)


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
