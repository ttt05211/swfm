"""Opt-in frozen V18 execution trials; no new weights or changed geometry.

Sparse integer majority evaluates ALL unknown cells, not a candidate cap.
Threshold/tie decisions still replay scipy's frozen float32 arithmetic.
"""
from contextlib import contextmanager
from collections import OrderedDict
import numpy as np
from scipy.ndimage import uniform_filter
import torch

MOTION_KEYS = ('features', 'local_semantic_tube', 'kta_displacement_xy_m',
               'frame_motion_features', 'target_source_mask_tube')


class FrozenV18Graph:
    """Bounded shape-specific graph; input transfer and output clone are live.

CPU labels are validated on EVERY call. No cached predictions or optimizer
path; refuse changed weights, training mode or an enabled autograd context.
"""
    def __init__(self, model, device, *, entries=2, budget_mib=256):
        if entries < 1 or budget_mib <= 0 or torch.device(device).type != 'cuda':
            raise ValueError('bounded CUDA graph configuration required')
        self.model, self.device = model, torch.device(device)
        self.entries, self.budget = entries, budget_mib*2**20
        self.weights = tuple(p._version for p in model.parameters())
        self.cache = OrderedDict()
        self.capture_seconds = 0.
        self.replays = 0

    def __call__(self, record):
        # A graph launch must use the graph's CUDA device, even when the caller
        # has another current device. No global device choice is left behind.
        with torch.cuda.device(self.device):
            return self._call(record)

    def _call(self, record):
        import time
        if (self.model.training or torch.is_grad_enabled()
                or any(p.requires_grad for p in self.model.parameters())
                or tuple(p._version for p in self.model.parameters()) != self.weights):
            raise RuntimeError('frozen V18 graph cannot train or use changed weights')
        values = []
        for key in MOTION_KEYS:
            value = record[key]
            if isinstance(value, torch.Tensor):
                if value.device.type != 'cpu':
                    raise ValueError('explicit CPU input audit required')
                a = value.numpy()
            else:
                a = np.asarray(value)
            if key in ('local_semantic_tube', 'target_source_mask_tube'):
                retained = a[:, -self.model.config.history_frames:]
                limit = 18 if key == 'local_semantic_tube' else 2
                if retained.size and ((retained < 0).any() or (retained >= limit).any()):
                    raise ValueError('invalid historical semantic/source-mask label')
            t = torch.as_tensor(np.ascontiguousarray(a), device=self.device)
            values.append(t.float() if key in ('features', 'kta_displacement_xy_m', 'frame_motion_features') else t)
        key = tuple((tuple(t.shape), t.dtype) for t in values)
        if len(values[0]) == 0:
            with torch.autocast('cuda', dtype=torch.bfloat16), reuse_v18_projections(self.model):
                return self.model(*values, return_latents=True, validate_label_values=False)
        if key not in self.cache:
            while len(self.cache) >= self.entries:
                self.cache.popitem(last=False)
            started = time.perf_counter()
            buffers = [t.clone() for t in values]
            stream = torch.cuda.Stream(device=self.device)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(stream), torch.autocast('cuda', dtype=torch.bfloat16), reuse_v18_projections(self.model):
                for _ in range(3):
                    self.model(*buffers, return_latents=True, validate_label_values=False)
            torch.cuda.current_stream(self.device).wait_stream(stream)
            before = torch.cuda.memory_allocated(self.device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph), torch.autocast('cuda', dtype=torch.bfloat16), reuse_v18_projections(self.model):
                output = self.model(*buffers, return_latents=True, validate_label_values=False)
            allocated = max(0, torch.cuda.memory_allocated(self.device)-before)
            allocated += sum(t.numel()*t.element_size() for t in buffers)
            total = sum(v[3] for v in self.cache.values())+allocated
            if total > self.budget:
                raise RuntimeError('frozen motion graph exceeded its VRAM budget; no silent fallback')
            self.cache[key] = (buffers, graph, output, allocated)
            self.capture_seconds += time.perf_counter()-started
        self.cache.move_to_end(key)
        buffers, graph, output, _ = self.cache[key]
        for destination, value in zip(buffers, values):
            destination.copy_(value)
        graph.replay(); self.replays += 1
        # Callers may retain old forecasts; graph storage must never alias them.
        return {name: value.clone() for name, value in output.items()}

    def close(self):
        self.cache.clear()


@contextmanager
def reuse_v18_projections(model):
    modules = [(model, 'reuse_source_projection')]
    modules += [(block, 'reuse_context_norm') for block in model.decoder]
    previous = [(m, name, hasattr(m, name), getattr(m, name, None)) for m, name in modules]
    try:
        for module, name in modules:
            setattr(module, name, True)
        yield
    finally:
        for module, name, exists, value in previous:
            if exists:
                setattr(module, name, value)
            else:
                delattr(module, name)


def _scipy_edges(out, semantics, known, coordinates, classes, min_fraction):
    if not len(coordinates):
        return
    offsets = np.array([(x, y) for x in range(-2, 3) for y in range(-2, 3)])
    x = coordinates[:, 0, None] + offsets[None, :, 0]
    y = coordinates[:, 1, None] + offsets[None, :, 1]
    z = coordinates[:, 2, None]
    valid = (x >= 0) & (x < out.shape[0]) & (y >= 0) & (y < out.shape[1])
    x, y = x.clip(0, out.shape[0]-1), y.clip(0, out.shape[1]-1)
    seen = (valid & known[x, y, z]).reshape(-1, 5, 5)
    labels = semantics[x, y, z].reshape(-1, 5, 5)
    denominator = uniform_filter(seen.astype(np.float32), (1, 5, 5), mode='constant')[:, 2, 2]
    masks = ((labels[:, None] == classes[None, :, None, None]) & seen[:, None]).astype(np.float32)
    scores = uniform_filter(masks, (1, 1, 5, 5), mode='constant')[:, :, 2, 2]
    scores /= np.maximum(denominator[:, None], np.float32(1e-6))
    best = scores.argmax(1)
    active = scores[np.arange(len(best)), best] >= min_fraction
    rows = coordinates[active]
    out[tuple(rows.T)] = classes[best[active]]


def majority_fill_native_exact(semantics, unknown_mask, *, kernel=(5, 5, 1),
                               min_fraction=.3, device=None):
    from .native_column_cpu import get_prepared_native
    if tuple(kernel) != (5, 5, 1) or abs(float(min_fraction)-.3) > 1e-12:
        raise ValueError('native majority is frozen to kernel=(5,5,1), threshold=.3')
    native = get_prepared_native()
    sem = np.asarray(semantics)
    unknown = np.asarray(unknown_mask, bool)
    if sem.ndim != 3 or sem.shape != unknown.shape:
        raise ValueError('semantic/unknown 3D shape mismatch')
    output, ambiguous = native.v18_majority(sem, unknown)
    coordinates = np.argwhere(ambiguous)
    if len(coordinates):
        known = ~unknown
        classes = np.unique(sem[known]).astype(np.int64)
        _scipy_edges(output, sem, known, coordinates, classes, min_fraction)
    return output


@torch.no_grad()
def majority_fill_sparse_cuda_exact(semantics, unknown_mask, *, kernel=(5, 5, 1),
                                    min_fraction=.3, device, chunk=32768):
    from .runtime_fastpath import majority_fill_sparse_5x5x1
    if torch.device(device).type != 'cuda':
        return majority_fill_sparse_5x5x1(semantics, unknown_mask, kernel=kernel, min_fraction=min_fraction)
    sem = np.asarray(semantics)
    unknown = np.asarray(unknown_mask, bool)
    if sem.ndim != 3 or sem.shape != unknown.shape or tuple(kernel) != (5, 5, 1):
        raise ValueError('frozen 3D grid and 5x5x1 kernel required')
    if abs(float(min_fraction)-.3) > 1e-12 or chunk < 1:
        raise ValueError('invalid frozen majority configuration')
    coordinates = np.argwhere(unknown)
    out = sem.copy()
    if not len(coordinates):
        return out
    known = ~unknown
    classes = np.unique(sem[known]).astype(np.int64)
    if not len(classes):
        return out
    if classes.min() < 0:
        raise ValueError('negative known label')
    # Ascending actual known classes retain the reference's tie order. Bound
    # temporary histogram/gather memory independently of the unknown band size.
    dev = torch.device(device)
    source = torch.as_tensor(np.ascontiguousarray(sem), device=dev)
    visible = torch.as_tensor(np.ascontiguousarray(known), device=dev)
    cls = torch.as_tensor(classes, device=dev)
    offsets = torch.tensor([(x, y) for x in range(-2, 3) for y in range(-2, 3)], device=dev)
    results = []
    for begin in range(0, len(coordinates), chunk):
        query = torch.as_tensor(coordinates[begin:begin+chunk], device=dev)
        x = query[:, 0, None] + offsets[None, :, 0]
        y = query[:, 1, None] + offsets[None, :, 1]
        valid = (x >= 0) & (x < sem.shape[0]) & (y >= 0) & (y < sem.shape[1])
        x, y = x.clamp(0, sem.shape[0]-1), y.clamp(0, sem.shape[1]-1)
        z = query[:, 2, None]
        seen = valid & visible[x, y, z]
        labels = source[x, y, z].long()
        at = torch.searchsorted(cls, labels)
        # Unknown labels need not occur among known classes; zero their weight.
        at = at.clamp_max(len(classes)-1)
        counts = torch.zeros((len(query), len(classes)), device=dev, dtype=torch.int32)
        counts.scatter_add_(1, at, seen.to(torch.int32))
        maximum, best = counts.max(1)
        ties = (counts == maximum[:, None]).sum(1)
        denominator = seen.sum(1)
        lhs, rhs = 10*maximum, 3*denominator
        fill = (lhs > rhs) & (ties == 1)
        ambiguous = ((denominator > 0) & (lhs == rhs)) | ((lhs > rhs) & (ties > 1))
        # One compact synchronization per chunk, not three whole-grid readbacks.
        results.append(torch.stack((cls[best], fill.long(), ambiguous.long()), -1).cpu().numpy())
    result = np.concatenate(results)
    rows = coordinates[result[:, 1].astype(bool)]
    out[tuple(rows.T)] = result[result[:, 1].astype(bool), 0]
    _scipy_edges(out, sem, known, coordinates[result[:, 2].astype(bool)], classes, min_fraction)
    return out
