"""Opt-in scheduling-only Strong inverse warp; all geometry arithmetic unchanged.

Only a live six-horizon invocation owns these buffers. No poses, semantic
predictions, learned features or previous-window outputs are cached.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import numpy as np
import torch

_BACKEND = ContextVar('swfm_strong_warp_backend', default='reference')


def selected_backend():
    return _BACKEND.get()


@contextmanager
def strong_warp_execution(backend):
    if backend not in ('reference', 'buffered'):
        raise ValueError('unknown Strong warp execution backend')
    token = _BACKEND.set(backend)
    try:
        yield
    finally:
        _BACKEND.reset(token)


@torch.no_grad()
def inverse_warp_sequence_buffered_exact(semantics, src_to_dst_seq, *, grid,
                                         free_label, device, boundary_tol_vox=5e-3):
    from .runtime_fastpath import _torch_voxel_centers_cached
    from .strong_w2det import _voxel_centers
    if torch.device(device).type != 'cuda':
        raise ValueError('CUDA exact-corrected inverse warp requires a CUDA device')
    sem = np.asarray(semantics)
    if tuple(sem.shape) != tuple(grid.shape_hwd):
        raise ValueError('semantic grid shape mismatch')
    shape = tuple(int(v) for v in grid.shape_hwd)
    origin = np.asarray([grid.x_min, grid.y_min, grid.z_min], np.float64)
    step = np.asarray(grid.voxel_size, np.float64)
    # Keep the same per-horizon GEMM dimensions/order, FP32 expressions, FP64
    # NumPy correction, tolerance and floor rules as the original implementation.
    centers32 = _torch_voxel_centers_cached(shape, tuple(origin), tuple(step), str(torch.device(device)))
    centers64 = _voxel_centers(grid)
    origin32 = torch.tensor(origin, dtype=torch.float32, device=device)
    step32 = torch.tensor(step, dtype=torch.float32, device=device)
    source = torch.from_numpy(sem.reshape(-1)).to(device=device)
    volume = len(centers64); X, Y, Z = shape; result = []
    poses = list(src_to_dst_seq)
    # Bound live index/correction buffers to six grids even for other callers.
    for begin in range(0, len(poses), 6):
        inverses, indices, uncertain = [], [], []
        for transform in poses[begin:begin+6]:
            inverse = np.linalg.inv(np.asarray(transform, np.float64)); inverses.append(inverse)
            matrix = torch.from_numpy(inverse[:3, :3].astype(np.float32)).to(device)
            translation = torch.from_numpy(inverse[:3, 3].astype(np.float32)).to(device)
            source_points = centers32 @ matrix.T + translation
            coordinate = (source_points - origin32) / step32
            indices.append(torch.floor(coordinate).to(torch.int64))
            uncertain.append(torch.any(torch.abs(coordinate-torch.round(coordinate)) <= float(boundary_tol_vox), dim=1))
        # One compact boundary readback instead of any/nonzero/readback at
        # EACH horizon. Flattened nonzero retains the original ascending order.
        positions_t = torch.nonzero(torch.stack(uncertain).reshape(-1), as_tuple=False).flatten()
        positions = positions_t.cpu().numpy()
        outputs, knowns = [], []
        for hi, (inverse, index) in enumerate(zip(inverses, indices)):
            lo, end = np.searchsorted(positions, [hi*volume, (hi+1)*volume])
            local = positions[lo:end]-hi*volume
            if len(local):
                reference_points = centers64[local] @ inverse[:3, :3].T + inverse[:3, 3]
                reference_index = np.floor((reference_points-origin[None])/step[None]).astype(np.int64)
                local_t = positions_t[int(lo):int(end)]-hi*volume
                index[local_t] = torch.from_numpy(reference_index).to(device=device)
            known = ((index[:, 0] >= 0) & (index[:, 0] < X)
                     & (index[:, 1] >= 0) & (index[:, 1] < Y)
                     & (index[:, 2] >= 0) & (index[:, 2] < Z))
            at = (index[:, 0]*Y + index[:, 1])*Z + index[:, 2]
            # No bool(any(known)) or variable-size CUDA gather/nonzero sync.
            # Invalid rows get the SAME free label; clamping only protects the
            # otherwise unused gather, never changes valid source coordinates.
            labels = source[at.clamp(0, volume-1)]
            outputs.append(torch.where(known, labels, torch.full_like(labels, int(free_label))))
            knowns.append(known)
        if sem.dtype == np.uint8:
            packed = torch.stack((torch.stack(outputs), torch.stack(knowns).to(torch.uint8)), dim=1).cpu().numpy()
            result.extend((row[0].reshape(shape), row[1].view(np.bool_).reshape(shape)) for row in packed)
        else:
            labels_cpu = torch.stack(outputs).cpu().numpy()
            known_cpu = torch.stack(knowns).cpu().numpy()
            result.extend((labels.reshape(shape), known.reshape(shape)) for labels, known in zip(labels_cpu, known_cpu))
    return result
