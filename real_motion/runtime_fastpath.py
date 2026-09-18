"""Bit-exact-oriented runtime fast paths for the frozen Strong/KTA + V18 pipeline.

These functions are engineering replacements for slow reference implementations.
They do not change the method contract.  Every fast path has a reference
equivalent and is intended to be gated by np.array_equal tests before paper
runtime numbers are accepted.

The main optimizations are:
- sparse 5x5x1 majority fill: evaluate only unknown voxels instead of running
  one full-grid scipy uniform_filter per semantic class;
- cropped connected components: run scipy label only inside the bounding box of
  each present dynamic class;
- direct A1 CLEAR mask construction: one dense boolean mask per horizon instead
  of allocating one 200x200x16 mask per source component.
"""
from __future__ import annotations

from typing import Iterable, Sequence
from functools import lru_cache

import numpy as np
from scipy.ndimage import generate_binary_structure, label, uniform_filter
import torch

from .metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS
from .rigid_transport import RasterizedRigidComponent
from .strong_w2det import StrongW2DetConfig, _transform_points, _voxel_centers


def majority_fill_sparse_5x5x1(
    semantics: np.ndarray,
    unknown_mask: np.ndarray,
    *,
    kernel: tuple[int, int, int] = (5, 5, 1),
    min_fraction: float = 0.3,
) -> np.ndarray:
    """Sparse bit-exact majority fill for the frozen 5x5x1 contract.

    The fast path first computes *integer* 5x5 neighborhood histograms only at
    unknown voxels.  For almost every queried voxel this is sufficient:
      - a unique integer winner safely above 0.3 is filled;
      - an integer winner safely below 0.3 is not filled.

    scipy.uniform_filter(float32) can differ from exact rational arithmetic only
    at threshold/tie edge cases because of separable float32 rounding.  We
    therefore replay scipy's exact filter path *only* for ambiguous queried
    voxels (ratio exactly 0.3, or a max-count tie above threshold), using compact
    5x5 patches.  This preserves the reference output bit-for-bit while avoiding
    full-volume per-class filtering.
    """
    if tuple(int(x) for x in kernel) != (5, 5, 1):
        raise ValueError("runtime majority fill only supports frozen kernel=(5,5,1)")

    sem = np.asarray(semantics)
    unknown = np.asarray(unknown_mask, dtype=bool)
    if sem.shape != unknown.shape:
        raise ValueError("semantics/unknown shape mismatch")
    if not bool(unknown.any()):
        return sem.copy()
    if sem.ndim != 3:
        raise ValueError("semantic occupancy must be 3-D")
    if abs(float(min_fraction) - 0.3) > 1e-12:
        raise ValueError("runtime sparse exact path is frozen to min_fraction=0.3")

    coords = np.argwhere(unknown)
    m = int(coords.shape[0])
    X, Y, Z = [int(x) for x in sem.shape]
    offsets = np.asarray(
        [(dx, dy) for dx in range(-2, 3) for dy in range(-2, 3)],
        dtype=np.int64,
    )
    nx = coords[:, 0:1] + offsets[None, :, 0]
    ny = coords[:, 1:2] + offsets[None, :, 1]
    z = coords[:, 2:3]
    inb = (nx >= 0) & (nx < X) & (ny >= 0) & (ny < Y)

    gx = np.clip(nx, 0, X - 1)
    gy = np.clip(ny, 0, Y - 1)
    gz = np.broadcast_to(z, gx.shape)
    known = ~unknown
    valid_known = inb & known[gx, gy, gz]
    neigh_label = sem[gx, gy, gz].astype(np.int64, copy=False)

    if sem.size and int(sem.min()) < 0:
        raise ValueError("negative semantic label unsupported")
    nlabels = int(sem.max()) + 1 if sem.size else 1
    rows = np.broadcast_to(np.arange(m, dtype=np.int64)[:, None], gx.shape)
    rr = rows[valid_known]
    ll = neigh_label[valid_known]
    counts = np.bincount(
        rr * nlabels + ll,
        minlength=m * nlabels,
    ).reshape(m, nlabels)
    denom_count = valid_known.sum(axis=1).astype(np.int64, copy=False)

    max_count = counts.max(axis=1)
    best_label = counts.argmax(axis=1).astype(sem.dtype, copy=False)
    tie_count = (counts == max_count[:, None]).sum(axis=1)

    # Compare max_count / denom_count with 3/10 using exact integers.
    lhs = 10 * max_count
    rhs = 3 * denom_count
    fill = (lhs > rhs) & (tie_count == 1)

    # Only these cells can be affected by scipy float32 rounding/tie behavior.
    # denom_count==0 is never ambiguous in the frozen reference:
    # denom is clamped to 1e-6, every class score is exactly zero, and the
    # voxel is not filled.  Excluding those cells avoids expensive scipy replay
    # over large fully-unknown boundary bands.
    ambiguous = (
        ((denom_count > 0) & (lhs == rhs))
        | ((lhs > rhs) & (tie_count > 1))
    )
    if bool(ambiguous.any()):
        ai = np.flatnonzero(ambiguous)
        # Each queried voxel already has its exact 5x5 neighborhood gathered.
        # Keeping query and class axes at filter-size 1 makes every patch
        # independent while reproducing the frozen 5x5 separable float path.
        patch_known = valid_known[ai].reshape(len(ai), 5, 5)
        patch_label = neigh_label[ai].reshape(len(ai), 5, 5)
        denom = uniform_filter(
            patch_known.astype(np.float32),
            size=(1, 5, 5),
            mode="constant",
        )[:, 2, 2]
        denom = np.maximum(denom, np.float32(1e-6))

        classes = np.unique(sem[known]).astype(np.int64, copy=False)
        masks = (
            (patch_label[:, None, :, :] == classes[None, :, None, None])
            & patch_known[:, None, :, :]
        ).astype(np.float32)
        scores = uniform_filter(
            masks,
            size=(1, 1, 5, 5),
            mode="constant",
        )[:, :, 2, 2]
        scores = scores / denom[:, None]

        # np.unique is ascending and the reference updates only on strict '>'.
        # argmax's first-maximum rule therefore reproduces the exact tie order.
        best_idx = np.argmax(scores, axis=1)
        best_score = scores[np.arange(len(ai)), best_idx]
        best_label[ai] = classes[best_idx].astype(sem.dtype, copy=False)
        fill[ai] = best_score >= float(min_fraction)

    out = sem.copy()
    fc = coords[fill]
    out[fc[:, 0], fc[:, 1], fc[:, 2]] = best_label[fill]
    return out

def extract_instances_cropped_exact(
    semantics: np.ndarray,
    ego_to_world: np.ndarray,
    *,
    grid,
    cfg: StrongW2DetConfig,
) -> list[dict]:
    """Same-class connected components with class-local cropped labeling.

    Component ordering remains class-major then scipy component-id, matching the
    frozen reference.  Coordinates from np.argwhere are C-order and therefore
    preserve the reference centroid summation order after the crop offset is
    restored.
    """
    sem = np.asarray(semantics)
    if tuple(sem.shape) != tuple(grid.shape_hwd):
        raise ValueError("semantic grid shape mismatch")
    structure = generate_binary_structure(3, int(cfg.connectivity))
    origin = np.asarray([grid.x_min, grid.y_min, grid.z_min], dtype=np.float64)
    step = np.asarray(grid.voxel_size, dtype=np.float64)
    out: list[dict] = []

    dyn = np.isin(sem, np.asarray(DYNAMIC_CLASS_IDS, dtype=sem.dtype))
    all_idx = np.argwhere(dyn)
    if len(all_idx) == 0:
        return out
    all_labels = sem[all_idx[:, 0], all_idx[:, 1], all_idx[:, 2]]

    for cls in DYNAMIC_CLASS_IDS:
        cls_pts = all_idx[all_labels == int(cls)]
        if len(cls_pts) < int(cfg.min_component_voxels):
            continue
        lo = cls_pts.min(axis=0)
        hi = cls_pts.max(axis=0) + 1
        sl = tuple(slice(int(lo[d]), int(hi[d])) for d in range(3))
        crop = sem[sl] == int(cls)
        comp_map, n = label(crop, structure=structure)
        for comp_id in range(1, int(n) + 1):
            local = np.argwhere(comp_map == comp_id)
            if len(local) < int(cfg.min_component_voxels):
                continue
            idx = local.astype(np.int64, copy=False) + lo[None].astype(np.int64)
            pts_ego = origin + (idx.astype(np.float64) + 0.5) * step
            centroid_world = _transform_points(
                ego_to_world, pts_ego.mean(axis=0, keepdims=True)
            )[0]
            out.append(
                {
                    "class_id": int(cls),
                    "voxel_indices": idx,
                    "centroid_world": centroid_world,
                    "voxel_count": int(len(idx)),
                }
            )
    return out


def component_lists_equal(a: Sequence[dict], b: Sequence[dict]) -> bool:
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if int(x["class_id"]) != int(y["class_id"]):
            return False
        if int(x["voxel_count"]) != int(y["voxel_count"]):
            return False
        if not np.array_equal(
            np.asarray(x["voxel_indices"], dtype=np.int64),
            np.asarray(y["voxel_indices"], dtype=np.int64),
        ):
            return False
        # The downstream matching/KTA contract depends on exact centroid values.
        if not np.array_equal(
            np.asarray(x["centroid_world"], dtype=np.float64),
            np.asarray(y["centroid_world"], dtype=np.float64),
        ):
            return False
    return True


def compose_component_replacements_fast_exact(
    anchor_occ: np.ndarray,
    baseline_components: Iterable[RasterizedRigidComponent],
    replacement_components: Iterable[RasterizedRigidComponent],
    *,
    dynamic_class_ids: Sequence[int],
    free_label: int,
    grid,
    precomputed_clear_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Frozen A1 compositor without one dense mask allocation per component."""
    anchor = np.asarray(anchor_occ)
    if tuple(anchor.shape) != tuple(grid.shape_hwd):
        raise ValueError("anchor_occ shape differs from occupancy grid")
    out = anchor.copy()

    if precomputed_clear_mask is None:
        clear = np.zeros(grid.shape_hwd, dtype=bool)
        for comp in baseline_components:
            idx = np.asarray(comp.voxel_indices, dtype=np.int64)
            if len(idx):
                clear[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    else:
        clear = np.asarray(precomputed_clear_mask, dtype=bool)
        if clear.shape != anchor.shape:
            raise ValueError("precomputed_clear_mask shape mismatch")

    lut = np.zeros(256, dtype=bool)
    for cid in dynamic_class_ids:
        lut[int(cid)] = True
    out[clear & lut[out.astype(np.uint8, copy=False)]] = int(free_label)

    # Preserve exact frozen source/input write order.
    for comp in replacement_components:
        idx = np.asarray(comp.voxel_indices, dtype=np.int64)
        if len(idx):
            out[idx[:, 0], idx[:, 1], idx[:, 2]] = int(comp.class_id)
    return out


def baseline_clear_mask(
    baseline_components: Iterable[RasterizedRigidComponent],
    *,
    grid,
) -> np.ndarray:
    clear = np.zeros(grid.shape_hwd, dtype=bool)
    for comp in baseline_components:
        idx = np.asarray(comp.voxel_indices, dtype=np.int64)
        if len(idx):
            clear[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return clear


@lru_cache(maxsize=8)
def _torch_voxel_centers_cached(
    shape: tuple[int, int, int],
    origin: tuple[float, float, float],
    step: tuple[float, float, float],
    device_key: str,
) -> torch.Tensor:
    # Construct through the frozen NumPy helper so coordinates originate from
    # exactly the same voxel-center convention as the reference inverse_warp.
    class _GridProxy:
        shape_hwd = shape
        x_min, y_min, z_min = origin
        voxel_size = step
    pts = _voxel_centers(_GridProxy())
    return torch.from_numpy(pts.astype(np.float32, copy=False)).to(device_key)


def inverse_warp_sequence_cuda_exact(
    semantics: np.ndarray,
    src_to_dst_seq: Sequence[np.ndarray],
    *,
    grid,
    free_label: int,
    device,
    boundary_tol_vox: float = 5e-3,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Accelerated six-horizon inverse warp with exact boundary correction.

    The dense affine transform is evaluated in float32 on CUDA.  A voxel index
    can differ from the frozen float64 NumPy reference only when a transformed
    coordinate lies close to an integer voxel boundary.  We conservatively
    detect those points in normalized voxel coordinates and recompute *only*
    them with the exact reference float64 expression before any gather.

    Formal use is always guarded by whole-grid np.array_equal comparison against
    the frozen CPU Strong/KTA implementation on multiple real validation
    windows.  CPU devices intentionally fall back to the reference caller.
    """
    if torch.device(device).type != "cuda":
        raise ValueError("CUDA exact-corrected inverse warp requires a CUDA device")

    sem = np.asarray(semantics)
    if tuple(sem.shape) != tuple(grid.shape_hwd):
        raise ValueError("semantic grid shape mismatch")

    shape = tuple(int(x) for x in grid.shape_hwd)
    origin_np = np.asarray([grid.x_min, grid.y_min, grid.z_min], dtype=np.float64)
    step_np = np.asarray(grid.voxel_size, dtype=np.float64)
    device_key = str(torch.device(device))
    dst_pts32 = _torch_voxel_centers_cached(
        shape,
        tuple(float(x) for x in origin_np),
        tuple(float(x) for x in step_np),
        device_key,
    )
    # Float64 NumPy centers are used only for the sparse correction set.
    dst_pts64 = _voxel_centers(grid)
    origin32 = torch.tensor(origin_np, dtype=torch.float32, device=device)
    step32 = torch.tensor(step_np, dtype=torch.float32, device=device)
    sem_t = torch.from_numpy(sem.reshape(-1)).to(device=device)
    X, Y, Z = shape
    tol = float(boundary_tol_vox)
    outputs = []

    for src_to_dst in src_to_dst_seq:
        # Preserve the exact reference inverse before converting to the fast
        # CUDA representation.
        dst_to_src64 = np.linalg.inv(np.asarray(src_to_dst, dtype=np.float64))
        mat32 = torch.from_numpy(dst_to_src64[:3, :3].astype(np.float32)).to(device)
        trans32 = torch.from_numpy(dst_to_src64[:3, 3].astype(np.float32)).to(device)

        src_pts32 = dst_pts32 @ mat32.T + trans32
        q32 = (src_pts32 - origin32) / step32
        idx = torch.floor(q32).to(torch.int64)

        # Distance to the nearest integer boundary in voxel-coordinate space.
        frac = torch.abs(q32 - torch.round(q32))
        uncertain = torch.any(frac <= tol, dim=1)
        if bool(torch.any(uncertain)):
            pos_t = torch.nonzero(uncertain, as_tuple=False).flatten()
            pos = pos_t.detach().cpu().numpy()
            ref_src = (
                dst_pts64[pos] @ dst_to_src64[:3, :3].T
                + dst_to_src64[:3, 3]
            )
            ref_idx = np.floor(
                (ref_src - origin_np[None]) / step_np[None]
            ).astype(np.int64)
            idx[pos_t] = torch.from_numpy(ref_idx).to(device=device)

        known = (
            (idx[:, 0] >= 0) & (idx[:, 0] < X)
            & (idx[:, 1] >= 0) & (idx[:, 1] < Y)
            & (idx[:, 2] >= 0) & (idx[:, 2] < Z)
        )
        out = torch.full(
            (idx.shape[0],),
            int(free_label),
            dtype=sem_t.dtype,
            device=device,
        )
        if bool(torch.any(known)):
            q = idx[known]
            flat_idx = (q[:, 0] * Y + q[:, 1]) * Z + q[:, 2]
            out[known] = sem_t[flat_idx]
        outputs.append(
            (
                out.reshape(shape).detach().cpu().numpy(),
                known.reshape(shape).detach().cpu().numpy(),
            )
        )
    return outputs
