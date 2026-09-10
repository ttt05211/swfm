"""Exact Strong-source footprint utilities for V17 transport supervision.

The V17 representation keeps its historical 0.8 m target-source mask.  This
module defines a separate *loss-only* footprint built from the exact t0 Strong
source voxels.  The footprint is projected to XY and cropped in the same
source-centered metric frame as the local STWM tube, but it remains at the
native occupancy-grid resolution (0.4 m on the current protocol).
"""
from __future__ import annotations

import numpy as np

from .geometry import OccupancyGrid

NATIVE_SOURCE_FOOTPRINT_CONTRACT = (
    "exact_strong_t0_xy_source_footprint_source_centered_native_grid_v1"
)


def native_source_footprint_patch(
    voxel_indices_xyz: np.ndarray,
    center_xy_m: np.ndarray,
    *,
    grid: OccupancyGrid = OccupancyGrid(),
    patch_size_m: float = 16.0,
) -> tuple[np.ndarray, float]:
    """Project one Strong source to a fixed native-resolution local XY patch.

    Args:
        voxel_indices_xyz: exact Strong component voxels in the t0 occupancy grid.
        center_xy_m: the cached source centroid in the t0 ego frame.
        grid: occupancy-grid metric contract.
        patch_size_m: local square size; must be an even number of native cells.

    Returns:
        ``(mask, coverage)`` where ``mask`` is uint8 ``[H,W]`` at the native XY
        voxel resolution and ``coverage`` is the fraction of the source's unique
        XY cells retained inside the local crop.

    The crop-origin rule intentionally matches ``extract_bev_patch`` used by the
    V16/V17 local semantic tube: the metric center is first mapped to its native
    grid cell with floor, then an even-sized crop is centered on that cell.
    """
    idx = np.asarray(voxel_indices_xyz, dtype=np.int64)
    if idx.ndim != 2 or idx.shape[1] != 3 or len(idx) == 0:
        raise ValueError("voxel_indices_xyz must be non-empty [N,3]")
    shape = tuple(int(x) for x in grid.shape_hwd)
    if bool((idx < 0).any()) or bool((idx[:, 0] >= shape[0]).any()) or bool((idx[:, 1] >= shape[1]).any()) or bool((idx[:, 2] >= shape[2]).any()):
        raise ValueError("source voxel index lies outside occupancy grid")

    vx, vy = float(grid.voxel_size[0]), float(grid.voxel_size[1])
    if vx <= 0 or vy <= 0 or abs(vx - vy) > 1e-12:
        raise ValueError("native footprint requires positive square XY voxels")
    raw_f = float(patch_size_m) / vx
    raw = int(round(raw_f))
    if raw <= 0 or raw % 2 or abs(raw_f - raw) > 1e-9:
        raise ValueError("patch_size_m must be an even integer number of native cells")

    center = np.asarray(center_xy_m, dtype=np.float64)
    if center.shape != (2,) or not bool(np.isfinite(center).all()):
        raise ValueError("center_xy_m must be finite [2]")
    cx = int(np.floor((center[0] - float(grid.x_min)) / vx))
    cy = int(np.floor((center[1] - float(grid.y_min)) / vy))
    x0, y0 = cx - raw // 2, cy - raw // 2

    xy = np.unique(idx[:, :2], axis=0)
    local = xy - np.asarray([x0, y0], dtype=np.int64)[None]
    inside = (
        (local[:, 0] >= 0)
        & (local[:, 0] < raw)
        & (local[:, 1] >= 0)
        & (local[:, 1] < raw)
    )
    mask = np.zeros((raw, raw), dtype=np.uint8)
    if bool(inside.any()):
        q = local[inside]
        mask[q[:, 0], q[:, 1]] = 1
    if not bool(mask.any()):
        raise RuntimeError("source-centered native footprint unexpectedly became empty")
    coverage = float(inside.sum()) / float(len(xy))
    return mask, coverage


def pool_binary_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    """OR-pool a binary mask by an integer factor without changing its origin."""
    x = np.asarray(mask, dtype=np.uint8)
    if x.ndim != 2 or int(factor) <= 0:
        raise ValueError("mask must be [H,W] and factor must be positive")
    f = int(factor)
    H, W = x.shape
    if H % f or W % f:
        raise ValueError("mask shape must be divisible by pooling factor")
    return x.reshape(H // f, f, W // f, f).any(axis=(1, 3)).astype(np.uint8)
