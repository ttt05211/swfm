"""Causal Emergence Tokens (CET) Stage-0 surface diagnostics.

This module intentionally contains no learned component.  It defines a narrow,
causal support immediately outside the union of historical BEV grid footprints
and two deterministic surface-continuation baselines.  Future ground truth is
accepted only by :func:`oracle_surface_proposal`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .v19_static_novelty import nearest_static_anchor_map


PROTOCOL = "p0_f9_v22_cet_surface_stage0_v1"
CORE_SURFACE_IDS = (11, 13)
EXTENDED_SURFACE_IDS = (11, 12, 13, 14)
DEFAULT_WIDTHS_M = (0.8, 1.6, 3.2)


@dataclass(frozen=True)
class SurfaceFrontier:
    """Causal BEV supports and their nearest historical surface anchors."""

    new_query_bev: np.ndarray
    scope_by_width: dict[float, np.ndarray]
    causal_by_width: dict[float, np.ndarray]
    nearest_x: np.ndarray
    nearest_y: np.ndarray
    nearest_distance_m: np.ndarray
    anchor_bev: np.ndarray


def _validate_widths(widths_m: Iterable[float]) -> tuple[float, ...]:
    widths = tuple(float(x) for x in widths_m)
    if not widths or any((not np.isfinite(x)) or x <= 0 for x in widths):
        raise ValueError("frontier widths must be finite positive values")
    if tuple(sorted(set(widths))) != widths:
        raise ValueError("frontier widths must be unique and ascending")
    return widths


def _masked_surface(
    static_render: np.ndarray,
    class_ids: Iterable[int],
    free_label: int,
) -> np.ndarray:
    sem = np.asarray(static_render, dtype=np.uint8)
    if sem.ndim != 3:
        raise ValueError("static_render must be [X,Y,Z]")
    ids = np.asarray(tuple(int(x) for x in class_ids), dtype=np.uint8)
    if ids.size == 0:
        raise ValueError("surface class set cannot be empty")
    return np.where(np.isin(sem, ids), sem, int(free_label)).astype(
        np.uint8, copy=False
    )


def build_surface_frontier(
    static_render: np.ndarray,
    history_footprint_bev: np.ndarray,
    *,
    class_ids: Iterable[int],
    widths_m: Iterable[float] = DEFAULT_WIDTHS_M,
    free_label: int = 17,
    voxel_size_xy_m: float = 0.4,
) -> SurfaceFrontier:
    """Build geometry-only and surface-supported frontier corridors.

    ``scope`` is the strip just outside the historical grid-footprint union.
    ``causal`` additionally requires a historical ground-surface column within
    the same metric width.  This separation makes the anchor-retention loss
    directly measurable.
    """
    widths = _validate_widths(widths_m)
    sem = _masked_surface(static_render, class_ids, free_label)
    footprint = np.asarray(history_footprint_bev, dtype=bool)
    if footprint.shape != sem.shape[:2]:
        raise ValueError("history footprint/static render shape mismatch")
    if not np.isfinite(voxel_size_xy_m) or float(voxel_size_xy_m) <= 0:
        raise ValueError("voxel_size_xy_m must be finite and positive")

    new_query = ~footprint

    # The generic exact-nearest helper is reused for both supports.  The dummy
    # one-bin semantic grid marks every historical-footprint cell as an anchor,
    # so its distance is a pure geometry/query-entry distance.
    dummy = np.full((*footprint.shape, 1), int(free_label), dtype=np.uint8)
    dummy[..., 0][footprint] = 0
    scope_cells, _, _, scope_valid = nearest_static_anchor_map(
        dummy, footprint, free_label=int(free_label)
    )
    surface_cells, ax, ay, surface_valid = nearest_static_anchor_map(
        sem, footprint, free_label=int(free_label)
    )
    scale = float(voxel_size_xy_m)
    scope_m = np.asarray(scope_cells, dtype=np.float32) * scale
    surface_m = np.asarray(surface_cells, dtype=np.float32) * scale
    anchor_bev = footprint & (sem != int(free_label)).any(axis=2)

    scope = {}
    causal = {}
    for width in widths:
        scope[width] = (
            new_query & scope_valid & (scope_m <= width + 1e-6)
        )
        causal[width] = (
            new_query & surface_valid & (surface_m <= width + 1e-6)
        )
        if bool((causal[width] & ~scope[width]).any()):
            raise RuntimeError("causal surface support escaped geometry scope")

    return SurfaceFrontier(
        new_query_bev=new_query,
        scope_by_width=scope,
        causal_by_width=causal,
        nearest_x=np.asarray(ax, dtype=np.int32),
        nearest_y=np.asarray(ay, dtype=np.int32),
        nearest_distance_m=surface_m,
        anchor_bev=anchor_bev,
    )


def oracle_surface_proposal(
    future_gt: np.ndarray,
    support_bev: np.ndarray,
    *,
    class_ids: Iterable[int],
    free_label: int = 17,
) -> np.ndarray:
    """Exact surface proposal for an upper bound; this is the only GT path."""
    gt = np.asarray(future_gt, dtype=np.uint8)
    support = np.asarray(support_bev, dtype=bool)
    if gt.ndim != 3 or support.shape != gt.shape[:2]:
        raise ValueError("future GT/support shape mismatch")
    keep = support[..., None] & np.isin(
        gt, np.asarray(tuple(int(x) for x in class_ids), dtype=np.uint8)
    )
    return np.where(keep, gt, int(free_label)).astype(np.uint8, copy=False)


def nearest_column_proposal(
    static_render: np.ndarray,
    support_bev: np.ndarray,
    nearest_x: np.ndarray,
    nearest_y: np.ndarray,
    *,
    class_ids: Iterable[int],
    free_label: int = 17,
) -> np.ndarray:
    """Copy the nearest historical surface column into each supported query."""
    sem = _masked_surface(static_render, class_ids, free_label)
    support = np.asarray(support_bev, dtype=bool)
    ax = np.asarray(nearest_x, dtype=np.int64)
    ay = np.asarray(nearest_y, dtype=np.int64)
    if not (support.shape == ax.shape == ay.shape == sem.shape[:2]):
        raise ValueError("surface support/anchor shape mismatch")
    out = np.full_like(sem, int(free_label))
    if bool(support.any()):
        copied = sem[ax, ay]
        out[support] = copied[support]
    return out


def tangent_plane_proposal(
    static_render: np.ndarray,
    support_bev: np.ndarray,
    nearest_x: np.ndarray,
    nearest_y: np.ndarray,
    *,
    class_ids: Iterable[int],
    free_label: int = 17,
    max_abs_slope: float = 1.0,
) -> np.ndarray:
    """Extend nearest columns while continuing the local surface tangent.

    A nearest-filled height field is differentiated only to estimate the local
    tangent at the historical anchor.  Semantics and column thickness are
    copied from that real historical column; only the vertical offset changes.
    Therefore this baseline uses history and ego geometry only.
    """
    sem = _masked_surface(static_render, class_ids, free_label)
    support = np.asarray(support_bev, dtype=bool)
    ax = np.asarray(nearest_x, dtype=np.int64)
    ay = np.asarray(nearest_y, dtype=np.int64)
    if not (support.shape == ax.shape == ay.shape == sem.shape[:2]):
        raise ValueError("surface support/anchor shape mismatch")
    if not np.isfinite(max_abs_slope) or float(max_abs_slope) < 0:
        raise ValueError("max_abs_slope must be finite and non-negative")

    occupied = sem != int(free_label)
    count = occupied.sum(axis=2)
    z = np.arange(sem.shape[2], dtype=np.float32)
    height = np.zeros(sem.shape[:2], dtype=np.float32)
    np.divide(
        (occupied * z[None, None]).sum(axis=2),
        count,
        out=height,
        where=count > 0,
    )
    anchor = count > 0

    def _axis_slope(axis: int) -> np.ndarray:
        previous_height = np.roll(height, 1, axis=axis)
        next_height = np.roll(height, -1, axis=axis)
        previous_valid = np.roll(anchor, 1, axis=axis)
        next_valid = np.roll(anchor, -1, axis=axis)
        if axis == 0:
            previous_valid[0] = False
            next_valid[-1] = False
        else:
            previous_valid[:, 0] = False
            next_valid[:, -1] = False
        slope = np.zeros_like(height)
        both = anchor & previous_valid & next_valid
        forward = anchor & ~previous_valid & next_valid
        backward = anchor & previous_valid & ~next_valid
        slope[both] = 0.5 * (next_height[both] - previous_height[both])
        slope[forward] = next_height[forward] - height[forward]
        slope[backward] = height[backward] - previous_height[backward]
        return slope

    gx, gy = _axis_slope(0), _axis_slope(1)
    limit = float(max_abs_slope)
    gx = np.clip(gx, -limit, limit)
    gy = np.clip(gy, -limit, limit)

    qx, qy = np.nonzero(support)
    out = np.full_like(sem, int(free_label))
    if not len(qx):
        return out
    sx, sy = ax[qx, qy], ay[qx, qy]
    shift = np.rint(
        gx[sx, sy] * (qx - sx) + gy[sx, sy] * (qy - sy)
    ).astype(np.int64)
    source_columns = sem[sx, sy]
    Z = sem.shape[2]
    for source_z in range(Z):
        values = source_columns[:, source_z]
        target_z = source_z + shift
        valid = (values != int(free_label)) & (target_z >= 0) & (target_z < Z)
        if bool(valid.any()):
            out[qx[valid], qy[valid], target_z[valid]] = values[valid]
    return out


def protected_surface_add(
    base: np.ndarray,
    proposal: np.ndarray,
    *,
    free_label: int = 17,
) -> np.ndarray:
    """V18-free add-only composition used by every CET Stage-0 variant."""
    out = np.asarray(base, dtype=np.uint8).copy()
    prop = np.asarray(proposal, dtype=np.uint8)
    if out.shape != prop.shape:
        raise ValueError("base/proposal shape mismatch")
    add = (out == int(free_label)) & (prop != int(free_label))
    out[add] = prop[add]
    return out
