"""Causal Emergence Tokens (CET) Stage-0 surface diagnostics.

This module intentionally contains no learned component.  It defines a narrow,
causal support immediately outside the union of historical BEV grid footprints
and two deterministic surface-continuation baselines.  Future ground truth is
accepted only by :func:`oracle_surface_proposal`.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .v19_static_novelty import nearest_static_anchor_map


PROTOCOL = "p0_f9_v22_cet_surface_stage0_v1"
CORE_SURFACE_IDS = (11, 13)
EXTENDED_SURFACE_IDS = (11, 12, 13, 14)
DEFAULT_WIDTHS_M = (0.8, 1.6, 3.2)
COMPREHENSIVE_WIDTHS_M = (0.8, 1.6, 3.2, 6.4)
FRONTIER_TYPES = ("GRID_ENTRY", "VISIBILITY_FRONTIER", "UNION")


def build_future_static_memory_only(
    history_semantics,
    history_observed,
    history_poses,
    future_poses,
    *,
    grid,
    dynamic_class_ids,
    free_label: int = 17,
    workers: int = 1,
    return_observed_bev: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Exact V19 Static-Memory mosaic without unused network BEV features.

    The collision/clearing primitives are shared with the formal V19 builder;
    this wrapper only avoids constructing 36 semantic/geometry feature maps
    that Stage-0 never consumes.
    """
    from .geometry import relative_transform
    from .v19_innovation import (
        _semantic_choices_from_pretransformed,
        _xyz_to_indices,
        prepare_history_alignment_frame,
    )

    if not (
        len(history_semantics)
        == len(history_observed)
        == len(history_poses)
        == 6
    ):
        raise ValueError("expected six history frames")
    if len(future_poses) != 6:
        raise ValueError("expected six future frames")
    shape = tuple(int(x) for x in grid.shape_hwd)
    _, Y, Z = shape
    prepared = [
        prepare_history_alignment_frame(
            sem,
            obs,
            pose,
            grid=grid,
            dynamic_class_ids=dynamic_class_ids,
        )
        for sem, obs, pose in zip(
            history_semantics, history_observed, history_poses
        )
    ]

    def _one(future_pose):
        out = np.full(shape, int(free_label), dtype=np.uint8)
        flat_out = out.reshape(-1)
        observed_bev = np.zeros(shape[:2], dtype=bool)
        for hpose, xyz, labels, usable_static in prepared:
            if not len(xyz):
                continue
            transform = relative_transform(
                np.asarray(hpose, dtype=np.float64),
                np.asarray(future_pose, dtype=np.float64),
            )
            dst_xyz = xyz @ transform[:3, :3].T + transform[:3, 3]
            ix, iy, iz, valid = _xyz_to_indices(dst_xyz, grid)
            ix = np.asarray(ix, dtype=np.int64)
            iy = np.asarray(iy, dtype=np.int64)
            iz = np.asarray(iz, dtype=np.int64)
            valid = np.asarray(valid, dtype=bool)
            if bool(valid.any()):
                observed_bev[ix[valid], iy[valid]] = True
            static_valid = valid & np.asarray(usable_static, dtype=bool)
            if bool(static_valid.any()):
                clear_flat = (
                    (ix[static_valid] * Y + iy[static_valid]) * Z
                    + iz[static_valid]
                ).astype(np.int64, copy=False)
                flat_out[clear_flat] = int(free_label)
            _, _, _, values, write_flat = _semantic_choices_from_pretransformed(
                labels,
                ix,
                iy,
                iz,
                dst_xyz,
                valid,
                usable_static,
                grid=grid,
                free_label=int(free_label),
            )
            if len(write_flat):
                flat_out[np.asarray(write_flat, dtype=np.int64)] = np.asarray(
                    values, dtype=np.uint8
                )
        return out, observed_bev

    nworkers = max(1, min(int(workers), len(future_poses)))
    if nworkers == 1:
        rows = [_one(pose) for pose in future_poses]
    else:
        with ThreadPoolExecutor(max_workers=nworkers) as pool:
            rows = list(pool.map(_one, future_poses))
    memory = np.stack([row[0] for row in rows], axis=0).astype(
        np.uint8, copy=False
    )
    if not bool(return_observed_bev):
        return memory
    observed = np.stack([row[1] for row in rows], axis=0).astype(
        bool, copy=False
    )
    return memory, observed


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


def build_surface_frontier_family(
    static_render: np.ndarray,
    history_footprint_bev: np.ndarray,
    history_observed_union_bev: np.ndarray,
    *,
    class_ids: Iterable[int],
    widths_m: Iterable[float] = COMPREHENSIVE_WIDTHS_M,
    free_label: int = 17,
    voxel_size_xy_m: float = 0.4,
) -> dict[str, SurfaceFrontier]:
    """Build grid-entry, visibility, and union surface frontiers together.

    Visibility candidates are inside a historical grid footprint but on the
    unknown side of the aligned historical ``mask_lidar`` BEV union.  They are
    limited to a metric strip around observed cells, so this is not dense
    unknown-volume completion.  The union is exactly the OR of the two frozen
    supports rather than a separately enlarged region.
    """
    widths = _validate_widths(widths_m)
    footprint = np.asarray(history_footprint_bev, dtype=bool)
    observed = np.asarray(history_observed_union_bev, dtype=bool)
    if footprint.shape != observed.shape:
        raise ValueError("history footprint/observed-union shape mismatch")
    grid = build_surface_frontier(
        static_render,
        footprint,
        class_ids=class_ids,
        widths_m=widths,
        free_label=free_label,
        voxel_size_xy_m=voxel_size_xy_m,
    )

    visibility_region = footprint & ~observed
    dummy = np.full((*observed.shape, 1), int(free_label), dtype=np.uint8)
    dummy[..., 0][observed] = 0
    distance_cells, _, _, valid = nearest_static_anchor_map(
        dummy, observed, free_label=int(free_label)
    )
    distance_m = np.asarray(distance_cells, dtype=np.float32) * float(
        voxel_size_xy_m
    )
    visibility_scope = {}
    visibility_causal = {}
    for width in widths:
        scope = visibility_region & valid & (distance_m <= width + 1e-6)
        causal = scope & (
            grid.nearest_distance_m <= width + 1e-6
        )
        visibility_scope[width] = scope
        visibility_causal[width] = causal

    visibility = SurfaceFrontier(
        new_query_bev=visibility_region,
        scope_by_width=visibility_scope,
        causal_by_width=visibility_causal,
        nearest_x=grid.nearest_x,
        nearest_y=grid.nearest_y,
        nearest_distance_m=grid.nearest_distance_m,
        anchor_bev=grid.anchor_bev,
    )
    union = SurfaceFrontier(
        new_query_bev=grid.new_query_bev | visibility_region,
        scope_by_width={
            width: grid.scope_by_width[width] | visibility_scope[width]
            for width in widths
        },
        causal_by_width={
            width: grid.causal_by_width[width] | visibility_causal[width]
            for width in widths
        },
        nearest_x=grid.nearest_x,
        nearest_y=grid.nearest_y,
        nearest_distance_m=grid.nearest_distance_m,
        anchor_bev=grid.anchor_bev,
    )
    for frontier in (visibility, union):
        for width in widths:
            if bool(
                (
                    frontier.causal_by_width[width]
                    & ~frontier.scope_by_width[width]
                ).any()
            ):
                raise RuntimeError("causal support escaped its frozen scope")
    return {
        "GRID_ENTRY": grid,
        "VISIBILITY_FRONTIER": visibility,
        "UNION": union,
    }


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


def _dominant_surface_class(
    sem: np.ndarray,
    class_ids: tuple[int, ...],
) -> np.ndarray:
    ids = np.asarray(class_ids, dtype=np.uint8)
    counts = np.stack(
        [(sem == int(class_id)).sum(axis=2) for class_id in ids], axis=2
    )
    return ids[counts.argmax(axis=2)]


def oracle_geometry_history_semantic_proposal(
    future_gt: np.ndarray,
    support_bev: np.ndarray,
    static_render: np.ndarray,
    nearest_x: np.ndarray,
    nearest_y: np.ndarray,
    *,
    class_ids: Iterable[int],
    free_label: int = 17,
) -> np.ndarray:
    """Use exact GT surface geometry but causal nearest-history semantics."""
    class_ids = tuple(int(x) for x in class_ids)
    gt = np.asarray(future_gt, dtype=np.uint8)
    support = np.asarray(support_bev, dtype=bool)
    sem = _masked_surface(static_render, class_ids, free_label)
    ax = np.asarray(nearest_x, dtype=np.int64)
    ay = np.asarray(nearest_y, dtype=np.int64)
    if not (support.shape == ax.shape == ay.shape == gt.shape[:2] == sem.shape[:2]):
        raise ValueError("GT/history/support shape mismatch")
    nearest_class = _dominant_surface_class(sem, class_ids)[ax, ay]
    target = support[..., None] & np.isin(
        gt, np.asarray(class_ids, dtype=np.uint8)
    )
    out = np.full_like(gt, int(free_label))
    broadcast_class = np.broadcast_to(nearest_class[..., None], gt.shape)
    out[target] = broadcast_class[target]
    return out


def nearest_geometry_gt_semantic_proposal(
    future_gt: np.ndarray,
    nearest_proposal: np.ndarray,
    *,
    class_ids: Iterable[int],
    free_label: int = 17,
) -> np.ndarray:
    """Keep causal nearest-column geometry and oracle-correct hit semantics."""
    gt = np.asarray(future_gt, dtype=np.uint8)
    out = np.asarray(nearest_proposal, dtype=np.uint8).copy()
    if gt.shape != out.shape:
        raise ValueError("future GT/nearest proposal shape mismatch")
    correctable = (out != int(free_label)) & np.isin(
        gt, np.asarray(tuple(int(x) for x in class_ids), dtype=np.uint8)
    )
    out[correctable] = gt[correctable]
    return out


def oracle_vertical_shift_proposal(
    future_gt: np.ndarray,
    static_render: np.ndarray,
    support_bev: np.ndarray,
    nearest_x: np.ndarray,
    nearest_y: np.ndarray,
    *,
    class_ids: Iterable[int],
    free_label: int = 17,
    max_abs_shift_bins: int = 4,
) -> np.ndarray:
    """Oracle-select one vertical shift per causal historical surface column.

    Column semantics and thickness remain historical.  Future GT is used only
    to select a shift in ``[-max_abs_shift_bins,+max_abs_shift_bins]``.  This
    diagnostic isolates the amount of error explainable by vertical alignment.
    """
    class_ids = tuple(int(x) for x in class_ids)
    gt = np.asarray(future_gt, dtype=np.uint8)
    support = np.asarray(support_bev, dtype=bool)
    sem = _masked_surface(static_render, class_ids, free_label)
    ax = np.asarray(nearest_x, dtype=np.int64)
    ay = np.asarray(nearest_y, dtype=np.int64)
    if not (support.shape == ax.shape == ay.shape == gt.shape[:2] == sem.shape[:2]):
        raise ValueError("GT/history/support shape mismatch")
    limit = int(max_abs_shift_bins)
    if limit < 0:
        raise ValueError("max_abs_shift_bins must be non-negative")
    source = sem[ax, ay]
    source_occupied = source != int(free_label)
    target = np.isin(gt, np.asarray(class_ids, dtype=np.uint8))
    best_score = np.full(support.shape, -1, dtype=np.int16)
    best_shift = np.zeros(support.shape, dtype=np.int8)
    shifts = [0]
    for value in range(1, limit + 1):
        shifts.extend((-value, value))
    Z = sem.shape[2]
    for shift in shifts:
        overlap = np.zeros(support.shape, dtype=np.int16)
        source_start = max(0, -shift)
        source_stop = min(Z, Z - shift)
        if source_start < source_stop:
            target_start = source_start + shift
            target_stop = source_stop + shift
            overlap = (
                source_occupied[..., source_start:source_stop]
                & target[..., target_start:target_stop]
            ).sum(axis=2, dtype=np.int16)
        improve = support & (overlap > best_score)
        best_score[improve] = overlap[improve]
        best_shift[improve] = int(shift)

    out = np.full_like(sem, int(free_label))
    qx, qy = np.nonzero(support)
    if not len(qx):
        return out
    shifts_for_query = best_shift[qx, qy].astype(np.int64)
    source_columns = source[qx, qy]
    for source_z in range(Z):
        values = source_columns[:, source_z]
        target_z = source_z + shifts_for_query
        valid = (values != int(free_label)) & (target_z >= 0) & (target_z < Z)
        if bool(valid.any()):
            out[qx[valid], qy[valid], target_z[valid]] = values[valid]
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
    class_ids = tuple(int(x) for x in class_ids)
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
    dominant_class = _dominant_surface_class(sem, class_ids)

    def _axis_slope(axis: int) -> np.ndarray:
        previous_height = np.roll(height, 1, axis=axis)
        next_height = np.roll(height, -1, axis=axis)
        previous_valid = np.roll(anchor, 1, axis=axis)
        next_valid = np.roll(anchor, -1, axis=axis)
        previous_class = np.roll(dominant_class, 1, axis=axis)
        next_class = np.roll(dominant_class, -1, axis=axis)
        # A curb/semantic transition is a discontinuity, not evidence for a
        # steep ground plane.  Estimate tangents only from same-class surface
        # neighbours and fall back to zero slope at class boundaries.
        previous_valid &= previous_class == dominant_class
        next_valid &= next_class == dominant_class
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
