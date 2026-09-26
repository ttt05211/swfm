"""V20 three-dimensional historical-evidence world-model contracts.

This module owns geometry, responsibility labels and protected composition.
It intentionally has no dataset-specific loader so every training/evaluation
entry point shares the same causal contract.

Inference may use:
  * six historical semantic OCC grids;
  * six historical lidar-observation masks;
  * six historical ego poses;
  * the same future ego trajectory conditioning already permitted by V18.

Future semantic occupancy, future observation masks and GT identities are
supervision/evaluation only and are rejected by the inference-input audit.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from functools import lru_cache
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch

from .metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS
from .motion_transport import FUTURE_FRAMES, HISTORY_FRAMES

V20_PROTOCOL = "p0_f9_v20_3d_history_world_model_v1"
FREE_LABEL = 17
DYNAMIC_IDS = tuple(int(x) for x in DYNAMIC_CLASS_IDS)
DYNAMIC_SET = frozenset(DYNAMIC_IDS)


@dataclass(frozen=True)
class CanonicalLattice:
    """Fixed canonical superset lattice Ωmax.

    The tensor shape and indices are fixed over the run.  High-resolution model
    work is still expected to be restricted by a per-window query mask.
    """

    origin_xyz_m: tuple[float, float, float]
    voxel_size_xyz_m: tuple[float, float, float]
    shape_xyz: tuple[int, int, int]

    def __post_init__(self):
        if len(self.origin_xyz_m) != 3 or len(self.voxel_size_xyz_m) != 3:
            raise ValueError("origin/voxel size must be xyz triples")
        if len(self.shape_xyz) != 3 or min(self.shape_xyz) <= 0:
            raise ValueError("shape_xyz must be positive")
        if min(self.voxel_size_xyz_m) <= 0:
            raise ValueError("voxel size must be positive")

    @property
    def max_xyz_m(self) -> np.ndarray:
        return np.asarray(self.origin_xyz_m, dtype=np.float64) + (
            np.asarray(self.shape_xyz, dtype=np.float64)
            * np.asarray(self.voxel_size_xyz_m, dtype=np.float64)
        )

    def world_to_index(self, xyz_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p = np.asarray(xyz_world, dtype=np.float64)
        if p.shape[-1] != 3:
            raise ValueError("xyz_world last dimension must be 3")
        origin = np.asarray(self.origin_xyz_m, dtype=np.float64)
        step = np.asarray(self.voxel_size_xyz_m, dtype=np.float64)
        idx = np.floor((p - origin) / step).astype(np.int64)
        shp = np.asarray(self.shape_xyz, dtype=np.int64)
        valid = ((idx >= 0) & (idx < shp)).all(axis=-1)
        return idx, valid

    def index_to_world_center(self, idx_xyz: np.ndarray) -> np.ndarray:
        idx = np.asarray(idx_xyz, dtype=np.float64)
        if idx.shape[-1] != 3:
            raise ValueError("idx last dimension must be 3")
        return (
            np.asarray(self.origin_xyz_m, dtype=np.float64)
            + (idx + 0.5) * np.asarray(self.voxel_size_xyz_m, dtype=np.float64)
        )


@dataclass(frozen=True)
class QueryMaskReport:
    mask: np.ndarray
    requested_voxels: int
    in_bounds_voxels: int
    out_of_bounds_voxels: int
    out_of_bounds_fraction: float


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    p = np.asarray(points, dtype=np.float64)
    if T.shape != (4, 4) or p.shape[-1] != 3:
        raise ValueError("expected T[4,4] and points[...,3]")
    return p @ T[:3, :3].T + T[:3, 3]



def poses_to_t0_canonical(
    ego_to_world: np.ndarray,
    t0_ego_to_world: np.ndarray,
) -> np.ndarray:
    """Convert ego poses to per-window t0-ego canonical transforms."""
    poses = np.asarray(ego_to_world, dtype=np.float64)
    t0 = np.asarray(t0_ego_to_world, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or t0.shape != (4, 4):
        raise ValueError("poses must be [N,4,4] and t0 pose [4,4]")
    world_to_t0 = np.linalg.inv(t0)
    return np.stack([world_to_t0 @ p for p in poses], axis=0)


@lru_cache(maxsize=16)
def _grid_centers_xyz_cached(
    shape: tuple[int, int, int],
    origin_t: tuple[float, float, float],
    step_t: tuple[float, float, float],
) -> np.ndarray:
    origin = np.asarray(origin_t, dtype=np.float64)
    step = np.asarray(step_t, dtype=np.float64)
    axes = [
        origin[d] + (np.arange(shape[d], dtype=np.float64) + 0.5) * step[d]
        for d in range(3)
    ]
    x, y, z = np.meshgrid(*axes, indexing="ij")
    out = np.stack((x, y, z), axis=-1)
    out.setflags(write=False)
    return out


def grid_centers_xyz(
    shape_xyz: Sequence[int],
    origin_xyz_m: Sequence[float],
    voxel_size_xyz_m: Sequence[float],
) -> np.ndarray:
    """Return immutable cached native-grid centers for repeated V20 geometry."""
    shape = tuple(int(x) for x in shape_xyz)
    origin = tuple(float(x) for x in origin_xyz_m)
    step = tuple(float(x) for x in voxel_size_xyz_m)
    if len(shape) != 3 or len(origin) != 3 or len(step) != 3:
        raise ValueError("grid center arguments must be xyz triples")
    return _grid_centers_xyz_cached(shape, origin, step)


def future_union_query_mask(
    lattice: CanonicalLattice,
    *,
    future_ego_to_canonical: np.ndarray,
    native_shape_xyz: Sequence[int],
    native_origin_xyz_m: Sequence[float],
    native_voxel_size_xyz_m: Sequence[float],
) -> QueryMaskReport:
    """Rasterize the union of six future native OCC views into Ωmax.

    This uses future ego *pose only*.  It never inspects future semantics or
    future lidar masks.  Duplicate mapped cells are naturally unioned.
    """
    poses = np.asarray(future_ego_to_canonical, dtype=np.float64)
    if poses.shape != (FUTURE_FRAMES, 4, 4):
        raise ValueError("future_ego_to_canonical must be [6,4,4]")
    local = grid_centers_xyz(
        native_shape_xyz, native_origin_xyz_m, native_voxel_size_xyz_m
    ).reshape(-1, 3)
    out = np.zeros(lattice.shape_xyz, dtype=bool)
    requested = 0
    in_bounds = 0
    oob = 0
    for T in poses:
        world = transform_points(T, local)
        idx, valid = lattice.world_to_index(world)
        requested += int(len(idx))
        in_bounds += int(valid.sum())
        oob += int((~valid).sum())
        good = idx[valid]
        if len(good):
            out[good[:, 0], good[:, 1], good[:, 2]] = True
    return QueryMaskReport(
        mask=out,
        requested_voxels=requested,
        in_bounds_voxels=in_bounds,
        out_of_bounds_voxels=oob,
        out_of_bounds_fraction=float(oob / max(requested, 1)),
    )


@dataclass(frozen=True)
class AlignedHistoryEvidence:
    semantic: np.ndarray
    observed: np.ndarray
    observed_free: np.ndarray
    unknown: np.ndarray
    conflict: np.ndarray
    out_of_bounds_samples: int


def observed_native_points(
    semantic: np.ndarray,
    observed: np.ndarray,
    *,
    native_origin_xyz_m: Sequence[float],
    native_voxel_size_xyz_m: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract observed native voxel centers/labels once in deterministic C-order."""
    sem = np.asarray(semantic, dtype=np.uint8)
    obs = np.asarray(observed, dtype=bool)
    if sem.shape != obs.shape or sem.ndim != 3:
        raise ValueError("semantic/observed must be one [X,Y,Z] frame")
    src_id = np.flatnonzero(obs.reshape(-1))
    if src_id.size == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.uint8)
    native_idx = np.column_stack(
        np.unravel_index(src_id, sem.shape)
    ).astype(np.float32, copy=False)
    origin = np.asarray(native_origin_xyz_m, dtype=np.float32)
    step = np.asarray(native_voxel_size_xyz_m, dtype=np.float32)
    local = origin[None] + (native_idx + 0.5) * step[None]
    labels = sem.reshape(-1)[src_id].astype(np.uint8, copy=True)
    return local.astype(np.float32, copy=False), labels


def align_sparse_history_once_to_canonical(
    lattice: CanonicalLattice,
    *,
    history_local_xyz: Sequence[np.ndarray],
    history_semantic_observed: Sequence[np.ndarray],
    history_ego_to_world: np.ndarray,
    t0_ego_to_world: np.ndarray,
    free_label: int = FREE_LABEL,
) -> AlignedHistoryEvidence:
    """Align cached sparse observed evidence without rescanning native dense grids.

    Each frame's points/labels must preserve native C-order.  The result is
    elementwise identical to align_history_once_to_canonical for the same
    observed semantic input.
    """
    if len(history_local_xyz) != HISTORY_FRAMES or len(history_semantic_observed) != HISTORY_FRAMES:
        raise ValueError("V20 requires exactly six sparse historical frames")
    poses_world = np.asarray(history_ego_to_world, dtype=np.float64)
    t0 = np.asarray(t0_ego_to_world, dtype=np.float64)
    if poses_world.shape != (HISTORY_FRAMES, 4, 4) or t0.shape != (4, 4):
        raise ValueError("history poses must be [6,4,4] and t0 pose [4,4]")
    world_to_t0 = np.linalg.inv(t0)
    poses = np.stack([world_to_t0 @ p for p in poses_world], axis=0)

    cshape = (HISTORY_FRAMES,) + tuple(lattice.shape_xyz)
    out_sem = np.full(cshape, int(free_label), dtype=np.uint8)
    out_obs = np.zeros(cshape, dtype=bool)
    conflict = np.zeros(cshape, dtype=bool)
    oob = 0

    for t in range(HISTORY_FRAMES):
        local = np.asarray(history_local_xyz[t], dtype=np.float32)
        labels = np.asarray(history_semantic_observed[t], dtype=np.uint8).reshape(-1)
        if local.ndim != 2 or local.shape[1] != 3 or local.shape[0] != labels.size:
            raise ValueError("sparse history frame must be local_xyz[N,3] + labels[N]")
        if labels.size == 0:
            continue
        world = transform_points(poses[t], local)
        idx, in_bounds = lattice.world_to_index(world)
        oob += int((~in_bounds).sum())
        frame_sem, frame_obs, frame_conflict = _rasterize_observed_frame_vectorized(
            idx,
            in_bounds,
            labels,
            np.ones(labels.size, dtype=bool),
            canonical_shape_xyz=lattice.shape_xyz,
            free_label=int(free_label),
        )
        out_sem[t] = frame_sem
        out_obs[t] = frame_obs
        conflict[t] = frame_conflict

    observed_free = out_obs & (out_sem == int(free_label))
    unknown = ~out_obs
    return AlignedHistoryEvidence(
        semantic=out_sem,
        observed=out_obs,
        observed_free=observed_free,
        unknown=unknown,
        conflict=conflict,
        out_of_bounds_samples=int(oob),
    )


def _rasterize_observed_frame_vectorized(
    idx_xyz: np.ndarray,
    in_bounds: np.ndarray,
    src_semantic_flat: np.ndarray,
    src_observed_flat: np.ndarray,
    *,
    canonical_shape_xyz: Sequence[int],
    free_label: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rasterize one observed frame with exact legacy collision semantics.

    Source C-order is preserved. For each canonical cell:
      * any observation marks the cell observed;
      * occupied overrides observed-free;
      * the first occupied source label wins deterministically;
      * multiple distinct occupied labels set conflict=True.
    """
    idx = np.asarray(idx_xyz, dtype=np.int64)
    bounds = np.asarray(in_bounds, dtype=bool)
    sem = np.asarray(src_semantic_flat).reshape(-1)
    obs = np.asarray(src_observed_flat, dtype=bool).reshape(-1)
    shape = tuple(int(x) for x in canonical_shape_xyz)
    if idx.shape != (sem.size, 3) or bounds.shape != (sem.size,) or obs.shape != (sem.size,):
        raise ValueError("frame raster input shape mismatch")
    out_sem = np.full(shape, int(free_label), dtype=np.uint8)
    out_obs = np.zeros(shape, dtype=bool)
    conflict = np.zeros(shape, dtype=bool)
    valid = bounds & obs
    if not bool(valid.any()):
        return out_sem, out_obs, conflict

    src_id = np.flatnonzero(valid)
    cells = idx[valid]
    labels = sem[valid].astype(np.uint8, copy=False)
    lin = np.ravel_multi_index(cells.T, shape)
    order = np.lexsort((src_id, lin))
    lin = lin[order]
    labels = labels[order]

    unique_cells = np.unique(lin)
    out_obs.reshape(-1)[unique_cells] = True

    occupied = labels != int(free_label)
    if bool(occupied.any()):
        occ_lin = lin[occupied]
        occ_lab = labels[occupied]
        first_lin, first_pos = np.unique(occ_lin, return_index=True)
        out_sem.reshape(-1)[first_lin] = occ_lab[first_pos]
        if len(occ_lin) > 1:
            transition = (
                (occ_lin[1:] == occ_lin[:-1])
                & (occ_lab[1:] != occ_lab[:-1])
            )
            if bool(transition.any()):
                conflict_cells = np.unique(occ_lin[1:][transition])
                conflict.reshape(-1)[conflict_cells] = True
    return out_sem, out_obs, conflict


def align_history_once_to_canonical(
    lattice: CanonicalLattice,
    *,
    history_semantic: np.ndarray,
    history_observed: np.ndarray,
    history_ego_to_world: np.ndarray,
    t0_ego_to_world: np.ndarray,
    native_origin_xyz_m: Sequence[float],
    native_voxel_size_xyz_m: Sequence[float],
    free_label: int = FREE_LABEL,
) -> AlignedHistoryEvidence:
    """Align each historical frame exactly once into the canonical lattice.

    The return keeps time as an explicit axis [T,X,Y,Z].  Unknown is never
    conflated with observed-free.  If more than one native voxel quantizes to a
    canonical cell, observed occupied evidence takes precedence over free;
    disagreeing occupied labels are marked as conflict.
    """
    sem = np.asarray(history_semantic)
    obs = np.asarray(history_observed, dtype=bool)
    poses_world = np.asarray(history_ego_to_world, dtype=np.float64)
    t0 = np.asarray(t0_ego_to_world, dtype=np.float64)
    if sem.shape != obs.shape or sem.ndim != 4:
        raise ValueError("history semantic/observed must be [T,X,Y,Z]")
    if sem.shape[0] != HISTORY_FRAMES or poses_world.shape != (HISTORY_FRAMES, 4, 4):
        raise ValueError("V20 requires exactly six historical frames")
    if t0.shape != (4, 4):
        raise ValueError("t0_ego_to_world must be [4,4]")
    world_to_t0 = np.linalg.inv(t0)
    poses = np.stack([world_to_t0 @ p for p in poses_world], axis=0)

    native_shape = sem.shape[1:]
    native_origin = np.asarray(native_origin_xyz_m, dtype=np.float64)
    native_step = np.asarray(native_voxel_size_xyz_m, dtype=np.float64)
    cshape = (HISTORY_FRAMES,) + tuple(lattice.shape_xyz)
    out_sem = np.full(cshape, int(free_label), dtype=np.uint8)
    out_obs = np.zeros(cshape, dtype=bool)
    conflict = np.zeros(cshape, dtype=bool)
    oob = 0

    for t in range(HISTORY_FRAMES):
        # Only lidar-observed native voxels can contribute evidence.  Mapping
        # every unknown voxel wastes most of the Stage-1 build time.  flatnonzero
        # preserves native C-order, so collision/first-label semantics stay exact.
        src_sem = sem[t].reshape(-1)
        src_obs = obs[t].reshape(-1)
        src_id = np.flatnonzero(src_obs)
        if src_id.size == 0:
            continue
        native_idx = np.column_stack(
            np.unravel_index(src_id, native_shape)
        ).astype(np.float64, copy=False)
        local = native_origin[None] + (native_idx + 0.5) * native_step[None]
        world = transform_points(poses[t], local)
        idx, in_bounds = lattice.world_to_index(world)
        oob += int((~in_bounds).sum())
        frame_sem, frame_obs, frame_conflict = (
            _rasterize_observed_frame_vectorized(
                idx,
                in_bounds,
                src_sem[src_id],
                np.ones(src_id.size, dtype=bool),
                canonical_shape_xyz=lattice.shape_xyz,
                free_label=int(free_label),
            )
        )
        out_sem[t] = frame_sem
        out_obs[t] = frame_obs
        conflict[t] = frame_conflict

    observed_free = out_obs & (out_sem == int(free_label))
    unknown = ~out_obs
    if bool((observed_free & unknown).any()):
        raise RuntimeError("observed-free and unknown must be disjoint")
    return AlignedHistoryEvidence(
        semantic=out_sem,
        observed=out_obs,
        observed_free=observed_free,
        unknown=unknown,
        conflict=conflict,
        out_of_bounds_samples=int(oob),
    )



@dataclass(frozen=True)
class FutureRenderIndex:
    indices_xyz: np.ndarray
    valid: np.ndarray
    out_of_bounds_voxels: int
    linear_index: np.ndarray | None = None


def future_native_to_canonical_indices(
    lattice: CanonicalLattice,
    *,
    future_ego_to_canonical: np.ndarray,
    native_shape_xyz: Sequence[int],
    native_origin_xyz_m: Sequence[float],
    native_voxel_size_xyz_m: Sequence[float],
) -> FutureRenderIndex:
    """Build the one authoritative canonical->future render lookup.

    indices_xyz/valid are [F,X,Y,Z,(3)].  The same lookup can be cached and
    reused by supervision and inference; no future semantic content is needed.
    """
    poses = np.asarray(future_ego_to_canonical, dtype=np.float64)
    if poses.shape != (FUTURE_FRAMES, 4, 4):
        raise ValueError("future_ego_to_canonical must be [6,4,4]")
    shape = tuple(int(x) for x in native_shape_xyz)
    local = grid_centers_xyz(
        shape, native_origin_xyz_m, native_voxel_size_xyz_m
    ).reshape(-1, 3)
    all_idx, all_valid = [], []
    oob = 0
    for T in poses:
        world = transform_points(T, local)
        idx, valid = lattice.world_to_index(world)
        all_idx.append(idx.reshape(shape + (3,)))
        all_valid.append(valid.reshape(shape))
        oob += int((~valid).sum())
    indices = np.stack(all_idx, axis=0)
    valid_all = np.stack(all_valid, axis=0)
    # Keep a compact flattened canonical lookup.  Runtime query-union and
    # rendering can then avoid repeatedly materializing Nx3 advanced-index
    # arrays. Invalid entries are zeroed and ignored through valid_all.
    safe = indices.copy()
    safe[~valid_all] = 0
    Y, Z = int(lattice.shape_xyz[1]), int(lattice.shape_xyz[2])
    linear = (
        safe[..., 0] * (Y * Z)
        + safe[..., 1] * Z
        + safe[..., 2]
    ).astype(np.int32, copy=False)
    return FutureRenderIndex(
        indices_xyz=indices,
        valid=valid_all,
        out_of_bounds_voxels=int(oob),
        linear_index=linear,
    )


def render_canonical_semantic_to_future(
    canonical_semantic: np.ndarray,
    render_index: FutureRenderIndex,
    *,
    free_label: int = FREE_LABEL,
) -> np.ndarray:
    """Nearest-cell deterministic render of one canonical semantic world."""
    world = np.asarray(canonical_semantic)
    idx = np.asarray(render_index.indices_xyz, dtype=np.int64)
    valid = np.asarray(render_index.valid, dtype=bool)
    if world.ndim != 3 or idx.shape[:-1] != valid.shape or idx.shape[-1] != 3:
        raise ValueError("canonical/render-index shape mismatch")
    out = np.full(valid.shape, int(free_label), dtype=world.dtype)
    linear = getattr(render_index, "linear_index", None)
    if linear is not None:
        lin = np.asarray(linear)
        out[valid] = world.reshape(-1)[lin[valid]]
    else:
        q = idx[valid]
        out[valid] = world[q[:, 0], q[:, 1], q[:, 2]]
    return out


def gather_canonical_logits_to_future(
    canonical_logits: torch.Tensor,
    indices_xyz: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Differentiably gather canonical logits into six future native grids.

    canonical_logits: [B,C,Xc,Yc,Zc]
    indices_xyz: [B,F,X,Y,Z,3]
    valid: [B,F,X,Y,Z]
    Returns [B,F,C,X,Y,Z], with invalid locations set to zero logits.
    """
    if canonical_logits.ndim != 5 or indices_xyz.ndim != 6:
        raise ValueError("unexpected canonical/render tensor rank")
    if indices_xyz.shape[-1] != 3 or valid.shape != indices_xyz.shape[:-1]:
        raise ValueError("render indices/valid mismatch")
    B, C, Xc, Yc, Zc = canonical_logits.shape
    if indices_xyz.shape[0] != B:
        raise ValueError("render batch mismatch")
    idx = indices_xyz.long()
    safe = idx.clone()
    safe[..., 0].clamp_(0, Xc - 1)
    safe[..., 1].clamp_(0, Yc - 1)
    safe[..., 2].clamp_(0, Zc - 1)
    linear = (safe[..., 0] * (Yc * Zc) + safe[..., 1] * Zc + safe[..., 2])
    flat = canonical_logits.reshape(B, C, Xc * Yc * Zc)
    gather_idx = linear.reshape(B, 1, -1).expand(-1, C, -1)
    out = torch.gather(flat, 2, gather_idx)
    out = out.reshape(B, C, *valid.shape[1:]).permute(0, 2, 1, 3, 4, 5)
    return out * valid[:, :, None].to(out.dtype)



def canonical_tile_grid_sample_coordinates(
    high_lattice: CanonicalLattice,
    coarse_lattice: CanonicalLattice,
    tile_start_xyz: Sequence[int],
    tile_shape_xyz: Sequence[int],
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Normalized grid_sample coordinates for one canonical high-res tile.

    V20 semantic tensors use Occ3D [X,Y,Z]. PyTorch Conv3d interprets these as
    [D,H,W]=[X,Y,Z], while grid_sample's coordinate tuple is ordered
    (W,H,D)=(Z,Y,X). This helper is the single place where that permutation is
    performed.
    """
    start = np.asarray(tuple(int(x) for x in tile_start_xyz), dtype=np.int64)
    shape = np.asarray(tuple(int(x) for x in tile_shape_xyz), dtype=np.int64)
    if start.shape != (3,) or shape.shape != (3,) or bool((shape <= 0).any()):
        raise ValueError("tile start/shape must be xyz triples")
    hi_shape = np.asarray(high_lattice.shape_xyz, dtype=np.int64)
    if bool((start < 0).any()) or bool((start + shape > hi_shape).any()):
        raise ValueError("tile lies outside high-resolution Ωmax")

    xs = np.arange(start[0], start[0] + shape[0], dtype=np.float64)
    ys = np.arange(start[1], start[1] + shape[1], dtype=np.float64)
    zs = np.arange(start[2], start[2] + shape[2], dtype=np.float64)
    ix, iy, iz = np.meshgrid(xs, ys, zs, indexing="ij")
    high_idx = np.stack((ix, iy, iz), axis=-1)
    xyz = high_lattice.index_to_world_center(high_idx)

    origin = np.asarray(coarse_lattice.origin_xyz_m, dtype=np.float64)
    step = np.asarray(coarse_lattice.voxel_size_xyz_m, dtype=np.float64)
    coarse_f = (xyz - origin) / step - 0.5
    cs = np.asarray(coarse_lattice.shape_xyz, dtype=np.float64)
    denom = np.maximum(cs - 1.0, 1.0)
    norm = 2.0 * coarse_f / denom - 1.0
    # grid_sample tuple order: z, y, x for an input laid out [X,Y,Z].
    grid = np.stack((norm[..., 2], norm[..., 1], norm[..., 0]), axis=-1)
    return torch.as_tensor(grid[None], dtype=dtype, device=device)


def native_sparse_to_canonical_indices(
    lattice: CanonicalLattice,
    *,
    native_indices_xyz: np.ndarray,
    ego_to_canonical: np.ndarray,
    native_origin_xyz_m: Sequence[float],
    native_voxel_size_xyz_m: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Map sparse native Occ3D indices into the canonical lattice."""
    idx = np.asarray(native_indices_xyz, dtype=np.int64)
    if idx.ndim != 2 or idx.shape[1] != 3:
        raise ValueError("native indices must be [N,3]")
    origin = np.asarray(native_origin_xyz_m, dtype=np.float64)
    step = np.asarray(native_voxel_size_xyz_m, dtype=np.float64)
    xyz = origin[None] + (idx.astype(np.float64) + 0.5) * step[None]
    canon = transform_points(np.asarray(ego_to_canonical, dtype=np.float64), xyz)
    return lattice.world_to_index(canon)


class DynamicResponsibility(IntEnum):
    IGNORE = 0
    CURRENT_ANCESTRAL = 1
    DORMANT_ANCESTRAL = 2
    BIRTH = 3


@dataclass(frozen=True)
class DynamicPartition:
    labels: np.ndarray
    current: tuple[int, ...]
    dormant: tuple[int, ...]
    birth: tuple[int, ...]
    ignore: tuple[int, ...]

    def assert_mutually_exclusive(self) -> None:
        groups = [set(self.current), set(self.dormant), set(self.birth), set(self.ignore)]
        union = set().union(*groups)
        if len(union) != sum(len(g) for g in groups):
            raise RuntimeError("dynamic responsibility groups overlap")


def partition_future_dynamic_instances(
    *,
    num_future_instances: int,
    matches_t0: Mapping[int, bool],
    matches_earlier_history: Mapping[int, bool],
    ambiguous: Iterable[int] = (),
) -> DynamicPartition:
    """Strict GT-only responsibility partition used for labels/evaluation.

    A BIRTH is *not* merely absent at t0: it must lack a reliable observed
    ancestor throughout all six history frames.
    """
    n = int(num_future_instances)
    amb = {int(x) for x in ambiguous}
    labels = np.full(n, int(DynamicResponsibility.IGNORE), dtype=np.uint8)
    current, dormant, birth, ignore = [], [], [], []
    for i in range(n):
        if i in amb:
            ignore.append(i)
            continue
        at_t0 = bool(matches_t0.get(i, False))
        earlier = bool(matches_earlier_history.get(i, False))
        if at_t0:
            labels[i] = int(DynamicResponsibility.CURRENT_ANCESTRAL)
            current.append(i)
        elif earlier:
            labels[i] = int(DynamicResponsibility.DORMANT_ANCESTRAL)
            dormant.append(i)
        else:
            labels[i] = int(DynamicResponsibility.BIRTH)
            birth.append(i)
    out = DynamicPartition(
        labels=labels,
        current=tuple(current),
        dormant=tuple(dormant),
        birth=tuple(birth),
        ignore=tuple(ignore),
    )
    out.assert_mutually_exclusive()
    return out


def static_supervision_mask(
    future_gt_semantic: np.ndarray,
    future_gt_observed: np.ndarray,
    *,
    free_label: int = FREE_LABEL,
    dynamic_class_ids: Sequence[int] = DYNAMIC_IDS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (valid, target) for static semantic supervision.

    Supervise only future observed voxels. Dynamic occupied GT is ignored.
    Observed free remains a valid negative/semantic class.
    """
    gt = np.asarray(future_gt_semantic)
    obs = np.asarray(future_gt_observed, dtype=bool)
    if gt.shape != obs.shape:
        raise ValueError("future GT semantic/observed shape mismatch")
    dyn = np.isin(gt, np.asarray(dynamic_class_ids, dtype=gt.dtype))
    valid = obs & ~dyn
    target = np.asarray(gt, dtype=np.uint8).copy()
    target[~valid] = int(free_label)
    return valid, target


def audit_inference_inputs(inputs: Mapping[str, object]) -> None:
    """Fail fast on accidental future-label leakage."""
    forbidden_exact = {
        "future_semantic",
        "future_occ",
        "future_occupancy",
        "future_mask_lidar",
        "future_observed",
        "future_instance_id",
        "future_instance_ids",
        "gt_instance_id",
        "gt_instance_ids",
    }
    bad = []
    for key in inputs:
        low = str(key).lower()
        if low in forbidden_exact:
            bad.append(str(key))
        elif low.startswith("future_gt") or low.startswith("gt_future"):
            bad.append(str(key))
    if bad:
        raise RuntimeError("future supervision leaked into inference inputs: " + ", ".join(sorted(bad)))


def protected_add_only(
    base_v18: torch.Tensor,
    *,
    static_world: torch.Tensor | None = None,
    birth: torch.Tensor | None = None,
    dormant: torch.Tensor | None = None,
    free_label: int = FREE_LABEL,
) -> torch.Tensor:
    """Compose V20 in fixed priority without overwriting V18 occupied voxels.

    Priority:
        V18 current source > Dormant > Birth > Static.

    All added branches write only where the original V18 tensor was free.
    Later lower-priority branches additionally cannot overwrite a higher-priority
    V20 branch.
    """
    out = base_v18.clone()
    base_free = base_v18.eq(int(free_label))

    def add(proposal: torch.Tensor | None) -> None:
        nonlocal out
        if proposal is None:
            return
        if proposal.shape != out.shape:
            raise ValueError("V20 proposal shape must match base V18")
        write = base_free & out.eq(int(free_label)) & proposal.ne(int(free_label))
        out[write] = proposal[write].to(out.dtype)

    add(dormant)
    add(birth)
    add(static_world)
    return out


def assert_zero_contribution_identity(
    base_v18: torch.Tensor,
    final_prediction: torch.Tensor,
) -> None:
    if not torch.equal(base_v18, final_prediction):
        diff = int((base_v18 != final_prediction).sum().item())
        raise AssertionError(
            f"zero-contribution V20 must equal V18 elementwise; differing voxels={diff}"
        )
