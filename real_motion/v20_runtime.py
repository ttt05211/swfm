"""V20 runtime helpers shared by formal evaluation and latency measurement."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from .v20_history_world import (
    CanonicalLattice,
    DYNAMIC_IDS,
    canonical_tile_grid_sample_coordinates,
    future_native_to_canonical_indices,
    future_union_query_mask,
    render_canonical_semantic_to_future,
)
from .v20_training import decode_static_logits


@dataclass(frozen=True)
class StaticRuntimeReport:
    canonical_semantic: np.ndarray
    future_semantic: np.ndarray
    query_voxels: int
    active_tiles: int
    out_of_bounds_voxels: int


def _tile_history_context(
    history_observed_coarse: np.ndarray,
    *,
    high_lattice: CanonicalLattice,
    coarse_lattice: CanonicalLattice,
    start_xyz: Sequence[int],
    shape_xyz: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    obs = np.asarray(history_observed_coarse, dtype=bool)
    if obs.shape != (6,) + tuple(coarse_lattice.shape_xyz):
        raise ValueError("history_observed_coarse shape mismatch")
    start = np.asarray(start_xyz, dtype=np.int64)
    shape = np.asarray(shape_xyz, dtype=np.int64)
    hi_step = np.asarray(high_lattice.voxel_size_xyz_m, dtype=np.float64)
    co_step = np.asarray(coarse_lattice.voxel_size_xyz_m, dtype=np.float64)
    factor = np.maximum(np.rint(co_step / hi_step).astype(np.int64), 1)
    x = np.arange(start[0], start[0] + shape[0]) // factor[0]
    y = np.arange(start[1], start[1] + shape[1]) // factor[1]
    z = np.arange(start[2], start[2] + shape[2]) // factor[2]
    x = np.clip(x, 0, coarse_lattice.shape_xyz[0] - 1)
    y = np.clip(y, 0, coarse_lattice.shape_xyz[1] - 1)
    z = np.clip(z, 0, coarse_lattice.shape_xyz[2] - 1)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    seen = obs[:, xx, yy, zz].any(axis=0)
    t0 = obs[-1, xx, yy, zz]
    return seen, seen & ~t0


def decode_static_world_tiled(
    model,
    scene_features: torch.Tensor,
    *,
    high_lattice: CanonicalLattice,
    coarse_lattice: CanonicalLattice,
    future_ego_to_canonical: np.ndarray,
    history_observed_coarse: np.ndarray,
    native_shape_xyz: Sequence[int],
    native_origin_xyz_m: Sequence[float],
    native_voxel_size_xyz_m: Sequence[float],
    tile_size_xyz: Sequence[int] = (32, 32, 16),
    free_label: int = 17,
) -> StaticRuntimeReport:
    """Decode one canonical Static world, then render it to all six futures."""
    if scene_features.shape[0] != 1:
        raise ValueError("runtime helper currently expects one window")
    q = future_union_query_mask(
        high_lattice,
        future_ego_to_canonical=np.asarray(future_ego_to_canonical),
        native_shape_xyz=native_shape_xyz,
        native_origin_xyz_m=native_origin_xyz_m,
        native_voxel_size_xyz_m=native_voxel_size_xyz_m,
    )
    world = np.full(high_lattice.shape_xyz, int(free_label), dtype=np.uint8)
    tile = np.asarray(tuple(int(x) for x in tile_size_xyz), dtype=np.int64)
    shape = np.asarray(high_lattice.shape_xyz, dtype=np.int64)
    starts = []
    for x in range(0, shape[0], tile[0]):
        for y in range(0, shape[1], tile[1]):
            for z in range(0, shape[2], tile[2]):
                stop = np.minimum(np.asarray([x, y, z]) + tile, shape)
                if q.mask[x:stop[0], y:stop[1], z:stop[2]].any():
                    starts.append((x, y, z))

    with torch.inference_mode():
        for start_t in starts:
            start = np.asarray(start_t, dtype=np.int64)
            stop = np.minimum(start + tile, shape)
            tshape = tuple((stop - start).tolist())
            grid = canonical_tile_grid_sample_coordinates(
                high_lattice,
                coarse_lattice,
                start,
                tshape,
                device=scene_features.device,
                dtype=scene_features.dtype,
            )
            qtile = q.mask[
                start[0]:stop[0],
                start[1]:stop[1],
                start[2]:stop[2],
            ]
            seen, missing = _tile_history_context(
                history_observed_coarse,
                high_lattice=high_lattice,
                coarse_lattice=coarse_lattice,
                start_xyz=start,
                shape_xyz=tshape,
            )
            logits = model.static.refine_tiles(
                scene_features,
                sample_grid=grid,
                query_mask=torch.from_numpy(qtile).to(scene_features.device).unsqueeze(0),
                seen_mask=torch.from_numpy(seen).to(scene_features.device).unsqueeze(0),
                t0_missing_mask=torch.from_numpy(missing).to(scene_features.device).unsqueeze(0),
            )
            pred = decode_static_logits(logits)[0].cpu().numpy().astype(np.uint8)
            # No prediction is allowed outside the future-union query domain.
            pred[~qtile] = int(free_label)
            world[
                start[0]:stop[0],
                start[1]:stop[1],
                start[2]:stop[2],
            ] = pred

    ri = future_native_to_canonical_indices(
        high_lattice,
        future_ego_to_canonical=np.asarray(future_ego_to_canonical),
        native_shape_xyz=native_shape_xyz,
        native_origin_xyz_m=native_origin_xyz_m,
        native_voxel_size_xyz_m=native_voxel_size_xyz_m,
    )
    future = render_canonical_semantic_to_future(
        world, ri, free_label=int(free_label)
    )
    return StaticRuntimeReport(
        canonical_semantic=world,
        future_semantic=future,
        query_voxels=int(q.mask.sum()),
        active_tiles=int(len(starts)),
        out_of_bounds_voxels=int(ri.out_of_bounds_voxels),
    )


def static_subset_masks(
    *,
    high_lattice: CanonicalLattice,
    history_observed: np.ndarray,
    history_ego_to_canonical: np.ndarray,
    future_ego_to_canonical: np.ndarray,
    future_observed: np.ndarray,
    future_gt_semantic: np.ndarray,
    native_origin_xyz_m: Sequence[float],
    native_voxel_size_xyz_m: Sequence[float],
    free_label: int = 17,
) -> dict[str, np.ndarray]:
    """Future-native subset domains derived only from historical observation geometry."""
    hist_obs = np.asarray(history_observed, dtype=bool)
    seen = np.zeros(high_lattice.shape_xyz, dtype=bool)
    t0_seen = np.zeros_like(seen)
    from .v20_history_world import native_sparse_to_canonical_indices
    for ti in range(6):
        native = np.argwhere(hist_obs[ti])
        idx, valid = native_sparse_to_canonical_indices(
            high_lattice,
            native_indices_xyz=native,
            ego_to_canonical=np.asarray(history_ego_to_canonical[ti]),
            native_origin_xyz_m=native_origin_xyz_m,
            native_voxel_size_xyz_m=native_voxel_size_xyz_m,
        )
        good = idx[valid]
        if len(good):
            seen[good[:, 0], good[:, 1], good[:, 2]] = True
            if ti == 5:
                t0_seen[good[:, 0], good[:, 1], good[:, 2]] = True
    ri = future_native_to_canonical_indices(
        high_lattice,
        future_ego_to_canonical=np.asarray(future_ego_to_canonical),
        native_shape_xyz=hist_obs.shape[1:],
        native_origin_xyz_m=native_origin_xyz_m,
        native_voxel_size_xyz_m=native_voxel_size_xyz_m,
    )
    seen_f = render_canonical_semantic_to_future(seen, ri, free_label=0).astype(bool)
    t0_f = render_canonical_semantic_to_future(t0_seen, ri, free_label=0).astype(bool)
    gt = np.asarray(future_gt_semantic)
    obs = np.asarray(future_observed, dtype=bool)
    dyn = np.isin(gt, np.asarray(DYNAMIC_IDS, dtype=gt.dtype))
    static_domain = obs & ~dyn
    return {
        "history_seen_t0_missing": static_domain & seen_f & ~t0_f,
        "never_seen_static_domain": static_domain & ~seen_f,
        "never_seen_static_positive": (
            static_domain & ~seen_f & (gt != int(free_label))
        ),
    }
