"""Causal runtime and composition for V20 unified completion."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

from .local_st_world_model import SEMANTIC_CLASSES
from .motion_transport import FUTURE_FRAMES
from .v20_history_world import FREE_LABEL, CanonicalLattice
from .v20_unified_data import (
    CompletionTile,
    CompletionTileQuery,
    iter_core_tiles,
    make_completion_tile,
    native_tile_points_to_t0,
    native_voxel_center_points,
    points_in_lattice,
)


@dataclass(frozen=True)
class UnifiedRuntimeReport:
    requested_voxels: int
    in_bounds_voxels: int
    out_of_bounds_voxels: int
    eligible_voxels: int


def completion_support(
    current_transport: torch.Tensor, geometry_query_valid: torch.Tensor
) -> torch.Tensor:
    if current_transport.shape != geometry_query_valid.shape:
        raise ValueError("transport and geometry validity must have identical shapes")
    return geometry_query_valid.bool() & (current_transport.long() == FREE_LABEL)


def dense_geometry_and_transport_condition(
    current_transport: torch.Tensor,
    future_ego_to_t0: torch.Tensor,
    *,
    coarse_lattice: CanonicalLattice,
    native_origin_xyz_m: Sequence[float],
    native_voxel_size_xyz_m: Sequence[float],
    chunk_shape_xyz: Sequence[int] = (64, 64, 32),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build query validity and a spatial 19-channel transport condition.

    Channels 0..17 are per-coarse-cell semantic proportions and channel 18 is
    geometric coverage. No future GT or supervision enters this path.
    """
    if current_transport.ndim != 5 or current_transport.shape[1] != FUTURE_FRAMES:
        raise ValueError("current_transport must be [B,6,X,Y,Z]")
    B = int(current_transport.shape[0])
    native = tuple(int(v) for v in current_transport.shape[2:])
    if tuple(future_ego_to_t0.shape) != (B, FUTURE_FRAMES, 4, 4):
        raise ValueError("future_ego_to_t0 must be [B,6,4,4]")

    device = current_transport.device
    valid_dense = torch.zeros(
        (B, FUTURE_FRAMES, *native), device=device, dtype=torch.bool
    )
    coarse_shape = tuple(int(v) for v in coarse_lattice.shape_xyz)
    coarse_voxels = int(coarse_shape[0] * coarse_shape[1] * coarse_shape[2])
    counts = torch.zeros(
        (B, FUTURE_FRAMES, coarse_voxels, SEMANTIC_CLASSES),
        device=device,
        dtype=torch.float32,
    )
    coverage_count = torch.zeros(
        (B, FUTURE_FRAMES, coarse_voxels, 1),
        device=device,
        dtype=torch.float32,
    )
    counts_flat = counts.reshape(-1, SEMANTIC_CLASSES)
    coverage_flat = coverage_count.reshape(-1, 1)
    coarse_origin = torch.as_tensor(
        coarse_lattice.origin_xyz_m, device=device, dtype=future_ego_to_t0.dtype
    )
    coarse_step = torch.as_tensor(
        coarse_lattice.voxel_size_xyz_m,
        device=device,
        dtype=future_ego_to_t0.dtype,
    )

    # Materialize the native lattice once and transform all horizons in one
    # batched operation.  The former chunk loop launched dozens of meshgrid,
    # bounds and index_add kernels per window despite ample GPU memory.
    native_points = native_voxel_center_points(
        start_xyz=(0, 0, 0),
        shape_xyz=native,
        device=device,
        dtype=future_ego_to_t0.dtype,
        native_origin_xyz_m=native_origin_xyz_m,
        native_voxel_size_xyz_m=native_voxel_size_xyz_m,
    ).reshape(-1, 3)
    points = torch.einsum(
        "nj,bhij->bhni", native_points, future_ego_to_t0[..., :3, :3]
    ) + future_ego_to_t0[..., :3, 3].unsqueeze(-2)
    coarse_idx = torch.floor((points - coarse_origin) / coarse_step).long()
    valid = (
        (coarse_idx[..., 0] >= 0)
        & (coarse_idx[..., 0] < coarse_shape[0])
        & (coarse_idx[..., 1] >= 0)
        & (coarse_idx[..., 1] < coarse_shape[1])
        & (coarse_idx[..., 2] >= 0)
        & (coarse_idx[..., 2] < coarse_shape[2])
    )
    valid_dense.copy_(valid.reshape(B, FUTURE_FRAMES, *native))
    linear = (
        coarse_idx[..., 0] * (coarse_shape[1] * coarse_shape[2])
        + coarse_idx[..., 1] * coarse_shape[2]
        + coarse_idx[..., 2]
    )
    offsets = torch.arange(
        B * FUTURE_FRAMES, device=device, dtype=linear.dtype
    ).view(B, FUTURE_FRAMES, 1) * coarse_voxels
    global_linear = (linear + offsets)[valid]
    coverage_flat.index_add_(
        0,
        global_linear,
        torch.ones(
            (global_linear.numel(), 1), device=device, dtype=torch.float32
        ),
    )
    labels = current_transport.reshape(B, FUTURE_FRAMES, -1)[valid].long()
    occupied = labels != FREE_LABEL
    counts_flat.index_add_(
        0,
        global_linear[occupied],
        F.one_hot(labels[occupied], SEMANTIC_CLASSES).to(torch.float32),
    )

    counts[..., FREE_LABEL] = (
        coverage_count[..., 0] - counts[..., :FREE_LABEL].sum(dim=-1)
    ).clamp_min_(0.0)
    proportions = counts / coverage_count.clamp_min(1.0)
    native_step = torch.as_tensor(
        native_voxel_size_xyz_m, dtype=torch.float32, device=device
    )
    coarse_step_f = torch.as_tensor(
        coarse_lattice.voxel_size_xyz_m, dtype=torch.float32, device=device
    )
    nominal = (
        coarse_step_f.prod() / native_step.prod().clamp_min(1.0e-12)
    ).clamp_min(1.0)
    coverage = (coverage_count / nominal).clamp_(0.0, 1.0)
    condition = torch.cat((proportions, coverage), dim=-1)
    condition = condition.view(
        B, FUTURE_FRAMES, *coarse_shape, SEMANTIC_CLASSES + 1
    ).permute(0, 1, 5, 2, 3, 4).contiguous()
    return valid_dense, condition.detach()


def transport_condition(
    current_transport: torch.Tensor,
    geometry_query_valid: torch.Tensor,
    future_ego_to_t0: torch.Tensor,
    *,
    coarse_lattice: CanonicalLattice,
    native_origin_xyz_m: Sequence[float],
    native_voxel_size_xyz_m: Sequence[float],
) -> torch.Tensor:
    valid, condition = dense_geometry_and_transport_condition(
        current_transport,
        future_ego_to_t0,
        coarse_lattice=coarse_lattice,
        native_origin_xyz_m=native_origin_xyz_m,
        native_voxel_size_xyz_m=native_voxel_size_xyz_m,
    )
    if geometry_query_valid.shape != valid.shape:
        raise ValueError("geometry_query_valid must match current_transport")
    if not torch.equal(geometry_query_valid.bool(), valid):
        raise RuntimeError(
            "provided geometry validity disagrees with spatial transport mapping"
        )
    return condition


def compose_completion_tiles(
    current_transport: torch.Tensor,
    tile_logits: Sequence[torch.Tensor],
    tile_queries: Sequence[CompletionTileQuery],
) -> torch.Tensor:
    """Compose tile predictions directly without dense 18-way logits."""
    if len(tile_logits) != len(tile_queries):
        raise ValueError("tile logit/query count mismatch")
    out = current_transport.clone()
    for logits, query in zip(tile_logits, tile_queries):
        tile = query.tile
        if tuple(logits.shape) != (*tile.halo_shape_xyz, SEMANTIC_CLASSES):
            raise ValueError("tile logits must cover the full halo tile")
        proposal = logits[(*tile.core_slice_xyz, slice(None))].argmax(dim=-1)
        support = query.support[tile.core_slice_xyz]
        start, size = tile.core_start_xyz, tile.core_shape_xyz
        sl = tuple(slice(start[d], start[d] + size[d]) for d in range(3))
        current_core = out[(tile.window_index, tile.horizon, *sl)]
        write = (
            support.bool()
            & (current_core.long() == FREE_LABEL)
            & (proposal != FREE_LABEL)
        )
        out[(tile.window_index, tile.horizon, *sl)] = torch.where(
            write, proposal.to(current_core.dtype), current_core
        )
    return out


def compose(
    current_transport: torch.Tensor,
    completion_logits: torch.Tensor,
    support: torch.Tensor,
) -> torch.Tensor:
    """Protected add-only composition; transport occupancy is immutable."""
    if completion_logits.shape[:-1] != current_transport.shape:
        raise ValueError("completion logits must be [...,18] over transport voxels")
    if completion_logits.shape[-1] != SEMANTIC_CLASSES or support.shape != current_transport.shape:
        raise ValueError("completion classes/support shape mismatch")
    proposal = completion_logits.argmax(dim=-1)
    write = support.bool() & (current_transport.long() == FREE_LABEL) & (proposal != FREE_LABEL)
    return torch.where(write, proposal.to(current_transport.dtype), current_transport)


def prepare_runtime_queries(
    current_transport: torch.Tensor,
    future_ego_to_t0: torch.Tensor,
    *,
    coarse_lattice: CanonicalLattice,
    native_origin_xyz_m: Sequence[float],
    native_voxel_size_xyz_m: Sequence[float],
    tiles: Sequence[CompletionTile] | None = None,
    core_shape_xyz: Sequence[int] = (32, 32, 16),
    halo: int = 2,
    dense_geometry_valid: torch.Tensor | None = None,
    dense_completion_support: torch.Tensor | None = None,
) -> tuple[list[CompletionTileQuery], UnifiedRuntimeReport]:
    """Prepare future-native queries without semantic supervision."""
    if current_transport.ndim != 5 or current_transport.shape[1] != FUTURE_FRAMES:
        raise ValueError("current_transport must be [B,6,X,Y,Z]")
    B = int(current_transport.shape[0])
    native = tuple(int(v) for v in current_transport.shape[2:])
    if tuple(future_ego_to_t0.shape) != (B, FUTURE_FRAMES, 4, 4):
        raise ValueError("future_ego_to_t0 must be [B,6,4,4]")
    if dense_geometry_valid is not None and dense_geometry_valid.shape != current_transport.shape:
        raise ValueError("dense_geometry_valid must match current_transport")
    if dense_completion_support is not None and dense_completion_support.shape != current_transport.shape:
        raise ValueError("dense_completion_support must match current_transport")
    if tiles is None:
        tiles = [
            make_completion_tile(
                window_index=b,
                horizon=h,
                core_start_xyz=start,
                core_shape_xyz=size,
                native_shape_xyz=native,
                halo=halo,
            )
            for b in range(B)
            for h in range(FUTURE_FRAMES)
            for start, size in iter_core_tiles(native, core_shape_xyz)
        ]
    queries: list[CompletionTileQuery] = []
    requested = 0
    in_bounds_counts: list[torch.Tensor] = []
    eligible_counts: list[torch.Tensor] = []
    tensor_cache: dict[
        tuple[int, int, tuple[int, int, int], tuple[int, int, int]],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ] = {}
    for tile in tiles:
        cache_key = (
            int(tile.window_index),
            int(tile.horizon),
            tuple(tile.halo_start_xyz),
            tuple(tile.halo_shape_xyz),
        )
        cached = tensor_cache.get(cache_key)
        if cached is None:
            points = native_tile_points_to_t0(
                tile,
                future_ego_to_t0,
                native_origin_xyz_m=native_origin_xyz_m,
                native_voxel_size_xyz_m=native_voxel_size_xyz_m,
            )
            hs = tile.halo_start_xyz
            he = tuple(hs[d] + tile.halo_shape_xyz[d] for d in range(3))
            sl = tuple(slice(hs[d], he[d]) for d in range(3))
            dense_index = (tile.window_index, tile.horizon, *sl)
            geometry_valid = (
                dense_geometry_valid[dense_index]
                if dense_geometry_valid is not None
                else points_in_lattice(points, coarse_lattice)
            )
            support = (
                dense_completion_support[dense_index]
                if dense_completion_support is not None
                else completion_support(current_transport[dense_index], geometry_valid)
            )
            cached = (points, geometry_valid, support)
            tensor_cache[cache_key] = cached
        points, geometry_valid, support = cached
        query = CompletionTileQuery(tile, points, geometry_valid, support)
        query.validate()
        queries.append(query)
        requested += int(geometry_valid.numel())
        in_bounds_counts.append(geometry_valid.sum(dtype=torch.int64))
        eligible_counts.append(support.sum(dtype=torch.int64))
    if in_bounds_counts:
        totals = torch.stack(
            (
                torch.stack(in_bounds_counts).sum(),
                torch.stack(eligible_counts).sum(),
            )
        ).detach().cpu().tolist()
        in_bounds, eligible = (int(totals[0]), int(totals[1]))
    else:
        in_bounds = eligible = 0
    return queries, UnifiedRuntimeReport(
        requested_voxels=requested,
        in_bounds_voxels=in_bounds,
        out_of_bounds_voxels=requested - in_bounds,
        eligible_voxels=eligible,
    )


def dense_geometry_query_valid(
    future_ego_to_t0: torch.Tensor,
    *,
    coarse_lattice: CanonicalLattice,
    native_shape_xyz: Sequence[int],
    native_origin_xyz_m: Sequence[float],
    native_voxel_size_xyz_m: Sequence[float],
    chunk_shape_xyz: Sequence[int] = (32, 32, 16),
) -> torch.Tensor:
    """Build the audited native query-valid mask without semantic inputs."""
    B = int(future_ego_to_t0.shape[0])
    native = tuple(int(v) for v in native_shape_xyz)
    out = torch.zeros(
        (B, FUTURE_FRAMES, *native),
        device=future_ego_to_t0.device,
        dtype=torch.bool,
    )
    for b in range(B):
        for h in range(FUTURE_FRAMES):
            for start, size in iter_core_tiles(native, chunk_shape_xyz):
                tile = make_completion_tile(
                    window_index=b,
                    horizon=h,
                    core_start_xyz=start,
                    core_shape_xyz=size,
                    native_shape_xyz=native,
                    halo=0,
                )
                points = native_tile_points_to_t0(
                    tile,
                    future_ego_to_t0,
                    native_origin_xyz_m=native_origin_xyz_m,
                    native_voxel_size_xyz_m=native_voxel_size_xyz_m,
                )
                valid = points_in_lattice(points, coarse_lattice)
                sl = tuple(slice(start[d], start[d] + size[d]) for d in range(3))
                out[(b, h, *sl)] = valid
    return out


def assemble_completion_logits(
    tile_logits: Sequence[torch.Tensor],
    tile_queries: Sequence[CompletionTileQuery],
    *,
    output_shape: Sequence[int],
    fill_bias_free: float = 1.0,
) -> torch.Tensor:
    """Write core logits only; repeated training tiles remain caller-owned."""
    if len(tile_logits) != len(tile_queries):
        raise ValueError("tile logit/query count mismatch")
    if len(output_shape) != 5:
        raise ValueError("output_shape must be [B,6,X,Y,Z]")
    if tile_logits:
        device, dtype = tile_logits[0].device, tile_logits[0].dtype
    else:
        device, dtype = torch.device("cpu"), torch.float32
    out = torch.zeros((*tuple(int(v) for v in output_shape), SEMANTIC_CLASSES), device=device, dtype=dtype)
    out[..., FREE_LABEL] = float(fill_bias_free)
    for logits, query in zip(tile_logits, tile_queries):
        tile = query.tile
        if tuple(logits.shape) != (*tile.halo_shape_xyz, SEMANTIC_CLASSES):
            raise ValueError("tile logits must cover the full halo tile")
        core_logits = logits[(*tile.core_slice_xyz, slice(None))]
        start, size = tile.core_start_xyz, tile.core_shape_xyz
        sl = tuple(slice(start[d], start[d] + size[d]) for d in range(3))
        out[(tile.window_index, tile.horizon, *sl, slice(None))] = core_logits
    return out
