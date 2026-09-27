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


def transport_condition(
    current_transport: torch.Tensor,
    geometry_query_valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Semantic proportions plus geometric coverage, with no GT dependency."""
    if current_transport.ndim != 5 or current_transport.shape[1] != FUTURE_FRAMES:
        raise ValueError("current_transport must be [B,6,X,Y,Z]")
    if geometry_query_valid is None:
        geometry_query_valid = torch.ones_like(current_transport, dtype=torch.bool)
    if geometry_query_valid.shape != current_transport.shape:
        raise ValueError("geometry_query_valid must match current_transport")
    valid = geometry_query_valid.bool()
    safe = current_transport.long().clamp(0, SEMANTIC_CLASSES - 1)
    one_hot = F.one_hot(safe, SEMANTIC_CLASSES).to(torch.float32)
    weights = valid.unsqueeze(-1).to(one_hot.dtype)
    counts = (one_hot * weights).sum(dim=(2, 3, 4))
    denom = valid.sum(dim=(2, 3, 4), keepdim=False).clamp_min(1).unsqueeze(-1)
    proportions = counts / denom
    total = current_transport.shape[2] * current_transport.shape[3] * current_transport.shape[4]
    coverage = valid.sum(dim=(2, 3, 4), dtype=torch.float32).unsqueeze(-1) / float(total)
    return torch.cat((proportions, coverage), dim=-1).detach()


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
) -> tuple[list[CompletionTileQuery], UnifiedRuntimeReport]:
    """Prepare future-native queries without semantic supervision."""
    if current_transport.ndim != 5 or current_transport.shape[1] != FUTURE_FRAMES:
        raise ValueError("current_transport must be [B,6,X,Y,Z]")
    B = int(current_transport.shape[0])
    native = tuple(int(v) for v in current_transport.shape[2:])
    if tuple(future_ego_to_t0.shape) != (B, FUTURE_FRAMES, 4, 4):
        raise ValueError("future_ego_to_t0 must be [B,6,4,4]")
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
    requested = in_bounds = eligible = 0
    for tile in tiles:
        points = native_tile_points_to_t0(
            tile,
            future_ego_to_t0,
            native_origin_xyz_m=native_origin_xyz_m,
            native_voxel_size_xyz_m=native_voxel_size_xyz_m,
        )
        geometry_valid = points_in_lattice(points, coarse_lattice)
        hs = tile.halo_start_xyz
        he = tuple(hs[d] + tile.halo_shape_xyz[d] for d in range(3))
        sl = tuple(slice(hs[d], he[d]) for d in range(3))
        transport_tile = current_transport[(tile.window_index, tile.horizon, *sl)]
        support = completion_support(transport_tile, geometry_valid)
        query = CompletionTileQuery(tile, points, geometry_valid, support)
        query.validate()
        queries.append(query)
        requested += int(geometry_valid.numel())
        in_bounds += int(geometry_valid.sum().item())
        eligible += int(support.sum().item())
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
