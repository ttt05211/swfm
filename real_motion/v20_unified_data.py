"""Data-only contracts for V20 unified transport completion.

Future ground truth deliberately lives in :class:`UnifiedSupervision`; none of
the model-facing dataclasses can carry it.  This separation is also checked by
the public inference-input audit below.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, fields
from typing import Iterator, Sequence

import torch
import torch.nn.functional as F

from .motion_transport import FUTURE_FRAMES, HISTORY_FRAMES
from .v20_history_world import FREE_LABEL, CanonicalLattice

V20_UNIFIED_PROTOCOL = "p0_f9_v20_unified_transport_completion_v1"

_NATIVE_POINTS_CACHE: OrderedDict[tuple, torch.Tensor] = OrderedDict()
_NATIVE_POINTS_CACHE_SIZE = 128


@dataclass(frozen=True)
class UnifiedHistoryInput:
    semantic: torch.Tensor
    observed: torch.Tensor
    observed_free: torch.Tensor
    future_ego_to_t0: torch.Tensor

    def validate(self) -> None:
        if self.semantic.ndim != 5 or self.semantic.shape[1] != HISTORY_FRAMES:
            raise ValueError("semantic must be [B,6,Xc,Yc,Zc]")
        if self.observed.shape != self.semantic.shape:
            raise ValueError("observed must match semantic")
        if self.observed_free.shape != self.semantic.shape:
            raise ValueError("observed_free must match semantic")
        expected = (self.semantic.shape[0], FUTURE_FRAMES, 4, 4)
        if tuple(self.future_ego_to_t0.shape) != expected:
            raise ValueError("future_ego_to_t0 must be [B,6,4,4]")


@dataclass(frozen=True)
class UnifiedSourceInput:
    """Observation-only source state already encoded by V18."""

    history_source_context: torch.Tensor
    future_transport_queries: torch.Tensor
    source_anchor_xyz_t0_m: torch.Tensor
    kta_displacement_xy_m: torch.Tensor
    window_index: torch.Tensor

    def validate(self, d_model: int | None = None) -> None:
        n = int(self.future_transport_queries.shape[0])
        if self.future_transport_queries.ndim != 3:
            raise ValueError("future_transport_queries must be [N,6,D]")
        if self.future_transport_queries.shape[1] != FUTURE_FRAMES:
            raise ValueError("six future transport queries are required")
        if d_model is not None and self.future_transport_queries.shape[2] != d_model:
            raise ValueError("transport query dimension does not match V18")
        if tuple(self.history_source_context.shape) != (
            n,
            self.future_transport_queries.shape[2],
        ):
            raise ValueError("history_source_context must be [N,D]")
        if tuple(self.source_anchor_xyz_t0_m.shape) != (n, 3):
            raise ValueError("source_anchor_xyz_t0_m must be [N,3]")
        if tuple(self.kta_displacement_xy_m.shape) != (n, FUTURE_FRAMES, 2):
            raise ValueError("kta_displacement_xy_m must be [N,6,2]")
        if tuple(self.window_index.shape) != (n,):
            raise ValueError("window_index must be [N]")


@dataclass(frozen=True)
class UnifiedSupervision:
    """Training/evaluation-only tensors; never accepted by model.forward."""

    future_semantic: torch.Tensor
    formal_valid: torch.Tensor

    def validate(self) -> None:
        if self.future_semantic.ndim != 5:
            raise ValueError("future_semantic must be [B,6,X,Y,Z]")
        if self.future_semantic.shape[1] != FUTURE_FRAMES:
            raise ValueError("six future labels are required")
        if self.formal_valid.shape != self.future_semantic.shape:
            raise ValueError("formal_valid must match future_semantic")


@dataclass(frozen=True)
class CompletionTile:
    """One future-native tile including its convolution halo."""

    window_index: int
    horizon: int
    core_start_xyz: tuple[int, int, int]
    core_shape_xyz: tuple[int, int, int]
    halo_start_xyz: tuple[int, int, int]
    halo_shape_xyz: tuple[int, int, int]
    core_slice_xyz: tuple[slice, slice, slice]


@dataclass(frozen=True)
class CompletionTileQuery:
    tile: CompletionTile
    points_xyz_t0_m: torch.Tensor
    geometry_valid: torch.Tensor
    support: torch.Tensor

    def validate(self) -> None:
        shape = self.tile.halo_shape_xyz
        if tuple(self.points_xyz_t0_m.shape) != (*shape, 3):
            raise ValueError("tile points must be [Xh,Yh,Zh,3]")
        if tuple(self.geometry_valid.shape) != shape:
            raise ValueError("geometry_valid must match halo shape")
        if tuple(self.support.shape) != shape:
            raise ValueError("support must match halo shape")


_FORBIDDEN_INFERENCE_FRAGMENTS = (
    "future_semantic",
    "future_gt",
    "ground_truth",
    "instance",
    "visibility",
    "formal_valid",
    "target",
)


def audit_inference_input_names(obj: object) -> None:
    """Reject accidental future-supervision fields in an inference object."""
    names = [f.name.lower() for f in fields(obj)] if hasattr(obj, "__dataclass_fields__") else []
    bad = [n for n in names if any(x in n for x in _FORBIDDEN_INFERENCE_FRAGMENTS)]
    if bad:
        raise ValueError(f"future supervision leaked into inference input: {bad}")


def iter_core_tiles(
    shape_xyz: Sequence[int], core_shape_xyz: Sequence[int] = (32, 32, 16)
) -> Iterator[tuple[tuple[int, int, int], tuple[int, int, int]]]:
    shape = tuple(int(v) for v in shape_xyz)
    core = tuple(int(v) for v in core_shape_xyz)
    if len(shape) != 3 or len(core) != 3 or min(shape) <= 0 or min(core) <= 0:
        raise ValueError("shape/core_shape must be positive xyz triples")
    for x in range(0, shape[0], core[0]):
        for y in range(0, shape[1], core[1]):
            for z in range(0, shape[2], core[2]):
                start = (x, y, z)
                size = tuple(min(core[d], shape[d] - start[d]) for d in range(3))
                yield start, size


def make_completion_tile(
    *,
    window_index: int,
    horizon: int,
    core_start_xyz: Sequence[int],
    core_shape_xyz: Sequence[int],
    native_shape_xyz: Sequence[int],
    halo: int = 2,
) -> CompletionTile:
    start = tuple(int(v) for v in core_start_xyz)
    core = tuple(int(v) for v in core_shape_xyz)
    native = tuple(int(v) for v in native_shape_xyz)
    if not 0 <= int(horizon) < FUTURE_FRAMES:
        raise ValueError("horizon must be in [0,5]")
    if halo < 0:
        raise ValueError("halo must be non-negative")
    hs = tuple(max(0, start[d] - halo) for d in range(3))
    he = tuple(min(native[d], start[d] + core[d] + halo) for d in range(3))
    hshape = tuple(he[d] - hs[d] for d in range(3))
    cs = tuple(slice(start[d] - hs[d], start[d] - hs[d] + core[d]) for d in range(3))
    return CompletionTile(
        window_index=int(window_index),
        horizon=int(horizon),
        core_start_xyz=start,
        core_shape_xyz=core,
        halo_start_xyz=hs,
        halo_shape_xyz=hshape,
        core_slice_xyz=cs,
    )


def sample_training_tiles(
    support: torch.Tensor,
    future_semantic: torch.Tensor,
    formal_valid: torch.Tensor,
    *,
    core_shape_xyz: Sequence[int] = (32, 32, 16),
    draws_per_horizon: int = 16,
    positive_draws: int = 8,
    halo: int = 2,
    generator: torch.Generator | None = None,
) -> list[CompletionTile]:
    """Draw tiles with replacement, preserving every repeated draw.

    Positive means any non-free target on valid completion support.  If there
    are no positive tiles all draws come from the uniform support pool.  If a
    horizon has no valid support it contributes no tile and therefore no loss.
    """
    if support.ndim != 5 or support.shape[1] != FUTURE_FRAMES:
        raise ValueError("support must be [B,6,X,Y,Z]")
    if future_semantic.shape != support.shape or formal_valid.shape != support.shape:
        raise ValueError("labels/valid masks must match support")
    if not 0 <= positive_draws <= draws_per_horizon:
        raise ValueError("positive_draws must be in [0,draws_per_horizon]")
    native = tuple(int(v) for v in support.shape[2:])
    grid = list(iter_core_tiles(native, core_shape_xyz))
    core = tuple(int(v) for v in core_shape_xyz)
    grid_shape = tuple((native[d] + core[d] - 1) // core[d] for d in range(3))
    padded_shape = tuple(grid_shape[d] * core[d] for d in range(3))
    pad = (
        0,
        padded_shape[2] - native[2],
        0,
        padded_shape[1] - native[1],
        0,
        padded_shape[0] - native[0],
    )
    valid_support = support.bool() & formal_valid.bool()
    positive_support = valid_support & (future_semantic != FREE_LABEL)

    def tile_any(mask: torch.Tensor) -> torch.Tensor:
        padded = F.pad(mask, pad, mode="constant", value=False)
        shaped = padded.reshape(
            mask.shape[0],
            mask.shape[1],
            grid_shape[0],
            core[0],
            grid_shape[1],
            core[1],
            grid_shape[2],
            core[2],
        )
        return torch.any(shaped, dim=(3, 5, 7))

    # Reduce every non-overlapping core tile in two vectorized kernels instead
    # of launching two ``any`` reductions for every tile and horizon.
    flags = torch.stack(
        (tile_any(valid_support), tile_any(positive_support)), dim=0
    ).detach().cpu()
    out: list[CompletionTile] = []
    for b in range(int(support.shape[0])):
        for h in range(FUTURE_FRAMES):
            uniform = torch.nonzero(
                flags[0, b, h].reshape(-1), as_tuple=False
            ).flatten().tolist()
            positive = torch.nonzero(
                flags[1, b, h].reshape(-1), as_tuple=False
            ).flatten().tolist()
            if not uniform:
                continue
            n_pos = positive_draws if positive else 0
            pools = [(positive, n_pos), (uniform, draws_per_horizon - n_pos)]
            for pool, count in pools:
                if count == 0:
                    continue
                chosen = torch.randint(len(pool), (count,), generator=generator).tolist()
                for j in chosen:
                    start, size = grid[pool[j]]
                    out.append(
                        make_completion_tile(
                            window_index=b,
                            horizon=h,
                            core_start_xyz=start,
                            core_shape_xyz=size,
                            native_shape_xyz=native,
                            halo=halo,
                        )
                    )
    return out


def native_voxel_center_points(
    *,
    start_xyz: Sequence[int],
    shape_xyz: Sequence[int],
    device: torch.device,
    dtype: torch.dtype,
    native_origin_xyz_m: Sequence[float],
    native_voxel_size_xyz_m: Sequence[float],
) -> torch.Tensor:
    """Return cached native-frame voxel centers for one rectangular region."""
    start = tuple(int(v) for v in start_xyz)
    shape = tuple(int(v) for v in shape_xyz)
    origin = tuple(float(v) for v in native_origin_xyz_m)
    step = tuple(float(v) for v in native_voxel_size_xyz_m)
    key = (str(device), str(dtype), origin, step, start, shape)
    cached = _NATIVE_POINTS_CACHE.get(key)
    if cached is not None:
        _NATIVE_POINTS_CACHE.move_to_end(key)
        return cached
    axes = []
    for d in range(3):
        idx = torch.arange(
            start[d], start[d] + shape[d], device=device, dtype=dtype
        )
        axes.append(origin[d] + (idx + 0.5) * step[d])
    value = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
    _NATIVE_POINTS_CACHE[key] = value
    while len(_NATIVE_POINTS_CACHE) > _NATIVE_POINTS_CACHE_SIZE:
        _NATIVE_POINTS_CACHE.popitem(last=False)
    return value


def native_tile_points_to_t0(
    tile: CompletionTile,
    future_ego_to_t0: torch.Tensor,
    *,
    native_origin_xyz_m: Sequence[float],
    native_voxel_size_xyz_m: Sequence[float],
) -> torch.Tensor:
    """Return future-native voxel centers transformed into the t0 frame."""
    if future_ego_to_t0.ndim != 4 or tuple(future_ego_to_t0.shape[-2:]) != (4, 4):
        raise ValueError("future_ego_to_t0 must be [B,6,4,4]")
    xyz = native_voxel_center_points(
        start_xyz=tile.halo_start_xyz,
        shape_xyz=tile.halo_shape_xyz,
        device=future_ego_to_t0.device,
        dtype=future_ego_to_t0.dtype,
        native_origin_xyz_m=native_origin_xyz_m,
        native_voxel_size_xyz_m=native_voxel_size_xyz_m,
    )
    T = future_ego_to_t0[tile.window_index, tile.horizon]
    return torch.einsum("...j,ij->...i", xyz, T[:3, :3]) + T[:3, 3]


def points_in_lattice(points_xyz: torch.Tensor, lattice: CanonicalLattice) -> torch.Tensor:
    origin = torch.as_tensor(lattice.origin_xyz_m, device=points_xyz.device, dtype=points_xyz.dtype)
    maximum = origin + torch.as_tensor(
        lattice.shape_xyz, device=points_xyz.device, dtype=points_xyz.dtype
    ) * torch.as_tensor(
        lattice.voxel_size_xyz_m, device=points_xyz.device, dtype=points_xyz.dtype
    )
    return ((points_xyz >= origin) & (points_xyz < maximum)).all(dim=-1)
