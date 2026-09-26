"""V20 runtime helpers shared by formal evaluation and latency measurement."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import torch

from .v20_history_world import (
    CanonicalLattice,
    DYNAMIC_IDS,
    QueryMaskReport,
    canonical_tile_grid_sample_coordinates,
    future_native_to_canonical_indices,
    future_union_query_mask,
    render_canonical_semantic_to_future,
)
from .v20_training import decode_static_logits


def sample_scene_features_at_t0_points(
    scene_features: torch.Tensor,
    points_xyz_t0_m: torch.Tensor,
    coarse_lattice: CanonicalLattice,
) -> torch.Tensor:
    """Trilinearly sample [N,C] local scene evidence at t0-canonical points."""
    import torch.nn.functional as F

    if scene_features.ndim != 5 or scene_features.shape[0] != 1:
        raise ValueError("scene_features must be [1,C,X,Y,Z]")
    pts = points_xyz_t0_m.to(device=scene_features.device, dtype=scene_features.dtype)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points_xyz_t0_m must be [N,3]")
    if pts.shape[0] == 0:
        return scene_features.new_empty((0, scene_features.shape[1]))
    origin = torch.as_tensor(
        coarse_lattice.origin_xyz_m, device=pts.device, dtype=pts.dtype
    )
    step = torch.as_tensor(
        coarse_lattice.voxel_size_xyz_m, device=pts.device, dtype=pts.dtype
    )
    shape = torch.as_tensor(
        coarse_lattice.shape_xyz, device=pts.device, dtype=pts.dtype
    )
    fidx = (pts - origin) / step - 0.5
    denom = torch.clamp(shape - 1.0, min=1.0)
    norm = 2.0 * fidx / denom - 1.0
    # Input is [D,H,W]=[X,Y,Z]; grid tuple is (W,H,D)=(Z,Y,X).
    grid = torch.stack((norm[:, 2], norm[:, 1], norm[:, 0]), dim=-1)
    grid = grid.view(1, pts.shape[0], 1, 1, 3)
    out = F.grid_sample(
        scene_features,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return out[0, :, :, 0, 0].transpose(0, 1)


@dataclass(frozen=True)
class StaticRuntimeReport:
    canonical_semantic: np.ndarray
    future_semantic: np.ndarray
    query_voxels: int
    active_tiles: int
    out_of_bounds_voxels: int
    render_index: object | None = None


@dataclass
class StaticRuntimeCache:
    """Geometry tensors invariant across V20 Static evaluation windows."""
    tile_grids: dict = field(default_factory=dict)
    tile_coarse_linear: dict = field(default_factory=dict)


def _runtime_tile_grid(
    cache,
    high_lattice,
    coarse_lattice,
    start_t,
    tshape,
    *,
    device,
    dtype,
):
    key = (
        tuple(int(x) for x in start_t),
        tuple(int(x) for x in tshape),
        str(device),
        str(dtype),
    )
    if cache is not None and key in cache.tile_grids:
        return cache.tile_grids[key]
    grid = canonical_tile_grid_sample_coordinates(
        high_lattice,
        coarse_lattice,
        start_t,
        tshape,
        device=device,
        dtype=dtype,
    )
    if cache is not None:
        cache.tile_grids[key] = grid
    return grid


def _runtime_tile_coarse_linear(
    cache,
    high_lattice,
    coarse_lattice,
    start_t,
    tshape,
):
    key = (
        tuple(int(x) for x in start_t),
        tuple(int(x) for x in tshape),
    )
    if cache is not None and key in cache.tile_coarse_linear:
        return cache.tile_coarse_linear[key]
    start = np.asarray(start_t, dtype=np.int64)
    shape = np.asarray(tshape, dtype=np.int64)
    hi_step = np.asarray(high_lattice.voxel_size_xyz_m, dtype=np.float64)
    co_step = np.asarray(coarse_lattice.voxel_size_xyz_m, dtype=np.float64)
    factor = np.maximum(np.rint(co_step / hi_step).astype(np.int64), 1)
    x = np.clip(
        np.arange(start[0], start[0] + shape[0]) // factor[0],
        0, coarse_lattice.shape_xyz[0] - 1,
    )
    y = np.clip(
        np.arange(start[1], start[1] + shape[1]) // factor[1],
        0, coarse_lattice.shape_xyz[1] - 1,
    )
    z = np.clip(
        np.arange(start[2], start[2] + shape[2]) // factor[2],
        0, coarse_lattice.shape_xyz[2] - 1,
    )
    Y, Z = int(coarse_lattice.shape_xyz[1]), int(coarse_lattice.shape_xyz[2])
    linear = (
        x[:, None, None] * (Y * Z)
        + y[None, :, None] * Z
        + z[None, None, :]
    ).reshape(-1)
    if cache is not None:
        cache.tile_coarse_linear[key] = linear
    return linear


def _query_mask_from_render_index(
    high_lattice: CanonicalLattice,
    render_index,
) -> QueryMaskReport:
    valid = np.asarray(render_index.valid, dtype=bool)
    out = np.zeros(high_lattice.shape_xyz, dtype=bool)
    linear = getattr(render_index, "linear_index", None)
    if linear is not None:
        lin = np.asarray(linear)
        if bool(valid.all()):
            out.reshape(-1)[lin.reshape(-1)] = True
        else:
            out.reshape(-1)[lin[valid]] = True
    else:
        idx = np.asarray(render_index.indices_xyz, dtype=np.int64)
        good = idx[valid]
        if len(good):
            out[good[:, 0], good[:, 1], good[:, 2]] = True
    requested = int(valid.size)
    in_bounds = int(valid.sum())
    oob = int(requested - in_bounds)
    return QueryMaskReport(
        mask=out,
        requested_voxels=requested,
        in_bounds_voxels=in_bounds,
        out_of_bounds_voxels=oob,
        out_of_bounds_fraction=float(oob / max(requested, 1)),
    )


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
    tile_batch_size: int = 32,
    free_label: int = 17,
    runtime_cache: StaticRuntimeCache | None = None,
) -> StaticRuntimeReport:
    """Decode one canonical Static world and render all six futures.

    The future native->canonical mapping is computed once and reused for both
    the query union and final render. Equal-shaped active tiles are refined in
    batches to avoid one Conv3D launch and GPU->CPU sync per tile.
    """
    if scene_features.shape[0] != 1:
        raise ValueError("runtime helper currently expects one window")

    ri = future_native_to_canonical_indices(
        high_lattice,
        future_ego_to_canonical=np.asarray(future_ego_to_canonical),
        native_shape_xyz=native_shape_xyz,
        native_origin_xyz_m=native_origin_xyz_m,
        native_voxel_size_xyz_m=native_voxel_size_xyz_m,
    )
    q = _query_mask_from_render_index(high_lattice, ri)

    world = np.full(high_lattice.shape_xyz, int(free_label), dtype=np.uint8)
    tile = np.asarray(tuple(int(x) for x in tile_size_xyz), dtype=np.int64)
    shape = np.asarray(high_lattice.shape_xyz, dtype=np.int64)
    starts = []
    buckets = {}
    for x in range(0, shape[0], tile[0]):
        for y in range(0, shape[1], tile[1]):
            for z in range(0, shape[2], tile[2]):
                start = np.asarray([x, y, z], dtype=np.int64)
                stop = np.minimum(start + tile, shape)
                if not q.mask[x:stop[0], y:stop[1], z:stop[2]].any():
                    continue
                start_t = (int(x), int(y), int(z))
                tshape = tuple((stop - start).tolist())
                starts.append(start_t)
                buckets.setdefault(tshape, []).append(start_t)

    bsz = max(int(tile_batch_size), 1)
    obs_arr = np.asarray(history_observed_coarse, dtype=bool)
    if obs_arr.shape != (6,) + tuple(coarse_lattice.shape_xyz):
        raise ValueError("history_observed_coarse shape mismatch")
    seen_coarse = obs_arr.any(axis=0).reshape(-1)
    t0_coarse = obs_arr[-1].reshape(-1)

    with torch.inference_mode():
        for tshape, bucket in buckets.items():
            for bi in range(0, len(bucket), bsz):
                chunk = bucket[bi:bi + bsz]
                grids = []
                qtiles = []
                seen_rows = []
                missing_rows = []
                stops = []
                for start_t in chunk:
                    start = np.asarray(start_t, dtype=np.int64)
                    stop = np.minimum(start + tile, shape)
                    stops.append(stop)
                    grids.append(
                        _runtime_tile_grid(
                            runtime_cache,
                            high_lattice,
                            coarse_lattice,
                            start_t,
                            tshape,
                            device=scene_features.device,
                            dtype=scene_features.dtype,
                        )
                    )
                    qtile = q.mask[
                        start[0]:stop[0],
                        start[1]:stop[1],
                        start[2]:stop[2],
                    ]
                    cmap = _runtime_tile_coarse_linear(
                        runtime_cache,
                        high_lattice,
                        coarse_lattice,
                        start_t,
                        tshape,
                    )
                    seen = seen_coarse[cmap].reshape(tshape)
                    t0_seen = t0_coarse[cmap].reshape(tshape)
                    missing = seen & ~t0_seen
                    qtiles.append(qtile)
                    seen_rows.append(seen)
                    missing_rows.append(missing)

                B = len(chunk)
                logits = model.static.refine_tiles(
                    scene_features,
                    sample_grid=torch.cat(grids, dim=0),
                    query_mask=torch.from_numpy(
                        np.stack(qtiles, axis=0)
                    ).to(scene_features.device, non_blocking=True),
                    seen_mask=torch.from_numpy(
                        np.stack(seen_rows, axis=0)
                    ).to(scene_features.device, non_blocking=True),
                    t0_missing_mask=torch.from_numpy(
                        np.stack(missing_rows, axis=0)
                    ).to(scene_features.device, non_blocking=True),
                )
                pred_batch = (
                    decode_static_logits(logits)
                    .cpu()
                    .numpy()
                    .astype(np.uint8, copy=False)
                )
                for local_b, (start_t, stop, qtile) in enumerate(
                    zip(chunk, stops, qtiles)
                ):
                    pred = pred_batch[local_b].copy()
                    pred[~qtile] = int(free_label)
                    start = np.asarray(start_t, dtype=np.int64)
                    world[
                        start[0]:stop[0],
                        start[1]:stop[1],
                        start[2]:stop[2],
                    ] = pred

    future = render_canonical_semantic_to_future(
        world, ri, free_label=int(free_label)
    )
    return StaticRuntimeReport(
        canonical_semantic=world,
        future_semantic=future,
        query_voxels=int(q.mask.sum()),
        active_tiles=int(len(starts)),
        out_of_bounds_voxels=int(ri.out_of_bounds_voxels),
        render_index=ri,
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
    future_render_index=None,
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
    ri = future_render_index
    if ri is None:
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


@dataclass(frozen=True)
class V18BirthCondition:
    current_source_tokens: torch.Tensor
    current_source_xyz_norm: torch.Tensor
    future_source_tokens: torch.Tensor
    future_source_xyz_norm: torch.Tensor
    future_source_xy_t0_m: torch.Tensor
    future_source_existence_prob: torch.Tensor
    current_source_class_id: torch.Tensor


def v18_birth_condition_from_output(
    record: Mapping[str, object],
    output: Mapping[str, torch.Tensor],
    *,
    device: torch.device,
    position_scale_m: float = 40.0,
) -> V18BirthCondition:
    """Convert one already-computed frozen V18 output into Birth context."""
    scale = float(position_scale_m)
    if scale <= 0:
        raise ValueError("position_scale_m must be positive")
    for key in (
        "history_source_context",
        "future_transport_queries",
        "residual_xy_m",
        "existence_logits",
    ):
        if key not in output:
            raise KeyError(f"V18 latent output missing {key}")
    current_tok = output["history_source_context"].float()
    future_tok = output["future_transport_queries"].float()
    N = int(current_tok.shape[0])
    if "source_centroid_xy_t0_m" in record:
        current_xy = record["source_centroid_xy_t0_m"].float().to(device)
    else:
        current_xy = record["features"][:, :2].float().to(device) * 40.0
    if current_xy.shape != (N, 2):
        raise RuntimeError("V18 source centroid count mismatch")
    future_xy = (
        record["anchors_xy_t0_m"].float().to(device)
        + output["residual_xy_m"].float()
    )
    if future_xy.shape != (N, 6, 2):
        raise RuntimeError("V18 future source position shape mismatch")
    cur_xyz = torch.zeros((N, 3), dtype=torch.float32, device=device)
    cur_xyz[:, :2] = current_xy
    fut_xyz = torch.zeros((N, 6, 3), dtype=torch.float32, device=device)
    fut_xyz[..., :2] = future_xy
    cls = record["source_class_id"].long().to(device)
    return V18BirthCondition(
        current_source_tokens=current_tok.unsqueeze(0),
        current_source_xyz_norm=(cur_xyz / scale).unsqueeze(0),
        future_source_tokens=future_tok.unsqueeze(0),
        future_source_xyz_norm=(fut_xyz / scale).unsqueeze(0),
        future_source_xy_t0_m=future_xy,
        future_source_existence_prob=torch.sigmoid(
            output["existence_logits"].float()
        ),
        current_source_class_id=cls,
    )


def v18_birth_condition_from_record(
    v18,
    record,
    *,
    device: torch.device,
    amp: bool,
    position_scale_m: float = 40.0,
) -> V18BirthCondition:
    """Run frozen V18 once and expose current/future source state for Birth."""
    def mv(name, dtype=None):
        x = record[name].to(device)
        return x.to(dtype) if dtype is not None else x
    ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if amp and device.type == "cuda" else nullcontext()
    )
    with torch.no_grad(), ctx:
        out = v18(
            mv("features", torch.float32),
            mv("local_semantic_tube"),
            mv("kta_displacement_xy_m", torch.float32),
            mv("frame_motion_features", torch.float32),
            mv("target_source_mask_tube"),
            return_latents=True,
        )
    return v18_birth_condition_from_output(
        record,
        out,
        device=device,
        position_scale_m=position_scale_m,
    )


def v18_source_tokens_from_arrays(
    v18,
    arrays: Mapping[str, torch.Tensor],
    *,
    device: torch.device,
    amp: bool,
) -> torch.Tensor:
    """Frozen V18 history-source context for arbitrary causal source arrays."""
    def mv(name, dtype=None):
        x = arrays[name].to(device)
        return x.to(dtype) if dtype is not None else x
    ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if amp and device.type == "cuda" else nullcontext()
    )
    with torch.no_grad(), ctx:
        out = v18(
            mv("features", torch.float32),
            mv("local_semantic_tube"),
            mv("kta_displacement_xy_m", torch.float32),
            mv("frame_motion_features", torch.float32),
            mv("target_source_mask_tube"),
            return_latents=True,
        )
    return out["history_source_context"].float()
