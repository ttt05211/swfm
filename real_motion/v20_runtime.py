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


@dataclass(frozen=True)
class V18BirthCondition:
    current_source_tokens: torch.Tensor
    current_source_xyz_norm: torch.Tensor
    future_source_tokens: torch.Tensor
    future_source_xyz_norm: torch.Tensor
    future_source_xy_t0_m: torch.Tensor
    future_source_existence_prob: torch.Tensor
    current_source_class_id: torch.Tensor


def v18_birth_condition_from_record(
    v18,
    record,
    *,
    device: torch.device,
    amp: bool,
    position_scale_m: float = 40.0,
) -> V18BirthCondition:
    """Expose frozen V18 current/future source state for Birth conditioning."""
    scale = float(position_scale_m)
    if scale <= 0:
        raise ValueError("position_scale_m must be positive")
    def mv(name, dtype=None):
        x = record[name].to(device)
        return x.to(dtype) if dtype is not None else x
    with torch.no_grad(), (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if amp and device.type == "cuda" else torch.autocast("cpu", enabled=False)
    ):
        out = v18(
            mv("features", torch.float32),
            mv("local_semantic_tube"),
            mv("kta_displacement_xy_m", torch.float32),
            mv("frame_motion_features", torch.float32),
            mv("target_source_mask_tube"),
            return_latents=True,
        )
    current_tok = out["history_source_context"].float()
    future_tok = out["future_transport_queries"].float()
    N = int(current_tok.shape[0])
    if "source_centroid_xy_t0_m" in record:
        current_xy = record["source_centroid_xy_t0_m"].float().to(device)
    else:
        # Frozen feature contract: first two channels are current x/y / 40 m.
        current_xy = record["features"][:, :2].float().to(device) * 40.0
    if current_xy.shape != (N, 2):
        raise RuntimeError("V18 source centroid count mismatch")
    future_xy = (
        record["anchors_xy_t0_m"].float().to(device)
        + out["residual_xy_m"].float()
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
            out["existence_logits"].float()
        ),
        current_source_class_id=cls,
    )
