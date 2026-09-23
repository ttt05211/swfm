"""Residual scene-innovation utilities for V19-MI.

The innovation branch is deliberately subordinate to memory + transport:
it predicts only add-only occupancy in regions not already explained by the
base forecast.  Historical geometry is aligned explicitly into each future ego
frame before learning; the network is therefore not asked to learn known ego
motion.

This file contains:
  * future-frame aligned history BEV construction using Occ3D visibility;
  * a small temporal CNN innovation head;
  * masked innovation losses;
  * protected add-only 3D decoding/composition helpers.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import (
    _occupied_indices_to_xyz,
    _xyz_to_indices,
    relative_transform,
    warp_semantic_and_mask,
)
from .local_st_world_model import SEMANTIC_CLASSES
from .motion_transport import FUTURE_FRAMES, HISTORY_FRAMES


INNOVATION_PROTOCOL = "v19_residual_innovation_future_aligned_bev_v1"
GEOMETRY_CHANNELS = 4  # coverage, top-height, bottom-height, occupied-count


def _future_aligned_bev_summary(
    semantics: np.ndarray,
    observed: np.ndarray,
    src_pose: np.ndarray,
    future_pose: np.ndarray,
    *,
    grid,
    free_label: int,
    return_aligned_observed: bool = False,
):
    """Return top semantic + geometric summary in one future ego frame."""
    sem = np.asarray(semantics, dtype=np.uint8)
    obs = np.asarray(observed, dtype=bool)
    if sem.shape != tuple(grid.shape_hwd) or obs.shape != sem.shape:
        raise ValueError("semantic/observed grid shape mismatch")

    T = relative_transform(
        np.asarray(src_pose, dtype=np.float64),
        np.asarray(future_pose, dtype=np.float64),
    )
    # Warp semantic evidence and lidar coverage with one coordinate transform.
    # This preserves the historical semantic collision rule while avoiding the
    # second full mask warp.
    aligned_sem, aligned_obs = warp_semantic_and_mask(
        sem,
        obs,
        T,
        grid=grid,
        free_label=int(free_label),
    )

    occupied = (aligned_sem != int(free_label)) & aligned_obs
    has = occupied.any(axis=2)
    X, Y, Z = aligned_sem.shape

    top_label = np.full((X, Y), int(free_label), dtype=np.uint8)
    top_h = np.zeros((X, Y), dtype=np.float32)
    bottom_h = np.zeros((X, Y), dtype=np.float32)
    count = occupied.sum(axis=2).astype(np.float32)

    if bool(has.any()):
        rev = occupied[:, :, ::-1]
        top_from_back = np.argmax(rev, axis=2)
        z_top = Z - 1 - top_from_back
        z_bottom = np.argmax(occupied, axis=2)
        ix, iy = np.nonzero(has)
        top_label[ix, iy] = aligned_sem[ix, iy, z_top[ix, iy]]
        denom = max(Z - 1, 1)
        top_h[ix, iy] = z_top[ix, iy].astype(np.float32) / float(denom)
        bottom_h[ix, iy] = z_bottom[ix, iy].astype(np.float32) / float(denom)

    coverage = aligned_obs.any(axis=2).astype(np.float32)
    count = count / float(max(Z, 1))
    geom = np.stack((coverage, top_h, bottom_h, count), axis=0).astype(np.float32)
    if return_aligned_observed:
        return top_label, geom, aligned_obs
    return top_label, geom


def build_future_aligned_history_bev(
    history_semantics: Sequence[np.ndarray],
    history_observed: Sequence[np.ndarray],
    history_poses: Sequence[np.ndarray],
    future_poses: Sequence[np.ndarray],
    *,
    grid,
    free_label: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build [F,T,H,W] semantic and [F,T,4,H,W] geometric history tensors."""
    if not (
        len(history_semantics)
        == len(history_observed)
        == len(history_poses)
        == HISTORY_FRAMES
    ):
        raise ValueError("expected six history semantic/obs/pose frames")
    if len(future_poses) != FUTURE_FRAMES:
        raise ValueError("expected six future poses")

    labels = []
    geometry = []
    for fpose in future_poses:
        lf, gf = [], []
        for sem, obs, hpose in zip(
            history_semantics, history_observed, history_poses
        ):
            lab, geo = _future_aligned_bev_summary(
                sem,
                obs,
                hpose,
                fpose,
                grid=grid,
                free_label=int(free_label),
            )
            lf.append(lab)
            gf.append(geo)
        labels.append(np.stack(lf, axis=0))
        geometry.append(np.stack(gf, axis=0))
    return (
        np.stack(labels, axis=0).astype(np.uint8),
        np.stack(geometry, axis=0).astype(np.float32),
    )


def build_future_aligned_history_bev_with_coverage(
    history_semantics: Sequence[np.ndarray],
    history_observed: Sequence[np.ndarray],
    history_poses: Sequence[np.ndarray],
    future_poses: Sequence[np.ndarray],
    *,
    grid,
    free_label: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build Innovation inputs plus exact 3D historical observation coverage.

    The coverage is accumulated from the same fused warps used to build the
    network inputs, avoiding a second 6x6 mask-warp pass in cache generation.
    """
    if not (
        len(history_semantics)
        == len(history_observed)
        == len(history_poses)
        == HISTORY_FRAMES
    ):
        raise ValueError("expected six history semantic/obs/pose frames")
    if len(future_poses) != FUTURE_FRAMES:
        raise ValueError("expected six future poses")

    labels = []
    geometry = []
    coverage_3d = []
    for fpose in future_poses:
        lf, gf = [], []
        cov = np.zeros(tuple(grid.shape_hwd), dtype=bool)
        for sem, obs, hpose in zip(
            history_semantics, history_observed, history_poses
        ):
            lab, geo, aligned_obs = _future_aligned_bev_summary(
                sem,
                obs,
                hpose,
                fpose,
                grid=grid,
                free_label=int(free_label),
                return_aligned_observed=True,
            )
            lf.append(lab)
            gf.append(geo)
            cov |= aligned_obs
        labels.append(np.stack(lf, axis=0))
        geometry.append(np.stack(gf, axis=0))
        coverage_3d.append(cov)
    return (
        np.stack(labels, axis=0).astype(np.uint8),
        np.stack(geometry, axis=0).astype(np.float32),
        np.stack(coverage_3d, axis=0),
    )


def _semantic_scatter_from_pretransformed(
    labels: np.ndarray,
    ix: np.ndarray,
    iy: np.ndarray,
    iz: np.ndarray,
    dst_xyz: np.ndarray,
    valid: np.ndarray,
    select: np.ndarray,
    *,
    grid,
    free_label: int,
) -> np.ndarray:
    """Exact semantic collision resolution after a shared point transform."""
    out = np.full(tuple(grid.shape_hwd), int(free_label), dtype=np.uint8)
    keep = np.asarray(valid, dtype=bool) & np.asarray(select, dtype=bool)
    if not bool(keep.any()):
        return out
    vals = np.asarray(labels, dtype=np.uint8)[keep]
    sx = np.asarray(ix)[keep]
    sy = np.asarray(iy)[keep]
    sz = np.asarray(iz)[keep]
    pts = np.asarray(dst_xyz)[keep]

    occupied = vals != int(free_label)
    if not bool(occupied.any()):
        return out
    vals = vals[occupied]
    sx, sy, sz = sx[occupied], sy[occupied], sz[occupied]
    pts = pts[occupied]

    vx, vy, vz = grid.voxel_size
    centers = np.stack(
        [
            grid.x_min + (sx + 0.5) * vx,
            grid.y_min + (sy + 0.5) * vy,
            grid.z_min + (sz + 0.5) * vz,
        ],
        axis=1,
    )
    dist2 = np.sum((pts - centers) ** 2, axis=1)
    _, Y, Z = grid.shape_hwd
    flat = (sx * Y + sy) * Z + sz
    order = np.lexsort((dist2, flat))
    flat_sorted = flat[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = flat_sorted[1:] != flat_sorted[:-1]
    chosen = order[first]
    out[sx[chosen], sy[chosen], sz[chosen]] = vals[chosen]
    return out


def build_future_aligned_history_and_static_memory(
    history_semantics: Sequence[np.ndarray],
    history_observed: Sequence[np.ndarray],
    history_poses: Sequence[np.ndarray],
    future_poses: Sequence[np.ndarray],
    *,
    grid,
    free_label: int,
    dynamic_class_ids: Sequence[int],
    workers: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exact fused V19 history alignment + deterministic Static Memory render.

    Each observed history frame is sparsified once.  For each history/future
    pair the point coordinates are transformed only once and reused for:
      * Innovation semantic/geometric history input;
      * full lidar-observation coverage;
      * static-memory free clearing and static semantic rendering.

    The outputs are bit-equivalent to the previous separate
    build_future_aligned_history_bev_with_coverage +
    render_static_history_mosaic path.
    """
    if not (
        len(history_semantics)
        == len(history_observed)
        == len(history_poses)
        == HISTORY_FRAMES
    ):
        raise ValueError("expected six history semantic/obs/pose frames")
    if len(future_poses) != FUTURE_FRAMES:
        raise ValueError("expected six future poses")

    dynamic_ids = np.asarray(tuple(int(x) for x in dynamic_class_ids), dtype=np.uint8)
    prepared = []
    for sem, obs, hpose in zip(
        history_semantics, history_observed, history_poses
    ):
        sem = np.asarray(sem, dtype=np.uint8)
        obs = np.asarray(obs, dtype=bool)
        if sem.shape != tuple(grid.shape_hwd) or obs.shape != sem.shape:
            raise ValueError("semantic/observed grid shape mismatch")
        idx = np.argwhere(obs)
        if len(idx):
            xyz = _occupied_indices_to_xyz(idx, grid)
            labels = sem[tuple(idx.T)]
            usable_static = ~np.isin(labels, dynamic_ids)
        else:
            xyz = np.zeros((0, 3), dtype=np.float64)
            labels = np.zeros((0,), dtype=np.uint8)
            usable_static = np.zeros((0,), dtype=bool)
        prepared.append(
            (
                np.asarray(hpose, dtype=np.float64),
                xyz,
                labels,
                usable_static,
            )
        )

    nworkers = max(1, int(workers))
    def _one_future(fpose):
        fpose = np.asarray(fpose, dtype=np.float64)
        lf, gf = [], []
        coverage = np.zeros(tuple(grid.shape_hwd), dtype=bool)
        static_out = np.full(
            tuple(grid.shape_hwd), int(free_label), dtype=np.uint8
        )

        for hpose, xyz, labels, usable_static in prepared:
            if len(xyz) == 0:
                aligned_sem = np.full(
                    tuple(grid.shape_hwd), int(free_label), dtype=np.uint8
                )
                aligned_obs = np.zeros(tuple(grid.shape_hwd), dtype=bool)
                static_sem = aligned_sem.copy()
                static_obs = aligned_obs.copy()
            else:
                T = relative_transform(hpose, fpose)
                dst_xyz = xyz @ T[:3, :3].T + T[:3, 3]
                ix, iy, iz, valid = _xyz_to_indices(dst_xyz, grid)

                aligned_obs = np.zeros(tuple(grid.shape_hwd), dtype=bool)
                if bool(valid.any()):
                    aligned_obs[ix[valid], iy[valid], iz[valid]] = True
                aligned_sem = _semantic_scatter_from_pretransformed(
                    labels,
                    ix,
                    iy,
                    iz,
                    dst_xyz,
                    valid,
                    np.ones(len(labels), dtype=bool),
                    grid=grid,
                    free_label=int(free_label),
                )

                static_valid = valid & usable_static
                static_obs = np.zeros(tuple(grid.shape_hwd), dtype=bool)
                if bool(static_valid.any()):
                    static_obs[
                        ix[static_valid],
                        iy[static_valid],
                        iz[static_valid],
                    ] = True
                static_sem = _semantic_scatter_from_pretransformed(
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

            occupied = (aligned_sem != int(free_label)) & aligned_obs
            has = occupied.any(axis=2)
            X, Y, Z = aligned_sem.shape
            top_label = np.full(
                (X, Y), int(free_label), dtype=np.uint8
            )
            top_h = np.zeros((X, Y), dtype=np.float32)
            bottom_h = np.zeros((X, Y), dtype=np.float32)
            count = occupied.sum(axis=2).astype(np.float32)
            if bool(has.any()):
                rev = occupied[:, :, ::-1]
                z_top = Z - 1 - np.argmax(rev, axis=2)
                z_bottom = np.argmax(occupied, axis=2)
                ax, ay = np.nonzero(has)
                top_label[ax, ay] = aligned_sem[
                    ax, ay, z_top[ax, ay]
                ]
                denom = max(Z - 1, 1)
                top_h[ax, ay] = (
                    z_top[ax, ay].astype(np.float32) / float(denom)
                )
                bottom_h[ax, ay] = (
                    z_bottom[ax, ay].astype(np.float32) / float(denom)
                )
            geom = np.stack(
                (
                    aligned_obs.any(axis=2).astype(np.float32),
                    top_h,
                    bottom_h,
                    count / float(max(Z, 1)),
                ),
                axis=0,
            ).astype(np.float32)
            lf.append(top_label)
            gf.append(geom)
            coverage |= aligned_obs

            # Exact oldest->newest static mosaic semantics.
            static_out[static_obs] = int(free_label)
            static_occ = static_sem != int(free_label)
            static_out[static_occ] = static_sem[static_occ]

        return (
            np.stack(lf, axis=0).astype(np.uint8),
            np.stack(gf, axis=0).astype(np.float32),
            coverage,
            static_out.astype(np.uint8),
        )


    if nworkers == 1:
        rows = [_one_future(fpose) for fpose in future_poses]
    else:
        with ThreadPoolExecutor(
            max_workers=min(nworkers, len(future_poses))
        ) as pool:
            rows = list(pool.map(_one_future, future_poses))

    all_labels, all_geometry, all_coverage, all_static = zip(*rows)
    return (
        np.stack(all_labels, axis=0).astype(np.uint8),
        np.stack(all_geometry, axis=0).astype(np.float32),
        np.stack(all_coverage, axis=0),
        np.stack(all_static, axis=0).astype(np.uint8),
    )


def base_explained_bev(
    base_future_occ: np.ndarray,
    *,
    free_label: int,
) -> np.ndarray:
    """Occupied support of an already-composed base forecast."""
    x = np.asarray(base_future_occ)
    if x.ndim != 4:
        raise ValueError("base_future_occ must be [F,X,Y,Z]")
    return (x != int(free_label)).any(axis=3).astype(np.float32)[:, None]


class ResidualInnovationHead(nn.Module):
    """Small future-aligned temporal CNN.

    It processes each history frame with a shared stride-2 stem, mixes the six
    temporal states with depthwise temporal convolution, injects the base
    explained map and future-time embedding, and upsamples once to the native
    BEV resolution.
    """

    def __init__(
        self,
        *,
        future_frames: int = FUTURE_FRAMES,
        history_frames: int = HISTORY_FRAMES,
        semantic_dim: int = 8,
        hidden_dim: int = 32,
        num_semantic_classes: int = 17,
        vertical_bins: int = 16,
    ):
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.future_frames = int(future_frames)
        self.history_frames = int(history_frames)
        self.num_semantic_classes = int(num_semantic_classes)
        self.vertical_bins = int(vertical_bins)
        self.semantic_embedding = nn.Embedding(
            SEMANTIC_CLASSES, int(semantic_dim)
        )
        in_frame = int(semantic_dim) + GEOMETRY_CHANNELS
        self.frame_stem = nn.Sequential(
            nn.Conv2d(in_frame, hidden_dim, 3, stride=2, padding=1),
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),
        )
        self.temporal_dw = nn.Conv3d(
            hidden_dim,
            hidden_dim,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0),
            groups=hidden_dim,
        )
        self.temporal_pw = nn.Conv3d(
            hidden_dim, hidden_dim, kernel_size=1
        )
        self.temporal_norm = nn.GroupNorm(1, hidden_dim)
        self.base_proj = nn.Sequential(
            nn.Conv2d(1, hidden_dim, 3, stride=2, padding=1),
            nn.GroupNorm(1, hidden_dim),
        )
        self.future_time_embedding = nn.Parameter(
            torch.zeros(1, self.future_frames, hidden_dim)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),
        )
        out_ch = 1 + self.num_semantic_classes + self.vertical_bins
        self.out_head = nn.Conv2d(hidden_dim, out_ch, 1)

        nn.init.trunc_normal_(self.future_time_embedding, std=0.02)
        # Conservative initialization: no innovation is preferred initially.
        nn.init.zeros_(self.out_head.weight)
        nn.init.zeros_(self.out_head.bias)
        with torch.no_grad():
            self.out_head.bias[0] = -4.0

    def forward(
        self,
        future_aligned_semantic: torch.Tensor,
        future_aligned_geometry: torch.Tensor,
        base_explained: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        lab = future_aligned_semantic.long()
        geo = future_aligned_geometry
        if lab.ndim != 5:
            raise ValueError("future_aligned_semantic must be [B,F,T,H,W]")
        B, Fh, T, H, W = lab.shape
        if Fh != self.future_frames or T != self.history_frames:
            raise ValueError("future/history frame count mismatch")
        if geo.shape != (B, Fh, T, GEOMETRY_CHANNELS, H, W):
            raise ValueError(
                "future_aligned_geometry must be [B,F,T,4,H,W]"
            )
        if base_explained.shape != (B, Fh, 1, H, W):
            raise ValueError("base_explained must be [B,F,1,H,W]")
        if bool((lab < 0).any()) or bool((lab >= SEMANTIC_CLASSES).any()):
            raise ValueError("semantic labels outside [0,17]")

        emb = self.semantic_embedding(lab)
        emb = emb.permute(0, 1, 2, 5, 3, 4)
        x = torch.cat((emb, geo.to(emb.dtype)), dim=3)
        x = x.reshape(
            B * Fh * T, x.shape[3], H, W
        )
        x = self.frame_stem(x)
        H2, W2 = x.shape[-2:]
        x = x.reshape(B * Fh, T, -1, H2, W2).permute(
            0, 2, 1, 3, 4
        )
        x = self.temporal_pw(self.temporal_dw(x))
        x = x.mean(dim=2)
        x = self.temporal_norm(x)

        base = base_explained.reshape(B * Fh, 1, H, W).to(x.dtype)
        x = x + self.base_proj(base)
        time = self.future_time_embedding.expand(B, -1, -1).reshape(
            B * Fh, -1
        )
        x = x + time[:, :, None, None].to(x.dtype)
        x = self.decoder(x)
        raw = self.out_head(x).reshape(B, Fh, -1, H, W)

        s0 = 1
        s1 = s0 + self.num_semantic_classes
        return {
            "add_presence_logits": raw[:, :, 0],
            "semantic_logits": raw[:, :, s0:s1],
            "vertical_occupancy_logits": raw[:, :, s1:],
        }



class ResidualInnovationIntervalHead(ResidualInnovationHead):
    """Innovation head with compact vertical interval geometry.

    The spatial/temporal trunk is identical to ResidualInnovationHead. Instead
    of 16 independent occupancy bits, each positive BEV cell predicts:
      * bottom z-bin (Z-way classification);
      * span length 1..Z (Z-way classification).

    This guarantees a contiguous decoded vertical extent.
    """

    def __init__(
        self,
        *,
        future_frames: int = FUTURE_FRAMES,
        history_frames: int = HISTORY_FRAMES,
        semantic_dim: int = 8,
        hidden_dim: int = 32,
        num_semantic_classes: int = 17,
        vertical_bins: int = 16,
    ):
        super().__init__(
            future_frames=future_frames,
            history_frames=history_frames,
            semantic_dim=semantic_dim,
            hidden_dim=hidden_dim,
            num_semantic_classes=num_semantic_classes,
            vertical_bins=vertical_bins,
        )
        out_ch = 1 + self.num_semantic_classes + 2 * self.vertical_bins
        self.out_head = nn.Conv2d(hidden_dim, out_ch, 1)
        nn.init.zeros_(self.out_head.weight)
        nn.init.zeros_(self.out_head.bias)
        with torch.no_grad():
            self.out_head.bias[0] = -4.0

    def forward(
        self,
        future_aligned_semantic: torch.Tensor,
        future_aligned_geometry: torch.Tensor,
        base_explained: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        lab = future_aligned_semantic.long()
        geo = future_aligned_geometry
        if lab.ndim != 5:
            raise ValueError("future_aligned_semantic must be [B,F,T,H,W]")
        B, Fh, T, H, W = lab.shape
        if Fh != self.future_frames or T != self.history_frames:
            raise ValueError("future/history frame count mismatch")
        if geo.shape != (B, Fh, T, GEOMETRY_CHANNELS, H, W):
            raise ValueError(
                "future_aligned_geometry must be [B,F,T,4,H,W]"
            )
        if base_explained.shape != (B, Fh, 1, H, W):
            raise ValueError("base_explained must be [B,F,1,H,W]")
        if bool((lab < 0).any()) or bool((lab >= SEMANTIC_CLASSES).any()):
            raise ValueError("semantic labels outside [0,17]")

        emb = self.semantic_embedding(lab)
        emb = emb.permute(0, 1, 2, 5, 3, 4)
        x = torch.cat((emb, geo.to(emb.dtype)), dim=3)
        x = x.reshape(B * Fh * T, x.shape[3], H, W)
        x = self.frame_stem(x)
        H2, W2 = x.shape[-2:]
        x = x.reshape(B * Fh, T, -1, H2, W2).permute(
            0, 2, 1, 3, 4
        )
        x = self.temporal_pw(self.temporal_dw(x))
        x = x.mean(dim=2)
        x = self.temporal_norm(x)

        base = base_explained.reshape(B * Fh, 1, H, W).to(x.dtype)
        x = x + self.base_proj(base)
        time = self.future_time_embedding.expand(B, -1, -1).reshape(
            B * Fh, -1
        )
        x = x + time[:, :, None, None].to(x.dtype)
        x = self.decoder(x)
        raw = self.out_head(x).reshape(B, Fh, -1, H, W)

        s0 = 1
        s1 = s0 + self.num_semantic_classes
        s2 = s1 + self.vertical_bins
        return {
            "add_presence_logits": raw[:, :, 0],
            "semantic_logits": raw[:, :, s0:s1],
            "bottom_logits": raw[:, :, s1:s2],
            "span_logits": raw[:, :, s2:],
        }


def innovation_interval_loss(
    outputs: dict[str, torch.Tensor],
    *,
    add_target: torch.Tensor,
    semantic_target: torch.Tensor,
    vertical_target: torch.Tensor,
    candidate_mask: torch.Tensor,
    weights = None,
    presence_hard_negative_ratio: float = 4.0,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Hard-negative presence + semantic + bottom/span interval objective."""
    if weights is None:
        weights = InnovationLossWeights()
    add_logits = outputs["add_presence_logits"]
    sem_logits = outputs["semantic_logits"]
    bottom_logits = outputs["bottom_logits"]
    span_logits = outputs["span_logits"]
    pos = add_target.bool()
    cand = candidate_mask.bool()
    if add_logits.shape != pos.shape or cand.shape != pos.shape:
        raise ValueError("presence/candidate shape mismatch")
    if semantic_target.shape != pos.shape:
        raise ValueError("semantic_target shape mismatch")
    if vertical_target.ndim != 5:
        raise ValueError("vertical_target must be [B,F,Z,H,W]")
    Z = int(vertical_target.shape[2])
    if bottom_logits.shape[2] != Z or span_logits.shape[2] != Z:
        raise ValueError("interval logits/vertical target bin mismatch")

    ratio = float(presence_hard_negative_ratio)
    if ratio <= 0:
        raise ValueError("presence_hard_negative_ratio must be positive")
    pos_rows = add_logits[pos & cand]
    neg_rows = add_logits[cand & ~pos]
    pos_loss = F.softplus(-pos_rows)
    neg_loss_all = F.softplus(neg_rows)
    npos = int(pos_rows.numel())
    nneg = int(neg_rows.numel())
    hard_negative_count = 0
    if npos > 0 and nneg > 0:
        hard_negative_count = min(
            nneg,
            max(1, int(np.ceil(ratio * npos))),
        )
        neg_loss = torch.topk(
            neg_loss_all,
            k=hard_negative_count,
            largest=True,
            sorted=False,
        ).values
        add = torch.cat((pos_loss, neg_loss), dim=0).mean()
    elif npos > 0:
        add = pos_loss.mean()
    elif nneg > 0:
        add = neg_loss_all.mean()
        hard_negative_count = nneg
    else:
        add = add_logits.sum() * 0.0

    if bool(pos.any()):
        sem_rows = sem_logits.permute(0, 1, 3, 4, 2)[pos]
        sem = F.cross_entropy(
            sem_rows,
            semantic_target[pos].long(),
        )
        z_rows = (
            vertical_target.permute(0, 1, 3, 4, 2)[pos].bool()
        )
        if not bool(z_rows.any(dim=1).all()):
            raise RuntimeError("positive BEV cell without vertical target")
        idx = torch.arange(Z, device=z_rows.device)[None]
        bottom_target = torch.where(
            z_rows,
            idx,
            torch.full_like(idx, Z),
        ).min(dim=1).values
        top_target = torch.where(
            z_rows,
            idx,
            torch.full_like(idx, -1),
        ).max(dim=1).values
        span_target = (top_target - bottom_target).long()

        bottom_rows = bottom_logits.permute(0, 1, 3, 4, 2)[pos]
        span_rows = span_logits.permute(0, 1, 3, 4, 2)[pos]
        bottom = F.cross_entropy(bottom_rows, bottom_target.long())
        span = F.cross_entropy(span_rows, span_target)
        vertical = 0.5 * (bottom + span)
    else:
        sem = sem_logits.sum() * 0.0
        bottom = bottom_logits.sum() * 0.0
        span = span_logits.sum() * 0.0
        vertical = bottom + span

    total = (
        add
        + float(weights.semantic) * sem
        + float(weights.vertical) * vertical
    )
    return total, {
        "loss": float(total.detach().cpu()),
        "add_bce": float(add.detach().cpu()),
        "semantic_ce": float(sem.detach().cpu()),
        "bottom_ce": float(bottom.detach().cpu()),
        "span_ce": float(span.detach().cpu()),
        "vertical_interval_ce": float(vertical.detach().cpu()),
        "positive_bev_cells": int(pos.sum().item()),
        "candidate_bev_cells": int(cand.sum().item()),
        "hard_negative_bev_cells": int(hard_negative_count),
    }


@dataclass(frozen=True)
class InnovationLossWeights:
    semantic: float = 1.0
    vertical: float = 1.0


def innovation_loss(
    outputs: dict[str, torch.Tensor],
    *,
    add_target: torch.Tensor,
    semantic_target: torch.Tensor,
    vertical_target: torch.Tensor,
    candidate_mask: torch.Tensor,
    weights: InnovationLossWeights = InnovationLossWeights(),
    positive_weight: float = 4.0,
    vertical_positive_weight: float = 1.0,
    presence_hard_negative_ratio: float | None = None,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Sparse innovation objective.

    Presence is supervised on all candidate cells.  Semantic and vertical
    geometry losses are evaluated only on innovation-positive cells.
    """
    add_logits = outputs["add_presence_logits"]
    sem_logits = outputs["semantic_logits"]
    z_logits = outputs["vertical_occupancy_logits"]
    pos = add_target.bool()
    cand = candidate_mask.bool()
    if add_logits.shape != pos.shape or cand.shape != pos.shape:
        raise ValueError("presence/candidate shape mismatch")
    if semantic_target.shape != pos.shape:
        raise ValueError("semantic_target shape mismatch")
    if vertical_target.shape != z_logits.shape:
        raise ValueError("vertical_target shape mismatch")

    hard_negative_count = 0
    if bool(cand.any()):
        if presence_hard_negative_ratio is None:
            target = pos.to(add_logits.dtype)
            pw = torch.as_tensor(
                float(positive_weight),
                dtype=add_logits.dtype,
                device=add_logits.device,
            )
            add = F.binary_cross_entropy_with_logits(
                add_logits[cand],
                target[cand],
                pos_weight=pw,
            )
        else:
            ratio = float(presence_hard_negative_ratio)
            if ratio <= 0:
                raise ValueError("presence_hard_negative_ratio must be positive")
            pos_rows = add_logits[pos & cand]
            neg_rows = add_logits[cand & ~pos]
            pos_loss = F.softplus(-pos_rows)
            neg_loss_all = F.softplus(neg_rows)
            npos = int(pos_rows.numel())
            nneg = int(neg_rows.numel())
            if npos > 0 and nneg > 0:
                hard_negative_count = min(
                    nneg,
                    max(1, int(np.ceil(ratio * npos))),
                )
                neg_loss = torch.topk(
                    neg_loss_all,
                    k=hard_negative_count,
                    largest=True,
                    sorted=False,
                ).values
                add = torch.cat((pos_loss, neg_loss), dim=0).mean()
            elif npos > 0:
                add = pos_loss.mean()
            elif nneg > 0:
                # All-negative batch: keep a real anti-hallucination signal.
                add = neg_loss_all.mean()
                hard_negative_count = nneg
            else:
                add = add_logits.sum() * 0.0
    else:
        add = add_logits.sum() * 0.0

    if bool(pos.any()):
        # Move class channel last so boolean BEV indexing produces [N,C].
        sem_rows = sem_logits.permute(0, 1, 3, 4, 2)[pos]
        sem = F.cross_entropy(
            sem_rows,
            semantic_target[pos].long(),
        )
        z_rows = z_logits.permute(0, 1, 3, 4, 2)[pos]
        z_tgt = vertical_target.permute(0, 1, 3, 4, 2)[pos].to(
            z_rows.dtype
        )
        z_pw = torch.as_tensor(
            float(vertical_positive_weight),
            dtype=z_rows.dtype,
            device=z_rows.device,
        )
        vertical = F.binary_cross_entropy_with_logits(
            z_rows,
            z_tgt,
            pos_weight=z_pw,
        )
    else:
        sem = sem_logits.sum() * 0.0
        vertical = z_logits.sum() * 0.0

    total = add + float(weights.semantic) * sem + float(weights.vertical) * vertical
    return total, {
        "loss": float(total.detach().cpu()),
        "add_bce": float(add.detach().cpu()),
        "semantic_ce": float(sem.detach().cpu()),
        "vertical_bce": float(vertical.detach().cpu()),
        "positive_bev_cells": int(pos.sum().item()),
        "candidate_bev_cells": int(cand.sum().item()),
        "hard_negative_bev_cells": int(hard_negative_count),
    }


def decode_innovation(
    outputs: dict[str, torch.Tensor],
    *,
    free_label: int = 17,
    add_threshold: float = 0.5,
    vertical_threshold: float = 0.5,
) -> torch.Tensor:
    """Decode add-only logits into [B,F,H,W,Z] semantic occupancy proposals."""
    add = torch.sigmoid(outputs["add_presence_logits"]) >= float(add_threshold)
    sem = outputs["semantic_logits"].argmax(dim=2)
    B, Fh, H, W = add.shape
    if "vertical_occupancy_logits" in outputs:
        z = torch.sigmoid(outputs["vertical_occupancy_logits"]) >= float(
            vertical_threshold
        )
        Z = int(z.shape[2])
        zmask = z.permute(0, 1, 3, 4, 2) & add[..., None]
    elif "bottom_logits" in outputs and "span_logits" in outputs:
        Z = int(outputs["bottom_logits"].shape[2])
        bottom = outputs["bottom_logits"].argmax(dim=2)
        span = outputs["span_logits"].argmax(dim=2) + 1
        top = torch.clamp(bottom + span - 1, max=Z - 1)
        zidx = torch.arange(Z, device=add.device).view(1, 1, 1, 1, Z)
        zmask = (
            (zidx >= bottom[..., None])
            & (zidx <= top[..., None])
            & add[..., None]
        )
    else:
        raise ValueError("unknown innovation vertical output parameterization")
    out = torch.full(
        (B, Fh, H, W, Z),
        int(free_label),
        dtype=torch.long,
        device=add.device,
    )
    labels = sem[..., None].expand(B, Fh, H, W, Z)
    out[zmask] = labels[zmask]
    return out


def protected_add_only_torch(
    base: torch.Tensor,
    proposal: torch.Tensor,
    *,
    free_label: int = 17,
    protected_mask: torch.Tensor | None = None,
    confidence_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Torch equivalent of V19 protected add-only composition."""
    if base.shape != proposal.shape:
        raise ValueError("base/proposal shape mismatch")
    out = base.clone()
    candidate = (proposal != int(free_label)) & (out == int(free_label))
    if protected_mask is not None:
        if protected_mask.shape != out.shape:
            raise ValueError("protected_mask shape mismatch")
        candidate &= ~protected_mask.bool()
    if confidence_mask is not None:
        if confidence_mask.shape != out.shape:
            raise ValueError("confidence_mask shape mismatch")
        candidate &= confidence_mask.bool()
    out[candidate] = proposal[candidate]
    return out
