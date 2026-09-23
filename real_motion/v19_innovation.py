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

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import relative_transform, warp_mask, warp_semantic_grid
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
) -> tuple[np.ndarray, np.ndarray]:
    """Return top semantic + geometric summary in one future ego frame."""
    sem = np.asarray(semantics, dtype=np.uint8)
    obs = np.asarray(observed, dtype=bool)
    if sem.shape != tuple(grid.shape_hwd) or obs.shape != sem.shape:
        raise ValueError("semantic/observed grid shape mismatch")

    T = relative_transform(
        np.asarray(src_pose, dtype=np.float64),
        np.asarray(future_pose, dtype=np.float64),
    )
    # Only genuinely observed occupied semantics are carried forward.  Free
    # evidence is represented by the separately warped coverage mask.
    observed_sem = sem.copy()
    observed_sem[~obs] = int(free_label)
    aligned_sem = warp_semantic_grid(
        observed_sem,
        T,
        grid=grid,
        free_label=int(free_label),
    )
    aligned_obs = warp_mask(obs, T, grid=grid)

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
    geom = np.stack((coverage, top_h, bottom_h, count), axis=0)
    return top_label, geom.astype(np.float32)


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

    if bool(cand.any()):
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
    z = torch.sigmoid(outputs["vertical_occupancy_logits"]) >= float(
        vertical_threshold
    )
    B, Fh, H, W = add.shape
    Z = int(z.shape[2])
    out = torch.full(
        (B, Fh, H, W, Z),
        int(free_label),
        dtype=torch.long,
        device=add.device,
    )
    zmask = z.permute(0, 1, 3, 4, 2) & add[..., None]
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
