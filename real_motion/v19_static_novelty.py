"""Static New-FOV Novelty model for the V19 Transport/Memory/Novelty split.

The branch predicts only ancestor-free *static* occupancy in BEV columns that
are geometrically outside every historical occupancy-grid footprint after exact
future-ego alignment.

The representation follows the measured target structure:
- New-FOV static columns are not contiguous enough for a bottom/top interval
  parameterization (about 75% contiguous in the current diagnostic);
- but the semantic class is usually shared within a positive column (about 93%
  single-semantic columns).

So the head predicts:
1. a direct Z-bin occupancy mask, allowing non-contiguous vertical structure;
2. one static semantic class per positive BEV column.

This is deliberately lighter than dense Z x C semantic logits while avoiding
the interval bottleneck that failed on non-contiguous targets.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import relative_transform
from .local_st_world_model import SEMANTIC_CLASSES
from .motion_transport import FUTURE_FRAMES, HISTORY_FRAMES
from .v19_innovation import GEOMETRY_CHANNELS


STATIC_NEW_FOV_PROTOCOL = "v19_static_new_fov_direct_z_v1"


def history_grid_footprint_bev(
    history_poses: np.ndarray,
    future_pose: np.ndarray,
    grid,
) -> np.ndarray:
    """Return future BEV cells covered by at least one historical XY grid.

    This uses only ego poses and fixed occupancy-grid geometry, so it is fully
    causal given the same future-ego conditioning contract as frozen V18.
    """
    X, Y, _ = tuple(int(x) for x in grid.shape_hwd)
    vx, vy, _ = tuple(float(x) for x in grid.voxel_size)
    xs = float(grid.x_min) + (np.arange(X, dtype=np.float64) + 0.5) * vx
    ys = float(grid.y_min) + (np.arange(Y, dtype=np.float64) + 0.5) * vy
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    pts_future = np.stack(
        (
            xx.reshape(-1),
            yy.reshape(-1),
            np.zeros(X * Y, dtype=np.float64),
            np.ones(X * Y, dtype=np.float64),
        ),
        axis=1,
    )

    covered = np.zeros(X * Y, dtype=bool)
    fpose = np.asarray(future_pose, dtype=np.float64)
    for hpose in np.asarray(history_poses, dtype=np.float64):
        future_to_history = relative_transform(
            fpose,
            np.asarray(hpose, dtype=np.float64),
        )
        ph = (future_to_history @ pts_future.T).T
        covered |= (
            (ph[:, 0] >= float(grid.x_min))
            & (ph[:, 0] < float(grid.x_max))
            & (ph[:, 1] >= float(grid.y_min))
            & (ph[:, 1] < float(grid.y_max))
        )
    return covered.reshape(X, Y)


def majority_semantic_per_column(
    positive_mask: np.ndarray,
    gt_occ: np.ndarray,
    *,
    num_classes: int = 17,
    ignore_label: int = 255,
) -> np.ndarray:
    """Majority occupied semantic class for each positive BEV column."""
    m = np.asarray(positive_mask, dtype=bool)
    gt = np.asarray(gt_occ, dtype=np.uint8)
    if m.shape != gt.shape or m.ndim != 3:
        raise ValueError("positive mask and GT must share [X,Y,Z]")
    out = np.full(m.shape[:2], int(ignore_label), dtype=np.uint8)
    pos_bev = m.any(axis=2)
    if not bool(pos_bev.any()):
        return out

    coords = np.argwhere(pos_bev)
    for x, y in coords:
        labels = gt[x, y][m[x, y]]
        if len(labels) == 0:
            continue
        counts = np.bincount(
            labels.astype(np.int64),
            minlength=int(num_classes),
        )[: int(num_classes)]
        out[x, y] = np.uint8(int(np.argmax(counts)))
    return out


class StaticNewFOVHead(nn.Module):
    """Future-aligned temporal BEV head for static New-FOV completion."""

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
        self.future_frames = int(future_frames)
        self.history_frames = int(history_frames)
        self.num_semantic_classes = int(num_semantic_classes)
        self.vertical_bins = int(vertical_bins)
        if min(
            self.future_frames,
            self.history_frames,
            self.num_semantic_classes,
            self.vertical_bins,
            int(hidden_dim),
        ) <= 0:
            raise ValueError("invalid StaticNewFOVHead dimensions")

        self.semantic_embedding = nn.Embedding(
            SEMANTIC_CLASSES,
            int(semantic_dim),
        )
        in_frame = int(semantic_dim) + int(GEOMETRY_CHANNELS)
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
        self.temporal_pw = nn.Conv3d(hidden_dim, hidden_dim, 1)
        self.temporal_norm = nn.GroupNorm(1, hidden_dim)

        # Both maps are causal/deployable:
        #   channel 0: Transport+Memory already explains this BEV column;
        #   channel 1: this future column lies in geometric New-FOV support.
        self.context_proj = nn.Sequential(
            nn.Conv2d(2, hidden_dim, 3, stride=2, padding=1),
            nn.GroupNorm(1, hidden_dim),
        )
        self.future_time_embedding = nn.Parameter(
            torch.zeros(1, self.future_frames, hidden_dim)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),
            nn.Upsample(
                scale_factor=2.0,
                mode="bilinear",
                align_corners=False,
            ),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),
        )
        self.out_head = nn.Conv2d(
            hidden_dim,
            self.vertical_bins + self.num_semantic_classes,
            1,
        )

        nn.init.trunc_normal_(self.future_time_embedding, std=0.02)
        nn.init.zeros_(self.out_head.weight)
        nn.init.zeros_(self.out_head.bias)
        self.set_occupancy_prior(0.05)

    def set_occupancy_prior(self, probability: float) -> None:
        p = min(max(float(probability), 1e-4), 1.0 - 1e-4)
        bias = math.log(p / (1.0 - p))
        with torch.no_grad():
            self.out_head.bias[: self.vertical_bins].fill_(float(bias))

    def forward(
        self,
        future_aligned_semantic: torch.Tensor,
        future_aligned_geometry: torch.Tensor,
        base_explained: torch.Tensor,
        new_fov_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        lab = future_aligned_semantic.long()
        geo = future_aligned_geometry
        if lab.ndim != 5:
            raise ValueError(
                "future_aligned_semantic must be [B,F,T,H,W]"
            )
        B, Fh, T, H, W = lab.shape
        if Fh != self.future_frames or T != self.history_frames:
            raise ValueError("future/history frame count mismatch")
        if geo.shape != (
            B,
            Fh,
            T,
            GEOMETRY_CHANNELS,
            H,
            W,
        ):
            raise ValueError(
                "future_aligned_geometry must be [B,F,T,4,H,W]"
            )
        if base_explained.shape != (B, Fh, 1, H, W):
            raise ValueError("base_explained must be [B,F,1,H,W]")
        if new_fov_mask.shape == (B, Fh, H, W):
            new_fov_mask = new_fov_mask.unsqueeze(2)
        if new_fov_mask.shape != (B, Fh, 1, H, W):
            raise ValueError("new_fov_mask must be [B,F,1,H,W]")
        if bool((lab < 0).any()) or bool(
            (lab >= SEMANTIC_CLASSES).any()
        ):
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
        x = self.temporal_norm(x.mean(dim=2))

        ctx = torch.cat(
            (
                base_explained,
                new_fov_mask.to(base_explained.dtype),
            ),
            dim=2,
        ).reshape(B * Fh, 2, H, W)
        x = x + self.context_proj(ctx.to(x.dtype))
        time = self.future_time_embedding.expand(B, -1, -1).reshape(
            B * Fh, -1
        )
        x = x + time[:, :, None, None].to(x.dtype)
        x = self.decoder(x)
        raw = self.out_head(x).reshape(B, Fh, -1, H, W)

        z = self.vertical_bins
        return {
            "occupancy_logits": raw[:, :, :z],
            "semantic_logits": raw[:, :, z:],
        }


def static_new_fov_loss(
    outputs: dict[str, torch.Tensor],
    *,
    occupancy_target: torch.Tensor,
    candidate_voxels: torch.Tensor,
    semantic_target: torch.Tensor,
    occupancy_positive_weight: float,
    semantic_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Voxel occupancy BCE in causal New-FOV support + column semantic CE."""
    occ_logits = outputs["occupancy_logits"]
    sem_logits = outputs["semantic_logits"]
    occ_tgt = occupancy_target.bool()
    cand = candidate_voxels.bool()
    if occ_logits.shape != occ_tgt.shape or cand.shape != occ_tgt.shape:
        raise ValueError("occupancy logits/target/candidate mismatch")
    if semantic_target.shape != occ_tgt.shape[:2] + occ_tgt.shape[3:]:
        # occ target [B,F,Z,H,W], semantic [B,F,H,W]
        raise ValueError("semantic target shape mismatch")

    logits = occ_logits[cand]
    target = occ_tgt[cand].to(occ_logits.dtype)
    if logits.numel() == 0:
        occ = occ_logits.sum() * 0.0
    else:
        pw = torch.as_tensor(
            float(occupancy_positive_weight),
            dtype=occ_logits.dtype,
            device=occ_logits.device,
        )
        occ = F.binary_cross_entropy_with_logits(
            logits,
            target,
            pos_weight=pw,
        )

    pos_bev = occ_tgt.any(dim=2)
    if bool(pos_bev.any()):
        rows = sem_logits.permute(0, 1, 3, 4, 2)[pos_bev]
        sem = F.cross_entropy(
            rows,
            semantic_target[pos_bev].long(),
        )
    else:
        sem = sem_logits.sum() * 0.0

    total = occ + float(semantic_weight) * sem
    return total, {
        "loss": float(total.detach().cpu()),
        "occupancy_bce": float(occ.detach().cpu()),
        "semantic_ce": float(sem.detach().cpu()),
        "positive_voxels": int((occ_tgt & cand).sum().item()),
        "candidate_voxels": int(cand.sum().item()),
        "positive_bev_columns": int(pos_bev.sum().item()),
    }


def decode_static_new_fov(
    outputs: dict[str, torch.Tensor],
    *,
    new_fov_mask: torch.Tensor,
    base_free_mask: torch.Tensor,
    free_label: int,
    occupancy_threshold: float,
) -> torch.Tensor:
    """Decode semantic 3D proposal, masked to causal support and base-free voxels."""
    occ_prob = torch.sigmoid(outputs["occupancy_logits"].float())
    occ = occ_prob >= float(occupancy_threshold)
    sem = outputs["semantic_logits"].argmax(dim=2)
    if new_fov_mask.ndim == 5:
        new_fov_mask = new_fov_mask[:, :, 0]
    support = (
        new_fov_mask.bool().unsqueeze(2)
        & base_free_mask.bool()
    )
    occ &= support

    B, Fh, Z, H, W = occ.shape
    proposal = torch.full(
        (B, Fh, Z, H, W),
        int(free_label),
        dtype=torch.long,
        device=occ.device,
    )
    cls = sem.unsqueeze(2).expand(B, Fh, Z, H, W)
    proposal[occ] = cls[occ]
    return proposal
