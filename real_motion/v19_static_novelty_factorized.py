"""Factorized Static New-FOV completion for V19 Novelty.

The failed direct-Z head mixed three different responsibilities in one dense
voxel loss: where a future static column exists, what semantic it has, and what
its vertical profile is.  This module factorizes those tasks:

1. BEV presence over causal geometric New-FOV support;
2. one semantic class per GT-positive column;
3. a direct 16-bin vertical profile trained only on GT-positive columns.

Nearest historical Static-Memory geometry is supplied only as context
(semantic, vertical profile and distance).  It is not copied or treated as a
hard geometry template.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .local_st_world_model import SEMANTIC_CLASSES
from .motion_transport import FUTURE_FRAMES, HISTORY_FRAMES
from .v19_innovation import GEOMETRY_CHANNELS


PROTOCOL = "v19_factorized_static_new_fov_v1"
ANCHOR_DISTANCE_MAX_M = 40.0


def quantize_anchor_distance_m(x, max_distance_m: float = ANCHOR_DISTANCE_MAX_M):
    import numpy as np

    a = np.asarray(x, dtype=np.float32)
    m = float(max_distance_m)
    if m <= 0:
        raise ValueError("max_distance_m must be positive")
    a = np.nan_to_num(a, nan=m, posinf=m, neginf=0.0)
    return np.rint(np.clip(a, 0.0, m) / m * 255.0).astype(np.uint8)


def dequantize_anchor_distance_torch(
    x: torch.Tensor,
    max_distance_m: float = ANCHOR_DISTANCE_MAX_M,
) -> torch.Tensor:
    m = float(max_distance_m)
    if m <= 0:
        raise ValueError("max_distance_m must be positive")
    return x.to(torch.float32) / 255.0 * m


class FactorizedStaticNewFOVHead(nn.Module):
    """BEV presence + semantic + conditional direct-Z profile.

    Historical semantic/geometry is encoded exactly as in the earlier
    future-aligned Innovation head.  The explicit anchor context contributes:
      * nearest known-static semantic embedding;
      * nearest known-static 16-bin occupancy profile;
      * nearest-anchor distance;
      * anchor validity;
      * causal geometric New-FOV mask;
      * frozen Transport+Memory explained BEV occupancy.

    The vertical logits are *not* residuals around the anchor profile.  The
    anchor only conditions prediction because boundary copying was shown to
    have insufficient 3D precision.
    """

    def __init__(
        self,
        *,
        future_frames: int = FUTURE_FRAMES,
        history_frames: int = HISTORY_FRAMES,
        semantic_dim: int = 8,
        anchor_semantic_dim: int = 4,
        hidden_dim: int = 32,
        num_semantic_classes: int = 17,
        vertical_bins: int = 16,
        anchor_distance_max_m: float = ANCHOR_DISTANCE_MAX_M,
    ):
        super().__init__()
        self.future_frames = int(future_frames)
        self.history_frames = int(history_frames)
        self.num_semantic_classes = int(num_semantic_classes)
        self.vertical_bins = int(vertical_bins)
        self.anchor_distance_max_m = float(anchor_distance_max_m)
        if min(
            self.future_frames,
            self.history_frames,
            int(semantic_dim),
            int(anchor_semantic_dim),
            int(hidden_dim),
            self.num_semantic_classes,
            self.vertical_bins,
        ) <= 0:
            raise ValueError("invalid factorized Static New-FOV dimensions")
        if self.anchor_distance_max_m <= 0:
            raise ValueError("anchor_distance_max_m must be positive")

        # Historical input uses all 18 Occ3D labels including free=17.
        self.semantic_embedding = nn.Embedding(
            SEMANTIC_CLASSES,
            int(semantic_dim),
        )
        self.anchor_semantic_embedding = nn.Embedding(
            SEMANTIC_CLASSES,
            int(anchor_semantic_dim),
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

        context_channels = (
            2  # frozen explained BEV + New-FOV
            + int(anchor_semantic_dim)
            + self.vertical_bins
            + 1  # normalized anchor distance
            + 1  # anchor valid
        )
        self.context_proj = nn.Sequential(
            nn.Conv2d(
                context_channels,
                hidden_dim,
                3,
                stride=2,
                padding=1,
            ),
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),
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
            1 + self.num_semantic_classes + self.vertical_bins,
            1,
        )

        nn.init.trunc_normal_(self.future_time_embedding, std=0.02)
        nn.init.zeros_(self.out_head.weight)
        nn.init.zeros_(self.out_head.bias)
        self.set_output_priors(
            presence_probability=0.30,
            vertical_probability=0.15,
        )

    @staticmethod
    def _logit(p: float) -> float:
        p = min(max(float(p), 1e-4), 1.0 - 1e-4)
        return math.log(p / (1.0 - p))

    def set_output_priors(
        self,
        *,
        presence_probability: float,
        vertical_probability: float,
    ) -> None:
        with torch.no_grad():
            self.out_head.bias[0] = float(
                self._logit(presence_probability)
            )
            z0 = 1 + self.num_semantic_classes
            self.out_head.bias[z0:].fill_(
                float(self._logit(vertical_probability))
            )

    def forward(
        self,
        future_aligned_semantic: torch.Tensor,
        future_aligned_geometry: torch.Tensor,
        base_explained: torch.Tensor,
        new_fov_mask: torch.Tensor,
        anchor_semantic: torch.Tensor,
        anchor_profile: torch.Tensor,
        anchor_distance_m: torch.Tensor,
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
        if new_fov_mask.shape == (B, Fh, 1, H, W):
            new_fov_mask = new_fov_mask[:, :, 0]
        if new_fov_mask.shape != (B, Fh, H, W):
            raise ValueError("new_fov_mask must be [B,F,H,W]")
        if anchor_semantic.shape != (B, Fh, H, W):
            raise ValueError("anchor_semantic must be [B,F,H,W]")
        if anchor_profile.shape != (
            B,
            Fh,
            self.vertical_bins,
            H,
            W,
        ):
            raise ValueError(
                "anchor_profile must be [B,F,Z,H,W]"
            )
        if anchor_distance_m.shape == (B, Fh, H, W):
            anchor_distance_m = anchor_distance_m.unsqueeze(2)
        if anchor_distance_m.shape != (B, Fh, 1, H, W):
            raise ValueError(
                "anchor_distance_m must be [B,F,1,H,W]"
            )
        if bool((lab < 0).any()) or bool(
            (lab >= SEMANTIC_CLASSES).any()
        ):
            raise ValueError("history semantic labels outside [0,17]")
        if bool((anchor_semantic < 0).any()) or bool(
            (anchor_semantic >= SEMANTIC_CLASSES).any()
        ):
            raise ValueError("anchor semantic labels outside [0,17]")

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

        anchor_emb = self.anchor_semantic_embedding(
            anchor_semantic.long()
        ).permute(0, 1, 4, 2, 3)
        valid = (
            anchor_semantic != (SEMANTIC_CLASSES - 1)
        ).to(base_explained.dtype).unsqueeze(2)
        dist = (
            anchor_distance_m.to(torch.float32)
            / self.anchor_distance_max_m
        ).clamp_(0.0, 1.0)
        context = torch.cat(
            (
                base_explained,
                new_fov_mask.to(base_explained.dtype).unsqueeze(2),
                anchor_emb.to(base_explained.dtype),
                anchor_profile.to(base_explained.dtype),
                dist.to(base_explained.dtype),
                valid,
            ),
            dim=2,
        ).reshape(B * Fh, -1, H, W)
        x = x + self.context_proj(context.to(x.dtype))

        time = self.future_time_embedding.expand(B, -1, -1).reshape(
            B * Fh, -1
        )
        x = x + time[:, :, None, None].to(x.dtype)
        x = self.decoder(x)
        raw = self.out_head(x).reshape(B, Fh, -1, H, W)

        s0 = 1
        s1 = s0 + self.num_semantic_classes
        return {
            "presence_logits": raw[:, :, 0],
            "semantic_logits": raw[:, :, s0:s1],
            "vertical_logits": raw[:, :, s1:],
        }


def factorized_static_new_fov_loss(
    outputs: dict[str, torch.Tensor],
    *,
    presence_target: torch.Tensor,
    candidate_bev: torch.Tensor,
    semantic_target: torch.Tensor,
    vertical_target: torch.Tensor,
    base_free: torch.Tensor,
    presence_positive_weight: float,
    vertical_positive_weight: float,
    semantic_weight: float = 1.0,
    vertical_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Factorized loss with vertical supervision only on GT-positive columns."""
    pres_logits = outputs["presence_logits"]
    sem_logits = outputs["semantic_logits"]
    vert_logits = outputs["vertical_logits"]

    tgt_bev = presence_target.bool()
    cand_bev = candidate_bev.bool()
    vert_tgt = vertical_target.bool()
    free = base_free.bool()
    if pres_logits.shape != tgt_bev.shape or cand_bev.shape != tgt_bev.shape:
        raise ValueError("presence target/candidate shape mismatch")
    if vert_logits.shape != vert_tgt.shape or free.shape != vert_tgt.shape:
        raise ValueError("vertical target/base-free shape mismatch")
    if sem_logits.shape[:2] + sem_logits.shape[3:] != tgt_bev.shape:
        raise ValueError("semantic logits shape mismatch")
    if semantic_target.shape != tgt_bev.shape:
        raise ValueError("semantic target shape mismatch")

    p_logits = pres_logits[cand_bev]
    p_target = tgt_bev[cand_bev].to(pres_logits.dtype)
    if p_logits.numel() == 0:
        presence = pres_logits.sum() * 0.0
    else:
        presence = F.binary_cross_entropy_with_logits(
            p_logits,
            p_target,
            pos_weight=torch.as_tensor(
                float(presence_positive_weight),
                dtype=pres_logits.dtype,
                device=pres_logits.device,
            ),
        )

    if bool(tgt_bev.any()):
        sem_rows = sem_logits.permute(0, 1, 3, 4, 2)[tgt_bev]
        semantic = F.cross_entropy(
            sem_rows,
            semantic_target[tgt_bev].long(),
        )

        positive_column_voxels = tgt_bev.unsqueeze(2) & free
        v_logits = vert_logits[positive_column_voxels]
        v_target = vert_tgt[positive_column_voxels].to(
            vert_logits.dtype
        )
        vertical = F.binary_cross_entropy_with_logits(
            v_logits,
            v_target,
            pos_weight=torch.as_tensor(
                float(vertical_positive_weight),
                dtype=vert_logits.dtype,
                device=vert_logits.device,
            ),
        )
    else:
        semantic = sem_logits.sum() * 0.0
        vertical = vert_logits.sum() * 0.0

    total = (
        presence
        + float(semantic_weight) * semantic
        + float(vertical_weight) * vertical
    )
    return total, {
        "loss": float(total.detach().cpu()),
        "presence_bce": float(presence.detach().cpu()),
        "semantic_ce": float(semantic.detach().cpu()),
        "vertical_bce": float(vertical.detach().cpu()),
        "positive_bev_columns": int(tgt_bev.sum().item()),
        "candidate_bev_columns": int(cand_bev.sum().item()),
        "positive_vertical_voxels": int(
            (vert_tgt & tgt_bev.unsqueeze(2) & free).sum().item()
        ),
        "vertical_supervised_voxels": int(
            (tgt_bev.unsqueeze(2) & free).sum().item()
        ),
    }


def decode_factorized_static_new_fov(
    outputs: dict[str, torch.Tensor],
    *,
    new_fov_mask: torch.Tensor,
    base_free: torch.Tensor,
    free_label: int,
    presence_threshold: float,
    vertical_threshold: float,
) -> torch.Tensor:
    """Decode [B,F,Z,H,W] semantic proposal under causal/add-only support."""
    pres = (
        torch.sigmoid(outputs["presence_logits"].float())
        >= float(presence_threshold)
    )
    vert = (
        torch.sigmoid(outputs["vertical_logits"].float())
        >= float(vertical_threshold)
    )
    if new_fov_mask.ndim == 5:
        new_fov_mask = new_fov_mask[:, :, 0]
    active = pres & new_fov_mask.bool()
    occ = active.unsqueeze(2) & vert & base_free.bool()

    sem = outputs["semantic_logits"].argmax(dim=2)
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
