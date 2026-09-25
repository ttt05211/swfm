"""Stage-0 semantic control for V20.

Geometry/presence/vertical support stay frozen from V19 Factorized New-FOV.
Only the previous one-class-per-BEV-column semantic prediction is replaced by
per-Z voxel semantics on candidate voxels.

This is deliberately a cheap semantic-control experiment, not the V20 3D scene
encoder.  It answers whether vertical semantic ambiguity alone explains the
Factorized New-FOV ceiling.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .local_st_world_model import SEMANTIC_CLASSES
from .v20_history_world import DYNAMIC_IDS

PROTOCOL = "p0_f9_v20_stage0_per_z_semantic_v1"


class PerZSemanticHead(nn.Module):
    """Predict static semantic logits per candidate voxel.

    Input is a low-cost stack of frozen V19 features at BEV resolution plus the
    16-bin candidate vertical support.  The head processes Z as an explicit
    spatial dimension using small 3D convolutions and may be called tile-wise.
    """

    def __init__(
        self,
        *,
        bev_feature_channels: int,
        vertical_bins: int = 16,
        hidden_dim: int = 32,
        num_classes: int = SEMANTIC_CLASSES - 1,
    ):
        super().__init__()
        self.vertical_bins = int(vertical_bins)
        self.num_classes = int(num_classes)
        if self.vertical_bins <= 0 or self.num_classes <= 1:
            raise ValueError("invalid Stage-0 dimensions")
        self.bev_proj = nn.Sequential(
            nn.Conv2d(int(bev_feature_channels), int(hidden_dim), 3, padding=1),
            nn.GroupNorm(1, int(hidden_dim)),
            nn.GELU(),
        )
        self.z_embedding = nn.Parameter(
            torch.zeros(1, int(hidden_dim), self.vertical_bins, 1, 1)
        )
        self.refine = nn.Sequential(
            nn.Conv3d(int(hidden_dim) + 1, int(hidden_dim), 3, padding=1),
            nn.GroupNorm(1, int(hidden_dim)),
            nn.GELU(),
            nn.Conv3d(int(hidden_dim), self.num_classes, 1),
        )
        nn.init.trunc_normal_(self.z_embedding, std=0.02)

    def forward(
        self,
        bev_features: torch.Tensor,
        candidate_vertical: torch.Tensor,
    ) -> torch.Tensor:
        if bev_features.ndim != 4:
            raise ValueError("bev_features must be [B,C,H,W]")
        if candidate_vertical.ndim != 4:
            raise ValueError("candidate_vertical must be [B,Z,H,W]")
        B, Z, H, W = candidate_vertical.shape
        if Z != self.vertical_bins or bev_features.shape[0] != B:
            raise ValueError("Stage-0 batch/Z mismatch")
        if tuple(bev_features.shape[-2:]) != (H, W):
            raise ValueError("Stage-0 spatial mismatch")
        x2 = self.bev_proj(bev_features)
        x3 = x2.unsqueeze(2).expand(B, -1, Z, H, W)
        x3 = x3 + self.z_embedding.to(x3.dtype)
        support = candidate_vertical.to(x3.dtype).unsqueeze(1)
        return self.refine(torch.cat((x3, support), dim=1))


def per_z_semantic_loss(
    logits: torch.Tensor,
    target_semantic: torch.Tensor,
    supervised_candidate: torch.Tensor,
    *,
    class_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross entropy only on the frozen candidate voxels used by Stage 0."""
    if logits.ndim != 5:
        raise ValueError("logits must be [B,C,Z,H,W]")
    if target_semantic.shape != supervised_candidate.shape:
        raise ValueError("target/candidate shape mismatch")
    if logits.shape[0] != target_semantic.shape[0] or logits.shape[2:] != target_semantic.shape[1:]:
        raise ValueError("logit/target shape mismatch")
    mask = supervised_candidate.bool()
    if not bool(mask.any()):
        return logits.sum() * 0.0
    rows = logits.permute(0, 2, 3, 4, 1)[mask].clone()
    target = target_semantic.long()[mask]
    # Stage 0 evaluates static New-FOV only. Dynamic logits are prohibited even
    # if numerical noise would otherwise make them win argmax.
    dyn = torch.as_tensor(DYNAMIC_IDS, dtype=torch.long, device=rows.device)
    rows[:, dyn] = torch.finfo(rows.dtype).min
    return F.cross_entropy(rows, target, weight=class_weight)


def decode_per_z_semantic(
    logits: torch.Tensor,
    candidate_vertical: torch.Tensor,
    *,
    free_label: int,
) -> torch.Tensor:
    """Return [B,Z,H,W] semantic proposal under frozen V19 geometry."""
    if logits.ndim != 5 or candidate_vertical.ndim != 4:
        raise ValueError("unexpected Stage-0 tensor rank")
    masked = logits.clone()
    dyn = torch.as_tensor(DYNAMIC_IDS, dtype=torch.long, device=masked.device)
    masked[:, dyn] = torch.finfo(masked.dtype).min
    cls = masked.argmax(dim=1)
    out = torch.full_like(cls, int(free_label))
    active = candidate_vertical.bool()
    out[active] = cls[active]
    return out
