"""Stage-0 semantic control for V20.

Geometry/presence/vertical support stay frozen from V19 Factorized New-FOV.
Only the previous one-class-per-BEV-column semantic prediction is replaced by
per-Z voxel semantics on candidate voxels.

This is deliberately a cheap semantic-control experiment, not the V20 3D scene
encoder.  It answers whether vertical semantic ambiguity alone explains the
Factorized New-FOV ceiling.
"""
from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .local_st_world_model import SEMANTIC_CLASSES
from .v20_history_world import DYNAMIC_IDS
from .v19_static_novelty_factorized import FactorizedStaticNewFOVHead

PROTOCOL = "p0_f9_v20_stage0_per_z_semantic_v1"
LABEL_SIDECAR_PROTOCOL = "p0_f9_v20_stage0_per_z_semantic_labels_v2"
LABEL_SIDECAR_KEYS = frozenset(
    {
        "protocol",
        "parent_v19_shard",
        "count",
        "voxel_semantic_target",
        "scene_name",
        "t0_token",
    }
)


def frozen_factorized_support(
    outputs: Mapping[str, torch.Tensor],
    *,
    new_fov_mask: torch.Tensor,
    base_free: torch.Tensor,
    presence_threshold: float,
    vertical_threshold: float,
) -> torch.Tensor:
    """Return the frozen V19 predicted occupancy support [B,F,Z,H,W].

    This is the only geometry input allowed into the Stage-0 semantic head.
    It depends on frozen V19 presence/vertical predictions plus causal
    New-FOV/base-free masks; no future semantic or vertical GT is accepted.
    """
    if "presence_logits" not in outputs or "vertical_logits" not in outputs:
        raise KeyError("frozen V19 outputs require presence_logits and vertical_logits")
    pres = outputs["presence_logits"]
    vert = outputs["vertical_logits"]
    if pres.ndim != 4 or vert.ndim != 5:
        raise ValueError("unexpected frozen V19 presence/vertical rank")
    if vert.shape[:2] != pres.shape[:2] or vert.shape[-2:] != pres.shape[-2:]:
        raise ValueError("frozen V19 presence/vertical spatial mismatch")
    if new_fov_mask.ndim == 5:
        if new_fov_mask.shape[2] != 1:
            raise ValueError("5D new_fov_mask must have singleton channel")
        new_fov_mask = new_fov_mask[:, :, 0]
    if new_fov_mask.shape != pres.shape:
        raise ValueError("new_fov_mask must match frozen presence logits")
    if base_free.shape != vert.shape:
        raise ValueError("base_free must match frozen vertical logits")
    active = (
        torch.sigmoid(pres.float()) >= float(presence_threshold)
    ) & new_fov_mask.bool()
    z_support = torch.sigmoid(vert.float()) >= float(vertical_threshold)
    return active.unsqueeze(2) & z_support & base_free.bool()


def validate_stage0_sidecar_pair(
    v19_shard: Mapping[str, object],
    label_shard: Mapping[str, object],
    *,
    expected_parent_shard: str,
) -> int:
    """Validate a label-only Stage-0 sidecar against one frozen V19 shard."""
    if label_shard.get("protocol") != LABEL_SIDECAR_PROTOCOL:
        raise RuntimeError(
            f"unexpected Stage-0 label protocol: {label_shard.get('protocol')}"
        )
    extra = set(label_shard) - set(LABEL_SIDECAR_KEYS)
    if extra:
        raise RuntimeError(
            "Stage-0 sidecar must be label-only; unexpected keys: "
            + ", ".join(sorted(str(x) for x in extra))
        )
    if str(label_shard.get("parent_v19_shard")) != str(expected_parent_shard):
        raise RuntimeError("Stage-0 sidecar parent shard mismatch")
    scenes = list(v19_shard.get("scene_name", []))
    tokens = list(v19_shard.get("t0_token", []))
    side_scenes = list(label_shard.get("scene_name", []))
    side_tokens = list(label_shard.get("t0_token", []))
    if scenes != side_scenes:
        raise RuntimeError("Stage-0 sidecar scene order mismatch")
    if tokens != side_tokens:
        raise RuntimeError("Stage-0 sidecar t0-token order mismatch")
    n = len(tokens)
    if len(scenes) != n:
        raise RuntimeError("malformed V19 shard identity arrays")
    if int(label_shard.get("count", -1)) != n:
        raise RuntimeError("Stage-0 sidecar count mismatch")
    target = label_shard.get("voxel_semantic_target")
    if not isinstance(target, torch.Tensor) or int(target.shape[0]) != n:
        raise RuntimeError("Stage-0 sidecar target batch mismatch")
    return n



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


class FrozenFactorizedFeatureAdapter(nn.Module):
    """Expose the frozen V19 decoder feature without changing its predictions."""

    def __init__(self, factorized: FactorizedStaticNewFOVHead):
        super().__init__()
        self.factorized = factorized
        for p in self.factorized.parameters():
            p.requires_grad = False
        self.factorized.eval()
        self._last_feature: torch.Tensor | None = None
        self._hook = self.factorized.decoder.register_forward_hook(
            self._capture_decoder_feature
        )

    def _capture_decoder_feature(self, module, inputs, output):
        self._last_feature = output

    @property
    def feature_channels(self) -> int:
        # The final decoder Conv2d preserves hidden_dim.
        for m in reversed(list(self.factorized.decoder.modules())):
            if isinstance(m, nn.Conv2d):
                return int(m.out_channels)
        raise RuntimeError("cannot infer frozen Factorized decoder channels")

    def forward(self, *args, **kwargs):
        self._last_feature = None
        with torch.no_grad():
            outputs = self.factorized(*args, **kwargs)
        feat = self._last_feature
        if feat is None:
            raise RuntimeError("Factorized decoder feature hook did not fire")
        B = int(args[0].shape[0])
        Fh = int(args[0].shape[1])
        if feat.shape[0] != B * Fh:
            raise RuntimeError("unexpected Factorized decoder batch shape")
        feat = feat.reshape(B, Fh, feat.shape[1], feat.shape[2], feat.shape[3])
        return outputs, feat

    def train(self, mode: bool = True):
        # Wrapper may be put in train mode by callers, but frozen V19 must stay eval.
        super().train(mode)
        self.factorized.eval()
        return self

