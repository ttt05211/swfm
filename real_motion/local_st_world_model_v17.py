"""V17 single-expert Local-STWM improvements.

V17 keeps the successful V16/KTA displacement contract and rigid transport, but
addresses two diagnosed mismatches without introducing routing or multiple
prediction branches:

1. Source-centered crops preserve local context but hide visual translation.
   V17 injects a *frame-specific* motion coordinate into every history frame and
   marks the causally tracked target source inside the local semantic tube.
2. Center SmoothL1 is not the final task.  A differentiable transport-overlap
   term measures whether translating the observed source footprint by the
   predicted displacement would overlap the same footprint translated by GT.

The v13/V16 displacement target is unchanged:
    residual = GT_displacement - KTA_displacement
and a fresh residual head is still exactly zero, so fresh V17 == KTA.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .local_st_world_model import (
    LOCAL_STWM_CACHE_VERSION,
    LOCAL_TUBE_CONTRACT,
    SEMANTIC_CLASSES,
    LocalSTWMConfig,
    LocalSpatialTemporalWorldModel,
)
from .motion_transport import FEATURE_DIM, FEATURE_NAMES, FUTURE_FRAMES, HISTORY_FRAMES

LOCAL_STWM_V17_CACHE_VERSION = "p0_f9_local_stwm_v2"
MODEL_PROTOCOL_V17 = "p0_f9_v17_local_spatial_temporal_world_model_v1"
REPRESENTATION_CONTRACT = "source_centered_semantic_tube_plus_frame_motion_and_target_mask_v1"
FRAME_MOTION_CONTRACT = "per_frame_offset_velocity_valid_from_v13_causal_history_v1"
SOURCE_MASK_CONTRACT = "center_gated_same_class_target_mask_from_causal_local_tube_v1"
TRANSPORT_OVERLAP_CONTRACT = "soft_iou_same_source_footprint_under_pred_vs_gt_displacement_v1"
FRAME_MOTION_DIM = 5

_NAME_TO_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}


def frame_motion_features_from_flat(features: torch.Tensor) -> torch.Tensor:
    """Recover explicit per-frame motion coordinates from the frozen v13 features.

    Output channels are [offset_x, offset_y, velocity_x, velocity_y, valid].
    Metric quantities retain the v13 normalization (offset / 20m, velocity /
    20m/s).  Interior-frame velocity is the average of the incoming and outgoing
    valid segment features; end frames use the adjacent segment.
    """
    if features.ndim != 2 or features.shape[-1] != FEATURE_DIM:
        raise ValueError(f"features must be [N,{FEATURE_DIM}]")
    N = features.shape[0]
    out = features.new_zeros((N, HISTORY_FRAMES, FRAME_MOTION_DIM))
    seg = features.new_zeros((N, HISTORY_FRAMES - 1, 2))
    for t in range(HISTORY_FRAMES - 1):
        seg[:, t, 0] = features[:, _NAME_TO_INDEX[f"hist_vel_{t}_x"]]
        seg[:, t, 1] = features[:, _NAME_TO_INDEX[f"hist_vel_{t}_y"]]
    for t in range(HISTORY_FRAMES):
        out[:, t, 0] = features[:, _NAME_TO_INDEX[f"hist_offset_{t}_x"]]
        out[:, t, 1] = features[:, _NAME_TO_INDEX[f"hist_offset_{t}_y"]]
        out[:, t, 4] = features[:, _NAME_TO_INDEX[f"hist_valid_{t}"]]
        if t == 0:
            vel = seg[:, 0]
        elif t == HISTORY_FRAMES - 1:
            vel = seg[:, -1]
        else:
            vel = 0.5 * (seg[:, t - 1] + seg[:, t])
        out[:, t, 2:4] = vel
    # Invalid frames must not inject invented motion into the temporal stream.
    out[:, :, :4] *= out[:, :, 4:5]
    return out


def target_source_mask_from_tube(
    local_semantic_tube: torch.Tensor,
    source_class_id: torch.Tensor,
    track_valid: torch.Tensor,
    features: torch.Tensor,
    *,
    patch_resolution_m: float = 0.8,
    margin_cells: float = 1.0,
) -> torch.Tensor:
    """Build a cheap causal target-source mask from an already-built V16 tube.

    The crop is centered on the tracked source.  We therefore retain same-class
    pixels inside a center gate derived from the t0 Strong component extent.
    This avoids accidentally marking a nearby same-class vehicle while requiring
    no annotations and no expensive re-running of Strong tracking.
    """
    tube = local_semantic_tube
    if tube.ndim != 4 or tube.shape[1] != HISTORY_FRAMES:
        raise ValueError("local_semantic_tube must be [N,6,H,W]")
    N, T, H, W = tube.shape
    if source_class_id.shape != (N,) or track_valid.shape != (N, T):
        raise ValueError("source class / track valid shape mismatch")
    if features.shape != (N, FEATURE_DIM):
        raise ValueError("features shape mismatch")
    if patch_resolution_m <= 0:
        raise ValueError("patch_resolution_m must be positive")

    extent_x = features[:, _NAME_TO_INDEX["extent_x_norm"]].float() * 10.0
    extent_y = features[:, _NAME_TO_INDEX["extent_y_norm"]].float() * 10.0
    # At least one cell either side of the crop center; extra one-cell margin
    # absorbs 2x pooling/centroid quantization without swallowing distant cars.
    half_x = torch.clamp(extent_x / (2.0 * float(patch_resolution_m)) + float(margin_cells), min=1.0)
    half_y = torch.clamp(extent_y / (2.0 * float(patch_resolution_m)) + float(margin_cells), min=1.0)
    gx = torch.arange(H, device=tube.device, dtype=torch.float32) - (H - 1) / 2.0
    gy = torch.arange(W, device=tube.device, dtype=torch.float32) - (W - 1) / 2.0
    gate_x = gx[None, None, :, None].abs() <= half_x[:, None, None, None]
    gate_y = gy[None, None, None, :].abs() <= half_y[:, None, None, None]
    same_class = tube.long() == source_class_id.long()[:, None, None, None]
    valid = track_valid.bool()[:, :, None, None]
    return (same_class & gate_x & gate_y & valid).to(torch.uint8)


@dataclass(frozen=True)
class LocalSTWMV17Config:
    d_model: int = 128
    semantic_dim: int = 32
    heads: int = 4
    blocks: int = 4
    decoder_blocks: int = 2
    tube_hw: int = 20
    history_frames: int = HISTORY_FRAMES
    future_frames: int = FUTURE_FRAMES
    use_representation: bool = True


class LocalSpatialTemporalWorldModelV17(LocalSpatialTemporalWorldModel):
    """Single V17 STWM with optional motion-preserving representation.

    With ``use_representation=False`` the forward path calls the V16 parent
    exactly.  This makes the V17-L loss-only ablation an architecture-controlled
    comparison rather than a different network.
    """

    def __init__(self, config: LocalSTWMV17Config = LocalSTWMV17Config()):
        base = LocalSTWMConfig(
            d_model=int(config.d_model),
            semantic_dim=int(config.semantic_dim),
            heads=int(config.heads),
            blocks=int(config.blocks),
            decoder_blocks=int(config.decoder_blocks),
            tube_hw=int(config.tube_hw),
            history_frames=int(config.history_frames),
            future_frames=int(config.future_frames),
        )
        # Construct the entire V16 core first.  Under an identical torch seed,
        # the disabled-representation variant therefore has identical core init.
        super().__init__(base)
        self.v17_config = config
        self.frame_motion_proj = nn.Sequential(
            nn.Linear(FRAME_MOTION_DIM, config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model),
        )
        self.source_mask_embedding = nn.Embedding(2, config.semantic_dim)
        nn.init.normal_(self.source_mask_embedding.weight, mean=0.0, std=0.02)

    def forward(
        self,
        features: torch.Tensor,
        local_semantic_tube: torch.Tensor,
        kta_displacement_xy_m: torch.Tensor,
        frame_motion_features: torch.Tensor | None = None,
        target_source_mask_tube: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if not bool(self.v17_config.use_representation):
            return super().forward(features, local_semantic_tube, kta_displacement_xy_m)

        cfg = self.config
        if features.ndim != 2 or features.shape[-1] != FEATURE_DIM:
            raise ValueError(f"features must be [B,{FEATURE_DIM}]")
        B = features.shape[0]
        if local_semantic_tube.ndim != 4 or tuple(local_semantic_tube.shape[1:]) != (
            HISTORY_FRAMES, cfg.tube_hw, cfg.tube_hw
        ):
            raise ValueError("local_semantic_tube shape mismatch")
        if kta_displacement_xy_m.shape != (B, FUTURE_FRAMES, 2):
            raise ValueError("kta_displacement_xy_m must be [B,6,2]")
        if frame_motion_features is None or frame_motion_features.shape != (B, HISTORY_FRAMES, FRAME_MOTION_DIM):
            raise ValueError("frame_motion_features must be [B,6,5]")
        if target_source_mask_tube is None or target_source_mask_tube.shape != local_semantic_tube.shape:
            raise ValueError("target_source_mask_tube must match local_semantic_tube")
        labels = local_semantic_tube.long()
        if bool((labels < 0).any()) or bool((labels >= SEMANTIC_CLASSES).any()):
            raise ValueError("semantic tube contains labels outside [0,17]")
        mask_labels = target_source_mask_tube.long()
        if bool((mask_labels < 0).any()) or bool((mask_labels > 1).any()):
            raise ValueError("target source mask must be binary")
        if B == 0:
            return {
                "residual_xy_m": features.new_empty((0, FUTURE_FRAMES, 2)),
                "existence_logits": features.new_empty((0, FUTURE_FRAMES)),
            }

        emb = self.semantic_embedding(labels) + self.source_mask_embedding(mask_labels)
        x = emb.permute(0, 1, 4, 2, 3).reshape(
            B * HISTORY_FRAMES, cfg.semantic_dim, cfg.tube_hw, cfg.tube_hw
        )
        x = self.spatial_stem(x)
        Hs, Ws = x.shape[-2:]
        x = x.reshape(B, HISTORY_FRAMES, cfg.d_model, Hs, Ws)
        obj = self.kinematic_proj(features).view(B, 1, cfg.d_model, 1, 1)
        fm = self.frame_motion_proj(frame_motion_features.to(x.dtype)).view(
            B, HISTORY_FRAMES, cfg.d_model, 1, 1
        )
        x = x + obj + fm + self.time_embedding + self.spatial_embedding
        for block in self.blocks:
            x = block(x)

        context = x.permute(0, 1, 3, 4, 2).reshape(B, HISTORY_FRAMES * Hs * Ws, cfg.d_model)
        q = self.future_query.expand(B, -1, -1) + self.future_time_embedding
        q = q + self.kinematic_proj(features).unsqueeze(1)
        q = q + self.kta_future_proj(kta_displacement_xy_m.to(q.dtype) / 20.0)
        for block in self.decoder:
            q = block(q, context)
        return {
            "residual_xy_m": self.residual_head(q),
            "existence_logits": self.existence_head(q)[..., 0],
        }


def soft_transport_overlap_loss(
    pred_residual_xy_m: torch.Tensor,
    target_residual_xy_m: torch.Tensor,
    source_footprint_mask: torch.Tensor,
    target_valid: torch.Tensor,
    *,
    patch_resolution_m: float = 0.8,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Differentiable Soft-IoU after translating the observed source footprint.

    The GT footprint is kept centered.  The predicted copy is shifted by the
    *displacement error* ``pred_residual-target_residual``.  Absolute future
    motion therefore never needs a large canvas, and no future occupancy is
    consumed.  SmoothL1 remains the long-range attraction term when overlap is
    zero; this term only rewards task-relevant overlap near the correct motion.
    """
    pred = pred_residual_xy_m
    target = target_residual_xy_m.to(pred.dtype)
    valid = target_valid.bool()
    mask = source_footprint_mask.to(pred.dtype)
    if pred.shape != target.shape or pred.ndim != 3 or pred.shape[-1] != 2:
        raise ValueError("residual tensors must be [B,6,2]")
    B, Hf, _ = pred.shape
    if Hf != FUTURE_FRAMES or mask.ndim != 3 or mask.shape[0] != B:
        raise ValueError("source footprint shape mismatch")
    if valid.shape != (B, Hf):
        raise ValueError("target_valid shape mismatch")
    if patch_resolution_m <= 0:
        raise ValueError("patch_resolution_m must be positive")

    H, W = int(mask.shape[-2]), int(mask.shape[-1])
    # One footprint width of padding per side: GT remains centered while the
    # predicted copy can move by a full local-patch width before clipping.
    canvas = F.pad(mask[:, None], (W, W, H, H))
    Hc, Wc = int(canvas.shape[-2]), int(canvas.shape[-1])
    inp = canvas[:, None].expand(B, Hf, 1, Hc, Wc).reshape(B * Hf, 1, Hc, Wc)

    ys = torch.linspace(-1.0, 1.0, Hc, device=pred.device, dtype=pred.dtype)
    xs = torch.linspace(-1.0, 1.0, Wc, device=pred.device, dtype=pred.dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    base_grid = torch.stack((gx, gy), dim=-1)[None].expand(B * Hf, Hc, Wc, 2).clone()

    err = (pred - target).reshape(B * Hf, 2)
    shift_x_cells = err[:, 0] / float(patch_resolution_m)
    shift_y_cells = err[:, 1] / float(patch_resolution_m)
    # grid_sample returns output(r,c)=input(grid(r,c)); subtracting the desired
    # source-space shift moves image content in the positive output direction.
    base_grid[..., 1] -= (2.0 * shift_x_cells / max(Hc - 1, 1))[:, None, None]
    base_grid[..., 0] -= (2.0 * shift_y_cells / max(Wc - 1, 1))[:, None, None]
    shifted = F.grid_sample(
        inp, base_grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )

    inter = (shifted * inp).sum(dim=(1, 2, 3))
    union = (shifted + inp - shifted * inp).sum(dim=(1, 2, 3))
    iou = (inter + eps) / (union + eps)
    footprint_present = canvas.flatten(1).sum(dim=1) > 0
    usable = valid & footprint_present[:, None]
    usable_flat = usable.reshape(-1)
    if bool(usable_flat.any()):
        loss = (1.0 - iou[usable_flat]).mean()
        mean_iou = float(iou[usable_flat].detach().mean().cpu())
        count = int(usable_flat.sum().item())
    else:
        loss = pred.sum() * 0.0
        mean_iou = float("nan")
        count = 0
    return loss, {"transport_soft_iou": mean_iou, "transport_overlap_labels": count}


def config_from_mapping_v17(raw: Mapping | None) -> LocalSTWMV17Config:
    if not raw:
        return LocalSTWMV17Config()
    fields = LocalSTWMV17Config.__dataclass_fields__
    vals = {}
    for k in fields:
        if k not in raw:
            continue
        vals[k] = bool(raw[k]) if k == "use_representation" else int(raw[k])
    return LocalSTWMV17Config(**vals)
