"""V18: SE(2) extension of the frozen V17-RL local motion predictor.

This module changes only the predicted planar rigid-motion state.  The historical
V17 representation, future-query decoder, XY residual head and existence head are
kept intact.  V18 adds one scalar relative-yaw output per future horizon and uses
source-centred SE(2) supervision that is geometrically consistent with a GT box
rigid transform even when the observed Strong source centroid is offset from the
annotation-box centre.

For a source centroid c_s, GT box centres a_0/a_h and relative rotation R_h:

    d_s^* = (a_h - a_0) + (R_h - I) (c_s - a_0)

and the renderer applies

    p'_h = R_h (p - c_s) + c_s + d_s^*.

Thus the observed source shape is never aligned to the absolute GT box centre.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .local_st_world_model import SEMANTIC_CLASSES
from .local_st_world_model_v17 import (
    FRAME_MOTION_DIM,
    LocalSTWMV17Config,
    LocalSpatialTemporalWorldModelV17,
)
from .motion_transport import FEATURE_DIM, FUTURE_FRAMES, HISTORY_FRAMES

MODEL_PROTOCOL_V18_SE2 = "p0_f9_v18_se2_local_spatial_temporal_world_model_v1"
SE2_CACHE_VERSION = "p0_f9_local_stwm_v18_se2_labels_v1"
SE2_TARGET_CONTRACT = "source_center_se2_gt_rigid_equivalent_v1"
SE2_SHAPE_CONTRACT = "soft_iou_same_source_footprint_pred_vs_gt_se2_v1"

# Pedestrian orientation is intentionally disabled in the first controlled
# experiment.  This set is deployment-known from source semantics only.
YAW_ENABLED_CLASS_IDS = (2, 3, 4, 5, 6, 9, 10)


@dataclass(frozen=True)
class SE2Target:
    source_displacement_xy_m: np.ndarray
    yaw_rad: float


def wrap_angle_np(angle: float) -> float:
    return float((float(angle) + math.pi) % (2.0 * math.pi) - math.pi)


def wrap_angle_tensor(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _rot2_np(angle: float) -> np.ndarray:
    c, s = math.cos(float(angle)), math.sin(float(angle))
    return np.asarray([[c, -s], [s, c]], dtype=np.float64)


def source_center_se2_target(
    source_center_xy: np.ndarray,
    gt_box_center_t0_xy: np.ndarray,
    gt_box_center_future_xy: np.ndarray,
    relative_yaw_rad: float,
) -> SE2Target:
    """Convert a GT box-centred rigid motion into source-centred SE(2).

    All inputs must be expressed in one frozen coordinate frame (the current t0
    ego frame in the formal pipeline).  The returned displacement is the motion
    of the observed source centroid itself, not the GT box-centre displacement.
    """
    cs = np.asarray(source_center_xy, dtype=np.float64)
    a0 = np.asarray(gt_box_center_t0_xy, dtype=np.float64)
    ah = np.asarray(gt_box_center_future_xy, dtype=np.float64)
    if cs.shape != (2,) or a0.shape != (2,) or ah.shape != (2,):
        raise ValueError("source/GT centres must be XY vectors")
    yaw = wrap_angle_np(relative_yaw_rad)
    R = _rot2_np(yaw)
    offset = cs - a0
    displacement = (ah - a0) + (R @ offset - offset)
    return SE2Target(displacement.astype(np.float32), yaw)


def apply_box_centered_rigid_xy(
    points_xy: np.ndarray,
    gt_box_center_t0_xy: np.ndarray,
    gt_box_center_future_xy: np.ndarray,
    relative_yaw_rad: float,
) -> np.ndarray:
    """Reference GT rigid transform used by geometry-equivalence tests."""
    pts = np.asarray(points_xy, dtype=np.float64)
    a0 = np.asarray(gt_box_center_t0_xy, dtype=np.float64)
    ah = np.asarray(gt_box_center_future_xy, dtype=np.float64)
    R = _rot2_np(relative_yaw_rad)
    return (pts - a0[None]) @ R.T + ah[None]


def apply_source_centered_rigid_xy(
    points_xy: np.ndarray,
    source_center_xy: np.ndarray,
    source_displacement_xy: np.ndarray,
    relative_yaw_rad: float,
) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float64)
    cs = np.asarray(source_center_xy, dtype=np.float64)
    ds = np.asarray(source_displacement_xy, dtype=np.float64)
    R = _rot2_np(relative_yaw_rad)
    return (pts - cs[None]) @ R.T + cs[None] + ds[None]


def heading_in_t0_from_world_yaw(yaw_world: float, t0_ego_to_world: np.ndarray) -> float:
    """Express a horizontal world heading in the frozen t0 ego frame."""
    v_world = np.asarray(
        [math.cos(float(yaw_world)), math.sin(float(yaw_world)), 0.0],
        dtype=np.float64,
    )
    R_ego_to_world = np.asarray(t0_ego_to_world, dtype=np.float64)[:3, :3]
    if R_ego_to_world.shape != (3, 3):
        raise ValueError("t0_ego_to_world must be 4x4")
    # Row-vector convention: world -> ego uses multiplication by R.
    v_t0 = v_world @ R_ego_to_world
    if float(np.linalg.norm(v_t0[:2])) < 1e-8:
        raise ValueError("heading projection into t0 XY is degenerate")
    return math.atan2(float(v_t0[1]), float(v_t0[0]))


def relative_yaw_in_t0(
    yaw0_world: float,
    yawh_world: float,
    t0_ego_to_world: np.ndarray,
) -> float:
    h0 = heading_in_t0_from_world_yaw(yaw0_world, t0_ego_to_world)
    hh = heading_in_t0_from_world_yaw(yawh_world, t0_ego_to_world)
    return wrap_angle_np(hh - h0)


class LocalSpatialTemporalWorldModelV18SE2(LocalSpatialTemporalWorldModelV17):
    """V17-RL plus one zero-initialized scalar yaw head per future query."""

    def __init__(self, config: LocalSTWMV17Config = LocalSTWMV17Config()):
        if not bool(config.use_representation):
            raise ValueError("V18-SE2 requires the V17 representation path")
        super().__init__(config)
        self.yaw_head = nn.Linear(int(config.d_model), 1)
        nn.init.zeros_(self.yaw_head.weight)
        nn.init.zeros_(self.yaw_head.bias)

    def forward(
        self,
        features: torch.Tensor,
        local_semantic_tube: torch.Tensor,
        kta_displacement_xy_m: torch.Tensor,
        frame_motion_features: torch.Tensor | None = None,
        target_source_mask_tube: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
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
        if frame_motion_features is None or frame_motion_features.shape != (
            B, HISTORY_FRAMES, FRAME_MOTION_DIM
        ):
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
                "yaw_delta_rad": features.new_empty((0, FUTURE_FRAMES)),
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

        context = x.permute(0, 1, 3, 4, 2).reshape(
            B, HISTORY_FRAMES * Hs * Ws, cfg.d_model
        )
        q = self.future_query.expand(B, -1, -1) + self.future_time_embedding
        q = q + self.kinematic_proj(features).unsqueeze(1)
        q = q + self.kta_future_proj(kta_displacement_xy_m.to(q.dtype) / 20.0)
        for block in self.decoder:
            q = block(q, context)

        return {
            "residual_xy_m": self.residual_head(q),
            "existence_logits": self.existence_head(q)[..., 0],
            "yaw_delta_rad": self.yaw_head(q)[..., 0],
        }


def periodic_yaw_loss(
    pred_yaw_rad: torch.Tensor,
    target_yaw_rad: torch.Tensor,
    yaw_enabled: torch.Tensor,
    yaw_label_valid: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    if pred_yaw_rad.shape != target_yaw_rad.shape:
        raise ValueError("pred/target yaw shape mismatch")
    if yaw_label_valid.shape != pred_yaw_rad.shape:
        raise ValueError("yaw_label_valid shape mismatch")
    enabled = yaw_enabled.bool()
    if enabled.ndim == 1:
        enabled = enabled[:, None].expand_as(pred_yaw_rad)
    if enabled.shape != pred_yaw_rad.shape:
        raise ValueError("yaw_enabled must be [B] or [B,H]")
    usable = enabled & yaw_label_valid.bool()
    if bool(usable.any()):
        delta = wrap_angle_tensor(pred_yaw_rad - target_yaw_rad.to(pred_yaw_rad.dtype))
        per = 1.0 - torch.cos(delta)
        loss = per[usable].mean()
        mae = delta[usable].abs().mean()
        count = int(usable.sum().item())
    else:
        loss = pred_yaw_rad.sum() * 0.0
        mae = pred_yaw_rad.new_tensor(float("nan"))
        count = 0
    return loss, {
        "yaw_periodic_loss": float(loss.detach().cpu()),
        "yaw_mae_rad": float(mae.detach().cpu()),
        "yaw_labels": count,
    }


def _warp_footprint_se2(
    source_mask: torch.Tensor,
    displacement_xy_m: torch.Tensor,
    yaw_rad: torch.Tensor,
    *,
    patch_resolution_m: float,
) -> torch.Tensor:
    """Warp one source footprint with differentiable source-centred SE(2).

    source_mask: [B,H,W]
    displacement_xy_m: [B,F,2]
    yaw_rad: [B,F]

    The existing V17 convention is retained: image row corresponds to metric X
    and image column corresponds to metric Y.
    """
    if source_mask.ndim != 3:
        raise ValueError("source_mask must be [B,H,W]")
    B, H, W = source_mask.shape
    if displacement_xy_m.ndim != 3 or displacement_xy_m.shape[0] != B or displacement_xy_m.shape[-1] != 2:
        raise ValueError("displacement must be [B,F,2]")
    Fh = int(displacement_xy_m.shape[1])
    if yaw_rad.shape != (B, Fh):
        raise ValueError("yaw must be [B,F]")
    if patch_resolution_m <= 0:
        raise ValueError("patch_resolution_m must be positive")

    # One source-patch width of padding on every side matches the historical V17
    # overlap canvas while leaving room for rotation.
    canvas = F.pad(source_mask[:, None].to(displacement_xy_m.dtype), (W, W, H, H))
    Hc, Wc = int(canvas.shape[-2]), int(canvas.shape[-1])
    inp = canvas[:, None].expand(B, Fh, 1, Hc, Wc).reshape(B * Fh, 1, Hc, Wc)

    row = (
        torch.arange(Hc, device=inp.device, dtype=inp.dtype)
        - (Hc - 1) / 2.0
    ) * float(patch_resolution_m)
    col = (
        torch.arange(Wc, device=inp.device, dtype=inp.dtype)
        - (Wc - 1) / 2.0
    ) * float(patch_resolution_m)
    x_out, y_out = torch.meshgrid(row, col, indexing="ij")
    x_out = x_out[None].expand(B * Fh, -1, -1)
    y_out = y_out[None].expand(B * Fh, -1, -1)

    disp = displacement_xy_m.reshape(B * Fh, 2)
    theta = yaw_rad.reshape(B * Fh)
    # Inverse map required by grid_sample:
    # x_in = R(-theta) @ (x_out - d).
    xo = x_out - disp[:, 0, None, None]
    yo = y_out - disp[:, 1, None, None]
    c = torch.cos(theta)[:, None, None]
    s = torch.sin(theta)[:, None, None]
    x_in = c * xo + s * yo
    y_in = -s * xo + c * yo

    row_in = x_in / float(patch_resolution_m) + (Hc - 1) / 2.0
    col_in = y_in / float(patch_resolution_m) + (Wc - 1) / 2.0
    gy = 2.0 * row_in / max(Hc - 1, 1) - 1.0
    gx = 2.0 * col_in / max(Wc - 1, 1) - 1.0
    grid = torch.stack((gx, gy), dim=-1)

    return F.grid_sample(
        inp, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    ).reshape(B, Fh, Hc, Wc)


def soft_se2_transport_overlap_loss(
    pred_source_displacement_xy_m: torch.Tensor,
    target_source_displacement_xy_m: torch.Tensor,
    pred_yaw_rad: torch.Tensor,
    target_yaw_rad: torch.Tensor,
    source_footprint_mask: torch.Tensor,
    target_valid: torch.Tensor,
    yaw_enabled: torch.Tensor,
    yaw_label_valid: torch.Tensor,
    *,
    patch_resolution_m: float = 0.8,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Soft-IoU between predicted and GT SE(2)-transported source footprints.

    For classes with yaw disabled, and for future labels without valid yaw, both
    predicted and GT effective yaw are forced to zero.  Translation supervision
    remains active.  Thus future GT validity never becomes a deployment gate.
    """
    pred_d = pred_source_displacement_xy_m
    tgt_d = target_source_displacement_xy_m.to(pred_d.dtype)
    if pred_d.shape != tgt_d.shape or pred_d.ndim != 3 or pred_d.shape[-1] != 2:
        raise ValueError("source displacement tensors must be [B,F,2]")
    B, Fh, _ = pred_d.shape
    if pred_yaw_rad.shape != (B, Fh) or target_yaw_rad.shape != (B, Fh):
        raise ValueError("yaw tensors must be [B,F]")
    if target_valid.shape != (B, Fh) or yaw_label_valid.shape != (B, Fh):
        raise ValueError("valid masks must be [B,F]")
    enabled = yaw_enabled.bool()
    if enabled.ndim != 1 or enabled.shape[0] != B:
        raise ValueError("yaw_enabled must be [B]")

    yaw_use = enabled[:, None] & yaw_label_valid.bool()
    pred_yaw_eff = torch.where(
        yaw_use, pred_yaw_rad, torch.zeros_like(pred_yaw_rad)
    )
    tgt_yaw_eff = torch.where(
        yaw_use, target_yaw_rad.to(pred_yaw_rad.dtype), torch.zeros_like(pred_yaw_rad)
    )

    pred_mask = _warp_footprint_se2(
        source_footprint_mask, pred_d, pred_yaw_eff,
        patch_resolution_m=float(patch_resolution_m),
    )
    tgt_mask = _warp_footprint_se2(
        source_footprint_mask, tgt_d, tgt_yaw_eff,
        patch_resolution_m=float(patch_resolution_m),
    )
    inter = (pred_mask * tgt_mask).sum(dim=(-2, -1))
    union = (pred_mask + tgt_mask - pred_mask * tgt_mask).sum(dim=(-2, -1))
    iou = (inter + float(eps)) / (union + float(eps))

    footprint_present = source_footprint_mask.flatten(1).sum(dim=1) > 0
    usable = target_valid.bool() & footprint_present[:, None]
    if bool(usable.any()):
        loss = (1.0 - iou[usable]).mean()
        mean_iou = float(iou[usable].detach().mean().cpu())
        count = int(usable.sum().item())
    else:
        loss = pred_d.sum() * 0.0 + pred_yaw_rad.sum() * 0.0
        mean_iou = float("nan")
        count = 0
    return loss, {
        "se2_transport_soft_iou": mean_iou,
        "se2_transport_overlap_labels": count,
        "se2_yaw_active_labels": int((usable & yaw_use).sum().item()),
    }


def yaw_enabled_from_class_id(class_id: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(class_id, dtype=torch.bool)
    for cid in YAW_ENABLED_CLASS_IDS:
        out |= class_id.long() == int(cid)
    return out
