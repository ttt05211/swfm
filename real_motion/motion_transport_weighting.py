"""Loss weighting helpers for P0-F9 v15 learned motion transport.

The v14 diagnostic showed two objective mismatches:
1) only about one third of valid dynamic-class observations are truly moving
   under the frozen Moving-mIoU v2 rule;
2) Moving-mIoU is macro-averaged over dynamic semantic classes while raw
   trajectory supervision is micro-averaged over observations.

This module keeps the v13 motion head and displacement target unchanged and only
changes how valid trajectory observations contribute to Smooth-L1.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS, SPEED_THRESHOLD_MPS

MODE_MOTION = "motion"
MODE_MOTION_CLASS = "motion_class"
WEIGHTING_MODES = (MODE_MOTION, MODE_MOTION_CLASS)
DEFAULT_MOTION_WEIGHT = 2.0


def true_moving_mask_torch(
    target_displacement_xy_m: torch.Tensor,
    target_valid: torch.Tensor,
    *,
    frame_dt_s: float = 0.5,
    speed_threshold_mps: float = SPEED_THRESHOLD_MPS,
) -> torch.Tensor:
    """Exact true-motion observation mask used by Moving-mIoU v2.

    Motion is decided by interval center speed from t0 to each future horizon.
    GT displacement is training/validation supervision only and never becomes a
    causal model input.
    """
    disp = target_displacement_xy_m
    valid = target_valid.bool()
    if disp.ndim != 3 or disp.shape[-1] != 2 or valid.shape != disp.shape[:2]:
        raise ValueError("expected displacement [N,H,2] and valid [N,H]")
    h = disp.shape[1]
    dt = (
        torch.arange(1, h + 1, device=disp.device, dtype=disp.dtype)
        * float(frame_dt_s)
    ).view(1, h)
    speed = torch.linalg.vector_norm(disp, dim=-1) / dt
    return valid & (speed >= float(speed_threshold_mps))


def compute_macro_class_weights(
    source_class_id: torch.Tensor,
    true_moving_mask: torch.Tensor,
    *,
    dynamic_classes: Sequence[int] = DYNAMIC_CLASS_IDS,
) -> dict[int, float]:
    """Equalize total true-moving weight across classes present in training.

    For class c with n_c true-moving observations, weight_c = N/(K*n_c), where
    N is the total true-moving observation count and K is the number of classes
    with at least one such observation.  Therefore the mean class factor over all
    true-moving training observations is exactly one: class balancing only
    redistributes true-moving mass and does not increase it.
    """
    cls = source_class_id.detach().cpu().long().numpy()
    moving = true_moving_mask.detach().cpu().bool().numpy()
    if cls.ndim != 1 or moving.ndim != 2 or moving.shape[0] != cls.shape[0]:
        raise ValueError("source_class_id must be [N] and true_moving_mask [N,H]")
    counts: dict[int, int] = {}
    for c in dynamic_classes:
        counts[int(c)] = int(moving[cls == int(c)].sum())
    present = [c for c, n in counts.items() if n > 0]
    total = int(sum(counts.values()))
    if total <= 0 or not present:
        raise ValueError("training cache has no true-moving observations")
    k = len(present)
    return {
        int(c): (float(total) / (float(k) * float(counts[int(c)])) if counts[int(c)] > 0 else 0.0)
        for c in dynamic_classes
    }


def make_observation_weights(
    source_class_id: torch.Tensor,
    true_moving_mask: torch.Tensor,
    target_valid: torch.Tensor,
    *,
    mode: str,
    class_weights: Mapping[int, float] | None = None,
    motion_weight: float = DEFAULT_MOTION_WEIGHT,
) -> torch.Tensor:
    """Return normalized-contract per-observation weights [N,H].

    Non-moving valid observations always have weight 1.  True-moving observations
    have weight ``motion_weight``.  In ``motion_class`` mode, only true-moving
    observations receive the additional training-set macro class factor.
    """
    if mode not in WEIGHTING_MODES:
        raise ValueError(f"unsupported weighting mode: {mode}")
    if float(motion_weight) <= 0:
        raise ValueError("motion_weight must be positive")
    valid = target_valid.bool()
    moving = true_moving_mask.bool()
    cls = source_class_id.long()
    if moving.shape != valid.shape or moving.shape[0] != cls.shape[0]:
        raise ValueError("weighting tensor shape mismatch")
    w = torch.ones(valid.shape, dtype=torch.float32, device=valid.device)
    w = torch.where(moving, torch.full_like(w, float(motion_weight)), w)
    if mode == MODE_MOTION_CLASS:
        if class_weights is None:
            raise ValueError("motion_class mode requires training-set class_weights")
        factor = torch.ones((cls.shape[0],), dtype=w.dtype, device=w.device)
        for c in DYNAMIC_CLASS_IDS:
            factor[cls == int(c)] = float(class_weights.get(int(c), 0.0))
        w = torch.where(moving, w * factor[:, None], w)
    return torch.where(valid, w, torch.zeros_like(w))


def weighted_motion_transport_loss(
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    observation_weights: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    """v13 loss with only trajectory-observation weighting changed.

    Existence BCE is intentionally identical to v13 so M/MC are controlled
    trajectory-objective experiments.
    """
    pred = outputs["residual_xy_m"]
    target = batch["target_residual_xy_m"].to(pred.dtype)
    valid = batch["target_valid"].bool()
    w = observation_weights.to(device=pred.device, dtype=pred.dtype)
    if w.shape != valid.shape:
        raise ValueError("observation weight shape mismatch")
    if bool(valid.any()):
        per_obs = F.smooth_l1_loss(pred, target, reduction="none", beta=1.0).mean(dim=-1)
        denom = w[valid].sum().clamp_min(1e-12)
        traj = (per_obs[valid] * w[valid]).sum() / denom
    else:
        traj = pred.sum() * 0.0

    logits = outputs["existence_logits"]
    labels = batch["existence"].to(logits.dtype)
    supervised = batch.get("supervised_source")
    if supervised is None:
        emask = torch.ones_like(labels, dtype=torch.bool)
    else:
        emask = supervised.bool().unsqueeze(-1).expand_as(labels)
    if bool(emask.any()):
        exist = F.binary_cross_entropy_with_logits(logits[emask], labels[emask])
    else:
        exist = logits.sum() * 0.0
    total = traj + exist
    return total, {
        "loss": float(total.detach().cpu()),
        "trajectory_smooth_l1_weighted": float(traj.detach().cpu()),
        "existence_bce": float(exist.detach().cpu()),
        "trajectory_labels": int(valid.sum().item()),
        "trajectory_weight_mass": float(w[valid].sum().detach().cpu()) if bool(valid.any()) else 0.0,
        "existence_labels": int(emask.sum().item()),
    }


def weight_contract_summary(
    source_class_id: torch.Tensor,
    true_moving_mask: torch.Tensor,
    target_valid: torch.Tensor,
    observation_weights: torch.Tensor,
    class_weights: Mapping[int, float] | None,
) -> dict:
    valid = target_valid.bool()
    moving = true_moving_mask.bool()
    w = observation_weights.float()
    valid_count = int(valid.sum().item())
    moving_count = int(moving.sum().item())
    per_class = {}
    cls = source_class_id.long()
    for c in DYNAMIC_CLASS_IDS:
        cm = moving & (cls[:, None] == int(c))
        per_class[int(c)] = {
            "true_moving_observations": int(cm.sum().item()),
            "class_factor": None if class_weights is None else float(class_weights.get(int(c), 0.0)),
            "weighted_mass": float(w[cm].sum().item()) if bool(cm.any()) else 0.0,
        }
    return {
        "valid_observations": valid_count,
        "true_moving_observations": moving_count,
        "true_moving_fraction": moving_count / max(valid_count, 1),
        "total_weight_mass": float(w[valid].sum().item()) if bool(valid.any()) else 0.0,
        "effective_true_moving_weight_fraction": (
            float(w[moving].sum().item()) / max(float(w[valid].sum().item()), 1e-12)
            if bool(valid.any()) else float("nan")
        ),
        "per_class": per_class,
    }
