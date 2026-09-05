"""Utilities for motion-associated flow-matching error diagnostics.

The functions here are intentionally independent of the World Model and decoder.
They convert the occupancy-space Moving-v2 support to latent BEV cells and
accumulate velocity-space errors in motion-associated vs non-motion cells.
"""
from __future__ import annotations

import math

import torch


def motion_support_occ_to_latent(
    support: torch.Tensor,
    *,
    latent_hw: tuple[int, int] = (50, 50),
) -> torch.Tensor:
    """Map [T,X,Y,Z] occupancy support to [T,H,W] latent support by any pooling.

    The mapping is deliberately conservative: first take ``any`` over height Z,
    then any-pool each exact X/Y block.  No extra dilation is introduced.
    """
    if not torch.is_tensor(support) or support.ndim != 4:
        raise ValueError("motion support must be a [T,X,Y,Z] tensor")
    t, x, y, _ = map(int, support.shape)
    lh, lw = map(int, latent_hw)
    if t <= 0 or x <= 0 or y <= 0 or lh <= 0 or lw <= 0:
        raise ValueError("motion support and latent dimensions must be positive")
    if x % lh != 0 or y % lw != 0:
        raise ValueError(
            f"occupancy BEV {(x, y)} must divide exactly into latent HW {(lh, lw)}"
        )
    fx, fy = x // lh, y // lw
    bev = support.bool().any(dim=-1)
    return bev.reshape(t, lh, fx, lw, fy).any(dim=(2, 4))


def _new_group() -> dict:
    return {
        "sse": 0.0,
        "target_ss": 0.0,
        "pred_ss": 0.0,
        "dot": 0.0,
        "elements": 0,
        "cells": 0,
        "cosine_sum": 0.0,
        "cosine_count": 0,
        "empty_groups": 0,
        "zero_norm_groups": 0,
    }


def _new_row() -> dict:
    return {"moving": _new_group(), "non_moving": _new_group()}


def new_grouped_velocity_accumulator(num_frames: int = 6) -> dict:
    if int(num_frames) <= 0:
        raise ValueError("num_frames must be positive")
    return {
        "num_frames": int(num_frames),
        "num_samples": 0,
        "overall": _new_row(),
        "by_frame": [_new_row() for _ in range(int(num_frames))],
    }


def _update_group(
    group: dict,
    pred: torch.Tensor,
    target: torch.Tensor,
    cell_mask: torch.Tensor,
) -> None:
    if pred.ndim != 4 or target.shape != pred.shape:
        raise ValueError("per-frame pred/target must be matching [N,C,H,W] tensors")
    if cell_mask.ndim != 3 or tuple(cell_mask.shape) != (
        int(pred.shape[0]),
        int(pred.shape[-2]),
        int(pred.shape[-1]),
    ):
        raise ValueError("per-frame cell mask must be [N,H,W] and match pred")

    cell_mask = cell_mask.bool()
    group["cells"] += int(cell_mask.sum().item())
    select = cell_mask[:, None].expand(-1, pred.shape[1], -1, -1)
    count = int(select.sum().item())
    group["elements"] += count
    if count == 0:
        group["empty_groups"] += 1
        return

    pf = pred.float()[select]
    tf = target.float()[select]
    err = pf - tf
    sse = float(err.square().sum().item())
    target_ss = float(tf.square().sum().item())
    pred_ss = float(pf.square().sum().item())
    dot = float((pf * tf).sum().item())
    group["sse"] += sse
    group["target_ss"] += target_ss
    group["pred_ss"] += pred_ss
    group["dot"] += dot

    denom = math.sqrt(pred_ss) * math.sqrt(target_ss)
    if denom <= 0.0:
        group["zero_norm_groups"] += 1
    else:
        group["cosine_sum"] += dot / denom
        group["cosine_count"] += 1


def update_grouped_velocity_accumulator(
    accumulator: dict,
    predicted_velocity: torch.Tensor,
    target_velocity: torch.Tensor,
    motion_mask: torch.Tensor,
) -> None:
    """Accumulate one sample's valid Top-K windows.

    Args:
        predicted_velocity/target_velocity: [N,T,C,H,W].
        motion_mask: [N,T,H,W].  Overlapping windows remain duplicated exactly
            as they are in the sparse FM training objective.

    Cosine is macro-averaged over sample/horizon groups. MSE/NMSE are micro
    statistics from exact sums and element counts.
    """
    if predicted_velocity.ndim != 5 or target_velocity.shape != predicted_velocity.shape:
        raise ValueError("predicted/target velocity must be matching [N,T,C,H,W]")
    if motion_mask.ndim != 4:
        raise ValueError("motion_mask must be [N,T,H,W]")
    n, t, _, h, w = map(int, predicted_velocity.shape)
    if tuple(motion_mask.shape) != (n, t, h, w):
        raise ValueError("motion_mask shape differs from velocity spatial contract")
    if int(accumulator["num_frames"]) != t:
        raise ValueError("accumulator future-frame count differs")

    pred = predicted_velocity.detach().float()
    target = target_velocity.detach().float()
    mask = motion_mask.detach().bool()
    for fi in range(t):
        p = pred[:, fi]
        y = target[:, fi]
        m = mask[:, fi]
        _update_group(accumulator["overall"]["moving"], p, y, m)
        _update_group(accumulator["overall"]["non_moving"], p, y, ~m)
        _update_group(accumulator["by_frame"][fi]["moving"], p, y, m)
        _update_group(accumulator["by_frame"][fi]["non_moving"], p, y, ~m)
    accumulator["num_samples"] += 1


def _finish_group(group: dict) -> dict:
    n = int(group["elements"])
    cells = int(group["cells"])
    target_ss = float(group["target_ss"])
    out = {
        "elements": n,
        "cells": cells,
        "sse": float(group["sse"]),
        "target_ss": target_ss,
        "pred_ss": float(group["pred_ss"]),
        "mse": (float(group["sse"]) / n) if n else None,
        "nmse": (float(group["sse"]) / target_ss) if target_ss > 0.0 else None,
        "target_rms": math.sqrt(target_ss / n) if n else None,
        "pred_rms": math.sqrt(float(group["pred_ss"]) / n) if n else None,
        "cosine_macro": (
            float(group["cosine_sum"]) / int(group["cosine_count"])
            if int(group["cosine_count"]) > 0
            else None
        ),
        "cosine_groups": int(group["cosine_count"]),
        "empty_groups": int(group["empty_groups"]),
        "zero_norm_groups": int(group["zero_norm_groups"]),
    }
    return out


def _finish_row(row: dict, *, motion_weight_lambda: float) -> dict:
    moving = _finish_group(row["moving"])
    non = _finish_group(row["non_moving"])
    total_elements = moving["elements"] + non["elements"]
    total_cells = moving["cells"] + non["cells"]
    total_sse = moving["sse"] + non["sse"]
    global_mse = total_sse / total_elements if total_elements else None
    recomposed = None
    if total_elements:
        recomposed = (
            moving["mse"] * moving["elements"] if moving["mse"] is not None else 0.0
        ) + (
            non["mse"] * non["elements"] if non["mse"] is not None else 0.0
        )
        recomposed /= total_elements

    lam = float(motion_weight_lambda)
    if lam < 0.0:
        raise ValueError("motion_weight_lambda must be non-negative")
    weighted_motion_mass = None
    if total_cells:
        weighted_motion_mass = (1.0 + lam) * moving["cells"] / (
            (1.0 + lam) * moving["cells"] + non["cells"]
        )
    return {
        "global_mse": global_mse,
        "moving": moving,
        "non_moving": non,
        "motion_cell_fraction": moving["cells"] / total_cells if total_cells else None,
        "motion_squared_error_share": moving["sse"] / total_sse if total_sse > 0.0 else None,
        "effective_motion_weight_mass": weighted_motion_mass,
        "motion_weight_lambda": lam,
        "recomposed_global_mse": recomposed,
        "recomposition_abs_error": (
            abs(global_mse - recomposed)
            if global_mse is not None and recomposed is not None
            else None
        ),
    }


def finalize_grouped_velocity_accumulator(
    accumulator: dict,
    *,
    motion_weight_lambda: float = 2.0,
) -> dict:
    return {
        "num_samples": int(accumulator["num_samples"]),
        "overall": _finish_row(
            accumulator["overall"], motion_weight_lambda=motion_weight_lambda
        ),
        "by_frame": [
            _finish_row(row, motion_weight_lambda=motion_weight_lambda)
            for row in accumulator["by_frame"]
        ],
        "cosine_aggregation": "macro over valid sample-horizon group vectors",
        "mse_nmse_aggregation": "micro exact sums over valid Top-K window elements",
    }
