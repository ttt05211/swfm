"""Dependency-light geometry helpers for V16-STWM versus v13 MLP diagnosis.

These helpers intentionally operate only on displacement residuals. They do not
know about nuScenes, occupancy rasterization, or model implementations, so the
geometric contracts can be unit-tested without heavyweight dependencies.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np


def residual_error_components(
    pred_residual_xy_m: np.ndarray,
    target_residual_xy_m: np.ndarray,
    target_displacement_xy_m: np.ndarray,
) -> dict[str, np.ndarray]:
    """Decompose displacement error into Euclidean, along- and cross-track terms.

    The reference direction is the GT t0->future displacement. Because prediction
    and target use the same KTA-residual contract, ``pred-target`` is exactly the
    final displacement error. Rows with zero GT displacement receive a zero
    reference direction and should normally be excluded from true-moving reports.
    """
    pred = np.asarray(pred_residual_xy_m, dtype=np.float64)
    target = np.asarray(target_residual_xy_m, dtype=np.float64)
    gt_disp = np.asarray(target_displacement_xy_m, dtype=np.float64)
    if pred.shape != target.shape or pred.shape != gt_disp.shape:
        raise ValueError("prediction/target/displacement shapes must match")
    if pred.ndim != 3 or pred.shape[-1] != 2:
        raise ValueError("expected [N,H,2] residual tensors")

    err = pred - target
    euclidean = np.linalg.norm(err, axis=-1)
    norm = np.linalg.norm(gt_disp, axis=-1, keepdims=True)
    direction = np.divide(gt_disp, norm, out=np.zeros_like(gt_disp), where=norm > 1e-12)
    perpendicular = np.stack((-direction[..., 1], direction[..., 0]), axis=-1)
    along_signed = np.sum(err * direction, axis=-1)
    cross_signed = np.sum(err * perpendicular, axis=-1)
    return {
        "euclidean_m": euclidean,
        "along_signed_m": along_signed,
        "cross_signed_m": cross_signed,
        "along_abs_m": np.abs(along_signed),
        "cross_abs_m": np.abs(cross_signed),
    }


def kta_error_components(
    target_residual_xy_m: np.ndarray,
    target_displacement_xy_m: np.ndarray,
) -> dict[str, np.ndarray]:
    """KTA is the zero-residual predictor under the frozen displacement contract."""
    target = np.asarray(target_residual_xy_m)
    return residual_error_components(np.zeros_like(target), target, target_displacement_xy_m)


def weighted_masked_mean(values: np.ndarray, weights: np.ndarray, mask: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    if x.shape != m.shape:
        raise ValueError("values/mask shape mismatch")
    if w.ndim == 1:
        if w.shape[0] != x.shape[0]:
            raise ValueError("source weight length mismatch")
        w = np.broadcast_to(w[:, None], x.shape)
    if w.shape != x.shape:
        raise ValueError("weights must be [N] or [N,H]")
    ww = w[m]
    if ww.size == 0 or float(ww.sum()) <= 0.0:
        return float("nan")
    return float(np.sum(x[m] * ww) / np.sum(ww))


def masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    if x.shape != m.shape:
        raise ValueError("values/mask shape mismatch")
    return float(x[m].mean()) if bool(m.any()) else float("nan")


def summarize_model_components(
    components: Mapping[str, np.ndarray],
    mask: np.ndarray,
    source_voxel_count: np.ndarray,
) -> dict[str, float | int]:
    m = np.asarray(mask, dtype=bool)
    return {
        "count": int(m.sum()),
        "ade_m": masked_mean(components["euclidean_m"], m),
        "voxel_weighted_ade_m": weighted_masked_mean(
            components["euclidean_m"], source_voxel_count, m
        ),
        "along_abs_m": masked_mean(components["along_abs_m"], m),
        "cross_abs_m": masked_mean(components["cross_abs_m"], m),
        "voxel_weighted_along_abs_m": weighted_masked_mean(
            components["along_abs_m"], source_voxel_count, m
        ),
        "voxel_weighted_cross_abs_m": weighted_masked_mean(
            components["cross_abs_m"], source_voxel_count, m
        ),
        "along_signed_bias_m": masked_mean(components["along_signed_m"], m),
        "cross_signed_bias_m": masked_mean(components["cross_signed_m"], m),
    }


def three_way_winner(
    kta_error_m: np.ndarray,
    mlp_error_m: np.ndarray,
    stwm_error_m: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Return 0=KTA, 1=MLP, 2=STWM; ties conservatively prefer earlier choices."""
    k = np.asarray(kta_error_m, dtype=np.float64)
    m = np.asarray(mlp_error_m, dtype=np.float64)
    s = np.asarray(stwm_error_m, dtype=np.float64)
    v = np.asarray(valid, dtype=bool)
    if k.shape != m.shape or k.shape != s.shape or k.shape != v.shape:
        raise ValueError("winner tensors must share [N,H] shape")
    stacked = np.stack((k, m, s), axis=-1)
    out = np.argmin(stacked, axis=-1).astype(np.int8)
    out[~v] = -1
    return out


def pairwise_winner(
    first_error_m: np.ndarray,
    second_error_m: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Return 0 for first and 1 for second; ties prefer first; invalid=-1."""
    a = np.asarray(first_error_m, dtype=np.float64)
    b = np.asarray(second_error_m, dtype=np.float64)
    v = np.asarray(valid, dtype=bool)
    if a.shape != b.shape or a.shape != v.shape:
        raise ValueError("winner tensors must share [N,H] shape")
    out = (b < a).astype(np.int8)
    out[~v] = -1
    return out
