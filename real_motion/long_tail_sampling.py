"""Long-tail source sampling for clean one-stage V18-SE2 training.

The sampler changes only which already-supervised source examples are drawn.
Model architecture, targets, losses and inference are untouched.

Two independent training-set-only factors are combined:

1. semantic class frequency
       w_cls(i) = n_{c_i}^{-1/2}
2. KTA-relative motion-target density
       q_i = mean_h ||d^*_{i,h} - d^KTA_{i,h}||_2
       w_motion(i) = p_tilde(q_i)^{-1/2}

where p_tilde is a Gaussian-smoothed histogram estimated only from the training
cache.  The product is normalized to mean one and capped before being passed to
WeightedRandomSampler.  No validation labels, model errors, motorcycle/bicycle
special cases, or inference-time gates are used.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


PROTOCOL = "p0_f9_v18_long_tail_balanced_source_sampler_v1"
CLASS_POWER = 0.5
MOTION_DENSITY_POWER = 0.5
MOTION_DENSITY_SIGMA_BINS = 1.0
MAX_NORMALIZED_WEIGHT = 4.0
MIN_AUTO_BINS = 16
MAX_AUTO_BINS = 128


@dataclass(frozen=True)
class BalancedSourceWeights:
    weights: torch.Tensor
    motion_difficulty_m: torch.Tensor
    class_weights: torch.Tensor
    motion_weights: torch.Tensor
    report: dict[str, object]


def source_motion_difficulty(
    target_source_residual_xy_m: torch.Tensor,
    se2_target_valid: torch.Tensor,
) -> torch.Tensor:
    """Mean valid-horizon KTA-relative source correction magnitude per source."""
    residual = torch.as_tensor(target_source_residual_xy_m, dtype=torch.float32)
    valid = torch.as_tensor(se2_target_valid, dtype=torch.bool)
    if residual.ndim != 3 or residual.shape[-1] != 2:
        raise ValueError("target_source_residual_xy_m must be [N,H,2]")
    if valid.shape != residual.shape[:2]:
        raise ValueError("se2_target_valid must be [N,H]")
    mag = torch.linalg.vector_norm(residual, dim=-1)
    count = valid.sum(dim=1)
    if bool((count == 0).any()):
        bad = torch.nonzero(count == 0, as_tuple=False).flatten().tolist()
        raise ValueError(f"sources without valid SE2 targets: {bad[:8]}")
    return (mag * valid.to(mag.dtype)).sum(dim=1) / count.to(mag.dtype)


def _auto_histogram_edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("motion difficulty values must be finite 1-D data")
    lo = float(values.min())
    hi = float(values.max())
    if not hi > lo:
        return np.asarray([lo - 0.5, hi + 0.5], dtype=np.float64)
    q25, q75 = np.quantile(values, [0.25, 0.75])
    iqr = float(q75 - q25)
    if iqr > 0.0:
        width = 2.0 * iqr * (values.size ** (-1.0 / 3.0))
        bins = int(math.ceil((hi - lo) / max(width, np.finfo(np.float64).eps)))
    else:
        bins = int(round(math.sqrt(values.size)))
    bins = max(MIN_AUTO_BINS, min(MAX_AUTO_BINS, bins))
    return np.linspace(lo, hi, bins + 1, dtype=np.float64)


def _gaussian_kernel_1d(sigma_bins: float) -> np.ndarray:
    sigma = float(sigma_bins)
    if sigma <= 0.0:
        raise ValueError("sigma_bins must be positive")
    radius = max(1, int(math.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def _class_weights(class_ids: np.ndarray, power: float) -> tuple[np.ndarray, dict[int, int]]:
    ids = np.asarray(class_ids, dtype=np.int64)
    if ids.ndim != 1 or ids.size == 0:
        raise ValueError("class_ids must be non-empty 1-D data")
    classes, counts = np.unique(ids, return_counts=True)
    table = {int(c): int(n) for c, n in zip(classes, counts)}
    out = np.asarray([table[int(c)] ** (-float(power)) for c in ids], dtype=np.float64)
    out /= float(out.mean())
    return out, table


def _motion_density_weights(
    difficulty: np.ndarray,
    *,
    power: float,
    sigma_bins: float,
) -> tuple[np.ndarray, dict[str, object]]:
    q = np.asarray(difficulty, dtype=np.float64)
    edges = _auto_histogram_edges(q)
    if len(edges) == 2 and np.allclose(q, q[0]):
        return np.ones_like(q), {
            "bin_edges_m": edges.tolist(),
            "raw_bin_counts": [int(q.size)],
            "smoothed_bin_counts": [float(q.size)],
        }
    counts, _ = np.histogram(q, bins=edges)
    kernel = _gaussian_kernel_1d(float(sigma_bins))
    smooth = np.convolve(counts.astype(np.float64), kernel, mode="same")
    # A strictly positive floor prevents an isolated occupied edge bin from
    # receiving an unbounded inverse-density weight after smoothing.
    smooth = np.maximum(smooth, 1.0 / max(int(q.size), 1))
    idx = np.searchsorted(edges, q, side="right") - 1
    idx = np.clip(idx, 0, len(counts) - 1)
    density = smooth[idx] / float(smooth.sum())
    out = density ** (-float(power))
    out /= float(out.mean())
    return out, {
        "bin_edges_m": edges.tolist(),
        "raw_bin_counts": [int(x) for x in counts],
        "smoothed_bin_counts": [float(x) for x in smooth],
    }


def _quantiles(values: np.ndarray) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    ps = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
    return {f"p{int(round(p * 100)):02d}": float(np.quantile(x, p)) for p in ps}


def _class_summary(class_ids: np.ndarray, q: np.ndarray, weights: np.ndarray) -> dict[str, object]:
    ids = np.asarray(class_ids, dtype=np.int64)
    out: dict[str, object] = {}
    total_weight = float(weights.sum())
    for cid in sorted(int(x) for x in np.unique(ids)):
        m = ids == cid
        qq = q[m]
        out[str(cid)] = {
            "sources": int(m.sum()),
            "source_fraction": float(m.mean()),
            "expected_sample_fraction": float(weights[m].sum() / total_weight),
            "motion_difficulty_m": {
                "mean": float(qq.mean()),
                "median": float(np.median(qq)),
                "p75": float(np.quantile(qq, 0.75)),
                "p90": float(np.quantile(qq, 0.90)),
                "p95": float(np.quantile(qq, 0.95)),
            },
            "mean_final_weight": float(weights[m].mean()),
        }
    return out


def build_balanced_source_weights(
    source_class_id: torch.Tensor,
    target_source_residual_xy_m: torch.Tensor,
    se2_target_valid: torch.Tensor,
    *,
    class_power: float = CLASS_POWER,
    motion_density_power: float = MOTION_DENSITY_POWER,
    sigma_bins: float = MOTION_DENSITY_SIGMA_BINS,
    max_normalized_weight: float = MAX_NORMALIZED_WEIGHT,
) -> BalancedSourceWeights:
    ids_t = torch.as_tensor(source_class_id, dtype=torch.long).cpu()
    if ids_t.ndim != 1:
        raise ValueError("source_class_id must be [N]")
    q_t = source_motion_difficulty(target_source_residual_xy_m, se2_target_valid).cpu()
    if q_t.shape != ids_t.shape:
        raise ValueError("source_class_id and targets have different N")
    ids = ids_t.numpy().astype(np.int64, copy=False)
    q = q_t.numpy().astype(np.float64, copy=False)

    cw, class_counts = _class_weights(ids, float(class_power))
    mw, density_report = _motion_density_weights(
        q, power=float(motion_density_power), sigma_bins=float(sigma_bins)
    )
    raw = cw * mw
    raw /= float(raw.mean())
    if float(max_normalized_weight) <= 0.0:
        raise ValueError("max_normalized_weight must be positive")
    weights = np.minimum(raw, float(max_normalized_weight))
    if not np.isfinite(weights).all() or bool((weights <= 0.0).any()):
        raise RuntimeError("balanced source weights must be finite and positive")

    top_threshold = float(np.quantile(q, 0.90))
    top = q >= top_threshold
    top_class_counts = {
        str(int(cid)): int(((ids == int(cid)) & top).sum())
        for cid in sorted(int(x) for x in np.unique(ids))
    }
    effective_n = float(weights.sum() ** 2 / np.square(weights).sum())
    report = {
        "protocol": PROTOCOL,
        "definition": {
            "motion_difficulty": "mean_valid_horizon_l2_norm_of_target_source_residual_xy_m",
            "class_weight": f"class_count^-{float(class_power):g}",
            "motion_weight": f"gaussian_smoothed_motion_density^-{float(motion_density_power):g}",
            "motion_density_sigma_bins": float(sigma_bins),
            "histogram_bins": "Freedman-Diaconis clipped to [16,128]",
            "combined": "normalize_mean_one_then_upper_cap",
            "max_normalized_weight": float(max_normalized_weight),
            "sampling": "WeightedRandomSampler replacement=True num_samples=N",
        },
        "sources": int(ids.size),
        "class_counts": {str(k): int(v) for k, v in class_counts.items()},
        "motion_difficulty_m_quantiles": _quantiles(q),
        "global_top10_motion_threshold_m": top_threshold,
        "global_top10_motion_class_counts": top_class_counts,
        "weight_quantiles": _quantiles(weights),
        "effective_sample_size": effective_n,
        "effective_sample_fraction": effective_n / float(ids.size),
        "classes": _class_summary(ids, q, weights),
        "motion_density": density_report,
    }
    return BalancedSourceWeights(
        weights=torch.as_tensor(weights, dtype=torch.double),
        motion_difficulty_m=q_t.float(),
        class_weights=torch.as_tensor(cw, dtype=torch.float32),
        motion_weights=torch.as_tensor(mw, dtype=torch.float32),
        report=report,
    )


def make_balanced_source_sampler(
    balanced: BalancedSourceWeights,
    *,
    seed: int,
) -> WeightedRandomSampler:
    gen = torch.Generator().manual_seed(int(seed))
    n = int(balanced.weights.numel())
    return WeightedRandomSampler(
        weights=balanced.weights,
        num_samples=n,
        replacement=True,
        generator=gen,
    )
