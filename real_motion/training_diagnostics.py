"""Small dependency-light utilities for P0-F9 training diagnostics.

The helpers in this module intentionally avoid any nuScenes/OccFM imports so
that class-distribution and gradient-statistic logic can be unit tested in CI.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
import torch


OCC3D_CLASS_NAMES = (
    "others",
    "barrier",
    "bicycle",
    "bus",
    "car",
    "construction_vehicle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "trailer",
    "truck",
    "driveable_surface",
    "other_flat",
    "sidewalk",
    "terrain",
    "manmade",
    "vegetation",
    "free",
)
OCC3D_FREE_ID = 17
OCC3D_DYNAMIC_IDS = (2, 3, 4, 5, 6, 7, 9, 10)


def class_histogram(labels, *, num_classes: int = 18, ignore_label: int | None = 255) -> np.ndarray:
    """Count semantic labels without silently folding invalid values into a class."""
    x = np.asarray(labels).reshape(-1)
    if ignore_label is not None:
        x = x[x != int(ignore_label)]
    if x.size == 0:
        return np.zeros(int(num_classes), dtype=np.int64)
    if int(x.min()) < 0 or int(x.max()) >= int(num_classes):
        bad = x[(x < 0) | (x >= int(num_classes))]
        raise ValueError(f"semantic labels outside [0,{num_classes}): {np.unique(bad)[:8].tolist()}")
    return np.bincount(x.astype(np.int64, copy=False), minlength=int(num_classes)).astype(np.int64)


def summarize_class_histogram(
    counts: Sequence[int] | np.ndarray,
    *,
    class_names: Sequence[str] = OCC3D_CLASS_NAMES,
    free_id: int = OCC3D_FREE_ID,
    dynamic_ids: Iterable[int] = OCC3D_DYNAMIC_IDS,
) -> dict:
    """Return all-voxel and occupied-only frequencies plus dynamic mass."""
    c = np.asarray(counts, dtype=np.int64)
    if c.ndim != 1 or c.shape[0] != len(class_names):
        raise ValueError("counts/class_names shape mismatch")
    if np.any(c < 0):
        raise ValueError("class counts must be non-negative")
    total = int(c.sum())
    free_id = int(free_id)
    if not 0 <= free_id < len(c):
        raise ValueError("free_id out of range")
    occupied = total - int(c[free_id])
    dyn_ids = tuple(int(i) for i in dynamic_ids)
    if any(i < 0 or i >= len(c) for i in dyn_ids):
        raise ValueError("dynamic class id out of range")
    dyn = int(c[list(dyn_ids)].sum()) if dyn_ids else 0

    all_frac = c.astype(np.float64) / float(total) if total > 0 else np.zeros_like(c, dtype=np.float64)
    occ_frac = np.zeros_like(c, dtype=np.float64)
    if occupied > 0:
        occ_frac = c.astype(np.float64) / float(occupied)
        occ_frac[free_id] = 0.0
    return {
        "counts": {str(class_names[i]): int(c[i]) for i in range(len(c))},
        "fraction_all_voxels": {str(class_names[i]): float(all_frac[i]) for i in range(len(c))},
        "fraction_occupied_only": {str(class_names[i]): float(occ_frac[i]) for i in range(len(c))},
        "total_voxels": total,
        "occupied_voxels": occupied,
        "occupied_fraction": float(occupied / total) if total > 0 else float("nan"),
        "dynamic_voxels": dyn,
        "dynamic_fraction_all_voxels": float(dyn / total) if total > 0 else float("nan"),
        "dynamic_fraction_occupied_only": float(dyn / occupied) if occupied > 0 else float("nan"),
    }


def enrichment_ratio(
    target_counts: Sequence[int] | np.ndarray,
    reference_counts: Sequence[int] | np.ndarray,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """Per-class ratio of normalized target frequency to normalized reference frequency."""
    a = np.asarray(target_counts, dtype=np.float64)
    b = np.asarray(reference_counts, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("target/reference histogram shape mismatch")
    if a.sum() <= 0 or b.sum() <= 0:
        raise ValueError("histograms must be non-empty")
    pa = a / a.sum()
    pb = b / b.sum()
    return pa / np.maximum(pb, float(eps))


def jensen_shannon_divergence(
    counts_a: Sequence[int] | np.ndarray,
    counts_b: Sequence[int] | np.ndarray,
    *,
    eps: float = 1e-12,
) -> float:
    """Jensen-Shannon divergence in nats between two count histograms."""
    a = np.asarray(counts_a, dtype=np.float64)
    b = np.asarray(counts_b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("histogram shape mismatch")
    if a.sum() <= 0 or b.sum() <= 0:
        raise ValueError("histograms must be non-empty")
    p = a / a.sum()
    q = b / b.sum()
    m = 0.5 * (p + q)

    def kl(x, y):
        mask = x > 0
        return float(np.sum(x[mask] * (np.log(np.maximum(x[mask], eps)) - np.log(np.maximum(y[mask], eps)))))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def gradient_pair_stats(
    grads_a: Sequence[torch.Tensor | None],
    grads_b: Sequence[torch.Tensor | None],
) -> dict:
    """Cosine/norm/sign-conflict statistics without concatenating large gradients."""
    if len(grads_a) != len(grads_b):
        raise ValueError("gradient list length mismatch")
    dot = 0.0
    norm_a_sq = 0.0
    norm_b_sq = 0.0
    conflict = 0
    both_nonzero = 0
    used_tensors = 0
    total_elements = 0
    for ga, gb in zip(grads_a, grads_b):
        if ga is None and gb is None:
            continue
        if ga is None:
            ga = torch.zeros_like(gb)
        if gb is None:
            gb = torch.zeros_like(ga)
        a = ga.detach().float()
        b = gb.detach().float()
        dot += float((a * b).sum().cpu())
        norm_a_sq += float(a.square().sum().cpu())
        norm_b_sq += float(b.square().sum().cpu())
        nz = (a != 0) & (b != 0)
        both_nonzero += int(nz.sum().cpu())
        conflict += int(((a * b < 0) & nz).sum().cpu())
        used_tensors += 1
        total_elements += int(a.numel())
    norm_a = math.sqrt(max(norm_a_sq, 0.0))
    norm_b = math.sqrt(max(norm_b_sq, 0.0))
    cosine = dot / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else float("nan")
    return {
        "dot": dot,
        "norm_a": norm_a,
        "norm_b": norm_b,
        "cosine": cosine,
        "opposite_sign_elements": conflict,
        "opposite_sign_fraction_on_joint_nonzero": (
            float(conflict / both_nonzero) if both_nonzero > 0 else float("nan")
        ),
        "joint_nonzero_elements": both_nonzero,
        "total_elements": total_elements,
        "used_tensors": used_tensors,
    }
