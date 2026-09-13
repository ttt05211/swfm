"""Source-level selective forecasting utilities for frozen KTA/V17 experts.

The feasibility experiment intentionally keeps both experts frozen:
  * physics expert: Strong-W2Det/KTA rigid transport
  * learned expert: V17-RL rigid transport with A1 source-order composition

Future GT is used only to build selector supervision/oracle scores. Deployable
selector inputs are the existing causal motion-transport features only.

The oracle label is an exact single-source marginal under the frozen
Moving-mIoU-v2 dataset accumulator: replace one KTA source by the V17 source at
all report horizons while every other source remains KTA, then measure the
resulting dataset-level Moving-mIoU change. This avoids center-error/ADE labels
and does not require GT instance matching.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from .geometry import OccupancyGrid
from .metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS, REPORT_HORIZONS_S
from .motion_transport import FEATURE_DIM, FEATURE_NAMES
from .rigid_transport import RasterizedRigidComponent

SELECTIVE_LABEL_CACHE_VERSION = "p0_f9_v17_selective_expert_labels_v1"
SELECTOR_PROTOCOL = "p0_f9_v17_correction_selector_v1"
UTILITY_CONTRACT = "exact_single_source_marginal_moving_miou_v2_a1_vs_kta_v1"
SELECTOR_INPUT_CONTRACT = "causal_motion_transport_features_only_v1"
BUDGET_CONTRACT = "per_window_top_fraction_selected_sources_v1"

DYNAMIC_CLASSES = tuple(int(c) for c in DYNAMIC_CLASS_IDS)
CLASS_TO_SLOT = {c: i for i, c in enumerate(DYNAMIC_CLASSES)}
SPEED_FEATURE_INDEX = FEATURE_NAMES.index("current_speed_norm")


def moving_counts(pred, gt, support) -> tuple[np.ndarray, np.ndarray]:
    """Return per-class Moving-mIoU intersection/union counts."""
    p = np.asarray(pred)
    g = np.asarray(gt)
    s = np.asarray(support, dtype=bool)
    if p.shape != g.shape or p.shape != s.shape:
        raise ValueError("moving_counts shape mismatch")
    inter = np.zeros(len(DYNAMIC_CLASSES), dtype=np.int64)
    union = np.zeros(len(DYNAMIC_CLASSES), dtype=np.int64)
    for j, c in enumerate(DYNAMIC_CLASSES):
        pm = (p == c) & s
        gm = (g == c) & s
        inter[j] = int((pm & gm).sum())
        union[j] = int((pm | gm).sum())
    return inter, union


def _flat_indices(idx: np.ndarray, grid: OccupancyGrid) -> np.ndarray:
    arr = np.asarray(idx, dtype=np.int64)
    if arr.size == 0:
        return np.zeros((0,), dtype=np.int64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("component indices must be [N,3]")
    _, ny, nz = [int(x) for x in grid.shape_hwd]
    return (arr[:, 0] * ny + arr[:, 1]) * nz + arr[:, 2]


def single_source_moving_delta_counts(
    anchor_occ: np.ndarray,
    gt_occ: np.ndarray,
    moving_support: np.ndarray,
    baseline: RasterizedRigidComponent,
    replacement: RasterizedRigidComponent,
    *,
    free_label: int = 17,
    grid: OccupancyGrid = OccupancyGrid(),
) -> tuple[np.ndarray, np.ndarray]:
    """Exact sparse count delta for replacing one KTA source with one V17 source.

    The operation is exactly the one-source A1 CLEAR/WRITE contract:
    1) clear baseline voxels only where the current anchor is dynamic;
    2) write the replacement semantic class;
    3) preserve every other voxel.

    Since only the union of baseline/replacement voxels can change, no dense
    composed occupancy copy is required.
    """
    anchor = np.asarray(anchor_occ)
    gt = np.asarray(gt_occ)
    support = np.asarray(moving_support, dtype=bool)
    if tuple(anchor.shape) != tuple(grid.shape_hwd):
        raise ValueError("anchor shape mismatch")
    if gt.shape != anchor.shape or support.shape != anchor.shape:
        raise ValueError("GT/support shape mismatch")
    if int(baseline.class_id) != int(replacement.class_id):
        raise ValueError("baseline/replacement class mismatch")

    bflat = np.unique(_flat_indices(baseline.voxel_indices, grid))
    rflat = np.unique(_flat_indices(replacement.voxel_indices, grid))
    affected = np.union1d(bflat, rflat)
    if affected.size == 0:
        z = np.zeros(len(DYNAMIC_CLASSES), dtype=np.int64)
        return z.copy(), z.copy()

    af = anchor.reshape(-1)
    gf = gt.reshape(-1)
    sf = support.reshape(-1)

    before = af[affected].astype(np.int64, copy=True)
    after = before.copy()

    if bflat.size:
        pos = np.searchsorted(affected, bflat)
        clearable = np.isin(before[pos], np.asarray(DYNAMIC_CLASSES, dtype=np.int64))
        after[pos[clearable]] = int(free_label)
    if rflat.size:
        pos = np.searchsorted(affected, rflat)
        after[pos] = int(replacement.class_id)

    mask = sf[affected]
    labels = gf[affected]
    d_inter = np.zeros(len(DYNAMIC_CLASSES), dtype=np.int64)
    d_union = np.zeros(len(DYNAMIC_CLASSES), dtype=np.int64)
    for j, c in enumerate(DYNAMIC_CLASSES):
        g = (labels == c) & mask
        pb = (before == c) & mask
        pa = (after == c) & mask
        d_inter[j] = int((pa & g).sum()) - int((pb & g).sum())
        d_union[j] = int((pa | g).sum()) - int((pb | g).sum())
    return d_inter, d_union


def moving_miou_from_counts(inter: np.ndarray, union: np.ndarray) -> float:
    """Frozen horizon-first Moving-mIoU from [H,C] counts."""
    i = np.asarray(inter, dtype=np.float64)
    u = np.asarray(union, dtype=np.float64)
    expected = (len(REPORT_HORIZONS_S), len(DYNAMIC_CLASSES))
    if i.shape != expected:
        raise ValueError(f"count shape must be {expected}")
    if u.shape != i.shape:
        raise ValueError("intersection/union shape mismatch")
    horizon_vals = []
    for h in range(i.shape[0]):
        valid = u[h] > 0
        if not bool(valid.any()):
            return float("nan")
        horizon_vals.append(float(np.mean(i[h, valid] / u[h, valid])))
    return 100.0 * float(np.mean(horizon_vals))


def marginal_utility_pp(
    anchor_inter: np.ndarray,
    anchor_union: np.ndarray,
    delta_inter: np.ndarray,
    delta_union: np.ndarray,
) -> float:
    base = moving_miou_from_counts(anchor_inter, anchor_union)
    new = moving_miou_from_counts(
        np.asarray(anchor_inter, dtype=np.int64) + np.asarray(delta_inter, dtype=np.int64),
        np.asarray(anchor_union, dtype=np.int64) + np.asarray(delta_union, dtype=np.int64),
    )
    return float(new - base)


def marginal_by_horizon_pp(
    anchor_inter: np.ndarray,
    anchor_union: np.ndarray,
    delta_inter: np.ndarray,
    delta_union: np.ndarray,
) -> np.ndarray:
    ai = np.asarray(anchor_inter, dtype=np.float64)
    au = np.asarray(anchor_union, dtype=np.float64)
    di = np.asarray(delta_inter, dtype=np.float64)
    du = np.asarray(delta_union, dtype=np.float64)
    out = np.zeros((len(REPORT_HORIZONS_S),), dtype=np.float64)
    for h in range(len(REPORT_HORIZONS_S)):
        valid = au[h] > 0
        if not bool(valid.any()):
            out[h] = np.nan
            continue
        base = np.mean(ai[h, valid] / au[h, valid])
        new_u = au[h, valid] + du[h, valid]
        if bool((new_u <= 0).any()):
            raise RuntimeError("marginal update produced non-positive union")
        new = np.mean((ai[h, valid] + di[h, valid]) / new_u)
        out[h] = 100.0 * float(new - base)
    return out.astype(np.float32)


class CorrectionSelector(nn.Module):
    """Tiny source-level ranker; inputs are causal 46-D motion features only."""

    def __init__(self, feature_dim: int = FEATURE_DIM, hidden_dim: int = 64):
        super().__init__()
        mid = max(16, int(hidden_dim) // 2)
        self.net = nn.Sequential(
            nn.Linear(int(feature_dim), int(hidden_dim)),
            nn.GELU(),
            nn.LayerNorm(int(hidden_dim)),
            nn.Linear(int(hidden_dim), mid),
            nn.GELU(),
            nn.Linear(mid, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
            raise ValueError(f"selector features must be [B,{FEATURE_DIM}]")
        return self.net(features).squeeze(-1)


@dataclass(frozen=True)
class SelectorNormalization:
    feature_mean: Sequence[float]
    feature_std: Sequence[float]
    target_mean: float
    target_std: float

    def normalize_features(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.feature_mean, dtype=x.dtype, device=x.device)
        std = torch.as_tensor(self.feature_std, dtype=x.dtype, device=x.device)
        return (x - mean) / std.clamp_min(1e-6)

    def denormalize_target(self, y: torch.Tensor) -> torch.Tensor:
        return y * float(self.target_std) + float(self.target_mean)


def deterministic_scene_split(
    scene_names: Sequence[str],
    *,
    val_fraction: float = 0.10,
    seed: int = 20260913,
) -> tuple[set[str], set[str]]:
    scenes = sorted(set(str(x) for x in scene_names))
    if len(scenes) < 2:
        raise ValueError("need at least two scenes for selector split")
    rng = random.Random(int(seed))
    rng.shuffle(scenes)
    n_val = max(1, min(len(scenes) - 1, int(round(len(scenes) * float(val_fraction)))))
    val = set(scenes[:n_val])
    train = set(scenes[n_val:])
    return train, val


def rankdata_simple(x: np.ndarray) -> np.ndarray:
    """Deterministic average ranks, implemented without scipy."""
    arr = np.asarray(x, dtype=np.float64)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(len(arr), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        v = arr[order[start]]
        while end < len(order) and arr[order[end]] == v:
            end += 1
        rank = 0.5 * (start + end - 1)
        ranks[order[start:end]] = rank
        start = end
    return ranks


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    a = rankdata_simple(np.asarray(x, dtype=np.float64))
    b = rankdata_simple(np.asarray(y, dtype=np.float64))
    if len(a) < 2 or float(a.std()) <= 0 or float(b.std()) <= 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def selected_count(n_sources: int, budget_percent: float) -> int:
    n = int(n_sources)
    q = float(budget_percent)
    if n <= 0 or q <= 0:
        return 0
    if q >= 100:
        return n
    return max(0, min(n, int(round(n * q / 100.0))))


def top_budget_mask(scores: Sequence[float], budget_percent: float) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float64)
    k = selected_count(len(s), budget_percent)
    out = np.zeros(len(s), dtype=bool)
    if k <= 0:
        return out
    if k >= len(s):
        out[:] = True
        return out
    order = np.argsort(-s, kind="mergesort")
    out[order[:k]] = True
    return out


def stable_random_scores(sample_id: str, n_sources: int, seed: int) -> np.ndarray:
    key = f"{int(seed)}::{sample_id}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    local_seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
    rng = np.random.default_rng(local_seed)
    return rng.random(int(n_sources), dtype=np.float64)
