"""Utilities for the KTA/V17 selective-forecasting feasibility study.

The experiment deliberately separates three concepts:
  1. semantic dynamic class;
  2. actual physical motion;
  3. whether the learned V17 expert is worth invoking instead of KTA.

Selector inputs are causal only.  Future GT is used exclusively to build an
offline utility target and oracle ranking for feasibility analysis.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from .motion_transport import FEATURE_DIM, FEATURE_NAMES, FUTURE_FRAMES, HISTORY_FRAMES

SELECTOR_CACHE_VERSION = "p0_f9_kta_v17_selector_utility_v1"
SELECTOR_PROTOCOL = "p0_f9_kta_v17_causal_mlp_selector_v1"
UTILITY_CONTRACT = "single_source_counterfactual_delta_correct_semantic_voxels_on_gt_moving_support_v1"
FEATURE_CONTRACT = (
    "v17_causal_motion_features_plus_frame_motion_plus_kta_displacements_no_v17_output_no_future_gt_v1"
)

SELECTOR_FEATURE_NAMES = (
    *tuple(f"v17_{x}" for x in FEATURE_NAMES),
    *tuple(
        f"frame_motion_t{t}_{name}"
        for t in range(HISTORY_FRAMES)
        for name in ("offset_x", "offset_y", "velocity_x", "velocity_y", "valid")
    ),
    *tuple(f"kta_future_t{t}_{axis}" for t in range(FUTURE_FRAMES) for axis in ("x", "y")),
)
SELECTOR_FEATURE_DIM = len(SELECTOR_FEATURE_NAMES)

_SPEED_X = FEATURE_NAMES.index("current_vx_norm")
_SPEED_Y = FEATURE_NAMES.index("current_vy_norm")


def selector_features(record: Mapping) -> torch.Tensor:
    """Build causal scalar selector features [N,D] from a V17 cache record."""
    base = torch.as_tensor(record["features"], dtype=torch.float32)
    frame = torch.as_tensor(record["frame_motion_features"], dtype=torch.float32)
    kta = torch.as_tensor(record["kta_displacement_xy_m"], dtype=torch.float32)
    if base.ndim != 2 or base.shape[1] != FEATURE_DIM:
        raise ValueError(f"features must be [N,{FEATURE_DIM}], got {tuple(base.shape)}")
    if frame.shape != (base.shape[0], HISTORY_FRAMES, 5):
        raise ValueError(f"frame_motion_features shape mismatch: {tuple(frame.shape)}")
    if kta.shape != (base.shape[0], FUTURE_FRAMES, 2):
        raise ValueError(f"kta_displacement shape mismatch: {tuple(kta.shape)}")
    # Keep every channel on roughly O(1) scale.  V17 base features are already
    # normalized.  Frame-motion offsets/velocities and KTA displacements are in
    # metres, so use the same conservative 20 m scale as the V17 feature code.
    # Use explicit flattened dimensions rather than -1.  Windows with
    # zero Strong sources are valid evaluation cases; torch.reshape(0, -1) is
    # ambiguous, while [0, fixed_dim] is well-defined and should propagate
    # through selector/ranking code as an empty source set.
    n = int(base.shape[0])
    out = torch.cat(
        [
            base,
            frame.reshape(n, HISTORY_FRAMES * 5) / 20.0,
            kta.reshape(n, FUTURE_FRAMES * 2) / 20.0,
        ],
        dim=1,
    )
    if out.shape[1] != SELECTOR_FEATURE_DIM:
        raise AssertionError((out.shape, SELECTOR_FEATURE_DIM))
    return out


def speed_score(record: Mapping) -> torch.Tensor:
    """Cheap causal rule baseline: current planar speed, normalized feature units."""
    x = torch.as_tensor(record["features"], dtype=torch.float32)
    return torch.sqrt(x[:, _SPEED_X] ** 2 + x[:, _SPEED_Y] ** 2)


def top_fraction_mask(scores: np.ndarray | torch.Tensor, fraction: float) -> np.ndarray:
    """Per-window deterministic top-Q mask. Q=0 selects none; Q=1 selects all."""
    s = scores.detach().cpu().numpy() if torch.is_tensor(scores) else np.asarray(scores)
    s = np.asarray(s, dtype=np.float64).reshape(-1)
    q = float(fraction)
    if not 0.0 <= q <= 1.0:
        raise ValueError("fraction must be in [0,1]")
    n = len(s)
    out = np.zeros(n, dtype=bool)
    if n == 0 or q <= 0.0:
        return out
    if q >= 1.0:
        out[:] = True
        return out
    k = max(1, int(np.ceil(q * n)))
    # Stable tie-break by original Strong source order.
    order = np.lexsort((np.arange(n, dtype=np.int64), -s))
    out[order[:k]] = True
    return out


def random_fraction_mask(n: int, fraction: float, seed: int, *, sample_id: str = "") -> np.ndarray:
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return np.zeros(0, dtype=bool)
    # Deterministic per sample and repeat; do not depend on Python hash randomization.
    h = 2166136261
    for b in sample_id.encode("utf-8"):
        h = ((h ^ int(b)) * 16777619) & 0xFFFFFFFF
    rng = np.random.default_rng((int(seed) + h) & 0xFFFFFFFF)
    return top_fraction_mask(rng.random(n), fraction)


class KtaV17Selector(nn.Module):
    """Tiny source-level utility regressor used only after oracle feasibility passes."""

    def __init__(self, input_dim: int = SELECTOR_FEATURE_DIM, hidden_dim: int = 96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.GELU(),
            nn.LayerNorm(int(hidden_dim)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.net[0].in_features:
            raise ValueError(f"selector input shape mismatch: {tuple(x.shape)}")
        return self.net(x).squeeze(-1)


@dataclass(frozen=True)
class UtilitySummary:
    num_sources: int
    positive_fraction: float
    zero_fraction: float
    mean: float
    std: float
    p90: float
    p99: float


def summarize_utility(values: Sequence[float]) -> UtilitySummary:
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return UtilitySummary(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return UtilitySummary(
        num_sources=int(x.size),
        positive_fraction=float((x > 0).mean()),
        zero_fraction=float((x == 0).mean()),
        mean=float(x.mean()),
        std=float(x.std()),
        p90=float(np.quantile(x, 0.90)),
        p99=float(np.quantile(x, 0.99)),
    )
