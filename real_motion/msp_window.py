"""Deployment-aligned window selection for MSP score maps.

The tiny MSP predicts dense ranking scores on the latent map, while the current
sparse World Model executes fixed spatial crops. This module bridges the two
without retraining MSP: aggregate future-horizon scores, then greedily select a
fixed number of spatial windows by marginal captured score.

One set of spatial windows is shared across all future horizons, matching the
local-WM execution unit used by the current SWFM pipeline.
"""
from __future__ import annotations

import torch

from .windows import WindowPlan


def _window_sums_float(x: torch.Tensor, wh: int, ww: int) -> torch.Tensor:
    """Exact sliding-window sums for one floating score map on CPU."""
    if x.ndim != 2:
        raise ValueError("score map must be [H,W]")
    H, W = map(int, x.shape)
    if wh <= 0 or ww <= 0 or wh > H or ww > W:
        raise ValueError("invalid window size")
    y = x.to(dtype=torch.float64, device="cpu")
    integral = torch.zeros((H + 1, W + 1), dtype=torch.float64)
    integral[1:, 1:] = y.cumsum(0).cumsum(1)
    return (
        integral[wh:, ww:]
        - integral[:-wh, ww:]
        - integral[wh:, :-ww]
        + integral[:-wh, :-ww]
    )


def plan_topk_score_windows(
    score_maps: torch.Tensor,
    *,
    window_hw: tuple[int, int] = (20, 20),
    max_windows: int = 1,
) -> WindowPlan:
    """Greedily select fixed windows by marginal MSP score capture.

    ``score_maps`` may be ``[B,T,H,W]`` or pre-aggregated ``[B,H,W]``. Future
    scores are summed before selection so every horizon shares one spatial plan.
    After choosing a window, its score is removed before choosing the next one;
    later windows therefore maximize marginal captured score rather than raw
    overlapping score.
    """
    if score_maps.ndim == 4:
        agg = score_maps.sum(dim=1)
    elif score_maps.ndim == 3:
        agg = score_maps
    else:
        raise ValueError("score_maps must be [B,T,H,W] or [B,H,W]")
    if max_windows <= 0:
        raise ValueError("max_windows must be positive")
    if not bool(torch.isfinite(agg).all()):
        raise ValueError("non-finite MSP score")
    if bool((agg < 0).any()):
        raise ValueError("MSP scores must be non-negative")

    out_device = agg.device
    scores = agg.detach().cpu().to(torch.float64)
    B, H, W = map(int, scores.shape)
    wh, ww = map(int, window_hw)
    if wh <= 0 or ww <= 0 or wh > H or ww > W:
        raise ValueError("window cannot be larger than score map")

    origins = torch.full((B, int(max_windows), 2), -1, dtype=torch.long)
    valid = torch.zeros((B, int(max_windows)), dtype=torch.bool)

    for b in range(B):
        remaining = scores[b].clone()
        for k in range(int(max_windows)):
            window_score = _window_sums_float(remaining, wh, ww)
            flat = int(window_score.reshape(-1).argmax())
            best = float(window_score.reshape(-1)[flat])
            if best <= 0.0:
                break
            out_w = int(window_score.shape[1])
            y0, x0 = divmod(flat, out_w)
            origins[b, k] = torch.tensor([y0, x0])
            valid[b, k] = True
            remaining[y0:y0 + wh, x0:x0 + ww] = 0.0

    return WindowPlan(
        origins=origins.to(out_device),
        valid=valid.to(out_device),
        window_hw=(wh, ww),
        full_hw=(H, W),
    )


def window_plan_support(plan: WindowPlan, *, device=None) -> torch.Tensor:
    """Return the union of selected windows as ``[B,H,W]`` boolean support."""
    device = device or plan.origins.device
    origins = plan.origins.to(device=device, dtype=torch.long)
    valid = plan.valid.to(device=device)
    B, K = map(int, valid.shape)
    H, W = map(int, plan.full_hw)
    wh, ww = map(int, plan.window_hw)
    support = torch.zeros((B, H, W), dtype=torch.bool, device=device)
    for b in range(B):
        for k in range(K):
            if not bool(valid[b, k]):
                continue
            y0 = int(origins[b, k, 0].item())
            x0 = int(origins[b, k, 1].item())
            support[b, y0:y0 + wh, x0:x0 + ww] = True
    return support


def score_capture_ratio(score_maps: torch.Tensor, plan: WindowPlan) -> torch.Tensor:
    """Fraction of total aggregated MSP score covered by selected windows."""
    if score_maps.ndim == 4:
        agg = score_maps.sum(dim=1)
    elif score_maps.ndim == 3:
        agg = score_maps
    else:
        raise ValueError("score_maps must be [B,T,H,W] or [B,H,W]")
    if tuple(agg.shape[-2:]) != tuple(plan.full_hw):
        raise ValueError("score map and plan shape mismatch")
    if int(agg.shape[0]) != int(plan.valid.shape[0]):
        raise ValueError("score map and plan batch mismatch")
    support = window_plan_support(plan, device=agg.device)
    numer = (agg * support.to(agg.dtype)).flatten(1).sum(1)
    denom = agg.flatten(1).sum(1)
    return torch.where(denom > 0, numer / denom, torch.ones_like(denom))
