"""Context-window utilities for sparse future prediction.

P0-F4 predicts a 20x20 future latent window while conditioning on a larger
40x40 crop of the full historical latent.  The context crop is shifted (never
padded) at scene boundaries so the prediction window is always contained in the
context window.
"""
from __future__ import annotations

import torch

from .windows import WindowPlan, crop_windows


def context_plan_from_prediction(
    plan: WindowPlan,
    *,
    context_hw: tuple[int, int] = (40, 40),
) -> WindowPlan:
    ph, pw = map(int, plan.window_hw)
    ch, cw = map(int, context_hw)
    fh, fw = map(int, plan.full_hw)
    if ch < ph or cw < pw:
        raise ValueError("context window must contain the prediction window")
    if ch > fh or cw > fw:
        raise ValueError("context window cannot exceed full latent grid")

    origins = plan.origins.long()
    center_y = origins[..., 0] + ph // 2
    center_x = origins[..., 1] + pw // 2
    cy = center_y - ch // 2
    cx = center_x - cw // 2
    cy = cy.clamp(min=0, max=fh - ch)
    cx = cx.clamp(min=0, max=fw - cw)
    ctx_origins = torch.stack([cy, cx], dim=-1)
    return WindowPlan(ctx_origins, plan.valid, (ch, cw), plan.full_hw)


def crop_prediction_and_context(
    full_history: torch.Tensor,
    plan: WindowPlan,
    *,
    context_hw: tuple[int, int] = (40, 40),
) -> tuple[torch.Tensor, torch.Tensor, WindowPlan]:
    """Return local prediction history plus its larger surrounding context."""
    local = crop_windows(full_history, plan)
    context_plan = context_plan_from_prediction(plan, context_hw=context_hw)
    context = crop_windows(full_history, context_plan)
    return local, context, context_plan
