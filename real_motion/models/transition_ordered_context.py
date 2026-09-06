"""Ordered-history residual context for the P0-F9 native transition.

The existing P0-F9 context path temporally averages the 6 history frames before
one stride-2 projection.  This module keeps that mean path unchanged and adds a
small residual path that concatenates H0..H5 along channels before a zero-init
3x3 stride-2 projection.  With zero initialization the ordered path is an exact
no-op, so an M-400 checkpoint can be lifted into this architecture without
changing its function at phase step 0.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .transition_native_physics import MotionWindowNativePhysicsTransition


ORDERED_CONTEXT_PROTOCOL = "concat_6_history_frames_96to128_stride2_3x3_zero_init_residual_v1"


class MotionWindowNativePhysicsOrderedContextTransition(MotionWindowNativePhysicsTransition):
    """P0-F9 transition with optional ordered-history residual context.

    ``ordered_context_enabled=False`` keeps the extra branch dormant.  The
    module and state_dict keys still exist so M-control and M+T checkpoints have
    the same architecture and differ only by whether this residual is allowed to
    participate in training/inference.
    """

    def __init__(
        self,
        *args,
        context_channels: int = 16,
        ordered_history_frames: int = 6,
        ordered_context_enabled: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, context_channels=context_channels, **kwargs)
        self.ordered_history_frames = int(ordered_history_frames)
        if self.ordered_history_frames <= 0:
            raise ValueError("ordered_history_frames must be positive")
        self.ordered_context_enabled = bool(ordered_context_enabled)
        self.ordered_context_proj = nn.Conv2d(
            int(context_channels) * self.ordered_history_frames,
            self.model_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=True,
        )
        nn.init.zeros_(self.ordered_context_proj.weight)
        nn.init.zeros_(self.ordered_context_proj.bias)

    @property
    def ordered_context_trainable_parameters(self) -> int:
        return sum(int(p.numel()) for p in self.ordered_context_proj.parameters() if p.requires_grad)

    def set_ordered_context_enabled(self, enabled: bool) -> None:
        self.ordered_context_enabled = bool(enabled)

    def _context_base(self, history_context, batch_size, device, dtype):
        base = super()._context_base(history_context, batch_size, device, dtype)
        if history_context is None or not self.ordered_context_enabled:
            return base
        if history_context.ndim != 5 or history_context.shape[0] != batch_size:
            raise ValueError("history_context must be [B,T,C,H,W]")
        if int(history_context.shape[1]) != self.ordered_history_frames:
            raise ValueError(
                f"ordered context expects {self.ordered_history_frames} history frames, "
                f"got {int(history_context.shape[1])}"
            )
        if int(history_context.shape[2]) * self.ordered_history_frames != int(
            self.ordered_context_proj.in_channels
        ):
            raise ValueError("ordered history/context channel contract mismatch")
        ctx = history_context.to(device=device, dtype=dtype).contiguous()
        # Concatenating T then C preserves H0,H1,... order explicitly.  Reversing
        # the temporal order leaves the mean path unchanged but changes this input.
        ordered = ctx.reshape(
            int(batch_size),
            int(ctx.shape[1]) * int(ctx.shape[2]),
            int(ctx.shape[3]),
            int(ctx.shape[4]),
        )
        residual = self.ordered_context_proj(ordered)
        return residual if base is None else base + residual


def ordered_context_is_exact_zero(transition) -> bool:
    """Return whether the ordered residual is exactly zero-impact by parameters."""
    proj = transition.ordered_context_proj
    return bool((proj.weight.detach() == 0).all() and (proj.bias.detach() == 0).all())
