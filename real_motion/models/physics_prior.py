"""Lightweight physics-prior fusion for P0-F9 native sparse forecasting.

The module borrows the *controlled condition injection* idea from strong
forecasting/world-model systems: keep the pretrained transition exactly intact at
initialization, then let training learn how much of the external physics prior to
use. The Strong-W2Det/KTA future latent is used only as conditioning evidence;
it is never the flow source or the prediction target.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class GatedPhysicsCrossAttention(nn.Module):
    """Per-frame cross-attention with an exact zero-impact initialization.

    Query tokens come from the native WM bottleneck. Key/value tokens come from
    the Strong-W2Det future prior. ``gate`` starts at exactly zero, so adding this
    module cannot change the pretrained OccFM forward before optimization.

    ``prior_proj`` deliberately has no bias. The aligned prior contains zero
    history slots; keeping zero input exactly zero prevents those history slots
    from turning into a learned constant pseudo-physics condition once the gate
    becomes nonzero.
    """

    def __init__(
        self,
        *,
        prior_channels: int = 16,
        hidden_size: int = 256,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.prior_channels = int(prior_channels)
        self.hidden_size = int(hidden_size)
        self.query_norm = nn.LayerNorm(self.hidden_size)
        self.prior_norm = nn.LayerNorm(self.hidden_size)
        self.prior_proj = nn.Linear(self.prior_channels, self.hidden_size, bias=False)
        self.attn = nn.MultiheadAttention(
            self.hidden_size,
            int(num_heads),
            dropout=0.0,
            batch_first=True,
        )
        # tanh keeps the learned authority bounded while preserving an exact
        # pretrained no-op at initialization.
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor, prior_future: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError("physics fusion expects WM state [B,C,F,H,W]")
        if prior_future is None:
            return x
        if prior_future.ndim != 5:
            raise ValueError("physics prior must be [B,F,C,H,W]")
        b, c, f, h, w = x.shape
        if c != self.hidden_size:
            raise ValueError(
                f"WM bottleneck channels {c} != configured physics hidden size {self.hidden_size}"
            )
        if prior_future.shape[0] != b or prior_future.shape[1] != f:
            raise ValueError("physics prior B/F must match WM bottleneck")
        if prior_future.shape[2] != self.prior_channels:
            raise ValueError("physics prior channel mismatch")

        prior = prior_future.to(device=x.device, dtype=x.dtype)
        prior = F.interpolate(
            prior.reshape(b * f, self.prior_channels, *prior.shape[-2:]),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )
        prior = rearrange(prior, "(b f) c h w -> (b f) (h w) c", b=b, f=f)
        prior = self.prior_norm(self.prior_proj(prior))

        query = rearrange(x, "b c f h w -> (b f) (h w) c")
        attn_out, _ = self.attn(
            self.query_norm(query),
            prior,
            prior,
            need_weights=False,
        )
        fused = query + torch.tanh(self.gate).to(dtype=query.dtype) * attn_out
        return rearrange(fused, "(b f) (h w) c -> b c f h w", b=b, f=f, h=h, w=w)

    @property
    def authority(self) -> torch.Tensor:
        """Current bounded physics-injection strength for logging."""
        return torch.tanh(self.gate.detach())
