"""Anchor-centered local flow matching for MSP-selected World-Model windows.

The expensive future transition is executed only on selected 20x20 windows.
Each future window starts from a causal semantic anchor latent and flows toward
full future GT. P0-F4 optionally conditions the transition on a larger crop of
full historical occupancy latent without changing the local future state size.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class AnchorWindowCFM(nn.Module):
    """Conditional flow from an anchor latent to the future GT latent."""

    def __init__(
        self,
        transition: nn.Module,
        *,
        rescale_factor: float = 10.0,
        sample_steps: int = 10,
        alpha_shift: float = 3.0,
        source_noise_std: float = 0.0,
    ):
        super().__init__()
        self.transition = transition
        self.rescale_factor = float(rescale_factor)
        self.sample_steps = int(sample_steps)
        self.alpha_shift = float(alpha_shift)
        self.source_noise_std = float(source_noise_std)
        self.time_scalar = 1000.0
        if self.sample_steps <= 0:
            raise ValueError("sample_steps must be positive")
        if self.source_noise_std < 0:
            raise ValueError("source_noise_std must be non-negative")

    @staticmethod
    def _sample_t(bs: int, device, dtype):
        return torch.sigmoid(torch.randn(bs, 1, 1, 1, 1, device=device, dtype=dtype))

    @staticmethod
    def _align_prior(anchor: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        if anchor.ndim != 5 or history.ndim != 5:
            raise ValueError("anchor/history must be [B,T,C,H,W]")
        zeros = torch.zeros(
            anchor.shape[0],
            history.shape[1],
            *anchor.shape[2:],
            device=anchor.device,
            dtype=anchor.dtype,
        )
        return torch.cat([zeros, anchor], dim=1)

    @staticmethod
    def _validate_history_context(history_context, history):
        if history_context is None:
            return
        if history_context.ndim != 5:
            raise ValueError("history_context must be [B,T,C,H,W]")
        if history_context.shape[:3] != history.shape[:3]:
            raise ValueError("history_context B,T,C must match local history")
        if history_context.shape[-2] < history.shape[-2] or history_context.shape[-1] < history.shape[-1]:
            raise ValueError("history_context must not be spatially smaller than local history")

    def flow_loss(
        self,
        history: torch.Tensor,
        target_future: torch.Tensor,
        anchor_future: torch.Tensor,
        *,
        history_context: torch.Tensor | None = None,
        trajectory: torch.Tensor | None = None,
        window_origins: torch.Tensor | None = None,
        t_override: float | torch.Tensor | None = None,
        source_noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        if target_future.shape != anchor_future.shape:
            raise ValueError("target_future and anchor_future must match")
        if history.ndim != 5 or target_future.ndim != 5:
            raise ValueError("history/target must be [B,T,C,H,W]")
        self._validate_history_context(history_context, history)

        hist = history * self.rescale_factor
        target = target_future * self.rescale_factor
        anchor = anchor_future * self.rescale_factor
        ctx = None if history_context is None else history_context * self.rescale_factor
        if source_noise is None:
            eps = torch.zeros_like(anchor) if self.source_noise_std == 0 else torch.randn_like(anchor)
        else:
            if source_noise.shape != anchor.shape:
                raise ValueError("source_noise shape mismatch")
            eps = source_noise.to(device=anchor.device, dtype=anchor.dtype)
        source = anchor + self.source_noise_std * eps

        bs = target.shape[0]
        if t_override is None:
            t = self._sample_t(bs, target.device, target.dtype)
        elif torch.is_tensor(t_override):
            t = t_override.to(device=target.device, dtype=target.dtype)
            if t.numel() == 1:
                t = t.reshape(1, 1, 1, 1, 1).expand(bs, 1, 1, 1, 1)
            elif tuple(t.shape) != (bs, 1, 1, 1, 1):
                raise ValueError("t_override tensor must be scalar or [B,1,1,1,1]")
        else:
            tv = float(t_override)
            if not 0.0 <= tv <= 1.0:
                raise ValueError("t_override must be in [0,1]")
            t = torch.full((bs, 1, 1, 1, 1), tv, device=target.device, dtype=target.dtype)

        xt = (1.0 - t) * source + t * target
        seq = torch.cat([hist, xt], dim=1)
        prior = self._align_prior(anchor_future, history) * self.rescale_factor
        batch = {
            "noised_sequence": seq,
            "timesteps": t[:, 0, 0, 0, 0] * self.time_scalar,
            "trajectory": trajectory,
            "prior_condition": prior,
            "history_context": ctx,
            "window_origins": window_origins,
        }
        pred = self.transition(batch)["predicted_latent"][:, history.shape[1]:]
        velocity_target = target - source
        loss = (pred - velocity_target).square().mean()

        with torch.no_grad():
            p = pred.float().reshape(pred.shape[0], -1)
            y = velocity_target.float().reshape(velocity_target.shape[0], -1)
            dot = (p * y).sum(dim=1)
            denom = p.norm(dim=1) * y.norm(dim=1)
            cosine = torch.where(denom > 0, dot / denom.clamp_min(1e-12), torch.zeros_like(dot))
        return loss, {
            "loss": float(loss.detach().cpu()),
            "target_rms": float(velocity_target.float().square().mean().sqrt().detach().cpu()),
            "pred_rms": float(pred.float().square().mean().sqrt().detach().cpu()),
            "cosine": float(cosine.mean().detach().cpu()),
        }

    @torch.no_grad()
    def sample(
        self,
        history: torch.Tensor,
        anchor_future: torch.Tensor,
        *,
        history_context: torch.Tensor | None = None,
        trajectory: torch.Tensor | None = None,
        window_origins: torch.Tensor | None = None,
        initial_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if history.ndim != 5 or anchor_future.ndim != 5:
            raise ValueError("history/anchor must be [B,T,C,H,W]")
        self._validate_history_context(history_context, history)
        hist = history * self.rescale_factor
        anchor = anchor_future * self.rescale_factor
        ctx = None if history_context is None else history_context * self.rescale_factor
        if initial_noise is None:
            eps = torch.zeros_like(anchor) if self.source_noise_std == 0 else torch.randn_like(anchor)
        else:
            if initial_noise.shape != anchor.shape:
                raise ValueError("initial_noise shape mismatch")
            eps = initial_noise.to(device=anchor.device, dtype=anchor.dtype)
        future = anchor + self.source_noise_std * eps
        prior = self._align_prior(anchor_future, history) * self.rescale_factor

        timesteps = torch.linspace(0, 1, self.sample_steps + 1, device=hist.device, dtype=hist.dtype)
        shifted = 1 - (self.alpha_shift * timesteps) / (1 + (self.alpha_shift - 1) * timesteps)
        shifted = shifted.flip(0)
        for tc, tp in zip(shifted[:-1], shifted[1:]):
            seq = torch.cat([hist, future], dim=1)
            batch = {
                "noised_sequence": seq,
                "timesteps": torch.full(
                    (hist.shape[0],),
                    float(tc * self.time_scalar),
                    device=hist.device,
                    dtype=hist.dtype,
                ),
                "trajectory": trajectory,
                "prior_condition": prior,
                "history_context": ctx,
                "window_origins": window_origins,
            }
            velocity = self.transition(batch)["predicted_latent"][:, history.shape[1]:]
            future = future + (tp - tc) * velocity
        return future / self.rescale_factor
