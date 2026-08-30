"""Anchor-centered local flow matching for MSP-selected World-Model windows.

The expensive future transition is executed only on selected 20x20 windows.
Each future window starts from a causal semantic anchor latent and flows toward
a target latent. P0-F4 optionally conditions the transition on a larger crop of
full historical occupancy latent and can restrict the single FM objective to a
causal sparse write mask. P0-F5 changes the endpoint to an encoded occupancy
repair target. P0-F6 can additionally request the differentiable endpoint
estimate from the same FM forward pass for decoder-aware semantic supervision.
Legacy callers keep the previous behavior by leaving ``return_endpoint=False``.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class AnchorWindowCFM(nn.Module):
    """Conditional flow from an anchor latent to a future target latent."""

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

    @staticmethod
    def _normalize_loss_mask(loss_mask, target):
        if loss_mask is None:
            return None
        m = loss_mask.to(device=target.device, dtype=target.dtype)
        if m.ndim == 4:
            m = m.unsqueeze(2)
        if m.ndim != 5:
            raise ValueError("loss_mask must be [B,T,H,W] or [B,T,1,H,W]")
        if m.shape[0] != target.shape[0] or m.shape[1] != target.shape[1] or m.shape[-2:] != target.shape[-2:]:
            raise ValueError("loss_mask must align with target B,T,H,W")
        if m.shape[2] not in (1, target.shape[2]):
            raise ValueError("loss_mask channel dimension must be 1 or match target")
        if m.shape[2] == 1:
            m = m.expand(-1, -1, target.shape[2], -1, -1)
        return m

    @staticmethod
    def _force_output_grad_dtype(x: torch.Tensor) -> torch.Tensor:
        """Keep custom AMP attention backward gradients in the forward dtype.

        The pinned OccFM FlashAttention autograd Function assumes ``grad_output``
        has the same dtype as its saved q/k/v tensors. Decoder-aware P0-F6 can
        feed an FP32 semantic gradient into a BF16 transition output, which the
        custom backward does not cast and then fails inside ``einsum``. Register
        the cast exactly at the transition-output boundary so the combined FM +
        semantic gradient enters the pinned transition in its native AMP dtype.
        This changes neither the forward value nor the loss definition.
        """
        if x.requires_grad and x.is_floating_point():
            dtype = x.dtype
            x.register_hook(lambda grad: grad.to(dtype=dtype))
        return x

    def flow_loss(
        self,
        history: torch.Tensor,
        target_future: torch.Tensor,
        anchor_future: torch.Tensor,
        *,
        history_context: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        trajectory: torch.Tensor | None = None,
        window_origins: torch.Tensor | None = None,
        t_override: float | torch.Tensor | None = None,
        source_noise: torch.Tensor | None = None,
        return_endpoint: bool = False,
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
        mask = self._normalize_loss_mask(loss_mask, target)
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
        pred = self._force_output_grad_dtype(pred)
        velocity_target = target - source
        sq = (pred - velocity_target).square()
        if mask is None:
            loss = sq.mean()
            metric_mask = None
        else:
            denom = mask.sum()
            if float(denom.detach().cpu()) <= 0:
                raise ValueError("loss_mask selects no latent elements")
            loss = (sq * mask).sum() / denom
            metric_mask = mask

        with torch.no_grad():
            pf = pred.float()
            yf = velocity_target.float()
            if metric_mask is not None:
                mf = metric_mask.float()
                pf = pf * mf
                yf = yf * mf
                target_rms = (velocity_target.float().square() * mf).sum() / mf.sum().clamp_min(1.0)
                pred_rms = (pred.float().square() * mf).sum() / mf.sum().clamp_min(1.0)
                target_rms = target_rms.sqrt()
                pred_rms = pred_rms.sqrt()
                active_fraction = float(metric_mask[:, :, :1].float().mean().detach().cpu())
            else:
                target_rms = velocity_target.float().square().mean().sqrt()
                pred_rms = pred.float().square().mean().sqrt()
                active_fraction = 1.0
            p = pf.reshape(pred.shape[0], -1)
            y = yf.reshape(velocity_target.shape[0], -1)
            dot = (p * y).sum(dim=1)
            denom = p.norm(dim=1) * y.norm(dim=1)
            cosine = torch.where(denom > 0, dot / denom.clamp_min(1e-12), torch.zeros_like(dot))
        info = {
            "loss": float(loss.detach().cpu()),
            "target_rms": float(target_rms.detach().cpu()),
            "pred_rms": float(pred_rms.detach().cpu()),
            "cosine": float(cosine.mean().detach().cpu()),
            "loss_active_fraction": active_fraction,
        }
        if return_endpoint:
            # For the linear conditional path x_t=(1-t)x_0+t*x_1 and velocity
            # v=x_1-x_0, the endpoint is x_1=x_t+(1-t)v.  Keep this tensor in
            # the graph so a frozen decoder can supervise final semantics.
            info["predicted_endpoint"] = (
                xt + (1.0 - t) * pred
            ) / self.rescale_factor
            info["sampled_t"] = t
        return loss, info

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
