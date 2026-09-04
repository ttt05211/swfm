"""Native noise-to-future conditional flow matching for P0-F9.

Unlike P0-F4..F8, the Strong-W2Det/KTA future is *not* the flow source. P0-F9
restores the official OccFM task: Gaussian source -> absolute future latent,
conditioned on history/trajectory. The physics future is supplied only as an
additional condition and remains the exact deployment fallback outside MSP
write support.

The pretrained OccFM-Fut checkpoint was trained with six history slots but only
the last four populated (HIST_LAST=4). P0-F9 preserves that native backbone
contract on the inherited path while the separate full-history context branch is
still allowed to see all six history frames.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class NativeFutureWindowCFM(nn.Module):
    def __init__(
        self,
        transition: nn.Module,
        *,
        rescale_factor: float = 10.0,
        sample_steps: int = 10,
        alpha_shift: float = 3.0,
        unconditional_probability: float = 0.2,
        guidance_scale: float = 2.0,
        hist_last: int = 4,
    ) -> None:
        super().__init__()
        self.transition = transition
        self.rescale_factor = float(rescale_factor)
        self.sample_steps = int(sample_steps)
        self.alpha_shift = float(alpha_shift)
        self.unconditional_probability = float(unconditional_probability)
        self.guidance_scale = float(guidance_scale)
        self.hist_last = int(hist_last)
        self.time_scalar = 1000.0
        if self.sample_steps <= 0:
            raise ValueError("sample_steps must be positive")
        if not 0.0 <= self.unconditional_probability < 1.0:
            raise ValueError("unconditional_probability must be in [0,1)")
        if self.guidance_scale < 0:
            raise ValueError("guidance_scale must be non-negative")
        if self.hist_last <= 0:
            raise ValueError("hist_last must be positive")

    @staticmethod
    def _sample_t(bs: int, device, dtype):
        return torch.sigmoid(torch.randn(bs, 1, 1, 1, 1, device=device, dtype=dtype))

    @staticmethod
    def _validate_context(history_context, history):
        if history_context is None:
            return
        if history_context.ndim != 5:
            raise ValueError("history_context must be [B,T,C,H,W]")
        if history_context.shape[:3] != history.shape[:3]:
            raise ValueError("history_context B/T/C must match history")

    def _native_backbone_condition(
        self,
        history: torch.Tensor,
        trajectory: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Match the released OccFM-Fut HIST_LAST contract on loaded weights.

        The official dataset keeps six condition slots but zeros the first
        ``6-HIST_LAST`` slots and the matching trajectory rows. The new 40x40
        context branch receives the untouched six-frame history separately.
        """
        if history.ndim != 5:
            raise ValueError("history must be [B,T,C,H,W]")
        hist_frames = int(history.shape[1])
        if self.hist_last > hist_frames:
            raise ValueError(
                f"hist_last={self.hist_last} exceeds history frames={hist_frames}"
            )
        repeat = hist_frames - self.hist_last
        if repeat <= 0:
            native_history = history
        else:
            native_history = history.clone()
            native_history[:, :repeat] = 0

        native_trajectory = trajectory
        if trajectory is not None:
            if trajectory.ndim != 3 or trajectory.shape[0] != history.shape[0]:
                raise ValueError("trajectory must be [B,L,2]")
            if trajectory.shape[-1] != 2 or trajectory.shape[1] < repeat:
                raise ValueError("trajectory shape is incompatible with HIST_LAST masking")
            native_trajectory = trajectory.clone()
            if repeat > 0:
                native_trajectory[:, :repeat] = 0
        return native_history, native_trajectory

    @staticmethod
    def _align_prior(physics_future: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        zeros = torch.zeros(
            physics_future.shape[0],
            history.shape[1],
            *physics_future.shape[2:],
            device=physics_future.device,
            dtype=physics_future.dtype,
        )
        return torch.cat([zeros, physics_future], dim=1)

    @staticmethod
    def _force_output_grad_dtype(x: torch.Tensor) -> torch.Tensor:
        if x.requires_grad and x.is_floating_point():
            dtype = x.dtype
            x.register_hook(lambda grad: grad.to(dtype=dtype))
        return x

    def flow_loss(
        self,
        history: torch.Tensor,
        target_future: torch.Tensor,
        physics_future: torch.Tensor,
        *,
        history_context: torch.Tensor | None = None,
        trajectory: torch.Tensor | None = None,
        window_origins: torch.Tensor | None = None,
        t_override: float | torch.Tensor | None = None,
        source_noise: torch.Tensor | None = None,
        return_endpoint: bool = False,
        force_conditioned: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        if history.ndim != 5 or target_future.ndim != 5 or physics_future.ndim != 5:
            raise ValueError("history/target/physics must be [B,T,C,H,W]")
        if target_future.shape != physics_future.shape:
            raise ValueError("target_future and physics_future must match")
        if history.shape[0] != target_future.shape[0] or history.shape[2:] != target_future.shape[2:]:
            raise ValueError("history and future B/C/H/W must match")
        self._validate_context(history_context, history)
        native_history, native_trajectory = self._native_backbone_condition(history, trajectory)

        hist = native_history * self.rescale_factor
        target = target_future * self.rescale_factor
        physics = physics_future * self.rescale_factor
        # Keep the new context path full-history; only the inherited native path
        # obeys HIST_LAST=4.
        ctx = None if history_context is None else history_context * self.rescale_factor
        if source_noise is None:
            source = torch.randn_like(target)
        else:
            if source_noise.shape != target.shape:
                raise ValueError("source_noise shape mismatch")
            source = source_noise.to(device=target.device, dtype=target.dtype)

        bs = int(target.shape[0])
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

        if self.training and not force_conditioned and self.unconditional_probability > 0:
            keep = (
                torch.rand(bs, 1, 1, 1, 1, device=target.device)
                >= self.unconditional_probability
            ).to(dtype=target.dtype)
            hist_cond = hist * keep
            physics_cond = physics * keep
            ctx_cond = None if ctx is None else ctx * keep
            conditioned_fraction = float(keep.float().mean().detach().cpu())
        else:
            hist_cond, physics_cond, ctx_cond = hist, physics, ctx
            conditioned_fraction = 1.0

        xt = (1.0 - t) * source + t * target
        seq = torch.cat([hist_cond, xt], dim=1)
        prior = self._align_prior(physics_cond, hist_cond)
        batch = {
            "noised_sequence": seq,
            "timesteps": t[:, 0, 0, 0, 0] * self.time_scalar,
            "trajectory": native_trajectory,
            "prior_condition": prior,
            "history_context": ctx_cond,
            "window_origins": window_origins,
        }
        pred = self.transition(batch)["predicted_latent"][:, history.shape[1]:]
        pred = self._force_output_grad_dtype(pred)
        target_velocity = target - source
        loss = (pred - target_velocity).square().mean()

        with torch.no_grad():
            pf = pred.float().reshape(bs, -1)
            yf = target_velocity.float().reshape(bs, -1)
            dot = (pf * yf).sum(dim=1)
            denom = pf.norm(dim=1) * yf.norm(dim=1)
            cosine = torch.where(
                denom > 0,
                dot / denom.clamp_min(1e-12),
                torch.zeros_like(dot),
            )
            target_rms = target_velocity.float().square().mean().sqrt()
            pred_rms = pred.float().square().mean().sqrt()
        info = {
            "loss": float(loss.detach().cpu()),
            "target_rms": float(target_rms.detach().cpu()),
            "pred_rms": float(pred_rms.detach().cpu()),
            "cosine": float(cosine.mean().detach().cpu()),
            "conditioned_fraction": conditioned_fraction,
            "hist_last": self.hist_last,
        }
        if return_endpoint:
            # Auxiliary x1 reconstruction from the sampled FM state. This is a
            # semantic-training surrogate, not a replacement for ODE sampling.
            info["predicted_endpoint"] = (
                xt + (1.0 - t) * pred
            ) / self.rescale_factor
            info["sampled_t"] = t
        return loss, info

    @torch.no_grad()
    def sample(
        self,
        history: torch.Tensor,
        physics_future: torch.Tensor,
        *,
        history_context: torch.Tensor | None = None,
        trajectory: torch.Tensor | None = None,
        window_origins: torch.Tensor | None = None,
        initial_noise: torch.Tensor | None = None,
        guidance_scale: float | None = None,
    ) -> torch.Tensor:
        if history.ndim != 5 or physics_future.ndim != 5:
            raise ValueError("history/physics must be [B,T,C,H,W]")
        if history.shape[0] != physics_future.shape[0] or history.shape[2:] != physics_future.shape[2:]:
            raise ValueError("history and physics B/C/H/W must match")
        self._validate_context(history_context, history)
        native_history, native_trajectory = self._native_backbone_condition(history, trajectory)

        hist = native_history * self.rescale_factor
        physics = physics_future * self.rescale_factor
        ctx = None if history_context is None else history_context * self.rescale_factor
        if initial_noise is None:
            future = torch.randn_like(physics)
        else:
            if initial_noise.shape != physics.shape:
                raise ValueError("initial_noise shape mismatch")
            future = initial_noise.to(device=physics.device, dtype=physics.dtype)
        scale = self.guidance_scale if guidance_scale is None else float(guidance_scale)

        timesteps = torch.linspace(
            0, 1, self.sample_steps + 1, device=hist.device, dtype=hist.dtype
        )
        shifted = 1 - (self.alpha_shift * timesteps) / (
            1 + (self.alpha_shift - 1) * timesteps
        )
        shifted = shifted.flip(0)
        for tc, tp in zip(shifted[:-1], shifted[1:]):
            cond_seq = torch.cat([hist, future], dim=1)
            if scale == 1.0:
                seq = cond_seq
                prior = self._align_prior(physics, hist)
                ctx_batch = ctx
                traj_batch = native_trajectory
                origins_batch = window_origins
            else:
                uncond_seq = torch.cat([torch.zeros_like(hist), future], dim=1)
                seq = torch.cat([uncond_seq, cond_seq], dim=0)
                prior_cond = self._align_prior(physics, hist)
                prior = torch.cat([torch.zeros_like(prior_cond), prior_cond], dim=0)
                ctx_batch = None if ctx is None else torch.cat([torch.zeros_like(ctx), ctx], dim=0)
                traj_batch = (
                    None if native_trajectory is None
                    else torch.cat([native_trajectory, native_trajectory], dim=0)
                )
                origins_batch = (
                    None if window_origins is None
                    else torch.cat([window_origins, window_origins], dim=0)
                )
            batch = {
                "noised_sequence": seq,
                "timesteps": torch.full(
                    (seq.shape[0],),
                    float(tc * self.time_scalar),
                    device=hist.device,
                    dtype=hist.dtype,
                ),
                "trajectory": traj_batch,
                "prior_condition": prior,
                "history_context": ctx_batch,
                "window_origins": origins_batch,
            }
            velocity = self.transition(batch)["predicted_latent"][:, history.shape[1]:]
            if scale != 1.0:
                uncond_v, cond_v = torch.chunk(velocity, 2, dim=0)
                velocity = uncond_v + scale * (cond_v - uncond_v)
            future = future + (tp - tc) * velocity
        return future / self.rescale_factor
