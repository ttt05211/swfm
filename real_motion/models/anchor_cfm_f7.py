"""P0-F7 innovation-weighted anchor-centered flow matching.

This keeps the P0-F6 endpoint/semantic path unchanged and changes only the base
FM reduction. Every latent element remains supervised, but cells whose encoded
repair endpoint differs strongly from the Strong-W2Det anchor receive more
relative weight. The weights are bounded and normalized to unit mean per
sample, so the global FM scale stays comparable to P0-F6.
"""
from __future__ import annotations

import torch

from .anchor_cfm import AnchorWindowCFM


class InnovationWeightedAnchorWindowCFM(AnchorWindowCFM):
    """Anchor CFM with bounded soft emphasis on repair-energy cells."""

    @staticmethod
    def innovation_weights(
        velocity_target: torch.Tensor,
        *,
        alpha: float,
        eps: float = 1e-6,
    ) -> tuple[torch.Tensor, dict]:
        """Return unit-mean soft weights derived from target repair energy.

        ``velocity_target`` is ``Z_repair - Z_anchor`` after the model's latent
        rescale. Energy is channel RMS per latent cell. The bounded focus

            focus = energy / (energy + mean_energy)

        lies in [0,1]. Raw weights are ``1 + alpha * focus`` and are normalized
        to mean 1 per sample. Therefore alpha=0 gives uniform weights exactly,
        while alpha>0 increases gradient share on true innovation without
        dropping supervision anywhere.
        """
        if alpha < 0:
            raise ValueError("innovation weight alpha must be non-negative")
        if velocity_target.ndim != 5:
            raise ValueError("velocity_target must be [B,T,C,H,W]")

        vf = velocity_target.detach().float()
        energy = vf.square().mean(dim=2, keepdim=True).sqrt()
        mean_energy = energy.mean(dim=(1, 3, 4), keepdim=True)
        focus = energy / (energy + mean_energy + float(eps))
        raw = 1.0 + float(alpha) * focus
        norm = raw.mean(dim=(1, 3, 4), keepdim=True).clamp_min(float(eps))
        weight = raw / norm
        info = {
            "innovation_energy_mean": float(energy.mean().cpu()),
            "innovation_focus_mean": float(focus.mean().cpu()),
            "innovation_weight_mean": float(weight.mean().cpu()),
            "innovation_weight_max": float(weight.max().cpu()),
            "innovation_weight_min": float(weight.min().cpu()),
        }
        return weight, info

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
        innovation_weight_alpha: float = 0.0,
    ) -> tuple[torch.Tensor, dict]:
        if target_future.shape != anchor_future.shape:
            raise ValueError("target_future and anchor_future must match")
        if history.ndim != 5 or target_future.ndim != 5:
            raise ValueError("history/target must be [B,T,C,H,W]")
        if innovation_weight_alpha < 0:
            raise ValueError("innovation_weight_alpha must be non-negative")
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
        sq_native = (pred - velocity_target).square()

        # Preserve the exact P0-F6 FM reduction for the alpha=0 ablation.
        if float(innovation_weight_alpha) == 0.0:
            weight, weight_info = self.innovation_weights(velocity_target, alpha=0.0)
            if mask is None:
                loss = sq_native.mean()
                unweighted_loss = loss
                metric_mask = None
            else:
                denom = mask.sum()
                if float(denom.detach().cpu()) <= 0:
                    raise ValueError("loss_mask selects no latent elements")
                loss = (sq_native * mask).sum() / denom
                unweighted_loss = loss
                metric_mask = mask
        else:
            sq = sq_native.float()
            weight, weight_info = self.innovation_weights(
                velocity_target,
                alpha=float(innovation_weight_alpha),
            )
            weight_full = weight.expand(-1, -1, sq.shape[2], -1, -1)
            if mask is None:
                loss = (sq * weight_full).mean()
                unweighted_loss = sq.mean()
                metric_mask = None
            else:
                mf = mask.float()
                denom = (mf * weight_full).sum()
                if float(denom.detach().cpu()) <= 0:
                    raise ValueError("loss_mask selects no latent elements")
                loss = (sq * mf * weight_full).sum() / denom
                unweighted_loss = (sq * mf).sum() / mf.sum().clamp_min(1.0)
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
            "unweighted_fm_loss": float(unweighted_loss.detach().cpu()),
            "target_rms": float(target_rms.detach().cpu()),
            "pred_rms": float(pred_rms.detach().cpu()),
            "cosine": float(cosine.mean().detach().cpu()),
            "loss_active_fraction": active_fraction,
            "innovation_weight_alpha": float(innovation_weight_alpha),
            **weight_info,
        }
        if return_endpoint:
            info["predicted_endpoint"] = (
                xt + (1.0 - t) * pred
            ) / self.rescale_factor
            info["sampled_t"] = t
        return loss, info
