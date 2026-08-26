import torch
import torch.nn as nn


class RealMotionWindowCFM(nn.Module):
    """CFM trainer/sampler for motion-window latent inpainting.

    Only ``active_mask`` cells are stochastic/generative. Cells outside the
    causal generation support are clamped to a known empty-latent canvas at
    *every* training/sampling step. This prevents unsupervised window margins
    from becoming dynamic false positives or influencing the active region as
    random noise.
    """
    def __init__(self, transition: nn.Module, rescale_factor=10.0,
                 sample_steps=10, alpha_shift=3.0):
        super().__init__()
        self.transition = transition
        self.rescale_factor = float(rescale_factor)
        self.sample_steps = int(sample_steps)
        self.alpha_shift = float(alpha_shift)
        self.time_scalar = 1000.0

    @staticmethod
    def _sample_t(bs, device, dtype):
        return torch.sigmoid(torch.randn(bs, 1, 1, 1, 1, device=device, dtype=dtype))

    @staticmethod
    def _expand_mask(active_mask, channels):
        mask = active_mask.bool()
        if mask.ndim != 5:
            raise ValueError("active_mask must be [B,F,1|C,H,W]")
        if mask.shape[2] == 1:
            mask = mask.expand(-1, -1, channels, -1, -1)
        elif mask.shape[2] != channels:
            raise ValueError("active_mask channel mismatch")
        return mask

    @staticmethod
    def _align_prior(prior, history, future_frames):
        """Return a prior aligned to the complete history+future sequence.

        The public cache contract stores **future-only** static/KTA priors.  We
        also accept an already aligned sequence for integration tests, but any
        other temporal length is rejected instead of being guessed from H/F.
        """
        if prior.ndim != 5:
            raise ValueError("prior must be [B,F|H+F,P,H,W]")
        if prior.shape[1] == int(future_frames):
            return torch.cat([
                torch.zeros(prior.shape[0], history.shape[1], *prior.shape[2:],
                            device=prior.device, dtype=prior.dtype),
                prior,
            ], dim=1)
        expected = history.shape[1] + int(future_frames)
        if prior.shape[1] == expected:
            return prior
        raise ValueError(
            f"prior has {prior.shape[1]} frames; expected future-only "
            f"{future_frames} or aligned history+future {expected}"
        )

    def flow_loss(self, history, future, prior, active_mask, known_future,
                  trajectory=None, window_origins=None):
        """Masked CFM loss.

        Args:
            history/future/known_future: [B,T,C,H,W] unscaled latent tensors.
            prior: [B,F,P,H,W] (static+KTA); history prior is prepended as zero.
            active_mask: [B,F,1,H,W] causal generation support.
            known_future: usually crops from E(empty), used outside active support.
        """
        if future.shape != known_future.shape:
            raise ValueError("future and known_future must have identical shape")
        mask = self._expand_mask(active_mask, future.shape[2])

        hist = history * self.rescale_factor
        target = torch.where(mask, future, known_future) * self.rescale_factor
        known = known_future * self.rescale_factor

        bs = future.shape[0]
        t = self._sample_t(bs, future.device, future.dtype)
        z0 = torch.randn_like(future)
        noised_active = t * target + (1.0 - t) * z0
        noised = torch.where(mask, noised_active, known)
        seq = torch.cat([hist, noised], dim=1)

        # Official OccFM scales latent states by RESCALE_FACTOR before the
        # transition. Static/KTA condition latents use the same frozen VAE, so
        # put them in the same numeric convention before the zero-init adapter.
        prior = self._align_prior(prior, history, future.shape[1]) * self.rescale_factor
        batch = {
            "noised_sequence": seq,
            "timesteps": t[:, 0, 0, 0, 0] * self.time_scalar,
            "trajectory": trajectory,
            "prior_condition": prior,
            "window_origins": window_origins,
        }
        pred = self.transition(batch)["predicted_latent"][:, history.shape[1]:]
        velocity_target = target - z0
        mask_f = mask.to(pred.dtype)
        denom = mask_f.sum().clamp_min(1.0)
        loss = ((pred - velocity_target).square() * mask_f).sum() / denom
        return loss, {
            "pred": pred,
            "target": velocity_target,
            "loss_mask_fraction": active_mask.float().mean().detach(),
        }

    @torch.no_grad()
    def sample(self, history, future_shape, prior, active_mask, known_future,
               trajectory=None, window_origins=None, initial_noise=None):
        """Single-forward-per-step sampler; no CFG by design.

        ``initial_noise`` can be cropped from one global 50x50 noise canvas.
        Overlapping windows therefore start from identical noise at the same
        global latent cell.
        """
        hist = history * self.rescale_factor
        if tuple(known_future.shape) != tuple(future_shape):
            raise ValueError("known_future shape mismatch")
        mask = self._expand_mask(active_mask, future_shape[2])
        known = known_future * self.rescale_factor

        if initial_noise is None:
            noise = torch.randn(future_shape, device=hist.device, dtype=hist.dtype)
        else:
            if tuple(initial_noise.shape) != tuple(future_shape):
                raise ValueError("initial_noise shape mismatch")
            noise = initial_noise.to(device=hist.device, dtype=hist.dtype)
        future = torch.where(mask, noise, known)

        timesteps = torch.linspace(0, 1, self.sample_steps + 1,
                                   device=hist.device, dtype=hist.dtype)
        shifted = 1 - (self.alpha_shift * timesteps) / (
            1 + (self.alpha_shift - 1) * timesteps
        )
        shifted = shifted.flip(0)
        prior = self._align_prior(prior, history, future_shape[1]) * self.rescale_factor

        for tc, tp in zip(shifted[:-1], shifted[1:]):
            seq = torch.cat([hist, future], dim=1)
            batch = {
                "noised_sequence": seq,
                "timesteps": torch.full((hist.shape[0],), float(tc * self.time_scalar),
                                         device=hist.device, dtype=hist.dtype),
                "trajectory": trajectory,
                "prior_condition": prior,
                "window_origins": window_origins,
            }
            v = self.transition(batch)["predicted_latent"][:, history.shape[1]:]
            updated = future + (tp - tc) * v
            future = torch.where(mask, updated, known)

        return future / self.rescale_factor
