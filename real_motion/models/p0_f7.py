"""Factory for P0-F7 innovation-weighted Strong-W2Det anchor World Model."""
from __future__ import annotations

from .anchor_cfm_f7 import InnovationWeightedAnchorWindowCFM
from .p0_f4 import make_p0_f4_model


def make_p0_f7_model(window=20, *, sample_steps=10, source_noise_std=0.0):
    """Reuse the exact P0-F6 transition and AMP patch, changing only FM reduction."""
    base = make_p0_f4_model(
        window,
        sample_steps=sample_steps,
        source_noise_std=source_noise_std,
    )
    return InnovationWeightedAnchorWindowCFM(
        base.transition,
        rescale_factor=base.rescale_factor,
        sample_steps=base.sample_steps,
        alpha_shift=base.alpha_shift,
        source_noise_std=base.source_noise_std,
    )
