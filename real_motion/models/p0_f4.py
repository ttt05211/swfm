"""Factory for the P0-F4 strong-anchor full-context Sparse World Model."""
from __future__ import annotations

from .anchor_cfm import AnchorWindowCFM
from .transition import MotionWindowFlowMatching


def make_p0_f4_model(window=20, *, sample_steps=10, source_noise_std=0.0):
    tr = MotionWindowFlowMatching(
        in_channels=16,
        out_channels=16,
        model_channels=128,
        channel_multi=[2, 4],
        input_size=[int(window), int(window)],
        trajectory_length=12,
        init_kernel_size=7,
        init_3d_conv_channels=64,
        attn_dim=32,
        temporal_attn_head=8,
        spatial_attn_head=8,
        prior_channels=16,
        context_channels=16,
        full_input_size=(50, 50),
    )
    return AnchorWindowCFM(
        tr,
        rescale_factor=10.0,
        sample_steps=int(sample_steps),
        alpha_shift=3.0,
        source_noise_std=float(source_noise_std),
    )
