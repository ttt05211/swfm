"""Factory for P0-F9 physics-conditioned native sparse forecasting."""
from __future__ import annotations

from real_motion.flash_attention_compat import patch_occfm_flash_attention_backward_dtype

from .native_cfm import NativeFutureWindowCFM
from .transition_native_physics import MotionWindowNativePhysicsTransition


P0_F9_PROTOCOL = "p0_f9_physics_conditioned_native_sparse_forecast_v1"


def make_p0_f9_model(
    window: int = 20,
    *,
    sample_steps: int = 10,
    unconditional_probability: float = 0.2,
    guidance_scale: float = 2.0,
):
    patch_occfm_flash_attention_backward_dtype()
    tr = MotionWindowNativePhysicsTransition(
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
        physics_mid_channels=256,
        physics_heads=8,
    )
    return NativeFutureWindowCFM(
        tr,
        rescale_factor=10.0,
        sample_steps=int(sample_steps),
        alpha_shift=3.0,
        unconditional_probability=float(unconditional_probability),
        guidance_scale=float(guidance_scale),
    )
