"""Factory for the P0-F4 strong-anchor full-context Sparse World Model."""
from __future__ import annotations

from real_motion.flash_attention_compat import patch_occfm_flash_attention_backward_dtype

from .anchor_cfm import AnchorWindowCFM
from .transition_full_context import MotionWindowFlowMatchingFullContext


def make_p0_f4_model(window=20, *, sample_steps=10, source_noise_std=0.0):
    # P0-F6 decoder-aware training can send FP32 semantic gradients into the
    # BF16 graph created by the pinned OccFM custom FlashAttention under AMP.
    # Patch that upstream backward boundary once per process; this is a no-op on
    # repeated calls and does not alter forward values or checkpoint contents.
    patch_occfm_flash_attention_backward_dtype()
    tr = MotionWindowFlowMatchingFullContext(
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
