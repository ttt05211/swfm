"""Runtime compatibility fix for the pinned OccFM custom FlashAttention backward.

The upstream autograd Function assumes grad_output has the same dtype as the
saved q/k/v tensors. Decoder-aware P0-F6 can legitimately send an FP32 gradient
back into a BF16 attention graph under AMP. Patch only that backward boundary:
cast grad_output to q.dtype before delegating to the untouched upstream
implementation. Forward values, parameters, and losses are unchanged.
"""
from __future__ import annotations

import torch


def _patch_flash_attention_class(cls) -> bool:
    """Install the dtype guard on one FlashAttention autograd Function class."""
    if getattr(cls, "_swfm_grad_dtype_patch", False):
        return False

    original_backward = cls.backward

    @torch.no_grad()
    def _patched_backward(ctx, grad_output):
        saved = ctx.saved_tensors
        if saved and torch.is_tensor(grad_output):
            q = saved[0]
            if grad_output.dtype != q.dtype:
                grad_output = grad_output.to(dtype=q.dtype)
        return original_backward(ctx, grad_output)

    cls.backward = staticmethod(_patched_backward)
    cls._swfm_grad_dtype_patch = True
    cls._swfm_original_backward = original_backward
    return True


def patch_occfm_flash_attention_backward_dtype() -> bool:
    """Patch the pinned OccFM FlashAttentionFunction once per process.

    Returns True when the patch is installed and False when it was already
    installed. The patch is intentionally local to runtime; the pinned upstream
    submodule remains byte-for-byte unchanged.
    """
    from forecast.ops.flash_attention.flash_attention import FlashAttentionFunction

    return _patch_flash_attention_class(FlashAttentionFunction)
