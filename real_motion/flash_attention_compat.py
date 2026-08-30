"""Runtime compatibility fix for the pinned OccFM custom FlashAttention backward.

The pinned custom autograd Function was written assuming all backward einsums
run in one dtype. Under decoder-aware P0-F6, PyTorch can re-enter its backward
with autocast active: saved q/k/v and grad_output may be FP32 while einsum is
autocast to BF16, producing ``p`` in BF16 and ``doc`` in FP32. The upstream
implementation then fails at ``einsum(p, doc)``.

Patch only the backward numerical boundary: cast grad_output to the saved q
dtype and run the untouched upstream backward with autocast explicitly disabled.
Forward values, parameters, losses, routing, and checkpoints are unchanged.
"""
from __future__ import annotations

from contextlib import nullcontext

import torch


def _autocast_disabled_for(tensor: torch.Tensor):
    """Return an autocast-disabled context for supported torch device types."""
    device_type = tensor.device.type
    if device_type in {"cuda", "cpu", "xpu", "mps"}:
        return torch.autocast(device_type=device_type, enabled=False)
    return nullcontext()


def _patch_flash_attention_class(cls) -> bool:
    """Install the dtype/autocast guard on one FlashAttention autograd class."""
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
            # Critical: the original custom backward contains torch.einsum calls.
            # If autocast remains active, those einsums may emit BF16 even when
            # saved q/k/v + grad_output are FP32, recreating p/doc mismatch.
            with _autocast_disabled_for(grad_output):
                return original_backward(ctx, grad_output)
        return original_backward(ctx, grad_output)

    cls.backward = staticmethod(_patched_backward)
    cls._swfm_grad_dtype_patch = True
    cls._swfm_original_backward = original_backward
    return True


def patch_occfm_flash_attention_backward_dtype() -> bool:
    """Patch the pinned OccFM FlashAttentionFunction once per process."""
    from forecast.ops.flash_attention.flash_attention import FlashAttentionFunction

    return _patch_flash_attention_class(FlashAttentionFunction)
