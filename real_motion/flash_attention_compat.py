"""Numerically safe backward for the pinned OccFM custom FlashAttention.

Why this is needed
------------------
The upstream forward keeps q/k/v/o in the AMP dtype (BF16 in P0-F6), but its
``all_row_sums`` / ``all_row_maxes`` tensors are created without an explicit
dtype. Consequently the saved log-sum-exp tensor ``lse`` is FP32. In the
upstream backward this makes

    p = exp(BF16_attn_weights - FP32_lse)

a FP32 tensor, while ``doc`` can still be BF16. The next ``einsum(p, doc)``
therefore fails with a BF16/FP32 dtype mismatch. Casting only grad_output or
disabling autocast cannot fix that promotion caused by the saved FP32 lse.

P0-F6 keeps the upstream forward byte-for-byte unchanged and replaces only the
runtime backward implementation. All backward algebra is evaluated in FP32,
then dq/dk/dv are cast back to the original q/k/v dtypes before returning.
This preserves the model, forward values, losses, checkpoints, and routing while
making decoder-aware gradients well-defined under BF16 AMP.
"""
from __future__ import annotations

import math

import torch


def _fp32_flash_attention_backward(ctx, grad_output):
    """Reproduce the pinned OccFM FlashAttention backward entirely in FP32."""
    causal, scale, mask, q_bucket_size, k_bucket_size = ctx.args
    q_saved, k_saved, v_saved, o_saved, lse_saved = ctx.saved_tensors

    q_dtype = q_saved.dtype
    k_dtype = k_saved.dtype
    v_dtype = v_saved.dtype
    device = q_saved.device

    # Critical contract: every floating tensor participating in backward
    # arithmetic is one dtype, irrespective of how the upstream forward mixed
    # BF16 q/k/v/o with FP32 lse.
    q = q_saved.float()
    k = k_saved.float()
    v = v_saved.float()
    o = o_saved.float()
    lse = lse_saved.float()
    do = grad_output.float()

    max_neg_value = -torch.finfo(torch.float32).max
    qk_len_diff = max(k.shape[-2] - q.shape[-2], 0)

    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)

    row_splits = zip(
        q.split(q_bucket_size, dim=-2),
        o.split(q_bucket_size, dim=-2),
        do.split(q_bucket_size, dim=-2),
        mask,
        lse.split(q_bucket_size, dim=-2),
        dq.split(q_bucket_size, dim=-2),
    )

    for ind, (qc, oc, doc, row_mask, lsec, dqc) in enumerate(row_splits):
        q_start_index = ind * q_bucket_size - qk_len_diff

        col_splits = zip(
            k.split(k_bucket_size, dim=-2),
            v.split(k_bucket_size, dim=-2),
            dk.split(k_bucket_size, dim=-2),
            dv.split(k_bucket_size, dim=-2),
            row_mask,
        )

        for k_ind, (kc, vc, dkc, dvc, col_mask) in enumerate(col_splits):
            k_start_index = k_ind * k_bucket_size

            attn_weights = torch.einsum("... i d, ... j d -> ... i j", qc, kc) * float(scale)

            if causal and q_start_index < (k_start_index + k_bucket_size - 1):
                causal_mask = torch.ones(
                    (qc.shape[-2], kc.shape[-2]),
                    dtype=torch.bool,
                    device=device,
                ).triu(q_start_index - k_start_index + 1)
                attn_weights.masked_fill_(causal_mask, max_neg_value)

            p = torch.exp(attn_weights - lsec)
            if col_mask is not None:
                p.masked_fill_(~col_mask, 0.0)

            # All operands below are guaranteed FP32.
            dv_chunk = torch.einsum("... i j, ... i d -> ... j d", p, doc)
            dp = torch.einsum("... i d, ... j d -> ... i j", doc, vc)
            D = (doc * oc).sum(dim=-1, keepdims=True)
            ds = p * float(scale) * (dp - D)
            dq_chunk = torch.einsum("... i j, ... j d -> ... i d", ds, kc)
            dk_chunk = torch.einsum("... i j, ... i d -> ... j d", ds, qc)

            dqc.add_(dq_chunk)
            dkc.add_(dk_chunk)
            dvc.add_(dv_chunk)

    return (
        dq.to(dtype=q_dtype),
        dk.to(dtype=k_dtype),
        dv.to(dtype=v_dtype),
        None,
        None,
        None,
        None,
    )


def _patch_flash_attention_class(cls) -> bool:
    """Install the FP32 backward once on a FlashAttention autograd class."""
    if getattr(cls, "_swfm_grad_dtype_patch", False):
        return False

    original_backward = cls.backward

    @torch.no_grad()
    def _patched_backward(ctx, grad_output):
        return _fp32_flash_attention_backward(ctx, grad_output)

    cls.backward = staticmethod(_patched_backward)
    cls._swfm_grad_dtype_patch = True
    cls._swfm_original_backward = original_backward
    cls._swfm_backward_mode = "fp32_full_backward"
    return True


def patch_occfm_flash_attention_backward_dtype() -> bool:
    """Patch the pinned OccFM FlashAttentionFunction once per process."""
    from forecast.ops.flash_attention.flash_attention import FlashAttentionFunction

    return _patch_flash_attention_class(FlashAttentionFunction)
