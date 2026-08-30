import math

import torch

from real_motion.flash_attention_compat import (
    _fp32_flash_attention_backward,
    _patch_flash_attention_class,
)


class _Ctx:
    pass


def _make_mixed_dtype_ctx():
    # Mimic the pinned forward contract under BF16 AMP: q/k/v/o are BF16,
    # while lse is FP32 because upstream row-sum/max buffers use default dtype.
    q = torch.tensor([[[[0.25, -0.50], [0.75, 0.10]]]], dtype=torch.bfloat16)
    k = torch.tensor([[[[0.20, 0.30], [-0.40, 0.80]]]], dtype=torch.bfloat16)
    v = torch.tensor([[[[0.50, -0.20], [0.10, 0.90]]]], dtype=torch.bfloat16)
    scale = q.shape[-1] ** -0.5

    qf, kf, vf = q.float(), k.float(), v.float()
    attn = torch.einsum("... i d, ... j d -> ... i j", qf, kf) * scale
    lse = torch.logsumexp(attn, dim=-1, keepdim=True).float()
    probs = torch.softmax(attn, dim=-1)
    o = torch.einsum("... i j, ... j d -> ... i d", probs, vf).to(torch.bfloat16)

    ctx = _Ctx()
    ctx.args = (False, scale, ((None,),), 2, 2)
    ctx.saved_tensors = (q, k, v, o, lse)
    return ctx


def test_fp32_backward_handles_bf16_qkv_with_fp32_lse_and_grad():
    ctx = _make_mixed_dtype_ctx()
    grad = torch.tensor([[[[1.0, -0.3], [0.2, 0.7]]]], dtype=torch.float32)

    dq, dk, dv, *rest = _fp32_flash_attention_backward(ctx, grad)

    assert dq.dtype == torch.bfloat16
    assert dk.dtype == torch.bfloat16
    assert dv.dtype == torch.bfloat16
    assert torch.isfinite(dq.float()).all()
    assert torch.isfinite(dk.float()).all()
    assert torch.isfinite(dv.float()).all()
    assert rest == [None, None, None, None]


def test_fp32_backward_matches_dense_reference_for_float32_case():
    q = torch.tensor([[[[0.25, -0.50], [0.75, 0.10]]]], dtype=torch.float32, requires_grad=True)
    k = torch.tensor([[[[0.20, 0.30], [-0.40, 0.80]]]], dtype=torch.float32, requires_grad=True)
    v = torch.tensor([[[[0.50, -0.20], [0.10, 0.90]]]], dtype=torch.float32, requires_grad=True)
    scale = q.shape[-1] ** -0.5
    attn = torch.einsum("... i d, ... j d -> ... i j", q, k) * scale
    probs = torch.softmax(attn, dim=-1)
    o = torch.einsum("... i j, ... j d -> ... i d", probs, v)
    lse = torch.logsumexp(attn.detach(), dim=-1, keepdim=True)
    grad = torch.tensor([[[[1.0, -0.3], [0.2, 0.7]]]], dtype=torch.float32)

    ref_dq, ref_dk, ref_dv = torch.autograd.grad(o, (q, k, v), grad_outputs=grad)

    ctx = _Ctx()
    ctx.args = (False, scale, ((None,),), 2, 2)
    ctx.saved_tensors = (q.detach(), k.detach(), v.detach(), o.detach(), lse.detach())
    dq, dk, dv, *_ = _fp32_flash_attention_backward(ctx, grad)

    assert torch.allclose(dq, ref_dq, atol=1e-5, rtol=1e-5)
    assert torch.allclose(dk, ref_dk, atol=1e-5, rtol=1e-5)
    assert torch.allclose(dv, ref_dv, atol=1e-5, rtol=1e-5)


def test_patch_is_idempotent_and_uses_fp32_backward():
    class FakeFlashAttentionFunction:
        @staticmethod
        def backward(ctx, grad_output):
            raise AssertionError("original backward must not be called")

    assert _patch_flash_attention_class(FakeFlashAttentionFunction) is True
    assert _patch_flash_attention_class(FakeFlashAttentionFunction) is False
    assert FakeFlashAttentionFunction._swfm_backward_mode == "fp32_full_backward"

    ctx = _make_mixed_dtype_ctx()
    grad = torch.ones_like(ctx.saved_tensors[4])
    # Reshape grad to attention output shape.
    grad = torch.ones_like(ctx.saved_tensors[3], dtype=torch.float32)
    dq, dk, dv, *_ = FakeFlashAttentionFunction.backward(ctx, grad)
    assert dq.dtype == torch.bfloat16
    assert dk.dtype == torch.bfloat16
    assert dv.dtype == torch.bfloat16
