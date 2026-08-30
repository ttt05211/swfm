import torch

from real_motion.flash_attention_compat import _patch_flash_attention_class


class _Ctx:
    def __init__(self, q):
        self.saved_tensors = (q,)


def test_flash_attention_backward_patch_casts_grad_output_to_saved_q_dtype():
    seen = []

    class FakeFlashAttentionFunction:
        @staticmethod
        def backward(ctx, grad_output):
            seen.append(grad_output.dtype)
            return grad_output

    assert _patch_flash_attention_class(FakeFlashAttentionFunction) is True
    assert _patch_flash_attention_class(FakeFlashAttentionFunction) is False

    q = torch.zeros(4, dtype=torch.bfloat16)
    grad = torch.ones(4, dtype=torch.float32)
    out = FakeFlashAttentionFunction.backward(_Ctx(q), grad)

    assert seen == [torch.bfloat16]
    assert out.dtype == torch.bfloat16


def test_flash_attention_backward_patch_disables_active_autocast():
    seen = []

    class FakeFlashAttentionFunction:
        @staticmethod
        def backward(ctx, grad_output):
            # The real pinned backward uses einsum/matmul-like ops. Record the
            # autocast state and execute one matmul so this test covers the same
            # numerical boundary rather than only checking a flag.
            seen.append(torch.is_autocast_enabled("cpu"))
            x = grad_output.reshape(2, 2)
            return x @ x

    assert _patch_flash_attention_class(FakeFlashAttentionFunction) is True
    q = torch.zeros(4, dtype=torch.float32)
    grad = torch.ones(4, dtype=torch.float32)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True):
        out = FakeFlashAttentionFunction.backward(_Ctx(q), grad)

    assert seen == [False]
    assert out.dtype == torch.float32
