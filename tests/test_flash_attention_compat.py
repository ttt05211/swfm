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
