import torch

from real_motion.models.anchor_cfm import AnchorWindowCFM


def test_force_output_grad_dtype_casts_fp32_semantic_grad_back_to_bf16():
    x = torch.randn(8, dtype=torch.bfloat16, requires_grad=True)
    y = x * torch.tensor(2.0, dtype=torch.bfloat16)
    seen = []

    def record(grad):
        seen.append(grad.dtype)
        return grad

    # Record the gradient that actually leaves the protected transition-output
    # boundary toward the BF16 producer.
    y.register_hook(record)
    protected = AnchorWindowCFM._force_output_grad_dtype(y)

    # Mimic a frozen FP32 decoder / semantic objective sending an FP32 gradient
    # back to a BF16 World-Model output.
    grad_out = torch.ones(protected.shape, dtype=torch.float32)
    torch.autograd.backward(protected, grad_tensors=grad_out)

    assert seen == [torch.bfloat16]
    assert x.grad is not None
    assert x.grad.dtype == torch.bfloat16
