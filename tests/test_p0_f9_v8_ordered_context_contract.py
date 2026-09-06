from pathlib import Path

import torch

from real_motion.models.p0_f9 import make_p0_f9_model
from real_motion.models.transition_ordered_context import (
    ORDERED_CONTEXT_PROTOCOL,
    ordered_context_is_exact_zero,
)


def test_ordered_context_zero_init_is_exact_mean_context_noop():
    model = make_p0_f9_model(
        20,
        sample_steps=10,
        unconditional_probability=0.0,
        guidance_scale=1.0,
        hist_last=4,
        ordered_context=True,
        ordered_context_enabled=True,
    )
    tr = model.transition
    assert ordered_context_is_exact_zero(tr)
    assert tr.ordered_context_proj.in_channels == 96
    assert tr.ordered_context_proj.out_channels == 128
    assert tr.ordered_context_proj.kernel_size == (3, 3)
    assert tr.ordered_context_proj.stride == (2, 2)

    torch.manual_seed(7)
    with torch.no_grad():
        tr.context_proj.weight.normal_(0.0, 0.01)
        tr.context_proj.bias.normal_(0.0, 0.01)
    ctx = torch.randn(2, 6, 16, 40, 40)
    enabled = tr._context_base(ctx, 2, torch.device("cpu"), torch.float32)
    tr.set_ordered_context_enabled(False)
    dormant = tr._context_base(ctx, 2, torch.device("cpu"), torch.float32)
    torch.testing.assert_close(enabled, dormant, rtol=0.0, atol=0.0)


def test_ordered_context_becomes_order_sensitive_after_nonzero_weights():
    model = make_p0_f9_model(
        20,
        ordered_context=True,
        ordered_context_enabled=True,
    )
    tr = model.transition
    torch.manual_seed(11)
    with torch.no_grad():
        tr.context_proj.weight.normal_(0.0, 0.01)
        tr.context_proj.bias.normal_(0.0, 0.01)
        tr.ordered_context_proj.weight.normal_(0.0, 0.01)
        tr.ordered_context_proj.bias.zero_()
    ctx = torch.randn(1, 6, 16, 40, 40)
    forward = tr._context_base(ctx, 1, torch.device("cpu"), torch.float32)
    reverse = tr._context_base(ctx.flip(1), 1, torch.device("cpu"), torch.float32)
    # Temporal mean is invariant to reversal. Any difference therefore comes
    # from the ordered residual and proves that H0..H5 order is represented.
    assert not torch.allclose(forward, reverse)


def test_base_factory_has_no_extra_ordered_state_by_default():
    base = make_p0_f9_model(20, ordered_context=False)
    ordered = make_p0_f9_model(20, ordered_context=True)
    base_keys = set(base.state_dict())
    ordered_keys = set(ordered.state_dict())
    assert ordered_keys - base_keys == {
        "transition.ordered_context_proj.weight",
        "transition.ordered_context_proj.bias",
    }


def test_ordered_phase_source_keeps_single_loss_and_milestone_policy():
    root = Path(__file__).resolve().parents[1]
    text = (root / "tools/real_motion/train_p0_f9_v8_ordered_context_phase.py").read_text()
    assert 'VARIANTS = ("MCTRL", "MT")' in text
    assert "normalized_motion_weighted_mse" in text
    assert "total_loss = normalized_motion_weighted_mse" in text
    assert "ordered_context_proj.requires_grad_(False)" in text
    assert "ordered_context_proj.requires_grad_(True)" in text
    assert 'ck_path = out / f"step_{phase_step:04d}.pt"' in text
    assert "ORDERED_CONTEXT_PROTOCOL" in text
    assert "semantic_loss_for_endpoint" not in text
    # Metadata explicitly records that the old auxiliary is disabled; this is
    # allowed. The contract we care about is that no Lovasz implementation is
    # imported/called anywhere in this phase trainer.
    assert '"lovasz_auxiliary": False' in text
    assert "lovasz_softmax" not in text
    assert ORDERED_CONTEXT_PROTOCOL.startswith("concat_6_history_frames")
