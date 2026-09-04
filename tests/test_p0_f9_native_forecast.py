import torch
import torch.nn as nn
import torch.nn.functional as F

from real_motion.edit_repair import WRITE_OFFSET
from real_motion.model_ema import ModelEMA
from real_motion.models.native_cfm import NativeFutureWindowCFM
from real_motion.models.physics_prior import GatedPhysicsCrossAttention
from real_motion.native_forecast import (
    absolute_future_semantic_loss,
    collapse_occ_logits_to_dynamic,
    deterministic_sample_seed,
)


class _CaptureTransition(nn.Module):
    def __init__(self):
        super().__init__()
        self.last = None

    def forward(self, batch):
        self.last = {k: v for k, v in batch.items()}
        batch = dict(batch)
        batch["predicted_latent"] = torch.zeros_like(batch["noised_sequence"])
        return batch


def test_native_cfm_uses_gaussian_source_not_physics_anchor():
    tr = _CaptureTransition()
    model = NativeFutureWindowCFM(
        tr,
        rescale_factor=10.0,
        unconditional_probability=0.0,
        guidance_scale=1.0,
        hist_last=2,
    )
    history = torch.ones(1, 2, 1, 1, 1)
    target = torch.full((1, 2, 1, 1, 1), 5.0)
    physics = torch.full_like(target, 99.0)
    noise = torch.full_like(target, 3.0)
    loss, info = model.flow_loss(
        history,
        target,
        physics,
        t_override=0.25,
        source_noise=noise,
        return_endpoint=True,
        force_conditioned=True,
    )

    # Target is rescaled to 50, but the Gaussian source stays unit-scale exactly
    # like the official OccFM CFM. The Strong-W2Det value 99 never enters x_t.
    expected_xt = 0.75 * 3.0 + 0.25 * 50.0
    captured_future = tr.last["noised_sequence"][:, history.shape[1]:]
    assert torch.allclose(captured_future, torch.full_like(captured_future, expected_xt))
    assert torch.allclose(tr.last["prior_condition"][:, history.shape[1]:], physics * 10.0)
    assert torch.allclose(loss, torch.tensor((50.0 - 3.0) ** 2))
    # Zero velocity endpoint is x_t / rescale, not the physics anchor.
    assert torch.allclose(info["predicted_endpoint"], captured_future / 10.0)


def test_native_hist_last_masks_loaded_backbone_but_not_full_context():
    tr = _CaptureTransition()
    model = NativeFutureWindowCFM(
        tr,
        rescale_factor=10.0,
        unconditional_probability=0.0,
        guidance_scale=1.0,
        hist_last=4,
    )
    history = torch.arange(1, 7, dtype=torch.float32).reshape(1, 6, 1, 1, 1)
    context = (100 + torch.arange(1, 7, dtype=torch.float32)).reshape(1, 6, 1, 1, 1)
    target = torch.ones(1, 6, 1, 1, 1)
    physics = torch.ones_like(target)
    trajectory = torch.arange(24, dtype=torch.float32).reshape(1, 12, 2) + 1
    model.flow_loss(
        history,
        target,
        physics,
        history_context=context,
        trajectory=trajectory,
        t_override=0.5,
        source_noise=torch.zeros_like(target),
        force_conditioned=True,
    )
    native_hist = tr.last["noised_sequence"][:, :6]
    assert torch.equal(native_hist[:, :2], torch.zeros_like(native_hist[:, :2]))
    assert torch.equal(native_hist[:, 2:], history[:, 2:] * 10.0)
    # New surrounding context intentionally sees all six frames.
    assert torch.equal(tr.last["history_context"], context * 10.0)
    assert torch.equal(tr.last["trajectory"][:, :2], torch.zeros_like(trajectory[:, :2]))
    assert torch.equal(tr.last["trajectory"][:, 2:], trajectory[:, 2:])


def test_native_sampler_starts_from_explicit_noise_not_physics():
    tr = _CaptureTransition()
    model = NativeFutureWindowCFM(
        tr,
        rescale_factor=10.0,
        sample_steps=2,
        unconditional_probability=0.0,
        guidance_scale=1.0,
        hist_last=2,
    )
    history = torch.zeros(1, 2, 1, 1, 1)
    physics = torch.full((1, 2, 1, 1, 1), 8.0)
    initial = torch.full_like(physics, 4.0)
    out = model.sample(history, physics, initial_noise=initial)
    # Fake transition predicts zero velocity, so ODE leaves the noise untouched.
    assert torch.allclose(out, initial / 10.0)


def test_zero_gated_physics_cross_attention_is_exact_noop_and_gate_learns():
    torch.manual_seed(7)
    fusion = GatedPhysicsCrossAttention(prior_channels=3, hidden_size=8, num_heads=2)
    x = torch.randn(2, 8, 3, 2, 2, requires_grad=True)
    prior = torch.randn(2, 3, 3, 4, 4)
    out = fusion(x, prior)
    assert torch.equal(out, x)
    out.square().mean().backward()
    assert fusion.gate.grad is not None
    assert torch.isfinite(fusion.gate.grad)
    assert float(fusion.gate.grad.abs()) > 0.0

    fusion.zero_grad(set_to_none=True)
    fusion.gate.data.fill_(0.2)
    changed = fusion(x.detach(), prior)
    assert not torch.equal(changed, x.detach())


def test_absent_physics_frame_stays_exact_noop_after_attention_biases_learn():
    torch.manual_seed(9)
    fusion = GatedPhysicsCrossAttention(prior_channels=3, hidden_size=8, num_heads=2)
    fusion.gate.data.fill_(0.7)
    # Simulate learned attention biases that would otherwise create a constant
    # pseudo-condition from an all-zero aligned history prior.
    with torch.no_grad():
        fusion.attn.in_proj_bias.fill_(0.31)
        fusion.attn.out_proj.bias.fill_(-0.27)
    x = torch.randn(1, 8, 2, 2, 2)
    prior = torch.randn(1, 2, 3, 4, 4)
    prior[:, 0].zero_()
    out = fusion(x, prior)
    assert torch.equal(out[:, :, 0], x[:, :, 0])
    assert not torch.equal(out[:, :, 1], x[:, :, 1])


def test_collapsed_dynamic_probabilities_exactly_marginalize_18_way_softmax():
    torch.manual_seed(3)
    logits = torch.randn(11, 18)
    collapsed = collapse_occ_logits_to_dynamic(logits)
    p18 = logits.softmax(dim=-1)
    p9 = collapsed.softmax(dim=-1)
    dynamic = [2, 3, 4, 5, 6, 7, 9, 10]
    non_dynamic = [i for i in range(18) if i not in dynamic]
    assert torch.allclose(p9[:, 0], p18[:, non_dynamic].sum(dim=-1), atol=1e-6, rtol=1e-6)
    assert torch.allclose(p9[:, 1:], p18[:, dynamic], atol=1e-6, rtol=1e-6)


def _tiny_record():
    return {
        "sample_id": "scene:token",
        "scene_name": "scene",
        "edit_flat_indices": torch.tensor([0], dtype=torch.int32),
        "edit_actions": torch.tensor([WRITE_OFFSET], dtype=torch.uint8),
        "edit_anchor_slots": torch.tensor([0], dtype=torch.uint8),
        "edit_result_slots": torch.tensor([1], dtype=torch.uint8),
        "edit_moving": torch.tensor([True]),
        "keep_flat_indices": torch.tensor([1], dtype=torch.int32),
        "keep_anchor_slots": torch.tensor([0], dtype=torch.uint8),
        "keep_priority": torch.tensor([0], dtype=torch.uint8),
    }


def test_absolute_future_semantic_loss_uses_result_semantics_not_edit_actions():
    logits = torch.zeros(2, 18, requires_grad=True)
    # First voxel should be dynamic slot 1 == Occ3D class 2; second is background.
    logits.data[0, 2] = 3.0
    logits.data[1, 17] = 3.0
    weights = torch.ones(9)
    loss, info = absolute_future_semantic_loss(
        [logits],
        [_tiny_record()],
        class_weights=weights,
        lovasz_weight=1.0,
    )
    assert torch.isfinite(loss)
    assert info["num_supervised_voxels"] == 2
    assert info["accuracy"] == 1.0
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_semantic_loss_equal_weights_horizons_not_voxel_population():
    stride = 200 * 200 * 16
    # One difficult dynamic voxel at horizon 0, nine easy background voxels at h1.
    rec = {
        "sample_id": "scene:balanced",
        "scene_name": "scene",
        "edit_flat_indices": torch.tensor([0], dtype=torch.int32),
        "edit_actions": torch.tensor([WRITE_OFFSET], dtype=torch.uint8),
        "edit_anchor_slots": torch.tensor([0], dtype=torch.uint8),
        "edit_result_slots": torch.tensor([1], dtype=torch.uint8),
        "edit_moving": torch.tensor([True]),
        "keep_flat_indices": torch.tensor([stride + i for i in range(9)], dtype=torch.int32),
        "keep_anchor_slots": torch.zeros(9, dtype=torch.uint8),
        "keep_priority": torch.zeros(9, dtype=torch.uint8),
    }
    raw = torch.zeros(10, 18)
    raw[1:, 17] = 8.0
    weights = torch.ones(9)
    loss, info = absolute_future_semantic_loss(
        [raw], [rec], class_weights=weights, lovasz_weight=0.0
    )
    collapsed = collapse_occ_logits_to_dynamic(raw)
    ce_h0 = F.cross_entropy(collapsed[:1], torch.tensor([1]))
    ce_h1 = F.cross_entropy(collapsed[1:], torch.zeros(9, dtype=torch.long))
    expected = 0.5 * (ce_h0 + ce_h1)
    population_weighted = (ce_h0 + 9.0 * ce_h1) / 10.0
    assert torch.allclose(loss, expected, atol=1e-7, rtol=1e-7)
    assert not torch.allclose(loss, population_weighted, atol=1e-4, rtol=1e-4)
    assert info["per_horizon_voxels"][:2] == [1, 9]


def test_deterministic_sample_seed_is_stable_and_stream_specific():
    a = deterministic_sample_seed("scene:token", 123, stream="history")
    b = deterministic_sample_seed("scene:token", 123, stream="history")
    c = deterministic_sample_seed("scene:token", 123, stream="future")
    d = deterministic_sample_seed("scene:other", 123, stream="history")
    assert a == b
    assert a != c
    assert a != d
    assert 0 <= a < 2**31 - 1


def test_model_ema_uses_fp32_ramped_shadow():
    model = nn.Linear(3, 2).to(dtype=torch.float32)
    ema = ModelEMA(model, decay=0.999)
    with torch.no_grad():
        model.weight.add_(1.0)
    d = ema.update(model)
    assert 0.0 < d < 0.999
    assert ema.updates == 1
    assert ema.model.weight.dtype == torch.float32
    assert torch.isfinite(ema.model.weight).all()
