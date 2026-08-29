import torch
import torch.nn as nn

from real_motion.models.anchor_cfm import AnchorWindowCFM
from real_motion.msp_wm_cache import validate_msp_wm_sample


class ZeroTransition(nn.Module):
    def forward(self, batch):
        batch = dict(batch)
        batch["predicted_latent"] = torch.zeros_like(batch["noised_sequence"])
        return batch


def test_anchor_equal_target_has_zero_flow_loss_with_zero_transition():
    model = AnchorWindowCFM(ZeroTransition(), rescale_factor=10.0, sample_steps=2)
    hist = torch.randn(2, 6, 16, 4, 4)
    anchor = torch.randn(2, 6, 16, 4, 4)
    loss, info = model.flow_loss(
        hist, anchor.clone(), anchor,
        t_override=0.5,
        source_noise=torch.zeros_like(anchor),
    )
    assert float(loss) == 0.0
    assert info["target_rms"] == 0.0


def test_zero_velocity_sampler_preserves_anchor_exactly():
    model = AnchorWindowCFM(ZeroTransition(), rescale_factor=10.0, sample_steps=3)
    hist = torch.randn(1, 6, 16, 4, 4)
    anchor = torch.randn(1, 6, 16, 4, 4)
    out = model.sample(hist, anchor)
    assert torch.equal(out, anchor)


def test_msp_wm_cache_contract_is_top2_and_shape_safe():
    sample = {
        "sample_id": "scene:token",
        "scene_name": "scene",
        "moving_history_latent": torch.zeros(6, 16, 50, 50),
        "anchor_future_latent": torch.zeros(6, 16, 50, 50),
        "gt_future_latent": torch.zeros(6, 16, 50, 50),
        "window_origins": torch.tensor([[0, 0], [20, 20]], dtype=torch.long),
        "window_valid": torch.tensor([True, True]),
        "trajectory": torch.zeros(12, 2),
    }
    assert validate_msp_wm_sample(sample, topk=2)
