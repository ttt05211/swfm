import torch
import torch.nn as nn

from real_motion.models.anchor_cfm_f7 import InnovationWeightedAnchorWindowCFM
from tools.real_motion.train_p0_f7_innovation_weighted_wm import _build_optimizer


def test_innovation_weights_alpha_zero_is_exact_uniform():
    target = torch.randn(2, 3, 4, 5, 5)
    weight, info = InnovationWeightedAnchorWindowCFM.innovation_weights(target, alpha=0.0)
    assert weight.shape == (2, 3, 1, 5, 5)
    assert torch.equal(weight, torch.ones_like(weight))
    assert info["innovation_weight_mean"] == 1.0


def test_innovation_weights_are_unit_mean_and_emphasize_high_energy_cells():
    target = torch.zeros(1, 2, 4, 3, 3)
    target[:, :, :, 1, 1] = 10.0
    weight, info = InnovationWeightedAnchorWindowCFM.innovation_weights(target, alpha=4.0)
    assert torch.allclose(weight.mean(), torch.tensor(1.0), atol=1e-6)
    assert weight[0, 0, 0, 1, 1] > weight[0, 0, 0, 0, 0]
    assert info["innovation_weight_max"] > 1.0
    assert info["innovation_weight_min"] < 1.0


def test_zero_innovation_stays_neutral_even_with_positive_alpha():
    target = torch.zeros(2, 3, 4, 5, 5)
    weight, info = InnovationWeightedAnchorWindowCFM.innovation_weights(target, alpha=4.0)
    assert torch.equal(weight, torch.ones_like(weight))
    assert info["innovation_focus_mean"] == 0.0


class _FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.transition = nn.Module()
        self.transition.loaded = nn.Linear(3, 3)
        self.transition.new = nn.Linear(3, 2)


def test_optimizer_uses_small_lr_only_for_upstream_loaded_parameters():
    model = _FakeModel()
    reuse = {
        "loaded_keys": ["loaded.weight", "loaded.bias"],
    }
    optimizer, summary = _build_optimizer(
        model,
        reuse,
        lr=2e-5,
        backbone_lr_scale=0.1,
        weight_decay=1e-2,
    )
    lrs = sorted(group["lr"] for group in optimizer.param_groups)
    assert lrs == [2e-6, 2e-5]
    assert summary["num_pretrained_tensors"] == 2
    assert summary["num_new_or_unloaded_tensors"] == 2
    assert set(summary["new_or_unloaded_names"]) == {
        "transition.new.weight",
        "transition.new.bias",
    }
