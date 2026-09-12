import math

import torch

from tools.real_motion.train_p0_f9_v17_scene_supervision_c0 import (
    _lr_scale,
    alpha_ramp,
    enable_all_parameters,
    preserve_rng_state,
)


def test_c0_alpha_ramp_contract():
    assert alpha_ramp(1, 120) == 0.0
    assert math.isclose(alpha_ramp(120, 120), 1.0)
    assert alpha_ramp(600, 120) == 1.0
    assert alpha_ramp(1, 1) == 1.0


def test_c0_enables_all_model_parameters():
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.LayerNorm(4), torch.nn.Linear(4, 2))
    for p in model[0].parameters():
        p.requires_grad_(False)
    report = enable_all_parameters(model)
    assert report["frozen_parameters"] == 0
    assert report["trainable_parameters"] == sum(p.numel() for p in model.parameters())
    assert all(p.requires_grad for p in model.parameters())


def test_c0_scene_rng_isolation_preserves_source_rng_path():
    device = torch.device("cpu")
    torch.manual_seed(123)
    first = torch.rand(5)
    with preserve_rng_state(device):
        _ = torch.rand(100)
    after = torch.rand(5)

    torch.manual_seed(123)
    expected_first = torch.rand(5)
    expected_after = torch.rand(5)
    assert torch.equal(first, expected_first)
    assert torch.equal(after, expected_after)


def test_c0_lr_schedule_matches_historical_cosine_contract():
    assert math.isclose(_lr_scale(0, 100), 1.0)
    assert math.isclose(_lr_scale(100, 100), 0.1)
    assert 0.1 < _lr_scale(50, 100) < 1.0
