import numpy as np
import torch

from real_motion.motion_transport_weighting import (
    MODE_MOTION,
    MODE_MOTION_CLASS,
    compute_macro_class_weights,
    make_observation_weights,
    true_moving_mask_torch,
    weighted_motion_transport_loss,
)


def test_true_motion_mask_uses_exact_half_meter_per_second_rule():
    # Horizons are 0.5, 1.0, 1.5 s, so thresholds are 0.25, 0.5, 0.75 m.
    disp = torch.tensor([[[0.24, 0.0], [0.50, 0.0], [0.76, 0.0]]])
    valid = torch.ones((1, 3), dtype=torch.bool)
    got = true_moving_mask_torch(disp, valid)
    assert got.tolist() == [[False, True, True]]


def test_motion_weight_two_makes_true_motion_half_effective_mass_near_one_third_raw():
    cls = torch.tensor([4, 4, 4])
    valid = torch.ones((3, 1), dtype=torch.bool)
    moving = torch.tensor([[True], [False], [False]])
    w = make_observation_weights(cls, moving, valid, mode=MODE_MOTION, motion_weight=2.0)
    assert torch.equal(w[:, 0], torch.tensor([2.0, 1.0, 1.0]))
    assert np.isclose(float(w[moving].sum() / w[valid].sum()), 0.5)


def test_macro_class_weights_equalize_true_moving_mass_without_changing_total():
    # class 4 has 3 moving observations; class 7 has 1.  Equal macro mass =>
    # factors 4/(2*3)=2/3 and 4/(2*1)=2.
    cls = torch.tensor([4, 4, 4, 7])
    moving = torch.ones((4, 1), dtype=torch.bool)
    valid = moving.clone()
    cw = compute_macro_class_weights(cls, moving)
    assert np.isclose(cw[4], 2.0 / 3.0)
    assert np.isclose(cw[7], 2.0)
    w = make_observation_weights(
        cls, moving, valid, mode=MODE_MOTION_CLASS, class_weights=cw, motion_weight=2.0
    )
    # Motion factor contributes total mass 2*N=8 both before and after class redistribution.
    assert np.isclose(float(w.sum()), 8.0)
    assert np.isclose(float(w[cls == 4].sum()), 4.0)
    assert np.isclose(float(w[cls == 7].sum()), 4.0)


def test_weighted_loss_matches_unweighted_when_all_weights_one():
    pred = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
    out = {
        "residual_xy_m": pred,
        "existence_logits": torch.zeros((1, 2)),
    }
    batch = {
        "target_residual_xy_m": torch.zeros_like(pred),
        "target_valid": torch.ones((1, 2), dtype=torch.bool),
        "existence": torch.ones((1, 2)),
        "supervised_source": torch.ones((1,), dtype=torch.bool),
    }
    loss, parts = weighted_motion_transport_loss(out, batch, torch.ones((1, 2)))
    # SmoothL1 beta=1: [1,0] -> element losses [.5,0], mean over xy=.25;
    # second horizon is zero, so observation mean=.125.
    assert np.isclose(parts["trajectory_smooth_l1_weighted"], 0.125)
    assert torch.isfinite(loss)


def test_invalid_observation_has_zero_weight():
    cls = torch.tensor([4])
    moving = torch.tensor([[True, False]])
    valid = torch.tensor([[False, True]])
    w = make_observation_weights(cls, moving, valid, mode=MODE_MOTION)
    assert w.tolist() == [[0.0, 1.0]]
