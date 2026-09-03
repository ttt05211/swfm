import math

import torch
import torch.nn as nn

from tools.real_motion.p0_f8_train_impl import (
    _build_optimizer,
    assess_all_keep_collapse,
    calibrate_edit_lambda,
)


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.transition = nn.Linear(1, 1, bias=False)
        self.edit_head = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.transition.weight.fill_(1.0)
            self.edit_head.weight.fill_(1.0)


def test_edit_lambda_calibration_uses_shared_transition_not_private_head():
    model = _ToyModel()
    w = model.transition.weight.sum()
    h = model.edit_head.weight.sum()
    fm_loss = w.square()                  # shared grad = 2
    edit_loss = (3.0 * w).square() + (100.0 * h).square()  # shared grad = 18

    lam, info = calibrate_edit_lambda(
        fm_loss,
        edit_loss,
        model,
        target_ratio=0.5,
        lambda_min=1e-6,
        lambda_max=1.0,
    )
    expected = 0.5 * 2.0 / 18.0
    assert math.isclose(lam, expected, rel_tol=1e-6)
    assert info["calibration_parameters"] == "shared_transition_only"
    assert math.isclose(info["realized_shared_grad_ratio"], 0.5, rel_tol=1e-6)
    # The intentionally huge private-head gradient is reported but cannot shrink
    # the WM edit coefficient.
    assert info["edit_head_grad_norm_unweighted"] > 1000.0


def test_optimizer_gives_edit_head_an_independent_learning_rate():
    model = _ToyModel()
    optimizer, summary = _build_optimizer(
        model,
        {"loaded_keys": ["weight"]},
        lr=2e-5,
        edit_head_lr=1e-3,
        backbone_lr_scale=0.1,
        weight_decay=1e-2,
    )
    groups = {group["group_name"]: group for group in optimizer.param_groups}
    assert set(groups) == {
        "pretrained_backbone",
        "edit_head",
    }
    assert math.isclose(groups["pretrained_backbone"]["lr"], 2e-6)
    assert math.isclose(groups["edit_head"]["lr"], 1e-3)
    assert summary["edit_head_names"] == ["edit_head.weight"]
    assert summary["num_edit_head_parameters"] == model.edit_head.weight.numel()


def test_all_keep_collapse_gate_only_stops_when_due_and_zero_edits():
    val = {
        "num_lovasz_voxels": 100,
        "num_pool_predicted_edits": 0,
    }
    assert assess_all_keep_collapse(val, step=100, check_step=200)["status"] == "NOT_DUE"
    failed = assess_all_keep_collapse(val, step=200, check_step=200)
    assert failed["status"] == "FAIL"
    assert failed["stop"] is True
    val["num_pool_predicted_edits"] = 1
    passed = assess_all_keep_collapse(val, step=200, check_step=200)
    assert passed["status"] == "PASS"
    assert passed["stop"] is False
