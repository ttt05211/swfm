import math

import torch
import torch.nn as nn

from tools.real_motion.p0_f8_train_impl import calibrate_edit_lambda


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
