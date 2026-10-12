"""Numerical checks for Shared-History execution-only speed candidates."""
import copy

import numpy as np
import pytest
import torch

from tools.shared_history_experiments import engine as reference
from tools.shared_history_experiments.speed.engine import (
    WideGroupedScene,
    forward_batch,
    train_arm_grouped,
    train_arm_packed,
)
from tools.joint_occ_experiments.batching.geometry import GroupedScene, grouped_loss
from tools.joint_occ_experiments.fast_autograd import RenderGradientBridge
from test_shared_history_pilot import fixture


def _optimizer(model):
    return torch.optim.AdamW([
        dict(params=model.transport.parameters(), lr=5e-4, initial_lr=5e-4),
        dict(params=model.columns.parameters(), lr=3e-4, initial_lr=3e-4),
        dict(params=model.ego.parameters(), lr=3e-4, initial_lr=3e-4),
    ])


def _assert_model_close(a, b, *, rtol=2e-4, atol=2e-5):
    left, right = a.state_dict(), b.state_dict()
    assert left.keys() == right.keys()
    for key in left:
        torch.testing.assert_close(left[key], right[key], rtol=rtol, atol=atol)


def test_wide_h6_renderer_matches_three_h2_groups_and_boundary_gradients():
    model, _, provider, row = fixture()
    model.eval()
    fw = reference.forward_window(model, provider, row)
    labels = np.asarray(row[1]["future_gt_occ"])

    def execute(wide):
        inputs = {
            key: fw[key].detach().clone().requires_grad_(fw[key].requires_grad)
            for key in ("poses", "targets", "yaws", "mapped", "actions")
        }
        scene = (WideGroupedScene if wide else GroupedScene)(
            fw["prep"], provider.pcfg.grid, provider.device
        )
        if wide:
            groups = [list(range(6))]
        else:
            groups = ([0, 1], [2, 3], [4, 5])
        probabilities, losses = [], []
        for hs in groups:
            probability = scene.render_group(
                hs,
                **inputs,
                evidence=fw["evidence"],
                plan=fw["plan"],
            )
            loss, _, _ = grouped_loss(probability, labels[hs])
            probabilities.append(probability)
            losses.append(loss)
        probability = torch.cat(probabilities, 0)
        loss = torch.cat(losses, 0)
        scalar = loss.sum()
        active = [inputs[key] for key in inputs if inputs[key].requires_grad]
        gradients = torch.autograd.grad(scalar, active, allow_unused=True)
        return probability, loss, gradients

    expected = execute(False)
    actual = execute(True)
    torch.testing.assert_close(actual[0], expected[0], rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(actual[1], expected[1], rtol=3e-6, atol=3e-6)
    assert len(actual[2]) == len(expected[2])
    for a, b in zip(actual[2], expected[2]):
        if a is None or b is None:
            assert a is b
        else:
            torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)


def test_packed_forward_matches_two_independent_shared_forwards():
    model, _, provider, row = fixture()
    model.eval()
    expected = [
        reference.forward_window(model, provider, row),
        reference.forward_window(model, provider, row),
    ]
    actual = forward_batch(model, provider, [row, row])
    assert len(actual) == 2
    for a, b in zip(actual, expected):
        for key in ("ego", "targets", "yaws", "mapped", "actions", "field"):
            torch.testing.assert_close(a[key], b[key], rtol=2e-5, atol=2e-6)
        for key in (
            "residual_xy_m",
            "yaw_delta_rad",
            "existence_logits",
            "history_source_context",
            "future_transport_queries",
        ):
            torch.testing.assert_close(
                a["output"][key], b["output"][key], rtol=2e-5, atol=2e-6
            )


@pytest.mark.parametrize("horizon_group", [3, 6])
def test_sequential_h3_h6_one_step_matches_current_h2(horizon_group):
    model, _, provider, row = fixture()
    baseline = copy.deepcopy(model)
    candidate = copy.deepcopy(model)
    opt_a, opt_b = _optimizer(baseline), _optimizer(candidate)

    rng = torch.get_rng_state()
    stats_a = reference.train_arm(
        baseline, opt_a, provider, [row, row], "gt",
        update=1, total_updates=4, probe=True,
    )
    torch.set_rng_state(rng)
    stats_b = train_arm_grouped(
        candidate, opt_b, provider, [row, row], "gt",
        update=1, total_updates=4, probe=True, horizon_group=horizon_group,
    )

    for key in ("CE", "Lovasz", "source_geometry", "existence", "ego_geometry"):
        np.testing.assert_allclose(stats_b[key], stats_a[key], rtol=2e-5, atol=2e-6)
    for key in stats_a["gradient_links"]:
        np.testing.assert_allclose(
            stats_b["gradient_links"][key],
            stats_a["gradient_links"][key],
            rtol=2e-4,
            atol=2e-5,
        )
    _assert_model_close(candidate, baseline)


def test_packed_h2_one_step_matches_current_h2_with_same_global_batch():
    model, _, provider, row = fixture()
    baseline = copy.deepcopy(model)
    candidate = copy.deepcopy(model)
    opt_a, opt_b = _optimizer(baseline), _optimizer(candidate)

    rng = torch.get_rng_state()
    stats_a = reference.train_arm(
        baseline, opt_a, provider, [row, row], "gt",
        update=1, total_updates=4, probe=True,
    )
    torch.set_rng_state(rng)
    stats_b = train_arm_packed(
        candidate, opt_b, provider, [row, row], "gt",
        update=1, total_updates=4, probe=True, horizon_group=2,
    )

    for key in ("CE", "Lovasz", "source_geometry", "existence", "ego_geometry"):
        np.testing.assert_allclose(stats_b[key], stats_a[key], rtol=2e-4, atol=2e-5)
    for key in stats_a["gradient_links"]:
        np.testing.assert_allclose(
            stats_b["gradient_links"][key],
            stats_a["gradient_links"][key],
            rtol=5e-4,
            atol=5e-5,
        )
    _assert_model_close(candidate, baseline, rtol=5e-4, atol=5e-5)
