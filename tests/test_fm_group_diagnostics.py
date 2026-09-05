import math

import torch

from real_motion.fm_group_diagnostics import (
    finalize_grouped_velocity_accumulator,
    motion_support_occ_to_latent,
    new_grouped_velocity_accumulator,
    update_grouped_velocity_accumulator,
)


def test_motion_support_occ_to_latent_exact_any_pool():
    support = torch.zeros(2, 8, 8, 3, dtype=torch.bool)
    support[0, 1, 2, 0] = True
    support[0, 7, 7, 2] = True
    support[1, 4, 0, 1] = True

    latent = motion_support_occ_to_latent(support, latent_hw=(2, 2))
    assert latent.shape == (2, 2, 2)
    assert torch.equal(
        latent,
        torch.tensor(
            [
                [[True, False], [False, True]],
                [[False, False], [True, False]],
            ],
            dtype=torch.bool,
        ),
    )


def test_grouped_mse_recomposes_global_exactly():
    # [N=1,T=2,C=2,H=1,W=2]
    target = torch.zeros(1, 2, 2, 1, 2)
    pred = torch.tensor(
        [
            [
                [[[1.0, 2.0]], [[3.0, 4.0]]],
                [[[5.0, 6.0]], [[7.0, 8.0]]],
            ]
        ]
    )
    mask = torch.tensor([[[[True, False]], [[False, True]]]])

    acc = new_grouped_velocity_accumulator(num_frames=2)
    update_grouped_velocity_accumulator(acc, pred, target, mask)
    report = finalize_grouped_velocity_accumulator(acc, motion_weight_lambda=2.0)

    expected_global = float(pred.square().mean())
    assert math.isclose(report["overall"]["global_mse"], expected_global, abs_tol=1e-12)
    assert report["overall"]["recomposition_abs_error"] <= 1e-12
    assert report["overall"]["moving"]["elements"] == 4
    assert report["overall"]["non_moving"]["elements"] == 4
    assert math.isclose(report["overall"]["motion_cell_fraction"], 0.5, abs_tol=1e-12)
    assert math.isclose(
        report["overall"]["effective_motion_weight_mass"], 0.75, abs_tol=1e-12
    )


def test_empty_motion_group_is_not_reported_as_zero_error():
    pred = torch.ones(1, 1, 1, 2, 2)
    target = torch.zeros_like(pred)
    mask = torch.zeros(1, 1, 2, 2, dtype=torch.bool)

    acc = new_grouped_velocity_accumulator(num_frames=1)
    update_grouped_velocity_accumulator(acc, pred, target, mask)
    report = finalize_grouped_velocity_accumulator(acc)

    moving = report["overall"]["moving"]
    assert moving["elements"] == 0
    assert moving["mse"] is None
    assert moving["nmse"] is None
    assert moving["cosine_macro"] is None
    assert moving["empty_groups"] == 1
