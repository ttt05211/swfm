from __future__ import annotations

import numpy as np
import torch

from real_motion.long_tail_sampling import (
    MAX_NORMALIZED_WEIGHT,
    build_balanced_source_weights,
    make_balanced_source_sampler,
    source_motion_difficulty,
)


def test_source_motion_difficulty_is_mean_valid_kta_relative_magnitude():
    residual = torch.tensor([
        [[3.0, 4.0], [0.0, 0.0], [6.0, 8.0]],
        [[1.0, 0.0], [0.0, 2.0], [0.0, 0.0]],
    ])
    valid = torch.tensor([
        [True, False, True],
        [True, True, False],
    ])
    got = source_motion_difficulty(residual, valid)
    assert torch.allclose(got, torch.tensor([7.5, 1.5]))


def test_source_without_valid_motion_target_is_retained_as_nan_difficulty():
    residual = torch.zeros(3, 2, 2)
    residual[0, :, 0] = 0.5
    residual[1, :, 0] = 1.0
    valid = torch.tensor([
        [True, True],
        [False, False],
        [True, False],
    ])
    got = source_motion_difficulty(residual, valid)
    assert torch.isfinite(got[0])
    assert torch.isnan(got[1])
    assert torch.isfinite(got[2])


def test_balanced_weights_raise_tail_class_and_motion_tail_without_special_cases():
    ids = torch.tensor([4] * 12 + [6] * 3, dtype=torch.long)
    residual = torch.zeros(15, 3, 2)
    valid = torch.ones(15, 3, dtype=torch.bool)
    residual[:12, :, 0] = torch.linspace(0.05, 0.6, 12)[:, None]
    residual[12:, :, 0] = torch.tensor([0.10, 0.20, 1.50])[:, None]

    out = build_balanced_source_weights(ids, residual, valid)
    w = out.weights.numpy()
    assert w[12:].mean() > w[:12].mean()
    assert w[14] > w[12]
    assert np.isfinite(w).all()
    assert (w > 0).all()
    assert float(w.max()) <= MAX_NORMALIZED_WEIGHT + 1e-12


def test_motion_unlabeled_source_gets_neutral_motion_factor_but_class_factor_remains():
    ids = torch.tensor([4, 4, 4, 6], dtype=torch.long)
    residual = torch.zeros(4, 2, 2)
    residual[0, :, 0] = 0.1
    residual[1, :, 0] = 0.2
    residual[2, :, 0] = 0.3
    residual[3, :, 0] = 0.8
    valid = torch.tensor([
        [True, True],
        [False, False],
        [True, True],
        [True, True],
    ])
    out = build_balanced_source_weights(ids, residual, valid)
    assert torch.isnan(out.motion_difficulty_m[1])
    assert abs(float(out.motion_weights[1]) - 1.0) < 1e-7
    assert np.isfinite(out.weights.numpy()).all()
    assert out.report["motion_unlabeled_sources"] == 1
    assert out.report["classes"]["4"]["motion_unlabeled_sources"] == 1


def test_balanced_sampler_keeps_exactly_n_draws_per_epoch_and_is_reproducible():
    ids = torch.tensor([2, 2, 4, 4, 4, 6], dtype=torch.long)
    residual = torch.zeros(6, 2, 2)
    residual[:, :, 0] = torch.tensor([0.1, 0.3, 0.1, 0.2, 0.4, 1.0])[:, None]
    valid = torch.ones(6, 2, dtype=torch.bool)
    out = build_balanced_source_weights(ids, residual, valid)

    a = list(make_balanced_source_sampler(out, seed=17))
    b = list(make_balanced_source_sampler(out, seed=17))
    assert len(a) == len(ids)
    assert a == b


def test_report_exposes_raw_and_expected_class_fractions():
    ids = torch.tensor([2, 4, 4, 4, 6, 6], dtype=torch.long)
    residual = torch.zeros(6, 2, 2)
    residual[:, :, 0] = torch.tensor([1.0, 0.1, 0.2, 0.3, 0.8, 1.2])[:, None]
    valid = torch.ones(6, 2, dtype=torch.bool)
    out = build_balanced_source_weights(ids, residual, valid)
    report = out.report

    assert report["sources"] == 6
    assert report["motion_labeled_sources"] == 6
    assert report["motion_unlabeled_sources"] == 0
    assert report["class_counts"] == {"2": 1, "4": 3, "6": 2}
    assert abs(sum(v["expected_sample_fraction"] for v in report["classes"].values()) - 1.0) < 1e-8
    assert "global_top10_motion_threshold_m" in report
    assert "effective_sample_fraction" in report
