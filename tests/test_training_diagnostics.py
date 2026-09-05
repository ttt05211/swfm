import math

import numpy as np
import torch

from real_motion.training_diagnostics import (
    class_histogram,
    enrichment_ratio,
    gradient_pair_stats,
    jensen_shannon_divergence,
    summarize_class_histogram,
)


def test_class_histogram_and_summary():
    labels = np.array([0, 2, 2, 4, 17, 17, 255], dtype=np.int64)
    hist = class_histogram(labels)
    assert hist.shape == (18,)
    assert hist[0] == 1
    assert hist[2] == 2
    assert hist[4] == 1
    assert hist[17] == 2
    assert hist.sum() == 6

    summary = summarize_class_histogram(hist)
    assert summary["total_voxels"] == 6
    assert summary["occupied_voxels"] == 4
    assert math.isclose(summary["occupied_fraction"], 4 / 6)
    # bicycle + car are both dynamic under the frozen Occ3D mapping.
    assert summary["dynamic_voxels"] == 3
    assert math.isclose(summary["dynamic_fraction_occupied_only"], 3 / 4)


def test_distribution_metrics_identity_and_enrichment():
    ref = np.array([1, 3, 6], dtype=np.int64)
    same = np.array([2, 6, 12], dtype=np.int64)
    assert math.isclose(jensen_shannon_divergence(ref, same), 0.0, abs_tol=1e-12)
    enrich = enrichment_ratio(same, ref)
    assert np.allclose(enrich, np.ones(3))

    shifted = np.array([5, 3, 2], dtype=np.int64)
    assert jensen_shannon_divergence(ref, shifted) > 0


def test_gradient_pair_stats_alignment_and_conflict():
    a = [torch.tensor([1.0, 2.0]), torch.tensor([3.0])]
    b_same = [torch.tensor([2.0, 4.0]), torch.tensor([6.0])]
    row = gradient_pair_stats(a, b_same)
    assert math.isclose(row["cosine"], 1.0, rel_tol=1e-6)
    assert row["opposite_sign_elements"] == 0
    assert math.isclose(row["opposite_sign_fraction_on_joint_nonzero"], 0.0)

    b_opp = [torch.tensor([-1.0, 2.0]), torch.tensor([-3.0])]
    row = gradient_pair_stats(a, b_opp)
    assert row["joint_nonzero_elements"] == 3
    assert row["opposite_sign_elements"] == 2
    assert math.isclose(row["opposite_sign_fraction_on_joint_nonzero"], 2 / 3)


def test_gradient_pair_stats_handles_unused_gradient():
    a = [torch.tensor([1.0, 0.0]), None]
    b = [None, torch.tensor([2.0])]
    row = gradient_pair_stats(a, b)
    assert math.isclose(row["norm_a"], 1.0)
    assert math.isclose(row["norm_b"], 2.0)
    assert math.isclose(row["dot"], 0.0)
    assert math.isclose(row["cosine"], 0.0)
