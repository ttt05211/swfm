from __future__ import annotations

import numpy as np

from real_motion.local_stwm_gap import (
    kta_error_components,
    pairwise_winner,
    residual_error_components,
    summarize_model_components,
    three_way_winner,
    weighted_masked_mean,
)


def test_along_cross_track_decomposition():
    # GT motion points along +x. Target residual says KTA is 2 m short in x.
    gt_disp = np.asarray([[[4.0, 0.0]]], dtype=np.float32)
    target_res = np.asarray([[[2.0, 0.0]]], dtype=np.float32)

    kta = kta_error_components(target_res, gt_disp)
    assert np.allclose(kta["euclidean_m"], [[2.0]])
    assert np.allclose(kta["along_abs_m"], [[2.0]])
    assert np.allclose(kta["cross_abs_m"], [[0.0]])

    # Learned residual fixes 1.5 m along track but introduces 0.5 m lateral error.
    pred = np.asarray([[[1.5, 0.5]]], dtype=np.float32)
    learned = residual_error_components(pred, target_res, gt_disp)
    assert np.allclose(learned["euclidean_m"], [[np.sqrt(0.5)]])
    assert np.allclose(learned["along_abs_m"], [[0.5]])
    assert np.allclose(learned["cross_abs_m"], [[0.5]])


def test_voxel_weighted_mean_exposes_large_source_error():
    values = np.asarray([[1.0], [3.0]], dtype=np.float64)
    weights = np.asarray([1.0, 9.0], dtype=np.float64)
    mask = np.ones((2, 1), dtype=bool)
    assert np.isclose(weighted_masked_mean(values, weights, mask), 2.8)


def test_three_way_and_pairwise_winners_are_conservative_on_ties():
    kta = np.asarray([[1.0, 2.0, 1.0]])
    mlp = np.asarray([[0.8, 2.0, 1.2]])
    stwm = np.asarray([[0.9, 1.5, 0.7]])
    valid = np.asarray([[True, True, False]])
    three = three_way_winner(kta, mlp, stwm, valid)
    assert three.tolist() == [[1, 2, -1]]

    pair = pairwise_winner(mlp, stwm, valid)
    assert pair.tolist() == [[0, 1, -1]]


def test_summary_reports_weighted_and_directional_metrics():
    pred = np.asarray([[[1.0, 0.0]], [[0.0, 2.0]]], dtype=np.float64)
    target = np.zeros_like(pred)
    gt_disp = np.asarray([[[4.0, 0.0]], [[0.0, 4.0]]], dtype=np.float64)
    comp = residual_error_components(pred, target, gt_disp)
    mask = np.ones((2, 1), dtype=bool)
    stats = summarize_model_components(comp, mask, np.asarray([1.0, 3.0]))
    assert stats["count"] == 2
    assert np.isclose(stats["ade_m"], 1.5)
    assert np.isclose(stats["voxel_weighted_ade_m"], 1.75)
    assert np.isclose(stats["cross_abs_m"], 0.0)
