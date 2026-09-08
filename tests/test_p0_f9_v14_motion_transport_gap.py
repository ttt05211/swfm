import numpy as np

from real_motion.motion_transport_gap import (
    displacement_errors,
    interpolate_displacement,
    learned_winner_mask,
    masked_error_stats,
    true_moving_observation_mask,
)


def test_true_moving_mask_matches_half_meter_per_second_contract():
    # H=3 at 0.5, 1.0, 1.5 seconds. Threshold displacement is 0.25,0.5,0.75 m.
    disp = np.asarray([[[0.24, 0.0], [0.50, 0.0], [0.76, 0.0]]], dtype=np.float32)
    valid = np.ones((1, 3), dtype=bool)
    got = true_moving_observation_mask(disp, valid, frame_dt_s=0.5, speed_threshold_mps=0.5)
    assert got.tolist() == [[False, True, True]]


def test_invalid_future_is_not_true_moving():
    disp = np.asarray([[[10.0, 0.0]]], dtype=np.float32)
    valid = np.asarray([[False]])
    got = true_moving_observation_mask(disp, valid)
    assert not bool(got[0, 0])


def test_selector_uses_learned_only_when_strictly_closer():
    target = np.asarray([[[2.0, 0.0], [1.0, 0.0], [1.0, 0.0]]])
    pred = np.asarray([[[1.5, 0.0], [2.0, 0.0], [0.0, 0.0]]])
    valid = np.ones((1, 3), dtype=bool)
    # h0: KTA error=2, learned=.5 => learned; h1 KTA=1, learned=1 tie => KTA;
    # h2 KTA=1, learned=1 tie => KTA.
    got = learned_winner_mask(pred, target, valid)
    assert got.tolist() == [[True, False, False]]


def test_error_stats_report_reduction_and_win_fraction():
    target = np.asarray([[[2.0, 0.0], [1.0, 0.0]]])
    pred = np.asarray([[[1.0, 0.0], [1.5, 0.0]]])
    kta, learned = displacement_errors(pred, target)
    stats = masked_error_stats(kta, learned, np.ones((1, 2), dtype=bool))
    assert np.isclose(stats["kta_ade_m"], 1.5)
    assert np.isclose(stats["learned_ade_m"], 0.75)
    assert np.isclose(stats["relative_ade_reduction"], 0.5)
    assert np.isclose(stats["learned_win_fraction"], 1.0)


def test_interpolation_endpoints_and_midpoint():
    kta = np.asarray([[0.0, 0.0]])
    gt = np.asarray([[4.0, 2.0]])
    assert np.allclose(interpolate_displacement(kta, gt, 0.0), kta)
    assert np.allclose(interpolate_displacement(kta, gt, 1.0), gt)
    assert np.allclose(interpolate_displacement(kta, gt, 0.5), [[2.0, 1.0]])
