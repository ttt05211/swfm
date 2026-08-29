import torch

from real_motion.msp_window import (
    plan_topk_score_windows,
    score_capture_ratio,
    window_plan_support,
)


def test_one_window_selects_score_hotspot_and_has_exact_area():
    score = torch.zeros((1, 6, 10, 10), dtype=torch.float32)
    score[:, :, 2:4, 3:5] = 1.0
    plan = plan_topk_score_windows(score, window_hw=(4, 4), max_windows=1)
    assert int(plan.valid.sum()) == 1
    support = window_plan_support(plan)
    assert support.shape == (1, 10, 10)
    assert int(support.sum()) == 16
    assert bool(support[0, 2:4, 3:5].all())
    capture = score_capture_ratio(score, plan)
    assert torch.allclose(capture, torch.ones_like(capture))


def test_second_window_uses_marginal_score_for_distant_hotspot():
    score = torch.zeros((1, 2, 12, 12), dtype=torch.float32)
    score[:, :, 1:3, 1:3] = 3.0
    score[:, :, 9:11, 9:11] = 2.0
    plan = plan_topk_score_windows(score, window_hw=(4, 4), max_windows=2)
    assert int(plan.valid.sum()) == 2
    support = window_plan_support(plan)
    assert bool(support[0, 1:3, 1:3].all())
    assert bool(support[0, 9:11, 9:11].all())
    assert float(score_capture_ratio(score, plan)[0]) == 1.0


def test_horizon_scores_are_aggregated_into_one_shared_plan():
    score = torch.zeros((1, 3, 10, 10), dtype=torch.float32)
    score[0, 0, 1, 1] = 2.0
    score[0, 1, 1, 1] = 2.0
    score[0, 2, 8, 8] = 3.0
    plan = plan_topk_score_windows(score, window_hw=(4, 4), max_windows=1)
    support = window_plan_support(plan)
    # Summed score at (1,1) is 4, so the shared window should prefer it over
    # the single-horizon score of 3 at (8,8).
    assert bool(support[0, 1, 1])
    assert not bool(support[0, 8, 8])


def test_zero_score_map_creates_no_fake_windows():
    score = torch.zeros((2, 6, 10, 10), dtype=torch.float32)
    plan = plan_topk_score_windows(score, window_hw=(4, 4), max_windows=3)
    assert not plan.valid.any()
    assert not window_plan_support(plan).any()
    assert torch.allclose(score_capture_ratio(score, plan), torch.ones(2))


def test_invalid_scores_fail_closed():
    neg = torch.zeros((1, 2, 8, 8), dtype=torch.float32)
    neg[0, 0, 0, 0] = -1.0
    try:
        plan_topk_score_windows(neg, window_hw=(4, 4), max_windows=1)
    except ValueError as e:
        assert "non-negative" in str(e)
    else:
        raise AssertionError("negative MSP scores must fail closed")
