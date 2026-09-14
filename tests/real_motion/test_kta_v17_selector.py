from __future__ import annotations

import numpy as np
import torch

from real_motion.kta_v17_selector import (
    SELECTOR_FEATURE_DIM,
    KtaV17Selector,
    random_fraction_mask,
    selector_features,
    top_fraction_mask,
    utility_pairwise_ranking_loss,
)
from real_motion.motion_transport import FEATURE_DIM, FUTURE_FRAMES, HISTORY_FRAMES


def _record(n=4):
    return {
        "features": torch.arange(n * FEATURE_DIM, dtype=torch.float32).reshape(n, FEATURE_DIM) / 100.0,
        "frame_motion_features": torch.arange(n * HISTORY_FRAMES * 5, dtype=torch.float32).reshape(n, HISTORY_FRAMES, 5) / 10.0,
        "kta_displacement_xy_m": torch.arange(n * FUTURE_FRAMES * 2, dtype=torch.float32).reshape(n, FUTURE_FRAMES, 2),
    }


def test_selector_features_are_fixed_shape_and_causal_record_only():
    r = _record(3)
    x = selector_features(r)
    assert x.shape == (3, SELECTOR_FEATURE_DIM)
    # The helper has no future-label argument; changing unrelated future-like
    # keys cannot alter the selector input.
    r["gt_utility_pct"] = torch.tensor([99.0, -99.0, 7.0])
    r["target_residual_xy_m"] = torch.randn(3, 6, 2)
    y = selector_features(r)
    assert torch.equal(x, y)


def test_selector_features_support_zero_source_windows():
    r = _record(0)
    x = selector_features(r)
    assert x.shape == (0, SELECTOR_FEATURE_DIM)
    assert x.dtype == torch.float32


def test_top_fraction_mask_is_deterministic_and_source_order_tie_broken():
    s = np.asarray([1.0, 3.0, 3.0, 2.0])
    assert top_fraction_mask(s, 0.0).tolist() == [False, False, False, False]
    assert top_fraction_mask(s, 1.0).tolist() == [True, True, True, True]
    # ceil(0.5 * 4)=2; tied sources 1/2 are resolved by original source order.
    assert top_fraction_mask(s, 0.5).tolist() == [False, True, True, False]
    tied = np.ones(4)
    assert top_fraction_mask(tied, 0.5).tolist() == [True, True, False, False]


def test_random_fraction_mask_reproducible_per_sample():
    a = random_fraction_mask(10, 0.2, 123, sample_id="scene:a")
    b = random_fraction_mask(10, 0.2, 123, sample_id="scene:a")
    c = random_fraction_mask(10, 0.2, 124, sample_id="scene:a")
    assert np.array_equal(a, b)
    assert int(a.sum()) == 2
    assert not np.array_equal(a, c)


def test_selector_forward_shape():
    model = KtaV17Selector(SELECTOR_FEATURE_DIM, hidden_dim=32)
    out = model(torch.zeros(5, SELECTOR_FEATURE_DIM))
    assert out.shape == (5,)


def test_utility_pairwise_ranking_prefers_positive_over_zero_over_negative():
    utility = torch.tensor([2.0, 0.5, 0.0, -0.25, -2.0])

    good = torch.tensor([3.0, 2.0, 0.0, -1.0, -3.0], requires_grad=True)
    bad = torch.tensor([-3.0, -2.0, 0.0, 1.0, 3.0], requires_grad=True)

    good_loss, good_stats = utility_pairwise_ranking_loss(good, utility)
    bad_loss, _ = utility_pairwise_ranking_loss(bad, utility)

    assert good_loss.item() < bad_loss.item()
    assert good_stats["positive_negative_pairs"] == 4
    assert good_stats["positive_zero_pairs"] == 2
    assert good_stats["zero_negative_pairs"] == 2

    good_loss.backward()
    assert good.grad is not None
    assert torch.isfinite(good.grad).all()


def test_utility_pairwise_ranking_handles_all_zero_window():
    scores = torch.randn(6, requires_grad=True)
    utility = torch.zeros(6)
    loss, stats = utility_pairwise_ranking_loss(scores, utility)

    assert loss.item() == 0.0
    assert stats["positive_negative_pairs"] == 0
    assert stats["positive_zero_pairs"] == 0
    assert stats["zero_negative_pairs"] == 0

    loss.backward()
    assert scores.grad is not None
    assert torch.equal(scores.grad, torch.zeros_like(scores))
