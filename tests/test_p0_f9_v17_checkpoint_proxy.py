import math

from tools.real_motion.diagnose_p0_f9_v17_checkpoint_proxy import (
    _corr,
    _rankdata,
    aggregate_proxy_samples,
)


def test_proxy_class_horizon_macro_is_not_micro_average():
    # Group (1s, car) has two samples; (3s, truck) has one.  Macro must give
    # equal weight to the two class/horizon groups, not to the three samples.
    rows = [
        {"horizon": 1.0, "class_id": 4, "soft_iou": 1.0, "hard_iou": 1.0, "error_m": 0.2, "area": 10.0},
        {"horizon": 1.0, "class_id": 4, "soft_iou": 1.0, "hard_iou": 1.0, "error_m": 0.2, "area": 10.0},
        {"horizon": 3.0, "class_id": 10, "soft_iou": 0.0, "hard_iou": 0.0, "error_m": 2.0, "area": 100.0},
    ]
    out = aggregate_proxy_samples(rows)
    assert math.isclose(out["tm_soft_iou_micro"], 2.0 / 3.0)
    assert math.isclose(out["tm_soft_iou_class_horizon_macro"], 0.5)
    assert math.isclose(out["tm_hard_iou_class_horizon_macro"], 0.5)
    assert math.isclose(out["tm_hit_0p4_class_horizon_macro"], 0.5)


def test_rankdata_uses_average_rank_for_ties():
    ranks = _rankdata([3.0, 1.0, 1.0, 2.0])
    assert list(ranks) == [4.0, 1.5, 1.5, 3.0]


def test_spearman_via_rank_correlation_detects_matching_order():
    a = _rankdata([0.2, 0.8, 0.5, 1.0])
    b = _rankdata([2.0, 8.0, 5.0, 10.0])
    assert math.isclose(_corr(a, b), 1.0, rel_tol=0.0, abs_tol=1e-12)
