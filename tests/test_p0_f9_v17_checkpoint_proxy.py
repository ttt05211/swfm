import math

import torch

from real_motion.local_st_world_model_v17 import soft_transport_overlap_loss
from tools.real_motion.diagnose_p0_f9_v17_checkpoint_proxy import (
    _corr,
    _rankdata,
    _transport_iou_per_label,
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


def test_proxy_soft_iou_matches_training_overlap_contract():
    pred = torch.zeros(2, 6, 2)
    target = torch.zeros_like(pred)
    pred[0, :, 0] = 0.8
    pred[1, :, 1] = -0.4
    mask = torch.zeros(2, 20, 20)
    mask[0, 8:12, 8:12] = 1.0
    mask[1, 9:12, 7:13] = 1.0
    valid = torch.ones(2, 6, dtype=torch.bool)

    _, stats = soft_transport_overlap_loss(
        pred,
        target,
        mask,
        valid,
        patch_resolution_m=0.8,
    )
    soft, _, present = _transport_iou_per_label(
        pred - target,
        mask,
        patch_resolution_m=0.8,
    )
    usable = valid & present[:, None]
    assert int(usable.sum()) == int(stats["transport_overlap_labels"])
    assert math.isclose(
        float(soft[usable].mean()),
        float(stats["transport_soft_iou"]),
        rel_tol=0.0,
        abs_tol=1e-6,
    )
