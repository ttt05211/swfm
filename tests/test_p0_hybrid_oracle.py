import numpy as np
import torch

from tools.real_motion.p0_hybrid_oracle_scan import (
    _build_moving_support,
    _partition_hits,
)


def test_hybrid_is_union_of_endpoint_and_swept_support():
    endpoint = torch.zeros((2, 7, 7), dtype=torch.bool)
    swept = torch.zeros_like(endpoint)
    endpoint[:, 5, 5] = True
    swept[:, 1:6, 1] = True
    cfg = {
        "geometry": "hybrid",
        "endpoint_radii": [1, 1],
        "swept_radii": [0, 0],
        "uncertain_radii": [0, 0],
    }
    out = _build_moving_support(endpoint, swept, cfg)
    # endpoint keeps its local cap and corridor remains a separate thin path.
    assert out[:, 4:7, 4:7].all()
    assert out[:, 1:6, 1].all()
    # The hybrid must not accidentally fill the rectangle between them.
    assert not out[:, 3, 3].any()


def test_future_arrival_partition_counts_only_actual_gt_voxels():
    gt = torch.zeros((2, 4, 4, 2), dtype=torch.bool)
    gt[0, 3, 3, 0] = True
    gt[1, 2, 2, 1] = True
    moving = torch.zeros((2, 4, 4), dtype=torch.bool)
    uncertain = torch.zeros_like(moving)
    moving[0, 3, 3] = True
    uncertain[1, 2, 2] = True

    total, moving_hit, uncertain_only, missed = _partition_hits(gt, moving, uncertain)
    assert total.tolist() == [1.0, 1.0]
    assert moving_hit.tolist() == [1.0, 0.0]
    assert uncertain_only.tolist() == [0.0, 1.0]
    assert missed.tolist() == [0.0, 0.0]
    assert np.array_equal(total, moving_hit + uncertain_only + missed)


def test_partition_rejects_bev_gt_mask():
    gt = torch.zeros((2, 4, 4), dtype=torch.bool)
    support = torch.zeros_like(gt)
    try:
        _partition_hits(gt, support, support)
    except ValueError as exc:
        assert "[F,X,Y,Z]" in str(exc)
    else:
        raise AssertionError("expected a shape validation error")
