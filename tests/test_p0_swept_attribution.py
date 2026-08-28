import numpy as np
import torch

from tools.real_motion.p0_swept_oracle_scan import _accumulate_support_hits


def _state():
    return {
        "gt_total": np.zeros(2, dtype=np.float64),
        "moving_hit": np.zeros(2, dtype=np.float64),
        "uncertain_hit_only": np.zeros(2, dtype=np.float64),
        "missed": np.zeros(2, dtype=np.float64),
    }


def test_support_hit_attribution_expands_bev_over_z_and_partitions_voxels():
    gt = torch.zeros((2, 3, 3, 2), dtype=torch.bool)
    # horizon 0: two GT voxels at same XY but different Z
    gt[0, 1, 1, 0] = True
    gt[0, 1, 1, 1] = True
    # horizon 1: one moving-covered, one uncertain-only, one missed
    gt[1, 0, 0, 0] = True
    gt[1, 1, 1, 0] = True
    gt[1, 2, 2, 1] = True

    moving = torch.zeros((2, 3, 3), dtype=torch.bool)
    uncertain = torch.zeros((2, 3, 3), dtype=torch.bool)
    moving[0, 1, 1] = True
    moving[1, 0, 0] = True
    uncertain[1, 1, 1] = True

    st = _state()
    _accumulate_support_hits(st, gt, moving, uncertain)

    assert st["gt_total"].tolist() == [2.0, 3.0]
    assert st["moving_hit"].tolist() == [2.0, 1.0]
    assert st["uncertain_hit_only"].tolist() == [0.0, 1.0]
    assert st["missed"].tolist() == [0.0, 1.0]
    assert np.array_equal(
        st["gt_total"],
        st["moving_hit"] + st["uncertain_hit_only"] + st["missed"],
    )


def test_support_hit_attribution_rejects_bev_gt_mask():
    st = _state()
    gt = torch.zeros((2, 3, 3), dtype=torch.bool)
    support = torch.zeros((2, 3, 3), dtype=torch.bool)
    try:
        _accumulate_support_hits(st, gt, support, support)
    except ValueError as exc:
        assert "[F,X,Y,Z]" in str(exc)
    else:
        raise AssertionError("expected shape validation to reject BEV gt support")
