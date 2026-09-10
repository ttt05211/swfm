import math

import numpy as np
import torch

from real_motion.geometry import OccupancyGrid
from real_motion.source_footprint import native_source_footprint_patch, pool_binary_mask
from tools.real_motion.diagnose_p0_f9_v17_native_footprint import exact_source_masks
from tools.real_motion.train_p0_f9_v17_native_footprint_pair import (
    CONTROL_OVERLAP_WEIGHT,
    NATIVE_OVERLAP_WEIGHT,
    _footprint_contract,
    _lr_scale,
)


def _grid():
    return OccupancyGrid(
        x_min=-2.0,
        y_min=-2.0,
        z_min=0.0,
        voxel_size=(0.4, 0.4, 0.4),
        shape_hwd=(20, 20, 4),
    )


def test_fixed_native_patch_matches_b0_common_coordinate_pooling():
    grid = _grid()
    voxels = np.asarray(
        [
            [4, 4, 0], [4, 5, 1], [5, 4, 0], [5, 5, 2],
            [6, 5, 0], [6, 6, 0],
        ],
        dtype=np.int64,
    )
    center = np.asarray([0.2, 0.2], dtype=np.float64)
    patch, coverage = native_source_footprint_patch(
        voxels, center, grid=grid, patch_size_m=4.0
    )
    _, pooled_b0, coverage_b0 = exact_source_masks(
        voxels,
        center,
        grid=grid,
        patch_size_m=4.0,
        pooled_resolution_m=0.8,
    )
    assert patch.shape == (10, 10)
    assert int(patch.sum()) == len(np.unique(voxels[:, :2], axis=0))
    assert np.array_equal(pool_binary_mask(patch, 2), pooled_b0)
    assert math.isclose(coverage, coverage_b0)


def test_pair_uses_same_legacy_eligibility_for_control_and_native():
    B = 3
    batch = {
        "target_source_mask_tube": torch.zeros(B, 6, 4, 4, dtype=torch.uint8),
        "target_valid": torch.tensor(
            [[1, 1, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0]],
            dtype=torch.bool,
        ),
        "native_source_footprint_mask": torch.zeros(B, 8, 8, dtype=torch.uint8),
    }
    batch["target_source_mask_tube"][0, -1, 1, 1] = 1
    batch["target_source_mask_tube"][2, -1, 1, 1] = 1
    batch["native_source_footprint_mask"][:, 2:4, 2:4] = 1

    legacy_mask, legacy_valid, legacy_res, legacy_w = _footprint_contract(
        batch, "B-C", legacy_resolution_m=0.8, native_resolution_m=0.4
    )
    native_mask, native_valid, native_res, native_w = _footprint_contract(
        batch, "B-S", legacy_resolution_m=0.8, native_resolution_m=0.4
    )
    assert legacy_mask.shape == (B, 4, 4)
    assert native_mask.shape == (B, 8, 8)
    assert torch.equal(legacy_valid, native_valid)
    # Source 1 has a native mask but no legacy t0 support: it must not become a
    # new overlap-supervised source in B-S.
    assert not bool(native_valid[1].any())
    assert legacy_res == 0.8 and native_res == 0.4
    assert legacy_w == CONTROL_OVERLAP_WEIGHT == 0.25
    assert native_w == NATIVE_OVERLAP_WEIGHT == 0.175


def test_remaining_cosine_schedule_is_continuous_at_epoch5_boundary():
    # The original 10-epoch schedule uses 100 batches/epoch in this synthetic
    # check.  A resumed epoch-5 checkpoint must start at the exact scale that
    # the original run had after 500 updates, then decrease smoothly.
    total = 1000
    start = 500
    s0 = _lr_scale(start, total)
    s1 = _lr_scale(start + 1, total)
    assert math.isclose(s0, 0.55, rel_tol=0.0, abs_tol=1e-12)
    assert s1 < s0
    assert math.isclose(_lr_scale(total, total), 0.1, rel_tol=0.0, abs_tol=1e-12)
