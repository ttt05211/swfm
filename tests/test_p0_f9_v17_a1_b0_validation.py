import math

import numpy as np
import torch

from real_motion.geometry import OccupancyGrid
from real_motion.local_st_world_model_v17 import soft_transport_overlap_loss
from real_motion.rigid_transport import (
    RasterizedRigidComponent,
    compose_component_replacements,
    compose_component_replacements_in_input_order,
)
from tools.real_motion.diagnose_p0_f9_v17_native_footprint import exact_source_masks, mask_counts


def _grid():
    return OccupancyGrid(
        x_min=-2.0,
        y_min=-2.0,
        z_min=0.0,
        voxel_size=(0.4, 0.4, 0.4),
        shape_hwd=(20, 20, 4),
    )


def _comp(class_id, idx, source_voxel_count):
    return RasterizedRigidComponent(
        int(class_id),
        np.asarray(idx, dtype=np.int64).reshape(-1, 3),
        int(source_voxel_count),
    )


def test_a1_changes_only_collision_winner_not_clear_contract():
    grid = _grid()
    free = 17
    anchor = np.full(grid.shape_hwd, free, dtype=np.uint8)
    anchor[2, 2, 0] = 4
    anchor[3, 2, 0] = 10
    baseline = [_comp(4, [[2, 2, 0]], 10), _comp(10, [[3, 2, 0]], 1)]

    # Input/source order is small object then large object.  Both write the same
    # destination voxel.  Legacy sorting writes large first then small, whereas
    # A1 preserves input order and therefore writes large last.
    replacements = [_comp(10, [[8, 8, 0]], 1), _comp(4, [[8, 8, 0]], 10)]
    legacy = compose_component_replacements(
        anchor, baseline, replacements, dynamic_class_ids=(4, 10), free_label=free, grid=grid
    )
    source_order = compose_component_replacements_in_input_order(
        anchor, baseline, replacements, dynamic_class_ids=(4, 10), free_label=free, grid=grid
    )
    assert legacy[2, 2, 0] == free
    assert source_order[2, 2, 0] == free
    assert legacy[3, 2, 0] == free
    assert source_order[3, 2, 0] == free
    assert legacy[8, 8, 0] == 10
    assert source_order[8, 8, 0] == 4


def test_a1_matches_legacy_when_replacements_do_not_collide():
    grid = _grid()
    free = 17
    anchor = np.full(grid.shape_hwd, free, dtype=np.uint8)
    replacements = [_comp(10, [[8, 8, 0]], 1), _comp(4, [[9, 9, 0]], 10)]
    legacy = compose_component_replacements(
        anchor, [], replacements, dynamic_class_ids=(4, 10), free_label=free, grid=grid
    )
    source_order = compose_component_replacements_in_input_order(
        anchor, [], replacements, dynamic_class_ids=(4, 10), free_label=free, grid=grid
    )
    assert np.array_equal(legacy, source_order)


def test_exact_source_mask_aligns_to_common_pooled_coordinates():
    grid = _grid()
    # Center 0.2 m lies in native cell (5,5) for x_min=y_min=-2 at 0.4 m.
    # With the 4 m local patch starting at native cell (0,0), cells 4/5 form
    # one 2x2 block and therefore one 0.8 m pooled support cell.
    voxels = np.asarray(
        [
            [4, 4, 0], [4, 5, 0], [5, 4, 1], [5, 5, 1],
        ],
        dtype=np.int64,
    )
    tight, pooled, coverage = exact_source_masks(
        voxels,
        np.asarray([0.2, 0.2]),
        grid=grid,
        patch_size_m=4.0,
        pooled_resolution_m=0.8,
    )
    assert tight.shape == (2, 2)
    assert int(tight.sum()) == 4
    assert pooled.shape == (5, 5)
    assert int(pooled.sum()) == 1
    assert math.isclose(coverage, 1.0)
    counts = mask_counts(pooled, pooled)
    assert counts["intersection"] == 1
    assert counts["legacy_extra_cells"] == 0
    assert counts["legacy_missed_cells"] == 0
    assert math.isclose(counts["iou"], 1.0)


def test_native_overlap_gradient_is_finite_and_nonzero_near_overlap():
    mask = torch.zeros(1, 5, 7)
    mask[0, 1:4, 1:3] = 1.0
    mask[0, 3, 3:6] = 1.0
    pred = torch.zeros(1, 6, 2, requires_grad=True)
    target = torch.zeros_like(pred)
    pred.data[0, 1, 0] = 0.2
    valid = torch.zeros(1, 6, dtype=torch.bool)
    valid[0, 1] = True
    loss, stats = soft_transport_overlap_loss(
        pred, target, mask, valid, patch_resolution_m=0.4
    )
    grad = torch.autograd.grad(loss, pred)[0]
    assert torch.isfinite(loss)
    assert torch.isfinite(grad).all()
    assert 0.0 < float(stats["transport_soft_iou"]) < 1.0
    assert float(torch.linalg.vector_norm(grad[0, 1])) > 0.0
