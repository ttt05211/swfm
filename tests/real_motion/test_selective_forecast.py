from __future__ import annotations

import numpy as np
import torch

from real_motion.geometry import OccupancyGrid
from real_motion.metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS
from real_motion.rigid_transport import (
    RasterizedRigidComponent,
    compose_component_replacements_in_input_order,
)
from real_motion.selective_forecast import (
    CorrectionSelector,
    deterministic_scene_split,
    marginal_utility_pp,
    moving_counts,
    moving_miou_from_counts,
    selected_count,
    single_source_moving_delta_counts,
    spearman_corr,
    top_budget_mask,
)


def _grid():
    return OccupancyGrid(
        x_min=0.0,
        y_min=0.0,
        z_min=0.0,
        voxel_size=(1.0, 1.0, 1.0),
        shape_hwd=(5, 5, 1),
    )


def test_sparse_single_source_delta_matches_dense_a1_composition():
    grid = _grid()
    free = 17
    cls = int(DYNAMIC_CLASS_IDS[0])
    anchor = np.full(grid.shape_hwd, free, dtype=np.uint8)
    # KTA source currently occupies (1,1) and (1,2).
    anchor[1, 1, 0] = cls
    anchor[1, 2, 0] = cls
    # Another source/class is untouched and proves locality.
    other = int(DYNAMIC_CLASS_IDS[1])
    anchor[4, 4, 0] = other

    gt = np.full_like(anchor, free)
    gt[2, 1, 0] = cls
    gt[2, 2, 0] = cls
    gt[4, 4, 0] = other
    support = np.ones_like(anchor, dtype=bool)

    baseline = RasterizedRigidComponent(
        cls,
        np.asarray([[1, 1, 0], [1, 2, 0]], dtype=np.int64),
        2,
    )
    replacement = RasterizedRigidComponent(
        cls,
        np.asarray([[2, 1, 0], [2, 2, 0]], dtype=np.int64),
        2,
    )

    di, du = single_source_moving_delta_counts(
        anchor, gt, support, baseline, replacement, free_label=free, grid=grid
    )
    dense = compose_component_replacements_in_input_order(
        anchor,
        [baseline],
        [replacement],
        dynamic_class_ids=DYNAMIC_CLASS_IDS,
        free_label=free,
        grid=grid,
    )
    bi, bu = moving_counts(anchor, gt, support)
    ai, au = moving_counts(dense, gt, support)
    assert np.array_equal(di, ai - bi)
    assert np.array_equal(du, au - bu)


def test_sparse_delta_respects_clear_only_dynamic_contract():
    grid = _grid()
    free = 17
    cls = int(DYNAMIC_CLASS_IDS[0])
    anchor = np.full(grid.shape_hwd, free, dtype=np.uint8)
    # Static class under the KTA mask must not be cleared.
    anchor[1, 1, 0] = 11
    gt = anchor.copy()
    support = np.ones_like(anchor, dtype=bool)
    baseline = RasterizedRigidComponent(
        cls, np.asarray([[1, 1, 0]], dtype=np.int64), 1
    )
    replacement = RasterizedRigidComponent(
        cls, np.asarray([[2, 1, 0]], dtype=np.int64), 1
    )
    dense = compose_component_replacements_in_input_order(
        anchor,
        [baseline],
        [replacement],
        dynamic_class_ids=DYNAMIC_CLASS_IDS,
        free_label=free,
        grid=grid,
    )
    assert dense[1, 1, 0] == 11
    di, du = single_source_moving_delta_counts(
        anchor, gt, support, baseline, replacement, free_label=free, grid=grid
    )
    bi, bu = moving_counts(anchor, gt, support)
    ai, au = moving_counts(dense, gt, support)
    assert np.array_equal(di, ai - bi)
    assert np.array_equal(du, au - bu)


def test_marginal_utility_is_exact_from_global_counts():
    c = len(DYNAMIC_CLASS_IDS)
    inter = np.full((3, c), 50, dtype=np.int64)
    union = np.full((3, c), 100, dtype=np.int64)
    di = np.zeros_like(inter)
    du = np.zeros_like(union)
    di[:, 0] = 5
    gain = marginal_utility_pp(inter, union, di, du)
    expected = moving_miou_from_counts(inter + di, union) - moving_miou_from_counts(
        inter, union
    )
    assert abs(gain - expected) < 1e-12
    assert gain > 0


def test_budget_masks_are_stable_and_exact():
    scores = np.asarray([0.1, 0.9, 0.2, 0.8, 0.3])
    assert selected_count(5, 0) == 0
    assert selected_count(5, 20) == 1
    assert selected_count(5, 40) == 2
    assert selected_count(5, 100) == 5
    m = top_budget_mask(scores, 40)
    assert m.tolist() == [False, True, False, True, False]


def test_scene_split_is_disjoint_and_deterministic():
    scenes = [f"scene-{i:03d}" for i in range(20) for _ in range(3)]
    a, b = deterministic_scene_split(scenes, val_fraction=0.2, seed=7)
    a2, b2 = deterministic_scene_split(scenes, val_fraction=0.2, seed=7)
    assert a == a2 and b == b2
    assert a.isdisjoint(b)
    assert a | b == set(scenes)


def test_spearman_and_selector_shapes():
    assert spearman_corr(np.arange(10), np.arange(10)) == pytest.approx(1.0)
    model = CorrectionSelector()
    y = model(torch.zeros((4, model.net[0].in_features)))
    assert y.shape == (4,)
