import numpy as np

from real_motion.v22_causal_emergence import (
    build_surface_frontier_family,
    build_future_static_memory_only,
    build_surface_frontier,
    nearest_column_proposal,
    nearest_geometry_gt_semantic_proposal,
    oracle_geometry_history_semantic_proposal,
    oracle_surface_proposal,
    oracle_vertical_shift_proposal,
    protected_surface_add,
    tangent_plane_proposal,
)
from real_motion.geometry import OccupancyGrid
from real_motion.v19_innovation import build_future_aligned_history_and_static_memory
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import Metrics
from tools.real_motion.eval_p0_f9_v22_cet_surface_stage0 import (
    _add_only_metric_counts,
)


FREE = 17


def _scene():
    static = np.full((7, 5, 5), FREE, dtype=np.uint8)
    footprint = np.zeros((7, 5), dtype=bool)
    footprint[:4] = True
    # Flat road and sidewalk surfaces on the known side.
    static[:3, :, 1] = 11
    static[3, :, 2] = 13
    return static, footprint


def test_surface_frontier_separates_geometry_scope_from_causal_surface_support():
    static, footprint = _scene()
    frontier = build_surface_frontier(
        static,
        footprint,
        class_ids=(11,),
        widths_m=(1.0, 2.0),
        voxel_size_xy_m=1.0,
        free_label=FREE,
    )
    # x=4 is one cell outside the historical grid and therefore in scope.
    assert frontier.scope_by_width[1.0][4].all()
    # The nearest eligible road anchor is x=2; sidewalk x=3 is excluded.
    assert not frontier.causal_by_width[1.0][4].any()
    assert frontier.causal_by_width[2.0][4].all()
    assert not frontier.scope_by_width[2.0][:4].any()
    assert np.all(frontier.causal_by_width[2.0] <= frontier.scope_by_width[2.0])


def test_oracle_is_exact_and_protected_add_never_overwrites_base():
    gt = np.full((3, 2, 3), FREE, dtype=np.uint8)
    gt[2, :, 1] = 11
    support = np.zeros((3, 2), dtype=bool)
    support[2] = True
    proposal = oracle_surface_proposal(
        gt, support, class_ids=(11, 13), free_label=FREE
    )
    assert np.array_equal(proposal[2, :, 1], np.asarray([11, 11], dtype=np.uint8))
    base = np.full_like(gt, FREE)
    base[2, 0, 1] = 4
    pred = protected_surface_add(base, proposal, free_label=FREE)
    assert pred[2, 0, 1] == 4
    assert pred[2, 1, 1] == 11


def test_history_only_baselines_copy_surface_semantics_and_height():
    static, footprint = _scene()
    frontier = build_surface_frontier(
        static,
        footprint,
        class_ids=(11, 13),
        widths_m=(1.0,),
        voxel_size_xy_m=1.0,
        free_label=FREE,
    )
    support = frontier.causal_by_width[1.0]
    nearest = nearest_column_proposal(
        static,
        support,
        frontier.nearest_x,
        frontier.nearest_y,
        class_ids=(11, 13),
        free_label=FREE,
    )
    tangent = tangent_plane_proposal(
        static,
        support,
        frontier.nearest_x,
        frontier.nearest_y,
        class_ids=(11, 13),
        free_label=FREE,
    )
    assert (nearest[4, :, 2] == 13).all()
    assert (tangent[4, :, 2] == 13).all()
    assert not (nearest[:4] != FREE).any()
    assert not (tangent[:4] != FREE).any()


def test_tangent_plane_continues_simple_ramp():
    static = np.full((6, 3, 6), FREE, dtype=np.uint8)
    footprint = np.zeros((6, 3), dtype=bool)
    footprint[:4] = True
    for x in range(4):
        static[x, :, x] = 11
    frontier = build_surface_frontier(
        static,
        footprint,
        class_ids=(11,),
        widths_m=(1.0,),
        voxel_size_xy_m=1.0,
        free_label=FREE,
    )
    support = frontier.causal_by_width[1.0]
    nearest = nearest_column_proposal(
        static, support, frontier.nearest_x, frontier.nearest_y,
        class_ids=(11,), free_label=FREE,
    )
    tangent = tangent_plane_proposal(
        static, support, frontier.nearest_x, frontier.nearest_y,
        class_ids=(11,), free_label=FREE,
    )
    assert (nearest[4, :, 3] == 11).all()
    assert (tangent[4, :, 4] == 11).all()


def test_static_only_fast_path_is_exactly_the_formal_v19_mosaic():
    grid = OccupancyGrid(
        x_min=0.0,
        y_min=0.0,
        z_min=0.0,
        voxel_size=(1.0, 1.0, 1.0),
        shape_hwd=(6, 5, 4),
    )
    history = np.full((6, *grid.shape_hwd), FREE, dtype=np.uint8)
    observed = np.ones_like(history, dtype=bool)
    for frame in range(6):
        history[frame, 1:5, :, 1] = 11
        history[frame, 2, 2, 2] = 4  # dynamic and therefore never in Static Memory
    history_poses = np.repeat(np.eye(4, dtype=np.float64)[None], 6, axis=0)
    future_poses = history_poses.copy()
    _, _, _, expected = build_future_aligned_history_and_static_memory(
        history,
        observed,
        history_poses,
        future_poses,
        grid=grid,
        free_label=FREE,
        dynamic_class_ids=(2, 3, 4, 5, 6, 7, 9, 10),
        workers=1,
        return_coverage=False,
    )
    actual = build_future_static_memory_only(
        history,
        observed,
        history_poses,
        future_poses,
        grid=grid,
        free_label=FREE,
        dynamic_class_ids=(2, 3, 4, 5, 6, 7, 9, 10),
        workers=2,
    )
    assert np.array_equal(actual, expected)

    actual_memory, actual_observed = build_future_static_memory_only(
        history,
        observed,
        history_poses,
        future_poses,
        grid=grid,
        free_label=FREE,
        dynamic_class_ids=(2, 3, 4, 5, 6, 7, 9, 10),
        workers=2,
        return_observed_bev=True,
    )
    assert np.array_equal(actual_memory, expected)
    assert actual_observed.shape == (6, *grid.shape_hwd[:2])
    assert actual_observed.all()


def test_frontier_family_keeps_visibility_narrow_and_union_exact():
    static = np.full((7, 3, 4), FREE, dtype=np.uint8)
    static[:2, :, 1] = 11
    footprint = np.zeros((7, 3), dtype=bool)
    footprint[:4] = True
    observed = np.zeros((7, 3), dtype=bool)
    observed[:2] = True
    family = build_surface_frontier_family(
        static,
        footprint,
        observed,
        class_ids=(11,),
        widths_m=(1.0, 3.0),
        voxel_size_xy_m=1.0,
        free_label=FREE,
    )
    grid = family["GRID_ENTRY"]
    visibility = family["VISIBILITY_FRONTIER"]
    union = family["UNION"]
    assert grid.scope_by_width[1.0][4].all()
    assert visibility.scope_by_width[1.0][2].all()
    assert not visibility.scope_by_width[1.0][3].any()
    assert np.array_equal(
        union.scope_by_width[width := 3.0],
        grid.scope_by_width[width] | visibility.scope_by_width[width],
    )
    assert np.array_equal(
        union.causal_by_width[width],
        grid.causal_by_width[width] | visibility.causal_by_width[width],
    )


def test_factorized_oracles_isolate_semantic_and_vertical_errors():
    static = np.full((4, 2, 6), FREE, dtype=np.uint8)
    static[1, :, 1] = 11
    support = np.zeros((4, 2), dtype=bool)
    support[2] = True
    ax = np.ones((4, 2), dtype=np.int32)
    ay = np.broadcast_to(np.arange(2, dtype=np.int32)[None], (4, 2))
    gt = np.full_like(static, FREE)
    gt[2, :, 3] = 13

    exact_geometry = oracle_geometry_history_semantic_proposal(
        gt,
        support,
        static,
        ax,
        ay,
        class_ids=(11, 13),
        free_label=FREE,
    )
    assert (exact_geometry[2, :, 3] == 11).all()

    nearest = nearest_column_proposal(
        static,
        support,
        ax,
        ay,
        class_ids=(11, 13),
        free_label=FREE,
    )
    gt_semantic = nearest_geometry_gt_semantic_proposal(
        gt, nearest, class_ids=(11, 13), free_label=FREE
    )
    # The nearest geometry misses the GT height, so semantic supervision alone
    # cannot create a correct voxel.
    assert (gt_semantic[2, :, 1] == 11).all()

    shifted = oracle_vertical_shift_proposal(
        gt,
        static,
        support,
        ax,
        ay,
        class_ids=(11, 13),
        free_label=FREE,
        max_abs_shift_bins=4,
    )
    assert (shifted[2, :, 3] == 11).all()


def test_add_only_metric_fast_path_matches_full_frozen_confusion():
    rng = np.random.default_rng(7)
    gt = rng.integers(0, 18, size=(7, 6, 4), dtype=np.uint8)
    baseline = rng.integers(0, 18, size=gt.shape, dtype=np.uint8)
    moving = rng.random(gt.shape) < 0.3
    prediction = baseline.copy()
    candidates = (baseline == FREE) & (rng.random(gt.shape) < 0.5)
    prediction[candidates] = rng.integers(
        0, 17, size=int(candidates.sum()), dtype=np.uint8
    )
    base_counts = Metrics.counts(baseline, gt, moving, FREE)
    expected = Metrics.counts(prediction, gt, moving, FREE)
    actual = _add_only_metric_counts(
        base_counts, baseline, prediction, gt, moving, FREE
    )
    for got, wanted in zip(actual, expected):
        assert np.array_equal(got, wanted)
