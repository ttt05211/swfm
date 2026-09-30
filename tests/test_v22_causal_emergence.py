import numpy as np

from real_motion.v22_causal_emergence import (
    build_future_static_memory_only,
    build_surface_frontier,
    nearest_column_proposal,
    oracle_surface_proposal,
    protected_surface_add,
    tangent_plane_proposal,
)
from real_motion.geometry import OccupancyGrid
from real_motion.v19_innovation import build_future_aligned_history_and_static_memory


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
