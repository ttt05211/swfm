import numpy as np

from real_motion.geometry import OccupancyGrid
from real_motion.rigid_transport import (
    RasterizedRigidComponent,
    compose_component_replacements_in_input_order,
)
from real_motion.runtime_fastpath import (
    component_lists_equal,
    compose_component_replacements_fast_exact,
    extract_instances_cropped_exact,
    majority_fill_sparse_5x5x1,
)
from real_motion.strong_w2det import (
    StrongW2DetConfig,
    extract_instances,
    majority_fill,
)


def test_sparse_majority_fill_matches_reference_random():
    rng = np.random.default_rng(20260918)
    for _ in range(12):
        sem = rng.integers(0, 18, size=(23, 21, 5), dtype=np.uint8)
        unknown = rng.random(sem.shape) < 0.17
        ref = majority_fill(
            sem, unknown, kernel=(5, 5, 1), min_fraction=0.3
        )
        fast = majority_fill_sparse_5x5x1(
            sem, unknown, kernel=(5, 5, 1), min_fraction=0.3
        )
        assert np.array_equal(ref, fast)


def test_cropped_component_extraction_matches_reference():
    grid = OccupancyGrid(
        x_min=-4.0, x_max=4.0,
        y_min=-4.0, y_max=4.0,
        z_min=-1.0, z_max=1.0,
        voxel_size=(0.4, 0.4, 0.4),
    )
    sem = np.full(grid.shape_hwd, 17, dtype=np.uint8)
    # Two same-class components and one different dynamic class.
    sem[2:5, 3:6, 1:3] = 4
    sem[12:16, 10:14, 2:4] = 4
    sem[6:9, 15:19, 0:2] = 7
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = np.asarray([1.25, -2.0, 0.3])
    cfg = StrongW2DetConfig(min_component_voxels=6)
    ref = extract_instances(sem, pose, grid=grid, cfg=cfg)
    fast = extract_instances_cropped_exact(sem, pose, grid=grid, cfg=cfg)
    assert component_lists_equal(ref, fast)


def test_fast_a1_compositor_matches_reference():
    grid = OccupancyGrid(
        x_min=-2.0, x_max=2.0,
        y_min=-2.0, y_max=2.0,
        z_min=-0.8, z_max=0.8,
        voxel_size=(0.4, 0.4, 0.4),
    )
    anchor = np.full(grid.shape_hwd, 17, dtype=np.uint8)
    anchor[2:5, 2:5, 1] = 4
    anchor[5:8, 5:8, 1] = 7
    b1 = RasterizedRigidComponent(4, np.asarray([[2,2,1],[2,3,1],[3,2,1]]), 3)
    b2 = RasterizedRigidComponent(7, np.asarray([[5,5,1],[5,6,1],[6,5,1]]), 3)
    r1 = RasterizedRigidComponent(4, np.asarray([[3,3,1],[3,4,1],[4,3,1]]), 3)
    r2 = RasterizedRigidComponent(7, np.asarray([[6,6,1],[6,7,1],[7,6,1]]), 3)
    ref = compose_component_replacements_in_input_order(
        anchor, [b1,b2], [r1,r2],
        dynamic_class_ids=(4,7), free_label=17, grid=grid,
    )
    fast = compose_component_replacements_fast_exact(
        anchor, [b1,b2], [r1,r2],
        dynamic_class_ids=(4,7), free_label=17, grid=grid,
    )
    assert np.array_equal(ref, fast)
