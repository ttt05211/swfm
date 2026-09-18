import numpy as np
import pytest
import torch

from real_motion.geometry import OccupancyGrid
from real_motion.rigid_transport import (
    RasterizedRigidComponent,
    compose_component_replacements_in_input_order,
)
from real_motion.runtime_fastpath import (
    component_lists_equal,
    compose_component_replacements_fast_exact,
    extract_instances_cropped_exact,
    inverse_warp_sequence_cuda_exact,
    majority_fill_sparse_5x5x1,
)
from real_motion.strong_w2det import (
    StrongW2DetConfig,
    extract_instances,
    inverse_warp,
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
        x_min=-4.0,
        y_min=-4.0,
        z_min=-1.0,
        voxel_size=(0.4, 0.4, 0.4),
        shape_hwd=(20, 20, 5),
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
        x_min=-2.0,
        y_min=-2.0,
        z_min=-0.8,
        voxel_size=(0.4, 0.4, 0.4),
        shape_hwd=(10, 10, 4),
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


def test_sparse_majority_fill_matches_reference_boundary_and_ties():
    sem = np.full((17, 19, 3), 17, dtype=np.uint8)
    # Equal-support classes around unknown cells exercise the frozen strict-'>' tie rule.
    sem[0:5, 0:5, 1] = 4
    sem[0:5:2, 0:5, 1] = 7
    unknown = np.zeros_like(sem, dtype=bool)
    unknown[0:3, 0:4, 1] = True
    unknown[8:11, 8:11, 1] = True
    ref = majority_fill(sem, unknown, kernel=(5, 5, 1), min_fraction=0.3)
    fast = majority_fill_sparse_5x5x1(
        sem, unknown, kernel=(5, 5, 1), min_fraction=0.3
    )
    assert np.array_equal(ref, fast)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime fast path")
def test_cuda_inverse_warp_matches_reference_small_grid():
    grid = OccupancyGrid(
        x_min=-4.0,
        y_min=-4.0,
        z_min=-1.0,
        voxel_size=(0.4, 0.4, 0.4),
        shape_hwd=(20, 20, 5),
    )
    rng = np.random.default_rng(20260918)
    sem = rng.integers(0, 18, size=grid.shape_hwd, dtype=np.uint8)

    transforms = []
    for yaw, tx, ty, tz in [
        (0.013, 0.17, -0.09, 0.01),
        (-0.027, 0.41, 0.18, -0.02),
        (0.061, -0.36, 0.22, 0.03),
    ]:
        c0, s0 = np.cos(yaw), np.sin(yaw)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = np.asarray(
            [[c0, -s0, 0.0], [s0, c0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        T[:3, 3] = np.asarray([tx, ty, tz], dtype=np.float64)
        transforms.append(T)

    fast = inverse_warp_sequence_cuda_exact(
        sem,
        transforms,
        grid=grid,
        free_label=17,
        device=torch.device("cuda"),
    )
    for T, (fast_sem, fast_known) in zip(transforms, fast):
        ref_sem, ref_known = inverse_warp(sem, T, grid, 17)
        assert np.array_equal(ref_sem, fast_sem)
        assert np.array_equal(ref_known, fast_known)
