import math

import numpy as np

from real_motion.geometry import OccupancyGrid
from real_motion.metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS
from real_motion.rigid_transport import (
    compose_component_replacements,
    rasterize_rigid_component,
    wrap_angle,
)


def _grid():
    return OccupancyGrid(
        x_min=-2.0,
        y_min=-2.0,
        z_min=0.0,
        voxel_size=(1.0, 1.0, 1.0),
        shape_hwd=(6, 6, 2),
    )


def test_wrap_angle_contract():
    assert abs(wrap_angle(0.0)) < 1e-12
    assert abs(wrap_angle(2.0 * math.pi)) < 1e-12
    assert abs(wrap_angle(1.5 * math.pi) + 0.5 * math.pi) < 1e-12


def test_rigid_transport_zero_motion_preserves_voxel():
    grid = _grid()
    idx = np.asarray([[2, 2, 0], [3, 2, 0]], dtype=np.int64)
    comp = rasterize_rigid_component(
        idx,
        4,
        np.eye(4),
        np.eye(4),
        source_center_world=(0.5, 0.5, 0.5),
        target_center_world=(0.5, 0.5, 0.5),
        grid=grid,
    )
    assert {tuple(x) for x in comp.voxel_indices.tolist()} == {(2, 2, 0), (3, 2, 0)}


def test_rigid_transport_translation_moves_one_cell():
    grid = _grid()
    idx = np.asarray([[2, 2, 0], [3, 2, 0]], dtype=np.int64)
    comp = rasterize_rigid_component(
        idx,
        4,
        np.eye(4),
        np.eye(4),
        source_center_world=(0.5, 0.5, 0.5),
        target_center_world=(1.5, 0.5, 0.5),
        grid=grid,
    )
    assert {tuple(x) for x in comp.voxel_indices.tolist()} == {(3, 2, 0), (4, 2, 0)}


def test_rigid_transport_yaw_rotates_about_pivot():
    grid = _grid()
    # Voxel center (1.5, 0.5, 0.5), pivot (0.5, 0.5, 0.5).
    # A +90 degree rotation moves it to (0.5, 1.5, 0.5).
    idx = np.asarray([[3, 2, 0]], dtype=np.int64)
    comp = rasterize_rigid_component(
        idx,
        4,
        np.eye(4),
        np.eye(4),
        source_center_world=(0.5, 0.5, 0.5),
        target_center_world=(0.5, 0.5, 0.5),
        yaw_delta_rad=0.5 * math.pi,
        grid=grid,
    )
    assert comp.voxel_indices.shape == (1, 3)
    assert tuple(comp.voxel_indices[0]) == (2, 3, 0)


def test_coherent_replacement_clears_kta_copy_then_writes_target():
    grid = _grid()
    free = 17
    anchor = np.full(grid.shape_hwd, free, dtype=np.uint8)
    anchor[2, 2, 0] = 4  # selected Strong/KTA copy
    anchor[5, 5, 0] = 10  # unrelated dynamic object must remain

    baseline = rasterize_rigid_component(
        np.asarray([[2, 2, 0]], dtype=np.int64),
        4,
        np.eye(4),
        np.eye(4),
        source_center_world=(0.5, 0.5, 0.5),
        target_center_world=(0.5, 0.5, 0.5),
        grid=grid,
    )
    target = rasterize_rigid_component(
        np.asarray([[2, 2, 0]], dtype=np.int64),
        4,
        np.eye(4),
        np.eye(4),
        source_center_world=(0.5, 0.5, 0.5),
        target_center_world=(1.5, 0.5, 0.5),
        grid=grid,
    )
    out = compose_component_replacements(
        anchor,
        [baseline],
        [target],
        dynamic_class_ids=DYNAMIC_CLASS_IDS,
        free_label=free,
        grid=grid,
    )
    assert out[2, 2, 0] == free
    assert out[3, 2, 0] == 4
    assert out[5, 5, 0] == 10


def test_disappearance_is_clear_without_write():
    grid = _grid()
    free = 17
    anchor = np.full(grid.shape_hwd, free, dtype=np.uint8)
    anchor[2, 2, 0] = 4
    baseline = rasterize_rigid_component(
        np.asarray([[2, 2, 0]], dtype=np.int64),
        4,
        np.eye(4),
        np.eye(4),
        source_center_world=(0.5, 0.5, 0.5),
        target_center_world=(0.5, 0.5, 0.5),
        grid=grid,
    )
    out = compose_component_replacements(
        anchor,
        [baseline],
        [],
        dynamic_class_ids=DYNAMIC_CLASS_IDS,
        free_label=free,
        grid=grid,
    )
    assert out[2, 2, 0] == free
