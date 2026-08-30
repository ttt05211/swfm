import numpy as np

from real_motion.geometry import OccupancyGrid
from real_motion.strong_w2det import StrongW2DetConfig, w2det_predict


def _cube(sem, x0, x1, y0, y1, z0, z1, cls=4):
    sem[x0:x1, y0:y1, z0:z1] = cls


def test_w2det_matched_component_uses_backward_constant_velocity():
    grid = OccupancyGrid(x_min=0.0, y_min=0.0, z_min=0.0,
                         voxel_size=(1.0, 1.0, 1.0), shape_hwd=(20, 20, 4))
    cfg = StrongW2DetConfig(free_label=17, min_component_voxels=6,
                            max_match_speed_mps=25.0, connectivity=2)
    prev = np.full(grid.shape_hwd, 17, dtype=np.uint8)
    cur = np.full(grid.shape_hwd, 17, dtype=np.uint8)
    _cube(prev, 5, 7, 5, 7, 1, 3)
    _cube(cur, 6, 8, 5, 7, 1, 3)
    eye = np.eye(4, dtype=np.float64)

    out = w2det_predict(
        cur, prev, eye, eye, eye,
        dt_future_s=1.0, dt_previous_s=0.5,
        grid=grid, cfg=cfg,
    )
    # One voxel displacement in 0.5 s -> 2 voxels/s. From current x=[6,8)
    # the 1 s future footprint must be x=[8,10).
    assert np.all(out[8:10, 5:7, 1:3] == 4)
    assert not np.any(out[6:8, 5:7, 1:3] == 4)


def test_w2det_unmatched_component_keeps_zero_object_velocity():
    grid = OccupancyGrid(x_min=0.0, y_min=0.0, z_min=0.0,
                         voxel_size=(1.0, 1.0, 1.0), shape_hwd=(20, 20, 4))
    cfg = StrongW2DetConfig(free_label=17, min_component_voxels=6,
                            max_match_speed_mps=25.0, connectivity=2)
    prev = np.full(grid.shape_hwd, 17, dtype=np.uint8)
    cur = np.full(grid.shape_hwd, 17, dtype=np.uint8)
    _cube(cur, 6, 8, 5, 7, 1, 3)
    eye = np.eye(4, dtype=np.float64)

    out = w2det_predict(
        cur, prev, eye, eye, eye,
        dt_future_s=2.0, dt_previous_s=0.5,
        grid=grid, cfg=cfg,
    )
    assert np.all(out[6:8, 5:7, 1:3] == 4)


def test_w2det_static_semantics_survive_identity_w1_branch():
    grid = OccupancyGrid(x_min=0.0, y_min=0.0, z_min=0.0,
                         voxel_size=(1.0, 1.0, 1.0), shape_hwd=(12, 12, 3))
    cfg = StrongW2DetConfig(free_label=17, min_component_voxels=6)
    prev = np.full(grid.shape_hwd, 17, dtype=np.uint8)
    cur = np.full(grid.shape_hwd, 17, dtype=np.uint8)
    cur[2:5, 2:5, 0:2] = 11  # driveable surface / non-motion semantic
    eye = np.eye(4, dtype=np.float64)
    out = w2det_predict(
        cur, prev, eye, eye, eye,
        dt_future_s=1.0, dt_previous_s=0.5,
        grid=grid, cfg=cfg,
    )
    assert np.array_equal(out, cur)
