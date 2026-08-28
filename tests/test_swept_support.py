import numpy as np

from real_motion.geometry import OccupancyGrid
from real_motion.kta import MotionComponent
from real_motion.swept_support import (
    swept_component_voxel_masks,
    swept_support_in_future_ego,
)


def _grid():
    return OccupancyGrid(-4, -4, -1, (1, 1, 1), (8, 8, 2))


def _component(vx=1.0, vy=0.0):
    vox = np.asarray([[2, 3, 0]], dtype=np.int64)
    return MotionComponent(
        class_id=4,
        bev_cells=vox[:, :2].copy(),
        voxel_indices=vox,
        centroid_xy_m=np.asarray([-1.5, -0.5]),
        velocity_xy_mps=np.asarray([vx, vy]),
        matched=True,
    )


def test_swept_support_contains_start_intermediate_and_endpoint():
    out = swept_component_voxel_masks([_component(vx=1.0)], [3.0], _grid())
    assert out.shape == (1, 8, 8, 2)
    # x: 2 -> 5, with the same observed object footprint swept through 3 and 4.
    for x in (2, 3, 4, 5):
        assert out[0, x, 3, 0]
    assert not out[0, 1, 3, 0]
    assert not out[0, 6, 3, 0]


def test_swept_support_keeps_components_separate():
    a = _component(vx=2.0)
    b = MotionComponent(
        class_id=4,
        bev_cells=np.asarray([[2, 6]], dtype=np.int64),
        voxel_indices=np.asarray([[2, 6, 0]], dtype=np.int64),
        centroid_xy_m=np.asarray([-1.5, 2.5]),
        velocity_xy_mps=np.asarray([2.0, 0.0]),
        matched=True,
    )
    out = swept_component_voxel_masks([a, b], [1.0], _grid())[0]
    # Two horizontal corridors should not create a filled rectangle between y=3 and y=6.
    assert out[3, 3, 0]
    assert out[3, 6, 0]
    assert not out[3, 4, 0]
    assert not out[3, 5, 0]


def test_future_identity_pose_preserves_swept_bev():
    g = _grid()
    comp = _component(vx=1.0)
    t0 = np.eye(4)
    out = swept_support_in_future_ego([comp], [2.0], t0, [np.eye(4)], g)
    raw = swept_component_voxel_masks([comp], [2.0], g).any(axis=3)
    assert np.array_equal(out, raw)


def test_future_ego_translation_uses_geometry_warp():
    g = _grid()
    comp = _component(vx=0.0)
    t0 = np.eye(4)
    future_pose = np.eye(4)
    future_pose[0, 3] = 1.0
    out = swept_support_in_future_ego([comp], [1.0], t0, [future_pose], g)
    # Future ego moved +1m in world x, so a stationary world point appears one x-cell back.
    assert out[0, 1, 3]
    assert not out[0, 2, 3]
