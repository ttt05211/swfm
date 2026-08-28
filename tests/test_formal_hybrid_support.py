import numpy as np

from real_motion.geometry import OccupancyGrid
from real_motion.kta import KTAConfig
from real_motion.motion import PersistenceMotionConfig
from real_motion.nuscenes_adapter import WindowTokens
from real_motion.prepared import PrepareConfig, prepare_nuscenes_window


class TinySource:
    def __init__(self):
        self.loaded = []

    def load_semantics(self, scene, token):
        self.loaded.append(token)
        x = np.full((8, 8, 1), 17, dtype=np.int64)
        if token == "h0":
            x[2, 4, 0] = 4
        elif token == "h1":
            x[3, 4, 0] = 4
        else:
            raise AssertionError("online include_gt=False must not load future semantics")
        return x

    def pose(self, token):
        return np.eye(4)

    def official_trajectory(self, history, future, hist_last=2, zero_prefix=0,
                            require_info=False):
        return np.zeros((4, 2), dtype=np.float32)


def test_formal_hybrid_corridor_is_write_support_not_kta_semantic_prior():
    src = TinySource()
    window = WindowTokens("s", ("h0", "h1"), "h1", ("f0", "f1"))
    grid = OccupancyGrid(-4, -4, -1, (1, 1, 1), (8, 8, 1))
    cfg = PrepareConfig(
        history_frames=2,
        future_frames=2,
        frame_dt_s=1.0,
        support_geometry="hybrid_endpoint_swept_v1",
        endpoint_tube_radii=(0, 0),
        swept_tube_radii=(0, 0),
        uncertain_tube_radii=(0, 0),
        trajectory_length=4,
        trajectory_hist_last=2,
        trajectory_zero_prefix=0,
        trajectory_protocol="unit_4step",
        require_temporal_info=False,
        grid=grid,
        motion=PersistenceMotionConfig(
            min_observed_frames=2,
            min_static_observations=2,
            history_dt_s=1.0,
            voxel_size_xy_m=(1, 1),
            moving_speed_mps=0.5,
        ),
        kta=KTAConfig(history_dt_s=1.0, max_match_distance_m=2.0),
    )

    out = prepare_nuscenes_window(src, window, cfg, include_gt=False)

    # Observed current car is x=3. Constant-velocity KTA endpoint at 1s is x=4.
    assert out["moving_kta_future_occ"][0, 4, 4, 0] == 4
    assert out["kta_future_occ"][0, 4, 4, 0] == 4

    # The swept corridor also contains the current x=3 cell, but that cell is
    # only write permission. It must not be copied into the semantic KTA prior.
    assert out["swept_generation_support"][0, 3, 4]
    assert out["generation_support_occ"][0, 3, 4]
    assert out["kta_future_occ"][0, 3, 4, 0] == 17

    # Endpoint remains included by the union.
    assert out["endpoint_generation_support"][0, 4, 4]
    assert out["generation_support_occ"][0, 4, 4]
    assert out["support_geometry"] == "hybrid_endpoint_swept_v1"
    assert src.loaded == ["h0", "h1"]
