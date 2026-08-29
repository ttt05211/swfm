import numpy as np
import torch

from real_motion.geometry import OccupancyGrid
from real_motion.kta import KTAConfig
from real_motion.motion import PersistenceMotionConfig
from real_motion.msp import build_probe_record
from real_motion.nuscenes_adapter import WindowTokens
from real_motion.prepared import PrepareConfig


class FakeNuScenes:
    def __init__(self, future_x):
        self.future_x = float(future_x)
        self.samples = {
            "h0": {"anns": ["a_h0"]},
            "h1": {"anns": ["a_h1"]},
            "f0": {"anns": ["a_f0"]},
            "f1": {"anns": ["a_f1"]},
        }
        self.anns = {
            "a_h0": self._ann(-0.5),
            "a_h1": self._ann(0.5),
            "a_f0": self._ann(self.future_x),
            "a_f1": self._ann(self.future_x + 1.0),
        }

    @staticmethod
    def _ann(x):
        return {
            "instance_token": "car-1",
            "category_name": "vehicle.car",
            "translation": [float(x), 0.5, 0.0],
            "rotation": [1.0, 0.0, 0.0, 0.0],
            "size": [1.0, 2.0, 1.0],
        }

    def get(self, table, token):
        if table == "sample":
            return self.samples[token]
        if table == "sample_annotation":
            return self.anns[token]
        raise KeyError((table, token))


class FakeSource:
    def __init__(self, future_x):
        self.nusc = FakeNuScenes(future_x)

    def _sem(self, token):
        x = np.full((6, 6, 1), 17, dtype=np.int64)
        xi = {"h0": 2, "h1": 3, "f0": 4, "f1": 5}[token]
        x[xi, 3, 0] = 4
        return x

    def load_occ3d(self, scene, token, require_lidar_mask=True):
        sem = self._sem(token)
        return sem, np.ones_like(sem, dtype=bool)

    def load_semantics(self, scene, token):
        return self._sem(token)

    def pose(self, token):
        return np.eye(4, dtype=np.float64)

    def official_trajectory(self, history, future, hist_last=2, zero_prefix=0,
                            require_info=False):
        return np.zeros((4, 2), dtype=np.float32)


def _cfg():
    grid = OccupancyGrid(-3, -3, -0.5, (1.0, 1.0, 1.0), (6, 6, 1))
    motion = PersistenceMotionConfig(
        min_observed_frames=2,
        min_static_observations=2,
        history_dt_s=0.5,
        voxel_size_xy_m=(1.0, 1.0),
        moving_speed_mps=0.5,
        component_max_step_m=3.0,
    )
    return PrepareConfig(
        history_frames=2,
        future_frames=2,
        frame_dt_s=0.5,
        trajectory_length=4,
        trajectory_hist_last=2,
        trajectory_zero_prefix=0,
        trajectory_protocol="unit_4step",
        require_temporal_info=False,
        grid=grid,
        motion=motion,
        kta=KTAConfig(history_dt_s=0.5, max_match_distance_m=3.0),
    )


def test_future_gt_changes_labels_but_not_causal_msp_inputs():
    w = WindowTokens("scene", ("h0", "h1"), "h1", ("f0", "f1"))
    cfg = _cfg()
    near = build_probe_record(FakeSource(1.5), w, cfg, match_max_distance_m=2.0)
    far = build_probe_record(FakeSource(3.5), w, cfg, match_max_distance_m=2.0)

    assert near["num_candidates"] == far["num_candidates"] == 1
    assert torch.equal(near["features"], far["features"])
    assert torch.equal(near["anchors_xy_t0_m"], far["anchors_xy_t0_m"])
    assert torch.equal(near["candidate_state"], far["candidate_state"])
    assert torch.equal(near["candidate_extent_xy_m"], far["candidate_extent_xy_m"])
    assert not torch.equal(near["target_xy_t0_m"], far["target_xy_t0_m"])
