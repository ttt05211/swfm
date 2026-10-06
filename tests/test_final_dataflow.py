from types import SimpleNamespace

import numpy as np
import torch

from tools.real_motion.final_dataflow import (
    _derive_kta,
    _history_only_record,
    batched_frozen_motion,
)
from tools.real_motion.benchmark_p0_f9_dense_forecast_fps import aggregate


def _component(x, y, cls=4):
    return {
        "centroid_world": np.asarray([x, y, 0.0], dtype=np.float64),
        "class_id": cls,
        "voxel_indices": np.asarray([[0, 0, 0]], dtype=np.int64),
    }


def test_history_only_record_cannot_smuggle_future_prediction_or_gt():
    record = {
        "sample_id": "x",
        "scene_name": "s",
        "history_tokens": ("a", "b", "c", "d"),
        "t0_token": "d",
        "future_tokens": ("e", "f", "g", "h", "i", "j"),
        "features": torch.zeros(1, 3),
        "local_semantic_tube": torch.zeros(1, 4, 2, 2, dtype=torch.uint8),
        "frame_motion_features": torch.zeros(1, 4, 5),
        "target_source_mask_tube": torch.zeros(1, 4, 2, 2, dtype=torch.uint8),
        "source_class_id": torch.tensor([4]),
        "source_voxel_count": torch.tensor([1]),
        "kta_displacement_xy_m": torch.ones(1, 6, 2),
        "anchors_xy_t0_m": torch.ones(1, 6, 2),
        "future_gt_occ": torch.ones(6, 1, 1, 1),
        "target_residual_xy_m": torch.ones(1, 6, 2),
    }
    out = _history_only_record(record)
    assert "features" in out and "source_class_id" in out
    assert "kta_displacement_xy_m" not in out
    assert "anchors_xy_t0_m" not in out
    assert "future_gt_occ" not in out
    assert "target_residual_xy_m" not in out


def test_kta_is_rebuilt_from_history_velocity_for_all_six_horizons():
    current = [_component(1.0, 2.0)]
    velocities = {0: np.asarray([2.0, -1.0, 0.0], dtype=np.float64)}
    pose = np.eye(4, dtype=np.float64)
    centers, kta, anchors = _derive_kta(current, velocities, pose, 0.5)
    assert np.array_equal(centers, np.asarray([[1.0, 2.0]]))
    expected = np.asarray(
        [[[1.0 * h, -0.5 * h] for h in range(1, 7)]],
        dtype=np.float32,
    )
    assert np.array_equal(kta, expected)
    assert np.array_equal(
        anchors, expected + np.asarray([[[1.0, 2.0]]], dtype=np.float32))


def test_dense_forecast_fps_uses_total_frames_over_total_time():
    rows = [
        dict(seconds=1.0, stages_seconds={"motion": 0.2}, key="a"),
        dict(seconds=3.0, stages_seconds={"motion": 0.4}, key="b"),
    ]
    out = aggregate(rows)
    assert out["future_frames"] == 12
    assert out["total_seconds"] == 4.0
    assert out["Dense_Forecast_FPS"] == 3.0
    assert out["mean_six_ms"] == 2000.0


class _FakeTeacher:
    def __init__(self):
        self.calls = 0

    def motion(self, record, device):
        self.calls += 1
        n = len(record["features"])
        base = torch.arange(n, dtype=torch.float32).reshape(n, 1)
        return {
            "residual_xy_m": base[:, None].repeat(1, 6, 2),
            "existence_logits": base.repeat(1, 6),
            "yaw_delta_rad": base.repeat(1, 6),
            "history_source_context": base.repeat(1, 4),
            "future_transport_queries": base[:, None].repeat(1, 6, 4),
        }


def _motion_record(n):
    return {
        "features": torch.zeros(n, 3),
        "local_semantic_tube": torch.zeros(n, 4, 2, 2, dtype=torch.uint8),
        "kta_displacement_xy_m": torch.zeros(n, 6, 2),
        "frame_motion_features": torch.zeros(n, 4, 5),
        "target_source_mask_tube": torch.zeros(n, 4, 2, 2, dtype=torch.uint8),
    }


def test_training_motion_is_one_batch_forward_then_split():
    teacher = _FakeTeacher()
    rows = [_motion_record(2), _motion_record(3)]
    out = batched_frozen_motion(teacher, rows, torch.device("cpu"))
    assert teacher.calls == 1
    assert len(out) == 2
    assert out[0]["history_source_context"].shape[0] == 2
    assert out[1]["history_source_context"].shape[0] == 3
    assert torch.equal(
        out[1]["history_source_context"][:, 0],
        torch.tensor([2.0, 3.0, 4.0]))
