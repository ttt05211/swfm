import numpy as np
import torch

from real_motion.final_dataflow import (
    CausalHistoryState,
    _slim_history_record,
    batch_frozen_motion,
    build_causal_motion_prior,
)


def _history(current, velocities, pose=None):
    pose = np.eye(4, dtype=np.float64) if pose is None else np.asarray(pose, np.float64)
    return CausalHistoryState(
        record={},
        window=None,
        raw_history={"future_gt_occ": None},
        current_sem=np.zeros((1, 1, 1), np.uint8),
        previous_sem=np.zeros((1, 1, 1), np.uint8),
        current_pose=pose,
        previous_pose=pose,
        future_poses=tuple(np.eye(4) for _ in range(6)),
        current=current,
        previous=[],
        velocities=velocities,
        source_world_points=[],
        source_rel_xy=[],
        source_z_t0=np.zeros(len(current)),
        registrations=[],
        source_audit={},
        canonical_evidence=None,
        gpu_history={},
        preparation_seconds={},
    )


def test_causal_motion_prior_rebuilds_six_horizon_kta():
    current = [{"centroid_world": np.array([2.0, 3.0, 0.0])}]
    h = _history(current, {0: np.array([2.0, -1.0, 0.0])})
    kta, anchors = build_causal_motion_prior(h, 0.5)
    expected = np.array([[[(i+1), -0.5*(i+1)] for i in range(6)]], np.float32)
    assert np.array_equal(kta, expected)
    assert np.array_equal(anchors, expected + np.array([[[2.0, 3.0]]], np.float32))


def test_slim_history_record_rejects_forecast_labels_and_kta():
    record = {
        "sample_id": "s",
        "scene_name": "scene",
        "t0_token": "t0",
        "features": torch.zeros(1, 3),
        "local_semantic_tube": torch.zeros(1, 4, 2, 2, dtype=torch.uint8),
        "frame_motion_features": torch.zeros(1, 4, 5),
        "target_source_mask_tube": torch.zeros(1, 4, 2, 2, dtype=torch.uint8),
        "source_class_id": torch.tensor([2]),
        "source_centroid_xy_t0_m": torch.zeros(1, 2),
        "kta_displacement_xy_m": torch.ones(1, 6, 2),
        "future_gt_occ": torch.ones(6, 1, 1, 1),
        "target_yaw_rad": torch.ones(1, 6),
    }
    slim = _slim_history_record(record)
    assert "kta_displacement_xy_m" not in slim
    assert "future_gt_occ" not in slim
    assert "target_yaw_rad" not in slim
    assert set(("features", "local_semantic_tube", "frame_motion_features",
                "target_source_mask_tube")).issubset(slim)


class _Teacher:
    def motion(self, record, device):
        x = torch.as_tensor(record["features"], device=device).float()
        n = len(x)
        # Mimic the first-dimension contract of frozen V18 outputs.
        return {
            "residual_xy_m": x[:, :1, None].expand(n, 6, 2).clone(),
            "existence_logits": x[:, 1:2].expand(n, 6).clone(),
            "yaw_delta_rad": x[:, 2:3].expand(n, 6).clone(),
            "history_source_context": x,
            "future_transport_queries": x[:, None, :].expand(n, 6, x.shape[-1]).clone(),
        }


def _row(values):
    x = torch.tensor(values, dtype=torch.float32)
    n = len(x)
    return ({
        "features": x,
        "local_semantic_tube": torch.zeros(n, 4, 2, 2, dtype=torch.uint8),
        "kta_displacement_xy_m": torch.zeros(n, 6, 2),
        "frame_motion_features": torch.zeros(n, 4, 5),
        "target_source_mask_tube": torch.zeros(n, 4, 2, 2, dtype=torch.uint8),
    }, {})


def test_batch_frozen_motion_matches_independent_windows_on_cpu():
    rows = [
        _row([[1, 2, 3], [4, 5, 6]]),
        _row([[7, 8, 9]]),
    ]
    teacher = _Teacher()
    batched = batch_frozen_motion(teacher, rows, torch.device("cpu"))
    refs = [teacher.motion(r, torch.device("cpu")) for r, _ in rows]
    assert len(batched) == len(refs)
    for got, ref in zip(batched, refs):
        assert got.keys() == ref.keys()
        for key in got:
            assert torch.equal(got[key], ref[key])
