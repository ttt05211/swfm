import numpy as np
import torch

from real_motion.geometry import OccupancyGrid
from real_motion.motion_transport import (
    FEATURE_DIM,
    FUTURE_FRAMES,
    HISTORY_FRAMES,
    MotionTransportHead,
    backward_component_tracks,
    build_motion_targets,
    build_source_features,
    motion_transport_loss,
)


def comp(cls, x, y, vox=8):
    idx = np.asarray([[10 + i % 2, 10 + (i // 2) % 2, i % 2] for i in range(vox)], dtype=np.int64)
    return {
        "class_id": int(cls),
        "centroid_world": np.asarray([x, y, 0.0], dtype=np.float64),
        "voxel_indices": idx,
        "voxel_count": int(len(idx)),
    }


def test_backward_tracks_and_feature_contract():
    frames = []
    for t in range(HISTORY_FRAMES):
        frames.append([comp(4, 0.5 * t, 1.0)])
    centers, valid = backward_component_tracks(frames, frame_dt_s=0.5, max_speed_mps=25.0)
    assert centers.shape == (1, HISTORY_FRAMES, 3)
    assert valid.all()
    vel = {0: np.asarray([1.0, 0.0, 0.0])}
    feat = build_source_features(frames[-1], vel, centers, valid, np.eye(4), grid=OccupancyGrid())
    assert feat.shape == (1, FEATURE_DIM)
    assert np.isfinite(feat).all()


def test_motion_targets_are_kta_residuals():
    current = [comp(4, 0.0, 0.0)]
    vel = {0: np.asarray([2.0, 0.0, 0.0])}
    token = "car0"
    maps = []
    for h in range(FUTURE_FRAMES):
        # Exact constant velocity: target residual must be zero.
        maps.append({token: {"instance_token": token, "class_id": 4,
                            "center_world": np.asarray([(h + 1) * 1.0, 0.0, 0.0])}})
    target = build_motion_targets(current, vel, [token], maps, np.eye(4), frame_dt_s=0.5)
    assert np.allclose(target["target_residual_xy_m"], 0.0)
    assert target["target_valid"].all()
    assert np.allclose(target["existence"], 1.0)


def test_head_zero_residual_initialization_and_loss():
    torch.manual_seed(0)
    model = MotionTransportHead(hidden_dim=32)
    x = torch.randn(4, FEATURE_DIM)
    out = model(x)
    assert out["residual_xy_m"].shape == (4, FUTURE_FRAMES, 2)
    assert torch.equal(out["residual_xy_m"], torch.zeros_like(out["residual_xy_m"]))
    batch = {
        "target_residual_xy_m": torch.zeros(4, FUTURE_FRAMES, 2),
        "existence": torch.ones(4, FUTURE_FRAMES),
        "target_valid": torch.ones(4, FUTURE_FRAMES, dtype=torch.bool),
        "supervised_source": torch.ones(4, dtype=torch.bool),
    }
    loss, parts = motion_transport_loss(out, batch)
    assert torch.isfinite(loss)
    assert parts["trajectory_smooth_l1"] == 0.0


def test_missing_future_trains_existence_but_not_trajectory():
    model = MotionTransportHead(hidden_dim=32)
    out = model(torch.zeros(1, FEATURE_DIM))
    batch = {
        "target_residual_xy_m": torch.zeros(1, FUTURE_FRAMES, 2),
        "existence": torch.zeros(1, FUTURE_FRAMES),
        "target_valid": torch.zeros(1, FUTURE_FRAMES, dtype=torch.bool),
        "supervised_source": torch.ones(1, dtype=torch.bool),
    }
    loss, parts = motion_transport_loss(out, batch)
    assert torch.isfinite(loss)
    assert parts["trajectory_labels"] == 0
    assert parts["existence_labels"] == FUTURE_FRAMES
