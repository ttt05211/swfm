import numpy as np
import torch

from real_motion.motion_transport_v2 import (
    FUTURE_FRAMES,
    build_motion_targets,
    displacement_residual_from_absolute_centers,
    upgrade_v1_record_targets,
)


def comp(x, y):
    return {
        "class_id": 4,
        "centroid_world": np.asarray([x, y, 0.0], dtype=np.float64),
        "voxel_indices": np.asarray([[10, 10, 0], [11, 10, 0]], dtype=np.int64),
        "voxel_count": 2,
    }


def test_residual_preserves_component_to_box_center_offset():
    # Visible component centroid is at x=0.5 while the true box center is x=1.0.
    # KTA and GT both move exactly 1 m over the horizon.  Correct displacement
    # residual is zero; the discarded v12 absolute-center target would be +0.5 m.
    got = displacement_residual_from_absolute_centers(
        np.asarray([0.5, 0.0]),
        np.asarray([1.5, 0.0]),
        np.asarray([1.0, 0.0]),
        np.asarray([2.0, 0.0]),
    )
    assert np.allclose(got, 0.0)


def test_build_motion_targets_uses_gt_displacement_not_absolute_alignment():
    current = [comp(0.5, 0.0)]
    velocity = {0: np.asarray([2.0, 0.0, 0.0])}
    token = "car0"
    ann0 = {token: {"instance_token": token, "class_id": 4, "center_world": np.asarray([1.0, 0.0, 0.0])}}
    future = []
    for h in range(FUTURE_FRAMES):
        # 2 m/s => +1m per 0.5 s, exactly matching KTA displacement.
        future.append({token: {"instance_token": token, "class_id": 4,
                              "center_world": np.asarray([1.0 + (h + 1), 0.0, 0.0])}})
    target = build_motion_targets(current, velocity, [token], ann0, future, np.eye(4), frame_dt_s=0.5)
    assert np.allclose(target["target_residual_xy_m"], 0.0)
    assert np.allclose(target["source_centroid_xy_t0_m"], [[0.5, 0.0]])
    assert np.allclose(target["gt_t0_xy_t0_m"], [[1.0, 0.0]])
    assert np.allclose(target["target_displacement_xy_m"][0, :, 0], np.arange(1, FUTURE_FRAMES + 1))


def test_v1_cache_upgrade_retargets_without_recomputing_features():
    n = 1
    features = torch.zeros(n, 46)
    # FEATURE_DIM is intentionally not hard-coded by behavior except first XY.
    # Pad if the implementation feature contract is longer.
    from real_motion.motion_transport_v2 import FEATURE_DIM
    features = torch.zeros(n, FEATURE_DIM)
    features[0, 0] = 0.5 / 40.0
    anchors = torch.zeros(n, FUTURE_FRAMES, 2)
    target_xy = torch.zeros(n, FUTURE_FRAMES, 2)
    valid = torch.ones(n, FUTURE_FRAMES, dtype=torch.bool)
    for h in range(FUTURE_FRAMES):
        anchors[0, h, 0] = 0.5 + (h + 1)
        target_xy[0, h, 0] = 1.0 + (h + 1)
    rec = {
        "features": features,
        "anchors_xy_t0_m": anchors,
        "target_xy_t0_m": target_xy,
        "target_valid": valid,
        "target_residual_xy_m": target_xy - anchors,
    }
    out = upgrade_v1_record_targets(rec, np.asarray([[1.0, 0.0]], dtype=np.float32))
    assert torch.allclose(out["target_residual_xy_m"], torch.zeros_like(out["target_residual_xy_m"]), atol=1e-6)
    assert torch.equal(out["features"], rec["features"])
