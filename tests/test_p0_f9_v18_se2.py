from __future__ import annotations

import math
from dataclasses import asdict

import numpy as np
import torch

from real_motion.local_st_world_model_v17 import (
    MODEL_PROTOCOL_V17,
    LocalSTWMV17Config,
    LocalSpatialTemporalWorldModelV17,
    frame_motion_features_from_flat,
)
from real_motion.local_st_world_model_v18_se2 import (
    LocalSpatialTemporalWorldModelV18SE2,
    apply_box_centered_rigid_xy,
    apply_source_centered_rigid_xy,
    kta_relative_safe_transport_loss,
    relative_yaw_in_t0,
    soft_se2_transport_overlap_loss,
    source_center_se2_target,
)
from real_motion.motion_transport import FEATURE_DIM, FUTURE_FRAMES, HISTORY_FRAMES
from tools.real_motion.train_p0_f9_v18_se2_pair import _build_model_optimizer


def _yaw_pose(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return T


def test_source_center_se2_is_exactly_box_center_rigid_equivalent():
    rng = np.random.default_rng(7)
    for _ in range(32):
        pts = rng.normal(size=(25, 2))
        a0 = rng.normal(size=2)
        ah = rng.normal(size=2)
        cs = rng.normal(size=2)
        yaw = float(rng.uniform(-1.2, 1.2))
        tgt = source_center_se2_target(cs, a0, ah, yaw)
        box = apply_box_centered_rigid_xy(pts, a0, ah, yaw)
        source = apply_source_centered_rigid_xy(
            pts, cs, tgt.source_displacement_xy_m, tgt.yaw_rad
        )
        assert np.allclose(box, source, atol=2e-6)


def test_source_center_se2_reduces_to_box_displacement_when_no_rotation():
    cs = np.asarray([4.0, -2.0])
    a0 = np.asarray([1.0, 3.0])
    ah = np.asarray([2.5, 4.5])
    tgt = source_center_se2_target(cs, a0, ah, 0.0)
    assert np.allclose(tgt.source_displacement_xy_m, ah - a0, atol=1e-7)
    assert abs(tgt.yaw_rad) < 1e-12


def test_relative_yaw_is_invariant_to_t0_ego_yaw_for_nonturning_object():
    for ego_yaw in (-2.0, -0.3, 0.0, 0.7, 2.4):
        got = relative_yaw_in_t0(0.37, 0.37, _yaw_pose(ego_yaw))
        assert abs(got) < 1e-8


def test_v18_loads_v17_exactly_and_zero_yaw_preserves_xy_outputs():
    cfg = LocalSTWMV17Config(
        d_model=32,
        semantic_dim=8,
        heads=4,
        blocks=1,
        decoder_blocks=1,
        tube_hw=20,
        use_representation=True,
    )
    torch.manual_seed(101)
    old = LocalSpatialTemporalWorldModelV17(cfg).eval()
    new = LocalSpatialTemporalWorldModelV18SE2(cfg).eval()
    incompatible = new.load_state_dict(old.state_dict(), strict=False)
    assert set(incompatible.missing_keys) == {"yaw_head.weight", "yaw_head.bias"}
    assert incompatible.unexpected_keys == []

    n = 3
    features = torch.randn(n, FEATURE_DIM)
    tube = torch.randint(
        0, 18, (n, HISTORY_FRAMES, 20, 20), dtype=torch.uint8
    )
    kta = torch.randn(n, FUTURE_FRAMES, 2)
    frame = frame_motion_features_from_flat(features)
    mask = torch.zeros_like(tube)
    with torch.no_grad():
        a = old(features, tube, kta, frame, mask)
        b = new(features, tube, kta, frame, mask)
    assert torch.equal(a["residual_xy_m"], b["residual_xy_m"])
    assert torch.equal(a["existence_logits"], b["existence_logits"])
    assert torch.equal(b["yaw_delta_rad"], torch.zeros_like(b["yaw_delta_rad"]))


def test_soft_se2_overlap_is_zero_at_exact_nonzero_transform():
    B = 1
    footprint = torch.zeros(B, 20, 20)
    footprint[:, 8:12, 5:15] = 1.0
    target_d = torch.zeros(B, FUTURE_FRAMES, 2)
    target_d[..., 0] = 1.2
    target_d[..., 1] = -0.8
    target_yaw = torch.full((B, FUTURE_FRAMES), 0.31)
    valid = torch.ones(B, FUTURE_FRAMES, dtype=torch.bool)
    enabled = torch.ones(B, dtype=torch.bool)
    yaw_valid = torch.ones(B, FUTURE_FRAMES, dtype=torch.bool)
    pred_d = target_d.clone().requires_grad_(True)
    pred_yaw = target_yaw.clone().requires_grad_(True)

    loss, stats = soft_se2_transport_overlap_loss(
        pred_d,
        target_d,
        pred_yaw,
        target_yaw,
        footprint,
        valid,
        enabled,
        yaw_valid,
    )
    assert float(loss.detach()) < 2e-5
    assert stats["se2_transport_soft_iou"] > 0.9999


def test_soft_se2_yaw_gradient_matches_finite_difference_away_from_kinks():
    B = 1
    footprint = torch.zeros(B, 20, 20)
    footprint[:, 8:12, 5:15] = 1.0
    target_d = torch.zeros(B, FUTURE_FRAMES, 2)
    target_yaw = torch.full((B, FUTURE_FRAMES), 0.37)
    pred_d = torch.zeros_like(target_d)
    valid = torch.ones(B, FUTURE_FRAMES, dtype=torch.bool)
    enabled = torch.ones(B, dtype=torch.bool)
    yaw_valid = torch.ones(B, FUTURE_FRAMES, dtype=torch.bool)

    p = torch.full((B, FUTURE_FRAMES), 0.13, requires_grad=True)
    loss, _ = soft_se2_transport_overlap_loss(
        pred_d, target_d, p, target_yaw, footprint,
        valid, enabled, yaw_valid,
    )
    loss.backward()
    g = float(p.grad[0, 0])

    eps = 1e-3
    def f(v):
        q = torch.full((B, FUTURE_FRAMES), 0.13)
        q[0, 0] = v
        z, _ = soft_se2_transport_overlap_loss(
            pred_d, target_d, q, target_yaw, footprint,
            valid, enabled, yaw_valid,
        )
        return float(z)
    fd = (f(0.13 + eps) - f(0.13 - eps)) / (2.0 * eps)
    assert np.isfinite(g) and np.isfinite(fd)
    assert abs(g - fd) <= 0.08 * max(abs(fd), 1e-3) + 2e-3


def test_yaw_disabled_class_receives_no_shape_yaw_gradient():
    B = 1
    footprint = torch.zeros(B, 20, 20)
    footprint[:, 8:12, 5:15] = 1.0
    pred_d = torch.zeros(B, FUTURE_FRAMES, 2, requires_grad=True)
    target_d = torch.zeros_like(pred_d)
    pred_yaw = torch.full((B, FUTURE_FRAMES), 0.7, requires_grad=True)
    target_yaw = torch.full((B, FUTURE_FRAMES), 0.2)
    valid = torch.ones(B, FUTURE_FRAMES, dtype=torch.bool)
    yaw_valid = torch.ones(B, FUTURE_FRAMES, dtype=torch.bool)
    enabled = torch.zeros(B, dtype=torch.bool)

    loss, _ = soft_se2_transport_overlap_loss(
        pred_d, target_d, pred_yaw, target_yaw, footprint,
        valid, enabled, yaw_valid,
    )
    loss.backward()
    assert pred_yaw.grad is not None
    assert torch.equal(pred_yaw.grad, torch.zeros_like(pred_yaw.grad))



def test_safe_loss_is_zero_when_prediction_matches_or_beats_kta():
    B = 1
    footprint = torch.zeros(B, 20, 20)
    footprint[:, 8:12, 5:15] = 1.0
    valid = torch.ones(B, FUTURE_FRAMES, dtype=torch.bool)
    enabled = torch.ones(B, dtype=torch.bool)
    yaw_valid = torch.ones(B, FUTURE_FRAMES, dtype=torch.bool)

    # GT is a nonzero translation. Prediction exactly matches GT; KTA is worse.
    target_d = torch.zeros(B, FUTURE_FRAMES, 2)
    target_d[..., 0] = 1.6
    target_d[..., 1] = -0.8
    pred_d = target_d.clone().requires_grad_(True)
    kta_d = torch.zeros_like(target_d)
    target_yaw = torch.full((B, FUTURE_FRAMES), 0.2)
    pred_yaw = target_yaw.clone().requires_grad_(True)

    loss, stats = kta_relative_safe_transport_loss(
        pred_d,
        kta_d,
        target_d,
        pred_yaw,
        target_yaw,
        footprint,
        valid,
        enabled,
        yaw_valid,
    )
    assert float(loss.detach()) < 1e-7
    assert stats["safe_active_labels"] == 0
    loss.backward()
    assert pred_d.grad is not None
    assert pred_yaw.grad is not None


def test_safe_loss_is_positive_when_prediction_is_worse_than_kta():
    B = 1
    footprint = torch.zeros(B, 20, 20)
    footprint[:, 8:12, 5:15] = 1.0
    valid = torch.ones(B, FUTURE_FRAMES, dtype=torch.bool)
    enabled = torch.ones(B, dtype=torch.bool)
    yaw_valid = torch.ones(B, FUTURE_FRAMES, dtype=torch.bool)

    # GT == KTA, while the learned transform introduces a harmful shift/yaw.
    target_d = torch.zeros(B, FUTURE_FRAMES, 2)
    kta_d = torch.zeros_like(target_d)
    pred_d = torch.zeros_like(target_d, requires_grad=True)
    pred_d.data[..., 0] = 2.4
    target_yaw = torch.zeros(B, FUTURE_FRAMES)
    pred_yaw = torch.full(
        (B, FUTURE_FRAMES), 0.35, requires_grad=True
    )

    loss, stats = kta_relative_safe_transport_loss(
        pred_d,
        kta_d,
        target_d,
        pred_yaw,
        target_yaw,
        footprint,
        valid,
        enabled,
        yaw_valid,
    )
    assert float(loss.detach()) > 0.05
    assert stats["safe_active_labels"] == B * FUTURE_FRAMES
    assert stats["safe_active_fraction"] == 1.0
    assert stats["safe_pred_soft_iou"] < stats["safe_kta_soft_iou"]
    loss.backward()
    assert torch.isfinite(pred_d.grad).all()
    assert torch.isfinite(pred_yaw.grad).all()


def test_safe_loss_margin_is_one_sided_tolerance():
    B = 1
    footprint = torch.zeros(B, 20, 20)
    footprint[:, 8:12, 5:15] = 1.0
    valid = torch.ones(B, FUTURE_FRAMES, dtype=torch.bool)
    enabled = torch.ones(B, dtype=torch.bool)
    yaw_valid = torch.ones(B, FUTURE_FRAMES, dtype=torch.bool)
    target_d = torch.zeros(B, FUTURE_FRAMES, 2)
    kta_d = torch.zeros_like(target_d)
    pred_d = torch.zeros_like(target_d)
    pred_d[..., 0] = 0.05
    target_yaw = torch.zeros(B, FUTURE_FRAMES)
    pred_yaw = torch.zeros_like(target_yaw)

    no_margin, _ = kta_relative_safe_transport_loss(
        pred_d, kta_d, target_d, pred_yaw, target_yaw,
        footprint, valid, enabled, yaw_valid, margin=0.0,
    )
    with_margin, _ = kta_relative_safe_transport_loss(
        pred_d, kta_d, target_d, pred_yaw, target_yaw,
        footprint, valid, enabled, yaw_valid, margin=1.0,
    )
    assert float(no_margin) >= 0.0
    assert float(with_margin) == 0.0


def test_v18_optimizer_resume_preserves_old_state_and_appends_yaw_group():
    cfg = LocalSTWMV17Config(
        d_model=32,
        semantic_dim=8,
        heads=4,
        blocks=1,
        decoder_blocks=1,
        tube_hw=20,
        use_representation=True,
    )
    old = LocalSpatialTemporalWorldModelV17(cfg)
    opt = torch.optim.AdamW(old.parameters(), lr=5e-4, weight_decay=1e-4)
    # Populate Adam moments without depending on a particular forward graph.
    for p in old.parameters():
        if p.requires_grad:
            p.grad = torch.full_like(p, 1e-3)
    opt.step()
    opt.zero_grad(set_to_none=True)

    ck = {
        "protocol": MODEL_PROTOCOL_V17,
        "epoch": 5,
        "variant": "RL",
        "use_representation": True,
        "overlap_weight": 0.25,
        "state_dict": old.state_dict(),
        "optimizer": opt.state_dict(),
        "model_config": asdict(cfg),
        "args": {"lr": 5e-4, "weight_decay": 1e-4},
    }
    new, resumed = _build_model_optimizer("Y", ck, torch.device("cpu"))
    assert isinstance(new, LocalSpatialTemporalWorldModelV18SE2)
    assert len(resumed.param_groups) == 2
    assert len(resumed.param_groups[1]["params"]) == 2
    # Historical Adam state exists only for inherited parameters at step 0.
    yaw_ids = {id(p) for p in new.yaw_head.parameters()}
    assert all(id(p) in yaw_ids for p in resumed.param_groups[1]["params"])
    assert all(p not in resumed.state for p in resumed.param_groups[1]["params"])

    safe_model, safe_opt = _build_model_optimizer("S", ck, torch.device("cpu"))
    assert isinstance(safe_model, LocalSpatialTemporalWorldModelV18SE2)
    assert len(safe_opt.param_groups) == 2
    assert len(safe_opt.param_groups[1]["params"]) == 2