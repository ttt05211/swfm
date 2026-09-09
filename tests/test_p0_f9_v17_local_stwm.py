from __future__ import annotations

import torch

from real_motion.local_st_world_model import LocalSTWMConfig, LocalSpatialTemporalWorldModel
from real_motion.local_st_world_model_v17 import (
    FRAME_MOTION_DIM,
    LocalSTWMV17Config,
    LocalSpatialTemporalWorldModelV17,
    frame_motion_features_from_flat,
    soft_transport_overlap_loss,
    target_source_mask_from_tube,
)
from real_motion.motion_transport import FEATURE_DIM, FEATURE_NAMES, FUTURE_FRAMES, HISTORY_FRAMES


def _idx(name: str) -> int:
    return FEATURE_NAMES.index(name)


def test_frame_motion_features_preserve_per_frame_offset_and_validity():
    x = torch.zeros(1, FEATURE_DIM)
    for t in range(HISTORY_FRAMES):
        x[0, _idx(f"hist_offset_{t}_x")] = 0.1 * t
        x[0, _idx(f"hist_offset_{t}_y")] = -0.05 * t
        x[0, _idx(f"hist_valid_{t}")] = 1.0
    for t in range(HISTORY_FRAMES - 1):
        x[0, _idx(f"hist_vel_{t}_x")] = 0.2 + 0.01 * t
        x[0, _idx(f"hist_vel_{t}_y")] = -0.1
    out = frame_motion_features_from_flat(x)
    assert out.shape == (1, HISTORY_FRAMES, FRAME_MOTION_DIM)
    assert torch.allclose(out[0, :, 0], torch.arange(HISTORY_FRAMES) * 0.1)
    assert torch.all(out[0, :, 4] == 1)
    x[0, _idx("hist_valid_2")] = 0.0
    out = frame_motion_features_from_flat(x)
    assert torch.equal(out[0, 2, :4], torch.zeros(4))


def test_target_source_mask_uses_center_gate_and_class():
    tube = torch.full((1, HISTORY_FRAMES, 20, 20), 17, dtype=torch.uint8)
    tube[:, :, 9:12, 9:12] = 4
    tube[:, :, 2:5, 2:5] = 4  # nearby same-class distractor outside center gate
    features = torch.zeros(1, FEATURE_DIM)
    features[0, _idx("extent_x_norm")] = 0.16  # 1.6m
    features[0, _idx("extent_y_norm")] = 0.16
    valid = torch.ones(1, HISTORY_FRAMES, dtype=torch.bool)
    mask = target_source_mask_from_tube(tube, torch.tensor([4]), valid, features)
    assert mask.shape == tube.shape
    assert int(mask[:, :, 9:12, 9:12].sum()) > 0
    assert int(mask[:, :, 2:5, 2:5].sum()) == 0
    valid[:, 0] = False
    mask = target_source_mask_from_tube(tube, torch.tensor([4]), valid, features)
    assert int(mask[:, 0].sum()) == 0


def test_v17_fresh_representation_model_is_exact_zero_residual():
    cfg = LocalSTWMV17Config(d_model=32, semantic_dim=8, heads=4, blocks=1, decoder_blocks=1, tube_hw=20, use_representation=True)
    model = LocalSpatialTemporalWorldModelV17(cfg)
    n = 2
    features = torch.randn(n, FEATURE_DIM)
    tube = torch.randint(0, 18, (n, HISTORY_FRAMES, 20, 20), dtype=torch.uint8)
    kta = torch.randn(n, FUTURE_FRAMES, 2)
    frame = frame_motion_features_from_flat(features)
    mask = torch.zeros_like(tube)
    out = model(features, tube, kta, frame, mask)
    assert torch.equal(out["residual_xy_m"], torch.zeros_like(out["residual_xy_m"]))
    assert torch.equal(out["existence_logits"], torch.zeros_like(out["existence_logits"]))


def test_v17_loss_only_forward_matches_v16_under_same_seed():
    kwargs = dict(d_model=32, semantic_dim=8, heads=4, blocks=1, decoder_blocks=1, tube_hw=20)
    torch.manual_seed(123)
    v16 = LocalSpatialTemporalWorldModel(LocalSTWMConfig(**kwargs)).eval()
    torch.manual_seed(123)
    v17 = LocalSpatialTemporalWorldModelV17(LocalSTWMV17Config(**kwargs, use_representation=False)).eval()
    n = 2
    features = torch.randn(n, FEATURE_DIM)
    tube = torch.randint(0, 18, (n, HISTORY_FRAMES, 20, 20), dtype=torch.uint8)
    kta = torch.randn(n, FUTURE_FRAMES, 2)
    with torch.no_grad():
        a = v16(features, tube, kta)
        b = v17(features, tube, kta)
    assert torch.equal(a["residual_xy_m"], b["residual_xy_m"])
    assert torch.equal(a["existence_logits"], b["existence_logits"])


def test_transport_overlap_is_best_at_exact_displacement_and_has_gradient():
    B = 1
    footprint = torch.zeros(B, 20, 20)
    footprint[:, 8:12, 7:13] = 1.0
    target = torch.zeros(B, FUTURE_FRAMES, 2)
    valid = torch.ones(B, FUTURE_FRAMES, dtype=torch.bool)
    exact = torch.zeros_like(target, requires_grad=True)
    loss0, stats0 = soft_transport_overlap_loss(exact, target, footprint, valid)
    assert float(loss0.detach()) < 1e-5
    assert stats0["transport_soft_iou"] > 0.999

    shifted = torch.zeros_like(target)
    shifted[..., 0] = 0.4
    shifted.requires_grad_()
    loss1, stats1 = soft_transport_overlap_loss(shifted, target, footprint, valid)
    assert float(loss1.detach()) > float(loss0.detach()) + 1e-3
    assert stats1["transport_soft_iou"] < stats0["transport_soft_iou"]
    loss1.backward()
    assert shifted.grad is not None
    assert float(shifted.grad.abs().sum()) > 0.0


def test_v17_accepts_zero_source_window():
    cfg = LocalSTWMV17Config(d_model=32, semantic_dim=8, heads=4, blocks=1, decoder_blocks=1, tube_hw=20, use_representation=True)
    model = LocalSpatialTemporalWorldModelV17(cfg)
    features = torch.empty((0, FEATURE_DIM))
    tube = torch.empty((0, HISTORY_FRAMES, 20, 20), dtype=torch.uint8)
    kta = torch.empty((0, FUTURE_FRAMES, 2))
    frame = torch.empty((0, HISTORY_FRAMES, FRAME_MOTION_DIM))
    mask = torch.empty_like(tube)
    out = model(features, tube, kta, frame, mask)
    assert out["residual_xy_m"].shape == (0, FUTURE_FRAMES, 2)
    assert out["existence_logits"].shape == (0, FUTURE_FRAMES)
