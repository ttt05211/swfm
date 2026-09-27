from __future__ import annotations

from dataclasses import dataclass
import inspect

import pytest
import torch

from real_motion.local_st_world_model import SEMANTIC_CLASSES
from real_motion.local_st_world_model_v17 import LocalSTWMV17Config
from real_motion.local_st_world_model_v18_se2 import LocalSpatialTemporalWorldModelV18SE2
from real_motion.motion_transport import FEATURE_DIM, FUTURE_FRAMES, HISTORY_FRAMES
from real_motion.v20_history_world import FREE_LABEL, CanonicalLattice
from real_motion.v20_unified_data import (
    UnifiedHistoryInput,
    audit_inference_input_names,
    native_tile_points_to_t0,
    make_completion_tile,
)
from real_motion.v20_unified_model import V20UnifiedTransportCompletion
from real_motion.v20_unified_runtime import completion_support, compose


def _v18() -> LocalSpatialTemporalWorldModelV18SE2:
    cfg = LocalSTWMV17Config(
        d_model=16,
        semantic_dim=4,
        heads=4,
        blocks=1,
        decoder_blocks=1,
        tube_hw=8,
        use_representation=True,
    )
    return LocalSpatialTemporalWorldModelV18SE2(cfg)


def test_v18_pure_decoder_preserves_original_forward_exactly():
    torch.manual_seed(1)
    model = _v18().eval()
    n = 2
    features = torch.randn(n, FEATURE_DIM)
    tube = torch.randint(0, 18, (n, HISTORY_FRAMES, 8, 8))
    kta = torch.randn(n, FUTURE_FRAMES, 2)
    frame = torch.randn(n, HISTORY_FRAMES, 5)
    mask = torch.randint(0, 2, tube.shape)
    with torch.no_grad():
        out = model(features, tube, kta, frame, mask, return_latents=True)
        decoded = model.decode_transport_queries(out["future_transport_queries"])
    for key in ("residual_xy_m", "existence_logits", "yaw_delta_rad"):
        assert torch.equal(out[key], decoded[key])


def test_inference_contract_has_no_future_supervision_argument():
    names = set(inspect.signature(V20UnifiedTransportCompletion.forward).parameters)
    assert not any("gt" in n or "target" in n and n != "target_source_mask_tube" for n in names)
    assert "future_semantic" not in names
    history = UnifiedHistoryInput(
        semantic=torch.zeros(1, 6, 2, 2, 2),
        observed=torch.ones(1, 6, 2, 2, 2, dtype=torch.bool),
        observed_free=torch.ones(1, 6, 2, 2, 2, dtype=torch.bool),
        future_ego_to_t0=torch.eye(4).view(1, 1, 4, 4).expand(1, 6, 4, 4),
    )
    audit_inference_input_names(history)

    @dataclass(frozen=True)
    class BadInput:
        future_gt: torch.Tensor

    with pytest.raises(ValueError, match="leaked"):
        audit_inference_input_names(BadInput(torch.zeros(1)))


def test_native_voxel_centers_transform_to_t0_without_axis_swap():
    tile = make_completion_tile(
        window_index=0,
        horizon=0,
        core_start_xyz=(0, 0, 0),
        core_shape_xyz=(2, 1, 1),
        native_shape_xyz=(2, 1, 1),
        halo=0,
    )
    transform = torch.eye(4).view(1, 1, 4, 4).repeat(1, 6, 1, 1)
    transform[0, 0, :3, 3] = torch.tensor([10.0, 20.0, 30.0])
    points = native_tile_points_to_t0(
        tile,
        transform,
        native_origin_xyz_m=(-1.0, -2.0, -3.0),
        native_voxel_size_xyz_m=(2.0, 4.0, 6.0),
    )
    assert torch.equal(points[0, 0, 0], torch.tensor([10.0, 20.0, 30.0]))
    assert torch.equal(points[1, 0, 0], torch.tensor([12.0, 20.0, 30.0]))


def test_each_horizon_uses_its_own_future_pose_label():
    transform = torch.eye(4).view(1, 1, 4, 4).repeat(1, 6, 1, 1)
    transform[0, :, 0, 3] = torch.arange(6, dtype=torch.float32) * 10.0
    points = []
    for horizon in range(6):
        tile = make_completion_tile(
            window_index=0,
            horizon=horizon,
            core_start_xyz=(0, 0, 0),
            core_shape_xyz=(1, 1, 1),
            native_shape_xyz=(1, 1, 1),
            halo=0,
        )
        points.append(
            native_tile_points_to_t0(
                tile,
                transform,
                native_origin_xyz_m=(0.0, 0.0, 0.0),
                native_voxel_size_xyz_m=(2.0, 2.0, 2.0),
            )[0, 0, 0, 0]
        )
    assert torch.equal(torch.stack(points), 1.0 + 10.0 * torch.arange(6))


def test_support_and_protected_add_only_composition():
    current = torch.full((1, 6, 2, 2, 1), FREE_LABEL)
    current[..., 0, 0, 0] = 4
    valid = torch.ones_like(current, dtype=torch.bool)
    valid[..., 1, 1, 0] = False
    support = completion_support(current, valid)
    assert not bool(support[..., 0, 0, 0].any())
    assert not bool(support[..., 1, 1, 0].any())
    logits = torch.zeros(*current.shape, SEMANTIC_CLASSES)
    logits[..., 8] = 3.0
    result = compose(current, logits, support)
    assert torch.equal(result[..., 0, 0, 0], current[..., 0, 0, 0])
    assert torch.equal(result[..., 1, 1, 0], current[..., 1, 1, 0])
    assert bool((result[..., 0, 1, 0] == 8).all())


def test_free_proposal_never_writes_even_on_support():
    current = torch.full((1, 6, 1, 1, 1), FREE_LABEL)
    logits = torch.zeros(*current.shape, SEMANTIC_CLASSES)
    logits[..., FREE_LABEL] = 5.0
    result = compose(current, logits, torch.ones_like(current, dtype=torch.bool))
    assert torch.equal(result, current)
