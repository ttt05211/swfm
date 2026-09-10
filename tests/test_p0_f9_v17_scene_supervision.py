import math

import numpy as np
import torch

from real_motion.geometry import OccupancyGrid
from real_motion.local_st_world_model_v17 import LocalSTWMV17Config, LocalSpatialTemporalWorldModelV17
from real_motion.local_stwm_scene_supervision import (
    calibrate_scene_alpha_from_gradients,
    freeze_history_encoder_for_scene_continuation,
    sparse_full_scene_ce_ordered,
    v17_base_loss_tensors,
)
from real_motion.motion_transport import FEATURE_DIM, FUTURE_FRAMES
from tools.real_motion.train_p0_f9_v17_local_stwm import objective_loss


def _grid(n=12):
    return OccupancyGrid(
        x_min=0.0,
        y_min=0.0,
        z_min=0.0,
        voxel_size=(0.4, 0.4, 0.4),
        shape_hwd=(n, n, 2),
    )


def _scene_inputs(grid, pred_dx=0.0, gt_shift=0):
    n = grid.shape_hwd[0]
    vox = np.asarray([[3, 3, 0]], dtype=np.int64)
    pred = torch.zeros(1, FUTURE_FRAMES, 2, requires_grad=True)
    with torch.no_grad():
        pred[:, :, 0] = float(pred_dx)
    kta = torch.zeros_like(pred)
    anchor = np.full((FUTURE_FRAMES, n, n, 2), 17, dtype=np.uint8)
    anchor[:, 3, 3, 0] = 4
    gt = anchor.copy()
    if gt_shift:
        gt[:, 3, 3, 0] = 17
        gt[:, 3 + int(gt_shift), 3, 0] = 4
    poses = np.repeat(np.eye(4, dtype=np.float64)[None], FUTURE_FRAMES, axis=0)
    return pred, kta, [vox], torch.tensor([4]), np.eye(4), poses, anchor, gt


def test_sparse_scene_ce_matches_correct_full_scene_at_identity():
    grid = _grid(12)
    args = _scene_inputs(grid, pred_dx=0.0)
    result = sparse_full_scene_ce_ordered(
        *args,
        grid=grid,
        halo_voxels=1,
        eps=1e-4,
        jitter_voxels=0.0,
    )
    # Every voxel is predicted with its GT class.  Epsilon-smoothed correct CE
    # is therefore the full-scene value, independent of sparse query size.
    expected = -math.log(1.0 - 18.0e-4)
    assert math.isclose(float(result.loss.detach()), expected, rel_tol=2e-5, abs_tol=2e-6)
    assert 0 < result.query_voxels < result.full_voxels


def test_scene_query_domain_does_not_depend_on_gt_support():
    grid = _grid(12)
    a = list(_scene_inputs(grid, pred_dx=0.4, gt_shift=0))
    b = list(_scene_inputs(grid, pred_dx=0.4, gt_shift=4))
    ra = sparse_full_scene_ce_ordered(*a, grid=grid, halo_voxels=1, jitter_voxels=0.0)
    rb = sparse_full_scene_ce_ordered(*b, grid=grid, halo_voxels=1, jitter_voxels=0.0)
    assert ra.query_voxels == rb.query_voxels
    assert not math.isclose(float(ra.loss.detach()), float(rb.loss.detach()))


def test_full_scene_normalization_not_roi_mean():
    grads = []
    for n in (12, 24):
        grid = _grid(n)
        args = _scene_inputs(grid, pred_dx=0.2, gt_shift=0)
        pred = args[0]
        result = sparse_full_scene_ce_ordered(
            *args, grid=grid, halo_voxels=1, eps=1e-4, jitter_voxels=0.0
        )
        g = torch.autograd.grad(result.loss, pred)[0]
        grads.append(float(torch.linalg.vector_norm(g)))
    # Same local variable region in a 4x larger XY scene must receive a much
    # smaller gradient under full-scene normalization.  An ROI mean would not.
    assert grads[0] > grads[1] * 3.5


def test_tensor_base_objective_matches_frozen_v17_objective():
    torch.manual_seed(7)
    B = 5
    outputs = {
        "residual_xy_m": torch.randn(B, FUTURE_FRAMES, 2, requires_grad=True),
        "existence_logits": torch.randn(B, FUTURE_FRAMES, requires_grad=True),
    }
    mask = torch.zeros(B, 6, 8, 8, dtype=torch.uint8)
    mask[:, -1, 2:6, 2:6] = 1
    batch = {
        "target_residual_xy_m": torch.randn(B, FUTURE_FRAMES, 2),
        "target_valid": torch.tensor(
            [[1,1,1,1,1,1],[1,1,1,0,0,0],[1,1,0,0,0,0],[1,0,0,0,0,0],[1,1,1,1,0,0]],
            dtype=torch.bool,
        ),
        "existence": torch.randint(0, 2, (B, FUTURE_FRAMES)).float(),
        "supervised_source": torch.tensor([1,1,1,1,1], dtype=torch.bool),
        "target_source_mask_tube": mask,
    }
    old, _ = objective_loss(outputs, batch, overlap_weight=0.25, patch_resolution_m=0.8)
    new = v17_base_loss_tensors(outputs, batch, overlap_weight=0.25, patch_resolution_m=0.8)["total"]
    assert torch.allclose(old, new, atol=1e-7, rtol=1e-6)


def test_freeze_contract_trains_only_future_decoder_and_heads():
    model = LocalSpatialTemporalWorldModelV17(
        LocalSTWMV17Config(d_model=32, semantic_dim=8, heads=4, blocks=1, decoder_blocks=1, tube_hw=8)
    )
    report = freeze_history_encoder_for_scene_continuation(model)
    assert report["trainable_parameters"] > 0
    assert report["frozen_parameters"] > 0
    for name, p in model.named_parameters():
        should_train = name.startswith((
            "future_query", "future_time_embedding", "kta_future_proj", "decoder", "residual_head", "existence_head"
        ))
        assert p.requires_grad == should_train, name
    assert not any(p.requires_grad for p in model.kinematic_proj.parameters())


def test_alpha_calibration_is_fixed_gradient_ratio_not_sweep():
    out = calibrate_scene_alpha_from_gradients(
        [2.0, 4.0, 6.0], [0.5, 1.0, 1.5], target_ratio=0.25, max_alpha=100.0
    )
    assert math.isclose(out["motion_grad_median"], 4.0)
    assert math.isclose(out["scene_unit_grad_median"], 1.0)
    assert math.isclose(out["alpha"], 1.0)
    assert out["alpha_clipped"] is False
