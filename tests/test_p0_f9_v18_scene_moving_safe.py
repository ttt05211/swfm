from __future__ import annotations

import numpy as np
import torch

from real_motion.geometry import OccupancyGrid
from real_motion.local_stwm_scene_moving_safe import (
    REPORT_FRAME_INDICES,
    _sample_source_alpha_se2,
    scene_moving_safe_loss,
)
from real_motion.local_stwm_scene_supervision import _sample_source_alpha
from real_motion.motion_transport import FUTURE_FRAMES


def _flat(idx, grid):
    x, y, z = [int(v) for v in idx]
    _, Y, Z = grid.shape_hwd
    return (x * Y + y) * Z + z


def test_se2_sampler_zero_yaw_matches_translation_sampler():
    grid = OccupancyGrid(
        x_min=0.0,
        y_min=0.0,
        z_min=0.0,
        voxel_size=(1.0, 1.0, 1.0),
        shape_hwd=(8, 8, 2),
    )
    vox = np.asarray([[2, 2, 0], [2, 3, 0], [3, 2, 0]], dtype=np.int64)
    pose = np.eye(4, dtype=np.float64)
    query = torch.tensor(
        [_flat((2, 2, 0), grid), _flat((3, 2, 0), grid)],
        dtype=torch.long,
    )
    d = torch.tensor([0.25, -0.15], dtype=torch.float32, requires_grad=True)
    old = _sample_source_alpha(
        vox, d, pose, pose, query, grid=grid, jitter_voxels=0.0
    )
    new = _sample_source_alpha_se2(
        vox,
        d,
        torch.tensor(0.0),
        pose,
        pose,
        query,
        grid=grid,
        jitter_voxels=0.0,
    )
    assert torch.allclose(old, new, atol=1e-6, rtol=1e-6)


def _one_source_scene(pred_dx):
    grid = OccupancyGrid(
        x_min=0.0,
        y_min=0.0,
        z_min=0.0,
        voxel_size=(1.0, 1.0, 1.0),
        shape_hwd=(8, 8, 2),
    )
    vox = np.asarray([[2, 2, 0]], dtype=np.int64)
    cls = 4
    pose = np.eye(4, dtype=np.float64)
    anchor = np.full(
        (FUTURE_FRAMES, *grid.shape_hwd), 17, dtype=np.uint8
    )
    gt = anchor.copy()
    anchor[:, 2, 2, 0] = cls
    gt[:, 2, 2, 0] = cls

    support = {
        int(hi): np.asarray([_flat((2, 2, 0), grid)], dtype=np.int64)
        for hi in REPORT_FRAME_INDICES
    }
    pred = torch.zeros(1, FUTURE_FRAMES, 2, requires_grad=True)
    with torch.no_grad():
        pred[..., 0] = float(pred_dx)
    yaw = torch.zeros(1, FUTURE_FRAMES, requires_grad=True)
    kta = torch.zeros_like(pred)

    result = scene_moving_safe_loss(
        pred,
        yaw,
        kta,
        [[vox]],
        [torch.tensor([cls])],
        [pose],
        [np.stack([pose] * FUTURE_FRAMES, axis=0)],
        [anchor],
        [gt],
        [support],
        [slice(0, 1)],
        grid=grid,
        halo_voxels=2,
        jitter_voxels=0.0,
    )
    return result, pred, yaw


def test_scene_moving_safe_zero_when_candidate_matches_kta():
    result, pred, yaw = _one_source_scene(0.0)
    assert float(result.loss.detach()) < 1e-7
    assert abs(float(result.pred_soft_moving_miou.detach()) - 1.0) < 1e-6
    assert abs(float(result.kta_moving_miou.detach()) - 1.0) < 1e-6
    assert not result.active


def test_scene_moving_safe_positive_and_differentiable_when_candidate_hurts_kta():
    result, pred, yaw = _one_source_scene(0.75)
    assert float(result.loss.detach()) > 0.0
    assert float(result.pred_soft_moving_miou.detach()) < float(
        result.kta_moving_miou.detach()
    )
    assert result.active
    result.loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert float(pred.grad.abs().sum()) > 0.0
    assert yaw.grad is not None
    assert torch.isfinite(yaw.grad).all()