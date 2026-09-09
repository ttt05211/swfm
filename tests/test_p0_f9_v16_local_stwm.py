from __future__ import annotations

import numpy as np
import torch

from real_motion.geometry import OccupancyGrid
from real_motion.local_st_world_model import (
    LocalSTWMConfig,
    LocalSpatialTemporalWorldModel,
    extract_bev_patch,
    priority_pool2x2,
    top_surface_semantic,
)
from real_motion.metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS
from real_motion.motion_transport import FEATURE_DIM, FUTURE_FRAMES, HISTORY_FRAMES


def test_top_surface_keeps_object_over_ground():
    grid = OccupancyGrid(shape_hwd=(4, 4, 4))
    sem = np.full(grid.shape_hwd, 17, dtype=np.uint8)
    sem[1, 2, 0] = 11
    sem[1, 2, 2] = int(DYNAMIC_CLASS_IDS[0])
    bev = top_surface_semantic(sem, grid=grid, free_label=17)
    assert int(bev[1, 2]) == int(DYNAMIC_CLASS_IDS[0])
    assert int(bev[0, 0]) == 17


def test_priority_pool_preserves_small_dynamic_cell():
    dyn = int(DYNAMIC_CLASS_IDS[0])
    x = np.full((4, 4), 17, dtype=np.uint8)
    x[0:2, 0:2] = 11
    x[1, 1] = dyn
    y = priority_pool2x2(x, free_label=17)
    assert y.shape == (2, 2)
    assert int(y[0, 0]) == dyn


def test_centered_patch_pads_outside_grid_with_free():
    grid = OccupancyGrid(shape_hwd=(8, 8, 2))
    bev = np.zeros((8, 8), dtype=np.uint8)
    patch = extract_bev_patch(
        bev,
        np.asarray([grid.x_min + 0.1, grid.y_min + 0.1], dtype=np.float32),
        grid=grid,
        patch_voxels=4,
        free_label=17,
    )
    assert patch.shape == (4, 4)
    assert (patch == 17).any()
    assert (patch == 0).any()


def test_fresh_local_stwm_is_exact_kta_residual_zero():
    cfg = LocalSTWMConfig(d_model=32, semantic_dim=8, heads=4, blocks=1, decoder_blocks=1, tube_hw=20)
    model = LocalSpatialTemporalWorldModel(cfg)
    n = 3
    features = torch.randn(n, FEATURE_DIM)
    tube = torch.randint(0, 18, (n, HISTORY_FRAMES, 20, 20), dtype=torch.uint8)
    kta = torch.randn(n, FUTURE_FRAMES, 2)
    out = model(features, tube, kta)
    assert out["residual_xy_m"].shape == (n, FUTURE_FRAMES, 2)
    assert out["existence_logits"].shape == (n, FUTURE_FRAMES)
    assert torch.equal(out["residual_xy_m"], torch.zeros_like(out["residual_xy_m"]))
    assert torch.equal(out["existence_logits"], torch.zeros_like(out["existence_logits"]))


def test_local_stwm_accepts_zero_source_window():
    cfg = LocalSTWMConfig(d_model=32, semantic_dim=8, heads=4, blocks=1, decoder_blocks=1, tube_hw=20)
    model = LocalSpatialTemporalWorldModel(cfg)
    features = torch.empty((0, FEATURE_DIM), dtype=torch.float32)
    tube = torch.empty((0, HISTORY_FRAMES, 20, 20), dtype=torch.uint8)
    kta = torch.empty((0, FUTURE_FRAMES, 2), dtype=torch.float32)
    out = model(features, tube, kta)
    assert out["residual_xy_m"].shape == (0, FUTURE_FRAMES, 2)
    assert out["existence_logits"].shape == (0, FUTURE_FRAMES)
    assert out["residual_xy_m"].dtype == features.dtype
    assert out["existence_logits"].dtype == features.dtype


def test_local_stwm_backpropagates_after_zero_head():
    cfg = LocalSTWMConfig(d_model=32, semantic_dim=8, heads=4, blocks=1, decoder_blocks=1, tube_hw=20)
    model = LocalSpatialTemporalWorldModel(cfg)
    n = 2
    features = torch.randn(n, FEATURE_DIM)
    tube = torch.randint(0, 18, (n, HISTORY_FRAMES, 20, 20), dtype=torch.uint8)
    kta = torch.randn(n, FUTURE_FRAMES, 2)
    target = torch.randn(n, FUTURE_FRAMES, 2)
    out = model(features, tube, kta)
    loss = torch.nn.functional.smooth_l1_loss(out["residual_xy_m"], target)
    loss.backward()
    assert model.residual_head.weight.grad is not None
    assert float(model.residual_head.weight.grad.abs().sum()) > 0.0
