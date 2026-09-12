from __future__ import annotations

import numpy as np
import pytest
import torch

from real_motion.geometry import OccupancyGrid
from real_motion.local_st_world_model_v17 import (
    LocalSpatialTemporalWorldModelV17,
    LocalSTWMV17Config,
)
from real_motion.motion_transport import FEATURE_DIM
from real_motion.v17_backtrace3d_probe import (
    Backtrace3DProbeConfig,
    LocalSpatialTemporalWorldModelV17Backtrace3D,
    build_kta_backtrace_3d_crops,
)


def _inputs(batch=2):
    return dict(
        features=torch.zeros(batch, FEATURE_DIM),
        tube=torch.full((batch, 6, 20, 20), 17, dtype=torch.uint8),
        kta=torch.zeros(batch, 6, 2),
        frame_motion=torch.zeros(batch, 6, 5),
        source_mask=torch.zeros(batch, 6, 20, 20, dtype=torch.uint8),
    )


def test_gamma_zero_is_exact_v17():
    torch.manual_seed(7)
    cfg = LocalSTWMV17Config()
    base = LocalSpatialTemporalWorldModelV17(cfg).eval()
    treatment = LocalSpatialTemporalWorldModelV17Backtrace3D(
        cfg, Backtrace3DProbeConfig(source_microbatch=1)
    ).eval()
    missing, unexpected = treatment.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected
    assert missing
    x = _inputs()
    with torch.no_grad():
        a = base(
            x["features"], x["tube"], x["kta"],
            x["frame_motion"], x["source_mask"],
        )
        b = treatment(
            x["features"], x["tube"], x["kta"],
            x["frame_motion"], x["source_mask"],
            branch_gamma=0.0,
        )
    assert torch.equal(a["residual_xy_m"], b["residual_xy_m"])
    assert torch.equal(a["existence_logits"], b["existence_logits"])


class _FakeSource:
    def __init__(self, sem, valid):
        self.sem = sem
        self.valid = valid

    def load_occ3d(self, scene_name, token, require_lidar_mask=True):
        return self.sem, self.valid

    def pose(self, token):
        return np.eye(4, dtype=np.float64)


def _fake_record():
    return {
        "scene_name": "scene",
        "history_tokens": tuple(f"t{i}" for i in range(6)),
        "source_centroid_xy_t0_m": torch.zeros((1, 2), dtype=torch.float32),
        "features": torch.zeros((1, FEATURE_DIM), dtype=torch.float32),
        "native_source_footprint_mask": torch.zeros((1, 40, 40), dtype=torch.uint8),
    }


def test_backtrace_crop_identity_history_cpu():
    grid = OccupancyGrid()
    sem = np.full(grid.shape_hwd, 17, dtype=np.uint8)
    valid = np.ones(grid.shape_hwd, dtype=bool)
    sem[100, 100, 5] = 4
    rec = _fake_record()
    rec["native_source_footprint_mask"][0, 20, 20] = 1
    crop = build_kta_backtrace_3d_crops(
        _FakeSource(sem, valid), rec, [0], grid=grid, frame_dt_s=0.5
    )
    assert crop["semantics"].shape == (1, 6, 64, 64, 16)
    assert crop["valid"].shape == crop["semantics"].shape
    assert crop["source_mask"].shape == (1, 64, 64)
    assert crop["relative_times"].shape == (1, 6)
    for t in range(1, 6):
        assert torch.equal(crop["semantics"][:, 0], crop["semantics"][:, t])
        assert torch.equal(crop["valid"][:, 0], crop["valid"][:, t])
    assert bool(crop["valid"].all())
    assert int((crop["semantics"] == 4).sum()) > 0
    assert int(crop["source_mask"].sum()) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_backtrace_crop_cuda_matches_cpu_identity():
    grid = OccupancyGrid()
    sem = np.full(grid.shape_hwd, 17, dtype=np.uint8)
    valid = np.ones(grid.shape_hwd, dtype=bool)
    sem[100, 100, 5] = 4
    rec = _fake_record()
    rec["native_source_footprint_mask"][0, 20, 20] = 1
    src = _FakeSource(sem, valid)
    cpu = build_kta_backtrace_3d_crops(src, rec, [0], grid=grid, frame_dt_s=0.5)
    gpu = build_kta_backtrace_3d_crops(
        src, rec, [0], grid=grid, frame_dt_s=0.5, device="cuda"
    )
    assert torch.equal(cpu["semantics"], gpu["semantics"].cpu())
    assert torch.equal(cpu["valid"], gpu["valid"].cpu())
    assert torch.equal(cpu["source_mask"], gpu["source_mask"].cpu())
    assert torch.equal(cpu["relative_times"], gpu["relative_times"].cpu())
