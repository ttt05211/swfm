from __future__ import annotations

import json
import numpy as np
import pytest
import torch

from real_motion.geometry import OccupancyGrid
from real_motion.local_st_world_model_v17 import (
    LocalSpatialTemporalWorldModelV17,
    LocalSTWMV17Config,
)
from real_motion.motion_transport import FEATURE_DIM, FEATURE_NAMES
from tools.real_motion.diagnose_p0_f9_v17_yaw_oracle import _decide_yaw
from tools.real_motion.summarize_p0_f9_v17_backtrace3d_probe import _load as _load_summary_row

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


def test_gamma_one_zero_init_branch_is_exact_v17():
    torch.manual_seed(11)
    cfg = LocalSTWMV17Config()
    base = LocalSpatialTemporalWorldModelV17(cfg).eval()
    treatment = LocalSpatialTemporalWorldModelV17Backtrace3D(
        cfg, Backtrace3DProbeConfig(source_microbatch=1)
    ).eval()
    missing, unexpected = treatment.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected
    assert missing
    x = _inputs(batch=1)
    semantics = torch.randint(0, 19, (1, 6, 64, 64, 16), dtype=torch.uint8)
    valid = torch.ones_like(semantics, dtype=torch.bool)
    source_mask = torch.zeros((1, 64, 64), dtype=torch.bool)
    source_mask[:, 31:34, 29:35] = True
    relative_times = torch.tensor([[-2.5, -2.0, -1.5, -1.0, -0.5, 0.0]])
    with torch.no_grad():
        a = base(
            x["features"], x["tube"], x["kta"],
            x["frame_motion"], x["source_mask"],
        )
        b = treatment(
            x["features"], x["tube"], x["kta"],
            x["frame_motion"], x["source_mask"],
            backtrace_semantics=semantics,
            backtrace_valid=valid,
            backtrace_source_mask=source_mask,
            backtrace_relative_times=relative_times,
            branch_gamma=1.0,
        )
    assert torch.count_nonzero(b["backtrace3d_query_residual"]) == 0
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


class _FakeSequenceSource:
    def __init__(self, sem_by_token, valid_by_token, pose_by_token):
        self.sem_by_token = sem_by_token
        self.valid_by_token = valid_by_token
        self.pose_by_token = pose_by_token

    def load_occ3d(self, scene_name, token, require_lidar_mask=True):
        return self.sem_by_token[token], self.valid_by_token[token]

    def pose(self, token):
        return self.pose_by_token[token]


def _pose(yaw_rad, tx, ty):
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    T = np.eye(4, dtype=np.float64)
    T[:2, :2] = np.asarray([[c, -s], [s, c]], dtype=np.float64)
    T[0, 3] = float(tx)
    T[1, 3] = float(ty)
    return T


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_backtrace_crop_cuda_matches_cpu_asymmetric_motion_pose():
    grid = OccupancyGrid()
    ix = np.arange(grid.shape_hwd[0], dtype=np.int64)[:, None, None]
    iy = np.arange(grid.shape_hwd[1], dtype=np.int64)[None, :, None]
    iz = np.arange(grid.shape_hwd[2], dtype=np.int64)[None, None, :]
    sem = ((3 * ix + 5 * iy + 7 * iz) % 17).astype(np.uint8)
    # Make two strongly asymmetric landmarks explicit as well.
    sem[121, 73, 6] = 4
    sem[84, 137, 11] = 7
    valid = np.ones(grid.shape_hwd, dtype=bool)

    rec = _fake_record()
    rec["source_centroid_xy_t0_m"][0] = torch.tensor([0.0, 0.0])
    vx = FEATURE_NAMES.index("current_vx_norm")
    vy = FEATURE_NAMES.index("current_vy_norm")
    # v13 normalization is /20 m/s => +/-0.8 m/s. Every 0.5 s is one
    # native 0.4 m cell, so the test exercises non-zero KTA backtracing.
    rec["features"][0, vx] = 0.04
    rec["features"][0, vy] = -0.04
    rec["native_source_footprint_mask"][0, 17, 24] = 1

    tokens = rec["history_tokens"]
    t0 = _pose(np.pi / 2.0, 0.8, -0.4)
    poses = {
        tokens[0]: np.eye(4, dtype=np.float64),
        tokens[1]: np.eye(4, dtype=np.float64),
        tokens[2]: _pose(np.pi / 2.0, 0.4, -0.8),
        tokens[3]: _pose(np.pi / 2.0, 0.4, -0.8),
        tokens[4]: t0,
        tokens[5]: t0,
    }
    src = _FakeSequenceSource(
        {t: sem for t in tokens},
        {t: valid for t in tokens},
        poses,
    )
    cpu = build_kta_backtrace_3d_crops(
        src, rec, [0], grid=grid, frame_dt_s=0.5
    )
    gpu = build_kta_backtrace_3d_crops(
        src, rec, [0], grid=grid, frame_dt_s=0.5, device="cuda"
    )
    assert torch.equal(cpu["semantics"], gpu["semantics"].cpu())
    assert torch.equal(cpu["valid"], gpu["valid"].cpu())
    assert torch.equal(cpu["source_mask"], gpu["source_mask"].cpu())
    assert torch.equal(cpu["relative_times"], gpu["relative_times"].cpu())


def test_yaw_add_requires_full_deployment_gain():
    decision, recommendation = _decide_yaw(
        coverage_ok=True,
        full_yaw_gain=-0.10,
        matched_yaw_gain=0.80,
        full_d2=0.20,
        full_d3=0.20,
        matched_d2=0.40,
        matched_d3=0.40,
    )
    assert decision == "YAW_OPTIONAL"
    assert recommendation == "YAW_OPTIONAL"

    decision, recommendation = _decide_yaw(
        coverage_ok=True,
        full_yaw_gain=0.60,
        matched_yaw_gain=0.10,
        full_d2=0.05,
        full_d3=0.07,
        matched_d2=0.01,
        matched_d3=0.02,
    )
    assert decision == "ADD_YAW_HEAD"
    assert recommendation == "ADD_YAW_HEAD"


def test_yaw_low_coverage_is_inconclusive():
    decision, recommendation = _decide_yaw(
        coverage_ok=False,
        full_yaw_gain=1.0,
        matched_yaw_gain=1.0,
        full_d2=1.0,
        full_d3=1.0,
        matched_d2=1.0,
        matched_d3=1.0,
    )
    assert decision == "INCONCLUSIVE_LOW_MATCH_COVERAGE"
    assert recommendation == "YAW_OPTIONAL"


def test_summary_identity_keeps_train_cache_and_branch_seed(tmp_path):
    report = {
        "protocol": "p0_f9_v17_local_stwm_rigid_transport_eval_v3",
        "checkpoint_protocol": "p0_f9_v17_local_spatial_temporal_world_model_v1",
        "backtrace3d_enabled": False,
        "variant": "RL",
        "use_representation": True,
        "overlap_weight": 0.25,
        "local_stwm_cache": "/val/nativefp.pt",
        "p0f9_cache": "/val/p0f9",
        "num_windows": 128,
        "target_contract": "target",
        "representation_contract": "representation",
        "occupancy_iou_contract": "occ",
        "a1_write_order_contract": "a1",
        "free_label": 17,
        "model_config": {"d_model": 128},
        "fast_probe_continuation": {
            "protocol": "p0_f9_v17_backtrace3d_paired_probe_train_v1",
            "arm": "control",
            "resume_checkpoint": "/start/epoch_0005.pt",
            "train_cache": "/train/nativefp.pt",
            "start_epoch": 5,
            "local_step": 300,
            "paired_shuffle_seed": 20260910,
            "branch_seed": 20260912,
            "source_batch_size": 256,
            "source_batch_contract": "paired-v1",
            "loss_contract": "L_pos+L_exist+0.25L_overlap",
            "branch_gamma_warmup_steps": 20,
            "scene_ce": False,
            "msp_routing": False,
        },
        "reports": {
            "local_stwm_center_always_source_order": {
                "occupancy": {"IoU": 50.0},
                "overall": {"mIoU": 40.0},
                "moving": {
                    "mIoU": 20.0,
                    "per_horizon": {
                        "1.0": {"mIoU": 30.0},
                        "2.0": {"mIoU": 20.0},
                        "3.0": {"mIoU": 10.0},
                    },
                },
            }
        },
        "diagnostics": {"learned_ade_m": 0.5, "learned_fde_m": 1.0},
    }
    path = tmp_path / "control300.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    row = _load_summary_row(path, expected_arm="control", expected_step=300)
    assert row["identity"]["train_cache"] == "/train/nativefp.pt"
    assert row["identity"]["branch_seed"] == 20260912
