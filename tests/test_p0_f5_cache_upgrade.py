import torch
import pytest

from real_motion.cache_upgrade import (
    P0_F5_LOSS_CONTRACT,
    P0_F5_REPAIR_CONTRACT,
    P0_F5_TARGET,
    REUSED_TENSOR_KEYS,
    build_upgraded_sample,
    make_p0_f5_upgrade_metadata,
    validate_p0_f4_upgrade_source,
)
from real_motion.msp_wm_cache import MSP_WM_CACHE_VERSION_V2


def _meta():
    return {
        "anchor_contract": "strong_w2det_occ_only_v1",
        "history_contract": "full_native_occ_history_6f",
        "target": "full_future_gt_latent",
        "topk": 2,
        "window_hw": [20, 20],
        "context_hw": [40, 40],
        "vae_mode": "mean",
        "include_eval_payload": False,
        "slot_compute_ratio": 0.315,
    }


def _sample():
    return {
        "sample_id": "scene-0001:abc",
        "scene_name": "scene-0001",
        "full_history_latent": torch.randn(6, 16, 50, 50),
        "anchor_future_latent": torch.randn(6, 16, 50, 50),
        "gt_future_latent": torch.randn(6, 16, 50, 50),
        "window_origins": torch.tensor([[0, 0], [20, 20]], dtype=torch.long),
        "window_valid": torch.tensor([True, True]),
        "msp_write_support_latent": torch.zeros(6, 50, 50, dtype=torch.bool),
        "trajectory": torch.randn(12, 2),
    }


def test_upgrade_metadata_changes_only_target_contract_fields():
    source = _meta()
    validate_p0_f4_upgrade_source(MSP_WM_CACHE_VERSION_V2, source)
    upgraded = make_p0_f5_upgrade_metadata(
        source,
        source_cache="/tmp/p0_f4",
        source_index_sha256="deadbeef",
    )
    assert upgraded["repair_endpoint_contract"] == P0_F5_REPAIR_CONTRACT
    assert upgraded["loss_contract"] == P0_F5_LOSS_CONTRACT
    assert upgraded["target"] == P0_F5_TARGET
    assert upgraded["latent_loss_mask"] == "none"
    assert upgraded["slot_compute_ratio"] == source["slot_compute_ratio"]
    assert upgraded["incremental_reused_tensor_keys"] == list(REUSED_TENSOR_KEYS)
    assert upgraded["incremental_new_gpu_encode"] == ["repair_target_latent"]


def test_upgraded_sample_reuses_frozen_tensors_and_drops_full_gt_target():
    source = _sample()
    repair_latent = torch.randn_like(source["anchor_future_latent"])
    out = build_upgraded_sample(source, repair_latent)
    assert "gt_future_latent" not in out
    assert torch.equal(out["repair_target_latent"], repair_latent)
    for key in REUSED_TENSOR_KEYS:
        assert torch.equal(out[key], source[key])
        assert out[key] is not source[key]


def test_upgraded_val_sample_preserves_eval_payload_and_adds_endpoint():
    source = _sample()
    source["eval_future_gt_occ"] = torch.zeros(6, 8, 8, 2, dtype=torch.uint8)
    source["eval_strong_anchor_occ"] = torch.ones(6, 8, 8, 2, dtype=torch.uint8)
    source["eval_gt_moving_support"] = torch.zeros(6, 8, 8, dtype=torch.bool)
    repair_occ = torch.full((6, 8, 8, 2), 17, dtype=torch.uint8)
    repair_latent = torch.randn_like(source["anchor_future_latent"])
    out = build_upgraded_sample(
        source,
        repair_latent,
        repair_target_occ=repair_occ,
    )
    assert torch.equal(out["eval_future_gt_occ"], source["eval_future_gt_occ"])
    assert torch.equal(out["eval_strong_anchor_occ"], source["eval_strong_anchor_occ"])
    assert torch.equal(out["eval_gt_moving_support"], source["eval_gt_moving_support"])
    assert torch.equal(out["eval_repair_target_occ"], repair_occ)


def test_wrong_source_contract_is_rejected():
    bad = _meta()
    bad["topk"] = 3
    with pytest.raises(RuntimeError, match="Top-2"):
        validate_p0_f4_upgrade_source(MSP_WM_CACHE_VERSION_V2, bad)
