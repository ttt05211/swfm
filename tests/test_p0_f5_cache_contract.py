import pytest
import torch

from real_motion.msp_wm_cache import collate_msp_wm, validate_msp_wm_sample


def _sample():
    return {
        "sample_id": "scene-x:token",
        "scene_name": "scene-x",
        "full_history_latent": torch.zeros(6, 16, 50, 50),
        "anchor_future_latent": torch.zeros(6, 16, 50, 50),
        "repair_target_latent": torch.ones(6, 16, 50, 50),
        "window_origins": torch.tensor([[0, 0], [20, 20]], dtype=torch.long),
        "window_valid": torch.tensor([True, True]),
        "msp_write_support_latent": torch.zeros(6, 50, 50, dtype=torch.bool),
        "trajectory": torch.zeros(12, 2),
    }


def test_v3_sample_requires_encoded_repair_target():
    sample = _sample()
    assert validate_msp_wm_sample(
        sample,
        topk=2,
        require_full_history=True,
        require_write_support=True,
        require_repair_target=True,
    )

    wrong = dict(sample)
    wrong["gt_future_latent"] = wrong.pop("repair_target_latent")
    with pytest.raises(KeyError):
        validate_msp_wm_sample(
            wrong,
            topk=2,
            require_full_history=True,
            require_write_support=True,
            require_repair_target=True,
        )


def test_collate_keeps_repair_target_not_full_gt_target():
    a = _sample()
    b = _sample()
    b["sample_id"] = "scene-y:token"
    b["scene_name"] = "scene-y"
    out = collate_msp_wm([a, b])
    assert tuple(out["repair_target_latent"].shape) == (2, 6, 16, 50, 50)
    assert "gt_future_latent" not in out
