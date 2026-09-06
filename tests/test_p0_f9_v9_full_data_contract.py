from pathlib import Path

import torch

from tools.real_motion.train_p0_f9_v9_full_m import (
    scene_balanced_epoch_indices,
    schedule_shape,
)


class _ToyDataset:
    def __init__(self):
        self.entries = [
            {"sample_id": "a0", "scene_name": "a"},
            {"sample_id": "a1", "scene_name": "a"},
            {"sample_id": "a2", "scene_name": "a"},
            {"sample_id": "b0", "scene_name": "b"},
        ]

    def __len__(self):
        return len(self.entries)


def test_full_schedule_is_frozen_by_schedule_epochs():
    s = schedule_shape(20_430, 8, 10)
    assert s["steps_per_epoch"] == 2554
    assert s["total_schedule_steps"] == 25_540
    assert s["schedule_epochs"] == 10


def test_scene_balanced_epoch_sampling_is_deterministic_and_epoch_specific():
    ds = _ToyDataset()
    a = scene_balanced_epoch_indices(ds, seed=20260906, epoch=1)
    b = scene_balanced_epoch_indices(ds, seed=20260906, epoch=1)
    c = scene_balanced_epoch_indices(ds, seed=20260906, epoch=2)
    assert a == b
    assert len(a) == len(ds)
    assert a != c
    assert all(0 <= i < len(ds) for i in a)


def test_v9_trainer_is_official_init_single_weighted_fm_no_ordered_or_semantic():
    root = Path(__file__).resolve().parents[1]
    text = (root / "tools/real_motion/train_p0_f9_v9_full_m.py").read_text()
    assert 'PROTOCOL = "p0_f9_v9_full_data_m_official_init_v1"' in text
    assert "load_shape_safe(model.transition, a.upstream_ckpt" in text
    assert "parent-checkpoint" not in text
    assert "normalized_motion_weighted_mse" in text
    assert "total_loss = normalized_motion_weighted_mse" in text
    assert "ordered_context=False" in text
    assert "semantic_loss_for_endpoint" not in text
    assert "lovasz_weight" not in text
    assert '"best": False' in text
    assert '"latest": True' in text


def test_full_cache_builder_is_direct_native_target_not_repair_intermediate():
    root = Path(__file__).resolve().parents[1]
    text = (root / "tools/real_motion/build_p0_f9_full_native_cache.py").read_text()
    assert "direct_full_data_build" in text
    assert '"target": "absolute_gt_future_vae_latent"' in text
    assert '"flow_source": "gaussian_noise_not_anchor"' in text
    assert 'mode="sample"' in text
    assert "repair_target_latent" not in text
    assert "build_dynamic_repair_endpoint" not in text


def test_full_msp_builder_explicitly_requests_all_eligible_windows():
    root = Path(__file__).resolve().parents[1]
    text = (root / "tools/real_motion/build_p0_f9_full_msp_cache.py").read_text()
    assert "max_windows=None" in text
    assert '"all_eligible_windows": True' in text
    assert "native_ids" in text
    assert "set(selected_ids) != native_ids" in text
