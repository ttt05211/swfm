from __future__ import annotations

import numpy as np
import torch

from real_motion.local_st_world_model_v17 import LocalSTWMV17Config
from real_motion.local_st_world_model_v18_se2 import LocalSpatialTemporalWorldModelV18SE2
from real_motion.motion_transport import FEATURE_DIM, FUTURE_FRAMES, HISTORY_FRAMES
from real_motion.v20_history_world import (
    CanonicalLattice,
    DynamicResponsibility,
    align_history_once_to_canonical,
    assert_zero_contribution_identity,
    audit_inference_inputs,
    future_union_query_mask,
    partition_future_dynamic_instances,
    protected_add_only,
)
from real_motion.v20_scene_model import HistoricalEvidence3DEncoder, V20SceneConfig
from real_motion.v20_stage0_voxel_semantic import (
    LABEL_SIDECAR_PROTOCOL,
    PerZSemanticHead,
    decode_per_z_semantic,
    frozen_factorized_support,
    per_z_semantic_loss,
    validate_stage0_sidecar_pair,
)
from real_motion.v20_training import choose_birth_query_count


def test_v18_optional_latents_do_not_change_default_predictions():
    torch.manual_seed(7)
    cfg = LocalSTWMV17Config(
        d_model=32,
        semantic_dim=8,
        heads=4,
        blocks=1,
        decoder_blocks=1,
        tube_hw=8,
    )
    model = LocalSpatialTemporalWorldModelV18SE2(cfg).eval()
    B = 2
    features = torch.randn(B, FEATURE_DIM)
    tube = torch.randint(0, 18, (B, HISTORY_FRAMES, cfg.tube_hw, cfg.tube_hw))
    kta = torch.randn(B, FUTURE_FRAMES, 2)
    fm = torch.randn(B, HISTORY_FRAMES, 5)
    mask = torch.randint(0, 2, tube.shape)
    with torch.no_grad():
        base = model(features, tube, kta, fm, mask)
        exposed = model(features, tube, kta, fm, mask, return_latents=True)
    assert set(base) == {"residual_xy_m", "existence_logits", "yaw_delta_rad"}
    for key in base:
        assert torch.equal(base[key], exposed[key])
    assert exposed["future_transport_queries"].shape == (B, FUTURE_FRAMES, cfg.d_model)
    assert exposed["history_source_context"].shape == (B, cfg.d_model)


def test_observed_free_and_unknown_are_distinct_after_one_alignment():
    lattice = CanonicalLattice((0, 0, 0), (1, 1, 1), (3, 3, 2))
    sem = np.full((HISTORY_FRAMES, 2, 2, 1), 17, dtype=np.uint8)
    obs = np.zeros_like(sem, dtype=bool)
    obs[:, 0, 0, 0] = True
    sem[:, 1, 0, 0] = 4
    obs[:, 1, 0, 0] = True
    poses = np.repeat(np.eye(4)[None], HISTORY_FRAMES, axis=0)
    ev = align_history_once_to_canonical(
        lattice,
        history_semantic=sem,
        history_observed=obs,
        history_ego_to_world=poses,
        t0_ego_to_world=np.eye(4),
        native_origin_xyz_m=(0, 0, 0),
        native_voxel_size_xyz_m=(1, 1, 1),
    )
    assert not np.any(ev.observed_free & ev.unknown)
    assert ev.observed_free[:, 0, 0, 0].all()
    assert ev.semantic[:, 1, 0, 0].tolist() == [4] * HISTORY_FRAMES


def test_future_union_query_reports_oob_instead_of_silent_clipping():
    lattice = CanonicalLattice((-1, -1, -1), (1, 1, 1), (2, 2, 2))
    poses = np.repeat(np.eye(4)[None], FUTURE_FRAMES, axis=0)
    poses[-1, 0, 3] = 100.0
    report = future_union_query_mask(
        lattice,
        future_ego_to_canonical=poses,
        native_shape_xyz=(2, 2, 2),
        native_origin_xyz_m=(-1, -1, -1),
        native_voxel_size_xyz_m=(1, 1, 1),
    )
    assert report.out_of_bounds_voxels > 0
    assert report.out_of_bounds_fraction > 0.0


def test_dynamic_partition_birth_is_not_t0_absence_only():
    p = partition_future_dynamic_instances(
        num_future_instances=4,
        matches_t0={0: True},
        matches_earlier_history={1: True},
        ambiguous={3},
    )
    assert p.labels.tolist() == [
        int(DynamicResponsibility.CURRENT_ANCESTRAL),
        int(DynamicResponsibility.DORMANT_ANCESTRAL),
        int(DynamicResponsibility.BIRTH),
        int(DynamicResponsibility.IGNORE),
    ]


def test_protected_compositor_priority_and_zero_identity():
    free = 17
    base = torch.full((1, 1, 2, 2, 2), free, dtype=torch.uint8)
    base[..., 0, 0, 0] = 4
    dormant = torch.full_like(base, free); dormant[..., 0, 0, 1] = 5; dormant[..., 0, 0, 0] = 6
    birth = torch.full_like(base, free); birth[..., 0, 0, 1] = 7; birth[..., 0, 1, 0] = 8
    static = torch.full_like(base, free); static[..., 0, 1, 0] = 9; static[..., 1, 0, 0] = 10
    out = protected_add_only(base, dormant=dormant, birth=birth, static_world=static)
    assert int(out[..., 0, 0, 0]) == 4
    assert int(out[..., 0, 0, 1]) == 5
    assert int(out[..., 0, 1, 0]) == 8
    assert int(out[..., 1, 0, 0]) == 10
    identity = protected_add_only(base)
    assert_zero_contribution_identity(base, identity)


def test_inference_leakage_audit_rejects_future_supervision():
    audit_inference_inputs({"history_occ": object(), "future_ego_to_world": object()})
    try:
        audit_inference_inputs({"future_occ": object()})
    except RuntimeError:
        pass
    else:
        raise AssertionError("future OCC leakage was not rejected")


def test_scene_encoder_keeps_z_spatial():
    cfg = V20SceneConfig(semantic_dim=4, base_dim=8)
    enc = HistoricalEvidence3DEncoder(cfg)
    sem = torch.randint(0, 18, (1, HISTORY_FRAMES, 16, 12, 8))
    obs = torch.rand_like(sem.float()) > 0.2
    free = obs & sem.eq(17)
    out = enc(sem, obs, free)
    assert out.ndim == 5
    assert out.shape[-1] > 1


def test_birth_q_selection_reports_gt_truncation():
    s = choose_birth_query_count([0, 0, 1, 1, 2, 3, 5], max_truncation_fraction=0.1)
    assert s["Q"] >= 1
    assert s["truncated_gt_fraction"] <= 0.1 + 1e-12
    assert s["windows_ge3"] == 2


def test_stage0_future_gt_changes_loss_not_model_inputs_or_geometry():
    torch.manual_seed(17)
    B, Fh, Z, H, W = 1, 2, 3, 2, 2
    frozen = {
        "presence_logits": torch.full((B, Fh, H, W), 8.0),
        "vertical_logits": torch.full((B, Fh, Z, H, W), 8.0),
    }
    new_fov = torch.ones((B, Fh, H, W), dtype=torch.bool)
    base_free = torch.ones((B, Fh, Z, H, W), dtype=torch.bool)

    support_a = frozen_factorized_support(
        frozen,
        new_fov_mask=new_fov,
        base_free=base_free,
        presence_threshold=0.5,
        vertical_threshold=0.5,
    )
    # Future GT is deliberately changed after the inference inputs are fixed.
    static_ids = [cid for cid in range(17) if cid not in set(DYNAMIC_CLASS_IDS)]
    assert len(static_ids) >= 2
    target_a = torch.full((B * Fh, Z, H, W), static_ids[0], dtype=torch.long)
    target_b = torch.full((B * Fh, Z, H, W), static_ids[1], dtype=torch.long)

    support_b = frozen_factorized_support(
        frozen,
        new_fov_mask=new_fov,
        base_free=base_free,
        presence_threshold=0.5,
        vertical_threshold=0.5,
    )
    assert torch.equal(support_a, support_b)

    feature = torch.randn(B * Fh, 4, H, W)
    head = PerZSemanticHead(
        bev_feature_channels=4,
        vertical_bins=Z,
        hidden_dim=4,
    ).eval()
    with torch.no_grad():
        pred_a = decode_per_z_semantic(
            head(feature, support_a.reshape(B * Fh, Z, H, W)),
            support_a.reshape(B * Fh, Z, H, W),
            free_label=17,
        )
        pred_b = decode_per_z_semantic(
            head(feature, support_b.reshape(B * Fh, Z, H, W)),
            support_b.reshape(B * Fh, Z, H, W),
            free_label=17,
        )
    assert torch.equal(pred_a, pred_b)
    assert torch.equal(pred_a.ne(17), support_a.reshape(B * Fh, Z, H, W))

    logits = torch.zeros((B * Fh, 17, Z, H, W))
    logits[:, static_ids[0]] = 3.0
    logits[:, static_ids[1]] = -2.0
    supervised = support_a.reshape(B * Fh, Z, H, W)
    loss_a = per_z_semantic_loss(logits, target_a, supervised)
    loss_b = per_z_semantic_loss(logits, target_b, supervised)
    assert not torch.equal(loss_a, loss_b)


def test_stage0_sidecar_pair_is_label_only_and_identity_checked():
    v19 = {
        "scene_name": ["scene-a", "scene-b"],
        "t0_token": ["t0-a", "t0-b"],
        "future_aligned_semantic": torch.zeros(2, 1),
    }
    labels = {
        "protocol": LABEL_SIDECAR_PROTOCOL,
        "parent_v19_shard": "shard_00000.pt",
        "count": 2,
        "voxel_semantic_target": torch.zeros((2, 6, 2, 2, 2), dtype=torch.uint8),
        "scene_name": ["scene-a", "scene-b"],
        "t0_token": ["t0-a", "t0-b"],
    }
    assert validate_stage0_sidecar_pair(
        v19,
        labels,
        expected_parent_shard="shard_00000.pt",
    ) == 2

    leaked = dict(labels)
    leaked["vertical_target_bits"] = torch.zeros(1)
    try:
        validate_stage0_sidecar_pair(
            v19,
            leaked,
            expected_parent_shard="shard_00000.pt",
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("Stage-0 sidecar accepted a copied V19/GT tensor")
