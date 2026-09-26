from __future__ import annotations

import numpy as np
import torch

from real_motion.metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS
from real_motion.local_st_world_model_v17 import LocalSTWMV17Config
from real_motion.local_st_world_model_v18_se2 import LocalSpatialTemporalWorldModelV18SE2
from real_motion.motion_transport import FEATURE_DIM, FUTURE_FRAMES, HISTORY_FRAMES
from real_motion.v20_history_world import (
    CanonicalLattice,
    DynamicResponsibility,
    align_history_once_to_canonical,
    align_sparse_history_once_to_canonical,
    observed_native_points,
    assert_zero_contribution_identity,
    audit_inference_inputs,
    future_union_query_mask,
    partition_future_dynamic_instances,
    protected_add_only,
    _rasterize_observed_frame_vectorized,
)
from real_motion.v20_stage1_codec import (
    pack_bool,
    pack_history_semantic,
    pack_static_supervision,
    unpack_bool,
    unpack_history_semantic,
    unpack_static_indices_and_labels,
)
from real_motion.v20_scene_model import (
    BirthQueryHead,
    HistoricalEvidence3DEncoder,
    V20HistoryWorldModel,
    V20SceneConfig,
)
from real_motion.v20_dormant import render_dormant_sources
from real_motion.v19_scene_memory import SourceTrack
from real_motion.v20_stage0_voxel_semantic import (
    LABEL_SIDECAR_PROTOCOL,
    PerZSemanticHead,
    decode_per_z_semantic,
    dense_targets_from_sparse,
    frozen_factorized_support,
    per_z_semantic_loss,
    validate_stage0_sidecar_pair,
)
from real_motion.v20_training import choose_birth_query_count
from real_motion.v20_birth import render_birth_queries


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
        "semantic_values": torch.tensor([1, 2, 3], dtype=torch.uint8),
        "semantic_offsets": torch.tensor([0, 1, 3], dtype=torch.int64),
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


def test_stage0_sparse_labels_expand_in_parent_mask_order():
    mask = torch.zeros((2, 1, 2, 2, 2), dtype=torch.bool)
    mask[0, 0, 0, 0, 1] = True
    mask[1, 0, 0, 1, 0] = True
    mask[1, 0, 1, 1, 1] = True
    values = torch.tensor([4, 6, 9], dtype=torch.uint8)
    counts = torch.tensor([1, 2], dtype=torch.int64)
    dense = dense_targets_from_sparse(
        mask,
        values,
        counts,
        ignore_label=255,
    )
    assert dense.shape == mask.shape
    assert dense[0, 0, 0, 0, 1].item() == 4
    assert dense[1, 0, 0, 1, 0].item() == 6
    assert dense[1, 0, 1, 1, 1].item() == 9
    assert torch.all(dense[~mask] == 255)


def test_stage0_sidecar_allows_strict_parent_prefix_for_smoke():
    v19 = {
        "scene_name": ["scene-a", "scene-a", "scene-b"],
        "t0_token": ["t0-a", "t0-b", "t0-c"],
    }
    labels = {
        "protocol": LABEL_SIDECAR_PROTOCOL,
        "parent_v19_shard": "shard_00000.pt",
        "count": 2,
        "semantic_values": torch.tensor([4, 5], dtype=torch.uint8),
        "semantic_offsets": torch.tensor([0, 1, 2], dtype=torch.int64),
        "scene_name": ["scene-a", "scene-a"],
        "t0_token": ["t0-a", "t0-b"],
    }
    assert validate_stage0_sidecar_pair(
        v19,
        labels,
        expected_parent_shard="shard_00000.pt",
    ) == 2


def test_dormant_renderer_uses_kta_plus_residual_and_existence():
    class Grid:
        x_min = -2.0
        y_min = -2.0
        z_min = -1.0
        voxel_size = (1.0, 1.0, 1.0)
        shape_hwd = (6, 6, 3)

    centers = np.zeros((HISTORY_FRAMES, 3), dtype=np.float64)
    centers[:, 0] = np.arange(HISTORY_FRAMES, dtype=np.float64) * 0.1
    valid = np.ones(HISTORY_FRAMES, dtype=bool)
    tr = SourceTrack(
        track_id=7,
        class_id=int(DYNAMIC_CLASS_IDS[0]),
        canonical_xyz_local=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
        centers_world=centers,
        valid_history=valid,
        velocity_world=np.zeros(3, dtype=np.float64),
        last_observed_frame=HISTORY_FRAMES - 2,
        confidence=1.0,
        provenance="observed_history",
        current_component_index=None,
        last_component_voxel_count=1,
    )
    out = {
        "residual_xy_m": torch.zeros((1, FUTURE_FRAMES, 2)),
        "yaw_delta_rad": torch.zeros((1, FUTURE_FRAMES)),
        "existence_logits": torch.full((1, FUTURE_FRAMES), -10.0),
    }
    out["existence_logits"][0, 0] = 10.0
    kta = torch.zeros((1, FUTURE_FRAMES, 2))
    kta[0, 0, 0] = 1.0
    poses = np.repeat(np.eye(4)[None], FUTURE_FRAMES, axis=0)
    report = render_dormant_sources(
        out,
        [tr],
        kta_displacement_xy_m=kta,
        t0_ego_to_world=np.eye(4),
        future_ego_to_world=poses,
        grid=Grid(),
        frame_dt_s=0.5,
        existence_threshold=0.5,
        free_label=17,
    )
    assert report.active_track_horizons == 1
    assert report.rendered_voxels == 1
    assert np.count_nonzero(report.future_semantic != 17) == 1
    assert np.count_nonzero(report.future_semantic[1:] != 17) == 0


def test_birth_head_accepts_spatial_scene_and_v18_source_context():
    torch.manual_seed(19)
    cfg = V20SceneConfig(
        semantic_dim=4,
        base_dim=8,
        source_dim=16,
        birth_queries=3,
        birth_shape_size_xyz=(4, 3, 2),
    )
    head = BirthQueryHead(scene_dim=16, cfg=cfg).eval()
    scene = torch.randn(1, 16, 8, 8, 4)
    cur = torch.randn(1, 2, 16)
    cur_pos = torch.randn(1, 2, 3)
    fut = torch.randn(1, 2, FUTURE_FRAMES, 16)
    fut_pos = torch.randn(1, 2, FUTURE_FRAMES, 3)
    with torch.no_grad():
        out = head(
            scene,
            current_source_tokens=cur,
            current_source_xyz_norm=cur_pos,
            future_source_tokens=fut,
            future_source_xyz_norm=fut_pos,
        )
    assert out["class_logits"].shape == (1, 3, len(DYNAMIC_CLASS_IDS) + 1)
    assert out["existence_logits"].shape == (1, 3, FUTURE_FRAMES)
    assert out["trajectory_xyz_yaw"].shape == (1, 3, FUTURE_FRAMES, 4)
    assert out["shape_logits"].shape == (1, 3, 4, 3, 2)
    # Zero-contribution init must still select no-object.
    assert torch.all(out["class_logits"].argmax(-1) == len(DYNAMIC_CLASS_IDS))


def test_birth_renderer_suppresses_duplicate_current_source_horizon():
    cid = int(DYNAMIC_CLASS_IDS[0])
    cfg = V20SceneConfig(
        birth_queries=1,
        birth_shape_size_xyz=(1, 1, 1),
    )
    outputs = {
        "class_logits": torch.full((1, 1, len(DYNAMIC_CLASS_IDS) + 1), -8.0),
        "existence_logits": torch.full((1, 1, FUTURE_FRAMES), -8.0),
        "trajectory_xyz_yaw": torch.zeros((1, 1, FUTURE_FRAMES, 4)),
        "shape_logits": torch.full((1, 1, 1, 1, 1), 8.0),
    }
    local = list(DYNAMIC_CLASS_IDS).index(cid)
    outputs["class_logits"][0, 0, local] = 8.0
    outputs["existence_logits"][0, 0, 0] = 8.0
    poses = np.repeat(np.eye(4)[None], FUTURE_FRAMES, axis=0)
    src_xy = np.zeros((1, FUTURE_FRAMES, 2), dtype=np.float32)
    src_ex = np.zeros((1, FUTURE_FRAMES), dtype=np.float32)
    src_ex[0, 0] = 1.0
    report = render_birth_queries(
        outputs,
        future_ego_to_canonical=poses,
        native_shape_xyz=(4, 4, 2),
        native_origin_xyz_m=(-2.0, -2.0, -1.0),
        native_voxel_size_xyz_m=(1.0, 1.0, 1.0),
        shape_voxel_size_m=1.0,
        current_source_future_xy_t0_m=src_xy,
        current_source_existence_prob=src_ex,
        current_source_class_id=np.asarray([cid], dtype=np.int64),
        duplicate_distance_m=1.0,
    )
    assert report.duplicate_suppressed_query_horizons == 1
    assert report.rendered_voxels == 0
    assert np.all(report.future_semantic == 17)


def test_birth_matching_uses_strict_birth_records_and_center_distance():
    cid = int(DYNAMIC_CLASS_IDS[0])
    local = list(DYNAMIC_CLASS_IDS).index(cid)
    outputs = {
        "class_logits": torch.full((1, 1, len(DYNAMIC_CLASS_IDS) + 1), -8.0),
        "existence_logits": torch.full((1, 1, FUTURE_FRAMES), -8.0),
        "trajectory_xyz_yaw": torch.zeros((1, 1, FUTURE_FRAMES, 4)),
    }
    outputs["class_logits"][0, 0, local] = 8.0
    outputs["existence_logits"][0, 0, 0] = 8.0
    row = {
        "dynamic_supervision": [
            {
                "responsibility_name": "BIRTH",
                "class_id": cid,
                "existence": [1, 0, 0, 0, 0, 0],
                "trajectory_xyz_yaw_t0": [
                    [0.5, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
            },
            {
                "responsibility_name": "CURRENT_ANCESTRAL",
                "class_id": cid,
                "existence": [1, 1, 1, 1, 1, 1],
                "trajectory_xyz_yaw_t0": [[0.0, 0.0, 0.0, 0.0]] * 6,
            },
        ]
    }
    from real_motion.v20_birth import birth_query_match_counts
    m = birth_query_match_counts(
        outputs,
        row,
        existence_threshold=0.5,
        distance_threshold_m=1.0,
    )
    assert m["gt_birth_instances"] == 1
    assert m["predicted_birth_queries"] == 1
    assert m["distance_matched"] == 1


def test_vectorized_history_raster_matches_legacy_collision_semantics():
    shape = (2, 2, 1)
    # Six native samples; several intentionally quantize to the same cells.
    idx = np.asarray(
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
        ],
        dtype=np.int64,
    )
    in_bounds = np.ones(6, dtype=bool)
    sem = np.asarray([17, 4, 5, 17, 6, 7], dtype=np.uint8)
    obs = np.asarray([1, 1, 1, 1, 1, 0], dtype=bool)

    ref_sem = np.full(shape, 17, dtype=np.uint8)
    ref_obs = np.zeros(shape, dtype=bool)
    ref_conflict = np.zeros(shape, dtype=bool)
    for cell, label in zip(idx[in_bounds & obs], sem[in_bounds & obs]):
        key = tuple(int(x) for x in cell)
        was = ref_obs[key]
        old = int(ref_sem[key])
        lab = int(label)
        if not was:
            ref_sem[key] = lab
            ref_obs[key] = True
        elif old == 17 and lab != 17:
            ref_sem[key] = lab
        elif old != 17 and lab != 17 and old != lab:
            ref_conflict[key] = True

    got_sem, got_obs, got_conflict = _rasterize_observed_frame_vectorized(
        idx,
        in_bounds,
        sem,
        obs,
        canonical_shape_xyz=shape,
        free_label=17,
    )
    assert np.array_equal(got_sem, ref_sem)
    assert np.array_equal(got_obs, ref_obs)
    assert np.array_equal(got_conflict, ref_conflict)
    assert int(got_sem[0, 0, 0]) == 4
    assert bool(got_conflict[0, 0, 0])
    assert int(got_sem[1, 0, 0]) == 6


def test_fresh_v20_heads_are_zero_contribution_before_training():
    torch.manual_seed(23)
    cfg = V20SceneConfig(
        semantic_dim=4,
        base_dim=8,
        source_dim=16,
        tile_dim=8,
        birth_queries=2,
        birth_shape_size_xyz=(2, 2, 2),
    )
    model = V20HistoryWorldModel(cfg).eval()
    scene = torch.randn(1, model.encoder.output_dim, 2, 2, 2)

    coarse = model.static.forward_coarse(scene)
    assert torch.all(coarse.argmax(1) == 17)

    sample_grid = torch.zeros((1, 2, 2, 2, 3))
    qmask = torch.ones((1, 2, 2, 2), dtype=torch.bool)
    zmask = torch.zeros_like(qmask)
    fine = model.static.refine_tiles(
        scene,
        sample_grid=sample_grid,
        query_mask=qmask,
        seen_mask=zmask,
        t0_missing_mask=zmask,
    )
    assert torch.all(fine.argmax(1) == 17)

    source = torch.randn(1, cfg.source_dim)
    local = torch.randn(1, model.encoder.output_dim)
    dout = model.dormant(source, local)
    assert torch.all(torch.sigmoid(dout["existence_logits"]) < 0.5)
    assert torch.count_nonzero(dout["residual_xy_m"]) == 0
    assert torch.count_nonzero(dout["yaw_delta_rad"]) == 0

    bout = model.birth(scene)
    assert torch.all(
        bout["class_logits"].argmax(-1) == len(DYNAMIC_CLASS_IDS)
    )
    assert torch.all(torch.sigmoid(bout["existence_logits"]) < 0.5)
    assert torch.all(torch.sigmoid(bout["shape_logits"]) < 0.5)

    base = torch.randint(0, 18, (6, 3, 3, 2), dtype=torch.uint8)
    free = torch.full_like(base, 17)
    combined = protected_add_only(
        base,
        dormant=free,
        birth=free,
        static_world=free,
        free_label=17,
    )
    assert_zero_contribution_identity(base, combined)


def test_stage1_v2_history_codec_roundtrip_is_exact():
    rng = np.random.default_rng(31)
    shape = (6, 4, 3, 2)
    obs = rng.random(shape) > 0.45
    sem = rng.integers(0, 17, size=shape, dtype=np.uint8)
    free = obs & (rng.random(shape) > 0.65)
    sem[~obs] = 17
    sem[free] = 17
    occupied = obs & ~free
    # Ensure occupied locations never use the free label.
    sem[occupied] %= 17
    packed = pack_history_semantic(sem, obs, free)
    row = dict(packed)
    obs2 = unpack_bool(pack_bool(obs), shape)
    free2 = unpack_bool(pack_bool(free), shape)
    sem2 = unpack_history_semantic(row, obs2, free2)
    assert np.array_equal(obs2, obs)
    assert np.array_equal(free2, free)
    assert np.array_equal(sem2, sem)


def test_stage1_v2_static_supervision_codec_roundtrip_is_exact():
    rng = np.random.default_rng(37)
    shape = (5, 4, 3)
    gt = rng.integers(0, 18, size=shape, dtype=np.uint8)
    obs = rng.random(shape) > 0.30
    sup = pack_static_supervision(gt, obs)
    idx, labels = unpack_static_indices_and_labels(sup, shape)
    dyn = np.isin(gt, np.asarray(DYNAMIC_CLASS_IDS, dtype=np.uint8))
    valid = obs & ~dyn
    assert np.array_equal(idx, np.argwhere(valid))
    assert np.array_equal(labels.astype(np.uint8), gt[valid])


def test_sparse_observed_history_alignment_matches_full_mapping_reference():
    rng = np.random.default_rng(41)
    native_shape = (4, 3, 2)
    lattice = CanonicalLattice((-1, -1, -1), (1, 1, 1), (7, 7, 5))
    sem = rng.integers(
        0, 18, size=(HISTORY_FRAMES,) + native_shape, dtype=np.uint8
    )
    obs = rng.random(sem.shape) > 0.55
    poses = np.repeat(np.eye(4)[None], HISTORY_FRAMES, axis=0)
    poses[:, 0, 3] = np.linspace(0.0, 0.4, HISTORY_FRAMES)

    got = align_history_once_to_canonical(
        lattice,
        history_semantic=sem,
        history_observed=obs,
        history_ego_to_world=poses,
        t0_ego_to_world=poses[-1],
        native_origin_xyz_m=(0, 0, 0),
        native_voxel_size_xyz_m=(1, 1, 1),
        free_label=17,
    )

    # Reference uses the previous dense-grid mapping formulation.
    local = np.stack(
        np.meshgrid(
            np.arange(native_shape[0], dtype=np.float64) + 0.5,
            np.arange(native_shape[1], dtype=np.float64) + 0.5,
            np.arange(native_shape[2], dtype=np.float64) + 0.5,
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 3)
    ref_sem = np.full_like(got.semantic, 17)
    ref_obs = np.zeros_like(got.observed)
    ref_conflict = np.zeros_like(got.conflict)
    world_to_t0 = np.linalg.inv(poses[-1])
    ref_oob = 0
    for t in range(HISTORY_FRAMES):
        T = world_to_t0 @ poses[t]
        world = local @ T[:3, :3].T + T[:3, 3]
        idx, valid = lattice.world_to_index(world)
        src_obs = obs[t].reshape(-1)
        ref_oob += int((src_obs & ~valid).sum())
        a, b, c = _rasterize_observed_frame_vectorized(
            idx,
            valid,
            sem[t].reshape(-1),
            src_obs,
            canonical_shape_xyz=lattice.shape_xyz,
            free_label=17,
        )
        ref_sem[t], ref_obs[t], ref_conflict[t] = a, b, c
    assert np.array_equal(got.semantic, ref_sem)
    assert np.array_equal(got.observed, ref_obs)
    assert np.array_equal(got.conflict, ref_conflict)
    assert got.out_of_bounds_samples == ref_oob


def test_cached_sparse_history_alignment_is_elementwise_identical():
    rng = np.random.default_rng(43)
    native_shape = (5, 4, 3)
    sem = rng.integers(
        0, 18, size=(HISTORY_FRAMES,) + native_shape, dtype=np.uint8
    )
    obs = rng.random(sem.shape) > 0.58
    poses = np.repeat(np.eye(4)[None], HISTORY_FRAMES, axis=0)
    poses[:, 0, 3] = np.linspace(-0.3, 0.2, HISTORY_FRAMES)
    poses[:, 1, 3] = np.linspace(0.1, -0.2, HISTORY_FRAMES)
    lattice = CanonicalLattice((-4, -4, -2), (0.5, 0.5, 0.5), (20, 20, 12))
    origin = (-1.0, -1.0, -0.5)
    step = (0.5, 0.5, 0.5)

    dense = align_history_once_to_canonical(
        lattice,
        history_semantic=sem,
        history_observed=obs,
        history_ego_to_world=poses,
        t0_ego_to_world=poses[-1],
        native_origin_xyz_m=origin,
        native_voxel_size_xyz_m=step,
        free_label=17,
    )
    xyz, labels = zip(*[
        observed_native_points(
            sem[t], obs[t],
            native_origin_xyz_m=origin,
            native_voxel_size_xyz_m=step,
        )
        for t in range(HISTORY_FRAMES)
    ])
    sparse = align_sparse_history_once_to_canonical(
        lattice,
        history_local_xyz=xyz,
        history_semantic_observed=labels,
        history_ego_to_world=poses,
        t0_ego_to_world=poses[-1],
        free_label=17,
    )
    assert np.array_equal(sparse.semantic, dense.semantic)
    assert np.array_equal(sparse.observed, dense.observed)
    assert np.array_equal(sparse.observed_free, dense.observed_free)
    assert np.array_equal(sparse.unknown, dense.unknown)
    assert np.array_equal(sparse.conflict, dense.conflict)
    assert sparse.out_of_bounds_samples == dense.out_of_bounds_samples


def test_static_training_confusion_supports_free_prediction_column():
    from tools.real_motion.train_p0_f9_v20_static import _miou

    conf = np.zeros((18, 18), dtype=np.int64)
    cid = next(
        i for i in range(17) if i not in set(DYNAMIC_CLASS_IDS)
    )
    # One correct static prediction and one static -> free miss.
    conf[cid, cid] = 1
    conf[cid, 17] = 1
    miou, per = _miou(conf)
    assert abs(per[str(cid)] - 0.5) < 1e-12
    assert np.isfinite(miou)


def test_static_vectorized_aggregate_matches_dict_reference():
    from real_motion.v20_stage1_codec import pack_static_supervision
    from tools.real_motion.train_p0_f9_v20_static import (
        _aggregate_sparse_targets,
    )
    from real_motion.v20_history_world import (
        native_sparse_to_canonical_indices,
    )

    rng = np.random.default_rng(47)
    native_shape = (6, 5, 3)
    lattice = CanonicalLattice((-2, -2, -1), (1, 1, 1), (8, 8, 4))
    rel = np.repeat(np.eye(4)[None], FUTURE_FRAMES, axis=0)
    rel[:, 0, 3] = np.linspace(0.0, 0.7, FUTURE_FRAMES)
    rel[:, 1, 3] = np.linspace(0.2, -0.4, FUTURE_FRAMES)

    sups = []
    dense_pairs = []
    for fi in range(FUTURE_FRAMES):
        gt = rng.integers(0, 18, size=native_shape, dtype=np.uint8)
        obs = rng.random(native_shape) > 0.25
        sup = pack_static_supervision(gt, obs)
        sups.append(sup)

    row = {
        "future_ego_to_t0": torch.from_numpy(rel.astype(np.float32)),
        "static_supervision": sups,
    }
    got_i, got_y = _aggregate_sparse_targets(
        row,
        lattice,
        native_shape,
        (-1.5, -1.5, -0.5),
        (1.0, 1.0, 1.0),
    )

    from real_motion.v20_stage1_codec import (
        unpack_static_indices_and_labels,
    )
    table = {}
    for fi, sup in enumerate(sups):
        native, labels = unpack_static_indices_and_labels(sup, native_shape)
        idx, valid = native_sparse_to_canonical_indices(
            lattice,
            native_indices_xyz=native,
            ego_to_canonical=rel[fi],
            native_origin_xyz_m=(-1.5, -1.5, -0.5),
            native_voxel_size_xyz_m=(1.0, 1.0, 1.0),
        )
        for cell, lab in zip(idx[valid], labels[valid]):
            key = tuple(int(x) for x in cell)
            lab = int(lab)
            old = table.get(key)
            if old is None:
                table[key] = lab
            elif old != lab:
                table[key] = -1
    ref_i = np.asarray(
        [k for k, v in table.items() if v >= 0], dtype=np.int64
    )
    ref_y = np.asarray(
        [v for v in table.values() if v >= 0], dtype=np.int64
    )
    assert np.array_equal(got_i, ref_i)
    assert np.array_equal(got_y, ref_y)


def test_static_paired_aggregation_matches_two_independent_aggregations():
    from real_motion.v20_stage1_codec import pack_static_supervision
    from tools.real_motion.train_p0_f9_v20_static import (
        _aggregate_sparse_targets,
        _aggregate_sparse_targets_pair,
    )

    rng = np.random.default_rng(53)
    native_shape = (7, 6, 3)
    coarse = CanonicalLattice((-3, -3, -1), (1.0, 1.0, 1.0), (10, 10, 5))
    high = CanonicalLattice((-3, -3, -1), (0.5, 0.5, 0.5), (20, 20, 10))
    rel = np.repeat(np.eye(4)[None], FUTURE_FRAMES, axis=0)
    rel[:, 0, 3] = np.linspace(-0.4, 0.5, FUTURE_FRAMES)
    rel[:, 1, 3] = np.linspace(0.3, -0.2, FUTURE_FRAMES)
    sups = []
    for _ in range(FUTURE_FRAMES):
        gt = rng.integers(0, 18, size=native_shape, dtype=np.uint8)
        obs = rng.random(native_shape) > 0.2
        sups.append(pack_static_supervision(gt, obs))
    row = {
        "future_ego_to_t0": torch.from_numpy(rel.astype(np.float32)),
        "static_supervision": sups,
    }
    args = (native_shape, (-1.5, -1.5, -0.5), (0.5, 0.5, 0.5))
    ci, cy = _aggregate_sparse_targets(row, coarse, *args)
    hi, hy = _aggregate_sparse_targets(row, high, *args)
    pci, pcy, phi, phy = _aggregate_sparse_targets_pair(
        row, coarse, high, *args
    )
    assert np.array_equal(pci, ci)
    assert np.array_equal(pcy, cy)
    assert np.array_equal(phi, hi)
    assert np.array_equal(phy, hy)


def test_equal_tile_weighted_ce_matches_per_tile_loop():
    from tools.real_motion.train_p0_f9_v20_static import (
        _equal_tile_weighted_ce,
    )

    g = torch.Generator().manual_seed(59)
    logits = torch.randn(13, 18, generator=g)
    targets = torch.tensor([0, 1, 2, 17, 3, 0, 5, 17, 1, 2, 3, 5, 0])
    tile_ids = torch.tensor([0,0,0,0,1,1,1,1,1,2,2,2,2])
    weights = torch.linspace(0.5, 2.0, 18)
    got = _equal_tile_weighted_ce(
        logits, targets, tile_ids, weights, 3
    )
    ref = sum(
        torch.nn.functional.cross_entropy(
            logits[tile_ids == tid],
            targets[tile_ids == tid],
            weight=weights,
        )
        for tid in range(3)
    )
    assert torch.allclose(got, ref, atol=1e-6, rtol=1e-6)


def test_static_prepared_rows_preserve_order_with_threads(tmp_path):
    from tools.real_motion.train_p0_f9_v20_static import _iter_prepared_rows
    # Contract-level source check: ordered prefetch uses a deque and yields
    # future results from the left, never completion order.
    import inspect
    src = inspect.getsource(_iter_prepared_rows)
    assert "pending.popleft()" in src
    assert "yield result" in src
    assert "ThreadPoolExecutor" in src


def test_static_resume_checkpoint_contract_is_present():
    import inspect
    from tools.real_motion import train_p0_f9_v20_static as m
    src = inspect.getsource(m.main)
    assert '"optimizer_state_dict"' in src
    assert '"training_progress"' in src
    assert '"rng_state"' in src
    assert '"resume_latest.pt"' in src


def test_runtime_query_mask_reuses_render_geometry_exactly():
    from real_motion.v20_history_world import (
        future_native_to_canonical_indices,
        future_union_query_mask,
    )
    from real_motion.v20_runtime import _query_mask_from_render_index

    lattice = CanonicalLattice(
        (-4.0, -4.0, -2.0), (0.5, 0.5, 0.5), (20, 20, 10)
    )
    poses = np.repeat(np.eye(4)[None], FUTURE_FRAMES, axis=0)
    poses[:, 0, 3] = np.linspace(-0.3, 0.4, FUTURE_FRAMES)
    poses[:, 1, 3] = np.linspace(0.2, -0.5, FUTURE_FRAMES)
    kwargs = dict(
        future_ego_to_canonical=poses,
        native_shape_xyz=(6, 5, 3),
        native_origin_xyz_m=(-1.5, -1.0, -0.5),
        native_voxel_size_xyz_m=(0.5, 0.5, 0.5),
    )
    direct = future_union_query_mask(lattice, **kwargs)
    ri = future_native_to_canonical_indices(lattice, **kwargs)
    reused = _query_mask_from_render_index(lattice, ri)
    assert np.array_equal(reused.mask, direct.mask)
    assert reused.requested_voxels == direct.requested_voxels
    assert reused.in_bounds_voxels == direct.in_bounds_voxels
    assert reused.out_of_bounds_voxels == direct.out_of_bounds_voxels


def test_runtime_batched_static_tiles_match_single_tile_decode():
    from real_motion.v20_runtime import decode_static_world_tiled

    static_id = next(
        i for i in range(17) if i not in set(DYNAMIC_CLASS_IDS)
    )

    class FakeStatic:
        def refine_tiles(
            self,
            scene_features,
            *,
            sample_grid,
            query_mask,
            seen_mask,
            t0_missing_mask,
        ):
            B, X, Y, Z = query_mask.shape
            logits = torch.zeros(
                (B, 18, X, Y, Z),
                dtype=scene_features.dtype,
                device=scene_features.device,
            )
            logits[:, static_id] = (
                1.0 + seen_mask.to(logits.dtype)
            )
            logits[:, 17] = (~query_mask).to(logits.dtype) * 5.0
            return logits

    class FakeModel:
        static = FakeStatic()

    high = CanonicalLattice(
        (-2.0, -2.0, -1.0), (0.5, 0.5, 0.5), (10, 10, 6)
    )
    coarse = CanonicalLattice(
        (-2.0, -2.0, -1.0), (1.0, 1.0, 1.0), (5, 5, 3)
    )
    poses = np.repeat(np.eye(4)[None], FUTURE_FRAMES, axis=0)
    hist = np.zeros((HISTORY_FRAMES, 5, 5, 3), dtype=bool)
    hist[:, 1:4, 1:4, :] = True
    scene = torch.zeros((1, 4, 5, 5, 3))
    kwargs = dict(
        model=FakeModel(),
        scene_features=scene,
        high_lattice=high,
        coarse_lattice=coarse,
        future_ego_to_canonical=poses,
        history_observed_coarse=hist,
        native_shape_xyz=(4, 4, 2),
        native_origin_xyz_m=(-1.0, -1.0, -0.5),
        native_voxel_size_xyz_m=(0.5, 0.5, 0.5),
        tile_size_xyz=(4, 4, 3),
    )
    a = decode_static_world_tiled(**kwargs, tile_batch_size=1)
    b = decode_static_world_tiled(**kwargs, tile_batch_size=8)
    assert np.array_equal(a.canonical_semantic, b.canonical_semantic)
    assert np.array_equal(a.future_semantic, b.future_semantic)
    assert a.query_voxels == b.query_voxels
    assert a.active_tiles == b.active_tiles
    assert a.out_of_bounds_voxels == b.out_of_bounds_voxels


def test_static_refine_packed_single_scene_matches_expanded_scene():
    from real_motion.v20_scene_model import StaticWorldHead, V20SceneConfig

    torch.manual_seed(61)
    cfg = V20SceneConfig(base_dim=4, tile_dim=6)
    head = StaticWorldHead(scene_dim=8, cfg=cfg).eval()
    B, D, H, W = 3, 4, 5, 3
    scene = torch.randn(1, 8, 7, 6, 5)
    grid = torch.empty(B, D, H, W, 3).uniform_(-1, 1)
    q = torch.rand(B, D, H, W) > 0.2
    seen = torch.rand(B, D, H, W) > 0.5
    miss = seen & (torch.rand(B, D, H, W) > 0.5)

    with torch.inference_mode():
        packed = head.refine_tiles(
            scene,
            sample_grid=grid,
            query_mask=q,
            seen_mask=seen,
            t0_missing_mask=miss,
        )
        expanded = head.refine_tiles(
            scene.expand(B, -1, -1, -1, -1).contiguous(),
            sample_grid=grid,
            query_mask=q,
            seen_mask=seen,
            t0_missing_mask=miss,
        )
    assert torch.allclose(packed, expanded, atol=1e-5, rtol=1e-5)


def test_decode_static_logits_allowed_channel_path_matches_mask_reference():
    from real_motion.v20_training import decode_static_logits

    torch.manual_seed(67)
    logits = torch.randn(3, 18, 4, 3, 2)
    # Include ties to verify ascending global-class tie breaking.
    logits[:, 0, 0, 0, 0] = 5.0
    logits[:, 17, 0, 0, 0] = 5.0
    ref = logits.clone()
    dyn = torch.as_tensor(DYNAMIC_CLASS_IDS, dtype=torch.long)
    ref[:, dyn] = torch.finfo(ref.dtype).min
    expected = ref.argmax(dim=1)
    got = decode_static_logits(logits)
    assert torch.equal(got, expected)


def test_future_render_linear_index_matches_xyz_render():
    from real_motion.v20_history_world import (
        FutureRenderIndex,
        future_native_to_canonical_indices,
        render_canonical_semantic_to_future,
    )

    lattice = CanonicalLattice(
        (-3.0, -3.0, -1.0), (0.5, 0.5, 0.5), (14, 14, 6)
    )
    poses = np.repeat(np.eye(4)[None], FUTURE_FRAMES, axis=0)
    poses[:, 0, 3] = np.linspace(-0.2, 0.4, FUTURE_FRAMES)
    ri = future_native_to_canonical_indices(
        lattice,
        future_ego_to_canonical=poses,
        native_shape_xyz=(5, 4, 2),
        native_origin_xyz_m=(-1.0, -1.0, -0.5),
        native_voxel_size_xyz_m=(0.5, 0.5, 0.5),
    )
    world = np.arange(np.prod(lattice.shape_xyz), dtype=np.int64).reshape(
        lattice.shape_xyz
    )
    fast = render_canonical_semantic_to_future(world, ri, free_label=-1)
    legacy = FutureRenderIndex(
        indices_xyz=ri.indices_xyz,
        valid=ri.valid,
        out_of_bounds_voxels=ri.out_of_bounds_voxels,
        linear_index=None,
    )
    slow = render_canonical_semantic_to_future(world, legacy, free_label=-1)
    assert np.array_equal(fast, slow)


def test_future_render_index_uses_compact_int32_xyz():
    from real_motion.v20_history_world import (
        future_native_to_canonical_indices,
    )

    lattice = CanonicalLattice(
        (-3.0, -3.0, -1.0), (0.5, 0.5, 0.5), (14, 14, 6)
    )
    poses = np.repeat(np.eye(4)[None], FUTURE_FRAMES, axis=0)
    ri = future_native_to_canonical_indices(
        lattice,
        future_ego_to_canonical=poses,
        native_shape_xyz=(5, 4, 2),
        native_origin_xyz_m=(-1.0, -1.0, -0.5),
        native_voxel_size_xyz_m=(0.5, 0.5, 0.5),
    )
    assert ri.indices_xyz.dtype == np.int32
    assert ri.linear_index.dtype == np.int32


def test_static_repair_support_codec_and_target_contract():
    from real_motion.v20_static_repair import (
        pack_v18_free_support,
        unpack_v18_free_support,
        repair_target_from_gt,
    )

    rng = np.random.default_rng(71)
    shape = (7, 6, 3)
    support = rng.random((FUTURE_FRAMES,) + shape) > 0.3
    packed = pack_v18_free_support(support)
    got = unpack_v18_free_support(packed, shape)
    assert np.array_equal(got, support)

    gt = rng.integers(0, 18, size=(FUTURE_FRAMES,) + shape, dtype=np.uint8)
    target = repair_target_from_gt(gt)
    dyn = np.isin(gt, np.asarray(DYNAMIC_CLASS_IDS, dtype=np.uint8))
    assert np.all(target[dyn] == 17)
    assert np.array_equal(target[~dyn], gt[~dyn])


def test_static_repair_confusion_keeps_free_to_static_false_positives():
    from real_motion.v20_static_repair import (
        repair_confusion,
        repair_diagnostics_from_confusion,
    )

    # 100 static positives and 900 free negatives, predict the static class
    # everywhere.  This is the failure mode the legacy training metric hid.
    cid = next(i for i in range(17) if i not in set(DYNAMIC_CLASS_IDS))
    target = np.full((1, 1000), 17, dtype=np.uint8)
    target[0, :100] = cid
    pred = np.full_like(target, cid)
    support = np.ones_like(target, dtype=bool)
    conf = repair_confusion(target, pred, support)
    diag = repair_diagnostics_from_confusion(conf)
    assert conf[17, cid] == 900
    assert conf[cid, cid] == 100
    assert abs(diag["addition_precision"] - 0.1) < 1e-12
    assert abs(diag["repair_support_semantic_miou"] - 0.1) < 1e-12


def test_static_repair_protocol_guards_are_wired():
    import inspect
    from tools.real_motion import train_p0_f9_v20_dormant as dormant
    from tools.real_motion import v20_validate_run_inputs as validate

    dsrc = inspect.getsource(dormant.main)
    vsrc = inspect.getsource(validate.main)
    assert "STATIC_REPAIR_PROTOCOL" in dsrc
    assert "STATIC_REPAIR_PROTOCOL" in vsrc
    assert "overfit_diagnostic_only" in dsrc
    assert "overfit_diagnostic_only" in vsrc


def test_static_repair_gpu_geometry_matches_formal_cpu_mapping():
    from real_motion.v20_history_world import (
        future_native_to_canonical_indices,
    )
    from tools.real_motion.train_p0_f9_v20_static_repair import _Geometry

    high = CanonicalLattice(
        (-4.0, -4.0, -2.0), (0.5, 0.5, 0.5), (20, 20, 10)
    )
    coarse = CanonicalLattice(
        (-4.0, -4.0, -2.0), (1.0, 1.0, 1.0), (10, 10, 5)
    )
    shape = (6, 5, 3)
    origin = (-1.5, -1.0, -0.5)
    step = (0.5, 0.5, 0.5)
    poses = np.repeat(np.eye(4)[None], FUTURE_FRAMES, axis=0)
    poses[:, 0, 3] = np.linspace(-0.2, 0.3, FUTURE_FRAMES)
    poses[:, 1, 3] = np.linspace(0.25, -0.15, FUTURE_FRAMES)

    geom = _Geometry(
        high, coarse, shape, origin, step, (4, 4, 2), torch.device("cpu")
    )
    linear, query = geom.future_linear_and_query(poses)
    ri = future_native_to_canonical_indices(
        high,
        future_ego_to_canonical=poses,
        native_shape_xyz=shape,
        native_origin_xyz_m=origin,
        native_voxel_size_xyz_m=step,
    )
    Y, Z = high.shape_xyz[1], high.shape_xyz[2]
    ref_linear = (
        ri.indices_xyz[..., 0].astype(np.int64) * (Y * Z)
        + ri.indices_xyz[..., 1].astype(np.int64) * Z
        + ri.indices_xyz[..., 2].astype(np.int64)
    ).reshape(FUTURE_FRAMES, -1)
    assert np.array_equal(linear.numpy(), ref_linear)

    ref_q = np.zeros(high.shape_xyz, dtype=bool)
    ref_q.reshape(-1)[ref_linear.reshape(-1)] = True
    assert np.array_equal(query.numpy(), ref_q)


def test_static_repair_preserves_conflicting_horizon_contributions():
    from real_motion.v20_static_repair import STATIC_ALLOWED_IDS
    from tools.real_motion.train_p0_f9_v20_static_repair import (
        _repair_loss_and_confusion,
    )

    allowed = list(STATIC_ALLOWED_IDS)
    cid = next(i for i in allowed if i != 17)
    A = len(allowed)
    world = torch.zeros((A, 1, 1, 1), dtype=torch.float32)
    # Make the shared canonical cell prefer cid. Two horizons supervise the
    # same cell with conflicting labels; both must remain in confusion/loss.
    world[allowed.index(cid), 0, 0, 0] = 2.0
    linear = torch.zeros((FUTURE_FRAMES, 1), dtype=torch.long)
    support = np.zeros((FUTURE_FRAMES, 1, 1, 1), dtype=bool)
    target = np.full((FUTURE_FRAMES, 1, 1, 1), 17, dtype=np.uint8)
    support[0, 0, 0, 0] = True
    support[1, 0, 0, 0] = True
    target[0, 0, 0, 0] = cid
    target[1, 0, 0, 0] = 17

    loss, conf = _repair_loss_and_confusion(
        world, linear, support, target, device=torch.device("cpu")
    )
    got = conf.numpy()
    assert torch.isfinite(loss)
    assert got[cid, cid] == 1
    assert got[17, cid] == 1
    assert got.sum() == 2


def test_legacy_static_trainer_requires_explicit_reproduction_flag():
    import inspect
    from tools.real_motion import train_p0_f9_v20_static as legacy

    src = inspect.getsource(legacy.main)
    assert "--allow-legacy-v1" in src
    assert "train_p0_f9_v20_static_repair.py" in src


def test_static_repair_builder_never_reads_future_gt_or_lidar_masks():
    import inspect
    from tools.real_motion import build_p0_f9_v20_static_repair_support as b

    src = inspect.getsource(b._lean_raw)
    assert "future_gt_occ" not in src
    assert "load_lidar_observation" not in src
    assert "load_semantics" in src  # only t-1/t0 causal V18 input semantics
