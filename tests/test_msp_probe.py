import numpy as np
import torch

from real_motion.msp import (
    FEATURE_DIM,
    SOURCE_A,
    SOURCE_B,
    SOURCE_C,
    STATE_DORMANT,
    STATE_OBSERVED_MOVING,
    MSPCandidate,
    MSPProbeHead,
    candidate_feature,
    collate_probe_records,
    latent_support_to_bev,
    match_candidates_to_instances,
    msp_probe_loss,
    rasterize_msp_scores,
    source_type_for_token,
    top_budget_support,
)


def _candidate(x, y, cls=4, state=STATE_OBSERVED_MOVING):
    return MSPCandidate(
        class_id=cls,
        state=state,
        centroid_xy_m=np.asarray([x, y], dtype=np.float32),
        velocity_xy_mps=np.asarray([1.0, 0.0], dtype=np.float32),
        extent_xy_m=np.asarray([2.0, 1.0], dtype=np.float32),
        voxel_count=8,
        kta_matched=True,
    )


def _record(n=2, h=2):
    features = torch.zeros((n, FEATURE_DIM), dtype=torch.float32)
    anchors = torch.zeros((n, h, 2), dtype=torch.float32)
    activation = torch.zeros((n, h), dtype=torch.float32)
    valid = torch.zeros((n, h), dtype=torch.bool)
    target = torch.zeros((n, h, 2), dtype=torch.float32)
    target_valid = torch.zeros((n, h), dtype=torch.bool)
    if n:
        features[:, 0] = torch.arange(n, dtype=torch.float32) * 0.1
        activation[:, :] = 1.0
        valid[:, :] = True
        target[..., 0] = 1.0
        target_valid[:, :] = True
    rel = torch.eye(4).reshape(1, 4, 4).repeat(h, 1, 1)
    return {
        "sample_id": f"sample-{n}",
        "scene_name": f"scene-{n}",
        "features": features,
        "anchors_xy_t0_m": anchors,
        "activation": activation,
        "activation_valid": valid,
        "target_xy_t0_m": target,
        "target_valid": target_valid,
        "future_rel_t0_to_ego": rel,
        "candidate_state": torch.zeros((n,), dtype=torch.long),
        "candidate_extent_xy_m": torch.ones((n, 2), dtype=torch.float32),
    }


def test_feature_contract_has_no_hidden_annotation_fields():
    f = candidate_feature(_candidate(4.0, -2.0))
    assert f.shape == (FEATURE_DIM,)
    assert np.isfinite(f).all()
    # The frozen probe feature is intentionally compact and contains only
    # causal component state, geometry, KTA velocity, and class one-hot.
    assert FEATURE_DIM == 19


def test_gt_label_matching_is_class_aware_one_to_one():
    candidates = [
        _candidate(0.0, 0.0, cls=4, state=STATE_OBSERVED_MOVING),
        _candidate(1.0, 0.0, cls=4, state=STATE_DORMANT),
        _candidate(0.0, 0.0, cls=7, state=STATE_DORMANT),
    ]
    instances = [
        {"instance_token": "car-a", "class_id": 4, "center_xy_t0_m": np.array([0.1, 0.0])},
        {"instance_token": "car-b", "class_id": 4, "center_xy_t0_m": np.array([1.1, 0.0])},
        {"instance_token": "ped-a", "class_id": 7, "center_xy_t0_m": np.array([0.2, 0.0])},
    ]
    tokens, inverse = match_candidates_to_instances(candidates, instances, max_distance_m=0.5)
    assert tokens == ["car-a", "car-b", "ped-a"]
    assert len(set(inverse.values())) == 3
    assert source_type_for_token("car-a", inverse, candidates) == SOURCE_A
    assert source_type_for_token("car-b", inverse, candidates) == SOURCE_B
    assert source_type_for_token("missing", inverse, candidates) == SOURCE_C


def test_model_loss_handles_padding_and_an_empty_window():
    batch = collate_probe_records([_record(2, 2), _record(0, 2)])
    model = MSPProbeHead(
        feature_dim=FEATURE_DIM, hidden_dim=32, num_heads=4,
        num_modes=2, future_frames=2, dropout=0.0,
    )
    out = model(batch["features"], batch["candidate_mask"])
    assert out["activation_logits"].shape == (2, 2, 2)
    assert out["mu_residual_xy_m"].shape == (2, 2, 2, 2, 2)
    loss, info = msp_probe_loss(out, batch)
    assert torch.isfinite(loss)
    assert info["num_activation_labels"] == 4
    assert info["num_location_labels"] == 4


def test_raster_budget_is_bounded_and_geometry_aligned():
    h = 2
    batch = {
        "anchors_xy_t0_m": torch.zeros((1, 1, h, 2), dtype=torch.float32),
        "future_rel_t0_to_ego": torch.eye(4).reshape(1, 1, 4, 4).repeat(1, h, 1, 1),
        "candidate_mask": torch.ones((1, 1), dtype=torch.bool),
        "candidate_extent_xy_m": torch.ones((1, 1, 2), dtype=torch.float32),
    }
    out = {
        "activation_logits": torch.full((1, 1, h), 8.0),
        "mu_residual_xy_m": torch.zeros((1, 1, h, 1, 2)),
        "raw_sigma": torch.full((1, 1, h, 1), -2.0),
        "mode_logits": torch.zeros((1, 1, h, 1)),
    }
    score = rasterize_msp_scores(out, batch, latent_hw=(10, 10))
    assert score.shape == (1, h, 10, 10)
    assert torch.isfinite(score).all()
    support = top_budget_support(score, 0.10)
    assert support.shape == score.shape
    assert torch.all(support.sum(dim=(-2, -1)) <= 10)
    assert torch.all(support.sum(dim=(-2, -1)) > 0)
    bev = latent_support_to_bev(support[0], (20, 20))
    assert bev.shape == (h, 20, 20)
    assert int(bev.sum()) == int(support[0].sum()) * 4


def test_zero_score_map_does_not_create_fake_budget_cells():
    score = torch.zeros((1, 3, 10, 10))
    support = top_budget_support(score, 0.15)
    assert not support.any()
