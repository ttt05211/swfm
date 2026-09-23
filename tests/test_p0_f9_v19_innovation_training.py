import numpy as np
import torch

from tools.real_motion.diagnose_p0_f9_v19_true_motion_responsibility import (
    _motion_status,
)
from tools.real_motion.diagnose_p0_f9_v19_novelty_candidate import (
    _accumulate_geometry_stats,
    _empty_geometry_stats,
    _finalize_geometry_stats,
)
from real_motion.geometry import OccupancyGrid
from real_motion.v19_static_novelty import (
    StaticNewFOVHead,
    decode_static_new_fov,
    history_grid_footprint_bev,
    majority_semantic_per_column,
    static_new_fov_loss,
)

from real_motion.v19_innovation import innovation_loss
from real_motion.v19_innovation_training import (
    build_true_motion_innovation_bev_supervision,
)
from real_motion.v19_innovation_v5 import (
    decode_ordered_endpoints,
    gaussian_endpoint_cross_entropy,
    soft_interval_iou_loss,
)
from real_motion.v19_innovation_targets import (
    DECOMPOSITION_CATEGORIES,
    NOVELTY_POSITIVE_CATEGORIES,
    TRANSPORT_REFINEMENT_CATEGORIES,
)
from real_motion.v19_innovation_training import (
    build_innovation_bev_supervision,
    dequantize_geometry_torch,
    pack_vertical_occupancy,
    quantize_geometry,
    unpack_vertical_occupancy_torch,
)


def _masks(shape):
    return {k: np.zeros(shape, dtype=bool) for k in DECOMPOSITION_CATEGORIES}


def test_innovation_supervision_ignores_excluded_and_mixed_columns():
    free = 17
    shape = (3, 3, 4)
    gt = np.full(shape, free, dtype=np.uint8)
    base = np.full(shape, free, dtype=np.uint8)
    masks = _masks(shape)

    gt[0, 0, 1] = 4
    masks["source_shape_innovation"][0, 0, 1] = True

    gt[1, 1, 2] = 4
    masks["current_source_transportable_miss"][1, 1, 2] = True

    gt[2, 2, 0] = 11
    gt[2, 2, 3] = 3
    masks["never_seen_static"][2, 2, 0] = True
    masks["history_source_recoverable"][2, 2, 3] = True

    sup = build_innovation_bev_supervision(
        gt, base, masks, free_label=free
    )
    assert sup["add_target"][0, 0] == 1
    assert sup["candidate_mask"][0, 0] == 1
    assert sup["semantic_target"][0, 0] == 4
    assert sup["vertical_target"][0, 0, 1] == 1

    assert sup["candidate_mask"][1, 1] == 0
    assert sup["add_target"][1, 1] == 0
    assert sup["candidate_mask"][2, 2] == 0
    assert sup["add_target"][2, 2] == 0
    assert sup["mixed_responsibility_bev_cells"] == 1


def test_innovation_supervision_drops_positive_blocked_by_explained_state():
    free = 17
    shape = (2, 2, 4)
    gt = np.full(shape, free, dtype=np.uint8)
    base = np.full(shape, free, dtype=np.uint8)
    masks = _masks(shape)
    gt[0, 0, 2] = 11
    masks["never_seen_static"][0, 0, 2] = True
    base[0, 0, 2] = 6

    sup = build_innovation_bev_supervision(
        gt, base, masks, free_label=free
    )
    assert sup["positive_voxels_raw"] == 1
    assert sup["positive_voxels_actionable"] == 0
    assert sup["blocked_positive_voxels"] == 1
    assert sup["add_target"][0, 0] == 0


def test_innovation_geometry_quantization_and_vertical_bitpack_roundtrip():
    geo = np.asarray(
        [0.0, 0.25, 0.5, 1.0], dtype=np.float32
    ).reshape(1, 1, 1, 1, 4)
    q = quantize_geometry(geo)
    dq = dequantize_geometry_torch(torch.from_numpy(q)).numpy()
    assert np.max(np.abs(dq - geo)) <= (1.0 / 255.0 + 1e-6)

    m = np.zeros((2, 3, 4, 16), dtype=bool)
    m[0, 1, 2, 0] = True
    m[0, 1, 2, 15] = True
    m[1, 0, 0, 7] = True
    bits = pack_vertical_occupancy(m)
    out = unpack_vertical_occupancy_torch(
        torch.from_numpy(bits)[None], 16
    )[0]
    out = out.permute(0, 2, 3, 1).bool().numpy()
    assert np.array_equal(out, m)


def test_dynamic_only_mode_ignores_never_seen_static():
    free = 17
    shape = (2, 2, 4)
    gt = np.full(shape, free, dtype=np.uint8)
    base = np.full(shape, free, dtype=np.uint8)
    masks = _masks(shape)

    gt[0, 0, 1] = 4
    masks["source_shape_innovation"][0, 0, 1] = True
    gt[1, 1, 2] = 11
    masks["never_seen_static"][1, 1, 2] = True

    sup = build_innovation_bev_supervision(
        gt,
        base,
        masks,
        free_label=free,
        positive_categories=(
            "future_birth_dynamic",
            "source_shape_innovation",
        ),
    )
    assert sup["add_target"][0, 0] == 1
    assert sup["candidate_mask"][0, 0] == 1
    assert sup["candidate_mask"][1, 1] == 0
    assert sup["add_target"][1, 1] == 0


def test_hard_negative_presence_keeps_bounded_negative_ratio():
    add_logits = torch.tensor(
        [[[[2.0, 1.5, 1.0], [0.5, 0.0, -0.5]]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    outputs = {
        "add_presence_logits": add_logits,
        "semantic_logits": torch.zeros((1, 1, 17, 2, 3)),
        "vertical_occupancy_logits": torch.zeros((1, 1, 4, 2, 3)),
    }
    add_target = torch.zeros((1, 1, 2, 3), dtype=torch.bool)
    add_target[0, 0, 0, 0] = True
    semantic_target = torch.zeros((1, 1, 2, 3), dtype=torch.long)
    vertical_target = torch.zeros((1, 1, 4, 2, 3))
    vertical_target[0, 0, 0, 0, 0] = 1.0
    candidate = torch.ones_like(add_target)

    loss, stats = innovation_loss(
        outputs,
        add_target=add_target,
        semantic_target=semantic_target,
        vertical_target=vertical_target,
        candidate_mask=candidate,
        positive_weight=1.0,
        vertical_positive_weight=1.0,
        presence_hard_negative_ratio=2.0,
    )
    assert torch.isfinite(loss)
    assert stats["positive_bev_cells"] == 1
    assert stats["hard_negative_bev_cells"] == 2
    loss.backward()
    assert add_logits.grad is not None


def test_true_motion_status_uses_instance_identity_not_spatial_overlap():
    common = {"moving-car", "parked-car"}
    moving = {"moving-car"}
    assert _motion_status(
        "moving-car",
        common_tokens=common,
        moving_tokens=moving,
    ) == "true_moving"
    assert _motion_status(
        "parked-car",
        common_tokens=common,
        moving_tokens=moving,
    ) == "common_nonmoving"
    assert _motion_status(
        "future-birth",
        common_tokens=common,
        moving_tokens=moving,
    ) == "not_common_unscored"
    assert _motion_status(
        None,
        common_tokens=common,
        moving_tokens=moving,
    ) == "unresolved"


def test_true_motion_supervision_makes_resolved_other_responsibility_negative():
    free = 17
    gt = np.full((3, 3, 4), free, dtype=np.uint8)
    base = np.full_like(gt, free)
    gt[1, 1, 1] = 4
    gt[2, 2, 1] = 4

    cats = {
        name: np.zeros_like(gt, dtype=bool)
        for name in (
            "history_source_recoverable",
            "t0_unrepresented_dynamic",
            "current_source_transportable_miss",
            "future_birth_dynamic",
            "source_shape_innovation",
            "dynamic_other_ambiguous",
            "history_static_recoverable",
            "history_static_seen_mismatch",
            "never_seen_static",
            "static_other_ambiguous",
        )
    }
    cats["source_shape_innovation"][1, 1, 1] = True
    cats["current_source_transportable_miss"][2, 2, 1] = True
    true_moving = np.zeros_like(gt, dtype=bool)
    true_moving[1, 1, 1] = True

    sup = build_true_motion_innovation_bev_supervision(
        gt,
        base,
        cats,
        true_moving,
        free_label=free,
    )
    assert sup["add_target"][1, 1] == 1
    # Resolved transport responsibility is no longer ignored: it is a negative.
    assert sup["candidate_mask"][2, 2] == 1
    assert sup["add_target"][2, 2] == 0


def test_gaussian_endpoint_ce_prefers_nearby_error():
    target = torch.tensor([5])
    near = torch.full((1, 16), -4.0)
    far = torch.full((1, 16), -4.0)
    near[0, 6] = 4.0
    far[0, 12] = 4.0
    ln = gaussian_endpoint_cross_entropy(near, target, sigma=0.75)
    lf = gaussian_endpoint_cross_entropy(far, target, sigma=0.75)
    assert float(ln) < float(lf)


def test_soft_interval_iou_rewards_matching_endpoints():
    b = torch.full((1, 8), -5.0)
    t = torch.full((1, 8), -5.0)
    b[0, 2] = 5.0
    t[0, 4] = 5.0
    good = soft_interval_iou_loss(
        b, t, torch.tensor([2]), torch.tensor([4])
    )
    bad = soft_interval_iou_loss(
        b, t, torch.tensor([5]), torch.tensor([7])
    )
    assert float(good) < float(bad)


def test_ordered_endpoint_decode_never_returns_top_below_bottom():
    bottom = torch.zeros((1, 1, 4, 1, 1))
    top = torch.zeros_like(bottom)
    bottom[0, 0, 3, 0, 0] = 10.0
    top[0, 0, 0, 0, 0] = 10.0
    b, t = decode_ordered_endpoints(bottom, top)
    assert int(t.item()) >= int(b.item())



def test_ancestor_free_novelty_contract_excludes_known_source_shape():
    assert set(NOVELTY_POSITIVE_CATEGORIES) == {
        "never_seen_static",
        "future_birth_dynamic",
    }
    assert "source_shape_innovation" not in NOVELTY_POSITIVE_CATEGORIES
    assert "source_shape_innovation" in TRANSPORT_REFINEMENT_CATEGORIES
    assert "current_source_transportable_miss" in TRANSPORT_REFINEMENT_CATEGORIES



def test_history_grid_footprint_detects_new_forward_fov():
    grid = OccupancyGrid(
        x_min=-2.0,
        y_min=-2.0,
        z_min=-1.0,
        voxel_size=(1.0, 1.0, 1.0),
        shape_hwd=(4, 4, 2),
    )
    history = np.eye(4, dtype=np.float64)[None, ...]
    future = np.eye(4, dtype=np.float64)
    future[0, 3] = 1.0
    covered = history_grid_footprint_bev(
        history,
        future,
        grid,
    )
    assert covered.shape == (4, 4)
    # The future ego moved +1 m in x, so its frontmost x row was outside
    # the previous [-2,2) grid footprint.
    assert covered[:3].all()
    assert not covered[3].any()



def test_static_geometry_stats_detect_contiguity_and_semantic_purity():
    mask = np.zeros((2, 2, 4), dtype=bool)
    gt = np.full((2, 2, 4), 17, dtype=np.uint8)

    # One pure contiguous road column of length 2.
    mask[0, 0, 0:2] = True
    gt[0, 0, 0:2] = 11

    # One non-contiguous mixed-semantic column spanning 0..2.
    mask[1, 1, 0] = True
    mask[1, 1, 2] = True
    gt[1, 1, 0] = 15
    gt[1, 1, 2] = 16

    stats = _empty_geometry_stats(4)
    _accumulate_geometry_stats(stats, mask, gt)
    out = _finalize_geometry_stats(stats)

    assert out["positive_columns"] == 2
    assert out["positive_voxels"] == 4
    assert out["contiguous_columns"] == 1
    assert out["contiguous_fraction"] == 0.5
    assert out["single_semantic_columns"] == 1
    assert out["single_semantic_column_fraction"] == 0.5
    assert out["bottom_histogram"][0] == 2
    assert out["top_histogram"][1] == 1
    assert out["top_histogram"][2] == 1



def test_static_new_fov_majority_semantic_and_decode():
    m = np.zeros((2, 2, 4), dtype=bool)
    gt = np.full((2, 2, 4), 17, dtype=np.uint8)
    m[0, 0, 0:3] = True
    gt[0, 0, 0] = 11
    gt[0, 0, 1:3] = 16
    sem = majority_semantic_per_column(m, gt)
    assert int(sem[0, 0]) == 16
    assert int(sem[1, 1]) == 255

    model = StaticNewFOVHead(
        future_frames=1,
        history_frames=1,
        semantic_dim=4,
        hidden_dim=8,
        num_semantic_classes=17,
        vertical_bins=4,
    ).eval()
    lab = torch.full((1, 1, 1, 2, 2), 17)
    geo = torch.zeros((1, 1, 1, 4, 2, 2))
    base = torch.zeros((1, 1, 1, 2, 2))
    support = torch.ones((1, 1, 1, 2, 2))
    with torch.inference_mode():
        out = model(lab, geo, base, support)
    assert out["occupancy_logits"].shape == (1, 1, 4, 2, 2)
    assert out["semantic_logits"].shape == (1, 1, 17, 2, 2)

    out = {k: v.clone() for k, v in out.items()}
    out["occupancy_logits"].fill_(-10)
    out["semantic_logits"].fill_(-10)
    out["occupancy_logits"][0, 0, 1, 0, 0] = 10
    out["semantic_logits"][0, 0, 11, 0, 0] = 10
    base_free = torch.ones((1, 1, 4, 2, 2), dtype=torch.bool)
    proposal = decode_static_new_fov(
        out,
        new_fov_mask=support,
        base_free_mask=base_free,
        free_label=17,
        occupancy_threshold=0.5,
    )
    assert int(proposal[0, 0, 1, 0, 0]) == 11
    assert int((proposal != 17).sum()) == 1


def test_static_new_fov_loss_masks_outside_support():
    occ_logits = torch.zeros((1, 1, 2, 1, 2))
    sem_logits = torch.zeros((1, 1, 17, 1, 2))
    target = torch.zeros_like(occ_logits, dtype=torch.bool)
    target[0, 0, 0, 0, 0] = True
    cand = torch.zeros_like(target)
    cand[0, 0, :, 0, 0] = True
    sem_tgt = torch.full((1, 1, 1, 2), 255, dtype=torch.long)
    sem_tgt[0, 0, 0, 0] = 11
    loss, stats = static_new_fov_loss(
        {
            "occupancy_logits": occ_logits,
            "semantic_logits": sem_logits,
        },
        occupancy_target=target,
        candidate_voxels=cand,
        semantic_target=sem_tgt,
        occupancy_positive_weight=1.0,
    )
    assert torch.isfinite(loss)
    assert stats["candidate_voxels"] == 2
    assert stats["positive_voxels"] == 1
