import numpy as np
import torch

from tools.real_motion.diagnose_p0_f9_v19_true_motion_responsibility import (
    _motion_status,
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
from real_motion.v19_innovation_targets import DECOMPOSITION_CATEGORIES
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
