import numpy as np
import torch

from real_motion.edit_repair import (
    CLEAR,
    DYNAMIC_IDS,
    KEEP,
    NUM_ACTIONS,
    WRITE_OFFSET,
    action_probs_to_result_probs,
    anchor_relative_edit_loss,
    apply_anchor_relative_actions,
    build_anchor_relative_edit_record,
    horizon_from_flat_indices,
    lovasz_softmax_flat,
    select_balanced_edit_supervision,
)
from real_motion.models.p0_f8 import AnchorRelativeEditHead


def _toy_record(easy_keep_limit=0):
    gt = np.full((6, 200, 200, 16), 17, dtype=np.uint8)
    anchor = gt.copy()
    c1, c2 = int(DYNAMIC_IDS[0]), int(DYNAMIC_IDS[1])
    # All four voxels lie in the first expanded latent support cell.
    anchor[0, 0, 0, 0] = c1
    gt[0, 0, 0, 0] = c1  # KEEP
    anchor[0, 0, 1, 0] = c1
    gt[0, 0, 1, 0] = 17  # CLEAR
    anchor[0, 0, 2, 0] = 17
    gt[0, 0, 2, 0] = c2  # WRITE creation
    anchor[0, 0, 3, 0] = c1
    gt[0, 0, 3, 0] = c2  # WRITE class change

    support = torch.zeros(6, 50, 50, dtype=torch.bool)
    support[0, 0, 0] = True
    moving = np.zeros((6, 200, 200, 16), dtype=bool)
    moving[0, 0, 0, 0] = True
    rec = build_anchor_relative_edit_record(
        sample_id="scene:sample",
        scene_name="scene",
        gt_future_occ=gt,
        strong_anchor_occ=anchor,
        write_support_latent=support,
        moving_support_bev=moving,
        easy_keep_limit=easy_keep_limit,
    )
    return rec, anchor, gt


def test_edit_record_encodes_keep_clear_and_write():
    rec, _, _ = _toy_record()
    assert rec["edit_flat_indices"].numel() == 3
    assert rec["keep_flat_indices"].numel() == 1
    actions = rec["edit_actions"].tolist()
    assert actions[0] == CLEAR
    assert actions[1] == WRITE_OFFSET + 1  # DYNAMIC_IDS[1] => slot 2 => action 3
    assert actions[2] == WRITE_OFFSET + 1
    assert rec["keep_priority"].tolist() == [2]


def test_balanced_sampling_keeps_all_edits_and_prioritizes_hard_keep():
    rec, _, _ = _toy_record(easy_keep_limit=32)
    sel = select_balanced_edit_supervision(
        rec,
        keep_ratio=1.0,
        deterministic=True,
    )
    assert sel["num_edits"] == 3
    assert sel["num_keeps"] == 3
    assert sel["actions"][:3].ne(KEEP).all()
    assert sel["actions"][3:].eq(KEEP).all()
    # The first selected KEEP is the exact true-moving dynamic KEEP (priority 2).
    assert int(sel["anchor_slots"][3]) > 0


def test_true_motion_priority_is_voxel_exact_not_bev_broadcast():
    rec, _, _ = _toy_record(easy_keep_limit=0)
    assert rec["keep_priority"].tolist() == [2]


def test_action_probability_projection_is_anchor_relative():
    logits = torch.full((2, NUM_ACTIONS), -8.0)
    logits[0, KEEP] = 8.0
    logits[1, CLEAR] = 8.0
    anchor_slots = torch.tensor([2, 2])
    result = action_probs_to_result_probs(logits, anchor_slots)
    assert int(result[0].argmax()) == 2
    assert int(result[1].argmax()) == 0
    assert torch.allclose(result.sum(dim=1), torch.ones(2), atol=1e-5)


def test_lovasz_and_edit_loss_prefer_correct_result():
    labels = torch.tensor([0, 1, 2, 2])
    perfect = torch.nn.functional.one_hot(labels, num_classes=3).float() * 0.98 + 0.02 / 3
    wrong = torch.full_like(perfect, 1 / 3)
    assert lovasz_softmax_flat(perfect, labels) < lovasz_softmax_flat(wrong, labels)

    actions = torch.tensor([KEEP, CLEAR, WRITE_OFFSET, WRITE_OFFSET + 1])
    anchor_slots = torch.tensor([1, 1, 0, 1])
    result_slots = torch.tensor([1, 0, 1, 2])
    good_logits = torch.full((4, NUM_ACTIONS), -5.0, requires_grad=True)
    good_logits.data[torch.arange(4), actions] = 5.0
    loss, info = anchor_relative_edit_loss(
        good_logits, actions, anchor_slots, result_slots, lovasz_weight=0.5
    )
    assert torch.isfinite(loss)
    assert info["accuracy"] == 1.0
    loss.backward()
    assert torch.isfinite(good_logits.grad).all()


def test_edit_head_starts_with_keep_bias_and_uses_horizon():
    head = AnchorRelativeEditHead(keep_bias=2.0)
    semantic = torch.zeros(6, 18)
    anchor = torch.zeros(6, dtype=torch.long)
    horizon = torch.arange(6)
    logits = head(semantic, anchor, horizon)
    assert logits.shape == (6, NUM_ACTIONS)
    assert logits.argmax(dim=-1).eq(KEEP).all()


def test_apply_actions_changes_only_requested_voxels():
    rec, anchor, _ = _toy_record()
    idx = rec["edit_flat_indices"].numpy().astype(np.int64)
    actions = rec["edit_actions"].numpy().astype(np.int64)
    out = apply_anchor_relative_actions(anchor, idx, actions)
    flat_a = anchor.reshape(-1)
    flat_o = out.reshape(-1)
    changed = np.flatnonzero(flat_a != flat_o)
    assert set(changed.tolist()) == set(idx.tolist())
    assert np.all(flat_o[idx] != flat_a[idx])


def test_horizon_from_flat_indices():
    stride = 200 * 200 * 16
    idx = torch.tensor([0, stride - 1, stride, 5 * stride])
    assert horizon_from_flat_indices(idx).tolist() == [0, 0, 1, 5]
