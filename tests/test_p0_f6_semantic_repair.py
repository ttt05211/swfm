import numpy as np
import torch

from real_motion.semantic_repair import (
    DYNAMIC_TO_SLOT,
    NUM_REPAIR_CLASSES,
    OCC_SHAPE,
    build_sparse_semantic_record,
    collapse_dynamic_logits,
    sparse_dynamic_semantic_loss,
    sparse_union_targets,
)


def test_sparse_union_targets_encodes_keep_remove_create():
    rec = {
        "sample_id": "scene-1:abc",
        "scene_name": "scene-1",
        # GT dynamic at 3 and 9. Anchor dynamic at 3 (keep) and 5 (remove).
        "gt_dynamic_flat_indices": torch.tensor([3, 9], dtype=torch.int32),
        "gt_dynamic_slots": torch.tensor([1, 4], dtype=torch.uint8),
        "anchor_dynamic_flat_indices": torch.tensor([3, 5], dtype=torch.int32),
    }
    idx, target = sparse_union_targets(rec, "cpu")
    assert idx.tolist() == [3, 5, 9]
    assert target.tolist() == [1, 0, 4]


def test_collapse_dynamic_logits_uses_grouped_background_logsumexp():
    logits = torch.full((1, 18), -10.0)
    logits[0, 0] = 0.0
    logits[0, 17] = 0.0
    logits[0, 2] = 1.0
    out = collapse_dynamic_logits(logits)
    assert tuple(out.shape) == (1, NUM_REPAIR_CLASSES)
    expected_bg = torch.logsumexp(logits[0, [0, 1, 8, 11, 12, 13, 14, 15, 16, 17]], dim=0)
    assert torch.allclose(out[0, 0], expected_bg.float())
    assert torch.allclose(out[0, 1], logits[0, 2].float())


def test_sparse_semantic_ce_supervises_background_and_dynamic():
    # Two sparse voxels, one removal/background and one dynamic class 2 slot.
    logits = torch.zeros((2, 18), requires_grad=True)
    targets = torch.tensor([0, DYNAMIC_TO_SLOT[4]], dtype=torch.long)
    loss, info = sparse_dynamic_semantic_loss([logits], [targets])
    assert torch.isfinite(loss)
    assert info["num_supervised_voxels"] == 2
    assert info["num_gt_dynamic_voxels"] == 1
    loss.backward()
    assert logits.grad is not None
    assert float(logits.grad.abs().sum()) > 0


def test_build_record_restricts_sparse_targets_to_write_support():
    gt = np.full(OCC_SHAPE, 17, dtype=np.uint8)
    anchor = np.full(OCC_SHAPE, 17, dtype=np.uint8)
    write = torch.zeros((6, 50, 50), dtype=torch.bool)
    write[0, 0, 0] = True  # expands to x/y [0:4,0:4]

    gt[0, 1, 1, 0] = 4       # inside support -> create
    gt[0, 10, 10, 0] = 4     # outside support -> ignored
    anchor[0, 2, 2, 0] = 4   # inside support -> remove if GT is background
    anchor[0, 20, 20, 0] = 4 # outside support -> ignored

    rec = build_sparse_semantic_record(
        sample_id="scene-1:abc",
        scene_name="scene-1",
        gt_future_occ=gt,
        anchor_decoded_occ=anchor,
        write_support_latent=write,
    )
    assert rec["gt_dynamic_flat_indices"].numel() == 1
    assert rec["anchor_dynamic_flat_indices"].numel() == 1
    idx, target = sparse_union_targets(rec, "cpu")
    assert idx.numel() == 2
    assert sorted(target.tolist()) == [0, DYNAMIC_TO_SLOT[4]]
