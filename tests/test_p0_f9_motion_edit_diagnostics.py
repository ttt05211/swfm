import math
import numpy as np

from real_motion.motion_edit_diagnostics import (
    MotionEditAccumulator,
    motion_edit_counts,
    summarize_motion_edit_counts,
)

DYN = (2, 3, 4, 5, 6, 7, 9, 10)
FREE = 17


def _fixture():
    anchor = np.full((2, 3, 1), FREE, dtype=np.uint8)
    gt = anchor.copy()
    proposal = anchor.copy()
    write = np.ones((2, 3), dtype=bool)

    # required CLEAR: anchor dynamic, GT moved away. Proposal fails to clear -> stale.
    anchor[0, 0, 0] = 4
    gt[0, 0, 0] = FREE
    proposal[0, 0, 0] = 4

    # required KEEP: GT still dynamic. Proposal incorrectly clears.
    anchor[0, 1, 0] = 4
    gt[0, 1, 0] = 4
    proposal[0, 1, 0] = FREE

    # required WRITE: anchor empty, GT dynamic. Proposal writes correct class.
    gt[0, 2, 0] = 4
    proposal[0, 2, 0] = 4

    # stable non-dynamic: proposal produces a false dynamic.
    proposal[1, 0, 0] = 3

    # required KEEP with GT relabel: proposal follows GT class.
    anchor[1, 1, 0] = 4
    gt[1, 1, 0] = 3
    proposal[1, 1, 0] = 3

    # stable non-dynamic correct.
    return anchor, proposal, gt, write


def test_clear_keep_write_event_accounting():
    anchor, proposal, gt, write = _fixture()
    c = motion_edit_counts(anchor, proposal, gt, write, dynamic_class_ids=DYN)
    assert c["required_clear"] == 1
    assert c["stale_dynamic"] == 1
    assert c["correct_clear"] == 0
    assert c["required_keep"] == 2
    assert c["wrong_clear"] == 1
    assert c["correct_keep_presence"] == 1
    assert c["required_relabel"] == 1
    assert c["correct_keep_class"] == 1
    assert c["required_write"] == 1
    assert c["correct_write_presence"] == 1
    assert c["correct_write_class"] == 1
    assert c["false_write"] == 1

    s = summarize_motion_edit_counts(c)
    assert s["stale_dynamic_rate"] == 1.0
    assert s["clear_recall"] == 0.0
    assert s["wrong_clear_rate"] == 0.5
    assert s["keep_presence_recall"] == 0.5
    assert s["write_recall"] == 1.0
    assert s["write_class_accuracy"] == 1.0


def test_support_excludes_outside_voxels():
    anchor, proposal, gt, write = _fixture()
    write[:] = False
    c = motion_edit_counts(anchor, proposal, gt, write, dynamic_class_ids=DYN)
    assert c["support_voxels"] == 0
    assert all(v == 0 for k, v in c.items() if k != "support_voxels")
    s = summarize_motion_edit_counts(c)
    assert math.isnan(s["clear_recall"])
    assert math.isnan(s["write_recall"])


def test_accumulator_matches_two_updates():
    anchor, proposal, gt, write = _fixture()
    single = motion_edit_counts(anchor, proposal, gt, write, dynamic_class_ids=DYN)
    acc = MotionEditAccumulator(DYN)
    acc.update(anchor, proposal, gt, write)
    acc.update(anchor, proposal, gt, write)
    got = acc.compute()["counts"]
    for key, value in single.items():
        assert got[key] == 2 * value
