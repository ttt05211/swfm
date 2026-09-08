import numpy as np

from real_motion.oracle_gap import (
    gt_edit_support_bev,
    oracle_clear_keep_presence,
    oracle_dynamic_semantics,
    oracle_event_presence,
    oracle_write_presence,
    support_coverage_counts,
)


DYN = (4, 10)
FREE = 17


def _toy():
    # [X,Y,Z] with Z=1.  Four BEV cells exercise CLEAR, KEEP, WRITE, stable-non.
    anchor = np.array([[[4], [4]], [[17], [17]]], dtype=np.uint8)
    gt = np.array([[[17], [4]], [[10], [17]]], dtype=np.uint8)
    proposal = np.array([[[4], [17]], [[17], [10]]], dtype=np.uint8)
    support = np.ones((2, 2), dtype=bool)
    return anchor, proposal, gt, support


def test_clear_keep_oracle_only_fixes_anchor_dynamic_presence():
    anchor, proposal, gt, support = _toy()
    out = oracle_clear_keep_presence(
        anchor, proposal, gt, support, dynamic_class_ids=DYN, free_label=FREE
    )
    assert int(out[0, 0, 0]) == FREE  # required CLEAR
    assert int(out[0, 1, 0]) == 4     # missing KEEP restored with anchor class
    assert int(out[1, 0, 0]) == 17    # WRITE site untouched
    assert int(out[1, 1, 0]) == 10    # false WRITE untouched


def test_write_oracle_only_fixes_anchor_non_dynamic_sites():
    anchor, proposal, gt, support = _toy()
    out = oracle_write_presence(
        anchor, proposal, gt, support, dynamic_class_ids=DYN, free_label=FREE
    )
    assert int(out[0, 0, 0]) == 4     # CLEAR site untouched
    assert int(out[0, 1, 0]) == 17    # KEEP site untouched
    assert int(out[1, 0, 0]) == 10    # required WRITE gets GT dynamic class
    assert int(out[1, 1, 0]) == FREE  # false WRITE removed


def test_event_oracle_combines_clear_keep_and_write():
    anchor, proposal, gt, support = _toy()
    out = oracle_event_presence(
        anchor, proposal, gt, support, dynamic_class_ids=DYN, free_label=FREE
    )
    assert np.array_equal(out[..., 0], np.array([[17, 4], [10, 17]], dtype=np.uint8))


def test_semantic_oracle_preserves_dynamic_presence_geometry():
    anchor = np.array([[[4], [17]], [[17], [17]]], dtype=np.uint8)
    gt = np.array([[[10], [10]], [[17], [17]]], dtype=np.uint8)
    proposal = np.array([[[4], [17]], [[4], [17]]], dtype=np.uint8)
    support = np.ones((2, 2), dtype=bool)
    out = oracle_dynamic_semantics(anchor, proposal, gt, support, dynamic_class_ids=DYN)
    assert int(out[0, 0, 0]) == 10  # class corrected where both are dynamic
    assert int(out[0, 1, 0]) == 17  # missing GT dynamic is not created
    assert int(out[1, 0, 0]) == 4   # false dynamic is not removed


def test_gt_edit_support_expands_only_required_dynamic_edit_cells():
    anchor, _, gt, _ = _toy()
    current = np.zeros((2, 2), dtype=bool)
    current[0, 1] = True  # unchanged KEEP cell only
    expanded = gt_edit_support_bev(anchor, gt, current, dynamic_class_ids=DYN)
    # CLEAR and WRITE are required; current KEEP cell is retained; stable-non is absent.
    assert expanded.tolist() == [[True, True], [True, False]]

    counts = support_coverage_counts(anchor, gt, current, dynamic_class_ids=DYN)
    assert counts["clear"] == {"total": 1, "covered": 0}
    assert counts["write"] == {"total": 1, "covered": 0}
    assert counts["relabel"] == {"total": 0, "covered": 0}
    assert counts["event"] == {"total": 2, "covered": 0}
