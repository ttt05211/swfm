import numpy as np
import pytest

from real_motion.msp import (
    MSPCandidate,
    STATE_DORMANT,
    STATE_OBSERVED_MOVING,
)
from real_motion.msp_audit import (
    C_T0_ANNOTATION_MISSING,
    C_NO_SAME_CLASS_CANDIDATE,
    C_DISTANCE_GATE,
    C_ONE_TO_ONE_CONFLICT,
    classify_unmatched_instance,
    distance_bin,
    mask_touches_xy_boundary,
)


def _candidate(x, y, cls=4, state=STATE_DORMANT):
    return MSPCandidate(
        class_id=cls,
        state=state,
        centroid_xy_m=np.asarray([x, y], dtype=np.float32),
        velocity_xy_mps=np.zeros(2, dtype=np.float32),
        extent_xy_m=np.asarray([2.0, 1.0], dtype=np.float32),
        voxel_count=4,
        kta_matched=True,
    )


def test_c_source_missing_t0_annotation_is_explicit_contract_failure_category():
    row = classify_unmatched_instance(
        class_id=4,
        center_xy_t0_m=np.zeros(2),
        candidates=[],
        candidate_tokens=[],
        match_max_distance_m=4.0,
        t0_annotation_present=False,
    )
    assert row.category == C_T0_ANNOTATION_MISSING
    assert row.nearest_distance_m is None


def test_c_source_no_same_class_candidate_is_not_confused_with_other_classes():
    row = classify_unmatched_instance(
        class_id=4,
        center_xy_t0_m=np.zeros(2),
        candidates=[_candidate(0.1, 0.0, cls=7)],
        candidate_tokens=[None],
        match_max_distance_m=4.0,
    )
    assert row.category == C_NO_SAME_CLASS_CANDIDATE
    assert row.nearest_candidate_index is None


def test_c_source_candidate_outside_distance_gate():
    row = classify_unmatched_instance(
        class_id=4,
        center_xy_t0_m=np.zeros(2),
        candidates=[_candidate(4.01, 0.0, cls=4)],
        candidate_tokens=[None],
        match_max_distance_m=4.0,
    )
    assert row.category == C_DISTANCE_GATE
    assert abs(row.nearest_distance_m - 4.01) < 1e-4
    assert not row.candidate_within_gate


def test_c_source_within_gate_means_one_to_one_association_conflict():
    row = classify_unmatched_instance(
        class_id=4,
        center_xy_t0_m=np.zeros(2),
        candidates=[
            _candidate(2.0, 0.0, cls=4, state=STATE_OBSERVED_MOVING),
            _candidate(0.5, 0.0, cls=4),
        ],
        candidate_tokens=["other-a", "other-b"],
        match_max_distance_m=4.0,
    )
    assert row.category == C_ONE_TO_ONE_CONFLICT
    assert row.nearest_candidate_index == 1
    assert row.nearest_assigned_token == "other-b"
    assert row.candidate_within_gate


def test_c_source_within_gate_unassigned_candidate_breaks_matcher_invariant():
    with pytest.raises(RuntimeError, match="matcher invariant is broken"):
        classify_unmatched_instance(
            class_id=4,
            center_xy_t0_m=np.zeros(2),
            candidates=[_candidate(0.5, 0.0, cls=4)],
            candidate_tokens=[None],
            match_max_distance_m=4.0,
        )


def test_distance_bins_have_stable_boundaries():
    assert distance_bin(None) == "no_candidate"
    assert distance_bin(0.999) == "lt_1m"
    assert distance_bin(1.0) == "1_to_2m"
    assert distance_bin(2.0) == "2_to_4m"
    assert distance_bin(4.0) == "4_to_6m"
    assert distance_bin(6.0) == "6_to_10m"
    assert distance_bin(10.0) == "ge_10m"


def test_xy_boundary_detection_only_flags_xy_edges():
    m = np.zeros((5, 6, 3), dtype=bool)
    m[2, 3, 0] = True
    assert not mask_touches_xy_boundary(m), "z boundary is irrelevant to XY/FOV truncation"
    m[0, 3, 1] = True
    assert mask_touches_xy_boundary(m)
