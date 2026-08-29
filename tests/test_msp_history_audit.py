import pytest

from real_motion.msp_history_audit import (
    HISTORY_NEVER_SEEN,
    HISTORY_SEEN,
    HistoryFrameEvidence,
    summarize_history_source,
)


def _row(age, *, ann=True, in_grid=True, exact=0, margin=0):
    return HistoryFrameEvidence(
        age_s=float(age),
        annotation_present=bool(ann),
        box_in_grid=bool(in_grid),
        exact_same_class_voxels=int(exact),
        margin_same_class_voxels=int(margin),
    )


def test_history_seen_if_any_past_frame_contains_same_class_occ():
    rows = [
        _row(2.5, ann=False, in_grid=False),
        _row(2.0, in_grid=False),
        _row(1.5, exact=0, margin=0),
        _row(1.0, exact=3, margin=5),
        _row(0.5, in_grid=False),
    ]
    s = summarize_history_source(rows)
    assert s.category == HISTORY_SEEN
    assert s.seen_frame_count == 1
    assert s.last_seen_age_s == 1.0
    assert s.oldest_seen_age_s == 1.0
    assert s.annotation_frame_count == 4
    assert s.in_grid_frame_count == 2


def test_margin_only_occ_is_a_valid_historical_source():
    rows = [
        _row(1.0, exact=0, margin=2),
        _row(0.5, exact=0, margin=0),
    ]
    s = summarize_history_source(rows)
    assert s.category == HISTORY_SEEN
    assert s.last_seen_age_s == 1.0


def test_never_seen_when_annotations_exist_but_no_same_class_occ():
    rows = [
        _row(1.5, in_grid=False),
        _row(1.0, exact=0, margin=0),
        _row(0.5, exact=0, margin=0),
    ]
    s = summarize_history_source(rows)
    assert s.category == HISTORY_NEVER_SEEN
    assert s.seen_frame_count == 0
    assert s.last_seen_age_s is None
    assert s.oldest_seen_age_s is None
    assert s.annotation_frame_count == 3
    assert s.in_grid_frame_count == 2


def test_last_seen_age_is_closest_seen_frame_to_t0():
    rows = [
        _row(2.5, exact=1, margin=1),
        _row(1.5, exact=2, margin=2),
        _row(0.5, exact=1, margin=1),
    ]
    s = summarize_history_source(rows)
    assert s.category == HISTORY_SEEN
    assert s.seen_frame_count == 3
    assert s.last_seen_age_s == 0.5
    assert s.oldest_seen_age_s == 2.5


def test_invalid_evidence_fails_closed():
    with pytest.raises(ValueError):
        _row(1.0, ann=False, in_grid=True)
    with pytest.raises(ValueError):
        _row(1.0, in_grid=False, margin=1)
    with pytest.raises(ValueError):
        _row(1.0, exact=2, margin=1)
    with pytest.raises(ValueError):
        summarize_history_source([])
    with pytest.raises(ValueError):
        summarize_history_source([_row(1.0), _row(1.0)])
