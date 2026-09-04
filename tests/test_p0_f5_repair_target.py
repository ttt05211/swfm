import numpy as np

from real_motion.repair_target import (
    apply_dynamic_repair,
    apply_dynamic_union,
    apply_dynamic_write_only,
    build_dynamic_repair_endpoint,
)

DYN = (2, 3, 4, 5, 6, 7, 9, 10)
FREE = 17


def test_repair_matches_frozen_dynamic_oracle_semantics():
    anchor = np.full((2, 4, 4, 2), FREE, dtype=np.uint8)
    proposal = anchor.copy()

    # Proposal dynamic is allowed to overwrite the anchor semantic inside support.
    anchor[0, 1, 1, 0] = 11
    proposal[0, 1, 1, 0] = 2

    # Dynamic anchor at an old location is cleared if proposal moves away.
    anchor[0, 1, 1, 1] = 2
    # New dynamic location is inserted from the proposal.
    proposal[0, 1, 2, 1] = 2

    # Non-dynamic proposal semantics are never copied.
    anchor[0, 1, 2, 0] = 12
    proposal[0, 1, 2, 0] = 13

    # Outside support proposal content must never leak into the endpoint.
    proposal[0, 3, 3, 1] = 3

    write = np.zeros((2, 4, 4), dtype=bool)
    write[0, 1, 1:3] = True

    out = build_dynamic_repair_endpoint(
        anchor, proposal, write, dynamic_class_ids=DYN, free_label=FREE
    )

    assert out[0, 1, 1, 0] == 2
    assert out[0, 1, 1, 1] == FREE
    assert out[0, 1, 2, 1] == 2
    assert out[0, 1, 2, 0] == 12
    assert out[0, 3, 3, 1] == anchor[0, 3, 3, 1]
    assert np.array_equal(out[~write], anchor[~write])


def test_empty_write_support_is_bit_exact_anchor():
    rng = np.random.default_rng(0)
    anchor = rng.integers(0, 18, size=(3, 5, 6, 2), dtype=np.uint8)
    proposal = rng.integers(0, 18, size=anchor.shape, dtype=np.uint8)
    write = np.zeros(anchor.shape[:-1], dtype=bool)
    for fn in (apply_dynamic_repair, apply_dynamic_write_only, apply_dynamic_union):
        kwargs = {"dynamic_class_ids": DYN}
        if fn is apply_dynamic_repair:
            kwargs["free_label"] = FREE
        out = fn(anchor, proposal, write, **kwargs)
        assert np.array_equal(out, anchor)


def test_single_horizon_contract_matches_temporal_contract():
    anchor = np.full((4, 4, 2), FREE, dtype=np.uint8)
    proposal = anchor.copy()
    anchor[1, 1, 0] = 2
    proposal[2, 1, 0] = 2
    write = np.zeros((4, 4), dtype=bool)
    write[1:3, 1] = True

    for fn in (apply_dynamic_repair, apply_dynamic_write_only, apply_dynamic_union):
        kwargs = {"dynamic_class_ids": DYN}
        if fn is apply_dynamic_repair:
            kwargs["free_label"] = FREE
        single = fn(anchor, proposal, write, **kwargs)
        stacked = fn(anchor[None], proposal[None], write[None], **kwargs)[0]
        assert np.array_equal(single, stacked)


def test_write_only_adds_new_dynamic_but_never_changes_existing_dynamic():
    anchor = np.full((1, 3, 3, 2), FREE, dtype=np.uint8)
    proposal = anchor.copy()
    write = np.ones((1, 3, 3), dtype=bool)

    # Existing anchor dynamic must survive even when proposal says free.
    anchor[0, 0, 0, 0] = 2
    proposal[0, 0, 0, 0] = FREE

    # Existing anchor dynamic class must not be relabelled by proposal.
    anchor[0, 0, 1, 0] = 2
    proposal[0, 0, 1, 0] = 3

    # Proposal may add dynamics over free and static/non-dynamic anchor semantics.
    proposal[0, 1, 0, 0] = 4
    anchor[0, 1, 1, 0] = 11
    proposal[0, 1, 1, 0] = 5

    out = apply_dynamic_write_only(anchor, proposal, write, dynamic_class_ids=DYN)
    assert out[0, 0, 0, 0] == 2
    assert out[0, 0, 1, 0] == 2
    assert out[0, 1, 0, 0] == 4
    assert out[0, 1, 1, 0] == 5


def test_dynamic_union_preserves_presence_but_allows_dynamic_class_correction():
    anchor = np.full((1, 3, 3, 2), FREE, dtype=np.uint8)
    proposal = anchor.copy()
    write = np.ones((1, 3, 3), dtype=bool)

    # Proposal background cannot erase anchor dynamic presence.
    anchor[0, 0, 0, 0] = 2
    proposal[0, 0, 0, 0] = FREE

    # Proposal dynamic may relabel an existing anchor dynamic class.
    anchor[0, 0, 1, 0] = 2
    proposal[0, 0, 1, 0] = 3

    # Proposal dynamic may also add a missing dynamic voxel.
    anchor[0, 1, 0, 0] = 11
    proposal[0, 1, 0, 0] = 4

    out = apply_dynamic_union(anchor, proposal, write, dynamic_class_ids=DYN)
    assert out[0, 0, 0, 0] == 2
    assert out[0, 0, 1, 0] == 3
    assert out[0, 1, 0, 0] == 4
