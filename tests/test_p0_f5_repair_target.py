import numpy as np

from real_motion.repair_target import apply_dynamic_repair, build_dynamic_repair_endpoint

DYN = (2, 3, 4, 5, 6, 7, 9, 10)
FREE = 17


def test_repair_preserves_outside_support_and_static_inside():
    anchor = np.full((2, 4, 4, 2), FREE, dtype=np.uint8)
    proposal = anchor.copy()

    # Static anchor semantic inside support must survive.
    anchor[0, 1, 1, 0] = 11
    proposal[0, 1, 1, 0] = 2

    # Dynamic anchor at an old location is cleared if proposal moves away.
    anchor[0, 1, 1, 1] = 2
    # New dynamic location is inserted from the proposal.
    proposal[0, 1, 2, 1] = 2

    # Outside support proposal content must never leak into the endpoint.
    proposal[0, 3, 3, 1] = 3

    write = np.zeros((2, 4, 4), dtype=bool)
    write[0, 1, 1:3] = True

    out = build_dynamic_repair_endpoint(
        anchor, proposal, write, dynamic_class_ids=DYN, free_label=FREE
    )

    assert out[0, 1, 1, 0] == 11
    assert out[0, 1, 1, 1] == FREE
    assert out[0, 1, 2, 1] == 2
    assert out[0, 3, 3, 1] == anchor[0, 3, 3, 1]
    assert np.array_equal(out[~write], anchor[~write])


def test_empty_write_support_is_bit_exact_anchor():
    rng = np.random.default_rng(0)
    anchor = rng.integers(0, 18, size=(3, 5, 6, 2), dtype=np.uint8)
    proposal = rng.integers(0, 18, size=anchor.shape, dtype=np.uint8)
    write = np.zeros(anchor.shape[:-1], dtype=bool)
    out = apply_dynamic_repair(
        anchor, proposal, write, dynamic_class_ids=DYN, free_label=FREE
    )
    assert np.array_equal(out, anchor)


def test_single_horizon_contract_matches_temporal_contract():
    anchor = np.full((4, 4, 2), FREE, dtype=np.uint8)
    proposal = anchor.copy()
    anchor[1, 1, 0] = 2
    proposal[2, 1, 0] = 2
    write = np.zeros((4, 4), dtype=bool)
    write[1:3, 1] = True

    single = apply_dynamic_repair(
        anchor, proposal, write, dynamic_class_ids=DYN, free_label=FREE
    )
    stacked = apply_dynamic_repair(
        anchor[None], proposal[None], write[None], dynamic_class_ids=DYN, free_label=FREE
    )[0]
    assert np.array_equal(single, stacked)
