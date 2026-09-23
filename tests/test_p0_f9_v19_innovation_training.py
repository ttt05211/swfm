import numpy as np
import torch

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
