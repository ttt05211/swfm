import numpy as np

from tools.real_motion.diagnose_p0_f9_v19_static_novelty import (
    _add_only_triage,
    _accumulate_threshold_hist,
    _baseline_counts,
    _evaluate_threshold_grid,
    _new_threshold_hist,
)


def test_add_only_triage_exact_categories():
    raw = {
        "a": {
            "occ_inter": [10, 20],
            "occ_union": [40, 60],
            "sem_inter": [[2, 3], [4, 5]],
        },
        "b": {
            "occ_inter": [13, 22],
            "occ_union": [44, 63],
            "sem_inter": [[3, 4], [5, 5]],
        },
    }
    got = _add_only_triage(raw, "a", "b")
    assert got["position_and_semantic_correct"] == 3
    assert got["position_correct_semantic_wrong"] == 2
    assert got["added_into_gt_free"] == 7
    assert got["added_voxels"] == 12


def test_threshold_hist_matches_manual_single_voxel_logic():
    pths = [0.4, 0.6]
    vths = [0.4, 0.6]
    hist = _new_threshold_hist(2, 2)

    presence = np.asarray([[0.7]], dtype=np.float32)
    vertical = np.asarray([[[0.7]]], dtype=np.float32)
    pred_cls = np.asarray([[3]], dtype=np.uint8)
    gt = np.asarray([[[3]]], dtype=np.uint8)
    eligible = np.asarray([[[True]]])

    _accumulate_threshold_hist(
        hist,
        1,
        presence,
        vertical,
        pred_cls,
        gt,
        eligible,
        pths,
        vths,
    )
    base = _baseline_counts()
    # Baseline prediction is free while GT is class 3.
    base["occ_union"][1] = 1
    base["sem_union"][1, 3] = 1

    rep = _evaluate_threshold_grid(base, hist, pths, vths)
    for row in rep["all"]:
        assert row["added_tp"] == 1
        assert row["added_fp"] == 0
        assert row["semantic_correct_additions"] == 1
