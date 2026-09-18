import numpy as np

from real_motion.v18_two_wheel_diagnostic import (
    add_count_rows,
    paired_scene_bootstrap_gap,
    renderer_yaw_delta,
    scene_leave_one_out_gap,
    semantic_count_row,
)


def test_two_wheel_yaw_intervention_is_class_only():
    assert renderer_yaw_delta(2, 0.7, zero_two_wheel_yaw=False) == 0.7
    assert renderer_yaw_delta(2, 0.7, zero_two_wheel_yaw=True) == 0.0
    assert renderer_yaw_delta(6, -0.4, zero_two_wheel_yaw=True) == 0.0
    assert renderer_yaw_delta(4, 0.3, zero_two_wheel_yaw=True) == 0.3
    assert renderer_yaw_delta(7, 0.9, zero_two_wheel_yaw=False) == 0.0
    assert renderer_yaw_delta(7, 0.9, zero_two_wheel_yaw=True) == 0.0


def test_semantic_counts_expose_tp_fp_fn_exactly():
    gt = np.array([[[2], [2]], [[0], [0]]], dtype=np.int64)
    pred = np.array([[[2], [0]], [[2], [0]]], dtype=np.int64)
    support = np.ones_like(gt, dtype=bool)
    row = semantic_count_row(pred, gt, support, 2)
    assert row["tp"] == 1
    assert row["fp"] == 1
    assert row["fn"] == 1
    assert row["intersection"] == 1
    assert row["union"] == 3
    assert row["pred_voxels"] == 2
    assert row["gt_voxels"] == 2
    assert abs(row["iou"] - 100.0 / 3.0) < 1e-12


def test_count_aggregation_recomputes_dataset_iou():
    a = {"tp": 1, "fp": 0, "fn": 1, "intersection": 1, "union": 2,
         "pred_voxels": 1, "gt_voxels": 2}
    b = {"tp": 9, "fp": 1, "fn": 0, "intersection": 9, "union": 10,
         "pred_voxels": 10, "gt_voxels": 9}
    out = add_count_rows([a, b])
    assert out["intersection"] == 10
    assert out["union"] == 12
    assert abs(out["iou"] - (1000.0 / 12.0)) < 1e-12


def test_scene_leave_one_out_uses_raw_counts_not_mean_scene_iou():
    ref = {
        "A": {"tp": 1, "fp": 0, "fn": 0, "intersection": 1, "union": 1,
              "pred_voxels": 1, "gt_voxels": 1},
        "B": {"tp": 90, "fp": 10, "fn": 0, "intersection": 90, "union": 100,
              "pred_voxels": 100, "gt_voxels": 90},
    }
    cand = {
        "A": {"tp": 1, "fp": 0, "fn": 0, "intersection": 1, "union": 1,
              "pred_voxels": 1, "gt_voxels": 1},
        "B": {"tp": 50, "fp": 50, "fn": 0, "intersection": 50, "union": 100,
              "pred_voxels": 100, "gt_voxels": 50},
    }
    out = scene_leave_one_out_gap(ref, cand)
    expected_gap = 100.0 * 51 / 101 - 100.0 * 91 / 101
    assert abs(out["candidate_minus_reference_iou_pp"] - expected_gap) < 1e-12
    b = next(x for x in out["scenes"] if x["scene"] == "B")
    assert abs(b["gap_without_scene_pp"]) < 1e-12
    assert b["removal_change_pp"] > 0


def test_paired_scene_bootstrap_is_deterministic():
    ref = {
        "A": {"tp": 8, "fp": 2, "fn": 0, "intersection": 8, "union": 10,
              "pred_voxels": 10, "gt_voxels": 8},
        "B": {"tp": 5, "fp": 5, "fn": 0, "intersection": 5, "union": 10,
              "pred_voxels": 10, "gt_voxels": 5},
    }
    cand = {
        "A": {"tp": 7, "fp": 3, "fn": 0, "intersection": 7, "union": 10,
              "pred_voxels": 10, "gt_voxels": 7},
        "B": {"tp": 6, "fp": 4, "fn": 0, "intersection": 6, "union": 10,
              "pred_voxels": 10, "gt_voxels": 6},
    }
    a = paired_scene_bootstrap_gap(ref, cand, samples=100, seed=123)
    b = paired_scene_bootstrap_gap(ref, cand, samples=100, seed=123)
    assert a == b
    assert a["samples"] == 100
