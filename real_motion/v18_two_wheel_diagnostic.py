"""Pure helpers for the V18 two-wheel deployment-yaw diagnostic.

This module is intentionally metric-agnostic. It does not alter frozen
Moving-mIoU v2 or the micro companion; it only provides count-level accounting,
a class-only yaw intervention, robust error summaries and scene leave-one-out
recomputation used by the offline diagnostic entrypoint.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
import math

import numpy as np

from .local_st_world_model_v18_se2 import YAW_ENABLED_CLASS_IDS

TWO_WHEEL_CLASS_IDS = (2, 6)
TWO_WHEEL_CLASS_NAMES = {2: "bicycle", 6: "motorcycle"}


def renderer_yaw_delta(
    class_id: int,
    predicted_yaw_rad: float,
    *,
    zero_two_wheel_yaw: bool,
) -> float:
    """Return renderer yaw while changing *only* bicycle/motorcycle rotation.

    The deployment-known semantic yaw-enable rule remains frozen. No GT label,
    source validity or future information participates in this intervention.
    """
    cid = int(class_id)
    if cid not in YAW_ENABLED_CLASS_IDS:
        return 0.0
    if bool(zero_two_wheel_yaw) and cid in TWO_WHEEL_CLASS_IDS:
        return 0.0
    return float(predicted_yaw_rad)


def semantic_count_row(pred, gt, support, class_id: int) -> dict[str, int | float]:
    """Exact TP/FP/FN accounting for one semantic class inside one support."""
    pred = np.asarray(pred)
    gt = np.asarray(gt)
    support = np.asarray(support, dtype=bool)
    if pred.shape != gt.shape or pred.shape != support.shape:
        raise ValueError("pred/gt/support shape mismatch")
    cid = int(class_id)
    p = (pred == cid) & support
    g = (gt == cid) & support
    tp = int((p & g).sum())
    fp = int((p & ~g).sum())
    fn = int((~p & g).sum())
    union = int(tp + fp + fn)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "intersection": tp,
        "union": union,
        "pred_voxels": int(p.sum()),
        "gt_voxels": int(g.sum()),
        "iou": 100.0 * float(tp) / float(union) if union else float("nan"),
    }


def add_count_rows(rows: Iterable[Mapping[str, int | float]]) -> dict[str, int | float]:
    keys = ("tp", "fp", "fn", "intersection", "union", "pred_voxels", "gt_voxels")
    out = {k: 0 for k in keys}
    for row in rows:
        for key in keys:
            out[key] += int(row.get(key, 0))
    out["iou"] = (
        100.0 * float(out["intersection"]) / float(out["union"])
        if int(out["union"]) > 0
        else float("nan")
    )
    return out


def robust_error_summary(values: Iterable[float]) -> dict[str, int | float]:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if not arr.size:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "p90": float("nan"),
        }
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(np.quantile(arr, 0.90)),
    }


def wrapped_abs_error_rad(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if pred.shape != target.shape:
        raise ValueError("pred/target shape mismatch")
    delta = np.arctan2(np.sin(pred - target), np.cos(pred - target))
    return np.abs(delta)


def scene_leave_one_out_gap(
    reference_by_scene: Mapping[str, Mapping[str, int | float]],
    candidate_by_scene: Mapping[str, Mapping[str, int | float]],
) -> dict[str, object]:
    """Recompute dataset IoU gap after removing each scene from raw counts.

    This deliberately does not average scene IoUs. Positive
    ``removal_change_pp`` means removing that scene makes candidate-reference
    larger, i.e. the scene contributes to the candidate deficit.
    """
    scenes = sorted(set(reference_by_scene) | set(candidate_by_scene))
    ref_total = add_count_rows(reference_by_scene.get(s, {}) for s in scenes)
    cand_total = add_count_rows(candidate_by_scene.get(s, {}) for s in scenes)
    gap = float(cand_total["iou"]) - float(ref_total["iou"])
    rows = []
    for scene in scenes:
        rr = reference_by_scene.get(scene, {})
        cr = candidate_by_scene.get(scene, {})
        ref_wo = {
            k: int(ref_total[k]) - int(rr.get(k, 0))
            for k in ("tp", "fp", "fn", "intersection", "union", "pred_voxels", "gt_voxels")
        }
        cand_wo = {
            k: int(cand_total[k]) - int(cr.get(k, 0))
            for k in ("tp", "fp", "fn", "intersection", "union", "pred_voxels", "gt_voxels")
        }
        ref_wo["iou"] = (
            100.0 * ref_wo["intersection"] / ref_wo["union"]
            if ref_wo["union"] else float("nan")
        )
        cand_wo["iou"] = (
            100.0 * cand_wo["intersection"] / cand_wo["union"]
            if cand_wo["union"] else float("nan")
        )
        gap_wo = float(cand_wo["iou"]) - float(ref_wo["iou"])
        rows.append(
            {
                "scene": scene,
                "gap_without_scene_pp": gap_wo,
                "removal_change_pp": gap_wo - gap,
                "reference_union": int(rr.get("union", 0)),
                "candidate_union": int(cr.get("union", 0)),
            }
        )
    rows.sort(key=lambda r: abs(float(r["removal_change_pp"])), reverse=True)
    return {
        "candidate_minus_reference_iou_pp": gap,
        "reference_total": ref_total,
        "candidate_total": cand_total,
        "scenes": rows,
    }


def paired_scene_bootstrap_gap(
    reference_by_scene: Mapping[str, Mapping[str, int | float]],
    candidate_by_scene: Mapping[str, Mapping[str, int | float]],
    *,
    samples: int,
    seed: int,
) -> dict[str, int | float]:
    scenes = sorted(set(reference_by_scene) | set(candidate_by_scene))
    if int(samples) <= 0 or not scenes:
        return {
            "samples": 0,
            "seed": int(seed),
            "mean_pp": float("nan"),
            "p2_5_pp": float("nan"),
            "p97_5_pp": float("nan"),
        }
    rng = np.random.default_rng(int(seed))
    vals = []
    for _ in range(int(samples)):
        draw = rng.choice(scenes, size=len(scenes), replace=True)
        rr = add_count_rows(reference_by_scene[str(s)] for s in draw)
        cr = add_count_rows(candidate_by_scene[str(s)] for s in draw)
        if int(rr["union"]) <= 0 or int(cr["union"]) <= 0:
            continue
        vals.append(float(cr["iou"]) - float(rr["iou"]))
    if not vals:
        return {
            "samples": 0,
            "seed": int(seed),
            "mean_pp": float("nan"),
            "p2_5_pp": float("nan"),
            "p97_5_pp": float("nan"),
        }
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "samples": int(arr.size),
        "seed": int(seed),
        "mean_pp": float(arr.mean()),
        "p2_5_pp": float(np.quantile(arr, 0.025)),
        "p97_5_pp": float(np.quantile(arr, 0.975)),
    }


def finite_or_none(value: float):
    return float(value) if math.isfinite(float(value)) else None
