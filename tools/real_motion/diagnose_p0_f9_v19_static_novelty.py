#!/usr/bin/env python3
"""One-pass diagnosis for the V19 Static-Memory / Factorized-New-FOV stack.

The script intentionally separates two questions that are otherwise easy to
conflate:

1) Existing full-eval JSON, zero GPU:
   decompose every add-only branch into
     * position + semantic correct,
     * position correct / semantic wrong,
     * added into GT-free space.
   This is exact because both Static Memory and Factorized Novelty are protected
   add-only branches.

2) One scene-stratified diagnostic pass, no retraining and no full-4369 rerun:
   * diagnose Static-Memory candidate reliability by class, observation count,
     recency and history-footprint boundary distance;
   * run the frozen factorized head once per sampled window, then evaluate a
     whole presence/vertical threshold grid from compact score histograms;
   * compare network semantic classes against nearest-anchor semantics and a
     GT-semantic diagnostic ceiling;
   * score the final context-only composition V18 + Novelty, while Static Memory
     remains conditioning/context only.

No threshold combination re-runs V18.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from real_motion.metrics.moving_miou_v2 import (
    DYNAMIC_CLASS_IDS,
    NUSCENES_LABELS,
)
from real_motion.prepared import load_nuscenes_window_raw
from real_motion.runtime_config import (
    add_config_args,
    load_runtime_config,
    make_prepare_config,
)
from real_motion.strong_w2det import StrongW2DetConfig
from real_motion.v19_innovation import base_explained_bev
from real_motion.v19_scene_memory import protected_add_only
from real_motion.v19_static_novelty import (
    history_grid_footprint_bev_sequence,
)
from real_motion.v19_static_novelty_factorized import (
    FactorizedStaticNewFOVHead,
)
from tools.real_motion import eval_p0_f9_v18_se2 as base
from tools.real_motion import eval_p0_f9_v18_full_validation as full
from tools.real_motion.benchmark_p0_f9_v18_runtime import (
    _forecast_once,
    _release_gpu_inputs,
    _stage_gpu_inputs,
)
from tools.real_motion.eval_p0_f9_v17_local_stwm import window_from_record
from tools.real_motion.eval_p0_f9_v19_factorized_static_new_fov import (
    CachedSource,
    _ComponentLRU,
    _HistoryAlignmentLRU,
    _HistoryFutureAlignmentLRU,
    _build_anchor_context_sequence,
    _build_history_static_from_pair_cache,
    _prepare_record_from_raw,
)
from tools.real_motion.train_p0_f9_v18_se2_clean import (
    PROTOCOL as CLEAN_PROTOCOL,
)
from tools.real_motion.train_p0_f9_v19_factorized_static_new_fov import (
    HEAD_TYPE,
    PROTOCOL as FACTORIZED_TRAIN_PROTOCOL,
)


PROTOCOL = "p0_f9_v19_static_novelty_onepass_diagnosis_v1"
HORIZONS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
MAIN_HI = (1, 3, 5)
SEM_CLASSES = 17
FREE_LABEL = 17
DYNAMIC_IDS = tuple(int(x) for x in DYNAMIC_CLASS_IDS)


def _parse_thresholds(text: str) -> list[float]:
    vals = sorted({float(x.strip()) for x in str(text).split(",") if x.strip()})
    if not vals:
        raise ValueError("threshold list is empty")
    if vals[0] < 0.0 or vals[-1] > 1.0:
        raise ValueError("thresholds must lie in [0,1]")
    return vals


def _with_threshold(values: list[float], x: float) -> list[float]:
    return sorted({*(float(v) for v in values), float(x)})


def _add_only_triage(raw_counts: dict, before: str, after: str) -> dict:
    a = raw_counts[before]
    b = raw_counts[after]
    occ_tp = int(
        np.asarray(b["occ_inter"], dtype=np.int64).sum()
        - np.asarray(a["occ_inter"], dtype=np.int64).sum()
    )
    free_fp = int(
        np.asarray(b["occ_union"], dtype=np.int64).sum()
        - np.asarray(a["occ_union"], dtype=np.int64).sum()
    )
    sem_tp = int(
        np.asarray(b["sem_inter"], dtype=np.int64).sum()
        - np.asarray(a["sem_inter"], dtype=np.int64).sum()
    )
    if min(occ_tp, free_fp, sem_tp) < 0 or sem_tp > occ_tp:
        raise RuntimeError(
            "branch is not behaving as protected add-only; raw-count triage "
            "assumptions are violated"
        )
    wrong_sem = int(occ_tp - sem_tp)
    total = int(occ_tp + free_fp)
    return {
        "added_voxels": total,
        "position_and_semantic_correct": sem_tp,
        "position_correct_semantic_wrong": wrong_sem,
        "added_into_gt_free": free_fp,
        "occupancy_precision": float(occ_tp / max(total, 1)),
        "semantic_precision_over_all_additions": float(sem_tp / max(total, 1)),
        "semantic_accuracy_given_position_correct": float(
            sem_tp / max(occ_tp, 1)
        ),
        "fractions": {
            "position_and_semantic_correct": float(sem_tp / max(total, 1)),
            "position_correct_semantic_wrong": float(wrong_sem / max(total, 1)),
            "added_into_gt_free": float(free_fp / max(total, 1)),
        },
    }


def diagnose_full_json(path: str) -> dict:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = obj.get("raw_counts")
    if not isinstance(raw, dict):
        raise RuntimeError("full eval JSON does not contain raw_counts")
    required = {
        "v18",
        "v18_static",
        "v18_static_factorized_new_fov",
    }
    if not required.issubset(raw):
        raise RuntimeError(
            f"full eval JSON missing variants: {sorted(required - set(raw))}"
        )
    return {
        "Static": _add_only_triage(raw, "v18", "v18_static"),
        "Novelty": _add_only_triage(
            raw,
            "v18_static",
            "v18_static_factorized_new_fov",
        ),
    }


def _scene_stratified_records(records: list, windows_per_scene: int) -> list:
    if int(windows_per_scene) <= 0:
        raise ValueError("windows-per-scene must be positive")
    grouped = defaultdict(list)
    for rec in records:
        grouped[str(window_from_record(rec).scene_name)].append(rec)

    selected = []
    for scene in sorted(grouped):
        rows = grouped[scene]
        k = min(int(windows_per_scene), len(rows))
        if k == len(rows):
            idx = list(range(len(rows)))
        else:
            raw_idx = np.linspace(
                0,
                len(rows) - 1,
                k + 2,
                dtype=np.float64,
            )[1:-1]
            idx = []
            for x in raw_idx:
                j = int(round(float(x)))
                if j not in idx:
                    idx.append(j)
            if len(idx) < k:
                for j in range(len(rows)):
                    if j not in idx:
                        idx.append(j)
                    if len(idx) == k:
                        break
            idx = sorted(idx[:k])
        selected.extend(rows[j] for j in idx)
    return selected


def _new_static_stat() -> dict:
    return {
        "candidates": 0,
        "occupancy_tp": 0,
        "semantic_tp": 0,
        "free_fp": 0,
    }


def _merge_static_stat(dst: dict, *, n: int, occ_tp: int, sem_tp: int) -> None:
    dst["candidates"] += int(n)
    dst["occupancy_tp"] += int(occ_tp)
    dst["semantic_tp"] += int(sem_tp)
    dst["free_fp"] += int(n - occ_tp)


def _finalize_static_stat(x: dict) -> dict:
    n = int(x["candidates"])
    tp = int(x["occupancy_tp"])
    sem = int(x["semantic_tp"])
    return {
        **{k: int(v) for k, v in x.items()},
        "occupancy_precision": float(tp / max(n, 1)),
        "semantic_precision_over_all_candidates": float(sem / max(n, 1)),
        "semantic_accuracy_given_position_correct": float(
            sem / max(tp, 1)
        ),
    }


def _boundary_bin(distance_m: np.ndarray) -> np.ndarray:
    # 0:<1m, 1:1-2m, 2:2-4m, 3:4-8m, 4:>=8m
    return np.digitize(
        np.asarray(distance_m, dtype=np.float32),
        np.asarray([1.0, 2.0, 4.0, 8.0], dtype=np.float32),
        right=False,
    ).astype(np.int8)


BOUNDARY_LABELS = ("<1m", "1-2m", "2-4m", "4-8m", ">=8m")


def _accumulate_grouped_static(
    store: dict,
    group_name: str,
    values: np.ndarray,
    candidate: np.ndarray,
    gt: np.ndarray,
    pred_semantic: np.ndarray,
) -> None:
    vals = np.asarray(values)
    if vals.shape != candidate.shape:
        vals = np.broadcast_to(vals, candidate.shape)
    for v in np.unique(vals[candidate]):
        mask = candidate & (vals == v)
        n = int(mask.sum())
        if n == 0:
            continue
        occ = mask & (gt != FREE_LABEL)
        sem = mask & (gt == pred_semantic)
        key = str(v.item() if hasattr(v, "item") else v)
        row = store.setdefault(group_name, {}).setdefault(
            key,
            _new_static_stat(),
        )
        _merge_static_stat(
            row,
            n=n,
            occ_tp=int(occ.sum()),
            sem_tp=int(sem.sum()),
        )


def _suffix2(x: np.ndarray) -> np.ndarray:
    y = np.asarray(x)
    y = y[:, ::-1, ::-1, ...]
    y = np.cumsum(np.cumsum(y, axis=1), axis=2)
    return y[:, ::-1, ::-1, ...]


def _baseline_counts():
    return {
        "occ_inter": np.zeros(6, dtype=np.int64),
        "occ_union": np.zeros(6, dtype=np.int64),
        "sem_inter": np.zeros((6, SEM_CLASSES), dtype=np.int64),
        "sem_union": np.zeros((6, SEM_CLASSES), dtype=np.int64),
    }


def _accumulate_v18_baseline(dst: dict, hi: int, pred: np.ndarray, gt: np.ndarray):
    p = np.asarray(pred, dtype=np.uint8).reshape(-1)
    g = np.asarray(gt, dtype=np.uint8).reshape(-1)
    po = p != FREE_LABEL
    go = g != FREE_LABEL
    dst["occ_inter"][hi] += int(np.count_nonzero(po & go))
    dst["occ_union"][hi] += int(np.count_nonzero(po | go))

    for cid in range(SEM_CLASSES):
        pp = p == cid
        gg = g == cid
        inter = int(np.count_nonzero(pp & gg))
        union = int(np.count_nonzero(pp | gg))
        dst["sem_inter"][hi, cid] += inter
        dst["sem_union"][hi, cid] += union


def _new_threshold_hist(P: int, V: int):
    return {
        "tp": np.zeros((6, P, V), dtype=np.int64),
        "fp": np.zeros((6, P, V), dtype=np.int64),
        "correct": np.zeros(
            (6, P, V, SEM_CLASSES),
            dtype=np.int64,
        ),
        "wrong_pred": np.zeros(
            (6, P, V, SEM_CLASSES),
            dtype=np.int64,
        ),
    }


def _hist2(cell: np.ndarray, size: int) -> np.ndarray:
    return np.bincount(cell, minlength=size).astype(np.int64, copy=False)


def _accumulate_threshold_hist(
    hist: dict,
    hi: int,
    presence_prob_xy: np.ndarray,
    vertical_prob_zxy: np.ndarray,
    pred_class_xy: np.ndarray,
    gt_xyz: np.ndarray,
    eligible_zxy: np.ndarray,
    presence_thresholds: list[float],
    vertical_thresholds: list[float],
) -> None:
    pths = np.asarray(presence_thresholds, dtype=np.float32)
    vths = np.asarray(vertical_thresholds, dtype=np.float32)
    P, V = len(pths), len(vths)

    pb_xy = np.searchsorted(
        pths,
        np.asarray(presence_prob_xy, dtype=np.float32),
        side="right",
    ) - 1
    vb = np.searchsorted(
        vths,
        np.asarray(vertical_prob_zxy, dtype=np.float32),
        side="right",
    ) - 1

    elig = np.asarray(eligible_zxy, dtype=bool)
    pb = np.broadcast_to(pb_xy[None], vb.shape)
    valid = elig & (pb >= 0) & (vb >= 0)
    if not bool(valid.any()):
        return

    pbv = pb[valid].astype(np.int64, copy=False)
    vbv = vb[valid].astype(np.int64, copy=False)
    cell = pbv * V + vbv
    cell_n = P * V

    gt_zxy = np.asarray(gt_xyz, dtype=np.uint8).transpose(2, 0, 1)
    g = gt_zxy[valid].astype(np.int64, copy=False)
    pred = np.broadcast_to(
        np.asarray(pred_class_xy, dtype=np.int64)[None],
        vb.shape,
    )[valid]

    occ = g < SEM_CLASSES
    hist["tp"][hi] += _hist2(cell[occ], cell_n).reshape(P, V)
    hist["fp"][hi] += _hist2(cell[~occ], cell_n).reshape(P, V)

    correct = occ & (pred == g)
    if bool(correct.any()):
        encoded = (
            cell[correct] * SEM_CLASSES
            + g[correct]
        )
        hist["correct"][hi] += _hist2(
            encoded,
            cell_n * SEM_CLASSES,
        ).reshape(P, V, SEM_CLASSES)

    wrong = ~correct
    if bool(wrong.any()):
        encoded = (
            cell[wrong] * SEM_CLASSES
            + pred[wrong]
        )
        hist["wrong_pred"][hi] += _hist2(
            encoded,
            cell_n * SEM_CLASSES,
        ).reshape(P, V, SEM_CLASSES)


def _safe_iou(inter: np.ndarray, union: np.ndarray) -> np.ndarray:
    inter = np.asarray(inter, dtype=np.float64)
    union = np.asarray(union, dtype=np.float64)
    out = np.full(inter.shape, np.nan, dtype=np.float64)
    np.divide(inter, union, out=out, where=union > 0)
    return 100.0 * out


def _baseline_report(base_counts: dict) -> dict:
    occ = _safe_iou(base_counts["occ_inter"], base_counts["occ_union"])
    sem = _safe_iou(base_counts["sem_inter"], base_counts["sem_union"])
    miou = np.nanmean(sem, axis=1)
    return {
        "IoU": float(np.nanmean(occ)),
        "mIoU": float(np.nanmean(miou)),
        "main_1_2_3s": {
            "IoU": float(np.nanmean(occ[list(MAIN_HI)])),
            "mIoU": float(np.nanmean(miou[list(MAIN_HI)])),
        },
        "per_horizon": {
            str(HORIZONS[i]): {
                "IoU": float(occ[i]),
                "mIoU": float(miou[i]),
            }
            for i in range(6)
        },
    }


def _evaluate_threshold_grid(
    base_counts: dict,
    hist: dict,
    presence_thresholds: list[float],
    vertical_thresholds: list[float],
) -> dict:
    tp = _suffix2(hist["tp"])
    fp = _suffix2(hist["fp"])
    correct = _suffix2(hist["correct"])
    wrong = _suffix2(hist["wrong_pred"])

    base_report = _baseline_report(base_counts)
    rows = []
    for pi, pt in enumerate(presence_thresholds):
        for vi, vt in enumerate(vertical_thresholds):
            oi = base_counts["occ_inter"] + tp[:, pi, vi]
            ou = base_counts["occ_union"] + fp[:, pi, vi]
            si = base_counts["sem_inter"] + correct[:, pi, vi]
            su = base_counts["sem_union"] + wrong[:, pi, vi]

            occ = _safe_iou(oi, ou)
            sem = _safe_iou(si, su)
            miou = np.nanmean(sem, axis=1)
            added_tp = int(tp[:, pi, vi].sum())
            added_fp = int(fp[:, pi, vi].sum())
            row = {
                "presence_threshold": float(pt),
                "vertical_threshold": float(vt),
                "IoU": float(np.nanmean(occ)),
                "mIoU": float(np.nanmean(miou)),
                "main_1_2_3s": {
                    "IoU": float(np.nanmean(occ[list(MAIN_HI)])),
                    "mIoU": float(np.nanmean(miou[list(MAIN_HI)])),
                },
                "delta_vs_v18": {
                    "IoU": float(
                        np.nanmean(occ) - base_report["IoU"]
                    ),
                    "mIoU": float(
                        np.nanmean(miou) - base_report["mIoU"]
                    ),
                    "main_1_2_3s_IoU": float(
                        np.nanmean(occ[list(MAIN_HI)])
                        - base_report["main_1_2_3s"]["IoU"]
                    ),
                    "main_1_2_3s_mIoU": float(
                        np.nanmean(miou[list(MAIN_HI)])
                        - base_report["main_1_2_3s"]["mIoU"]
                    ),
                },
                "added_tp": added_tp,
                "added_fp": added_fp,
                "addition_precision": float(
                    added_tp / max(added_tp + added_fp, 1)
                ),
                "semantic_correct_additions": int(
                    correct[:, pi, vi].sum()
                ),
            }
            rows.append(row)

    rows.sort(
        key=lambda x: (
            x["main_1_2_3s"]["mIoU"],
            x["main_1_2_3s"]["IoU"],
        ),
        reverse=True,
    )
    return {
        "baseline_v18": base_report,
        "best_by_main_1_2_3s_mIoU": rows[0],
        "top10": rows[:10],
        "all": rows,
    }


def _lookup_threshold_row(
    report: dict,
    presence_threshold: float,
    vertical_threshold: float,
) -> dict:
    for row in report["all"]:
        if (
            abs(row["presence_threshold"] - float(presence_threshold)) < 1e-9
            and abs(row["vertical_threshold"] - float(vertical_threshold)) < 1e-9
        ):
            return row
    raise KeyError("checkpoint threshold pair not found in grid")


def _autocast(device: torch.device, enabled: bool):
    return torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=bool(enabled and device.type == "cuda"),
    )


def _static_summary_finalize(static_diag: dict) -> dict:
    out = {
        "overall": _finalize_static_stat(static_diag["overall"]),
        "headroom": static_diag["headroom"],
        "t0_contradiction_filtered": int(
            static_diag["t0_contradiction_filtered"]
        ),
    }
    for group in (
        "by_class",
        "by_observation_count",
        "by_recent_age_s",
        "by_boundary_distance",
    ):
        rows = {}
        for key, val in static_diag.get(group, {}).items():
            row = _finalize_static_stat(val)
            if group == "by_class":
                cid = int(key)
                row["class_name"] = (
                    NUSCENES_LABELS[cid]
                    if 0 <= cid < len(NUSCENES_LABELS)
                    else str(cid)
                )
            rows[str(key)] = row
        out[group] = rows

    h = out["headroom"]
    h["candidate_occupancy_tp_coverage_of_eligible_gt_static_miss"] = float(
        h["candidate_occupancy_tp"]
        / max(h["eligible_gt_static_miss"], 1)
    )
    h[
        "candidate_occupancy_tp_coverage_of_history_observed_gt_static_miss"
    ] = float(
        h["candidate_occupancy_tp"]
        / max(h["history_observed_gt_static_miss"], 1)
    )
    return out


def run_stratified_diagnosis(a, full_triage: dict) -> dict:
    required = (
        "val_cache",
        "base_checkpoint",
        "novelty_checkpoint",
        "dataroot",
        "info_pkl",
        "output",
    )
    missing = [x for x in required if not getattr(a, x, None)]
    if missing:
        raise ValueError(
            "stratified diagnosis requires: "
            + ", ".join("--" + x.replace("_", "-") for x in missing)
        )

    pcfg = make_prepare_config(load_runtime_config(a.config, a.override))
    _, records = base.load_cache(a.val_cache)
    selected = _scene_stratified_records(
        list(records),
        int(a.windows_per_scene),
    )
    if int(a.max_scenes) > 0:
        keep_scenes = sorted(
            {
                str(window_from_record(r).scene_name)
                for r in selected
            }
        )[: int(a.max_scenes)]
        keep = set(keep_scenes)
        selected = [
            r for r in selected
            if str(window_from_record(r).scene_name) in keep
        ]
    if not selected:
        raise RuntimeError("empty scene-stratified sample")

    selected_scenes = sorted(
        {
            str(window_from_record(r).scene_name)
            for r in selected
        }
    )

    device = torch.device(
        a.device
        if a.device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    amp = device.type == "cuda" and not bool(a.no_amp)

    _, base_model, _ = full._load_model(
        a.base_checkpoint,
        CLEAN_PROTOCOL,
        device,
    )
    nov_ck = torch.load(
        a.novelty_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    if nov_ck.get("protocol") != FACTORIZED_TRAIN_PROTOCOL:
        raise RuntimeError(
            f"unexpected novelty checkpoint protocol: {nov_ck.get('protocol')}"
        )
    if nov_ck.get("head_type") != HEAD_TYPE:
        raise RuntimeError(
            f"unexpected novelty head type: {nov_ck.get('head_type')}"
        )
    novelty = FactorizedStaticNewFOVHead(
        **dict(nov_ck["architecture"])
    ).to(device)
    novelty.load_state_dict(nov_ck["model_state_dict"], strict=True)
    novelty.eval()

    selected_presence = float(nov_ck["selected_presence_threshold"])
    selected_vertical = float(nov_ck["selected_vertical_threshold"])
    pths = _with_threshold(
        _parse_thresholds(a.presence_thresholds),
        selected_presence,
    )
    vths = _with_threshold(
        _parse_thresholds(a.vertical_thresholds),
        selected_vertical,
    )

    source = CachedSource(a.dataroot, info_pkl=a.info_pkl, verbose=False)
    strong_cfg = StrongW2DetConfig(free_label=int(pcfg.free_label))
    component_cache = _ComponentLRU(maxsize=1024)
    history_cache = _HistoryAlignmentLRU(maxsize=96)
    pair_cache = _HistoryFutureAlignmentLRU(maxsize=192)

    base_counts = _baseline_counts()
    hists = {
        "network_semantic": _new_threshold_hist(len(pths), len(vths)),
        "anchor_semantic_fallback_network": _new_threshold_hist(
            len(pths), len(vths)
        ),
        "gt_semantic_oracle_on_true_additions": _new_threshold_hist(
            len(pths), len(vths)
        ),
    }
    static_diag = {
        "overall": _new_static_stat(),
        "by_class": {},
        "by_observation_count": {},
        "by_recent_age_s": {},
        "by_boundary_distance": {},
        "t0_contradiction_filtered": 0,
        "headroom": {
            "eligible_gt_static_miss": 0,
            "history_observed_gt_static_miss": 0,
            "candidate_occupancy_tp": 0,
        },
    }

    pair_hits = pair_misses = 0
    started = time.perf_counter()

    for wi, rec in enumerate(selected, start=1):
        w = window_from_record(rec)
        raw = load_nuscenes_window_raw(
            source,
            w,
            pcfg,
            include_gt=True,
        )
        state = _prepare_record_from_raw(
            rec,
            raw,
            source,
            pcfg,
            strong_cfg,
            device,
            component_cache,
        )
        _stage_gpu_inputs(state, device)
        try:
            pred_all = _forecast_once(
                base_model,
                state,
                pcfg,
                strong_cfg,
                device,
            )
        finally:
            _release_gpu_inputs(state)

        prepared_history = [
            history_cache.get_or_build(
                str(w.scene_name),
                str(tok),
                raw["history_occ"][ti],
                raw["history_observed"][ti],
                raw["history_poses"][ti],
                grid=pcfg.grid,
                dynamic_class_ids=DYNAMIC_IDS,
            )
            for ti, tok in enumerate(w.history_tokens)
        ]
        sem, geo, static_all, pair_stats = (
            _build_history_static_from_pair_cache(
                scene=str(w.scene_name),
                history_tokens=w.history_tokens,
                future_tokens=w.future_tokens,
                prepared_history=prepared_history,
                future_poses=raw["future_poses"],
                pair_cache=pair_cache,
                grid=pcfg.grid,
                free_label=int(pcfg.free_label),
                workers=int(a.alignment_workers),
            )
        )
        pair_hits += int(pair_stats["hits"])
        pair_misses += int(pair_stats["misses"])

        history_poses = np.asarray(raw["history_poses"], dtype=np.float64)
        future_poses = np.asarray(raw["future_poses"], dtype=np.float64)
        footprint_all = history_grid_footprint_bev_sequence(
            history_poses,
            future_poses,
            pcfg.grid,
            workers=int(a.alignment_workers),
        )

        pred_stack = np.asarray(pred_all, dtype=np.uint8)
        static_all = np.asarray(static_all, dtype=np.uint8)
        explained = protected_add_only(
            pred_stack,
            static_all,
            free_label=int(pcfg.free_label),
        )
        explained_free = explained == int(pcfg.free_label)
        new_fov = ~footprint_all
        anchor_sem, anchor_profile, anchor_dist = (
            _build_anchor_context_sequence(
                static_all,
                footprint_all,
                free_label=int(pcfg.free_label),
                voxel_size_xy_m=float(pcfg.grid.voxel_size[0]),
                workers=int(a.alignment_workers),
            )
        )

        sem_t = torch.from_numpy(sem[None]).to(device)
        geo_t = torch.from_numpy(geo[None]).to(device)
        base_t = torch.from_numpy(
            base_explained_bev(
                explained,
                free_label=int(pcfg.free_label),
            )[None]
        ).to(device)
        nf_t = torch.from_numpy(new_fov[None]).to(device)
        anchor_sem_t = torch.from_numpy(anchor_sem[None]).to(device)
        anchor_profile_t = torch.from_numpy(
            anchor_profile.transpose(0, 3, 1, 2)[None]
        ).to(device)
        anchor_dist_t = torch.from_numpy(anchor_dist[None]).to(device)

        with torch.inference_mode(), _autocast(device, amp):
            out = novelty(
                sem_t,
                geo_t,
                base_t,
                nf_t,
                anchor_sem_t,
                anchor_profile_t,
                anchor_dist_t,
            )
        presence_prob = (
            torch.sigmoid(out["presence_logits"].float())[0]
            .cpu()
            .numpy()
        )
        vertical_prob = (
            torch.sigmoid(out["vertical_logits"].float())[0]
            .cpu()
            .numpy()
        )
        network_sem = (
            out["semantic_logits"].float().argmax(dim=2)[0]
            .cpu()
            .numpy()
            .astype(np.uint8)
        )

        # Static-Memory reliability diagnosis.
        for hi in range(6):
            pred = pred_stack[hi]
            gt = np.asarray(raw["future_gt_occ"][hi], dtype=np.uint8)
            static_render = static_all[hi]
            footprint = footprint_all[hi]
            shape = gt.shape

            obs_count = np.zeros(shape, dtype=np.uint8)
            recent_idx = np.full(shape, -1, dtype=np.int8)
            t0_clear = np.zeros(shape, dtype=bool)
            t0_write = np.zeros(shape, dtype=bool)
            for ti, htok in enumerate(w.history_tokens):
                pair = pair_cache.get(
                    str(w.scene_name),
                    str(htok),
                    str(w.future_tokens[hi]),
                )
                if pair is None:
                    raise RuntimeError("expected history/future pair cache hit")
                clear_flat = np.asarray(pair[3], dtype=np.int64)
                write_flat = np.asarray(pair[4], dtype=np.int64)
                if len(write_flat):
                    flat_count = obs_count.reshape(-1)
                    flat_recent = recent_idx.reshape(-1)
                    flat_count[write_flat] += np.uint8(1)
                    flat_recent[write_flat] = np.int8(ti)
                if ti == 5:
                    if len(clear_flat):
                        t0_clear.reshape(-1)[np.unique(clear_flat)] = True
                    if len(write_flat):
                        t0_write.reshape(-1)[write_flat] = True

            t0_observed_free = t0_clear & ~t0_write
            raw_candidate = (
                (static_render != FREE_LABEL)
                & (pred == FREE_LABEL)
                & (obs_count > 0)
            )
            filtered = raw_candidate & ~t0_observed_free
            static_diag["t0_contradiction_filtered"] += int(
                (raw_candidate & t0_observed_free).sum()
            )

            n = int(filtered.sum())
            occ = filtered & (gt != FREE_LABEL)
            sem_ok = filtered & (gt == static_render)
            _merge_static_stat(
                static_diag["overall"],
                n=n,
                occ_tp=int(occ.sum()),
                sem_tp=int(sem_ok.sum()),
            )

            _accumulate_grouped_static(
                static_diag,
                "by_class",
                static_render,
                filtered,
                gt,
                static_render,
            )
            _accumulate_grouped_static(
                static_diag,
                "by_observation_count",
                obs_count,
                filtered,
                gt,
                static_render,
            )
            recent_age = np.where(
                recent_idx >= 0,
                (5 - recent_idx.astype(np.int16))
                * float(pcfg.frame_dt_s),
                -1.0,
            ).astype(np.float32)
            _accumulate_grouped_static(
                static_diag,
                "by_recent_age_s",
                recent_age,
                filtered,
                gt,
                static_render,
            )

            try:
                from scipy.ndimage import distance_transform_edt

                boundary_m = (
                    distance_transform_edt(footprint)
                    * float(pcfg.grid.voxel_size[0])
                )
                bbin = _boundary_bin(boundary_m)
                _accumulate_grouped_static(
                    static_diag,
                    "by_boundary_distance",
                    bbin[:, :, None],
                    filtered,
                    gt,
                    static_render,
                )
            except ImportError:
                pass

            footprint3d = np.broadcast_to(
                footprint[:, :, None],
                shape,
            )
            gt_static = (
                (gt != FREE_LABEL)
                & ~np.isin(
                    gt,
                    np.asarray(DYNAMIC_IDS, dtype=np.uint8),
                )
            )
            eligible_gt = (
                gt_static
                & (pred == FREE_LABEL)
                & footprint3d
                & ~t0_observed_free
            )
            observed_gt = eligible_gt & (obs_count > 0)
            h = static_diag["headroom"]
            h["eligible_gt_static_miss"] += int(eligible_gt.sum())
            h["history_observed_gt_static_miss"] += int(
                observed_gt.sum()
            )
            h["candidate_occupancy_tp"] += int(occ.sum())

            # Baseline V18 counts and context-only Novelty threshold histograms.
            _accumulate_v18_baseline(base_counts, hi, pred, gt)
            eligible_zxy = (
                new_fov[hi][None]
                & explained_free[hi].transpose(2, 0, 1)
            )
            net_cls = network_sem[hi]
            anc_cls = np.asarray(anchor_sem[hi], dtype=np.uint8)
            anc_fallback = np.where(
                anc_cls < SEM_CLASSES,
                anc_cls,
                net_cls,
            ).astype(np.uint8)

            gt_zxy = gt.transpose(2, 0, 1)
            oracle_cls_zxy = np.where(
                gt_zxy < SEM_CLASSES,
                gt_zxy,
                np.broadcast_to(net_cls[None], gt_zxy.shape),
            )
            # The histogram helper expects one class per BEV column.  For the
            # GT oracle, semantic class varies by Z only when the GT column is
            # semantically mixed; accumulate it directly below.
            _accumulate_threshold_hist(
                hists["network_semantic"],
                hi,
                presence_prob[hi],
                vertical_prob[hi],
                net_cls,
                gt,
                eligible_zxy,
                pths,
                vths,
            )
            _accumulate_threshold_hist(
                hists["anchor_semantic_fallback_network"],
                hi,
                presence_prob[hi],
                vertical_prob[hi],
                anc_fallback,
                gt,
                eligible_zxy,
                pths,
                vths,
            )

            # GT-semantic ceiling: all true additions are class-correct, while
            # GT-free false positives keep the network class.  This is not a
            # deployable variant and is reported only as a diagnostic ceiling.
            pths_np = np.asarray(pths, dtype=np.float32)
            vths_np = np.asarray(vths, dtype=np.float32)
            pb_xy = np.searchsorted(
                pths_np,
                presence_prob[hi].astype(np.float32),
                side="right",
            ) - 1
            vb = np.searchsorted(
                vths_np,
                vertical_prob[hi].astype(np.float32),
                side="right",
            ) - 1
            pb = np.broadcast_to(pb_xy[None], vb.shape)
            valid = eligible_zxy & (pb >= 0) & (vb >= 0)
            if bool(valid.any()):
                P, V = len(pths), len(vths)
                cell = (
                    pb[valid].astype(np.int64) * V
                    + vb[valid].astype(np.int64)
                )
                g = gt_zxy[valid].astype(np.int64)
                net_z = np.broadcast_to(net_cls[None], vb.shape)[valid]
                occ_mask = g < SEM_CLASSES
                oh = hists["gt_semantic_oracle_on_true_additions"]
                oh["tp"][hi] += _hist2(
                    cell[occ_mask],
                    P * V,
                ).reshape(P, V)
                oh["fp"][hi] += _hist2(
                    cell[~occ_mask],
                    P * V,
                ).reshape(P, V)
                if bool(occ_mask.any()):
                    enc = cell[occ_mask] * SEM_CLASSES + g[occ_mask]
                    oh["correct"][hi] += _hist2(
                        enc,
                        P * V * SEM_CLASSES,
                    ).reshape(P, V, SEM_CLASSES)
                if bool((~occ_mask).any()):
                    enc = (
                        cell[~occ_mask] * SEM_CLASSES
                        + net_z[~occ_mask]
                    )
                    oh["wrong_pred"][hi] += _hist2(
                        enc,
                        P * V * SEM_CLASSES,
                    ).reshape(P, V, SEM_CLASSES)

        if wi == 1 or wi % 25 == 0 or wi == len(selected):
            elapsed = max(time.perf_counter() - started, 1e-9)
            print(
                f"static_novelty_diag {wi}/{len(selected)} "
                f"rate={wi/elapsed:.3f} win/s",
                flush=True,
            )

    threshold_reports = {
        mode: _evaluate_threshold_grid(
            base_counts,
            hist,
            pths,
            vths,
        )
        for mode, hist in hists.items()
    }
    for mode, rep in threshold_reports.items():
        rep["checkpoint_selected_thresholds"] = _lookup_threshold_row(
            rep,
            selected_presence,
            selected_vertical,
        )
        if mode == "gt_semantic_oracle_on_true_additions":
            rep["diagnostic_only_future_gt_semantic"] = True

    static_out = _static_summary_finalize(static_diag)
    # Human-readable boundary labels.
    if static_out.get("by_boundary_distance"):
        static_out["by_boundary_distance"] = {
            BOUNDARY_LABELS[int(k)]: v
            for k, v in static_out["by_boundary_distance"].items()
            if 0 <= int(k) < len(BOUNDARY_LABELS)
        }

    result = {
        "protocol": PROTOCOL,
        "full_json_triage": full_triage,
        "selection": {
            "strategy": "scene_stratified_even_interior_positions",
            "windows_per_scene": int(a.windows_per_scene),
            "num_scenes": int(len(selected_scenes)),
            "num_windows": int(len(selected)),
            "scenes": selected_scenes,
        },
        "checkpoint": {
            "base": str(Path(a.base_checkpoint).resolve()),
            "novelty": str(Path(a.novelty_checkpoint).resolve()),
            "selected_presence_threshold": selected_presence,
            "selected_vertical_threshold": selected_vertical,
        },
        "static_memory_candidate_reliability": static_out,
        "novelty_context_only_threshold_sweep": {
            "composition": (
                "Static Memory is used as factorized-head context/base-explained "
                "but is NOT written to final occupancy; proposals are added to V18."
            ),
            "presence_thresholds": pths,
            "vertical_thresholds": vths,
            "semantic_modes": threshold_reports,
        },
        "history_future_pair_cache": {
            "hits": int(pair_hits),
            "misses": int(pair_misses),
            "hit_rate": float(
                pair_hits / max(pair_hits + pair_misses, 1)
            ),
        },
        "elapsed_s": float(time.perf_counter() - started),
    }
    return result


def _print_triage(triage: dict) -> None:
    print("\n=== FULL JSON ADD-ONLY TRIAGE ===")
    for name in ("Static", "Novelty"):
        x = triage[name]
        print(
            f"{name:8s} added={x['added_voxels']} "
            f"sem_correct={x['position_and_semantic_correct']} "
            f"pos_correct_sem_wrong={x['position_correct_semantic_wrong']} "
            f"gt_free_fp={x['added_into_gt_free']} "
            f"occP={100*x['occupancy_precision']:.2f}% "
            f"semAcc|TP={100*x['semantic_accuracy_given_position_correct']:.2f}%"
        )


def _print_stratified(result: dict) -> None:
    print("\n=== SCENE-STRATIFIED STATIC / NOVELTY DIAGNOSIS ===")
    s = result["selection"]
    print(
        f"sample scenes={s['num_scenes']} windows={s['num_windows']} "
        f"windows_per_scene={s['windows_per_scene']}"
    )
    st = result["static_memory_candidate_reliability"]
    x = st["overall"]
    print(
        "Static candidates "
        f"n={x['candidates']} "
        f"occP={100*x['occupancy_precision']:.2f}% "
        f"semP={100*x['semantic_precision_over_all_candidates']:.2f}% "
        f"semAcc|TP={100*x['semantic_accuracy_given_position_correct']:.2f}%"
    )
    h = st["headroom"]
    print(
        "Static headroom "
        f"eligible_gt_miss={h['eligible_gt_static_miss']} "
        f"history_observed_gt_miss={h['history_observed_gt_static_miss']} "
        f"candidate_tp={h['candidate_occupancy_tp']} "
        f"coverage={100*h['candidate_occupancy_tp_coverage_of_history_observed_gt_static_miss']:.2f}%"
    )

    modes = result["novelty_context_only_threshold_sweep"]["semantic_modes"]
    for mode, rep in modes.items():
        b = rep["best_by_main_1_2_3s_mIoU"]
        c = rep["checkpoint_selected_thresholds"]
        print(
            f"{mode}: best=({b['presence_threshold']:.2f},"
            f"{b['vertical_threshold']:.2f}) "
            f"main_mIoU={b['main_1_2_3s']['mIoU']:.3f} "
            f"d={b['delta_vs_v18']['main_1_2_3s_mIoU']:+.3f}; "
            f"checkpoint d={c['delta_vs_v18']['main_1_2_3s_mIoU']:+.3f}"
        )


def main():
    p = argparse.ArgumentParser()
    add_config_args(p)
    p.add_argument("--full-json", required=True)
    p.add_argument("--json-only", action="store_true")
    p.add_argument("--val-cache", default="")
    p.add_argument("--base-checkpoint", default="")
    p.add_argument("--novelty-checkpoint", default="")
    p.add_argument("--dataroot", default="")
    p.add_argument("--info-pkl", default="")
    p.add_argument("--output", default="")
    p.add_argument("--windows-per-scene", type=int, default=3)
    p.add_argument("--max-scenes", type=int, default=0)
    p.add_argument("--alignment-workers", type=int, default=6)
    p.add_argument(
        "--presence-thresholds",
        default="0.30,0.40,0.50,0.60,0.70,0.80,0.90",
    )
    p.add_argument(
        "--vertical-thresholds",
        default="0.30,0.40,0.50,0.60,0.70,0.80,0.90",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-amp", action="store_true")
    a = p.parse_args()

    triage = diagnose_full_json(a.full_json)
    _print_triage(triage)
    if bool(a.json_only):
        if a.output:
            Path(a.output).write_text(
                json.dumps(
                    {
                        "protocol": PROTOCOL,
                        "full_json_triage": triage,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"saved {a.output}")
        return

    result = run_stratified_diagnosis(a, triage)
    _print_stratified(result)
    op = Path(a.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"saved {op}")


if __name__ == "__main__":
    main()
