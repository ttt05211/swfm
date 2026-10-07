#!/usr/bin/env python3
"""Read-only Point CCR quality-gap diagnostic against epoch19 Local.

Answers one question before any further training:
  Is the remaining Local-vs-CCR gap mainly learnability/training budget, or is
  useful Local behavior outside the current CCR support/action contract?

The restricted oracle uses future GT ONLY to choose correct actions inside the
already-built CCR support and legality mask, then passes those actions through
the exact official CCR compositor. GT never changes support, source identity,
projection, ownership, fallback, or motion.

No checkpoint writes, training, cache expansion or promotion.  With
--full-action-diagnostics, a predefined DEV threshold grid is evaluated only
to diagnose ranking/calibration; those DEV optima are explicitly non-deployable.
"""
from __future__ import annotations

import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
from collections import defaultdict
from contextlib import nullcontext
import json
import math
import time

import numpy as np
import torch

from real_motion.canonical_causal_repair import repair_targets, compose_canonical
from real_motion.canonical_repair_execution import CanonicalCpuExecution
from real_motion.causal_geometry_cache import CausalGeometryCache
from real_motion.ccr_val_history_cache import (
    namespace as val_history_cache_namespace,
    validate_manifest as validate_val_history_cache_manifest,
)
from real_motion.column_runtime_pipeline import CachedColumnSource, prefetch_raw_columns
from real_motion.nuscenes_adapter import NuScenesWindowSource, gt_moving_support_sequence
from real_motion.runtime_config import add_config_args, load_runtime_config, make_prepare_config
from real_motion.source_evidence_audit import edit_quality
from real_motion.v21_source_induction import stable_json_fingerprint
from tools.real_motion import causal_column_common as columns
from tools.real_motion.ccr_screen_common import build_inputs, map_inputs, old_execution
from tools.real_motion.eval_p0_f9_v18_se2 import load_cache
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import (
    Metrics, align_records, delta, load_manifest, sha256,
)
from tools.real_motion.pilot_p0_f9_canonical_causal_repair import probabilities
from tools.real_motion.point_ccr_v18_fps_common import load_point_head
from tools.real_motion.run_p0_f9_shared_evidence_pilot import PilotProvider, require_cuda
from tools.real_motion.shared_evidence_pilot_common import GATES, moving_support_masks
from tools.real_motion.static_evidence_selector_common import CLEAN_SHA256, write_json, finite_json
from tools.real_motion.train_p0_f9_causal_columns import record_keys
from tools.real_motion.train_p0_f9_joint_causal_columns import load_joint
from real_motion.causal_column_completion import actions_from_probabilities, compose_dense


PROTOCOL = "p0_f9_ccr_quality_gap_restricted_oracle_v2"
REPORT = columns.REPORT
PROB_BINS = 2048
ADD_DIAG_THRESHOLDS = (0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70)
REMOVE_DIAG_THRESHOLDS = (0.10, 0.20, 0.40, 0.60, 0.80, 0.90, 0.95)
VARIANTS = (
    "current_joint",
    "current_static",
    "current_dynamic",
    "current_add_only",
    "current_remove_only",
    "restricted_oracle",
    "oracle_static",
    "oracle_dynamic",
    "oracle_add_only",
    "oracle_remove_only",
    "old_local",
)


def _safe_div(a, b):
    return None if not b else float(a) / float(b)


def _action_bucket():
    return dict(valid=0, target_pos=0, pred_pos=0, tp=0, fp=0, fn=0)


def _update_action(bucket, valid, target, pred):
    valid = np.asarray(valid, bool)
    target = np.asarray(target, bool) & valid
    pred = np.asarray(pred, bool) & valid
    bucket["valid"] += int(valid.sum())
    bucket["target_pos"] += int(target.sum())
    bucket["pred_pos"] += int(pred.sum())
    bucket["tp"] += int((target & pred).sum())
    bucket["fp"] += int((~target & pred & valid).sum())
    bucket["fn"] += int((target & ~pred).sum())


def _finish_action(bucket):
    out = dict(bucket)
    out["precision"] = _safe_div(out["tp"], out["tp"] + out["fp"])
    out["recall"] = _safe_div(out["tp"], out["tp"] + out["fn"])
    p, r = out["precision"], out["recall"]
    out["f1"] = None if p is None or r is None or p + r == 0 else 2 * p * r / (p + r)
    out["positive_rate"] = _safe_div(out["target_pos"], out["valid"])
    return out


def _ranking_bucket():
    return dict(
        pos=np.zeros(PROB_BINS, np.int64),
        neg=np.zeros(PROB_BINS, np.int64),
        score_sum=np.zeros(PROB_BINS, np.float64),
        score_sq_sum=0.0,
        positive_score_sum=0.0,
        count=0,
        positives=0,
    )


def _update_ranking(bucket, scores, target, mask):
    mask=np.asarray(mask,bool)
    if not np.any(mask):
        return
    score=np.asarray(scores,np.float32)[mask]
    y=np.asarray(target,bool)[mask]
    score=np.clip(score,0.0,1.0)
    ids=np.minimum((score*(PROB_BINS-1)).astype(np.int32),PROB_BINS-1)
    pos_ids=ids[y];neg_ids=ids[~y]
    bucket["pos"] += np.bincount(pos_ids,minlength=PROB_BINS)
    bucket["neg"] += np.bincount(neg_ids,minlength=PROB_BINS)
    bucket["score_sum"] += np.bincount(ids,weights=score,minlength=PROB_BINS)
    bucket["score_sq_sum"] += float(np.square(score.astype(np.float64,copy=False)).sum())
    bucket["positive_score_sum"] += float(score[y].sum(dtype=np.float64))
    bucket["count"] += int(len(score))
    bucket["positives"] += int(y.sum())


def _hist_quantile(hist, q):
    total=int(np.asarray(hist,np.int64).sum())
    if total<=0:
        return None
    rank=float(q)*max(total-1,0)
    idx=int(np.searchsorted(np.cumsum(hist),rank+1,side="left"))
    return float(idx/(PROB_BINS-1))


def _ranking_at_threshold(pos,neg,threshold):
    idx=min(PROB_BINS-1,max(0,int(np.ceil(float(threshold)*(PROB_BINS-1)))))
    tp=int(pos[idx:].sum());fp=int(neg[idx:].sum())
    positives=int(pos.sum());negatives=int(neg.sum())
    fn=positives-tp
    precision=_safe_div(tp,tp+fp);recall=_safe_div(tp,positives)
    f1=None if precision is None or recall is None or precision+recall==0 else 2*precision*recall/(precision+recall)
    return dict(
        threshold=float(threshold),tp=tp,fp=fp,fn=fn,
        precision=precision,recall=recall,f1=f1,
        predicted_positive_rate=_safe_div(tp+fp,positives+negatives),
    )


def _finish_ranking(bucket):
    pos=np.asarray(bucket["pos"],np.int64);neg=np.asarray(bucket["neg"],np.int64)
    positives=int(pos.sum());negatives=int(neg.sum());count=positives+negatives
    if count==0:
        return dict(count=0,positives=0)
    tp=np.cumsum(pos[::-1]);fp=np.cumsum(neg[::-1])
    precision=np.divide(tp,tp+fp,out=np.ones_like(tp,dtype=np.float64),where=(tp+fp)>0)
    recall=tp/max(positives,1)
    delta_recall=pos[::-1]/max(positives,1)
    average_precision=float(np.sum(precision*delta_recall)) if positives else None
    f1=np.divide(2*precision*recall,precision+recall,
                 out=np.zeros_like(precision),where=(precision+recall)>0)
    best_i=int(np.argmax(f1)) if len(f1) else 0
    best_threshold=float((PROB_BINS-1-best_i)/(PROB_BINS-1))
    neg_below=np.cumsum(neg)-neg
    auc=(float(np.sum(pos*(neg_below+0.5*neg)))/(positives*negatives)
         if positives and negatives else None)

    # ECE on 20 equal-width groups, using actual probability sums rather than
    # bin midpoints. This is diagnosis only; no threshold is promoted.
    edges=np.linspace(0,PROB_BINS,21,dtype=int)
    ece=0.0
    for lo,hi in zip(edges[:-1],edges[1:]):
        n=int(pos[lo:hi].sum()+neg[lo:hi].sum())
        if not n: continue
        conf=float(bucket["score_sum"][lo:hi].sum())/n
        acc=float(pos[lo:hi].sum())/n
        ece += n/count*abs(conf-acc)
    sum_p=float(bucket["score_sum"].sum())
    brier=(float(bucket["score_sq_sum"])-2*float(bucket["positive_score_sum"])+positives)/count
    pmean=(float(bucket["positive_score_sum"])/positives if positives else None)
    nsum=sum_p-float(bucket["positive_score_sum"])
    nmean=(nsum/negatives if negatives else None)
    threshold_rows={}
    for t in (0.02,0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.95):
        threshold_rows[f"{t:.2f}"]=_ranking_at_threshold(pos,neg,t)
    return dict(
        count=count,positives=positives,negatives=negatives,
        prevalence=_safe_div(positives,count),
        average_precision=average_precision,
        ap_lift_over_prevalence=(_safe_div(average_precision,_safe_div(positives,count))
                                 if average_precision is not None else None),
        auroc=auc,brier=float(brier),ece20=float(ece),
        mean_score_positive=pmean,mean_score_negative=nmean,
        positive_quantiles={str(q):_hist_quantile(pos,q) for q in (0.1,0.25,0.5,0.75,0.9)},
        negative_quantiles={str(q):_hist_quantile(neg,q) for q in (0.1,0.25,0.5,0.75,0.9)},
        best_f1=dict(
            threshold=best_threshold,
            f1=float(f1[best_i]) if len(f1) else None,
            precision=float(precision[best_i]) if len(precision) else None,
            recall=float(recall[best_i]) if len(recall) else None,
        ),
        fixed_thresholds=threshold_rows,
        approximation=dict(
            probability_bins=PROB_BINS,
            note="AP/AUROC/quantiles use a fixed streaming probability histogram; no raw validation probabilities are persisted.",
        ),
    )


def _new_threshold_metrics():
    return {
        "add_all": {f"{t:.2f}": Metrics() for t in ADD_DIAG_THRESHOLDS},
        "add_static": {f"{t:.2f}": Metrics() for t in ADD_DIAG_THRESHOLDS},
        "add_dynamic": {f"{t:.2f}": Metrics() for t in ADD_DIAG_THRESHOLDS},
        "remove_all": {f"{t:.2f}": Metrics() for t in REMOVE_DIAG_THRESHOLDS},
    }


def _threshold_summary(metrics):
    out={}
    for family,rows in metrics.items():
        out[family]={t:m.compute() for t,m in rows.items()}
    return out


def _best_threshold_rows(sweep):
    result={}
    for family,rows in sweep.items():
        if not rows: continue
        best_m=max(rows.items(),key=lambda kv:kv[1]["mIoU"])
        best_mov=max(rows.items(),key=lambda kv:kv[1]["MovingMicro"])
        result[family]=dict(
            best_mIoU_threshold=float(best_m[0]),
            best_mIoU=float(best_m[1]["mIoU"]),
            best_MovingMicro_threshold=float(best_mov[0]),
            best_MovingMicro=float(best_mov[1]["MovingMicro"]),
        )
    return result


def _checkpoint_curve(saved):
    reports = saved.get("reports", {})
    initial_old = (
        reports.get("initial_dev64", {})
        .get("variants", {})
        .get("old_joint", {})
        .get("metrics")
    )
    rows = []
    for row in reports.get("epochs", []):
        metrics = row.get("evaluation", {}).get("variants", {}).get("joint", {}).get("metrics")
        if not metrics:
            continue
        item = {
            "epoch": int(row.get("epoch", len(rows) + 1)),
            "update": int(row.get("update", 0)),
            "mIoU": float(metrics["mIoU"]),
            "MovingMicro": float(metrics["MovingMicro"]),
            "per_horizon": metrics["per_horizon"],
        }
        if initial_old:
            item["vs_old_mIoU_pp"] = item["mIoU"] - float(initial_old["mIoU"])
            item["vs_old_MovingMicro_pp"] = item["MovingMicro"] - float(initial_old["MovingMicro"])
        rows.append(item)
    return {
        "initial_old_dev64": initial_old,
        "epochs": rows,
        "last_epoch_delta": (
            None if len(rows) < 2 else {
                "mIoU_pp": rows[-1]["mIoU"] - rows[-2]["mIoU"],
                "MovingMicro_pp": rows[-1]["MovingMicro"] - rows[-2]["MovingMicro"],
            }
        ),
    }


def _role_masks(evidence):
    return {
        "static": np.asarray(evidence.actor) < 0,
        "dynamic": np.asarray(evidence.actor) >= 0,
    }


def _compose_variants(prep, evidence, plan, p, target):
    zeros = np.zeros_like(p[..., 0], dtype=np.float32)
    ones_add = target[..., 0].astype(np.float32)
    ones_remove = target[..., 1].astype(np.float32)
    return {
        "current_joint": compose_canonical(
            prep.baseline, evidence, plan, p[..., 0], p[..., 1], thresholds=(.5, .95), role="all"
        ),
        "current_static": compose_canonical(
            prep.baseline, evidence, plan, p[..., 0], p[..., 1], thresholds=(.5, .95), role="static"
        ),
        "current_dynamic": compose_canonical(
            prep.baseline, evidence, plan, p[..., 0], p[..., 1], thresholds=(.5, .95), role="dynamic"
        ),
        "current_add_only": compose_canonical(
            prep.baseline, evidence, plan, p[..., 0], zeros, thresholds=(.5, .95), role="all"
        ),
        "current_remove_only": compose_canonical(
            prep.baseline, evidence, plan, zeros, p[..., 1], thresholds=(.5, .95), role="all"
        ),
        "restricted_oracle": compose_canonical(
            prep.baseline, evidence, plan, ones_add, ones_remove, thresholds=(.5, .95), role="all"
        ),
        "oracle_static": compose_canonical(
            prep.baseline, evidence, plan, ones_add, ones_remove, thresholds=(.5, .95), role="static"
        ),
        "oracle_dynamic": compose_canonical(
            prep.baseline, evidence, plan, ones_add, ones_remove, thresholds=(.5, .95), role="dynamic"
        ),
        "oracle_add_only": compose_canonical(
            prep.baseline, evidence, plan, ones_add, zeros, thresholds=(.5, .95), role="all"
        ),
        "oracle_remove_only": compose_canonical(
            prep.baseline, evidence, plan, zeros, ones_remove, thresholds=(.5, .95), role="all"
        ),
    }


def _old_helpful_coverage(before, old_dense, gt, evidence, plan, target, h):
    """How many Local fixes are exactly expressible by one legal CCR action."""
    before = np.asarray(before)
    old_dense = np.asarray(old_dense)
    gt = np.asarray(gt)
    helpful = (old_dense == gt) & (before != gt)
    harmful = (old_dense != gt) & (before == gt) & (old_dense != before)
    changed = old_dense != before

    positives = target[:, h, 0] | target[:, h, 1]
    flats = np.asarray(plan.flat[:, h], np.int64)
    recoverable = np.unique(flats[positives & (flats >= 0)])
    helpful_flat = np.flatnonzero(helpful.ravel())
    reachable = np.isin(helpful_flat, recoverable, assume_unique=False)

    rows = {
        "old_changed": int(changed.sum()),
        "old_helpful": int(helpful.sum()),
        "old_harmful": int(harmful.sum()),
        "old_helpful_reachable_by_exact_CCR_action": int(reachable.sum()),
        "old_helpful_unreachable": int(len(helpful_flat) - reachable.sum()),
    }
    rows["reachable_fraction"] = _safe_div(
        rows["old_helpful_reachable_by_exact_CCR_action"], rows["old_helpful"]
    )

    # Useful Local edits by GT semantic class, with current CCR reachability.
    per_class = {}
    flat_gt = gt.ravel()
    for cid in np.unique(flat_gt[helpful_flat]) if len(helpful_flat) else ():
        ids = helpful_flat[flat_gt[helpful_flat] == cid]
        ok = np.isin(ids, recoverable, assume_unique=False)
        per_class[str(int(cid))] = {
            "old_helpful": int(len(ids)),
            "reachable": int(ok.sum()),
            "unreachable": int(len(ids) - ok.sum()),
            "reachable_fraction": _safe_div(int(ok.sum()), int(len(ids))),
        }
    rows["per_gt_class"] = per_class
    return rows


def _merge_coverage(total, row):
    for key in (
        "old_changed", "old_helpful", "old_harmful",
        "old_helpful_reachable_by_exact_CCR_action", "old_helpful_unreachable",
    ):
        total[key] += int(row[key])
    for cid, values in row["per_gt_class"].items():
        dst = total["per_gt_class"][cid]
        for key in ("old_helpful", "reachable", "unreachable"):
            dst[key] += int(values[key])


def _finish_coverage(total):
    out = {k: int(v) for k, v in total.items() if k != "per_gt_class"}
    out["reachable_fraction"] = _safe_div(
        out["old_helpful_reachable_by_exact_CCR_action"], out["old_helpful"]
    )
    out["per_gt_class"] = {}
    for cid, values in sorted(total["per_gt_class"].items(), key=lambda x: int(x[0])):
        row = dict(values)
        row["reachable_fraction"] = _safe_div(row["reachable"], row["old_helpful"])
        out["per_gt_class"][cid] = row
    return out


def _top_class_gaps(metrics):
    rows = []
    current = metrics["current_joint"]
    old = metrics["old_local"]
    oracle = metrics["restricted_oracle"]
    for h in ("1.0", "2.0", "3.0"):
        c = current["per_horizon"][h]["semantic_per_class"]
        o = old["per_horizon"][h]["semantic_per_class"]
        q = oracle["per_horizon"][h]["semantic_per_class"]
        for cid in c:
            if all(np.isfinite(x[cid]) for x in (c, o, q)):
                rows.append({
                    "horizon_s": h,
                    "class_id": int(cid),
                    "old_minus_current_pp": float(o[cid] - c[cid]),
                    "oracle_minus_current_pp": float(q[cid] - c[cid]),
                    "oracle_minus_old_pp": float(q[cid] - o[cid]),
                    "current_iou": float(c[cid]),
                    "old_iou": float(o[cid]),
                    "oracle_iou": float(q[cid]),
                })
    rows.sort(key=lambda x: x["old_minus_current_pp"], reverse=True)
    return rows[:20]


def _decision(metrics):
    current = metrics["current_joint"]
    old = metrics["old_local"]
    oracle = metrics["restricted_oracle"]
    tol = .20
    oracle_gate = {
        "mIoU_within_0_20pp_of_old": oracle["mIoU"] >= old["mIoU"] - tol,
        "MovingMicro_within_0_20pp_of_old": oracle["MovingMicro"] >= old["MovingMicro"] - tol,
        "all_horizons_MovingMicro_within_0_20pp_of_old": all(
            oracle["per_horizon"][h]["MovingMicro"] >= old["per_horizon"][h]["MovingMicro"] - tol
            for h in ("1.0", "2.0", "3.0")
        ),
    }
    oracle_gate["pass"] = all(oracle_gate.values())
    return {
        "current_vs_old_pp": delta(current, old),
        "oracle_vs_current_pp": delta(oracle, current),
        "oracle_vs_old_pp": delta(oracle, old),
        "restricted_oracle_gate": oracle_gate,
        "route": (
            "SUPPORT_SUFFICIENT__test_training_budget_full_TRAIN20430_x3_warm_start"
            if oracle_gate["pass"]
            else "SUPPORT_OR_COMPOSITOR_LIMIT__do_not_blindly_add_epochs; change one support/context axis first"
        ),
        "note": (
            "Restricted oracle is diagnostic only: GT chooses actions inside the current legal CCR domain; "
            "it is not an arbitrary-scene GT upper bound."
        ),
    }


def _summary(result):
    """Compact terminal summary; full diagnostics stay in quality_gap.json."""
    lines = ["===== CCR QUALITY GAP =====", "status=" + result["status"]]
    m = result.get("metrics")
    if m:
        for name, label in (
            ("current_joint", "CURRENT"),
            ("old_local", "OLD_LOCAL"),
            ("restricted_oracle", "ORACLE"),
        ):
            x = m[name]
            lines.append(
                f'{label:9s} mIoU={x["mIoU"]:.4f}  MovingMicro={x["MovingMicro"]:.4f}'
            )
    d = result.get("decision")
    if d:
        cv = d["current_vs_old_pp"]
        ov = d["oracle_vs_old_pp"]
        gate = d["restricted_oracle_gate"]
        lines.append(
            f'GAP current-old: mIoU={cv["mIoU"]:+.4f}  Moving={cv["MovingMicro"]:+.4f}'
        )
        lines.append(
            f'ORACLE-old:      mIoU={ov["mIoU"]:+.4f}  Moving={ov["MovingMicro"]:+.4f}'
        )
        lines.append("ORACLE_GATE=" + ("PASS" if gate["pass"] else "FAIL"))
    if result.get("old_helpful_coverage"):
        c = result["old_helpful_coverage"]["all_report_horizons"]
        frac = c["reachable_fraction"]
        text = "n/a" if frac is None else f"{100*frac:.1f}%"
        lines.append(
            f'Local helpful edits reachable by CCR: '
            f'{c["old_helpful_reachable_by_exact_CCR_action"]}/{c["old_helpful"]} ({text})'
        )
    curve = result.get("training_curve", {}).get("epochs", [])
    if curve:
        compact = "  ".join(
            f'E{x["epoch"]}:mIoU={x["mIoU"]:.3f}/Mov={x["MovingMicro"]:.3f}'
            for x in curve
        )
        lines.append("DEV64 TRAIN CURVE  " + compact)
        last = result["training_curve"].get("last_epoch_delta")
        if last:
            lines.append(
                f'LAST EPOCH DELTA: mIoU={last["mIoU_pp"]:+.4f}  '
                f'Moving={last["MovingMicro_pp"]:+.4f}'
            )
    action = result.get("action_learning", {})
    if action:
        parts = []
        for key, label in (
            ("static/ADD/all", "sADD"),
            ("static/REMOVE/all", "sREM"),
            ("dynamic/ADD/all", "dADD"),
            ("dynamic/REMOVE/all", "dREM"),
        ):
            row = action.get(key)
            if row:
                r = row.get("recall")
                parts.append(f'{label}R=' + ("n/a" if r is None else f"{100*r:.1f}%"))
        if parts:
            lines.append("ACTION RECALL  " + "  ".join(parts))
    ranking=result.get("action_ranking_calibration",{})
    if ranking:
        for key,label in (("static/ADD/all","sADD"),("dynamic/ADD/all","dADD"),
                          ("static/REMOVE/all","sREM"),("dynamic/REMOVE/all","dREM")):
            row=ranking.get(key,{})
            if row:
                lines.append(
                    f'{label} RANK AP={row.get("average_precision"):.4f} '
                    f'base={row.get("prevalence"):.4f} '
                    f'lift={row.get("ap_lift_over_prevalence"):.2f} '
                    f'AUROC={row.get("auroc"):.4f} '
                    f'bestF1@{row.get("best_f1",{}).get("threshold"):.3f}'
                )
    best=result.get("diagnostic_threshold_best",{})
    if best:
        for family in ("add_all","add_static","add_dynamic","remove_all"):
            row=best.get(family)
            if row:
                lines.append(
                    f'{family} DENSE best-mIoU@{row["best_mIoU_threshold"]:.2f}={row["best_mIoU"]:.4f} '
                    f'best-Moving@{row["best_MovingMicro_threshold"]:.2f}={row["best_MovingMicro"]:.4f}'
                )
    if d:
        lines.append("ROUTE=" + d["route"])
    if "error" in result:
        lines.append("error=" + result["error"])
    lines.append("full details: quality_gap.json")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    for key in (
        "checkpoint", "ccr-checkpoint", "base-checkpoint", "dev-cache",
        "population-manifest", "dataroot", "dev-info", "out-dir",
    ):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--windows", type=int, default=64, choices=(64, 512))
    parser.add_argument("--cpu-workers", type=int, default=8)
    parser.add_argument("--ccr-cpu-execution", choices=("numpy", "native", "native_parallel"), default="native_parallel")
    parser.add_argument("--ccr-cpu-workers", type=int, default=4)
    parser.add_argument("--ccr-val-history-cache",
                        help="persistent VAL fixed-history geometry cache; quality evaluation only")
    parser.add_argument("--ccr-val-history-cache-ram-mib", type=int, default=512)
    parser.add_argument(
        "--full-action-diagnostics", action="store_true",
        help="one-pass read-only probability ranking/calibration + fixed-grid dense threshold diagnostics; never promotes thresholds",
    )
    args = parser.parse_args(argv)

    out = Path(args.out_dir)
    if out.exists():
        parser.error("fresh output required")
    for key in (
        "config", "checkpoint", "ccr_checkpoint", "base_checkpoint",
        "dev_cache", "population_manifest", "dev_info",
    ):
        if not Path(getattr(args, key) or "").is_file():
            parser.error("missing " + key)
    if (not Path(args.dataroot).is_dir() or not 1 <= args.cpu_workers <= 16
            or not 1 <= args.ccr_cpu_workers <= 8
            or not 0 <= args.ccr_val_history_cache_ram_mib <= 16384):
        parser.error("invalid paths/workers")

    device = require_cuda(args.device)
    torch.set_num_threads(1)
    out.mkdir(parents=True)
    started = time.perf_counter()
    result = {"status": "running", "protocol": PROTOCOL}
    def persist():
        write_json(out / "quality_gap.json", result)
        (out / "summary.txt").write_text(_summary(result), encoding="utf-8")
    persist()

    execution = None
    val_history_cache = None
    try:
        cfg = load_runtime_config(args.config, args.override)
        config_fp = stable_json_fingerprint(cfg)
        epoch19_sha = sha256(args.checkpoint)
        if sha256(args.base_checkpoint) != CLEAN_SHA256:
            raise RuntimeError("Clean-E14 base checkpoint fingerprint mismatch")
        ck, teacher = load_joint(
            args.checkpoint, device, reference_sha=CLEAN_SHA256,
            config_sha=config_fp, allow_diagnostic=True,
        )
        if (
            teacher.transport.config.history_frames != 4
            or ck.get("cursor_epoch") != 19
            or ck["model_configs"].get("adaptive_context") is not None
        ):
            raise RuntimeError("selected four-history epoch19 Local checkpoint required")
        teacher.eval().requires_grad_(False)
        for path, expected in (
            (args.dev_cache, ck["cache_fingerprints"]["dev"]),
            (args.dev_info, ck["info_fingerprints"]["dev"]),
        ):
            if sha256(path) != expected:
                raise RuntimeError("epoch19/data provenance mismatch: " + path)

        saved = torch.load(args.ccr_checkpoint, map_location="cpu", weights_only=False)
        head = load_point_head(
            saved,
            teacher_sha256=epoch19_sha,
            config_fingerprint=config_fp,
            source_dim=teacher.columns.source_dim,
            device=device,
            allow_completed_epoch_boundary=True,
        )
        head.eval().requires_grad_(False)
        result["training_curve"] = _checkpoint_curve(saved)

        manifest, keys64, _ = load_manifest(args.population_manifest)
        parent = tuple(map(tuple, manifest["parent_keys"]))
        if (
            len(keys64) != 64 or len(parent) != 512
            or manifest["manifest_fingerprint"] != ck["dev_manifest_fingerprint"]
            or parent != tuple(map(tuple, ck["dev_keys"]))
            or saved["contract"]["dev_manifest_fingerprint"] != manifest["manifest_fingerprint"]
        ):
            raise RuntimeError("frozen dev manifest mismatch")
        keys = keys64 if args.windows == 64 else parent
        _, all_dev = load_cache(args.dev_cache)
        record_keys(all_dev)
        records = align_records(all_dev, keys)
        del all_dev

        provider = PilotProvider(
            args.base_checkpoint, CLEAN_SHA256, make_prepare_config(cfg),
            device, args.cpu_workers, teacher, None,
        )
        execution = CanonicalCpuExecution(args.ccr_cpu_execution, args.ccr_cpu_workers)
        provider.ccr_execution = execution
        source = CachedColumnSource(
            NuScenesWindowSource(args.dataroot, info_pkl=args.dev_info, verbose=False), 256
        )
        if args.ccr_val_history_cache:
            root=Path(__file__).resolve().parents[2]
            namespace_input=val_history_cache_namespace(provider,args,root)
            val_history_cache=CausalGeometryCache(
                args.ccr_val_history_cache,namespace_input,max_bytes=0,
                ram_bytes=int(args.ccr_val_history_cache_ram_mib)*2**20,
                reserve_bytes=0,compression_level=6)
            manifest_cache=validate_val_history_cache_manifest(val_history_cache,args)
            provider.ccr_history_cache=val_history_cache
            provider.ccr_history_cache_mode='require'
            provider.ccr_history_cache_source=source
            result['val_history_cache']=dict(
                mode='require',root=str(Path(args.ccr_val_history_cache).resolve()),
                namespace=val_history_cache.namespace,
                ram_mib=args.ccr_val_history_cache_ram_mib,
                manifest_disk_gib=manifest_cache.get('disk_gib'))
            print('CCR_VAL_HISTORY_CACHE '+json.dumps(result['val_history_cache'],sort_keys=True),flush=True)

        base = Metrics()
        metric_obj = {name: Metrics() for name in VARIANTS}
        quality = {name: defaultdict(int) for name in VARIANTS}
        scenes = defaultdict(lambda: {name: Metrics() for name in ("baseline", *VARIANTS)})

        action = defaultdict(_action_bucket)
        ranking = defaultdict(_ranking_bucket) if args.full_action_diagnostics else None
        threshold_metrics = _new_threshold_metrics() if args.full_action_diagnostics else None
        threshold_parity_checked = False
        coverage_all = {
            "old_changed": 0,
            "old_helpful": 0,
            "old_harmful": 0,
            "old_helpful_reachable_by_exact_CCR_action": 0,
            "old_helpful_unreachable": 0,
            "per_gt_class": defaultdict(lambda: {"old_helpful": 0, "reachable": 0, "unreachable": 0}),
        }
        coverage_h = {
            str(.5 * (h + 1)): {
                "old_changed": 0,
                "old_helpful": 0,
                "old_harmful": 0,
                "old_helpful_reachable_by_exact_CCR_action": 0,
                "old_helpful_unreachable": 0,
                "per_gt_class": defaultdict(lambda: {"old_helpful": 0, "reachable": 0, "unreachable": 0}),
            }
            for h in REPORT
        }
        duplicate_positive_destinations = 0
        positive_proposals = 0

        with old_execution(teacher, provider):
            for wi, (record, raw) in enumerate(prefetch_raw_columns(provider, source, records), 1):
                output = teacher.motion(record, device)
                cached_causal=raw.get('_column_causal_preparation')
                if cached_causal is not None and not getattr(provider,'columns_checked',False):
                    # One live preflight preserves the frozen V18 exactness gate;
                    # all following windows use the verified persistent VAL cache.
                    del raw['_column_causal_preparation']
                    try:
                        prep=provider.prepare_columns(
                            source,record,include_gt=True,raw_window=raw,outputs=output)
                    finally:
                        raw['_column_causal_preparation']=cached_causal
                    print('CCR_VAL_CACHE_LIVE_EXACTNESS_PREFLIGHT PASS',flush=True)
                else:
                    prep = provider.prepare_columns(
                        source, record, include_gt=True, raw_window=raw, outputs=output
                    )
                evidence = build_inputs(provider, prep)
                plan = map_inputs(provider, evidence, prep)
                p = probabilities(head, evidence, plan, output, device)
                target, valid = repair_targets(evidence, plan, raw["future_gt_occ"])
                predictions = _compose_variants(prep, evidence, plan, p, target)

                # Old Local only needs report horizons; preserve its official gates.
                old_by_h = {}
                for h in REPORT:
                    old_plan = columns.candidate_plan(prep, h, provider.pcfg.grid, teacher.columns.config)
                    old_p = columns.predict_probabilities(
                        teacher.columns, prep, h, old_plan, provider.pcfg.grid, device, 256
                    )
                    old_by_h[h] = compose_dense(
                        prep.baseline[h], old_plan,
                        actions_from_probabilities(old_plan, old_p, GATES),
                    )

                support = gt_moving_support_sequence(
                    source.nusc, prep.window.t0_token, prep.window.future_tokens,
                    tuple(.5 * (h + 1) for h in range(6)),
                    grid=provider.pcfg.grid, workers=provider.workers,
                )
                moving = moving_support_masks(support, provider.pcfg.grid.shape_hwd)

                # Learned-action quality over the complete legal CCR domain.
                roles = _role_masks(evidence)
                pred_actions = np.stack(
                    (p[..., 0] >= .5, p[..., 1] >= .95), axis=-1
                ) & plan.legal
                for role_name, role_mask in roles.items():
                    for action_id, action_name in enumerate(("ADD", "REMOVE")):
                        for h in range(6):
                            mask = role_mask & valid[:, h, action_id]
                            key = f"{role_name}/{action_name}/h{h+1}"
                            _update_action(
                                action[key],
                                mask,
                                target[:, h, action_id],
                                pred_actions[:, h, action_id],
                            )
                            if ranking is not None:
                                _update_ranking(
                                    ranking[key],p[:,h,action_id],target[:,h,action_id],mask)
                        # Aggregate across six horizons without changing priors.
                        role6 = np.broadcast_to(role_mask[:, None], target[..., action_id].shape)
                        key = f"{role_name}/{action_name}/all"
                        aggregate_mask=role6 & valid[..., action_id]
                        _update_action(
                            action[key],
                            aggregate_mask,
                            target[..., action_id],
                            pred_actions[..., action_id],
                        )
                        if ranking is not None:
                            _update_ranking(
                                ranking[key],p[...,action_id],target[...,action_id],aggregate_mask)
                        for cid in np.unique(evidence.classes[role_mask]):
                            class_mask = role_mask & (evidence.classes == cid)
                            class6 = np.broadcast_to(class_mask[:, None], target[..., action_id].shape)
                            key = f"{role_name}/{action_name}/class_{int(cid)}"
                            class_valid=class6 & valid[..., action_id]
                            _update_action(
                                action[key],
                                class_valid,
                                target[..., action_id],
                                pred_actions[..., action_id],
                            )
                            if ranking is not None and action_id==0:
                                _update_ranking(
                                    ranking[key],p[...,action_id],target[...,action_id],class_valid)

                # Duplicate GT-positive proposals are reported because the
                # restricted oracle remains subject to the official compositor.
                for h in range(6):
                    positive = target[:, h, 0] | target[:, h, 1]
                    flats = plan.flat[positive, h]
                    flats = flats[flats >= 0]
                    positive_proposals += int(len(flats))
                    if len(flats):
                        _, count = np.unique(flats, return_counts=True)
                        duplicate_positive_destinations += int((count > 1).sum())

                if threshold_metrics is not None:
                    zeros=np.zeros_like(p[...,0],dtype=np.float32)
                    # Diagnostic-only fixed threshold axes. Convert decisions to
                    # binary masks and pass them through the OFFICIAL compositor
                    # at its safe 0.5 gate; this does not weaken deployment
                    # threshold guards or alter composition semantics.
                    for t in ADD_DIAG_THRESHOLDS:
                        add_mask=(p[...,0]>=t).astype(np.float32)
                        for family,role in (("add_all","all"),("add_static","static"),("add_dynamic","dynamic")):
                            dense=compose_canonical(
                                prep.baseline,evidence,plan,add_mask,zeros,
                                thresholds=(.5,None),role=role)
                            for ri,h in enumerate(REPORT):
                                threshold_metrics[family][f"{t:.2f}"].update(
                                    ri,dense[h],raw["future_gt_occ"][h],moving[h])
                        if (not threshold_parity_checked) and abs(t-.5)<1e-12:
                            diagnostic_default=compose_canonical(
                                prep.baseline,evidence,plan,add_mask,zeros,
                                thresholds=(.5,None),role="all")
                            if any(not np.array_equal(a,b) for a,b in
                                   zip(diagnostic_default,predictions["current_add_only"])):
                                raise RuntimeError("diagnostic ADD mask compositor parity failed")
                            threshold_parity_checked=True
                    for t in REMOVE_DIAG_THRESHOLDS:
                        remove_mask=(p[...,1]>=t).astype(np.float32)
                        dense=compose_canonical(
                            prep.baseline,evidence,plan,zeros,remove_mask,
                            thresholds=(None,.5),role="all")
                        for ri,h in enumerate(REPORT):
                            threshold_metrics["remove_all"][f"{t:.2f}"].update(
                                ri,dense[h],raw["future_gt_occ"][h],moving[h])

                scene = scenes[str(record["scene_name"])]
                for ri, h in enumerate(REPORT):
                    gt = raw["future_gt_occ"][h]
                    before = prep.baseline[h]
                    base.update(ri, before, gt, moving[h])
                    scene["baseline"].update(ri, before, gt, moving[h])

                    pred_h = {name: predictions[name][h] for name in predictions}
                    pred_h["old_local"] = old_by_h[h]
                    for name in VARIANTS:
                        dense = pred_h[name]
                        metric_obj[name].update(ri, dense, gt, moving[h])
                        scene[name].update(ri, dense, gt, moving[h])
                        for key, value in edit_quality(before, dense, gt).items():
                            quality[name][key] += value

                    cov = _old_helpful_coverage(
                        before, old_by_h[h], gt, evidence, plan, target, h
                    )
                    _merge_coverage(coverage_all, cov)
                    _merge_coverage(coverage_h[str(.5 * (h + 1))], cov)

                if wi == 1 or wi % 16 == 0 or wi == len(records):
                    print(f"CCR_GAP {wi}/{len(records)}", flush=True)

        baseline_metrics = base.compute()
        metrics = {"baseline": baseline_metrics}
        for name in VARIANTS:
            metrics[name] = metric_obj[name].compute()
        result["metrics"] = metrics
        result["delta_vs_transport_pp"] = {
            name: delta(metrics[name], baseline_metrics) for name in VARIANTS
        }
        result["decision"] = _decision(metrics)
        result["top_semantic_gaps"] = _top_class_gaps(metrics)
        result["action_learning"] = {
            key: _finish_action(value) for key, value in sorted(action.items())
        }
        if ranking is not None:
            result["action_ranking_calibration"] = {
                key:_finish_ranking(value) for key,value in sorted(ranking.items())
            }
            result["diagnostic_threshold_sweep"]=_threshold_summary(threshold_metrics)
            result["diagnostic_threshold_best"]=_best_threshold_rows(
                result["diagnostic_threshold_sweep"])
            result["diagnostic_threshold_contract"]=dict(
                add_thresholds=list(ADD_DIAG_THRESHOLDS),
                remove_thresholds=list(REMOVE_DIAG_THRESHOLDS),
                axes=(
                    "ADD-only all/static/dynamic role sweeps and REMOVE-only all-role sweep; "
                    "one axis at a time, no 2-D dev tuning"
                ),
                compositor="official compose_canonical fed binary action masks",
                promotion_allowed=False,
                note=(
                    "DEV diagnostic only. Any selected threshold must be frozen before an "
                    "independent evaluation; these DEV512 optima are not deployable/reportable tuned results."
                ),
            )
        result["old_helpful_coverage"] = {
            "all_report_horizons": _finish_coverage(coverage_all),
            "per_horizon": {
                h: _finish_coverage(value) for h, value in coverage_h.items()
            },
        }
        result["oracle_conflicts"] = {
            "positive_proposals": positive_proposals,
            "destinations_with_multiple_positive_proposals": duplicate_positive_destinations,
            "note": "duplicates are not removed; restricted oracle uses the official compositor exactly",
        }
        result["quality"] = {}
        for name in VARIANTS:
            q = dict(quality[name])
            added, removed = q.get("added", 0), q.get("removed", 0)
            q["addition_semantic_precision"] = _safe_div(q.get("added_semantic_tp", 0), added)
            q["removal_false_occupancy_fraction"] = _safe_div(q.get("removed_false_occupancy", 0), removed)
            result["quality"][name] = q

        result["population"] = {
            "windows": len(records),
            "mode": "dev64" if args.windows == 64 else "dev512",
            "manifest_fingerprint": manifest["manifest_fingerprint"],
            "key_fingerprint": stable_json_fingerprint([list(x) for x in keys]),
        }
        result.update(
            status="complete",
            elapsed_seconds=time.perf_counter() - started,
            read_only=True,
            future_GT_model_input=False,
            threshold_search=False,
            diagnostic_threshold_sweep=bool(args.full_action_diagnostics),
            threshold_selection_for_deployment=False,
            route=result["decision"]["route"],
        )
        persist()
        print(_summary(result), flush=True)
        return 0
    except BaseException as exc:
        result.update(
            status="failed",
            error=type(exc).__name__ + ": " + str(exc),
            elapsed_seconds=time.perf_counter() - started,
        )
        persist()
        raise
    finally:
        if val_history_cache is not None:
            result['val_history_cache_stats']=val_history_cache.stats()
            val_history_cache.close()
            persist()
        if execution is not None:
            execution.close()


if __name__ == "__main__":
    sys.exit(main())
