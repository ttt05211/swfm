#!/usr/bin/env python3
"""Evaluate source-selective KTA/V17 forecasting at fixed per-window budgets.

Policies:
  * oracle: future-GT single-source marginal utility (upper bound only)
  * selector: tiny causal MLP trained on train-scene oracle labels
  * speed: current causal source speed
  * random: repeated deterministic random source ranking

Q=0 is pure KTA; Q=100 is dense V17. All mixed predictions use the frozen A1
source-order CLEAR/WRITE compositor and the same Moving-mIoU-v2 evaluator.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from real_motion.metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS
from real_motion.metrics.occupancy_iou import OccupancyIoUMultiHorizon
from real_motion.motion_transport import FEATURE_DIM
from real_motion.msp_wm_cache import MSPWorldModelCacheDataset
from real_motion.nuscenes_adapter import NuScenesWindowSource
from real_motion.rigid_transport import (
    compose_component_replacements_in_input_order,
    rasterize_rigid_component,
)
from real_motion.runtime_config import add_config_args, load_runtime_config, make_prepare_config
from real_motion.selective_forecast import (
    BUDGET_CONTRACT,
    CorrectionSelector,
    SELECTIVE_LABEL_CACHE_VERSION,
    SELECTOR_INPUT_CONTRACT,
    SELECTOR_PROTOCOL,
    SPEED_FEATURE_INDEX,
    UTILITY_CONTRACT,
    stable_random_scores,
    top_budget_mask,
)
from real_motion.strong_w2det import StrongW2DetConfig, extract_instances, match_instances
from tools.real_motion import eval_p0_f9_frozen_sparse_occfm as safe
from tools.real_motion.eval_p0_f9_v17_local_stwm import (
    OCCUPANCY_IOU_CONTRACT,
    load_cache,
    t0_xy_to_world_preserve_source_z,
    window_from_record,
)

PROTOCOL = "p0_f9_v17_selective_forecast_eval_v1"
POLICIES = ("oracle", "selector", "speed")


def _parse_budgets(raw):
    vals = []
    for x in str(raw).split(","):
        x = x.strip()
        if not x:
            continue
        q = float(x)
        if q < 0 or q > 100:
            raise ValueError("budgets must be in [0,100]")
        vals.append(q)
    vals = sorted(set(vals))
    if not vals or vals[0] != 0.0 or vals[-1] != 100.0:
        raise ValueError("budgets must include 0 and 100")
    return vals


def _qname(q):
    return f"{float(q):g}"


def load_label_cache(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if obj.get("version") != SELECTIVE_LABEL_CACHE_VERSION:
        raise RuntimeError("selective label cache version mismatch")
    meta = obj.get("metadata") or {}
    if meta.get("utility_contract") != UTILITY_CONTRACT:
        raise RuntimeError("utility contract mismatch")
    if meta.get("selector_input_contract") != SELECTOR_INPUT_CONTRACT:
        raise RuntimeError("selector input contract mismatch")
    records = obj.get("records") or []
    if not records:
        raise RuntimeError("empty selective label cache")
    return meta, records


def load_selector(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("protocol") != SELECTOR_PROTOCOL:
        raise RuntimeError(f"selector protocol mismatch: {ck.get('protocol')}")
    if ck.get("selector_input_contract") != SELECTOR_INPUT_CONTRACT:
        raise RuntimeError("selector input contract mismatch")
    if ck.get("utility_contract") != UTILITY_CONTRACT:
        raise RuntimeError("selector utility contract mismatch")
    if int(ck.get("feature_dim", -1)) != FEATURE_DIM:
        raise RuntimeError("selector feature dimension mismatch")
    model = CorrectionSelector(
        feature_dim=FEATURE_DIM, hidden_dim=int(ck.get("hidden_dim", 64))
    ).to(device)
    model.load_state_dict(ck["state_dict"], strict=True)
    model.eval()
    norm = ck.get("normalization") or {}
    mean = torch.tensor(norm["feature_mean"], dtype=torch.float32, device=device)
    std = torch.tensor(norm["feature_std"], dtype=torch.float32, device=device)
    tmean = float(norm["target_mean"])
    tstd = float(norm["target_std"])
    return ck, model, mean, std, tmean, tstd


@torch.no_grad()
def selector_scores(model, features, mean, std, tmean, tstd, device):
    x = features.float().to(device)
    pred = model((x - mean) / std.clamp_min(1e-6))
    return (pred * tstd + tmean).float().cpu().numpy()


def _new_state():
    return {
        "semantic": safe._new_metrics(),
        "occupancy": OccupancyIoUMultiHorizon(),
    }


def _update_state(state, horizon, pred, gt, moving):
    safe._update(state["semantic"], float(horizon), pred, gt, moving)
    state["occupancy"].update(float(horizon), pred, gt)


def _report_state(state):
    r = safe._report(state["semantic"])
    r["occupancy"] = state["occupancy"].compute()
    return r


def _moving(report):
    return float(report["moving"]["mIoU"])


def _miou(report):
    return float(report["overall"]["mIoU"])


def _iou(report):
    return float(report["occupancy"]["IoU"])


def main():
    p = argparse.ArgumentParser()
    add_config_args(p)
    p.add_argument("--label-cache", required=True)
    p.add_argument("--v17-cache", required=True)
    p.add_argument("--p0f9-cache", required=True)
    p.add_argument("--selector-checkpoint", required=True)
    p.add_argument("--dataroot", required=True)
    p.add_argument("--info-pkl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--budgets", default="0,10,20,40,100")
    p.add_argument("--random-repeats", type=int, default=5)
    p.add_argument("--random-seed", type=int, default=20260913)
    p.add_argument("--max-windows", type=int, default=0)
    p.add_argument("--reference-v17-json", default="")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    budgets = _parse_budgets(a.budgets)
    if a.random_repeats <= 0:
        raise ValueError("random-repeats must be positive")

    cfg = load_runtime_config(a.config, a.override)
    pcfg = make_prepare_config(cfg)
    label_meta, label_records = load_label_cache(a.label_cache)
    cache_meta, v17_records = load_cache(a.v17_cache)

    label_by_id = {str(r["sample_id"]): r for r in label_records}
    v17_by_id = {str(r["sample_id"]): r for r in v17_records}
    sample_ids = [str(r["sample_id"]) for r in label_records]
    if int(a.max_windows) > 0:
        sample_ids = sample_ids[: min(len(sample_ids), int(a.max_windows))]
    missing = [sid for sid in sample_ids if sid not in v17_by_id]
    if missing:
        raise RuntimeError(f"V17 cache misses label samples: {missing[:5]}")

    expected_p0 = str(Path(a.p0f9_cache).resolve())
    if str(label_meta.get("p0f9_cache")) != expected_p0:
        raise RuntimeError(
            "label cache P0-F9 provenance differs from evaluation P0-F9 cache"
        )
    expected_v17 = str(Path(a.v17_cache).resolve())
    if str(label_meta.get("v17_cache")) != expected_v17:
        raise RuntimeError(
            "label cache V17 provenance differs from evaluation V17 cache"
        )

    ds = MSPWorldModelCacheDataset(a.p0f9_cache)
    sid_to_idx = {str(e["sample_id"]): i for i, e in enumerate(ds.entries)}
    missing = [sid for sid in sample_ids if sid not in sid_to_idx]
    if missing:
        raise RuntimeError(f"P0-F9 cache misses selected samples: {missing[:5]}")

    device = torch.device(
        a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    selector_ck, selector, fmean, fstd, tmean, tstd = load_selector(
        a.selector_checkpoint, device
    )
    selector_label_meta = selector_ck.get("label_cache_metadata") or {}
    if selector_label_meta.get("checkpoint") != label_meta.get("checkpoint"):
        raise RuntimeError(
            "selector was trained against a different frozen V17 expert checkpoint"
        )
    eval_scenes = {str(label_by_id[sid]["scene_name"]) for sid in sample_ids}
    selector_fit_scenes = set(selector_ck.get("train_scenes") or []) | set(
        selector_ck.get("val_scenes") or []
    )
    overlap = sorted(eval_scenes & selector_fit_scenes)
    if overlap:
        raise RuntimeError(
            "selector train/internal-val scenes overlap evaluation scenes: "
            f"{overlap[:10]}"
        )
    if not bool(ds.metadata.get("include_eval_payload", False)):
        raise RuntimeError(
            "selective evaluation requires P0-F9 validation cache with eval payload"
        )
    source = NuScenesWindowSource(a.dataroot, info_pkl=a.info_pkl, verbose=False)
    strong_cfg = StrongW2DetConfig(free_label=int(pcfg.free_label))

    states = {
        policy: {_qname(q): _new_state() for q in budgets}
        for policy in POLICIES
    }
    random_states = [
        {_qname(q): _new_state() for q in budgets}
        for _ in range(int(a.random_repeats))
    ]
    selected_counts = {
        policy: {_qname(q): 0 for q in budgets}
        for policy in POLICIES
    }
    random_selected_counts = [
        {_qname(q): 0 for q in budgets}
        for _ in range(int(a.random_repeats))
    ]
    source_total = 0

    for wi, sid in enumerate(sample_ids):
        rec = v17_by_id[sid]
        lab = label_by_id[sid]
        payload = safe._sample_payload(ds[sid_to_idx[sid]], torch.device("cpu"))
        w = window_from_record(rec)
        history_occ = [source.load_semantics(w.scene_name, tok) for tok in w.history_tokens]
        history_poses = [
            np.asarray(source.pose(tok), dtype=np.float64) for tok in w.history_tokens
        ]
        future_poses = [
            np.asarray(source.pose(tok), dtype=np.float64) for tok in w.future_tokens
        ]
        current = extract_instances(
            history_occ[-1], history_poses[-1], grid=pcfg.grid, cfg=strong_cfg
        )
        previous = extract_instances(
            history_occ[-2], history_poses[-2], grid=pcfg.grid, cfg=strong_cfg
        )
        velocities = match_instances(
            previous,
            current,
            float(pcfg.frame_dt_s),
            max_speed_mps=strong_cfg.max_match_speed_mps,
        )
        n = len(current)
        if n != int(rec["features"].shape[0]) or n != int(lab["features"].shape[0]):
            raise RuntimeError(f"{sid}: source count mismatch")
        if [int(c["class_id"]) for c in current] != [
            int(x) for x in rec["source_class_id"].tolist()
        ]:
            raise RuntimeError(f"{sid}: Strong source order/class mismatch")

        if not torch.equal(rec["features"].float(), lab["features"].float()):
            raise RuntimeError(f"{sid}: label-cache causal features differ from V17 cache")

        utilities = lab["utility_pp"].float().numpy()
        residual = lab["v17_residual_xy_m"].float().numpy()
        learned_scores = selector_scores(
            selector, rec["features"], fmean, fstd, tmean, tstd, device
        )
        speed_scores = rec["features"][:, SPEED_FEATURE_INDEX].float().numpy()

        masks = {
            "oracle": {
                _qname(q): top_budget_mask(utilities, q) for q in budgets
            },
            "selector": {
                _qname(q): top_budget_mask(learned_scores, q) for q in budgets
            },
            "speed": {
                _qname(q): top_budget_mask(speed_scores, q) for q in budgets
            },
        }
        random_masks = []
        for rr in range(int(a.random_repeats)):
            score = stable_random_scores(
                sid, n, int(a.random_seed) + 1009 * rr
            )
            random_masks.append({
                _qname(q): top_budget_mask(score, q) for q in budgets
            })

        for policy in POLICIES:
            for q in budgets:
                selected_counts[policy][_qname(q)] += int(
                    masks[policy][_qname(q)].sum()
                )
        for rr in range(int(a.random_repeats)):
            for q in budgets:
                random_selected_counts[rr][_qname(q)] += int(
                    random_masks[rr][_qname(q)].sum()
                )
        source_total += n

        t0_pose = history_poses[-1]
        for horizon, hi in safe.REPORT.items():
            gt = payload["gt"][hi]
            anchor = payload["anchor"][hi]
            moving = payload["moving"][hi]
            dt = (hi + 1) * float(pcfg.frame_dt_s)

            baseline = []
            replacement = []
            for i, comp in enumerate(current):
                src_center = np.asarray(comp["centroid_world"], dtype=np.float64)
                v = np.asarray(velocities.get(i, np.zeros(3)), dtype=np.float64)
                baseline.append(
                    rasterize_rigid_component(
                        comp["voxel_indices"],
                        int(comp["class_id"]),
                        t0_pose,
                        future_poses[hi],
                        source_center_world=src_center,
                        target_center_world=src_center + v * dt,
                        yaw_delta_rad=0.0,
                        grid=pcfg.grid,
                    )
                )
                xy_pred = rec["anchors_xy_t0_m"][i, hi].numpy() + residual[i, hi]
                center_pred = t0_xy_to_world_preserve_source_z(
                    xy_pred, src_center, t0_pose
                )
                replacement.append(
                    rasterize_rigid_component(
                        comp["voxel_indices"],
                        int(comp["class_id"]),
                        t0_pose,
                        future_poses[hi],
                        source_center_world=src_center,
                        target_center_world=center_pred,
                        yaw_delta_rad=0.0,
                        grid=pcfg.grid,
                    )
                )

            pred_cache = {}
            def compose(mask):
                key = np.packbits(np.asarray(mask, dtype=np.uint8)).tobytes()
                if key in pred_cache:
                    return pred_cache[key]
                ids = np.flatnonzero(mask)
                if len(ids) == 0:
                    pred = anchor
                else:
                    pred = compose_component_replacements_in_input_order(
                        anchor,
                        [baseline[int(i)] for i in ids],
                        [replacement[int(i)] for i in ids],
                        dynamic_class_ids=DYNAMIC_CLASS_IDS,
                        free_label=int(pcfg.free_label),
                        grid=pcfg.grid,
                    )
                pred_cache[key] = pred
                return pred

            for policy in POLICIES:
                for q in budgets:
                    pred = compose(masks[policy][_qname(q)])
                    _update_state(
                        states[policy][_qname(q)], horizon, pred, gt, moving
                    )
            for rr in range(int(a.random_repeats)):
                for q in budgets:
                    pred = compose(random_masks[rr][_qname(q)])
                    _update_state(
                        random_states[rr][_qname(q)], horizon, pred, gt, moving
                    )

        if wi == 0 or (wi + 1) % 8 == 0 or wi + 1 == len(sample_ids):
            print(
                f"selective_eval {wi+1}/{len(sample_ids)} sid={sid} sources={n}",
                flush=True,
            )

    reports = {
        policy: {
            _qname(q): _report_state(states[policy][_qname(q)])
            for q in budgets
        }
        for policy in POLICIES
    }
    random_repeat_reports = [
        {
            _qname(q): _report_state(random_states[rr][_qname(q)])
            for q in budgets
        }
        for rr in range(int(a.random_repeats))
    ]

    random_summary = {}
    for q in budgets:
        key = _qname(q)
        rows = [r[key] for r in random_repeat_reports]
        random_summary[key] = {
            "IoU_mean": float(np.mean([_iou(x) for x in rows])),
            "IoU_std": float(np.std([_iou(x) for x in rows])),
            "mIoU_mean": float(np.mean([_miou(x) for x in rows])),
            "mIoU_std": float(np.std([_miou(x) for x in rows])),
            "Moving_mean": float(np.mean([_moving(x) for x in rows])),
            "Moving_std": float(np.std([_moving(x) for x in rows])),
            "selected_fraction": float(
                np.mean([
                    random_selected_counts[rr][key] / max(source_total, 1)
                    for rr in range(int(a.random_repeats))
                ])
            ),
        }

    curve = {}
    for policy in POLICIES:
        curve[policy] = {}
        for q in budgets:
            key = _qname(q)
            r = reports[policy][key]
            curve[policy][key] = {
                "IoU": _iou(r),
                "mIoU": _miou(r),
                "Moving": _moving(r),
                "selected_sources": int(selected_counts[policy][key]),
                "selected_fraction": selected_counts[policy][key] / max(source_total, 1),
            }
    curve["random"] = random_summary

    kta = curve["oracle"][_qname(0.0)]["Moving"]
    dense = curve["oracle"][_qname(100.0)]["Moving"]
    analysis = {
        "KTA_Moving": kta,
        "dense_V17_Moving": dense,
        "dense_V17_gain_vs_KTA": dense - kta,
    }
    if 20.0 in budgets:
        q = _qname(20.0)
        oracle_gain = curve["oracle"][q]["Moving"] - kta
        selector_gain = curve["selector"][q]["Moving"] - kta
        analysis.update({
            "q20_oracle_gain_vs_KTA": oracle_gain,
            "q20_selector_gain_vs_KTA": selector_gain,
            "q20_selector_minus_speed": (
                curve["selector"][q]["Moving"] - curve["speed"][q]["Moving"]
            ),
            "q20_selector_minus_random_mean": (
                curve["selector"][q]["Moving"] - curve["random"][q]["Moving_mean"]
            ),
            "q20_selector_oracle_gain_retention": (
                selector_gain / oracle_gain if oracle_gain > 1e-12 else float("nan")
            ),
        })
    oracle_gains = [
        curve["oracle"][_qname(q)]["Moving"] - kta for q in budgets
    ]
    best_idx = int(np.argmax(np.asarray(oracle_gains)))
    analysis["best_oracle_budget_percent"] = float(budgets[best_idx])
    analysis["best_oracle_gain_vs_KTA"] = float(oracle_gains[best_idx])
    if 20.0 in budgets and analysis["best_oracle_gain_vs_KTA"] > 1e-12:
        analysis["q20_oracle_concentration_ratio"] = float(
            analysis["q20_oracle_gain_vs_KTA"]
            / analysis["best_oracle_gain_vs_KTA"]
        )

    reference_check = None
    if a.reference_v17_json:
        ref = json.loads(Path(a.reference_v17_json).read_text(encoding="utf-8"))
        branch = (ref.get("reports") or {}).get(
            "local_stwm_center_always_source_order"
        )
        if branch is None:
            raise RuntimeError("reference V17 JSON misses A1 report branch")
        strong = (ref.get("reports") or {}).get("strong_anchor")
        if strong is None:
            raise RuntimeError("reference V17 JSON misses Strong/KTA report branch")
        reference_check = {
            "reference_path": str(Path(a.reference_v17_json).resolve()),
            "reference_KTA_IoU": float(strong["occupancy"]["IoU"]),
            "reference_KTA_mIoU": float(strong["overall"]["mIoU"]),
            "reference_KTA_Moving": float(strong["moving"]["mIoU"]),
            "reference_V17_IoU": float(branch["occupancy"]["IoU"]),
            "reference_V17_mIoU": float(branch["overall"]["mIoU"]),
            "reference_V17_Moving": float(branch["moving"]["mIoU"]),
            "q0_IoU": curve["oracle"][_qname(0.0)]["IoU"],
            "q0_mIoU": curve["oracle"][_qname(0.0)]["mIoU"],
            "q0_Moving": curve["oracle"][_qname(0.0)]["Moving"],
            "q100_IoU": curve["oracle"][_qname(100.0)]["IoU"],
            "q100_mIoU": curve["oracle"][_qname(100.0)]["mIoU"],
            "q100_Moving": curve["oracle"][_qname(100.0)]["Moving"],
        }
        diffs = [
            abs(reference_check["reference_KTA_IoU"] - reference_check["q0_IoU"]),
            abs(reference_check["reference_KTA_mIoU"] - reference_check["q0_mIoU"]),
            abs(reference_check["reference_KTA_Moving"] - reference_check["q0_Moving"]),
            abs(reference_check["reference_V17_IoU"] - reference_check["q100_IoU"]),
            abs(reference_check["reference_V17_mIoU"] - reference_check["q100_mIoU"]),
            abs(reference_check["reference_V17_Moving"] - reference_check["q100_Moving"]),
        ]
        reference_check["max_abs_metric_diff"] = max(diffs)
        reference_check["pass_1e-6"] = bool(reference_check["max_abs_metric_diff"] <= 1e-6)
        if not reference_check["pass_1e-6"]:
            raise RuntimeError(
                "Q=0/100 do not reproduce frozen KTA/V17 reference metrics: "
                f"max diff={reference_check['max_abs_metric_diff']}"
            )

    result = {
        "protocol": PROTOCOL,
        "budget_contract": BUDGET_CONTRACT,
        "utility_contract": UTILITY_CONTRACT,
        "selector_input_contract": SELECTOR_INPUT_CONTRACT,
        "label_cache": str(Path(a.label_cache).resolve()),
        "v17_cache": str(Path(a.v17_cache).resolve()),
        "p0f9_cache": str(Path(a.p0f9_cache).resolve()),
        "selector_checkpoint": str(Path(a.selector_checkpoint).resolve()),
        "selector_train_label_cache": selector_ck.get("label_cache"),
        "selector_frozen_v17_checkpoint": selector_label_meta.get("checkpoint"),
        "eval_frozen_v17_checkpoint": label_meta.get("checkpoint"),
        "selector_fit_scenes": sorted(selector_fit_scenes),
        "evaluation_scenes": sorted(eval_scenes),
        "scene_overlap_checked_zero": True,
        "num_windows": len(sample_ids),
        "num_sources": int(source_total),
        "budgets_percent": budgets,
        "random_repeats": int(a.random_repeats),
        "random_seed": int(a.random_seed),
        "curve": curve,
        "full_reports": reports,
        "random_repeat_reports": random_repeat_reports,
        "analysis": analysis,
        "reference_v17_check": reference_check,
        "occupancy_iou_contract": OCCUPANCY_IOU_CONTRACT,
        "moving_metric_protocol": "interval_displacement_v2",
        "cache_metadata": cache_meta,
        "label_cache_metadata": label_meta,
    }

    print("\n=== SELECTIVE FORECAST BUDGET CURVE ===")
    print(
        f"{'policy':10s} {'Q%':>6s} {'IoU':>9s} {'mIoU':>9s} "
        f"{'Moving':>9s} {'selected':>10s}"
    )
    for policy in POLICIES:
        for q in budgets:
            row = curve[policy][_qname(q)]
            print(
                f"{policy:10s} {q:6.1f} {row['IoU']:9.4f} "
                f"{row['mIoU']:9.4f} {row['Moving']:9.4f} "
                f"{100*row['selected_fraction']:9.2f}%"
            )
    print("\n=== RANDOM ===")
    for q in budgets:
        row = curve["random"][_qname(q)]
        print(
            f"random     {q:6.1f} Moving={row['Moving_mean']:.4f}"
            f"+/-{row['Moving_std']:.4f} selected={100*row['selected_fraction']:.2f}%"
        )
    print("\n=== KEY ANALYSIS ===")
    print(json.dumps(analysis, indent=2))
    if reference_check is not None:
        print("\n=== DENSE V17 REFERENCE CHECK ===")
        print(json.dumps(reference_check, indent=2))

    op = Path(a.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"saved {op}")


if __name__ == "__main__":
    main()
