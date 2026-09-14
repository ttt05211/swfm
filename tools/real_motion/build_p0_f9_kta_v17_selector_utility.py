#!/usr/bin/env python3
"""Build causal KTA/V17 selector labels and optional GT-assisted budget curves.

This is a feasibility experiment, not the final method. For each Strong source
we compare two frozen experts under the final A1 source-order compositor:
  * KTA rigid transport;
  * V17-RL epoch5 rigid transport.

Future GT is used only to assign an offline source utility label: the change in
correctly classified semantic voxels on GT Moving support when exactly this
source is switched from KTA to V17 while all other sources remain KTA. Selector
inputs stored in the output are causal and contain neither V17 output nor future
GT.

When --evaluate-curves is set, the script also composes exact final occupancy
predictions for per-window Q budgets using GT utility ranking, current-speed
ranking and deterministic random ranking. This answers whether correction
benefit is sparse before any selector is trained.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from real_motion.kta_v17_selector import (
    FEATURE_CONTRACT,
    SELECTOR_CACHE_VERSION,
    SELECTOR_FEATURE_DIM,
    SELECTOR_FEATURE_NAMES,
    UTILITY_CONTRACT,
    random_fraction_mask,
    selector_features,
    speed_score,
    summarize_utility,
    top_fraction_mask,
)
from real_motion.metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS
from real_motion.metrics.occupancy_iou import OccupancyIoUMultiHorizon
from real_motion.msp_wm_cache import MSPWorldModelCacheDataset
from real_motion.nuscenes_adapter import NuScenesWindowSource
from real_motion.rigid_transport import (
    compose_component_replacements_in_input_order,
    rasterize_rigid_component,
)
from real_motion.runtime_config import add_config_args, load_runtime_config, make_prepare_config
from real_motion.strong_w2det import (
    StrongW2DetConfig,
    extract_instances,
    match_instances,
    strong_w2det_sequence,
)
from tools.real_motion import build_p0_f5_cache_direct as wm_base
from tools.real_motion import eval_p0_f9_frozen_sparse_occfm as safe
from tools.real_motion.eval_p0_f9_v17_local_stwm import (
    load_cache,
    load_model,
    t0_xy_to_world_preserve_source_z,
    window_from_record,
)

PROTOCOL = "p0_f9_kta_v17_selector_utility_builder_v1"


def _parse_budgets(raw: str):
    vals = []
    for x in str(raw).split(","):
        x = x.strip()
        if x:
            q = float(x)
            if q > 1.0:
                q /= 100.0
            if not 0.0 <= q <= 1.0:
                raise ValueError("budgets must be percentages or fractions in [0,1]")
            vals.append(q)
    vals.extend([0.0, 1.0])
    return tuple(sorted(set(vals)))


def _scene_balanced_cap(records, cap: int, seed: int):
    """Round-robin a deterministic per-scene shuffle instead of taking first N."""
    if cap <= 0 or len(records) <= cap:
        return list(records)
    groups = {}
    for r in records:
        groups.setdefault(str(r["scene_name"]), []).append(r)
    rng = np.random.default_rng(int(seed))
    scenes = sorted(groups)
    scenes = [scenes[i] for i in rng.permutation(len(scenes))]
    queues = {}
    for scene in scenes:
        rows = groups[scene]
        order = rng.permutation(len(rows)).tolist()
        queues[scene] = [rows[i] for i in order]
    out = []
    depth = 0
    while len(out) < cap:
        added = False
        for scene in scenes:
            q = queues[scene]
            if depth < len(q):
                out.append(q[depth])
                added = True
                if len(out) >= cap:
                    return out
        if not added:
            break
        depth += 1
    return out


def _new_state(free_label: int):
    return {
        "safe": safe._new_metrics(),
        "occ": OccupancyIoUMultiHorizon(free_label=int(free_label)),
        "selected": 0,
        "sources": 0,
    }


def _update_state(st, horizon, pred, gt, moving, selected, sources):
    safe._update(st["safe"], float(horizon), pred, gt, moving)
    st["occ"].update(float(horizon), pred, gt)
    st["selected"] += int(selected)
    st["sources"] += int(sources)


def _state_report(st):
    rep = safe._report(st["safe"])
    rep["occupancy"] = st["occ"].compute()
    rep["selection"] = {
        "selected_source_events": int(st["selected"]),
        "source_events": int(st["sources"]),
        "ratio": float(st["selected"] / max(st["sources"], 1)),
    }
    return rep


def _mixture(anchor, baseline, learned, select, *, free_label, grid):
    repl = [learned[i] if bool(select[i]) else baseline[i] for i in range(len(baseline))]
    return compose_component_replacements_in_input_order(
        anchor,
        baseline,
        repl,
        dynamic_class_ids=DYNAMIC_CLASS_IDS,
        free_label=int(free_label),
        grid=grid,
    )


def _moving_correct(pred, gt, moving):
    m = np.asarray(moving, dtype=bool)
    if not np.any(m):
        return 0
    return int((np.asarray(pred)[m] == np.asarray(gt)[m]).sum())


def _selector_eval_payload(
    sample,
    *,
    source,
    window,
    pcfg,
    strong_cfg,
    history_occ,
    history_poses,
    future_poses,
):
    """Return GT/Strong-anchor/Moving payload for utility labels.

    Validation P0-F9 caches carry an evaluation-only payload and we reuse it
    exactly.  Train P0-F9 caches intentionally omit that payload to save space;
    in that case reconstruct the same quantities from the exact nuScenes window.
    Future GT remains label-only and never enters selector inputs.
    """
    required = (
        "eval_future_gt_occ",
        "eval_strong_anchor_occ",
        "eval_gt_moving_support",
    )
    if all(k in sample for k in required):
        return {
            "gt": sample["eval_future_gt_occ"].cpu().numpy(),
            "anchor": sample["eval_strong_anchor_occ"].cpu().numpy(),
            "moving": sample["eval_gt_moving_support"].cpu().numpy().astype(bool),
        }, "cached_eval"

    gt = np.stack(
        [
            np.asarray(source.load_semantics(window.scene_name, tok), dtype=np.uint8)
            for tok in window.future_tokens
        ],
        axis=0,
    )
    anchor = strong_w2det_sequence(
        history_occ,
        history_poses,
        future_poses,
        frame_dt_s=float(pcfg.frame_dt_s),
        grid=pcfg.grid,
        cfg=strong_cfg,
    ).astype(np.uint8, copy=False)
    moving = wm_base._gt_moving_support(source, window, pcfg).astype(bool, copy=False)
    if gt.shape != anchor.shape or gt.shape != moving.shape:
        raise RuntimeError(
            f"{sample['sample_id']}: reconstructed payload shape mismatch "
            f"gt={gt.shape} anchor={anchor.shape} moving={moving.shape}"
        )
    return {"gt": gt, "anchor": anchor, "moving": moving}, "raw_reconstructed"


def main():
    p = argparse.ArgumentParser()
    add_config_args(p)
    p.add_argument("--local-stwm-cache", required=True)
    p.add_argument("--p0f9-cache", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataroot", required=True)
    p.add_argument("--info-pkl", required=True)
    p.add_argument("--output-cache", required=True)
    p.add_argument("--output-report", required=True)
    p.add_argument("--budgets", default="0,10,20,40,100")
    p.add_argument("--random-repeats", type=int, default=5)
    p.add_argument("--random-seed", type=int, default=20260913)
    p.add_argument("--selection-seed", type=int, default=20260913)
    p.add_argument("--max-windows", type=int, default=0)
    p.add_argument("--evaluate-curves", action="store_true")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    if a.random_repeats <= 0:
        raise ValueError("random-repeats must be positive")
    budgets = _parse_budgets(a.budgets)
    cfg = load_runtime_config(a.config, a.override)
    pcfg = make_prepare_config(cfg)
    cache_meta, records = load_cache(a.local_stwm_cache)
    ds = MSPWorldModelCacheDataset(a.p0f9_cache)
    sid_to_idx = {str(e["sample_id"]): i for i, e in enumerate(ds.entries)}
    records = [r for r in records if str(r["sample_id"]) in sid_to_idx]
    if a.max_windows > 0:
        records = _scene_balanced_cap(records, int(a.max_windows), int(a.selection_seed))
    if not records:
        raise RuntimeError("no V17/P0F9 sample intersection")

    device = torch.device(a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu")
    ck, model = load_model(a.checkpoint, device)
    if ck.get("protocol") != "p0_f9_v17_local_spatial_temporal_world_model_v1":
        raise RuntimeError("selector feasibility requires the standard frozen V17 checkpoint")
    if not bool(ck.get("use_representation", False)):
        raise RuntimeError("selector feasibility requires V17 representation checkpoint")

    source = NuScenesWindowSource(a.dataroot, info_pkl=a.info_pkl, verbose=False)
    strong_cfg = StrongW2DetConfig(free_label=int(pcfg.free_label))

    oracle_states = {q: _new_state(pcfg.free_label) for q in budgets}
    speed_states = {q: _new_state(pcfg.free_label) for q in budgets}
    random_states = {
        rep: {q: _new_state(pcfg.free_label) for q in budgets}
        for rep in range(int(a.random_repeats))
    }
    out_records = []
    all_utility = []
    total_sources = 0
    payload_source_counts = {"cached_eval": 0, "raw_reconstructed": 0}

    for wi, rec in enumerate(records):
        sid = str(rec["sample_id"])
        sample = ds[sid_to_idx[sid]]
        w = window_from_record(rec)
        history_occ = [source.load_semantics(w.scene_name, tok) for tok in w.history_tokens]
        history_poses = [np.asarray(source.pose(tok), dtype=np.float64) for tok in w.history_tokens]
        future_poses = [np.asarray(source.pose(tok), dtype=np.float64) for tok in w.future_tokens]
        payload, payload_source = _selector_eval_payload(
            sample,
            source=source,
            window=w,
            pcfg=pcfg,
            strong_cfg=strong_cfg,
            history_occ=history_occ,
            history_poses=history_poses,
            future_poses=future_poses,
        )
        payload_source_counts[payload_source] += 1
        current = extract_instances(history_occ[-1], history_poses[-1], grid=pcfg.grid, cfg=strong_cfg)
        previous = extract_instances(history_occ[-2], history_poses[-2], grid=pcfg.grid, cfg=strong_cfg)
        velocities = match_instances(
            previous, current, float(pcfg.frame_dt_s),
            max_speed_mps=strong_cfg.max_match_speed_mps,
        )
        n = len(current)
        if n != int(rec["features"].shape[0]):
            raise RuntimeError(f"{sid}: Strong/V17 source count mismatch")
        if [int(x["class_id"]) for x in current] != [int(x) for x in rec["source_class_id"].tolist()]:
            raise RuntimeError(f"{sid}: Strong/V17 source order mismatch")

        features = rec["features"].float().to(device)
        tube = rec["local_semantic_tube"].to(device)
        kta_disp = rec["kta_displacement_xy_m"].float().to(device)
        frame_motion = rec["frame_motion_features"].float().to(device)
        source_mask = rec["target_source_mask_tube"].to(device)
        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            pred = model(features, tube, kta_disp, frame_motion, source_mask)
        residual = pred["residual_xy_m"].float().cpu().numpy()

        baseline_by_hi = {}
        learned_by_hi = {}
        base_pred_by_hi = {}
        t0_pose = history_poses[-1]
        for horizon, hi in safe.REPORT.items():
            baseline = []
            learned = []
            dt = (hi + 1) * float(pcfg.frame_dt_s)
            for i, comp in enumerate(current):
                src_center = np.asarray(comp["centroid_world"], dtype=np.float64)
                vel = np.asarray(velocities.get(i, np.zeros(3)), dtype=np.float64)
                baseline.append(
                    rasterize_rigid_component(
                        comp["voxel_indices"], int(comp["class_id"]),
                        t0_pose, future_poses[hi],
                        source_center_world=src_center,
                        target_center_world=src_center + vel * dt,
                        yaw_delta_rad=0.0, grid=pcfg.grid,
                    )
                )
                xy = rec["anchors_xy_t0_m"][i, hi].numpy() + residual[i, hi]
                dst = t0_xy_to_world_preserve_source_z(xy, src_center, t0_pose)
                learned.append(
                    rasterize_rigid_component(
                        comp["voxel_indices"], int(comp["class_id"]),
                        t0_pose, future_poses[hi],
                        source_center_world=src_center,
                        target_center_world=dst,
                        yaw_delta_rad=0.0, grid=pcfg.grid,
                    )
                )
            anchor = payload["anchor"][hi]
            baseline_by_hi[hi] = baseline
            learned_by_hi[hi] = learned
            base_pred_by_hi[hi] = _mixture(
                anchor, baseline, learned, np.zeros(n, dtype=bool),
                free_label=pcfg.free_label, grid=pcfg.grid,
            )

        utility_raw = np.zeros(n, dtype=np.float64)
        utility_h = np.zeros((n, len(safe.REPORT)), dtype=np.float64)
        moving_voxels = 0
        for hj, (horizon, hi) in enumerate(safe.REPORT.items()):
            gt = payload["gt"][hi]
            moving = payload["moving"][hi]
            moving_voxels += int(np.asarray(moving, dtype=bool).sum())
            base_correct = _moving_correct(base_pred_by_hi[hi], gt, moving)
            for i in range(n):
                sel = np.zeros(n, dtype=bool)
                sel[i] = True
                one = _mixture(
                    payload["anchor"][hi], baseline_by_hi[hi], learned_by_hi[hi], sel,
                    free_label=pcfg.free_label, grid=pcfg.grid,
                )
                delta = _moving_correct(one, gt, moving) - base_correct
                utility_raw[i] += float(delta)
                utility_h[i, hj] = float(delta)
        utility_pct = utility_raw / max(moving_voxels, 1) * 100.0
        all_utility.extend(utility_pct.tolist())
        total_sources += n

        causal = selector_features(rec)
        spd = speed_score(rec)
        out_records.append({
            "sample_id": sid,
            "scene_name": str(rec["scene_name"]),
            "features": causal.cpu(),
            "speed_score": spd.cpu(),
            "gt_utility_pct": torch.from_numpy(utility_pct.astype(np.float32)),
            "gt_utility_raw_correct_voxels": torch.from_numpy(utility_raw.astype(np.float32)),
            "gt_utility_per_report_horizon": torch.from_numpy(utility_h.astype(np.float32)),
            "num_gt_moving_voxels_report_horizons": int(moving_voxels),
            "source_class_id": rec["source_class_id"].clone().cpu(),
        })

        if a.evaluate_curves:
            strategy_masks = {
                "oracle": {q: top_fraction_mask(utility_pct, q) for q in budgets},
                "speed": {q: top_fraction_mask(spd, q) for q in budgets},
            }
            random_masks = {
                rep: {
                    q: random_fraction_mask(
                        n, q, int(a.random_seed) + rep * 100003, sample_id=sid
                    )
                    for q in budgets
                }
                for rep in range(int(a.random_repeats))
            }
            for horizon, hi in safe.REPORT.items():
                gt = payload["gt"][hi]
                moving = payload["moving"][hi]
                anchor = payload["anchor"][hi]
                for q in budgets:
                    om = strategy_masks["oracle"][q]
                    opred = _mixture(
                        anchor, baseline_by_hi[hi], learned_by_hi[hi], om,
                        free_label=pcfg.free_label, grid=pcfg.grid,
                    )
                    _update_state(oracle_states[q], horizon, opred, gt, moving, om.sum(), n)
                    sm = strategy_masks["speed"][q]
                    spred = _mixture(
                        anchor, baseline_by_hi[hi], learned_by_hi[hi], sm,
                        free_label=pcfg.free_label, grid=pcfg.grid,
                    )
                    _update_state(speed_states[q], horizon, spred, gt, moving, sm.sum(), n)
                    for rep in range(int(a.random_repeats)):
                        rm = random_masks[rep][q]
                        rpred = _mixture(
                            anchor, baseline_by_hi[hi], learned_by_hi[hi], rm,
                            free_label=pcfg.free_label, grid=pcfg.grid,
                        )
                        _update_state(
                            random_states[rep][q], horizon, rpred, gt, moving, rm.sum(), n
                        )

        if wi == 0 or (wi + 1) % 16 == 0 or wi + 1 == len(records):
            print(
                f"selector utility {wi+1}/{len(records)} sid={sid} sources={n} "
                f"positive={(utility_pct > 0).mean() if n else 0:.3f}",
                flush=True,
            )

    utility_summary = asdict(summarize_utility(all_utility))
    metadata = {
        "version": SELECTOR_CACHE_VERSION,
        "protocol": PROTOCOL,
        "feature_contract": FEATURE_CONTRACT,
        "feature_dim": SELECTOR_FEATURE_DIM,
        "feature_names": list(SELECTOR_FEATURE_NAMES),
        "utility_contract": UTILITY_CONTRACT,
        "local_stwm_cache": str(Path(a.local_stwm_cache).resolve()),
        "p0f9_cache": str(Path(a.p0f9_cache).resolve()),
        "checkpoint": str(Path(a.checkpoint).resolve()),
        "num_windows": len(out_records),
        "num_sources": total_sources,
        "scene_names": sorted({str(r["scene_name"]) for r in out_records}),
        "selection_protocol": (
            "all_common_records" if a.max_windows <= 0
            else f"scene_balanced_round_robin_cap_{int(a.max_windows)}_seed_{int(a.selection_seed)}"
        ),
        "utility_summary": utility_summary,
        "budgets": list(budgets),
        "gt_usage": "labels_and_oracle_ranking_only_never_selector_inputs",
        "eval_payload_source_counts": payload_source_counts,
        "eval_payload_contract": (
            "reuse_cached_validation_payload_else_reconstruct_exact_raw_nuscenes_v1"
        ),
        "expert_contract": "KTA_vs_V17_RL_epoch5_A1_source_order",
    }
    op = Path(a.output_cache)
    op.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"version": SELECTOR_CACHE_VERSION, "metadata": metadata, "records": out_records}, op)

    report = {"metadata": metadata}
    if a.evaluate_curves:
        oracle = {str(q): _state_report(oracle_states[q]) for q in budgets}
        speed = {str(q): _state_report(speed_states[q]) for q in budgets}
        random_rows = {
            str(q): [_state_report(random_states[r][q]) for r in range(int(a.random_repeats))]
            for q in budgets
        }
        random_summary = {}
        for q in budgets:
            rows = random_rows[str(q)]
            random_summary[str(q)] = {
                "Moving_mean": float(np.mean([x["moving"]["mIoU"] for x in rows])),
                "Moving_std": float(np.std([x["moving"]["mIoU"] for x in rows])),
                "mIoU_mean": float(np.mean([x["overall"]["mIoU"] for x in rows])),
                "IoU_mean": float(np.mean([x["occupancy"]["IoU"] for x in rows])),
                "repeats": rows,
            }
        q0 = oracle[str(0.0)]["moving"]["mIoU"]
        for table in (oracle, speed):
            for q in budgets:
                table[str(q)]["delta_Moving_vs_Q0"] = float(
                    table[str(q)]["moving"]["mIoU"] - q0
                )
        best_oracle = max(float(x["moving"]["mIoU"]) for x in oracle.values())
        report.update({
            "oracle_gt_utility_rank": oracle,
            "speed_rule": speed,
            "random": random_summary,
            "curve_summary": {
                "Q0_KTA_Moving": float(q0),
                "Q100_V17_Moving": float(oracle[str(1.0)]["moving"]["mIoU"]),
                "best_oracle_Moving": float(best_oracle),
                "best_oracle_gain_vs_KTA": float(best_oracle - q0),
                "note": (
                    "GT utility rank is a one-source counterfactual ranking, not an exact "
                    "combinatorial oracle. Every reported curve point is nevertheless composed "
                    "and evaluated exactly under the final A1 protocol."
                ),
            },
        })

    rp = Path(a.output_report)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("=== KTA/V17 SELECTOR UTILITY CACHE ===")
    print(json.dumps(metadata, indent=2))
    if a.evaluate_curves:
        print("=== GT-ASSISTED BUDGET CURVE ===")
        print(f"{'Q':>7s} {'oracle':>10s} {'speed':>10s} {'random':>10s}")
        for q in budgets:
            print(
                f"{100*q:6.1f}% "
                f"{oracle[str(q)]['moving']['mIoU']:10.4f} "
                f"{speed[str(q)]['moving']['mIoU']:10.4f} "
                f"{random_summary[str(q)]['Moving_mean']:10.4f}"
            )
        print(json.dumps(report["curve_summary"], indent=2))
    print(f"saved cache {op}")
    print(f"saved report {rp}")


if __name__ == "__main__":
    main()
