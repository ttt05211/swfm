#!/usr/bin/env python3
"""Diagnose why v13 trajectory gains convert weakly into Moving-mIoU.

No training is performed.  The frozen v13 motion head is evaluated once on the
same scene-disjoint 128 windows, then the diagnostic separates four questions:

1. Does ADE/FDE also improve on *true-moving* source/horizon observations under
   the exact Moving-mIoU v2 0.5 m/s center-motion rule?
2. Are KTA and the learned correction complementary?  A GT-only selector oracle
   chooses the lower center-error option per source/horizon.
3. How nonlinear is Moving-mIoU with respect to center accuracy?  KTA motion is
   interpolated 25/50/75/100% toward GT displacement before rigid transport.
4. Which horizon, semantic class and source-size bucket retain the motion error?

The script preserves the v13 displacement contract and never aligns a visible
Strong component centroid to an annotation-box center.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from real_motion.metrics.moving_miou_v2 import (
    DYNAMIC_CLASS_IDS,
    NUSCENES_LABELS,
    SPEED_THRESHOLD_MPS,
)
from real_motion.motion_transport_gap import (
    displacement_errors,
    learned_winner_mask,
    masked_error_stats,
    summarize_rows,
    true_moving_observation_mask,
)
from real_motion.motion_transport_v2 import (
    FEATURE_DIM,
    FUTURE_FRAMES,
    MOTION_TRANSPORT_CACHE_VERSION,
    TARGET_CONTRACT,
    MotionTransportHead,
    annotation_map,
    dynamic_annotations,
    match_sources_to_annotations,
    world_points_to_t0,
)
from real_motion.msp_wm_cache import MSPWorldModelCacheDataset
from real_motion.nuscenes_adapter import NuScenesWindowSource, WindowTokens
from real_motion.rigid_transport import compose_component_replacements, rasterize_rigid_component
from real_motion.runtime_config import add_config_args, load_runtime_config, make_prepare_config
from real_motion.strong_w2det import StrongW2DetConfig, extract_instances, match_instances
from tools.real_motion import eval_p0_f9_frozen_sparse_occfm as safe

PROTOCOL = "p0_f9_v14_motion_transport_gap_diagnostic_v1"
CHECKPOINT_PROTOCOL = "p0_f9_v13_learned_motion_transport_v2"
ALPHAS = (0.25, 0.50, 0.75)
BASE_VARIANTS = (
    "strong_anchor",
    "learned_center_rigid",
    "kta_learned_selector_oracle",
    "gt_center_rigid",
)


def load_motion_cache(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if obj.get("version") != MOTION_TRANSPORT_CACHE_VERSION:
        raise RuntimeError("motion cache version mismatch")
    meta = obj.get("metadata") or {}
    if meta.get("target_contract") != TARGET_CONTRACT:
        raise RuntimeError("motion cache target contract mismatch")
    records = obj.get("records") or []
    if not records:
        raise RuntimeError("motion cache has no records")
    return meta, records


def load_model(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("protocol") != CHECKPOINT_PROTOCOL:
        raise RuntimeError("checkpoint protocol mismatch")
    if int(ck.get("feature_dim", -1)) != FEATURE_DIM:
        raise RuntimeError("checkpoint feature dimension mismatch")
    model = MotionTransportHead(
        feature_dim=FEATURE_DIM,
        hidden_dim=int(ck.get("hidden_dim", 128)),
        future_frames=int(ck.get("future_frames", FUTURE_FRAMES)),
    ).to(device)
    model.load_state_dict(ck["state_dict"], strict=True)
    model.eval()
    return ck, model


def window_from_record(r):
    return WindowTokens(
        scene_name=str(r["scene_name"]),
        history_tokens=tuple(str(x) for x in r["history_tokens"]),
        t0_token=str(r["t0_token"]),
        future_tokens=tuple(str(x) for x in r["future_tokens"]),
    )


def t0_xy_to_world_preserve_source_z(xy_t0, source_center_world, t0_pose):
    src_t0 = world_points_to_t0(np.asarray(source_center_world, dtype=np.float64)[None], t0_pose)[0]
    p = np.asarray([float(xy_t0[0]), float(xy_t0[1]), float(src_t0[2]), 1.0], dtype=np.float64)
    return (np.asarray(t0_pose, dtype=np.float64) @ p)[:3]


def metric_pair(report):
    return float(report["overall"]["mIoU"]), float(report["moving"]["mIoU"])


def _mean_class_across_horizons(report, class_id):
    vals = []
    for h in safe.REPORT:
        row = report["moving"]["per_horizon"]
        r = row[float(h)] if float(h) in row else row[str(float(h))]
        pc = r["per_class"]
        v = pc.get(int(class_id), pc.get(str(int(class_id)), float("nan")))
        if not np.isnan(float(v)):
            vals.append(float(v))
    return float(np.mean(vals)) if vals else float("nan")


def _size_report(true_rows):
    if not true_rows:
        return {"edges_voxels": [], "buckets": {}}
    sizes = np.asarray([float(r["voxel_count"]) for r in true_rows], dtype=np.float64)
    q1, q2 = [float(x) for x in np.quantile(sizes, [1/3, 2/3])]
    buckets = {
        "small": [r for r in true_rows if float(r["voxel_count"]) <= q1],
        "medium": [r for r in true_rows if q1 < float(r["voxel_count"]) <= q2],
        "large": [r for r in true_rows if float(r["voxel_count"]) > q2],
    }
    return {
        "edges_voxels": [q1, q2],
        "buckets": {name: summarize_rows(rows) for name, rows in buckets.items()},
    }


def main():
    p = argparse.ArgumentParser()
    add_config_args(p)
    p.add_argument("--motion-cache", required=True)
    p.add_argument("--p0f9-cache", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataroot", required=True)
    p.add_argument("--info-pkl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--existence-threshold", type=float, default=0.5)
    p.add_argument("--match-max-distance-m", type=float, default=4.0)
    p.add_argument("--max-windows", type=int, default=0)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    cfg = load_runtime_config(a.config, a.override)
    pcfg = make_prepare_config(cfg)
    motion_meta, records = load_motion_cache(a.motion_cache)
    ds = MSPWorldModelCacheDataset(a.p0f9_cache)
    sid_to_idx = {str(e["sample_id"]): i for i, e in enumerate(ds.entries)}
    missing = [str(r["sample_id"]) for r in records if str(r["sample_id"]) not in sid_to_idx]
    if missing:
        raise RuntimeError(f"P0-F9 cache misses motion-val samples: {missing[:5]}")
    if int(a.max_windows) > 0:
        records = records[: min(len(records), int(a.max_windows))]

    device = torch.device(a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu")
    ck, model = load_model(a.checkpoint, device)
    source = NuScenesWindowSource(a.dataroot, info_pkl=a.info_pkl, verbose=False)
    strong_cfg = StrongW2DetConfig(free_label=int(pcfg.free_label))

    variant_names = list(BASE_VARIANTS) + [f"gt_interp_{alpha:.2f}" for alpha in ALPHAS]
    states = {name: safe._new_metrics() for name in variant_names}
    trajectory_rows = []
    all_kta_fde, all_learned_fde = [], []
    moving_kta_fde, moving_learned_fde = [], []
    source_count = supervised_count = 0

    for wi, rec in enumerate(records):
        sid = str(rec["sample_id"])
        sample = ds[sid_to_idx[sid]]
        payload = safe._sample_payload(sample, torch.device("cpu"))
        w = window_from_record(rec)
        history_occ = [source.load_semantics(w.scene_name, tok) for tok in w.history_tokens]
        history_poses = [np.asarray(source.pose(tok), dtype=np.float64) for tok in w.history_tokens]
        future_poses = [np.asarray(source.pose(tok), dtype=np.float64) for tok in w.future_tokens]
        current = extract_instances(history_occ[-1], history_poses[-1], grid=pcfg.grid, cfg=strong_cfg)
        previous = extract_instances(history_occ[-2], history_poses[-2], grid=pcfg.grid, cfg=strong_cfg)
        velocities = match_instances(
            previous,
            current,
            float(pcfg.frame_dt_s),
            max_speed_mps=strong_cfg.max_match_speed_mps,
        )
        if len(current) != int(rec["features"].shape[0]):
            raise RuntimeError(f"{sid}: reconstructed Strong source count differs from motion cache")
        if [int(c["class_id"]) for c in current] != [int(x) for x in rec["source_class_id"].tolist()]:
            raise RuntimeError(f"{sid}: reconstructed Strong source ordering differs from motion cache")

        features = rec["features"].float().to(device)
        with torch.no_grad():
            out = model(features)
            pred_residual = out["residual_xy_m"].cpu().numpy()
            exist_prob = torch.sigmoid(out["existence_logits"]).cpu().numpy()

        valid = rec["target_valid"].bool().numpy()
        target_res = rec["target_residual_xy_m"].float().numpy()
        target_disp = rec["target_displacement_xy_m"].float().numpy()
        supervised = rec["supervised_source"].bool().numpy()
        true_moving = true_moving_observation_mask(
            target_disp,
            valid,
            frame_dt_s=float(pcfg.frame_dt_s),
            speed_threshold_mps=SPEED_THRESHOLD_MPS,
        )
        kta_err, learned_err = displacement_errors(pred_residual, target_res)
        winner = learned_winner_mask(pred_residual, target_res, valid)
        source_count += len(current)
        supervised_count += int(supervised.sum())

        for i in range(len(current)):
            ids = np.flatnonzero(valid[i])
            if len(ids):
                h = int(ids[-1])
                all_kta_fde.append(float(kta_err[i, h]))
                all_learned_fde.append(float(learned_err[i, h]))
            mids = np.flatnonzero(true_moving[i])
            if len(mids):
                h = int(mids[-1])
                moving_kta_fde.append(float(kta_err[i, h]))
                moving_learned_fde.append(float(learned_err[i, h]))
            for hi in ids:
                trajectory_rows.append({
                    "sample_id": sid,
                    "source_index": int(i),
                    "horizon_index": int(hi),
                    "horizon_s": float((hi + 1) * pcfg.frame_dt_s),
                    "class_id": int(rec["source_class_id"][i]),
                    "voxel_count": int(rec["source_voxel_count"][i]),
                    "true_moving": bool(true_moving[i, hi]),
                    "kta_error_m": float(kta_err[i, hi]),
                    "learned_error_m": float(learned_err[i, hi]),
                    "learned_wins": bool(winner[i, hi]),
                })

        anns0 = dynamic_annotations(source.nusc, w.t0_token)
        source_tokens = match_sources_to_annotations(current, anns0, max_distance_m=float(a.match_max_distance_m))
        future_maps = [annotation_map(source.nusc, tok) for tok in w.future_tokens]
        ann0_by_token = {str(x["instance_token"]): x for x in anns0}
        t0_pose = history_poses[-1]

        for horizon, hi in safe.REPORT.items():
            gt = payload["gt"][hi]
            anchor = payload["anchor"][hi]
            moving = payload["moving"][hi]
            safe._update(states["strong_anchor"], float(horizon), anchor, gt, moving)

            baseline_all = []
            learned_exist = []
            selector_baseline = []
            selector_repl = []
            gt_baseline = []
            gt_repl = []
            interp_baseline = {alpha: [] for alpha in ALPHAS}
            interp_repl = {alpha: [] for alpha in ALPHAS}
            dt = (hi + 1) * float(pcfg.frame_dt_s)

            for i, comp in enumerate(current):
                src_center = np.asarray(comp["centroid_world"], dtype=np.float64)
                v = np.asarray(velocities.get(i, np.zeros(3)), dtype=np.float64)
                kta_disp_world = v * dt
                kta_center = src_center + kta_disp_world
                base = rasterize_rigid_component(
                    comp["voxel_indices"], int(comp["class_id"]), t0_pose, future_poses[hi],
                    source_center_world=src_center, target_center_world=kta_center,
                    yaw_delta_rad=0.0, grid=pcfg.grid,
                )
                baseline_all.append(base)

                xy_pred = rec["anchors_xy_t0_m"][i, hi].numpy() + pred_residual[i, hi]
                center_pred = t0_xy_to_world_preserve_source_z(xy_pred, src_center, t0_pose)
                learned_repl = rasterize_rigid_component(
                    comp["voxel_indices"], int(comp["class_id"]), t0_pose, future_poses[hi],
                    source_center_world=src_center, target_center_world=center_pred,
                    yaw_delta_rad=0.0, grid=pcfg.grid,
                )
                if exist_prob[i, hi] >= float(a.existence_threshold):
                    learned_exist.append(learned_repl)

                # GT-only selector: leave KTA untouched unless learned center is
                # strictly closer to the correct displacement at this horizon.
                if bool(winner[i, hi]):
                    selector_baseline.append(base)
                    selector_repl.append(learned_repl)

                token = source_tokens[i]
                if token is None:
                    continue
                ann0 = ann0_by_token.get(str(token))
                annh = future_maps[hi].get(str(token))
                if ann0 is None or annh is None:
                    continue
                gt_disp_world = (
                    np.asarray(annh["center_world"], dtype=np.float64)
                    - np.asarray(ann0["center_world"], dtype=np.float64)
                )
                gt_baseline.append(base)
                gt_center = src_center + gt_disp_world
                gt_repl.append(rasterize_rigid_component(
                    comp["voxel_indices"], int(comp["class_id"]), t0_pose, future_poses[hi],
                    source_center_world=src_center, target_center_world=gt_center,
                    yaw_delta_rad=0.0, grid=pcfg.grid,
                ))
                for alpha in ALPHAS:
                    interp_baseline[alpha].append(base)
                    disp = kta_disp_world + float(alpha) * (gt_disp_world - kta_disp_world)
                    interp_repl[alpha].append(rasterize_rigid_component(
                        comp["voxel_indices"], int(comp["class_id"]), t0_pose, future_poses[hi],
                        source_center_world=src_center, target_center_world=src_center + disp,
                        yaw_delta_rad=0.0, grid=pcfg.grid,
                    ))

            learned_pred = compose_component_replacements(
                anchor, baseline_all, learned_exist,
                dynamic_class_ids=DYNAMIC_CLASS_IDS,
                free_label=int(pcfg.free_label), grid=pcfg.grid,
            )
            selector_pred = compose_component_replacements(
                anchor, selector_baseline, selector_repl,
                dynamic_class_ids=DYNAMIC_CLASS_IDS,
                free_label=int(pcfg.free_label), grid=pcfg.grid,
            )
            gt_pred = compose_component_replacements(
                anchor, gt_baseline, gt_repl,
                dynamic_class_ids=DYNAMIC_CLASS_IDS,
                free_label=int(pcfg.free_label), grid=pcfg.grid,
            )
            safe._update(states["learned_center_rigid"], float(horizon), learned_pred, gt, moving)
            safe._update(states["kta_learned_selector_oracle"], float(horizon), selector_pred, gt, moving)
            safe._update(states["gt_center_rigid"], float(horizon), gt_pred, gt, moving)
            for alpha in ALPHAS:
                pred = compose_component_replacements(
                    anchor, interp_baseline[alpha], interp_repl[alpha],
                    dynamic_class_ids=DYNAMIC_CLASS_IDS,
                    free_label=int(pcfg.free_label), grid=pcfg.grid,
                )
                safe._update(states[f"gt_interp_{alpha:.2f}"], float(horizon), pred, gt, moving)

        if wi == 0 or (wi + 1) % 8 == 0 or wi + 1 == len(records):
            print(f"motion_gap_v14 {wi+1}/{len(records)} sid={sid} sources={len(current)}")

    reports = {name: safe._report(state) for name, state in states.items()}
    all_rows = trajectory_rows
    true_rows = [r for r in trajectory_rows if bool(r["true_moving"])]
    all_stats = summarize_rows(all_rows)
    true_stats = summarize_rows(true_rows)
    all_stats.update({
        "fde_count": len(all_kta_fde),
        "kta_fde_m": float(np.mean(all_kta_fde)) if all_kta_fde else float("nan"),
        "learned_fde_m": float(np.mean(all_learned_fde)) if all_learned_fde else float("nan"),
    })
    true_stats.update({
        "fde_count": len(moving_kta_fde),
        "kta_fde_m": float(np.mean(moving_kta_fde)) if moving_kta_fde else float("nan"),
        "learned_fde_m": float(np.mean(moving_learned_fde)) if moving_learned_fde else float("nan"),
        "fraction_of_valid_observations": len(true_rows) / max(len(all_rows), 1),
    })

    by_horizon = {}
    for h in safe.REPORT:
        rows = [r for r in true_rows if abs(float(r["horizon_s"]) - float(h)) < 1e-9]
        by_horizon[str(float(h))] = summarize_rows(rows)
    by_class = {}
    for c in DYNAMIC_CLASS_IDS:
        rows = [r for r in true_rows if int(r["class_id"]) == int(c)]
        by_class[str(int(c))] = {
            "class_name": NUSCENES_LABELS[int(c)],
            **summarize_rows(rows),
        }
    size_report = _size_report(true_rows)

    so, sm = metric_pair(reports["strong_anchor"])
    _, lm = metric_pair(reports["learned_center_rigid"])
    _, selm = metric_pair(reports["kta_learned_selector_oracle"])
    _, gm = metric_pair(reports["gt_center_rigid"])
    sensitivity = [{"alpha": 0.0, "overall": so, "moving": sm}]
    for alpha in ALPHAS:
        o, m = metric_pair(reports[f"gt_interp_{alpha:.2f}"])
        sensitivity.append({"alpha": float(alpha), "overall": o, "moving": m})
    go, gm2 = metric_pair(reports["gt_center_rigid"])
    sensitivity.append({"alpha": 1.0, "overall": go, "moving": gm2})

    class_occ = {}
    for c in DYNAMIC_CLASS_IDS:
        class_occ[str(int(c))] = {
            "class_name": NUSCENES_LABELS[int(c)],
            "strong": _mean_class_across_horizons(reports["strong_anchor"], c),
            "learned": _mean_class_across_horizons(reports["learned_center_rigid"], c),
            "selector_oracle": _mean_class_across_horizons(reports["kta_learned_selector_oracle"], c),
            "gt_center": _mean_class_across_horizons(reports["gt_center_rigid"], c),
        }

    diagnostic = {
        "source_count": int(source_count),
        "supervised_source_count": int(supervised_count),
        "source_supervision_fraction": supervised_count / max(source_count, 1),
        "all_valid_trajectory": all_stats,
        "true_moving_trajectory": true_stats,
        "true_moving_by_report_horizon": by_horizon,
        "true_moving_by_class": by_class,
        "true_moving_by_source_size": size_report,
        "moving_selector_oracle_gain_over_learned": selm - lm,
        "moving_selector_oracle_gain_over_kta": selm - sm,
        "moving_gt_center_headroom": gm - sm,
        "learned_moving_recovery_fraction": (lm - sm) / (gm - sm) if abs(gm - sm) > 1e-12 else float("nan"),
        "selector_moving_recovery_fraction": (selm - sm) / (gm - sm) if abs(gm - sm) > 1e-12 else float("nan"),
        "center_sensitivity": sensitivity,
        "moving_per_class_mean_over_horizons": class_occ,
    }

    print("\n=== V14 MOTION GAP: OCCUPANCY ORACLES ===")
    print(f"{'variant':30s} {'Overall':>9s} {'Moving':>9s} {'dOverall':>10s} {'dMoving':>9s}")
    for name in ["strong_anchor", "learned_center_rigid", "kta_learned_selector_oracle", "gt_center_rigid"]:
        o, m = metric_pair(reports[name])
        print(f"{name:30s} {o:9.4f} {m:9.4f} {o-so:+10.4f} {m-sm:+9.4f}")

    print("\n=== V14 TRUE-MOVING TRAJECTORY ERROR ===")
    print(
        f"all_valid count={all_stats['count']} KTA_ADE={all_stats['kta_ade_m']:.4f} "
        f"Learned_ADE={all_stats['learned_ade_m']:.4f} reduction={100*all_stats['relative_ade_reduction']:.2f}% "
        f"learned_win={100*all_stats['learned_win_fraction']:.2f}%"
    )
    print(
        f"true_moving count={true_stats['count']} fraction={100*true_stats['fraction_of_valid_observations']:.2f}% "
        f"KTA_ADE={true_stats['kta_ade_m']:.4f} Learned_ADE={true_stats['learned_ade_m']:.4f} "
        f"reduction={100*true_stats['relative_ade_reduction']:.2f}% learned_win={100*true_stats['learned_win_fraction']:.2f}%"
    )
    print(
        f"true_moving_FDE count={true_stats['fde_count']} KTA={true_stats['kta_fde_m']:.4f} "
        f"Learned={true_stats['learned_fde_m']:.4f}"
    )

    print("\n=== V14 TRUE-MOVING BY REPORT HORIZON ===")
    for h, row in by_horizon.items():
        print(
            f"{h}s count={row['count']} KTA={row['kta_ade_m']:.4f} Learned={row['learned_ade_m']:.4f} "
            f"reduction={100*row['relative_ade_reduction']:.2f}% win={100*row['learned_win_fraction']:.2f}%"
        )

    print("\n=== V14 KTA/LEARNED SELECTOR SIGNAL ===")
    print(f"learned Moving={lm:.4f}")
    print(f"selector-oracle Moving={selm:.4f} gain_over_learned={selm-lm:+.4f} gain_over_KTA={selm-sm:+.4f}")
    print(f"GT-center Moving={gm:.4f}")

    print("\n=== V14 CENTER-ERROR SENSITIVITY ===")
    for row in sensitivity:
        print(f"alpha={row['alpha']:.2f} Overall={row['overall']:.4f} Moving={row['moving']:.4f}")

    print("\n=== V14 TRUE-MOVING BY CLASS ===")
    for c in DYNAMIC_CLASS_IDS:
        row = by_class[str(int(c))]
        print(
            f"{row['class_name']:22s} count={row['count']:5d} KTA={row['kta_ade_m']:.4f} "
            f"Learned={row['learned_ade_m']:.4f} reduction={100*row['relative_ade_reduction']:.2f}% "
            f"win={100*row['learned_win_fraction']:.2f}%"
        )

    print("\n=== V14 TRUE-MOVING BY SOURCE SIZE ===")
    print(f"tertile_edges_voxels={size_report['edges_voxels']}")
    for name, row in size_report["buckets"].items():
        print(
            f"{name:8s} count={row['count']:5d} KTA={row['kta_ade_m']:.4f} Learned={row['learned_ade_m']:.4f} "
            f"reduction={100*row['relative_ade_reduction']:.2f}% win={100*row['learned_win_fraction']:.2f}%"
        )

    result = {
        "protocol": PROTOCOL,
        "checkpoint": str(Path(a.checkpoint).resolve()),
        "checkpoint_epoch": int(ck.get("epoch", -1)),
        "motion_cache": str(Path(a.motion_cache).resolve()),
        "p0f9_cache": str(Path(a.p0f9_cache).resolve()),
        "num_windows": len(records),
        "moving_speed_threshold_mps": SPEED_THRESHOLD_MPS,
        "reports": reports,
        "diagnostic": diagnostic,
        "motion_cache_metadata": motion_meta,
    }
    op = Path(a.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"saved {op}")


if __name__ == "__main__":
    main()
