#!/usr/bin/env python3
"""Evaluate learned center prediction through real rigid occupancy transport.

This is the decisive deployment test for P0-F9 v12.  It does not score only
ADE/FDE: predicted KTA-residual centers are used to coherently CLEAR each
Strong/KTA source footprint and WRITE the observed t0 3D source shape at the
predicted future position.  The result is evaluated with the frozen 128-window
Overall and Moving-v2 protocol.
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

from real_motion.motion_transport import (
    FEATURE_DIM,
    FUTURE_FRAMES,
    MOTION_TRANSPORT_CACHE_VERSION,
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
from real_motion.metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS
from tools.real_motion import eval_p0_f9_frozen_sparse_occfm as safe

PROTOCOL = "p0_f9_v12_learned_motion_transport_eval_v1"
VARIANTS = ("strong_anchor", "learned_center_always", "learned_center_rigid", "gt_center_rigid")


def load_motion_cache(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if obj.get("version") != MOTION_TRANSPORT_CACHE_VERSION:
        raise RuntimeError("motion cache version mismatch")
    records = obj.get("records") or []
    if not records:
        raise RuntimeError("motion cache has no records")
    return obj.get("metadata") or {}, records


def load_model(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("protocol") != "p0_f9_v12_learned_motion_transport_v1":
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
        scene_name=str(r["scene_name"]), history_tokens=tuple(str(x) for x in r["history_tokens"]),
        t0_token=str(r["t0_token"]), future_tokens=tuple(str(x) for x in r["future_tokens"]),
    )


def t0_xy_to_world_preserve_source_z(xy_t0, source_center_world, t0_pose):
    src_t0 = world_points_to_t0(np.asarray(source_center_world, dtype=np.float64)[None], t0_pose)[0]
    p = np.asarray([float(xy_t0[0]), float(xy_t0[1]), float(src_t0[2]), 1.0], dtype=np.float64)
    return (np.asarray(t0_pose, dtype=np.float64) @ p)[:3]


def metric_states():
    return {name: safe._new_metrics() for name in VARIANTS}


def metric_pair(report):
    return float(report["overall"]["mIoU"]), float(report["moving"]["mIoU"])


def main():
    p = argparse.ArgumentParser()
    add_config_args(p)
    p.add_argument("--motion-cache", required=True, help="scene-disjoint Strong-source motion val cache")
    p.add_argument("--p0f9-cache", required=True, help="frozen P0-F9 128-window eval cache")
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
    states = metric_states()

    kta_errors = []
    learned_errors = []
    kta_fde = []
    learned_fde = []
    exist_tp = exist_fp = exist_fn = exist_correct = exist_total = 0
    source_count = supervised_count = 0

    for wi, rec in enumerate(records):
        sid = str(rec["sample_id"])
        s = ds[sid_to_idx[sid]]
        payload = safe._sample_payload(s, torch.device("cpu"))
        w = window_from_record(rec)
        history_occ = [source.load_semantics(w.scene_name, tok) for tok in w.history_tokens]
        history_poses = [np.asarray(source.pose(tok), dtype=np.float64) for tok in w.history_tokens]
        future_poses = [np.asarray(source.pose(tok), dtype=np.float64) for tok in w.future_tokens]
        current = extract_instances(history_occ[-1], history_poses[-1], grid=pcfg.grid, cfg=strong_cfg)
        previous = extract_instances(history_occ[-2], history_poses[-2], grid=pcfg.grid, cfg=strong_cfg)
        velocities = match_instances(
            previous, current, float(pcfg.frame_dt_s), max_speed_mps=strong_cfg.max_match_speed_mps
        )
        if len(current) != int(rec["features"].shape[0]):
            raise RuntimeError(f"{sid}: reconstructed Strong source count differs from motion cache")
        got_classes = [int(c["class_id"]) for c in current]
        if got_classes != [int(x) for x in rec["source_class_id"].tolist()]:
            raise RuntimeError(f"{sid}: reconstructed Strong source ordering differs from motion cache")

        features = rec["features"].float().to(device)
        with torch.no_grad():
            out = model(features)
            pred_residual = out["residual_xy_m"].cpu().numpy()
            exist_prob = torch.sigmoid(out["existence_logits"]).cpu().numpy()

        # Trajectory/existence diagnostics use labels only after causal inference.
        valid = rec["target_valid"].bool().numpy()
        target_res = rec["target_residual_xy_m"].float().numpy()
        supervised = rec["supervised_source"].bool().numpy()
        source_count += len(current); supervised_count += int(supervised.sum())
        for i in range(len(current)):
            ids = np.flatnonzero(valid[i])
            for h in ids:
                kta_errors.append(float(np.linalg.norm(target_res[i, h])))
                learned_errors.append(float(np.linalg.norm(pred_residual[i, h] - target_res[i, h])))
            if len(ids):
                h = int(ids[-1])
                kta_fde.append(float(np.linalg.norm(target_res[i, h])))
                learned_fde.append(float(np.linalg.norm(pred_residual[i, h] - target_res[i, h])))
            if supervised[i]:
                gt_e = rec["existence"][i].numpy() > 0.5
                pr_e = exist_prob[i] >= float(a.existence_threshold)
                exist_correct += int((gt_e == pr_e).sum()); exist_total += FUTURE_FRAMES
                exist_tp += int((gt_e & pr_e).sum()); exist_fp += int((~gt_e & pr_e).sum()); exist_fn += int((gt_e & ~pr_e).sum())

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
            learned_all = []
            learned_exist = []
            oracle_baseline = []
            oracle_repl = []
            dt = (hi + 1) * float(pcfg.frame_dt_s)
            for i, comp in enumerate(current):
                src_center = np.asarray(comp["centroid_world"], dtype=np.float64)
                v = np.asarray(velocities.get(i, np.zeros(3)), dtype=np.float64)
                kta_center = src_center + v * dt
                base = rasterize_rigid_component(
                    comp["voxel_indices"], int(comp["class_id"]), t0_pose, future_poses[hi],
                    source_center_world=src_center, target_center_world=kta_center,
                    yaw_delta_rad=0.0, grid=pcfg.grid,
                )
                baseline_all.append(base)
                xy_pred = rec["anchors_xy_t0_m"][i, hi].numpy() + pred_residual[i, hi]
                center_pred = t0_xy_to_world_preserve_source_z(xy_pred, src_center, t0_pose)
                repl = rasterize_rigid_component(
                    comp["voxel_indices"], int(comp["class_id"]), t0_pose, future_poses[hi],
                    source_center_world=src_center, target_center_world=center_pred,
                    yaw_delta_rad=0.0, grid=pcfg.grid,
                )
                learned_all.append(repl)
                if exist_prob[i, hi] >= float(a.existence_threshold):
                    learned_exist.append(repl)

                token = source_tokens[i]
                if token is not None:
                    oracle_baseline.append(base)
                    annh = future_maps[hi].get(str(token))
                    if annh is not None:
                        oracle_repl.append(rasterize_rigid_component(
                            comp["voxel_indices"], int(comp["class_id"]), t0_pose, future_poses[hi],
                            source_center_world=src_center,
                            target_center_world=np.asarray(annh["center_world"], dtype=np.float64),
                            yaw_delta_rad=0.0, grid=pcfg.grid,
                        ))

            pred_always = compose_component_replacements(
                anchor, baseline_all, learned_all, dynamic_class_ids=DYNAMIC_CLASS_IDS,
                free_label=int(pcfg.free_label), grid=pcfg.grid,
            )
            pred_exist = compose_component_replacements(
                anchor, baseline_all, learned_exist, dynamic_class_ids=DYNAMIC_CLASS_IDS,
                free_label=int(pcfg.free_label), grid=pcfg.grid,
            )
            pred_oracle = compose_component_replacements(
                anchor, oracle_baseline, oracle_repl, dynamic_class_ids=DYNAMIC_CLASS_IDS,
                free_label=int(pcfg.free_label), grid=pcfg.grid,
            )
            safe._update(states["learned_center_always"], float(horizon), pred_always, gt, moving)
            safe._update(states["learned_center_rigid"], float(horizon), pred_exist, gt, moving)
            safe._update(states["gt_center_rigid"], float(horizon), pred_oracle, gt, moving)

        if wi == 0 or (wi + 1) % 8 == 0 or wi + 1 == len(records):
            print(f"learned_transport_eval {wi+1}/{len(records)} sid={sid} sources={len(current)}")

    reports = {name: safe._report(state) for name, state in states.items()}
    so, sm = metric_pair(reports["strong_anchor"])
    go, gm = metric_pair(reports["gt_center_rigid"])
    lo, lm = metric_pair(reports["learned_center_rigid"])
    denom = gm - sm
    recovery = (lm - sm) / denom if abs(denom) > 1e-12 else float("nan")
    prec = exist_tp / max(exist_tp + exist_fp, 1)
    rec_e = exist_tp / max(exist_tp + exist_fn, 1)
    diagnostics = {
        "source_count": source_count,
        "supervised_source_count": supervised_count,
        "source_supervision_fraction": supervised_count / max(source_count, 1),
        "kta_ade_m": float(np.mean(kta_errors)) if kta_errors else float("nan"),
        "learned_ade_m": float(np.mean(learned_errors)) if learned_errors else float("nan"),
        "kta_fde_m": float(np.mean(kta_fde)) if kta_fde else float("nan"),
        "learned_fde_m": float(np.mean(learned_fde)) if learned_fde else float("nan"),
        "existence_accuracy": exist_correct / max(exist_total, 1),
        "existence_precision": prec,
        "existence_recall": rec_e,
        "existence_f1": 2 * prec * rec_e / max(prec + rec_e, 1e-12),
        "moving_oracle_recovery_fraction": recovery,
    }

    print("\n=== P0-F9 LEARNED MOTION TRANSPORT ===")
    print(f"{'variant':24s} {'Overall':>9s} {'Moving':>9s} {'dOverall':>10s} {'dMoving':>9s}")
    for name in VARIANTS:
        o, m = metric_pair(reports[name])
        print(f"{name:24s} {o:9.4f} {m:9.4f} {o-so:+10.4f} {m-sm:+9.4f}")
    print("\n=== MOVING BY HORIZON ===")
    for name in VARIANTS:
        vals = reports[name]["moving"]["per_horizon"]
        row = []
        for h in safe.REPORT:
            v = vals[float(h)] if float(h) in vals else vals[str(float(h))]
            row.append(float(v["mIoU"]))
        print(f"{name:24s} 1s={row[0]:.4f} 2s={row[1]:.4f} 3s={row[2]:.4f}")
    print("\n=== TRAJECTORY / EXISTENCE DIAGNOSTICS ===")
    for k, v in diagnostics.items():
        print(f"{k}={v}")

    result = {
        "protocol": PROTOCOL,
        "checkpoint": str(Path(a.checkpoint).resolve()),
        "checkpoint_epoch": int(ck.get("epoch", -1)),
        "motion_cache": str(Path(a.motion_cache).resolve()),
        "p0f9_cache": str(Path(a.p0f9_cache).resolve()),
        "existence_threshold": float(a.existence_threshold),
        "num_windows": len(records),
        "reports": reports,
        "diagnostics": diagnostics,
        "motion_cache_metadata": motion_meta,
    }
    op = Path(a.output); op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"saved {op}")


if __name__ == "__main__":
    main()
