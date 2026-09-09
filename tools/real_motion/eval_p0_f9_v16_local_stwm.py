#!/usr/bin/env python3
"""Evaluate v16 local ST world-model motion through rigid occupancy transport."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from real_motion.local_st_world_model import (
    LOCAL_STWM_CACHE_VERSION,
    LOCAL_TUBE_CONTRACT,
    MODEL_PROTOCOL,
    LocalSpatialTemporalWorldModel,
    config_from_mapping,
)
from real_motion.metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS
from real_motion.motion_transport import FUTURE_FRAMES
from real_motion.motion_transport_v2 import TARGET_CONTRACT, annotation_map, dynamic_annotations, match_sources_to_annotations, world_points_to_t0
from real_motion.msp_wm_cache import MSPWorldModelCacheDataset
from real_motion.nuscenes_adapter import NuScenesWindowSource, WindowTokens
from real_motion.rigid_transport import compose_component_replacements, rasterize_rigid_component
from real_motion.runtime_config import add_config_args, load_runtime_config, make_prepare_config
from real_motion.strong_w2det import StrongW2DetConfig, extract_instances, match_instances
from tools.real_motion import eval_p0_f9_frozen_sparse_occfm as safe

PROTOCOL = "p0_f9_v16_local_stwm_rigid_transport_eval_v1"
VARIANTS = ("strong_anchor", "local_stwm_center_always", "local_stwm_center_rigid", "gt_center_rigid")


def load_cache(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if obj.get("version") != LOCAL_STWM_CACHE_VERSION:
        raise RuntimeError("local STWM cache version mismatch")
    meta = obj.get("metadata") or {}
    if meta.get("target_contract") != TARGET_CONTRACT:
        raise RuntimeError("target contract mismatch")
    if meta.get("local_tube_contract") != LOCAL_TUBE_CONTRACT:
        raise RuntimeError("local tube contract mismatch")
    records = obj.get("records") or []
    if not records:
        raise RuntimeError("local STWM cache has no records")
    return meta, records


def load_model(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("protocol") != MODEL_PROTOCOL:
        raise RuntimeError(f"checkpoint protocol mismatch: {ck.get('protocol')}")
    cfg = config_from_mapping(ck.get("model_config"))
    model = LocalSpatialTemporalWorldModel(cfg).to(device)
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


def main():
    p = argparse.ArgumentParser(); add_config_args(p)
    p.add_argument("--local-stwm-cache", required=True)
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

    cfg = load_runtime_config(a.config, a.override); pcfg = make_prepare_config(cfg)
    cache_meta, records = load_cache(a.local_stwm_cache)
    ds = MSPWorldModelCacheDataset(a.p0f9_cache)
    sid_to_idx = {str(e["sample_id"]): i for i, e in enumerate(ds.entries)}
    missing = [str(r["sample_id"]) for r in records if str(r["sample_id"]) not in sid_to_idx]
    if missing: raise RuntimeError(f"P0-F9 cache misses local-STWM samples: {missing[:5]}")
    if int(a.max_windows) > 0: records = records[:min(len(records), int(a.max_windows))]

    device = torch.device(a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu")
    ck, model = load_model(a.checkpoint, device)
    source = NuScenesWindowSource(a.dataroot, info_pkl=a.info_pkl, verbose=False)
    strong_cfg = StrongW2DetConfig(free_label=int(pcfg.free_label))
    states = {name: safe._new_metrics() for name in VARIANTS}

    kta_errors = []; learned_errors = []; kta_fde = []; learned_fde = []
    exist_tp = exist_fp = exist_fn = exist_correct = exist_total = 0
    source_count = supervised_count = 0

    for wi, rec in enumerate(records):
        sid = str(rec["sample_id"]); sample = ds[sid_to_idx[sid]]
        payload = safe._sample_payload(sample, torch.device("cpu")); w = window_from_record(rec)
        history_occ = [source.load_semantics(w.scene_name, tok) for tok in w.history_tokens]
        history_poses = [np.asarray(source.pose(tok), dtype=np.float64) for tok in w.history_tokens]
        future_poses = [np.asarray(source.pose(tok), dtype=np.float64) for tok in w.future_tokens]
        current = extract_instances(history_occ[-1], history_poses[-1], grid=pcfg.grid, cfg=strong_cfg)
        previous = extract_instances(history_occ[-2], history_poses[-2], grid=pcfg.grid, cfg=strong_cfg)
        velocities = match_instances(previous, current, float(pcfg.frame_dt_s), max_speed_mps=strong_cfg.max_match_speed_mps)
        if len(current) != int(rec["features"].shape[0]):
            raise RuntimeError(f"{sid}: reconstructed Strong source count differs from v16 cache")
        got_classes = [int(c["class_id"]) for c in current]
        if got_classes != [int(x) for x in rec["source_class_id"].tolist()]:
            raise RuntimeError(f"{sid}: reconstructed Strong source ordering differs from v16 cache")

        features = rec["features"].float().to(device)
        tube = rec["local_semantic_tube"].to(device)
        kta_disp = rec["kta_displacement_xy_m"].float().to(device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            out = model(features, tube, kta_disp)
        pred_residual = out["residual_xy_m"].float().cpu().numpy()
        exist_prob = torch.sigmoid(out["existence_logits"].float()).cpu().numpy()

        valid = rec["target_valid"].bool().numpy(); target_res = rec["target_residual_xy_m"].float().numpy(); supervised = rec["supervised_source"].bool().numpy()
        source_count += len(current); supervised_count += int(supervised.sum())
        for i in range(len(current)):
            ids = np.flatnonzero(valid[i])
            for h in ids:
                kta_errors.append(float(np.linalg.norm(target_res[i, h])))
                learned_errors.append(float(np.linalg.norm(pred_residual[i, h] - target_res[i, h])))
            if len(ids):
                h = int(ids[-1]); kta_fde.append(float(np.linalg.norm(target_res[i, h]))); learned_fde.append(float(np.linalg.norm(pred_residual[i, h] - target_res[i, h])))
            if supervised[i]:
                gt_e = rec["existence"][i].numpy() > 0.5; pr_e = exist_prob[i] >= float(a.existence_threshold)
                exist_correct += int((gt_e == pr_e).sum()); exist_total += FUTURE_FRAMES
                exist_tp += int((gt_e & pr_e).sum()); exist_fp += int((~gt_e & pr_e).sum()); exist_fn += int((gt_e & ~pr_e).sum())

        anns0 = dynamic_annotations(source.nusc, w.t0_token)
        source_tokens = match_sources_to_annotations(current, anns0, max_distance_m=float(a.match_max_distance_m))
        future_maps = [annotation_map(source.nusc, tok) for tok in w.future_tokens]
        ann0_by_token = {str(x["instance_token"]): x for x in anns0}; t0_pose = history_poses[-1]

        for horizon, hi in safe.REPORT.items():
            gt = payload["gt"][hi]; anchor = payload["anchor"][hi]; moving = payload["moving"][hi]
            safe._update(states["strong_anchor"], float(horizon), anchor, gt, moving)
            baseline_all = []; learned_all = []; learned_exist = []; oracle_baseline = []; oracle_repl = []
            dt = (hi + 1) * float(pcfg.frame_dt_s)
            for i, comp in enumerate(current):
                src_center = np.asarray(comp["centroid_world"], dtype=np.float64)
                v = np.asarray(velocities.get(i, np.zeros(3)), dtype=np.float64)
                kta_center = src_center + v * dt
                base = rasterize_rigid_component(
                    comp["voxel_indices"], int(comp["class_id"]), t0_pose, future_poses[hi],
                    source_center_world=src_center, target_center_world=kta_center, yaw_delta_rad=0.0, grid=pcfg.grid,
                )
                baseline_all.append(base)
                xy_pred = rec["anchors_xy_t0_m"][i, hi].numpy() + pred_residual[i, hi]
                center_pred = t0_xy_to_world_preserve_source_z(xy_pred, src_center, t0_pose)
                repl = rasterize_rigid_component(
                    comp["voxel_indices"], int(comp["class_id"]), t0_pose, future_poses[hi],
                    source_center_world=src_center, target_center_world=center_pred, yaw_delta_rad=0.0, grid=pcfg.grid,
                )
                learned_all.append(repl)
                if exist_prob[i, hi] >= float(a.existence_threshold): learned_exist.append(repl)

                token = source_tokens[i]
                if token is not None:
                    oracle_baseline.append(base)
                    ann0 = ann0_by_token.get(str(token)); annh = future_maps[hi].get(str(token))
                    if ann0 is not None and annh is not None:
                        gt_disp_world = np.asarray(annh["center_world"], dtype=np.float64) - np.asarray(ann0["center_world"], dtype=np.float64)
                        target_center = src_center + gt_disp_world
                        oracle_repl.append(rasterize_rigid_component(
                            comp["voxel_indices"], int(comp["class_id"]), t0_pose, future_poses[hi],
                            source_center_world=src_center, target_center_world=target_center, yaw_delta_rad=0.0, grid=pcfg.grid,
                        ))

            pred_always = compose_component_replacements(anchor, baseline_all, learned_all, dynamic_class_ids=DYNAMIC_CLASS_IDS, free_label=int(pcfg.free_label), grid=pcfg.grid)
            pred_exist = compose_component_replacements(anchor, baseline_all, learned_exist, dynamic_class_ids=DYNAMIC_CLASS_IDS, free_label=int(pcfg.free_label), grid=pcfg.grid)
            pred_oracle = compose_component_replacements(anchor, oracle_baseline, oracle_repl, dynamic_class_ids=DYNAMIC_CLASS_IDS, free_label=int(pcfg.free_label), grid=pcfg.grid)
            safe._update(states["local_stwm_center_always"], float(horizon), pred_always, gt, moving)
            safe._update(states["local_stwm_center_rigid"], float(horizon), pred_exist, gt, moving)
            safe._update(states["gt_center_rigid"], float(horizon), pred_oracle, gt, moving)

        if wi == 0 or (wi + 1) % 8 == 0 or wi + 1 == len(records):
            print(f"local_stwm_eval {wi+1}/{len(records)} sid={sid} sources={len(current)}")

    reports = {name: safe._report(state) for name, state in states.items()}
    so, sm = metric_pair(reports["strong_anchor"]); _, gm = metric_pair(reports["gt_center_rigid"]); _, lm = metric_pair(reports["local_stwm_center_rigid"])
    denom = gm - sm; recovery = (lm - sm) / denom if abs(denom) > 1e-12 else float("nan")
    prec = exist_tp / max(exist_tp + exist_fp, 1); rec_e = exist_tp / max(exist_tp + exist_fn, 1)
    diagnostics = {
        "source_count": source_count, "supervised_source_count": supervised_count, "source_supervision_fraction": supervised_count / max(source_count, 1),
        "kta_ade_m": float(np.mean(kta_errors)) if kta_errors else float("nan"), "learned_ade_m": float(np.mean(learned_errors)) if learned_errors else float("nan"),
        "kta_fde_m": float(np.mean(kta_fde)) if kta_fde else float("nan"), "learned_fde_m": float(np.mean(learned_fde)) if learned_fde else float("nan"),
        "existence_accuracy": exist_correct / max(exist_total, 1), "existence_precision": prec, "existence_recall": rec_e,
        "existence_f1": 2 * prec * rec_e / max(prec + rec_e, 1e-12), "moving_oracle_recovery_fraction": recovery,
    }

    print("\n=== P0-F9 LOCAL SPATIAL-TEMPORAL WORLD MODEL ===")
    print(f"{'variant':28s} {'Overall':>9s} {'Moving':>9s} {'dOverall':>10s} {'dMoving':>9s}")
    for name in VARIANTS:
        o, m = metric_pair(reports[name]); print(f"{name:28s} {o:9.4f} {m:9.4f} {o-so:+10.4f} {m-sm:+9.4f}")
    print("\n=== MOVING BY HORIZON ===")
    for name in VARIANTS:
        vals = reports[name]["moving"]["per_horizon"]; row = []
        for h in safe.REPORT:
            v = vals[float(h)] if float(h) in vals else vals[str(float(h))]; row.append(float(v["mIoU"]))
        print(f"{name:28s} 1s={row[0]:.4f} 2s={row[1]:.4f} 3s={row[2]:.4f}")
    print("\n=== TRAJECTORY / EXISTENCE DIAGNOSTICS ===")
    for k, v in diagnostics.items(): print(f"{k}={v}")

    result = {
        "protocol": PROTOCOL,
        "checkpoint": str(Path(a.checkpoint).resolve()), "checkpoint_epoch": int(ck.get("epoch", -1)),
        "local_stwm_cache": str(Path(a.local_stwm_cache).resolve()), "p0f9_cache": str(Path(a.p0f9_cache).resolve()),
        "existence_threshold": float(a.existence_threshold), "num_windows": len(records),
        "reports": reports, "diagnostics": diagnostics, "model_config": ck.get("model_config"),
        "gt_center_oracle_contract": "translate_source_component_by_gt_annotation_displacement_v11_compatible",
        "target_contract": TARGET_CONTRACT, "local_tube_contract": LOCAL_TUBE_CONTRACT,
        "cache_metadata": cache_meta,
    }
    op = Path(a.output); op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(result, indent=2), encoding="utf-8"); print(f"saved {op}")


if __name__ == "__main__":
    main()
