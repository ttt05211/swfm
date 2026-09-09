#!/usr/bin/env python3
"""Diagnose why V16-STWM improves trajectory error but loses Moving-mIoU vs v13 MLP.

No training is performed.  The diagnostic evaluates the frozen v13 MLP and the
frozen V16 local spatial-temporal world model on the same 128-window cache and
asks four concrete questions:

1. On exact Moving-mIoU-v2 true-moving observations, where do MLP/STWM errors lie
   relative to the GT motion direction (along-track versus cross-track)?
2. Does the ranking reverse when trajectory errors are weighted by transported
   source voxel count, which is closer to occupancy-overlap importance?
3. Which horizon, semantic class, and source-size regimes favor MLP or STWM?
4. How much occupancy headroom exists if a GT-only selector can choose KTA, MLP,
   or STWM per source/horizon?

The script preserves the v13 displacement contract and rigid source-shape
transport.  It does not touch MT-V1-STPN code or checkpoints.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
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
    MODEL_PROTOCOL as STWM_PROTOCOL,
    LocalSpatialTemporalWorldModel,
    config_from_mapping,
)
from real_motion.local_stwm_gap import (
    kta_error_components,
    pairwise_winner,
    residual_error_components,
    three_way_winner,
)
from real_motion.metrics.moving_miou_v2 import (
    DYNAMIC_CLASS_IDS,
    NUSCENES_LABELS,
    SPEED_THRESHOLD_MPS,
)
from real_motion.motion_transport_gap import true_moving_observation_mask
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

PROTOCOL = "p0_f9_v16_vs_v13_mlp_geometric_gap_v1"
MLP_PROTOCOL = "p0_f9_v13_learned_motion_transport_v2"
VARIANTS = (
    "strong_anchor",
    "mlp_center_always",
    "stwm_center_always",
    "kta_mlp_selector_oracle",
    "kta_stwm_selector_oracle",
    "mlp_stwm_selector_oracle",
    "kta_mlp_stwm_selector_oracle",
    "gt_center_rigid",
)
MODELS = ("kta", "mlp", "stwm")


def load_local_cache(path: str):
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


def load_mlp(path: str, device: torch.device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("protocol") != MLP_PROTOCOL:
        raise RuntimeError(f"MLP checkpoint protocol mismatch: {ck.get('protocol')}")
    if int(ck.get("feature_dim", -1)) != FEATURE_DIM:
        raise RuntimeError("MLP feature dimension mismatch")
    model = MotionTransportHead(
        feature_dim=FEATURE_DIM,
        hidden_dim=int(ck.get("hidden_dim", 128)),
        future_frames=int(ck.get("future_frames", FUTURE_FRAMES)),
    ).to(device)
    model.load_state_dict(ck["state_dict"], strict=True)
    model.eval()
    return ck, model


def load_stwm(path: str, device: torch.device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("protocol") != STWM_PROTOCOL:
        raise RuntimeError(f"STWM checkpoint protocol mismatch: {ck.get('protocol')}")
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


def _mean(rows, key, *, weighted=False):
    if not rows:
        return float("nan")
    x = np.asarray([float(r[key]) for r in rows], dtype=np.float64)
    if not weighted:
        return float(x.mean())
    w = np.asarray([float(r["voxel_count"]) for r in rows], dtype=np.float64)
    return float(np.sum(x * w) / np.sum(w)) if float(w.sum()) > 0 else float("nan")


def summarize_rows(rows):
    out = {"count": len(rows)}
    for model in MODELS:
        out[model] = {
            "ade_m": _mean(rows, f"{model}_error_m"),
            "voxel_weighted_ade_m": _mean(rows, f"{model}_error_m", weighted=True),
            "along_abs_m": _mean(rows, f"{model}_along_abs_m"),
            "cross_abs_m": _mean(rows, f"{model}_cross_abs_m"),
            "voxel_weighted_along_abs_m": _mean(rows, f"{model}_along_abs_m", weighted=True),
            "voxel_weighted_cross_abs_m": _mean(rows, f"{model}_cross_abs_m", weighted=True),
            "along_signed_bias_m": _mean(rows, f"{model}_along_signed_m"),
            "cross_signed_bias_m": _mean(rows, f"{model}_cross_signed_m"),
        }
    if rows:
        out["mlp_beats_stwm_fraction"] = float(np.mean([
            float(r["mlp_error_m"]) < float(r["stwm_error_m"]) for r in rows
        ]))
        out["stwm_beats_mlp_fraction"] = float(np.mean([
            float(r["stwm_error_m"]) < float(r["mlp_error_m"]) for r in rows
        ]))
        winners = [int(r["three_way_winner"]) for r in rows]
        out["three_way_winner_fraction"] = {
            "kta": float(np.mean(np.asarray(winners) == 0)),
            "mlp": float(np.mean(np.asarray(winners) == 1)),
            "stwm": float(np.mean(np.asarray(winners) == 2)),
        }
    else:
        out["mlp_beats_stwm_fraction"] = float("nan")
        out["stwm_beats_mlp_fraction"] = float("nan")
        out["three_way_winner_fraction"] = {k: float("nan") for k in MODELS}
    return out


def latest_fde(rows):
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["sample_id"], int(r["source_index"]))].append(r)
    picked = []
    for group in grouped.values():
        picked.append(max(group, key=lambda x: int(x["horizon_index"])))
    return {
        "count": len(picked),
        **{f"{m}_fde_m": _mean(picked, f"{m}_error_m") for m in MODELS},
    }


def size_report(rows):
    if not rows:
        return {"edges_voxels": [], "buckets": {}}
    sizes = np.asarray([float(r["voxel_count"]) for r in rows], dtype=np.float64)
    q1, q2 = [float(x) for x in np.quantile(sizes, [1 / 3, 2 / 3])]
    buckets = {
        "small": [r for r in rows if float(r["voxel_count"]) <= q1],
        "medium": [r for r in rows if q1 < float(r["voxel_count"]) <= q2],
        "large": [r for r in rows if float(r["voxel_count"]) > q2],
    }
    return {
        "edges_voxels": [q1, q2],
        "buckets": {name: summarize_rows(group) for name, group in buckets.items()},
    }


def main():
    p = argparse.ArgumentParser()
    add_config_args(p)
    p.add_argument("--local-stwm-cache", required=True)
    p.add_argument("--p0f9-cache", required=True)
    p.add_argument("--mlp-checkpoint", required=True)
    p.add_argument("--stwm-checkpoint", required=True)
    p.add_argument("--dataroot", required=True)
    p.add_argument("--info-pkl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--match-max-distance-m", type=float, default=4.0)
    p.add_argument("--max-windows", type=int, default=0)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    cfg = load_runtime_config(a.config, a.override)
    pcfg = make_prepare_config(cfg)
    cache_meta, records = load_local_cache(a.local_stwm_cache)
    if int(a.max_windows) > 0:
        records = records[: min(len(records), int(a.max_windows))]

    ds = MSPWorldModelCacheDataset(a.p0f9_cache)
    sid_to_idx = {str(e["sample_id"]): i for i, e in enumerate(ds.entries)}
    missing = [str(r["sample_id"]) for r in records if str(r["sample_id"]) not in sid_to_idx]
    if missing:
        raise RuntimeError(f"P0-F9 cache misses local-STWM samples: {missing[:5]}")

    device = torch.device(a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu")
    mlp_ck, mlp = load_mlp(a.mlp_checkpoint, device)
    stwm_ck, stwm = load_stwm(a.stwm_checkpoint, device)
    source = NuScenesWindowSource(a.dataroot, info_pkl=a.info_pkl, verbose=False)
    strong_cfg = StrongW2DetConfig(free_label=int(pcfg.free_label))
    states = {name: safe._new_metrics() for name in VARIANTS}

    rows = []
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
            raise RuntimeError(f"{sid}: reconstructed Strong source count differs from V16 cache")
        got_classes = [int(c["class_id"]) for c in current]
        cached_classes = [int(x) for x in rec["source_class_id"].tolist()]
        if got_classes != cached_classes:
            raise RuntimeError(f"{sid}: reconstructed Strong source ordering differs from V16 cache")

        features = rec["features"].float().to(device)
        tube = rec["local_semantic_tube"].to(device)
        kta_disp = rec["kta_displacement_xy_m"].float().to(device)
        with torch.no_grad():
            mlp_out = mlp(features)
            mlp_res = mlp_out["residual_xy_m"].float().cpu().numpy()
        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")
        ):
            stwm_out = stwm(features, tube, kta_disp)
        stwm_res = stwm_out["residual_xy_m"].float().cpu().numpy()

        valid = rec["target_valid"].bool().numpy()
        target_res = rec["target_residual_xy_m"].float().numpy()
        target_disp = rec["target_displacement_xy_m"].float().numpy()
        voxels = rec["source_voxel_count"].cpu().numpy().astype(np.float64)
        supervised = rec["supervised_source"].bool().numpy()
        true_moving = true_moving_observation_mask(
            target_disp,
            valid,
            frame_dt_s=float(pcfg.frame_dt_s),
            speed_threshold_mps=SPEED_THRESHOLD_MPS,
        )

        comps = {
            "kta": kta_error_components(target_res, target_disp),
            "mlp": residual_error_components(mlp_res, target_res, target_disp),
            "stwm": residual_error_components(stwm_res, target_res, target_disp),
        }
        three = three_way_winner(
            comps["kta"]["euclidean_m"],
            comps["mlp"]["euclidean_m"],
            comps["stwm"]["euclidean_m"],
            valid,
        )
        pair = pairwise_winner(
            comps["mlp"]["euclidean_m"], comps["stwm"]["euclidean_m"], valid
        )
        source_count += len(current)
        supervised_count += int(supervised.sum())

        for i in range(len(current)):
            for hi in np.flatnonzero(valid[i]):
                row = {
                    "sample_id": sid,
                    "source_index": int(i),
                    "horizon_index": int(hi),
                    "horizon_s": float((hi + 1) * pcfg.frame_dt_s),
                    "class_id": int(rec["source_class_id"][i]),
                    "voxel_count": int(voxels[i]),
                    "true_moving": bool(true_moving[i, hi]),
                    "three_way_winner": int(three[i, hi]),
                    "mlp_stwm_winner": int(pair[i, hi]),
                }
                for model in MODELS:
                    row[f"{model}_error_m"] = float(comps[model]["euclidean_m"][i, hi])
                    row[f"{model}_along_abs_m"] = float(comps[model]["along_abs_m"][i, hi])
                    row[f"{model}_cross_abs_m"] = float(comps[model]["cross_abs_m"][i, hi])
                    row[f"{model}_along_signed_m"] = float(comps[model]["along_signed_m"][i, hi])
                    row[f"{model}_cross_signed_m"] = float(comps[model]["cross_signed_m"][i, hi])
                rows.append(row)

        anns0 = dynamic_annotations(source.nusc, w.t0_token)
        source_tokens = match_sources_to_annotations(
            current, anns0, max_distance_m=float(a.match_max_distance_m)
        )
        future_maps = [annotation_map(source.nusc, tok) for tok in w.future_tokens]
        ann0_by_token = {str(x["instance_token"]): x for x in anns0}
        t0_pose = history_poses[-1]

        for horizon, hi in safe.REPORT.items():
            gt = payload["gt"][hi]
            anchor = payload["anchor"][hi]
            moving = payload["moving"][hi]
            safe._update(states["strong_anchor"], float(horizon), anchor, gt, moving)

            baseline_all = []
            mlp_all = []
            stwm_all = []
            kta_mlp_base, kta_mlp_repl = [], []
            kta_stwm_base, kta_stwm_repl = [], []
            mlp_stwm_base, mlp_stwm_repl = [], []
            three_base, three_repl = [], []
            gt_base, gt_repl = [], []
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

                def learned_raster(residual):
                    xy = rec["anchors_xy_t0_m"][i, hi].numpy() + residual[i, hi]
                    center = t0_xy_to_world_preserve_source_z(xy, src_center, t0_pose)
                    return rasterize_rigid_component(
                        comp["voxel_indices"], int(comp["class_id"]), t0_pose, future_poses[hi],
                        source_center_world=src_center, target_center_world=center,
                        yaw_delta_rad=0.0, grid=pcfg.grid,
                    )

                mr = learned_raster(mlp_res)
                sr = learned_raster(stwm_res)
                mlp_all.append(mr)
                stwm_all.append(sr)

                if bool(valid[i, hi]):
                    ke = float(comps["kta"]["euclidean_m"][i, hi])
                    me = float(comps["mlp"]["euclidean_m"][i, hi])
                    se = float(comps["stwm"]["euclidean_m"][i, hi])
                    if me < ke:
                        kta_mlp_base.append(base); kta_mlp_repl.append(mr)
                    if se < ke:
                        kta_stwm_base.append(base); kta_stwm_repl.append(sr)
                    mlp_stwm_base.append(base)
                    mlp_stwm_repl.append(sr if se < me else mr)
                    winner = int(np.argmin([ke, me, se]))
                    if winner != 0:
                        three_base.append(base)
                        three_repl.append(mr if winner == 1 else sr)

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
                gt_base.append(base)
                gt_repl.append(rasterize_rigid_component(
                    comp["voxel_indices"], int(comp["class_id"]), t0_pose, future_poses[hi],
                    source_center_world=src_center, target_center_world=src_center + gt_disp_world,
                    yaw_delta_rad=0.0, grid=pcfg.grid,
                ))

            preds = {
                "mlp_center_always": compose_component_replacements(
                    anchor, baseline_all, mlp_all, dynamic_class_ids=DYNAMIC_CLASS_IDS,
                    free_label=int(pcfg.free_label), grid=pcfg.grid,
                ),
                "stwm_center_always": compose_component_replacements(
                    anchor, baseline_all, stwm_all, dynamic_class_ids=DYNAMIC_CLASS_IDS,
                    free_label=int(pcfg.free_label), grid=pcfg.grid,
                ),
                "kta_mlp_selector_oracle": compose_component_replacements(
                    anchor, kta_mlp_base, kta_mlp_repl, dynamic_class_ids=DYNAMIC_CLASS_IDS,
                    free_label=int(pcfg.free_label), grid=pcfg.grid,
                ),
                "kta_stwm_selector_oracle": compose_component_replacements(
                    anchor, kta_stwm_base, kta_stwm_repl, dynamic_class_ids=DYNAMIC_CLASS_IDS,
                    free_label=int(pcfg.free_label), grid=pcfg.grid,
                ),
                "mlp_stwm_selector_oracle": compose_component_replacements(
                    anchor, mlp_stwm_base, mlp_stwm_repl, dynamic_class_ids=DYNAMIC_CLASS_IDS,
                    free_label=int(pcfg.free_label), grid=pcfg.grid,
                ),
                "kta_mlp_stwm_selector_oracle": compose_component_replacements(
                    anchor, three_base, three_repl, dynamic_class_ids=DYNAMIC_CLASS_IDS,
                    free_label=int(pcfg.free_label), grid=pcfg.grid,
                ),
                "gt_center_rigid": compose_component_replacements(
                    anchor, gt_base, gt_repl, dynamic_class_ids=DYNAMIC_CLASS_IDS,
                    free_label=int(pcfg.free_label), grid=pcfg.grid,
                ),
            }
            for name, pred in preds.items():
                safe._update(states[name], float(horizon), pred, gt, moving)

        if wi == 0 or (wi + 1) % 8 == 0 or wi + 1 == len(records):
            print(f"v16_mlp_gap {wi+1}/{len(records)} sid={sid} sources={len(current)}")

    reports = {name: safe._report(state) for name, state in states.items()}
    all_rows = rows
    true_rows = [r for r in rows if bool(r["true_moving"])]
    all_summary = summarize_rows(all_rows)
    true_summary = summarize_rows(true_rows)
    all_summary["fde"] = latest_fde(all_rows)
    true_summary["fde"] = latest_fde(true_rows)
    true_summary["fraction_of_valid_observations"] = len(true_rows) / max(len(all_rows), 1)

    by_horizon = {}
    for h in safe.REPORT:
        group = [r for r in true_rows if abs(float(r["horizon_s"]) - float(h)) < 1e-9]
        by_horizon[str(float(h))] = summarize_rows(group)

    by_class = {}
    for c in DYNAMIC_CLASS_IDS:
        group = [r for r in true_rows if int(r["class_id"]) == int(c)]
        by_class[str(int(c))] = {
            "class_name": NUSCENES_LABELS[int(c)],
            **summarize_rows(group),
        }
    by_size = size_report(true_rows)

    so, sm = metric_pair(reports["strong_anchor"])
    occupancy = {name: {"overall": metric_pair(reports[name])[0], "moving": metric_pair(reports[name])[1]}
                 for name in VARIANTS}

    diagnostic = {
        "source_count": int(source_count),
        "supervised_source_count": int(supervised_count),
        "source_supervision_fraction": supervised_count / max(source_count, 1),
        "all_valid": all_summary,
        "true_moving": true_summary,
        "true_moving_by_report_horizon": by_horizon,
        "true_moving_by_class": by_class,
        "true_moving_by_source_size": by_size,
        "occupancy": occupancy,
    }

    print("\n=== V16 vs MLP: OCCUPANCY SELECTOR ORACLES ===")
    print(f"{'variant':31s} {'Overall':>9s} {'Moving':>9s} {'dOverall':>10s} {'dMoving':>9s}")
    for name in VARIANTS:
        o, m = metric_pair(reports[name])
        print(f"{name:31s} {o:9.4f} {m:9.4f} {o-so:+10.4f} {m-sm:+9.4f}")

    def print_summary(label, s):
        print(f"\n=== {label} ===")
        print(f"count={s['count']}")
        print(f"{'model':8s} {'ADE':>8s} {'voxADE':>8s} {'along':>8s} {'cross':>8s} {'vAlong':>8s} {'vCross':>8s}")
        for m in MODELS:
            x = s[m]
            print(
                f"{m:8s} {x['ade_m']:8.4f} {x['voxel_weighted_ade_m']:8.4f} "
                f"{x['along_abs_m']:8.4f} {x['cross_abs_m']:8.4f} "
                f"{x['voxel_weighted_along_abs_m']:8.4f} {x['voxel_weighted_cross_abs_m']:8.4f}"
            )
        print(
            f"MLP_vs_STWM win: MLP={100*s['mlp_beats_stwm_fraction']:.2f}% "
            f"STWM={100*s['stwm_beats_mlp_fraction']:.2f}%"
        )
        w = s["three_way_winner_fraction"]
        print(f"three-way winner: KTA={100*w['kta']:.2f}% MLP={100*w['mlp']:.2f}% STWM={100*w['stwm']:.2f}%")

    print_summary("V16 vs MLP: ALL VALID GEOMETRIC ERROR", all_summary)
    print_summary("V16 vs MLP: TRUE-MOVING GEOMETRIC ERROR", true_summary)
    print(
        f"true-moving FDE: KTA={true_summary['fde']['kta_fde_m']:.4f} "
        f"MLP={true_summary['fde']['mlp_fde_m']:.4f} STWM={true_summary['fde']['stwm_fde_m']:.4f}"
    )

    print("\n=== V16 vs MLP: TRUE-MOVING BY REPORT HORIZON ===")
    for h, s in by_horizon.items():
        print(
            f"{h}s n={s['count']:4d} "
            f"MLP ADE={s['mlp']['ade_m']:.4f} vox={s['mlp']['voxel_weighted_ade_m']:.4f} "
            f"cross={s['mlp']['cross_abs_m']:.4f} | "
            f"STWM ADE={s['stwm']['ade_m']:.4f} vox={s['stwm']['voxel_weighted_ade_m']:.4f} "
            f"cross={s['stwm']['cross_abs_m']:.4f}"
        )

    print("\n=== V16 vs MLP: TRUE-MOVING BY CLASS ===")
    for c in DYNAMIC_CLASS_IDS:
        s = by_class[str(int(c))]
        print(
            f"{s['class_name']:22s} n={s['count']:4d} "
            f"MLP={s['mlp']['ade_m']:.4f}/{s['mlp']['voxel_weighted_ade_m']:.4f} "
            f"STWM={s['stwm']['ade_m']:.4f}/{s['stwm']['voxel_weighted_ade_m']:.4f} "
            f"dADE={s['stwm']['ade_m']-s['mlp']['ade_m']:+.4f}"
        )

    print("\n=== V16 vs MLP: TRUE-MOVING BY SOURCE SIZE ===")
    print(f"tertile_edges_voxels={by_size['edges_voxels']}")
    for name, s in by_size["buckets"].items():
        print(
            f"{name:8s} n={s['count']:4d} "
            f"MLP={s['mlp']['ade_m']:.4f}/{s['mlp']['voxel_weighted_ade_m']:.4f} "
            f"STWM={s['stwm']['ade_m']:.4f}/{s['stwm']['voxel_weighted_ade_m']:.4f} "
            f"MLPcross={s['mlp']['cross_abs_m']:.4f} STWMcross={s['stwm']['cross_abs_m']:.4f}"
        )

    print("\n=== V16 vs MLP: KEY CONTRASTS ===")
    print(f"true-moving ADE STWM-MLP={true_summary['stwm']['ade_m']-true_summary['mlp']['ade_m']:+.4f} m")
    print(
        "true-moving voxel-weighted ADE STWM-MLP="
        f"{true_summary['stwm']['voxel_weighted_ade_m']-true_summary['mlp']['voxel_weighted_ade_m']:+.4f} m"
    )
    print(f"true-moving cross-track STWM-MLP={true_summary['stwm']['cross_abs_m']-true_summary['mlp']['cross_abs_m']:+.4f} m")
    print(
        "selector Moving: "
        f"KTA/MLP={occupancy['kta_mlp_selector_oracle']['moving']:.4f} "
        f"KTA/STWM={occupancy['kta_stwm_selector_oracle']['moving']:.4f} "
        f"MLP/STWM={occupancy['mlp_stwm_selector_oracle']['moving']:.4f} "
        f"KTA/MLP/STWM={occupancy['kta_mlp_stwm_selector_oracle']['moving']:.4f}"
    )

    result = {
        "protocol": PROTOCOL,
        "mlp_checkpoint": str(Path(a.mlp_checkpoint).resolve()),
        "mlp_checkpoint_epoch": int(mlp_ck.get("epoch", -1)),
        "stwm_checkpoint": str(Path(a.stwm_checkpoint).resolve()),
        "stwm_checkpoint_epoch": int(stwm_ck.get("epoch", -1)),
        "local_stwm_cache": str(Path(a.local_stwm_cache).resolve()),
        "p0f9_cache": str(Path(a.p0f9_cache).resolve()),
        "num_windows": len(records),
        "moving_speed_threshold_mps": SPEED_THRESHOLD_MPS,
        "target_contract": TARGET_CONTRACT,
        "local_tube_contract": LOCAL_TUBE_CONTRACT,
        "reports": reports,
        "diagnostic": diagnostic,
        "cache_metadata": cache_meta,
    }
    op = Path(a.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"saved {op}")


if __name__ == "__main__":
    main()
