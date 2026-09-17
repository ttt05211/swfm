#!/usr/bin/env python3
"""Expanded scene-disjoint validation for frozen V18 checkpoints.

This evaluator is a confirmation-only protocol.  It compares the frozen
Strong-W2Det/KTA anchor, historical Y600 V18 checkpoint, and clean one-stage
V18 checkpoint on one identical validation record stream.

Unlike the development evaluator, no P0-F9 latent cache is required.  The
compact evaluation payload is reconstructed from raw nuScenes with the same
frozen contracts used to create that cache:

- GT future occupancy from the exact 6+6 window;
- Strong-W2Det occupancy anchor from causal history only;
- Moving-mIoU v2 support from ``gt_moving_support_for_horizon``;
- original Strong source order and hard A1 component replacement compositor.

The intended input cache contains exactly one midpoint window per validation
scene.  A previous 128-scene development cache can be supplied so the report
separates those scenes from validation scenes never used by the earlier tuning
loop.  Paired bootstrap resamples *scenes* and recomputes metrics from summed
intersection/union counts; it never averages per-scene IoUs.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from real_motion.local_st_world_model_v17 import config_from_mapping_v17
from real_motion.local_st_world_model_v18_se2 import LocalSpatialTemporalWorldModelV18SE2
from real_motion.metrics.moving_miou_v2 import (
    DYNAMIC_CLASS_IDS,
    NUSCENES_LABELS,
    REPORT_HORIZONS_S,
)
from real_motion.nuscenes_adapter import gt_moving_support_for_horizon
from real_motion.prepared import load_nuscenes_window_raw
from real_motion.rigid_transport import (
    compose_component_replacements_in_input_order,
    rasterize_rigid_component,
)
from real_motion.runtime_config import add_config_args, load_runtime_config, make_prepare_config
from real_motion.strong_w2det import StrongW2DetConfig, extract_instances, match_instances, strong_w2det_sequence
from real_motion.v18_two_wheel_diagnostic import renderer_yaw_delta
from tools.real_motion import eval_p0_f9_v18_se2 as base
from tools.real_motion.eval_p0_f9_v17_local_stwm import (
    t0_xy_to_world_preserve_source_z,
    window_from_record,
)
from tools.real_motion.train_p0_f9_v18_se2_clean import PROTOCOL as CLEAN_PROTOCOL
from tools.real_motion.train_p0_f9_v18_se2_pair import PROTOCOL as PAIR_PROTOCOL


PROTOCOL = "p0_f9_v18_expanded_scene_disjoint_validation_v1"
VARIANTS = ("strong_anchor", "y600_pred", "clean_pred")
REPORT = {1.0: 1, 2.0: 3, 3.0: 5}
SEMANTIC_CLASSES = tuple(range(17))
BOOTSTRAP_SEED = 20260917


def _load_model(path: str, expected_protocol: str, device: torch.device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if str(ck.get("protocol")) != str(expected_protocol):
        raise RuntimeError(
            f"{path}: protocol={ck.get('protocol')!r}, expected {expected_protocol!r}"
        )
    if str(ck.get("arm")) != "Y":
        raise RuntimeError(f"{path}: expected arm=Y, got {ck.get('arm')!r}")
    cfg = config_from_mapping_v17(ck.get("model_config"))
    model = LocalSpatialTemporalWorldModelV18SE2(cfg).to(device)
    model.load_state_dict(ck["state_dict"], strict=True)
    model.eval()
    return ck, model, cfg


def _empty_scene_row():
    return {
        "occ_inter": 0,
        "occ_union": 0,
        "sem_inter": np.zeros(len(SEMANTIC_CLASSES), dtype=np.int64),
        "sem_union": np.zeros(len(SEMANTIC_CLASSES), dtype=np.int64),
        "mov_inter": np.zeros(len(DYNAMIC_CLASS_IDS), dtype=np.int64),
        "mov_union": np.zeros(len(DYNAMIC_CLASS_IDS), dtype=np.int64),
    }


def _count_row(pred, gt, moving, free_label: int):
    pred = np.asarray(pred)
    gt = np.asarray(gt)
    moving = np.asarray(moving, dtype=bool)
    if pred.shape != gt.shape or moving.shape != gt.shape:
        raise ValueError("metric shape mismatch")
    row = _empty_scene_row()
    po = pred != int(free_label)
    go = gt != int(free_label)
    row["occ_inter"] = int((po & go).sum())
    row["occ_union"] = int((po | go).sum())
    for j, cid in enumerate(SEMANTIC_CLASSES):
        p = pred == int(cid)
        g = gt == int(cid)
        row["sem_inter"][j] = int((p & g).sum())
        row["sem_union"][j] = int((p | g).sum())
    for j, cid in enumerate(DYNAMIC_CLASS_IDS):
        p = (pred == int(cid)) & moving
        g = (gt == int(cid)) & moving
        row["mov_inter"][j] = int((p & g).sum())
        row["mov_union"][j] = int((p | g).sum())
    return row


def _stack_scene_rows(scene_rows, scene_names):
    """Return variant -> raw arrays with axis 0 aligned to scene_names."""
    sidx = {str(s): i for i, s in enumerate(scene_names)}
    hidx = {float(h): i for i, h in enumerate(REPORT_HORIZONS_S)}
    out = {}
    for variant in VARIANTS:
        out[variant] = {
            "occ_inter": np.zeros((len(scene_names), 3), dtype=np.int64),
            "occ_union": np.zeros((len(scene_names), 3), dtype=np.int64),
            "sem_inter": np.zeros((len(scene_names), 3, len(SEMANTIC_CLASSES)), dtype=np.int64),
            "sem_union": np.zeros((len(scene_names), 3, len(SEMANTIC_CLASSES)), dtype=np.int64),
            "mov_inter": np.zeros((len(scene_names), 3, len(DYNAMIC_CLASS_IDS)), dtype=np.int64),
            "mov_union": np.zeros((len(scene_names), 3, len(DYNAMIC_CLASS_IDS)), dtype=np.int64),
        }
    for (scene, variant, horizon), row in scene_rows.items():
        i, h = sidx[str(scene)], hidx[float(horizon)]
        dst = out[variant]
        dst["occ_inter"][i, h] += int(row["occ_inter"])
        dst["occ_union"][i, h] += int(row["occ_union"])
        dst["sem_inter"][i, h] += row["sem_inter"]
        dst["sem_union"][i, h] += row["sem_union"]
        dst["mov_inter"][i, h] += row["mov_inter"]
        dst["mov_union"][i, h] += row["mov_union"]
    return out


def _safe_iou(inter, union):
    inter = np.asarray(inter, dtype=np.float64)
    union = np.asarray(union, dtype=np.float64)
    out = np.full(np.broadcast_shapes(inter.shape, union.shape), np.nan, dtype=np.float64)
    np.divide(inter, union, out=out, where=union > 0)
    return 100.0 * out


def _metric_from_raw(raw, indices=None):
    if indices is None:
        oi = raw["occ_inter"].sum(axis=0)
        ou = raw["occ_union"].sum(axis=0)
        si = raw["sem_inter"].sum(axis=0)
        su = raw["sem_union"].sum(axis=0)
        mi = raw["mov_inter"].sum(axis=0)
        mu = raw["mov_union"].sum(axis=0)
    else:
        idx = np.asarray(indices, dtype=np.int64)
        oi = raw["occ_inter"][idx].sum(axis=0)
        ou = raw["occ_union"][idx].sum(axis=0)
        si = raw["sem_inter"][idx].sum(axis=0)
        su = raw["sem_union"][idx].sum(axis=0)
        mi = raw["mov_inter"][idx].sum(axis=0)
        mu = raw["mov_union"][idx].sum(axis=0)

    occ_h = _safe_iou(oi, ou)
    sem_iou = _safe_iou(si, su)
    mov_iou = _safe_iou(mi, mu)
    sem_h = np.nanmean(sem_iou, axis=1)
    macro_h = np.nanmean(mov_iou, axis=1)
    micro_h = _safe_iou(mi.sum(axis=1), mu.sum(axis=1))

    per_horizon = {}
    for hi, h in enumerate(REPORT_HORIZONS_S):
        per_horizon[str(float(h))] = {
            "IoU": float(occ_h[hi]),
            "mIoU": float(sem_h[hi]),
            "MovingMacro": float(macro_h[hi]),
            "MovingMicro": float(micro_h[hi]),
            "moving_per_class": {
                str(cid): float(mov_iou[hi, j])
                for j, cid in enumerate(DYNAMIC_CLASS_IDS)
            },
            "moving_intersection": {
                str(cid): int(mi[hi, j]) for j, cid in enumerate(DYNAMIC_CLASS_IDS)
            },
            "moving_union": {
                str(cid): int(mu[hi, j]) for j, cid in enumerate(DYNAMIC_CLASS_IDS)
            },
        }
    return {
        "IoU": float(np.mean(occ_h)),
        "mIoU": float(np.mean(sem_h)),
        "MovingMacro": float(np.mean(macro_h)),
        "MovingMicro": float(np.mean(micro_h)),
        "per_horizon": per_horizon,
    }


def _subset_indices(all_scenes, subset_scenes):
    wanted = set(str(x) for x in subset_scenes)
    return np.asarray([i for i, s in enumerate(all_scenes) if str(s) in wanted], dtype=np.int64)


def _bootstrap_delta(raw_ref, raw_cand, scene_indices, *, samples: int, seed: int):
    idx = np.asarray(scene_indices, dtype=np.int64)
    if idx.size == 0:
        return None
    rng = np.random.default_rng(int(seed))
    keys = ("IoU", "mIoU", "MovingMacro", "MovingMicro")
    vals = {k: [] for k in keys}
    class_vals = {
        str(cid): {str(float(h)): [] for h in REPORT_HORIZONS_S}
        for cid in DYNAMIC_CLASS_IDS
    }
    for _ in range(int(samples)):
        draw = rng.choice(idx, size=idx.size, replace=True)
        rr = _metric_from_raw(raw_ref, draw)
        cc = _metric_from_raw(raw_cand, draw)
        for k in keys:
            d = float(cc[k]) - float(rr[k])
            if np.isfinite(d):
                vals[k].append(d)
        for cid in DYNAMIC_CLASS_IDS:
            for h in REPORT_HORIZONS_S:
                a = rr["per_horizon"][str(float(h))]["moving_per_class"][str(cid)]
                b = cc["per_horizon"][str(float(h))]["moving_per_class"][str(cid)]
                d = float(b) - float(a)
                if np.isfinite(d):
                    class_vals[str(cid)][str(float(h))].append(d)

    point_ref = _metric_from_raw(raw_ref, idx)
    point_cand = _metric_from_raw(raw_cand, idx)

    def summary(x, point):
        arr = np.asarray(x, dtype=np.float64)
        if arr.size == 0:
            return {"point_delta_pp": float(point), "valid_samples": 0,
                    "mean_pp": float("nan"), "p2_5_pp": float("nan"),
                    "p97_5_pp": float("nan")}
        return {
            "point_delta_pp": float(point),
            "valid_samples": int(arr.size),
            "mean_pp": float(arr.mean()),
            "p2_5_pp": float(np.quantile(arr, 0.025)),
            "p97_5_pp": float(np.quantile(arr, 0.975)),
        }

    out = {
        "samples": int(samples),
        "seed": int(seed),
        "num_scenes": int(idx.size),
        "aggregation": "paired_scene_resample_then_sum_raw_intersection_union",
        "metrics": {},
        "per_class_horizon": {},
    }
    for k in keys:
        out["metrics"][k] = summary(
            vals[k], float(point_cand[k]) - float(point_ref[k])
        )
    for cid in DYNAMIC_CLASS_IDS:
        out["per_class_horizon"][str(cid)] = {}
        for h in REPORT_HORIZONS_S:
            hs = str(float(h))
            a = point_ref["per_horizon"][hs]["moving_per_class"][str(cid)]
            b = point_cand["per_horizon"][hs]["moving_per_class"][str(cid)]
            point = float(b) - float(a) if np.isfinite(a) and np.isfinite(b) else float("nan")
            out["per_class_horizon"][str(cid)][hs] = summary(
                class_vals[str(cid)][hs], point
            )
    return out


def _load_dev_subset(path: str):
    if not path:
        return set(), set()
    _, records = base.load_cache(path)
    sample_ids = {str(r["sample_id"]) for r in records}
    scenes = {str(window_from_record(r).scene_name) for r in records}
    if len(sample_ids) != len(records) or len(scenes) != len(records):
        raise RuntimeError("development cache is not one-window-per-scene")
    return sample_ids, scenes


def main():
    p = argparse.ArgumentParser()
    add_config_args(p)
    p.add_argument("--full-val-cache", required=True,
                   help="SE2 cache containing one midpoint window for every validation scene")
    p.add_argument("--development-cache", default="",
                   help="previous 128-scene SE2 cache; used only to define seen/heldout scene subsets")
    p.add_argument("--y600-checkpoint", required=True)
    p.add_argument("--clean-checkpoint", required=True)
    p.add_argument("--dataroot", required=True)
    p.add_argument("--info-pkl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--bootstrap-samples", type=int, default=2000)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    cfg = load_runtime_config(a.config, a.override)
    pcfg = make_prepare_config(cfg)
    cache_meta, records = base.load_cache(a.full_val_cache)
    if not records:
        raise RuntimeError("full validation cache is empty")
    scenes = [str(window_from_record(r).scene_name) for r in records]
    sample_ids = [str(r["sample_id"]) for r in records]
    if len(set(sample_ids)) != len(records):
        raise RuntimeError("full validation cache contains duplicate sample IDs")
    if len(set(scenes)) != len(records):
        raise RuntimeError(
            "expanded confirmation protocol requires exactly one window per scene; "
            "do not pass an all-overlapping-window validation cache"
        )

    dev_ids, dev_scenes = _load_dev_subset(a.development_cache)
    if dev_ids:
        missing = sorted(dev_ids - set(sample_ids))
        if missing:
            raise RuntimeError(
                f"expanded full-val cache does not contain previous development samples: {missing[:5]}"
            )
    all_scenes = tuple(scenes)
    full_scene_set = set(all_scenes)
    heldout_scenes = full_scene_set - dev_scenes if dev_scenes else set()

    device = torch.device(a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu")
    y_ck, y_model, y_cfg = _load_model(a.y600_checkpoint, PAIR_PROTOCOL, device)
    c_ck, c_model, c_cfg = _load_model(a.clean_checkpoint, CLEAN_PROTOCOL, device)
    if y_cfg != c_cfg:
        raise RuntimeError("Y600 and Clean model_config differ")

    source = base.NuScenesWindowSource(a.dataroot, info_pkl=a.info_pkl, verbose=False)
    strong_cfg = StrongW2DetConfig(free_label=int(pcfg.free_label))
    scene_rows = {}
    instance_counts = {
        str(cid): {str(float(h)): {"instances": 0, "scenes": set()} for h in REPORT_HORIZONS_S}
        for cid in DYNAMIC_CLASS_IDS
    }

    for wi, rec in enumerate(records):
        sid = str(rec["sample_id"])
        w = window_from_record(rec)
        scene = str(w.scene_name)
        raw = load_nuscenes_window_raw(source, w, pcfg, include_gt=True)
        history_occ = np.asarray(raw["history_occ"], dtype=np.uint8)
        history_poses = [np.asarray(x, dtype=np.float64) for x in raw["history_poses"]]
        future_poses = [np.asarray(x, dtype=np.float64) for x in raw["future_poses"]]
        future_gt = np.asarray(raw["future_gt_occ"], dtype=np.uint8)
        anchor_all = strong_w2det_sequence(
            history_occ,
            history_poses,
            future_poses,
            frame_dt_s=float(pcfg.frame_dt_s),
            grid=pcfg.grid,
            cfg=strong_cfg,
        ).astype(np.uint8, copy=False)

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
        if len(current) != int(rec["features"].shape[0]):
            raise RuntimeError(f"{sid}: Strong/source count mismatch")
        current_classes = [int(c["class_id"]) for c in current]
        cached_classes = [int(x) for x in rec["source_class_id"].tolist()]
        if current_classes != cached_classes:
            raise RuntimeError(f"{sid}: Strong/source class order mismatch")

        nsrc = len(current)
        if nsrc:
            features = rec["features"].float().to(device)
            tube = rec["local_semantic_tube"].to(device)
            kta = rec["kta_displacement_xy_m"].float().to(device)
            frame_motion = rec["frame_motion_features"].float().to(device)
            source_mask = rec["target_source_mask_tube"].to(device)
            with torch.no_grad(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                yo = y_model(features, tube, kta, frame_motion, source_mask)
                co = c_model(features, tube, kta, frame_motion, source_mask)
            y_res = yo["residual_xy_m"].float().cpu().numpy()
            c_res = co["residual_xy_m"].float().cpu().numpy()
            y_yaw = yo["yaw_delta_rad"].float().cpu().numpy()
            c_yaw = co["yaw_delta_rad"].float().cpu().numpy()
        else:
            y_res = c_res = np.zeros((0, 6, 2), dtype=np.float32)
            y_yaw = c_yaw = np.zeros((0, 6), dtype=np.float32)

        t0_pose = history_poses[-1]
        for horizon, hi in REPORT.items():
            gt = future_gt[hi]
            anchor = anchor_all[hi]
            moving, moving_records, _ = gt_moving_support_for_horizon(
                source.nusc,
                w.t0_token,
                w.future_tokens[hi],
                float(horizon),
                grid=pcfg.grid,
            )
            for mr in moving_records:
                cid = int(mr["class_id"])
                if cid in DYNAMIC_CLASS_IDS:
                    entry = instance_counts[str(cid)][str(float(horizon))]
                    entry["instances"] += 1
                    entry["scenes"].add(scene)

            baseline = []
            y_components = []
            c_components = []
            dt = (hi + 1) * float(pcfg.frame_dt_s)
            for i, comp in enumerate(current):
                cid = int(comp["class_id"])
                src_center = np.asarray(comp["centroid_world"], dtype=np.float64)
                v = np.asarray(velocities.get(i, np.zeros(3)), dtype=np.float64)
                baseline.append(rasterize_rigid_component(
                    comp["voxel_indices"], cid, t0_pose, future_poses[hi],
                    source_center_world=src_center,
                    target_center_world=src_center + v * dt,
                    yaw_delta_rad=0.0,
                    grid=pcfg.grid,
                ))

                y_xy = rec["anchors_xy_t0_m"][i, hi].numpy() + y_res[i, hi]
                c_xy = rec["anchors_xy_t0_m"][i, hi].numpy() + c_res[i, hi]
                y_center = t0_xy_to_world_preserve_source_z(y_xy, src_center, t0_pose)
                c_center = t0_xy_to_world_preserve_source_z(c_xy, src_center, t0_pose)
                y_components.append(rasterize_rigid_component(
                    comp["voxel_indices"], cid, t0_pose, future_poses[hi],
                    source_center_world=src_center,
                    target_center_world=y_center,
                    yaw_delta_rad=renderer_yaw_delta(
                        cid, y_yaw[i, hi], zero_two_wheel_yaw=False
                    ),
                    grid=pcfg.grid,
                ))
                c_components.append(rasterize_rigid_component(
                    comp["voxel_indices"], cid, t0_pose, future_poses[hi],
                    source_center_world=src_center,
                    target_center_world=c_center,
                    yaw_delta_rad=renderer_yaw_delta(
                        cid, c_yaw[i, hi], zero_two_wheel_yaw=False
                    ),
                    grid=pcfg.grid,
                ))

            y_pred = compose_component_replacements_in_input_order(
                anchor, baseline, y_components,
                dynamic_class_ids=DYNAMIC_CLASS_IDS,
                free_label=int(pcfg.free_label), grid=pcfg.grid,
            )
            c_pred = compose_component_replacements_in_input_order(
                anchor, baseline, c_components,
                dynamic_class_ids=DYNAMIC_CLASS_IDS,
                free_label=int(pcfg.free_label), grid=pcfg.grid,
            )
            preds = {
                "strong_anchor": anchor,
                "y600_pred": y_pred,
                "clean_pred": c_pred,
            }
            for name, pred in preds.items():
                scene_rows[(scene, name, float(horizon))] = _count_row(
                    pred, gt, moving, int(pcfg.free_label)
                )

        if wi == 0 or (wi + 1) % 16 == 0 or wi + 1 == len(records):
            print(f"expanded_val {wi+1}/{len(records)} {sid}", flush=True)

    raw_by_variant = _stack_scene_rows(scene_rows, all_scenes)
    subsets = {"full": set(all_scenes)}
    if dev_scenes:
        subsets["development_seen"] = set(dev_scenes)
        subsets["heldout_scenes"] = set(heldout_scenes)

    report_subsets = {}
    bootstrap_subsets = {}
    for subset_name, subset_scene_set in subsets.items():
        idx = _subset_indices(all_scenes, subset_scene_set)
        report_subsets[subset_name] = {
            "num_scenes": int(idx.size),
            "scene_names": sorted(str(x) for x in subset_scene_set),
            "variants": {
                name: _metric_from_raw(raw_by_variant[name], idx)
                for name in VARIANTS
            },
        }
        bootstrap_subsets[subset_name] = _bootstrap_delta(
            raw_by_variant["y600_pred"],
            raw_by_variant["clean_pred"],
            idx,
            samples=int(a.bootstrap_samples),
            seed=BOOTSTRAP_SEED,
        )

    instance_json = {}
    for cid in DYNAMIC_CLASS_IDS:
        instance_json[str(cid)] = {}
        for h in REPORT_HORIZONS_S:
            e = instance_counts[str(cid)][str(float(h))]
            instance_json[str(cid)][str(float(h))] = {
                "class_name": NUSCENES_LABELS[int(cid)],
                "instances": int(e["instances"]),
                "scenes": int(len(e["scenes"])),
            }

    result = {
        "protocol": PROTOCOL,
        "status": "completed_evaluation",
        "full_val_cache": str(Path(a.full_val_cache).resolve()),
        "development_cache": str(Path(a.development_cache).resolve()) if a.development_cache else None,
        "cache_metadata": cache_meta,
        "num_windows": len(records),
        "num_scenes": len(all_scenes),
        "one_window_per_scene": True,
        "report_horizons_s": list(REPORT_HORIZONS_S),
        "dynamic_class_ids": list(DYNAMIC_CLASS_IDS),
        "y600_checkpoint": str(Path(a.y600_checkpoint).resolve()),
        "clean_checkpoint": str(Path(a.clean_checkpoint).resolve()),
        "y600_checkpoint_protocol": y_ck.get("protocol"),
        "clean_checkpoint_protocol": c_ck.get("protocol"),
        "y600_step": int(y_ck.get("continuation_step", -1)),
        "clean_epoch": int(c_ck.get("epoch", -1)),
        "clean_global_step": int(c_ck.get("global_step", -1)),
        "hard_compositor": "legacy_clear_plus_original_strong_source_write_order_v1",
        "yaw_mode": "pred",
        "moving_support": "interval_displacement_v2_gt_moving_support_for_horizon",
        "subsets": report_subsets,
        "moving_instance_counts_full": instance_json,
        "paired_bootstrap_clean_minus_y600": bootstrap_subsets,
        "notes": {
            "bootstrap_unit": "scene",
            "bootstrap_reaggregation": "raw intersections/unions are summed after each paired scene resample",
            "scene_iou_is_not_averaged": True,
            "no_training": True,
            "development_seen_definition": "scene/sample IDs in the supplied previous development cache",
            "heldout_definition": "full validation scenes absent from the supplied previous development cache",
        },
    }
    op = Path(a.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n=== EXPANDED SCENE-DISJOINT VALIDATION ===")
    for subset_name in ("full", "development_seen", "heldout_scenes"):
        if subset_name not in report_subsets:
            continue
        rr = report_subsets[subset_name]
        print(f"\n[{subset_name}] scenes={rr['num_scenes']}")
        print(f"{'variant':18s} {'IoU':>9s} {'mIoU':>9s} {'MacroMov':>10s} {'MicroMov':>10s}")
        for name in VARIANTS:
            m = rr["variants"][name]
            print(
                f"{name:18s} {m['IoU']:9.4f} {m['mIoU']:9.4f} "
                f"{m['MovingMacro']:10.4f} {m['MovingMicro']:10.4f}"
            )
        boot = bootstrap_subsets.get(subset_name)
        if boot:
            print("Clean-Y600 paired scene bootstrap:")
            for k in ("IoU", "mIoU", "MovingMacro", "MovingMicro"):
                b = boot["metrics"][k]
                print(
                    f"  {k:11s} point={b['point_delta_pp']:+.4f} "
                    f"95%CI=[{b['p2_5_pp']:+.4f},{b['p97_5_pp']:+.4f}]"
                )

    print("\n=== FULL-VAL MOVING PER CLASS x HORIZON ===")
    full = report_subsets["full"]["variants"]
    for cid in DYNAMIC_CLASS_IDS:
        name = NUSCENES_LABELS[int(cid)]
        meta = instance_json[str(cid)]
        for h in REPORT_HORIZONS_S:
            hs = str(float(h))
            y = full["y600_pred"]["per_horizon"][hs]["moving_per_class"][str(cid)]
            c = full["clean_pred"]["per_horizon"][hs]["moving_per_class"][str(cid)]
            b = bootstrap_subsets["full"]["per_class_horizon"][str(cid)][hs]
            print(
                f"{name:22s} {h:.1f}s Y600={y:8.3f} Clean={c:8.3f} "
                f"d={c-y:+8.3f} inst={meta[hs]['instances']:4d} "
                f"scenes={meta[hs]['scenes']:3d} "
                f"CI=[{b['p2_5_pp']:+.3f},{b['p97_5_pp']:+.3f}]"
            )
    print(f"saved {op}")


if __name__ == "__main__":
    main()
