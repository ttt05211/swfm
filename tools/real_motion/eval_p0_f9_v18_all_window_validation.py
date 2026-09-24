#!/usr/bin/env python3
"""Official-style all-window validation for frozen V18 motion checkpoints.

Evaluates Strong/KTA, historical Y600, and Clean-E14 on every eligible stride-1
6-history + 6-future window in the official nuScenes validation scene split.
Overlapping windows from the same scene are accumulated into one scene-level raw
intersection/union block before paired bootstrap; windows are never treated as
independent bootstrap units.

The SE2 cache supplies the exact V18 causal source features/representation for
each window.  GT future occupancy, Strong-W2Det anchor and frozen Moving-mIoU v2
support are reconstructed from raw nuScenes under the same contracts as the
midpoint evaluator.
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

from real_motion.metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS, NUSCENES_LABELS, REPORT_HORIZONS_S
from real_motion.nuscenes_adapter import gt_moving_support_for_horizon
from real_motion.prepared import load_nuscenes_window_raw
from real_motion.rigid_transport import compose_component_replacements_in_input_order, rasterize_rigid_component
from real_motion.runtime_config import add_config_args, load_runtime_config, make_prepare_config
from real_motion.strong_w2det import StrongW2DetConfig, extract_instances, match_instances, strong_w2det_sequence
from real_motion.v18_two_wheel_diagnostic import renderer_yaw_delta
from tools.real_motion import eval_p0_f9_v18_full_validation as mid
from tools.real_motion.eval_p0_f9_v17_local_stwm import t0_xy_to_world_preserve_source_z, window_from_record


PROTOCOL = "p0_f9_v18_all_window_validation_v1"


def _accumulate_count_row(dst: dict, src: dict) -> None:
    dst["occ_inter"] += int(src["occ_inter"])
    dst["occ_union"] += int(src["occ_union"])
    dst["sem_inter"] += src["sem_inter"]
    dst["sem_union"] += src["sem_union"]
    dst["mov_inter"] += src["mov_inter"]
    dst["mov_union"] += src["mov_union"]


def _scene_record_counts(records):
    out = {}
    for r in records:
        s = str(window_from_record(r).scene_name)
        out[s] = out.get(s, 0) + 1
    return out


def main():
    p = argparse.ArgumentParser()
    add_config_args(p)
    p.add_argument("--full-val-cache", required=True,
                   help="SE2 cache containing every eligible validation 6+6 window")
    p.add_argument("--development-cache", default="",
                   help="previous 128-scene midpoint cache; only defines seen-vs-heldout scene subsets")
    p.add_argument("--y600-checkpoint", required=True)
    p.add_argument("--clean-checkpoint", required=True)
    p.add_argument("--dataroot", required=True)
    p.add_argument("--info-pkl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--bootstrap-samples", type=int, default=2000)
    p.add_argument("--expected-windows", type=int, default=0)
    p.add_argument("--expected-scenes", type=int, default=0)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    cfg = load_runtime_config(a.config, a.override)
    pcfg = make_prepare_config(cfg)
    cache_meta, records = mid.base.load_cache(a.full_val_cache)
    if not records:
        raise RuntimeError("full validation cache is empty")

    sample_ids = [str(r["sample_id"]) for r in records]
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("all-window validation cache contains duplicate sample IDs")
    scene_stream = [str(window_from_record(r).scene_name) for r in records]
    all_scenes = tuple(dict.fromkeys(scene_stream))
    if int(a.expected_windows) > 0 and len(records) != int(a.expected_windows):
        raise RuntimeError(
            f"validation cache windows {len(records)} != expected {a.expected_windows}"
        )
    if int(a.expected_scenes) > 0 and len(all_scenes) != int(a.expected_scenes):
        raise RuntimeError(
            f"validation cache scenes {len(all_scenes)} != expected {a.expected_scenes}"
        )
    if len(records) <= len(all_scenes):
        raise RuntimeError(
            "all-window evaluator expected overlapping validation windows, but cache is one-window-per-scene"
        )

    dev_ids, dev_scenes = mid._load_dev_subset(a.development_cache)
    if dev_ids:
        missing = sorted(dev_ids - set(sample_ids))
        if missing:
            raise RuntimeError(
                f"all-window cache does not contain previous development midpoint samples: {missing[:5]}"
            )
    full_scene_set = set(all_scenes)
    heldout_scenes = full_scene_set - dev_scenes if dev_scenes else set()

    device = torch.device(a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu")
    y_ck, y_model, y_cfg = mid._load_model(a.y600_checkpoint, mid.PAIR_PROTOCOL, device)
    c_ck, c_model, c_cfg = mid._load_model(a.clean_checkpoint, mid.CLEAN_PROTOCOL, device)
    if y_cfg != c_cfg:
        raise RuntimeError("Y600 and Clean model_config differ")

    source = mid.base.NuScenesWindowSource(a.dataroot, info_pkl=a.info_pkl, verbose=False)
    strong_cfg = StrongW2DetConfig(free_label=int(pcfg.free_label))

    # One accumulated raw-count row per (scene, variant, horizon).  Every
    # overlapping window in that scene contributes to the same row.
    scene_rows = {}
    support_counts = {
        str(cid): {
            str(float(h)): {
                "window_instance_occurrences": 0,
                "instance_tokens": set(),
                "scenes": set(),
            }
            for h in REPORT_HORIZONS_S
        }
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
        for horizon, hi in mid.REPORT.items():
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
                    e = support_counts[str(cid)][str(float(horizon))]
                    e["window_instance_occurrences"] += 1
                    e["instance_tokens"].add(str(mr["instance_token"]))
                    e["scenes"].add(scene)

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
                key = (scene, name, float(horizon))
                row = mid._count_row(pred, gt, moving, int(pcfg.free_label))
                if key not in scene_rows:
                    scene_rows[key] = row
                else:
                    _accumulate_count_row(scene_rows[key], row)

        if wi == 0 or (wi + 1) % 50 == 0 or wi + 1 == len(records):
            print(f"all_window_val {wi+1}/{len(records)} {sid}", flush=True)

    raw_by_variant = mid._stack_scene_rows(scene_rows, all_scenes)
    subsets = {"full": set(all_scenes)}
    if dev_scenes:
        subsets["development_scenes_all_windows"] = set(dev_scenes)
        subsets["heldout_scenes_all_windows"] = set(heldout_scenes)

    report_subsets = {}
    bootstrap_subsets = {}
    for subset_name, subset_scene_set in subsets.items():
        idx = mid._subset_indices(all_scenes, subset_scene_set)
        report_subsets[subset_name] = {
            "num_scenes": int(idx.size),
            "num_windows": int(sum(1 for s in scene_stream if s in subset_scene_set)),
            "scene_names": sorted(str(x) for x in subset_scene_set),
            "variants": {
                name: mid._metric_from_raw(raw_by_variant[name], idx)
                for name in mid.VARIANTS
            },
        }
        bootstrap_subsets[subset_name] = mid._bootstrap_delta(
            raw_by_variant["y600_pred"],
            raw_by_variant["clean_pred"],
            idx,
            samples=int(a.bootstrap_samples),
            seed=mid.BOOTSTRAP_SEED,
        )

    support_json = {}
    for cid in DYNAMIC_CLASS_IDS:
        support_json[str(cid)] = {}
        for h in REPORT_HORIZONS_S:
            e = support_counts[str(cid)][str(float(h))]
            support_json[str(cid)][str(float(h))] = {
                "class_name": NUSCENES_LABELS[int(cid)],
                "window_instance_occurrences": int(e["window_instance_occurrences"]),
                "unique_instance_tokens": int(len(e["instance_tokens"])),
                "scenes": int(len(e["scenes"])),
            }

    per_scene_windows = _scene_record_counts(records)
    result = {
        "protocol": PROTOCOL,
        "status": "completed_evaluation",
        "full_val_cache": str(Path(a.full_val_cache).resolve()),
        "development_cache": str(Path(a.development_cache).resolve()) if a.development_cache else None,
        "cache_metadata": cache_meta,
        "num_windows": len(records),
        "num_scenes": len(all_scenes),
        "windows_per_scene": per_scene_windows,
        "all_eligible_overlapping_windows": True,
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
        "moving_support_counts_full": support_json,
        "paired_bootstrap_clean_minus_y600": bootstrap_subsets,
        "notes": {
            "bootstrap_unit": "scene",
            "bootstrap_reaggregation": "all overlapping windows are first summed inside scene; paired scene resample then sums raw intersections/unions",
            "overlapping_windows_are_not_independent_bootstrap_units": True,
            "scene_iou_is_not_averaged": True,
            "no_training": True,
            "development_subset_definition": "all full-validation windows from scenes represented in the supplied 128-scene development cache",
            "heldout_subset_definition": "all full-validation windows from scenes absent from the supplied development cache",
            "moving_support_count_note": "window_instance_occurrences counts repeated appearances across overlapping windows; unique_instance_tokens removes those repeats",
        },
    }

    op = Path(a.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n=== OFFICIAL-STYLE ALL-WINDOW VALIDATION ===")
    for subset_name in ("full", "development_scenes_all_windows", "heldout_scenes_all_windows"):
        if subset_name not in report_subsets:
            continue
        rr = report_subsets[subset_name]
        print(f"\n[{subset_name}] scenes={rr['num_scenes']} windows={rr['num_windows']}")
        print(f"{'variant':18s} {'IoU':>9s} {'mIoU':>9s} {'MacroMov':>10s} {'MicroMov':>10s}")
        for name in mid.VARIANTS:
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

    print("\n=== ALL-WINDOW MOVING PER CLASS x HORIZON ===")
    full = report_subsets["full"]["variants"]
    for cid in DYNAMIC_CLASS_IDS:
        name = NUSCENES_LABELS[int(cid)]
        meta = support_json[str(cid)]
        for h in REPORT_HORIZONS_S:
            hs = str(float(h))
            y = full["y600_pred"]["per_horizon"][hs]["moving_per_class"][str(cid)]
            c = full["clean_pred"]["per_horizon"][hs]["moving_per_class"][str(cid)]
            b = bootstrap_subsets["full"]["per_class_horizon"][str(cid)][hs]
            m = meta[hs]
            print(
                f"{name:22s} {h:.1f}s Y600={y:8.3f} Clean={c:8.3f} "
                f"d={c-y:+8.3f} occ={m['window_instance_occurrences']:5d} "
                f"uniq={m['unique_instance_tokens']:4d} scenes={m['scenes']:3d} "
                f"CI=[{b['p2_5_pp']:+.3f},{b['p97_5_pp']:+.3f}]"
            )
    print(f"saved {op}")


if __name__ == "__main__":
    main()
