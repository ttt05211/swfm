#!/usr/bin/env python3
"""Paired diagnosis for V18 bicycle/motorcycle Moving-IoU regression.

No training is performed and no frozen metric is modified. The script evaluates
Y600 continuation and Clean one-stage V18 on one shared record stream. Each
network is forwarded exactly once per window; the cached XY/yaw outputs are then
hard-rendered twice:

  1. predicted yaw (the deployed V18 behavior),
  2. predicted yaw for all classes except bicycle/motorcycle, whose yaw is set
     to zero immediately before the hard renderer.

Thus the intervention changes only two-wheel deployment rotation. XY, source
geometry, source order, KTA CLEAR footprints, class IDs, GT/moving support and
all other predicted yaw values are bit-identical within each checkpoint pair.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from real_motion.local_st_world_model_v17 import config_from_mapping_v17
from real_motion.local_st_world_model_v18_se2 import (
    LocalSpatialTemporalWorldModelV18SE2,
    SE2_TARGET_CONTRACT,
)
from real_motion.metrics.moving_micro_iou import MovingMicroIoUMultiHorizon
from real_motion.metrics.moving_miou_v2 import (
    DYNAMIC_CLASS_IDS,
    NUSCENES_LABELS,
    REPORT_HORIZONS_S,
    is_moving_world,
    moving_support_from_world_motion,
)
from real_motion.metrics.occupancy_iou import OccupancyIoUMultiHorizon
from real_motion.msp_wm_cache import MSPWorldModelCacheDataset
from real_motion.rigid_transport import (
    compose_component_replacements_in_input_order,
    rasterize_rigid_component,
)
from real_motion.runtime_config import add_config_args, load_runtime_config, make_prepare_config
from real_motion.strong_w2det import StrongW2DetConfig, extract_instances, match_instances
from real_motion.v18_two_wheel_diagnostic import (
    TWO_WHEEL_CLASS_IDS,
    add_count_rows,
    paired_scene_bootstrap_gap,
    renderer_yaw_delta,
    robust_error_summary,
    scene_leave_one_out_gap,
    semantic_count_row,
    wrapped_abs_error_rad,
)
from tools.real_motion import eval_p0_f9_frozen_sparse_occfm as safe
from tools.real_motion import eval_p0_f9_v18_se2 as base
from tools.real_motion.eval_p0_f9_v17_local_stwm import (
    t0_xy_to_world_preserve_source_z,
    window_from_record,
)
from tools.real_motion.train_p0_f9_v18_se2_clean import PROTOCOL as CLEAN_PROTOCOL
from tools.real_motion.train_p0_f9_v18_se2_pair import PROTOCOL as PAIR_PROTOCOL


PROTOCOL = "p0_f9_v18_two_wheel_regression_diagnostic_v1"
VARIANTS = (
    "y600_pred",
    "y600_two_wheel_zero_yaw",
    "clean_pred",
    "clean_two_wheel_zero_yaw",
)
BOOTSTRAP_SEED = 20260917


def _load_v18_checkpoint(path: str, expected_protocol: str, device: torch.device):
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


def _git_info() -> dict[str, object]:
    def run(*args):
        try:
            p = subprocess.run(
                ["git", *args], cwd=ROOT, text=True, capture_output=True, check=True
            )
            return p.stdout.strip()
        except Exception:
            return None
    return {
        "head": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


def _sample_order_sha256(records) -> str:
    h = hashlib.sha256()
    for rec in records:
        h.update(str(rec["sample_id"]).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def _new_variant_states(free_label: int):
    return {
        name: {
            "macro": safe._new_metrics(),
            "micro": MovingMicroIoUMultiHorizon(),
            "occupancy": OccupancyIoUMultiHorizon(free_label=int(free_label)),
        }
        for name in VARIANTS
    }


def _update_variant(state, horizon, pred, gt, moving):
    safe._update(state["macro"], float(horizon), pred, gt, moving)
    state["micro"].update(float(horizon), pred, gt, moving)
    state["occupancy"].update(float(horizon), pred, gt)


def _variant_report(state):
    out = safe._report(state["macro"])
    out["moving_micro"] = state["micro"].compute()
    out["occupancy"] = state["occupancy"].compute()
    return out


def _candidate_signature(report) -> dict[str, float]:
    return {
        "IoU": float(report["occupancy"]["IoU"]),
        "mIoU": float(report["overall"]["mIoU"]),
        "MovingMacro": float(report["moving"]["mIoU"]),
        "MovingMicro": float(report["moving_micro"]["micro_IoU"]),
    }


def _read_reference(path: str, *, checkpoint: str, protocol: str, num_windows: int):
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if str(obj.get("yaw_mode")) != "pred":
        raise RuntimeError(f"{path}: reference must be yaw_mode=pred")
    if str(obj.get("checkpoint_protocol")) != str(protocol):
        raise RuntimeError(f"{path}: checkpoint protocol mismatch")
    if int(obj.get("num_windows", -1)) != int(num_windows):
        raise RuntimeError(f"{path}: num_windows mismatch")
    got_ck = obj.get("checkpoint")
    if got_ck and Path(str(got_ck)).resolve() != Path(checkpoint).resolve():
        raise RuntimeError(f"{path}: reference checkpoint differs from requested checkpoint")
    if obj.get("se2_target_contract") != SE2_TARGET_CONTRACT:
        raise RuntimeError(f"{path}: SE2 target contract mismatch")
    if obj.get("a1_write_order_contract") != "legacy_clear_plus_original_strong_source_write_order_v1":
        raise RuntimeError(f"{path}: A1 composition contract mismatch")
    cand = obj.get("reports", {}).get("candidate")
    if not cand or "moving_micro" not in cand:
        raise RuntimeError(f"{path}: missing candidate micro/macro report")
    return obj, _candidate_signature(cand)


def _assert_signature(label: str, got: dict[str, float], expected: dict[str, float], tol: float):
    bad = {}
    for key in expected:
        d = float(got[key]) - float(expected[key])
        if abs(d) > float(tol):
            bad[key] = {"got": got[key], "expected": expected[key], "delta": d}
    if bad:
        raise RuntimeError(f"{label}: predicted-yaw reference reproduction failed: {bad}")


def _matched_support(rec, ann0_map, future_map, future_pose, horizon, grid):
    out = np.zeros(grid.shape_hwd, dtype=bool)
    for i, token in enumerate(tuple(rec["source_instance_token"])):
        if token is None:
            continue
        token = str(token)
        ann0 = ann0_map.get(token)
        annh = future_map.get(token)
        if ann0 is None or annh is None:
            continue
        cid = int(rec["source_class_id"][i].item())
        if int(ann0["class_id"]) != cid or int(annh["class_id"]) != cid:
            raise RuntimeError(f"source token {token}: semantic class mismatch")
        out |= moving_support_from_world_motion(
            ann0["center_world"], annh["center_world"],
            base._box_future_ego(ann0, future_pose),
            base._box_future_ego(annh, future_pose),
            float(horizon), grid=grid,
        )
    return out


def _moving_instance_sets(ann0_map, future_map, horizon, source_tokens):
    full = {int(c): set() for c in DYNAMIC_CLASS_IDS}
    matched = {int(c): set() for c in DYNAMIC_CLASS_IDS}
    source_tokens = set(str(x) for x in source_tokens if x is not None)
    for token, ann0 in ann0_map.items():
        annh = future_map.get(str(token))
        if annh is None:
            continue
        cid = int(ann0["class_id"])
        if cid not in full or int(annh["class_id"]) != cid:
            continue
        if not is_moving_world(ann0["center_world"], annh["center_world"], float(horizon)):
            continue
        full[cid].add(str(token))
        if str(token) in source_tokens:
            matched[cid].add(str(token))
    return full, matched


def _accumulate_count(store, key, row):
    store[key] = add_count_rows([store.get(key, {}), row])


def _write_csv(path: Path, aggregate_counts, scene_counts, instance_meta):
    fields = [
        "level", "scene", "variant", "support_scope", "class_id", "class_name",
        "horizon_s", "tp", "fp", "fn", "intersection", "union", "pred_voxels",
        "gt_voxels", "iou", "gt_moving_instances", "matched_moving_instances",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for (variant, scope, cid, horizon), row in sorted(aggregate_counts.items()):
            meta = instance_meta.get((int(cid), float(horizon)), {})
            w.writerow({
                "level": "aggregate", "scene": "", "variant": variant,
                "support_scope": scope, "class_id": cid,
                "class_name": NUSCENES_LABELS[int(cid)], "horizon_s": horizon,
                **row,
                "gt_moving_instances": int(meta.get("full_count", 0)),
                "matched_moving_instances": int(meta.get("matched_count", 0)),
            })
        for (scene, variant, scope, cid, horizon), row in sorted(scene_counts.items()):
            meta = instance_meta.get((int(cid), float(horizon)), {}).get("by_scene", {}).get(str(scene), {})
            w.writerow({
                "level": "scene", "scene": scene, "variant": variant,
                "support_scope": scope, "class_id": cid,
                "class_name": NUSCENES_LABELS[int(cid)], "horizon_s": horizon,
                **row,
                "gt_moving_instances": int(meta.get("full_count", 0)),
                "matched_moving_instances": int(meta.get("matched_count", 0)),
            })


def main():
    p = argparse.ArgumentParser()
    add_config_args(p)
    p.add_argument("--local-stwm-cache", required=True)
    p.add_argument("--p0f9-cache", required=True)
    p.add_argument("--y600-checkpoint", required=True)
    p.add_argument("--clean-checkpoint", required=True)
    p.add_argument("--y600-reference-json", default="")
    p.add_argument("--clean-reference-json", default="")
    p.add_argument("--dataroot", required=True)
    p.add_argument("--info-pkl", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--output-csv", required=True)
    p.add_argument("--reference-score-tol", type=float, default=1e-4)
    p.add_argument("--skip-reference-check", action="store_true")
    p.add_argument("--bootstrap-samples", type=int, default=2000)
    p.add_argument("--max-windows", type=int, default=0)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    cfg = load_runtime_config(a.config, a.override)
    pcfg = make_prepare_config(cfg)
    cache_meta, records = base.load_cache(a.local_stwm_cache)
    if int(a.max_windows) > 0:
        records = records[: min(len(records), int(a.max_windows))]
    if not records:
        raise RuntimeError("no SE2 records selected")

    ds = MSPWorldModelCacheDataset(a.p0f9_cache)
    ids = [str(r["sample_id"]) for r in records]
    sid_to_idx = {str(e["sample_id"]): i for i, e in enumerate(ds.entries)}
    missing = [sid for sid in ids if sid not in sid_to_idx]
    if missing:
        raise RuntimeError(f"P0-F9 cache misses SE2 samples: {missing[:5]}")

    device = torch.device(a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu")
    y_ck, y_model, y_cfg = _load_v18_checkpoint(a.y600_checkpoint, PAIR_PROTOCOL, device)
    c_ck, c_model, c_cfg = _load_v18_checkpoint(a.clean_checkpoint, CLEAN_PROTOCOL, device)
    if y_cfg != c_cfg:
        raise RuntimeError("Y600 and Clean model_config differ")

    if not bool(a.skip_reference_check) and (not a.y600_reference_json or not a.clean_reference_json):
        raise ValueError("full diagnosis requires both reference JSONs")
    if bool(a.skip_reference_check):
        y_ref_obj = c_ref_obj = y_ref_sig = c_ref_sig = None
    else:
        y_ref_obj, y_ref_sig = _read_reference(
            a.y600_reference_json, checkpoint=a.y600_checkpoint,
            protocol=PAIR_PROTOCOL, num_windows=len(records),
        )
        c_ref_obj, c_ref_sig = _read_reference(
            a.clean_reference_json, checkpoint=a.clean_checkpoint,
            protocol=CLEAN_PROTOCOL, num_windows=len(records),
        )
        if y_ref_obj.get("cache_metadata") != c_ref_obj.get("cache_metadata"):
            raise RuntimeError("reference JSONs use different SE2 cache metadata")
        if y_ref_obj.get("cache_metadata") != cache_meta:
            raise RuntimeError("reference JSON cache metadata differs from requested cache")

    source = base.NuScenesWindowSource(a.dataroot, info_pkl=a.info_pkl, verbose=False)
    strong_cfg = StrongW2DetConfig(free_label=int(pcfg.free_label))
    states = _new_variant_states(int(pcfg.free_label))
    aggregate_counts, scene_counts = {}, {}
    source_errors = []
    full_instance_evals = defaultdict(int)
    matched_instance_evals = defaultdict(int)
    full_scenes = defaultdict(set)
    matched_scenes = defaultdict(set)
    instance_scene_counts = defaultdict(lambda: defaultdict(lambda: {"full_count": 0, "matched_count": 0}))

    for wi, rec in enumerate(records):
        sid = str(rec["sample_id"])
        w = window_from_record(rec)
        scene_name = str(w.scene_name)
        payload = safe._sample_payload(ds[sid_to_idx[sid]], torch.device("cpu"))
        history_occ = [source.load_semantics(w.scene_name, t) for t in w.history_tokens]
        history_poses = [np.asarray(source.pose(t), dtype=np.float64) for t in w.history_tokens]
        future_poses = [np.asarray(source.pose(t), dtype=np.float64) for t in w.future_tokens]
        current = extract_instances(history_occ[-1], history_poses[-1], grid=pcfg.grid, cfg=strong_cfg)
        previous = extract_instances(history_occ[-2], history_poses[-2], grid=pcfg.grid, cfg=strong_cfg)
        velocities = match_instances(previous, current, float(pcfg.frame_dt_s), max_speed_mps=strong_cfg.max_match_speed_mps)
        if len(current) != int(rec["features"].shape[0]):
            raise RuntimeError(f"{sid}: Strong/source count mismatch")
        if [int(c["class_id"]) for c in current] != [int(x) for x in rec["source_class_id"].tolist()]:
            raise RuntimeError(f"{sid}: Strong/source order mismatch")

        features = rec["features"].float().to(device)
        tube = rec["local_semantic_tube"].to(device)
        kta = rec["kta_displacement_xy_m"].float().to(device)
        frame_motion = rec["frame_motion_features"].float().to(device)
        source_mask = rec["target_source_mask_tube"].to(device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            y_out = y_model(features, tube, kta, frame_motion, source_mask)
            c_out = c_model(features, tube, kta, frame_motion, source_mask)
        y_res = y_out["residual_xy_m"].float().cpu().numpy()
        c_res = c_out["residual_xy_m"].float().cpu().numpy()
        y_yaw = y_out["yaw_delta_rad"].float().cpu().numpy()
        c_yaw = c_out["yaw_delta_rad"].float().cpu().numpy()

        valid = rec["se2_target_valid"].bool().numpy()
        target_res = rec["target_source_residual_xy_m"].float().numpy()
        yaw_valid = rec["yaw_label_valid"].bool().numpy()
        target_yaw = rec["target_yaw_rad"].float().numpy()
        classes = [int(x) for x in rec["source_class_id"].tolist()]
        for i, cid in enumerate(classes):
            if cid not in TWO_WHEEL_CLASS_IDS:
                continue
            for h in range(len(target_res[i])):
                if not bool(valid[i, h]):
                    continue
                row = {
                    "scene": scene_name, "sample_id": sid, "class_id": cid,
                    "horizon_s": float((h + 1) * float(pcfg.frame_dt_s)),
                    "y600_xy_error_m": float(np.linalg.norm(y_res[i, h] - target_res[i, h])),
                    "clean_xy_error_m": float(np.linalg.norm(c_res[i, h] - target_res[i, h])),
                }
                if bool(yaw_valid[i, h]):
                    row.update({
                        "y600_yaw_abs_error_deg": math.degrees(float(wrapped_abs_error_rad(np.asarray([y_yaw[i, h]]), np.asarray([target_yaw[i, h]]))[0])),
                        "clean_yaw_abs_error_deg": math.degrees(float(wrapped_abs_error_rad(np.asarray([c_yaw[i, h]]), np.asarray([target_yaw[i, h]]))[0])),
                        "zero_yaw_abs_error_deg": math.degrees(abs(float(target_yaw[i, h]))),
                    })
                source_errors.append(row)

        ann0_map = base._dynamic_ann_map(source.nusc, w.t0_token)
        future_maps = [base._dynamic_ann_map(source.nusc, t) for t in w.future_tokens]
        source_tokens = tuple(rec["source_instance_token"])
        t0_pose = history_poses[-1]

        for horizon, hi in safe.REPORT.items():
            gt, anchor, moving = payload["gt"][hi], payload["anchor"][hi], payload["moving"][hi]
            matched = _matched_support(rec, ann0_map, future_maps[hi], future_poses[hi], float(horizon), pcfg.grid)
            full_inst, matched_inst = _moving_instance_sets(ann0_map, future_maps[hi], float(horizon), source_tokens)
            for cid in DYNAMIC_CLASS_IDS:
                key = (int(cid), float(horizon))
                full_instance_evals[key] += len(full_inst[int(cid)])
                matched_instance_evals[key] += len(matched_inst[int(cid)])
                if full_inst[int(cid)]:
                    full_scenes[key].add(scene_name)
                    instance_scene_counts[key][scene_name]["full_count"] += len(full_inst[int(cid)])
                if matched_inst[int(cid)]:
                    matched_scenes[key].add(scene_name)
                    instance_scene_counts[key][scene_name]["matched_count"] += len(matched_inst[int(cid)])

            baseline_all = []
            comps = {name: [] for name in VARIANTS}
            dt = (hi + 1) * float(pcfg.frame_dt_s)
            for i, comp in enumerate(current):
                cid = int(comp["class_id"])
                src_center = np.asarray(comp["centroid_world"], dtype=np.float64)
                v = np.asarray(velocities.get(i, np.zeros(3)), dtype=np.float64)
                baseline_all.append(rasterize_rigid_component(
                    comp["voxel_indices"], cid, t0_pose, future_poses[hi],
                    source_center_world=src_center, target_center_world=src_center + v * dt,
                    yaw_delta_rad=0.0, grid=pcfg.grid,
                ))
                y_xy = rec["anchors_xy_t0_m"][i, hi].numpy() + y_res[i, hi]
                c_xy = rec["anchors_xy_t0_m"][i, hi].numpy() + c_res[i, hi]
                y_center = t0_xy_to_world_preserve_source_z(y_xy, src_center, t0_pose)
                c_center = t0_xy_to_world_preserve_source_z(c_xy, src_center, t0_pose)
                yaw_map = {
                    "y600_pred": renderer_yaw_delta(cid, y_yaw[i, hi], zero_two_wheel_yaw=False),
                    "y600_two_wheel_zero_yaw": renderer_yaw_delta(cid, y_yaw[i, hi], zero_two_wheel_yaw=True),
                    "clean_pred": renderer_yaw_delta(cid, c_yaw[i, hi], zero_two_wheel_yaw=False),
                    "clean_two_wheel_zero_yaw": renderer_yaw_delta(cid, c_yaw[i, hi], zero_two_wheel_yaw=True),
                }
                center_map = {
                    "y600_pred": y_center, "y600_two_wheel_zero_yaw": y_center,
                    "clean_pred": c_center, "clean_two_wheel_zero_yaw": c_center,
                }
                for name in VARIANTS:
                    comps[name].append(rasterize_rigid_component(
                        comp["voxel_indices"], cid, t0_pose, future_poses[hi],
                        source_center_world=src_center, target_center_world=center_map[name],
                        yaw_delta_rad=yaw_map[name], grid=pcfg.grid,
                    ))

            predictions = {}
            for name in VARIANTS:
                predictions[name] = compose_component_replacements_in_input_order(
                    anchor, baseline_all, comps[name], dynamic_class_ids=DYNAMIC_CLASS_IDS,
                    free_label=int(pcfg.free_label), grid=pcfg.grid,
                )
                _update_variant(states[name], float(horizon), predictions[name], gt, moving)

            for name, pred in predictions.items():
                for scope, support in (("full", moving), ("matched", matched)):
                    for cid in DYNAMIC_CLASS_IDS:
                        row = semantic_count_row(pred, gt, support, int(cid))
                        _accumulate_count(aggregate_counts, (name, scope, int(cid), float(horizon)), row)
                        if int(cid) in TWO_WHEEL_CLASS_IDS:
                            _accumulate_count(scene_counts, (scene_name, name, scope, int(cid), float(horizon)), row)

        if wi == 0 or (wi + 1) % 16 == 0 or wi + 1 == len(records):
            print(f"two_wheel_diag {wi+1}/{len(records)} {sid}", flush=True)

    reports = {name: _variant_report(state) for name, state in states.items()}
    if not bool(a.skip_reference_check):
        _assert_signature("Y600", _candidate_signature(reports["y600_pred"]), y_ref_sig, a.reference_score_tol)
        _assert_signature("Clean", _candidate_signature(reports["clean_pred"]), c_ref_sig, a.reference_score_tol)

    instance_meta = {}
    for cid in DYNAMIC_CLASS_IDS:
        for horizon in REPORT_HORIZONS_S:
            key = (int(cid), float(horizon))
            instance_meta[key] = {
                "full_count": int(full_instance_evals[key]),
                "matched_count": int(matched_instance_evals[key]),
                "full_scene_count": int(len(full_scenes[key])),
                "matched_scene_count": int(len(matched_scenes[key])),
                "by_scene": dict(instance_scene_counts[key]),
            }

    scene_attribution = {}
    for cid in TWO_WHEEL_CLASS_IDS:
        scene_attribution[str(cid)] = {}
        for horizon in REPORT_HORIZONS_S:
            y_by_scene = {scene: row for (scene, name, scope, c, h), row in scene_counts.items() if name == "y600_pred" and scope == "full" and int(c) == cid and float(h) == float(horizon)}
            c_by_scene = {scene: row for (scene, name, scope, c, h), row in scene_counts.items() if name == "clean_pred" and scope == "full" and int(c) == cid and float(h) == float(horizon)}
            loo = scene_leave_one_out_gap(y_by_scene, c_by_scene)
            loo["paired_scene_bootstrap"] = paired_scene_bootstrap_gap(y_by_scene, c_by_scene, samples=a.bootstrap_samples, seed=BOOTSTRAP_SEED)
            scene_attribution[str(cid)][str(float(horizon))] = loo

    position_yaw = {}
    for model in ("y600", "clean"):
        position_yaw[model] = {}
        for cid in TWO_WHEEL_CLASS_IDS:
            position_yaw[model][str(cid)] = {}
            for horizon in REPORT_HORIZONS_S:
                rows = [r for r in source_errors if int(r["class_id"]) == cid and abs(float(r["horizon_s"]) - float(horizon)) < 1e-8]
                position_yaw[model][str(cid)][str(float(horizon))] = {
                    "source_center_xy_error_m": robust_error_summary(r[f"{model}_xy_error_m"] for r in rows),
                    "wrapped_yaw_abs_error_deg": robust_error_summary(r[f"{model}_yaw_abs_error_deg"] for r in rows if f"{model}_yaw_abs_error_deg" in r),
                    "zero_yaw_abs_error_deg": robust_error_summary(r["zero_yaw_abs_error_deg"] for r in rows if "zero_yaw_abs_error_deg" in r),
                }

    intervention = {}
    for model, pred_name, zero_name in (("y600", "y600_pred", "y600_two_wheel_zero_yaw"), ("clean", "clean_pred", "clean_two_wheel_zero_yaw")):
        ps, zs = _candidate_signature(reports[pred_name]), _candidate_signature(reports[zero_name])
        per_class = {}
        for cid in DYNAMIC_CLASS_IDS:
            per_class[str(cid)] = {}
            for h in REPORT_HORIZONS_S:
                pr = aggregate_counts[(pred_name, "full", int(cid), float(h))]
                zr = aggregate_counts[(zero_name, "full", int(cid), float(h))]
                per_class[str(cid)][str(float(h))] = {
                    "pred_yaw_iou": float(pr["iou"]),
                    "two_wheel_zero_yaw_iou": float(zr["iou"]),
                    "yaw_gain_pp": float(pr["iou"]) - float(zr["iou"]),
                    "pred_tp": int(pr["tp"]), "pred_fp": int(pr["fp"]), "pred_fn": int(pr["fn"]),
                    "zero_tp": int(zr["tp"]), "zero_fp": int(zr["fp"]), "zero_fn": int(zr["fn"]),
                }
        intervention[model] = {
            "pred_yaw": ps,
            "two_wheel_zero_yaw": zs,
            "yaw_gain_pred_minus_zero": {k: ps[k] - zs[k] for k in ps},
            "per_class_horizon": per_class,
        }

    fp_fn_delta = {}
    for cid in TWO_WHEEL_CLASS_IDS:
        fp_fn_delta[str(cid)] = {}
        for h in REPORT_HORIZONS_S:
            yr = aggregate_counts[("y600_pred", "full", int(cid), float(h))]
            cr = aggregate_counts[("clean_pred", "full", int(cid), float(h))]
            fp_fn_delta[str(cid)][str(float(h))] = {
                "clean_minus_y600_iou_pp": float(cr["iou"]) - float(yr["iou"]),
                "delta_tp": int(cr["tp"]) - int(yr["tp"]),
                "delta_fp": int(cr["fp"]) - int(yr["fp"]),
                "delta_fn": int(cr["fn"]) - int(yr["fn"]),
                "delta_pred_voxels": int(cr["pred_voxels"]) - int(yr["pred_voxels"]),
                "gt_voxels": int(cr["gt_voxels"]),
            }

    csv_path = Path(a.output_csv)
    _write_csv(csv_path, aggregate_counts, scene_counts, instance_meta)
    result = {
        "protocol": PROTOCOL,
        "status": "completed_evaluation",
        "provenance": {
            "git": _git_info(),
            "local_stwm_cache": str(Path(a.local_stwm_cache).resolve()),
            "p0f9_cache": str(Path(a.p0f9_cache).resolve()),
            "y600_checkpoint": str(Path(a.y600_checkpoint).resolve()),
            "clean_checkpoint": str(Path(a.clean_checkpoint).resolve()),
            "y600_checkpoint_protocol": y_ck.get("protocol"),
            "clean_checkpoint_protocol": c_ck.get("protocol"),
            "y600_continuation_step": int(y_ck.get("continuation_step", -1)),
            "clean_epoch": int(c_ck.get("epoch", -1)),
            "clean_global_step": int(c_ck.get("global_step", -1)),
            "num_windows": len(records),
            "sample_id_order_sha256": _sample_order_sha256(records),
            "se2_target_contract": SE2_TARGET_CONTRACT,
            "a1_composition": "legacy_clear_plus_original_strong_source_write_order_v1",
            "yaw_intervention_contract": "same_network_output_same_xy_same_sources_only_class_2_6_renderer_yaw_zero_v1",
            "cache_metadata": cache_meta,
        },
        "reference_reproduction": {
            "skipped": bool(a.skip_reference_check),
            "y600_reference": y_ref_sig,
            "y600_recomputed": _candidate_signature(reports["y600_pred"]),
            "clean_reference": c_ref_sig,
            "clean_recomputed": _candidate_signature(reports["clean_pred"]),
            "tolerance": float(a.reference_score_tol),
        },
        "four_group_reports": reports,
        "four_group_summary": {name: _candidate_signature(reports[name]) for name in VARIANTS},
        "yaw_intervention": intervention,
        "two_wheel_fp_fn_delta_clean_minus_y600": fp_fn_delta,
        "moving_instance_and_scene_counts": {f"{cid}:{h}": instance_meta[(cid, h)] for cid in TWO_WHEEL_CLASS_IDS for h in REPORT_HORIZONS_S},
        "scene_attribution_clean_minus_y600": scene_attribution,
        "same_valid_set_position_and_yaw": position_yaw,
        "csv": str(csv_path.resolve()),
        "notes": {
            "full_support": "exact frozen payload Moving-mIoU v2 true-moving support",
            "matched_support": "same Moving-v2 construction restricted to t0 GT-instance tokens attached to reconstructed predicted Strong sources",
            "gt_not_used_for_yaw_switch": True,
            "no_training": True,
        },
    }
    op = Path(a.output_json)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n=== FOUR-GROUP TWO-WHEEL YAW INTERVENTION ===")
    print(f"{'variant':30s} {'IoU':>9s} {'mIoU':>9s} {'MacroMov':>10s} {'MicroMov':>10s}")
    for name in VARIANTS:
        s = _candidate_signature(reports[name])
        print(f"{name:30s} {s['IoU']:9.4f} {s['mIoU']:9.4f} {s['MovingMacro']:10.4f} {s['MovingMicro']:10.4f}")
    print("\n=== TWO-WHEEL CLEAN - Y600, PREDICTED YAW ===")
    for cid in TWO_WHEEL_CLASS_IDS:
        for h in REPORT_HORIZONS_S:
            row = fp_fn_delta[str(cid)][str(float(h))]
            print(f"{NUSCENES_LABELS[cid]:10s} {h:.1f}s dIoU={row['clean_minus_y600_iou_pp']:+.3f} dTP={row['delta_tp']:+d} dFP={row['delta_fp']:+d} dFN={row['delta_fn']:+d}")
    print(f"saved {op}")
    print(f"saved {csv_path}")


if __name__ == "__main__":
    main()
