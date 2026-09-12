#!/usr/bin/env python3
"""Yaw value diagnostic for V17 rigid source transport.

No training is performed.  The probe asks whether adding oracle yaw to the
already-predicted V17 XY trajectory materially improves Moving-mIoU, using the
same A1 source-order compositor as the V17 headline evaluation.

Oracle transforms are fitted in the renderer's source-point coordinates.  GT
boxes are used only to generate corresponding target points; translation/yaw
passed to the renderer come from a planar Procrustes fit, so box-center/source-
centroid pivot mismatch cannot artificially depress the yaw oracle.
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

from real_motion.geometry import quaternion_yaw
from real_motion.local_st_world_model_v17 import MODEL_PROTOCOL_V17
from real_motion.metrics.moving_miou_v2 import (
    Box3D,
    DYNAMIC_CLASS_IDS,
    MovingMIoUV2MultiHorizon,
    NUSCENES_LABELS,
    moving_support_from_world_motion,
)
from real_motion.metrics.occupancy_iou import OccupancyIoUMultiHorizon
from real_motion.msp_wm_cache import MSPWorldModelCacheDataset
from real_motion.nuscenes_adapter import NuScenesWindowSource, category_to_dynamic_class
from real_motion.rigid_transport import (
    compose_component_replacements_in_input_order,
    rasterize_rigid_component,
    wrap_angle,
)
from real_motion.runtime_config import add_config_args, load_runtime_config, make_prepare_config
from real_motion.strong_w2det import StrongW2DetConfig, extract_instances, match_instances
from tools.real_motion import eval_p0_f9_frozen_sparse_occfm as safe
from tools.real_motion.eval_p0_f9_v17_local_stwm import (
    load_cache,
    load_model,
    t0_xy_to_world_preserve_source_z,
    window_from_record,
)

PROTOCOL = "p0_f9_v17_yaw_renderer_consistent_oracle_v1"
VARIANTS = ("v17_xy", "v17_xy_gt_yaw", "gt_xy_fit", "gt_rigid_fit")
VEHICLE_CLASSES = (3, 4, 5, 9, 10)


def _dynamic_annotations(nusc, sample_token):
    sample = nusc.get("sample", str(sample_token))
    out = []
    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        cid = category_to_dynamic_class(ann["category_name"])
        if cid is None:
            continue
        size = np.asarray(ann["size"], dtype=np.float64)
        out.append({
            "instance_token": str(ann["instance_token"]),
            "class_id": int(cid),
            "center_world": np.asarray(ann["translation"], dtype=np.float64),
            "yaw_world": float(quaternion_yaw(ann["rotation"])),
            "size_lwh": np.asarray([size[1], size[0], size[2]], dtype=np.float64),
        })
    out.sort(key=lambda x: (x["class_id"], x["instance_token"]))
    return out


def _ann_map(nusc, token):
    return {x["instance_token"]: x for x in _dynamic_annotations(nusc, token)}


def _match_components(components, anns, max_distance_m):
    pairs = []
    for ci, comp in enumerate(components):
        c = np.asarray(comp["centroid_world"], dtype=np.float64)
        for ai, ann in enumerate(anns):
            if int(comp["class_id"]) != int(ann["class_id"]):
                continue
            d = float(np.linalg.norm(c[:2] - ann["center_world"][:2]))
            if d <= float(max_distance_m):
                pairs.append((d, ci, ai, ann["instance_token"]))
    pairs.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
    used_c, used_a = set(), set()
    out = {}
    for _, ci, ai, _ in pairs:
        if ci in used_c or ai in used_a:
            continue
        used_c.add(ci); used_a.add(ai)
        out[int(ci)] = anns[int(ai)]
    return out


def _voxel_centers_world(voxel_indices, ego_to_world, grid):
    idx = np.asarray(voxel_indices, dtype=np.int64)
    origin = np.asarray([grid.x_min, grid.y_min, grid.z_min], dtype=np.float64)
    step = np.asarray(grid.voxel_size, dtype=np.float64)
    p = origin[None] + (idx.astype(np.float64) + 0.5) * step[None]
    T = np.asarray(ego_to_world, dtype=np.float64)
    return p @ T[:3, :3].T + T[:3, 3]


def _gt_target_points(source_points_world, ann0, annh):
    p = np.asarray(source_points_world, dtype=np.float64)
    dyaw = wrap_angle(float(annh["yaw_world"]) - float(ann0["yaw_world"]))
    c, s = math.cos(dyaw), math.sin(dyaw)
    R = np.asarray([[c, -s], [s, c]], dtype=np.float64)
    rel = p[:, :2] - np.asarray(ann0["center_world"], dtype=np.float64)[None, :2]
    out = p.copy()
    out[:, :2] = np.asarray(annh["center_world"], dtype=np.float64)[None, :2] + rel @ R.T
    return out


def _fit_translation(source_xy, target_xy, source_center_xy):
    shift = np.asarray(target_xy, dtype=np.float64).mean(0) - np.asarray(
        source_xy, dtype=np.float64
    ).mean(0)
    return np.asarray(source_center_xy, dtype=np.float64) + shift


def _fit_rigid(source_xy, target_xy, source_center_xy):
    x = np.asarray(source_xy, dtype=np.float64)
    y = np.asarray(target_xy, dtype=np.float64)
    if len(x) < 2:
        return _fit_translation(x, y, source_center_xy), 0.0, False
    xm = x.mean(0); ym = y.mean(0)
    xc = x - xm; yc = y - ym
    if float(np.linalg.norm(xc)) < 1e-8:
        return _fit_translation(x, y, source_center_xy), 0.0, False
    H = xc.T @ yc
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1.0
        R = Vt.T @ U.T
    yaw = wrap_angle(math.atan2(float(R[1, 0]), float(R[0, 0])))
    t = ym - xm @ R.T
    target_center = np.asarray(source_center_xy, dtype=np.float64) @ R.T + t
    return target_center, yaw, True


def _ego_yaw(T):
    R = np.asarray(T, dtype=np.float64)[:3, :3]
    return float(math.atan2(R[1, 0], R[0, 0]))


def _box_future_ego(ann, future_pose):
    T = np.linalg.inv(np.asarray(future_pose, dtype=np.float64))
    c = (T @ np.r_[ann["center_world"], 1.0])[:3]
    return Box3D(
        token=str(ann["instance_token"]),
        class_id=int(ann["class_id"]),
        center_xyz=tuple(float(x) for x in c),
        size_lwh=tuple(float(x) for x in ann["size_lwh"]),
        yaw=wrap_angle(float(ann["yaw_world"]) - _ego_yaw(future_pose)),
    )


def _matched_support(matches, future_map, future_pose, horizon, grid):
    out = np.zeros(grid.shape_hwd, dtype=bool)
    for ann0 in matches.values():
        annh = future_map.get(ann0["instance_token"])
        if annh is None:
            continue
        out |= moving_support_from_world_motion(
            ann0["center_world"],
            annh["center_world"],
            _box_future_ego(ann0, future_pose),
            _box_future_ego(annh, future_pose),
            float(horizon),
            grid=grid,
        )
    return out


def _update(states, occ_states, matched_states, name, horizon, pred, gt, full_support, matched_support):
    safe._update(states[name], float(horizon), pred, gt, full_support)
    occ_states[name].update(float(horizon), pred, gt)
    matched_states[name].update(float(horizon), pred, gt, matched_support)


def _row(report):
    return {
        "IoU": float(report["occupancy"]["IoU"]),
        "mIoU": float(report["overall"]["mIoU"]),
        "Moving": float(report["moving"]["mIoU"]),
    }


def main():
    p = argparse.ArgumentParser()
    add_config_args(p)
    p.add_argument("--local-stwm-cache", required=True)
    p.add_argument("--p0f9-cache", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataroot", required=True)
    p.add_argument("--info-pkl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--match-max-distance-m", type=float, default=4.0)
    p.add_argument("--max-windows", type=int, default=0)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    cfg = load_runtime_config(a.config, a.override)
    pcfg = make_prepare_config(cfg)
    _, records = load_cache(a.local_stwm_cache)
    if a.max_windows > 0:
        records = records[: min(len(records), int(a.max_windows))]

    ds = MSPWorldModelCacheDataset(a.p0f9_cache)
    sid_to_idx = {str(e["sample_id"]): i for i, e in enumerate(ds.entries)}
    missing = [r["sample_id"] for r in records if str(r["sample_id"]) not in sid_to_idx]
    if missing:
        raise RuntimeError(f"P0-F9 cache misses V17 samples: {missing[:5]}")

    device = torch.device(a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu")
    ck, model, is_backtrace3d = load_model(a.checkpoint, device, return_mode=True)
    if ck.get("protocol") != MODEL_PROTOCOL_V17 or is_backtrace3d:
        raise RuntimeError("yaw oracle must use a standard V17 checkpoint")
    if not bool(ck.get("use_representation", False)):
        raise RuntimeError("yaw oracle expects V17 representation checkpoint")

    source = NuScenesWindowSource(a.dataroot, info_pkl=a.info_pkl, verbose=False)
    strong_cfg = StrongW2DetConfig(free_label=int(pcfg.free_label))
    states = {k: safe._new_metrics() for k in VARIANTS}
    occ_states = {
        k: OccupancyIoUMultiHorizon(free_label=int(pcfg.free_label)) for k in VARIANTS
    }
    matched_states = {k: MovingMIoUV2MultiHorizon() for k in VARIANTS}
    totals = {
        "windows": 0,
        "sources": 0,
        "gt_matches": 0,
        "future_match_pairs": 0,
        "rigid_fit_valid": 0,
        "rigid_fit_degenerate": 0,
    }
    support_full = 0
    support_matched = 0
    support_inter = 0

    for wi, rec in enumerate(records):
        sid = str(rec["sample_id"])
        payload = safe._sample_payload(ds[sid_to_idx[sid]], torch.device("cpu"))
        w = window_from_record(rec)
        history_occ = [source.load_semantics(w.scene_name, t) for t in w.history_tokens]
        history_poses = [np.asarray(source.pose(t), dtype=np.float64) for t in w.history_tokens]
        future_poses = [np.asarray(source.pose(t), dtype=np.float64) for t in w.future_tokens]
        current = extract_instances(
            history_occ[-1], history_poses[-1], grid=pcfg.grid, cfg=strong_cfg
        )
        previous = extract_instances(
            history_occ[-2], history_poses[-2], grid=pcfg.grid, cfg=strong_cfg
        )
        velocities = match_instances(
            previous, current, float(pcfg.frame_dt_s),
            max_speed_mps=strong_cfg.max_match_speed_mps,
        )
        if len(current) != int(rec["features"].shape[0]):
            raise RuntimeError(f"{sid}: Strong/V17 source count mismatch")

        features = rec["features"].float().to(device)
        tube = rec["local_semantic_tube"].to(device)
        kta = rec["kta_displacement_xy_m"].float().to(device)
        fm = rec["frame_motion_features"].float().to(device)
        sm = rec["target_source_mask_tube"].to(device)
        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            pred = model(features, tube, kta, fm, sm)
        residual = pred["residual_xy_m"].float().cpu().numpy()

        ann0_list = _dynamic_annotations(source.nusc, w.t0_token)
        matches = _match_components(current, ann0_list, a.match_max_distance_m)
        fmap = [_ann_map(source.nusc, t) for t in w.future_tokens]
        t0_pose = history_poses[-1]

        totals["windows"] += 1
        totals["sources"] += len(current)
        totals["gt_matches"] += len(matches)

        source_points = {
            i: _voxel_centers_world(c["voxel_indices"], t0_pose, pcfg.grid)
            for i, c in enumerate(current)
        }

        for horizon, hi in safe.REPORT.items():
            gt = payload["gt"][hi]
            anchor = payload["anchor"][hi]
            full_moving = payload["moving"][hi]
            matched_moving = _matched_support(
                matches, fmap[hi], future_poses[hi], float(horizon), pcfg.grid
            )
            support_full += int(full_moving.sum())
            support_matched += int(matched_moving.sum())
            support_inter += int((full_moving & matched_moving).sum())

            baseline_all = []
            v17_repl = []
            v17_yaw_repl = []
            oracle_baseline = []
            gt_xy_repl = []
            gt_rigid_repl = []
            dt = (hi + 1) * float(pcfg.frame_dt_s)

            for i, comp in enumerate(current):
                src_center = np.asarray(comp["centroid_world"], dtype=np.float64)
                v = np.asarray(velocities.get(i, np.zeros(3)), dtype=np.float64)
                base = rasterize_rigid_component(
                    comp["voxel_indices"], int(comp["class_id"]),
                    t0_pose, future_poses[hi],
                    source_center_world=src_center,
                    target_center_world=src_center + v * dt,
                    yaw_delta_rad=0.0, grid=pcfg.grid,
                )
                baseline_all.append(base)

                xy_pred = rec["anchors_xy_t0_m"][i, hi].numpy() + residual[i, hi]
                pred_center = t0_xy_to_world_preserve_source_z(
                    xy_pred, src_center, t0_pose
                )
                v17 = rasterize_rigid_component(
                    comp["voxel_indices"], int(comp["class_id"]),
                    t0_pose, future_poses[hi],
                    source_center_world=src_center,
                    target_center_world=pred_center,
                    yaw_delta_rad=0.0, grid=pcfg.grid,
                )
                v17_repl.append(v17)

                ann0 = matches.get(i)
                annh = None if ann0 is None else fmap[hi].get(ann0["instance_token"])
                if annh is None:
                    v17_yaw_repl.append(v17)
                    continue

                totals["future_match_pairs"] += 1
                target_points = _gt_target_points(source_points[i], ann0, annh)
                trans_center_xy = _fit_translation(
                    source_points[i][:, :2],
                    target_points[:, :2],
                    src_center[:2],
                )
                rigid_center_xy, fit_yaw, fit_valid = _fit_rigid(
                    source_points[i][:, :2],
                    target_points[:, :2],
                    src_center[:2],
                )
                totals["rigid_fit_valid" if fit_valid else "rigid_fit_degenerate"] += 1

                v17_yaw_repl.append(
                    rasterize_rigid_component(
                        comp["voxel_indices"], int(comp["class_id"]),
                        t0_pose, future_poses[hi],
                        source_center_world=src_center,
                        target_center_world=pred_center,
                        yaw_delta_rad=fit_yaw, grid=pcfg.grid,
                    )
                )
                oracle_baseline.append(base)
                gt_xy_repl.append(
                    rasterize_rigid_component(
                        comp["voxel_indices"], int(comp["class_id"]),
                        t0_pose, future_poses[hi],
                        source_center_world=src_center,
                        target_center_world=np.asarray(
                            [trans_center_xy[0], trans_center_xy[1], src_center[2]]
                        ),
                        yaw_delta_rad=0.0, grid=pcfg.grid,
                    )
                )
                gt_rigid_repl.append(
                    rasterize_rigid_component(
                        comp["voxel_indices"], int(comp["class_id"]),
                        t0_pose, future_poses[hi],
                        source_center_world=src_center,
                        target_center_world=np.asarray(
                            [rigid_center_xy[0], rigid_center_xy[1], src_center[2]]
                        ),
                        yaw_delta_rad=fit_yaw, grid=pcfg.grid,
                    )
                )

            outputs = {
                "v17_xy": compose_component_replacements_in_input_order(
                    anchor, baseline_all, v17_repl,
                    dynamic_class_ids=DYNAMIC_CLASS_IDS,
                    free_label=int(pcfg.free_label), grid=pcfg.grid,
                ),
                "v17_xy_gt_yaw": compose_component_replacements_in_input_order(
                    anchor, baseline_all, v17_yaw_repl,
                    dynamic_class_ids=DYNAMIC_CLASS_IDS,
                    free_label=int(pcfg.free_label), grid=pcfg.grid,
                ),
                "gt_xy_fit": compose_component_replacements_in_input_order(
                    anchor, oracle_baseline, gt_xy_repl,
                    dynamic_class_ids=DYNAMIC_CLASS_IDS,
                    free_label=int(pcfg.free_label), grid=pcfg.grid,
                ),
                "gt_rigid_fit": compose_component_replacements_in_input_order(
                    anchor, oracle_baseline, gt_rigid_repl,
                    dynamic_class_ids=DYNAMIC_CLASS_IDS,
                    free_label=int(pcfg.free_label), grid=pcfg.grid,
                ),
            }
            for name, out in outputs.items():
                _update(
                    states, occ_states, matched_states, name,
                    float(horizon), out, gt, full_moving, matched_moving,
                )

        if wi == 0 or (wi + 1) % 8 == 0 or wi + 1 == len(records):
            print(
                f"yaw_oracle {wi+1}/{len(records)} sid={sid} "
                f"sources={len(current)} matched={len(matches)}",
                flush=True,
            )

    reports = {k: safe._report(v) for k, v in states.items()}
    matched_reports = {k: v.compute() for k, v in matched_states.items()}
    for k in VARIANTS:
        reports[k]["occupancy"] = occ_states[k].compute()

    base = _row(reports["v17_xy"])
    yaw = _row(reports["v17_xy_gt_yaw"])
    gxy = _row(reports["gt_xy_fit"])
    grid = _row(reports["gt_rigid_fit"])
    decision_gain = yaw["Moving"] - base["Moving"]
    yaw_gtxy_gain = grid["Moving"] - gxy["Moving"]
    if decision_gain < 0.20:
        decision = "NO_YAW_HEAD"
    elif decision_gain > 0.50:
        ph = reports["v17_xy_gt_yaw"]["moving"]["per_horizon"]
        pb = reports["v17_xy"]["moving"]["per_horizon"]
        d2 = float(ph[2.0]["mIoU"] - pb[2.0]["mIoU"])
        d3 = float(ph[3.0]["mIoU"] - pb[3.0]["mIoU"])
        decision = "ADD_YAW_HEAD" if d2 > 0 and d3 > 0 else "YAW_OPTIONAL"
    else:
        decision = "YAW_OPTIONAL"

    result = {
        "protocol": PROTOCOL,
        "checkpoint": str(Path(a.checkpoint).resolve()),
        "variants": reports,
        "matched_source_moving": matched_reports,
        "totals": totals,
        "matched_support": {
            "full_voxels": support_full,
            "matched_voxels": support_matched,
            "intersection_voxels": support_inter,
            "full_recall": support_inter / max(support_full, 1),
            "matched_precision": support_inter / max(support_matched, 1),
        },
        "yaw_gain_given_v17_xy_pp": decision_gain,
        "yaw_gain_given_gt_xy_pp": yaw_gtxy_gain,
        "decision": decision,
        "thresholds_pp": {
            "no_yaw": 0.20,
            "optional_upper": 0.50,
            "formal_add_requires_2s_3s_positive": True,
        },
    }

    print("\n=== V17 RENDERER-CONSISTENT YAW ORACLE ===")
    print(f"{'variant':20s} {'IoU':>9s} {'mIoU':>9s} {'Moving':>9s} {'MatchedM':>10s}")
    for name in VARIANTS:
        r = _row(reports[name])
        print(
            f"{name:20s} {r['IoU']:9.4f} {r['mIoU']:9.4f} {r['Moving']:9.4f} "
            f"{float(matched_reports[name]['mIoU']):10.4f}"
        )
    print("\n=== MOVING BY HORIZON ===")
    for name in VARIANTS:
        ph = reports[name]["moving"]["per_horizon"]
        print(
            f"{name:20s} 1s={ph[1.0]['mIoU']:.4f} "
            f"2s={ph[2.0]['mIoU']:.4f} 3s={ph[3.0]['mIoU']:.4f}"
        )
    print("\n=== VEHICLE MOVING IoU (horizon-averaged display) ===")
    for name in VARIANTS:
        ph = reports[name]["moving"]["per_horizon"]
        vals = {}
        for cid in VEHICLE_CLASSES:
            hs = [float(ph[h]["per_class"].get(cid, float("nan"))) for h in (1.0, 2.0, 3.0)]
            good = [x for x in hs if np.isfinite(x)]
            vals[NUSCENES_LABELS[cid]] = float(np.mean(good)) if good else float("nan")
        print(name, json.dumps(vals))
    print(
        f"\nyaw|V17={decision_gain:+.4f} pp "
        f"yaw|GTXY={yaw_gtxy_gain:+.4f} pp decision={decision}"
    )
    print(
        "matched_support_recall="
        f"{100*result['matched_support']['full_recall']:.2f}%"
    )

    op = Path(a.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"saved {op}")


if __name__ == "__main__":
    main()
