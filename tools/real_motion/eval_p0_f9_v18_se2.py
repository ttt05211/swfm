#!/usr/bin/env python3
"""Hard A1 evaluation for paired V17 control / V18-SE2 checkpoints."""
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
from real_motion.local_st_world_model_v17 import (
    LocalSpatialTemporalWorldModelV17,
    config_from_mapping_v17,
)
from real_motion.local_st_world_model_v18_se2 import (
    LocalSpatialTemporalWorldModelV18SE2,
    SE2_CACHE_VERSION,
    SE2_TARGET_CONTRACT,
    YAW_ENABLED_CLASS_IDS,
    wrap_angle_np,
)
from real_motion.metrics.moving_miou_v2 import (
    Box3D,
    DYNAMIC_CLASS_IDS,
    MovingMIoUV2MultiHorizon,
    moving_support_from_world_motion,
)
from real_motion.metrics.occupancy_iou import OccupancyIoUMultiHorizon
from real_motion.msp_wm_cache import MSPWorldModelCacheDataset
from real_motion.nuscenes_adapter import NuScenesWindowSource, category_to_dynamic_class
from real_motion.rigid_transport import (
    compose_component_replacements_in_input_order,
    rasterize_rigid_component,
)
from real_motion.runtime_config import (
    add_config_args,
    load_runtime_config,
    make_prepare_config,
)
from real_motion.strong_w2det import StrongW2DetConfig, extract_instances, match_instances
from tools.real_motion import eval_p0_f9_frozen_sparse_occfm as safe
from tools.real_motion.eval_p0_f9_v17_local_stwm import (
    t0_xy_to_world_preserve_source_z,
    window_from_record,
)
from tools.real_motion.train_p0_f9_v18_se2_pair import PROTOCOL

EVAL_PROTOCOL = "p0_f9_v18_se2_hard_a1_eval_v1"
TURN_BINS = ("straight", "mild", "strong")


def load_cache(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if obj.get("version") != SE2_CACHE_VERSION:
        raise RuntimeError(f"expected {SE2_CACHE_VERSION}")
    meta = obj.get("metadata") or {}
    if meta.get("se2_target_contract") != SE2_TARGET_CONTRACT:
        raise RuntimeError("SE2 target contract mismatch")
    records = obj.get("records") or []
    if not records:
        raise RuntimeError("SE2 cache has no records")
    return meta, records


def load_model(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("protocol") != PROTOCOL:
        raise RuntimeError(f"checkpoint protocol mismatch: {ck.get('protocol')}")
    arm = str(ck.get("arm"))
    cfg = config_from_mapping_v17(ck.get("model_config"))
    if arm == "C":
        model = LocalSpatialTemporalWorldModelV17(cfg).to(device)
    elif arm == "Y":
        model = LocalSpatialTemporalWorldModelV18SE2(cfg).to(device)
    else:
        raise RuntimeError(f"unknown paired arm {arm}")
    model.load_state_dict(ck["state_dict"], strict=True)
    model.eval()
    return ck, model, arm


def _dynamic_ann_map(nusc, sample_token):
    sample = nusc.get("sample", str(sample_token))
    out = {}
    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        cid = category_to_dynamic_class(ann["category_name"])
        if cid is None:
            continue
        size = np.asarray(ann["size"], dtype=np.float64)
        out[str(ann["instance_token"])] = {
            "instance_token": str(ann["instance_token"]),
            "class_id": int(cid),
            "center_world": np.asarray(ann["translation"], dtype=np.float64),
            "yaw_world": float(quaternion_yaw(ann["rotation"])),
            "size_lwh": np.asarray([size[1], size[0], size[2]], dtype=np.float64),
        }
    return out


def _ego_yaw(T):
    R = np.asarray(T, dtype=np.float64)[:3, :3]
    return math.atan2(float(R[1, 0]), float(R[0, 0]))


def _box_future_ego(ann, future_pose):
    T = np.linalg.inv(np.asarray(future_pose, dtype=np.float64))
    c = (T @ np.r_[ann["center_world"], 1.0])[:3]
    return Box3D(
        token=str(ann["instance_token"]),
        class_id=int(ann["class_id"]),
        center_xyz=tuple(float(x) for x in c),
        size_lwh=tuple(float(x) for x in ann["size_lwh"]),
        yaw=wrap_angle_np(float(ann["yaw_world"]) - _ego_yaw(future_pose)),
    )


def _turn_bin(abs_yaw_rad, mild_deg, strong_deg):
    deg = math.degrees(abs(float(abs_yaw_rad)))
    if deg < float(mild_deg):
        return "straight"
    if deg < float(strong_deg):
        return "mild"
    return "strong"


def _turn_supports(rec, ann0_map, future_maps, future_poses, horizon, hi, grid, mild_deg, strong_deg):
    out = {name: np.zeros(grid.shape_hwd, dtype=bool) for name in TURN_BINS}
    tokens = tuple(rec["source_instance_token"])
    target_yaw = rec["target_yaw_rad"].float().numpy()
    yaw_valid = rec["yaw_label_valid"].bool().numpy()
    yaw_enabled = rec["yaw_enabled"].bool().numpy()
    for i, token in enumerate(tokens):
        if token is None or not bool(yaw_enabled[i]) or not bool(yaw_valid[i, -1]):
            continue
        token = str(token)
        ann0 = ann0_map.get(token)
        annh = future_maps[hi].get(token)
        if ann0 is None or annh is None:
            continue
        name = _turn_bin(abs(float(target_yaw[i, -1])), mild_deg, strong_deg)
        out[name] |= moving_support_from_world_motion(
            ann0["center_world"],
            annh["center_world"],
            _box_future_ego(ann0, future_poses[hi]),
            _box_future_ego(annh, future_poses[hi]),
            float(horizon),
            grid=grid,
        )
    return out


def _update(states, occ_states, name, horizon, pred, gt, moving):
    safe._update(states[name], float(horizon), pred, gt, moving)
    occ_states[name].update(float(horizon), pred, gt)


def _latest(values, valid):
    out = []
    for i in range(valid.shape[0]):
        ids = np.flatnonzero(valid[i])
        if len(ids):
            out.append(float(values[i, int(ids[-1])]))
    return out


def main():
    p = argparse.ArgumentParser()
    add_config_args(p)
    p.add_argument("--local-stwm-cache", required=True)
    p.add_argument("--p0f9-cache", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataroot", required=True)
    p.add_argument("--info-pkl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--turn-mild-deg", type=float, default=5.0)
    p.add_argument("--turn-strong-deg", type=float, default=15.0)
    p.add_argument("--max-windows", type=int, default=0)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    if not (0.0 < a.turn_mild_deg < a.turn_strong_deg):
        raise ValueError("turn thresholds must satisfy 0 < mild < strong")

    cfg = load_runtime_config(a.config, a.override)
    pcfg = make_prepare_config(cfg)
    cache_meta, records = load_cache(a.local_stwm_cache)
    if int(a.max_windows) > 0:
        records = records[: min(len(records), int(a.max_windows))]

    ds = MSPWorldModelCacheDataset(a.p0f9_cache)
    sid_to_idx = {str(e["sample_id"]): i for i, e in enumerate(ds.entries)}
    missing = [str(r["sample_id"]) for r in records if str(r["sample_id"]) not in sid_to_idx]
    if missing:
        raise RuntimeError(f"P0-F9 cache misses SE2 samples: {missing[:5]}")

    device = torch.device(
        a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    ck, model, arm = load_model(a.checkpoint, device)
    source = NuScenesWindowSource(a.dataroot, info_pkl=a.info_pkl, verbose=False)
    strong_cfg = StrongW2DetConfig(free_label=int(pcfg.free_label))

    names = ("strong_anchor", "candidate")
    states = {n: safe._new_metrics() for n in names}
    occ_states = {
        n: OccupancyIoUMultiHorizon(free_label=int(pcfg.free_label))
        for n in names
    }
    turn_states = {
        n: MovingMIoUV2MultiHorizon() for n in TURN_BINS
    }

    xy_err = []
    xy_fde = []
    yaw_err = []
    zero_yaw_err = []
    yaw_by_h = {h: [] for h in range(6)}
    sources = 0
    yaw_enabled_sources = 0

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
            raise RuntimeError(f"{sid}: Strong/source count mismatch")
        if [int(c["class_id"]) for c in current] != [
            int(x) for x in rec["source_class_id"].tolist()
        ]:
            raise RuntimeError(f"{sid}: Strong/source order mismatch")

        features = rec["features"].float().to(device)
        tube = rec["local_semantic_tube"].to(device)
        kta = rec["kta_displacement_xy_m"].float().to(device)
        frame_motion = rec["frame_motion_features"].float().to(device)
        source_mask = rec["target_source_mask_tube"].to(device)
        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            out = model(features, tube, kta, frame_motion, source_mask)
        residual = out["residual_xy_m"].float().cpu().numpy()
        pred_yaw = (
            out["yaw_delta_rad"].float().cpu().numpy()
            if arm == "Y"
            else np.zeros((len(current), FUTURE_FRAMES), dtype=np.float32)
        )

        valid = (
            rec["se2_target_valid"].bool().numpy()
            if arm == "Y"
            else rec["target_valid"].bool().numpy()
        )
        target_res = (
            rec["target_source_residual_xy_m"].float().numpy()
            if arm == "Y"
            else rec["target_residual_xy_m"].float().numpy()
        )
        err = np.linalg.norm(residual - target_res, axis=-1)
        xy_err.extend(err[valid].tolist())
        xy_fde.extend(_latest(err, valid))

        yaw_enabled = rec["yaw_enabled"].bool().numpy()
        yaw_valid = rec["yaw_label_valid"].bool().numpy()
        yaw_target = rec["target_yaw_rad"].float().numpy()
        sources += len(current)
        yaw_enabled_sources += int(yaw_enabled.sum())
        ym = yaw_enabled[:, None] & yaw_valid & rec["se2_target_valid"].bool().numpy()
        if arm == "Y" and bool(ym.any()):
            de = np.arctan2(
                np.sin(pred_yaw - yaw_target),
                np.cos(pred_yaw - yaw_target),
            )
            yaw_err.extend(np.abs(de[ym]).tolist())
            zero_yaw_err.extend(np.abs(yaw_target[ym]).tolist())
            for h in range(FUTURE_FRAMES):
                mh = ym[:, h]
                if bool(mh.any()):
                    yaw_by_h[h].extend(np.abs(de[:, h][mh]).tolist())

        ann0_map = _dynamic_ann_map(source.nusc, w.t0_token)
        future_maps = [_dynamic_ann_map(source.nusc, t) for t in w.future_tokens]
        t0_pose = history_poses[-1]

        for horizon, hi in safe.REPORT.items():
            gt = payload["gt"][hi]
            anchor = payload["anchor"][hi]
            moving = payload["moving"][hi]
            baseline_all = []
            candidate_all = []
            dt = (hi + 1) * float(pcfg.frame_dt_s)

            for i, comp in enumerate(current):
                src_center = np.asarray(comp["centroid_world"], dtype=np.float64)
                v = np.asarray(velocities.get(i, np.zeros(3)), dtype=np.float64)
                base = rasterize_rigid_component(
                    comp["voxel_indices"],
                    int(comp["class_id"]),
                    t0_pose,
                    future_poses[hi],
                    source_center_world=src_center,
                    target_center_world=src_center + v * dt,
                    yaw_delta_rad=0.0,
                    grid=pcfg.grid,
                )
                baseline_all.append(base)

                xy_pred = rec["anchors_xy_t0_m"][i, hi].numpy() + residual[i, hi]
                pred_center = t0_xy_to_world_preserve_source_z(
                    xy_pred, src_center, t0_pose
                )
                use_yaw = (
                    arm == "Y"
                    and int(comp["class_id"]) in set(YAW_ENABLED_CLASS_IDS)
                )
                yaw = float(pred_yaw[i, hi]) if use_yaw else 0.0
                candidate_all.append(
                    rasterize_rigid_component(
                        comp["voxel_indices"],
                        int(comp["class_id"]),
                        t0_pose,
                        future_poses[hi],
                        source_center_world=src_center,
                        target_center_world=pred_center,
                        yaw_delta_rad=yaw,
                        grid=pcfg.grid,
                    )
                )

            candidate = compose_component_replacements_in_input_order(
                anchor,
                baseline_all,
                candidate_all,
                dynamic_class_ids=DYNAMIC_CLASS_IDS,
                free_label=int(pcfg.free_label),
                grid=pcfg.grid,
            )
            _update(states, occ_states, "strong_anchor", horizon, anchor, gt, moving)
            _update(states, occ_states, "candidate", horizon, candidate, gt, moving)

            turn_support = _turn_supports(
                rec, ann0_map, future_maps, future_poses,
                float(horizon), hi, pcfg.grid,
                float(a.turn_mild_deg), float(a.turn_strong_deg),
            )
            for name in TURN_BINS:
                turn_states[name].update(
                    float(horizon), candidate, gt, turn_support[name]
                )

        if wi == 0 or (wi + 1) % 16 == 0 or wi + 1 == len(records):
            print(
                f"se2 eval {wi+1}/{len(records)} {sid}",
                flush=True,
            )

    reports = {n: safe._report(states[n]) for n in names}
    for n in names:
        reports[n]["occupancy"] = occ_states[n].compute()
    turn_reports = {n: turn_states[n].compute() for n in TURN_BINS}

    diag = {
        "arm": arm,
        "source_count": sources,
        "yaw_enabled_source_count": yaw_enabled_sources,
        "trajectory_error_definition": (
            "source_center_ADE_FDE"
            if arm == "Y"
            else "legacy_box_displacement_equivalent_ADE_FDE"
        ),
        "learned_ade_m": float(np.mean(xy_err)) if xy_err else float("nan"),
        "learned_fde_m": float(np.mean(xy_fde)) if xy_fde else float("nan"),
        "wrapped_yaw_mae_deg": (
            math.degrees(float(np.mean(yaw_err))) if yaw_err else float("nan")
        ),
        "zero_yaw_baseline_mae_deg": (
            math.degrees(float(np.mean(zero_yaw_err)))
            if zero_yaw_err else float("nan")
        ),
        "wrapped_yaw_mae_deg_by_horizon": {
            str(h + 1): (
                math.degrees(float(np.mean(yaw_by_h[h])))
                if yaw_by_h[h] else float("nan")
            )
            for h in range(FUTURE_FRAMES)
        },
        "turn_subset_definition": {
            "straight": f"abs(gt_yaw_3s) < {a.turn_mild_deg} deg",
            "mild": (
                f"{a.turn_mild_deg} <= abs(gt_yaw_3s) < "
                f"{a.turn_strong_deg} deg"
            ),
            "strong": f"abs(gt_yaw_3s) >= {a.turn_strong_deg} deg",
            "diagnostic_only": True,
        },
        "turn_subset_moving_mIoU": {
            n: float(turn_reports[n]["mIoU"]) for n in TURN_BINS
        },
    }

    print("\n=== V17/V18 HARD A1 EVAL ===")
    print(f"arm={arm} step={ck.get('continuation_step')}")
    print(f"{'variant':20s} {'IoU':>9s} {'mIoU':>9s} {'Moving':>9s}")
    for n in names:
        rr = reports[n]
        print(
            f"{n:20s} "
            f"{float(rr['occupancy']['IoU']):9.4f} "
            f"{float(rr['overall']['mIoU']):9.4f} "
            f"{float(rr['moving']['mIoU']):9.4f}"
        )
    print(json.dumps(diag, indent=2))

    result = {
        "protocol": EVAL_PROTOCOL,
        "checkpoint": str(Path(a.checkpoint).resolve()),
        "checkpoint_protocol": ck.get("protocol"),
        "arm": arm,
        "continuation_step": int(ck.get("continuation_step", -1)),
        "num_windows": len(records),
        "reports": reports,
        "diagnostics": diag,
        "turn_reports": turn_reports,
        "se2_target_contract": SE2_TARGET_CONTRACT,
        "yaw_enabled_class_ids": list(YAW_ENABLED_CLASS_IDS),
        "a1_write_order_contract": "legacy_clear_plus_original_strong_source_write_order_v1",
        "cache_metadata": cache_meta,
    }
    op = Path(a.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"saved {op}")


if __name__ == "__main__":
    main()
