#!/usr/bin/env python3
"""Augment an existing V17 cache with geometrically consistent SE(2) targets.

This is intentionally a label-only upgrade.  It reuses all expensive V17 causal
features/tubes and reads only nuScenes annotations/poses needed to construct the
new source-centred SE(2) supervision.
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

from real_motion.geometry import quaternion_yaw
from real_motion.local_st_world_model_v17 import (
    LOCAL_STWM_V17_CACHE_VERSION,
    REPRESENTATION_CONTRACT,
)
from real_motion.local_st_world_model_v18_se2 import (
    SE2_CACHE_VERSION,
    SE2_TARGET_CONTRACT,
    YAW_ENABLED_CLASS_IDS,
    relative_yaw_in_t0,
    source_center_se2_target,
)
from real_motion.motion_transport import FUTURE_FRAMES, world_points_to_t0
from real_motion.motion_transport_v2 import TARGET_CONTRACT
from real_motion.nuscenes_adapter import NuScenesWindowSource


def _instance_annotation_map(nusc, sample_token):
    sample = nusc.get("sample", str(sample_token))
    out = {}
    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        out[str(ann["instance_token"])] = ann
    return out


def _center_t0(ann, t0_pose):
    p = np.asarray(ann["translation"], dtype=np.float64)[None]
    return world_points_to_t0(p, t0_pose)[0, :2]


def _upgrade_record(rec, source, *, strict=True):
    r = dict(rec)
    required = (
        "source_instance_token",
        "source_centroid_xy_t0_m",
        "kta_displacement_xy_m",
        "target_valid",
        "source_class_id",
        "t0_token",
        "future_tokens",
    )
    missing = [k for k in required if k not in r]
    if missing:
        raise RuntimeError(f"{r.get('sample_id','?')}: missing V17 fields {missing}")

    tokens = tuple(r["source_instance_token"])
    source_xy = r["source_centroid_xy_t0_m"].float().numpy().astype(np.float64)
    kta = r["kta_displacement_xy_m"].float().numpy().astype(np.float64)
    old_valid = r["target_valid"].bool().numpy()
    class_id = r["source_class_id"].long().numpy()
    n = int(source_xy.shape[0])
    if len(tokens) != n or kta.shape != (n, FUTURE_FRAMES, 2):
        raise RuntimeError(f"{r.get('sample_id','?')}: source-shape mismatch")

    t0_pose = np.asarray(source.pose(str(r["t0_token"])), dtype=np.float64)
    ann0_map = _instance_annotation_map(source.nusc, str(r["t0_token"]))
    future_maps = [
        _instance_annotation_map(source.nusc, str(tok))
        for tok in tuple(r["future_tokens"])
    ]
    if len(future_maps) != FUTURE_FRAMES:
        raise RuntimeError("SE2 target builder requires six future frames")

    target_disp = np.zeros((n, FUTURE_FRAMES, 2), dtype=np.float32)
    target_res = np.zeros_like(target_disp)
    target_yaw = np.zeros((n, FUTURE_FRAMES), dtype=np.float32)
    yaw_valid = np.zeros((n, FUTURE_FRAMES), dtype=bool)
    se2_valid = np.zeros((n, FUTURE_FRAMES), dtype=bool)
    yaw_enabled = np.isin(class_id, np.asarray(YAW_ENABLED_CLASS_IDS, dtype=np.int64))

    for i, token in enumerate(tokens):
        if token is None:
            if bool(old_valid[i].any()) and strict:
                raise RuntimeError(
                    f"{r.get('sample_id','?')}: valid target without source token at {i}"
                )
            continue
        token = str(token)
        ann0 = ann0_map.get(token)
        if ann0 is None:
            if strict:
                raise RuntimeError(
                    f"{r.get('sample_id','?')}: t0 annotation missing for {token}"
                )
            continue
        a0 = _center_t0(ann0, t0_pose)
        yaw0_world = float(quaternion_yaw(ann0["rotation"]))

        for h in range(FUTURE_FRAMES):
            annh = future_maps[h].get(token)
            if annh is None:
                if bool(old_valid[i, h]) and strict:
                    raise RuntimeError(
                        f"{r.get('sample_id','?')}: old valid target missing future ann "
                        f"source={i} horizon={h}"
                    )
                continue

            ah = _center_t0(annh, t0_pose)
            yawh_world = float(quaternion_yaw(annh["rotation"]))
            raw_dyaw = relative_yaw_in_t0(yaw0_world, yawh_world, t0_pose)

            # Classes disabled for yaw must remain the historical translation-only
            # contract in BOTH training and deployment.  Therefore their XY target
            # also uses an effective zero rotation rather than absorbing an
            # unmodelled pivot-rotation term into translation.
            effective_dyaw = raw_dyaw if bool(yaw_enabled[i]) else 0.0
            tgt = source_center_se2_target(source_xy[i], a0, ah, effective_dyaw)

            target_disp[i, h] = tgt.source_displacement_xy_m
            target_res[i, h] = (
                tgt.source_displacement_xy_m.astype(np.float64) - kta[i, h]
            ).astype(np.float32)
            target_yaw[i, h] = np.float32(raw_dyaw)
            yaw_valid[i, h] = True
            se2_valid[i, h] = True

    # The new supervision must not silently create labels where V17 had none.
    if strict and bool(np.any(se2_valid != old_valid)):
        bad = np.argwhere(se2_valid != old_valid)[:8].tolist()
        raise RuntimeError(
            f"{r.get('sample_id','?')}: SE2/legacy validity mismatch at {bad}"
        )
    se2_valid &= old_valid
    yaw_valid &= old_valid

    r["target_source_displacement_xy_m"] = torch.from_numpy(target_disp)
    r["target_source_residual_xy_m"] = torch.from_numpy(target_res)
    r["target_yaw_rad"] = torch.from_numpy(target_yaw)
    r["yaw_label_valid"] = torch.from_numpy(yaw_valid)
    r["se2_target_valid"] = torch.from_numpy(se2_valid)
    r["yaw_enabled"] = torch.from_numpy(yaw_enabled)
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="existing V17 cache")
    p.add_argument("--output", required=True, help="new label-augmented cache")
    p.add_argument("--dataroot", required=True)
    p.add_argument("--info-pkl", required=True)
    p.add_argument("--max-windows", type=int, default=0)
    p.add_argument("--non-strict", action="store_true")
    a = p.parse_args()

    src = torch.load(a.input, map_location="cpu", weights_only=False)
    if src.get("version") != LOCAL_STWM_V17_CACHE_VERSION:
        raise RuntimeError(
            f"expected V17 cache version {LOCAL_STWM_V17_CACHE_VERSION}, got "
            f"{src.get('version')}"
        )
    meta = dict(src.get("metadata") or {})
    if meta.get("target_contract") != TARGET_CONTRACT:
        raise RuntimeError("legacy XY target contract mismatch")
    if meta.get("representation_contract") != REPRESENTATION_CONTRACT:
        raise RuntimeError("V17 representation contract mismatch")
    records = list(src.get("records") or [])
    if not records:
        raise RuntimeError("input cache has no records")
    if int(a.max_windows) > 0:
        records = records[: min(len(records), int(a.max_windows))]

    source = NuScenesWindowSource(
        a.dataroot, info_pkl=a.info_pkl, verbose=False
    )
    out_records = []
    nsrc = nvalid = nyaw = nyaw_src = 0
    for wi, rec in enumerate(records, start=1):
        out = _upgrade_record(rec, source, strict=not bool(a.non_strict))
        out_records.append(out)
        nsrc += int(out["features"].shape[0])
        nvalid += int(out["se2_target_valid"].sum().item())
        nyaw += int(
            (
                out["yaw_label_valid"]
                & out["yaw_enabled"][:, None]
            ).sum().item()
        )
        nyaw_src += int(out["yaw_enabled"].sum().item())
        if wi == 1 or wi % 500 == 0 or wi == len(records):
            print(
                f"se2_label_upgrade {wi}/{len(records)} "
                f"sources={nsrc} labels={nvalid} yaw_labels={nyaw}",
                flush=True,
            )

    new_meta = dict(meta)
    new_meta.update(
        {
            "version": SE2_CACHE_VERSION,
            "se2_augmented_from_version": LOCAL_STWM_V17_CACHE_VERSION,
            "se2_augmented_from_path": str(Path(a.input).resolve()),
            "se2_target_contract": SE2_TARGET_CONTRACT,
            "se2_target_frame": "frozen_t0_ego",
            "se2_rotation_pivot": "strong_source_centroid",
            "se2_gt_motion_source": "matched_nuscenes_annotation_relative_rigid_transform",
            "yaw_encoding": "scalar_relative_yaw_rad",
            "yaw_enabled_class_ids": list(YAW_ENABLED_CLASS_IDS),
            "yaw_disabled_xy_contract": "legacy_translation_only_box_center_displacement",
            "yaw_deployment_rule": "semantic_class_only_no_future_validity_gate",
            "num_windows": len(out_records),
            "num_sources": nsrc,
            "num_se2_target_labels": nvalid,
            "num_yaw_enabled_sources": nyaw_src,
            "num_yaw_supervision_labels": nyaw,
        }
    )
    payload = {
        "version": SE2_CACHE_VERSION,
        "metadata": new_meta,
        "records": out_records,
    }
    op = Path(a.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, op)
    op.with_suffix(".summary.json").write_text(
        json.dumps(new_meta, indent=2), encoding="utf-8"
    )
    print(json.dumps(new_meta, indent=2))
    print(f"saved {op}")


if __name__ == "__main__":
    main()
