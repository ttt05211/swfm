#!/usr/bin/env python3
"""Build sparse frozen true-moving support for the final V18 Moving-Safe probe.

The source window set is the existing V17 scene-supervision cache.  Only the
frozen Moving-mIoU v2 support is added here; no model input, prediction, GT
occupancy, or Strong/KTA anchor is recomputed.
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
from real_motion.local_stwm_moving_safe import (
    MOVING_SAFE_LOSS_CONTRACT,
    MOVING_SUPPORT_CACHE_VERSION,
    REPORT_FRAME_INDICES,
)
from real_motion.local_stwm_scene_supervision import V17SceneCacheDataset
from real_motion.metrics.moving_miou_v2 import (
    Box3D,
    PROTOCOL as MOVING_PROTOCOL,
    moving_support_from_world_motion,
)
from real_motion.nuscenes_adapter import NuScenesWindowSource, category_to_dynamic_class
from real_motion.runtime_config import add_config_args, load_runtime_config, make_prepare_config


def _ego_yaw(T):
    R = np.asarray(T, dtype=np.float64)[:3, :3]
    return math.atan2(float(R[1, 0]), float(R[0, 0]))


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
            "size_lwh": np.asarray(
                [size[1], size[0], size[2]], dtype=np.float64
            ),
        }
    return out


def _box_future_ego(ann, future_pose):
    T = np.linalg.inv(np.asarray(future_pose, dtype=np.float64))
    c = (T @ np.r_[ann["center_world"], 1.0])[:3]
    return Box3D(
        token=str(ann["instance_token"]),
        class_id=int(ann["class_id"]),
        center_xyz=tuple(float(x) for x in c),
        size_lwh=tuple(float(x) for x in ann["size_lwh"]),
        yaw=float(ann["yaw_world"]) - _ego_yaw(future_pose),
    )


def main():
    p = argparse.ArgumentParser()
    add_config_args(p)
    p.add_argument("--scene-cache", required=True)
    p.add_argument("--dataroot", required=True)
    p.add_argument("--info-pkl", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()

    cfg = load_runtime_config(a.config, a.override)
    pcfg = make_prepare_config(cfg)
    ds = V17SceneCacheDataset(a.scene_cache)
    frame_dt = float(ds.metadata.get("frame_dt_s", pcfg.frame_dt_s))
    source = NuScenesWindowSource(a.dataroot, info_pkl=a.info_pkl, verbose=False)

    records = {}
    total = {int(h): 0 for h in REPORT_FRAME_INDICES}
    nonempty = {int(h): 0 for h in REPORT_FRAME_INDICES}

    for i in range(len(ds)):
        row = ds[i]
        sid = str(row["sample_id"])
        t0_token = str(row["t0_token"])
        future_tokens = tuple(str(x) for x in row["future_tokens"])
        future_poses = np.asarray(row["future_ego_to_world"], dtype=np.float64)
        if len(future_tokens) != future_poses.shape[0]:
            raise RuntimeError(f"{sid}: future token/pose count mismatch")

        ann0 = _dynamic_ann_map(source.nusc, t0_token)
        future_maps = {
            int(hi): _dynamic_ann_map(source.nusc, future_tokens[int(hi)])
            for hi in REPORT_FRAME_INDICES
        }
        supports = {}
        for hi in REPORT_FRAME_INDICES:
            support = np.zeros(pcfg.grid.shape_hwd, dtype=bool)
            amap = future_maps[int(hi)]
            pose = future_poses[int(hi)]
            dt = (int(hi) + 1) * frame_dt
            for token, a0 in ann0.items():
                ah = amap.get(token)
                if ah is None:
                    continue
                support |= moving_support_from_world_motion(
                    a0["center_world"],
                    ah["center_world"],
                    _box_future_ego(a0, pose),
                    _box_future_ego(ah, pose),
                    dt,
                    grid=pcfg.grid,
                )
            flat = np.flatnonzero(support.reshape(-1)).astype(np.int32)
            supports[int(hi)] = torch.from_numpy(flat)
            total[int(hi)] += int(len(flat))
            nonempty[int(hi)] += int(len(flat) > 0)

        records[sid] = {
            "sample_id": sid,
            "scene_name": str(row["scene_name"]),
            "moving_support_flat_by_horizon": supports,
        }
        if i == 0 or (i + 1) % 32 == 0 or i + 1 == len(ds):
            print(
                f"moving_support {i+1}/{len(ds)} sid={sid} "
                + " ".join(
                    f"h{h}={len(supports[h])}" for h in REPORT_FRAME_INDICES
                ),
                flush=True,
            )

    obj = {
        "version": MOVING_SUPPORT_CACHE_VERSION,
        "metadata": {
            "moving_safe_loss_contract": MOVING_SAFE_LOSS_CONTRACT,
            "moving_metric_protocol": MOVING_PROTOCOL,
            "source_scene_cache": str(Path(a.scene_cache).resolve()),
            "num_windows": len(records),
            "frame_dt_s": frame_dt,
            "report_frame_indices": list(REPORT_FRAME_INDICES),
            "support_voxels_total_by_frame_index": {
                str(k): int(v) for k, v in total.items()
            },
            "nonempty_windows_by_frame_index": {
                str(k): int(v) for k, v in nonempty.items()
            },
            "grid_shape_hwd": [int(x) for x in pcfg.grid.shape_hwd],
            "grid_voxel_size": [float(x) for x in pcfg.grid.voxel_size],
            "support_is_training_only_gt_signal": True,
        },
        "records": records,
    }
    op = Path(a.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    if op.exists():
        raise FileExistsError(op)
    torch.save(obj, op)
    print("=== V18 MOVING SUPPORT CACHE ===")
    print(json.dumps(obj["metadata"], indent=2))
    print(f"saved {op}")


if __name__ == "__main__":
    main()
