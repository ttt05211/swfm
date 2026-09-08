#!/usr/bin/env python3
"""Upgrade v12 motion-transport cache targets to the v13 displacement contract.

This reuses the expensive Strong-source six-frame features/tracks already stored
in a v12 cache.  Only t0 GT centers are read from nuScenes annotations so the
incorrect centroid-to-box-center residual target can be replaced by the correct
GT-displacement-minus-KTA-displacement target.
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

from real_motion.motion_transport import MOTION_TRANSPORT_CACHE_VERSION as V1_VERSION
from real_motion.motion_transport_v2 import (
    FEATURE_DIM,
    MOTION_TRANSPORT_CACHE_VERSION,
    TARGET_CONTRACT,
    annotation_map,
    upgrade_v1_record_targets,
    world_points_to_t0,
)
from real_motion.nuscenes_adapter import NuScenesWindowSource


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="existing p0_f9_motion_transport_v1 .pt cache")
    p.add_argument("--output", required=True, help="new v2 cache path")
    p.add_argument("--dataroot", required=True)
    p.add_argument("--info-pkl", required=True)
    a = p.parse_args()

    src_path = Path(a.input).resolve()
    dst_path = Path(a.output).resolve()
    if src_path == dst_path:
        raise ValueError("upgrade must write a new path; do not overwrite the only v1 cache")
    obj = torch.load(src_path, map_location="cpu", weights_only=False)
    if obj.get("version") != V1_VERSION:
        raise RuntimeError(f"expected {V1_VERSION}, got {obj.get('version')}")
    records = obj.get("records") or []
    if not records:
        raise RuntimeError("input cache has no records")
    meta = dict(obj.get("metadata") or {})
    if int(meta.get("feature_dim", -1)) != FEATURE_DIM:
        raise RuntimeError("feature dimension mismatch")

    source = NuScenesWindowSource(a.dataroot, info_pkl=a.info_pkl, verbose=False)
    upgraded = []
    supervised = valid_targets = 0
    for ri, rec in enumerate(records, start=1):
        t0_pose = np.asarray(source.pose(str(rec["t0_token"])), dtype=np.float64)
        ann0 = annotation_map(source.nusc, str(rec["t0_token"]))
        tokens = tuple(rec.get("source_instance_token") or ())
        if len(tokens) != int(rec["features"].shape[0]):
            raise RuntimeError(f"{rec['sample_id']}: source_instance_token length mismatch")
        gt0 = np.zeros((len(tokens), 2), dtype=np.float32)
        sup = rec["supervised_source"].bool().numpy()
        for i, token in enumerate(tokens):
            if token is None:
                if sup[i]:
                    raise RuntimeError(f"{rec['sample_id']}: supervised source lacks token")
                continue
            row = ann0.get(str(token))
            if row is None:
                raise RuntimeError(f"{rec['sample_id']}: t0 annotation token disappeared: {token}")
            gt0[i] = world_points_to_t0(
                np.asarray(row["center_world"], dtype=np.float64)[None], t0_pose
            )[0, :2].astype(np.float32)
        out = upgrade_v1_record_targets(rec, gt0)
        upgraded.append(out)
        supervised += int(out["supervised_source"].sum().item())
        valid_targets += int(out["target_valid"].sum().item())
        if ri == 1 or ri % 500 == 0 or ri == len(records):
            print(f"motion_cache_upgrade {ri}/{len(records)}")

    meta["version"] = MOTION_TRANSPORT_CACHE_VERSION
    meta["target_contract"] = TARGET_CONTRACT
    meta["upgraded_from_version"] = V1_VERSION
    meta["upgraded_from_path"] = str(src_path)
    summary = dict(meta.get("summary") or {})
    summary["num_supervised_sources"] = supervised
    summary["num_future_center_labels"] = valid_targets
    meta["summary"] = summary

    payload = {"version": MOTION_TRANSPORT_CACHE_VERSION, "metadata": meta, "records": upgraded}
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, dst_path)
    dst_path.with_suffix(".summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(dst_path),
        "version": MOTION_TRANSPORT_CACHE_VERSION,
        "target_contract": TARGET_CONTRACT,
        "num_records": len(upgraded),
        "num_supervised_sources": supervised,
        "num_future_center_labels": valid_targets,
    }, indent=2))


if __name__ == "__main__":
    main()
