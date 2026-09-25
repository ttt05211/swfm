#!/usr/bin/env python3
"""Scan future-union geometric extent before freezing V20 Ωmax.

Input JSONL rows must contain:
  {"scene": "...", "future_ego_to_world": [[[...4x4...]], ... six ...]}

Only future ego poses are consumed. No future semantic/mask/identity is read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from real_motion.motion_transport import FUTURE_FRAMES
from real_motion.v20_history_world import transform_points


def _parse3(text, typ=float):
    vals = tuple(typ(x) for x in text.split(","))
    if len(vals) != 3:
        raise ValueError("expected x,y,z")
    return vals


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--windows-jsonl", required=True)
    p.add_argument("--native-shape", default="200,200,16")
    p.add_argument("--native-origin-m", default="-40,-40,-1")
    p.add_argument("--voxel-size-m", default="0.4,0.4,0.4")
    p.add_argument("--margin-m", type=float, default=2.0)
    p.add_argument("--output", required=True)
    a = p.parse_args()

    shape = np.asarray(_parse3(a.native_shape, int), dtype=np.int64)
    origin = np.asarray(_parse3(a.native_origin_m, float), dtype=np.float64)
    step = np.asarray(_parse3(a.voxel_size_m, float), dtype=np.float64)
    if np.any(shape <= 0) or np.any(step <= 0):
        raise ValueError("invalid grid")
    # Outer grid corners, sufficient for a rigidly transformed box extent.
    lo = origin
    hi = origin + shape * step
    corners = np.asarray(
        [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])],
        dtype=np.float64,
    )
    mins = np.full(3, np.inf)
    maxs = np.full(3, -np.inf)
    windows = 0
    scenes = set()
    max_abs_yaw_proxy = 0.0
    with Path(a.windows_jsonl).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            poses = np.asarray(row["future_ego_to_world"], dtype=np.float64)
            if poses.shape != (FUTURE_FRAMES, 4, 4):
                raise ValueError("future_ego_to_world must be [6,4,4]")
            for T in poses:
                w = transform_points(T, corners)
                mins = np.minimum(mins, w.min(axis=0))
                maxs = np.maximum(maxs, w.max(axis=0))
                # Diagnostic only: planar rotation angle magnitude.
                max_abs_yaw_proxy = max(max_abs_yaw_proxy, abs(float(np.arctan2(T[1,0], T[0,0]))))
            windows += 1
            if row.get("scene") is not None:
                scenes.add(str(row["scene"]))
    if windows == 0:
        raise RuntimeError("no windows")
    margin = float(a.margin_m)
    mins -= margin
    maxs += margin
    size = np.ceil((maxs - mins) / step).astype(np.int64)
    report = {
        "protocol": "p0_f9_v20_query_extent_scan_v1",
        "windows": windows,
        "scenes": len(scenes),
        "native_shape_xyz": shape.tolist(),
        "native_origin_xyz_m": origin.tolist(),
        "voxel_size_xyz_m": step.tolist(),
        "margin_m": margin,
        "observed_world_min_xyz_m": mins.tolist(),
        "observed_world_max_xyz_m": maxs.tolist(),
        "recommended_origin_xyz_m": mins.tolist(),
        "recommended_shape_xyz": size.tolist(),
        "recommended_max_xyz_m": (mins + size * step).tolist(),
        "max_abs_planar_pose_yaw_rad": max_abs_yaw_proxy,
    }
    Path(a.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
