#!/usr/bin/env python3
"""Audit train/dev future-query OOB for a candidate V20 Ωmax lattice."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import yaml

from real_motion.v20_history_world import CanonicalLattice, future_union_query_mask, poses_to_t0_canonical

PROTOCOL = "p0_f9_v20_query_extent_audit_v1"


def _lattice_from_config(path):
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    lc = cfg["canonical_lattice"]
    return CanonicalLattice(
        tuple(float(x) for x in lc["origin_xyz_m"]),
        tuple(float(x) for x in lc["voxel_size_xyz_m"]),
        tuple(int(x) for x in lc["shape_xyz"]),
    )


def _lattice_from_scan(path):
    r = json.loads(Path(path).read_text(encoding="utf-8"))
    if r.get("protocol") != "p0_f9_v20_query_extent_scan_v1":
        raise RuntimeError("unexpected extent scan protocol")
    return CanonicalLattice(
        tuple(float(x) for x in r["recommended_origin_xyz_m"]),
        tuple(float(x) for x in r["voxel_size_xyz_m"]),
        tuple(int(x) for x in r["recommended_shape_xyz"]),
    )


def _parse3(text, typ=float):
    vals = tuple(typ(x) for x in str(text).split(","))
    if len(vals) != 3:
        raise ValueError("expected xyz triple")
    return vals


def main():
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--v20-config")
    src.add_argument("--scan-report")
    p.add_argument("--windows-jsonl", required=True)
    p.add_argument("--native-shape", default="200,200,16")
    p.add_argument("--native-origin-m", default="-40,-40,-1")
    p.add_argument("--voxel-size-m", default="0.4,0.4,0.4")
    p.add_argument("--output", required=True)
    p.add_argument("--require-zero-oob", action="store_true")
    p.add_argument("--worst-k", type=int, default=20)
    a = p.parse_args()

    lattice = (
        _lattice_from_config(a.v20_config)
        if a.v20_config else _lattice_from_scan(a.scan_report)
    )
    native_shape = _parse3(a.native_shape, int)
    native_origin = _parse3(a.native_origin_m, float)
    native_step = _parse3(a.voxel_size_m, float)

    windows = 0
    requested = in_bounds = oob = 0
    rows = []
    scenes = set()
    with Path(a.windows_jsonl).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            t0 = np.asarray(row["t0_ego_to_world"], dtype=np.float64)
            future = np.asarray(row["future_ego_to_world"], dtype=np.float64)
            rel = poses_to_t0_canonical(future, t0)
            q = future_union_query_mask(
                lattice,
                future_ego_to_canonical=rel,
                native_shape_xyz=native_shape,
                native_origin_xyz_m=native_origin,
                native_voxel_size_xyz_m=native_step,
            )
            windows += 1
            requested += int(q.requested_voxels)
            in_bounds += int(q.in_bounds_voxels)
            oob += int(q.out_of_bounds_voxels)
            scenes.add(str(row.get("scene", "")))
            rows.append({
                "scene": str(row.get("scene", "")),
                "t0_token": str(row.get("t0_token", "")),
                "out_of_bounds_voxels": int(q.out_of_bounds_voxels),
                "out_of_bounds_fraction": float(q.out_of_bounds_fraction),
            })
    if windows == 0:
        raise RuntimeError("no pose windows")
    rows.sort(
        key=lambda x: (
            -float(x["out_of_bounds_fraction"]),
            -int(x["out_of_bounds_voxels"]),
            x["scene"],
            x["t0_token"],
        )
    )
    report = {
        "protocol": PROTOCOL,
        "windows": int(windows),
        "scenes": int(len(scenes)),
        "lattice": {
            "origin_xyz_m": list(lattice.origin_xyz_m),
            "voxel_size_xyz_m": list(lattice.voxel_size_xyz_m),
            "shape_xyz": list(lattice.shape_xyz),
            "max_xyz_m": lattice.max_xyz_m.tolist(),
        },
        "requested_voxels": int(requested),
        "in_bounds_voxels": int(in_bounds),
        "out_of_bounds_voxels": int(oob),
        "out_of_bounds_fraction": float(oob / max(requested, 1)),
        "windows_with_oob": int(sum(int(x["out_of_bounds_voxels"]) > 0 for x in rows)),
        "worst_windows": rows[: max(0, int(a.worst_k))],
    }
    op = Path(a.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if a.require_zero_oob and oob:
        raise SystemExit(
            f"candidate Ωmax has {oob} future-query OOB voxels; refusing success"
        )


if __name__ == "__main__":
    main()
