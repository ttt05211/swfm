#!/usr/bin/env python3
"""Fast containment audit for a candidate V20 Omega-max lattice.

The scan itself is defined by rigidly transforming the 8 outer corners of the
native OCC box.  For a rigid/affine transform, every coordinate extremum of the
box is attained at a corner.  Therefore corner containment is sufficient to
prove that *all* native voxel centers are in bounds.

This audit intentionally does not rasterize 200x200x16 voxels for every view.
If containment fails, it reports the offending windows/views and exits; exact
OOB voxel counts are unnecessary for deciding whether Omega-max may be frozen.
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
import yaml

from real_motion.v20_history_world import (
    CanonicalLattice,
    poses_to_t0_canonical,
    transform_points,
)

PROTOCOL = "p0_f9_v20_query_extent_audit_v3"


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
    if r.get("protocol") != "p0_f9_v20_query_extent_scan_v2":
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


def _outer_corners(native_shape, native_origin, native_step):
    shape = np.asarray(native_shape, dtype=np.float64)
    origin = np.asarray(native_origin, dtype=np.float64)
    step = np.asarray(native_step, dtype=np.float64)
    lo = origin
    hi = origin + shape * step
    return np.asarray(
        [
            [x, y, z]
            for x in (lo[0], hi[0])
            for y in (lo[1], hi[1])
            for z in (lo[2], hi[2])
        ],
        dtype=np.float64,
    )


def _audit_views(poses, corners, lattice):
    lo = np.asarray(lattice.origin_xyz_m, dtype=np.float64)
    hi = np.asarray(lattice.max_xyz_m, dtype=np.float64)
    failing = 0
    min_clearance = np.full(3, np.inf, dtype=np.float64)
    worst_violation = 0.0
    for T in np.asarray(poses, dtype=np.float64):
        world = transform_points(T, corners)
        wmin = world.min(axis=0)
        wmax = world.max(axis=0)
        clearance = np.minimum(wmin - lo, hi - wmax)
        min_clearance = np.minimum(min_clearance, clearance)
        violation = np.maximum(-clearance, 0.0)
        if bool((violation > 1e-9).any()):
            failing += 1
            worst_violation = max(worst_violation, float(violation.max()))
    return {
        "views": int(len(poses)),
        "failing_views": int(failing),
        "min_clearance_xyz_m": min_clearance.tolist(),
        "min_clearance_m": float(min_clearance.min()),
        "worst_violation_m": float(worst_violation),
    }


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
    p.add_argument("--progress-every", type=int, default=5000)
    a = p.parse_args()

    lattice = (
        _lattice_from_config(a.v20_config)
        if a.v20_config else _lattice_from_scan(a.scan_report)
    )
    native_shape = _parse3(a.native_shape, int)
    native_origin = _parse3(a.native_origin_m, float)
    native_step = _parse3(a.voxel_size_m, float)
    corners = _outer_corners(native_shape, native_origin, native_step)

    windows = 0
    scenes = set()
    future_fail_windows = 0
    history_fail_windows = 0
    future_fail_views = 0
    history_fail_views = 0
    future_min_clearance = np.full(3, np.inf, dtype=np.float64)
    history_min_clearance = np.full(3, np.inf, dtype=np.float64)
    future_worst_violation = 0.0
    history_worst_violation = 0.0
    worst_rows = []

    with Path(a.windows_jsonl).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            t0 = np.asarray(row["t0_ego_to_world"], dtype=np.float64)
            history = np.asarray(row["history_ego_to_world"], dtype=np.float64)
            future = np.asarray(row["future_ego_to_world"], dtype=np.float64)
            hist_rel = poses_to_t0_canonical(history, t0)
            future_rel = poses_to_t0_canonical(future, t0)

            ha = _audit_views(hist_rel, corners, lattice)
            fa = _audit_views(future_rel, corners, lattice)

            windows += 1
            scene = str(row.get("scene", ""))
            scenes.add(scene)
            history_fail_views += int(ha["failing_views"])
            future_fail_views += int(fa["failing_views"])
            history_min_clearance = np.minimum(
                history_min_clearance,
                np.asarray(ha["min_clearance_xyz_m"], dtype=np.float64),
            )
            future_min_clearance = np.minimum(
                future_min_clearance,
                np.asarray(fa["min_clearance_xyz_m"], dtype=np.float64),
            )
            history_worst_violation = max(
                history_worst_violation, float(ha["worst_violation_m"])
            )
            future_worst_violation = max(
                future_worst_violation, float(fa["worst_violation_m"])
            )

            failed = False
            if int(ha["failing_views"]) > 0:
                history_fail_windows += 1
                failed = True
            if int(fa["failing_views"]) > 0:
                future_fail_windows += 1
                failed = True
            if failed:
                worst_rows.append({
                    "scene": scene,
                    "t0_token": str(row.get("t0_token", "")),
                    "history_failing_views": int(ha["failing_views"]),
                    "future_failing_views": int(fa["failing_views"]),
                    "history_min_clearance_m": float(ha["min_clearance_m"]),
                    "future_min_clearance_m": float(fa["min_clearance_m"]),
                    "history_worst_violation_m": float(ha["worst_violation_m"]),
                    "future_worst_violation_m": float(fa["worst_violation_m"]),
                })

            if (
                windows == 1
                or (int(a.progress_every) > 0 and windows % int(a.progress_every) == 0)
            ):
                print(
                    f"v20_extent_audit {windows} windows "
                    f"history_fail={history_fail_windows} "
                    f"future_fail={future_fail_windows}",
                    flush=True,
                )

    if windows == 0:
        raise RuntimeError("no pose windows")

    worst_rows.sort(
        key=lambda x: (
            -max(
                float(x["history_worst_violation_m"]),
                float(x["future_worst_violation_m"]),
            ),
            x["scene"],
            x["t0_token"],
        )
    )
    history_ok = history_fail_windows == 0
    future_ok = future_fail_windows == 0

    report = {
        "protocol": PROTOCOL,
        "audit_method": "rigid_outer_box_8_corner_containment",
        "windows": int(windows),
        "scenes": int(len(scenes)),
        "lattice": {
            "origin_xyz_m": list(lattice.origin_xyz_m),
            "voxel_size_xyz_m": list(lattice.voxel_size_xyz_m),
            "shape_xyz": list(lattice.shape_xyz),
            "max_xyz_m": lattice.max_xyz_m.tolist(),
        },
        "future_query": {
            "views": int(windows * 6),
            "views_with_oob": int(future_fail_views),
            "windows_with_oob": int(future_fail_windows),
            "out_of_bounds_voxels": 0 if future_ok else None,
            "out_of_bounds_fraction": 0.0 if future_ok else None,
            "min_clearance_xyz_m": future_min_clearance.tolist(),
            "min_clearance_m": float(future_min_clearance.min()),
            "worst_violation_m": float(future_worst_violation),
        },
        "history_view": {
            "views": int(windows * 6),
            "views_with_oob": int(history_fail_views),
            "windows_with_oob": int(history_fail_windows),
            "out_of_bounds_voxels": 0 if history_ok else None,
            "out_of_bounds_fraction": 0.0 if history_ok else None,
            "min_clearance_xyz_m": history_min_clearance.tolist(),
            "min_clearance_m": float(history_min_clearance.min()),
            "worst_violation_m": float(history_worst_violation),
        },
        "worst_windows": worst_rows[: max(0, int(a.worst_k))],
    }
    op = Path(a.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

    if a.require_zero_oob and not (history_ok and future_ok):
        raise SystemExit(
            "candidate Omega-max containment failed: "
            f"history_windows={history_fail_windows}, "
            f"future_windows={future_fail_windows}; refusing success"
        )


if __name__ == "__main__":
    main()
