#!/usr/bin/env python3
"""Fixed-time summary for the V17 C-C vs C-S scene-supervision experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

BRANCH = "local_stwm_center_always_source_order"


def _load(path: str):
    x = json.loads(Path(path).read_text(encoding="utf-8"))
    reports = x.get("reports") or {}
    if BRANCH not in reports:
        raise RuntimeError(f"{path}: missing required A1 source-order branch {BRANCH}")
    r = reports[BRANCH]
    occ = r.get("occupancy") or {}
    if "IoU" not in occ:
        raise RuntimeError(f"{path}: evaluator JSON lacks true occupancy IoU")
    return {
        "path": str(path),
        "IoU": float(occ["IoU"]),
        "mIoU": float(r["overall"]["mIoU"]),
        "Moving": float(r["moving"]["mIoU"]),
        "checkpoint_epoch": x.get("checkpoint_epoch"),
    }


def _delta(a, b):
    return {k: float(a[k] - b[k]) for k in ("IoU", "mIoU", "Moving")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True)
    p.add_argument("--control-mid", required=True)
    p.add_argument("--scene-mid", required=True)
    p.add_argument("--control-end", required=True)
    p.add_argument("--scene-end", required=True)
    p.add_argument("--output", default="")
    a = p.parse_args()

    rows = {
        "shared_start": _load(a.start),
        "C-C_mid": _load(a.control_mid),
        "C-S_mid": _load(a.scene_mid),
        "C-C_end": _load(a.control_end),
        "C-S_end": _load(a.scene_end),
    }
    out = {
        "report_branch": BRANCH,
        "rows": rows,
        "scene_minus_control": {
            "mid": _delta(rows["C-S_mid"], rows["C-C_mid"]),
            "end": _delta(rows["C-S_end"], rows["C-C_end"]),
        },
        "scene_minus_start": {
            "mid": _delta(rows["C-S_mid"], rows["shared_start"]),
            "end": _delta(rows["C-S_end"], rows["shared_start"]),
        },
    }

    print("=== V17 SCENE SUPERVISION PAIRED FIXED-TIME COMPARISON ===")
    print(f"report_branch={BRANCH}")
    for name, r in rows.items():
        print(f"{name:16s} IoU={r['IoU']:8.4f} mIoU={r['mIoU']:8.4f} Moving={r['Moving']:8.4f}")
    print("\n=== C-S minus C-C ===")
    for key in ("mid", "end"):
        d = out["scene_minus_control"][key]
        print(f"{key:4s} dIoU={d['IoU']:+8.4f} dmIoU={d['mIoU']:+8.4f} dMoving={d['Moving']:+8.4f}")
    print("\n=== C-S minus shared RL-epoch5+A1 start ===")
    for key in ("mid", "end"):
        d = out["scene_minus_start"][key]
        print(f"{key:4s} dIoU={d['IoU']:+8.4f} dmIoU={d['mIoU']:+8.4f} dMoving={d['Moving']:+8.4f}")
    if a.output:
        Path(a.output).parent.mkdir(parents=True, exist_ok=True)
        Path(a.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"saved {a.output}")


if __name__ == "__main__":
    main()
