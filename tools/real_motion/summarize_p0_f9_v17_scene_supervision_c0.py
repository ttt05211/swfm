#!/usr/bin/env python3
"""Summarize C0 control fidelity and the short C0-C/C0-S pair."""
from __future__ import annotations
import argparse, json
from pathlib import Path

BRANCH = "local_stwm_center_always_source_order"


def _load(path):
    x = json.loads(Path(path).read_text(encoding="utf-8"))
    r = (x.get("reports") or {}).get(BRANCH)
    if r is None:
        raise RuntimeError(f"{path}: missing {BRANCH}")
    occ = r.get("occupancy") or {}
    return {
        "path": str(path),
        "IoU": float(occ["IoU"]),
        "mIoU": float(r["overall"]["mIoU"]),
        "Moving": float(r["moving"]["mIoU"]),
    }


def _delta(a, b):
    return {k: float(a[k] - b[k]) for k in ("IoU", "mIoU", "Moving")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True)
    p.add_argument("--historical-control-e8", required=True)
    p.add_argument("--c0-control-e8", required=True)
    p.add_argument("--control-mid", required=True)
    p.add_argument("--scene-mid", required=True)
    p.add_argument("--control-end", required=True)
    p.add_argument("--scene-end", required=True)
    p.add_argument("--output", default="")
    a = p.parse_args()

    rows = {
        "shared_start": _load(a.start),
        "historical_BC_e8": _load(a.historical_control_e8),
        "C0C_e8": _load(a.c0_control_e8),
        "C0C_step300": _load(a.control_mid),
        "C0S_step300": _load(a.scene_mid),
        "C0C_step600": _load(a.control_end),
        "C0S_step600": _load(a.scene_end),
    }
    out = {
        "report_branch": BRANCH,
        "rows": rows,
        "control_fidelity_C0C_minus_historical_BC_e8": _delta(rows["C0C_e8"], rows["historical_BC_e8"]),
        "scene_minus_control": {
            "step300": _delta(rows["C0S_step300"], rows["C0C_step300"]),
            "step600": _delta(rows["C0S_step600"], rows["C0C_step600"]),
        },
        "scene_minus_start": {
            "step300": _delta(rows["C0S_step300"], rows["shared_start"]),
            "step600": _delta(rows["C0S_step600"], rows["shared_start"]),
        },
    }

    print("=== V17 C0 CONTROL FIDELITY + SCENE PAIR ===")
    print(f"report_branch={BRANCH}")
    for name, r in rows.items():
        print(f"{name:18s} IoU={r['IoU']:8.4f} mIoU={r['mIoU']:8.4f} Moving={r['Moving']:8.4f}")
    d = out["control_fidelity_C0C_minus_historical_BC_e8"]
    print("\n=== C0-C e8 minus historical B-C e8 ===")
    print(f"dIoU={d['IoU']:+8.4f} dmIoU={d['mIoU']:+8.4f} dMoving={d['Moving']:+8.4f}")
    print("\n=== C0-S minus C0-C ===")
    for key in ("step300", "step600"):
        d = out["scene_minus_control"][key]
        print(f"{key:7s} dIoU={d['IoU']:+8.4f} dmIoU={d['mIoU']:+8.4f} dMoving={d['Moving']:+8.4f}")
    print("\n=== C0-S minus shared start ===")
    for key in ("step300", "step600"):
        d = out["scene_minus_start"][key]
        print(f"{key:7s} dIoU={d['IoU']:+8.4f} dmIoU={d['mIoU']:+8.4f} dMoving={d['Moving']:+8.4f}")

    if a.output:
        op = Path(a.output)
        op.parent.mkdir(parents=True, exist_ok=True)
        op.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"saved {op}")


if __name__ == "__main__":
    main()
