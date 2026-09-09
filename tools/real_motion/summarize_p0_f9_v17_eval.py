#!/usr/bin/env python3
"""Summarize V17 rigid-transport JSON reports with occupancy IoU metrics.

The V17 evaluator already stores semantic per-class/per-horizon intersections in
its report.  This utility makes the final task metrics explicit next to trajectory
metrics:

- Overall semantic mIoU (17 occupied semantic classes, evaluator contract);
- Dynamic-mIoU (mean over the frozen dynamic classes, then report horizons);
- Moving-mIoU v2 and its 1s/2s/3s values;
- ADE/FDE and existence F1.

No metric is recomputed from predictions; the utility only aggregates the saved
evaluator report, so it is cheap and deterministic.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from real_motion.metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS, REPORT_HORIZONS_S


def _lookup(mapping, key):
    if key in mapping:
        return mapping[key]
    skey = str(key)
    if skey in mapping:
        return mapping[skey]
    fkey = str(float(key))
    if fkey in mapping:
        return mapping[fkey]
    raise KeyError(key)


def dynamic_miou_from_report(report: dict) -> tuple[float, dict[str, float]]:
    per_h = report["overall"]["per_horizon"]
    by_h = {}
    for horizon in REPORT_HORIZONS_S:
        row = _lookup(per_h, horizon)
        per_class = row["per_class"]
        vals = []
        for class_id in DYNAMIC_CLASS_IDS:
            try:
                value = float(_lookup(per_class, int(class_id)))
            except KeyError:
                continue
            if np.isfinite(value):
                vals.append(value)
        by_h[str(float(horizon))] = float(np.mean(vals)) if vals else float("nan")
    finite = [v for v in by_h.values() if np.isfinite(v)]
    return (float(np.mean(finite)) if finite else float("nan"), by_h)


def horizon_miou(report: dict, family: str, horizon: float) -> float:
    row = _lookup(report[family]["per_horizon"], horizon)
    return float(row["mIoU"])


def summarize(path: Path, branch: str) -> dict:
    obj = json.loads(path.read_text())
    if branch not in obj["reports"]:
        raise KeyError(f"{path}: missing report branch {branch}")
    report = obj["reports"][branch]
    dynamic, dynamic_h = dynamic_miou_from_report(report)
    d = obj["diagnostics"]
    return {
        "name": path.stem,
        "variant": str(obj.get("variant", "?")),
        "epoch": int(obj.get("checkpoint_epoch", -1)),
        "overall_mIoU": float(report["overall"]["mIoU"]),
        "dynamic_mIoU": dynamic,
        "moving_mIoU": float(report["moving"]["mIoU"]),
        "overall_1s": horizon_miou(report, "overall", 1.0),
        "overall_2s": horizon_miou(report, "overall", 2.0),
        "overall_3s": horizon_miou(report, "overall", 3.0),
        "dynamic_1s": dynamic_h["1.0"],
        "dynamic_2s": dynamic_h["2.0"],
        "dynamic_3s": dynamic_h["3.0"],
        "moving_1s": horizon_miou(report, "moving", 1.0),
        "moving_2s": horizon_miou(report, "moving", 2.0),
        "moving_3s": horizon_miou(report, "moving", 3.0),
        "ADE_m": float(d["learned_ade_m"]),
        "FDE_m": float(d["learned_fde_m"]),
        "exist_F1": float(d["existence_f1"]),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("json", nargs="+", help="V17 evaluator JSON files")
    p.add_argument("--branch", default="local_stwm_center_always",
                   choices=("strong_anchor", "local_stwm_center_always", "local_stwm_center_rigid", "gt_center_rigid"))
    a = p.parse_args()
    rows = [summarize(Path(x), a.branch) for x in a.json]

    print("\n=== V17 FINAL OCCUPANCY / TRAJECTORY SUMMARY ===")
    print(f"report_branch={a.branch}")
    print(
        f"{'name':28s} {'ep':>3s} {'Overall':>8s} {'Dynamic':>8s} {'Moving':>8s} "
        f"{'M@1s':>7s} {'M@2s':>7s} {'M@3s':>7s} {'ADE':>7s} {'FDE':>7s} {'ExF1':>7s}"
    )
    for r in rows:
        print(
            f"{r['name'][:28]:28s} {r['epoch']:3d} {r['overall_mIoU']:8.4f} {r['dynamic_mIoU']:8.4f} "
            f"{r['moving_mIoU']:8.4f} {r['moving_1s']:7.4f} {r['moving_2s']:7.4f} {r['moving_3s']:7.4f} "
            f"{r['ADE_m']:7.4f} {r['FDE_m']:7.4f} {r['exist_F1']:7.4f}"
        )

    print("\n=== IoU BY HORIZON ===")
    for r in rows:
        print(
            f"{r['name']}: "
            f"Overall=[{r['overall_1s']:.4f}, {r['overall_2s']:.4f}, {r['overall_3s']:.4f}] "
            f"Dynamic=[{r['dynamic_1s']:.4f}, {r['dynamic_2s']:.4f}, {r['dynamic_3s']:.4f}] "
            f"Moving=[{r['moving_1s']:.4f}, {r['moving_2s']:.4f}, {r['moving_3s']:.4f}]"
        )


if __name__ == "__main__":
    main()
