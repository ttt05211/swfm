#!/usr/bin/env python3
"""Summarize V17 rigid-transport evaluation JSONs.

Metrics are kept explicitly distinct:
- IoU: binary occupied-vs-free occupancy IoU from the evaluator;
- mIoU: 17-class semantic occupancy mIoU (free excluded);
- Dynamic-mIoU: semantic mIoU over the frozen dynamic classes;
- Moving-mIoU v2: semantic mIoU restricted to frozen true-moving support;
- ADE/FDE and existence F1.

Legacy V17 JSONs produced before evaluator protocol v2 do not contain binary
occupancy IoU and are rejected instead of silently relabeling semantic mIoU.
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


def horizon_metric(report: dict, family: str, horizon: float, field: str) -> float:
    row = _lookup(report[family]["per_horizon"], horizon)
    return float(row[field])


def summarize(path: Path, branch: str) -> dict:
    obj = json.loads(path.read_text())
    if branch not in obj["reports"]:
        raise KeyError(f"{path}: missing report branch {branch}")
    report = obj["reports"][branch]
    if "occupancy" not in report:
        raise RuntimeError(
            f"{path}: missing true binary occupancy IoU. "
            "This is a legacy V17 evaluator JSON; rerun eval_p0_f9_v17_local_stwm.py."
        )
    dynamic, dynamic_h = dynamic_miou_from_report(report)
    d = obj["diagnostics"]
    return {
        "name": path.stem,
        "variant": str(obj.get("variant", "?")),
        "epoch": int(obj.get("checkpoint_epoch", -1)),
        "occupancy_IoU": float(report["occupancy"]["IoU"]),
        "semantic_mIoU": float(report["overall"]["mIoU"]),
        "dynamic_mIoU": dynamic,
        "moving_mIoU": float(report["moving"]["mIoU"]),
        "iou_1s": horizon_metric(report, "occupancy", 1.0, "IoU"),
        "iou_2s": horizon_metric(report, "occupancy", 2.0, "IoU"),
        "iou_3s": horizon_metric(report, "occupancy", 3.0, "IoU"),
        "miou_1s": horizon_metric(report, "overall", 1.0, "mIoU"),
        "miou_2s": horizon_metric(report, "overall", 2.0, "mIoU"),
        "miou_3s": horizon_metric(report, "overall", 3.0, "mIoU"),
        "dynamic_1s": dynamic_h["1.0"],
        "dynamic_2s": dynamic_h["2.0"],
        "dynamic_3s": dynamic_h["3.0"],
        "moving_1s": horizon_metric(report, "moving", 1.0, "mIoU"),
        "moving_2s": horizon_metric(report, "moving", 2.0, "mIoU"),
        "moving_3s": horizon_metric(report, "moving", 3.0, "mIoU"),
        "ADE_m": float(d["learned_ade_m"]),
        "FDE_m": float(d["learned_fde_m"]),
        "exist_F1": float(d["existence_f1"]),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("json", nargs="+", help="V17 evaluator v2 JSON files")
    p.add_argument(
        "--branch",
        default="local_stwm_center_always",
        choices=(
            "strong_anchor",
            "local_stwm_center_always",
            "local_stwm_center_rigid",
            "gt_center_rigid",
        ),
    )
    a = p.parse_args()
    rows = [summarize(Path(x), a.branch) for x in a.json]

    print("\n=== V17 FINAL OCCUPANCY / TRAJECTORY SUMMARY ===")
    print(f"report_branch={a.branch}")
    print(
        f"{'name':28s} {'ep':>3s} {'IoU':>8s} {'mIoU':>8s} {'Dynamic':>8s} {'Moving':>8s} "
        f"{'M@1s':>7s} {'M@2s':>7s} {'M@3s':>7s} {'ADE':>7s} {'FDE':>7s} {'ExF1':>7s}"
    )
    for r in rows:
        print(
            f"{r['name'][:28]:28s} {r['epoch']:3d} {r['occupancy_IoU']:8.4f} "
            f"{r['semantic_mIoU']:8.4f} {r['dynamic_mIoU']:8.4f} {r['moving_mIoU']:8.4f} "
            f"{r['moving_1s']:7.4f} {r['moving_2s']:7.4f} {r['moving_3s']:7.4f} "
            f"{r['ADE_m']:7.4f} {r['FDE_m']:7.4f} {r['exist_F1']:7.4f}"
        )

    print("\n=== METRICS BY HORIZON ===")
    for r in rows:
        print(
            f"{r['name']}: "
            f"IoU=[{r['iou_1s']:.4f}, {r['iou_2s']:.4f}, {r['iou_3s']:.4f}] "
            f"mIoU=[{r['miou_1s']:.4f}, {r['miou_2s']:.4f}, {r['miou_3s']:.4f}] "
            f"Dynamic=[{r['dynamic_1s']:.4f}, {r['dynamic_2s']:.4f}, {r['dynamic_3s']:.4f}] "
            f"Moving=[{r['moving_1s']:.4f}, {r['moving_2s']:.4f}, {r['moving_3s']:.4f}]"
        )


if __name__ == "__main__":
    main()
