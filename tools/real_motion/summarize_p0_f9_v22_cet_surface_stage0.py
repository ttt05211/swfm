#!/usr/bin/env python3
"""Print the compact decision table from a CET surface Stage-0 JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _percent(value):
    return "None" if value is None else f"{100.0 * float(value):.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    obj = json.loads(Path(args.input).read_text(encoding="utf-8"))
    print("===== CET STATIC SURFACE STAGE-0 =====")
    print("protocol:", obj["protocol"])
    print("windows:", obj["population"]["windows"])
    print("\n===== ALL CONFIGURATIONS =====")
    grouped = {}
    for row in obj["comparison_table"]:
        grouped.setdefault(row["configuration"], {})[row["method"]] = row
    for configuration, methods in grouped.items():
        nearest = methods["NEAREST_COLUMN"]
        tangent = methods["TANGENT_PLANE"]
        support = nearest["support"]
        print(
            f"{configuration:52s} "
            f"scope={nearest['scope_gt_delta_mIoU_pp']:+.5f} "
            f"causal={nearest['causal_gt_delta_mIoU_pp']:+.5f} "
            f"retain={_percent(nearest['causal_gt_retention_of_scope']):>7s} "
            f"GTgeo+HistSem={nearest['gt_geometry_history_semantic_delta_mIoU_pp']:+.5f} "
            f"Zoracle={nearest['oracle_z_shift_history_semantic_delta_mIoU_pp']:+.5f} "
            f"Near+GTsem={nearest['nearest_geometry_gt_semantic_delta_mIoU_pp']:+.5f} "
            f"nearest={nearest['deterministic_delta_mIoU_pp']:+.5f} "
            f"tangent={tangent['deterministic_delta_mIoU_pp']:+.5f}"
        )
    maximum = obj["maximum_scope_headroom"]
    z_shift = obj["best_oracle_vertical_shift"]
    best = obj["best_deterministic"]
    scene = obj["best_deterministic_scene_delta_mIoU_pp"]
    print("\n===== MAXIMUM STATIC SCOPE =====")
    print("configuration:", maximum["configuration"])
    print("scope_gt_delta_mIoU_pp:", maximum["scope_gt_delta_mIoU_pp"])
    print("causal_gt_delta_mIoU_pp:", maximum["causal_gt_delta_mIoU_pp"])
    print("causal_gt_retention:", maximum["causal_gt_retention_of_scope"])
    print(
        "gt_geometry_history_semantic_delta_mIoU_pp:",
        maximum["gt_geometry_history_semantic_delta_mIoU_pp"],
    )
    print("\n===== BEST ORACLE Z SHIFT =====")
    print("configuration:", z_shift["configuration"])
    print(
        "oracle_z_shift_history_semantic_delta_mIoU_pp:",
        z_shift["oracle_z_shift_history_semantic_delta_mIoU_pp"],
    )
    print("\n===== BEST DETERMINISTIC =====")
    print("variant:", best["variant"])
    print("scope_gt_delta_mIoU_pp:", best["scope_gt_delta_mIoU_pp"])
    print("causal_gt_delta_mIoU_pp:", best["causal_gt_delta_mIoU_pp"])
    print("causal_gt_retention:", best["causal_gt_retention_of_scope"])
    print("deterministic_delta_mIoU_pp:", best["deterministic_delta_mIoU_pp"])
    print("deterministic_delta_IoU_pp:", best["deterministic_delta_IoU_pp"])
    print("addition_quality:", best["addition_quality"])
    print("scene_delta:", scene)
    print("\n===== DECISION =====")
    print("gate:", obj["stage0_gate"])
    print("route:", obj["recommended_next_route"])
    print("performance:", obj["performance"])


if __name__ == "__main__":
    main()
