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
    for row in obj["comparison_table"]:
        q = row["addition_quality"]
        support = row["support"]
        print(
            f"{row['configuration']:31s} {row['method']:14s} "
            f"scope={row['scope_gt_delta_mIoU_pp']:+.6f} "
            f"causal={row['causal_gt_delta_mIoU_pp']:+.6f} "
            f"retain={_percent(row['causal_gt_retention_of_scope']):>8s} "
            f"actual={row['deterministic_delta_mIoU_pp']:+.6f} "
            f"precision={_percent(q['addition_occupancy_precision']):>8s} "
            f"target_recall={_percent(q['scope_target_semantic_recall']):>8s} "
            f"positive_bev={_percent(support['causal_positive_bev_fraction']):>8s}"
        )
    best = obj["best_deterministic"]
    scene = obj["best_deterministic_scene_delta_mIoU_pp"]
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
