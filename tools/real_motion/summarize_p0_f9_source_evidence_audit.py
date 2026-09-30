#!/usr/bin/env python3
"""Print a compact evidence audit summary; no Torch or nuScenes required."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def number(value):
    return "N/A" if value is None else f"{value:+.6f}"


def summarize(result):
    if result.get("protocol") != "p0_f9_source_evidence_audit_v1":
        raise ValueError("not a source evidence audit v1 artifact")
    print("===== SOURCE EVIDENCE AUDIT =====")
    print("windows:", result["windows"])
    print("commit:", result["git_commit"])
    print("population:", result["selected_key_fingerprint"])
    print("\n===== ONE-PASS COUNTERFACTUALS (pp) =====")
    for name, row in result["variants"].items():
        d, q = row["delta_vs_v18_pp"], row["edit_quality"]
        precision = q.get("addition_semantic_precision")
        precision = "N/A" if precision is None else f"{100 * precision:.2f}%"
        print(f"{name:40s} dMiOU={number(d['mIoU'])} dIoU={number(d['IoU'])} "
              f"dMovingMicro={number(d['MovingMicro'])} added={q.get('added', 0)} "
              f"semantic_precision={precision} damaged={q.get('damaged', 0)}")
    print("\n===== PER HORIZON =====")
    for name in ("T0_GT_MOTION", "HISTORY_GT_ALIGN_GT_MOTION", "HISTORY_CAUSAL_ALIGN_PRED_MOTION",
                 "HISTORY_GT_SELECT_PRED_MOTION", "STATIC_MEMORY", "STATIC_PATCH_GT_SELECT"):
        row = result["variants"][name]
        print(name, {h: d["mIoU"] for h, d in row["delta_vs_v18_pp"]["per_horizon"].items()})
    for key in ("geometry_audit", "error_mass", "resource_triage", "performance"):
        print(f"\n===== {key.upper()} =====")
        print(json.dumps(result[key], ensure_ascii=False, indent=2, allow_nan=False))
    print("\nGT motion/alignment/selection are diagnostics, NOT deployable gains. "
          "VOXEL_GT_FILTER rows are hindsight precision ceilings only. No training was run.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    summarize(json.loads(args.result.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
