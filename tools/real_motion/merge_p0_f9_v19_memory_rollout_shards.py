#!/usr/bin/env python3
"""Merge exact raw-count shards from the V19 zero-training memory rollout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from tools.real_motion.eval_p0_f9_v18_zero_shot_long_rollout import _finalize
from tools.real_motion.eval_p0_f9_v19_memory_rollout import PROTOCOL, VARIANTS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()

    rows = [json.loads(Path(x).read_text()) for x in a.inputs]
    if not rows:
        raise RuntimeError("no shard inputs")
    for r in rows:
        if r.get("protocol") != PROTOCOL:
            raise RuntimeError(f"unexpected protocol: {r.get('protocol')}")
        if "raw_counts" not in r:
            raise RuntimeError("missing raw_counts")

    ref = rows[0]
    for r in rows[1:]:
        for key in (
            "checkpoint",
            "checkpoint_epoch",
            "population_total_windows",
            "future_gt_used_for_prediction",
            "future_ego_pose_used_through_s",
            "reconciliation_config",
            "variant_contracts",
        ):
            if r.get(key) != ref.get(key):
                raise RuntimeError(f"shard metadata mismatch for {key}")

    num_shards = int(ref["shard"]["num_shards"])
    indices = sorted(int(r["shard"]["shard_index"]) for r in rows)
    if indices != list(range(num_shards)):
        raise RuntimeError(
            f"need all shard indices 0..{num_shards - 1}; got {indices}"
        )

    merged = {}
    for variant in VARIANTS:
        merged[variant] = {}
        for key in ref["raw_counts"][variant]:
            arrays = [
                np.asarray(r["raw_counts"][variant][key], dtype=np.int64)
                for r in rows
            ]
            if any(x.shape != arrays[0].shape for x in arrays):
                raise RuntimeError(f"shape mismatch: {variant}/{key}")
            merged[variant][key] = np.sum(
                arrays, axis=0, dtype=np.int64
            )

    num_windows = sum(int(r["num_windows"]) for r in rows)
    expected = int(ref["population_total_windows"])
    if num_windows != expected:
        raise RuntimeError(
            f"merged windows {num_windows} != expected {expected}"
        )


    reconciliation_totals = None
    if "reconciliation_totals" in ref:
        sum_keys = (
            "detected_sources",
            "memory_sources",
            "matched",
            "unmatched_detected",
            "unmatched_memory",
            "selected_memory_only",
            "dropped_memory_age",
            "dropped_memory_confidence",
            "match_distance_count",
        )
        reconciliation_totals = {
            key: int(sum(int(r["reconciliation_totals"][key]) for r in rows))
            for key in sum_keys
        }
        reconciliation_totals["match_distance_sum_m"] = float(
            sum(
                float(r["reconciliation_totals"]["match_distance_sum_m"])
                for r in rows
            )
        )
        reconciliation_totals["mean_match_distance_m"] = (
            reconciliation_totals["match_distance_sum_m"]
            / max(reconciliation_totals["match_distance_count"], 1)
        )
        reconciliation_totals["matched_fraction_of_detected"] = (
            reconciliation_totals["matched"]
            / max(reconciliation_totals["detected_sources"], 1)
        )
        reconciliation_totals["selected_memory_fraction"] = (
            reconciliation_totals["selected_memory_only"]
            / max(reconciliation_totals["memory_sources"], 1)
        )

    metrics = {v: _finalize(merged[v]) for v in VARIANTS}
    result = {
        "protocol": PROTOCOL,
        "merged_from_shards": True,
        "checkpoint": ref["checkpoint"],
        "checkpoint_epoch": ref["checkpoint_epoch"],
        "population_total_windows": expected,
        "num_windows": num_windows,
        "future_gt_used_for_prediction": False,
        "future_ego_pose_used_through_s": ref[
            "future_ego_pose_used_through_s"
        ],
        "reconciliation_config": ref.get("reconciliation_config"),
        "reconciliation_totals": reconciliation_totals,
        "variant_contracts": ref["variant_contracts"],
        "metrics": metrics,
        "raw_counts": {
            v: {k: x.tolist() for k, x in merged[v].items()}
            for v in VARIANTS
        },
        "source_shards": [str(Path(x).resolve()) for x in a.inputs],
    }
    op = Path(a.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\\n=== MERGED V19 ZERO-TRAINING MEMORY ROLLOUT ===")
    for v in VARIANTS:
        print("\\n", v)
        for h, row in metrics[v]["per_horizon"].items():
            print(
                f"{h}s IoU={row['IoU']:.3f} mIoU={row['mIoU']:.3f} "
                f"MovMacro={row['MovingMacro']:.3f} "
                f"MovMicro={row['MovingMicro']:.3f}"
            )
        print("AVG4/5/6", json.dumps(metrics[v]["average_4s_5s_6s"]))
    print(f"saved {op}")


if __name__ == "__main__":
    main()
