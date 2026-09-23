#!/usr/bin/env python3
"""Merge sharded V19 Innovation evaluation JSON files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools.real_motion.eval_p0_f9_v19_innovation import (
    PROTOCOL,
    VARIANTS,
)
from tools.real_motion.eval_p0_f9_v19_memory_ablation import (
    _delta,
    _finalize,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()

    rows = [
        json.loads(Path(x).read_text(encoding="utf-8"))
        for x in a.inputs
    ]
    if any(r.get("protocol") != PROTOCOL for r in rows):
        raise RuntimeError("unexpected V19 eval protocol")

    for key in (
        "base_checkpoint",
        "base_checkpoint_epoch",
        "innovation_checkpoint",
        "innovation_epoch",
    ):
        vals = {json.dumps(r.get(key), sort_keys=True) for r in rows}
        if len(vals) != 1:
            raise RuntimeError(f"eval shard mismatch for {key}")

    declared = {int(r.get("num_shards", 1)) for r in rows}
    if len(declared) != 1:
        raise RuntimeError("num_shards mismatch")
    num_shards = declared.pop()
    shard_ids = sorted(int(r.get("shard_index", 0)) for r in rows)
    if num_shards > 1 and shard_ids != list(range(num_shards)):
        raise RuntimeError(
            f"expected shard indexes 0..{num_shards-1}, got {shard_ids}"
        )

    raw = {}
    for v in VARIANTS:
        keys = rows[0]["raw_counts"][v].keys()
        raw[v] = {
            k: np.sum(
                [
                    np.asarray(r["raw_counts"][v][k], dtype=np.int64)
                    for r in rows
                ],
                axis=0,
            )
            for k in keys
        }
    metrics = {v: _finalize(raw[v]) for v in VARIANTS}
    proposed = sum(
        int(r["proposal_audit"]["proposed_voxels"]) for r in rows
    )
    added = sum(
        int(r["proposal_audit"]["added_voxels_after_protection"])
        for r in rows
    )
    windows_with = sum(
        int(r["proposal_audit"]["windows_with_additions"]) for r in rows
    )
    elapsed = max(float(r.get("timing", {}).get("elapsed_s", 0.0)) for r in rows)
    num_windows = sum(int(r["num_windows"]) for r in rows)

    out = {
        "protocol": PROTOCOL,
        "num_windows": int(num_windows),
        "merged_parallel_shards": len(rows),
        "base_checkpoint": rows[0]["base_checkpoint"],
        "base_checkpoint_epoch": rows[0]["base_checkpoint_epoch"],
        "innovation_checkpoint": rows[0]["innovation_checkpoint"],
        "innovation_epoch": rows[0]["innovation_epoch"],
        "future_gt_used_for_prediction": False,
        "metrics": metrics,
        "delta_vs_v18": {
            v: _delta(metrics[v], metrics["v18"])
            for v in VARIANTS
            if v != "v18"
        },
        "delta_innovation_vs_static": _delta(
            metrics["v18_static_innovation"],
            metrics["v18_static"],
        ),
        "proposal_audit": {
            "proposed_voxels": int(proposed),
            "added_voxels_after_protection": int(added),
            "windows_with_additions": int(windows_with),
        },
        "raw_counts": {
            v: {k: np.asarray(x).tolist() for k, x in raw[v].items()}
            for v in VARIANTS
        },
        "parallel_timing": {
            "wall_clock_upper_bound_s": float(elapsed),
            "aggregate_windows_per_s": float(
                num_windows / max(elapsed, 1e-9)
            ),
        },
    }
    op = Path(a.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\n=== MERGED V19 INNOVATION EVAL ===")
    for v in VARIANTS:
        m = metrics[v]
        print(
            f"{v:24s} IoU={m['IoU']:.3f} mIoU={m['mIoU']:.3f} "
            f"MovMacro={m['MovingMacro']:.3f} MovMicro={m['MovingMicro']:.3f}"
        )
    print("innovation_vs_static", json.dumps(out["delta_innovation_vs_static"]))
    print("proposal_audit", json.dumps(out["proposal_audit"]))
    print("parallel_timing", json.dumps(out["parallel_timing"]))
    print(f"saved {op}")


if __name__ == "__main__":
    main()
