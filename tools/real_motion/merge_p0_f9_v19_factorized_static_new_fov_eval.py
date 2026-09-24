#!/usr/bin/env python3
"""Merge disjoint factorized Static New-FOV evaluation shards exactly.

The evaluator stores additive raw confusion counts. Summing those counts and
re-running the standard finalizer reproduces the single-process full-population
metrics exactly, while allowing one evaluator process per GPU.
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

from tools.real_motion.eval_p0_f9_v19_factorized_static_new_fov import (
    PROTOCOL,
    VARIANTS,
    _effective_addition_quality,
)
from tools.real_motion.eval_p0_f9_v19_static_new_fov import (
    _delta,
    _finalize,
    _new_raw,
)


IDENTICAL_KEYS = (
    "protocol",
    "selected_population_windows",
    "num_shards",
    "base_checkpoint",
    "base_checkpoint_epoch",
    "novelty_checkpoint",
    "novelty_epoch",
    "presence_threshold",
    "vertical_threshold",
    "future_gt_used_for_prediction",
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()

    paths = [Path(x).resolve() for x in a.inputs]
    rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in paths
    ]
    if not rows:
        raise RuntimeError("no evaluation shards")
    ref = rows[0]
    if ref.get("protocol") != PROTOCOL:
        raise RuntimeError(
            f"unexpected protocol: {ref.get('protocol')}"
        )

    for i, row in enumerate(rows[1:], start=1):
        for key in IDENTICAL_KEYS:
            if row.get(key) != ref.get(key):
                raise RuntimeError(
                    f"evaluation shard {i} mismatch for {key}: "
                    f"{row.get(key)!r} != {ref.get(key)!r}"
                )

    expected = int(ref.get("num_shards", 1))
    if len(rows) != expected:
        raise RuntimeError(
            f"expected {expected} shard JSONs, got {len(rows)}"
        )
    shard_ids = sorted(int(x.get("shard_index", -1)) for x in rows)
    if shard_ids != list(range(expected)):
        raise RuntimeError(
            f"shard indices are not exactly 0..{expected-1}: {shard_ids}"
        )

    merged_raw = {v: _new_raw() for v in VARIANTS}
    total_windows = 0
    audit = {
        "new_fov_bev_columns": 0,
        "active_bev_columns": 0,
        "proposed_voxels": 0,
        "added_voxels_after_protection": 0,
        "windows_with_additions": 0,
    }
    elapsed_parallel_max = 0.0
    elapsed_sum = 0.0

    for row in rows:
        total_windows += int(row["num_windows"])
        for v in VARIANTS:
            raw = row["raw_counts"][v]
            for key in merged_raw[v]:
                merged_raw[v][key] += np.asarray(
                    raw[key],
                    dtype=np.int64,
                )
        for key in audit:
            audit[key] += int(row["proposal_audit"][key])
        elapsed = float(row.get("timing", {}).get("elapsed_s", 0.0))
        elapsed_parallel_max = max(elapsed_parallel_max, elapsed)
        elapsed_sum += elapsed

    expected_population = int(ref["selected_population_windows"])
    if total_windows != expected_population:
        raise RuntimeError(
            f"merged windows {total_windows} != selected population "
            f"{expected_population}"
        )

    metrics = {
        v: _finalize(merged_raw[v])
        for v in VARIANTS
    }
    delta = _delta(
        metrics["v18_static_factorized_new_fov"],
        metrics["v18_static"],
    )
    effective = _effective_addition_quality(
        merged_raw["v18_static"],
        merged_raw["v18_static_factorized_new_fov"],
    )

    result = {
        "protocol": PROTOCOL,
        "merged_multi_gpu": True,
        "num_windows": int(total_windows),
        "selected_population_windows": int(expected_population),
        "num_shards": int(expected),
        "shard_index": None,
        "input_shards": [str(x) for x in paths],
        "base_checkpoint": ref["base_checkpoint"],
        "base_checkpoint_epoch": int(ref["base_checkpoint_epoch"]),
        "novelty_checkpoint": ref["novelty_checkpoint"],
        "novelty_epoch": int(ref["novelty_epoch"]),
        "presence_threshold": float(ref["presence_threshold"]),
        "vertical_threshold": float(ref["vertical_threshold"]),
        "future_gt_used_for_prediction": bool(
            ref["future_gt_used_for_prediction"]
        ),
        "metrics": metrics,
        "delta_factorized_vs_static": delta,
        "effective_addition_quality": effective,
        "proposal_audit": audit,
        "raw_counts": {
            v: {
                k: np.asarray(x).tolist()
                for k, x in merged_raw[v].items()
            }
            for v in VARIANTS
        },
        "timing": {
            "sum_worker_elapsed_s": float(elapsed_sum),
            "parallel_wall_estimate_s": float(elapsed_parallel_max),
            "effective_parallel_windows_per_s": float(
                total_windows / max(elapsed_parallel_max, 1e-9)
            ),
        },
    }

    op = Path(a.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print("\n=== MERGED V19 FACTORIZED STATIC NEW-FOV EVAL ===")
    for v in VARIANTS:
        m = metrics[v]
        print(
            f"{v:32s} "
            f"IoU={m['IoU']:.3f} "
            f"mIoU={m['mIoU']:.3f} "
            f"MovMacro={m['MovingMacro']:.3f} "
            f"MovMicro={m['MovingMicro']:.3f} "
            f"main123_mIoU={m['main_1_2_3s']['mIoU']:.3f}"
        )
    print("factorized_vs_static", json.dumps(delta))
    print("effective_addition", json.dumps(effective))
    print("proposal_audit", json.dumps(audit))
    print("timing", json.dumps(result["timing"]))
    print(f"saved {op}")


if __name__ == "__main__":
    main()
