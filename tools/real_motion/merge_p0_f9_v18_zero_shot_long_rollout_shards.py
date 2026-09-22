#!/usr/bin/env python3
"""Merge exact raw-count shards from the V18 zero-shot long rollout evaluator."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from tools.real_motion.eval_p0_f9_v18_zero_shot_long_rollout import (
    PROTOCOL,
    _finalize,
)


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
            raise RuntimeError("shard is missing raw_counts; rerun with sharded evaluator")

    ref = rows[0]
    for r in rows[1:]:
        for key in (
            "checkpoint",
            "checkpoint_epoch",
            "checkpoint_global_step",
            "val_cache",
            "population_total_windows",
            "history_frames_per_block",
            "relative_future_frames_per_block",
            "rollout_blocks",
            "report_horizons_s",
            "evaluation_mode",
            "second_block_history",
            "future_gt_used_for_prediction",
            "conditioning_contract",
            "rollout_contract",
            "population_contract",
        ):
            if r.get(key) != ref.get(key):
                raise RuntimeError(f"shard metadata mismatch for {key}")

    shard_meta = [r.get("shard") or {} for r in rows]
    num_shards = int(shard_meta[0].get("num_shards", -1))
    got_indices = sorted(int(s.get("shard_index", -1)) for s in shard_meta)
    if num_shards <= 0 or got_indices != list(range(num_shards)):
        raise RuntimeError(
            f"need all shard indices 0..{num_shards-1}; got {got_indices}"
        )

    merged = {}
    for key in ref["raw_counts"]:
        arrays = [np.asarray(r["raw_counts"][key], dtype=np.int64) for r in rows]
        shape = arrays[0].shape
        if any(x.shape != shape for x in arrays):
            raise RuntimeError(f"raw count shape mismatch for {key}")
        merged[key] = np.sum(arrays, axis=0, dtype=np.int64)

    num_windows = sum(int(r["num_windows"]) for r in rows)
    expected = int(ref["population_total_windows"])
    if num_windows != expected:
        raise RuntimeError(
            f"merged windows {num_windows} != population_total_windows {expected}"
        )

    metrics = _finalize(merged)
    result = {
        "protocol": PROTOCOL,
        "merged_from_shards": True,
        "checkpoint": ref["checkpoint"],
        "checkpoint_epoch": ref["checkpoint_epoch"],
        "checkpoint_global_step": ref["checkpoint_global_step"],
        "val_cache": ref["val_cache"],
        "population_total_windows": expected,
        "num_windows": num_windows,
        "num_scenes_note": (
            "scene counts are not summed because scenes may span shards; "
            "population is defined by exact merged window count"
        ),
        "history_frames_per_block": ref["history_frames_per_block"],
        "relative_future_frames_per_block": ref["relative_future_frames_per_block"],
        "rollout_blocks": ref["rollout_blocks"],
        "report_horizons_s": ref["report_horizons_s"],
        "evaluation_mode": ref.get("evaluation_mode"),
        "second_block_history": ref.get("second_block_history"),
        "future_gt_used_for_prediction": ref["future_gt_used_for_prediction"],
        "future_ego_pose_used_through_s": ref["future_ego_pose_used_through_s"],
        "conditioning_contract": ref["conditioning_contract"],
        "rollout_contract": ref["rollout_contract"],
        "exactness_gate": ref["exactness_gate"],
        "population_contract": ref["population_contract"],
        "metrics": metrics,
        "raw_counts": {k: v.tolist() for k, v in merged.items()},
        "source_shards": [str(Path(x).resolve()) for x in a.inputs],
    }
    op = Path(a.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n=== MERGED V18 ZERO-SHOT BLOCK ROLLOUT 1--6s ===")
    print(
        f"{'horizon':>8s} {'IoU':>9s} {'mIoU':>9s} "
        f"{'MovMacro':>10s} {'MovMicro':>10s}"
    )
    for h in ref["report_horizons_s"]:
        x = metrics["per_horizon"][str(float(h))]
        print(
            f"{float(h):8.1f} {x['IoU']:9.3f} {x['mIoU']:9.3f} "
            f"{x['MovingMacro']:10.3f} {x['MovingMicro']:10.3f}"
        )
    print("AVG 1/2/3:", json.dumps(metrics["average_1s_2s_3s"]))
    print("AVG 4/5/6:", json.dumps(metrics["average_4s_5s_6s"]))
    print(f"saved {op}")


if __name__ == "__main__":
    main()
