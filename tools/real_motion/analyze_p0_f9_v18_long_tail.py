#!/usr/bin/env python3
"""Audit semantic and KTA-relative motion long tails in a V18 SE2 train cache.

No model is loaded and no training is performed. The script uses exactly the
flattened supervised-source population consumed by clean V18 training and
reports the deterministic balanced-sampler weights that would be used by the
balanced one-stage trainer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from real_motion.long_tail_sampling import build_balanced_source_weights
from tools.real_motion.train_p0_f9_v18_se2_pair import flatten_supervised, load_se2_cache


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-cache", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()

    meta, records = load_se2_cache(a.train_cache)
    train = flatten_supervised(records)
    balanced = build_balanced_source_weights(
        train["source_class_id"],
        train["target_source_residual_xy_m"],
        train["se2_target_valid"],
    )
    report = {
        "train_cache": str(Path(a.train_cache).resolve()),
        "train_cache_metadata": meta,
        **balanced.report,
    }
    op = Path(a.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=== V18 LONG-TAIL TRAIN DISTRIBUTION ===")
    print(f"sources={report['sources']}")
    print(f"effective_sample_fraction={report['effective_sample_fraction']:.4f}")
    print(
        f"global top10 motion threshold="
        f"{report['global_top10_motion_threshold_m']:.4f} m"
    )
    print(
        f"{'class':>5s} {'N':>8s} {'raw%':>8s} {'sample%':>9s} "
        f"{'q50':>8s} {'q90':>8s} {'q95':>8s} {'mean_w':>8s}"
    )
    for cid, row in report["classes"].items():
        q = row["motion_difficulty_m"]
        print(
            f"{cid:>5s} {row['sources']:8d} "
            f"{100*row['source_fraction']:8.3f} "
            f"{100*row['expected_sample_fraction']:9.3f} "
            f"{q['median']:8.3f} {q['p90']:8.3f} {q['p95']:8.3f} "
            f"{row['mean_final_weight']:8.3f}"
        )
    print(f"saved {op}")


if __name__ == "__main__":
    main()
