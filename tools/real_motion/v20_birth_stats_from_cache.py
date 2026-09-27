#!/usr/bin/env python3
"""Compute strict V20 Birth statistics directly from Stage-1 caches."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from real_motion.v20_training import choose_birth_query_count
from tools.real_motion.build_p0_f9_v20_history_cache import PROTOCOL as CACHE_PROTOCOL

PROTOCOL = "p0_f9_v20_birth_distribution_from_cache_v1"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-cache", required=True)
    p.add_argument("--max-truncation-fraction", type=float, default=0.01)
    p.add_argument("--output", required=True)
    a = p.parse_args()

    root = Path(a.train_cache)
    idx = json.loads((root / "index.json").read_text(encoding="utf-8"))
    if idx.get("protocol") != CACHE_PROTOCOL:
        raise RuntimeError("Birth statistics require V20 Stage-1 cache")

    counts = []
    classes = Counter()
    horizons = Counter()
    sizes = []
    ambiguity = 0
    future_dynamic = 0
    for shard in idx["shards"]:
        obj = torch.load(root / shard["file"], map_location="cpu", weights_only=False)
        if obj.get("protocol") != CACHE_PROTOCOL:
            raise RuntimeError("bad V20 cache shard")
        for row in obj["rows"]:
            births = [
                r for r in row["dynamic_supervision"]
                if r["responsibility_name"] == "BIRTH"
            ]
            counts.append(len(births))
            for r in row["dynamic_supervision"]:
                future_dynamic += 1
                if r["responsibility_name"] == "IGNORE":
                    ambiguity += 1
            for r in births:
                classes[str(int(r["class_id"]))] += 1
                horizons[str(int(r["first_horizon"]))] += 1
                if r.get("size_lwh_m") is not None:
                    sizes.append([float(x) for x in r["size_lwh_m"]])

    q = choose_birth_query_count(
        counts,
        max_truncation_fraction=float(a.max_truncation_fraction),
    )
    arr = np.asarray(sizes, dtype=np.float64) if sizes else np.zeros((0, 3))
    report = {
        "protocol": PROTOCOL,
        "source_cache_protocol": CACHE_PROTOCOL,
        **q,
        "num_scenes": int(idx.get("num_scenes", 0)),
        "class_counts": dict(sorted(classes.items())),
        "first_horizon_counts_zero_based": dict(sorted(horizons.items())),
        "future_dynamic_instances": int(future_dynamic),
        "ambiguous_ignore_instances": int(ambiguity),
        "ambiguous_fraction": float(ambiguity / max(future_dynamic, 1)),
        "size_lwh_m": {
            "count": int(len(arr)),
            "p50": np.percentile(arr, 50, axis=0).tolist() if len(arr) else None,
            "p90": np.percentile(arr, 90, axis=0).tolist() if len(arr) else None,
            "p99": np.percentile(arr, 99, axis=0).tolist() if len(arr) else None,
            "max": arr.max(axis=0).tolist() if len(arr) else None,
        },
    }
    op = Path(a.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
