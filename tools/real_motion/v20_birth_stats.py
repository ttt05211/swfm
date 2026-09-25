#!/usr/bin/env python3
"""Report strict V20 birth statistics and select Q before Birth training.

Input JSONL is produced by the V20 label builder. Each row contains a window
and a list named births; each birth may contain class_id, first_horizon,
size_xyz_m and ambiguous=false. Rows may additionally contain ambiguous_dynamic.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from real_motion.v20_training import choose_birth_query_count


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--labels-jsonl", required=True)
    p.add_argument("--max-truncation-fraction", type=float, default=0.01)
    p.add_argument("--output", required=True)
    a = p.parse_args()

    counts = []
    classes = Counter()
    horizons = Counter()
    sizes = []
    ambiguous = 0
    dynamic_total = 0
    scenes = set()
    with Path(a.labels_jsonl).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            births = list(row.get("births", []))
            counts.append(len(births))
            ambiguous += int(row.get("ambiguous_dynamic", 0))
            dynamic_total += int(row.get("future_dynamic_instances", len(births)))
            if row.get("scene") is not None:
                scenes.add(str(row["scene"]))
            for b in births:
                classes[str(b["class_id"])] += 1
                horizons[str(b["first_horizon"])] += 1
                if "size_xyz_m" in b:
                    sizes.append([float(x) for x in b["size_xyz_m"]])

    q = choose_birth_query_count(
        counts,
        max_truncation_fraction=float(a.max_truncation_fraction),
    )
    arr = np.asarray(sizes, dtype=np.float64) if sizes else np.zeros((0, 3))
    report = {
        "protocol": "p0_f9_v20_birth_distribution_v1",
        **q,
        "scenes": len(scenes),
        "class_counts": dict(sorted(classes.items())),
        "first_horizon_counts": dict(sorted(horizons.items())),
        "ambiguous_dynamic_instances": ambiguous,
        "future_dynamic_instances": dynamic_total,
        "ambiguous_fraction": float(ambiguous / max(dynamic_total, 1)),
        "size_xyz_m": {
            "count": int(len(arr)),
            "p50": np.percentile(arr, 50, axis=0).tolist() if len(arr) else None,
            "p90": np.percentile(arr, 90, axis=0).tolist() if len(arr) else None,
            "p99": np.percentile(arr, 99, axis=0).tolist() if len(arr) else None,
        },
    }
    Path(a.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
