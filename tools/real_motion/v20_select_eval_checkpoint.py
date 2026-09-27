#!/usr/bin/env python3
"""Select an evaluated checkpoint only from identical formal dev populations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _get(obj, path):
    cur = obj
    for key in str(path).split("."):
        if not isinstance(cur, dict) or key not in cur:
            raise KeyError(f"metric path not found: {path}")
        cur = cur[key]
    return float(cur)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-json", nargs="+", required=True)
    p.add_argument("--metric-path", required=True)
    p.add_argument("--mode", choices=("max", "min"), default="max")
    p.add_argument("--output", required=True)
    a = p.parse_args()

    rows = []
    population = None
    base_checkpoint = None
    for path in a.eval_json:
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        fp = obj.get("population_fingerprint_sha256")
        if not fp:
            raise RuntimeError(f"{path}: missing population fingerprint")
        if population is None:
            population = fp
        elif fp != population:
            raise RuntimeError("refusing checkpoint selection across different populations")
        base = str(obj.get("base_checkpoint", ""))
        if base_checkpoint is None:
            base_checkpoint = base
        elif base != base_checkpoint:
            raise RuntimeError("refusing selection across different V18 checkpoints")
        if obj.get("checkpoint_eligible_for_formal_selection") is False:
            raise RuntimeError(
                f"{path}: checkpoint is explicitly ineligible for formal selection"
            )
        checkpoint = (
            obj.get("v20_checkpoint")
            or obj.get("stage0_checkpoint")
            or obj.get("checkpoint")
        )
        if not checkpoint:
            raise RuntimeError(f"{path}: no checkpoint field found")
        rows.append({
            "eval_json": str(Path(path).resolve()),
            "checkpoint": str(checkpoint),
            "score": _get(obj, a.metric_path),
            "protocol": obj.get("protocol"),
        })
    rows.sort(
        key=lambda x: x["score"],
        reverse=(a.mode == "max"),
    )
    result = {
        "protocol": "p0_f9_v20_formal_dev_checkpoint_selection_v1",
        "metric_path": str(a.metric_path),
        "mode": str(a.mode),
        "population_fingerprint_sha256": population,
        "base_checkpoint": base_checkpoint,
        "selected": rows[0],
        "candidates": rows,
        "note": (
            "This tool only selects among already-computed formal dev evaluations. "
            "It never uses training/validation loss as a success criterion."
        ),
    }
    op = Path(a.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
