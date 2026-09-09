#!/usr/bin/env python3
"""Select a V17 checkpoint that is balanced across IoU, mIoU and Moving-mIoU.

This is intentionally a *final selection* utility for checkpoints that already
have full evaluator-v2 JSONs. It does not replace the cheap Moving-aware proxy
used to screen checkpoints before expensive full evaluation.

Selection contract:
1. read true binary occupancy IoU, semantic mIoU, and Moving-mIoU;
2. discard Pareto-dominated checkpoints from the candidate frontier;
3. for each metric compute absolute percentage-point regret to the best observed
   value among the supplied checkpoints;
4. select the Pareto checkpoint with minimum worst-metric regret (minimax);
5. deterministic ties: lower mean regret, then lower sum regret, then earlier
   epoch, then lexical path.

The default uses equal percentage-point importance because all three metrics are
reported in the same units. No range normalization is used: normalizing by the
observed checkpoint spread would make the selector unstable to which checkpoints
happened to be supplied.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from tools.real_motion.summarize_p0_f9_v17_eval import summarize

METRICS = ("occupancy_IoU", "semantic_mIoU", "moving_mIoU")
PROTOCOL = "p0_f9_v17_balanced_checkpoint_selection_v1"


def pareto_front(rows: list[dict], metrics=METRICS, atol: float = 1e-12) -> list[bool]:
    """Return True for non-dominated rows; larger is better for every metric."""
    out = []
    for i, row in enumerate(rows):
        dominated = False
        for j, other in enumerate(rows):
            if i == j:
                continue
            ge = all(float(other[m]) >= float(row[m]) - atol for m in metrics)
            gt = any(float(other[m]) > float(row[m]) + atol for m in metrics)
            if ge and gt:
                dominated = True
                break
        out.append(not dominated)
    return out


def select_balanced(rows: list[dict], metrics=METRICS) -> dict:
    """Minimize the worst absolute percentage-point regret on the Pareto front."""
    if not rows:
        raise ValueError("no checkpoint rows")
    for row in rows:
        for metric in metrics:
            if metric not in row or not np.isfinite(float(row[metric])):
                raise ValueError(f"row {row.get('name', '?')} has invalid metric {metric}")

    best = {metric: max(float(r[metric]) for r in rows) for metric in metrics}
    front = pareto_front(rows, metrics=metrics)
    enriched = []
    for row, is_front in zip(rows, front):
        regrets = {metric: best[metric] - float(row[metric]) for metric in metrics}
        values = [regrets[metric] for metric in metrics]
        enriched.append({
            **row,
            "pareto": bool(is_front),
            "regret_pp": regrets,
            "max_regret_pp": float(max(values)),
            "mean_regret_pp": float(np.mean(values)),
            "sum_regret_pp": float(np.sum(values)),
        })

    candidates = [row for row in enriched if row["pareto"]]
    selected = min(
        candidates,
        key=lambda row: (
            float(row["max_regret_pp"]),
            float(row["mean_regret_pp"]),
            float(row["sum_regret_pp"]),
            int(row.get("epoch", 10**9)),
            str(row.get("name", "")),
        ),
    )
    return {
        "protocol": PROTOCOL,
        "metrics": list(metrics),
        "best_by_metric": best,
        "rows": enriched,
        "selected": selected,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json", nargs="+", help="full V17 evaluator-v2 JSON files")
    parser.add_argument(
        "--branch",
        default="local_stwm_center_always",
        choices=(
            "strong_anchor",
            "local_stwm_center_always",
            "local_stwm_center_rigid",
            "gt_center_rigid",
        ),
    )
    parser.add_argument("--output", default="", help="optional machine-readable selection JSON")
    args = parser.parse_args()

    paths = [Path(x) for x in args.json]
    rows = [summarize(path, args.branch) for path in paths]
    for row, path in zip(rows, paths):
        row["eval_json"] = str(path.resolve())

    result = select_balanced(rows)
    best = result["best_by_metric"]

    print("\n=== V17 BALANCED CHECKPOINT SELECTION ===")
    print(f"report_branch={args.branch}")
    print(
        "contract=minimize worst absolute percentage-point regret to the observed "
        "best IoU/mIoU/Moving value, after Pareto filtering"
    )
    print(
        f"observed_best: IoU={best['occupancy_IoU']:.4f} "
        f"mIoU={best['semantic_mIoU']:.4f} Moving={best['moving_mIoU']:.4f}"
    )
    print(
        f"{'name':28s} {'ep':>3s} {'P':>1s} {'IoU':>8s} {'mIoU':>8s} {'Moving':>8s} "
        f"{'rIoU':>7s} {'rmIoU':>7s} {'rMov':>7s} {'maxR':>7s} {'meanR':>7s}"
    )
    for row in result["rows"]:
        regret = row["regret_pp"]
        print(
            f"{row['name'][:28]:28s} {int(row['epoch']):3d} "
            f"{'*' if row['pareto'] else '-':>1s} "
            f"{float(row['occupancy_IoU']):8.4f} {float(row['semantic_mIoU']):8.4f} "
            f"{float(row['moving_mIoU']):8.4f} {regret['occupancy_IoU']:7.4f} "
            f"{regret['semantic_mIoU']:7.4f} {regret['moving_mIoU']:7.4f} "
            f"{float(row['max_regret_pp']):7.4f} {float(row['mean_regret_pp']):7.4f}"
        )

    selected = result["selected"]
    print("\nselected_checkpoint:")
    print(f"  name={selected['name']}")
    print(f"  epoch={selected['epoch']}")
    print(f"  eval_json={selected['eval_json']}")
    print(f"  max_regret_pp={selected['max_regret_pp']:.4f}")
    print(f"  mean_regret_pp={selected['mean_regret_pp']:.4f}")

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"saved {output}")


if __name__ == "__main__":
    main()
