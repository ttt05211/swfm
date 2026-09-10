#!/usr/bin/env python3
"""Summarize B-C/B-S full evaluations at fixed matched training epochs.

This utility intentionally does not use SoftCH or balanced checkpoint selection.
It checks that the compared control/treatment JSONs are from the same global
epoch and reports direct treatment-minus-control deltas for IoU, mIoU and
Moving-mIoU.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.real_motion.summarize_p0_f9_v17_eval import summarize

BRANCH = "local_stwm_center_always"


def _row(path: str):
    return summarize(Path(path), BRANCH)


def _print(row, label):
    print(
        f"{label:16s} ep={int(row['epoch']):2d} "
        f"IoU={float(row['occupancy_IoU']):8.4f} "
        f"mIoU={float(row['semantic_mIoU']):8.4f} "
        f"Moving={float(row['moving_mIoU']):8.4f}"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True, help="shared epoch-5 RL full-eval JSON")
    p.add_argument("--control-mid", required=True)
    p.add_argument("--native-mid", required=True)
    p.add_argument("--control-end", required=True)
    p.add_argument("--native-end", required=True)
    a = p.parse_args()

    start = _row(a.start)
    cm = _row(a.control_mid); nm = _row(a.native_mid)
    ce = _row(a.control_end); ne = _row(a.native_end)
    if int(cm["epoch"]) != int(nm["epoch"]):
        raise RuntimeError("midpoint control/treatment epochs differ")
    if int(ce["epoch"]) != int(ne["epoch"]):
        raise RuntimeError("endpoint control/treatment epochs differ")

    print("\n=== V17 NATIVE FOOTPRINT PAIRED FIXED-TIME COMPARISON ===")
    print(f"report_branch={BRANCH}")
    _print(start, "shared_start")
    _print(cm, "B-C_mid")
    _print(nm, "B-S_mid")
    _print(ce, "B-C_end")
    _print(ne, "B-S_end")

    print("\n=== B-S minus B-C ===")
    for name, c, n in (("mid", cm, nm), ("end", ce, ne)):
        print(
            f"{name:4s} ep={int(c['epoch']):2d} "
            f"dIoU={float(n['occupancy_IoU'])-float(c['occupancy_IoU']):+8.4f} "
            f"dmIoU={float(n['semantic_mIoU'])-float(c['semantic_mIoU']):+8.4f} "
            f"dMoving={float(n['moving_mIoU'])-float(c['moving_mIoU']):+8.4f}"
        )


if __name__ == "__main__":
    main()
