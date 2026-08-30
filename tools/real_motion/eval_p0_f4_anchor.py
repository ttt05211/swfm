#!/usr/bin/env python3
"""Preflight P0-F4 strong W2Det anchor and same-support GT oracle.

This is a contract check before Sparse-WM training: it uses only the already
built validation cache, performs no learned future prediction, and confirms that
we restored the strong causal baseline rather than the P0-F3 weak anchor.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from real_motion.metrics.moving_miou_v2 import (
    DYNAMIC_CLASS_IDS,
    MovingMIoUV2MultiHorizon,
    REPORT_HORIZONS_S,
    SemanticIoUAccumulator,
)
from real_motion.msp import latent_support_to_bev
from real_motion.msp_wm_cache import MSP_WM_CACHE_VERSION_V2, MSPWorldModelCacheDataset

REPORT = {1.0: 1, 2.0: 3, 3.0: 5}
DYNAMIC = np.asarray(DYNAMIC_CLASS_IDS, dtype=np.int64)
FREE = 17


def _new_metrics():
    return {
        "overall": {h: SemanticIoUAccumulator() for h in REPORT_HORIZONS_S},
        "moving": MovingMIoUV2MultiHorizon(),
    }


def _update(state, h, pred, gt, support):
    state["overall"][h].update(pred, gt)
    state["moving"].update(h, pred, gt, support)


def _report(state):
    oh = {h: state["overall"][h].compute() for h in REPORT_HORIZONS_S}
    return {
        "overall": {
            "mIoU": float(np.nanmean([oh[h]["mIoU"] for h in REPORT_HORIZONS_S])),
            "per_horizon": oh,
        },
        "moving": state["moving"].compute(),
    }


def _gt_repair(anchor, gt, write_bev):
    write = np.asarray(write_bev, dtype=bool)[..., None]
    anchor = np.asarray(anchor)
    gt = np.asarray(gt)
    out = anchor.copy()
    out[write & np.isin(anchor, DYNAMIC)] = FREE
    dyn_gt = np.isin(gt, DYNAMIC)
    out[write & dyn_gt] = gt[write & dyn_gt]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--output", default=None)
    a = p.parse_args()

    ds = MSPWorldModelCacheDataset(a.cache)
    if ds.version != MSP_WM_CACHE_VERSION_V2:
        raise RuntimeError("P0-F4 preflight requires v2 cache")
    if not bool(ds.metadata.get("include_eval_payload", False)):
        raise RuntimeError("validation cache must include eval payload")

    anchor_state = _new_metrics()
    oracle_state = _new_metrics()
    for i in range(len(ds)):
        s = ds[i]
        gt = s["eval_future_gt_occ"].numpy()
        anchor = s["eval_strong_anchor_occ"].numpy()
        support = s["eval_gt_moving_support"].numpy().astype(bool)
        write = latent_support_to_bev(s["msp_write_support_latent"], (200, 200)).numpy().astype(bool)
        for h, fi in REPORT.items():
            _update(anchor_state, h, anchor[fi], gt[fi], support[fi])
            _update(oracle_state, h, _gt_repair(anchor[fi], gt[fi], write[fi]), gt[fi], support[fi])
        if i % 16 == 0:
            print("preflight", i, s["sample_id"])

    ar = _report(anchor_state)
    oracle = _report(oracle_state)
    am = float(ar["moving"]["mIoU"])
    om = float(oracle["moving"]["mIoU"])
    out = {
        "protocol": "p0_f4_strong_w2det_anchor_preflight_v1",
        "num_windows": len(ds),
        "write_budget_ratio": ds.metadata.get("write_budget_ratio"),
        "slot_compute_ratio": ds.metadata.get("slot_compute_ratio"),
        "strong_w2det_anchor": ar,
        "same_support_gt_repair_oracle": oracle,
        "oracle_delta_Moving_vs_strong_anchor": om - am,
    }
    if a.output:
        op = Path(a.output)
        op.parent.mkdir(parents=True, exist_ok=True)
        op.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
