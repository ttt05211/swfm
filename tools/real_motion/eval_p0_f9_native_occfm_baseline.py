#!/usr/bin/env python3
"""Evaluate the frozen official dense OccFM-Fut checkpoint on the P0-F9 split.

This is a paper baseline, not a diagnostic: it answers how strong the native
History->Future pretrained World Model is under exactly the same 128-window
Overall/Moving protocol used for Strong-W2Det and P0-F9.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
UP = ROOT / "upstream_occfm"
sys.path[:0] = [str(ROOT), str(UP)]

import numpy as np
import torch

from real_motion.metrics.moving_miou_v2 import (
    MovingMIoUV2MultiHorizon,
    REPORT_HORIZONS_S,
    SemanticIoUAccumulator,
)
from real_motion.msp_wm_cache import MSP_WM_CACHE_VERSION_V2, MSPWorldModelCacheDataset
from real_motion.occfm_io import load_official_wm, run_frozen_occfm_forecast
from tools.real_motion.build_p0_f9_cache_fast import P0_F9_CACHE_PROTOCOL

REPORT = {1.0: 1, 2.0: 3, 3.0: 5}


def _new_metrics():
    return {
        "overall": {h: SemanticIoUAccumulator() for h in REPORT_HORIZONS_S},
        "moving": MovingMIoUV2MultiHorizon(),
    }


def _report(state):
    h = {t: state["overall"][t].compute() for t in REPORT_HORIZONS_S}
    return {
        "overall": {
            "mIoU": float(np.nanmean([h[t]["mIoU"] for t in REPORT_HORIZONS_S])),
            "per_horizon": h,
        },
        "moving": state["moving"].compute(),
    }


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--occfm-ckpt", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=20260904)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    device = torch.device(a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("the pinned official OccFM sampler requires CUDA")
    ds = MSPWorldModelCacheDataset(a.cache)
    if ds.version != MSP_WM_CACHE_VERSION_V2 or ds.metadata.get("protocol") != P0_F9_CACHE_PROTOCOL:
        raise RuntimeError("native OccFM baseline requires a P0-F9 validation cache")
    if not bool(ds.metadata.get("include_eval_payload", False)):
        raise RuntimeError("native OccFM baseline requires validation eval payload")

    wm, cfg = load_official_wm(UP, a.occfm_ckpt, device)
    state = _new_metrics()
    for i in range(len(ds)):
        s = ds[i]
        for key in ("eval_future_gt_occ", "eval_gt_moving_support"):
            if key not in s:
                raise RuntimeError(f"{s['sample_id']}: missing {key}")
        pred = run_frozen_occfm_forecast(
            wm,
            s["full_history_latent"],
            s["gt_future_latent"],
            trajectory=s["trajectory"],
            seed=int(a.seed) + i,
            hist_last=4,
        ).numpy()
        gt = s["eval_future_gt_occ"].numpy()
        moving = s["eval_gt_moving_support"].numpy().astype(bool)
        for horizon, fi in REPORT.items():
            state["overall"][horizon].update(pred[fi], gt[fi])
            state["moving"].update(horizon, pred[fi], gt[fi], moving[fi])
        if i % 8 == 0:
            print("native_occfm_eval", i, s["sample_id"])

    report = {
        "protocol": "p0_f9_official_occfm_native_baseline_v1",
        "num_windows": len(ds),
        "checkpoint": str(Path(a.occfm_ckpt).resolve()),
        "sample_steps": int(cfg.LOSS.SAMPLE_STEP),
        "seed": int(a.seed),
        "metrics": _report(state),
    }
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
