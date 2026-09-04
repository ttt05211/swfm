#!/usr/bin/env python3
"""Evaluate the frozen official dense OccFM-Fut checkpoint on the P0-F9 split.

This is a paper baseline, not a diagnostic. The audited P0-F9 cache stores
posterior-sampled history latents, matching the distribution used by the released
OccFM-Fut cache. ``run_frozen_occfm_forecast`` then applies the official
HIST_LAST=4 condition masking and released sampler.
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
from real_motion.native_forecast import deterministic_sample_seed
from real_motion.occfm_io import file_sha256, load_official_wm, run_frozen_occfm_forecast
from tools.real_motion.build_p0_f9_cache_fast import P0_F9_CACHE_PROTOCOL

REPORT = {1.0: 1, 2.0: 3, 3.0: 5}
HIST_LAST = 4


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
        raise RuntimeError("native OccFM baseline requires the audited P0-F9 validation cache")
    if ds.metadata.get("vae_mode") != "sample":
        raise RuntimeError("native OccFM baseline requires posterior-sampled P0-F9 latents")
    if int(ds.metadata.get("native_backbone_hist_last", -1)) != HIST_LAST:
        raise RuntimeError("native OccFM baseline requires HIST_LAST=4 cache provenance")
    if not bool(ds.metadata.get("include_eval_payload", False)):
        raise RuntimeError("native OccFM baseline requires validation eval payload")

    wm, cfg = load_official_wm(UP, a.occfm_ckpt, device)
    if int(cfg.DATA_CONFIG.HIST_LAST) != HIST_LAST:
        raise RuntimeError(f"official OccFM config HIST_LAST changed: {cfg.DATA_CONFIG.HIST_LAST}")
    state = _new_metrics()
    for i in range(len(ds)):
        s = ds[i]
        for key in ("eval_future_gt_occ", "eval_gt_moving_support"):
            if key not in s:
                raise RuntimeError(f"{s['sample_id']}: missing {key}")
        sample_seed = deterministic_sample_seed(
            str(s["sample_id"]), a.seed, stream="forecast"
        )
        pred = run_frozen_occfm_forecast(
            wm,
            s["full_history_latent"],
            s["gt_future_latent"],
            trajectory=s["trajectory"],
            seed=sample_seed,
            hist_last=HIST_LAST,
        ).numpy()
        gt = s["eval_future_gt_occ"].numpy()
        moving = s["eval_gt_moving_support"].numpy().astype(bool)
        for horizon, fi in REPORT.items():
            state["overall"][horizon].update(pred[fi], gt[fi])
            state["moving"].update(horizon, pred[fi], gt[fi], moving[fi])
        if i % 8 == 0:
            print("native_occfm_eval", i, s["sample_id"])

    report = {
        "protocol": "p0_f9_official_occfm_native_baseline_v2",
        "num_windows": len(ds),
        "checkpoint": str(Path(a.occfm_ckpt).resolve()),
        "checkpoint_sha256": file_sha256(a.occfm_ckpt),
        "cache_index_sha256": file_sha256(ds.root / "index.json"),
        "latent_distribution": "posterior_sample",
        "hist_last": HIST_LAST,
        "sample_steps": int(cfg.LOSS.SAMPLE_STEP),
        "seed": int(a.seed),
        "sample_seed_contract": "sha256(base,forecast,sample_id)",
        "metrics": _report(state),
    }
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
