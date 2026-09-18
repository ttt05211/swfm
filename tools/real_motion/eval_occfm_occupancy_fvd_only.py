#!/usr/bin/env python3
"""FVD-only wrapper for occupancy temporal diagnostics.

Uses the same corrected actual-prediction feature contract as
eval_occfm_occupancy_inception_metrics.py but skips single-frame FID/KID work.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from tools.real_motion.eval_occfm_occupancy_inception_metrics import (
    _clip_files,
    _extract_fvd,
    _frechet,
    _load_occfm_model,
    _seed_all,
)

PROTOCOL = "p0_f9_occfm_occupancy_fvd_only_v1"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--occfm-root", required=True)
    p.add_argument("--clip-dir", required=True)
    p.add_argument("--fvd-cfg", required=True)
    p.add_argument("--fvd-ckpt", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--feature-cache", default="")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--max-clips", type=int, default=0)
    a = p.parse_args()

    root = Path(a.occfm_root).resolve()
    clip_dir = Path(a.clip_dir).resolve()
    files = _clip_files(clip_dir)
    if a.max_clips > 0:
        files = files[: min(len(files), int(a.max_clips))]
    if len(files) < 2:
        raise RuntimeError("need at least two clips")

    device = torch.device(
        a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    _seed_all(int(a.seed))

    cache = Path(a.feature_cache).resolve() if a.feature_cache else None
    if cache is not None and cache.exists():
        with np.load(cache) as x:
            pred = np.asarray(x["fvd_pred"], dtype=np.float32)
            gt = np.asarray(x["fvd_gt"], dtype=np.float32)
        if len(pred) != len(files):
            raise RuntimeError(
                f"feature cache samples {len(pred)} != selected clips {len(files)}"
            )
        print(f"loaded FVD feature cache {cache}", flush=True)
    else:
        model, _, _ = _load_occfm_model(
            root, Path(a.fvd_cfg).resolve(), Path(a.fvd_ckpt).resolve(), device
        )
        pred, gt, _ = _extract_fvd(
            model,
            files,
            device,
            reorder_gt_sanity=False,
            reorder_seed=int(a.seed) + 17,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, fvd_pred=pred, fvd_gt=gt)
            print(f"saved FVD feature cache {cache}", flush=True)

    fvd = _frechet(pred, gt)
    fvd["x1e3"] = float(fvd["value"] * 1e3)
    result = {
        "protocol": PROTOCOL,
        "num_clips": int(len(files)),
        "clip_dir": str(clip_dir),
        "seed": int(a.seed),
        "feature_extractor": {
            "cfg": str(Path(a.fvd_cfg).resolve()),
            "checkpoint": str(Path(a.fvd_ckpt).resolve()),
            "contract": (
                "OccFM released temporal occupancy 3D-VAE; actual prediction; "
                "sampled latent; adaptive_avg_pool2d 5x5; six-frame flatten"
            ),
        },
        "fvd_3s_6frames": fvd,
        "feature_cache": str(cache) if cache is not None else None,
    }
    op = Path(a.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\n=== OCCUPANCY FVD ONLY ===")
    print(json.dumps(result, indent=2))
    print(f"saved {op}")


if __name__ == "__main__":
    main()
