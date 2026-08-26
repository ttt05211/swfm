#!/usr/bin/env python3
"""P0-A: Full vs causal-static-only vs moving/uncertain-only frozen OccFM."""
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
UP = ROOT / "upstream_occfm"
sys.path[:0] = [str(UP), str(ROOT)]

import numpy as np
import torch
from real_motion.prepared import PreparedShardDataset
from real_motion.occfm_io import (
    load_official_vae, load_official_wm, OccFMVAEAdapter,
    run_frozen_occfm_forecast,
)
from real_motion.metrics.moving_miou_v2 import (
    MovingMIoUV2MultiHorizon, SemanticIoUAccumulator, REPORT_HORIZONS_S,
)
from real_motion.nuscenes_adapter import gt_true_static_mask

REPORT_INDEX = {1.0: 1, 2.0: 3, 3.0: 5}  # six future keyframes at 0.5 s


def mean_horizon(acc):
    per = {h: a.compute() for h, a in acc.items()}
    vals = [per[h]["mIoU"] for h in REPORT_HORIZONS_S]
    return {"mIoU": float(np.nanmean(vals)), "per_horizon": per}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prepared", required=True)
    p.add_argument("--vae-ckpt", required=True)
    p.add_argument("--wm-ckpt", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-windows", type=int, default=None)
    p.add_argument("--vae-mode", choices=["sample", "mean"], default="sample")
    p.add_argument("--seed", type=int, default=20260826)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    ds = PreparedShardDataset(a.prepared)
    n = len(ds) if a.max_windows is None else min(len(ds), a.max_windows)
    if n == 0:
        raise RuntimeError("no prepared windows")
    vae, _ = load_official_vae(UP, a.vae_ckpt, a.device)
    wm, _ = load_official_wm(UP, a.wm_ckpt, a.device)
    adapter = OccFMVAEAdapter(vae)

    branches = ("full", "static", "moving")
    moving_metric = {b: MovingMIoUV2MultiHorizon() for b in branches}
    overall = {b: {h: SemanticIoUAccumulator() for h in REPORT_HORIZONS_S} for b in branches}
    true_static = {b: {h: SemanticIoUAccumulator() for h in REPORT_HORIZONS_S} for b in branches}

    for i in range(n):
        s = ds[i]
        seed = a.seed + i * 101
        # Same epsilon seed across the three branch encodings is intentional.
        z_full = adapter.encode(torch.from_numpy(s["full_history_occ"]).unsqueeze(0),
                                mode=a.vae_mode, seed=seed)[0].cpu()
        z_static = adapter.encode(torch.from_numpy(s["static_history_occ"]).unsqueeze(0),
                                  mode=a.vae_mode, seed=seed)[0].cpu()
        z_moving = adapter.encode(torch.from_numpy(s["moving_history_occ"]).unsqueeze(0),
                                  mode=a.vae_mode, seed=seed)[0].cpu()
        # Values are not exposed to the sampler; this tensor only establishes
        # the official six-frame future shape and loss target.
        z_future = adapter.encode(torch.from_numpy(s["future_gt_occ"]).unsqueeze(0),
                                  mode=a.vae_mode, seed=seed + 1)[0].cpu()
        traj = torch.as_tensor(s["trajectory"], dtype=torch.float32)

        pred = {}
        for bi, (name, hist) in enumerate((("full", z_full), ("static", z_static), ("moving", z_moving))):
            pred[name] = run_frozen_occfm_forecast(
                wm, hist, z_future, trajectory=traj,
                seed=a.seed + i * 1009, hist_last=4,
            ).numpy()

        for h in REPORT_HORIZONS_S:
            fi = REPORT_INDEX[h]
            gt = s["future_gt_occ"][fi]
            support = s["gt_moving_support"][fi]
            static_mask = gt_true_static_mask(gt, support)
            for name in branches:
                overall[name][h].update(pred[name][fi], gt)
                true_static[name][h].update(pred[name][fi], gt, static_mask)
                moving_metric[name].update(h, pred[name][fi], gt, support)

        if i % 10 == 0:
            print("P0-A", i, "/", n, s["sample_id"])

    report = {
        "protocol": "P0-A_frozen_causal_real_motion_decomposition",
        "num_windows": n,
        "vae_mode": a.vae_mode,
        "branches": {},
    }
    for name in branches:
        report["branches"][name] = {
            "overall_mIoU": mean_horizon(overall[name]),
            "true_static_mIoU": mean_horizon(true_static[name]),
            "Moving-mIoU_v2": moving_metric[name].compute(),
        }
    # Convenient deltas for the intended separability test.
    report["separability"] = {
        "static_only_minus_full_true_static_pp": (
            report["branches"]["static"]["true_static_mIoU"]["mIoU"]
            - report["branches"]["full"]["true_static_mIoU"]["mIoU"]
        ),
        "moving_only_minus_full_moving_pp": (
            report["branches"]["moving"]["Moving-mIoU_v2"]["mIoU"]
            - report["branches"]["full"]["Moving-mIoU_v2"]["mIoU"]
        ),
    }
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["separability"], indent=2))
    print("saved", a.output)


if __name__ == "__main__":
    main()
