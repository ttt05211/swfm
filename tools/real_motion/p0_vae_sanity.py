#!/usr/bin/env python3
"""P0-E: frozen VAE moving-only reconstruction and sparse-canvas sanity."""
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
UP = ROOT / "upstream_occfm"
sys.path[:0] = [str(UP), str(ROOT)]

import torch
from real_motion.prepared import PreparedShardDataset
from real_motion.occfm_io import load_official_vae, OccFMVAEAdapter
from real_motion.support import downsample_support
from real_motion.metrics.moving_miou_v2 import MovingMIoUV2MultiHorizon

REPORT = {1.0: 1, 2.0: 3, 3.0: 5}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prepared", required=True)
    p.add_argument("--vae-ckpt", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-windows", type=int, default=None)
    p.add_argument("--mode", choices=["sample", "mean"], default="sample")
    p.add_argument("--seed", type=int, default=20260826)
    p.add_argument("--latent-extra-radius", type=int, default=1)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    ds = PreparedShardDataset(a.prepared)
    n = len(ds) if a.max_windows is None else min(len(ds), a.max_windows)
    vae, _ = load_official_vae(UP, a.vae_ckpt, a.device)
    adapter = OccFMVAEAdapter(vae)
    empty = adapter.empty_latent(mode=a.mode, seed=a.seed + 999)
    full_metric = MovingMIoUV2MultiHorizon()
    causal_sparse_metric = MovingMIoUV2MultiHorizon()
    gt_support_sparse_metric = MovingMIoUV2MultiHorizon()
    rms_sum, rms_n = 0.0, 0

    for i in range(n):
        s = ds[i]
        z = adapter.encode(torch.from_numpy(s["future_moving_occ"]).unsqueeze(0),
                           mode=a.mode, seed=a.seed + i)[0]
        pred_full = adapter.decode_labels(z).cpu().numpy()

        causal = downsample_support(
            torch.from_numpy(s["generation_support_occ"]).bool(), (50,50),
            extra_radius=a.latent_extra_radius,
        ).to(z.device)
        gt_lat = downsample_support(
            torch.from_numpy(s["gt_moving_support"]).any(dim=-1), (50,50),
            extra_radius=a.latent_extra_radius,
        ).to(z.device)
        empty_seq = empty[None].expand(z.shape[0], -1, -1, -1).to(z.dtype)
        z_causal = torch.where(causal[:,None], z, empty_seq)
        z_gtmask = torch.where(gt_lat[:,None], z, empty_seq)
        pred_causal = adapter.decode_labels(z_causal).cpu().numpy()
        pred_gtmask = adapter.decode_labels(z_gtmask).cpu().numpy()

        outside = ~causal[:,None].expand_as(z)
        if bool(outside.any()):
            rms_sum += float(((z_causal[outside] - empty_seq[outside]) ** 2).sum().cpu())
            rms_n += int(outside.sum().cpu())

        for h, fi in REPORT.items():
            gt = s["future_gt_occ"][fi]
            support = s["gt_moving_support"][fi]
            full_metric.update(h, pred_full[fi], gt, support)
            causal_sparse_metric.update(h, pred_causal[fi], gt, support)
            gt_support_sparse_metric.update(h, pred_gtmask[fi], gt, support)
        if i % 25 == 0:
            print("VAE sanity", i, "/", n)

    report = {
        "num_windows": n,
        "latent_mode": a.mode,
        "full_moving_reconstruction": full_metric.compute(),
        "causal_generation_support_canvas": causal_sparse_metric.compute(),
        "gt_support_canvas_diagnostic": gt_support_sparse_metric.compute(),
        "outside_causal_support_empty_latent_rms": (rms_sum / max(rms_n,1)) ** 0.5,
    }
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
