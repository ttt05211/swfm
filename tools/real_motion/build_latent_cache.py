#!/usr/bin/env python3
"""Encode prepared occupancy branches with the frozen official OccFM VAE."""
import argparse, sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
UP = ROOT / "upstream_occfm"
sys.path[:0] = [str(UP), str(ROOT)]

import torch
from real_motion.prepared import PreparedShardDataset
from real_motion.cache import save_sharded_cache
from real_motion.occfm_io import load_official_vae, OccFMVAEAdapter, file_sha256
from real_motion.support import downsample_support


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prepared", required=True)
    p.add_argument("--vae-ckpt", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--empty-latent", required=True)
    p.add_argument("--mode", choices=["sample", "mean"], default="sample")
    p.add_argument("--seed", type=int, default=20260826)
    p.add_argument("--shard-size", type=int, default=256)
    p.add_argument("--latent-extra-radius", type=int, default=1)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    vae, _ = load_official_vae(UP, a.vae_ckpt, a.device)
    adapter = OccFMVAEAdapter(vae)
    prepared = PreparedShardDataset(a.prepared)

    # One fixed empty sample is part of the representation contract. Use the
    # same latent mode and a dedicated fixed seed; never substitute numeric 0.
    empty = adapter.empty_latent(mode=a.mode, seed=a.seed + 999).detach().cpu()
    Path(a.empty_latent).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"empty_latent": empty, "mode": a.mode, "seed": a.seed + 999}, a.empty_latent)

    def samples():
        for i in range(len(prepared)):
            s = prepared[i]
            # Deterministic per-sample stochastic latents. P0 paired branch tests
            # intentionally share epsilon across branches; training cache branches
            # use independent but reproducible seeds, matching a stochastic VAE.
            base_seed = a.seed + i * 17
            moving_hist = adapter.encode(torch.from_numpy(s["moving_history_occ"]).unsqueeze(0),
                                         mode=a.mode, seed=base_seed)[0].cpu()
            future_moving = adapter.encode(torch.from_numpy(s["future_dynamic_target_occ"]).unsqueeze(0),
                                           mode=a.mode, seed=base_seed + 1)[0].cpu()
            static_future = adapter.encode(torch.from_numpy(s["static_future_occ"]).unsqueeze(0),
                                           mode=a.mode, seed=base_seed + 2)[0].cpu()
            kta_future = adapter.encode(torch.from_numpy(s["kta_future_occ"]).unsqueeze(0),
                                        mode=a.mode, seed=base_seed + 3)[0].cpu()

            gen = downsample_support(
                torch.from_numpy(s["generation_support_occ"]).bool(), (50, 50),
                extra_radius=a.latent_extra_radius,
            ).cpu()
            hist_context = downsample_support(
                torch.from_numpy(s["history_candidate_support"]).bool(), (50, 50),
                extra_radius=a.latent_extra_radius,
            ).cpu()
            planning = torch.cat([hist_context, gen], dim=0)
            sample = {
                "sample_id": s["sample_id"],
                "moving_history_latent": moving_hist,
                "future_dynamic_target_latent": future_moving,
                "static_future_latent": static_future,
                "kta_future_latent": kta_future,
                "generation_support": gen,
                "planning_support": planning,
                "trajectory": torch.as_tensor(s["trajectory"], dtype=torch.float32),
            }
            if i % 50 == 0:
                print("encoded", i, s["sample_id"], "active", float(gen.float().mean()))
            yield sample

    meta = {
        "prepared": str(Path(a.prepared).resolve()),
        "vae_ckpt": str(Path(a.vae_ckpt).resolve()),
        "vae_ckpt_sha256": file_sha256(a.vae_ckpt),
        "latent_mode": a.mode,
        "seed": a.seed,
        "latent_extra_radius": a.latent_extra_radius,
    }
    index = save_sharded_cache(a.output, samples(), a.shard_size, meta)
    print("saved", index["num_samples"], "latent samples to", a.output)
    print("empty latent:", a.empty_latent)


if __name__ == "__main__":
    main()
