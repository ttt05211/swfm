#!/usr/bin/env python3
"""Build the frozen Top-2 MSP cache for anchor-centered Sparse-WM training.

The router is frozen.  Future GT is used only to build the World-Model target;
it never enters MSP features or window selection.  The VAE uses deterministic
FP32 mean latents so the target ``z_gt-z_anchor`` is not polluted by independent
posterior sampling noise.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UP = ROOT / "upstream_occfm"
sys.path[:0] = [str(ROOT), str(UP)]

import numpy as np
import torch

from real_motion.composition import static_protected_compose
from real_motion.metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS
from real_motion.msp import (
    FEATURE_DIM, MSPProbeHead, collate_probe_records, rasterize_msp_scores,
)
from real_motion.msp_window import plan_topk_score_windows, score_capture_ratio
from real_motion.msp_wm_cache import save_msp_wm_shards
from real_motion.occfm_io import OccFMVAEAdapter, file_sha256, load_official_vae
from real_motion.prepared import PreparedShardDataset, PREPARED_VERSION
from real_motion.runtime_config import get_cfg, make_prepare_config


def _load_probe_cache(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    records = obj.get("records", [])
    meta = obj.get("metadata", {})
    if not records:
        raise RuntimeError("MSP probe cache has no records")
    return meta, records


def _load_msp(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if int(ck.get("feature_dim", -1)) != FEATURE_DIM:
        raise RuntimeError("MSP checkpoint feature contract mismatch")
    model = MSPProbeHead(
        feature_dim=FEATURE_DIM,
        hidden_dim=int(ck["hidden_dim"]),
        num_heads=int(ck["num_heads"]),
        num_modes=int(ck["num_modes"]),
        future_frames=int(ck["future_frames"]),
    )
    model.load_state_dict(ck["state_dict"], strict=True)
    return ck, model.to(device).eval()


def _compose_anchor(sample):
    static = torch.from_numpy(np.asarray(sample["static_future_occ"]))
    kta = torch.from_numpy(np.asarray(sample["kta_future_occ"]))
    protected = torch.from_numpy(np.asarray(sample["confident_static_future_mask"]))
    return static_protected_compose(
        static, kta, protected, DYNAMIC_CLASS_IDS, write_support=None
    ).cpu()


def _prepared_index(ds):
    out = {}
    for i, e in enumerate(ds.entries):
        sid = str(e.get("sample_id", ""))
        if sid in out:
            raise RuntimeError(f"duplicate prepared sample_id {sid}")
        out[sid] = i
    return out


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prepared", required=True)
    p.add_argument("--msp-cache", required=True)
    p.add_argument("--msp-checkpoint", required=True)
    p.add_argument("--vae-ckpt", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--topk", type=int, default=2)
    p.add_argument("--route-batch-size", type=int, default=16)
    p.add_argument("--vae-batch-size", type=int, default=4)
    p.add_argument("--shard-size", type=int, default=32)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    if a.topk != 2:
        raise ValueError("P0-F3 main training is frozen to Top-2; test Top-1/3 only after Top-2")
    if min(a.route_batch_size, a.vae_batch_size, a.shard_size) <= 0:
        raise ValueError("batch/shard sizes must be positive")

    device = torch.device(a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu")
    probe_meta, records = _load_probe_cache(a.msp_cache)
    msp_ck, msp = _load_msp(a.msp_checkpoint, device)
    cfg = msp_ck["resolved_config"]
    pcfg = make_prepare_config(cfg)
    latent_hw = tuple(int(v) for v in get_cfg(cfg, "UPSTREAM.LATENT_HW", [50, 50]))
    window_hw = tuple(int(v) for v in get_cfg(cfg, "MODEL.WINDOW_HW", [20, 20]))
    if latent_hw != (50, 50) or window_hw != (20, 20):
        raise RuntimeError("P0-F3 frozen route expects 50x50 latent and 20x20 WM windows")
    if int(msp_ck["future_frames"]) != 6:
        raise RuntimeError("P0-F3 expects 6 future frames")

    prepared = PreparedShardDataset(a.prepared)
    pmap = _prepared_index(prepared)
    missing = [str(r["sample_id"]) for r in records if str(r["sample_id"]) not in pmap]
    if missing:
        raise RuntimeError(
            f"prepared dataset misses {len(missing)} MSP samples, e.g. {missing[:3]}; "
            "build prepared assets covering the exact MSP probe windows"
        )

    vae, _ = load_official_vae(UP, a.vae_ckpt, device)
    va = OccFMVAEAdapter(vae)
    scenes = sorted({str(r["scene_name"]) for r in records})
    route_capture = []

    def generated_samples():
        for route_start in range(0, len(records), a.route_batch_size):
            route_records = records[route_start:route_start + a.route_batch_size]
            batch = collate_probe_records(route_records)
            bdev = {
                k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                for k, v in batch.items()
            }
            pred = msp(bdev["features"], bdev["candidate_mask"])
            scores = rasterize_msp_scores(pred, bdev, latent_hw=latent_hw, grid=pcfg.grid)
            plan = plan_topk_score_windows(
                scores, window_hw=window_hw, max_windows=a.topk
            )
            capture = score_capture_ratio(scores, plan).detach().cpu().tolist()
            route_capture.extend(float(x) for x in capture)
            origins_all = plan.origins.detach().cpu()
            valid_all = plan.valid.detach().cpu()

            for sub_start in range(0, len(route_records), a.vae_batch_size):
                sub = route_records[sub_start:sub_start + a.vae_batch_size]
                rows = [prepared[pmap[str(r["sample_id"])]] for r in sub]
                moving = torch.from_numpy(np.stack([np.asarray(s["moving_history_occ"]) for s in rows]))
                anchor = torch.stack([_compose_anchor(s) for s in rows])
                gt = torch.from_numpy(np.stack([np.asarray(s["future_gt_occ"]) for s in rows]))

                # Deterministic FP32 posterior means: no VAE sample noise is
                # allowed to masquerade as a motion residual target.
                zh = va.encode(moving, mode="mean").float().cpu()
                za = va.encode(anchor, mode="mean").float().cpu()
                zg = va.encode(gt, mode="mean").float().cpu()

                off = sub_start
                for j, (r, s) in enumerate(zip(sub, rows)):
                    pi = off + j
                    yield {
                        "sample_id": str(r["sample_id"]),
                        "scene_name": str(r["scene_name"]),
                        "moving_history_latent": zh[j],
                        "anchor_future_latent": za[j],
                        "gt_future_latent": zg[j],
                        "window_origins": origins_all[pi].clone(),
                        "window_valid": valid_all[pi].clone(),
                        "trajectory": torch.as_tensor(s["trajectory"], dtype=torch.float32),
                    }
            done = min(route_start + a.route_batch_size, len(records))
            print(f"routed+encoded {done}/{len(records)}")

    metadata = {
        "protocol": "p0_f3_top2_anchor_wm_cache_v1",
        "prepared": str(Path(a.prepared).resolve()),
        "prepared_version": PREPARED_VERSION,
        "msp_probe_cache": str(Path(a.msp_cache).resolve()),
        "msp_checkpoint": str(Path(a.msp_checkpoint).resolve()),
        "msp_checkpoint_sha256": file_sha256(a.msp_checkpoint),
        "vae_checkpoint": str(Path(a.vae_ckpt).resolve()),
        "vae_checkpoint_sha256": file_sha256(a.vae_ckpt),
        "vae_mode": "mean",
        "vae_amp": False,
        "topk": int(a.topk),
        "latent_hw": list(latent_hw),
        "window_hw": list(window_hw),
        "shared_windows_across_horizons": True,
        "score_aggregation": "sum_over_6_future_horizons",
        "selection": "greedy_marginal_msp_score",
        "target": "full_gt_latent_in_selected_window",
        "source": "causal_kta_zero_anchor_latent",
        "loss_contract": "anchor_to_gt_local_flow_no_auxiliary_losses",
        "trajectory_length": 12,
        "num_records": len(records),
        "num_unique_scenes": len(scenes),
        "scene_names": scenes,
        "probe_selection": probe_meta.get("selection"),
        "probe_seed": probe_meta.get("seed"),
    }
    idx = save_msp_wm_shards(
        a.output, generated_samples(), shard_size=a.shard_size, metadata=metadata
    )
    ip = Path(a.output) / "index.json"
    obj = json.loads(ip.read_text(encoding="utf-8"))
    obj["metadata"]["mean_score_capture_ratio"] = float(np.mean(route_capture)) if route_capture else 0.0
    obj["metadata"]["mean_valid_windows"] = float(
        sum(int(s) for s in [])
    ) if False else None
    ip.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    print("saved", idx["num_samples"], "samples to", a.output)
    print("mean MSP score capture:", obj["metadata"]["mean_score_capture_ratio"])


if __name__ == "__main__":
    main()
