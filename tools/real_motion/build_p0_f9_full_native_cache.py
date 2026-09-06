#!/usr/bin/env python3
"""Build the full-data P0-F9 native cache directly from the full MSP probe.

This avoids constructing an intermediate P0-F7 repair-endpoint cache.  It keeps
exactly the scientific contracts needed by the final P0-F9/M recipe:

- all eligible train 6+6 windows from ``build_p0_f9_full_msp_cache.py``;
- frozen MSP checkpoint, Top-2 20x20 routing and 15% write support;
- Strong-W2Det future as physics condition/fallback only;
- official OccFM VAE posterior *samples* for history, physics and absolute GT;
- Gaussian source -> absolute future native FM target.

No semantic decoder targets and no ordered-context artifacts are produced.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
UP = ROOT / "upstream_occfm"
sys.path[:0] = [str(ROOT), str(UP)]

import numpy as np
import torch

from real_motion.cache_pipeline import bounded_ordered_parallel_map
from real_motion.native_forecast import deterministic_sample_seed
from real_motion.occfm_io import OccFMVAEAdapter, file_sha256, load_official_vae
from real_motion.prepared import load_nuscenes_window_raw
from real_motion.runtime_config import get_cfg, make_prepare_config
from real_motion.strong_w2det import StrongW2DetConfig, strong_w2det_sequence
from tools.real_motion import build_p0_f5_cache_direct as base
from tools.real_motion.build_p0_f7_cache_fast import _resolve_device, _routes
from tools.real_motion.build_p0_f9_cache_fast import (
    P0_F9_CACHE_PROTOCOL,
    _Writer,
    _host_occ,
)
from tools.real_motion.build_p0_f9_full_msp_cache import PROTOCOL as FULL_MSP_PROTOCOL


PROTOCOL = "p0_f9_full_native_cache_direct_v1"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--msp-cache", required=True)
    p.add_argument("--msp-checkpoint", required=True)
    p.add_argument("--dataroot", required=True)
    p.add_argument("--info-pkl", required=True)
    p.add_argument("--vae-ckpt", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--write-budget-ratio", type=float, default=0.15)
    p.add_argument("--route-batch-size", type=int, default=128)
    p.add_argument("--vae-batch-size", type=int, default=16)
    p.add_argument("--prepare-workers", type=int, default=0)
    p.add_argument("--prefetch-windows", type=int, default=0)
    p.add_argument("--shard-size", type=int, default=32)
    p.add_argument("--latent-seed", type=int, default=20260904)
    p.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    if not 0.0 < float(a.write_budget_ratio) <= 1.0:
        raise ValueError("write-budget-ratio must be in (0,1]")
    if min(a.route_batch_size, a.vae_batch_size, a.shard_size) <= 0:
        raise ValueError("route/vae/shard sizes must be positive")

    device = _resolve_device(a.device)
    if device.type == "cuda":
        torch.cuda.set_device(int(device.index))
    workers = int(a.prepare_workers) if int(a.prepare_workers) > 0 else min(
        16, max(1, os.cpu_count() or 1)
    )
    prefetch = int(a.prefetch_windows) if int(a.prefetch_windows) > 0 else 4 * workers

    probe_meta, records, cfg = base._load_probe(a.msp_cache)
    if probe_meta.get("protocol") != FULL_MSP_PROTOCOL:
        raise RuntimeError("--msp-cache is not the audited full eligible MSP cache")
    if probe_meta.get("mode") != "train" or not bool(probe_meta.get("all_eligible_windows", False)):
        raise RuntimeError("full native cache requires all eligible train MSP windows")
    if int(probe_meta.get("num_windows", -1)) != len(records):
        raise RuntimeError("full MSP metadata count differs from records")
    sample_ids = [str(r["sample_id"]) for r in records]
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("full MSP cache contains duplicate sample IDs")

    msp_ck, msp = base._load_msp(a.msp_checkpoint, device)
    pcfg = make_prepare_config(cfg)
    if int(pcfg.history_frames) != 6 or int(pcfg.future_frames) != 6:
        raise RuntimeError("full P0-F9 native cache requires 6+6 windows")
    latent_hw = tuple(int(v) for v in get_cfg(cfg, "UPSTREAM.LATENT_HW", [50, 50]))
    window_hw = tuple(int(v) for v in get_cfg(cfg, "MODEL.WINDOW_HW", [20, 20]))
    if latent_hw != (50, 50) or window_hw != (20, 20):
        raise RuntimeError("full P0-F9 requires 50x50 latent and Top-2 20x20 windows")
    if int(msp_ck["future_frames"]) != int(pcfg.future_frames):
        raise RuntimeError("MSP checkpoint future-frame contract mismatch")

    print("routing full eligible MSP population with frozen checkpoint")
    route_map, captures, valid_counts, write_ratios = _routes(
        records,
        msp=msp,
        pcfg=pcfg,
        device=device,
        latent_hw=latent_hw,
        window_hw=window_hw,
        batch_size=int(a.route_batch_size),
        write_budget=float(a.write_budget_ratio),
    )
    if set(route_map) != set(sample_ids):
        raise RuntimeError("frozen MSP routing did not cover the exact full population")

    msp_sha = file_sha256(a.msp_cache)
    msp_ck_sha = file_sha256(a.msp_checkpoint)
    vae_sha = file_sha256(a.vae_ckpt)
    scenes = sorted({str(r["scene_name"]) for r in records})
    w2cfg = StrongW2DetConfig(free_label=int(pcfg.free_label))
    metadata = {
        "protocol": P0_F9_CACHE_PROTOCOL,
        "build_protocol": PROTOCOL,
        "direct_full_data_build": True,
        "all_eligible_windows": True,
        "native_eligible_window_count": int(probe_meta.get("native_eligible_window_count", len(records))),
        # Kept explicitly null so the existing resumable v2 writer can reuse its
        # provenance checks without pretending an intermediate v3 cache exists.
        "source_v3_cache": None,
        "source_v3_cache_index_sha256": None,
        "source_msp_cache": str(Path(a.msp_cache).resolve()),
        "source_msp_cache_sha256": msp_sha,
        "source_msp_mode": "train",
        "source_msp_selection": probe_meta.get("selection"),
        "msp_checkpoint": str(Path(a.msp_checkpoint).resolve()),
        "msp_checkpoint_sha256": msp_ck_sha,
        "vae_checkpoint": str(Path(a.vae_ckpt).resolve()),
        "vae_checkpoint_sha256": vae_sha,
        "vae_mode": "sample",
        "vae_sample_seed_base": int(a.latent_seed),
        "vae_sample_seed_contract": "sha256(base,stream,sample_id)_per_video_sample_v1",
        "latent_dtype": "float32",
        "topk": 2,
        "latent_hw": [50, 50],
        "window_hw": [20, 20],
        "context_hw": [40, 40],
        "trajectory_length": int(pcfg.trajectory_length),
        "write_budget_ratio": float(a.write_budget_ratio),
        "mean_write_latent_ratio": float(np.mean(write_ratios)) if write_ratios else 0.0,
        "mean_score_capture_ratio": float(np.mean(captures)) if captures else 0.0,
        "mean_valid_windows": float(np.mean(valid_counts)) if valid_counts else 0.0,
        "slot_compute_ratio": (
            float(np.mean(valid_counts) * 400.0 / 2500.0) if valid_counts else 0.0
        ),
        "history_contract": "full_native_occ_history_6f",
        "native_backbone_hist_last": 4,
        "anchor_contract": "strong_w2det_occ_only_v1",
        "w2det_min_component_voxels": int(w2cfg.min_component_voxels),
        "w2det_max_match_speed_mps": float(w2cfg.max_match_speed_mps),
        "w2det_connectivity": int(w2cfg.connectivity),
        "target": "absolute_gt_future_vae_latent",
        "flow_source": "gaussian_noise_not_anchor",
        "num_unique_scenes": len(scenes),
        "scene_names": scenes,
        "include_eval_payload": False,
        "reencoded_tensor_keys": [
            "full_history_latent",
            "anchor_future_latent",
            "gt_future_latent",
        ],
        "reuse_note": (
            "direct full-data build: exact MSP records are routed by frozen MSP; "
            "history/Strong-W2Det/absolute-GT are posterior-sampled with the official VAE"
        ),
    }
    writer = _Writer(Path(a.output), metadata, shard_size=int(a.shard_size), resume=bool(a.resume))

    source = base.CachedNuScenesWindowSource(a.dataroot, info_pkl=a.info_pkl, verbose=False)
    vae_model, _ = load_official_vae(UP, a.vae_ckpt, device)
    vae = OccFMVAEAdapter(vae_model)

    record_by_id = {str(r["sample_id"]): r for r in records}
    work = [sid for sid in sample_ids if sid not in writer.seen]
    started = time.perf_counter()

    def prepare_one(sid: str):
        rec = record_by_id[str(sid)]
        window = base._window_from_record(rec, pcfg.history_frames, pcfg.future_frames)
        raw = load_nuscenes_window_raw(source, window, pcfg, include_gt=True)
        history = np.asarray(raw["history_occ"], dtype=np.uint8)
        gt = np.asarray(raw["future_gt_occ"], dtype=np.uint8)
        anchor = strong_w2det_sequence(
            history,
            raw["history_poses"],
            raw["future_poses"],
            frame_dt_s=float(pcfg.frame_dt_s),
            grid=pcfg.grid,
            cfg=w2cfg,
        ).astype(np.uint8, copy=False)
        origins, valid, write = route_map[str(sid)]
        meta = {
            "sample_id": str(sid),
            "scene_name": str(window.scene_name),
            "window_origins": origins,
            "window_valid": valid,
            "msp_write_support_latent": write.bool(),
            "trajectory": torch.as_tensor(raw["trajectory"], dtype=torch.float32),
        }
        return meta, history, anchor, gt

    pending = []
    prepared = 0
    encoded = 0

    def flush() -> None:
        nonlocal pending, encoded
        if not pending:
            return
        pin = bool(a.pin_memory and device.type == "cuda")
        history = _host_occ(pending, 1, pin=pin)
        anchor = _host_occ(pending, 2, pin=pin)
        gt = _host_occ(pending, 3, pin=pin)
        if device.type == "cuda":
            history = history.to(device=device, non_blocking=pin)
            anchor = anchor.to(device=device, non_blocking=pin)
            gt = gt.to(device=device, non_blocking=pin)
        sids = [str(row[0]["sample_id"]) for row in pending]
        hist_seed = [deterministic_sample_seed(s, a.latent_seed, stream="history") for s in sids]
        anchor_seed = [deterministic_sample_seed(s, a.latent_seed, stream="physics") for s in sids]
        gt_seed = [deterministic_sample_seed(s, a.latent_seed, stream="future") for s in sids]
        zh = vae.encode(history, mode="sample", seed=hist_seed).float().cpu()
        za = vae.encode(anchor, mode="sample", seed=anchor_seed).float().cpu()
        zg = vae.encode(gt, mode="sample", seed=gt_seed).float().cpu()
        for j, (meta, _, _, _) in enumerate(pending):
            writer.add({
                "sample_id": meta["sample_id"],
                "scene_name": meta["scene_name"],
                "full_history_latent": zh[j],
                "anchor_future_latent": za[j],
                "gt_future_latent": zg[j],
                "window_origins": meta["window_origins"].cpu(),
                "window_valid": meta["window_valid"].cpu(),
                "msp_write_support_latent": meta["msp_write_support_latent"].cpu(),
                "trajectory": meta["trajectory"].cpu(),
            })
        encoded += len(pending)
        pending = []

    try:
        for row in bounded_ordered_parallel_map(
            prepare_one,
            work,
            max_workers=workers,
            max_in_flight=prefetch,
            thread_name_prefix="p0-f9-full-native",
        ):
            pending.append(row)
            prepared += 1
            if len(pending) >= int(a.vae_batch_size):
                flush()
            done = len(writer.entries) + len(writer.current) + len(pending)
            if prepared == 1 or done % 128 == 0 or prepared == len(work):
                elapsed = max(time.perf_counter() - started, 1e-9)
                print(
                    f"full P0-F9 cache {done}/{len(records)} "
                    f"prep_rate={prepared/elapsed:.2f} win/s encoded={encoded} "
                    f"occ_cache={source.load_occ3d.cache_info()} pose_cache={source.pose.cache_info()}"
                )
        flush()
        index = writer.close()
    except BaseException:
        if writer.current:
            writer._commit(writer.current)
            writer.current = []
        raise

    if int(index["num_samples"]) != len(records):
        raise RuntimeError(
            f"full P0-F9 cache incomplete: {index['num_samples']} != {len(records)}"
        )
    elapsed = max(time.perf_counter() - started, 1e-9)
    print(json.dumps({
        "output": str(Path(a.output).resolve()),
        "num_samples": int(index["num_samples"]),
        "num_scenes": len(scenes),
        "mean_valid_windows": metadata["mean_valid_windows"],
        "slot_compute_ratio": metadata["slot_compute_ratio"],
        "mean_write_latent_ratio": metadata["mean_write_latent_ratio"],
        "score_capture": metadata["mean_score_capture_ratio"],
        "vae_mode": "sample",
        "latent_seed": int(a.latent_seed),
        "elapsed_seconds": elapsed,
    }, indent=2))


if __name__ == "__main__":
    main()
