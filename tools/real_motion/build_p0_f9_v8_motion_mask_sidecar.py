#!/usr/bin/env python3
"""Build the P0-F9 v8 true-motion latent-mask training sidecar.

The train P0-F9 cache intentionally does not carry evaluation-only future
instance support.  For motion-weighted FM we therefore reconstruct each exact
nuScenes window from the frozen MSP provenance, compute the same Moving-mIoU-v2
dual-box support used by validation, then exact-any-pool 200x200 BEV support to
50x50 latent cells.  Only the boolean latent masks are persisted.

This artifact is training-label-only; it is never an inference input.
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

import torch

from real_motion.cache_pipeline import bounded_ordered_parallel_map
from real_motion.fm_group_diagnostics import motion_support_occ_to_latent
from real_motion.motion_mask_sidecar import (
    EXPECTED_MASK_SHAPE,
    MOTION_MASK_PROTOCOL,
    MOTION_MASK_SIDECAR_VERSION,
    effective_motion_weight_mass,
)
from real_motion.msp_wm_cache import MSP_WM_CACHE_VERSION_V2, MSPWorldModelCacheDataset
from real_motion.occfm_io import file_sha256
from real_motion.runtime_config import make_prepare_config
from real_motion.windows import WindowPlan, crop_windows
from tools.real_motion import build_p0_f4_cache_direct as base
from tools.real_motion.build_p0_f9_cache_fast import P0_F9_CACHE_PROTOCOL


PROTOCOL = "p0_f9_v8_train_motion_mask_builder_v1"


def _atomic_save(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _routed_counts(mask: torch.Tensor, sample: dict) -> tuple[int, int]:
    origins = sample["window_origins"].unsqueeze(0).long()
    valid = sample["window_valid"].unsqueeze(0).bool()
    plan = WindowPlan(origins, valid, (20, 20), (50, 50))
    full = mask[:, None].unsqueeze(0).float()  # [1,T,1,50,50]
    windows = crop_windows(full, plan)  # [1,K,T,1,20,20]
    flat_valid = plan.valid.reshape(-1)
    if not bool(flat_valid.any()):
        return 0, 0
    flat = windows.reshape(
        int(plan.valid.numel()), *windows.shape[2:]
    )[flat_valid]
    routed = flat[:, :, 0].bool()
    return int(routed.sum().item()), int(routed.numel())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-cache", required=True)
    p.add_argument("--msp-cache", required=True)
    p.add_argument("--dataroot", required=True)
    p.add_argument("--info-pkl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--prefetch-samples", type=int, default=32)
    p.add_argument("--motion-weight-lambda", type=float, default=2.0)
    a = p.parse_args()

    if a.workers <= 0 or a.prefetch_samples <= 0:
        raise ValueError("workers/prefetch-samples must be positive")
    if a.motion_weight_lambda < 0:
        raise ValueError("motion-weight-lambda must be non-negative")

    output = Path(a.output).expanduser().resolve()
    if output.exists():
        raise RuntimeError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    ds = MSPWorldModelCacheDataset(a.train_cache)
    if ds.version != MSP_WM_CACHE_VERSION_V2:
        raise RuntimeError("v8 motion masks require the P0-F9 v2 absolute-future cache")
    meta = ds.metadata
    checks = {
        "protocol": P0_F9_CACHE_PROTOCOL,
        "source_msp_mode": "train",
        "target": "absolute_gt_future_vae_latent",
        "topk": 2,
        "window_hw": [20, 20],
        "latent_hw": [50, 50],
    }
    for key, expected in checks.items():
        if meta.get(key) != expected:
            raise RuntimeError(
                f"train cache mismatch for {key}: {meta.get(key)!r} != {expected!r}"
            )

    msp_path = Path(a.msp_cache).expanduser().resolve()
    msp_sha = file_sha256(msp_path)
    expected_msp = meta.get("source_msp_cache_sha256")
    if expected_msp and msp_sha != expected_msp:
        raise RuntimeError("--msp-cache differs from train-cache provenance")

    probe_meta, records, cfg = base._load_probe(msp_path)
    if probe_meta.get("mode") != "train":
        raise RuntimeError("MSP probe is not the train split")
    record_map = {str(r["sample_id"]): r for r in records}
    if len(record_map) != len(records):
        raise RuntimeError("MSP probe has duplicate sample IDs")
    train_ids = [str(e["sample_id"]) for e in ds.entries]
    missing = sorted(set(train_ids) - set(record_map))
    if missing:
        raise RuntimeError(f"MSP probe misses train samples, e.g. {missing[:3]}")

    pcfg = make_prepare_config(cfg)
    source = base.CachedNuScenesWindowSource(
        a.dataroot, info_pkl=a.info_pkl, verbose=False
    )

    started = time.perf_counter()

    def build_one(i: int):
        sample = ds[int(i)]
        sid = str(sample["sample_id"])
        rec = record_map[sid]
        window = base._window_from_record(rec, pcfg.history_frames, pcfg.future_frames)
        if str(window.scene_name) != str(sample["scene_name"]):
            raise RuntimeError(f"{sid}: scene mismatch between train cache and MSP probe")
        moving_occ = base._gt_moving_support(source, window, pcfg)
        latent = motion_support_occ_to_latent(
            torch.from_numpy(moving_occ), latent_hw=(50, 50)
        ).bool().contiguous()
        if tuple(latent.shape) != EXPECTED_MASK_SHAPE:
            raise RuntimeError(f"{sid}: unexpected latent motion-mask shape {tuple(latent.shape)}")
        routed_motion, routed_total = _routed_counts(latent, sample)
        return {
            "sample_id": sid,
            "scene_name": str(sample["scene_name"]),
            "motion_mask_latent": latent,
            "full_motion_cells": int(latent.sum().item()),
            "full_cells": int(latent.numel()),
            "routed_motion_cells": int(routed_motion),
            "routed_cells": int(routed_total),
        }

    rows = []
    for row in bounded_ordered_parallel_map(
        build_one,
        range(len(ds)),
        max_workers=int(a.workers),
        max_in_flight=int(a.prefetch_samples),
        thread_name_prefix="v8-motion-mask",
    ):
        rows.append(row)
        n = len(rows)
        if n == 1 or n % 128 == 0 or n == len(ds):
            elapsed = max(time.perf_counter() - started, 1e-9)
            print(f"v8 motion masks {n}/{len(ds)} rate={n/elapsed:.2f} sample/s")

    if [r["sample_id"] for r in rows] != train_ids:
        raise RuntimeError("motion-mask sidecar order differs from train cache")

    full_motion = sum(r["full_motion_cells"] for r in rows)
    full_total = sum(r["full_cells"] for r in rows)
    routed_motion = sum(r["routed_motion_cells"] for r in rows)
    routed_total = sum(r["routed_cells"] for r in rows)
    full_fraction = full_motion / max(full_total, 1)
    routed_fraction = routed_motion / max(routed_total, 1)
    lam = float(a.motion_weight_lambda)

    metadata = {
        "protocol": PROTOCOL,
        "motion_mask_protocol": MOTION_MASK_PROTOCOL,
        "source_train_cache": str(Path(a.train_cache).expanduser().resolve()),
        "source_train_cache_index_sha256": file_sha256(Path(a.train_cache) / "index.json"),
        "source_msp_cache": str(msp_path),
        "source_msp_cache_sha256": msp_sha,
        "source_msp_mode": probe_meta.get("mode"),
        "source_msp_selection": probe_meta.get("selection"),
        "num_samples": len(rows),
        "mask_shape": list(EXPECTED_MASK_SHAPE),
        "occupancy_to_latent": "height_any_then_exact_4x4_xy_any_pool_no_extra_dilation",
        "motion_definition": (
            "Moving-mIoU-v2 dual-box old/future support; >=0.5m/s world-XY interval speed; "
            "0.5m box margin; training labels only"
        ),
        "full_latent_motion_fraction": full_fraction,
        "routed_top2_motion_fraction": routed_fraction,
        "preview_motion_lambda": lam,
        "preview_effective_motion_weight_mass": effective_motion_weight_mass(
            routed_fraction, lam
        ),
    }
    records = [
        {
            "sample_id": r["sample_id"],
            "scene_name": r["scene_name"],
            "motion_mask_latent": r["motion_mask_latent"],
        }
        for r in rows
    ]
    _atomic_save(
        output,
        {
            "version": MOTION_MASK_SIDECAR_VERSION,
            "metadata": metadata,
            "records": records,
        },
    )
    elapsed = max(time.perf_counter() - started, 1e-9)
    print(json.dumps({
        "output": str(output),
        "num_samples": len(rows),
        "full_latent_motion_fraction": full_fraction,
        "routed_top2_motion_fraction": routed_fraction,
        "lambda": lam,
        "effective_motion_weight_mass": metadata["preview_effective_motion_weight_mass"],
        "elapsed_seconds": elapsed,
    }, indent=2))


if __name__ == "__main__":
    main()
