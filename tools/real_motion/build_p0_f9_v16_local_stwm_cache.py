#!/usr/bin/env python3
"""Augment the v13 Strong-source motion cache with causal local semantic tubes.

The expensive Strong decomposition, source matching, history tracking, kinematic
features and corrected displacement targets are reused verbatim from v13.  This
builder reads only the six *history* occupancy grids, ego-compensates them to t0,
and extracts source-centered local semantic BEV tubes.  No future occupancy or
future annotation is read.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from real_motion.cache_pipeline import bounded_ordered_parallel_map
from real_motion.local_st_world_model import (
    DEFAULT_PATCH_RESOLUTION_M,
    DEFAULT_PATCH_SIZE_M,
    LOCAL_STWM_CACHE_VERSION,
    LOCAL_TUBE_CONTRACT,
    build_local_semantic_tubes,
    history_offsets_from_features,
)
from real_motion.motion_transport import FEATURE_DIM, HISTORY_FRAMES
from real_motion.motion_transport_v2 import MOTION_TRANSPORT_CACHE_VERSION, TARGET_CONTRACT
from real_motion.nuscenes_adapter import NuScenesWindowSource
from real_motion.runtime_config import add_config_args, config_fingerprint, load_runtime_config, make_prepare_config


class CachedSource(NuScenesWindowSource):
    @lru_cache(maxsize=1024)
    def load_semantics(self, scene_name, token):
        return super().load_semantics(scene_name, token)

    @lru_cache(maxsize=4096)
    def pose(self, token):
        return super().pose(token)


def load_base_cache(path: str):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if obj.get("version") != MOTION_TRANSPORT_CACHE_VERSION:
        raise RuntimeError(f"expected v13 motion cache {MOTION_TRANSPORT_CACHE_VERSION}")
    meta = dict(obj.get("metadata") or {})
    if meta.get("target_contract") != TARGET_CONTRACT:
        raise RuntimeError("base cache target contract mismatch")
    if int(meta.get("feature_dim", -1)) != FEATURE_DIM:
        raise RuntimeError("base cache feature dimension mismatch")
    records = obj.get("records") or []
    if not records:
        raise RuntimeError("base cache contains no records")
    return meta, records


def augment_one(source, rec, pcfg, patch_size_m: float, patch_resolution_m: float):
    history_tokens = tuple(str(x) for x in rec["history_tokens"])
    if len(history_tokens) != HISTORY_FRAMES:
        raise RuntimeError(f"{rec['sample_id']}: history length mismatch")
    history_occ = [source.load_semantics(str(rec["scene_name"]), tok) for tok in history_tokens]
    history_poses = [np.asarray(source.pose(tok), dtype=np.float64) for tok in history_tokens]
    source_xy = rec["source_centroid_xy_t0_m"].float().numpy()
    offsets = history_offsets_from_features(rec["features"])
    valid = rec["track_valid"].bool().numpy()
    tube = build_local_semantic_tubes(
        history_occ,
        history_poses,
        source_xy,
        offsets,
        valid,
        grid=pcfg.grid,
        free_label=int(pcfg.free_label),
        patch_size_m=float(patch_size_m),
        patch_resolution_m=float(patch_resolution_m),
    )
    out = dict(rec)
    out["local_semantic_tube"] = torch.from_numpy(tube)
    return out


def main():
    p = argparse.ArgumentParser()
    add_config_args(p)
    p.add_argument("--base-cache", required=True, help="v13 displacement-preserving motion cache")
    p.add_argument("--dataroot", required=True)
    p.add_argument("--info-pkl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--patch-size-m", type=float, default=DEFAULT_PATCH_SIZE_M)
    p.add_argument("--patch-resolution-m", type=float, default=DEFAULT_PATCH_RESOLUTION_M)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--prefetch-windows", type=int, default=0)
    p.add_argument("--max-windows", type=int, default=0)
    a = p.parse_args()

    cfg = load_runtime_config(a.config, a.override)
    pcfg = make_prepare_config(cfg)
    base_meta, base_records = load_base_cache(a.base_cache)
    expected = base_meta.get("config_contract_sha256")
    got = config_fingerprint(cfg, "cache")
    if expected and expected != got:
        raise RuntimeError("runtime config differs from base motion-cache contract")
    if int(a.max_windows) > 0:
        base_records = base_records[: min(len(base_records), int(a.max_windows))]

    workers = int(a.workers) if int(a.workers) > 0 else min(8, max(1, os.cpu_count() or 1))
    prefetch = int(a.prefetch_windows) if int(a.prefetch_windows) > 0 else 4 * workers
    source = CachedSource(a.dataroot, info_pkl=a.info_pkl, verbose=False)

    print(json.dumps({
        "protocol": "p0_f9_v16_local_stwm_cache_builder_v1",
        "windows": len(base_records),
        "workers": workers,
        "prefetch_windows": prefetch,
        "patch_size_m": float(a.patch_size_m),
        "patch_resolution_m": float(a.patch_resolution_m),
        "base_cache_version": MOTION_TRANSPORT_CACHE_VERSION,
        "target_contract": TARGET_CONTRACT,
    }))

    def fn(rec):
        return augment_one(
            source, rec, pcfg,
            patch_size_m=float(a.patch_size_m),
            patch_resolution_m=float(a.patch_resolution_m),
        )

    started = time.perf_counter()
    records = []
    total_sources = 0
    for i, out in enumerate(bounded_ordered_parallel_map(
        fn,
        base_records,
        max_workers=workers,
        max_in_flight=prefetch,
        thread_name_prefix="local-stwm",
    ), start=1):
        records.append(out)
        total_sources += int(out["local_semantic_tube"].shape[0])
        if i == 1 or i % 50 == 0 or i == len(base_records):
            elapsed = max(time.perf_counter() - started, 1e-9)
            print(
                f"local_stwm_cache {i}/{len(base_records)} sources={total_sources} "
                f"rate={i/elapsed:.2f} win/s occ_cache={source.load_semantics.cache_info()} "
                f"pose_cache={source.pose.cache_info()}"
            )

    tube_hw = 0
    if records and records[0]["local_semantic_tube"].ndim == 4:
        tube_hw = int(records[0]["local_semantic_tube"].shape[-1])
    metadata = dict(base_meta)
    metadata.update({
        "version": LOCAL_STWM_CACHE_VERSION,
        "base_cache": str(Path(a.base_cache).resolve()),
        "base_cache_version": MOTION_TRANSPORT_CACHE_VERSION,
        "target_contract": TARGET_CONTRACT,
        "local_tube_contract": LOCAL_TUBE_CONTRACT,
        "local_tube_is_causal": True,
        "local_tube_reads_future_occupancy": False,
        "local_tube_reads_future_annotations": False,
        "patch_size_m": float(a.patch_size_m),
        "patch_resolution_m": float(a.patch_resolution_m),
        "tube_hw": tube_hw,
        "tube_history_frames": HISTORY_FRAMES,
        "tube_dtype": "uint8",
        "num_windows": len(records),
        "num_sources": total_sources,
    })
    payload = {"version": LOCAL_STWM_CACHE_VERSION, "metadata": metadata, "records": records}
    op = Path(a.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, op)
    op.with_suffix(".summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(f"saved {op}")


if __name__ == "__main__":
    main()
