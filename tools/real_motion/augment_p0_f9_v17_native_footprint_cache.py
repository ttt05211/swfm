#!/usr/bin/env python3
"""Augment a frozen V17 cache with exact native Strong-source XY footprints.

This is a one-time data preparation step for the paired B-C/B-S continuation.
It reads only the t0 occupancy/pose needed to reconstruct the same Strong sources
already represented in the V17 cache.  No future occupancy or future annotation
is read.  The existing V17 representation tensors are preserved verbatim.
"""
from __future__ import annotations

from functools import lru_cache
import argparse
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
from real_motion.local_st_world_model_v17 import LOCAL_STWM_V17_CACHE_VERSION
from real_motion.nuscenes_adapter import NuScenesWindowSource
from real_motion.runtime_config import add_config_args, load_runtime_config, make_prepare_config
from real_motion.source_footprint import (
    NATIVE_SOURCE_FOOTPRINT_CONTRACT,
    native_source_footprint_patch,
)
from real_motion.strong_w2det import StrongW2DetConfig, extract_instances
from tools.real_motion.train_p0_f9_v17_local_stwm import load_cache

FIELD = "native_source_footprint_mask"
PROTOCOL = "p0_f9_v17_native_source_footprint_cache_augment_v1"


class CachedSource(NuScenesWindowSource):
    @lru_cache(maxsize=2048)
    def load_semantics(self, scene_name, token):
        return super().load_semantics(scene_name, token)

    @lru_cache(maxsize=4096)
    def pose(self, token):
        return super().pose(token)


def _augment_one(source, rec, *, pcfg, patch_size_m: float):
    scene = str(rec["scene_name"])
    t0_token = str(rec.get("t0_token") or rec["history_tokens"][-1])
    sem = source.load_semantics(scene, t0_token)
    pose = np.asarray(source.pose(t0_token), dtype=np.float64)
    strong_cfg = StrongW2DetConfig(free_label=int(pcfg.free_label))
    current = extract_instances(sem, pose, grid=pcfg.grid, cfg=strong_cfg)

    n_cached = int(rec["features"].shape[0])
    if len(current) != n_cached:
        raise RuntimeError(
            f"{rec['sample_id']}: Strong source count {len(current)} != cached {n_cached}"
        )
    classes = [int(c["class_id"]) for c in current]
    cached_classes = [int(x) for x in rec["source_class_id"].tolist()]
    if classes != cached_classes:
        raise RuntimeError(f"{rec['sample_id']}: Strong source ordering/class mismatch")

    source_xy = rec["source_centroid_xy_t0_m"].float().numpy()
    masks = []
    coverages = []
    for i, comp in enumerate(current):
        mask, coverage = native_source_footprint_patch(
            comp["voxel_indices"],
            source_xy[i],
            grid=pcfg.grid,
            patch_size_m=float(patch_size_m),
        )
        masks.append(torch.from_numpy(mask))
        coverages.append(float(coverage))

    if masks:
        native = torch.stack(masks, dim=0).to(torch.uint8)
    else:
        native_res = float(pcfg.grid.voxel_size[0])
        raw = int(round(float(patch_size_m) / native_res))
        native = torch.zeros((0, raw, raw), dtype=torch.uint8)

    out = dict(rec)
    out[FIELD] = native
    return out, coverages


def main():
    p = argparse.ArgumentParser()
    add_config_args(p)
    p.add_argument("--input", required=True, help="existing V17 cache")
    p.add_argument("--dataroot", required=True)
    p.add_argument("--info-pkl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--prefetch-windows", type=int, default=0)
    p.add_argument("--max-windows", type=int, default=0)
    a = p.parse_args()

    cfg = load_runtime_config(a.config, a.override)
    pcfg = make_prepare_config(cfg)
    meta, records = load_cache(a.input)
    if int(a.max_windows) > 0:
        records = records[: min(len(records), int(a.max_windows))]
    if not records:
        raise RuntimeError("no records selected")

    patch_size_m = float(meta.get("patch_size_m", 16.0))
    native_res = float(pcfg.grid.voxel_size[0])
    if abs(float(pcfg.grid.voxel_size[1]) - native_res) > 1e-12:
        raise RuntimeError("native footprint augmentation requires square XY voxels")
    raw_f = patch_size_m / native_res
    raw = int(round(raw_f))
    if raw <= 0 or raw % 2 or abs(raw_f - raw) > 1e-9:
        raise RuntimeError("V17 patch size is incompatible with native grid resolution")

    workers = int(a.workers) if int(a.workers) > 0 else min(8, max(1, os.cpu_count() or 1))
    prefetch = int(a.prefetch_windows) if int(a.prefetch_windows) > 0 else 4 * workers
    source = CachedSource(a.dataroot, info_pkl=a.info_pkl, verbose=False)

    print(json.dumps({
        "protocol": PROTOCOL,
        "input_version": LOCAL_STWM_V17_CACHE_VERSION,
        "windows": len(records),
        "workers": workers,
        "prefetch_windows": prefetch,
        "patch_size_m": patch_size_m,
        "native_resolution_m": native_res,
        "native_hw": raw,
        "no_future_occupancy": True,
        "no_future_annotations": True,
    }), flush=True)

    def fn(rec):
        return _augment_one(source, rec, pcfg=pcfg, patch_size_m=patch_size_m)

    out_records = []
    total_sources = 0
    coverage_sum = 0.0
    coverage_min = 1.0
    started = time.perf_counter()
    for wi, (rec, coverages) in enumerate(
        bounded_ordered_parallel_map(
            fn,
            records,
            max_workers=workers,
            max_in_flight=prefetch,
            thread_name_prefix="v17-native-fp",
        ),
        start=1,
    ):
        out_records.append(rec)
        n = int(rec[FIELD].shape[0])
        total_sources += n
        if coverages:
            coverage_sum += float(sum(coverages))
            coverage_min = min(coverage_min, float(min(coverages)))
        if wi == 1 or wi % 100 == 0 or wi == len(records):
            elapsed = max(time.perf_counter() - started, 1e-9)
            print(
                f"native_fp_cache {wi}/{len(records)} sources={total_sources} "
                f"rate={wi/elapsed:.2f} win/s occ_cache={source.load_semantics.cache_info()} "
                f"pose_cache={source.pose.cache_info()}",
                flush=True,
            )

    new_meta = dict(meta)
    new_meta.update({
        "native_source_footprint_contract": NATIVE_SOURCE_FOOTPRINT_CONTRACT,
        "native_source_footprint_field": FIELD,
        "native_source_footprint_dtype": "uint8",
        "native_source_footprint_resolution_m": native_res,
        "native_source_footprint_hw": raw,
        "native_source_footprint_patch_size_m": patch_size_m,
        "native_source_footprint_reads_future_occupancy": False,
        "native_source_footprint_reads_future_annotations": False,
        "native_source_footprint_num_sources": total_sources,
        "native_source_footprint_mean_crop_coverage": coverage_sum / max(total_sources, 1),
        "native_source_footprint_min_crop_coverage": coverage_min if total_sources else float("nan"),
        "native_source_footprint_augmented_from": str(Path(a.input).resolve()),
        "native_source_footprint_augment_protocol": PROTOCOL,
    })
    payload = {
        "version": LOCAL_STWM_V17_CACHE_VERSION,
        "metadata": new_meta,
        "records": out_records,
    }
    op = Path(a.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, op)
    op.with_suffix(".summary.json").write_text(json.dumps(new_meta, indent=2), encoding="utf-8")
    print(json.dumps({
        "saved": str(op),
        "windows": len(out_records),
        "sources": total_sources,
        "mean_crop_coverage": new_meta["native_source_footprint_mean_crop_coverage"],
        "min_crop_coverage": new_meta["native_source_footprint_min_crop_coverage"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
