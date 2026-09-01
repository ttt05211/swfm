#!/usr/bin/env python3
"""Build P0-F8 anchor-relative edit targets from existing P0-F5/F7 WM caches.

The expensive WM latent cache is reused unchanged.  For validation caches with
an eval payload, exact Strong-W2Det / GT / true-moving support are read directly.
For training caches, the builder reconstructs the same exact raw window and
Strong-W2Det anchor from the frozen MSP provenance, then writes only a compact
sparse edit sidecar.
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
from real_motion.edit_repair import (
    DYNAMIC_IDS,
    OCC_SHAPE,
    P0_F8_EDIT_CACHE_VERSION,
    build_anchor_relative_edit_record,
)
from real_motion.msp_wm_cache import MSP_WM_CACHE_VERSION_V3, MSPWorldModelCacheDataset
from real_motion.occfm_io import file_sha256
from real_motion.prepared import load_nuscenes_window_raw
from real_motion.runtime_config import make_prepare_config
from real_motion.strong_w2det import StrongW2DetConfig, strong_w2det_sequence
from tools.real_motion import build_p0_f4_cache_direct as wm_base

PARTIAL_SUFFIX = ".partial"
PROTOCOL = "p0_f8_anchor_relative_edit_targets_v1"


def _save(path: Path, metadata: dict, records: list[dict]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save({
        "version": P0_F8_EDIT_CACHE_VERSION,
        "metadata": metadata,
        "records": records,
    }, tmp)
    os.replace(tmp, path)


def _validate_source(ds: MSPWorldModelCacheDataset) -> None:
    if ds.version != MSP_WM_CACHE_VERSION_V3:
        raise RuntimeError("P0-F8 edit targets require a P0-F5/v3 WM cache")
    meta = ds.metadata
    if int(meta.get("topk", -1)) != 2:
        raise RuntimeError("P0-F8 is frozen to Top-2")
    if meta.get("anchor_contract") != "strong_w2det_occ_only_v1":
        raise RuntimeError("source cache does not use Strong W2Det")
    if meta.get("repair_endpoint_contract") != "strong_anchor_outside_support_gt_dynamic_inside_support_v1":
        raise RuntimeError("source cache repair contract mismatch")


def _resolve_file(explicit, metadata, keys, label):
    value = explicit
    if not value:
        for key in keys:
            value = metadata.get(key)
            if value:
                break
    if not value:
        raise RuntimeError(f"{label} path is required")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _load_probe(path: Path):
    meta, records, cfg = wm_base._load_probe(path)
    record_map = {str(r["sample_id"]): r for r in records}
    if len(record_map) != len(records):
        raise RuntimeError("MSP cache contains duplicate sample IDs")
    return meta, records, record_map, cfg


def _metadata(source_root, source_ids, *, probe_path, anchor_source, moving_source, easy_keep_limit):
    return {
        "protocol": PROTOCOL,
        "source_wm_cache": str(source_root),
        "source_wm_cache_index_sha256": file_sha256(source_root / "index.json"),
        "source_sample_ids": list(source_ids),
        "msp_probe_cache": str(probe_path) if probe_path is not None else None,
        "msp_probe_cache_sha256": file_sha256(probe_path) if probe_path is not None else None,
        "anchor_source": str(anchor_source),
        "moving_source": str(moving_source),
        "occupancy_shape": list(OCC_SHAPE),
        "dynamic_class_ids": list(DYNAMIC_IDS),
        "actions": ["KEEP", "CLEAR"] + [f"WRITE:{cid}" for cid in DYNAMIC_IDS],
        "target_contract": "exact_strong_anchor_relative_keep_clear_write_inside_causal_msp_support",
        "keep_pool": "all_correct_dynamic_anchor_voxels_plus_compact_background_keep_near_edits",
        "true_motion_use": "priority_for_hard_KEEP_sampling_only; never an inference input",
        "easy_keep_limit_per_sample": int(easy_keep_limit),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-cache", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--msp-cache", default=None)
    p.add_argument("--dataroot", default=None)
    p.add_argument("--info-pkl", default=None)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--prefetch-samples", type=int, default=0)
    p.add_argument("--checkpoint-every", type=int, default=512)
    p.add_argument("--easy-keep-limit", type=int, default=4096)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    if a.checkpoint_every <= 0 or a.easy_keep_limit < 0:
        raise ValueError("checkpoint-every must be positive and easy-keep-limit non-negative")

    source_root = Path(a.source_cache).expanduser().resolve()
    ds = MSPWorldModelCacheDataset(source_root)
    _validate_source(ds)
    source_ids = [str(e["sample_id"]) for e in ds.entries]
    if len(source_ids) != len(set(source_ids)):
        raise RuntimeError("source WM cache contains duplicate sample IDs")

    first = ds[0]
    cached_eval = all(k in first for k in (
        "eval_future_gt_occ", "eval_strong_anchor_occ", "eval_gt_moving_support"
    ))

    probe_path = None
    probe_meta = None
    record_map = None
    pcfg = None
    raw_source = None
    if not cached_eval:
        probe_path = _resolve_file(
            a.msp_cache,
            ds.metadata,
            ("incremental_msp_probe_cache", "source_msp_cache"),
            "MSP probe cache",
        )
        expected = ds.metadata.get("source_msp_cache_sha256")
        if expected and file_sha256(probe_path) != expected:
            raise RuntimeError("MSP probe cache differs from source WM cache provenance")
        if not a.dataroot or not a.info_pkl:
            raise RuntimeError("train edit sidecar requires --dataroot and --info-pkl")
        probe_meta, _, record_map, cfg = _load_probe(probe_path)
        pcfg = make_prepare_config(cfg)
        raw_source = wm_base.CachedNuScenesWindowSource(
            a.dataroot, info_pkl=a.info_pkl, verbose=False
        )
        missing = sorted(set(source_ids) - set(record_map))
        if missing:
            raise RuntimeError(f"{len(missing)} WM samples missing from MSP probe cache")
        anchor_source = "recomputed_exact_strong_w2det_from_raw_history"
        moving_source = "nuScenes_gt_moving_support_for_horizon_training_labels_only"
    else:
        anchor_source = "cached_eval_strong_anchor_occ"
        moving_source = "cached_eval_gt_moving_support"

    metadata = _metadata(
        source_root,
        source_ids,
        probe_path=probe_path,
        anchor_source=anchor_source,
        moving_source=moving_source,
        easy_keep_limit=int(a.easy_keep_limit),
    )
    if probe_meta is not None:
        metadata["source_msp_mode"] = probe_meta.get("mode")
        metadata["source_msp_selection"] = probe_meta.get("selection")

    workers = int(a.workers) if int(a.workers) > 0 else min(16, max(1, os.cpu_count() or 1))
    prefetch = int(a.prefetch_samples) if int(a.prefetch_samples) > 0 else 4 * workers
    metadata["build_performance"] = {
        "builder": "p0_f8_edit_targets_parallel_v1",
        "workers": workers,
        "prefetch_samples": prefetch,
        "checkpoint_every": int(a.checkpoint_every),
    }

    output = Path(a.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + PARTIAL_SUFFIX)
    if output.exists():
        raise RuntimeError(f"{output} already exists")

    records = []
    done = set()
    if partial.exists():
        if not a.resume:
            raise RuntimeError(f"{partial} exists; use --resume or remove it")
        obj = torch.load(partial, map_location="cpu", weights_only=False)
        pm = obj.get("metadata") or {}
        for key in ("protocol", "source_wm_cache_index_sha256", "msp_probe_cache_sha256"):
            if pm.get(key) != metadata.get(key):
                raise RuntimeError(f"partial sidecar provenance mismatch for {key}")
        records = list(obj.get("records") or [])
        done = {str(r["sample_id"]) for r in records}
        if len(done) != len(records):
            raise RuntimeError("partial sidecar contains duplicate sample IDs")

    last_checkpoint = len(records)
    started = time.perf_counter()
    totals = {
        "edit": sum(int(r["edit_flat_indices"].numel()) for r in records),
        "moving_edit": sum(int(r["edit_moving"].sum().item()) for r in records),
        "keep_pool": sum(int(r["keep_flat_indices"].numel()) for r in records),
    }

    def source_items():
        for i in range(len(ds)):
            sample = ds[i]
            if str(sample["sample_id"]) not in done:
                yield sample

    w2cfg = StrongW2DetConfig(free_label=int(pcfg.free_label)) if pcfg is not None else None

    def build_one(sample):
        sid = str(sample["sample_id"])
        if cached_eval:
            gt = sample["eval_future_gt_occ"].cpu().numpy()
            anchor = sample["eval_strong_anchor_occ"].cpu().numpy()
            moving = sample["eval_gt_moving_support"].cpu().numpy().astype(bool)
        else:
            rec = record_map[sid]
            w = wm_base._window_from_record(rec, pcfg.history_frames, pcfg.future_frames)
            raw = load_nuscenes_window_raw(raw_source, w, pcfg, include_gt=True)
            anchor = strong_w2det_sequence(
                raw["history_occ"],
                raw["history_poses"],
                raw["future_poses"],
                frame_dt_s=float(pcfg.frame_dt_s),
                grid=pcfg.grid,
                cfg=w2cfg,
            )
            gt = raw["future_gt_occ"]
            moving = wm_base._gt_moving_support(raw_source, w, pcfg)
        return build_anchor_relative_edit_record(
            sample_id=sid,
            scene_name=str(sample["scene_name"]),
            gt_future_occ=gt,
            strong_anchor_occ=anchor,
            write_support_latent=sample["msp_write_support_latent"],
            moving_support_bev=moving,
            easy_keep_limit=int(a.easy_keep_limit),
        )

    for rec in bounded_ordered_parallel_map(
        build_one,
        source_items(),
        max_workers=workers,
        max_in_flight=prefetch,
        thread_name_prefix="f8-edit",
    ):
        records.append(rec)
        done.add(str(rec["sample_id"]))
        totals["edit"] += int(rec["edit_flat_indices"].numel())
        totals["moving_edit"] += int(rec["edit_moving"].sum().item())
        totals["keep_pool"] += int(rec["keep_flat_indices"].numel())
        if len(records) - last_checkpoint >= int(a.checkpoint_every):
            _save(partial, metadata, records)
            last_checkpoint = len(records)
        n = len(done)
        if n == 1 or n % 128 == 0 or n == len(ds):
            elapsed = max(time.perf_counter() - started, 1e-9)
            print(
                f"F8 edit targets {n}/{len(ds)} rate={n/elapsed:.2f} sample/s "
                f"edit={totals['edit']} moving_edit={totals['moving_edit']} "
                f"keep_pool={totals['keep_pool']}"
            )

    if done != set(source_ids):
        raise RuntimeError(f"P0-F8 sidecar sample mismatch source={len(source_ids)} output={len(done)}")
    _save(output, metadata, records)
    if partial.exists():
        partial.unlink()
    elapsed = max(time.perf_counter() - started, 1e-9)
    print(json.dumps({
        "output": str(output),
        "num_samples": len(records),
        "elapsed_seconds": elapsed,
        "samples_per_second": len(records) / elapsed,
        "total_edit_voxels": totals["edit"],
        "total_true_moving_edit_voxels": totals["moving_edit"],
        "total_keep_pool_voxels": totals["keep_pool"],
        "anchor_source": anchor_source,
        "moving_source": moving_source,
        "wm_latents_rebuilt": False,
        "vae_recomputed": False,
        "strong_w2det_recomputed": not cached_eval,
    }, indent=2))


if __name__ == "__main__":
    main()
