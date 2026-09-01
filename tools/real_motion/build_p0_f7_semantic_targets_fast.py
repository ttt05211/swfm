#!/usr/bin/env python3
"""High-throughput P0-F6 semantic sidecar builder for larger P0-F7 caches.

The semantic contract is unchanged.  Compared with the original builder this
version overlaps raw future-GT reads with GPU anchor decoding, uses larger decode
batches, pinned transfers, shared source caches, and checkpoints the growing
single-file sidecar only periodically instead of re-serializing it after every
small batch.
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
from real_motion.msp_wm_cache import MSPWorldModelCacheDataset
from real_motion.occfm_io import OccFMVAEAdapter, file_sha256, load_official_vae
from real_motion.semantic_repair import (
    DYNAMIC_IDS,
    DYNAMIC_TO_SLOT,
    OCC_SHAPE,
    P0_F6_SEMANTIC_CACHE_VERSION,
    build_sparse_semantic_record,
)
from tools.real_motion import build_p0_f5_cache_direct as wm_base
from tools.real_motion import build_p0_f6_semantic_targets as sem_base


def _metadata(source_root, source_ids, vae_path, vae_sha, probe_path, cached_gt):
    return {
        "protocol": "p0_f6_decoder_aware_sparse_dynamic_semantics_v1",
        "source_wm_cache": str(source_root),
        "source_wm_cache_index_sha256": file_sha256(source_root / "index.json"),
        "source_sample_ids": source_ids,
        "vae_checkpoint": str(vae_path),
        "vae_checkpoint_sha256": vae_sha,
        "msp_probe_cache": str(probe_path) if probe_path is not None else None,
        "gt_source": "p0_f5_eval_payload" if cached_gt else "raw_future_occ3d_semantics",
        "anchor_semantic_source": "frozen_vae_decode(anchor_future_latent)",
        "occupancy_shape": list(OCC_SHAPE),
        "dynamic_class_ids": list(DYNAMIC_IDS),
        "dynamic_to_slot": {str(k): int(v) for k, v in DYNAMIC_TO_SLOT.items()},
        "background_slot": 0,
        "supervision": "causal_write_support AND (gt_dynamic OR anchor_decode_dynamic)",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-cache", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--vae-ckpt", default=None)
    p.add_argument("--msp-cache", default=None)
    p.add_argument("--dataroot", default=None)
    p.add_argument("--batch-size", type=int, default=32,
                   help="frozen VAE anchor-decode batch")
    p.add_argument("--workers", type=int, default=0,
                   help="future-GT I/O threads; 0=auto min(16,cpu_count)")
    p.add_argument("--prefetch-samples", type=int, default=0,
                   help="bounded GT-read queue; 0=4x workers")
    p.add_argument("--checkpoint-every", type=int, default=512,
                   help="partial sidecar serialization interval")
    p.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    if min(a.batch_size, a.checkpoint_every) <= 0:
        raise ValueError("batch-size/checkpoint-every must be positive")

    workers = int(a.workers) if int(a.workers) > 0 else min(
        16, max(1, os.cpu_count() or 1)
    )
    prefetch = int(a.prefetch_samples) if int(a.prefetch_samples) > 0 else 4 * workers

    source_root = Path(a.source_cache).expanduser().resolve()
    ds = MSPWorldModelCacheDataset(source_root)
    sem_base._validate_source(ds)
    source_ids = [str(e["sample_id"]) for e in ds.entries]
    if len(source_ids) != len(set(source_ids)):
        raise RuntimeError("source WM cache contains duplicate sample IDs")

    vae_path = sem_base._resolve_file(
        a.vae_ckpt, ds.metadata, ("vae_checkpoint",), "VAE checkpoint"
    )
    vae_sha = file_sha256(vae_path)
    expected_vae_sha = ds.metadata.get("vae_checkpoint_sha256")
    if expected_vae_sha and expected_vae_sha != vae_sha:
        raise RuntimeError("VAE checkpoint differs from source WM cache")

    first = ds[0]
    cached_gt = "eval_future_gt_occ" in first
    record_map = None
    raw_source = None
    probe_path = None
    if not cached_gt:
        probe_path = sem_base._resolve_file(
            a.msp_cache,
            ds.metadata,
            ("incremental_msp_probe_cache", "source_msp_cache"),
            "MSP probe cache",
        )
        expected_probe_sha = ds.metadata.get("source_msp_cache_sha256")
        if expected_probe_sha and file_sha256(probe_path) != expected_probe_sha:
            raise RuntimeError("MSP probe cache differs from source WM cache provenance")
        if not a.dataroot:
            raise RuntimeError("train sidecar requires --dataroot")
        record_map = sem_base._load_probe(probe_path)
        missing = sorted(set(source_ids) - set(record_map))
        if missing:
            raise RuntimeError(f"{len(missing)} source samples missing from MSP probe cache")
        raw_source = wm_base.CachedNuScenesWindowSource(a.dataroot, verbose=False)

    output = Path(a.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + sem_base.PARTIAL_SUFFIX)
    metadata = _metadata(
        source_root, source_ids, vae_path, vae_sha, probe_path, cached_gt
    )
    metadata["build_performance"] = {
        "builder": "p0_f7_semantic_targets_high_throughput_v1",
        "batch_size": int(a.batch_size),
        "workers": workers,
        "prefetch_samples": prefetch,
        "checkpoint_every": int(a.checkpoint_every),
        "pin_memory": bool(a.pin_memory),
    }

    records = []
    done = set()
    if output.exists():
        raise RuntimeError(f"{output} already exists")
    if partial.exists():
        if not a.resume:
            raise RuntimeError(f"{partial} exists; use --resume or remove it")
        obj = torch.load(partial, map_location="cpu", weights_only=False)
        pm = obj.get("metadata") or {}
        for key in ("source_wm_cache_index_sha256", "vae_checkpoint_sha256", "protocol"):
            if pm.get(key) != metadata.get(key):
                raise RuntimeError(f"partial sidecar provenance mismatch for {key}")
        records = list(obj.get("records") or [])
        done = {str(r["sample_id"]) for r in records}
        if len(done) != len(records):
            raise RuntimeError("partial sidecar contains duplicate sample IDs")

    device = torch.device(a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu")
    vae_model, _ = load_official_vae(UP, vae_path, device)
    va = OccFMVAEAdapter(vae_model)

    # The iterator itself reads cache shards in the main thread, avoiding the
    # mutable single-shard cache race in MSPWorldModelCacheDataset. Worker
    # threads only perform raw future-GT I/O.
    def source_items():
        for i in range(len(ds)):
            sample = ds[i]
            if str(sample["sample_id"]) not in done:
                yield sample

    def load_gt(sample):
        gt = sem_base._load_gt_for_sample(
            sample, record_map=record_map, source=raw_source
        )
        return sample, gt

    pending = []
    total_gt_dyn = sum(int(r["gt_dynamic_flat_indices"].numel()) for r in records)
    total_anchor_dyn = sum(int(r["anchor_dynamic_flat_indices"].numel()) for r in records)
    last_checkpoint_count = len(records)
    started = time.perf_counter()

    def checkpoint_if_needed(force=False):
        nonlocal last_checkpoint_count
        if force or len(records) - last_checkpoint_count >= int(a.checkpoint_every):
            sem_base._save(partial, metadata, records)
            last_checkpoint_count = len(records)

    def flush(rows):
        nonlocal total_gt_dyn, total_anchor_dyn
        if not rows:
            return
        anchors = torch.stack([row[0]["anchor_future_latent"] for row in rows], dim=0)
        if bool(a.pin_memory and device.type == "cuda"):
            anchors = anchors.pin_memory().to(device=device, non_blocking=True)
        anchor_labels = va.decode_labels(anchors).cpu().numpy()
        for j, (sample, gt) in enumerate(rows):
            rec = build_sparse_semantic_record(
                sample_id=str(sample["sample_id"]),
                scene_name=str(sample["scene_name"]),
                gt_future_occ=gt,
                anchor_decoded_occ=anchor_labels[j],
                write_support_latent=sample["msp_write_support_latent"],
            )
            records.append(rec)
            done.add(str(rec["sample_id"]))
            total_gt_dyn += int(rec["gt_dynamic_flat_indices"].numel())
            total_anchor_dyn += int(rec["anchor_dynamic_flat_indices"].numel())
        checkpoint_if_needed()

    for sample, gt in bounded_ordered_parallel_map(
        load_gt,
        source_items(),
        max_workers=workers,
        max_in_flight=prefetch,
        thread_name_prefix="future-gt",
    ):
        pending.append((sample, gt))
        if len(pending) >= int(a.batch_size):
            flush(pending)
            pending = []
        prepared = len(done) + len(pending)
        if prepared % 128 == 0 or prepared == len(ds):
            elapsed = max(time.perf_counter() - started, 1e-9)
            print(
                f"F7 semantic targets {prepared}/{len(ds)} "
                f"rate={(prepared - (len(records)-len(done)))/elapsed:.2f} sample/s "
                f"gt_dynamic={total_gt_dyn} anchor_dynamic={total_anchor_dyn}"
            )
    flush(pending)

    if len(done) != len(source_ids) or done != set(source_ids):
        raise RuntimeError(
            f"semantic sidecar sample mismatch: source={len(source_ids)} output={len(done)}"
        )
    sem_base._save(output, metadata, records)
    if partial.exists():
        partial.unlink()
    elapsed = max(time.perf_counter() - started, 1e-9)
    print(json.dumps({
        "output": str(output),
        "num_samples": len(records),
        "elapsed_seconds": elapsed,
        "samples_per_second": max(len(records), 1) / elapsed,
        "gt_source": metadata["gt_source"],
        "total_gt_dynamic_voxels": total_gt_dyn,
        "total_anchor_dynamic_voxels": total_anchor_dyn,
        "strong_w2det_recomputed": False,
        "msp_recomputed": False,
        "vae_encoded": False,
        "vae_anchor_decoded_once": True,
    }, indent=2))


if __name__ == "__main__":
    main()
