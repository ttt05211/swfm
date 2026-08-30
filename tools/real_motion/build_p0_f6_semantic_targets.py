#!/usr/bin/env python3
"""Build the compact decoder-aware semantic target sidecar for P0-F6.

This builder does NOT rerun Strong W2Det, MSP routing, or any VAE encoding.
It reuses the frozen P0-F5 cache and performs only:

1. one frozen VAE *decode* of the cached Strong-W2Det anchor latent, and
2. reading six future GT semantic occupancy frames when the source cache does
   not already contain the validation eval payload.

Only sparse dynamic voxel coordinates inside the causal MSP write support are
stored, so the sidecar stays small and training never rereads nuScenes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
UP = ROOT / "upstream_occfm"
sys.path[:0] = [str(ROOT), str(UP)]

import numpy as np
import torch

from real_motion.msp import MSP_CACHE_VERSION
from real_motion.msp_wm_cache import MSP_WM_CACHE_VERSION_V3, MSPWorldModelCacheDataset
from real_motion.nuscenes_adapter import NuScenesWindowSource
from real_motion.occfm_io import OccFMVAEAdapter, file_sha256, load_official_vae
from real_motion.semantic_repair import (
    DYNAMIC_IDS,
    DYNAMIC_TO_SLOT,
    OCC_SHAPE,
    P0_F6_SEMANTIC_CACHE_VERSION,
    build_sparse_semantic_record,
)

PARTIAL_SUFFIX = ".partial.pt"


def _resolve_file(cli_value: str | None, meta: dict, keys, label: str) -> Path:
    value = cli_value
    if not value:
        for key in keys:
            if meta.get(key):
                value = meta[key]
                break
    if not value:
        raise RuntimeError(f"{label} path is required")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _load_probe(path: Path) -> dict[str, dict]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if obj.get("version") != MSP_CACHE_VERSION:
        raise RuntimeError(f"unsupported MSP probe cache version {obj.get('version')}")
    records = obj.get("records") or []
    out = {}
    for r in records:
        sid = str(r.get("sample_id"))
        if not sid or sid in out:
            raise RuntimeError("MSP probe cache has missing/duplicate sample_id")
        required = ("scene_name", "future_tokens")
        missing = [k for k in required if k not in r]
        if missing:
            raise RuntimeError(f"MSP probe record {sid} missing {missing}")
        out[sid] = r
    if not out:
        raise RuntimeError("empty MSP probe cache")
    return out


def _validate_source(ds: MSPWorldModelCacheDataset) -> None:
    if ds.version != MSP_WM_CACHE_VERSION_V3:
        raise RuntimeError("P0-F6 semantic targets require a P0-F5/v3 WM cache")
    m = ds.metadata
    if m.get("target") != "occupancy_sparse_repair_endpoint_vae_latent":
        raise RuntimeError("source WM cache is not the P0-F5 repair-endpoint contract")
    if m.get("anchor_contract") != "strong_w2det_occ_only_v1":
        raise RuntimeError("source WM cache is not Strong W2Det")
    if int(m.get("topk", -1)) != 2:
        raise RuntimeError("P0-F6 is frozen to P0-F5 Top-2")
    if float(m.get("write_budget_ratio", -1)) != 0.15:
        raise RuntimeError("P0-F6 is frozen to the 15% MSP write budget")


def _load_gt_for_sample(
    sample: dict,
    *,
    record_map: dict[str, dict] | None,
    source: NuScenesWindowSource | None,
) -> np.ndarray:
    if "eval_future_gt_occ" in sample:
        gt = sample["eval_future_gt_occ"].cpu().numpy()
        if tuple(gt.shape) != OCC_SHAPE:
            raise RuntimeError(f"{sample['sample_id']}: invalid cached GT shape {gt.shape}")
        return gt
    if record_map is None or source is None:
        raise RuntimeError("train semantic targets require MSP probe cache + dataroot")
    sid = str(sample["sample_id"])
    if sid not in record_map:
        raise RuntimeError(f"{sid}: missing from MSP probe cache")
    r = record_map[sid]
    scene = str(r["scene_name"])
    future_tokens = tuple(str(x) for x in r["future_tokens"])
    if len(future_tokens) != OCC_SHAPE[0]:
        raise RuntimeError(f"{sid}: expected 6 future tokens, got {len(future_tokens)}")
    rows = [source.load_semantics(scene, tok) for tok in future_tokens]
    gt = np.stack(rows, axis=0)
    if tuple(gt.shape) != OCC_SHAPE:
        raise RuntimeError(f"{sid}: invalid raw GT shape {gt.shape}")
    return gt


def _save(path: Path, metadata: dict, records: list[dict]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(
        {
            "version": P0_F6_SEMANTIC_CACHE_VERSION,
            "metadata": metadata,
            "records": records,
        },
        tmp,
    )
    tmp.replace(path)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-cache", required=True, help="P0-F5 train/val cache")
    p.add_argument("--output", required=True, help="single .pt semantic sidecar")
    p.add_argument("--vae-ckpt", default=None)
    p.add_argument("--msp-cache", default=None, help="train only; defaults to source metadata")
    p.add_argument("--dataroot", default=None, help="train only; future GT labels")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    if a.batch_size <= 0:
        raise ValueError("batch-size must be positive")

    source_root = Path(a.source_cache).expanduser().resolve()
    ds = MSPWorldModelCacheDataset(source_root)
    _validate_source(ds)
    source_index = source_root / "index.json"
    source_index_sha = file_sha256(source_index)
    source_ids = [str(e["sample_id"]) for e in ds.entries]
    if len(source_ids) != len(set(source_ids)):
        raise RuntimeError("source P0-F5 cache contains duplicate sample IDs")

    vae_path = _resolve_file(a.vae_ckpt, ds.metadata, ("vae_checkpoint",), "VAE checkpoint")
    expected_vae_sha = ds.metadata.get("vae_checkpoint_sha256")
    actual_vae_sha = file_sha256(vae_path)
    if expected_vae_sha and expected_vae_sha != actual_vae_sha:
        raise RuntimeError("VAE checkpoint differs from the P0-F5 cache")

    first = ds[0]
    cached_gt = "eval_future_gt_occ" in first
    record_map = None
    raw_source = None
    probe_path = None
    if not cached_gt:
        probe_path = _resolve_file(
            a.msp_cache,
            ds.metadata,
            ("incremental_msp_probe_cache", "source_msp_cache"),
            "MSP probe cache",
        )
        expected_probe_sha = ds.metadata.get("source_msp_cache_sha256")
        if expected_probe_sha and file_sha256(probe_path) != expected_probe_sha:
            raise RuntimeError("MSP probe cache differs from P0-F5 routing provenance")
        if not a.dataroot:
            raise RuntimeError("train sidecar requires --dataroot")
        record_map = _load_probe(probe_path)
        missing = sorted(set(source_ids) - set(record_map))
        if missing:
            raise RuntimeError(f"{len(missing)} source samples missing from MSP probe cache")
        raw_source = NuScenesWindowSource(a.dataroot, verbose=False)

    output = Path(a.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + PARTIAL_SUFFIX)
    metadata = {
        "protocol": "p0_f6_decoder_aware_sparse_dynamic_semantics_v1",
        "source_wm_cache": str(source_root),
        "source_wm_cache_index_sha256": source_index_sha,
        "source_sample_ids": source_ids,
        "vae_checkpoint": str(vae_path),
        "vae_checkpoint_sha256": actual_vae_sha,
        "msp_probe_cache": str(probe_path) if probe_path is not None else None,
        "gt_source": "p0_f5_eval_payload" if cached_gt else "raw_future_occ3d_semantics",
        "anchor_semantic_source": "frozen_vae_decode(anchor_future_latent)",
        "occupancy_shape": list(OCC_SHAPE),
        "dynamic_class_ids": list(DYNAMIC_IDS),
        "dynamic_to_slot": {str(k): int(v) for k, v in DYNAMIC_TO_SLOT.items()},
        "background_slot": 0,
        "supervision": "causal_write_support AND (gt_dynamic OR anchor_decode_dynamic)",
    }

    records: list[dict] = []
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
        if not done.issubset(set(source_ids)):
            raise RuntimeError("partial sidecar contains unknown sample IDs")

    device = torch.device(a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu")
    vae, _ = load_official_vae(UP, vae_path, device)
    va = OccFMVAEAdapter(vae)

    pending: list[tuple[dict, np.ndarray]] = []
    total_gt_dyn = sum(int(r["gt_dynamic_flat_indices"].numel()) for r in records)
    total_anchor_dyn = sum(int(r["anchor_dynamic_flat_indices"].numel()) for r in records)

    def flush(rows):
        nonlocal total_gt_dyn, total_anchor_dyn
        if not rows:
            return
        anchors = torch.stack([row[0]["anchor_future_latent"] for row in rows], dim=0)
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
        _save(partial, metadata, records)

    for i in range(len(ds)):
        sample = ds[i]
        sid = str(sample["sample_id"])
        if sid in done:
            continue
        gt = _load_gt_for_sample(sample, record_map=record_map, source=raw_source)
        pending.append((sample, gt))
        if len(pending) >= int(a.batch_size):
            flush(pending)
            pending = []
        prepared = len(done) + len(pending)
        if prepared % 64 == 0 or prepared == len(ds):
            print(
                f"P0-F6 semantic targets {prepared}/{len(ds)} "
                f"gt_dynamic={total_gt_dyn} anchor_dynamic={total_anchor_dyn}"
            )
    flush(pending)

    if len(done) != len(source_ids) or done != set(source_ids):
        raise RuntimeError(
            f"semantic sidecar sample mismatch: source={len(source_ids)} output={len(done)}"
        )
    _save(output, metadata, records)
    if partial.exists():
        partial.unlink()
    print(json.dumps({
        "output": str(output),
        "num_samples": len(records),
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
