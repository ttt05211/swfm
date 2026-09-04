#!/usr/bin/env python3
"""Upgrade a routed P0-F7/P0-F8 v3 cache to the P0-F9 native-future cache.

P0-F9 reuses the expensive, already frozen assets bit-for-bit:
- full 6-frame history latent;
- Strong-W2Det/KTA future latent (physics condition + deployment fallback);
- frozen MSP Top-2 route and horizon-wise write support;
- ego trajectory and compact evaluation payload when present.

Only the *absolute GT future* is newly encoded with the same frozen VAE mean.
The repair-target latent is intentionally discarded. Output uses the existing
MSP v2 tensor contract (``gt_future_latent``) with P0-F9-specific provenance.
"""
from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
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
from real_motion.msp_wm_cache import (
    MSP_WM_CACHE_VERSION_V2,
    MSP_WM_CACHE_VERSION_V3,
    MSPWorldModelCacheDataset,
    validate_msp_wm_sample,
)
from real_motion.occfm_io import OccFMVAEAdapter, file_sha256, load_official_vae
from real_motion.prepared import load_nuscenes_window_raw
from real_motion.runtime_config import make_prepare_config
from tools.real_motion import build_p0_f5_cache_direct as base

PROGRESS_NAME = ".p0_f9_index.partial.json"
P0_F9_CACHE_PROTOCOL = "p0_f9_absolute_future_native_sparse_cache_v1"


def _atomic_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    os.replace(tmp, path)


class _Writer:
    def __init__(self, root: Path, metadata: dict, *, shard_size: int, resume: bool):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.metadata = metadata
        self.shard_size = int(shard_size)
        self.entries = []
        self.seen = set()
        self.shard_id = 0
        self.current = []
        partial = self.root / PROGRESS_NAME
        final = self.root / "index.json"
        if final.exists():
            raise RuntimeError(f"{final} already exists; choose a fresh output directory")
        for p in self.root.glob("*.pt.tmp"):
            p.unlink()
        if partial.exists():
            if not resume:
                raise RuntimeError(f"{partial} exists; pass --resume or choose a fresh output")
            obj = json.loads(partial.read_text(encoding="utf-8"))
            if obj.get("version") != MSP_WM_CACHE_VERSION_V2:
                raise RuntimeError("P0-F9 partial cache version mismatch")
            if obj.get("metadata", {}).get("source_v3_cache_index_sha256") != metadata.get(
                "source_v3_cache_index_sha256"
            ):
                raise RuntimeError("P0-F9 resume source cache differs")
            self.entries = list(obj.get("entries", []))
            self.seen = {str(e["sample_id"]) for e in self.entries}
            ids = [int(Path(e["shard"]).stem.split("_")[-1]) for e in self.entries]
            self.shard_id = max(ids) + 1 if ids else 0

    def _commit(self, items):
        if not items:
            return
        name = f"shard_{self.shard_id:05d}.pt"
        dst = self.root / name
        tmp = self.root / (name + ".tmp")
        torch.save(items, tmp)
        os.replace(tmp, dst)
        for j, sample in enumerate(items):
            self.entries.append({
                "shard": name,
                "index": j,
                "sample_id": str(sample["sample_id"]),
                "scene_name": str(sample["scene_name"]),
            })
        self.shard_id += 1
        _atomic_json(self.root / PROGRESS_NAME, {
            "version": MSP_WM_CACHE_VERSION_V2,
            "metadata": self.metadata,
            "num_samples": len(self.entries),
            "entries": self.entries,
        })
        print(f"committed {name}: durable_total={len(self.entries)}")

    def add(self, sample):
        sid = str(sample["sample_id"])
        if sid in self.seen:
            return
        validate_msp_wm_sample(
            sample,
            topk=2,
            require_full_history=True,
            require_write_support=True,
            require_gt_target=True,
        )
        self.current.append(sample)
        self.seen.add(sid)
        if len(self.current) >= self.shard_size:
            items, self.current = self.current, []
            self._commit(items)

    def close(self):
        if self.current:
            items, self.current = self.current, []
            self._commit(items)
        index = {
            "version": MSP_WM_CACHE_VERSION_V2,
            "metadata": self.metadata,
            "num_samples": len(self.entries),
            "entries": self.entries,
        }
        _atomic_json(self.root / "index.json", index)
        partial = self.root / PROGRESS_NAME
        if partial.exists():
            partial.unlink()
        return index


def _resolve_device(spec: str) -> torch.device:
    requested = torch.device(spec)
    if requested.type != "cuda":
        return requested
    if not torch.cuda.is_available():
        return torch.device("cpu")
    idx = requested.index if requested.index is not None else int(torch.cuda.current_device())
    return torch.device("cuda", idx)


def _copy_eval_payload(sample: dict) -> dict:
    keys = (
        "eval_future_gt_occ",
        "eval_strong_anchor_occ",
        "eval_repair_target_occ",
        "eval_gt_moving_support",
    )
    return {k: sample[k] for k in keys if k in sample}


def _host_gt(rows, *, pin: bool):
    x = torch.from_numpy(np.stack([np.asarray(r[2], dtype=np.uint8) for r in rows], axis=0))
    return x.pin_memory() if pin else x


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-cache", required=True,
                   help="existing P0-F7/P0-F8 v3 cache with frozen route/history/anchor")
    p.add_argument("--msp-cache", required=True,
                   help="exact MSP probe cache used to recover raw nuScenes windows")
    p.add_argument("--dataroot", required=True)
    p.add_argument("--info-pkl", required=True)
    p.add_argument("--vae-ckpt", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--vae-batch-size", type=int, default=16)
    p.add_argument("--prepare-workers", type=int, default=0)
    p.add_argument("--prefetch-windows", type=int, default=0)
    p.add_argument("--shard-size", type=int, default=32)
    p.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    if min(a.vae_batch_size, a.shard_size) <= 0:
        raise ValueError("vae-batch-size and shard-size must be positive")

    device = _resolve_device(a.device)
    if device.type == "cuda":
        torch.cuda.set_device(int(device.index))
    workers = int(a.prepare_workers) if int(a.prepare_workers) > 0 else min(
        16, max(1, os.cpu_count() or 1)
    )
    prefetch = int(a.prefetch_windows) if int(a.prefetch_windows) > 0 else 4 * workers

    source_ds = MSPWorldModelCacheDataset(a.source_cache)
    if source_ds.version != MSP_WM_CACHE_VERSION_V3:
        raise RuntimeError("P0-F9 upgrader requires an existing P0-F5/P0-F7 v3 cache")
    sm = source_ds.metadata
    if sm.get("history_contract") != "full_native_occ_history_6f":
        raise RuntimeError("source cache must contain full native occupancy history")
    if sm.get("anchor_contract") != "strong_w2det_occ_only_v1":
        raise RuntimeError("source cache must use the frozen Strong-W2Det anchor")
    if int(sm.get("topk", -1)) != 2 or list(sm.get("window_hw", [])) != [20, 20]:
        raise RuntimeError("source cache must use frozen Top-2 20x20 routing")

    probe_meta, records, cfg = base._load_probe(a.msp_cache)
    pcfg = make_prepare_config(cfg)
    record_by_id = {str(r["sample_id"]): r for r in records}
    source_ids = [str(e["sample_id"]) for e in source_ds.entries]
    missing = [sid for sid in source_ids if sid not in record_by_id]
    if missing:
        raise RuntimeError(f"MSP cache is missing {len(missing)} source windows, e.g. {missing[:3]}")

    vae_sha = file_sha256(a.vae_ckpt)
    expected_vae = sm.get("vae_checkpoint_sha256")
    if expected_vae and vae_sha != expected_vae:
        raise RuntimeError("P0-F9 VAE differs from the source cache VAE")
    source_index_sha = file_sha256(Path(a.source_cache) / "index.json")
    msp_cache_sha = file_sha256(a.msp_cache)
    scenes = sorted({str(e["scene_name"]) for e in source_ds.entries})
    metadata = {
        "protocol": P0_F9_CACHE_PROTOCOL,
        "source_v3_cache": str(Path(a.source_cache).resolve()),
        "source_v3_cache_index_sha256": source_index_sha,
        "source_msp_cache": str(Path(a.msp_cache).resolve()),
        "source_msp_cache_sha256": msp_cache_sha,
        "source_msp_mode": sm.get("source_msp_mode", probe_meta.get("mode")),
        "source_msp_selection": sm.get("source_msp_selection", probe_meta.get("selection")),
        "msp_checkpoint_sha256": sm.get("msp_checkpoint_sha256"),
        "vae_checkpoint": str(Path(a.vae_ckpt).resolve()),
        "vae_checkpoint_sha256": vae_sha,
        "vae_mode": "mean",
        "latent_dtype": "float32",
        "topk": 2,
        "latent_hw": list(sm.get("latent_hw", [50, 50])),
        "window_hw": [20, 20],
        "context_hw": list(sm.get("context_hw", [40, 40])),
        "trajectory_length": int(sm.get("trajectory_length", 12)),
        "write_budget_ratio": float(sm.get("write_budget_ratio", 0.15)),
        "mean_write_latent_ratio": sm.get("mean_write_latent_ratio"),
        "mean_score_capture_ratio": sm.get("mean_score_capture_ratio"),
        "history_contract": "full_native_occ_history_6f",
        "anchor_contract": "strong_w2det_occ_only_v1",
        "target": "absolute_gt_future_vae_latent",
        "flow_source": "gaussian_noise_not_anchor",
        "num_unique_scenes": len(scenes),
        "scene_names": scenes,
        "include_eval_payload": bool(sm.get("include_eval_payload", False)),
        "upgrade_note": (
            "history/anchor/route/trajectory copied bit-for-bit from source v3; "
            "only absolute GT future latent is newly encoded"
        ),
    }
    writer = _Writer(Path(a.output), metadata, shard_size=a.shard_size, resume=a.resume)

    nusc_source = base.CachedNuScenesWindowSource(a.dataroot, info_pkl=a.info_pkl, verbose=False)
    vae_model, _ = load_official_vae(UP, a.vae_ckpt, device)
    vae = OccFMVAEAdapter(vae_model)

    work = [i for i, e in enumerate(source_ds.entries) if str(e["sample_id"]) not in writer.seen]
    t0 = time.perf_counter()

    def prepare_one(i):
        sample = source_ds[int(i)]
        sid = str(sample["sample_id"])
        record = record_by_id[sid]
        window = base._window_from_record(record, pcfg.history_frames, pcfg.future_frames)
        if "eval_future_gt_occ" in sample:
            gt = sample["eval_future_gt_occ"].cpu().numpy()
        else:
            raw = load_nuscenes_window_raw(nusc_source, window, pcfg, include_gt=True)
            gt = np.asarray(raw["future_gt_occ"], dtype=np.uint8)
        return int(i), sample, gt

    pending = []
    prepared = 0
    encoded = 0

    def flush():
        nonlocal pending, encoded
        if not pending:
            return
        pin = bool(a.pin_memory and device.type == "cuda")
        gt = _host_gt(pending, pin=pin)
        if device.type == "cuda":
            gt = gt.to(device=device, non_blocking=pin)
        zg = vae.encode(gt, mode="mean").float().cpu()
        for j, (_, src, _) in enumerate(pending):
            sample = {
                "sample_id": str(src["sample_id"]),
                "scene_name": str(src["scene_name"]),
                "full_history_latent": src["full_history_latent"].float().cpu(),
                "anchor_future_latent": src["anchor_future_latent"].float().cpu(),
                "gt_future_latent": zg[j],
                "window_origins": src["window_origins"].cpu(),
                "window_valid": src["window_valid"].cpu(),
                "msp_write_support_latent": src["msp_write_support_latent"].bool().cpu(),
                "trajectory": src["trajectory"].float().cpu(),
            }
            sample.update(_copy_eval_payload(src))
            writer.add(sample)
        encoded += len(pending)
        pending = []

    try:
        for row in bounded_ordered_parallel_map(
            prepare_one,
            work,
            max_workers=workers,
            max_in_flight=prefetch,
            thread_name_prefix="p0-f9-gt",
        ):
            pending.append(row)
            prepared += 1
            if len(pending) >= int(a.vae_batch_size):
                flush()
            done = len(writer.entries) + len(writer.current) + len(pending)
            if prepared == 1 or done % 50 == 0 or prepared == len(work):
                elapsed = max(time.perf_counter() - t0, 1e-9)
                print(
                    f"P0-F9 cache {done}/{len(source_ds)} prep_rate={prepared/elapsed:.2f} win/s "
                    f"encoded={encoded} occ_cache={nusc_source.load_occ3d.cache_info()}"
                )
        flush()
        index = writer.close()
    except BaseException:
        # Do not finalize an incomplete index; committed shards remain resumable.
        if writer.current:
            writer._commit(writer.current)
            writer.current = []
        raise

    elapsed = max(time.perf_counter() - t0, 1e-9)
    print(json.dumps({
        "output": str(Path(a.output).resolve()),
        "num_samples": index["num_samples"],
        "num_scenes": len(scenes),
        "elapsed_seconds": elapsed,
        "new_windows_per_second": prepared / elapsed if prepared else 0.0,
        "target": metadata["target"],
        "flow_source": metadata["flow_source"],
        "source_v3_cache_index_sha256": source_index_sha,
    }, indent=2))


if __name__ == "__main__":
    main()
