#!/usr/bin/env python3
"""High-throughput builder for the P0-F5/v3 cache contract used by P0-F7.

This is a performance-only implementation of the existing P0-F5 cache builder.
It preserves Strong W2Det, frozen MSP Top-2 routing, 15% causal write support,
occupancy-space repair endpoint, and FP32 VAE posterior-mean target semantics.

Speed comes from:
- larger batched frozen-MSP routing;
- bounded multithreaded raw I/O + Strong-W2Det + repair preparation;
- shared occupancy/pose LRUs across preparation threads;
- pinned host batches and non-blocking CPU->GPU transfers;
- larger VAE microbatches;
- asynchronous shard serialization overlapped with preparation/encoding;
- optional exact sample reuse from a compatible older v3 cache.

The output is the same ``MSP_WM_CACHE_VERSION_V3`` contract consumed by P0-F6
and P0-F7 trainers.  Sample order stays deterministic.
"""
from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import json
import math
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
from real_motion.metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS
from real_motion.msp import (
    collate_probe_records,
    latent_support_to_bev,
    rasterize_msp_scores,
    top_budget_support,
)
from real_motion.msp_window import (
    plan_topk_score_windows,
    score_capture_ratio,
    window_plan_support,
)
from real_motion.msp_wm_cache import (
    MSP_WM_CACHE_VERSION_V3,
    MSPWorldModelCacheDataset,
    validate_msp_wm_sample,
)
from real_motion.occfm_io import OccFMVAEAdapter, file_sha256, load_official_vae
from real_motion.prepared import load_nuscenes_window_raw
from real_motion.repair_target import build_dynamic_repair_endpoint
from real_motion.runtime_config import get_cfg, make_prepare_config
from real_motion.strong_w2det import StrongW2DetConfig, strong_w2det_sequence
from tools.real_motion import build_p0_f5_cache_direct as base

PROGRESS_NAME = base.PROGRESS_NAME
REPAIR_CONTRACT = "strong_anchor_outside_support_gt_dynamic_inside_support_v1"
LOSS_CONTRACT = "strong_anchor_to_occ_repair_endpoint_local_flow_full_history_context_no_auxiliary_losses"


def _atomic_json(path: Path, obj) -> None:
    base._atomic_json(path, obj)


class AsyncShardWriter:
    """Bounded single-thread serializer that overlaps disk writes with GPU work."""

    def __init__(
        self,
        root: Path,
        metadata: dict,
        *,
        shard_size: int,
        resume: bool,
        async_write: bool,
        max_pending_shards: int = 2,
    ):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.metadata = metadata
        self.shard_size = int(shard_size)
        self.async_write = bool(async_write)
        self.max_pending = max(1, int(max_pending_shards))
        self.entries, self.seen, self.shard_id = base._load_progress(
            self.root, resume=bool(resume)
        )
        self.current = []
        self.pending: list[tuple[Future | None, str, list[dict], list[dict]]] = []
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cache-writer") \
            if self.async_write else None

    @property
    def num_samples(self):
        return len(self.entries) + sum(len(x[3]) for x in self.pending) + len(self.current)

    def _save_file(self, items, name):
        dst = self.root / name
        tmp = self.root / (name + ".tmp")
        torch.save(items, tmp)
        os.replace(tmp, dst)
        return name

    def _finalize_oldest(self):
        if not self.pending:
            return
        future, name, _, entry_rows = self.pending.pop(0)
        if future is not None:
            future.result()
        self.entries.extend(entry_rows)
        _atomic_json(self.root / PROGRESS_NAME, {
            "version": MSP_WM_CACHE_VERSION_V3,
            "metadata": self.metadata,
            "num_samples": len(self.entries),
            "entries": self.entries,
        })
        print(f"committed {name}: durable_total={len(self.entries)}")

    def _submit(self, items):
        if not items:
            return
        name = f"shard_{self.shard_id:05d}.pt"
        rows = [{
            "shard": name,
            "index": j,
            "sample_id": str(s["sample_id"]),
            "scene_name": str(s["scene_name"]),
        } for j, s in enumerate(items)]
        if self.pool is None:
            self._save_file(items, name)
            future = None
        else:
            future = self.pool.submit(self._save_file, items, name)
        self.pending.append((future, name, items, rows))
        self.shard_id += 1
        while len(self.pending) >= self.max_pending:
            self._finalize_oldest()

    def add(self, sample):
        sid = str(sample["sample_id"])
        if sid in self.seen:
            return
        validate_msp_wm_sample(
            sample,
            topk=2,
            require_full_history=True,
            require_write_support=True,
            require_repair_target=True,
        )
        self.current.append(sample)
        self.seen.add(sid)
        if len(self.current) >= self.shard_size:
            items, self.current = self.current, []
            self._submit(items)

    def close(self):
        try:
            if self.current:
                items, self.current = self.current, []
                self._submit(items)
            while self.pending:
                self._finalize_oldest()
        finally:
            if self.pool is not None:
                self.pool.shutdown(wait=True)
        index = {
            "version": MSP_WM_CACHE_VERSION_V3,
            "metadata": self.metadata,
            "num_samples": len(self.entries),
            "entries": self.entries,
        }
        _atomic_json(self.root / "index.json", index)
        partial = self.root / PROGRESS_NAME
        if partial.exists():
            partial.unlink()
        return index


def _validate_reuse_cache(path, *, msp_sha, vae_sha, write_budget, include_eval):
    if not path:
        return None, {}
    ds = MSPWorldModelCacheDataset(path)
    if ds.version != MSP_WM_CACHE_VERSION_V3:
        raise RuntimeError("reuse cache must be P0-F5/v3")
    meta = ds.metadata
    checks = {
        "topk": 2,
        "msp_checkpoint_sha256": msp_sha,
        "vae_checkpoint_sha256": vae_sha,
        "write_budget_ratio": float(write_budget),
        "anchor_contract": "strong_w2det_occ_only_v1",
        "history_contract": "full_native_occ_history_6f",
        "repair_endpoint_contract": REPAIR_CONTRACT,
        "target": "occupancy_sparse_repair_endpoint_vae_latent",
    }
    for key, value in checks.items():
        if meta.get(key) != value:
            raise RuntimeError(
                f"reuse WM cache incompatible for {key}: {meta.get(key)!r} != {value!r}"
            )
    if include_eval and not bool(meta.get("include_eval_payload", False)):
        raise RuntimeError("reuse cache lacks eval payload required by this build")
    lookup = {str(e["sample_id"]): i for i, e in enumerate(ds.entries)}
    if len(lookup) != len(ds.entries):
        raise RuntimeError("reuse WM cache has duplicate sample IDs")
    return ds, lookup


def _routes(records, *, msp, pcfg, device, latent_hw, window_hw, batch_size, write_budget):
    route_map = {}
    captures = []
    valid_counts = []
    write_ratios = []
    for start in range(0, len(records), int(batch_size)):
        rb = records[start : start + int(batch_size)]
        batch = collate_probe_records(rb)
        bdev = {
            k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()
        }
        pred = msp(bdev["features"], bdev["candidate_mask"])
        scores = rasterize_msp_scores(pred, bdev, latent_hw=latent_hw, grid=pcfg.grid)
        plan = plan_topk_score_windows(scores, window_hw=window_hw, max_windows=2)
        capture = score_capture_ratio(scores, plan).detach().cpu()
        spatial_window = window_plan_support(plan).bool()
        write = top_budget_support(scores, float(write_budget))
        write = write & spatial_window[:, None, :, :]
        origins = plan.origins.detach().cpu()
        valid = plan.valid.detach().cpu()
        write_cpu = write.detach().cpu()
        for j, r in enumerate(rb):
            sid = str(r["sample_id"])
            route_map[sid] = (
                origins[j].clone(), valid[j].clone(), write_cpu[j].clone()
            )
            captures.append(float(capture[j]))
            valid_counts.append(int(valid[j].sum()))
            write_ratios.append(float(write_cpu[j].float().mean().item()))
    return route_map, captures, valid_counts, write_ratios


def _route_matches(sample, route):
    origins, valid, write = route
    return (
        torch.equal(sample["window_origins"].cpu(), origins)
        and torch.equal(sample["window_valid"].cpu(), valid)
        and torch.equal(sample["msp_write_support_latent"].cpu(), write)
    )


def _host_tensor(arrays, *, pin: bool):
    # Keep semantic labels compact on host / PCIe. OccFMVAEAdapter casts uint8
    # to long on the GPU when the tensor is already resident there.
    x = torch.from_numpy(np.stack([
        np.asarray(a, dtype=np.uint8) for a in arrays
    ], axis=0))
    return x.pin_memory() if pin else x


def _encode_batch(va, rows, *, device, pin_memory: bool):
    pin = bool(pin_memory and device.type == "cuda")
    hist = _host_tensor([x[1] for x in rows], pin=pin)
    anchor = _host_tensor([x[2] for x in rows], pin=pin)
    repair = _host_tensor([x[3] for x in rows], pin=pin)

    # Transfer uint8 labels first; casting to long happens on GPU. This avoids
    # materializing/transferring an 8x larger int64 host buffer.
    if device.type == "cuda":
        hist = hist.to(device=device, non_blocking=pin)
        anchor = anchor.to(device=device, non_blocking=pin)
        repair = repair.to(device=device, non_blocking=pin)

    zh = va.encode(hist, mode="mean").float().cpu()
    za = va.encode(anchor, mode="mean").float().cpu()
    zr = va.encode(repair, mode="mean").float().cpu()
    out = []
    for j, (meta, _, _, _, payload) in enumerate(rows):
        sample = {
            "sample_id": meta["sample_id"],
            "scene_name": meta["scene_name"],
            "full_history_latent": zh[j],
            "anchor_future_latent": za[j],
            "repair_target_latent": zr[j],
            "window_origins": meta["window_origins"],
            "window_valid": meta["window_valid"],
            "msp_write_support_latent": meta["msp_write_support_latent"],
            "trajectory": meta["trajectory"],
        }
        if payload is not None:
            sample.update(payload)
        out.append(sample)
    return out


def _resolve_device(spec: str) -> torch.device:
    """Resolve bare ``cuda`` to an indexed device accepted by set_device()."""
    requested = torch.device(spec)
    if requested.type != "cuda":
        return requested
    if not torch.cuda.is_available():
        return torch.device("cpu")
    index = requested.index
    if index is None:
        index = int(torch.cuda.current_device())
    return torch.device("cuda", index)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--msp-cache", required=True)
    p.add_argument("--msp-checkpoint", required=True)
    p.add_argument("--dataroot", required=True)
    p.add_argument("--info-pkl", required=True)
    p.add_argument("--vae-ckpt", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--topk", type=int, default=2)
    p.add_argument("--write-budget-ratio", type=float, default=0.15)
    p.add_argument("--route-batch-size", type=int, default=128)
    p.add_argument("--vae-batch-size", type=int, default=16)
    p.add_argument("--shard-size", type=int, default=32)
    p.add_argument("--prepare-workers", type=int, default=0,
                   help="CPU I/O/W2Det threads; 0=auto min(16,cpu_count)")
    p.add_argument("--prefetch-windows", type=int, default=0,
                   help="bounded prepared jobs; 0=4x workers")
    p.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--async-write", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-pending-shards", type=int, default=2)
    p.add_argument("--reuse-cache", default=None,
                   help="optional compatible v3 cache; route-identical samples are copied")
    p.add_argument("--include-eval-payload", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    if a.topk != 2:
        raise ValueError("P0-F7 cache contract is frozen to Top-2")
    if not 0.0 < float(a.write_budget_ratio) <= 1.0:
        raise ValueError("write-budget-ratio must be in (0,1]")
    if min(a.route_batch_size, a.vae_batch_size, a.shard_size, a.max_pending_shards) <= 0:
        raise ValueError("batch/shard settings must be positive")

    workers = int(a.prepare_workers) if int(a.prepare_workers) > 0 else min(
        16, max(1, os.cpu_count() or 1)
    )
    prefetch = int(a.prefetch_windows) if int(a.prefetch_windows) > 0 else 4 * workers
    device = _resolve_device(a.device)
    if device.type == "cuda":
        torch.cuda.set_device(int(device.index))

    probe_meta, records, cfg = base._load_probe(a.msp_cache)
    msp_ck, msp = base._load_msp(a.msp_checkpoint, device)
    pcfg = make_prepare_config(cfg)
    latent_hw = tuple(int(v) for v in get_cfg(cfg, "UPSTREAM.LATENT_HW", [50, 50]))
    window_hw = tuple(int(v) for v in get_cfg(cfg, "MODEL.WINDOW_HW", [20, 20]))
    if latent_hw != (50, 50) or window_hw != (20, 20):
        raise RuntimeError("P0-F7 expects 50x50 latent and 20x20 prediction windows")
    if int(msp_ck["future_frames"]) != pcfg.future_frames:
        raise RuntimeError("MSP future-frame contract mismatch")
    if len({str(r["sample_id"]) for r in records}) != len(records):
        raise RuntimeError("MSP cache contains duplicate sample IDs")

    print(f"routing {len(records)} windows with batch={a.route_batch_size}")
    t_route = time.perf_counter()
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
    route_seconds = time.perf_counter() - t_route
    print(f"routing done in {route_seconds:.1f}s ({len(records)/max(route_seconds,1e-9):.1f} win/s)")

    source = base.CachedNuScenesWindowSource(a.dataroot, info_pkl=a.info_pkl, verbose=False)
    windows = []
    for r in records:
        w = base._window_from_record(r, pcfg.history_frames, pcfg.future_frames)
        ts = int(source.nusc.get("sample", w.t0_token)["timestamp"])
        windows.append((w.scene_name, ts, str(r["sample_id"]), w))
    windows.sort(key=lambda x: (x[0], x[1]))

    vae_sha = file_sha256(a.vae_ckpt)
    msp_sha = file_sha256(a.msp_checkpoint)
    reuse_ds, reuse_lookup = _validate_reuse_cache(
        a.reuse_cache,
        msp_sha=msp_sha,
        vae_sha=vae_sha,
        write_budget=float(a.write_budget_ratio),
        include_eval=bool(a.include_eval_payload),
    )

    out_root = Path(a.output)
    old_entries, already_done, _ = base._load_progress(
        out_root, resume=a.resume
    ) if out_root.exists() else ([], set(), 0)
    del old_entries

    w2cfg = StrongW2DetConfig(free_label=int(pcfg.free_label))
    bev_hw = tuple(int(v) for v in pcfg.grid.shape_hwd[:2])
    vae, _ = load_official_vae(UP, a.vae_ckpt, device)
    va = OccFMVAEAdapter(vae)

    scenes = sorted({str(r["scene_name"]) for r in records})
    metadata = {
        "protocol": "p0_f5_strong_w2det_occ_repair_endpoint_top2_v1",
        "direct_from_exact_msp_windows": True,
        "source_msp_cache": str(Path(a.msp_cache).resolve()),
        "source_msp_cache_sha256": file_sha256(a.msp_cache),
        "source_msp_mode": probe_meta.get("mode"),
        "source_msp_selection": probe_meta.get("selection"),
        "msp_checkpoint": str(Path(a.msp_checkpoint).resolve()),
        "msp_checkpoint_sha256": msp_sha,
        "vae_checkpoint": str(Path(a.vae_ckpt).resolve()),
        "vae_checkpoint_sha256": vae_sha,
        "vae_mode": "mean",
        "latent_dtype": "float32",
        "topk": 2,
        "latent_hw": list(latent_hw),
        "window_hw": [20, 20],
        "context_hw": [40, 40],
        "trajectory_length": int(pcfg.trajectory_length),
        "num_unique_scenes": len(scenes),
        "scene_names": scenes,
        "mean_score_capture_ratio": float(np.mean(captures)) if captures else 0.0,
        "mean_valid_windows": float(np.mean(valid_counts)) if valid_counts else 0.0,
        "slot_compute_ratio": float(np.mean(valid_counts) * 400.0 / 2500.0) if valid_counts else 0.0,
        "write_budget_ratio": float(a.write_budget_ratio),
        "mean_write_latent_ratio": float(np.mean(write_ratios)) if write_ratios else 0.0,
        "history_contract": "full_native_occ_history_6f",
        "anchor_contract": "strong_w2det_occ_only_v1",
        "repair_endpoint_contract": REPAIR_CONTRACT,
        "w2det_min_component_voxels": int(w2cfg.min_component_voxels),
        "w2det_max_match_speed_mps": float(w2cfg.max_match_speed_mps),
        "w2det_connectivity": int(w2cfg.connectivity),
        "loss_contract": LOSS_CONTRACT,
        "source": "strong_w2det_anchor_latent",
        "target": "occupancy_sparse_repair_endpoint_vae_latent",
        "latent_loss_mask": "none",
        "include_eval_payload": bool(a.include_eval_payload),
        "build_performance": {
            "builder": "p0_f7_high_throughput_v1",
            "route_batch_size": int(a.route_batch_size),
            "vae_batch_size": int(a.vae_batch_size),
            "prepare_workers": workers,
            "prefetch_windows": prefetch,
            "pin_memory": bool(a.pin_memory),
            "async_write": bool(a.async_write),
            "max_pending_shards": int(a.max_pending_shards),
            "reuse_cache": str(Path(a.reuse_cache).resolve()) if a.reuse_cache else None,
            "reuse_note": "optional speed path; main scientific build may omit reuse to keep one VAE batching path",
        },
    }

    writer = AsyncShardWriter(
        out_root,
        metadata,
        shard_size=int(a.shard_size),
        resume=bool(a.resume),
        async_write=bool(a.async_write),
        max_pending_shards=int(a.max_pending_shards),
    )

    work = [x for x in windows if x[2] not in already_done]
    reused_count = 0
    prepared_count = 0
    encoded_count = 0
    t_build = time.perf_counter()

    def prepare_one(item):
        _, _, sid, w = item
        if reuse_ds is not None and sid in reuse_lookup:
            sample = reuse_ds[reuse_lookup[sid]]
            if not _route_matches(sample, route_map[sid]):
                raise RuntimeError(f"{sid}: reuse cache route differs from current frozen MSP")
            return ("reuse", sample)

        raw = load_nuscenes_window_raw(source, w, pcfg, include_gt=True)
        origins, valid, write = route_map[sid]
        strong_anchor = strong_w2det_sequence(
            raw["history_occ"],
            raw["history_poses"],
            raw["future_poses"],
            frame_dt_s=float(pcfg.frame_dt_s),
            grid=pcfg.grid,
            cfg=w2cfg,
        )
        write_bev = latent_support_to_bev(write.bool(), bev_hw).numpy().astype(bool)
        repair_endpoint = build_dynamic_repair_endpoint(
            strong_anchor,
            raw["future_gt_occ"],
            write_bev,
            dynamic_class_ids=DYNAMIC_CLASS_IDS,
            free_label=int(pcfg.free_label),
        )
        meta = {
            "sample_id": sid,
            "scene_name": str(w.scene_name),
            "window_origins": origins,
            "window_valid": valid,
            "msp_write_support_latent": write.bool(),
            "trajectory": torch.as_tensor(raw["trajectory"], dtype=torch.float32),
        }
        payload = None
        if a.include_eval_payload:
            payload = {
                "eval_future_gt_occ": torch.from_numpy(np.asarray(raw["future_gt_occ"], dtype=np.uint8)),
                "eval_strong_anchor_occ": torch.from_numpy(np.asarray(strong_anchor, dtype=np.uint8)),
                "eval_repair_target_occ": torch.from_numpy(np.asarray(repair_endpoint, dtype=np.uint8)),
                "eval_gt_moving_support": torch.from_numpy(base._gt_moving_support(source, w, pcfg)),
            }
        return (
            "encode",
            meta,
            np.asarray(raw["history_occ"], dtype=np.uint8),
            np.asarray(strong_anchor, dtype=np.uint8),
            np.asarray(repair_endpoint, dtype=np.uint8),
            payload,
        )

    pending_encode = []

    def flush_encode():
        nonlocal pending_encode, encoded_count
        if not pending_encode:
            return
        samples = _encode_batch(
            va,
            pending_encode,
            device=device,
            pin_memory=bool(a.pin_memory),
        )
        for sample in samples:
            writer.add(sample)
        encoded_count += len(samples)
        pending_encode = []

    try:
        for result in bounded_ordered_parallel_map(
            prepare_one,
            work,
            max_workers=workers,
            max_in_flight=prefetch,
            thread_name_prefix="w2det-prep",
        ):
            prepared_count += 1
            if result[0] == "reuse":
                flush_encode()  # preserve exact deterministic sample order
                writer.add(result[1])
                reused_count += 1
            else:
                pending_encode.append(result[1:])
                if len(pending_encode) >= int(a.vae_batch_size):
                    flush_encode()

            done = prepared_count + len(already_done)
            if prepared_count == 1 or done % 50 == 0 or done == len(windows):
                elapsed = max(time.perf_counter() - t_build, 1e-9)
                print(
                    f"F7 cache {done}/{len(windows)} prep_rate={prepared_count/elapsed:.2f} win/s "
                    f"encoded={encoded_count} reused={reused_count} "
                    f"occ_cache={source.load_occ3d.cache_info()} pose_cache={source.pose.cache_info()}"
                )
        flush_encode()
        index = writer.close()
    except BaseException:
        # Close the writer so already submitted shards become durable and the
        # partial index is useful with --resume, then re-raise the original error.
        try:
            if pending_encode:
                pending_encode = []
            while writer.pending:
                writer._finalize_oldest()
            if writer.pool is not None:
                writer.pool.shutdown(wait=True)
        finally:
            raise

    elapsed = max(time.perf_counter() - t_build, 1e-9)
    print(json.dumps({
        "output": str(out_root),
        "num_samples": index["num_samples"],
        "num_scenes": len(scenes),
        "new_or_reused_windows_per_second": prepared_count / elapsed,
        "encoded_samples": encoded_count,
        "reused_samples": reused_count,
        "route_seconds": route_seconds,
        "prepare_workers": workers,
        "prefetch_windows": prefetch,
        "vae_batch_size": int(a.vae_batch_size),
        "mean_valid_windows": metadata["mean_valid_windows"],
        "slot_compute_ratio": metadata["slot_compute_ratio"],
        "mean_write_latent_ratio": metadata["mean_write_latent_ratio"],
        "score_capture": metadata["mean_score_capture_ratio"],
        "target": metadata["target"],
    }, indent=2))


if __name__ == "__main__":
    main()
