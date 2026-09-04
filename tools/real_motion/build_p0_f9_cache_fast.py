#!/usr/bin/env python3
"""Build the P0-F9 native-future cache from a frozen routed v3 cache.

The first P0-F9 draft reused deterministic VAE posterior means from P0-F7/F8.
That is not the latent distribution used to train the released OccFM-Fut model:
the official VAE cache stores posterior ``sampled_features``. This audited
builder therefore re-encodes history, Strong-W2Det physics anchor, and absolute
GT future with deterministic per-sample posterior draws while reusing the
expensive scientific contracts bit-for-bit:

- frozen Real-Motion/MSP Top-2 route and horizon-wise write support;
- exact sample/window identity;
- cached trajectory;
- Strong-W2Det occupancy contract;
- compact evaluation payload when present.

No MSP model is rerun. For train windows the Strong-W2Det occupancy anchor is
recomputed from the exact raw history because the old cache only persisted its
mean latent. For validation the exact cached anchor occupancy is reused.
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
from real_motion.msp_wm_cache import (
    MSP_WM_CACHE_VERSION_V2,
    MSP_WM_CACHE_VERSION_V3,
    MSPWorldModelCacheDataset,
    validate_msp_wm_sample,
)
from real_motion.native_forecast import deterministic_sample_seed
from real_motion.occfm_io import OccFMVAEAdapter, file_sha256, load_official_vae
from real_motion.prepared import load_nuscenes_window_raw
from real_motion.runtime_config import make_prepare_config
from real_motion.strong_w2det import StrongW2DetConfig, strong_w2det_sequence
from tools.real_motion import build_p0_f5_cache_direct as base

PROGRESS_NAME = ".p0_f9_index.partial.json"
P0_F9_CACHE_PROTOCOL = "p0_f9_absolute_future_native_sparse_cache_v2"


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
            old_meta = obj.get("metadata", {})
            for key in (
                "protocol",
                "source_v3_cache_index_sha256",
                "source_msp_cache_sha256",
                "vae_checkpoint_sha256",
                "vae_mode",
                "vae_sample_seed_base",
            ):
                if old_meta.get(key) != metadata.get(key):
                    raise RuntimeError(f"P0-F9 resume metadata differs for {key}")
            self.entries = list(obj.get("entries", []))
            self.seen = {str(e["sample_id"]) for e in self.entries}
            if len(self.seen) != len(self.entries):
                raise RuntimeError("P0-F9 partial cache contains duplicate sample IDs")
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


def _host_occ(rows, pos: int, *, pin: bool):
    x = torch.from_numpy(np.stack([
        np.asarray(r[pos], dtype=np.uint8) for r in rows
    ], axis=0))
    return x.pin_memory() if pin else x


def _validate_w2det_contract(meta: dict, cfg: StrongW2DetConfig) -> None:
    checks = {
        "w2det_min_component_voxels": int(cfg.min_component_voxels),
        "w2det_max_match_speed_mps": float(cfg.max_match_speed_mps),
        "w2det_connectivity": int(cfg.connectivity),
    }
    for key, value in checks.items():
        old = meta.get(key)
        if old is not None and old != value:
            raise RuntimeError(f"source Strong-W2Det contract differs for {key}: {old} != {value}")


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-cache", required=True,
                   help="existing P0-F7/P0-F8 v3 cache with frozen route provenance")
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
    p.add_argument("--latent-seed", type=int, default=20260904)
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
        raise RuntimeError("P0-F9 builder requires an existing P0-F5/P0-F7 v3 cache")
    sm = source_ds.metadata
    if sm.get("history_contract") != "full_native_occ_history_6f":
        raise RuntimeError("source cache must contain full native occupancy history")
    if sm.get("anchor_contract") != "strong_w2det_occ_only_v1":
        raise RuntimeError("source cache must use the frozen Strong-W2Det anchor")
    if int(sm.get("topk", -1)) != 2 or list(sm.get("window_hw", [])) != [20, 20]:
        raise RuntimeError("source cache must use frozen Top-2 20x20 routing")
    source_ids = [str(e["sample_id"]) for e in source_ds.entries]
    if len(source_ids) != len(set(source_ids)):
        raise RuntimeError("source v3 cache contains duplicate sample IDs")

    msp_cache_sha = file_sha256(a.msp_cache)
    expected_msp_cache_sha = sm.get("source_msp_cache_sha256")
    if expected_msp_cache_sha and msp_cache_sha != expected_msp_cache_sha:
        raise RuntimeError("supplied MSP probe cache differs from source v3 provenance")

    probe_meta, records, cfg = base._load_probe(a.msp_cache)
    if len({str(r["sample_id"]) for r in records}) != len(records):
        raise RuntimeError("MSP probe cache contains duplicate sample IDs")
    pcfg = make_prepare_config(cfg)
    record_by_id = {str(r["sample_id"]): r for r in records}
    missing = [sid for sid in source_ids if sid not in record_by_id]
    if missing:
        raise RuntimeError(f"MSP cache is missing {len(missing)} source windows, e.g. {missing[:3]}")
    for key, probe_key in (("source_msp_mode", "mode"), ("source_msp_selection", "selection")):
        old = sm.get(key)
        cur = probe_meta.get(probe_key)
        if old is not None and cur is not None and old != cur:
            raise RuntimeError(f"MSP probe provenance differs for {key}: {old!r} != {cur!r}")

    vae_sha = file_sha256(a.vae_ckpt)
    expected_vae = sm.get("vae_checkpoint_sha256")
    if expected_vae and vae_sha != expected_vae:
        raise RuntimeError("P0-F9 VAE differs from the source cache VAE")
    source_index_sha = file_sha256(Path(a.source_cache) / "index.json")
    scenes = sorted({str(e["scene_name"]) for e in source_ds.entries})
    w2cfg = StrongW2DetConfig(free_label=int(pcfg.free_label))
    _validate_w2det_contract(sm, w2cfg)

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
        "vae_mode": "sample",
        "vae_sample_seed_base": int(a.latent_seed),
        "vae_sample_seed_contract": "sha256(base,stream,sample_id)_per_video_sample_v1",
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
        "native_backbone_hist_last": 4,
        "anchor_contract": "strong_w2det_occ_only_v1",
        "w2det_min_component_voxels": int(w2cfg.min_component_voxels),
        "w2det_max_match_speed_mps": float(w2cfg.max_match_speed_mps),
        "w2det_connectivity": int(w2cfg.connectivity),
        "target": "absolute_gt_future_vae_latent",
        "flow_source": "gaussian_noise_not_anchor",
        "num_unique_scenes": len(scenes),
        "scene_names": scenes,
        "include_eval_payload": bool(sm.get("include_eval_payload", False)),
        "reencoded_tensor_keys": [
            "full_history_latent",
            "anchor_future_latent",
            "gt_future_latent",
        ],
        "reuse_note": (
            "sample identity, trajectory, Top-2 route, write support and eval payload copied from source v3; "
            "history/anchor/GT latents re-encoded as deterministic posterior samples to match OccFM-Fut training distribution"
        ),
    }
    writer = _Writer(Path(a.output), metadata, shard_size=a.shard_size, resume=a.resume)

    nusc_source = base.CachedNuScenesWindowSource(a.dataroot, info_pkl=a.info_pkl, verbose=False)
    vae_model, _ = load_official_vae(UP, a.vae_ckpt, device)
    vae = OccFMVAEAdapter(vae_model)

    work = [i for i, e in enumerate(source_ds.entries) if str(e["sample_id"]) not in writer.seen]
    t0 = time.perf_counter()

    def prepare_one(i):
        src = source_ds[int(i)]
        sid = str(src["sample_id"])
        record = record_by_id[sid]
        window = base._window_from_record(record, pcfg.history_frames, pcfg.future_frames)
        if str(window.scene_name) != str(src["scene_name"]):
            raise RuntimeError(f"{sid}: source/MSP scene mismatch")
        raw = load_nuscenes_window_raw(nusc_source, window, pcfg, include_gt=True)
        raw_traj = torch.as_tensor(raw["trajectory"], dtype=torch.float32)
        if not torch.allclose(raw_traj, src["trajectory"].float().cpu(), atol=1e-6, rtol=1e-6):
            raise RuntimeError(f"{sid}: raw trajectory differs from frozen source cache")

        history = np.asarray(raw["history_occ"], dtype=np.uint8)
        raw_gt = np.asarray(raw["future_gt_occ"], dtype=np.uint8)
        if "eval_future_gt_occ" in src:
            gt = src["eval_future_gt_occ"].cpu().numpy().astype(np.uint8, copy=False)
            if not np.array_equal(gt, raw_gt):
                raise RuntimeError(f"{sid}: cached validation GT differs from raw exact window")
        else:
            gt = raw_gt

        if "eval_strong_anchor_occ" in src:
            anchor = src["eval_strong_anchor_occ"].cpu().numpy().astype(np.uint8, copy=False)
        else:
            anchor = strong_w2det_sequence(
                history,
                raw["history_poses"],
                raw["future_poses"],
                frame_dt_s=float(pcfg.frame_dt_s),
                grid=pcfg.grid,
                cfg=w2cfg,
            ).astype(np.uint8, copy=False)
        return src, history, anchor, gt

    pending = []
    prepared = 0
    encoded = 0

    def flush():
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
        for j, (src, _, _, _) in enumerate(pending):
            sample = {
                "sample_id": str(src["sample_id"]),
                "scene_name": str(src["scene_name"]),
                "full_history_latent": zh[j],
                "anchor_future_latent": za[j],
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
            thread_name_prefix="p0-f9-native",
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
                    f"encoded={encoded} occ_cache={nusc_source.load_occ3d.cache_info()} "
                    f"pose_cache={nusc_source.pose.cache_info()}"
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
        "vae_mode": metadata["vae_mode"],
        "latent_seed": int(a.latent_seed),
        "source_v3_cache_index_sha256": source_index_sha,
    }, indent=2))


if __name__ == "__main__":
    main()
