#!/usr/bin/env python3
"""Build P0-F3 Top-2 Sparse-WM latent cache directly from exact MSP windows.

This avoids materializing the very large full ``prepared`` dataset. Each exact
MSP window is prepared in memory, routed by the frozen MSP, encoded by the
frozen OccFM VAE, and immediately discarded. Only the three FP32 latent tensors
needed for training are persisted. Validation may additionally store a compact
uint8/bool evaluation payload so real occupancy metrics need no prepared cache.

Shard writes are atomic and progress is checkpointed after every shard. A rerun
with ``--resume`` skips already committed sample IDs.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
UP = ROOT / "upstream_occfm"
sys.path[:0] = [str(ROOT), str(UP)]

import numpy as np
import torch

from real_motion.composition import static_protected_compose
from real_motion.metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS
from real_motion.msp import (
    FEATURE_DIM, MSP_CACHE_VERSION, MSPProbeHead, collate_probe_records,
    rasterize_msp_scores,
)
from real_motion.msp_window import plan_topk_score_windows, score_capture_ratio
from real_motion.msp_wm_cache import MSP_WM_CACHE_VERSION, validate_msp_wm_sample
from real_motion.nuscenes_adapter import NuScenesWindowSource, WindowTokens
from real_motion.occfm_io import OccFMVAEAdapter, file_sha256, load_official_vae
from real_motion.prepared import prepare_nuscenes_window
from real_motion.runtime_config import get_cfg, make_prepare_config

PROGRESS_NAME = ".index.partial.json"


class CachedNuScenesWindowSource(NuScenesWindowSource):
    @lru_cache(maxsize=256)
    def load_occ3d(self, scene_name, token, require_lidar_mask=True):
        return super().load_occ3d(scene_name, token, require_lidar_mask=require_lidar_mask)

    @lru_cache(maxsize=2048)
    def pose(self, token):
        return super().pose(token)


def _atomic_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _load_probe(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if obj.get("version") != MSP_CACHE_VERSION:
        raise RuntimeError(f"unsupported MSP cache version {obj.get('version')}")
    records = obj.get("records") or []
    meta = obj.get("metadata") or {}
    if not records:
        raise RuntimeError("MSP cache contains no records")
    cfg = meta.get("resolved_config")
    if cfg is None:
        raise RuntimeError("MSP cache lacks resolved_config")
    return meta, records, cfg


def _load_msp(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if int(ck.get("feature_dim", -1)) != FEATURE_DIM:
        raise RuntimeError("MSP checkpoint feature contract mismatch")
    model = MSPProbeHead(
        feature_dim=FEATURE_DIM,
        hidden_dim=int(ck["hidden_dim"]),
        num_heads=int(ck["num_heads"]),
        num_modes=int(ck["num_modes"]),
        future_frames=int(ck["future_frames"]),
    )
    model.load_state_dict(ck["state_dict"], strict=True)
    return ck, model.to(device).eval()


def _window_from_record(r, history_frames, future_frames):
    required = ("sample_id", "scene_name", "history_tokens", "t0_token", "future_tokens")
    missing = [k for k in required if k not in r]
    if missing:
        raise KeyError(f"MSP record missing exact-window keys {missing}")
    scene = str(r["scene_name"])
    hist = tuple(str(x) for x in r["history_tokens"])
    t0 = str(r["t0_token"])
    fut = tuple(str(x) for x in r["future_tokens"])
    sid = str(r["sample_id"])
    if len(hist) != int(history_frames) or len(fut) != int(future_frames):
        raise RuntimeError(f"{sid}: window length mismatch")
    if hist[-1] != t0 or sid != f"{scene}:{t0}":
        raise RuntimeError(f"{sid}: exact-window token contract mismatch")
    return WindowTokens(scene, hist, t0, fut)


def _compose_anchor(sample):
    static = torch.from_numpy(np.asarray(sample["static_future_occ"]))
    kta = torch.from_numpy(np.asarray(sample["kta_future_occ"]))
    protected = torch.from_numpy(np.asarray(sample["confident_static_future_mask"]))
    return static_protected_compose(
        static, kta, protected, DYNAMIC_CLASS_IDS, write_support=None
    ).cpu()


def _compact_eval_payload(sample):
    # Semantic labels are in [0,17], so uint8 is lossless. Masks remain bool.
    return {
        "eval_future_gt_occ": torch.from_numpy(np.asarray(sample["future_gt_occ"], dtype=np.uint8)),
        "eval_static_future_occ": torch.from_numpy(np.asarray(sample["static_future_occ"], dtype=np.uint8)),
        "eval_confident_static_future_mask": torch.from_numpy(
            np.asarray(sample["confident_static_future_mask"], dtype=bool)
        ),
        "eval_kta_future_occ": torch.from_numpy(np.asarray(sample["kta_future_occ"], dtype=np.uint8)),
        "eval_gt_moving_support": torch.from_numpy(np.asarray(sample["gt_moving_support"], dtype=bool)),
    }


def _load_progress(root: Path, *, resume: bool):
    final = root / "index.json"
    partial = root / PROGRESS_NAME
    if final.exists():
        raise RuntimeError(f"{final} already exists; cache is complete")
    # A failed atomic torch.save may leave only a .tmp file. It is never committed.
    for p in root.glob("*.pt.tmp"):
        p.unlink()
    if not partial.exists():
        return [], set(), 0
    if not resume:
        raise RuntimeError(f"{partial} exists; use --resume or choose a fresh output directory")
    obj = json.loads(partial.read_text(encoding="utf-8"))
    if obj.get("version") != MSP_WM_CACHE_VERSION:
        raise RuntimeError("partial cache version mismatch")
    entries = obj.get("entries", [])
    seen = {str(e["sample_id"]) for e in entries}
    shard_ids = [int(Path(e["shard"]).stem.split("_")[-1]) for e in entries]
    return entries, seen, (max(shard_ids) + 1 if shard_ids else 0)


def _write_cache(root: Path, samples, metadata, *, shard_size: int, resume: bool):
    root.mkdir(parents=True, exist_ok=True)
    entries, seen, shard_id = _load_progress(root, resume=resume)
    shard = []

    def commit(items, sid):
        nonlocal entries
        if not items:
            return
        name = f"shard_{sid:05d}.pt"
        dst = root / name
        tmp = root / (name + ".tmp")
        torch.save(items, tmp)
        os.replace(tmp, dst)
        for j, s in enumerate(items):
            entries.append({
                "shard": name,
                "index": j,
                "sample_id": str(s["sample_id"]),
                "scene_name": str(s["scene_name"]),
            })
        _atomic_json(root / PROGRESS_NAME, {
            "version": MSP_WM_CACHE_VERSION,
            "metadata": metadata,
            "num_samples": len(entries),
            "entries": entries,
        })
        print(f"committed shard {sid}: total={len(entries)}")

    for sample in samples:
        sid = str(sample["sample_id"])
        if sid in seen:
            continue
        validate_msp_wm_sample(sample, topk=2)
        shard.append(sample)
        seen.add(sid)
        if len(shard) >= shard_size:
            commit(shard, shard_id)
            shard = []
            shard_id += 1
    if shard:
        commit(shard, shard_id)
    index = {
        "version": MSP_WM_CACHE_VERSION,
        "metadata": metadata,
        "num_samples": len(entries),
        "entries": entries,
    }
    _atomic_json(root / "index.json", index)
    partial = root / PROGRESS_NAME
    if partial.exists():
        partial.unlink()
    return index


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--msp-cache", required=True)
    p.add_argument("--msp-checkpoint", required=True)
    p.add_argument("--dataroot", required=True)
    p.add_argument("--info-pkl", required=True)
    p.add_argument("--vae-ckpt", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--topk", type=int, default=2)
    p.add_argument("--route-batch-size", type=int, default=16)
    p.add_argument("--vae-batch-size", type=int, default=4)
    p.add_argument("--shard-size", type=int, default=8)
    p.add_argument("--include-eval-payload", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    if a.topk != 2:
        raise ValueError("P0-F3 main training is frozen to Top-2")
    if min(a.route_batch_size, a.vae_batch_size, a.shard_size) <= 0:
        raise ValueError("batch/shard sizes must be positive")

    device = torch.device(a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu")
    probe_meta, records, cfg = _load_probe(a.msp_cache)
    msp_ck, msp = _load_msp(a.msp_checkpoint, device)
    pcfg = make_prepare_config(cfg)
    latent_hw = tuple(int(v) for v in get_cfg(cfg, "UPSTREAM.LATENT_HW", [50, 50]))
    window_hw = tuple(int(v) for v in get_cfg(cfg, "MODEL.WINDOW_HW", [20, 20]))
    if latent_hw != (50, 50) or window_hw != (20, 20):
        raise RuntimeError("P0-F3 expects 50x50 latent and 20x20 windows")
    if int(msp_ck["future_frames"]) != pcfg.future_frames:
        raise RuntimeError("MSP future-frame contract mismatch")

    source = CachedNuScenesWindowSource(a.dataroot, info_pkl=a.info_pkl, verbose=False)
    by_id = {str(r["sample_id"]): r for r in records}
    if len(by_id) != len(records):
        raise RuntimeError("MSP cache contains duplicate sample IDs")

    # Route the frozen records first. This is cheap and avoids keeping full raw
    # prepared samples in RAM while MSP inference is batched.
    route_map = {}
    captures = []
    valid_counts = []
    for start in range(0, len(records), a.route_batch_size):
        rb = records[start:start + a.route_batch_size]
        batch = collate_probe_records(rb)
        bdev = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        pred = msp(bdev["features"], bdev["candidate_mask"])
        scores = rasterize_msp_scores(pred, bdev, latent_hw=latent_hw, grid=pcfg.grid)
        plan = plan_topk_score_windows(scores, window_hw=window_hw, max_windows=2)
        cap = score_capture_ratio(scores, plan).detach().cpu()
        origins = plan.origins.detach().cpu()
        valid = plan.valid.detach().cpu()
        for j, r in enumerate(rb):
            sid = str(r["sample_id"])
            route_map[sid] = (origins[j].clone(), valid[j].clone())
            captures.append(float(cap[j]))
            valid_counts.append(int(valid[j].sum()))

    # Reorder the exact same set only for locality. This materially improves NAS
    # reuse because overlapping windows from the same scene become adjacent.
    windows = []
    for r in records:
        w = _window_from_record(r, pcfg.history_frames, pcfg.future_frames)
        ts = int(source.nusc.get("sample", w.t0_token)["timestamp"])
        windows.append((w.scene_name, ts, str(r["sample_id"]), w))
    windows.sort(key=lambda x: (x[0], x[1]))

    vae, _ = load_official_vae(UP, a.vae_ckpt, device)
    va = OccFMVAEAdapter(vae)
    out_root = Path(a.output)
    _, already_done, _ = _load_progress(out_root, resume=a.resume) if out_root.exists() else ([], set(), 0)

    def encoded_samples():
        pending = []

        def flush_pending(rows):
            if not rows:
                return []
            moving = torch.from_numpy(np.stack([x[1] for x in rows]))
            anchor = torch.stack([x[2] for x in rows])
            gt = torch.from_numpy(np.stack([x[3] for x in rows]))
            # Deterministic FP32 posterior means; no sampling/bf16 target noise.
            zh = va.encode(moving, mode="mean").float().cpu()
            za = va.encode(anchor, mode="mean").float().cpu()
            zg = va.encode(gt, mode="mean").float().cpu()
            out = []
            for j, (meta, _, _, _, payload) in enumerate(rows):
                sample = {
                    "sample_id": meta["sample_id"],
                    "scene_name": meta["scene_name"],
                    "moving_history_latent": zh[j],
                    "anchor_future_latent": za[j],
                    "gt_future_latent": zg[j],
                    "window_origins": meta["window_origins"],
                    "window_valid": meta["window_valid"],
                    "trajectory": meta["trajectory"],
                }
                if payload is not None:
                    sample.update(payload)
                out.append(sample)
            return out

        prepared_count = len(already_done)
        for _, _, sid, w in windows:
            if sid in already_done:
                continue
            raw = prepare_nuscenes_window(source, w, pcfg, include_gt=True)
            origins, valid = route_map[sid]
            meta = {
                "sample_id": sid,
                "scene_name": str(raw["scene_name"]),
                "window_origins": origins,
                "window_valid": valid,
                "trajectory": torch.as_tensor(raw["trajectory"], dtype=torch.float32),
            }
            moving = np.asarray(raw["moving_history_occ"])
            anchor = _compose_anchor(raw)
            gt = np.asarray(raw["future_gt_occ"])
            payload = _compact_eval_payload(raw) if a.include_eval_payload else None
            # Keep only the small tensors needed for the imminent VAE batch;
            # release the large full prepared dictionary immediately.
            pending.append((meta, moving, anchor, gt, payload))
            prepared_count += 1
            if len(pending) >= a.vae_batch_size:
                for s in flush_pending(pending):
                    yield s
                pending = []
            if prepared_count % 25 == 0:
                print(
                    f"direct prepared {prepared_count}/{len(windows)} "
                    f"occ_cache={source.load_occ3d.cache_info()} pose_cache={source.pose.cache_info()}"
                )
        if pending:
            for s in flush_pending(pending):
                yield s

    scenes = sorted({str(r["scene_name"]) for r in records})
    metadata = {
        "protocol": "p0_f3_top2_direct_exact_latent_cache_v1",
        "direct_from_exact_msp_windows": True,
        "source_msp_cache": str(Path(a.msp_cache).resolve()),
        "source_msp_cache_sha256": file_sha256(a.msp_cache),
        "source_msp_mode": probe_meta.get("mode"),
        "source_msp_selection": probe_meta.get("selection"),
        "source_msp_seed": probe_meta.get("seed"),
        "msp_checkpoint": str(Path(a.msp_checkpoint).resolve()),
        "msp_checkpoint_sha256": file_sha256(a.msp_checkpoint),
        "vae_checkpoint": str(Path(a.vae_ckpt).resolve()),
        "vae_checkpoint_sha256": file_sha256(a.vae_ckpt),
        "vae_mode": "mean",
        "vae_amp": False,
        "topk": 2,
        "latent_hw": list(latent_hw),
        "window_hw": list(window_hw),
        "shared_windows_across_horizons": True,
        "score_aggregation": "sum_over_6_future_horizons",
        "selection": "greedy_marginal_msp_score",
        "target": "full_gt_latent_in_selected_window",
        "source": "causal_kta_zero_anchor_latent",
        "loss_contract": "anchor_to_gt_local_flow_no_auxiliary_losses",
        "trajectory_length": 12,
        "num_records": len(records),
        "num_unique_scenes": len(scenes),
        "scene_names": scenes,
        "mean_score_capture_ratio": float(np.mean(captures)),
        "mean_valid_windows": float(np.mean(valid_counts)),
        "slot_compute_ratio": float(np.mean(valid_counts) * 400.0 / 2500.0),
        "include_eval_payload": bool(a.include_eval_payload),
        "eval_payload_dtypes": "semantic=uint8, masks=bool" if a.include_eval_payload else None,
    }
    idx = _write_cache(
        out_root, encoded_samples(), metadata,
        shard_size=int(a.shard_size), resume=bool(a.resume),
    )
    if int(idx["num_samples"]) != len(records):
        raise RuntimeError(f"cache saved {idx['num_samples']} but expected {len(records)}")
    print(json.dumps({
        "output": str(out_root),
        "num_samples": idx["num_samples"],
        "num_unique_scenes": len(scenes),
        "mean_score_capture_ratio": metadata["mean_score_capture_ratio"],
        "mean_valid_windows": metadata["mean_valid_windows"],
        "slot_compute_ratio": metadata["slot_compute_ratio"],
        "include_eval_payload": metadata["include_eval_payload"],
    }, indent=2))


if __name__ == "__main__":
    main()
