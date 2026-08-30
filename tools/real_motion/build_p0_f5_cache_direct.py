#!/usr/bin/env python3
"""Build P0-F5 Strong-W2Det anchor-preserving repair-endpoint cache.

Contract
--------
- Frozen Real-Motion MSP keeps the exact P0-F2/P0-F4 Top-2 routing.
- The future source/anchor is the occupancy-only strong W2Det contract.
- History conditioning is full native occupancy history.
- MSP scores define the same causal horizon-wise write support inside Top-2.
- The *training endpoint is first constructed in occupancy space*: outside the
  causal write support it is exact Strong W2Det; inside support, Strong-W2Det
  dynamic voxels are replaced by GT dynamic semantics.
- Frozen VAE then encodes this sparse repair endpoint with deterministic FP32
  posterior means. No hard latent mask defines the target.
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

from real_motion.metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS
from real_motion.msp import (
    FEATURE_DIM,
    MSP_CACHE_VERSION,
    MSPProbeHead,
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
    validate_msp_wm_sample,
)
from real_motion.nuscenes_adapter import (
    NuScenesWindowSource,
    WindowTokens,
    gt_moving_support_for_horizon,
)
from real_motion.occfm_io import OccFMVAEAdapter, file_sha256, load_official_vae
from real_motion.prepared import load_nuscenes_window_raw
from real_motion.repair_target import build_dynamic_repair_endpoint
from real_motion.runtime_config import get_cfg, make_prepare_config
from real_motion.strong_w2det import StrongW2DetConfig, strong_w2det_sequence

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


def _gt_moving_support(source, window, pcfg):
    rows = []
    for hi, tok in enumerate(window.future_tokens):
        dt = (hi + 1) * float(pcfg.frame_dt_s)
        support, _, _ = gt_moving_support_for_horizon(
            source.nusc,
            window.t0_token,
            tok,
            dt,
            pcfg.grid,
        )
        rows.append(np.asarray(support, dtype=bool))
    return np.stack(rows, axis=0)


def _load_progress(root: Path, *, resume: bool):
    final = root / "index.json"
    partial = root / PROGRESS_NAME
    if final.exists():
        raise RuntimeError(f"{final} already exists; cache is complete")
    for p in root.glob("*.pt.tmp"):
        p.unlink()
    if not partial.exists():
        return [], set(), 0
    if not resume:
        raise RuntimeError(f"{partial} exists; use --resume or choose a fresh output directory")
    obj = json.loads(partial.read_text(encoding="utf-8"))
    if obj.get("version") != MSP_WM_CACHE_VERSION_V3:
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
            "version": MSP_WM_CACHE_VERSION_V3,
            "metadata": metadata,
            "num_samples": len(entries),
            "entries": entries,
        })
        print(f"committed shard {sid}: total={len(entries)}")

    for sample in samples:
        sid = str(sample["sample_id"])
        if sid in seen:
            continue
        validate_msp_wm_sample(
            sample,
            topk=2,
            require_full_history=True,
            require_write_support=True,
            require_repair_target=True,
        )
        shard.append(sample)
        seen.add(sid)
        if len(shard) >= int(shard_size):
            commit(shard, shard_id)
            shard = []
            shard_id += 1
    if shard:
        commit(shard, shard_id)

    index = {
        "version": MSP_WM_CACHE_VERSION_V3,
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
    p.add_argument("--write-budget-ratio", type=float, default=0.15)
    p.add_argument("--route-batch-size", type=int, default=16)
    p.add_argument("--vae-batch-size", type=int, default=4)
    p.add_argument("--shard-size", type=int, default=8)
    p.add_argument("--include-eval-payload", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    if a.topk != 2:
        raise ValueError("P0-F5 main model is frozen to Top-2")
    if not 0.0 < float(a.write_budget_ratio) <= 1.0:
        raise ValueError("write-budget-ratio must be in (0,1]")
    if min(a.route_batch_size, a.vae_batch_size, a.shard_size) <= 0:
        raise ValueError("batch/shard sizes must be positive")

    device = torch.device(a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu")
    probe_meta, records, cfg = _load_probe(a.msp_cache)
    msp_ck, msp = _load_msp(a.msp_checkpoint, device)
    pcfg = make_prepare_config(cfg)
    latent_hw = tuple(int(v) for v in get_cfg(cfg, "UPSTREAM.LATENT_HW", [50, 50]))
    window_hw = tuple(int(v) for v in get_cfg(cfg, "MODEL.WINDOW_HW", [20, 20]))
    if latent_hw != (50, 50) or window_hw != (20, 20):
        raise RuntimeError("P0-F5 expects 50x50 latent and 20x20 prediction windows")
    if int(msp_ck["future_frames"]) != pcfg.future_frames:
        raise RuntimeError("MSP future-frame contract mismatch")

    source = CachedNuScenesWindowSource(a.dataroot, info_pkl=a.info_pkl, verbose=False)
    if len({str(r["sample_id"]) for r in records}) != len(records):
        raise RuntimeError("MSP cache contains duplicate sample IDs")

    # Keep routing bit-exact with P0-F4: only the target endpoint changes.
    route_map = {}
    captures = []
    valid_counts = []
    write_ratios = []
    for start in range(0, len(records), a.route_batch_size):
        rb = records[start : start + a.route_batch_size]
        batch = collate_probe_records(rb)
        bdev = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        pred = msp(bdev["features"], bdev["candidate_mask"])
        scores = rasterize_msp_scores(pred, bdev, latent_hw=latent_hw, grid=pcfg.grid)
        plan = plan_topk_score_windows(scores, window_hw=window_hw, max_windows=2)
        capture = score_capture_ratio(scores, plan).detach().cpu()
        spatial_window = window_plan_support(plan).bool()
        write = top_budget_support(scores, float(a.write_budget_ratio))
        write = write & spatial_window[:, None, :, :]

        origins = plan.origins.detach().cpu()
        valid = plan.valid.detach().cpu()
        write_cpu = write.detach().cpu()
        for j, r in enumerate(rb):
            sid = str(r["sample_id"])
            route_map[sid] = (
                origins[j].clone(),
                valid[j].clone(),
                write_cpu[j].clone(),
            )
            captures.append(float(capture[j]))
            valid_counts.append(int(valid[j].sum()))
            write_ratios.append(float(write_cpu[j].float().mean().item()))

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
    w2cfg = StrongW2DetConfig(free_label=int(pcfg.free_label))
    bev_hw = tuple(int(v) for v in pcfg.grid.shape_hwd[:2])

    def encoded_samples():
        pending = []

        def flush(rows):
            if not rows:
                return []
            hist = torch.from_numpy(np.stack([x[1] for x in rows]))
            anchor = torch.from_numpy(np.stack([x[2] for x in rows]))
            repair = torch.from_numpy(np.stack([x[3] for x in rows]))
            # Deterministic FP32 posterior means. P0-F5 target is Enc(repair endpoint),
            # not a masked subset of Enc(full GT).
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

        prepared_count = len(already_done)
        for _, _, sid, w in windows:
            if sid in already_done:
                continue
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
                    "eval_gt_moving_support": torch.from_numpy(_gt_moving_support(source, w, pcfg)),
                }
            pending.append((
                meta,
                np.asarray(raw["history_occ"]),
                np.asarray(strong_anchor),
                np.asarray(repair_endpoint),
                payload,
            ))
            prepared_count += 1
            if len(pending) >= int(a.vae_batch_size):
                for sample in flush(pending):
                    yield sample
                pending = []
            if prepared_count % 25 == 0:
                print(
                    f"P0-F5 prepared {prepared_count}/{len(windows)} "
                    f"occ_cache={source.load_occ3d.cache_info()} pose_cache={source.pose.cache_info()}"
                )
        if pending:
            for sample in flush(pending):
                yield sample

    scenes = sorted({str(r["scene_name"]) for r in records})
    metadata = {
        "protocol": "p0_f5_strong_w2det_occ_repair_endpoint_top2_v1",
        "direct_from_exact_msp_windows": True,
        "source_msp_cache": str(Path(a.msp_cache).resolve()),
        "source_msp_cache_sha256": file_sha256(a.msp_cache),
        "source_msp_mode": probe_meta.get("mode"),
        "source_msp_selection": probe_meta.get("selection"),
        "msp_checkpoint": str(Path(a.msp_checkpoint).resolve()),
        "msp_checkpoint_sha256": file_sha256(a.msp_checkpoint),
        "vae_checkpoint": str(Path(a.vae_ckpt).resolve()),
        "vae_checkpoint_sha256": file_sha256(a.vae_ckpt),
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
        "repair_endpoint_contract": "strong_anchor_outside_support_gt_dynamic_inside_support_v1",
        "w2det_min_component_voxels": int(w2cfg.min_component_voxels),
        "w2det_max_match_speed_mps": float(w2cfg.max_match_speed_mps),
        "w2det_connectivity": int(w2cfg.connectivity),
        "loss_contract": "strong_anchor_to_occ_repair_endpoint_local_flow_full_history_context_no_auxiliary_losses",
        "source": "strong_w2det_anchor_latent",
        "target": "occupancy_sparse_repair_endpoint_vae_latent",
        "latent_loss_mask": "none",
        "include_eval_payload": bool(a.include_eval_payload),
    }
    index = _write_cache(
        out_root,
        encoded_samples(),
        metadata,
        shard_size=int(a.shard_size),
        resume=bool(a.resume),
    )
    print(json.dumps({
        "output": str(out_root),
        "num_samples": index["num_samples"],
        "num_scenes": len(scenes),
        "mean_valid_windows": metadata["mean_valid_windows"],
        "slot_compute_ratio": metadata["slot_compute_ratio"],
        "mean_write_latent_ratio": metadata["mean_write_latent_ratio"],
        "score_capture": metadata["mean_score_capture_ratio"],
        "target": metadata["target"],
    }, indent=2))


if __name__ == "__main__":
    main()
