#!/usr/bin/env python3
"""Incrementally upgrade an existing P0-F4 cache to the P0-F5 target contract.

This path intentionally does *not* rerun frozen MSP routing or re-encode the
already-cached full-history / Strong-W2Det anchor latents. GPU work is only:

    repair endpoint occupancy -> frozen VAE mean -> repair_target_latent

For a validation cache that already contains compact eval payload, future GT and
Strong-W2Det occupancy are reused directly, so no nuScenes/W2Det recomputation is
needed. For the train cache (which normally has no eval payload), exact window
tokens are recovered from the frozen MSP probe cache; raw future GT and the
Strong-W2Det occupancy anchor are reconstructed on CPU before the single new VAE
encode.
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

from real_motion.cache_upgrade import (
    P0_F5_LOSS_CONTRACT,
    P0_F5_REPAIR_CONTRACT,
    P0_F5_TARGET,
    build_upgraded_sample,
    make_p0_f5_upgrade_metadata,
    validate_p0_f4_upgrade_source,
)
from real_motion.metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS
from real_motion.msp import MSP_CACHE_VERSION, latent_support_to_bev
from real_motion.msp_wm_cache import (
    MSP_WM_CACHE_VERSION_V3,
    MSPWorldModelCacheDataset,
    validate_msp_wm_sample,
)
from real_motion.nuscenes_adapter import NuScenesWindowSource, WindowTokens
from real_motion.occfm_io import OccFMVAEAdapter, file_sha256, load_official_vae
from real_motion.prepared import load_nuscenes_window_raw
from real_motion.repair_target import build_dynamic_repair_endpoint
from real_motion.runtime_config import make_prepare_config
from real_motion.strong_w2det import StrongW2DetConfig, strong_w2det_sequence

PROGRESS_NAME = ".index.partial.json"
EVAL_OCC_KEYS = ("eval_future_gt_occ", "eval_strong_anchor_occ")
FREE_LABEL = 17
PROGRESS_PROVENANCE_KEYS = (
    "incremental_upgrade",
    "incremental_upgrade_source_index_sha256",
    "repair_endpoint_contract",
    "target",
    "vae_checkpoint_sha256",
)


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


def _validate_partial_metadata(partial_metadata: dict, expected_metadata: dict) -> None:
    for key in PROGRESS_PROVENANCE_KEYS:
        if partial_metadata.get(key) != expected_metadata.get(key):
            raise RuntimeError(
                f"partial cache provenance mismatch for {key}: "
                f"{partial_metadata.get(key)!r} != {expected_metadata.get(key)!r}. "
                "Do not resume a direct-builder or different-source cache into this output."
            )


def _load_progress(root: Path, *, resume: bool, expected_metadata: dict | None = None):
    final = root / "index.json"
    partial = root / PROGRESS_NAME
    if final.exists():
        raise RuntimeError(f"{final} already exists; upgraded cache is complete")
    for p in root.glob("*.pt.tmp"):
        p.unlink()
    if not partial.exists():
        return [], set(), 0
    if not resume:
        raise RuntimeError(f"{partial} exists; use --resume or choose a fresh output directory")
    obj = json.loads(partial.read_text(encoding="utf-8"))
    if obj.get("version") != MSP_WM_CACHE_VERSION_V3:
        raise RuntimeError("partial cache version is not P0-F5/v3")
    if expected_metadata is not None:
        _validate_partial_metadata(obj.get("metadata") or {}, expected_metadata)
    entries = obj.get("entries", [])
    seen = {str(e["sample_id"]) for e in entries}
    if len(seen) != len(entries):
        raise RuntimeError("partial cache contains duplicate sample IDs")
    shard_ids = [int(Path(e["shard"]).stem.split("_")[-1]) for e in entries]
    return entries, seen, (max(shard_ids) + 1 if shard_ids else 0)


def _write_cache(root: Path, samples, metadata, *, shard_size: int, resume: bool):
    root.mkdir(parents=True, exist_ok=True)
    entries, seen, shard_id = _load_progress(
        root,
        resume=resume,
        expected_metadata=metadata,
    )
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
        for j, sample in enumerate(items):
            entries.append({
                "shard": name,
                "index": j,
                "sample_id": str(sample["sample_id"]),
                "scene_name": str(sample["scene_name"]),
            })
        _atomic_json(root / PROGRESS_NAME, {
            "version": MSP_WM_CACHE_VERSION_V3,
            "metadata": metadata,
            "num_samples": len(entries),
            "entries": entries,
        })
        print(f"committed upgraded shard {sid}: total={len(entries)}")

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


def _resolve_file(cli_value: str | None, metadata: dict, key: str, label: str) -> Path:
    value = cli_value or metadata.get(key)
    if not value:
        raise RuntimeError(f"{label} path is required; pass it explicitly")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _load_probe(path: Path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if obj.get("version") != MSP_CACHE_VERSION:
        raise RuntimeError(f"unsupported MSP probe cache version {obj.get('version')}")
    records = obj.get("records") or []
    metadata = obj.get("metadata") or {}
    cfg = metadata.get("resolved_config")
    if not records or cfg is None:
        raise RuntimeError("MSP probe cache lacks records/resolved_config")
    if len({str(r["sample_id"]) for r in records}) != len(records):
        raise RuntimeError("MSP probe cache contains duplicate sample IDs")
    return metadata, records, cfg


def _window_from_record(r, history_frames: int, future_frames: int) -> WindowTokens:
    required = ("sample_id", "scene_name", "history_tokens", "t0_token", "future_tokens")
    missing = [k for k in required if k not in r]
    if missing:
        raise KeyError(f"MSP record missing exact-window keys {missing}")
    sid = str(r["sample_id"])
    scene = str(r["scene_name"])
    hist = tuple(str(x) for x in r["history_tokens"])
    t0 = str(r["t0_token"])
    fut = tuple(str(x) for x in r["future_tokens"])
    if len(hist) != int(history_frames) or len(fut) != int(future_frames):
        raise RuntimeError(f"{sid}: exact-window length mismatch")
    if hist[-1] != t0 or sid != f"{scene}:{t0}":
        raise RuntimeError(f"{sid}: exact-window token contract mismatch")
    return WindowTokens(scene, hist, t0, fut)


def _assert_w2det_metadata(meta: dict, cfg: StrongW2DetConfig) -> None:
    checks = {
        "w2det_min_component_voxels": int(cfg.min_component_voxels),
        "w2det_max_match_speed_mps": float(cfg.max_match_speed_mps),
        "w2det_connectivity": int(cfg.connectivity),
    }
    for key, expected in checks.items():
        if key in meta and meta[key] != expected:
            raise RuntimeError(f"source P0-F4 {key}={meta[key]} differs from current StrongW2Det {expected}")


def _repair_from_cached_eval(sample: dict) -> np.ndarray:
    missing = [k for k in EVAL_OCC_KEYS if k not in sample]
    if missing:
        raise RuntimeError(f"{sample['sample_id']}: source eval payload missing {missing}")
    anchor = sample["eval_strong_anchor_occ"].cpu().numpy()
    gt = sample["eval_future_gt_occ"].cpu().numpy()
    bev_hw = tuple(int(v) for v in anchor.shape[1:3])
    write_bev = latent_support_to_bev(
        sample["msp_write_support_latent"].bool(), bev_hw
    ).cpu().numpy().astype(bool)
    return build_dynamic_repair_endpoint(
        anchor,
        gt,
        write_bev,
        dynamic_class_ids=DYNAMIC_CLASS_IDS,
        free_label=FREE_LABEL,
    )


def _repair_from_raw(
    sample: dict,
    *,
    record_map: dict,
    source: CachedNuScenesWindowSource,
    pcfg,
    w2cfg: StrongW2DetConfig,
) -> np.ndarray:
    sid = str(sample["sample_id"])
    if sid not in record_map:
        raise RuntimeError(f"{sid}: not found in frozen MSP probe cache")
    window = _window_from_record(record_map[sid], pcfg.history_frames, pcfg.future_frames)
    raw = load_nuscenes_window_raw(source, window, pcfg, include_gt=True)

    # Cheap exact-window guard: the trajectory already stored in P0-F4 must
    # match the reconstructed record. This catches a wrong train info/probe file
    # without paying for any extra VAE encode.
    cached_traj = sample["trajectory"].cpu().numpy()
    raw_traj = np.asarray(raw["trajectory"], dtype=np.float32)
    if cached_traj.shape != raw_traj.shape or not np.allclose(
        cached_traj, raw_traj, rtol=1e-5, atol=1e-5
    ):
        raise RuntimeError(f"{sid}: reconstructed exact window does not match cached trajectory")

    anchor = strong_w2det_sequence(
        raw["history_occ"],
        raw["history_poses"],
        raw["future_poses"],
        frame_dt_s=float(pcfg.frame_dt_s),
        grid=pcfg.grid,
        cfg=w2cfg,
    )
    bev_hw = tuple(int(v) for v in pcfg.grid.shape_hwd[:2])
    write_bev = latent_support_to_bev(
        sample["msp_write_support_latent"].bool(), bev_hw
    ).cpu().numpy().astype(bool)
    return build_dynamic_repair_endpoint(
        anchor,
        np.asarray(raw["future_gt_occ"]),
        write_bev,
        dynamic_class_ids=DYNAMIC_CLASS_IDS,
        free_label=int(pcfg.free_label),
    )


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-cache", required=True, help="existing P0-F4/v2 cache")
    p.add_argument("--output", required=True, help="new P0-F5/v3 cache directory")
    p.add_argument("--vae-ckpt", default=None, help="defaults to P0-F4 metadata path")
    p.add_argument("--msp-cache", default=None, help="needed only when train cache lacks eval payload")
    p.add_argument("--dataroot", default=None, help="needed only for train/raw fallback")
    p.add_argument("--info-pkl", default=None, help="needed only for train/raw fallback")
    p.add_argument("--vae-batch-size", type=int, default=8)
    p.add_argument("--shard-size", type=int, default=16)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    if min(a.vae_batch_size, a.shard_size) <= 0:
        raise ValueError("vae-batch-size/shard-size must be positive")

    source_root = Path(a.source_cache).expanduser().resolve()
    out_root = Path(a.output).expanduser().resolve()
    if out_root == source_root:
        raise RuntimeError("--output must be different from --source-cache")
    if not (source_root / "index.json").is_file():
        raise FileNotFoundError(f"P0-F4 cache index not found: {source_root / 'index.json'}")
    source_ds = MSPWorldModelCacheDataset(source_root)
    source_meta = source_ds.metadata
    validate_p0_f4_upgrade_source(source_ds.version, source_meta)
    source_ids = [str(e["sample_id"]) for e in source_ds.entries]
    if len(set(source_ids)) != len(source_ids):
        raise RuntimeError("source P0-F4 cache contains duplicate sample IDs")

    vae_path = _resolve_file(a.vae_ckpt, source_meta, "vae_checkpoint", "VAE checkpoint")
    expected_vae_sha = source_meta.get("vae_checkpoint_sha256")
    actual_vae_sha = file_sha256(vae_path)
    if expected_vae_sha and actual_vae_sha != expected_vae_sha:
        raise RuntimeError("VAE checkpoint differs from the one used by P0-F4 cache")

    # Validation caches already carry both occupancy tensors needed to build the
    # repair endpoint. Train caches usually do not and therefore take the raw
    # fallback, but routing/history/anchor latents are still reused unchanged.
    first = source_ds[0]
    has_eval_payload = all(k in first for k in EVAL_OCC_KEYS)
    if bool(source_meta.get("include_eval_payload", False)) != has_eval_payload:
        raise RuntimeError("P0-F4 include_eval_payload metadata does not match sample payload")

    pcfg = None
    record_map = None
    raw_source = None
    w2cfg = StrongW2DetConfig(free_label=FREE_LABEL)
    _assert_w2det_metadata(source_meta, w2cfg)
    probe_path = None
    if not has_eval_payload:
        if not a.dataroot or not a.info_pkl:
            raise RuntimeError(
                "train P0-F4 cache has no occupancy eval payload; pass --dataroot and --info-pkl"
            )
        probe_path = _resolve_file(a.msp_cache, source_meta, "source_msp_cache", "MSP probe cache")
        expected_probe_sha = source_meta.get("source_msp_cache_sha256")
        if expected_probe_sha and file_sha256(probe_path) != expected_probe_sha:
            raise RuntimeError("MSP probe cache differs from the one that built P0-F4")
        _, records, cfg = _load_probe(probe_path)
        pcfg = make_prepare_config(cfg)
        if int(pcfg.free_label) != FREE_LABEL:
            raise RuntimeError(f"P0-F5 repair contract expects free label {FREE_LABEL}")
        record_map = {str(r["sample_id"]): r for r in records}
        missing_ids = sorted(set(source_ids) - set(record_map))
        if missing_ids:
            raise RuntimeError(f"{len(missing_ids)} P0-F4 sample IDs missing from MSP probe cache")
        raw_source = CachedNuScenesWindowSource(a.dataroot, info_pkl=a.info_pkl, verbose=False)

    metadata = make_p0_f5_upgrade_metadata(
        source_meta,
        source_cache=source_root,
        source_index_sha256=file_sha256(source_root / "index.json"),
    )
    metadata["vae_checkpoint"] = str(vae_path)
    metadata["vae_checkpoint_sha256"] = actual_vae_sha
    metadata["include_eval_payload"] = bool(has_eval_payload)
    metadata["incremental_raw_fallback"] = not has_eval_payload
    metadata["incremental_msp_probe_cache"] = str(probe_path) if probe_path is not None else None
    if metadata.get("repair_endpoint_contract") != P0_F5_REPAIR_CONTRACT:
        raise AssertionError("unexpected P0-F5 repair contract")
    if metadata.get("loss_contract") != P0_F5_LOSS_CONTRACT or metadata.get("target") != P0_F5_TARGET:
        raise AssertionError("unexpected P0-F5 target metadata")

    _, already_done, _ = (
        _load_progress(out_root, resume=a.resume, expected_metadata=metadata)
        if out_root.exists()
        else ([], set(), 0)
    )
    unknown_done = sorted(already_done - set(source_ids))
    if unknown_done:
        raise RuntimeError("partial output contains sample IDs not present in source P0-F4 cache")

    # Do not even instantiate the VAE if a resume already contains every source
    # sample and only needs its final index committed.
    va = None
    if len(already_done) < len(source_ds):
        device = torch.device(a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu")
        vae, _ = load_official_vae(UP, vae_path, device)
        va = OccFMVAEAdapter(vae)

    def upgraded_samples():
        pending = []
        prepared = len(already_done)
        reused_eval = 0
        rebuilt_raw = 0

        def flush(rows):
            if not rows:
                return []
            if va is None:
                raise RuntimeError("internal error: VAE is unavailable for pending repair endpoints")
            repair_batch = torch.from_numpy(np.stack([row[1] for row in rows]))
            zr = va.encode(repair_batch, mode="mean").float().cpu()
            out = []
            for j, (src_sample, repair_occ) in enumerate(rows):
                out.append(build_upgraded_sample(
                    src_sample,
                    zr[j],
                    repair_target_occ=(
                        torch.from_numpy(np.asarray(repair_occ, dtype=np.uint8))
                        if has_eval_payload else None
                    ),
                ))
            return out

        for idx, entry in enumerate(source_ds.entries):
            sid = str(entry["sample_id"])
            if sid in already_done:
                continue
            sample = source_ds[idx]
            if str(sample["sample_id"]) != sid:
                raise RuntimeError("source cache index/sample mismatch")
            if has_eval_payload:
                repair = _repair_from_cached_eval(sample)
                reused_eval += 1
            else:
                repair = _repair_from_raw(
                    sample,
                    record_map=record_map,
                    source=raw_source,
                    pcfg=pcfg,
                    w2cfg=w2cfg,
                )
                rebuilt_raw += 1
            pending.append((sample, np.asarray(repair)))
            prepared += 1
            if len(pending) >= int(a.vae_batch_size):
                for upgraded in flush(pending):
                    yield upgraded
                pending = []
            if prepared % 25 == 0 or prepared == len(source_ds):
                msg = (
                    f"P0-F4->F5 upgraded {prepared}/{len(source_ds)} "
                    f"eval_payload={reused_eval} raw_fallback={rebuilt_raw}"
                )
                if raw_source is not None:
                    msg += (
                        f" occ_cache={raw_source.load_occ3d.cache_info()}"
                        f" pose_cache={raw_source.pose.cache_info()}"
                    )
                print(msg)
        if pending:
            for upgraded in flush(pending):
                yield upgraded

    index = _write_cache(
        out_root,
        upgraded_samples(),
        metadata,
        shard_size=int(a.shard_size),
        resume=bool(a.resume),
    )
    final_ids = [str(e["sample_id"]) for e in index["entries"]]
    if len(final_ids) != len(source_ids) or set(final_ids) != set(source_ids):
        raise RuntimeError(
            f"upgraded cache sample set mismatch: source={len(source_ids)} output={len(final_ids)}"
        )
    print(json.dumps({
        "output": str(out_root),
        "num_samples": index["num_samples"],
        "source_cache": str(source_root),
        "incremental_upgrade": True,
        "reused_fields": metadata["incremental_reused_tensor_keys"],
        "new_gpu_encode": metadata["incremental_new_gpu_encode"],
        "raw_fallback": metadata["incremental_raw_fallback"],
        "target": metadata["target"],
    }, indent=2))


if __name__ == "__main__":
    main()
