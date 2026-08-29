#!/usr/bin/env python3
"""Prepare exactly the windows frozen in an MSP probe cache.

This is the formal P0-F3 bridge from the already-validated MSP sample set to
Sparse-WM training assets. It never re-selects windows from nuScenes: the
scene/history/t0/future tokens are read verbatim from the probe cache.

For I/O locality the same window set is reordered by (scene, t0 timestamp)
before preparation, and a small in-process LRU caches repeated Occ3D frames and
poses. Reordering changes neither the sample set nor any training target.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from real_motion.msp import MSP_CACHE_VERSION
from real_motion.nuscenes_adapter import NuScenesWindowSource, WindowTokens
from real_motion.occfm_io import file_sha256
from real_motion.prepared import PREPARED_VERSION, prepare_nuscenes_window, save_prepared_shards
from real_motion.runtime_config import get_cfg, make_prepare_config, save_resolved_config


class CachedNuScenesWindowSource(NuScenesWindowSource):
    """Small read-through cache for overlapping exact windows.

    The formal preparation code stacks/copies loaded arrays before geometry and
    motion processing, so cached source arrays are treated as read-only.
    """

    @lru_cache(maxsize=256)
    def load_occ3d(self, scene_name, token, require_lidar_mask=True):
        return super().load_occ3d(scene_name, token, require_lidar_mask=require_lidar_mask)

    @lru_cache(maxsize=2048)
    def pose(self, token):
        return super().pose(token)


def _load_probe(path: str):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if obj.get("version") != MSP_CACHE_VERSION:
        raise RuntimeError(
            f"expected MSP cache version {MSP_CACHE_VERSION}, got {obj.get('version')}"
        )
    meta = obj.get("metadata") or {}
    records = obj.get("records") or []
    if not records:
        raise RuntimeError("MSP cache contains no records")
    cfg = meta.get("resolved_config")
    if cfg is None:
        raise RuntimeError("MSP cache lacks resolved_config; exact preparation must reuse it")
    return meta, records, cfg


def _window_from_record(record, *, history_frames: int, future_frames: int) -> WindowTokens:
    required = ("sample_id", "scene_name", "history_tokens", "t0_token", "future_tokens")
    missing = [k for k in required if k not in record]
    if missing:
        raise KeyError(f"MSP record missing exact-window keys {missing}")
    scene = str(record["scene_name"])
    hist = tuple(str(x) for x in record["history_tokens"])
    t0 = str(record["t0_token"])
    fut = tuple(str(x) for x in record["future_tokens"])
    sid = str(record["sample_id"])
    if len(hist) != int(history_frames) or len(fut) != int(future_frames):
        raise RuntimeError(
            f"{sid}: token lengths {len(hist)}+{len(fut)} do not match "
            f"{history_frames}+{future_frames}"
        )
    if not hist or hist[-1] != t0:
        raise RuntimeError(f"{sid}: history does not terminate at t0 token")
    expected_sid = f"{scene}:{t0}"
    if sid != expected_sid:
        raise RuntimeError(f"sample_id mismatch: cache={sid}, expected={expected_sid}")
    return WindowTokens(scene, hist, t0, fut)


def _t0_timestamp(source: NuScenesWindowSource, w: WindowTokens) -> int:
    return int(source.nusc.get("sample", w.t0_token)["timestamp"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--msp-cache", required=True)
    p.add_argument("--dataroot", required=True)
    p.add_argument("--info-pkl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--shard-size", type=int, default=None)
    a = p.parse_args()

    meta, records, cfg = _load_probe(a.msp_cache)
    pcfg = make_prepare_config(cfg)
    shard_size = int(a.shard_size or get_cfg(cfg, "CACHE.PREPARED_SHARD_SIZE", 16))
    if shard_size <= 0:
        raise ValueError("shard-size must be positive")

    out = Path(a.output)
    if (out / "index.json").exists():
        raise RuntimeError(
            f"{out}/index.json already exists; refusing to mix/overwrite prepared assets"
        )

    source = CachedNuScenesWindowSource(a.dataroot, info_pkl=a.info_pkl, verbose=False)
    windows = [
        _window_from_record(
            r,
            history_frames=pcfg.history_frames,
            future_frames=pcfg.future_frames,
        )
        for r in records
    ]
    sample_ids = [f"{w.scene_name}:{w.t0_token}" for w in windows]
    if len(set(sample_ids)) != len(sample_ids):
        raise RuntimeError("MSP cache contains duplicate exact windows")

    # Preserve the exact set, but cluster by scene/time for substantially better
    # locality on NAS and to make the small frame/pose LRU useful.
    windows.sort(key=lambda w: (w.scene_name, _t0_timestamp(source, w)))

    expected_ids = set(sample_ids)
    produced_ids = set()

    def gen():
        for i, w in enumerate(windows):
            sample = prepare_nuscenes_window(source, w, pcfg, include_gt=True)
            sid = str(sample["sample_id"])
            if sid not in expected_ids:
                raise RuntimeError(f"prepared unexpected sample {sid}")
            if sid in produced_ids:
                raise RuntimeError(f"prepared duplicate sample {sid}")
            produced_ids.add(sid)
            if i % 25 == 0:
                occ_info = source.load_occ3d.cache_info()
                pose_info = source.pose.cache_info()
                print(
                    f"prepared exact {i}/{len(windows)} {sid} "
                    f"occ_cache={occ_info.hits}/{occ_info.misses} "
                    f"pose_cache={pose_info.hits}/{pose_info.misses}"
                )
            yield sample

    scenes = sorted({w.scene_name for w in windows})
    prepared_meta = {
        "prepared_version": PREPARED_VERSION,
        "exact_window_source": "msp_probe_cache_records_v1",
        "source_msp_cache": str(Path(a.msp_cache).resolve()),
        "source_msp_cache_sha256": file_sha256(a.msp_cache),
        "source_msp_version": meta.get("version", MSP_CACHE_VERSION),
        "source_msp_mode": meta.get("mode"),
        "source_msp_selection": meta.get("selection"),
        "source_msp_seed": meta.get("seed"),
        "source_msp_num_windows": int(meta.get("num_windows", len(records))),
        "source_msp_num_unique_scenes": int(meta.get("num_unique_scenes", len(scenes))),
        "num_exact_windows": len(windows),
        "num_unique_scenes": len(scenes),
        "scene_names": scenes,
        "prepared_order": "scene_then_t0_timestamp_for_io_locality",
        "dataroot": str(Path(a.dataroot).resolve()),
        "info_pkl": str(Path(a.info_pkl).resolve()),
        "history_frames": pcfg.history_frames,
        "future_frames": pcfg.future_frames,
        "frame_dt_s": pcfg.frame_dt_s,
        "trajectory_protocol": pcfg.trajectory_protocol,
        "trajectory_length": pcfg.trajectory_length,
        "trajectory_hist_last": pcfg.trajectory_hist_last,
        "trajectory_zero_prefix": pcfg.trajectory_zero_prefix,
        "require_temporal_info": pcfg.require_temporal_info,
        "support_geometry": pcfg.support_geometry,
        "endpoint_tube_radii": list(pcfg.endpoint_tube_radii),
        "swept_tube_radii": list(pcfg.swept_tube_radii),
        "uncertain_tube_radii": list(pcfg.uncertain_tube_radii),
        "motion_config": asdict(pcfg.motion),
        "kta_config": asdict(pcfg.kta),
        "grid": asdict(pcfg.grid),
        "causal": True,
        "resolved_config": cfg,
    }

    idx = save_prepared_shards(out, gen(), shard_size=shard_size, metadata=prepared_meta)
    if produced_ids != expected_ids:
        missing = sorted(expected_ids - produced_ids)
        extra = sorted(produced_ids - expected_ids)
        raise RuntimeError(
            f"exact prepared set mismatch: missing={len(missing)} extra={len(extra)} "
            f"examples_missing={missing[:3]} examples_extra={extra[:3]}"
        )
    if int(idx["num_samples"]) != len(records):
        raise RuntimeError(
            f"saved {idx['num_samples']} prepared samples but MSP cache has {len(records)}"
        )
    save_resolved_config(cfg, out / "resolved_config.yaml")
    print(
        f"saved exact prepared: samples={idx['num_samples']} scenes={len(scenes)} "
        f"selection={meta.get('selection')} output={out}"
    )
    print("final occ cache:", source.load_occ3d.cache_info())
    print("final pose cache:", source.pose.cache_info())


if __name__ == "__main__":
    main()
