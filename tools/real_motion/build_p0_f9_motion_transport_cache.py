#!/usr/bin/env python3
"""Build Strong-source learned-motion training/validation caches.

The window population is taken verbatim from an existing MSP cache so train and
scene-disjoint validation splits stay frozen.  For each window we reconstruct
Strong-W2Det dynamic components from the six causal history occupancies, build a
six-frame occupancy-only backward track, and attach future center/existence
labels by nuScenes instance identity *only for supervision*.

No VAE, OccFM, future occupancy tensor, or world-model cache is involved.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from real_motion.cache_pipeline import bounded_ordered_parallel_map
from real_motion.motion_transport import (
    FEATURE_DIM,
    FEATURE_NAMES,
    FUTURE_FRAMES,
    HISTORY_FRAMES,
    MOTION_TRANSPORT_CACHE_VERSION,
    annotation_map,
    backward_component_tracks,
    build_motion_targets,
    build_source_features,
    dynamic_annotations,
    match_sources_to_annotations,
)
from real_motion.msp import MSP_CACHE_VERSION
from real_motion.nuscenes_adapter import NuScenesWindowSource, WindowTokens
from real_motion.runtime_config import (
    add_config_args,
    config_fingerprint,
    load_runtime_config,
    make_prepare_config,
    save_resolved_config,
)
from real_motion.strong_w2det import StrongW2DetConfig, extract_instances, match_instances


class CachedSource(NuScenesWindowSource):
    @lru_cache(maxsize=1024)
    def load_semantics(self, scene_name, token):
        return super().load_semantics(scene_name, token)

    @lru_cache(maxsize=4096)
    def pose(self, token):
        return super().pose(token)


def load_window_records(path: str):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if obj.get("version") != MSP_CACHE_VERSION:
        raise RuntimeError("window cache must be an MSP probe cache")
    meta = obj.get("metadata") or {}
    records = obj.get("records") or []
    if not records:
        raise RuntimeError("window cache contains no records")
    ids = [str(r["sample_id"]) for r in records]
    if len(set(ids)) != len(ids):
        raise RuntimeError("window cache contains duplicate sample IDs")
    return meta, records


def window_from_record(rec) -> WindowTokens:
    return WindowTokens(
        scene_name=str(rec["scene_name"]),
        history_tokens=tuple(str(x) for x in rec["history_tokens"]),
        t0_token=str(rec["t0_token"]),
        future_tokens=tuple(str(x) for x in rec["future_tokens"]),
    )


def build_one(source, rec, pcfg, *, match_max_distance_m: float):
    w = window_from_record(rec)
    history_occ = [source.load_semantics(w.scene_name, tok) for tok in w.history_tokens]
    history_poses = [np.asarray(source.pose(tok), dtype=np.float64) for tok in w.history_tokens]
    strong_cfg = StrongW2DetConfig(free_label=int(pcfg.free_label))
    components_by_frame = [
        extract_instances(sem, pose, grid=pcfg.grid, cfg=strong_cfg)
        for sem, pose in zip(history_occ, history_poses)
    ]
    current = components_by_frame[-1]
    previous = components_by_frame[-2]
    velocities = match_instances(
        previous, current, float(pcfg.frame_dt_s), max_speed_mps=strong_cfg.max_match_speed_mps
    )
    tracks, track_valid = backward_component_tracks(
        components_by_frame,
        frame_dt_s=float(pcfg.frame_dt_s),
        max_speed_mps=float(strong_cfg.max_match_speed_mps),
    )
    features = build_source_features(
        current, velocities, tracks, track_valid, history_poses[-1],
        frame_dt_s=float(pcfg.frame_dt_s), grid=pcfg.grid,
    )

    anns0 = dynamic_annotations(source.nusc, w.t0_token)
    source_tokens = match_sources_to_annotations(
        current, anns0, max_distance_m=float(match_max_distance_m)
    )
    future_maps = [annotation_map(source.nusc, tok) for tok in w.future_tokens]
    targets = build_motion_targets(
        current, velocities, source_tokens, future_maps, history_poses[-1],
        frame_dt_s=float(pcfg.frame_dt_s),
    )
    class_ids = np.asarray([int(c["class_id"]) for c in current], dtype=np.int64)
    voxel_counts = np.asarray([int(c.get("voxel_count", len(c["voxel_indices"]))) for c in current], dtype=np.int64)
    return {
        "sample_id": str(rec["sample_id"]),
        "scene_name": str(rec["scene_name"]),
        "history_tokens": tuple(w.history_tokens),
        "t0_token": str(w.t0_token),
        "future_tokens": tuple(w.future_tokens),
        "features": torch.from_numpy(features),
        "anchors_xy_t0_m": torch.from_numpy(targets["anchors_xy_t0_m"]),
        "target_xy_t0_m": torch.from_numpy(targets["target_xy_t0_m"]),
        "target_residual_xy_m": torch.from_numpy(targets["target_residual_xy_m"]),
        "existence": torch.from_numpy(targets["existence"]),
        "target_valid": torch.from_numpy(targets["target_valid"]),
        "supervised_source": torch.from_numpy(targets["supervised_source"]),
        "source_class_id": torch.from_numpy(class_ids),
        "source_voxel_count": torch.from_numpy(voxel_counts),
        "source_instance_token": tuple(source_tokens),
        "track_valid": torch.from_numpy(track_valid),
        "num_sources": int(len(current)),
        "num_supervised_sources": int(np.asarray(targets["supervised_source"]).sum()),
        "num_kta_velocity_sources": int(len(velocities)),
    }


def summarize(records):
    nsrc = sum(int(r["num_sources"]) for r in records)
    nsup = sum(int(r["num_supervised_sources"]) for r in records)
    nkta = sum(int(r["num_kta_velocity_sources"]) for r in records)
    future_labels = sum(int(r["target_valid"].sum()) for r in records)
    exist_labels = sum(int(r["supervised_source"].sum()) * FUTURE_FRAMES for r in records)
    full_track = sum(int(r["track_valid"].all(dim=1).sum()) for r in records)
    return {
        "num_windows": len(records),
        "num_sources": nsrc,
        "num_supervised_sources": nsup,
        "source_gt_match_fraction": nsup / max(nsrc, 1),
        "num_kta_velocity_sources": nkta,
        "kta_velocity_match_fraction": nkta / max(nsrc, 1),
        "num_future_center_labels": future_labels,
        "num_existence_labels": exist_labels,
        "full_6frame_track_fraction": full_track / max(nsrc, 1),
    }


def main():
    p = argparse.ArgumentParser()
    add_config_args(p)
    p.add_argument("--window-cache", required=True,
                   help="frozen MSP train-full or scene-disjoint val cache used only for window identity")
    p.add_argument("--dataroot", required=True)
    p.add_argument("--info-pkl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--match-max-distance-m", type=float, default=4.0)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--prefetch-windows", type=int, default=0)
    p.add_argument("--max-windows", type=int, default=0)
    a = p.parse_args()

    cfg = load_runtime_config(a.config, a.override)
    pcfg = make_prepare_config(cfg)
    if int(pcfg.history_frames) != HISTORY_FRAMES or int(pcfg.future_frames) != FUTURE_FRAMES:
        raise RuntimeError("motion transport v1 requires the frozen 6-history + 6-future contract")
    meta, base_records = load_window_records(a.window_cache)
    expected = meta.get("config_contract_sha256")
    got = config_fingerprint(cfg, "cache")
    if expected and expected != got:
        raise RuntimeError("runtime config differs from window-cache contract")
    if int(a.max_windows) > 0:
        base_records = base_records[: min(len(base_records), int(a.max_windows))]

    workers = int(a.workers) if int(a.workers) > 0 else min(8, max(1, os.cpu_count() or 1))
    prefetch = int(a.prefetch_windows) if int(a.prefetch_windows) > 0 else 4 * workers
    source = CachedSource(a.dataroot, info_pkl=a.info_pkl, verbose=False)

    def fn(rec):
        return build_one(source, rec, pcfg, match_max_distance_m=float(a.match_max_distance_m))

    print(json.dumps({
        "windows": len(base_records), "workers": workers, "prefetch_windows": prefetch,
        "feature_dim": FEATURE_DIM,
    }))
    started = time.perf_counter()
    records = []
    for i, out in enumerate(bounded_ordered_parallel_map(
        fn, base_records, max_workers=workers, max_in_flight=prefetch,
        thread_name_prefix="motion-transport",
    ), start=1):
        records.append(out)
        if i == 1 or i % 50 == 0 or i == len(base_records):
            elapsed = max(time.perf_counter() - started, 1e-9)
            print(
                f"motion_transport_cache {i}/{len(base_records)} rate={i/elapsed:.2f} win/s "
                f"occ_cache={source.load_semantics.cache_info()} pose_cache={source.pose.cache_info()}"
            )

    summary = summarize(records)
    metadata = {
        "version": MOTION_TRANSPORT_CACHE_VERSION,
        "window_cache": str(Path(a.window_cache).resolve()),
        "window_cache_version": MSP_CACHE_VERSION,
        "config_contract_sha256": got,
        "feature_dim": FEATURE_DIM,
        "feature_names": list(FEATURE_NAMES),
        "feature_contract": "strong_source_6frame_backward_occ_only_v1",
        "target_contract": "gt_center_and_existence_training_only_kta_residual_v1",
        "match_max_distance_m": float(a.match_max_distance_m),
        "frame_dt_s": float(pcfg.frame_dt_s),
        "summary": summary,
        "resolved_config": cfg,
    }
    payload = {"version": MOTION_TRANSPORT_CACHE_VERSION, "metadata": metadata, "records": records}
    op = Path(a.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, op)
    op.with_suffix(".summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    save_resolved_config(cfg, op.with_suffix(".resolved.yaml"))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
