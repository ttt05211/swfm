#!/usr/bin/env python3
"""Build a bounded scene cache for the V17-RL C experiment.

The cache is intentionally a *short-experiment* cache, not a new benchmark.  A
scene-balanced subset of the existing V17 training windows is selected without
using any future GT content.  For each selected window we materialize only what
is expensive to reconstruct repeatedly during C training:

- exact future semantic occupancy (training supervision only),
- exact Strong-W2Det/KTA future anchor,
- original-Strong-order t0 source voxel indices,
- t0/future ego poses.

The V17 model inputs/trajectory targets remain in the existing V17 cache and are
not duplicated here.  No GT moving mask or future annotation identity is used.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from real_motion.local_stwm_scene_supervision import (
    SCENE_CACHE_VERSION,
    SCENE_LOSS_CONTRACT,
    SCENE_QUERY_CONTRACT,
)
from real_motion.nuscenes_adapter import NuScenesWindowSource, WindowTokens
from real_motion.runtime_config import add_config_args, load_runtime_config, make_prepare_config
from real_motion.strong_w2det import StrongW2DetConfig, extract_instances, strong_w2det_sequence
from tools.real_motion.train_p0_f9_v17_local_stwm import load_cache

SELECTION_CONTRACT = "scene_round_robin_v17_train_record_order_no_future_gt_v1"


def window_from_record(r) -> WindowTokens:
    return WindowTokens(
        scene_name=str(r["scene_name"]),
        history_tokens=tuple(str(x) for x in r["history_tokens"]),
        t0_token=str(r["t0_token"]),
        future_tokens=tuple(str(x) for x in r["future_tokens"]),
    )


def scene_balanced_indices(records, max_windows: int) -> list[int]:
    if int(max_windows) <= 0:
        return list(range(len(records)))
    by_scene: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        by_scene[str(r["scene_name"])].append(i)
    scenes = sorted(by_scene)
    ptr = {s: 0 for s in scenes}
    out: list[int] = []
    while len(out) < min(int(max_windows), len(records)):
        added = False
        for s in scenes:
            j = ptr[s]
            if j >= len(by_scene[s]):
                continue
            out.append(by_scene[s][j])
            ptr[s] = j + 1
            added = True
            if len(out) >= int(max_windows):
                break
        if not added:
            break
    return out


def _ids_digest(rows) -> str:
    h = hashlib.sha256()
    for r in rows:
        h.update(str(r["sample_id"]).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    add_config_args(p)
    p.add_argument("--v17-train-cache", required=True)
    p.add_argument("--dataroot", required=True)
    p.add_argument("--info-pkl", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-windows", type=int, default=1024)
    p.add_argument("--shard-size", type=int, default=8)
    a = p.parse_args()
    if a.max_windows == 0 or a.max_windows < -1 or a.shard_size <= 0:
        raise ValueError("max-windows must be positive or -1; shard-size must be positive")

    cfg = load_runtime_config(a.config, a.override)
    pcfg = make_prepare_config(cfg)
    meta, records = load_cache(a.v17_train_cache)
    selected_idx = scene_balanced_indices(records, len(records) if a.max_windows < 0 else a.max_windows)
    selected = [records[i] for i in selected_idx]
    if not selected:
        raise RuntimeError("scene cache selection is empty")
    sample_ids = [str(r["sample_id"]) for r in selected]
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("selected V17 sample ids are not unique")

    source = NuScenesWindowSource(a.dataroot, info_pkl=a.info_pkl, verbose=False)
    strong_cfg = StrongW2DetConfig(free_label=int(pcfg.free_label))
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Refuse accidental append into an unrelated cache.
    index_path = out / "index.json"
    if index_path.exists():
        raise FileExistsError(f"output already contains index.json: {index_path}")

    entries = []
    shard_rows = []
    shard_id = 0
    total_sources = 0

    def flush():
        nonlocal shard_rows, shard_id
        if not shard_rows:
            return
        name = f"shard_{shard_id:05d}.pt"
        torch.save(shard_rows, out / name)
        for j, row in enumerate(shard_rows):
            entries.append({
                "sample_id": str(row["sample_id"]),
                "scene_name": str(row["scene_name"]),
                "shard": name,
                "index": int(j),
                "source_count": int(len(row["source_voxel_indices_t0"])),
            })
        shard_rows = []
        shard_id += 1

    for k, rec in enumerate(selected, start=1):
        w = window_from_record(rec)
        # Strong only needs the final two causal occupancy frames.
        prev = source.load_semantics(w.scene_name, w.history_tokens[-2])
        cur = source.load_semantics(w.scene_name, w.history_tokens[-1])
        prev_pose = np.asarray(source.pose(w.history_tokens[-2]), dtype=np.float64)
        t0_pose = np.asarray(source.pose(w.history_tokens[-1]), dtype=np.float64)
        future_poses = np.stack(
            [np.asarray(source.pose(tok), dtype=np.float64) for tok in w.future_tokens], axis=0
        )
        current = extract_instances(cur, t0_pose, grid=pcfg.grid, cfg=strong_cfg)
        if len(current) != int(rec["features"].shape[0]):
            raise RuntimeError(
                f"{rec['sample_id']}: Strong source count {len(current)} != V17 {rec['features'].shape[0]}"
            )
        got_cls = [int(x["class_id"]) for x in current]
        expected_cls = [int(x) for x in rec["source_class_id"].tolist()]
        if got_cls != expected_cls:
            raise RuntimeError(f"{rec['sample_id']}: Strong source ordering/class mismatch")

        anchor = strong_w2det_sequence(
            np.stack([prev, cur], axis=0),
            [prev_pose, t0_pose],
            list(future_poses),
            frame_dt_s=float(pcfg.frame_dt_s),
            grid=pcfg.grid,
            cfg=strong_cfg,
        )
        gt = np.stack(
            [source.load_semantics(w.scene_name, tok) for tok in w.future_tokens], axis=0
        )
        expected_shape = (6, *tuple(pcfg.grid.shape_hwd))
        if tuple(anchor.shape) != expected_shape or tuple(gt.shape) != expected_shape:
            raise RuntimeError(f"{rec['sample_id']}: future occupancy shape mismatch")
        if int(gt.min()) < 0 or int(gt.max()) >= 18 or int(anchor.min()) < 0 or int(anchor.max()) >= 18:
            raise RuntimeError(f"{rec['sample_id']}: semantic label outside [0,17]")

        voxels = [
            torch.as_tensor(np.asarray(x["voxel_indices"], dtype=np.int16), dtype=torch.int16)
            for x in current
        ]
        row = {
            "sample_id": str(rec["sample_id"]),
            "scene_name": str(rec["scene_name"]),
            "t0_token": str(rec["t0_token"]),
            "future_tokens": tuple(str(x) for x in rec["future_tokens"]),
            "future_gt_occ": torch.as_tensor(gt, dtype=torch.uint8),
            "strong_anchor_occ": torch.as_tensor(anchor, dtype=torch.uint8),
            "source_voxel_indices_t0": voxels,
            "source_class_id": rec["source_class_id"].long().clone(),
            "t0_ego_to_world": torch.as_tensor(t0_pose, dtype=torch.float64),
            "future_ego_to_world": torch.as_tensor(future_poses, dtype=torch.float64),
        }
        shard_rows.append(row)
        total_sources += len(current)
        if len(shard_rows) >= int(a.shard_size):
            flush()
        if k == 1 or k % 16 == 0 or k == len(selected):
            print(
                f"scene_cache {k}/{len(selected)} sid={rec['sample_id']} "
                f"sources={len(current)} total_sources={total_sources}",
                flush=True,
            )
    flush()

    selection_scenes = sorted({str(r["scene_name"]) for r in selected})
    index = {
        "version": SCENE_CACHE_VERSION,
        "metadata": {
            "scene_loss_contract": SCENE_LOSS_CONTRACT,
            "scene_query_contract": SCENE_QUERY_CONTRACT,
            "selection_contract": SELECTION_CONTRACT,
            "selection_sample_ids_sha256": _ids_digest(selected),
            "v17_cache_version": meta.get("representation_contract"),
            "source_v17_train_cache": str(Path(a.v17_train_cache).resolve()),
            "num_windows": len(selected),
            "num_scenes": len(selection_scenes),
            "num_sources": int(total_sources),
            "scene_names": selection_scenes,
            "grid_shape_hwd": [int(x) for x in pcfg.grid.shape_hwd],
            "grid_voxel_size": [float(x) for x in pcfg.grid.voxel_size],
            "free_label": int(pcfg.free_label),
            "frame_dt_s": float(pcfg.frame_dt_s),
            "future_gt_used_only_as_training_target": True,
            "gt_moving_filter_used": False,
        },
        "num_samples": len(entries),
        "entries": entries,
    }
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    print("=== V17 SCENE SUPERVISION CACHE ===")
    print(json.dumps(index["metadata"], indent=2))
    print(f"saved {index_path}")


if __name__ == "__main__":
    main()
