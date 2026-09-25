#!/usr/bin/env python3
"""Build a label-only Stage-0 sidecar for frozen V19 Factorized caches.

The parent V19 cache remains the sole source of inference tensors.  This tool
writes only per-Z future semantic supervision and identity metadata, so future
GT cannot travel through the model-input dictionary by construction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from real_motion.metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS
from real_motion.prepared import load_nuscenes_window_raw
from real_motion.runtime_config import add_config_args, load_runtime_config, make_prepare_config
from real_motion.v19_innovation_training import unpack_vertical_occupancy_torch
from real_motion.v20_stage0_voxel_semantic import LABEL_SIDECAR_PROTOCOL
from tools.real_motion import eval_p0_f9_v18_se2 as base
from tools.real_motion.diagnose_p0_f9_v19_innovation_decomposition import CachedSource
from tools.real_motion.eval_p0_f9_v17_local_stwm import window_from_record
from tools.real_motion.build_p0_f9_v19_factorized_static_new_fov_cache import (
    PROTOCOL as V19_CACHE_PROTOCOL,
)

PROTOCOL = LABEL_SIDECAR_PROTOCOL
IGNORE_LABEL = 255
DYNAMIC_IDS = tuple(int(x) for x in DYNAMIC_CLASS_IDS)


def _record_map(source_cache: str):
    _, records = base.load_cache(source_cache)
    out = {}
    for rec in records:
        w = window_from_record(rec)
        key = (str(w.scene_name), str(w.t0_token))
        if key in out:
            raise RuntimeError(f"duplicate scene/t0 pair in source cache: {key}")
        out[key] = (rec, w)
    return out


def main():
    p = argparse.ArgumentParser()
    add_config_args(p)
    p.add_argument("--v19-cache", required=True)
    p.add_argument("--source-cache", default="")
    p.add_argument("--dataroot", required=True)
    p.add_argument("--info-pkl", required=True)
    p.add_argument("--output-dir", required=True)
    a = p.parse_args()

    src_root = Path(a.v19_cache)
    idx = json.loads((src_root / "index.json").read_text(encoding="utf-8"))
    if idx.get("protocol") != V19_CACHE_PROTOCOL:
        raise RuntimeError(f"unexpected V19 cache protocol: {idx.get('protocol')}")
    source_cache = str(a.source_cache or idx.get("source_cache", ""))
    if not source_cache:
        raise RuntimeError("source cache is required")

    out_dir = Path(a.output_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty output dir: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    pcfg = make_prepare_config(load_runtime_config(a.config, a.override))
    if list(pcfg.grid.shape_hwd) != list(idx["grid_shape_hwd"]):
        raise RuntimeError("runtime grid differs from frozen V19 cache")
    z = int(pcfg.grid.shape_hwd[2])
    free_label = int(pcfg.free_label)
    records = _record_map(source_cache)
    source = CachedSource(a.dataroot, info_pkl=a.info_pkl, verbose=False)

    out_shards = []
    class_hist = {str(i): 0 for i in range(17)}
    supervised_voxels = 0
    sidecar_bytes = 0
    for shard_id, row in enumerate(idx["shards"]):
        parent_name = str(row["file"])
        obj = torch.load(
            src_root / parent_name,
            map_location="cpu",
            weights_only=False,
        )
        if obj.get("protocol") != V19_CACHE_PROTOCOL:
            raise RuntimeError(f"bad shard protocol: {parent_name}")
        n = int(row["count"])
        if len(obj["scene_name"]) != n or len(obj["t0_token"]) != n:
            raise RuntimeError(f"malformed V19 shard identities: {parent_name}")

        voxel_targets = []
        for bi in range(n):
            scene = str(obj["scene_name"][bi])
            token = str(obj["t0_token"][bi])
            key = (scene, token)
            if key not in records:
                raise KeyError(f"scene/t0 pair not in source cache: {key}")
            _, w = records[key]
            if str(w.scene_name) != scene or str(w.t0_token) != token:
                raise RuntimeError(f"source-cache identity mismatch: {key}")

            raw = load_nuscenes_window_raw(source, w, pcfg, include_gt=True)
            gt = np.asarray(raw["future_gt_occ"], dtype=np.uint8)
            if gt.shape != (6,) + tuple(pcfg.grid.shape_hwd):
                raise RuntimeError(f"future GT shape mismatch for {key}: {gt.shape}")

            # vertical_target_bits is GT-derived supervision from the frozen V19
            # cache.  It is used only to choose which semantic labels are valid;
            # it is never copied to the sidecar and never becomes a model input.
            target_bits = obj["vertical_target_bits"][bi : bi + 1]
            target = unpack_vertical_occupancy_torch(target_bits, z)[0]
            target = target.permute(0, 2, 3, 1).bool().numpy()
            sem = np.full(gt.shape, IGNORE_LABEL, dtype=np.uint8)
            sem[target] = gt[target]
            bad_free = target & (gt == free_label)
            bad_dyn = target & np.isin(gt, np.asarray(DYNAMIC_IDS, dtype=np.uint8))
            if bool(bad_free.any()) or bool(bad_dyn.any()):
                raise RuntimeError(
                    f"Stage-0 target contract violated for {key}: "
                    f"free={int(bad_free.sum())} dynamic={int(bad_dyn.sum())}"
                )
            for cid in range(17):
                class_hist[str(cid)] += int((target & (gt == cid)).sum())
            supervised_voxels += int(target.sum())
            voxel_targets.append(torch.from_numpy(sem))

        payload = {
            "protocol": PROTOCOL,
            "parent_v19_shard": parent_name,
            "count": n,
            "voxel_semantic_target": torch.stack(voxel_targets),
            "scene_name": list(obj["scene_name"]),
            "t0_token": list(obj["t0_token"]),
        }
        name = f"shard_{shard_id:05d}.pt"
        torch.save(payload, out_dir / name)
        nbytes = int((out_dir / name).stat().st_size)
        sidecar_bytes += nbytes
        out_shards.append(
            {
                "file": name,
                "count": n,
                "parent_v19_shard": parent_name,
                "bytes": nbytes,
            }
        )

    out_idx = {
        "protocol": PROTOCOL,
        "parent_v19_cache": str(src_root.resolve()),
        "parent_v19_protocol": V19_CACHE_PROTOCOL,
        "source_cache": str(Path(source_cache).resolve()),
        "num_windows": int(idx["num_windows"]),
        "num_scenes": int(idx.get("num_scenes", len(set(idx.get("scene_names", []))))),
        "scene_names": list(idx.get("scene_names", [])),
        "grid_shape_hwd": [int(x) for x in idx["grid_shape_hwd"]],
        "free_label": int(idx.get("free_label", free_label)),
        "anchor_distance_max_m": float(idx.get("anchor_distance_max_m", 40.0)),
        "future_gt_semantic_is_supervision_only": True,
        "label_only_sidecar": True,
        "model_input_tensors_copied_from_v19": False,
        "pairing_contract": "shard + scene_name + t0_token exact order",
        "stage0_frozen_geometry_contract": (
            "presence/vertical support is recomputed from the frozen V19 "
            "checkpoint during train/eval; GT vertical_target is label-only"
        ),
        "voxel_semantic_ignore_label": IGNORE_LABEL,
        "voxel_semantic_supervised_voxels": int(supervised_voxels),
        "voxel_semantic_class_histogram": class_hist,
        "sidecar_bytes": int(sidecar_bytes),
        "shards": out_shards,
    }
    (out_dir / "index.json").write_text(
        json.dumps(out_idx, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "protocol": PROTOCOL,
                "num_windows": int(idx["num_windows"]),
                "supervised_voxels": int(supervised_voxels),
                "sidecar_bytes": int(sidecar_bytes),
                "class_histogram": class_hist,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
