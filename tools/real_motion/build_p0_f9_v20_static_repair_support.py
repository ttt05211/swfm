#!/usr/bin/env python3
"""Build frozen-V18 full-grid support for V20 Static Repair v2.

This cache contains no future GT semantics.  It stores only the causal frozen
V18 free/occupied decision used to define where protected-add-only Static has
deployment authority.  Future GT is loaded later by the trainer from the same
semantic source used by formal evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from real_motion.runtime_config import (
    add_config_args,
    load_runtime_config,
    make_prepare_config,
)
from real_motion.strong_w2det import StrongW2DetConfig
from real_motion.v20_static_repair import (
    SUPPORT_CACHE_PROTOCOL,
    pack_v18_prediction,
    population_fingerprint,
)
from tools.real_motion import eval_p0_f9_v18_se2 as base
from tools.real_motion import eval_p0_f9_v18_full_validation as full
from tools.real_motion.benchmark_p0_f9_v18_runtime import (
    _forecast_once,
    _release_gpu_inputs,
    _stage_gpu_inputs,
)
from tools.real_motion.build_p0_f9_v20_history_cache import (
    PROTOCOL as STAGE1_PROTOCOL,
)
from tools.real_motion.eval_p0_f9_v17_local_stwm import window_from_record
from tools.real_motion.eval_p0_f9_v19_factorized_static_new_fov import (
    CachedSource,
    _ComponentLRU,
    _prepare_record_from_raw,
)
from tools.real_motion.train_p0_f9_v18_se2_clean import (
    PROTOCOL as CLEAN_PROTOCOL,
)


def _load_stage1_index(root):
    root = Path(root)
    idx = json.loads((root / "index.json").read_text(encoding="utf-8"))
    if idx.get("protocol") != STAGE1_PROTOCOL:
        raise RuntimeError(
            f"unexpected Stage1 protocol: {idx.get('protocol')}"
        )
    return root, idx


def _lean_raw(source, w):
    # Frozen V18 preparation needs only t-1/t0 semantics, their poses and the
    # six future ego poses.  No future semantic GT or lidar mask is read.
    toks = tuple(str(x) for x in w.history_tokens[-2:])
    return {
        "history_occ": np.stack([
            source.load_semantics(str(w.scene_name), tok) for tok in toks
        ]),
        "history_poses": [
            np.asarray(source.pose(tok), dtype=np.float64) for tok in toks
        ],
        "future_poses": [
            np.asarray(source.pose(str(tok)), dtype=np.float64)
            for tok in w.future_tokens
        ],
    }


def main():
    p = argparse.ArgumentParser()
    add_config_args(p)
    p.add_argument("--stage1-cache", required=True)
    p.add_argument("--v18-cache", required=True)
    p.add_argument("--base-checkpoint", required=True)
    p.add_argument("--dataroot", required=True)
    p.add_argument("--info-pkl", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-windows", type=int, default=0)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    pcfg = make_prepare_config(load_runtime_config(a.config, a.override))
    stage_root, stage_idx = _load_stage1_index(a.stage1_cache)
    _, v18_records = base.load_cache(a.v18_cache)

    v18_by_key = {}
    for rec in v18_records:
        key = (str(rec["scene_name"]), str(rec["t0_token"]))
        if key in v18_by_key:
            raise RuntimeError(f"duplicate V18 row: {key}")
        v18_by_key[key] = rec

    device = torch.device(
        a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    _, v18, _ = full._load_model(
        a.base_checkpoint, CLEAN_PROTOCOL, device
    )
    v18.eval()

    source = CachedSource(a.dataroot, info_pkl=a.info_pkl, verbose=False)
    strong_cfg = StrongW2DetConfig(free_label=int(pcfg.free_label))
    component_cache = _ComponentLRU(maxsize=1024)

    out = Path(a.output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing non-empty output dir: {out}")
    out.mkdir(parents=True, exist_ok=True)

    limit = int(a.max_windows)
    total_target = (
        min(int(stage_idx["num_windows"]), limit)
        if limit > 0 else int(stage_idx["num_windows"])
    )
    if total_target <= 0:
        raise RuntimeError("empty Stage1 population")

    shards = []
    identity_rows = []
    windows_done = 0
    free_counts = np.zeros(6, dtype=np.int64)
    voxel_counts = np.zeros(6, dtype=np.int64)
    packed_bytes = 0
    started = time.perf_counter()

    for si, shard_meta in enumerate(stage_idx["shards"]):
        if windows_done >= total_target:
            break
        obj = torch.load(
            stage_root / shard_meta["file"],
            map_location="cpu",
            weights_only=False,
        )
        if obj.get("protocol") != STAGE1_PROTOCOL:
            raise RuntimeError(f"bad Stage1 shard: {shard_meta['file']}")
        out_rows = []
        for srow in obj["rows"]:
            if windows_done >= total_target:
                break
            key = (str(srow["scene_name"]), str(srow["t0_token"]))
            rec = v18_by_key.get(key)
            if rec is None:
                raise RuntimeError(f"V18 cache missing Stage1 row {key}")
            w = window_from_record(rec)
            raw = _lean_raw(source, w)
            state = _prepare_record_from_raw(
                rec,
                raw,
                source,
                pcfg,
                strong_cfg,
                device,
                component_cache,
            )
            _stage_gpu_inputs(state, device)
            try:
                pred = np.asarray(
                    _forecast_once(v18, state, pcfg, strong_cfg, device),
                    dtype=np.uint8,
                )
            finally:
                _release_gpu_inputs(state)

            free = pred == int(pcfg.free_label)
            packed = pack_v18_prediction(
                pred, free_label=int(pcfg.free_label)
            )
            free_counts += free.reshape(6, -1).sum(axis=1)
            voxel_counts += free.reshape(6, -1).shape[1]
            packed_bytes += int(packed["v18_free_bits"].numel())
            packed_bytes += int(
                packed["v18_occupied_semantic_5bit"].numel()
            )
            row = {
                "scene_name": key[0],
                "t0_token": key[1],
                "future_tokens": tuple(str(x) for x in w.future_tokens),
                "native_shape_xyz": tuple(int(x) for x in free.shape[1:]),
                **packed,
            }
            out_rows.append(row)
            identity_rows.append(row)
            windows_done += 1

            if (
                windows_done == 1
                or windows_done % 25 == 0
                or windows_done == total_target
            ):
                elapsed = max(time.perf_counter() - started, 1e-9)
                print(
                    f"v20_static_repair_support "
                    f"{windows_done}/{total_target} "
                    f"rate={windows_done/elapsed:.3f} win/s",
                    flush=True,
                )

        if out_rows:
            name = f"shard_{len(shards):05d}.pt"
            torch.save(
                {"protocol": SUPPORT_CACHE_PROTOCOL, "rows": out_rows},
                out / name,
            )
            shards.append({
                "file": name,
                "count": len(out_rows),
                "source_stage1_shard": str(shard_meta["file"]),
                "bytes": int((out / name).stat().st_size),
            })

    if windows_done != total_target:
        raise RuntimeError(
            f"built {windows_done} windows, expected {total_target}"
        )

    index = {
        "protocol": SUPPORT_CACHE_PROTOCOL,
        "num_windows": int(windows_done),
        "num_scenes": int(len({str(x["scene_name"]) for x in identity_rows})),
        "population_fingerprint_sha256": population_fingerprint(identity_rows),
        "stage1_cache": str(Path(a.stage1_cache).resolve()),
        "stage1_protocol": STAGE1_PROTOCOL,
        "v18_cache": str(Path(a.v18_cache).resolve()),
        "base_checkpoint": str(Path(a.base_checkpoint).resolve()),
        "free_label": int(pcfg.free_label),
        "native_shape_xyz": list(identity_rows[0]["native_shape_xyz"]),
        "future_gt_used": False,
        "future_lidar_mask_used": False,
        "support_contract": (
            "formal_full_grid_positions_where_frozen_v18_prediction_is_free"
        ),
        "contains_lossless_v18_semantic_prediction": True,
        "v18_free_fraction_by_horizon": (
            free_counts / np.maximum(voxel_counts, 1)
        ).tolist(),
        "packed_support_bytes": int(packed_bytes),
        "mean_packed_support_bytes_per_window": float(
            packed_bytes / max(windows_done, 1)
        ),
        "shards": shards,
    }
    (out / "index.json").write_text(
        json.dumps(index, indent=2), encoding="utf-8"
    )
    print(json.dumps(index, indent=2))


if __name__ == "__main__":
    main()
