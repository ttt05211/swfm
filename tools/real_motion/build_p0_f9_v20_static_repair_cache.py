#!/usr/bin/env python3
"""Build compact full-grid V20 Static Repair v2 supervision.

This is a target/cache build, not model training. Frozen V18 is run once per
window to define the only locations where Static can affect protected add-only
composition. Full future semantic GT is then converted to the exact Repair-v2
target contract and compressed losslessly.

Output shards deliberately mirror the Stage-1 history-cache shard boundaries so
training can stream paired rows without loading either full cache into memory.
Interrupted builds are restartable with --resume; completed shards are verified
and reused.
"""
from __future__ import annotations

import argparse
import hashlib
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
    REPAIR_CACHE_PROTOCOL,
    pack_static_repair_supervision,
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


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            x = f.read(1 << 20)
            if not x:
                break
            h.update(x)
    return h.hexdigest()


def _population_hash_update(h, scene_name, t0_token):
    h.update(str(scene_name).encode("utf-8"))
    h.update(b"\0")
    h.update(str(t0_token).encode("utf-8"))
    h.update(b"\n")


def _atomic_torch_save(obj, path: Path):
    tmp = path.with_name("." + path.name + ".tmp")
    try:
        if tmp.exists():
            tmp.unlink()
        torch.save(obj, tmp)
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _minimal_raw(source, w):
    scene = str(w.scene_name)
    hist_tokens = tuple(str(x) for x in w.history_tokens)
    fut_tokens = tuple(str(x) for x in w.future_tokens)
    history_occ = np.stack([
        np.asarray(source.load_semantics(scene, tok), dtype=np.uint8)
        for tok in hist_tokens[-2:]
    ])
    history_poses = np.stack([
        np.asarray(source.pose(tok), dtype=np.float64)
        for tok in hist_tokens[-2:]
    ])
    future_gt = np.stack([
        np.asarray(source.load_semantics(scene, tok), dtype=np.uint8)
        for tok in fut_tokens
    ])
    future_poses = np.stack([
        np.asarray(source.pose(tok), dtype=np.float64)
        for tok in fut_tokens
    ])
    return {
        "history_occ": history_occ,
        "history_poses": history_poses,
        "future_gt_occ": future_gt,
        "future_poses": future_poses,
    }


def _verify_existing_shard(path, history_name, expected_count):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if obj.get("protocol") != REPAIR_CACHE_PROTOCOL:
        raise RuntimeError(f"bad existing Repair-v2 shard protocol: {path}")
    if str(obj.get("source_history_shard")) != str(history_name):
        raise RuntimeError(f"Repair-v2 resume history-shard mismatch: {path}")
    rows = obj.get("rows") or []
    if len(rows) != int(expected_count):
        raise RuntimeError(f"Repair-v2 resume count mismatch: {path}")
    return obj


def main():
    p = argparse.ArgumentParser()
    add_config_args(p)
    p.add_argument("--history-cache", required=True)
    p.add_argument("--source-v18-cache", required=True)
    p.add_argument("--v18-checkpoint", required=True)
    p.add_argument("--dataroot", required=True)
    p.add_argument("--info-pkl", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-windows", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    history_root = Path(a.history_cache)
    history_index_path = history_root / "index.json"
    hidx = json.loads(history_index_path.read_text(encoding="utf-8"))
    if hidx.get("protocol") != STAGE1_PROTOCOL:
        raise RuntimeError("Static Repair builder requires V20 Stage-1 history cache")

    _, source_records = base.load_cache(a.source_v18_cache)
    source_map = {}
    for rec in source_records:
        key = (str(rec["scene_name"]), str(rec["t0_token"]))
        if key in source_map:
            raise RuntimeError(f"duplicate source V18 identity: {key}")
        source_map[key] = rec

    max_windows = int(a.max_windows)
    if max_windows < 0:
        raise ValueError("--max-windows must be >= 0")

    pcfg = make_prepare_config(load_runtime_config(a.config, a.override))
    device = torch.device(
        a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    _, v18, _ = full._load_model(
        a.v18_checkpoint, CLEAN_PROTOCOL, device
    )
    v18.eval()
    source = CachedSource(a.dataroot, info_pkl=a.info_pkl, verbose=False)
    strong_cfg = StrongW2DetConfig(free_label=int(pcfg.free_label))
    component_cache = _ComponentLRU(maxsize=1024)

    out = Path(a.output_dir)
    if out.exists() and any(out.iterdir()) and not bool(a.resume):
        raise FileExistsError(
            f"refusing non-empty output dir without --resume: {out}"
        )
    out.mkdir(parents=True, exist_ok=True)

    stats = {
        "windows": 0,
        "native_voxels": 0,
        "v18_occupied": 0,
        "support": 0,
        "static_positive": 0,
        "free_noadd": 0,
        "dynamic_noadd": 0,
    }
    shards = []
    pop = hashlib.sha256()
    started = time.perf_counter()
    processed = 0

    for si, hs in enumerate(hidx["shards"]):
        if max_windows and processed >= max_windows:
            break
        history_name = str(hs["file"])
        hobj = torch.load(
            history_root / history_name,
            map_location="cpu",
            weights_only=False,
        )
        if hobj.get("protocol") != STAGE1_PROTOCOL:
            raise RuntimeError(f"bad Stage-1 shard: {history_name}")
        hrows = list(hobj["rows"])
        if max_windows:
            hrows = hrows[: max(0, max_windows - processed)]
        if not hrows:
            break

        repair_name = f"shard_{si:05d}.pt"
        repair_path = out / repair_name
        if repair_path.exists() and bool(a.resume):
            existing = _verify_existing_shard(
                repair_path, history_name, len(hrows)
            )
            rows = list(existing["rows"])
            shard_stats = dict(existing.get("stats") or {})
        else:
            rows = []
            shard_stats = {
                "windows": 0,
                "native_voxels": 0,
                "v18_occupied": 0,
                "support": 0,
                "static_positive": 0,
                "free_noadd": 0,
                "dynamic_noadd": 0,
            }
            for hrow in hrows:
                key = (str(hrow["scene_name"]), str(hrow["t0_token"]))
                rec = source_map.get(key)
                if rec is None:
                    raise RuntimeError(
                        f"Stage-1 row missing from source V18 cache: {key}"
                    )
                w = window_from_record(rec)
                raw = _minimal_raw(source, w)
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
                        _forecast_once(
                            v18, state, pcfg, strong_cfg, device
                        ),
                        dtype=np.uint8,
                    )
                finally:
                    _release_gpu_inputs(state)
                gt = np.asarray(raw["future_gt_occ"], dtype=np.uint8)
                packed = pack_static_repair_supervision(
                    pred, gt, free_label=int(pcfg.free_label)
                )
                row = {
                    "scene_name": key[0],
                    "t0_token": key[1],
                    **packed,
                }
                rows.append(row)
                shard_stats["windows"] += 1
                shard_stats["native_voxels"] += int(gt.size)
                shard_stats["v18_occupied"] += int(
                    packed["v18_occupied_count"]
                )
                shard_stats["support"] += int(packed["support_count"])
                shard_stats["static_positive"] += int(
                    packed["static_positive_count"]
                )
                shard_stats["free_noadd"] += int(
                    packed["free_noadd_count"]
                )
                shard_stats["dynamic_noadd"] += int(
                    packed["dynamic_noadd_count"]
                )

            payload = {
                "protocol": REPAIR_CACHE_PROTOCOL,
                "source_history_shard": history_name,
                "rows": rows,
                "stats": shard_stats,
            }
            _atomic_torch_save(payload, repair_path)

        if len(rows) != len(hrows):
            raise RuntimeError("Repair-v2/history row count mismatch")
        for hrow, rrow in zip(hrows, rows):
            hk = (str(hrow["scene_name"]), str(hrow["t0_token"]))
            rk = (str(rrow["scene_name"]), str(rrow["t0_token"]))
            if hk != rk:
                raise RuntimeError(
                    f"Repair-v2/history ordered identity mismatch: {hk} != {rk}"
                )
            _population_hash_update(pop, *hk)

        for k in stats:
            stats[k] += int(shard_stats.get(k, 0))
        nbytes = int(repair_path.stat().st_size)
        shards.append({
            "file": repair_name,
            "count": len(rows),
            "bytes": nbytes,
            "source_history_shard": history_name,
        })
        processed += len(rows)
        elapsed = max(time.perf_counter() - started, 1e-9)
        if si == 0 or (si + 1) % 25 == 0 or processed == min(
            int(hidx["num_windows"]),
            max_windows if max_windows else int(hidx["num_windows"]),
        ):
            print(
                f"v20_static_repair_cache {processed}/"
                f"{min(int(hidx['num_windows']), max_windows if max_windows else int(hidx['num_windows']))} "
                f"rate={processed/elapsed:.3f} win/s",
                flush=True,
            )

    if processed == 0:
        raise RuntimeError("Static Repair cache build selected zero windows")

    scene_names = set()
    # Scene names are already present in Stage-1 metadata; for max-window smoke
    # derive exact names from repair rows without loading all payloads at once.
    for sh in shards:
        obj = torch.load(out / sh["file"], map_location="cpu", weights_only=False)
        scene_names.update(str(r["scene_name"]) for r in obj["rows"])

    index = {
        "protocol": REPAIR_CACHE_PROTOCOL,
        "num_windows": int(processed),
        "num_scenes": int(len(scene_names)),
        "scene_names": sorted(scene_names),
        "population_fingerprint_sha256": pop.hexdigest(),
        "source_history_cache": str(history_root.resolve()),
        "source_history_index_sha256": _sha256_file(history_index_path),
        "source_v18_cache": str(Path(a.source_v18_cache).resolve()),
        "v18_checkpoint": str(Path(a.v18_checkpoint).resolve()),
        "formal_supervision_domain": "full_native_future_grid",
        "composition_support": "frozen_v18_prediction_equals_free",
        "static_positive_target": "original_static_semantic",
        "free_target": "free_no_add",
        "dynamic_target": "free_no_add",
        "future_lidar_mask_used_for_target_validity": False,
        "horizon_conflicts": "retained_as_independent_future_voxel_contributions",
        "class_weighting": "none",
        "native_grid": dict(hidx["native_grid"]),
        "highres_lattice": dict(hidx["highres_lattice"]),
        "coarse_lattice": dict(hidx["coarse_lattice"]),
        "stats": stats,
        "shards": shards,
    }
    (out / "index.json").write_text(
        json.dumps(index, indent=2), encoding="utf-8"
    )
    print("=== V20 STATIC REPAIR V2 CACHE COMPLETE ===")
    print(json.dumps({
        "output": str(out.resolve()),
        "windows": processed,
        "scenes": len(scene_names),
        "support_fraction": float(
            stats["support"] / max(stats["native_voxels"], 1)
        ),
        "static_positive_fraction_of_support": float(
            stats["static_positive"] / max(stats["support"], 1)
        ),
        "bytes": int(sum(int(x["bytes"]) for x in shards)),
    }, indent=2))


if __name__ == "__main__":
    main()
