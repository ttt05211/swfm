#!/usr/bin/env python3
"""Build an MSP probe cache over every eligible train window.

This is the full-data counterpart of ``p0_msp_build_dataset.py``.  It preserves
that builder's exact 6+6 window, feature, target, matching, config and seed
contracts, but deliberately removes the 1024/4096 development cap.  The output
contains every eligible chronological window in the supplied train split.

The frozen MSP checkpoint is *not* used here; this cache stores the causal MSP
features/GT training labels that are later routed by the fixed MSP head when the
P0-F9 native cache is built.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from real_motion.cache_pipeline import bounded_ordered_parallel_map
from real_motion.msp import (
    FEATURE_DIM,
    FEATURE_NAMES,
    MSP_CACHE_VERSION,
    build_probe_record,
    validate_probe_record,
)
from real_motion.runtime_config import (
    add_config_args,
    config_fingerprint,
    load_runtime_config,
    make_prepare_config,
    save_resolved_config,
)
from tools.real_motion import p0_msp_build_dataset as base


PROTOCOL = "p0_f9_full_all_eligible_msp_probe_v1"
SELECTION = "scene_balanced_round_robin_all_eligible_v1"


def main() -> None:
    p = argparse.ArgumentParser()
    add_config_args(p)
    p.add_argument("--dataroot", required=True)
    p.add_argument("--info-pkl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--seed", type=int, default=base.DEFAULT_SEED)
    p.add_argument("--match-max-distance-m", type=float, default=4.0)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--prefetch-windows", type=int, default=0)
    p.add_argument(
        "--reuse-cache",
        default=None,
        help="optional compatible smaller train MSP cache, e.g. the existing 4096 cache",
    )
    a = p.parse_args()

    if a.stride <= 0:
        raise ValueError("stride must be positive")
    output = Path(a.output).expanduser().resolve()
    if output.exists():
        raise RuntimeError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    cfg = load_runtime_config(a.config, a.override)
    pcfg = make_prepare_config(cfg)
    if int(pcfg.history_frames) != 6 or int(pcfg.future_frames) != 6:
        raise RuntimeError("formal full-data P0-F9 requires the frozen 6+6 window contract")

    workers = int(a.workers) if int(a.workers) > 0 else min(
        16, max(1, os.cpu_count() or 1)
    )
    prefetch = int(a.prefetch_windows) if int(a.prefetch_windows) > 0 else 4 * workers

    source = base.CachedNuScenesWindowSource(a.dataroot, info_pkl=a.info_pkl, verbose=False)
    # max_windows=None is intentional here: this is the entire eligible train population.
    windows = base.select_windows(
        source,
        mode="train",
        history=pcfg.history_frames,
        future=pcfg.future_frames,
        stride=int(a.stride),
        max_windows=None,
        seed=int(a.seed),
    )
    if not windows:
        raise RuntimeError("train split contains no eligible 6+6 windows")

    # Independent count through the native chronological iterator.  The set must
    # match exactly; otherwise a future change to either selector is caught here.
    native_ids = {
        f"{w.scene_name}:{w.t0_token}"
        for w in source.iter_windows(
            history=pcfg.history_frames,
            future=pcfg.future_frames,
            stride=int(a.stride),
            max_windows=None,
        )
    }
    selected_ids = [base._window_id(w) for w in windows]
    if len(selected_ids) != len(set(selected_ids)):
        raise RuntimeError("full MSP selection contains duplicate window IDs")
    if set(selected_ids) != native_ids:
        missing = sorted(native_ids - set(selected_ids))
        extra = sorted(set(selected_ids) - native_ids)
        raise RuntimeError(
            "full MSP selection differs from native eligible population: "
            f"missing={missing[:3]} extra={extra[:3]}"
        )

    contract_sha = config_fingerprint(cfg, "cache")
    reuse = base._load_reuse_records(
        a.reuse_cache,
        expected={
            "mode": "train",
            "seed": int(a.seed),
            "stride": int(a.stride),
            "match_max_distance_m": float(a.match_max_distance_m),
            "config_contract_sha256": contract_sha,
        },
    )

    records = [None] * len(windows)
    missing = []
    reused = 0
    for i, w in enumerate(windows):
        sid = base._window_id(w)
        old = reuse.get(sid)
        if old is not None:
            validate_probe_record(old, future_frames=pcfg.future_frames)
            records[i] = old
            reused += 1
        else:
            missing.append((i, w))

    def build_one(item):
        i, w = item
        rec = build_probe_record(
            source,
            w,
            pcfg,
            match_max_distance_m=float(a.match_max_distance_m),
        )
        validate_probe_record(rec, future_frames=pcfg.future_frames)
        return i, rec

    print(json.dumps({
        "protocol": PROTOCOL,
        "all_eligible_windows": len(windows),
        "reused_records": reused,
        "to_build": len(missing),
        "workers": workers,
        "prefetch_windows": prefetch,
    }))
    started = time.perf_counter()
    built = 0
    for i, rec in bounded_ordered_parallel_map(
        build_one,
        missing,
        max_workers=workers,
        max_in_flight=prefetch,
        thread_name_prefix="p0-f9-full-msp",
    ):
        records[i] = rec
        built += 1
        done = reused + built
        if built == 1 or done % 256 == 0 or done == len(windows):
            elapsed = max(time.perf_counter() - started, 1e-9)
            print(
                f"full MSP built/reused {done}/{len(windows)} "
                f"new_rate={built/elapsed:.2f} win/s "
                f"occ_cache={source.load_occ3d.cache_info()} pose_cache={source.pose.cache_info()}"
            )

    if any(r is None for r in records):
        raise RuntimeError("internal error: missing full MSP records")
    if [str(r["sample_id"]) for r in records] != selected_ids:
        raise RuntimeError("full MSP record order differs from deterministic window selection")

    scenes = sorted({str(r["scene_name"]) for r in records})
    metadata = {
        "version": MSP_CACHE_VERSION,
        "protocol": PROTOCOL,
        "mode": "train",
        "selection": SELECTION,
        "all_eligible_windows": True,
        "native_eligible_window_count": len(native_ids),
        "seed": int(a.seed),
        "stride": int(a.stride),
        "match_max_distance_m": float(a.match_max_distance_m),
        "feature_dim": FEATURE_DIM,
        "feature_names": list(FEATURE_NAMES),
        "feature_contract": base.FEATURE_CONTRACT,
        "target_contract": base.TARGET_CONTRACT,
        "config_contract_sha256": contract_sha,
        "num_windows": len(records),
        "num_unique_scenes": len(scenes),
        "scene_names": scenes,
        "summary": base.summarize(records),
        "resolved_config": cfg,
        "build_performance": {
            "workers": workers,
            "prefetch_windows": prefetch,
            "reused_records": reused,
            "reuse_cache": str(Path(a.reuse_cache).resolve()) if a.reuse_cache else None,
        },
    }
    payload = {"version": MSP_CACHE_VERSION, "metadata": metadata, "records": records}
    torch.save(payload, output)
    save_resolved_config(cfg, output.with_suffix(".resolved.yaml"))
    output.with_suffix(".summary.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    elapsed = max(time.perf_counter() - started, 1e-9)
    print(json.dumps({
        "output": str(output),
        "num_windows": len(records),
        "num_scenes": len(scenes),
        "reused_records": reused,
        "new_records": built,
        "elapsed_seconds": elapsed,
    }, indent=2))


if __name__ == "__main__":
    main()
