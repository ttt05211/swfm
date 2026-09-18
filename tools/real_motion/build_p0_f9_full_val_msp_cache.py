#!/usr/bin/env python3
"""Build an MSP probe cache over every eligible validation 6+6 window.

This is the validation counterpart of ``build_p0_f9_full_msp_cache.py``.
It uses the same official temporal-info split and the same causal MSP feature /
GT target contracts, but keeps *all* stride-1 6-history + 6-future windows
instead of the development protocol's one midpoint window per scene.

An existing 128-scene midpoint validation cache may be supplied with
``--reuse-cache``; matching records are reused bit-for-bit and all remaining
windows are built in deterministic native chronological order.
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


PROTOCOL = "p0_f9_full_val_all_eligible_msp_probe_v1"
SELECTION = "all_eligible_stride1_chronological_v1"


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
        help="optional compatible validation MSP cache, e.g. msp_probe_val_128.pt",
    )
    p.add_argument(
        "--expected-windows",
        type=int,
        default=0,
        help="optional hard count check; use 4369 for the current official val temporal pickle",
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
        raise RuntimeError("formal full validation requires the frozen 6+6 window contract")

    workers = int(a.workers) if int(a.workers) > 0 else min(
        16, max(1, os.cpu_count() or 1)
    )
    prefetch = int(a.prefetch_windows) if int(a.prefetch_windows) > 0 else 4 * workers

    source = base.CachedNuScenesWindowSource(
        a.dataroot, info_pkl=a.info_pkl, verbose=False
    )

    # Native chronological population: every eligible validation window.
    windows = list(source.iter_windows(
        history=int(pcfg.history_frames),
        future=int(pcfg.future_frames),
        stride=int(a.stride),
        max_windows=None,
    ))
    if not windows:
        raise RuntimeError("validation split contains no eligible 6+6 windows")
    if int(a.expected_windows) > 0 and len(windows) != int(a.expected_windows):
        raise RuntimeError(
            f"eligible validation window count {len(windows)} != expected {a.expected_windows}"
        )

    selected_ids = [base._window_id(w) for w in windows]
    if len(selected_ids) != len(set(selected_ids)):
        raise RuntimeError("full validation selection contains duplicate window IDs")

    # Independent selector-set audit.  Train mode only changes ordering; with
    # max_windows=None it enumerates the same complete eligible population.
    audit_windows = base.select_windows(
        source,
        mode="train",
        history=int(pcfg.history_frames),
        future=int(pcfg.future_frames),
        stride=int(a.stride),
        max_windows=None,
        seed=int(a.seed),
    )
    audit_ids = {base._window_id(w) for w in audit_windows}
    if set(selected_ids) != audit_ids:
        missing = sorted(audit_ids - set(selected_ids))
        extra = sorted(set(selected_ids) - audit_ids)
        raise RuntimeError(
            "native chronological validation population differs from independent selector: "
            f"missing={missing[:3]} extra={extra[:3]}"
        )

    contract_sha = config_fingerprint(cfg, "cache")
    reuse = base._load_reuse_records(
        a.reuse_cache,
        expected={
            "mode": "val",
            "seed": int(a.seed),
            "stride": int(a.stride),
            "match_max_distance_m": float(a.match_max_distance_m),
            "config_contract_sha256": contract_sha,
        },
    )

    records = [None] * len(windows)
    missing_work = []
    reused = 0
    for i, w in enumerate(windows):
        sid = base._window_id(w)
        old = reuse.get(sid)
        if old is not None:
            validate_probe_record(old, future_frames=pcfg.future_frames)
            records[i] = old
            reused += 1
        else:
            missing_work.append((i, w))

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
        "unique_scenes": len({w.scene_name for w in windows}),
        "reused_records": reused,
        "to_build": len(missing_work),
        "workers": workers,
        "prefetch_windows": prefetch,
    }), flush=True)

    started = time.perf_counter()
    built = 0
    for i, rec in bounded_ordered_parallel_map(
        build_one,
        missing_work,
        max_workers=workers,
        max_in_flight=prefetch,
        thread_name_prefix="p0-f9-full-val-msp",
    ):
        records[i] = rec
        built += 1
        done = reused + built
        if built == 1 or done % 256 == 0 or done == len(windows):
            elapsed = max(time.perf_counter() - started, 1e-9)
            print(
                f"full val MSP built/reused {done}/{len(windows)} "
                f"new_rate={built/elapsed:.2f} win/s "
                f"occ_cache={source.load_occ3d.cache_info()} "
                f"pose_cache={source.pose.cache_info()}",
                flush=True,
            )

    if any(r is None for r in records):
        raise RuntimeError("internal error: missing full validation MSP records")
    if [str(r["sample_id"]) for r in records] != selected_ids:
        raise RuntimeError("full validation MSP record order differs from native chronology")

    scenes = sorted({str(r["scene_name"]) for r in records})
    metadata = {
        "version": MSP_CACHE_VERSION,
        "protocol": PROTOCOL,
        "mode": "val",
        "selection": SELECTION,
        "all_eligible_windows": True,
        "native_eligible_window_count": len(windows),
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
    payload = {
        "version": MSP_CACHE_VERSION,
        "metadata": metadata,
        "records": records,
    }
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
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
