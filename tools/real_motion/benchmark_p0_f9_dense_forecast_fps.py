#!/usr/bin/env python3
"""Official single-boundary Dense Forecast FPS for V18 + Point CCR.

Formal boundary:
    prepared CausalHistoryState
        -> fresh KTA/Strong
        -> frozen V18 motion + SE(2) transport
        -> future projection/ownership
        -> Point CCR shared encode + six readouts
        -> six finished dense semantic occupancy grids

History-only I/O/source association/registration/canonical evidence are excluded.
Future GT, metrics, correctness hashing and disk output are excluded.
"""
import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import os
import signal
import threading
import time

import numpy as np
import torch

from real_motion.column_runtime_pipeline import CachedColumnSource
from real_motion.native_column_cpu import prepare_native, get_prepared_native
from real_motion.nuscenes_adapter import NuScenesWindowSource
from real_motion.runtime_config import add_config_args, load_runtime_config, make_prepare_config
from real_motion.v21_source_induction import stable_json_fingerprint
from tools.real_motion.eval_p0_f9_v18_se2 import load_cache
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import load_manifest, sha256
from tools.real_motion.joint_training_recovery import snapshot_checkpoint
from tools.real_motion.run_p0_f9_shared_evidence_pilot import PilotProvider, require_cuda
from tools.real_motion.static_evidence_selector_common import CLEAN_SHA256, write_json, finite_json
from tools.real_motion.train_p0_f9_causal_columns import record_keys
from tools.real_motion.train_p0_f9_joint_causal_columns import load_joint
from tools.real_motion.point_ccr_v18_fps_common import (
    select_population,
    load_point_head,
    forecast as legacy_forecast,
)
from tools.real_motion.final_dataflow import prepare_history, forecast_six
from tools.real_motion import benchmark_p0_f9_v18_runtime as runtime

PROTOCOL = "p0_f9_dense_forecast_fps_v1"


def aggregate(trials):
    if not trials:
        return {}
    seconds = np.asarray([row["seconds"] for row in trials], dtype=np.float64)
    if not np.isfinite(seconds).all() or np.any(seconds <= 0):
        raise RuntimeError("invalid formal latency sample")
    total_frames = 6 * len(seconds)
    total_seconds = float(seconds.sum())
    stage_keys = sorted(set().union(*(row["stages_seconds"] for row in trials)))
    return {
        "future_frames": total_frames,
        "samples": len(seconds),
        "windows": len({row["key"] for row in trials}),
        "total_seconds": total_seconds,
        "Dense_Forecast_FPS": total_frames / total_seconds,
        "mean_six_ms": 1000.0 * float(seconds.mean()),
        "p50_six_ms": 1000.0 * float(np.median(seconds)),
        "p90_six_ms": 1000.0 * float(np.percentile(seconds, 90)),
        "stage_host_mean_ms": {
            key: 1000.0 * float(np.mean([
                row["stages_seconds"].get(key, 0.0) for row in trials
            ]))
            for key in stage_keys
        },
    }


def brief(result):
    lines = [
        "===== OFFICIAL DENSE FORECAST FPS =====",
        "status=" + result["status"],
        "protocol=" + PROTOCOL,
        "ONE formal boundary only: CausalHistoryState -> SIX finished dense semantic occupancy grids.",
        "INCLUDED: fresh KTA/Strong, V18 motion, SE(2) transport/layering, future projection/ownership, CCR, dense composition.",
        "EXCLUDED: history I/O, history-only source extraction/association/registration/canonical evidence, compile/warmup, GT/metrics/hash/save.",
        "Timing: batch=1 synchronized CPU+GPU wall-clock via perf_counter; FPS=total future frames / total elapsed seconds.",
        "cached Strong/KTA is NOT a formal result.",
    ]
    if result.get("aggregate"):
        r = result["aggregate"]
        lines.append(
            f"Dense Forecast FPS={r['Dense_Forecast_FPS']:.3f} "
            f"mean6={r['mean_six_ms']:.3f}ms "
            f"P50={r['p50_six_ms']:.3f}ms P90={r['p90_six_ms']:.3f}ms "
            f"windows={r['windows']} samples={r['samples']}"
        )
        lines.append("host_stage_mean_ms=" + json.dumps(
            r["stage_host_mean_ms"], sort_keys=True))
    prep = result.get("history_prepare_seconds_excluded", [])
    if prep:
        lines.append(
            f"history_prepare_excluded_mean_ms={1000*float(np.mean(prep)):.3f}")
    if "parity" in result:
        lines.append("parity=" + json.dumps(result["parity"], sort_keys=True))
    if "population" in result:
        lines.append("population=" + json.dumps(
            result["population"], ensure_ascii=False))
    if "error" in result:
        lines.append("error=" + result["error"])
    return "\n".join(lines) + "\n"


def main(stop_event=None, argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    add_config_args(p)
    for key in (
        "checkpoint", "ccr-checkpoint", "base-checkpoint", "dev-cache",
        "population-manifest", "dataroot", "dev-info", "out-dir",
    ):
        p.add_argument("--" + key, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--windows", type=int, default=20)
    p.add_argument("--stress-windows", type=int, default=2)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--cpu-workers", type=int, default=8)
    p.add_argument("--ccr-cpu-workers", type=int, default=4)
    p.add_argument(
        "--skip-legacy-parity", action="store_true",
        help="debug only; formal release should keep old/new exact parity enabled")
    a = p.parse_args(argv)
    out = Path(a.out_dir)
    if out.exists():
        p.error("fresh output required")
    for key in (
        "config", "checkpoint", "ccr_checkpoint", "base_checkpoint",
        "dev_cache", "population_manifest", "dev_info",
    ):
        if not Path(getattr(a, key) or "").is_file():
            p.error("missing " + key)
    if (
        not Path(a.dataroot).is_dir()
        or not 1 <= a.windows <= 64
        or not 0 <= a.stress_windows < a.windows
        or a.repeats < 2
        or not 1 <= a.cpu_workers <= 16
        or not 1 <= a.ccr_cpu_workers <= 8
    ):
        p.error("invalid benchmark budget")

    device = require_cuda(a.device)
    torch.set_num_threads(1)
    out.mkdir(parents=True)
    started = time.perf_counter()
    result = {
        "status": "running",
        "protocol": PROTOCOL,
        "GPU": torch.cuda.get_device_name(device),
        "torch_version": str(torch.__version__),
        "repeats": a.repeats,
        "trials": [],
        "history_prepare_seconds_excluded": [],
        "formal_boundary": {
            "start": "prepared CausalHistoryState",
            "stop": "six finished dense semantic occupancy grids after CUDA synchronize",
            "fresh_strong_kta_included": True,
            "cached_strong_kta_allowed": False,
            "future_gt_loaded": False,
        },
    }

    def persist():
        write_json(out / "dense_forecast_fps.json", result)
        (out / "summary.txt").write_text(brief(result), encoding="utf-8")

    persist()
    original_backend = os.environ.get("SWFM_COLUMN_CPU_BACKEND")
    execution_pool = ThreadPoolExecutor(max_workers=a.ccr_cpu_workers)
    try:
        os.environ["SWFM_COLUMN_CPU_BACKEND"] = "numpy"
        tick = time.perf_counter()
        result["native_preflight"] = prepare_native(out / "native_build")
        result["compile_seconds_excluded"] = time.perf_counter() - tick

        sources = {
            "epoch19": a.checkpoint,
            "point_ccr": a.ccr_checkpoint,
            "clean_e14": a.base_checkpoint,
        }
        snapshots = {k: out / (k + "_snapshot.pt") for k in sources}
        digests = {
            k: snapshot_checkpoint(path, snapshots[k])
            for k, path in sources.items()
        }
        result["checkpoint_sha256"] = digests

        cfg = load_runtime_config(a.config, a.override)
        config_fp = stable_json_fingerprint(cfg)
        ck, teacher = load_joint(
            snapshots["epoch19"], device, reference_sha=CLEAN_SHA256,
            config_sha=config_fp, allow_diagnostic=True)
        if (
            teacher.transport.config.history_frames != 4
            or ck.get("cursor_epoch") != 19
            or ck["model_configs"].get("adaptive_context") is not None
        ):
            raise RuntimeError("selected FOUR-history epoch19 transport required")
        teacher.eval().requires_grad_(False)

        for path, expected in (
            (a.dev_cache, ck["cache_fingerprints"]["dev"]),
            (a.dev_info, ck["info_fingerprints"]["dev"]),
            (snapshots["clean_e14"], CLEAN_SHA256),
        ):
            if sha256(path) != expected:
                raise RuntimeError("checkpoint/data provenance mismatch: " + str(path))

        saved = torch.load(
            snapshots["point_ccr"], map_location="cpu", weights_only=False)
        head = load_point_head(
            saved, teacher_sha256=digests["epoch19"],
            config_fingerprint=config_fp,
            source_dim=teacher.columns.source_dim, device=device)

        manifest, keys64, _ = load_manifest(a.population_manifest)
        parent = tuple(map(tuple, manifest["parent_keys"]))
        if (
            len(keys64) != 64
            or len(parent) != 512
            or manifest["manifest_fingerprint"] != ck["dev_manifest_fingerprint"]
            or tuple(map(tuple, ck["dev_keys"])) != parent
            or saved["contract"]["dev_manifest_fingerprint"]
                != manifest["manifest_fingerprint"]
            or tuple(map(tuple, saved["contract"]["final_dev_keys"])) != parent
        ):
            raise RuntimeError("frozen dev population identity/order mismatch")

        _, records = load_cache(a.dev_cache)
        record_keys(records)
        chosen, population = select_population(
            records, keys64, windows=a.windows,
            stress_windows=a.stress_windows)
        del records, saved
        result["population"] = {
            "windows": len(chosen),
            "scenes": len({r["scene_name"] for r in chosen}),
            "manifest_fingerprint": manifest["manifest_fingerprint"],
            "key_fingerprint": stable_json_fingerprint(
                [m["key"] for m in population]),
            "selection": (
                "scene-balanced round-robin + source-count-only stress; "
                "no GT/errors/latency selection"
            ),
        }
        write_json(
            out / "fps_manifest.json",
            {"protocol": PROTOCOL, **result["population"], "keys": population})

        provider = PilotProvider(
            snapshots["clean_e14"], CLEAN_SHA256, make_prepare_config(cfg),
            device, a.cpu_workers, teacher, None)
        provider.reference.eval().requires_grad_(False)
        model = teacher.transport
        source = CachedColumnSource(
            NuScenesWindowSource(
                a.dataroot, info_pkl=a.dev_info, verbose=False),
            128,
        )
        kernels = get_prepared_native()
        result["parity"] = {
            "enabled": not a.skip_legacy_parity,
            "windows": 0,
            "exact_signatures": True,
        }
        completed = 0

        with (out / "progress.jsonl").open("x", encoding="utf-8") as log:
            for index, (record, meta) in enumerate(zip(chosen, population)):
                if stop_event is not None and stop_event.is_set():
                    break

                prep_tick = time.perf_counter()
                history, future = prepare_history(
                    provider, source, record, device=device,
                    kernels=kernels, executor=execution_pool)
                torch.cuda.synchronize(device)
                result["history_prepare_seconds_excluded"].append(
                    time.perf_counter() - prep_tick)

                # One old/new equivalence check per window, completely outside
                # official timing aggregation. It also warms kernels/models.
                if not a.skip_legacy_parity:
                    legacy_raw = provider.load_raw_columns(
                        source, record, include_gt=False)
                    legacy_case = {
                        "record": record,
                        "raw": legacy_raw,
                        "gpu": runtime._gpu_inputs(record, device),
                    }
                    legacy = legacy_forecast(
                        legacy_case, provider, model, head,
                        native=False, boundary="fresh_prior",
                        kernels=kernels, executor=execution_pool)
                    formal = forecast_six(
                        history, future, provider, model, head,
                        kernels=kernels, executor=execution_pool,
                        optimize_motion=True, native_majority=True,
                        return_signature=True)
                    if legacy["signature"] != formal["signature"]:
                        raise RuntimeError(
                            "final dataflow differs from legacy exact output: "
                            + "/".join(meta["key"]))
                    result["parity"]["windows"] += 1
                    del legacy_case, legacy_raw, legacy, formal
                else:
                    warm = forecast_six(
                        history, future, provider, model, head,
                        kernels=kernels, executor=execution_pool,
                        optimize_motion=True, native_majority=True,
                        return_signature=False)
                    del warm

                for repeat in range(a.repeats):
                    if stop_event is not None and stop_event.is_set():
                        break
                    row = forecast_six(
                        history, future, provider, model, head,
                        kernels=kernels, executor=execution_pool,
                        optimize_motion=True, native_majority=True,
                        return_signature=False)
                    trial = {
                        "seconds": row["seconds"],
                        "stages_seconds": row["stages_seconds"],
                        "prior_profile_ms": row["prior_profile_ms"],
                        "key": "/".join(meta["key"]),
                        "stratum": meta["stratum"],
                        "repeat": repeat + 1,
                        "six_complete_dense": row["six_complete_dense"],
                    }
                    result["trials"].append(trial)
                    log.write(json.dumps(
                        finite_json(trial), allow_nan=False) + "\n")
                    log.flush()
                    del row
                else:
                    completed += 1
                    print(
                        f"DENSE_FPS {completed}/{len(chosen)} "
                        f"sources={meta['sources']} stratum={meta['stratum']}",
                        flush=True)
                    persist()
                    del history, future
                    continue
                break

        for name, path in sources.items():
            if sha256(path) != digests[name]:
                raise RuntimeError(
                    "source checkpoint changed during read-only FPS: " + name)

        complete = completed == len(chosen)
        result.update(
            status="complete" if complete else "stopped",
            aggregate=aggregate(result["trials"]),
            elapsed_seconds=time.perf_counter() - started,
            no_training=True,
            no_saved_scientific_updates=True,
            route="formal_dense_forecast_fps_only",
        )
        persist()
        print(brief(result), flush=True)
        return 0 if complete else 130

    except Exception as exc:
        result.update(
            status="failed",
            error=str(exc),
            elapsed_seconds=time.perf_counter() - started)
        persist()
        raise
    finally:
        execution_pool.shutdown(wait=True)
        if original_backend is None:
            os.environ.pop("SWFM_COLUMN_CPU_BACKEND", None)
        else:
            os.environ["SWFM_COLUMN_CPU_BACKEND"] = original_backend


if __name__ == "__main__":
    stopped = threading.Event()

    def request_stop(signum, frame):
        stopped.set()
        print(
            "Stop requested: finish current formal window; no checkpoint touched.",
            flush=True)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, request_stop)
    sys.exit(main(stopped))
