#!/usr/bin/env python3
"""Final single-boundary Dense Forecast FPS benchmark.

Official boundary:
  prepared CausalHistoryState
      -> live KTA/Strong prior
      -> frozen V18 motion + SE(2) transport
      -> Point CCR shared encoding + six horizon readouts
      -> constrained six-frame dense semantic occupancy.

History-only representation, disk I/O, GT, metrics, checkpoint loading,
compilation and warm-up are excluded. CPU+GPU work inside forecast_six is
measured with synchronized wall-clock time.
"""
from __future__ import annotations

import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import math
import signal
import threading
import time

import numpy as np
import torch

from real_motion.column_runtime_pipeline import CachedColumnSource
from real_motion.final_dataflow import prepare_history, forecast_six, build_causal_motion_prior
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
    load_point_head,
    select_population,
    forecast as legacy_forecast,
    result_signature,
)
from tools.real_motion.pilot_p0_f9_canonical_causal_repair import probabilities
from tools.real_motion import benchmark_p0_f9_v18_runtime as runtime


PROTOCOL = "p0_f9_dense_forecast_fps_final_v1"
FUTURE_FRAMES = 6


def _aggregate(trials):
    seconds = np.asarray([row["seconds"] for row in trials], dtype=np.float64)
    if not len(seconds) or not np.isfinite(seconds).all() or np.any(seconds <= 0):
        raise RuntimeError("invalid Dense Forecast timing population")
    total_future_frames = FUTURE_FRAMES * len(seconds)
    total_seconds = float(seconds.sum())
    stage_keys = sorted(set().union(*(row["host_stages_seconds"] for row in trials)))
    return {
        "samples": len(seconds),
        "windows": len({row["key"] for row in trials}),
        "future_frames": total_future_frames,
        "total_seconds": total_seconds,
        "dense_forecast_fps": total_future_frames / total_seconds,
        "mean_six_ms": 1000.0 * float(seconds.mean()),
        "p50_six_ms": 1000.0 * float(np.median(seconds)),
        "p90_six_ms": 1000.0 * float(np.percentile(seconds, 90)),
        "host_stage_mean_ms": {
            key: 1000.0 * float(np.mean([row["host_stages_seconds"].get(key, 0.0) for row in trials]))
            for key in stage_keys
        },
        "formula": "total_future_frames / total_synchronized_wall_time",
    }


def _summary(result):
    lines = [
        "===== FINAL DENSE FORECAST FPS =====",
        "status=" + result["status"],
        "protocol=" + PROTOCOL,
        "boundary=CausalHistoryState -> six finished dense semantic occupancy frames",
        "INCLUDED: live KTA/Strong + V18 motion + SE(2) transport + future projection/ownership + CCR + dense composition",
        "EXCLUDED: history-only representation + disk I/O + GT/metrics + checkpoint load + compile/warmup",
        "timing=torch.cuda.synchronize + time.perf_counter wall clock, batch=1",
    ]
    if result.get("aggregate"):
        a = result["aggregate"]
        lines += [
            f'Dense Forecast FPS={a["dense_forecast_fps"]:.3f}',
            f'six-frame latency mean={a["mean_six_ms"]:.3f} ms '
            f'p50={a["p50_six_ms"]:.3f} ms p90={a["p90_six_ms"]:.3f} ms',
            f'samples={a["samples"]} windows={a["windows"]} total_future_frames={a["future_frames"]}',
            "host stage means (diagnostic; official total is uninterrupted synchronized wall time)="
            + json.dumps(a["host_stage_mean_ms"], sort_keys=True),
        ]
    prep = result.get("history_preparation", [])
    if prep:
        lines.append(
            "excluded history preparation mean ms="
            + f'{1000*np.mean([x["total"] for x in prep]):.3f}'
        )
    if result.get("parity"):
        lines.append("legacy/new parity=" + json.dumps(result["parity"], sort_keys=True))
    lines += [
        "Point CCR quality status is unchanged by this speed benchmark.",
        "No training, threshold search, GT scoring, checkpoint mutation or deployment promotion.",
    ]
    if "error" in result:
        lines.append("error=" + result["error"])
    return "\n".join(lines) + "\n"


def main(stop_event=None, argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    for key in (
        "checkpoint", "ccr-checkpoint", "base-checkpoint", "dev-cache",
        "population-manifest", "dataroot", "dev-info", "out-dir",
    ):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--windows", type=int, default=20)
    parser.add_argument("--stress-windows", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cpu-workers", type=int, default=8)
    parser.add_argument("--ccr-cpu-workers", type=int, default=4)
    parser.add_argument("--parity-windows", type=int, default=20)
    args = parser.parse_args(argv)

    out = Path(args.out_dir)
    if out.exists():
        parser.error("fresh output required")
    for key in (
        "config", "checkpoint", "ccr_checkpoint", "base_checkpoint",
        "dev_cache", "population_manifest", "dev_info",
    ):
        if not Path(getattr(args, key) or "").is_file():
            parser.error("missing " + key)
    if (
        not Path(args.dataroot).is_dir()
        or not 1 <= args.windows <= 64
        or not 0 <= args.stress_windows < args.windows
        or args.repeats < 2
        or not 1 <= args.cpu_workers <= 16
        or not 1 <= args.ccr_cpu_workers <= 8
        or not 0 <= args.parity_windows <= args.windows
    ):
        parser.error("invalid Dense Forecast benchmark budget")

    device = require_cuda(args.device)
    torch.set_num_threads(1)
    out.mkdir(parents=True)
    started = time.perf_counter()
    result = {
        "status": "running",
        "protocol": PROTOCOL,
        "GPU": torch.cuda.get_device_name(device),
        "torch_version": str(torch.__version__),
        "repeats": args.repeats,
        "official_boundary": "CausalHistoryState_to_six_dense_semantic_occupancy",
        "no_training": True,
        "trials": [],
        "history_preparation": [],
        "parity": {"checked_windows": 0, "passed_windows": 0},
    }

    def persist():
        write_json(out / "dense_forecast_fps.json", result)
        (out / "summary.txt").write_text(_summary(result), encoding="utf-8")

    persist()
    pool = ThreadPoolExecutor(max_workers=args.ccr_cpu_workers)
    try:
        tick = time.perf_counter()
        result["native_preflight"] = prepare_native(out / "native_build")
        result["compile_seconds_excluded"] = time.perf_counter() - tick
        kernels = get_prepared_native()

        sources = {
            "epoch19": args.checkpoint,
            "point_ccr": args.ccr_checkpoint,
            "clean_e14": args.base_checkpoint,
        }
        snapshots = {name: out / (name + "_snapshot.pt") for name in sources}
        digests = {name: snapshot_checkpoint(path, snapshots[name]) for name, path in sources.items()}
        result["checkpoint_sha256"] = digests

        cfg = load_runtime_config(args.config, args.override)
        config_fp = stable_json_fingerprint(cfg)
        ck, teacher = load_joint(
            snapshots["epoch19"],
            device,
            reference_sha=CLEAN_SHA256,
            config_sha=config_fp,
            allow_diagnostic=True,
        )
        if (
            teacher.transport.config.history_frames != 4
            or ck.get("cursor_epoch") != 19
            or ck["model_configs"].get("adaptive_context") is not None
        ):
            raise RuntimeError("selected four-history epoch19 Local transport required")
        teacher.eval().requires_grad_(False)
        for path, expected in (
            (args.dev_cache, ck["cache_fingerprints"]["dev"]),
            (args.dev_info, ck["info_fingerprints"]["dev"]),
            (snapshots["clean_e14"], CLEAN_SHA256),
        ):
            if sha256(path) != expected:
                raise RuntimeError("checkpoint/data provenance mismatch: " + str(path))

        saved = torch.load(snapshots["point_ccr"], map_location="cpu", weights_only=False)
        head = load_point_head(
            saved,
            teacher_sha256=digests["epoch19"],
            config_fingerprint=config_fp,
            source_dim=teacher.columns.source_dim,
            device=device,
        )
        manifest, keys64, _ = load_manifest(args.population_manifest)
        parent = tuple(map(tuple, manifest["parent_keys"]))
        if (
            len(keys64) != 64
            or len(parent) != 512
            or manifest["manifest_fingerprint"] != ck["dev_manifest_fingerprint"]
            or tuple(map(tuple, ck["dev_keys"])) != parent
            or saved["contract"]["dev_manifest_fingerprint"] != manifest["manifest_fingerprint"]
            or tuple(map(tuple, saved["contract"]["final_dev_keys"])) != parent
        ):
            raise RuntimeError("frozen dev population identity/order mismatch")

        _, records = load_cache(args.dev_cache)
        record_keys(records)
        chosen, population = select_population(
            records, keys64, windows=args.windows, stress_windows=args.stress_windows
        )
        del records, saved
        result["population"] = {
            "windows": len(chosen),
            "scenes": len({r["scene_name"] for r in chosen}),
            "manifest_fingerprint": manifest["manifest_fingerprint"],
            "key_fingerprint": stable_json_fingerprint([m["key"] for m in population]),
            "selection": "scene-balanced round-robin + source-count-only stress; no GT/error/latency selection",
        }
        write_json(out / "fps_manifest.json", {
            "protocol": PROTOCOL, **result["population"], "keys": population
        })

        provider = PilotProvider(
            snapshots["clean_e14"], CLEAN_SHA256, make_prepare_config(cfg),
            device, args.cpu_workers, teacher, None,
        )
        source = CachedColumnSource(
            NuScenesWindowSource(args.dataroot, info_pkl=args.dev_info, verbose=False), 128
        )
        persist()

        with (out / "progress.jsonl").open("x", encoding="utf-8") as log:
            for wi, (record, meta) in enumerate(zip(chosen, population)):
                if stop_event is not None and stop_event.is_set():
                    break

                history = prepare_history(
                    provider, source, record, kernels=kernels, executor=pool
                )
                result["history_preparation"].append(history.preparation_seconds)

                # History-only cleanup must preserve the frozen cache's KTA/anchor
                # contract. This check is outside the official timer.
                live_kta, live_anchors = build_causal_motion_prior(
                    history, provider.pcfg.frame_dt_s
                )
                if not np.allclose(
                    live_kta,
                    torch.as_tensor(record["kta_displacement_xy_m"]).cpu().numpy(),
                    rtol=0, atol=1e-6,
                ):
                    raise RuntimeError("live KTA/cache mismatch")
                if not np.allclose(
                    live_anchors,
                    torch.as_tensor(record["anchors_xy_t0_m"]).cpu().numpy(),
                    rtol=0, atol=1e-5,
                ):
                    raise RuntimeError("live KTA anchors/cache mismatch")

                # Compile/kernel/model warm-up is explicitly outside FPS.
                forecast_six(
                    history, provider, teacher.transport, head, probabilities,
                    kernels=kernels, executor=pool, majority_backend="native",
                )
                torch.cuda.synchronize(device)

                # Strict old/new parity gate on the same window. Legacy path is
                # measurement infrastructure only and is never used for timing.
                if wi < args.parity_windows:
                    old_raw = provider.load_raw_columns(source, record, include_gt=False)
                    old_case = {
                        "record": record,
                        "raw": old_raw,
                        "gpu": runtime._gpu_inputs(record, device),
                    }
                    legacy = legacy_forecast(
                        old_case, provider, teacher.transport, head,
                        native=True, boundary="fresh_prior",
                        kernels=kernels, executor=pool,
                    )
                    new = forecast_six(
                        history, provider, teacher.transport, head, probabilities,
                        kernels=kernels, executor=pool, majority_backend="native",
                    )
                    torch.cuda.synchronize(device)
                    new_sig = result_signature(new["dense"], new["probability"], new["motion"])
                    result["parity"]["checked_windows"] += 1
                    if new_sig != legacy["signature"]:
                        raise RuntimeError("legacy/new dense forecast byte parity mismatch")
                    result["parity"]["passed_windows"] += 1
                    del old_case, old_raw, legacy, new

                for repeat in range(args.repeats):
                    torch.cuda.synchronize(device)
                    tick = time.perf_counter()
                    output = forecast_six(
                        history, provider, teacher.transport, head, probabilities,
                        kernels=kernels, executor=pool, majority_backend="native",
                    )
                    torch.cuda.synchronize(device)
                    elapsed = time.perf_counter() - tick
                    if not math.isfinite(elapsed) or elapsed <= 0:
                        raise RuntimeError("invalid synchronized wall time")
                    row = {
                        "key": "/".join(meta["key"]),
                        "stratum": meta["stratum"],
                        "sources": meta["sources"],
                        "repeat": repeat + 1,
                        "seconds": elapsed,
                        "host_stages_seconds": output["stages_seconds"],
                        "six_complete_dense": output["six_complete_dense"],
                    }
                    result["trials"].append(row)
                    log.write(json.dumps(finite_json(row), allow_nan=False) + "\n")
                    log.flush()

                result["aggregate"] = _aggregate(result["trials"])
                persist()
                print(
                    f'DENSE_FPS windows={wi+1}/{len(chosen)} '
                    f'FPS={result["aggregate"]["dense_forecast_fps"]:.3f} '
                    f'sources={meta["sources"]} stratum={meta["stratum"]}',
                    flush=True,
                )
                del history

        complete = len({x["key"] for x in result["trials"]}) == len(chosen)
        for name, path in sources.items():
            if sha256(path) != digests[name]:
                raise RuntimeError("source checkpoint changed: " + name)

        result.update(
            status="complete" if complete else "stopped",
            aggregate=_aggregate(result["trials"]) if result["trials"] else {},
            elapsed_seconds=time.perf_counter()-started,
            exactness={
                "live_kta_recomputed": True,
                "history_representation_built_once_per_window": True,
                "future_GT_used": False,
                "legacy_new_byte_parity_required": args.parity_windows,
                "source_checkpoints_unchanged": True,
            },
            route="single_official_dense_forecast_fps_only",
        )
        persist()
        print(_summary(result), flush=True)
        return 0 if complete else 130
    except Exception as exc:
        result.update(status="failed", error=str(exc), elapsed_seconds=time.perf_counter()-started)
        persist()
        raise
    finally:
        pool.shutdown(wait=True)


if __name__ == "__main__":
    stopped = threading.Event()
    def request_stop(signum, frame):
        stopped.set()
        print("Stop requested: finish current window; no checkpoint touched.", flush=True)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, request_stop)
    sys.exit(main(stopped))
