#!/usr/bin/env python3
"""End-to-end latency benchmark for the V20 full six-frame pipeline.

Timing boundaries are explicit and non-overlapping at the top level:
  * raw_window_load: dataset/cache I/O and raw history preparation;
  * v18_input_preparation: Strong/KTA/source preparation + GPU staging;
  * v18_forward: one frozen Clean-E14 forward with latent exposure;
  * v18_six_frame_render: six dense current-source renders using that output;
  * V20 detailed components: history alignment, shared encoder, Static,
    Dormant and Birth network/render work;
  * final_composition: V18 > Dormant > Birth > Static protected add-only.

The benchmark never loads future semantic GT and never computes metrics.
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

from real_motion.prepared import load_nuscenes_window_raw
from real_motion.runtime_config import add_config_args, load_runtime_config, make_prepare_config
from real_motion.strong_w2det import StrongW2DetConfig
from real_motion.v20_history_world import CanonicalLattice, protected_add_only
from real_motion.v20_pipeline import run_v20_modules
from real_motion.v20_training import load_v20_checkpoint
from tools.real_motion import eval_p0_f9_v18_se2 as base
from tools.real_motion import eval_p0_f9_v18_full_validation as full
from tools.real_motion.benchmark_p0_f9_v18_runtime import (
    _forecast_once,
    _model_forward,
    _release_gpu_inputs,
    _stage_gpu_inputs,
)
from tools.real_motion.eval_p0_f9_v17_local_stwm import window_from_record
from tools.real_motion.eval_p0_f9_v19_factorized_static_new_fov import (
    CachedSource,
    _ComponentLRU,
    _prepare_record_from_raw,
)
from tools.real_motion.train_p0_f9_v18_se2_clean import PROTOCOL as CLEAN_PROTOCOL

PROTOCOL = "p0_f9_v20_full_runtime_benchmark_v1"


def _lattice(d):
    return CanonicalLattice(
        tuple(float(x) for x in d["origin_xyz_m"]),
        tuple(float(x) for x in d["voxel_size_xyz_m"]),
        tuple(int(x) for x in d["shape_xyz"]),
    )


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _summary(values):
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return {}
    return {
        "n": int(x.size),
        "mean_ms": float(x.mean()),
        "median_ms": float(np.median(x)),
        "p90_ms": float(np.quantile(x, 0.90)),
        "p95_ms": float(np.quantile(x, 0.95)),
        "min_ms": float(x.min()),
        "max_ms": float(x.max()),
        "windows_per_s_from_mean": float(1000.0 / max(x.mean(), 1e-12)),
        "future_frames_per_s_from_mean": float(6000.0 / max(x.mean(), 1e-12)),
    }


def _compose(base_np, vp, free_label):
    return protected_add_only(
        torch.from_numpy(np.asarray(base_np)),
        dormant=torch.from_numpy(vp.dormant_future),
        birth=torch.from_numpy(vp.birth_future),
        static_world=torch.from_numpy(vp.static_future),
        free_label=int(free_label),
    ).numpy()


def main():
    p = argparse.ArgumentParser()
    add_config_args(p)
    p.add_argument("--val-cache", required=True)
    p.add_argument("--base-checkpoint", required=True)
    p.add_argument("--v20-checkpoint", required=True)
    p.add_argument("--dataroot", required=True)
    p.add_argument("--info-pkl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--warmup-windows", type=int, default=2)
    p.add_argument("--measure-windows", type=int, default=20)
    p.add_argument("--dormant-existence-threshold", type=float, default=0.5)
    p.add_argument("--birth-existence-threshold", type=float, default=0.5)
    p.add_argument("--birth-shape-threshold", type=float, default=0.5)
    p.add_argument("--birth-duplicate-distance-m", type=float, default=2.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-amp", action="store_true")
    a = p.parse_args()

    if int(a.warmup_windows) < 0 or int(a.measure_windows) <= 0:
        raise ValueError("warmup must be >=0 and measure-windows >0")
    pcfg = make_prepare_config(load_runtime_config(a.config, a.override))
    _, records = base.load_cache(a.val_cache)
    need = int(a.warmup_windows) + int(a.measure_windows)
    if len(records) < need:
        raise RuntimeError(
            f"benchmark needs {need} windows but cache has only {len(records)}"
        )
    records = records[:need]

    device = torch.device(
        a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    amp = device.type == "cuda" and not bool(a.no_amp)
    _, v18, _ = full._load_model(a.base_checkpoint, CLEAN_PROTOCOL, device)
    v18.eval()
    model, vck = load_v20_checkpoint(a.v20_checkpoint, map_location="cpu")
    if str(vck.get("stage")) not in {"birth", "full"}:
        raise RuntimeError("full runtime benchmark requires Birth-capable V20 checkpoint")
    extra = dict(vck.get("extra") or {})
    high = _lattice(extra["highres_lattice"])
    coarse = _lattice(extra["coarse_lattice"])
    tile_size = tuple(int(x) for x in extra.get("tile_size_xyz", [32, 32, 16]))
    model.to(device).eval()

    source = CachedSource(a.dataroot, info_pkl=a.info_pkl, verbose=False)
    strong_cfg = StrongW2DetConfig(free_label=int(pcfg.free_label))
    component_cache = _ComponentLRU(maxsize=1024)
    measured = []
    peak_cuda = []
    audits = {
        "dormant_tracks": 0,
        "static_query_voxels": 0,
        "static_active_tiles": 0,
        "history_observed_oob": 0,
        "static_render_oob": 0,
        "dormant_render_oob": 0,
        "birth_render_oob": 0,
        "birth_duplicate_suppressed_query_horizons": 0,
    }

    for wi, rec in enumerate(records):
        w = window_from_record(rec)
        is_measured = wi >= int(a.warmup_windows)
        row = {}

        _sync(device)
        wall_start = time.perf_counter()
        t = time.perf_counter()
        raw = load_nuscenes_window_raw(
            source, w, pcfg, include_gt=False
        )
        row["raw_window_load"] = (time.perf_counter() - t) * 1000.0

        if is_measured and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        _sync(device)
        in_memory_start = time.perf_counter()
        t = time.perf_counter()
        state = _prepare_record_from_raw(
            rec, raw, source, pcfg, strong_cfg, device, component_cache
        )
        _stage_gpu_inputs(state, device)
        _sync(device)
        row["v18_input_preparation"] = (time.perf_counter() - t) * 1000.0

        try:
            _sync(device)
            t = time.perf_counter()
            v18_out = _model_forward(
                v18, state["gpu"], device, return_latents=True
            )
            _sync(device)
            row["v18_forward"] = (time.perf_counter() - t) * 1000.0

            _sync(device)
            t = time.perf_counter()
            pred_stack = np.asarray(
                _forecast_once(
                    v18, state, pcfg, strong_cfg, device,
                    precomputed_out=v18_out,
                ),
                dtype=np.uint8,
            )
            _sync(device)
            row["v18_six_frame_render"] = (
                time.perf_counter() - t
            ) * 1000.0
        finally:
            _release_gpu_inputs(state)

        vp = run_v20_modules(
            model,
            v18,
            rec,
            raw,
            pcfg=pcfg,
            strong_cfg=strong_cfg,
            high_lattice=high,
            coarse_lattice=coarse,
            tile_size_xyz=tile_size,
            device=device,
            amp=amp,
            v18_output_with_latents=v18_out,
            enable_static=True,
            enable_dormant=True,
            enable_birth=True,
            dormant_existence_threshold=float(a.dormant_existence_threshold),
            birth_existence_threshold=float(a.birth_existence_threshold),
            birth_shape_threshold=float(a.birth_shape_threshold),
            birth_duplicate_distance_m=float(a.birth_duplicate_distance_m),
        )
        for k, v in vp.timings_ms.items():
            row[f"v20_{k}"] = float(v)

        _sync(device)
        t = time.perf_counter()
        final_pred = _compose(pred_stack, vp, pcfg.free_label)
        _sync(device)
        row["final_composition"] = (time.perf_counter() - t) * 1000.0
        if final_pred.shape != pred_stack.shape:
            raise RuntimeError("full V20 composition shape mismatch")

        row["in_memory_total"] = (
            time.perf_counter() - in_memory_start
        ) * 1000.0
        row["end_to_end_with_raw_io"] = (
            time.perf_counter() - wall_start
        ) * 1000.0
        row["v18_current_path"] = (
            row["v18_input_preparation"]
            + row["v18_forward"]
            + row["v18_six_frame_render"]
        )
        row["v20_added_path"] = sum(
            float(v)
            for k, v in row.items()
            if k.startswith("v20_")
        ) + row["final_composition"]

        if is_measured:
            measured.append(row)
            if device.type == "cuda":
                peak_cuda.append(
                    int(torch.cuda.max_memory_allocated(device))
                )
            audits["dormant_tracks"] += int(vp.dormant_tracks)
            audits["history_observed_oob"] += int(
                vp.history_out_of_bounds_observed_samples
            )
            if vp.static_report is not None:
                audits["static_query_voxels"] += int(
                    vp.static_report.query_voxels
                )
                audits["static_active_tiles"] += int(
                    vp.static_report.active_tiles
                )
                audits["static_render_oob"] += int(
                    vp.static_report.out_of_bounds_voxels
                )
            if vp.dormant_report is not None:
                audits["dormant_render_oob"] += int(
                    vp.dormant_report.out_of_bounds_voxels
                )
            if vp.birth_report is not None:
                audits["birth_render_oob"] += int(
                    vp.birth_report.out_of_bounds_voxels
                )
                audits["birth_duplicate_suppressed_query_horizons"] += int(
                    vp.birth_report.duplicate_suppressed_query_horizons
                )

        print(
            f"v20_runtime {wi + 1}/{len(records)} "
            f"measured={len(measured)}/{a.measure_windows}",
            flush=True,
        )

    names = sorted(set().union(*(r.keys() for r in measured)))
    component_summary = {
        name: _summary([r[name] for r in measured])
        for name in names
    }
    result = {
        "protocol": PROTOCOL,
        "warmup_windows": int(a.warmup_windows),
        "measured_windows": int(len(measured)),
        "base_checkpoint": str(Path(a.base_checkpoint).resolve()),
        "v20_checkpoint": str(Path(a.v20_checkpoint).resolve()),
        "v20_stage": str(vck.get("stage")),
        "future_gt_loaded": False,
        "timing_scope": {
            "raw_window_load": "dataset/cache I/O and six-frame raw history/future poses",
            "v18_input_preparation": "Strong/KTA/source preparation plus GPU staging",
            "v18_forward": "exactly one frozen Clean-E14 neural forward with latent exposure",
            "v18_six_frame_render": "six-horizon current-source rigid render/composition using the same forward",
            "v20_history_alignment": "six historical OCC frames aligned once to t0 canonical Ωmax",
            "v20_v20_encoder": "shared 3D historical-evidence encoder",
            "v20_static_network_and_render": "all active M_query Static tiles plus canonical-to-six-future render",
            "v20_dormant_prepare": "causal source-memory rebuild and dormant selection",
            "v20_dormant_network": "frozen V18 dormant-token extraction + local scene sample + Dormant head",
            "v20_dormant_render": "six-frame observed source-shape Dormant render",
            "v20_birth_network": "persistent Birth queries with spatial/V18-source conditioning",
            "v20_birth_render": "six-frame local-shape Birth render and duplicate suppression",
            "final_composition": "V18 > Dormant > Birth > Static protected add-only",
            "in_memory_total": "all online work after raw window is loaded",
            "end_to_end_with_raw_io": "raw load plus complete online six-frame forecast",
        },
        "components": component_summary,
        "peak_cuda_memory": (
            {
                "mean_bytes": float(np.mean(peak_cuda)),
                "max_bytes": int(max(peak_cuda)),
                "max_gib": float(max(peak_cuda) / (1024 ** 3)),
            }
            if peak_cuda else None
        ),
        "audit": audits,
        "raw_measured_ms": measured,
        "notes": {
            "static": (
                "Static currently decodes every M_query-intersecting fine tile. "
                "This benchmark is the prerequisite for any later calibrated "
                "coarse candidate filtering."
            ),
            "comparison": (
                "Report timing boundary explicitly; do not compare end-to-end "
                "numbers against methods that start from cached latents."
            ),
        },
    }
    op = Path(a.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "measured_windows": result["measured_windows"],
        "components": {
            k: v for k, v in component_summary.items()
            if k in {
                "v18_current_path",
                "v20_added_path",
                "in_memory_total",
                "end_to_end_with_raw_io",
            }
        },
        "peak_cuda_memory": result["peak_cuda_memory"],
        "audit": audits,
    }, indent=2))
    print(f"saved {op}")


if __name__ == "__main__":
    main()
