#!/usr/bin/env python3
"""Profile incremental learned-stage latency for selective V17 execution.

This measures only the stage whose compute changes with routing:
    causal selector score + V17 forward on selected sources.

Strong/KTA source extraction, occupancy I/O, rigid rasterization and final
composition are common to all policies and intentionally excluded. The report
therefore supports an *incremental learned-stage* efficiency claim, not an
end-to-end FPS claim.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from real_motion.motion_transport import FEATURE_DIM
from real_motion.selective_forecast import (
    CorrectionSelector,
    SELECTOR_INPUT_CONTRACT,
    SELECTOR_PROTOCOL,
    top_budget_mask,
)
from tools.real_motion.eval_p0_f9_v17_local_stwm import load_cache, load_model


PROTOCOL = "p0_f9_v17_selective_learned_stage_latency_v1"


def _parse_budgets(raw):
    vals = sorted(set(float(x.strip()) for x in str(raw).split(",") if x.strip()))
    if not vals or any(q < 0 or q > 100 for q in vals):
        raise ValueError("invalid budget list")
    if 100.0 not in vals:
        raise ValueError("latency budgets must include 100")
    return vals


def load_selector(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("protocol") != SELECTOR_PROTOCOL:
        raise RuntimeError("selector checkpoint protocol mismatch")
    if ck.get("selector_input_contract") != SELECTOR_INPUT_CONTRACT:
        raise RuntimeError("selector input contract mismatch")
    if int(ck.get("feature_dim", -1)) != FEATURE_DIM:
        raise RuntimeError("selector feature dimension mismatch")
    model = CorrectionSelector(
        FEATURE_DIM, int(ck.get("hidden_dim", 64))
    ).to(device)
    model.load_state_dict(ck["state_dict"], strict=True)
    model.eval()
    norm = ck["normalization"]
    mean = torch.tensor(norm["feature_mean"], dtype=torch.float32, device=device)
    std = torch.tensor(norm["feature_std"], dtype=torch.float32, device=device)
    return ck, model, mean, std


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--v17-cache", required=True)
    p.add_argument("--v17-checkpoint", required=True)
    p.add_argument("--selector-checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--budgets", default="0,10,20,40,100")
    p.add_argument("--warmup-windows", type=int, default=16)
    p.add_argument("--timing-windows", type=int, default=128)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-amp", action="store_true")
    a = p.parse_args()

    budgets = _parse_budgets(a.budgets)
    if min(a.warmup_windows, a.timing_windows, a.repeats) < 0 or a.repeats == 0:
        raise ValueError("invalid timing settings")

    meta, records = load_cache(a.v17_cache)
    if not records:
        raise RuntimeError("empty V17 cache")
    device = torch.device(
        a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    v17_ck, v17 = load_model(a.v17_checkpoint, device)
    if str(v17_ck.get("variant")) != "RL" or not bool(
        v17_ck.get("use_representation", False)
    ):
        raise RuntimeError("latency probe requires V17-RL representation checkpoint")
    selector_ck, selector, fmean, fstd = load_selector(
        a.selector_checkpoint, device
    )
    label_meta = selector_ck.get("label_cache_metadata") or {}
    if label_meta.get("checkpoint") != str(Path(a.v17_checkpoint).resolve()):
        raise RuntimeError("selector was trained for a different V17 checkpoint")

    amp = device.type == "cuda" and not bool(a.no_amp)
    max_needed = int(a.warmup_windows) + int(a.timing_windows)
    chosen = records[: min(len(records), max_needed)]
    if len(chosen) <= int(a.warmup_windows):
        raise RuntimeError("not enough cache windows after warmup")

    def one(rec, q, *, timed):
        f_cpu = rec["features"].float()
        n = int(f_cpu.shape[0])
        t0 = time.perf_counter()
        if float(q) >= 100.0:
            # Dense V17 is the no-router reference; do not charge it selector
            # overhead just to manufacture an all-true mask.
            score_cpu = np.zeros((n,), dtype=np.float64)
            _sync(device)
            t1 = time.perf_counter()
            mask = np.ones((n,), dtype=bool)
        else:
            f = f_cpu.to(device, non_blocking=False)
            score = selector((f - fmean) / fstd.clamp_min(1e-6))
            score_cpu = score.float().cpu().numpy()
            _sync(device)
            t1 = time.perf_counter()
            mask = top_budget_mask(score_cpu, q)
        ids_np = np.flatnonzero(mask)
        if len(ids_np) == 0:
            _sync(device)
            t2 = time.perf_counter()
            return {
                "sources": n,
                "selected": 0,
                "selector_ms": 1000.0 * (t1 - t0),
                "v17_ms": 0.0,
                "total_ms": 1000.0 * (t2 - t0),
            }

        ids = torch.as_tensor(ids_np, dtype=torch.long)
        # Index on CPU first, then transfer only the routed sources.
        fs = rec["features"][ids].float().to(device)
        tube = rec["local_semantic_tube"][ids].to(device)
        kta = rec["kta_displacement_xy_m"][ids].float().to(device)
        fm = rec["frame_motion_features"][ids].float().to(device)
        sm = rec["target_source_mask_tube"][ids].to(device)
        _sync(device)
        t_v17 = time.perf_counter()
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=amp
        ):
            _ = v17(fs, tube, kta, fm, sm)
        _sync(device)
        t2 = time.perf_counter()
        return {
            "sources": n,
            "selected": int(len(ids_np)),
            "selector_ms": 1000.0 * (t1 - t0),
            "v17_ms": 1000.0 * (t2 - t_v17),
            "total_ms": 1000.0 * (t2 - t0),
        }

    # Warm all kernels/memory paths at dense budget.
    for rec in chosen[: int(a.warmup_windows)]:
        one(rec, 100.0, timed=False)

    timed_records = chosen[
        int(a.warmup_windows):
        int(a.warmup_windows) + int(a.timing_windows)
    ]
    results = {}
    for q in budgets:
        rows = []
        for rr in range(int(a.repeats)):
            for rec in timed_records:
                rows.append(one(rec, q, timed=True))
        def stat(key):
            vals = [float(r[key]) for r in rows]
            return {
                "mean": float(statistics.mean(vals)),
                "median": float(statistics.median(vals)),
                "p90": float(np.quantile(vals, 0.90)),
            }
        results[str(float(q))] = {
            "budget_percent": float(q),
            "num_measurements": len(rows),
            "num_unique_windows": len(timed_records),
            "mean_sources": float(np.mean([r["sources"] for r in rows])),
            "mean_selected_sources": float(np.mean([r["selected"] for r in rows])),
            "mean_selected_fraction": float(np.mean([
                r["selected"] / max(r["sources"], 1) for r in rows
            ])),
            "selector_ms": stat("selector_ms"),
            "v17_ms": stat("v17_ms"),
            "learned_stage_total_ms": stat("total_ms"),
        }
        print(
            f"Q={q:g}% selected={100*results[str(float(q))]['mean_selected_fraction']:.2f}% "
            f"selector={results[str(float(q))]['selector_ms']['mean']:.3f}ms "
            f"V17={results[str(float(q))]['v17_ms']['mean']:.3f}ms "
            f"total={results[str(float(q))]['learned_stage_total_ms']['mean']:.3f}ms",
            flush=True,
        )

    dense_ms = results[str(100.0)]["learned_stage_total_ms"]["mean"]
    for row in results.values():
        total = row["learned_stage_total_ms"]["mean"]
        row["learned_stage_speedup_vs_dense"] = (
            dense_ms / total if total > 0 else float("inf")
        )
        row["learned_stage_latency_reduction_fraction"] = (
            1.0 - total / dense_ms if dense_ms > 0 else float("nan")
        )

    report = {
        "protocol": PROTOCOL,
        "scope": (
            "incremental learned-stage only: selector + routed V17 forward; "
            "excludes common Strong/KTA, occupancy IO, rasterization and composition"
        ),
        "v17_cache": str(Path(a.v17_cache).resolve()),
        "v17_checkpoint": str(Path(a.v17_checkpoint).resolve()),
        "selector_checkpoint": str(Path(a.selector_checkpoint).resolve()),
        "device": str(device),
        "amp_bf16": bool(amp),
        "warmup_windows": int(a.warmup_windows),
        "timing_windows": len(timed_records),
        "repeats": int(a.repeats),
        "results": results,
        "cache_metadata": meta,
    }
    op = Path(a.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved {op}")


if __name__ == "__main__":
    main()
