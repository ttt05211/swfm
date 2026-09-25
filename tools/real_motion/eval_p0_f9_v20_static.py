#!/usr/bin/env python3
"""Formal Stage-2 evaluation for V20 canonical Static world."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from real_motion.metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS
from real_motion.prepared import load_nuscenes_window_raw
from real_motion.runtime_config import add_config_args, load_runtime_config, make_prepare_config
from real_motion.strong_w2det import StrongW2DetConfig
from real_motion.v20_evaluation import population_fingerprint
from real_motion.v20_history_world import (
    CanonicalLattice,
    align_history_once_to_canonical,
    poses_to_t0_canonical,
    protected_add_only,
)
from real_motion.v20_runtime import decode_static_world_tiled, static_subset_masks
from real_motion.v20_training import load_v20_checkpoint
from tools.real_motion import eval_p0_f9_v18_se2 as base
from tools.real_motion import eval_p0_f9_v18_full_validation as full
from tools.real_motion.benchmark_p0_f9_v18_runtime import (
    _forecast_once,
    _release_gpu_inputs,
    _stage_gpu_inputs,
)
from tools.real_motion.eval_p0_f9_v17_local_stwm import window_from_record
from tools.real_motion.eval_p0_f9_v19_factorized_static_new_fov import (
    CachedSource,
    _ComponentLRU,
    _moving_support_sequence,
    _prepare_record_from_raw,
)
from tools.real_motion.eval_p0_f9_v19_static_new_fov import (
    HORIZONS,
    _delta,
    _finalize,
    _new_raw,
    _update_many,
)
from tools.real_motion.train_p0_f9_v18_se2_clean import PROTOCOL as CLEAN_PROTOCOL

PROTOCOL = "p0_f9_v20_static_eval_v1"
VARIANTS = ("v18", "v20_static")


def _lattice(d):
    return CanonicalLattice(
        tuple(float(x) for x in d["origin_xyz_m"]),
        tuple(float(x) for x in d["voxel_size_xyz_m"]),
        tuple(int(x) for x in d["shape_xyz"]),
    )


def _autocast(device, enabled):
    if enabled and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def _subset_empty():
    return {
        "domain_voxels": 0,
        "gt_positive_voxels": 0,
        "added_voxels": 0,
        "added_tp": 0,
        "added_fp": 0,
        "semantic_correct_tp": 0,
    }


def _update_subset(row, domain, base_pred, final_pred, gt, free_label):
    d = np.asarray(domain, dtype=bool)
    added = d & (base_pred == int(free_label)) & (final_pred != int(free_label))
    occ = gt != int(free_label)
    tp = added & occ
    fp = added & ~occ
    row["domain_voxels"] += int(d.sum())
    row["gt_positive_voxels"] += int((d & occ).sum())
    row["added_voxels"] += int(added.sum())
    row["added_tp"] += int(tp.sum())
    row["added_fp"] += int(fp.sum())
    row["semantic_correct_tp"] += int((tp & (final_pred == gt)).sum())


def _finish_subset(row):
    x = dict(row)
    x["addition_precision"] = float(x["added_tp"] / max(x["added_tp"] + x["added_fp"], 1))
    x["positive_recall_from_additions"] = float(x["added_tp"] / max(x["gt_positive_voxels"], 1))
    x["semantic_accuracy_on_added_tp"] = float(
        x["semantic_correct_tp"] / max(x["added_tp"], 1)
    )
    return x


def main():
    p = argparse.ArgumentParser()
    add_config_args(p)
    p.add_argument("--val-cache", required=True)
    p.add_argument("--base-checkpoint", required=True)
    p.add_argument("--v20-checkpoint", required=True)
    p.add_argument("--dataroot", required=True)
    p.add_argument("--info-pkl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-windows", type=int, default=0)
    p.add_argument("--alignment-workers", type=int, default=6)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-amp", action="store_true")
    a = p.parse_args()

    pcfg = make_prepare_config(load_runtime_config(a.config, a.override))
    _, records = base.load_cache(a.val_cache)
    if int(a.max_windows) > 0:
        records = records[:min(len(records), int(a.max_windows))]
    if not records:
        raise RuntimeError("empty validation population")
    device = torch.device(a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and not bool(a.no_amp)

    base_ck, base_model, _ = full._load_model(a.base_checkpoint, CLEAN_PROTOCOL, device)
    model, vck = load_v20_checkpoint(a.v20_checkpoint, map_location="cpu")
    if str(vck.get("stage")) != "static":
        raise RuntimeError(f"expected V20 static checkpoint, got {vck.get('stage')}")
    extra = dict(vck.get("extra") or {})
    high = _lattice(extra["highres_lattice"])
    coarse = _lattice(extra["coarse_lattice"])
    tile_size = tuple(int(x) for x in extra.get("tile_size_xyz", [32, 32, 16]))
    model.to(device).eval()

    source = CachedSource(a.dataroot, info_pkl=a.info_pkl, verbose=False)
    strong_cfg = StrongW2DetConfig(free_label=int(pcfg.free_label))
    component_cache = _ComponentLRU(maxsize=1024)
    raw_by_variant = {v: _new_raw() for v in VARIANTS}
    subset = {
        "history_seen_t0_missing": _subset_empty(),
        "never_seen_static_domain": _subset_empty(),
    }
    total_query = total_tiles = total_oob = total_history_oob = 0
    started = time.perf_counter()

    for wi, rec in enumerate(records, start=1):
        w = window_from_record(rec)
        raw = load_nuscenes_window_raw(source, w, pcfg, include_gt=True)
        future_obs = np.stack([
            source.load_lidar_observation(str(w.scene_name), str(tok))
            for tok in w.future_tokens
        ])
        state = _prepare_record_from_raw(
            rec, raw, source, pcfg, strong_cfg, device, component_cache
        )
        _stage_gpu_inputs(state, device)
        try:
            pred_all = _forecast_once(base_model, state, pcfg, strong_cfg, device)
        finally:
            _release_gpu_inputs(state)
        pred_stack = np.asarray(pred_all, dtype=np.uint8)

        hist_pose = np.asarray(raw["history_poses"], dtype=np.float64)
        future_pose = np.asarray(raw["future_poses"], dtype=np.float64)
        t0 = hist_pose[-1]
        hist_rel = poses_to_t0_canonical(hist_pose, t0)
        future_rel = poses_to_t0_canonical(future_pose, t0)
        aligned = align_history_once_to_canonical(
            coarse,
            history_semantic=np.asarray(raw["history_occ"], dtype=np.uint8),
            history_observed=np.asarray(raw["history_observed"], dtype=bool),
            history_ego_to_world=hist_pose,
            t0_ego_to_world=t0,
            native_origin_xyz_m=(pcfg.grid.x_min, pcfg.grid.y_min, pcfg.grid.z_min),
            native_voxel_size_xyz_m=pcfg.grid.voxel_size,
            free_label=int(pcfg.free_label),
        )
        sem = torch.from_numpy(aligned.semantic).to(device).unsqueeze(0)
        obs = torch.from_numpy(aligned.observed).to(device).unsqueeze(0)
        obsfree = torch.from_numpy(aligned.observed_free).to(device).unsqueeze(0)

        with torch.inference_mode(), _autocast(device, amp):
            scene = model.encode_history(sem, obs, obsfree)
            static = decode_static_world_tiled(
                model,
                scene,
                high_lattice=high,
                coarse_lattice=coarse,
                future_ego_to_canonical=future_rel,
                history_observed_coarse=aligned.observed,
                native_shape_xyz=pcfg.grid.shape_hwd,
                native_origin_xyz_m=(pcfg.grid.x_min, pcfg.grid.y_min, pcfg.grid.z_min),
                native_voxel_size_xyz_m=pcfg.grid.voxel_size,
                tile_size_xyz=tile_size,
                free_label=int(pcfg.free_label),
            )
        final_t = protected_add_only(
            torch.from_numpy(pred_stack),
            static_world=torch.from_numpy(static.future_semantic),
            free_label=int(pcfg.free_label),
        )
        final = final_t.numpy().astype(np.uint8)
        total_query += int(static.query_voxels)
        total_tiles += int(static.active_tiles)
        total_oob += int(static.out_of_bounds_voxels)
        total_history_oob += int(aligned.out_of_bounds_samples)

        masks = static_subset_masks(
            high_lattice=high,
            history_observed=np.asarray(raw["history_observed"], dtype=bool),
            history_ego_to_canonical=hist_rel,
            future_ego_to_canonical=future_rel,
            future_observed=future_obs,
            future_gt_semantic=np.asarray(raw["future_gt_occ"], dtype=np.uint8),
            native_origin_xyz_m=(pcfg.grid.x_min, pcfg.grid.y_min, pcfg.grid.z_min),
            native_voxel_size_xyz_m=pcfg.grid.voxel_size,
            free_label=int(pcfg.free_label),
        )
        for fi in range(6):
            gt = np.asarray(raw["future_gt_occ"][fi], dtype=np.uint8)
            _update_subset(
                subset["history_seen_t0_missing"],
                masks["history_seen_t0_missing"][fi],
                pred_stack[fi], final[fi], gt, int(pcfg.free_label),
            )
            _update_subset(
                subset["never_seen_static_domain"],
                masks["never_seen_static_domain"][fi],
                pred_stack[fi], final[fi], gt, int(pcfg.free_label),
            )

        moving_rows = _moving_support_sequence(
            source, w, grid=pcfg.grid, workers=int(a.alignment_workers)
        )
        for hi, _ in enumerate(HORIZONS):
            gt = np.asarray(raw["future_gt_occ"][hi], dtype=np.uint8)
            _update_many(
                raw_by_variant,
                hi,
                {"v18": pred_stack[hi], "v20_static": final[hi]},
                gt,
                moving_rows[hi],
                int(pcfg.free_label),
            )

        if wi == 1 or wi % 25 == 0 or wi == len(records):
            elapsed = max(time.perf_counter() - started, 1e-9)
            print(f"v20_static_eval {wi}/{len(records)} rate={wi/elapsed:.3f} win/s", flush=True)

    metrics = {v: _finalize(raw_by_variant[v]) for v in VARIANTS}
    elapsed = max(time.perf_counter() - started, 1e-9)
    result = {
        "protocol": PROTOCOL,
        "num_windows": len(records),
        "population_fingerprint_sha256": population_fingerprint(records),
        "base_checkpoint": str(Path(a.base_checkpoint).resolve()),
        "base_checkpoint_epoch": int(base_ck.get("epoch", -1)),
        "v20_checkpoint": str(Path(a.v20_checkpoint).resolve()),
        "future_gt_used_for_prediction": False,
        "static_memory_unconditional_output": False,
        "metrics": metrics,
        "delta_v20_static_vs_v18": _delta(metrics["v20_static"], metrics["v18"]),
        "static_capability_subsets": {k: _finish_subset(v) for k, v in subset.items()},
        "geometry_audit": {
            "query_voxels": int(total_query),
            "active_tiles": int(total_tiles),
            "render_out_of_bounds_voxels": int(total_oob),
            "history_observed_out_of_bounds_samples": int(total_history_oob),
        },
        "raw_counts": {
            v: {k: np.asarray(x).tolist() for k, x in raw_by_variant[v].items()}
            for v in VARIANTS
        },
        "timing": {
            "elapsed_s": float(elapsed),
            "windows_per_s": float(len(records) / elapsed),
            "note": "end-to-end evaluation wall time; dedicated benchmark reports decomposed latency",
        },
    }
    op = Path(a.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "metrics": metrics,
        "delta": result["delta_v20_static_vs_v18"],
        "subsets": result["static_capability_subsets"],
        "geometry_audit": result["geometry_audit"],
    }, indent=2))
    print(f"saved {op}")


if __name__ == "__main__":
    main()
