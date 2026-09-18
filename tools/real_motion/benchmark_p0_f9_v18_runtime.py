#!/usr/bin/env python3
"""Frozen runtime benchmark for the final Clean-E14 V18 SE(2) forecaster.

The benchmark exposes multiple timing boundaries instead of hiding preparation
costs inside one FPS number:

1) neural_model:
   cached causal source tensors resident on GPU -> six-horizon motion outputs.

2) cached_representation_forecast_6frames (main OccFM-comparable boundary):
   cached causal source representation + precomputed deterministic Strong/KTA
   prior -> six dense future occupancy grids.  It includes Clean-E14 forward,
   SE(2) rigid rendering and hard-A1 composition, while excluding disk I/O, GT
   and metrics.  OccFM's released cfm_eval timer likewise starts after cached
   latent preparation and includes its generative sampling + decoder.

3) full_causal_forecast_in_memory_6frames:
   the above plus deterministic Strong/KTA prior reconstruction.

4) source_extract_match:
   t-1/t0 occupancy + poses -> Strong connected components and causal matching,
   reported separately as representation/preparation cost.

The first measured window performs exactness checks against the historical
Strong-W2Det and rigid-raster implementations.  This file changes no model or
prediction contract; it is measurement infrastructure only.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from real_motion.geometry import relative_transform
from real_motion.metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS
from real_motion.rigid_transport import (
    RasterizedRigidComponent,
    _deduplicate_indices,
    _indices_to_ego_xyz,
    _metric_to_indices,
    _transform_points as rigid_transform_points,
    compose_component_replacements_in_input_order,
    rasterize_rigid_component,
)
from real_motion.runtime_config import add_config_args, load_runtime_config, make_prepare_config
from real_motion.runtime_fastpath import (
    baseline_clear_mask,
    component_lists_equal,
    compose_component_replacements_fast_exact,
    extract_instances_cropped_exact,
    majority_fill_sparse_5x5x1,
)
from real_motion.strong_w2det import (
    StrongW2DetConfig,
    _in_grid,
    _metric_to_voxel,
    _transform_points as strong_transform_points,
    extract_instances,
    inverse_warp,
    majority_fill,
    match_instances,
    strong_w2det_sequence,
)
from real_motion.v18_two_wheel_diagnostic import renderer_yaw_delta
from tools.real_motion import eval_p0_f9_v18_full_validation as mid
from tools.real_motion.eval_p0_f9_v17_local_stwm import (
    t0_xy_to_world_preserve_source_z,
    window_from_record,
)
from tools.real_motion.train_p0_f9_v18_se2_clean import PROTOCOL as CLEAN_PROTOCOL

PROTOCOL = "p0_f9_v18_clean_runtime_benchmark_v1"
FUTURE_FRAMES = 6


class CachedSource(mid.base.NuScenesWindowSource):
    """Small LRU cache only for benchmark preparation; not part of timed regions."""

    from functools import lru_cache

    @lru_cache(maxsize=256)
    def load_semantics(self, scene_name, token):
        return super().load_semantics(scene_name, token)

    @lru_cache(maxsize=1024)
    def pose(self, token):
        return super().pose(token)


def _summary_ms(values):
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return {}
    return {
        "n": int(x.size),
        "mean_ms": float(x.mean()),
        "std_ms": float(x.std(ddof=1)) if x.size > 1 else 0.0,
        "median_ms": float(np.median(x)),
        "p90_ms": float(np.quantile(x, 0.90)),
        "p95_ms": float(np.quantile(x, 0.95)),
        "min_ms": float(x.min()),
        "max_ms": float(x.max()),
        "windows_per_s_from_mean": float(1000.0 / x.mean()),
        "future_frames_per_s_from_mean": float(6000.0 / x.mean()),
    }


def _precompute_source_world(current, current_pose, grid):
    rows = []
    for comp in current:
        idx = np.asarray(comp["voxel_indices"], dtype=np.int64)
        pts_ego = _indices_to_ego_xyz(idx, grid)
        rows.append(rigid_transform_points(current_pose, pts_ego))
    return rows


def _fast_rasterize_from_world(
    source_points_world,
    class_id,
    source_voxel_count,
    source_center_world,
    target_center_world,
    yaw_delta_rad,
    world_to_future,
    grid,
):
    pts_world = np.asarray(source_points_world, dtype=np.float64)
    if len(pts_world) == 0:
        return RasterizedRigidComponent(
            int(class_id), np.zeros((0, 3), dtype=np.int64), int(source_voxel_count)
        )
    source_center = np.asarray(source_center_world, dtype=np.float64)
    target_center = np.asarray(target_center_world, dtype=np.float64)
    theta = float(yaw_delta_rad)
    c, s = math.cos(theta), math.sin(theta)
    rel = pts_world[:, :2] - source_center[None, :2]
    moved_world = pts_world.copy()
    moved_world[:, 0] = target_center[0] + c * rel[:, 0] - s * rel[:, 1]
    moved_world[:, 1] = target_center[1] + s * rel[:, 0] + c * rel[:, 1]
    moved_future = rigid_transform_points(world_to_future, moved_world)
    dst_idx, valid = _metric_to_indices(moved_future, grid)
    dst_idx = _deduplicate_indices(dst_idx[valid], grid)
    return RasterizedRigidComponent(int(class_id), dst_idx, int(source_voxel_count))


def _strong_all_horizons(
    current_semantics,
    current_pose,
    future_poses,
    current,
    velocities,
    source_world_points,
    *,
    frame_dt_s,
    grid,
    cfg,
):
    """Exact Strong-W2Det for all six horizons while reusing source extraction."""
    sem0 = np.asarray(current_semantics)
    dyn = np.isin(sem0, np.asarray(DYNAMIC_CLASS_IDS, dtype=sem0.dtype))
    static_src = sem0.copy()
    static_src[dyn] = int(cfg.free_label)

    covered = np.zeros_like(dyn, dtype=bool)
    for comp in current:
        idx = np.asarray(comp["voxel_indices"], dtype=np.int64)
        covered[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    rest = dyn & ~covered
    if bool(rest.any()):
        ridx = np.argwhere(rest)
        origin = np.asarray([grid.x_min, grid.y_min, grid.z_min], dtype=np.float64)
        step = np.asarray(grid.voxel_size, dtype=np.float64)
        rest_ego = origin + (ridx.astype(np.float64) + 0.5) * step
        rest_world = strong_transform_points(current_pose, rest_ego)
        rest_labels = sem0[ridx[:, 0], ridx[:, 1], ridx[:, 2]]
    else:
        rest_world = np.zeros((0, 3), dtype=np.float64)
        rest_labels = np.zeros((0,), dtype=sem0.dtype)

    outputs, baselines = [], []
    for hi, future_pose in enumerate(future_poses):
        future_pose = np.asarray(future_pose, dtype=np.float64)
        t_future_from_current = relative_transform(current_pose, future_pose)
        static_dst, known = inverse_warp(
            static_src, t_future_from_current, grid, int(cfg.free_label)
        )
        out = majority_fill_sparse_5x5x1(
            static_dst,
            ~known,
            kernel=cfg.fill_kernel,
            min_fraction=cfg.fill_min_fraction,
        )

        dt = (hi + 1) * float(frame_dt_s)
        world_parts, label_parts, slices = [], [], []
        cursor = 0
        for j, comp in enumerate(current):
            pts = np.asarray(source_world_points[j], dtype=np.float64)
            v = np.asarray(velocities.get(j, np.zeros(3)), dtype=np.float64)
            moved = pts + v[None] * dt
            world_parts.append(moved)
            label_parts.append(
                np.full(len(moved), int(comp["class_id"]), dtype=sem0.dtype)
            )
            slices.append(slice(cursor, cursor + len(moved)))
            cursor += len(moved)
        if len(rest_world):
            world_parts.append(rest_world)
            label_parts.append(rest_labels)

        baseline_components = []
        if world_parts:
            moved_world = np.concatenate(world_parts, axis=0)
            labels = np.concatenate(label_parts, axis=0)
            world_to_future = np.linalg.inv(future_pose)
            moved_future = strong_transform_points(world_to_future, moved_world)
            idx_all = _metric_to_voxel(moved_future, grid)
            valid_all = _in_grid(idx_all, grid)
            idx = idx_all[valid_all]
            labels_valid = labels[valid_all]
            out[idx[:, 0], idx[:, 1], idx[:, 2]] = labels_valid
            for comp, sl in zip(current, slices):
                cidx = idx_all[sl][valid_all[sl]]
                baseline_components.append(
                    RasterizedRigidComponent(
                        int(comp["class_id"]), cidx, int(len(comp["voxel_indices"]))
                    )
                )

        outputs.append(out.astype(np.uint8, copy=False))
        baselines.append(baseline_components)
    return outputs, baselines


def _gpu_inputs(rec, device):
    return {
        "features": rec["features"].float().to(device),
        "tube": rec["local_semantic_tube"].to(device),
        "kta": rec["kta_displacement_xy_m"].float().to(device),
        "frame_motion": rec["frame_motion_features"].float().to(device),
        "source_mask": rec["target_source_mask_tube"].to(device),
    }


def _model_forward(model, gi, device):
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        return model(
            gi["features"], gi["tube"], gi["kta"],
            gi["frame_motion"], gi["source_mask"],
        )


def _prepare_record(rec, source, pcfg, strong_cfg, device):
    w = window_from_record(rec)
    scene = str(w.scene_name)
    current_token = str(w.t0_token)
    previous_token = str(w.history_tokens[-2])
    current_sem = np.asarray(source.load_semantics(scene, current_token), dtype=np.uint8)
    previous_sem = np.asarray(source.load_semantics(scene, previous_token), dtype=np.uint8)
    current_pose = np.asarray(source.pose(current_token), dtype=np.float64)
    previous_pose = np.asarray(source.pose(previous_token), dtype=np.float64)
    future_poses = [
        np.asarray(source.pose(str(tok)), dtype=np.float64) for tok in w.future_tokens
    ]
    current = extract_instances(current_sem, current_pose, grid=pcfg.grid, cfg=strong_cfg)
    previous = extract_instances(previous_sem, previous_pose, grid=pcfg.grid, cfg=strong_cfg)
    velocities = match_instances(
        previous, current, float(pcfg.frame_dt_s),
        max_speed_mps=strong_cfg.max_match_speed_mps,
    )
    if len(current) != int(rec["features"].shape[0]):
        raise RuntimeError(f"{rec['sample_id']}: Strong/source count mismatch")
    got = [int(c["class_id"]) for c in current]
    exp = [int(x) for x in rec["source_class_id"].tolist()]
    if got != exp:
        raise RuntimeError(f"{rec['sample_id']}: Strong/source order mismatch")
    source_world_points = _precompute_source_world(current, current_pose, pcfg.grid)
    # Frozen deterministic prior is prepared outside the OccFM-comparable timing
    # boundary, analogous to OccFM loading/preparing its cached latent input.
    anchors, baseline_by_hi = _strong_all_horizons(
        current_sem, current_pose, future_poses, current, velocities,
        source_world_points,
        frame_dt_s=float(pcfg.frame_dt_s), grid=pcfg.grid, cfg=strong_cfg,
    )
    baseline_clear_by_hi = [
        baseline_clear_mask(rows, grid=pcfg.grid) for rows in baseline_by_hi
    ]
    world_to_future = [
        np.linalg.inv(np.asarray(p, dtype=np.float64)) for p in future_poses
    ]
    return {
        "rec": rec,
        "window": w,
        "scene": scene,
        "current_sem": current_sem,
        "previous_sem": previous_sem,
        "current_pose": current_pose,
        "previous_pose": previous_pose,
        "future_poses": future_poses,
        "current": current,
        "previous": previous,
        "velocities": velocities,
        "source_world_points": source_world_points,
        "anchors": anchors,
        "baseline_by_hi": baseline_by_hi,
        "baseline_clear_by_hi": baseline_clear_by_hi,
        "world_to_future": world_to_future,
        "gpu": _gpu_inputs(rec, device),
    }


def _forecast_once(model, state, pcfg, strong_cfg, device):
    rec = state["rec"]
    current = state["current"]
    current_pose = state["current_pose"]
    future_poses = state["future_poses"]
    source_world_points = state["source_world_points"]

    anchors = state["anchors"]
    baseline_by_hi = state["baseline_by_hi"]

    out = _model_forward(model, state["gpu"], device)
    pred_res = out["residual_xy_m"].float().cpu().numpy()
    pred_yaw = out["yaw_delta_rad"].float().cpu().numpy()

    preds = []
    for hi in range(FUTURE_FRAMES):
        world_to_future = state["world_to_future"][hi]
        repl = []
        for i, comp in enumerate(current):
            cid = int(comp["class_id"])
            src_center = np.asarray(comp["centroid_world"], dtype=np.float64)
            xy = rec["anchors_xy_t0_m"][i, hi].numpy() + pred_res[i, hi]
            center = t0_xy_to_world_preserve_source_z(xy, src_center, current_pose)
            repl.append(
                _fast_rasterize_from_world(
                    source_world_points[i], cid, len(comp["voxel_indices"]),
                    src_center, center,
                    renderer_yaw_delta(cid, pred_yaw[i, hi], zero_two_wheel_yaw=False),
                    world_to_future, pcfg.grid,
                )
            )
        pred = compose_component_replacements_fast_exact(
            anchors[hi], baseline_by_hi[hi], repl,
            dynamic_class_ids=DYNAMIC_CLASS_IDS,
            free_label=int(pcfg.free_label), grid=pcfg.grid,
            precomputed_clear_mask=state["baseline_clear_by_hi"][hi],
        )
        preds.append(pred)
    return preds


def _forecast_once_with_prior_rebuild(model, state, pcfg, strong_cfg, device):
    """Full in-memory causal forecast including deterministic Strong/KTA rebuild."""
    anchors, baseline_by_hi = _strong_all_horizons(
        state["current_sem"], state["current_pose"], state["future_poses"],
        state["current"], state["velocities"], state["source_world_points"],
        frame_dt_s=float(pcfg.frame_dt_s), grid=pcfg.grid, cfg=strong_cfg,
    )
    clear_by_hi = [baseline_clear_mask(rows, grid=pcfg.grid) for rows in baseline_by_hi]
    old_a, old_b, old_c = (
        state["anchors"], state["baseline_by_hi"], state["baseline_clear_by_hi"]
    )
    state["anchors"], state["baseline_by_hi"], state["baseline_clear_by_hi"] = (
        anchors, baseline_by_hi, clear_by_hi
    )
    try:
        return _forecast_once(model, state, pcfg, strong_cfg, device)
    finally:
        state["anchors"], state["baseline_by_hi"], state["baseline_clear_by_hi"] = (
            old_a, old_b, old_c
        )


def _time_prior_rebuild(state, pcfg, strong_cfg):
    t0 = time.perf_counter()
    _strong_all_horizons(
        state["current_sem"], state["current_pose"], state["future_poses"],
        state["current"], state["velocities"], state["source_world_points"],
        frame_dt_s=float(pcfg.frame_dt_s), grid=pcfg.grid, cfg=strong_cfg,
    )
    return (time.perf_counter() - t0) * 1000.0


def _exactness_check(model, state, pcfg, strong_cfg, device):
    fast_cur = extract_instances_cropped_exact(
        state["current_sem"], state["current_pose"], grid=pcfg.grid, cfg=strong_cfg
    )
    fast_prev = extract_instances_cropped_exact(
        state["previous_sem"], state["previous_pose"], grid=pcfg.grid, cfg=strong_cfg
    )
    if not component_lists_equal(fast_cur, state["current"]):
        raise RuntimeError("runtime fast current-component extraction mismatch")
    if not component_lists_equal(fast_prev, state["previous"]):
        raise RuntimeError("runtime fast previous-component extraction mismatch")

    history2 = np.stack([state["previous_sem"], state["current_sem"]], axis=0)
    ref_anchor = strong_w2det_sequence(
        history2,
        [state["previous_pose"], state["current_pose"]],
        state["future_poses"],
        frame_dt_s=float(pcfg.frame_dt_s), grid=pcfg.grid, cfg=strong_cfg,
    )
    fast_anchor, fast_base = _strong_all_horizons(
        state["current_sem"], state["current_pose"], state["future_poses"],
        state["current"], state["velocities"], state["source_world_points"],
        frame_dt_s=float(pcfg.frame_dt_s), grid=pcfg.grid, cfg=strong_cfg,
    )
    for hi in range(FUTURE_FRAMES):
        if not np.array_equal(ref_anchor[hi], fast_anchor[hi]):
            n = int(np.count_nonzero(ref_anchor[hi] != fast_anchor[hi]))
            raise RuntimeError(f"runtime fast Strong mismatch hi={hi} voxels={n}")

    out = _model_forward(model, state["gpu"], device)
    pred_res = out["residual_xy_m"].float().cpu().numpy()
    pred_yaw = out["yaw_delta_rad"].float().cpu().numpy()
    if state["current"]:
        i = 0
        hi = 5
        comp = state["current"][i]
        cid = int(comp["class_id"])
        src_center = np.asarray(comp["centroid_world"], dtype=np.float64)
        xy = state["rec"]["anchors_xy_t0_m"][i, hi].numpy() + pred_res[i, hi]
        center = t0_xy_to_world_preserve_source_z(
            xy, src_center, state["current_pose"]
        )
        yaw = renderer_yaw_delta(cid, pred_yaw[i, hi], zero_two_wheel_yaw=False)
        fast_comp = _fast_rasterize_from_world(
            state["source_world_points"][i], cid, len(comp["voxel_indices"]),
            src_center, center, yaw, np.linalg.inv(state["future_poses"][hi]),
            pcfg.grid,
        )
        ref_comp = rasterize_rigid_component(
            comp["voxel_indices"], cid, state["current_pose"], state["future_poses"][hi],
            source_center_world=src_center, target_center_world=center,
            yaw_delta_rad=yaw, grid=pcfg.grid,
        )
        if not np.array_equal(fast_comp.voxel_indices, ref_comp.voxel_indices):
            raise RuntimeError("runtime fast rigid raster mismatch")
    # Full A1 output equivalence of the optimized compositor against the frozen
    # reference compositor, using identical predicted replacement components.
    out = _model_forward(model, state["gpu"], device)
    pred_res = out["residual_xy_m"].float().cpu().numpy()
    pred_yaw = out["yaw_delta_rad"].float().cpu().numpy()
    for hi in range(FUTURE_FRAMES):
        repl = []
        for i, comp in enumerate(state["current"]):
            cid = int(comp["class_id"])
            src_center = np.asarray(comp["centroid_world"], dtype=np.float64)
            xy = state["rec"]["anchors_xy_t0_m"][i, hi].numpy() + pred_res[i, hi]
            center = t0_xy_to_world_preserve_source_z(
                xy, src_center, state["current_pose"]
            )
            repl.append(_fast_rasterize_from_world(
                state["source_world_points"][i], cid, len(comp["voxel_indices"]),
                src_center, center,
                renderer_yaw_delta(cid, pred_yaw[i, hi], zero_two_wheel_yaw=False),
                state["world_to_future"][hi], pcfg.grid,
            ))
        ref_pred = compose_component_replacements_in_input_order(
            state["anchors"][hi], state["baseline_by_hi"][hi], repl,
            dynamic_class_ids=DYNAMIC_CLASS_IDS,
            free_label=int(pcfg.free_label), grid=pcfg.grid,
        )
        fast_pred = compose_component_replacements_fast_exact(
            state["anchors"][hi], state["baseline_by_hi"][hi], repl,
            dynamic_class_ids=DYNAMIC_CLASS_IDS,
            free_label=int(pcfg.free_label), grid=pcfg.grid,
            precomputed_clear_mask=state["baseline_clear_by_hi"][hi],
        )
        if not np.array_equal(ref_pred, fast_pred):
            neq = int(np.count_nonzero(ref_pred != fast_pred))
            raise RuntimeError(f"runtime fast A1 compositor mismatch hi={hi} voxels={neq}")
    print(
        "RUNTIME EXACTNESS: components + Strong all-6 + rigid raster + A1 PASS",
        flush=True,
    )


def _time_neural(model, state, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = _model_forward(model, state["gpu"], device)
        end.record()
        torch.cuda.synchronize(device)
        _ = out["residual_xy_m"]
        return float(start.elapsed_time(end))
    t0 = time.perf_counter()
    _model_forward(model, state["gpu"], device)
    return (time.perf_counter() - t0) * 1000.0


def _time_forecast(model, state, pcfg, strong_cfg, device):
    """OccFM-comparable core: prepared causal representation -> 6 dense grids."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    _forecast_once(model, state, pcfg, strong_cfg, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return (time.perf_counter() - t0) * 1000.0


def _time_full_in_memory(model, state, pcfg, strong_cfg, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    _forecast_once_with_prior_rebuild(model, state, pcfg, strong_cfg, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return (time.perf_counter() - t0) * 1000.0


def _time_source_extract(state, pcfg, strong_cfg):
    t0 = time.perf_counter()
    cur = extract_instances_cropped_exact(
        state["current_sem"], state["current_pose"], grid=pcfg.grid, cfg=strong_cfg
    )
    prv = extract_instances_cropped_exact(
        state["previous_sem"], state["previous_pose"], grid=pcfg.grid, cfg=strong_cfg
    )
    match_instances(
        prv, cur, float(pcfg.frame_dt_s),
        max_speed_mps=strong_cfg.max_match_speed_mps,
    )
    return (time.perf_counter() - t0) * 1000.0


def _measure_model_flops(model, state, device):
    """Best-effort neural-forward FLOPs; scope excludes deterministic transport."""
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with FlopCounterMode(display=False) as mode:
            _model_forward(model, state["gpu"], device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        flops = int(mode.get_total_flops())
        return {
            "method": "torch.utils.flop_counter.FlopCounterMode",
            "model_forward_flops": flops,
            "model_forward_gflops": float(flops / 1e9),
            "scope": "Clean neural_model only; deterministic Strong/KTA/raster not included",
        }
    except Exception as primary:
        if not hasattr(torch, "profiler"):
            return {"error": repr(primary)}
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        try:
            with torch.profiler.profile(activities=activities, with_flops=True) as prof:
                _model_forward(model, state["gpu"], device)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
            flops = sum(int(getattr(x, "flops", 0) or 0) for x in prof.key_averages())
            return {
                "method": "torch.profiler.with_flops_fallback",
                "model_forward_flops": int(flops),
                "model_forward_gflops": float(flops / 1e9),
                "scope": "Clean neural_model only; supported operators only",
                "warning": "fallback profiler may under-count unsupported operators",
                "primary_error": repr(primary),
            }
        except Exception as secondary:
            return {"primary_error": repr(primary), "fallback_error": repr(secondary)}


def main():
    p = argparse.ArgumentParser()
    add_config_args(p)
    p.add_argument("--val-cache", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataroot", required=True)
    p.add_argument("--info-pkl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--warmup-windows", type=int, default=20)
    p.add_argument("--measure-windows", type=int, default=200)
    p.add_argument("--seed", type=int, default=20260918)
    p.add_argument("--device", default="cuda")
    p.add_argument("--profile-flops", action="store_true")
    a = p.parse_args()

    cfg = load_runtime_config(a.config, a.override)
    pcfg = make_prepare_config(cfg)
    _, records = mid.base.load_cache(a.val_cache)
    if not records:
        raise RuntimeError("validation cache is empty")
    if int(a.measure_windows) <= 0 or int(a.warmup_windows) < 0:
        raise ValueError("invalid warmup/measure counts")

    device = torch.device(
        a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    ck, model, _ = mid._load_model(a.checkpoint, CLEAN_PROTOCOL, device)
    # CLEAN_PROTOCOL is the authoritative compatibility check performed by
    # _load_model.  Historical Clean-E14 checkpoints may predate the
    # training_mode metadata field, while Balanced checkpoints deliberately
    # reuse the same model protocol.  Reject known non-main variants/modes, but
    # do not reject a valid historical Clean checkpoint merely because that
    # optional metadata key is absent.
    training_mode = str(ck.get("training_mode") or "")
    variant = str(ck.get("variant") or "")
    if "balanced" in training_mode.lower() or "balanced" in variant.lower():
        raise RuntimeError(
            f"runtime benchmark refuses balanced checkpoint: "
            f"training_mode={training_mode!r} variant={variant!r}"
        )
    allowed_main_modes = {
        "",
        "clean_one_stage_from_scratch_v1",
        "clean_one_stage_from_scratch_v1_tail_continuation",
    }
    if training_mode not in allowed_main_modes:
        raise RuntimeError(
            f"unexpected frozen-main checkpoint training_mode={training_mode!r}"
        )
    if training_mode.endswith("_tail_continuation") and int(ck.get("epoch", -1)) != 14:
        raise RuntimeError(
            "formal runtime benchmark expects the frozen Clean-E14 tail checkpoint; "
            f"got epoch={ck.get('epoch')!r}"
        )
    print(
        "RUNTIME CHECKPOINT "
        + json.dumps({
            "protocol": ck.get("protocol"),
            "training_mode": ck.get("training_mode"),
            "variant": ck.get("variant"),
            "epoch": ck.get("epoch"),
            "global_step": ck.get("global_step"),
        }),
        flush=True,
    )

    rng = np.random.default_rng(int(a.seed))
    total_need = min(
        len(records), int(a.warmup_windows) + int(a.measure_windows)
    )
    ids = np.sort(rng.choice(len(records), size=total_need, replace=False))
    selected = [records[int(i)] for i in ids]

    source = CachedSource(a.dataroot, info_pkl=a.info_pkl, verbose=False)
    strong_cfg = StrongW2DetConfig(free_label=int(pcfg.free_label))
    prepared = [
        _prepare_record(r, source, pcfg, strong_cfg, device) for r in selected
    ]
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    if prepared:
        _exactness_check(model, prepared[0], pcfg, strong_cfg, device)

    nw = min(int(a.warmup_windows), len(prepared))
    for state in prepared[:nw]:
        _model_forward(model, state["gpu"], device)
        _forecast_once(model, state, pcfg, strong_cfg, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    measured = prepared[nw:]
    neural_ms, core_ms, prior_ms, full_ms, source_ms = [], [], [], [], []
    source_counts = []
    for j, state in enumerate(measured, start=1):
        source_counts.append(len(state["current"]))
        neural_ms.append(_time_neural(model, state, device))
        core_ms.append(_time_forecast(model, state, pcfg, strong_cfg, device))
        prior_ms.append(_time_prior_rebuild(state, pcfg, strong_cfg))
        full_ms.append(_time_full_in_memory(model, state, pcfg, strong_cfg, device))
        source_ms.append(_time_source_extract(state, pcfg, strong_cfg))
        if j == 1 or j % 25 == 0 or j == len(measured):
            print(
                f"runtime {j}/{len(measured)} "
                f"model={neural_ms[-1]:.3f}ms "
                f"core6={core_ms[-1]:.3f}ms "
                f"full6={full_ms[-1]:.3f}ms "
                f"sources={source_counts[-1]}",
                flush=True,
            )

    params = sum(int(x.numel()) for x in model.parameters())
    trainable = sum(int(x.numel()) for x in model.parameters() if x.requires_grad)
    peak_mem = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )
    flops = None
    if bool(a.profile_flops) and measured:
        median_idx = int(np.argsort(np.asarray(source_counts))[len(source_counts) // 2])
        flops = _measure_model_flops(model, measured[median_idx], device)

    result = {
        "protocol": PROTOCOL,
        "checkpoint": str(Path(a.checkpoint).resolve()),
        "checkpoint_epoch": int(ck.get("epoch", -1)),
        "checkpoint_global_step": int(ck.get("global_step", -1)),
        "val_cache": str(Path(a.val_cache).resolve()),
        "device": str(device),
        "gpu_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "amp": "bf16" if device.type == "cuda" else "none",
        "future_frames_per_window": 6,
        "parameters": params,
        "trainable_parameters": trainable,
        "peak_cuda_memory_bytes": peak_mem,
        "warmup_windows": nw,
        "measured_windows": len(measured),
        "selection_seed": int(a.seed),
        "source_count": {
            "mean": float(np.mean(source_counts)) if source_counts else float("nan"),
            "median": float(np.median(source_counts)) if source_counts else float("nan"),
            "min": int(min(source_counts)) if source_counts else 0,
            "max": int(max(source_counts)) if source_counts else 0,
        },
        "timing": {
            "neural_model": _summary_ms(neural_ms),
            "cached_representation_forecast_6frames": _summary_ms(core_ms),
            "strong_kta_prior_rebuild_6frames": _summary_ms(prior_ms),
            "full_causal_forecast_in_memory_6frames": _summary_ms(full_ms),
            "source_extract_match": _summary_ms(source_ms),
        },
        "timing_boundaries": {
            "neural_model": (
                "cached causal source tensors resident on GPU -> all six motion outputs"
            ),
            "cached_representation_forecast_6frames": (
                "frozen cached causal source representation + precomputed deterministic "
                "Strong/KTA prior -> six dense future occupancy grids; includes Clean "
                "forward, predicted SE2 rigid render and hard-A1 compose; excludes "
                "disk I/O/GT/metrics. This is the closest boundary to OccFM cfm_eval, "
                "which starts after cached latent preparation."
            ),
            "strong_kta_prior_rebuild_6frames": (
                "prepared current sources/velocities + future ego poses -> deterministic "
                "Strong/KTA dense prior for all six horizons"
            ),
            "full_causal_forecast_in_memory_6frames": (
                "prepared source representation but without precomputed dense prior -> "
                "Strong/KTA prior + Clean forward + SE2 render + A1 compose for six frames"
            ),
            "source_extract_match": (
                "in-memory t-1/t0 occupancy and poses -> Strong components + matching; "
                "reported separately as causal representation preparation"
            ),
            "occfm_comparison_note": (
                "OccFM released cfm_eval uses CUDA events after cached latent preparation "
                "and includes flow sampling + decoder. For a hardware-matched main-table "
                "comparison use cached_representation_forecast_6frames and report "
                "neural_model plus full_causal_forecast_in_memory_6frames as transparency "
                "rows. Never derive FPS from full validation wall-clock."
            ),
        },
        "model_flops": flops,
    }
    op = Path(a.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\n=== V18 FROZEN RUNTIME BENCHMARK ===")
    print(json.dumps(result, indent=2))
    print(f"saved {op}")


if __name__ == "__main__":
    main()
