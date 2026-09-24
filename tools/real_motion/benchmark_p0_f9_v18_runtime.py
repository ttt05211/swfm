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
from real_motion.motion_transport import world_points_to_t0
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
    baseline_clear_flat_indices,
    baseline_clear_mask,
    component_lists_equal,
    compose_component_replacements_fast_exact,
    extract_instances_cropped_exact,
    inverse_warp_sequence_cuda_exact,
    majority_fill_cuda_exact,
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

PROTOCOL = "p0_f9_v18_clean_runtime_benchmark_v3"
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


def _occworld_style_summary(
    encode_ms,
    autoreg_ms,
    *,
    future_frames: int = FUTURE_FRAMES,
):
    """Paired OccWorld-style FPS summary.

    OccWorld defines per-frame time as:
        encode_time + autoregressive_time / N_future
    and reports FPS = 1 / per_frame_time.

    For V18 we intentionally use the most conservative currently measured
    online boundary available in this benchmark:
      encode_ms  := source extraction + causal matching from in-memory t-1/t0
      autoreg_ms := full causal six-frame forecast from the prepared source
                    representation, including Strong/KTA, Clean inference,
                    SE(2) rendering and final dense A1 composition.

    Important limitation: V18's learned source features/local semantic tube are
    loaded from the frozen causal cache, so this is an OccWorld-formula
    compatibility metric, not raw-occupancy end-to-end latency.  The JSON scope
    string records this explicitly.
    """
    enc = np.asarray(encode_ms, dtype=np.float64)
    aut = np.asarray(autoreg_ms, dtype=np.float64)
    if enc.shape != aut.shape:
        raise ValueError("OccWorld-style encode/autoreg timing arrays must align")
    if enc.size == 0:
        return {}
    nf = int(future_frames)
    if nf <= 0:
        raise ValueError("future_frames must be positive")
    per_frame = enc + aut / float(nf)
    return {
        "n": int(per_frame.size),
        "future_frames": nf,
        "formula": "encode_ms + autoreg_6frames_ms / future_frames",
        "encode_mean_ms": float(enc.mean()),
        "autoreg_6frames_mean_ms": float(aut.mean()),
        "per_frame_mean_ms": float(per_frame.mean()),
        "per_frame_median_ms": float(np.median(per_frame)),
        "per_frame_p90_ms": float(np.quantile(per_frame, 0.90)),
        "per_frame_p95_ms": float(np.quantile(per_frame, 0.95)),
        "fps_from_mean": float(1000.0 / per_frame.mean()),
        "scope": (
            "OccWorld-formula compatibility metric: encode=runtime source "
            "extraction+matching from in-memory t-1/t0 occupancy; "
            "autoreg=full causal six-frame Strong/KTA+Clean+SE2+A1 dense "
            "forecast. Frozen causal learned representation tensors "
            "(features/local tube/frame-motion/source-mask) remain cached."
        ),
        "comparison_note": (
            "Use only when explicitly stating the boundary. It is more "
            "conservative than cached-representation FPS because deterministic "
            "prior/render/compose are included, but it is not raw-occupancy "
            "end-to-end because learned representation construction is cached."
        ),
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


def _rasterize_all_sources_horizon(
    current,
    source_world_points,
    source_rel_xy,
    target_centers_world,
    yaw_values,
    world_to_future,
    grid,
):
    """Rasterize every source with one world->future transform per horizon.

    Object-centric rotation is still performed source-by-source in frozen input
    order; only the expensive affine transform and metric->voxel conversion are
    fused across sources.  Components are deduplicated independently exactly as
    in the reference renderer.
    """
    if not current:
        return []
    lengths = [int(len(x)) for x in source_world_points]
    slices = []
    cursor = 0
    for n in lengths:
        slices.append(slice(cursor, cursor + n))
        cursor += n
    if cursor == 0:
        return [
            RasterizedRigidComponent(int(comp["class_id"]), np.zeros((0, 3), dtype=np.int64), 0)
            for comp in current
        ]
    moved_world = np.concatenate(source_world_points, axis=0).copy()
    for i, (comp, sl) in enumerate(zip(current, slices)):
        theta = float(yaw_values[i])
        cc, ss = math.cos(theta), math.sin(theta)
        rel = source_rel_xy[i]
        target = np.asarray(target_centers_world[i], dtype=np.float64)
        moved_world[sl, 0] = target[0] + cc * rel[:, 0] - ss * rel[:, 1]
        moved_world[sl, 1] = target[1] + ss * rel[:, 0] + cc * rel[:, 1]
        # Planar contract: preserve each observed source point's world Z.
    moved_future = rigid_transform_points(world_to_future, moved_world)
    idx_all, valid_all = _metric_to_indices(moved_future, grid)
    out = []
    for comp, sl in zip(current, slices):
        cidx = _deduplicate_indices(idx_all[sl][valid_all[sl]], grid)
        out.append(
            RasterizedRigidComponent(
                int(comp["class_id"]), cidx, int(len(comp["voxel_indices"]))
            )
        )
    return out


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
    profile=None,
    runtime_device=None,
):
    """Bit-exact-gated Strong/KTA runtime path for all six horizons.

    Source extraction is reused across horizons; static hole filling is sparse;
    dynamic point clouds, labels and per-point velocities are concatenated once
    and only the horizon-dependent affine transport is repeated.
    """
    _t_pre = time.perf_counter() if profile is not None else None
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

    base_parts = []
    vel_parts = []
    label_parts = []
    slices = []
    cursor = 0
    for j, comp in enumerate(current):
        pts = np.asarray(source_world_points[j], dtype=np.float64)
        n = int(len(pts))
        v = np.asarray(velocities.get(j, np.zeros(3)), dtype=np.float64)
        base_parts.append(pts)
        vel_parts.append(np.broadcast_to(v[None], (n, 3)))
        label_parts.append(np.full(n, int(comp["class_id"]), dtype=sem0.dtype))
        slices.append(slice(cursor, cursor + n))
        cursor += n
    if len(rest_world):
        base_parts.append(rest_world)
        vel_parts.append(np.zeros_like(rest_world, dtype=np.float64))
        label_parts.append(rest_labels)

    if base_parts:
        base_world = np.concatenate(base_parts, axis=0)
        point_velocity = np.concatenate(vel_parts, axis=0)
        labels = np.concatenate(label_parts, axis=0)
    else:
        base_world = np.zeros((0, 3), dtype=np.float64)
        point_velocity = np.zeros((0, 3), dtype=np.float64)
        labels = np.zeros((0,), dtype=sem0.dtype)

    if profile is not None:
        profile["precompute_ms"] = profile.get("precompute_ms", 0.0) + (
            time.perf_counter() - _t_pre
        ) * 1000.0

    future_pose_arr = [np.asarray(p, dtype=np.float64) for p in future_poses]
    src_to_dst_seq = [
        relative_transform(current_pose, p) for p in future_pose_arr
    ]
    accelerated_inverse = None
    if runtime_device is not None and torch.device(runtime_device).type == "cuda":
        _t_inv_all = time.perf_counter() if profile is not None else None
        accelerated_inverse = inverse_warp_sequence_cuda_exact(
            static_src,
            src_to_dst_seq,
            grid=grid,
            free_label=int(cfg.free_label),
            device=runtime_device,
        )
        if profile is not None:
            profile["inverse_warp_ms"] = (
                profile.get("inverse_warp_ms", 0.0)
                + (time.perf_counter() - _t_inv_all) * 1000.0
            )

    outputs, baselines = [], []
    for hi, future_pose in enumerate(future_pose_arr):
        t_future_from_current = src_to_dst_seq[hi]
        if accelerated_inverse is None:
            _t = time.perf_counter() if profile is not None else None
            static_dst, known = inverse_warp(
                static_src, t_future_from_current, grid, int(cfg.free_label)
            )
            if profile is not None:
                profile["inverse_warp_ms"] = profile.get(
                    "inverse_warp_ms", 0.0
                ) + (time.perf_counter() - _t) * 1000.0
        else:
            static_dst, known = accelerated_inverse[hi]
        _t = time.perf_counter() if profile is not None else None
        if runtime_device is not None and torch.device(runtime_device).type == "cuda":
            out = majority_fill_cuda_exact(
                static_dst,
                ~known,
                kernel=cfg.fill_kernel,
                min_fraction=cfg.fill_min_fraction,
                device=runtime_device,
            )
        else:
            out = majority_fill_sparse_5x5x1(
                static_dst,
                ~known,
                kernel=cfg.fill_kernel,
                min_fraction=cfg.fill_min_fraction,
            )
        if profile is not None:
            profile["majority_fill_ms"] = profile.get("majority_fill_ms", 0.0) + (
                time.perf_counter() - _t
            ) * 1000.0

        _t = time.perf_counter() if profile is not None else None
        baseline_components = []
        if len(base_world):
            dt = (hi + 1) * float(frame_dt_s)
            moved_world = base_world + point_velocity * dt
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

        if profile is not None:
            profile["dynamic_transport_scatter_ms"] = profile.get(
                "dynamic_transport_scatter_ms", 0.0
            ) + (time.perf_counter() - _t) * 1000.0
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


def _stage_gpu_inputs(state, device):
    if state.get("gpu") is not None:
        raise RuntimeError("GPU inputs already staged for this state")
    state["gpu"] = _gpu_inputs(state["rec"], device)


def _release_gpu_inputs(state):
    state["gpu"] = None


def _model_forward(model, gi, device):
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        return model(
            gi["features"], gi["tube"], gi["kta"],
            gi["frame_motion"], gi["source_mask"],
        )


def _prepare_record(
    rec,
    source,
    pcfg,
    strong_cfg,
    device,
    *,
    raw_window=None,
):
    w = window_from_record(rec)
    scene = str(w.scene_name)
    current_token = str(w.t0_token)
    previous_token = str(w.history_tokens[-2])

    if raw_window is None:
        current_sem = np.asarray(
            source.load_semantics(scene, current_token),
            dtype=np.uint8,
        )
        previous_sem = np.asarray(
            source.load_semantics(scene, previous_token),
            dtype=np.uint8,
        )
        current_pose = np.asarray(
            source.pose(current_token),
            dtype=np.float64,
        )
        previous_pose = np.asarray(
            source.pose(previous_token),
            dtype=np.float64,
        )
        future_poses = [
            np.asarray(source.pose(str(tok)), dtype=np.float64)
            for tok in w.future_tokens
        ]
    else:
        history_occ = np.asarray(
            raw_window["history_occ"],
            dtype=np.uint8,
        )
        history_poses = np.asarray(
            raw_window["history_poses"],
            dtype=np.float64,
        )
        raw_future_poses = np.asarray(
            raw_window["future_poses"],
            dtype=np.float64,
        )
        if history_occ.shape[0] < 2 or history_poses.shape[0] < 2:
            raise ValueError("raw_window must contain at least two history frames")
        if raw_future_poses.shape[0] != FUTURE_FRAMES:
            raise ValueError("raw_window future pose count mismatch")
        current_sem = history_occ[-1]
        previous_sem = history_occ[-2]
        current_pose = history_poses[-1]
        previous_pose = history_poses[-2]
        future_poses = [
            np.asarray(x, dtype=np.float64)
            for x in raw_future_poses
        ]
    # Deployment/runtime path uses the cropped implementation.  Formal
    # exactness checks recompute the frozen full-grid reference extraction.
    current = extract_instances_cropped_exact(
        current_sem, current_pose, grid=pcfg.grid, cfg=strong_cfg
    )
    previous = extract_instances_cropped_exact(
        previous_sem, previous_pose, grid=pcfg.grid, cfg=strong_cfg
    )
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
    source_rel_xy = [
        np.asarray(pts, dtype=np.float64)[:, :2]
        - np.asarray(comp["centroid_world"], dtype=np.float64)[None, :2]
        for pts, comp in zip(source_world_points, current)
    ]
    # Preserve the frozen conversion exactly, but only once per source.
    source_z_t0 = np.asarray(
        [
            world_points_to_t0(
                np.asarray(comp["centroid_world"], dtype=np.float64)[None],
                current_pose,
            )[0, 2]
            for comp in current
        ],
        dtype=np.float64,
    )
    # Frozen deterministic prior is prepared outside the OccFM-comparable timing
    # boundary, analogous to OccFM loading/preparing its cached latent input.
    anchors, baseline_by_hi = _strong_all_horizons(
        current_sem, current_pose, future_poses, current, velocities,
        source_world_points,
        frame_dt_s=float(pcfg.frame_dt_s), grid=pcfg.grid, cfg=strong_cfg,
        runtime_device=device,
    )
    baseline_clear_by_hi = [
        baseline_clear_mask(rows, grid=pcfg.grid) for rows in baseline_by_hi
    ]
    baseline_clear_flat_by_hi = [
        baseline_clear_flat_indices(rows, grid=pcfg.grid) for rows in baseline_by_hi
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
        "source_rel_xy": source_rel_xy,
        "source_z_t0": source_z_t0,
        "anchors": anchors,
        "baseline_by_hi": baseline_by_hi,
        "baseline_clear_by_hi": baseline_clear_by_hi,
        "baseline_clear_flat_by_hi": baseline_clear_flat_by_hi,
        "world_to_future": world_to_future,
        # Staged one window at a time immediately before exactness/timing.
        # This keeps the reported peak CUDA memory representative of one
        # forecasting window instead of all selected benchmark windows.
        "gpu": None,
    }


def _target_world_from_xy_cached(xy_t0, source_z_t0, t0_pose):
    # Exact same final homogeneous matvec as
    # t0_xy_to_world_preserve_source_z, but the source t0-Z has already been
    # computed once during record preparation instead of recomputing inv(t0)
    # for every source at every horizon.
    p = np.asarray(
        [float(xy_t0[0]), float(xy_t0[1]), float(source_z_t0), 1.0],
        dtype=np.float64,
    )
    return (np.asarray(t0_pose, dtype=np.float64) @ p)[:3]


def _forecast_once(model, state, pcfg, strong_cfg, device, profile=None):
    rec = state["rec"]
    current = state["current"]
    current_pose = state["current_pose"]
    future_poses = state["future_poses"]
    source_world_points = state["source_world_points"]

    anchors = state["anchors"]
    baseline_by_hi = state["baseline_by_hi"]

    _t_model = time.perf_counter() if profile is not None else None
    out = _model_forward(model, state["gpu"], device)
    pred_res = out["residual_xy_m"].float().cpu().numpy()
    pred_yaw = out["yaw_delta_rad"].float().cpu().numpy()
    if profile is not None:
        profile["model_and_output_transfer_ms"] = (
            time.perf_counter() - _t_model
        ) * 1000.0

    preds = []
    for hi in range(FUTURE_FRAMES):
        world_to_future = state["world_to_future"][hi]
        _t_target = time.perf_counter() if profile is not None else None
        target_centers = []
        yaw_values = []
        for i, comp in enumerate(current):
            cid = int(comp["class_id"])
            src_center = np.asarray(comp["centroid_world"], dtype=np.float64)
            xy = rec["anchors_xy_t0_m"][i, hi].numpy() + pred_res[i, hi]
            target_centers.append(
                _target_world_from_xy_cached(
                    xy, state["source_z_t0"][i], current_pose
                )
            )
            yaw_values.append(
                renderer_yaw_delta(
                    cid, pred_yaw[i, hi], zero_two_wheel_yaw=False
                )
            )
        if profile is not None:
            profile["target_and_yaw_ms"] = profile.get(
                "target_and_yaw_ms", 0.0
            ) + (time.perf_counter() - _t_target) * 1000.0
        _t_raster = time.perf_counter() if profile is not None else None
        repl = _rasterize_all_sources_horizon(
            current,
            source_world_points,
            state["source_rel_xy"],
            target_centers,
            yaw_values,
            world_to_future,
            pcfg.grid,
        )
        if profile is not None:
            profile["rigid_raster_ms"] = profile.get(
                "rigid_raster_ms", 0.0
            ) + (time.perf_counter() - _t_raster) * 1000.0
        _t_comp = time.perf_counter() if profile is not None else None
        pred = compose_component_replacements_fast_exact(
            anchors[hi], baseline_by_hi[hi], repl,
            dynamic_class_ids=DYNAMIC_CLASS_IDS,
            free_label=int(pcfg.free_label), grid=pcfg.grid,
            precomputed_clear_flat_indices=state["baseline_clear_flat_by_hi"][hi],
        )
        if profile is not None:
            profile["a1_compose_ms"] = profile.get(
                "a1_compose_ms", 0.0
            ) + (time.perf_counter() - _t_comp) * 1000.0
        preds.append(pred)
    return preds


def _forecast_once_with_prior_rebuild(model, state, pcfg, strong_cfg, device):
    """Full in-memory causal forecast including deterministic Strong/KTA rebuild."""
    anchors, baseline_by_hi = _strong_all_horizons(
        state["current_sem"], state["current_pose"], state["future_poses"],
        state["current"], state["velocities"], state["source_world_points"],
        frame_dt_s=float(pcfg.frame_dt_s), grid=pcfg.grid, cfg=strong_cfg,
        runtime_device=device,
    )
    clear_flat_by_hi = [
        baseline_clear_flat_indices(rows, grid=pcfg.grid) for rows in baseline_by_hi
    ]
    old_a, old_b, old_d = (
        state["anchors"],
        state["baseline_by_hi"],
        state["baseline_clear_flat_by_hi"],
    )
    state["anchors"] = anchors
    state["baseline_by_hi"] = baseline_by_hi
    state["baseline_clear_flat_by_hi"] = clear_flat_by_hi
    try:
        return _forecast_once(model, state, pcfg, strong_cfg, device)
    finally:
        state["anchors"] = old_a
        state["baseline_by_hi"] = old_b
        state["baseline_clear_flat_by_hi"] = old_d


def _time_prior_rebuild(state, pcfg, strong_cfg, device):
    profile = {}
    t0 = time.perf_counter()
    _strong_all_horizons(
        state["current_sem"], state["current_pose"], state["future_poses"],
        state["current"], state["velocities"], state["source_world_points"],
        frame_dt_s=float(pcfg.frame_dt_s), grid=pcfg.grid, cfg=strong_cfg,
        profile=profile,
        runtime_device=device,
    )
    total = (time.perf_counter() - t0) * 1000.0
    profile["total_ms"] = total
    return total, profile


def _exactness_check(model, state, pcfg, strong_cfg, device):
    ref_cur = extract_instances(
        state["current_sem"], state["current_pose"], grid=pcfg.grid, cfg=strong_cfg
    )
    ref_prev = extract_instances(
        state["previous_sem"], state["previous_pose"], grid=pcfg.grid, cfg=strong_cfg
    )
    if not component_lists_equal(ref_cur, state["current"]):
        raise RuntimeError("runtime cropped current-component extraction mismatch")
    if not component_lists_equal(ref_prev, state["previous"]):
        raise RuntimeError("runtime cropped previous-component extraction mismatch")

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
        runtime_device=device,
    )
    for hi in range(FUTURE_FRAMES):
        if not np.array_equal(ref_anchor[hi], fast_anchor[hi]):
            n = int(np.count_nonzero(ref_anchor[hi] != fast_anchor[hi]))
            raise RuntimeError(f"runtime fast Strong mismatch hi={hi} voxels={n}")

        # The dense Strong anchor can be identical even if a per-source
        # baseline footprint differs. A1 CLEAR depends on those footprints, so
        # verify every source against the frozen rigid renderer as well.
        ref_baseline = []
        dt = (hi + 1) * float(pcfg.frame_dt_s)
        for i, comp in enumerate(state["current"]):
            cid = int(comp["class_id"])
            src_center = np.asarray(comp["centroid_world"], dtype=np.float64)
            v = np.asarray(state["velocities"].get(i, np.zeros(3)), dtype=np.float64)
            rb = rasterize_rigid_component(
                comp["voxel_indices"],
                cid,
                state["current_pose"],
                state["future_poses"][hi],
                source_center_world=src_center,
                target_center_world=src_center + v * dt,
                yaw_delta_rad=0.0,
                grid=pcfg.grid,
            )
            ref_baseline.append(rb)
            if i >= len(fast_base[hi]):
                raise RuntimeError(
                    f"runtime Strong baseline source-count mismatch hi={hi}"
                )
            fast_idx = _deduplicate_indices(
                np.asarray(fast_base[hi][i].voxel_indices, dtype=np.int64),
                pcfg.grid,
            )
            if not np.array_equal(rb.voxel_indices, fast_idx):
                raise RuntimeError(
                    f"runtime Strong baseline footprint mismatch hi={hi} source={i}"
                )
        if len(ref_baseline) != len(fast_base[hi]):
            raise RuntimeError(
                f"runtime Strong baseline source-count mismatch hi={hi}: "
                f"ref={len(ref_baseline)} fast={len(fast_base[hi])}"
            )
        ref_clear = baseline_clear_mask(ref_baseline, grid=pcfg.grid)
        if not np.array_equal(ref_clear, state["baseline_clear_by_hi"][hi]):
            raise RuntimeError(f"runtime Strong baseline CLEAR mismatch hi={hi}")
        ref_clear_flat = np.flatnonzero(ref_clear.reshape(-1)).astype(
            np.int64, copy=False
        )
        if not np.array_equal(
            ref_clear_flat, state["baseline_clear_flat_by_hi"][hi]
        ):
            raise RuntimeError(
                f"runtime Strong sparse baseline CLEAR mismatch hi={hi}"
            )

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
        repl_ref = []
        target_centers = []
        yaw_values = []
        for i, comp in enumerate(state["current"]):
            cid = int(comp["class_id"])
            src_center = np.asarray(comp["centroid_world"], dtype=np.float64)
            xy = state["rec"]["anchors_xy_t0_m"][i, hi].numpy() + pred_res[i, hi]
            center = t0_xy_to_world_preserve_source_z(
                xy, src_center, state["current_pose"]
            )
            cached_center = _target_world_from_xy_cached(
                xy, state["source_z_t0"][i], state["current_pose"]
            )
            if not np.array_equal(center, cached_center):
                raise RuntimeError(
                    f"runtime cached target-center mismatch hi={hi} source={i}"
                )
            yaw = renderer_yaw_delta(
                cid, pred_yaw[i, hi], zero_two_wheel_yaw=False
            )
            target_centers.append(center)
            yaw_values.append(yaw)
            repl_ref.append(_fast_rasterize_from_world(
                state["source_world_points"][i], cid, len(comp["voxel_indices"]),
                src_center, center, yaw, state["world_to_future"][hi], pcfg.grid,
            ))

        repl_fast = _rasterize_all_sources_horizon(
            state["current"],
            state["source_world_points"],
            state["source_rel_xy"],
            target_centers,
            yaw_values,
            state["world_to_future"][hi],
            pcfg.grid,
        )
        if len(repl_ref) != len(repl_fast):
            raise RuntimeError("runtime vectorized raster source-count mismatch")
        for i, (rr, ff) in enumerate(zip(repl_ref, repl_fast)):
            if (
                int(rr.class_id) != int(ff.class_id)
                or int(rr.source_voxel_count) != int(ff.source_voxel_count)
                or not np.array_equal(rr.voxel_indices, ff.voxel_indices)
            ):
                raise RuntimeError(
                    f"runtime vectorized rigid raster mismatch hi={hi} source={i}"
                )

        ref_pred = compose_component_replacements_in_input_order(
            state["anchors"][hi], state["baseline_by_hi"][hi], repl_ref,
            dynamic_class_ids=DYNAMIC_CLASS_IDS,
            free_label=int(pcfg.free_label), grid=pcfg.grid,
        )
        fast_pred = compose_component_replacements_fast_exact(
            state["anchors"][hi], state["baseline_by_hi"][hi], repl_fast,
            dynamic_class_ids=DYNAMIC_CLASS_IDS,
            free_label=int(pcfg.free_label), grid=pcfg.grid,
            precomputed_clear_flat_indices=state["baseline_clear_flat_by_hi"][hi],
        )
        if not np.array_equal(ref_pred, fast_pred):
            neq = int(np.count_nonzero(ref_pred != fast_pred))
            raise RuntimeError(
                f"runtime vectorized raster/A1 mismatch hi={hi} voxels={neq}"
            )
    print(
        "RUNTIME EXACTNESS: components + Strong all-6 + KTA footprints/CLEAR + vectorized rigid raster + sparse A1 PASS",
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
    profile = {}
    t0 = time.perf_counter()
    _forecast_once(model, state, pcfg, strong_cfg, device, profile=profile)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    total = (time.perf_counter() - t0) * 1000.0
    profile["total_ms"] = total
    return total, profile


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
    p.add_argument(
        "--exactness-windows", type=int, default=8,
        help="number of selected windows checked against frozen slow reference before timing",
    )
    p.add_argument(
        "--preflight-only",
        action="store_true",
        help="prepare/check only exactness windows, then exit before timing benchmark",
    )
    p.add_argument("--seed", type=int, default=20260918)
    p.add_argument("--device", default="cuda")
    p.add_argument("--profile-flops", action="store_true")
    a = p.parse_args()

    cfg = load_runtime_config(a.config, a.override)
    pcfg = make_prepare_config(cfg)
    _, records = mid.base.load_cache(a.val_cache)
    if not records:
        raise RuntimeError("validation cache is empty")
    if int(a.warmup_windows) < 0 or int(a.exactness_windows) < 0:
        raise ValueError("warmup/exactness counts must be non-negative")
    if bool(a.preflight_only):
        if int(a.exactness_windows) <= 0:
            raise ValueError("--preflight-only requires --exactness-windows > 0")
        if int(a.measure_windows) < 0:
            raise ValueError("measure count must be non-negative")
    elif int(a.measure_windows) <= 0:
        raise ValueError("normal benchmark requires --measure-windows > 0")

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
    requested_total = (
        int(a.exactness_windows)
        if bool(a.preflight_only)
        else int(a.warmup_windows) + int(a.measure_windows)
    )
    total_need = min(len(records), max(requested_total, int(a.exactness_windows)))
    ids = np.sort(rng.choice(len(records), size=total_need, replace=False))
    selected = [records[int(i)] for i in ids]

    source = CachedSource(a.dataroot, info_pkl=a.info_pkl, verbose=False)
    strong_cfg = StrongW2DetConfig(free_label=int(pcfg.free_label))
    exact_n = min(int(a.exactness_windows), len(selected))
    prepared = []
    prep_started = time.perf_counter()
    for pi, rec in enumerate(selected, start=1):
        tprep = time.perf_counter()
        state = _prepare_record(rec, source, pcfg, strong_cfg, device)
        prep_ms = (time.perf_counter() - tprep) * 1000.0
        prepared.append(state)

        if pi <= exact_n:
            print(
                f"RUNTIME PREPARE exactness {pi}/{exact_n}: "
                f"{prep_ms:.1f} ms",
                flush=True,
            )
            _stage_gpu_inputs(state, device)
            try:
                _exactness_check(model, state, pcfg, strong_cfg, device)
            finally:
                _release_gpu_inputs(state)
            print(f"RUNTIME EXACTNESS WINDOW {pi}/{exact_n}: PASS", flush=True)
        elif pi == exact_n + 1 or pi % 10 == 0 or pi == len(selected):
            elapsed = time.perf_counter() - prep_started
            print(
                f"RUNTIME PREPARE {pi}/{len(selected)} "
                f"last={prep_ms:.1f} ms elapsed={elapsed:.1f}s",
                flush=True,
            )

        if bool(a.preflight_only) and pi >= exact_n:
            print(
                "RUNTIME PREFLIGHT: exactness gate passed; exiting before timing.",
                flush=True,
            )
            return

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    nw = min(int(a.warmup_windows), len(prepared))
    for state in prepared[:nw]:
        _stage_gpu_inputs(state, device)
        try:
            _model_forward(model, state["gpu"], device)
            _forecast_once(model, state, pcfg, strong_cfg, device)
        finally:
            _release_gpu_inputs(state)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    # Never let extra exactness-only records silently increase the requested
    # measurement population.
    measured = prepared[nw : nw + int(a.measure_windows)]
    neural_ms, core_ms, prior_ms, full_ms, source_ms = [], [], [], [], []
    prior_breakdown_rows = []
    core_breakdown_rows = []
    source_counts = []
    for j, state in enumerate(measured, start=1):
        source_counts.append(len(state["current"]))
        _stage_gpu_inputs(state, device)
        try:
            neural_value = _time_neural(model, state, device)
            core_result = _time_forecast(model, state, pcfg, strong_cfg, device)
            if (
                not isinstance(core_result, tuple)
                or len(core_result) != 2
                or not isinstance(core_result[1], dict)
            ):
                raise TypeError(
                    "_time_forecast must return (total_ms: float, profile: dict)"
                )
            core_total, core_parts = core_result
            prior_result = _time_prior_rebuild(
                state, pcfg, strong_cfg, device
            )
            if (
                not isinstance(prior_result, tuple)
                or len(prior_result) != 2
                or not isinstance(prior_result[1], dict)
            ):
                raise TypeError(
                    "_time_prior_rebuild must return (total_ms: float, profile: dict)"
                )
            prior_total, prior_parts = prior_result
            full_value = _time_full_in_memory(
                model, state, pcfg, strong_cfg, device
            )
        finally:
            _release_gpu_inputs(state)

        neural_ms.append(float(neural_value))
        core_ms.append(float(core_total))
        core_breakdown_rows.append(core_parts)
        prior_ms.append(float(prior_total))
        prior_breakdown_rows.append(prior_parts)
        full_ms.append(float(full_value))
        source_ms.append(float(_time_source_extract(state, pcfg, strong_cfg)))

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
        median_idx = int(
            np.argsort(np.asarray(source_counts))[len(source_counts) // 2]
        )
        flop_state = measured[median_idx]
        _stage_gpu_inputs(flop_state, device)
        try:
            flops = _measure_model_flops(model, flop_state, device)
        finally:
            _release_gpu_inputs(flop_state)

    prior_breakdown_mean = {}
    if prior_breakdown_rows:
        for key in (
            "precompute_ms",
            "inverse_warp_ms",
            "majority_fill_ms",
            "dynamic_transport_scatter_ms",
            "total_ms",
        ):
            vals = [float(row.get(key, 0.0)) for row in prior_breakdown_rows]
            prior_breakdown_mean[key] = float(np.mean(vals))

    core_breakdown_mean = {}
    if core_breakdown_rows:
        for key in (
            "model_and_output_transfer_ms",
            "target_and_yaw_ms",
            "rigid_raster_ms",
            "a1_compose_ms",
            "total_ms",
        ):
            vals = [float(row.get(key, 0.0)) for row in core_breakdown_rows]
            core_breakdown_mean[key] = float(np.mean(vals))

    occworld_style = _occworld_style_summary(
        source_ms,
        full_ms,
        future_frames=FUTURE_FRAMES,
    )

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
        "exactness_windows": exact_n,
        "exactness_gate": (
            "reference components + six Strong anchors + per-source KTA baseline "
            "footprints/CLEAR + cached target centers + vectorized SE2 raster + A1"
        ),
        "gpu_input_staging": "one_window_at_a_time_outside_timed_regions",
        "component_extraction_runtime": "cropped_connected_components_bit_exact_gated",
        "majority_fill_runtime": "cuda_integer_box_counts_with_exact_scipy_edge_replay",
        "a1_clear_runtime": "precomputed_sparse_flat_clear_indices",
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
        "occworld_style_fps": (
            float(occworld_style["fps_from_mean"]) if occworld_style else None
        ),
        "occworld_style": occworld_style,
        "strong_kta_prior_breakdown_mean_ms": prior_breakdown_mean,
        "cached_forecast_breakdown_mean_ms": core_breakdown_mean,
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
            "occworld_style_fps": (
                "OccWorld formula: per-frame time = encode + autoreg/N_future. "
                "Here encode is source extraction+matching and autoreg is the full "
                "causal six-frame dense forecast; cached learned representation "
                "construction is explicitly excluded."
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
