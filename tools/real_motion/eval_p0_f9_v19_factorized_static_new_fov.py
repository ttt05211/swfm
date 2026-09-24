#!/usr/bin/env python3
"""Formal frozen-base evaluation for factorized Static New-FOV completion."""
from __future__ import annotations

import argparse
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
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
from real_motion.nuscenes_adapter import (
    NuScenesWindowSource,
    gt_moving_support_for_horizon,
)
from real_motion.motion_transport import world_points_to_t0
from real_motion.prepared import load_nuscenes_window_raw
from real_motion.runtime_config import (
    add_config_args,
    load_runtime_config,
    make_prepare_config,
)
from real_motion.runtime_fastpath import (
    baseline_clear_flat_indices,
    extract_instances_cropped_exact,
)
from real_motion.strong_w2det import StrongW2DetConfig, match_instances
from real_motion.v19_innovation import (
    base_explained_bev,
    align_prepared_history_frame_to_future,
    build_future_aligned_history_and_static_memory,
    prepare_history_alignment_frame,
)
from real_motion.v19_scene_memory import protected_add_only
from real_motion.v19_static_novelty import (
    history_grid_footprint_bev_sequence,
    majority_semantic_per_column,
    nearest_static_anchor_map,
)
from real_motion.v19_static_novelty_factorized import (
    FactorizedStaticNewFOVHead,
    decode_factorized_static_new_fov,
)
from tools.real_motion import eval_p0_f9_v18_se2 as base
from tools.real_motion import eval_p0_f9_v18_full_validation as full
from tools.real_motion.benchmark_p0_f9_v18_runtime import (
    _forecast_once,
    _precompute_source_world,
    _release_gpu_inputs,
    _stage_gpu_inputs,
    _strong_all_horizons,
)
from tools.real_motion.eval_p0_f9_v17_local_stwm import window_from_record
from tools.real_motion.eval_p0_f9_v19_static_new_fov import (
    HORIZONS,
    _delta,
    _finalize,
    _new_raw,
    _update_many,
)
from tools.real_motion.train_p0_f9_v18_se2_clean import (
    PROTOCOL as CLEAN_PROTOCOL,
)
from tools.real_motion.train_p0_f9_v19_factorized_static_new_fov import (
    HEAD_TYPE,
    PROTOCOL as FACTORIZED_TRAIN_PROTOCOL,
)


PROTOCOL = "p0_f9_v19_factorized_static_new_fov_eval_v1"
VARIANTS = (
    "v18",
    "v18_static",
    "v18_static_factorized_new_fov",
)


class CachedSource(NuScenesWindowSource):
    from functools import lru_cache

    @lru_cache(maxsize=768)
    def load_semantics(self, scene_name, token):
        return super().load_semantics(scene_name, token)

    @lru_cache(maxsize=768)
    def load_occ3d(self, scene_name, token, require_lidar_mask=True):
        return super().load_occ3d(
            scene_name,
            token,
            require_lidar_mask=require_lidar_mask,
        )

    @lru_cache(maxsize=4096)
    def pose(self, token):
        return super().pose(token)


class _HistoryAlignmentLRU:
    """Cache sparsified history frames across overlapping validation windows."""

    def __init__(self, maxsize: int = 64):
        self.maxsize = max(1, int(maxsize))
        self.data = OrderedDict()

    def get_or_build(
        self,
        scene,
        token,
        semantics,
        observed,
        pose,
        *,
        grid,
        dynamic_class_ids,
    ):
        key = (str(scene), str(token))
        if key in self.data:
            value = self.data.pop(key)
            self.data[key] = value
            return value
        value = prepare_history_alignment_frame(
            semantics,
            observed,
            pose,
            grid=grid,
            dynamic_class_ids=dynamic_class_ids,
        )
        self.data[key] = value
        while len(self.data) > self.maxsize:
            self.data.popitem(last=False)
        return value


class _HistoryFutureAlignmentLRU:
    """Reuse exact history->future sparse alignment pairs across windows.

    Consecutive stride-1 windows share 25/36 history/future frame pairs.  The
    cache stores only BEV features plus sparse Static-Memory clear/write
    indices, so reuse is exact without retaining dense per-pair 3D volumes.
    """

    def __init__(self, maxsize: int = 96):
        self.maxsize = max(1, int(maxsize))
        self.data = OrderedDict()

    def get(self, scene, history_token, future_token):
        key = (str(scene), str(history_token), str(future_token))
        if key not in self.data:
            return None
        value = self.data.pop(key)
        self.data[key] = value
        return value

    def put(self, scene, history_token, future_token, value):
        key = (str(scene), str(history_token), str(future_token))
        if key in self.data:
            self.data.pop(key)
        self.data[key] = value
        while len(self.data) > self.maxsize:
            self.data.popitem(last=False)


def _build_history_static_from_pair_cache(
    *,
    scene,
    history_tokens,
    future_tokens,
    prepared_history,
    future_poses,
    pair_cache,
    grid,
    free_label,
    workers,
):
    """Build exact six-future history features/static memory with pair reuse."""
    history_tokens = tuple(str(x) for x in history_tokens)
    future_tokens = tuple(str(x) for x in future_tokens)
    if len(history_tokens) != 6 or len(future_tokens) != 6:
        raise ValueError("expected six history and six future tokens")
    if len(prepared_history) != 6 or len(future_poses) != 6:
        raise ValueError("prepared history/future pose count mismatch")

    rows = {}
    misses = []
    for fi, ftok in enumerate(future_tokens):
        for ti, htok in enumerate(history_tokens):
            value = pair_cache.get(scene, htok, ftok)
            key = (ti, fi)
            if value is None:
                misses.append((key, htok, ftok))
            else:
                rows[key] = value

    def _build(item):
        key, htok, ftok = item
        ti, fi = key
        value = align_prepared_history_frame_to_future(
            prepared_history[ti],
            future_poses[fi],
            grid=grid,
            free_label=int(free_label),
            return_coverage=False,
        )
        return key, htok, ftok, value

    if misses:
        nworkers = max(1, min(int(workers), len(misses)))
        if nworkers == 1:
            built = [_build(x) for x in misses]
        else:
            with ThreadPoolExecutor(max_workers=nworkers) as pool:
                built = list(pool.map(_build, misses))
        for key, htok, ftok, value in built:
            pair_cache.put(scene, htok, ftok, value)
            rows[key] = value

    shape = tuple(int(v) for v in grid.shape_hwd)
    sem_futures = []
    geo_futures = []
    static_futures = []
    for fi in range(6):
        static_out = np.full(shape, int(free_label), dtype=np.uint8)
        static_flat = static_out.reshape(-1)
        sem_hist = []
        geo_hist = []
        for ti in range(6):
            (
                top_label,
                geom,
                _,
                clear_flat,
                write_flat,
                write_vals,
            ) = rows[(ti, fi)]
            sem_hist.append(top_label)
            geo_hist.append(geom)
            if len(clear_flat):
                static_flat[clear_flat] = int(free_label)
            if len(write_flat):
                static_flat[write_flat] = write_vals
        sem_futures.append(np.stack(sem_hist, axis=0))
        geo_futures.append(np.stack(geo_hist, axis=0))
        static_futures.append(static_out)

    return (
        np.stack(sem_futures, axis=0).astype(np.uint8, copy=False),
        np.stack(geo_futures, axis=0).astype(np.float32, copy=False),
        np.stack(static_futures, axis=0).astype(np.uint8, copy=False),
        {
            "hits": int(36 - len(misses)),
            "misses": int(len(misses)),
        },
    )


class _ComponentLRU:
    """Reuse exact Strong components for overlapping validation windows."""

    def __init__(self, maxsize: int = 1024):
        self.maxsize = max(1, int(maxsize))
        self.data = OrderedDict()

    def get_or_build(self, scene, token, semantics, pose, *, grid, cfg):
        key = (str(scene), str(token))
        if key in self.data:
            value = self.data.pop(key)
            self.data[key] = value
            return value
        value = extract_instances_cropped_exact(
            np.asarray(semantics, dtype=np.uint8),
            np.asarray(pose, dtype=np.float64),
            grid=grid,
            cfg=cfg,
        )
        self.data[key] = value
        while len(self.data) > self.maxsize:
            self.data.popitem(last=False)
        return value


def _prepare_record_from_raw(
    rec,
    raw,
    source,
    pcfg,
    strong_cfg,
    device,
    component_cache,
):
    """Lean evaluator preparation using already-loaded raw window tensors."""
    w = window_from_record(rec)
    scene = str(w.scene_name)
    current_sem = np.asarray(raw["history_occ"][-1], dtype=np.uint8)
    previous_sem = np.asarray(raw["history_occ"][-2], dtype=np.uint8)
    current_pose = np.asarray(raw["history_poses"][-1], dtype=np.float64)
    previous_pose = np.asarray(raw["history_poses"][-2], dtype=np.float64)
    future_poses = [
        np.asarray(x, dtype=np.float64)
        for x in raw["future_poses"]
    ]

    current = component_cache.get_or_build(
        scene,
        str(w.t0_token),
        current_sem,
        current_pose,
        grid=pcfg.grid,
        cfg=strong_cfg,
    )
    previous = component_cache.get_or_build(
        scene,
        str(w.history_tokens[-2]),
        previous_sem,
        previous_pose,
        grid=pcfg.grid,
        cfg=strong_cfg,
    )
    velocities = match_instances(
        previous,
        current,
        float(pcfg.frame_dt_s),
        max_speed_mps=strong_cfg.max_match_speed_mps,
    )

    if len(current) != int(rec["features"].shape[0]):
        raise RuntimeError(
            f"{rec['sample_id']}: Strong/source count mismatch"
        )
    got = [int(x["class_id"]) for x in current]
    expected = [int(x) for x in rec["source_class_id"].tolist()]
    if got != expected:
        raise RuntimeError(
            f"{rec['sample_id']}: Strong/source order mismatch"
        )

    source_world_points = _precompute_source_world(
        current,
        current_pose,
        pcfg.grid,
    )
    source_rel_xy = [
        np.asarray(pts, dtype=np.float64)[:, :2]
        - np.asarray(comp["centroid_world"], dtype=np.float64)[None, :2]
        for pts, comp in zip(source_world_points, current)
    ]
    source_z_t0 = np.asarray(
        [
            world_points_to_t0(
                np.asarray(
                    comp["centroid_world"],
                    dtype=np.float64,
                )[None],
                current_pose,
            )[0, 2]
            for comp in current
        ],
        dtype=np.float64,
    )

    anchors, baseline_by_hi = _strong_all_horizons(
        current_sem,
        current_pose,
        future_poses,
        current,
        velocities,
        source_world_points,
        frame_dt_s=float(pcfg.frame_dt_s),
        grid=pcfg.grid,
        cfg=strong_cfg,
        runtime_device=device,
    )
    baseline_clear_flat_by_hi = [
        baseline_clear_flat_indices(rows, grid=pcfg.grid)
        for rows in baseline_by_hi
    ]
    world_to_future = [
        np.linalg.inv(np.asarray(p, dtype=np.float64))
        for p in future_poses
    ]
    return {
        "rec": rec,
        "window": w,
        "scene": scene,
        "current_pose": current_pose,
        "future_poses": future_poses,
        "current": current,
        "velocities": velocities,
        "source_world_points": source_world_points,
        "source_rel_xy": source_rel_xy,
        "source_z_t0": source_z_t0,
        "anchors": anchors,
        "baseline_by_hi": baseline_by_hi,
        "baseline_clear_flat_by_hi": baseline_clear_flat_by_hi,
        "world_to_future": world_to_future,
        "gpu": None,
    }


def _build_anchor_context_sequence(
    static_all,
    footprints,
    *,
    free_label,
    voxel_size_xy_m,
    workers,
):
    static_all = np.asarray(static_all, dtype=np.uint8)
    footprints = np.asarray(footprints, dtype=bool)
    if static_all.ndim != 4:
        raise ValueError("static_all must be [F,X,Y,Z]")
    if footprints.shape != static_all.shape[:3]:
        raise ValueError("footprint/static shape mismatch")

    def _one(fi):
        static_render = static_all[int(fi)]
        footprint = footprints[int(fi)]
        dist_cells, ax, ay, valid = nearest_static_anchor_map(
            static_render,
            footprint,
            free_label=int(free_label),
        )
        static_occ = static_render != int(free_label)
        sem_source = majority_semantic_per_column(
            static_occ,
            static_render,
            num_classes=17,
            ignore_label=int(free_label),
        )
        aq = np.full(
            footprint.shape,
            int(free_label),
            dtype=np.uint8,
        )
        profile = np.zeros(static_occ.shape, dtype=bool)
        if bool(valid.any()):
            aq[valid] = sem_source[ax[valid], ay[valid]]
            copied = static_occ[ax, ay]
            profile[valid] = copied[valid]
        dist_m = (
            np.asarray(dist_cells, dtype=np.float32)
            * float(voxel_size_xy_m)
        )
        return aq, profile, dist_m

    ids = list(range(static_all.shape[0]))
    nworkers = max(1, int(workers))
    if nworkers == 1:
        rows = [_one(i) for i in ids]
    else:
        with ThreadPoolExecutor(
            max_workers=min(nworkers, len(ids))
        ) as pool:
            rows = list(pool.map(_one, ids))
    asem, aprof, adist = zip(*rows)
    return (
        np.stack(asem, axis=0),
        np.stack(aprof, axis=0),
        np.stack(adist, axis=0),
    )


def _moving_support_sequence(source, window, *, grid, workers):
    def _one(item):
        hi, h = item
        moving, _, _ = gt_moving_support_for_horizon(
            source.nusc,
            str(window.t0_token),
            str(window.future_tokens[int(hi)]),
            float(h),
            grid=grid,
        )
        return moving

    items = list(enumerate(HORIZONS))
    nworkers = max(1, int(workers))
    if nworkers == 1:
        return [_one(x) for x in items]
    with ThreadPoolExecutor(
        max_workers=min(nworkers, len(items))
    ) as pool:
        return list(pool.map(_one, items))


def _autocast(device, enabled):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _effective_addition_quality(base_raw, variant_raw):
    rows = []
    tp_all = fp_all = 0
    for hi, h in enumerate(HORIZONS):
        tp = int(
            variant_raw["occ_inter"][hi]
            - base_raw["occ_inter"][hi]
        )
        fp = int(
            variant_raw["occ_union"][hi]
            - base_raw["occ_union"][hi]
        )
        tp_all += tp
        fp_all += fp
        rows.append(
            {
                "horizon_s": float(h),
                "added_tp": tp,
                "added_fp": fp,
                "precision": float(tp / max(tp + fp, 1)),
            }
        )
    return {
        "per_horizon": rows,
        "added_tp": int(tp_all),
        "added_fp": int(fp_all),
        "precision": float(tp_all / max(tp_all + fp_all, 1)),
    }


def _cuda_sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _profile_add(profile, name, elapsed_s):
    row = profile.setdefault(
        str(name),
        {"total_s": 0.0, "calls": 0},
    )
    row["total_s"] += float(elapsed_s)
    row["calls"] += 1


def _profile_finalize(profile, measured_windows):
    total = sum(float(v["total_s"]) for v in profile.values())
    rows = {}
    for name, row in sorted(
        profile.items(),
        key=lambda kv: float(kv[1]["total_s"]),
        reverse=True,
    ):
        s = float(row["total_s"])
        calls = int(row["calls"])
        rows[name] = {
            "total_s": s,
            "share_pct": float(100.0 * s / max(total, 1e-12)),
            "mean_ms_per_window": float(
                1000.0 * s / max(int(measured_windows), 1)
            ),
            "mean_ms_per_call": float(
                1000.0 * s / max(calls, 1)
            ),
            "calls": calls,
        }
    return {
        "measured_windows": int(measured_windows),
        "summed_stage_s": float(total),
        "stages": rows,
    }


def main():
    p = argparse.ArgumentParser()
    add_config_args(p)
    p.add_argument("--val-cache", required=True)
    p.add_argument("--base-checkpoint", required=True)
    p.add_argument("--novelty-checkpoint", required=True)
    p.add_argument("--dataroot", required=True)
    p.add_argument("--info-pkl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-windows", type=int, default=0)
    p.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="split the selected validation population into disjoint shards",
    )
    p.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="0-based shard index used with --num-shards",
    )
    p.add_argument("--alignment-workers", type=int, default=6)
    p.add_argument(
        "--profile-windows",
        type=int,
        default=0,
        help=(
            "measure exact stage wall time on this many windows after warmup; "
            "CUDA stages are synchronized only while profiling"
        ),
    )
    p.add_argument(
        "--profile-warmup-windows",
        type=int,
        default=2,
        help="unmeasured warmup windows before stage profiling starts",
    )
    p.add_argument(
        "--profile-only",
        action="store_true",
        help=(
            "when profiling, evaluate only warmup+profile windows and exit "
            "after reporting the stage breakdown"
        ),
    )
    p.add_argument(
        "--presence-threshold",
        type=float,
        default=-1.0,
        help="<0 uses checkpoint-selected validation threshold",
    )
    p.add_argument(
        "--vertical-threshold",
        type=float,
        default=-1.0,
        help="<0 uses checkpoint-selected validation threshold",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-amp", action="store_true")
    a = p.parse_args()

    if int(a.alignment_workers) <= 0:
        raise ValueError("alignment-workers must be positive")
    if int(a.profile_windows) < 0 or int(a.profile_warmup_windows) < 0:
        raise ValueError("profile window counts must be non-negative")
    if bool(a.profile_only) and int(a.profile_windows) <= 0:
        raise ValueError("--profile-only requires --profile-windows > 0")
    if int(a.num_shards) <= 0:
        raise ValueError("num-shards must be positive")
    if not 0 <= int(a.shard_index) < int(a.num_shards):
        raise ValueError("shard-index must be in [0,num-shards)")

    pcfg = make_prepare_config(load_runtime_config(a.config, a.override))
    _, records = base.load_cache(a.val_cache)
    if int(a.max_windows) > 0:
        records = records[: min(len(records), int(a.max_windows))]
    selected_population = int(len(records))
    if int(a.num_shards) > 1:
        n = len(records)
        lo = n * int(a.shard_index) // int(a.num_shards)
        hi = n * (int(a.shard_index) + 1) // int(a.num_shards)
        records = records[lo:hi]
    if not records:
        raise RuntimeError("empty validation cache shard")
    if bool(a.profile_only):
        need = int(a.profile_warmup_windows) + int(a.profile_windows)
        records = records[: min(len(records), need)]

    device = torch.device(
        a.device
        if a.device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    amp = device.type == "cuda" and not bool(a.no_amp)

    base_ck, base_model, _ = full._load_model(
        a.base_checkpoint,
        CLEAN_PROTOCOL,
        device,
    )
    nov_ck = torch.load(
        a.novelty_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    if nov_ck.get("protocol") != FACTORIZED_TRAIN_PROTOCOL:
        raise RuntimeError(
            f"unexpected factorized checkpoint protocol: "
            f"{nov_ck.get('protocol')}"
        )
    if nov_ck.get("head_type") != HEAD_TYPE:
        raise RuntimeError(
            f"unexpected factorized head type: {nov_ck.get('head_type')}"
        )
    novelty = FactorizedStaticNewFOVHead(
        **dict(nov_ck["architecture"])
    ).to(device)
    novelty.load_state_dict(nov_ck["model_state_dict"], strict=True)
    novelty.eval()

    presence_threshold = (
        float(a.presence_threshold)
        if float(a.presence_threshold) >= 0
        else float(nov_ck["selected_presence_threshold"])
    )
    vertical_threshold = (
        float(a.vertical_threshold)
        if float(a.vertical_threshold) >= 0
        else float(nov_ck["selected_vertical_threshold"])
    )

    source = CachedSource(a.dataroot, info_pkl=a.info_pkl, verbose=False)
    strong_cfg = StrongW2DetConfig(free_label=int(pcfg.free_label))
    component_cache = _ComponentLRU(maxsize=1024)
    history_alignment_cache = _HistoryAlignmentLRU(maxsize=64)
    history_future_pair_cache = _HistoryFutureAlignmentLRU(maxsize=96)
    raw_by_variant = {v: _new_raw() for v in VARIANTS}
    proposed = 0
    added = 0
    windows_with_additions = 0
    active_bev_columns = 0
    new_fov_bev_columns = 0
    started = time.perf_counter()
    stage_profile = {}
    profiled_windows = 0
    pair_cache_hits = 0
    pair_cache_misses = 0
    profile_lo = int(a.profile_warmup_windows) + 1
    profile_hi = int(a.profile_warmup_windows) + int(a.profile_windows)

    for wi, rec in enumerate(records, start=1):
        profile_this = (
            int(a.profile_windows) > 0
            and profile_lo <= wi <= profile_hi
        )
        w = window_from_record(rec)

        t = time.perf_counter()
        raw = load_nuscenes_window_raw(source, w, pcfg, include_gt=True)
        if profile_this:
            _profile_add(
                stage_profile,
                "raw_window_load",
                time.perf_counter() - t,
            )

        t = time.perf_counter()
        state = _prepare_record_from_raw(
            rec,
            raw,
            source,
            pcfg,
            strong_cfg,
            device,
            component_cache,
        )
        if profile_this:
            _profile_add(
                stage_profile,
                "v18_prepare_record",
                time.perf_counter() - t,
            )

        if profile_this:
            _cuda_sync(device)
        t = time.perf_counter()
        _stage_gpu_inputs(state, device)
        if profile_this:
            _cuda_sync(device)
            _profile_add(
                stage_profile,
                "v18_gpu_stage_inputs",
                time.perf_counter() - t,
            )

        if profile_this:
            _cuda_sync(device)
        t = time.perf_counter()
        try:
            pred_all = _forecast_once(
                base_model,
                state,
                pcfg,
                strong_cfg,
                device,
            )
            if profile_this:
                _cuda_sync(device)
                _profile_add(
                    stage_profile,
                    "v18_forecast_6frames",
                    time.perf_counter() - t,
                )
        finally:
            _release_gpu_inputs(state)

        t = time.perf_counter()
        prepared_history = [
            history_alignment_cache.get_or_build(
                str(w.scene_name),
                str(tok),
                raw["history_occ"][ti],
                raw["history_observed"][ti],
                raw["history_poses"][ti],
                grid=pcfg.grid,
                dynamic_class_ids=DYNAMIC_CLASS_IDS,
            )
            for ti, tok in enumerate(w.history_tokens)
        ]
        sem, geo, static_all, pair_cache_stats = (
            _build_history_static_from_pair_cache(
                scene=str(w.scene_name),
                history_tokens=w.history_tokens,
                future_tokens=w.future_tokens,
                prepared_history=prepared_history,
                future_poses=raw["future_poses"],
                pair_cache=history_future_pair_cache,
                grid=pcfg.grid,
                free_label=int(pcfg.free_label),
                workers=int(a.alignment_workers),
            )
        )
        pair_cache_hits += int(pair_cache_stats["hits"])
        pair_cache_misses += int(pair_cache_stats["misses"])
        if profile_this:
            _profile_add(
                stage_profile,
                "history_align_and_static_memory",
                time.perf_counter() - t,
            )

        history_poses = np.asarray(
            raw["history_poses"],
            dtype=np.float64,
        )
        future_poses = np.asarray(
            raw["future_poses"],
            dtype=np.float64,
        )

        t_causal = time.perf_counter()
        pred_stack = np.asarray(pred_all, dtype=np.uint8)
        static_all = np.asarray(static_all, dtype=np.uint8)
        explained = protected_add_only(
            pred_stack,
            static_all,
            free_label=int(pcfg.free_label),
        )
        base_free = explained == int(pcfg.free_label)

        footprint_all = history_grid_footprint_bev_sequence(
            history_poses,
            future_poses,
            pcfg.grid,
            workers=int(a.alignment_workers),
        )
        new_fov = ~footprint_all
        new_fov_bev_columns += int(new_fov.sum())

        anchor_sem, anchor_profile, anchor_dist = (
            _build_anchor_context_sequence(
                static_all,
                footprint_all,
                free_label=int(pcfg.free_label),
                voxel_size_xy_m=float(pcfg.grid.voxel_size[0]),
                workers=int(a.alignment_workers),
            )
        )
        if profile_this:
            _profile_add(
                stage_profile,
                "new_fov_anchor_context",
                time.perf_counter() - t_causal,
            )

        t = time.perf_counter()

        sem_t = torch.from_numpy(sem[None]).to(device)
        geo_t = torch.from_numpy(geo[None]).to(device)
        base_t = torch.from_numpy(
            base_explained_bev(
                explained,
                free_label=int(pcfg.free_label),
            )[None]
        ).to(device)
        nf_t = torch.from_numpy(new_fov[None]).to(device)
        anchor_sem_t = torch.from_numpy(anchor_sem[None]).to(device)
        anchor_profile_t = torch.from_numpy(
            anchor_profile.transpose(0, 3, 1, 2)[None]
        ).to(device)
        anchor_dist_t = torch.from_numpy(anchor_dist[None]).to(device)
        free_t = torch.from_numpy(
            base_free.transpose(0, 3, 1, 2)[None]
        ).to(device)
        if profile_this:
            _cuda_sync(device)
            _profile_add(
                stage_profile,
                "novelty_tensor_staging",
                time.perf_counter() - t,
            )

        if profile_this:
            _cuda_sync(device)
        t = time.perf_counter()
        with torch.inference_mode(), _autocast(device, amp):
            out = novelty(
                sem_t,
                geo_t,
                base_t,
                nf_t,
                anchor_sem_t,
                anchor_profile_t,
                anchor_dist_t,
            )
            active_bev_columns += int(
                (
                    (
                        torch.sigmoid(
                            out["presence_logits"].float()
                        )
                        >= float(presence_threshold)
                    )
                    & nf_t.bool()
                ).sum().item()
            )
            proposal_zxy = decode_factorized_static_new_fov(
                out,
                new_fov_mask=nf_t,
                base_free=free_t,
                free_label=int(pcfg.free_label),
                presence_threshold=presence_threshold,
                vertical_threshold=vertical_threshold,
            )
        if profile_this:
            _cuda_sync(device)
            _profile_add(
                stage_profile,
                "novelty_forward_and_decode",
                time.perf_counter() - t,
            )

        t = time.perf_counter()
        proposal = (
            proposal_zxy[0]
            .permute(0, 2, 3, 1)
            .cpu()
            .numpy()
            .astype(np.uint8)
        )
        proposed += int((proposal != int(pcfg.free_label)).sum())

        final = protected_add_only(
            explained,
            proposal,
            free_label=int(pcfg.free_label),
        )
        added_mask = (
            (final != int(pcfg.free_label))
            & (explained == int(pcfg.free_label))
        )
        window_added = int(added_mask.sum())
        added += window_added
        windows_with_additions += int(window_added > 0)
        if profile_this:
            _profile_add(
                stage_profile,
                "proposal_cpu_and_compose",
                time.perf_counter() - t,
            )

        t_moving = time.perf_counter()
        moving_rows = _moving_support_sequence(
            source,
            w,
            grid=pcfg.grid,
            workers=int(a.alignment_workers),
        )
        if profile_this:
            _profile_add(
                stage_profile,
                "moving_support_gt_metric_prep",
                time.perf_counter() - t_moving,
            )

        t_metric = time.perf_counter()
        for hi, h in enumerate(HORIZONS):
            gt = np.asarray(raw["future_gt_occ"][hi], dtype=np.uint8)
            moving = moving_rows[hi]
            _update_many(
                raw_by_variant,
                hi,
                {
                    "v18": pred_stack[hi],
                    "v18_static": explained[hi],
                    "v18_static_factorized_new_fov": final[hi],
                },
                gt,
                moving,
                int(pcfg.free_label),
            )

        if profile_this:
            _profile_add(
                stage_profile,
                "metric_accumulation_3_variants",
                time.perf_counter() - t_metric,
            )
            profiled_windows += 1

        if wi == 1 or wi % 25 == 0 or wi == len(records):
            elapsed = max(time.perf_counter() - started, 1e-9)
            print(
                f"v19_factorized_new_fov_eval {wi}/{len(records)} "
                f"rate={wi/elapsed:.3f} win/s",
                flush=True,
            )

    metrics = {v: _finalize(raw_by_variant[v]) for v in VARIANTS}
    delta = _delta(
        metrics["v18_static_factorized_new_fov"],
        metrics["v18_static"],
    )
    effective = _effective_addition_quality(
        raw_by_variant["v18_static"],
        raw_by_variant["v18_static_factorized_new_fov"],
    )
    elapsed = max(time.perf_counter() - started, 1e-9)
    profile_report = (
        _profile_finalize(stage_profile, profiled_windows)
        if profiled_windows > 0
        else None
    )
    result = {
        "protocol": PROTOCOL,
        "num_windows": int(len(records)),
        "selected_population_windows": int(selected_population),
        "num_shards": int(a.num_shards),
        "shard_index": int(a.shard_index),
        "base_checkpoint": str(Path(a.base_checkpoint).resolve()),
        "base_checkpoint_epoch": int(base_ck.get("epoch", -1)),
        "novelty_checkpoint": str(Path(a.novelty_checkpoint).resolve()),
        "novelty_epoch": int(nov_ck.get("epoch", -1)),
        "presence_threshold": float(presence_threshold),
        "vertical_threshold": float(vertical_threshold),
        "future_gt_used_for_prediction": False,
        "metrics": metrics,
        "delta_factorized_vs_static": delta,
        "effective_addition_quality": effective,
        "proposal_audit": {
            "new_fov_bev_columns": int(new_fov_bev_columns),
            "active_bev_columns": int(active_bev_columns),
            "proposed_voxels": int(proposed),
            "added_voxels_after_protection": int(added),
            "windows_with_additions": int(windows_with_additions),
        },
        "raw_counts": {
            v: {
                k: np.asarray(x).tolist()
                for k, x in raw_by_variant[v].items()
            }
            for v in VARIANTS
        },
        "timing": {
            "elapsed_s": float(elapsed),
            "windows_per_s": float(len(records) / elapsed),
        },
        "stage_timing_profile": profile_report,
        "history_future_pair_cache": {
            "hits": int(pair_cache_hits),
            "misses": int(pair_cache_misses),
            "hit_rate": float(
                pair_cache_hits / max(pair_cache_hits + pair_cache_misses, 1)
            ),
        },
    }
    op = Path(a.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n=== V19 FACTORIZED STATIC NEW-FOV EVAL ===")
    for v in VARIANTS:
        m = metrics[v]
        print(
            f"{v:32s} "
            f"IoU={m['IoU']:.3f} "
            f"mIoU={m['mIoU']:.3f} "
            f"MovMacro={m['MovingMacro']:.3f} "
            f"MovMicro={m['MovingMicro']:.3f} "
            f"main123_mIoU={m['main_1_2_3s']['mIoU']:.3f}"
        )
    print("factorized_vs_static", json.dumps(delta))
    print("effective_addition", json.dumps(effective))
    print("proposal_audit", json.dumps(result["proposal_audit"]))
    if profile_report is not None:
        print("\n=== STAGE TIMING PROFILE ===")
        print(
            f"measured_windows={profile_report['measured_windows']} "
            f"summed_stage_s={profile_report['summed_stage_s']:.3f}"
        )
        for name, row in profile_report["stages"].items():
            print(
                f"{name:34s} "
                f"{row['share_pct']:6.2f}% "
                f"{row['mean_ms_per_window']:9.2f} ms/window "
                f"calls={row['calls']}"
            )
    print(f"saved {op}")


if __name__ == "__main__":
    main()
