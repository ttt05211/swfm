#!/usr/bin/env python3
"""Zero-training V19-M rollout diagnostic.

This evaluator keeps the frozen Clean-E14 first block unchanged, but replaces
second-block component re-detection/identity matching with persistent source
memory built directly from the first block's predicted source trajectories.

It reports two causal variants:
  * persistent_source: persistent dynamic source identity only;
  * persistent_source_static: plus six-frame lidar-observed StaticWorldMemory.

No V19 trainable adapter or innovation head is used here.  The purpose is to
measure the zero-training value of explicit state maintenance before adding
learned capacity.
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

from real_motion.metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS
from real_motion.motion_transport import (
    FUTURE_FRAMES,
    HISTORY_FRAMES,
    world_points_to_t0,
)
from real_motion.nuscenes_adapter import (
    NuScenesWindowSource,
    gt_moving_support_for_horizon,
)
from real_motion.prepared import load_nuscenes_window_raw
from real_motion.runtime_config import (
    add_config_args,
    load_runtime_config,
    make_prepare_config,
)
from real_motion.runtime_fastpath import (
    baseline_clear_flat_indices,
    baseline_clear_mask,
)
from real_motion.strong_w2det import StrongW2DetConfig
from real_motion.v19_scene_memory import (
    StaticWorldMemory,
    persistent_tracks_from_v18_predictions,
    prepare_causal_arrays_from_tracks,
    protected_add_only,
)
from tools.real_motion import eval_p0_f9_v18_se2 as base
from tools.real_motion import eval_p0_f9_v18_full_validation as full
from tools.real_motion.benchmark_p0_f9_v18_runtime import (
    _forecast_once,
    _model_forward,
    _prepare_record,
    _precompute_source_world,
    _release_gpu_inputs,
    _stage_gpu_inputs,
    _strong_all_horizons,
)
from tools.real_motion.eval_p0_f9_v17_local_stwm import window_from_record
from tools.real_motion.eval_p0_f9_v18_zero_shot_long_rollout import (
    REPORT_HORIZONS,
    REPORT_INDEX,
    _finalize,
    _new_raw,
    _update_raw,
)
from tools.real_motion.train_p0_f9_v18_se2_clean import PROTOCOL as CLEAN_PROTOCOL


PROTOCOL = "p0_f9_v19_memory_zero_training_rollout_6s_v1"
VARIANTS = ("persistent_source", "persistent_source_static")


class CachedSource(NuScenesWindowSource):
    from functools import lru_cache

    @lru_cache(maxsize=768)
    def load_semantics(self, scene_name, token):
        return super().load_semantics(scene_name, token)

    @lru_cache(maxsize=768)
    def load_occ3d(self, scene_name, token, require_lidar_mask=True):
        return super().load_occ3d(
            scene_name, token, require_lidar_mask=require_lidar_mask
        )

    @lru_cache(maxsize=4096)
    def pose(self, token):
        return super().pose(token)


def _world_points_to_indices(points_world, ego_to_world, grid):
    pts = np.asarray(points_world, dtype=np.float64)
    W2E = np.linalg.inv(np.asarray(ego_to_world, dtype=np.float64))
    ego = pts @ W2E[:3, :3].T + W2E[:3, 3]
    origin = np.asarray(
        [grid.x_min, grid.y_min, grid.z_min], dtype=np.float64
    )
    step = np.asarray(grid.voxel_size, dtype=np.float64)
    idx = np.floor((ego - origin[None]) / step[None]).astype(np.int64)
    shape = np.asarray(grid.shape_hwd, dtype=np.int64)
    valid = np.all((idx >= 0) & (idx < shape[None]), axis=1)
    return idx[valid]


def _state_from_persistent_tracks(
    tracks,
    rec2,
    history_occ,
    history_poses,
    future_poses,
    pcfg,
    strong_cfg,
    device,
):
    current_pose = np.asarray(history_poses[-1], dtype=np.float64)
    current_sem = np.asarray(history_occ[-1], dtype=np.uint8)
    # Persistent memory owns the dynamic source state.  Remove dense dynamic
    # labels before rebuilding Strong so they cannot be re-detected as "rest"
    # and duplicated with zero velocity.
    static_sem = current_sem.copy()
    static_sem[
        np.isin(
            static_sem,
            np.asarray(DYNAMIC_CLASS_IDS, dtype=static_sem.dtype),
        )
    ] = int(pcfg.free_label)

    current = []
    velocities = {}
    source_world_points = []
    source_rel_xy = []
    source_z_t0 = []
    for i, tr in enumerate(tracks):
        center = tr.anchor_center_world(float(pcfg.frame_dt_s))
        pts = center[None] + np.asarray(
            tr.canonical_xyz_local, dtype=np.float64
        )
        idx = _world_points_to_indices(pts, current_pose, pcfg.grid)
        current.append(
            {
                "class_id": int(tr.class_id),
                "centroid_world": np.asarray(center, dtype=np.float64),
                "voxel_indices": idx,
                "voxel_count": int(len(tr.canonical_xyz_local)),
            }
        )
        velocities[i] = np.asarray(tr.velocity_world, dtype=np.float64)
        source_world_points.append(pts)
        source_rel_xy.append(
            np.asarray(tr.canonical_xyz_local, dtype=np.float64)[:, :2]
        )
        source_z_t0.append(
            world_points_to_t0(
                np.asarray(center, dtype=np.float64)[None], current_pose
            )[0, 2]
        )

    anchors, baseline_by_hi = _strong_all_horizons(
        static_sem,
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
    clear_by_hi = [
        baseline_clear_mask(rows, grid=pcfg.grid)
        for rows in baseline_by_hi
    ]
    clear_flat_by_hi = [
        baseline_clear_flat_indices(rows, grid=pcfg.grid)
        for rows in baseline_by_hi
    ]
    return {
        "rec": rec2,
        "window": None,
        "scene": None,
        "current_sem": static_sem,
        "previous_sem": np.asarray(history_occ[-2], dtype=np.uint8),
        "current_pose": current_pose,
        "previous_pose": np.asarray(history_poses[-2], dtype=np.float64),
        "future_poses": [
            np.asarray(x, dtype=np.float64) for x in future_poses
        ],
        "current": current,
        "previous": [],
        "velocities": velocities,
        "source_world_points": source_world_points,
        "source_rel_xy": source_rel_xy,
        "source_z_t0": np.asarray(source_z_t0, dtype=np.float64),
        "anchors": anchors,
        "baseline_by_hi": baseline_by_hi,
        "baseline_clear_by_hi": clear_by_hi,
        "baseline_clear_flat_by_hi": clear_flat_by_hi,
        "world_to_future": [
            np.linalg.inv(np.asarray(p, dtype=np.float64))
            for p in future_poses
        ],
        "gpu": None,
    }


def main():
    p = argparse.ArgumentParser()
    add_config_args(p)
    p.add_argument("--val-cache", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataroot", required=True)
    p.add_argument("--info-pkl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--expected-windows", type=int, default=3469)
    p.add_argument("--max-windows", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--progress-every", type=int, default=25)
    a = p.parse_args()

    cfg = load_runtime_config(a.config, a.override)
    pcfg = make_prepare_config(cfg)
    _, records = base.load_cache(a.val_cache)
    rec_by_t0 = {str(r["t0_token"]): r for r in records}

    device = torch.device(
        a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    ck, model, _ = full._load_model(a.checkpoint, CLEAN_PROTOCOL, device)
    source = CachedSource(a.dataroot, info_pkl=a.info_pkl, verbose=False)
    strong_cfg = StrongW2DetConfig(free_label=int(pcfg.free_label))

    selected = []
    for w in source.iter_windows(
        history=HISTORY_FRAMES,
        future=12,
        stride=1,
        max_windows=None,
    ):
        rec = rec_by_t0.get(str(w.t0_token))
        if rec is None:
            continue
        rw = window_from_record(rec)
        if tuple(rw.history_tokens) != tuple(w.history_tokens):
            raise RuntimeError(f"{w.t0_token}: history mismatch")
        if tuple(rw.future_tokens) != tuple(w.future_tokens[:6]):
            raise RuntimeError(f"{w.t0_token}: first future block mismatch")
        selected.append((w, rec))

    total_selected = len(selected)
    if int(a.expected_windows) > 0 and total_selected != int(a.expected_windows):
        raise RuntimeError(
            f"eligible windows {total_selected} != expected {a.expected_windows}"
        )
    ns, si = int(a.num_shards), int(a.shard_index)
    if ns <= 0 or si < 0 or si >= ns:
        raise ValueError("invalid shard specification")
    selected = selected[si::ns]
    if int(a.max_windows) > 0:
        selected = selected[: min(len(selected), int(a.max_windows))]
    if not selected:
        raise RuntimeError("empty selected shard")

    raw = {name: _new_raw() for name in VARIANTS}
    started = time.perf_counter()

    for wi, (w, rec) in enumerate(selected, start=1):
        # First block is the exact frozen main path.
        state1 = _prepare_record(rec, source, pcfg, strong_cfg, device)
        _stage_gpu_inputs(state1, device)
        try:
            out1 = _model_forward(model, state1["gpu"], device)
            pred_res = out1["residual_xy_m"].float().cpu()
            pred_yaw = out1["yaw_delta_rad"].float().cpu()
            pred1 = _forecast_once(
                model, state1, pcfg, strong_cfg, device
            )
        finally:
            _release_gpu_inputs(state1)

        tracks = persistent_tracks_from_v18_predictions(
            state1["current"],
            state1["source_world_points"],
            state1["current_pose"],
            rec["anchors_xy_t0_m"],
            pred_res,
            pred_yaw,
            frame_dt_s=float(pcfg.frame_dt_s),
        )

        poses1 = [
            np.asarray(source.pose(tok), dtype=np.float64)
            for tok in w.future_tokens[:6]
        ]
        poses2 = [
            np.asarray(source.pose(tok), dtype=np.float64)
            for tok in w.future_tokens[6:12]
        ]
        rec2 = prepare_causal_arrays_from_tracks(
            tracks,
            pred1,
            poses1,
            grid=pcfg.grid,
            free_label=int(pcfg.free_label),
            frame_dt_s=float(pcfg.frame_dt_s),
        )
        state2 = _state_from_persistent_tracks(
            tracks,
            rec2,
            pred1,
            poses1,
            poses2,
            pcfg,
            strong_cfg,
            device,
        )
        _stage_gpu_inputs(state2, device)
        try:
            pred2 = _forecast_once(
                model, state2, pcfg, strong_cfg, device
            )
        finally:
            _release_gpu_inputs(state2)

        # Static memory is built only from actual lidar-observed history.
        raw_hist = load_nuscenes_window_raw(
            source, w, pcfg, include_gt=False
        )
        static_mem = StaticWorldMemory.from_history(
            raw_hist["history_occ"],
            raw_hist["history_observed"],
            raw_hist["history_poses"],
            grid=pcfg.grid,
            free_label=int(pcfg.free_label),
        )
        pred2_static = []
        for h in range(FUTURE_FRAMES):
            proposal = static_mem.render(
                poses2[h],
                grid=pcfg.grid,
                free_label=int(pcfg.free_label),
            )
            pred2_static.append(
                protected_add_only(
                    pred2[h],
                    proposal,
                    free_label=int(pcfg.free_label),
                )
            )

        for hi, horizon in enumerate(REPORT_HORIZONS):
            block, rel_idx = REPORT_INDEX[horizon]
            abs_idx = int(
                round(horizon / float(pcfg.frame_dt_s))
            ) - 1
            ftok = str(w.future_tokens[abs_idx])
            gt = np.asarray(
                source.load_semantics(w.scene_name, ftok), dtype=np.uint8
            )
            moving, _, _ = gt_moving_support_for_horizon(
                source.nusc,
                str(w.t0_token),
                ftok,
                float(horizon),
                grid=pcfg.grid,
            )
            if block == "first":
                p_source = pred1[rel_idx]
                p_static = pred1[rel_idx]
            else:
                p_source = pred2[rel_idx]
                p_static = pred2_static[rel_idx]
            _update_raw(
                raw["persistent_source"],
                hi,
                p_source,
                gt,
                moving,
                int(pcfg.free_label),
            )
            _update_raw(
                raw["persistent_source_static"],
                hi,
                p_static,
                gt,
                moving,
                int(pcfg.free_label),
            )

        if wi == 1 or wi % int(max(a.progress_every, 1)) == 0 or wi == len(selected):
            elapsed = max(time.perf_counter() - started, 1e-9)
            print(
                f"v19_memory_rollout {wi}/{len(selected)} "
                f"rate={wi/elapsed:.3f} win/s "
                f"persistent_sources={len(tracks)} static_voxels={len(static_mem)}",
                flush=True,
            )

    metrics = {name: _finalize(v) for name, v in raw.items()}
    result = {
        "protocol": PROTOCOL,
        "checkpoint": str(Path(a.checkpoint).resolve()),
        "checkpoint_epoch": int(ck.get("epoch", -1)),
        "population_total_windows": int(total_selected),
        "num_windows": int(len(selected)),
        "shard": {
            "num_shards": ns,
            "shard_index": si,
            "selection": "global_selected[shard_index::num_shards]",
        },
        "future_gt_used_for_prediction": False,
        "future_ego_pose_used_through_s": 6.0,
        "variant_contracts": {
            "persistent_source": (
                "first block frozen Clean-E14; second block reuses predicted "
                "source trajectories/canonical geometry directly without "
                "component re-detection or identity matching"
            ),
            "persistent_source_static": (
                "persistent_source plus add-only six-frame lidar-observed "
                "non-dynamic world-coordinate static memory"
            ),
        },
        "metrics": metrics,
        "raw_counts": {
            name: {k: np.asarray(v).tolist() for k, v in rr.items()}
            for name, rr in raw.items()
        },
    }
    op = Path(a.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\\n=== V19 ZERO-TRAINING MEMORY ROLLOUT ===")
    for name in VARIANTS:
        print("\\n", name)
        for h in REPORT_HORIZONS:
            row = metrics[name]["per_horizon"][str(h)]
            print(
                f"{h:4.1f}s IoU={row['IoU']:.3f} mIoU={row['mIoU']:.3f} "
                f"MovMacro={row['MovingMacro']:.3f} "
                f"MovMicro={row['MovingMicro']:.3f}"
            )
        print("AVG4/5/6", json.dumps(metrics[name]["average_4s_5s_6s"]))
    print(f"saved {op}")


if __name__ == "__main__":
    main()
