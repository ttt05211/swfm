#!/usr/bin/env python3
"""Zero-shot 1--6 s block rollout for frozen Clean-E14 V18.

The frozen model was trained for six relative future horizons (0.5--3.0 s) and
therefore cannot be made a 12-query model by simply changing FUTURE_FRAMES.
This evaluator instead reuses the exact frozen six-query model twice:

    real history [-2.5, ..., 0.0]
      -> Clean-E14 -> predicted [0.5, ..., 3.0]

    predicted [0.5, ..., 3.0] + their ego poses
      -> rebuild causal sources/tracks/KTA/local tubes
      -> same Clean-E14 -> predicted [3.5, ..., 6.0]

No weights, queries, losses or training data are changed.  The second block is a
true open-loop distribution-shift test because its six history occupancy grids
are model predictions.

The first block uses the frozen cached causal representation exactly.  On the
first selected window, a from-occupancy rebuild is required to reproduce the
cached representation and dense first-block forecast before the long rollout is
allowed to continue.  This gates the synthetic second-block representation
builder against the already-frozen deployment contract.

Future ego poses through 6 s are provided, matching the existing V18/OccFM-Fut
conditioning contract.  Future semantic occupancy and future instance
annotations are used only for metrics.
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

from real_motion.local_st_world_model import (
    build_local_semantic_tubes,
    history_offsets_from_features,
)
from real_motion.local_st_world_model_v17 import (
    frame_motion_features_from_flat,
    target_source_mask_from_tube,
)
from real_motion.metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS
from real_motion.motion_transport import (
    FUTURE_FRAMES,
    HISTORY_FRAMES,
    backward_component_tracks,
    build_source_features,
    world_points_to_t0,
    world_vec_to_t0,
)
from real_motion.nuscenes_adapter import (
    NuScenesWindowSource,
    gt_moving_support_for_horizon,
)
from real_motion.runtime_config import (
    add_config_args,
    load_runtime_config,
    make_prepare_config,
)
from real_motion.runtime_fastpath import (
    baseline_clear_flat_indices,
    baseline_clear_mask,
)
from real_motion.strong_w2det import (
    StrongW2DetConfig,
    extract_instances,
    match_instances,
)
from tools.real_motion import eval_p0_f9_v18_se2 as base
from tools.real_motion import eval_p0_f9_v18_full_validation as full
from tools.real_motion.benchmark_p0_f9_v18_runtime import (
    _forecast_once,
    _precompute_source_world,
    _prepare_record,
    _release_gpu_inputs,
    _stage_gpu_inputs,
    _strong_all_horizons,
)
from tools.real_motion.eval_p0_f9_v17_local_stwm import window_from_record
from tools.real_motion.train_p0_f9_v18_se2_clean import PROTOCOL as CLEAN_PROTOCOL

PROTOCOL = "p0_f9_v18_zero_shot_block_rollout_6s_v1"
REPORT_HORIZONS = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
# first block: 1/2/3 s -> 0-based future indices 1/3/5
# second block starts at 3 s: 4/5/6 s -> relative indices 1/3/5
REPORT_INDEX = {
    1.0: ("first", 1),
    2.0: ("first", 3),
    3.0: ("first", 5),
    4.0: ("second", 1),
    5.0: ("second", 3),
    6.0: ("second", 5),
}
SEMANTIC_CLASSES = tuple(range(17))


class CachedSource(NuScenesWindowSource):
    from functools import lru_cache

    @lru_cache(maxsize=768)
    def load_semantics(self, scene_name, token):
        return super().load_semantics(scene_name, token)

    @lru_cache(maxsize=4096)
    def pose(self, token):
        return super().pose(token)


def _new_raw():
    H = len(REPORT_HORIZONS)
    return {
        "occ_inter": np.zeros(H, dtype=np.int64),
        "occ_union": np.zeros(H, dtype=np.int64),
        "sem_inter": np.zeros((H, len(SEMANTIC_CLASSES)), dtype=np.int64),
        "sem_union": np.zeros((H, len(SEMANTIC_CLASSES)), dtype=np.int64),
        "mov_inter": np.zeros((H, len(DYNAMIC_CLASS_IDS)), dtype=np.int64),
        "mov_union": np.zeros((H, len(DYNAMIC_CLASS_IDS)), dtype=np.int64),
    }


def _update_raw(raw, hi, pred, gt, moving, free_label):
    pred = np.asarray(pred)
    gt = np.asarray(gt)
    moving = np.asarray(moving, dtype=bool)
    if pred.shape != gt.shape or moving.shape != gt.shape:
        raise ValueError("metric shape mismatch")
    po = pred != int(free_label)
    go = gt != int(free_label)
    raw["occ_inter"][hi] += int(np.logical_and(po, go).sum())
    raw["occ_union"][hi] += int(np.logical_or(po, go).sum())
    for j, cid in enumerate(SEMANTIC_CLASSES):
        p = pred == int(cid)
        g = gt == int(cid)
        raw["sem_inter"][hi, j] += int(np.logical_and(p, g).sum())
        raw["sem_union"][hi, j] += int(np.logical_or(p, g).sum())
    for j, cid in enumerate(DYNAMIC_CLASS_IDS):
        p = (pred == int(cid)) & moving
        g = (gt == int(cid)) & moving
        raw["mov_inter"][hi, j] += int(np.logical_and(p, g).sum())
        raw["mov_union"][hi, j] += int(np.logical_or(p, g).sum())


def _safe_iou(inter, union):
    inter = np.asarray(inter, dtype=np.float64)
    union = np.asarray(union, dtype=np.float64)
    out = np.full(inter.shape, np.nan, dtype=np.float64)
    np.divide(inter, union, out=out, where=union > 0)
    return 100.0 * out


def _finalize(raw):
    occ = _safe_iou(raw["occ_inter"], raw["occ_union"])
    sem = _safe_iou(raw["sem_inter"], raw["sem_union"])
    mov = _safe_iou(raw["mov_inter"], raw["mov_union"])
    sem_h = np.nanmean(sem, axis=1)
    mov_macro = np.nanmean(mov, axis=1)
    mov_micro = _safe_iou(
        raw["mov_inter"].sum(axis=1),
        raw["mov_union"].sum(axis=1),
    )
    per = {}
    for hi, h in enumerate(REPORT_HORIZONS):
        per[str(h)] = {
            "IoU": float(occ[hi]),
            "mIoU": float(sem_h[hi]),
            "MovingMacro": float(mov_macro[hi]),
            "MovingMicro": float(mov_micro[hi]),
            "moving_per_class": {
                str(int(cid)): float(mov[hi, j])
                for j, cid in enumerate(DYNAMIC_CLASS_IDS)
            },
        }
    return {
        "per_horizon": per,
        "average_1s_2s_3s": {
            "IoU": float(np.nanmean(occ[:3])),
            "mIoU": float(np.nanmean(sem_h[:3])),
            "MovingMacro": float(np.nanmean(mov_macro[:3])),
            "MovingMicro": float(np.nanmean(mov_micro[:3])),
        },
        "average_4s_5s_6s": {
            "IoU": float(np.nanmean(occ[3:])),
            "mIoU": float(np.nanmean(sem_h[3:])),
            "MovingMacro": float(np.nanmean(mov_macro[3:])),
            "MovingMicro": float(np.nanmean(mov_micro[3:])),
        },
    }


def _kta_tensors(current, velocities, current_pose, frame_dt_s):
    """Rebuild frozen source/KTA/anchor tensors with the cache arithmetic order.

    The cache builder computes each future anchor in float64 as
    anchor = cur_t0 + v_t0 * dt and only then casts to float32.
    Adding source_xy(float32) + kta(float32) changes the rounding order and
    can differ by a few float32 ULPs.  Preserve the original construction.
    """
    n = len(current)
    source_xy = np.zeros((n, 2), dtype=np.float32)
    kta = np.zeros((n, FUTURE_FRAMES, 2), dtype=np.float32)
    anchors = np.zeros((n, FUTURE_FRAMES, 2), dtype=np.float32)
    for i, comp in enumerate(current):
        center = np.asarray(comp["centroid_world"], dtype=np.float64)
        c0 = world_points_to_t0(center[None], current_pose)[0, :2]
        source_xy[i] = c0.astype(np.float32)
        vw = np.asarray(velocities.get(i, np.zeros(3)), dtype=np.float64)
        vt0 = world_vec_to_t0(vw, current_pose)[:2]
        for h in range(FUTURE_FRAMES):
            dt = (h + 1) * float(frame_dt_s)
            kd = vt0 * dt
            kta[i, h] = kd.astype(np.float32)
            anchors[i, h] = (c0 + kd).astype(np.float32)
    return source_xy, kta, anchors


def _build_block_state(
    history_occ,
    history_poses,
    future_poses,
    pcfg,
    strong_cfg,
    device,
):
    if len(history_occ) != HISTORY_FRAMES or len(history_poses) != HISTORY_FRAMES:
        raise ValueError("block history must contain six frames")
    if len(future_poses) != FUTURE_FRAMES:
        raise ValueError("block future poses must contain six frames")

    history_occ = [np.asarray(x, dtype=np.uint8) for x in history_occ]
    history_poses = [np.asarray(x, dtype=np.float64) for x in history_poses]
    future_poses = [np.asarray(x, dtype=np.float64) for x in future_poses]

    components_by_frame = [
        extract_instances(sem, pose, grid=pcfg.grid, cfg=strong_cfg)
        for sem, pose in zip(history_occ, history_poses)
    ]
    current = components_by_frame[-1]
    previous = components_by_frame[-2]
    velocities = match_instances(
        previous,
        current,
        float(pcfg.frame_dt_s),
        max_speed_mps=strong_cfg.max_match_speed_mps,
    )
    tracks, track_valid = backward_component_tracks(
        components_by_frame,
        frame_dt_s=float(pcfg.frame_dt_s),
        max_speed_mps=float(strong_cfg.max_match_speed_mps),
    )
    features_np = build_source_features(
        current,
        velocities,
        tracks,
        track_valid,
        history_poses[-1],
        frame_dt_s=float(pcfg.frame_dt_s),
        grid=pcfg.grid,
    )
    features = torch.from_numpy(features_np)
    source_xy, kta_np, anchors_np = _kta_tensors(
        current,
        velocities,
        history_poses[-1],
        float(pcfg.frame_dt_s),
    )
    kta = torch.from_numpy(kta_np)
    offsets = history_offsets_from_features(features)
    tube_np = build_local_semantic_tubes(
        history_occ,
        history_poses,
        source_xy,
        offsets,
        track_valid,
        grid=pcfg.grid,
        free_label=int(pcfg.free_label),
    )
    tube = torch.from_numpy(tube_np)
    class_ids = torch.as_tensor(
        [int(c["class_id"]) for c in current], dtype=torch.long
    )
    track_valid_t = torch.from_numpy(track_valid)
    frame_motion = frame_motion_features_from_flat(features)
    source_mask = target_source_mask_from_tube(
        tube,
        class_ids,
        track_valid_t,
        features,
    )
    anchors_xy = torch.from_numpy(anchors_np)

    rec = {
        "features": features,
        "local_semantic_tube": tube,
        "kta_displacement_xy_m": kta,
        "frame_motion_features": frame_motion,
        "target_source_mask_tube": source_mask,
        "source_class_id": class_ids,
        "anchors_xy_t0_m": anchors_xy,
    }

    current_pose = history_poses[-1]
    current_sem = history_occ[-1]
    source_world_points = _precompute_source_world(
        current, current_pose, pcfg.grid
    )
    source_rel_xy = [
        np.asarray(pts, dtype=np.float64)[:, :2]
        - np.asarray(comp["centroid_world"], dtype=np.float64)[None, :2]
        for pts, comp in zip(source_world_points, current)
    ]
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
    baseline_clear_by_hi = [
        baseline_clear_mask(rows, grid=pcfg.grid) for rows in baseline_by_hi
    ]
    baseline_clear_flat_by_hi = [
        baseline_clear_flat_indices(rows, grid=pcfg.grid)
        for rows in baseline_by_hi
    ]
    return {
        "rec": rec,
        "window": None,
        "scene": None,
        "current_sem": current_sem,
        "previous_sem": history_occ[-2],
        "current_pose": current_pose,
        "previous_pose": history_poses[-2],
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
        "world_to_future": [
            np.linalg.inv(np.asarray(p, dtype=np.float64)) for p in future_poses
        ],
        "gpu": None,
    }


def _assert_tensor_close(name, a, b):
    aa = torch.as_tensor(a).cpu()
    bb = torch.as_tensor(b).cpu()
    if aa.shape != bb.shape:
        raise RuntimeError(f"{name}: shape mismatch {tuple(aa.shape)} != {tuple(bb.shape)}")
    if aa.dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
        if not torch.allclose(aa.float(), bb.float(), rtol=0.0, atol=1e-6):
            d = float((aa.float() - bb.float()).abs().max().item()) if aa.numel() else 0.0
            raise RuntimeError(f"{name}: max abs mismatch {d}")
    elif not torch.equal(aa, bb):
        raise RuntimeError(f"{name}: tensor mismatch")


def _gate_synthetic_builder(
    rec,
    source,
    pcfg,
    strong_cfg,
    device,
    model,
):
    w = window_from_record(rec)
    history_occ = [
        np.asarray(source.load_semantics(w.scene_name, tok), dtype=np.uint8)
        for tok in w.history_tokens
    ]
    history_poses = [
        np.asarray(source.pose(tok), dtype=np.float64)
        for tok in w.history_tokens
    ]
    future_poses = [
        np.asarray(source.pose(tok), dtype=np.float64)
        for tok in w.future_tokens
    ]
    syn = _build_block_state(
        history_occ, history_poses, future_poses, pcfg, strong_cfg, device
    )
    for key in (
        "features",
        "local_semantic_tube",
        "kta_displacement_xy_m",
        "frame_motion_features",
        "target_source_mask_tube",
        "source_class_id",
        "anchors_xy_t0_m",
    ):
        _assert_tensor_close(key, rec[key], syn["rec"][key])

    ref = _prepare_record(rec, source, pcfg, strong_cfg, device)
    _stage_gpu_inputs(ref, device)
    _stage_gpu_inputs(syn, device)
    try:
        ref_pred = _forecast_once(model, ref, pcfg, strong_cfg, device)
        syn_pred = _forecast_once(model, syn, pcfg, strong_cfg, device)
    finally:
        _release_gpu_inputs(ref)
        _release_gpu_inputs(syn)
    for h, (a, b) in enumerate(zip(ref_pred, syn_pred)):
        if not np.array_equal(a, b):
            n = int(np.count_nonzero(np.asarray(a) != np.asarray(b)))
            raise RuntimeError(
                f"synthetic block builder dense forecast mismatch h={h}: {n} voxels"
            )
    print(
        "ZERO-SHOT ROLLOUT EXACTNESS: synthetic real-history rebuild reproduces "
        "frozen cached representation + six-frame dense forecast PASS",
        flush=True,
    )


def _progress_snapshot(path, processed, raw, total):
    payload = {
        "protocol": PROTOCOL,
        "processed_windows": int(processed),
        "total_windows": int(total),
        "partial_metrics": _finalize(raw),
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


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
    p.add_argument("--device", default="cuda")
    p.add_argument("--progress-every", type=int, default=25)
    a = p.parse_args()

    cfg = load_runtime_config(a.config, a.override)
    pcfg = make_prepare_config(cfg)
    if int(pcfg.history_frames) != HISTORY_FRAMES or int(pcfg.future_frames) != FUTURE_FRAMES:
        raise RuntimeError("frozen V18 requires the 6-history + 6-future relative block contract")

    _, records = base.load_cache(a.val_cache)
    rec_by_t0 = {str(r["t0_token"]): r for r in records}
    if len(rec_by_t0) != len(records):
        raise RuntimeError("validation cache has duplicate t0 tokens")

    device = torch.device(
        a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    ck, model, _ = full._load_model(a.checkpoint, CLEAN_PROTOCOL, device)
    source = CachedSource(a.dataroot, info_pkl=a.info_pkl, verbose=False)
    strong_cfg = StrongW2DetConfig(free_label=int(pcfg.free_label))

    long_windows = list(
        source.iter_windows(history=6, future=12, stride=1, max_windows=None)
    )
    selected = []
    for w in long_windows:
        rec = rec_by_t0.get(str(w.t0_token))
        if rec is None:
            continue
        rw = window_from_record(rec)
        if tuple(str(x) for x in rw.history_tokens) != tuple(str(x) for x in w.history_tokens):
            raise RuntimeError(f"{w.t0_token}: history token mismatch")
        if tuple(str(x) for x in rw.future_tokens) != tuple(str(x) for x in w.future_tokens[:6]):
            raise RuntimeError(f"{w.t0_token}: first-block future token mismatch")
        selected.append((w, rec))

    if int(a.expected_windows) > 0 and len(selected) != int(a.expected_windows):
        raise RuntimeError(
            f"eligible 6+12 cached windows {len(selected)} != expected {a.expected_windows}"
        )
    if int(a.max_windows) > 0:
        selected = selected[: min(len(selected), int(a.max_windows))]
    if not selected:
        raise RuntimeError("no eligible long-horizon windows")

    # Validate the synthetic history builder against the frozen main path once.
    _gate_synthetic_builder(
        selected[0][1], source, pcfg, strong_cfg, device, model
    )

    raw = _new_raw()
    started = time.perf_counter()
    progress_path = str(Path(a.output).with_suffix(".progress.json"))

    for wi, (w, rec) in enumerate(selected, start=1):
        # Block 1: exact frozen deployment path from real history.
        state1 = _prepare_record(rec, source, pcfg, strong_cfg, device)
        _stage_gpu_inputs(state1, device)
        try:
            pred1 = _forecast_once(model, state1, pcfg, strong_cfg, device)
        finally:
            _release_gpu_inputs(state1)

        # Block 2: exact same six-query model from its own six predicted frames.
        poses1 = [
            np.asarray(source.pose(tok), dtype=np.float64)
            for tok in w.future_tokens[:6]
        ]
        poses2 = [
            np.asarray(source.pose(tok), dtype=np.float64)
            for tok in w.future_tokens[6:12]
        ]
        state2 = _build_block_state(
            pred1, poses1, poses2, pcfg, strong_cfg, device
        )
        _stage_gpu_inputs(state2, device)
        try:
            pred2 = _forecast_once(model, state2, pcfg, strong_cfg, device)
        finally:
            _release_gpu_inputs(state2)

        for hi, h in enumerate(REPORT_HORIZONS):
            block, rel_idx = REPORT_INDEX[h]
            pred = pred1[rel_idx] if block == "first" else pred2[rel_idx]
            abs_idx = int(round(h / float(pcfg.frame_dt_s))) - 1
            ftok = str(w.future_tokens[abs_idx])
            gt = np.asarray(
                source.load_semantics(w.scene_name, ftok), dtype=np.uint8
            )
            moving, _, _ = gt_moving_support_for_horizon(
                source.nusc,
                str(w.t0_token),
                ftok,
                float(h),
                grid=pcfg.grid,
            )
            _update_raw(
                raw,
                hi,
                pred,
                gt,
                moving,
                int(pcfg.free_label),
            )

        if wi == 1 or wi % int(max(a.progress_every, 1)) == 0 or wi == len(selected):
            elapsed = max(time.perf_counter() - started, 1e-9)
            print(
                f"zero_shot_long_rollout {wi}/{len(selected)} "
                f"rate={wi/elapsed:.3f} win/s "
                f"block2_sources={len(state2['current'])}",
                flush=True,
            )
            _progress_snapshot(progress_path, wi, raw, len(selected))

    metrics = _finalize(raw)
    result = {
        "protocol": PROTOCOL,
        "checkpoint": str(Path(a.checkpoint).resolve()),
        "checkpoint_epoch": int(ck.get("epoch", -1)),
        "checkpoint_global_step": int(ck.get("global_step", -1)),
        "val_cache": str(Path(a.val_cache).resolve()),
        "num_windows": int(len(selected)),
        "num_scenes": int(len({str(w.scene_name) for w, _ in selected})),
        "history_frames_per_block": HISTORY_FRAMES,
        "relative_future_frames_per_block": FUTURE_FRAMES,
        "rollout_blocks": 2,
        "report_horizons_s": list(REPORT_HORIZONS),
        "future_gt_used_for_prediction": False,
        "future_ego_pose_used_through_s": 6.0,
        "conditioning_contract": (
            "same GT-future-ego information class as frozen V18/OccFM-Fut; "
            "second block consumes only first-block predicted occupancies plus poses"
        ),
        "rollout_contract": (
            "frozen Clean-E14 0.5--3.0 s parallel block, then rebuild causal "
            "Strong sources/backward tracks/KTA/local semantic tubes from those "
            "six predictions and reuse the same frozen checkpoint for 3.5--6.0 s"
        ),
        "exactness_gate": (
            "first selected real-history window synthetic rebuild must reproduce "
            "cached tensors and all six dense frozen outputs before evaluation"
        ),
        "population_contract": (
            "all validation windows with 6 history + 12 contiguous future frames "
            "that also have a frozen V18 validation-cache record"
        ),
        "metrics": metrics,
    }
    op = Path(a.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n=== V18 ZERO-SHOT BLOCK ROLLOUT 1--6s ===")
    print(
        f"{'horizon':>8s} {'IoU':>9s} {'mIoU':>9s} "
        f"{'MovMacro':>10s} {'MovMicro':>10s}"
    )
    for h in REPORT_HORIZONS:
        x = metrics["per_horizon"][str(h)]
        print(
            f"{h:8.1f} {x['IoU']:9.3f} {x['mIoU']:9.3f} "
            f"{x['MovingMacro']:10.3f} {x['MovingMicro']:10.3f}"
        )
    print("AVG 1/2/3:", json.dumps(metrics["average_1s_2s_3s"]))
    print("AVG 4/5/6:", json.dumps(metrics["average_4s_5s_6s"]))
    print(f"saved {op}")


if __name__ == "__main__":
    main()
