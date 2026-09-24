#!/usr/bin/env python3
"""Temporal-consistency diagnosis for the frozen V18 forecasting stack.

The diagnosis is deliberately non-training:

1. export Strong/KTA and Y600 six-frame clips on the exact all-window V18 val set;
2. reuse the already exported frozen Clean-E14 clips;
3. compare source-trajectory velocity/acceleration/jerk and yaw-rate/yaw-accel
   errors against the frozen SE(2) supervision;
4. compare frame-to-frame occupancy/semantic change-mask statistics against GT.

FVD itself is computed afterwards with eval_occfm_occupancy_fvd_only.py so all
three variants use the same released OccFM temporal 3D-VAE extractor.

Nothing in this file changes the frozen model, targets or evaluator.
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

from real_motion.runtime_config import add_config_args, load_runtime_config, make_prepare_config
from real_motion.strong_w2det import StrongW2DetConfig
from tools.real_motion import eval_p0_f9_v18_full_validation as full
from tools.real_motion.benchmark_p0_f9_v18_runtime import (
    CachedSource,
    _exactness_check,
    _forecast_once,
    _model_forward,
    _prepare_record,
    _release_gpu_inputs,
    _stage_gpu_inputs,
)
from tools.real_motion.eval_p0_f9_v17_local_stwm import window_from_record
from tools.real_motion.train_p0_f9_v18_se2_clean import PROTOCOL as CLEAN_PROTOCOL
from tools.real_motion.train_p0_f9_v18_se2_pair import PROTOCOL as PAIR_PROTOCOL

PROTOCOL = "p0_f9_v18_temporal_consistency_diagnostic_v1"
DT = 0.5
VARIANTS = ("strong_anchor", "y600_pred", "clean_pred")


def _wrap(x):
    return np.arctan2(np.sin(x), np.cos(x))


def _append(store, key, x):
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size:
        store.setdefault(key, []).extend(arr.tolist())


def _summary(x):
    a = np.asarray(x, dtype=np.float64)
    if a.size == 0:
        return {"n": 0, "mean": float("nan"), "median": float("nan"), "p90": float("nan")}
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p90": float(np.quantile(a, 0.90)),
    }


def _motion_stats_append(dst, pred_disp, pred_yaw, rec):
    gt = rec["target_source_displacement_xy_m"].float().numpy()
    valid = rec["se2_target_valid"].bool().numpy()
    sup = rec["supervised_source"].bool().numpy()
    valid = valid & sup[:, None]
    if pred_disp.shape != gt.shape:
        raise RuntimeError(f"motion shape mismatch pred={pred_disp.shape} gt={gt.shape}")

    # Relative displacements are from t0, so prepend the exact t0 origin.
    p = np.concatenate([np.zeros((len(pred_disp), 1, 2)), pred_disp], axis=1)
    g = np.concatenate([np.zeros((len(gt), 1, 2)), gt], axis=1)
    vp = np.diff(p, axis=1) / DT
    vg = np.diff(g, axis=1) / DT

    vel_valid = valid.copy()
    _append(dst, "velocity_error_mps", np.linalg.norm(vp - vg, axis=-1)[vel_valid])

    ap = np.diff(vp, axis=1) / DT
    ag = np.diff(vg, axis=1) / DT
    acc_valid = valid[:, 1:] & valid[:, :-1]
    _append(dst, "acceleration_error_mps2", np.linalg.norm(ap - ag, axis=-1)[acc_valid])
    _append(dst, "pred_acceleration_magnitude_mps2", np.linalg.norm(ap, axis=-1)[acc_valid])
    _append(dst, "gt_acceleration_magnitude_mps2", np.linalg.norm(ag, axis=-1)[acc_valid])

    jp = np.diff(ap, axis=1) / DT
    jg = np.diff(ag, axis=1) / DT
    jerk_valid = valid[:, 2:] & valid[:, 1:-1] & valid[:, :-2]
    _append(dst, "jerk_error_mps3", np.linalg.norm(jp - jg, axis=-1)[jerk_valid])
    _append(dst, "pred_jerk_magnitude_mps3", np.linalg.norm(jp, axis=-1)[jerk_valid])
    _append(dst, "gt_jerk_magnitude_mps3", np.linalg.norm(jg, axis=-1)[jerk_valid])

    gt_yaw = rec["target_yaw_rad"].float().numpy()
    yaw_valid = (
        rec["yaw_label_valid"].bool().numpy()
        & valid
        & rec["yaw_enabled"].bool().numpy()[:, None]
    )
    py = np.concatenate([np.zeros((len(pred_yaw), 1)), pred_yaw], axis=1)
    gy = np.concatenate([np.zeros((len(gt_yaw), 1)), gt_yaw], axis=1)
    p_rate = _wrap(np.diff(py, axis=1)) / DT
    g_rate = _wrap(np.diff(gy, axis=1)) / DT
    _append(dst, "yaw_rate_error_radps", np.abs(_wrap((p_rate - g_rate) * DT) / DT)[yaw_valid])

    p_acc = _wrap(np.diff(p_rate, axis=1) * DT) / (DT * DT)
    g_acc = _wrap(np.diff(g_rate, axis=1) * DT) / (DT * DT)
    ya_valid = yaw_valid[:, 1:] & yaw_valid[:, :-1]
    _append(dst, "yaw_accel_error_radps2", np.abs(p_acc - g_acc)[ya_valid])


def _change_stats_append(dst, pred, gt, free_label):
    pred = np.asarray(pred, dtype=np.uint8)
    gt = np.asarray(gt, dtype=np.uint8)
    if pred.shape != (6, 200, 200, 16) or gt.shape != pred.shape:
        raise RuntimeError(f"clip shape mismatch pred={pred.shape} gt={gt.shape}")
    for t in range(1, 6):
        ps = pred[t] != pred[t - 1]
        gs = gt[t] != gt[t - 1]
        po = (pred[t] != free_label) != (pred[t - 1] != free_label)
        go = (gt[t] != free_label) != (gt[t - 1] != free_label)
        _append(dst, "semantic_change_rate", [ps.mean()])
        _append(dst, "gt_semantic_change_rate", [gs.mean()])
        _append(dst, "semantic_change_excess", [ps.mean() - gs.mean()])
        _append(dst, "occupancy_flip_rate", [po.mean()])
        _append(dst, "gt_occupancy_flip_rate", [go.mean()])
        _append(dst, "occupancy_flip_excess", [po.mean() - go.mean()])
        su = np.logical_or(ps, gs).sum()
        ou = np.logical_or(po, go).sum()
        _append(dst, "semantic_change_mask_iou", [
            np.logical_and(ps, gs).sum() / su if su else 1.0
        ])
        _append(dst, "occupancy_flip_mask_iou", [
            np.logical_and(po, go).sum() / ou if ou else 1.0
        ])


def _valid_clip(path: Path):
    if not path.exists():
        return False
    try:
        with np.load(path) as x:
            p = np.asarray(x["pred"])
            g = np.asarray(x["gt"])
        return p.shape == (6, 200, 200, 16) and g.shape == p.shape
    except Exception:
        return False


def _save_clip(path, pred, gt, sid, scene):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        pred=np.asarray(pred, dtype=np.uint8),
        gt=np.asarray(gt, dtype=np.uint8),
        sample_id=np.asarray(str(sid)),
        scene_name=np.asarray(str(scene)),
    )


def _load_clean(path: Path, expected_sid: str):
    with np.load(path) as x:
        pred = np.asarray(x["pred"], dtype=np.uint8)
        gt = np.asarray(x["gt"], dtype=np.uint8)
        sid = str(x["sample_id"].item()) if "sample_id" in x.files else ""
    if sid and sid != expected_sid:
        raise RuntimeError(f"clean clip ordering mismatch: {sid} != {expected_sid}")
    return pred, gt


def main():
    p = argparse.ArgumentParser()
    add_config_args(p)
    p.add_argument("--val-cache", required=True)
    p.add_argument("--y600-checkpoint", required=True)
    p.add_argument("--clean-checkpoint", required=True)
    p.add_argument("--dataroot", required=True)
    p.add_argument("--info-pkl", required=True)
    p.add_argument("--clean-clip-dir", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--expected-windows", type=int, default=4369)
    p.add_argument("--max-windows", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()

    cfg = load_runtime_config(a.config, a.override)
    pcfg = make_prepare_config(cfg)
    _, records = full.base.load_cache(a.val_cache)
    if a.expected_windows and len(records) != int(a.expected_windows):
        raise RuntimeError(f"val windows {len(records)} != expected {a.expected_windows}")
    if a.max_windows > 0:
        records = records[: min(len(records), a.max_windows)]

    device = torch.device(a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu")
    y_ck, y_model, y_cfg = full._load_model(a.y600_checkpoint, PAIR_PROTOCOL, device)
    c_ck, c_model, c_cfg = full._load_model(a.clean_checkpoint, CLEAN_PROTOCOL, device)
    if y_cfg != c_cfg:
        raise RuntimeError("Y600/Clean model configs differ")

    source = CachedSource(a.dataroot, info_pkl=a.info_pkl, verbose=False)
    strong_cfg = StrongW2DetConfig(free_label=int(pcfg.free_label))
    out_root = Path(a.output_root)
    strong_dir = out_root / "strong_anchor"
    y600_dir = out_root / "y600_pred"
    clean_dir = Path(a.clean_clip_dir)

    motion = {name: {} for name in VARIANTS}
    change = {name: {} for name in VARIANTS}
    exactness_done = False
    started = time.perf_counter()

    for i, rec in enumerate(records):
        sid = str(rec["sample_id"])
        state = _prepare_record(rec, source, pcfg, strong_cfg, device)
        _stage_gpu_inputs(state, device)
        try:
            if not exactness_done:
                _exactness_check(c_model, state, pcfg, strong_cfg, device)
                exactness_done = True

            # One model pass per learned variant for trajectory diagnostics.
            yo = _model_forward(y_model, state["gpu"], device)
            co = _model_forward(c_model, state["gpu"], device)
            y_res = yo["residual_xy_m"].float().cpu().numpy()
            c_res = co["residual_xy_m"].float().cpu().numpy()
            y_yaw = yo["yaw_delta_rad"].float().cpu().numpy()
            c_yaw = co["yaw_delta_rad"].float().cpu().numpy()

            kta_disp = rec["kta_displacement_xy_m"].float().numpy()
            _motion_stats_append(motion["strong_anchor"], kta_disp, np.zeros(kta_disp.shape[:2]), rec)
            _motion_stats_append(motion["y600_pred"], kta_disp + y_res, y_yaw, rec)
            _motion_stats_append(motion["clean_pred"], kta_disp + c_res, c_yaw, rec)

            y_path = y600_dir / f"clip_{i:05d}.npz"
            if bool(a.resume) and _valid_clip(y_path):
                with np.load(y_path) as x:
                    y_pred = np.asarray(x["pred"], dtype=np.uint8)
            else:
                y_pred = np.stack(
                    _forecast_once(y_model, state, pcfg, strong_cfg, device), axis=0
                ).astype(np.uint8, copy=False)
        finally:
            _release_gpu_inputs(state)

        strong_pred = np.stack(state["anchors"], axis=0).astype(np.uint8, copy=False)
        clean_path = clean_dir / f"clip_{i:05d}.npz"
        if not _valid_clip(clean_path):
            raise FileNotFoundError(f"missing frozen Clean clip: {clean_path}")
        clean_pred, clean_gt = _load_clean(clean_path, sid)

        w = window_from_record(rec)
        gt = np.stack(
            [
                np.asarray(source.load_semantics(str(w.scene_name), str(tok)), dtype=np.uint8)
                for tok in w.future_tokens
            ],
            axis=0,
        )
        if not np.array_equal(clean_gt, gt):
            raise RuntimeError(f"{sid}: frozen Clean clip GT does not match current val window")
        if i == 0:
            # Verify the frozen Clean clip was generated by the same final model/path.
            _stage_gpu_inputs(state, device)
            try:
                clean_ref = np.stack(
                    _forecast_once(c_model, state, pcfg, strong_cfg, device), axis=0
                ).astype(np.uint8, copy=False)
            finally:
                _release_gpu_inputs(state)
            if not np.array_equal(clean_ref, clean_pred):
                n = int(np.count_nonzero(clean_ref != clean_pred))
                raise RuntimeError(f"frozen Clean clip mismatch on first window: {n} voxels")

        s_path = strong_dir / f"clip_{i:05d}.npz"
        y_path = y600_dir / f"clip_{i:05d}.npz"
        if not (bool(a.resume) and _valid_clip(s_path)):
            _save_clip(s_path, strong_pred, gt, sid, str(w.scene_name))
        if not (bool(a.resume) and _valid_clip(y_path)):
            _save_clip(y_path, y_pred, gt, sid, str(w.scene_name))

        _change_stats_append(change["strong_anchor"], strong_pred, gt, int(pcfg.free_label))
        _change_stats_append(change["y600_pred"], y_pred, gt, int(pcfg.free_label))
        _change_stats_append(change["clean_pred"], clean_pred, gt, int(pcfg.free_label))

        done = i + 1
        if done == 1 or done % 50 == 0 or done == len(records):
            dt = max(time.perf_counter() - started, 1e-9)
            print(f"temporal_diag {done}/{len(records)} rate={done/dt:.3f} win/s", flush=True)

    result = {
        "protocol": PROTOCOL,
        "num_windows": len(records),
        "val_cache": str(Path(a.val_cache).resolve()),
        "y600_checkpoint": str(Path(a.y600_checkpoint).resolve()),
        "clean_checkpoint": str(Path(a.clean_checkpoint).resolve()),
        "y600_epoch": int(y_ck.get("epoch", -1)),
        "clean_epoch": int(c_ck.get("epoch", -1)),
        "clip_dirs": {
            "strong_anchor": str(strong_dir.resolve()),
            "y600_pred": str(y600_dir.resolve()),
            "clean_pred": str(clean_dir.resolve()),
        },
        "trajectory": {
            name: {k: _summary(v) for k, v in motion[name].items()}
            for name in VARIANTS
        },
        "frame_change": {
            name: {k: _summary(v) for k, v in change[name].items()}
            for name in VARIANTS
        },
        "interpretation_contract": {
            "trajectory": (
                "GT-relative six-horizon source displacement; velocity/acceleration/jerk "
                "computed at 0.5 s spacing on supervised valid sources"
            ),
            "frame_change": (
                "five adjacent future-frame transitions per clip; compares semantic "
                "and occupied/free change masks to GT"
            ),
            "next_step": (
                "run the same corrected OccFM temporal-3D-VAE FVD on strong_anchor, "
                "y600_pred and clean_pred, then combine FVD direction with trajectory "
                "and change-mask evidence before any training change"
            ),
        },
    }
    op = Path(a.output_json)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\n=== V18 TEMPORAL CONSISTENCY DIAGNOSTIC ===")
    print(json.dumps(result, indent=2))
    print(f"saved {op}")


if __name__ == "__main__":
    main()
