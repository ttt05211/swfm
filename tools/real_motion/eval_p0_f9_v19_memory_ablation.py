#!/usr/bin/env python3
"""Zero-training 1--3 s V19 memory ablation on the frozen Clean-E14 protocol.

Variants:
  base                  frozen Clean-E14
  dormant_kta           + recently missing history sources, add-only KTA
  static_memory         + lidar-observed six-frame static world memory
  dormant_kta_static    + both

No V19 parameter is trained.  This is the causal gate before fitting the memory
adapter or residual innovation head.
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
from real_motion.strong_w2det import StrongW2DetConfig
from real_motion.v19_scene_memory import (
    StaticWorldMemory,
    build_dynamic_source_memory,
    protected_add_only,
    render_track_kta_add_only,
)
from tools.real_motion import eval_p0_f9_v18_se2 as base
from tools.real_motion import eval_p0_f9_v18_full_validation as full
from tools.real_motion.benchmark_p0_f9_v18_runtime import (
    _forecast_once,
    _prepare_record,
    _release_gpu_inputs,
    _stage_gpu_inputs,
)
from tools.real_motion.eval_p0_f9_v17_local_stwm import window_from_record
from tools.real_motion.train_p0_f9_v18_se2_clean import PROTOCOL as CLEAN_PROTOCOL


PROTOCOL = "p0_f9_v19_zero_training_memory_ablation_v1"
HORIZONS = (1.0, 2.0, 3.0)
REPORT = {1.0: 1, 2.0: 3, 3.0: 5}
SEM_CLASSES = tuple(range(17))
VARIANTS = (
    "base",
    "dormant_kta",
    "static_memory",
    "dormant_kta_static",
)


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


def _new_raw():
    H = len(HORIZONS)
    return {
        "occ_inter": np.zeros(H, dtype=np.int64),
        "occ_union": np.zeros(H, dtype=np.int64),
        "sem_inter": np.zeros((H, len(SEM_CLASSES)), dtype=np.int64),
        "sem_union": np.zeros((H, len(SEM_CLASSES)), dtype=np.int64),
        "mov_inter": np.zeros((H, len(DYNAMIC_CLASS_IDS)), dtype=np.int64),
        "mov_union": np.zeros((H, len(DYNAMIC_CLASS_IDS)), dtype=np.int64),
    }


def _update(raw, hi, pred, gt, moving, free_label):
    p, g = np.asarray(pred), np.asarray(gt)
    m = np.asarray(moving, dtype=bool)
    po, go = p != int(free_label), g != int(free_label)
    raw["occ_inter"][hi] += int((po & go).sum())
    raw["occ_union"][hi] += int((po | go).sum())
    for j, cid in enumerate(SEM_CLASSES):
        pp, gg = p == cid, g == cid
        raw["sem_inter"][hi, j] += int((pp & gg).sum())
        raw["sem_union"][hi, j] += int((pp | gg).sum())
    for j, cid in enumerate(DYNAMIC_CLASS_IDS):
        pp = (p == int(cid)) & m
        gg = (g == int(cid)) & m
        raw["mov_inter"][hi, j] += int((pp & gg).sum())
        raw["mov_union"][hi, j] += int((pp | gg).sum())


def _safe(i, u):
    i = np.asarray(i, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    out = np.full(i.shape, np.nan, dtype=np.float64)
    np.divide(i, u, out=out, where=u > 0)
    return 100.0 * out


def _finalize(raw):
    occ = _safe(raw["occ_inter"], raw["occ_union"])
    sem = _safe(raw["sem_inter"], raw["sem_union"])
    mov = _safe(raw["mov_inter"], raw["mov_union"])
    miou = np.nanmean(sem, axis=1)
    mmacro = np.nanmean(mov, axis=1)
    mmicro = _safe(
        raw["mov_inter"].sum(axis=1), raw["mov_union"].sum(axis=1)
    )
    per = {}
    for hi, h in enumerate(HORIZONS):
        per[str(h)] = {
            "IoU": float(occ[hi]),
            "mIoU": float(miou[hi]),
            "MovingMacro": float(mmacro[hi]),
            "MovingMicro": float(mmicro[hi]),
        }
    return {
        "IoU": float(np.nanmean(occ)),
        "mIoU": float(np.nanmean(miou)),
        "MovingMacro": float(np.nanmean(mmacro)),
        "MovingMicro": float(np.nanmean(mmicro)),
        "per_horizon": per,
    }


def _delta(a, b):
    return {
        k: float(a[k]) - float(b[k])
        for k in ("IoU", "mIoU", "MovingMacro", "MovingMicro")
    }


def main():
    p = argparse.ArgumentParser()
    add_config_args(p)
    p.add_argument("--val-cache", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataroot", required=True)
    p.add_argument("--info-pkl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-windows", type=int, default=0)
    p.add_argument("--max-missing-s", type=float, default=1.5)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    cfg = load_runtime_config(a.config, a.override)
    pcfg = make_prepare_config(cfg)
    _, records = base.load_cache(a.val_cache)
    if int(a.max_windows) > 0:
        records = records[: min(len(records), int(a.max_windows))]
    if not records:
        raise RuntimeError("empty validation cache")

    device = torch.device(
        a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    ck, model, _ = full._load_model(a.checkpoint, CLEAN_PROTOCOL, device)
    source = CachedSource(a.dataroot, info_pkl=a.info_pkl, verbose=False)
    strong_cfg = StrongW2DetConfig(free_label=int(pcfg.free_label))

    raw_by_variant = {v: _new_raw() for v in VARIANTS}
    dormant_tracks_total = 0
    windows_with_dormant = 0
    static_voxels_total = 0
    started = time.perf_counter()

    for wi, rec in enumerate(records, start=1):
        w = window_from_record(rec)
        raw = load_nuscenes_window_raw(
            source, w, pcfg, include_gt=True
        )
        state = _prepare_record(rec, source, pcfg, strong_cfg, device)
        _stage_gpu_inputs(state, device)
        try:
            pred_all = _forecast_once(
                model, state, pcfg, strong_cfg, device
            )
        finally:
            _release_gpu_inputs(state)

        tracks, comps = build_dynamic_source_memory(
            raw["history_occ"],
            raw["history_poses"],
            grid=pcfg.grid,
            strong_cfg=strong_cfg,
            frame_dt_s=float(pcfg.frame_dt_s),
            max_missing_s=float(a.max_missing_s),
        )
        n_current = len(comps[-1])
        dormant = tracks[n_current:]
        dormant_tracks_total += len(dormant)
        windows_with_dormant += int(bool(dormant))

        static_mem = StaticWorldMemory.from_history(
            raw["history_occ"],
            raw["history_observed"],
            raw["history_poses"],
            grid=pcfg.grid,
            free_label=int(pcfg.free_label),
        )
        static_voxels_total += len(static_mem)

        for hi, h in enumerate(HORIZONS):
            fi = REPORT[h]
            pred = np.asarray(pred_all[fi], dtype=np.uint8)
            future_pose = np.asarray(raw["future_poses"][fi], dtype=np.float64)

            p_dormant = pred.copy()
            for tr in dormant:
                p_dormant = render_track_kta_add_only(
                    p_dormant,
                    tr,
                    future_pose,
                    horizon_s=float(h),
                    frame_dt_s=float(pcfg.frame_dt_s),
                    grid=pcfg.grid,
                    free_label=int(pcfg.free_label),
                )

            static_prop = static_mem.render(
                future_pose,
                grid=pcfg.grid,
                free_label=int(pcfg.free_label),
            )
            p_static = protected_add_only(
                pred,
                static_prop,
                free_label=int(pcfg.free_label),
            )
            p_both = protected_add_only(
                p_dormant,
                static_prop,
                free_label=int(pcfg.free_label),
            )

            gt = np.asarray(raw["future_gt_occ"][fi], dtype=np.uint8)
            ftok = str(w.future_tokens[fi])
            moving, _, _ = gt_moving_support_for_horizon(
                source.nusc,
                str(w.t0_token),
                ftok,
                float(h),
                grid=pcfg.grid,
            )
            for variant, pp in (
                ("base", pred),
                ("dormant_kta", p_dormant),
                ("static_memory", p_static),
                ("dormant_kta_static", p_both),
            ):
                _update(
                    raw_by_variant[variant],
                    hi,
                    pp,
                    gt,
                    moving,
                    int(pcfg.free_label),
                )

        if wi == 1 or wi % 25 == 0 or wi == len(records):
            elapsed = max(time.perf_counter() - started, 1e-9)
            print(
                f"v19_memory_ablation {wi}/{len(records)} "
                f"rate={wi/elapsed:.3f} win/s",
                flush=True,
            )

    metrics = {v: _finalize(r) for v, r in raw_by_variant.items()}
    base_m = metrics["base"]
    result = {
        "protocol": PROTOCOL,
        "checkpoint": str(Path(a.checkpoint).resolve()),
        "checkpoint_epoch": int(ck.get("epoch", -1)),
        "num_windows": int(len(records)),
        "future_gt_used_for_prediction": False,
        "max_missing_s": float(a.max_missing_s),
        "dormant_tracks_total": int(dormant_tracks_total),
        "windows_with_dormant": int(windows_with_dormant),
        "mean_dormant_tracks_per_window": float(
            dormant_tracks_total / max(len(records), 1)
        ),
        "mean_static_world_voxels_per_window": float(
            static_voxels_total / max(len(records), 1)
        ),
        "metrics": metrics,
        "delta_vs_base": {
            v: _delta(metrics[v], base_m)
            for v in VARIANTS
            if v != "base"
        },
        "raw_counts": {
            v: {k: np.asarray(x).tolist() for k, x in r.items()}
            for v, r in raw_by_variant.items()
        },
    }
    op = Path(a.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\\n=== V19 ZERO-TRAINING MEMORY ABLATION ===")
    for v in VARIANTS:
        m = metrics[v]
        print(
            f"{v:22s} IoU={m['IoU']:.3f} mIoU={m['mIoU']:.3f} "
            f"MovMacro={m['MovingMacro']:.3f} MovMicro={m['MovingMicro']:.3f}"
        )
        if v != "base":
            print("  delta", json.dumps(result["delta_vs_base"][v]))
    print(f"saved {op}")


if __name__ == "__main__":
    main()
