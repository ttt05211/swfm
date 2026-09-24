#!/usr/bin/env python3
"""Formal frozen-base evaluation for Static New-FOV Novelty."""
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
from real_motion.v19_innovation import (
    base_explained_bev,
    build_future_aligned_history_and_static_memory,
)
from real_motion.v19_scene_memory import protected_add_only
from real_motion.v19_static_novelty import (
    StaticNewFOVHead,
    decode_static_new_fov,
    history_grid_footprint_bev,
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
from tools.real_motion.train_p0_f9_v18_se2_clean import (
    PROTOCOL as CLEAN_PROTOCOL,
)
from tools.real_motion.train_p0_f9_v19_static_new_fov import (
    HEAD_TYPE,
    PROTOCOL as NOVELTY_TRAIN_PROTOCOL,
)


PROTOCOL = "p0_f9_v19_static_new_fov_eval_v1"
HORIZONS = tuple(0.5 * (i + 1) for i in range(6))
SEM_CLASSES = tuple(range(17))
VARIANTS = (
    "v18",
    "v18_static",
    "v18_static_new_fov",
)


class CachedSource(NuScenesWindowSource):
    from functools import lru_cache

    @lru_cache(maxsize=768)
    def load_semantics(self, scene_name, token):
        return super().load_semantics(scene_name, token)

    @lru_cache(maxsize=768)
    def load_occ3d(
        self,
        scene_name,
        token,
        require_lidar_mask=True,
    ):
        return super().load_occ3d(
            scene_name,
            token,
            require_lidar_mask=require_lidar_mask,
        )

    @lru_cache(maxsize=4096)
    def pose(self, token):
        return super().pose(token)


def _autocast(device, enabled):
    if enabled and device.type == "cuda":
        return torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        )
    return nullcontext()


def _new_raw():
    H = len(HORIZONS)
    return {
        "occ_inter": np.zeros(H, dtype=np.int64),
        "occ_union": np.zeros(H, dtype=np.int64),
        "sem_inter": np.zeros(
            (H, len(SEM_CLASSES)),
            dtype=np.int64,
        ),
        "sem_union": np.zeros(
            (H, len(SEM_CLASSES)),
            dtype=np.int64,
        ),
        "mov_inter": np.zeros(
            (H, len(DYNAMIC_CLASS_IDS)),
            dtype=np.int64,
        ),
        "mov_union": np.zeros(
            (H, len(DYNAMIC_CLASS_IDS)),
            dtype=np.int64,
        ),
    }


def _update(raw, hi, pred, gt, moving, free_label):
    """Vectorized exact confusion accumulation.

    This replaces per-class full-volume boolean scans with bincount-based
    histograms.  It is algebraically identical to the previous implementation
    but substantially cheaper for full 3D validation.
    """
    p = np.asarray(pred, dtype=np.int64)
    g = np.asarray(gt, dtype=np.int64)
    m = np.asarray(moving, dtype=bool)
    if p.shape != g.shape or m.shape != g.shape:
        raise ValueError("pred/gt/moving shape mismatch")

    free = int(free_label)
    po = p != free
    go = g != free
    raw["occ_inter"][hi] += int(np.count_nonzero(po & go))
    raw["occ_union"][hi] += int(np.count_nonzero(po | go))

    # Semantic IoU for classes 0..16.  Labels outside that range (free=17)
    # simply do not contribute to per-class pred/GT histograms.
    flat_p = p.ravel()
    flat_g = g.ravel()
    sem_n = len(SEM_CLASSES)
    p_valid = (flat_p >= 0) & (flat_p < sem_n)
    g_valid = (flat_g >= 0) & (flat_g < sem_n)
    p_count = np.bincount(
        flat_p[p_valid],
        minlength=sem_n,
    )[:sem_n]
    g_count = np.bincount(
        flat_g[g_valid],
        minlength=sem_n,
    )[:sem_n]
    same = p_valid & g_valid & (flat_p == flat_g)
    inter = np.bincount(
        flat_p[same],
        minlength=sem_n,
    )[:sem_n]
    raw["sem_inter"][hi] += inter.astype(np.int64, copy=False)
    raw["sem_union"][hi] += (
        p_count + g_count - inter
    ).astype(np.int64, copy=False)

    # Moving metric is the same class-wise IoU restricted to GT-moving support.
    # Build one small histogram on that support, then gather dynamic class IDs.
    mp = flat_p[m.ravel()]
    mg = flat_g[m.ravel()]
    if mp.size:
        label_n = max(
            free + 1,
            max(int(x) for x in DYNAMIC_CLASS_IDS) + 1,
        )
        mp_valid = (mp >= 0) & (mp < label_n)
        mg_valid = (mg >= 0) & (mg < label_n)
        pc = np.bincount(
            mp[mp_valid],
            minlength=label_n,
        )
        gc = np.bincount(
            mg[mg_valid],
            minlength=label_n,
        )
        eq = mp_valid & mg_valid & (mp == mg)
        ic = np.bincount(
            mp[eq],
            minlength=label_n,
        )
        for j, cid in enumerate(DYNAMIC_CLASS_IDS):
            cid = int(cid)
            raw["mov_inter"][hi, j] += int(ic[cid])
            raw["mov_union"][hi, j] += int(
                pc[cid] + gc[cid] - ic[cid]
            )


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
        raw["mov_inter"].sum(axis=1),
        raw["mov_union"].sum(axis=1),
    )
    per = {}
    for hi, h in enumerate(HORIZONS):
        per[str(h)] = {
            "IoU": float(occ[hi]),
            "mIoU": float(miou[hi]),
            "MovingMacro": float(mmacro[hi]),
            "MovingMicro": float(mmicro[hi]),
        }
    main_idx = [1, 3, 5]
    return {
        "IoU": float(np.nanmean(occ)),
        "mIoU": float(np.nanmean(miou)),
        "MovingMacro": float(np.nanmean(mmacro)),
        "MovingMicro": float(np.nanmean(mmicro)),
        "main_1_2_3s": {
            "IoU": float(np.nanmean(occ[main_idx])),
            "mIoU": float(np.nanmean(miou[main_idx])),
            "MovingMacro": float(
                np.nanmean(mmacro[main_idx])
            ),
            "MovingMicro": float(
                np.nanmean(mmicro[main_idx])
            ),
        },
        "per_horizon": per,
    }


def _delta(a, b):
    keys = ("IoU", "mIoU", "MovingMacro", "MovingMicro")
    return {
        **{
            k: float(a[k]) - float(b[k])
            for k in keys
        },
        "main_1_2_3s": {
            k: float(a["main_1_2_3s"][k])
            - float(b["main_1_2_3s"][k])
            for k in keys
        },
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
    p.add_argument("--alignment-workers", type=int, default=6)
    p.add_argument(
        "--occupancy-threshold",
        type=float,
        default=-1.0,
        help="<0 uses checkpoint-selected validation threshold",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-amp", action="store_true")
    a = p.parse_args()

    if int(a.alignment_workers) <= 0:
        raise ValueError("alignment-workers must be positive")

    pcfg = make_prepare_config(
        load_runtime_config(a.config, a.override)
    )
    _, records = base.load_cache(a.val_cache)
    if int(a.max_windows) > 0:
        records = records[: min(len(records), int(a.max_windows))]
    if not records:
        raise RuntimeError("empty validation cache")

    device = torch.device(
        a.device
        if a.device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    amp = (
        device.type == "cuda"
        and not bool(a.no_amp)
    )
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
    if nov_ck.get("protocol") != NOVELTY_TRAIN_PROTOCOL:
        raise RuntimeError(
            f"unexpected Novelty checkpoint protocol: "
            f"{nov_ck.get('protocol')}"
        )
    if nov_ck.get("head_type") != HEAD_TYPE:
        raise RuntimeError(
            f"unexpected Novelty head type: "
            f"{nov_ck.get('head_type')}"
        )
    novelty = StaticNewFOVHead(
        **dict(nov_ck["architecture"])
    ).to(device)
    novelty.load_state_dict(
        nov_ck["model_state_dict"],
        strict=True,
    )
    novelty.eval()

    threshold = (
        float(a.occupancy_threshold)
        if float(a.occupancy_threshold) >= 0
        else float(nov_ck["selected_threshold"])
    )

    source = CachedSource(
        a.dataroot,
        info_pkl=a.info_pkl,
        verbose=False,
    )
    strong_cfg = StrongW2DetConfig(
        free_label=int(pcfg.free_label)
    )
    raw_by_variant = {
        v: _new_raw() for v in VARIANTS
    }
    proposed = 0
    added = 0
    windows_with_additions = 0
    new_fov_bev_columns = 0
    started = time.perf_counter()

    for wi, rec in enumerate(records, start=1):
        w = window_from_record(rec)
        raw = load_nuscenes_window_raw(
            source,
            w,
            pcfg,
            include_gt=True,
        )
        state = _prepare_record(
            rec,
            source,
            pcfg,
            strong_cfg,
            device,
        )
        _stage_gpu_inputs(state, device)
        try:
            pred_all = _forecast_once(
                base_model,
                state,
                pcfg,
                strong_cfg,
                device,
            )
        finally:
            _release_gpu_inputs(state)

        sem, geo, _, static_all = (
            build_future_aligned_history_and_static_memory(
                raw["history_occ"],
                raw["history_observed"],
                raw["history_poses"],
                raw["future_poses"],
                grid=pcfg.grid,
                free_label=int(pcfg.free_label),
                dynamic_class_ids=tuple(
                    int(x) for x in DYNAMIC_CLASS_IDS
                ),
                workers=int(a.alignment_workers),
            )
        )
        explained = []
        new_fov = []
        base_free = []
        for fi in range(len(raw["future_poses"])):
            pred = np.asarray(
                pred_all[fi],
                dtype=np.uint8,
            )
            exp = protected_add_only(
                pred,
                static_all[fi],
                free_label=int(pcfg.free_label),
            )
            explained.append(exp)
            base_free.append(
                exp == int(pcfg.free_label)
            )
            footprint = history_grid_footprint_bev(
                np.asarray(
                    raw["history_poses"],
                    dtype=np.float64,
                ),
                np.asarray(
                    raw["future_poses"][fi],
                    dtype=np.float64,
                ),
                pcfg.grid,
            )
            nf = ~footprint
            new_fov.append(nf)
            new_fov_bev_columns += int(nf.sum())

        explained = np.stack(
            explained,
            axis=0,
        ).astype(np.uint8)
        base_free = np.stack(base_free, axis=0)
        new_fov = np.stack(new_fov, axis=0)

        sem_t = torch.from_numpy(
            sem[None]
        ).to(device)
        geo_t = torch.from_numpy(
            geo[None]
        ).to(device)
        base_t = torch.from_numpy(
            base_explained_bev(
                explained,
                free_label=int(pcfg.free_label),
            )[None]
        ).to(device)
        nf_t = torch.from_numpy(
            new_fov[None]
        ).to(device)
        free_t = torch.from_numpy(
            base_free.transpose(0, 3, 1, 2)[None]
        ).to(device)

        with torch.inference_mode(), _autocast(device, amp):
            out = novelty(
                sem_t,
                geo_t,
                base_t,
                nf_t,
            )
            proposal_zxy = decode_static_new_fov(
                out,
                new_fov_mask=nf_t,
                base_free_mask=free_t,
                free_label=int(pcfg.free_label),
                occupancy_threshold=threshold,
            )

        proposal = (
            proposal_zxy[0]
            .permute(0, 2, 3, 1)
            .cpu()
            .numpy()
            .astype(np.uint8)
        )
        proposed += int(
            (proposal != int(pcfg.free_label)).sum()
        )

        final = []
        window_added = 0
        for fi in range(len(explained)):
            f = protected_add_only(
                explained[fi],
                proposal[fi],
                free_label=int(pcfg.free_label),
            )
            n = int(
                (
                    (f != int(pcfg.free_label))
                    & (
                        explained[fi]
                        == int(pcfg.free_label)
                    )
                ).sum()
            )
            added += n
            window_added += n
            final.append(f)
        windows_with_additions += int(window_added > 0)

        for hi, h in enumerate(HORIZONS):
            gt = np.asarray(
                raw["future_gt_occ"][hi],
                dtype=np.uint8,
            )
            moving, _, _ = gt_moving_support_for_horizon(
                source.nusc,
                str(w.t0_token),
                str(w.future_tokens[hi]),
                float(h),
                grid=pcfg.grid,
            )
            for name, pred in (
                ("v18", np.asarray(pred_all[hi], dtype=np.uint8)),
                ("v18_static", explained[hi]),
                ("v18_static_new_fov", final[hi]),
            ):
                _update(
                    raw_by_variant[name],
                    hi,
                    pred,
                    gt,
                    moving,
                    int(pcfg.free_label),
                )

        if (
            wi == 1
            or wi % 25 == 0
            or wi == len(records)
        ):
            elapsed = max(
                time.perf_counter() - started,
                1e-9,
            )
            print(
                f"v19_static_new_fov_eval "
                f"{wi}/{len(records)} "
                f"rate={wi/elapsed:.3f} win/s",
                flush=True,
            )

    metrics = {
        v: _finalize(raw_by_variant[v])
        for v in VARIANTS
    }
    elapsed = max(
        time.perf_counter() - started,
        1e-9,
    )
    result = {
        "protocol": PROTOCOL,
        "num_windows": int(len(records)),
        "base_checkpoint": str(
            Path(a.base_checkpoint).resolve()
        ),
        "base_checkpoint_epoch": int(
            base_ck.get("epoch", -1)
        ),
        "novelty_checkpoint": str(
            Path(a.novelty_checkpoint).resolve()
        ),
        "novelty_epoch": int(nov_ck.get("epoch", -1)),
        "occupancy_threshold": float(threshold),
        "future_gt_used_for_prediction": False,
        "metrics": metrics,
        "delta_new_fov_vs_static": _delta(
            metrics["v18_static_new_fov"],
            metrics["v18_static"],
        ),
        "proposal_audit": {
            "new_fov_bev_columns": int(
                new_fov_bev_columns
            ),
            "proposed_voxels": int(proposed),
            "added_voxels_after_protection": int(added),
            "windows_with_additions": int(
                windows_with_additions
            ),
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
            "windows_per_s": float(
                len(records) / elapsed
            ),
        },
    }
    op = Path(a.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print("\n=== V19 STATIC NEW-FOV FROZEN-BASE EVAL ===")
    for v in VARIANTS:
        m = metrics[v]
        print(
            f"{v:24s} "
            f"IoU={m['IoU']:.3f} "
            f"mIoU={m['mIoU']:.3f} "
            f"MovMacro={m['MovingMacro']:.3f} "
            f"MovMicro={m['MovingMicro']:.3f} "
            f"main123_mIoU={m['main_1_2_3s']['mIoU']:.3f}"
        )
    print(
        "new_fov_vs_static",
        json.dumps(result["delta_new_fov_vs_static"]),
    )
    print(
        "proposal_audit",
        json.dumps(result["proposal_audit"]),
    )
    print(f"saved {op}")


if __name__ == "__main__":
    main()
