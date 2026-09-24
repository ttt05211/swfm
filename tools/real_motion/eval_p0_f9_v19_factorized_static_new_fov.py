#!/usr/bin/env python3
"""Formal frozen-base evaluation for factorized Static New-FOV completion."""
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
    history_grid_footprint_bev,
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
    _prepare_record,
    _release_gpu_inputs,
    _stage_gpu_inputs,
)
from tools.real_motion.eval_p0_f9_v17_local_stwm import window_from_record
from tools.real_motion.eval_p0_f9_v19_static_new_fov import (
    HORIZONS,
    _delta,
    _finalize,
    _new_raw,
    _update,
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
    raw_by_variant = {v: _new_raw() for v in VARIANTS}
    proposed = 0
    added = 0
    windows_with_additions = 0
    active_bev_columns = 0
    new_fov_bev_columns = 0
    started = time.perf_counter()

    for wi, rec in enumerate(records, start=1):
        w = window_from_record(rec)
        raw = load_nuscenes_window_raw(source, w, pcfg, include_gt=True)
        state = _prepare_record(rec, source, pcfg, strong_cfg, device)
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
        base_free = []
        new_fov = []
        anchor_sem = []
        anchor_profile = []
        anchor_dist = []
        history_poses = np.asarray(
            raw["history_poses"],
            dtype=np.float64,
        )

        for fi in range(len(raw["future_poses"])):
            pred = np.asarray(pred_all[fi], dtype=np.uint8)
            static_render = np.asarray(static_all[fi], dtype=np.uint8)
            exp = protected_add_only(
                pred,
                static_render,
                free_label=int(pcfg.free_label),
            )
            explained.append(exp)
            free = exp == int(pcfg.free_label)
            base_free.append(free)

            footprint = history_grid_footprint_bev(
                history_poses,
                np.asarray(raw["future_poses"][fi], dtype=np.float64),
                pcfg.grid,
            )
            nf = ~footprint
            new_fov.append(nf)
            new_fov_bev_columns += int(nf.sum())

            dist_cells, ax, ay, valid = nearest_static_anchor_map(
                static_render,
                footprint,
                free_label=int(pcfg.free_label),
            )
            static_occ = static_render != int(pcfg.free_label)
            sem_source = majority_semantic_per_column(
                static_occ,
                static_render,
                num_classes=17,
                ignore_label=int(pcfg.free_label),
            )
            aq = np.full(
                nf.shape,
                int(pcfg.free_label),
                dtype=np.uint8,
            )
            profile = np.zeros(static_occ.shape, dtype=bool)
            if bool(valid.any()):
                aq[valid] = sem_source[ax[valid], ay[valid]]
                copied = static_occ[ax, ay]
                profile[valid] = copied[valid]
            anchor_sem.append(aq)
            anchor_profile.append(profile)
            anchor_dist.append(
                np.asarray(dist_cells, dtype=np.float32)
                * float(pcfg.grid.voxel_size[0])
            )

        explained = np.stack(explained, axis=0).astype(np.uint8)
        base_free = np.stack(base_free, axis=0)
        new_fov = np.stack(new_fov, axis=0)
        anchor_sem = np.stack(anchor_sem, axis=0)
        anchor_profile = np.stack(anchor_profile, axis=0)
        anchor_dist = np.stack(anchor_dist, axis=0)

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

        proposal = (
            proposal_zxy[0]
            .permute(0, 2, 3, 1)
            .cpu()
            .numpy()
            .astype(np.uint8)
        )
        proposed += int((proposal != int(pcfg.free_label)).sum())

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
                    & (explained[fi] == int(pcfg.free_label))
                ).sum()
            )
            added += n
            window_added += n
            final.append(f)
        windows_with_additions += int(window_added > 0)

        for hi, h in enumerate(HORIZONS):
            gt = np.asarray(raw["future_gt_occ"][hi], dtype=np.uint8)
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
                ("v18_static_factorized_new_fov", final[hi]),
            ):
                _update(
                    raw_by_variant[name],
                    hi,
                    pred,
                    gt,
                    moving,
                    int(pcfg.free_label),
                )

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
    print(f"saved {op}")


if __name__ == "__main__":
    main()
