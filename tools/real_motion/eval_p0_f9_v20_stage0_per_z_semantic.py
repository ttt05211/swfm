#!/usr/bin/env python3
"""Formal Stage-0 evaluation: frozen V19 geometry, per-Z semantics only."""
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

from real_motion.metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS, NUSCENES_LABELS
from real_motion.prepared import load_nuscenes_window_raw
from real_motion.runtime_config import add_config_args, load_runtime_config, make_prepare_config
from real_motion.strong_w2det import StrongW2DetConfig
from real_motion.v19_innovation import base_explained_bev
from real_motion.v19_static_novelty import history_grid_footprint_bev_sequence
from real_motion.v19_static_novelty_factorized import (
    FactorizedStaticNewFOVHead,
    decode_factorized_static_new_fov,
)
from real_motion.v20_evaluation import population_fingerprint
from real_motion.v20_stage0_voxel_semantic import (
    FrozenFactorizedFeatureAdapter,
    PerZSemanticHead,
    decode_per_z_semantic,
    frozen_factorized_support,
)
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
    _HistoryAlignmentLRU,
    _HistoryFutureAlignmentLRU,
    _build_anchor_context_sequence,
    _build_history_static_from_pair_cache,
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
from tools.real_motion.train_p0_f9_v19_factorized_static_new_fov import (
    HEAD_TYPE as V19_HEAD_TYPE,
    PROTOCOL as V19_TRAIN_PROTOCOL,
)
from tools.real_motion.train_p0_f9_v20_stage0_per_z_semantic import (
    PROTOCOL as STAGE0_TRAIN_PROTOCOL,
)

PROTOCOL = "p0_f9_v20_stage0_per_z_semantic_eval_v2"
VARIANTS = (
    "v18",
    "v18_static",
    "v18_static_factorized_column",
    "v18_static_v20_per_z",
)


def _autocast(device, enabled):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _per_class(raw):
    inter = np.asarray(raw["sem_inter"], dtype=np.float64)
    union = np.asarray(raw["sem_union"], dtype=np.float64)
    main = [1, 3, 5]
    rows = {}
    for cid in range(17):
        all_i = float(inter[:, cid].sum())
        all_u = float(union[:, cid].sum())
        main_i = float(inter[main, cid].sum())
        main_u = float(union[main, cid].sum())
        rows[str(cid)] = {
            "name": str(NUSCENES_LABELS[cid]),
            "all_six_iou": float(100.0 * all_i / all_u) if all_u else float("nan"),
            "main_1_2_3s_iou": float(100.0 * main_i / main_u) if main_u else float("nan"),
            "union_all_six": int(all_u),
        }
    return rows


def _per_class_delta(a, b):
    out = {}
    for cid in range(17):
        key = str(cid)
        out[key] = {
            "name": a[key]["name"],
            "all_six_iou": float(a[key]["all_six_iou"] - b[key]["all_six_iou"]),
            "main_1_2_3s_iou": float(a[key]["main_1_2_3s_iou"] - b[key]["main_1_2_3s_iou"]),
        }
    return out


def main():
    p = argparse.ArgumentParser()
    add_config_args(p)
    p.add_argument("--val-cache", required=True)
    p.add_argument("--base-checkpoint", required=True)
    p.add_argument("--factorized-checkpoint", required=True)
    p.add_argument("--stage0-checkpoint", required=True)
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

    fck = torch.load(a.factorized_checkpoint, map_location="cpu", weights_only=False)
    if fck.get("protocol") != V19_TRAIN_PROTOCOL or fck.get("head_type") != V19_HEAD_TYPE:
        raise RuntimeError("factorized checkpoint protocol mismatch")
    factorized = FactorizedStaticNewFOVHead(**dict(fck["architecture"])).to(device)
    factorized.load_state_dict(fck["model_state_dict"], strict=True)
    adapter = FrozenFactorizedFeatureAdapter(factorized).to(device).eval()
    pth = float(fck["selected_presence_threshold"])
    vth = float(fck["selected_vertical_threshold"])

    sck = torch.load(a.stage0_checkpoint, map_location="cpu", weights_only=False)
    if sck.get("protocol") != STAGE0_TRAIN_PROTOCOL:
        raise RuntimeError("Stage-0 checkpoint protocol mismatch")
    expected_factorized = Path(str(sck["factorized_checkpoint"])).name
    if Path(a.factorized_checkpoint).name != expected_factorized:
        raise RuntimeError(
            "Stage-0 checkpoint was trained against a different Factorized checkpoint"
        )
    shead = PerZSemanticHead(**dict(sck["head_architecture"])).to(device)
    shead.load_state_dict(sck["head_state_dict"], strict=True)
    shead.eval()

    source = CachedSource(a.dataroot, info_pkl=a.info_pkl, verbose=False)
    strong_cfg = StrongW2DetConfig(free_label=int(pcfg.free_label))
    component_cache = _ComponentLRU(maxsize=1024)
    history_alignment_cache = _HistoryAlignmentLRU(maxsize=64)
    history_future_pair_cache = _HistoryFutureAlignmentLRU(maxsize=96)
    raw_by_variant = {v: _new_raw() for v in VARIANTS}
    proposed_original = proposed_stage0 = 0
    stage0_semantic_changes = 0
    started = time.perf_counter()

    for wi, rec in enumerate(records, start=1):
        w = window_from_record(rec)
        raw = load_nuscenes_window_raw(source, w, pcfg, include_gt=True)
        state = _prepare_record_from_raw(
            rec, raw, source, pcfg, strong_cfg, device, component_cache
        )
        _stage_gpu_inputs(state, device)
        try:
            pred_all = _forecast_once(base_model, state, pcfg, strong_cfg, device)
        finally:
            _release_gpu_inputs(state)

        prepared_history = [
            history_alignment_cache.get_or_build(
                str(w.scene_name), str(tok),
                raw["history_occ"][ti], raw["history_observed"][ti], raw["history_poses"][ti],
                grid=pcfg.grid, dynamic_class_ids=DYNAMIC_CLASS_IDS,
            )
            for ti, tok in enumerate(w.history_tokens)
        ]
        sem, geo, static_all, _ = _build_history_static_from_pair_cache(
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
        pred_stack = np.asarray(pred_all, dtype=np.uint8)
        static_all = np.asarray(static_all, dtype=np.uint8)
        free_label = int(pcfg.free_label)
        explained = pred_stack.copy()
        static_add = (static_all != free_label) & (explained == free_label)
        np.copyto(explained, static_all, where=static_add)
        base_free = explained == free_label

        history_poses = np.asarray(raw["history_poses"], dtype=np.float64)
        future_poses = np.asarray(raw["future_poses"], dtype=np.float64)
        footprint_all = history_grid_footprint_bev_sequence(
            history_poses, future_poses, pcfg.grid, workers=int(a.alignment_workers)
        )
        new_fov = ~footprint_all
        anchor_sem, anchor_profile, anchor_dist = _build_anchor_context_sequence(
            static_all, footprint_all,
            free_label=free_label,
            voxel_size_xy_m=float(pcfg.grid.voxel_size[0]),
            workers=int(a.alignment_workers),
        )

        sem_t = torch.from_numpy(sem[None]).to(device)
        geo_t = torch.from_numpy(geo[None]).to(device)
        base_t = torch.from_numpy(
            base_explained_bev(explained, free_label=free_label)[None]
        ).to(device)
        nf_t = torch.from_numpy(new_fov[None]).to(device)
        asem_t = torch.from_numpy(anchor_sem[None]).to(device)
        aprof_t = torch.from_numpy(anchor_profile.transpose(0, 3, 1, 2)[None]).to(device)
        adist_t = torch.from_numpy(anchor_dist[None]).to(device)
        free_t = torch.from_numpy(base_free.transpose(0, 3, 1, 2)[None]).to(device)

        with torch.inference_mode(), _autocast(device, amp):
            fout, feature = adapter(
                sem_t, geo_t, base_t, nf_t, asem_t, aprof_t, adist_t
            )
            original_zxy = decode_factorized_static_new_fov(
                fout,
                new_fov_mask=nf_t,
                base_free=free_t,
                free_label=free_label,
                presence_threshold=pth,
                vertical_threshold=vth,
            )
            support = frozen_factorized_support(
                fout,
                new_fov_mask=nf_t,
                base_free=free_t,
                presence_threshold=pth,
                vertical_threshold=vth,
            )
            if not torch.equal(support, original_zxy.ne(free_label)):
                raise RuntimeError(
                    "column decoder geometry disagrees with frozen V19 support"
                )
            B, Fh, C, H, W = feature.shape
            feat = feature.reshape(B * Fh, C, H, W)
            support_bf = support.reshape(B * Fh, support.shape[2], H, W)
            slogits = shead(feat, support_bf)
            stage0_bf = decode_per_z_semantic(
                slogits, support_bf, free_label=free_label
            )
            stage0_zxy = stage0_bf.reshape(B, Fh, support.shape[2], H, W)

        original = original_zxy[0].permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        stage0 = stage0_zxy[0].permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        if not np.array_equal(original != free_label, stage0 != free_label):
            raise RuntimeError("Stage-0 changed frozen Factorized occupancy geometry")
        active = stage0 != free_label
        stage0_semantic_changes += int((active & (original != stage0)).sum())
        proposed_original += int((original != free_label).sum())
        proposed_stage0 += int(active.sum())

        final_original = explained.copy()
        np.copyto(final_original, original, where=original != free_label)
        final_stage0 = explained.copy()
        np.copyto(final_stage0, stage0, where=active)

        moving_rows = _moving_support_sequence(
            source, w, grid=pcfg.grid, workers=int(a.alignment_workers)
        )
        for hi, _ in enumerate(HORIZONS):
            gt = np.asarray(raw["future_gt_occ"][hi], dtype=np.uint8)
            _update_many(
                raw_by_variant,
                hi,
                {
                    "v18": pred_stack[hi],
                    "v18_static": explained[hi],
                    "v18_static_factorized_column": final_original[hi],
                    "v18_static_v20_per_z": final_stage0[hi],
                },
                gt,
                moving_rows[hi],
                free_label,
            )
        if wi == 1 or wi % 25 == 0 or wi == len(records):
            elapsed = max(time.perf_counter() - started, 1e-9)
            print(f"v20_stage0_eval {wi}/{len(records)} rate={wi/elapsed:.3f} win/s", flush=True)

    metrics = {v: _finalize(raw_by_variant[v]) for v in VARIANTS}
    pc_orig = _per_class(raw_by_variant["v18_static_factorized_column"])
    pc_v20 = _per_class(raw_by_variant["v18_static_v20_per_z"])
    result = {
        "protocol": PROTOCOL,
        "num_windows": len(records),
        "population_fingerprint_sha256": population_fingerprint(records),
        "base_checkpoint": str(Path(a.base_checkpoint).resolve()),
        "base_checkpoint_epoch": int(base_ck.get("epoch", -1)),
        "factorized_checkpoint": str(Path(a.factorized_checkpoint).resolve()),
        "stage0_checkpoint": str(Path(a.stage0_checkpoint).resolve()),
        "presence_threshold": pth,
        "vertical_threshold": vth,
        "future_gt_used_for_prediction": False,
        "stage0_geometry_source": "frozen_v19_predicted_presence_and_vertical_support",
        "gt_vertical_target_used_as_model_input": False,
        "geometry_identity_check": {
            "original_proposed_voxels": proposed_original,
            "stage0_proposed_voxels": proposed_stage0,
            "identical": bool(proposed_original == proposed_stage0),
        },
        "semantic_changes_on_frozen_support": int(stage0_semantic_changes),
        "metrics": metrics,
        "delta_per_z_vs_column": _delta(
            metrics["v18_static_v20_per_z"],
            metrics["v18_static_factorized_column"],
        ),
        "per_class_iou": {
            "factorized_column": pc_orig,
            "v20_per_z": pc_v20,
            "delta": _per_class_delta(pc_v20, pc_orig),
        },
        "raw_counts": {
            v: {k: np.asarray(x).tolist() for k, x in raw_by_variant[v].items()}
            for v in VARIANTS
        },
        "timing": {
            "elapsed_s": float(max(time.perf_counter() - started, 1e-9)),
            "windows_per_s": float(
                len(records) / max(time.perf_counter() - started, 1e-9)
            ),
        },
    }
    op = Path(a.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "metrics": metrics,
        "delta_per_z_vs_column": result["delta_per_z_vs_column"],
        "geometry_identity_check": result["geometry_identity_check"],
    }, indent=2))
    print(f"saved {op}")


if __name__ == "__main__":
    main()
