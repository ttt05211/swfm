#!/usr/bin/env python3
"""Build sharded training cache for the frozen-base V19 Innovation Head.

The cache is generated from the exact Clean-E14/V18 forecast path plus
six-frame deterministic Static Memory. Future GT/instance identity is used
only to construct the training responsibility masks; it is never part of the
inference input.
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

from real_motion.local_st_world_model_v18_se2 import YAW_ENABLED_CLASS_IDS
from real_motion.metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS
from real_motion.motion_transport import (
    dynamic_annotations,
    match_sources_to_annotations,
)
from real_motion.prepared import load_nuscenes_window_raw
from real_motion.rigid_transport import rasterize_rigid_component
from real_motion.runtime_config import (
    add_config_args,
    load_runtime_config,
    make_prepare_config,
)
from real_motion.runtime_fastpath import extract_instances_cropped_exact
from real_motion.strong_w2det import StrongW2DetConfig
from real_motion.v19_innovation import (
    base_explained_bev,
    build_future_aligned_history_bev_with_coverage,
)
from real_motion.v19_innovation_targets import (
    DECOMPOSITION_CATEGORIES,
    INNOVATION_POSITIVE_CATEGORIES,
    match_future_components_many_to_one,
)
from real_motion.v19_innovation_training import (
    INNOVATION_CACHE_PROTOCOL,
    build_innovation_bev_supervision,
    pack_vertical_occupancy,
    quantize_geometry,
)
from real_motion.v19_scene_memory import (
    protected_add_only,
    render_static_history_mosaic,
)
from tools.real_motion import eval_p0_f9_v18_se2 as base
from tools.real_motion import eval_p0_f9_v18_full_validation as full
from tools.real_motion.benchmark_p0_f9_v18_runtime import (
    _forecast_once,
    _prepare_record,
    _release_gpu_inputs,
    _stage_gpu_inputs,
)
from tools.real_motion.diagnose_p0_f9_v19_innovation_decomposition import (
    CachedSource,
    _ann_map,
    _grid_spec,
    _same_class_history_evidence,
    _source_target,
)
from tools.real_motion.eval_p0_f9_v17_local_stwm import window_from_record
from tools.real_motion.train_p0_f9_v18_se2_clean import (
    PROTOCOL as CLEAN_PROTOCOL,
)

PROTOCOL = INNOVATION_CACHE_PROTOCOL
_DYNAMIC = tuple(int(x) for x in DYNAMIC_CLASS_IDS)
POSITIVE_MODES = {
    "core": tuple(INNOVATION_POSITIVE_CATEGORIES),
    "dynamic": (
        "future_birth_dynamic",
        "source_shape_innovation",
    ),
}


def _category_masks_for_future(
    *,
    gt,
    pred_v18,
    future_pose,
    future_token,
    state,
    source_tokens,
    represented,
    history_occ,
    history_obs,
    history_poses,
    history_tokens,
    ann_hist,
    ann0,
    t0_tokens,
    source,
    pcfg,
    future_component_cfg,
    metric_grid,
    match_max_distance_m,
    history_coverage,
):
    gt = np.asarray(gt, dtype=np.uint8)
    pred = np.asarray(pred_v18, dtype=np.uint8)
    gt_occ = gt != int(pcfg.free_label)
    addable = gt_occ & (pred == int(pcfg.free_label))
    masks = {
        cat: np.zeros(gt.shape, dtype=bool)
        for cat in DECOMPOSITION_CATEGORIES
    }
    gt_dynamic = gt_occ & np.isin(
        gt, np.asarray(_DYNAMIC, dtype=gt.dtype)
    )
    gt_static = gt_occ & ~gt_dynamic

    annh = _ann_map(source.nusc, str(future_token))
    represented_transport_by_token = {}
    for src_i, comp in enumerate(state["current"]):
        tok = source_tokens[src_i]
        if tok is None:
            continue
        tok = str(tok)
        a0 = ann0.get(tok)
        ah = annh.get(tok)
        if a0 is None or ah is None:
            continue
        target, dyaw = _source_target(
            np.asarray(comp["centroid_world"], dtype=np.float64),
            a0,
            ah,
            int(comp["class_id"]) in set(YAW_ENABLED_CLASS_IDS),
        )
        rc = rasterize_rigid_component(
            comp["voxel_indices"],
            int(comp["class_id"]),
            state["current_pose"],
            np.asarray(future_pose, dtype=np.float64),
            source_center_world=np.asarray(
                comp["centroid_world"], dtype=np.float64
            ),
            target_center_world=target,
            yaw_delta_rad=float(dyaw),
            grid=pcfg.grid,
        )
        m = represented_transport_by_token.setdefault(
            tok, np.zeros(gt.shape, dtype=bool)
        )
        idx = np.asarray(rc.voxel_indices, dtype=np.int64)
        if len(idx):
            m[idx[:, 0], idx[:, 1], idx[:, 2]] = True

    future_components = extract_instances_cropped_exact(
        gt,
        np.asarray(future_pose, dtype=np.float64),
        grid=pcfg.grid,
        cfg=future_component_cfg,
    )
    links = match_future_components_many_to_one(
        future_components,
        annh,
        max_distance_m=float(match_max_distance_m),
    )
    dynamic_assigned = np.zeros(gt.shape, dtype=bool)

    for comp, (tok, _nearest_d) in zip(future_components, links):
        idx = np.asarray(comp["voxel_indices"], dtype=np.int64)
        if len(idx) == 0:
            continue
        cm = np.zeros(gt.shape, dtype=bool)
        cm[idx[:, 0], idx[:, 1], idx[:, 2]] = True
        ca = cm & addable & gt_dynamic
        if not bool(ca.any()):
            continue

        if tok is None:
            masks["dynamic_other_ambiguous"] |= ca
            dynamic_assigned |= ca
            continue

        tok = str(tok)
        ah = annh.get(tok)
        if (
            ah is None
            or int(ah["class_id"]) != int(comp["class_id"])
        ):
            raise RuntimeError("future component ancestry mismatch")

        if tok in represented:
            tr = represented_transport_by_token.get(tok)
            if tr is None:
                raise RuntimeError(
                    "represented future token lacks GT transport support"
                )
            masks["current_source_transportable_miss"] |= ca & tr
            masks["source_shape_innovation"] |= ca & ~tr
            dynamic_assigned |= ca
            continue

        seen_pre_t0 = _same_class_history_evidence(
            tok,
            int(ah["class_id"]),
            tuple(history_tokens[:-1]),
            history_occ[:-1],
            history_poses[:-1],
            ann_hist[:-1],
            metric_grid,
        )
        if seen_pre_t0:
            masks["history_source_recoverable"] |= ca
        elif tok in t0_tokens:
            masks["t0_unrepresented_dynamic"] |= ca
        else:
            masks["future_birth_dynamic"] |= ca
        dynamic_assigned |= ca

    residual_dynamic = addable & gt_dynamic & ~dynamic_assigned
    if bool(residual_dynamic.any()):
        masks["dynamic_other_ambiguous"] |= residual_dynamic

    static_render = render_static_history_mosaic(
        history_occ,
        history_obs,
        history_poses,
        np.asarray(future_pose, dtype=np.float64),
        grid=pcfg.grid,
        free_label=int(pcfg.free_label),
    )
    hist_static = addable & gt_static & (static_render == gt)
    masks["history_static_recoverable"] = hist_static

    hist_coverage = np.asarray(history_coverage, dtype=bool)
    if hist_coverage.shape != gt.shape:
        raise ValueError("history coverage shape mismatch")

    masks["history_static_seen_mismatch"] = (
        addable & gt_static & hist_coverage & ~hist_static
    )
    masks["never_seen_static"] = (
        addable & gt_static & ~hist_coverage & ~hist_static
    )
    masks["static_other_ambiguous"] = (
        addable
        & gt_static
        & ~masks["history_static_recoverable"]
        & ~masks["history_static_seen_mismatch"]
        & ~masks["never_seen_static"]
    )

    assigned = np.zeros(gt.shape, dtype=bool)
    for cat in DECOMPOSITION_CATEGORIES:
        if bool((assigned & masks[cat]).any()):
            raise RuntimeError(
                f"innovation category overlap: {cat}"
            )
        assigned |= masks[cat]

    if int(assigned.sum()) != int(addable.sum()):
        raise RuntimeError(
            "innovation decomposition not exhaustive: "
            f"assigned={int(assigned.sum())} "
            f"addable={int(addable.sum())}"
        )

    return masks, static_render


def _flush_shard(
    out_dir: Path,
    shard_id: int,
    rows: list[dict],
) -> dict:
    if not rows:
        raise ValueError(
            "cannot flush empty innovation cache shard"
        )

    name = f"shard_{int(shard_id):05d}.pt"
    payload = {
        "protocol": PROTOCOL,
        "future_aligned_semantic": torch.stack(
            [r["future_aligned_semantic"] for r in rows]
        ),
        "future_aligned_geometry_q": torch.stack(
            [r["future_aligned_geometry_q"] for r in rows]
        ),
        "base_explained": torch.stack(
            [r["base_explained"] for r in rows]
        ),
        "add_target": torch.stack(
            [r["add_target"] for r in rows]
        ),
        "semantic_target": torch.stack(
            [r["semantic_target"] for r in rows]
        ),
        "vertical_bits": torch.stack(
            [r["vertical_bits"] for r in rows]
        ),
        "candidate_mask": torch.stack(
            [r["candidate_mask"] for r in rows]
        ),
        "scene_name": [r["scene_name"] for r in rows],
        "t0_token": [r["t0_token"] for r in rows],
    }
    torch.save(payload, out_dir / name)
    return {"file": name, "count": len(rows)}


def main():
    p = argparse.ArgumentParser()
    add_config_args(p)
    p.add_argument(
        "--source-cache",
        required=True,
        help="V18 prepared train/val cache",
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataroot", required=True)
    p.add_argument("--info-pkl", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-windows", type=int, default=0)
    p.add_argument(
        "--preserve-record-order",
        action="store_true",
        help="Disable default stable grouping by scene used to improve frame-cache reuse.",
    )
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-size", type=int, default=8)
    p.add_argument(
        "--match-max-distance-m",
        type=float,
        default=4.0,
    )
    p.add_argument(
        "--positive-mode",
        choices=tuple(POSITIVE_MODES),
        default="core",
        help="Innovation responsibility used to build add targets.",
    )
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    if int(a.shard_size) <= 0:
        raise ValueError("shard-size must be positive")
    if int(a.num_shards) <= 0:
        raise ValueError("num-shards must be positive")
    if not 0 <= int(a.shard_index) < int(a.num_shards):
        raise ValueError("shard-index must be in [0,num-shards)")
    positive_categories = tuple(POSITIVE_MODES[str(a.positive_mode)])

    out_dir = Path(a.output_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(
            f"refusing non-empty output dir: {out_dir}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_runtime_config(a.config, a.override)
    pcfg = make_prepare_config(cfg)
    source_meta, records = base.load_cache(a.source_cache)
    if int(a.max_windows) > 0:
        records = records[
            : min(len(records), int(a.max_windows))
        ]
    global_num_windows = int(len(records))
    if not bool(a.preserve_record_order):
        # Stable grouping preserves the original temporal order inside each
        # scene while making overlapping history/future frames adjacent.  This
        # materially improves the NuScenes LRU hit rate on full train caches.
        records = sorted(
            records,
            key=lambda r: str(window_from_record(r).scene_name),
        )
    if int(a.num_shards) > 1:
        n = len(records)
        lo = n * int(a.shard_index) // int(a.num_shards)
        hi = n * (int(a.shard_index) + 1) // int(a.num_shards)
        records = records[lo:hi]
    if not records:
        raise RuntimeError("empty source cache shard")

    device = torch.device(
        a.device
        if a.device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    ck, model, _ = full._load_model(
        a.checkpoint, CLEAN_PROTOCOL, device
    )
    source = CachedSource(
        a.dataroot,
        info_pkl=a.info_pkl,
        verbose=False,
    )
    strong_cfg = StrongW2DetConfig(
        free_label=int(pcfg.free_label)
    )
    future_component_cfg = StrongW2DetConfig(
        free_label=int(pcfg.free_label),
        min_component_voxels=1,
        max_match_speed_mps=float(
            strong_cfg.max_match_speed_mps
        ),
        connectivity=int(strong_cfg.connectivity),
        fill_kernel=tuple(strong_cfg.fill_kernel),
        fill_min_fraction=float(
            strong_cfg.fill_min_fraction
        ),
    )
    metric_grid = _grid_spec(pcfg.grid)

    totals = {
        "positive_voxels_raw": 0,
        "positive_voxels_actionable": 0,
        "blocked_positive_voxels": 0,
        "positive_bev_cells": 0,
        "candidate_bev_cells": 0,
        "mixed_responsibility_bev_cells": 0,
        "excluded_bev_cells": 0,
    }
    category_voxels = {
        cat: 0 for cat in DECOMPOSITION_CATEGORIES
    }
    shards = []
    shard_rows = []
    scenes = set()
    started = time.perf_counter()
    stage_s = {
        "raw_load": 0.0,
        "base_forecast": 0.0,
        "ancestry_setup": 0.0,
        "history_alignment": 0.0,
        "future_targets_and_static": 0.0,
        "pack_and_write": 0.0,
    }

    for wi, rec in enumerate(records, start=1):
        tw = time.perf_counter()
        w = window_from_record(rec)
        scenes.add(str(w.scene_name))
        raw = load_nuscenes_window_raw(
            source, w, pcfg, include_gt=True
        )
        stage_s["raw_load"] += time.perf_counter() - tw

        tw = time.perf_counter()
        state = _prepare_record(
            rec, source, pcfg, strong_cfg, device
        )
        _stage_gpu_inputs(state, device)
        try:
            pred_all = _forecast_once(
                model, state, pcfg, strong_cfg, device
            )
        finally:
            _release_gpu_inputs(state)
        stage_s["base_forecast"] += time.perf_counter() - tw

        tw = time.perf_counter()
        history_occ = np.asarray(
            raw["history_occ"], dtype=np.uint8
        )
        history_obs = np.asarray(
            raw["history_observed"], dtype=bool
        )
        history_poses = np.asarray(
            raw["history_poses"], dtype=np.float64
        )
        future_poses = np.asarray(
            raw["future_poses"], dtype=np.float64
        )
        ann_hist = [
            _ann_map(source.nusc, tok)
            for tok in w.history_tokens
        ]
        ann0 = ann_hist[-1]
        t0_tokens = set(ann0)

        source_tokens = match_sources_to_annotations(
            state["current"],
            dynamic_annotations(
                source.nusc, w.t0_token
            ),
            max_distance_m=float(
                a.match_max_distance_m
            ),
        )
        represented = {
            str(x)
            for x in source_tokens
            if x is not None
        }
        stage_s["ancestry_setup"] += time.perf_counter() - tw

        tw = time.perf_counter()
        aligned_sem, aligned_geo, history_coverage_all = (
            build_future_aligned_history_bev_with_coverage(
                history_occ,
                history_obs,
                history_poses,
                future_poses,
                grid=pcfg.grid,
                free_label=int(pcfg.free_label),
            )
        )
        stage_s["history_alignment"] += time.perf_counter() - tw

        tw = time.perf_counter()
        explained_all = []
        add_targets = []
        semantic_targets = []
        vertical_targets = []
        candidate_masks = []

        for fi in range(len(future_poses)):
            gt = np.asarray(
                raw["future_gt_occ"][fi],
                dtype=np.uint8,
            )
            pred_v18 = np.asarray(
                pred_all[fi],
                dtype=np.uint8,
            )

            masks, static_render = (
                _category_masks_for_future(
                    gt=gt,
                    pred_v18=pred_v18,
                    future_pose=future_poses[fi],
                    future_token=w.future_tokens[fi],
                    state=state,
                    source_tokens=source_tokens,
                    represented=represented,
                    history_occ=history_occ,
                    history_obs=history_obs,
                    history_poses=history_poses,
                    history_tokens=tuple(
                        w.history_tokens
                    ),
                    ann_hist=ann_hist,
                    ann0=ann0,
                    t0_tokens=t0_tokens,
                    source=source,
                    pcfg=pcfg,
                    future_component_cfg=(
                        future_component_cfg
                    ),
                    metric_grid=metric_grid,
                    match_max_distance_m=float(
                        a.match_max_distance_m
                    ),
                    history_coverage=history_coverage_all[fi],
                )
            )

            for cat in DECOMPOSITION_CATEGORIES:
                category_voxels[cat] += int(
                    masks[cat].sum()
                )

            explained = protected_add_only(
                pred_v18,
                static_render,
                free_label=int(pcfg.free_label),
            )

            sup = build_innovation_bev_supervision(
                gt,
                explained,
                masks,
                free_label=int(pcfg.free_label),
                positive_categories=positive_categories,
            )
            for k in totals:
                totals[k] += int(sup[k])

            explained_all.append(explained)
            add_targets.append(sup["add_target"])
            semantic_targets.append(
                sup["semantic_target"]
            )
            vertical_targets.append(
                sup["vertical_target"]
            )
            candidate_masks.append(
                sup["candidate_mask"]
            )

        explained_all = np.stack(
            explained_all, axis=0
        )
        vertical_targets = np.stack(
            vertical_targets, axis=0
        )
        stage_s["future_targets_and_static"] += time.perf_counter() - tw

        tw = time.perf_counter()
        row = {
            "future_aligned_semantic": torch.from_numpy(
                aligned_sem.astype(np.uint8)
            ),
            "future_aligned_geometry_q": torch.from_numpy(
                quantize_geometry(aligned_geo)
            ),
            "base_explained": torch.from_numpy(
                base_explained_bev(
                    explained_all,
                    free_label=int(pcfg.free_label),
                ).astype(np.uint8)
            ),
            "add_target": torch.from_numpy(
                np.stack(
                    add_targets, axis=0
                ).astype(np.uint8)
            ),
            "semantic_target": torch.from_numpy(
                np.stack(
                    semantic_targets, axis=0
                ).astype(np.uint8)
            ),
            "vertical_bits": torch.from_numpy(
                pack_vertical_occupancy(
                    vertical_targets
                )
            ),
            "candidate_mask": torch.from_numpy(
                np.stack(
                    candidate_masks, axis=0
                ).astype(np.uint8)
            ),
            "scene_name": str(w.scene_name),
            "t0_token": str(w.t0_token),
        }
        shard_rows.append(row)

        if len(shard_rows) >= int(a.shard_size):
            shards.append(
                _flush_shard(
                    out_dir,
                    len(shards),
                    shard_rows,
                )
            )
            shard_rows = []
        stage_s["pack_and_write"] += time.perf_counter() - tw

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
                f"v19_innovation_cache "
                f"{wi}/{len(records)} "
                f"rate={wi/elapsed:.3f} win/s",
                flush=True,
            )

    if shard_rows:
        shards.append(
            _flush_shard(
                out_dir,
                len(shards),
                shard_rows,
            )
        )

    index = {
        "protocol": PROTOCOL,
        "source_cache": str(
            Path(a.source_cache).resolve()
        ),
        "source_cache_metadata_keys": sorted(
            str(k) for k in source_meta.keys()
        ),
        "base_checkpoint": str(
            Path(a.checkpoint).resolve()
        ),
        "base_checkpoint_epoch": int(
            ck.get("epoch", -1)
        ),
        "num_windows": int(len(records)),
        "global_num_windows_before_shard": int(global_num_windows),
        "num_shards": int(a.num_shards),
        "shard_index": int(a.shard_index),
        "num_scenes": int(len(scenes)),
        "scene_names": sorted(scenes),
        "future_frames": int(
            len(future_poses)
        ),
        "history_frames": int(
            len(history_poses)
        ),
        "grid_shape_hwd": [
            int(x)
            for x in pcfg.grid.shape_hwd
        ],
        "free_label": int(pcfg.free_label),
        "geometry_quantization": (
            "uint8_linear_0_1_255"
        ),
        "vertical_target_packing": (
            "uint16_lsb_z0"
        ),
        "explained_state": (
            "frozen_clean_e14_v18_plus_"
            "deterministic_static_memory"
        ),
        "future_gt_used_for_inference_input": False,
        "positive_mode": str(a.positive_mode),
        "positive_categories": list(positive_categories),
        "responsibility_policy": (
            "positive=" + "+".join(positive_categories)
            + "; all other addable decomposition categories=ignore; "
            "mixed BEV columns=ignore"
        ),
        "totals": totals,
        "category_voxels": category_voxels,
        "shards": shards,
        "timing_profile": {
            "elapsed_s": float(max(time.perf_counter() - started, 1e-9)),
            "windows_per_s": float(
                len(records) / max(time.perf_counter() - started, 1e-9)
            ),
            "stage_seconds": {
                k: float(v) for k, v in stage_s.items()
            },
            "scene_grouped": not bool(a.preserve_record_order),
        },
    }

    (out_dir / "index.json").write_text(
        json.dumps(index, indent=2),
        encoding="utf-8",
    )
    print(
        "\n=== V19 INNOVATION TRAINING CACHE ==="
    )
    print(
        json.dumps(
            {
                k: index[k]
                for k in (
                    "num_windows",
                    "num_scenes",
                    "grid_shape_hwd",
                    "totals",
                )
            },
            indent=2,
        )
    )
    print(f"saved {out_dir / 'index.json'}")


if __name__ == "__main__":
    main()
