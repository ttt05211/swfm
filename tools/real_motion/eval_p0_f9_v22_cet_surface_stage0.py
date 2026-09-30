#!/usr/bin/env python3
"""One-pass Stage-0 audit for CET static surface-emergence tokens.

The evaluator trains nothing and never uses future GT in either deterministic
baseline.  It evaluates both frozen semantic scopes and all three corridor
widths in one raw-data/V18 pass.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from real_motion.metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS
from real_motion.nuscenes_adapter import NuScenesWindowSource, gt_moving_support_sequence
from real_motion.prepared import load_nuscenes_window_raw
from real_motion.runtime_config import add_config_args, load_runtime_config, make_prepare_config
from real_motion.strong_w2det import StrongW2DetConfig
from real_motion.v19_static_novelty import history_grid_footprint_bev_all
from real_motion.v22_causal_emergence import (
    COMPREHENSIVE_WIDTHS_M,
    CORE_SURFACE_IDS,
    EXTENDED_SURFACE_IDS,
    FRONTIER_TYPES,
    PROTOCOL,
    build_future_static_memory_only,
    build_surface_frontier_family,
    nearest_column_proposal,
    nearest_geometry_gt_semantic_proposal,
    oracle_geometry_history_semantic_proposal,
    oracle_surface_proposal,
    oracle_vertical_shift_proposal,
    protected_surface_add,
    tangent_plane_proposal,
)
from tools.real_motion import benchmark_p0_f9_v18_runtime as runtime
from tools.real_motion import eval_p0_f9_v18_full_validation as full
from tools.real_motion import eval_p0_f9_v18_se2 as base
from tools.real_motion.eval_p0_f9_v17_local_stwm import window_from_record
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import (
    Metrics,
    align_records,
    assert_forward_exact,
    delta,
    load_manifest,
    validate_clean_e14_checkpoint,
)
from tools.real_motion.train_p0_f9_v18_se2_clean import PROTOCOL as CLEAN_PROTOCOL


EVAL_PROTOCOL = "p0_f9_v22_cet_surface_stage0_audit_v2_all_frontiers_factorized"
ALL_HORIZONS_S = tuple(0.5 * (i + 1) for i in range(6))
REPORT_INDICES = (1, 3, 5)
REPORT_HORIZONS_S = (1.0, 2.0, 3.0)
CLASS_SETS = {
    "CORE_ROAD_SIDEWALK": CORE_SURFACE_IDS,
    "EXTENDED_GROUND": EXTENDED_SURFACE_IDS,
}
METHODS = (
    "SCOPE_GT",
    "CAUSAL_GT",
    "GT_GEOMETRY_HISTORY_SEMANTIC",
    "ORACLE_Z_SHIFT_HISTORY_SEMANTIC",
    "NEAREST_GEOMETRY_GT_SEMANTIC",
    "NEAREST_COLUMN",
    "TANGENT_PLANE",
)


def _width_tag(width: float) -> str:
    return f"R{float(width):.1f}".replace(".", "P")


def _variant(
    frontier_type: str, scope_name: str, width: float, method: str
) -> str:
    return f"{frontier_type}_{scope_name}_{_width_tag(width)}_{method}"


def _seed_counts() -> dict[str, int]:
    return {
        "added": 0,
        "occ_tp": 0,
        "semantic_tp": 0,
        "target_addable": 0,
        "target_recovered": 0,
    }


def _addition_counts(
    baseline: np.ndarray,
    prediction: np.ndarray,
    gt: np.ndarray,
    scope_target: np.ndarray,
    free_label: int,
) -> dict[str, int]:
    base_free = np.asarray(baseline) == int(free_label)
    added = base_free & (np.asarray(prediction) != int(free_label))
    target = np.asarray(scope_target, dtype=bool) & base_free
    return {
        "added": int(added.sum()),
        "occ_tp": int((added & (gt != int(free_label))).sum()),
        "semantic_tp": int((added & (prediction == gt) & (gt != int(free_label))).sum()),
        "target_addable": int(target.sum()),
        "target_recovered": int((target & (prediction == gt) & (gt != int(free_label))).sum()),
    }


def _add_only_metric_counts(
    baseline_counts,
    baseline: np.ndarray,
    prediction: np.ndarray,
    gt: np.ndarray,
    moving: np.ndarray,
    free_label: int,
):
    """Update frozen metric counts by scanning only add-only differences."""
    base_pred = np.asarray(baseline, dtype=np.uint8)
    pred = np.asarray(prediction, dtype=np.uint8)
    truth = np.asarray(gt, dtype=np.uint8)
    moving_mask = np.asarray(moving, dtype=bool)
    changed = pred != base_pred
    if bool((changed & (base_pred != int(free_label))).any()):
        raise RuntimeError("CET variant overwrote a non-free V18 voxel")
    if bool((changed & (pred == int(free_label))).any()):
        raise RuntimeError("CET variant removed occupancy instead of add-only")

    oi, ou, si, su, mi, mu = baseline_counts
    oi = int(oi)
    ou = int(ou)
    si = np.asarray(si, dtype=np.int64).copy()
    su = np.asarray(su, dtype=np.int64).copy()
    mi = np.asarray(mi, dtype=np.int64).copy()
    mu = np.asarray(mu, dtype=np.int64).copy()
    if not bool(changed.any()):
        return oi, ou, si, su, mi, mu

    p = pred[changed].astype(np.int64, copy=False)
    g = truth[changed].astype(np.int64, copy=False)
    free = int(free_label)
    oi += int((g != free).sum())
    ou += int((g == free).sum())
    confusion = np.bincount(g * 18 + p, minlength=18 * 18).reshape(18, 18)
    diagonal = np.diag(confusion)
    si += diagonal[:17]
    su += confusion.sum(axis=0)[:17] - diagonal[:17]

    moving_added = moving_mask[changed]
    if bool(moving_added.any()):
        mp = p[moving_added]
        mg = g[moving_added]
        moving_confusion = np.bincount(
            mg * 18 + mp, minlength=18 * 18
        ).reshape(18, 18)
        moving_diagonal = np.diag(moving_confusion)
        dynamic_ids = np.asarray(
            tuple(int(x) for x in DYNAMIC_CLASS_IDS), dtype=np.int64
        )
        mi += moving_diagonal[dynamic_ids]
        mu += (
            moving_confusion.sum(axis=0)[dynamic_ids]
            - moving_diagonal[dynamic_ids]
        )
    return oi, ou, si, su, mi, mu


def _merge_counts(dst: dict[str, int], src: dict[str, int]) -> None:
    for key, value in src.items():
        dst[key] += int(value)


def _quality(row: dict[str, int]) -> dict:
    added = int(row["added"])
    occ_tp = int(row["occ_tp"])
    target = int(row["target_addable"])
    return {
        **{k: int(v) for k, v in row.items()},
        "addition_occupancy_precision": float(occ_tp / added) if added else None,
        "addition_semantic_precision": float(row["semantic_tp"] / added) if added else None,
        "semantic_accuracy_given_occupied": (
            float(row["semantic_tp"] / occ_tp) if occ_tp else None
        ),
        "scope_target_semantic_recall": (
            float(row["target_recovered"] / target) if target else None
        ),
    }


def _new_support_audit() -> dict[str, int]:
    return {
        "new_query_bev_cells": 0,
        "historical_surface_anchor_bev_cells": 0,
        "scope_bev_cells": 0,
        "causal_bev_cells": 0,
        "scope_positive_bev_cells": 0,
        "causal_positive_bev_cells": 0,
        "scope_target_voxels": 0,
        "causal_target_voxels": 0,
    }


def _finalize_support_audit(row: dict[str, int]) -> dict:
    scope_bev = int(row["scope_bev_cells"])
    causal_bev = int(row["causal_bev_cells"])
    scope_pos = int(row["scope_positive_bev_cells"])
    causal_pos = int(row["causal_positive_bev_cells"])
    return {
        **{k: int(v) for k, v in row.items()},
        "causal_retention_of_scope_bev": (
            float(causal_bev / scope_bev) if scope_bev else None
        ),
        "scope_positive_bev_fraction": (
            float(scope_pos / scope_bev) if scope_bev else None
        ),
        "causal_positive_bev_fraction": (
            float(causal_pos / causal_bev) if causal_bev else None
        ),
        "causal_target_voxel_retention": (
            float(row["causal_target_voxels"] / row["scope_target_voxels"])
            if row["scope_target_voxels"]
            else None
        ),
    }


def _scene_delta(scene_states: dict, variant: str) -> dict:
    values = {}
    for scene_name, states in sorted(scene_states.items()):
        baseline = states["V18_BASE"].compute()
        current = states[variant].compute()
        value = float(current["mIoU"] - baseline["mIoU"])
        if np.isfinite(value):
            values[str(scene_name)] = value
    x = np.asarray(list(values.values()), dtype=np.float64)
    eps = 1e-12
    return {
        "scenes": int(len(x)),
        "positive": int((x > eps).sum()),
        "zero": int((np.abs(x) <= eps).sum()),
        "negative": int((x < -eps).sum()),
        "mean": float(x.mean()) if len(x) else None,
        "median": float(np.median(x)) if len(x) else None,
        "minimum": float(x.min()) if len(x) else None,
        "maximum": float(x.max()) if len(x) else None,
        "by_scene": values,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    add_config_args(parser)
    for name in (
        "val-cache",
        "population-manifest",
        "checkpoint",
        "expected-checkpoint-sha256",
        "dataroot",
        "info-pkl",
        "output",
    ):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-workers", type=int, default=0)
    parser.add_argument("--moving-workers", type=int, default=0)
    parser.add_argument("--exactness-windows", type=int, default=1)
    args = parser.parse_args()
    if args.cpu_workers < 0 or args.moving_workers < 0 or args.exactness_windows < 0:
        raise ValueError("worker/exactness arguments must be non-negative")
    cpu_workers = (
        max(1, min(8, (os.cpu_count() or 2) - 1))
        if args.cpu_workers == 0
        else max(1, int(args.cpu_workers))
    )
    moving_workers = cpu_workers if args.moving_workers == 0 else max(1, args.moving_workers)

    pcfg = make_prepare_config(load_runtime_config(args.config, args.override))
    manifest, keys, _ = load_manifest(args.population_manifest)
    _, raw_records = base.load_cache(args.val_cache)
    records = align_records(raw_records, keys)
    source = NuScenesWindowSource(args.dataroot, info_pkl=args.info_pkl, verbose=False)
    strong = StrongW2DetConfig(free_label=int(pcfg.free_label))
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    checkpoint, model, _ = full._load_model(args.checkpoint, CLEAN_PROTOCOL, device)
    checkpoint_sha = validate_clean_e14_checkpoint(
        checkpoint, args.checkpoint, args.expected_checkpoint_sha256
    )

    variants = tuple(
        _variant(frontier_type, scope_name, width, method)
        for frontier_type in FRONTIER_TYPES
        for scope_name in CLASS_SETS
        for width in COMPREHENSIVE_WIDTHS_M
        for method in METHODS
    )
    states = {"V18_BASE": Metrics(), **{name: Metrics() for name in variants}}
    counts = {name: _seed_counts() for name in variants}
    horizon_counts = {
        name: [_seed_counts() for _ in REPORT_INDICES] for name in variants
    }
    support_audit = {
        f"{frontier_type}_{scope_name}_{_width_tag(width)}": _new_support_audit()
        for frontier_type in FRONTIER_TYPES
        for scope_name in CLASS_SETS
        for width in COMPREHENSIVE_WIDTHS_M
    }
    class_target_voxels = defaultdict(int)
    class_added = {name: defaultdict(int) for name in variants}
    class_semantic_correct = {name: defaultdict(int) for name in variants}
    scene_states = defaultdict(
        lambda: {"V18_BASE": Metrics(), **{name: Metrics() for name in variants}}
    )
    timings = defaultdict(float)
    checked = 0
    started = time.perf_counter()

    for window_index, record in enumerate(records, 1):
        window_started = time.perf_counter()
        window = window_from_record(record)

        stage = time.perf_counter()
        raw = load_nuscenes_window_raw(
            source, window, pcfg, include_gt=True, io_workers=cpu_workers
        )
        timings["raw_load"] += time.perf_counter() - stage

        stage = time.perf_counter()
        prepared = runtime._prepare_record(
            record, source, pcfg, strong, device, raw_window=raw
        )
        runtime._stage_gpu_inputs(prepared, device)
        try:
            if checked < args.exactness_windows:
                assert_forward_exact(model, prepared, device)
                runtime._exactness_check(model, prepared, pcfg, strong, device)
                checked += 1
            base_prediction = runtime._forecast_once(model, prepared, pcfg, strong, device)
        finally:
            runtime._release_gpu_inputs(prepared)
        timings["v18_prepare_forecast"] += time.perf_counter() - stage

        stage = time.perf_counter()
        static_memory, observed_union_bev = build_future_static_memory_only(
            np.asarray(raw["history_occ"], dtype=np.uint8),
            np.asarray(raw["history_observed"], dtype=bool),
            np.asarray(raw["history_poses"], dtype=np.float64),
            np.asarray(raw["future_poses"], dtype=np.float64),
            grid=pcfg.grid,
            free_label=int(pcfg.free_label),
            dynamic_class_ids=tuple(int(x) for x in DYNAMIC_CLASS_IDS),
            workers=cpu_workers,
            return_observed_bev=True,
        )
        footprints = history_grid_footprint_bev_all(
            np.asarray(raw["history_poses"], dtype=np.float64),
            np.asarray(raw["future_poses"], dtype=np.float64),
            pcfg.grid,
        )
        timings["causal_surface_memory"] += time.perf_counter() - stage

        stage = time.perf_counter()
        moving_rows = gt_moving_support_sequence(
            source.nusc,
            str(window.t0_token),
            window.future_tokens,
            ALL_HORIZONS_S,
            grid=pcfg.grid,
            workers=moving_workers,
        )
        moving = np.stack([row[0] for row in moving_rows], axis=0)
        timings["moving_support"] += time.perf_counter() - stage

        stage = time.perf_counter()
        for report_position, horizon_index in enumerate(REPORT_INDICES):
            gt = np.asarray(raw["future_gt_occ"][horizon_index], dtype=np.uint8)
            baseline = np.asarray(base_prediction[horizon_index], dtype=np.uint8)
            baseline_counts = states["V18_BASE"].update(
                report_position,
                baseline,
                gt,
                moving[horizon_index],
                pcfg.free_label,
            )
            scene_states[window.scene_name]["V18_BASE"].update(
                report_position, counts=baseline_counts
            )
            if not np.array_equal(
                protected_surface_add(
                    baseline,
                    np.full_like(baseline, int(pcfg.free_label)),
                    free_label=pcfg.free_label,
                ),
                baseline,
            ):
                raise RuntimeError("zero-contribution CET identity failed")

            for scope_name, class_ids in CLASS_SETS.items():
                frontier_family = build_surface_frontier_family(
                    static_memory[horizon_index],
                    footprints[horizon_index],
                    observed_union_bev[horizon_index],
                    class_ids=class_ids,
                    widths_m=COMPREHENSIVE_WIDTHS_M,
                    free_label=pcfg.free_label,
                    voxel_size_xy_m=float(pcfg.grid.voxel_size[0]),
                )
                for frontier_type, frontier in frontier_family.items():
                    max_support = frontier.causal_by_width[
                        max(COMPREHENSIVE_WIDTHS_M)
                    ]
                    nearest_max = nearest_column_proposal(
                        static_memory[horizon_index],
                        max_support,
                        frontier.nearest_x,
                        frontier.nearest_y,
                        class_ids=class_ids,
                        free_label=pcfg.free_label,
                    )
                    tangent_max = tangent_plane_proposal(
                        static_memory[horizon_index],
                        max_support,
                        frontier.nearest_x,
                        frontier.nearest_y,
                        class_ids=class_ids,
                        free_label=pcfg.free_label,
                    )
                    gt_geometry_history_semantic_max = (
                        oracle_geometry_history_semantic_proposal(
                            gt,
                            max_support,
                            static_memory[horizon_index],
                            frontier.nearest_x,
                            frontier.nearest_y,
                            class_ids=class_ids,
                            free_label=pcfg.free_label,
                        )
                    )
                    oracle_z_shift_max = oracle_vertical_shift_proposal(
                        gt,
                        static_memory[horizon_index],
                        max_support,
                        frontier.nearest_x,
                        frontier.nearest_y,
                        class_ids=class_ids,
                        free_label=pcfg.free_label,
                    )
                    nearest_gt_semantic_max = (
                        nearest_geometry_gt_semantic_proposal(
                            gt,
                            nearest_max,
                            class_ids=class_ids,
                            free_label=pcfg.free_label,
                        )
                    )
                    for width in COMPREHENSIVE_WIDTHS_M:
                        prefix = (
                            f"{frontier_type}_{scope_name}_{_width_tag(width)}"
                        )
                        scope_support = frontier.scope_by_width[float(width)]
                        causal_support = frontier.causal_by_width[float(width)]
                        scope_oracle = oracle_surface_proposal(
                            gt,
                            scope_support,
                            class_ids=class_ids,
                            free_label=pcfg.free_label,
                        )
                        causal_oracle = oracle_surface_proposal(
                            gt,
                            causal_support,
                            class_ids=class_ids,
                            free_label=pcfg.free_label,
                        )

                        def _crop(proposal):
                            return np.where(
                                causal_support[..., None],
                                proposal,
                                int(pcfg.free_label),
                            ).astype(np.uint8, copy=False)

                        nearest = _crop(nearest_max)
                        tangent = _crop(tangent_max)
                        proposals = {
                            "SCOPE_GT": scope_oracle,
                            "CAUSAL_GT": causal_oracle,
                            "GT_GEOMETRY_HISTORY_SEMANTIC": _crop(
                                gt_geometry_history_semantic_max
                            ),
                            "ORACLE_Z_SHIFT_HISTORY_SEMANTIC": _crop(
                                oracle_z_shift_max
                            ),
                            "NEAREST_GEOMETRY_GT_SEMANTIC": _crop(
                                nearest_gt_semantic_max
                            ),
                            "NEAREST_COLUMN": nearest,
                            "TANGENT_PLANE": tangent,
                        }
                        target_mask = scope_oracle != int(pcfg.free_label)
                        audit = support_audit[prefix]
                        audit["new_query_bev_cells"] += int(
                            frontier.new_query_bev.sum()
                        )
                        audit["historical_surface_anchor_bev_cells"] += int(
                            frontier.anchor_bev.sum()
                        )
                        audit["scope_bev_cells"] += int(scope_support.sum())
                        audit["causal_bev_cells"] += int(causal_support.sum())
                        audit["scope_positive_bev_cells"] += int(
                            target_mask.any(axis=2).sum()
                        )
                        audit["causal_positive_bev_cells"] += int(
                            (causal_oracle != int(pcfg.free_label)).any(axis=2).sum()
                        )
                        audit["scope_target_voxels"] += int(target_mask.sum())
                        audit["causal_target_voxels"] += int(
                            (causal_oracle != int(pcfg.free_label)).sum()
                        )
                        for class_id in class_ids:
                            class_target_voxels[
                                f"{prefix}/class_{int(class_id)}"
                            ] += int(
                                (
                                    (scope_oracle == int(class_id))
                                    & (baseline == pcfg.free_label)
                                ).sum()
                            )

                        for method, proposal in proposals.items():
                            name = _variant(
                                frontier_type, scope_name, width, method
                            )
                            prediction = protected_surface_add(
                                baseline, proposal, free_label=pcfg.free_label
                            )
                            metric_counts = _add_only_metric_counts(
                                baseline_counts,
                                baseline,
                                prediction,
                                gt,
                                moving[horizon_index],
                                pcfg.free_label,
                            )
                            if (
                                window_index <= args.exactness_windows
                                and report_position == 0
                                and width == min(COMPREHENSIVE_WIDTHS_M)
                            ):
                                reference_counts = Metrics.counts(
                                    prediction,
                                    gt,
                                    moving[horizon_index],
                                    pcfg.free_label,
                                )
                                if not all(
                                    np.array_equal(actual, reference)
                                    for actual, reference in zip(
                                        metric_counts, reference_counts
                                    )
                                ):
                                    raise RuntimeError(
                                        "add-only metric delta != frozen full "
                                        f"confusion for {name}"
                                    )
                            states[name].update(
                                report_position, counts=metric_counts
                            )
                            scene_states[window.scene_name][name].update(
                                report_position, counts=metric_counts
                            )
                            addition = _addition_counts(
                                baseline,
                                prediction,
                                gt,
                                target_mask,
                                pcfg.free_label,
                            )
                            _merge_counts(counts[name], addition)
                            _merge_counts(
                                horizon_counts[name][report_position], addition
                            )
                            added = (
                                (baseline == pcfg.free_label)
                                & (prediction != pcfg.free_label)
                            )
                            for class_id in class_ids:
                                class_added[name][str(int(class_id))] += int(
                                    (
                                        added
                                        & (prediction == int(class_id))
                                    ).sum()
                                )
                                class_semantic_correct[name][
                                    str(int(class_id))
                                ] += int(
                                    (
                                        added
                                        & (prediction == int(class_id))
                                        & (gt == int(class_id))
                                    ).sum()
                                )
        timings["frontier_render_metrics"] += time.perf_counter() - stage
        timings["total_window"] += time.perf_counter() - window_started
        if window_index == 1 or window_index % 8 == 0 or window_index == len(records):
            print(
                f"v22_cet_surface {window_index}/{len(records)} "
                f"seconds_per_window={timings['total_window']/window_index:.2f} "
                f"elapsed_seconds={time.perf_counter()-started:.1f}",
                flush=True,
            )

    baseline_report = states["V18_BASE"].compute()
    reports = {}
    for name in variants:
        metrics = states[name].compute()
        reports[name] = {
            "metrics": metrics,
            "delta_vs_v18_pp": delta(metrics, baseline_report),
            "addition_quality": _quality(counts[name]),
            "addition_quality_by_horizon": {
                str(horizon): _quality(horizon_counts[name][index])
                for index, horizon in enumerate(REPORT_HORIZONS_S)
            },
            "per_predicted_surface_class": {
                class_id: {
                    "added": int(value),
                    "semantic_correct": int(class_semantic_correct[name][class_id]),
                    "precision": (
                        float(class_semantic_correct[name][class_id] / value)
                        if value
                        else None
                    ),
                }
                for class_id, value in sorted(class_added[name].items())
            },
        }

    exactness = {}
    rows = []
    for frontier_type in FRONTIER_TYPES:
        for scope_name in CLASS_SETS:
            for width in COMPREHENSIVE_WIDTHS_M:
                prefix = (
                    f"{frontier_type}_{scope_name}_{_width_tag(width)}"
                )
                names = {
                    method: _variant(
                        frontier_type, scope_name, width, method
                    )
                    for method in METHODS
                }
                gains = {
                    method: float(
                        reports[name]["delta_vs_v18_pp"]["mIoU"]
                    )
                    for method, name in names.items()
                }
                scope_gain = gains["SCOPE_GT"]
                causal_gain = gains["CAUSAL_GT"]
                exact = counts[names["SCOPE_GT"]]
                exactness[prefix] = {
                    "all_added_voxels_are_occupied_gt": (
                        exact["added"] == exact["occ_tp"]
                    ),
                    "all_added_voxels_match_gt_semantics": (
                        exact["added"] == exact["semantic_tp"]
                    ),
                    "all_addable_scope_targets_recovered": (
                        exact["target_addable"] == exact["target_recovered"]
                    ),
                }
                if not all(exactness[prefix].values()):
                    raise RuntimeError(
                        f"scope GT exactness failed for {prefix}: "
                        f"{exactness[prefix]}"
                    )
                for method in ("NEAREST_COLUMN", "TANGENT_PLANE"):
                    name = names[method]
                    rows.append(
                        {
                            "configuration": prefix,
                            "frontier_type": frontier_type,
                            "class_scope": scope_name,
                            "width_m": float(width),
                            "method": method,
                            "variant": name,
                            "scope_gt_delta_mIoU_pp": scope_gain,
                            "causal_gt_delta_mIoU_pp": causal_gain,
                            "causal_gt_retention_of_scope": (
                                float(causal_gain / scope_gain)
                                if scope_gain > 0
                                else None
                            ),
                            "gt_geometry_history_semantic_delta_mIoU_pp": gains[
                                "GT_GEOMETRY_HISTORY_SEMANTIC"
                            ],
                            "gt_geometry_history_semantic_retention_of_causal_gt": (
                                gains["GT_GEOMETRY_HISTORY_SEMANTIC"]
                                / causal_gain
                                if causal_gain > 0
                                else None
                            ),
                            "oracle_z_shift_history_semantic_delta_mIoU_pp": gains[
                                "ORACLE_Z_SHIFT_HISTORY_SEMANTIC"
                            ],
                            "oracle_z_shift_retention_of_causal_gt": (
                                gains["ORACLE_Z_SHIFT_HISTORY_SEMANTIC"]
                                / causal_gain
                                if causal_gain > 0
                                else None
                            ),
                            "nearest_geometry_gt_semantic_delta_mIoU_pp": gains[
                                "NEAREST_GEOMETRY_GT_SEMANTIC"
                            ],
                            "deterministic_delta_mIoU_pp": gains[method],
                            "deterministic_delta_IoU_pp": float(
                                reports[name]["delta_vs_v18_pp"]["IoU"]
                            ),
                            "addition_quality": reports[name][
                                "addition_quality"
                            ],
                            "support": _finalize_support_audit(
                                support_audit[prefix]
                            ),
                        }
                    )

    best = max(rows, key=lambda row: row["deterministic_delta_mIoU_pp"])
    max_scope = max(rows, key=lambda row: row["scope_gt_delta_mIoU_pp"])
    best_z_shift = max(
        rows,
        key=lambda row: row[
            "oracle_z_shift_history_semantic_delta_mIoU_pp"
        ],
    )
    best_variant = str(best["variant"])
    best_scene = _scene_delta(scene_states, best_variant)
    precision = best["addition_quality"]["addition_occupancy_precision"]
    causal_retention = max_scope["causal_gt_retention_of_scope"]
    gate = {
        "global_scope_gt_delta_mIoU_ge_1_00": (
            max_scope["scope_gt_delta_mIoU_pp"] >= 1.00
        ),
        "causal_gt_retention_ge_0_70": (
            causal_retention is not None and causal_retention >= 0.70
        ),
        "deterministic_delta_mIoU_ge_0_10": (
            best["deterministic_delta_mIoU_pp"] >= 0.10
        ),
        "deterministic_addition_precision_ge_0_60": (
            precision is not None and precision >= 0.60
        ),
        "scene_positive_not_fewer_than_negative": (
            best_scene["positive"] >= best_scene["negative"]
        ),
    }
    gate["pass"] = all(gate.values())
    if gate["pass"]:
        route = "freeze_best_surface_scope_then_build_surface_emergence_tokens"
    elif max_scope["scope_gt_delta_mIoU_pp"] < 1.00:
        route = "stop_surface_emergence_all_frontiers_have_insufficient_headroom"
    elif causal_retention is None or causal_retention < 0.70:
        route = "revise_causal_surface_anchor_support_before_training"
    elif (
        max_scope["gt_geometry_history_semantic_retention_of_causal_gt"]
        is not None
        and max_scope[
            "gt_geometry_history_semantic_retention_of_causal_gt"
        ]
        >= 0.80
    ):
        route = "headroom_exists_semantics_are_sufficient_test_learned_surface_geometry_tokens_on_dev64"
    else:
        route = "headroom_exists_but_geometry_and_semantics_are_coupled_test_joint_surface_tokens_on_dev64_only"

    result = {
        "protocol": EVAL_PROTOCOL,
        "cet_protocol": PROTOCOL,
        "scientific_baseline": {
            "branch": "freeze/v18-main-final-20260918",
            "commit": "ccf7d77e65e9773f441b35083d625b06791bfeaa",
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_training_mode": checkpoint.get("training_mode"),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "forward_exactness": "literal frozen ccf7d77 forward elementwise comparison",
            "renderer_exactness": "frozen runtime exactness check",
            "zero_contribution_identity_windows": len(records),
            "exactness_windows": checked,
        },
        "population": {
            "manifest": str(Path(args.population_manifest).resolve()),
            "windows": len(records),
            "selected_key_fingerprint": manifest["selected_key_fingerprint"],
            "manifest_fingerprint": manifest["manifest_fingerprint"],
        },
        "class_sets": {key: list(value) for key, value in CLASS_SETS.items()},
        "frontier_types": list(FRONTIER_TYPES),
        "widths_m": list(COMPREHENSIVE_WIDTHS_M),
        "v18_base": baseline_report,
        "variants": reports,
        "comparison_table": rows,
        "maximum_scope_headroom": max_scope,
        "best_oracle_vertical_shift": best_z_shift,
        "best_deterministic": best,
        "best_deterministic_scene_delta_mIoU_pp": best_scene,
        "scope_oracle_exactness": exactness,
        "support_audit": {
            key: _finalize_support_audit(value)
            for key, value in support_audit.items()
        },
        "scope_target_voxels_by_class": dict(sorted(class_target_voxels.items())),
        "stage0_gate": gate,
        "recommended_next_route": route,
        "performance": {
            "cpu_workers": cpu_workers,
            "moving_workers": moving_workers,
            "seconds_total": time.perf_counter() - started,
            "seconds_by_stage": {key: float(value) for key, value in timings.items()},
            "mean_seconds_per_window": timings["total_window"] / max(len(records), 1),
        },
        "contracts": {
            "main_branch": "Surface Emergence Tokens",
            "future_entity_branch": "auxiliary only; not evaluated in this Stage-0",
            "grid_entry_scope": "future query cells outside all historical grid footprints and within a frozen metric boundary width",
            "visibility_frontier_scope": "inside historical grid footprints, on the unknown side of the aligned historical mask_lidar BEV union, and within the frozen width of observed cells",
            "union_scope": "exact OR of grid-entry and visibility-frontier supports; never dense unknown-volume completion",
            "causal_support": "scope cells with a class-eligible historical static surface column within the same width",
            "scope_gt": "future GT semantics used only as exact diagnostic target/upper bound",
            "gt_geometry_history_semantic": "exact GT surface occupancy with nearest historical surface semantics; semantic-capacity diagnostic only",
            "oracle_z_shift_history_semantic": "nearest historical surface column with one per-column GT-selected vertical shift in +/-4 bins; vertical-alignment diagnostic only",
            "nearest_geometry_gt_semantic": "nearest historical geometry with GT-corrected semantics only at occupied surface hits; semantic-error diagnostic only",
            "nearest_column": "history-only nearest surface semantic/height/thickness continuation",
            "tangent_plane": "history-only nearest surface column with local tangent height continuation",
            "compositor": "V18-free add-only; never overwrites frozen V18 occupied voxels",
            "report_horizons_s": list(REPORT_HORIZONS_S),
            "moving_metric": "frozen Moving-mIoU v2; unchanged",
            "v20_completion_used": False,
            "transformer_used": False,
            "training_used": False,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "windows": len(records),
                "best_variant": best_variant,
                "maximum_scope_configuration": max_scope["configuration"],
                "maximum_scope_gt_delta_mIoU_pp": max_scope[
                    "scope_gt_delta_mIoU_pp"
                ],
                "maximum_scope_causal_gt_delta_mIoU_pp": max_scope[
                    "causal_gt_delta_mIoU_pp"
                ],
                "maximum_scope_causal_gt_retention": causal_retention,
                "deterministic_delta_mIoU_pp": best["deterministic_delta_mIoU_pp"],
                "deterministic_addition_precision": precision,
                "stage0_gate_pass": gate["pass"],
                "recommended_next_route": route,
                "seconds_total": time.perf_counter() - started,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
