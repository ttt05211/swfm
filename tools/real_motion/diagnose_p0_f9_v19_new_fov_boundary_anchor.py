#!/usr/bin/env python3
"""Boundary-anchor diagnostic for Static New-FOV Novelty.

This is the final causal probe before replacing the failed absolute direct-Z
decoder.  It asks whether ancestor-free New-FOV static geometry is locally
continuous with the nearest *known static-memory* column on the historical
side of the geometric field-of-view boundary.

For each New-FOV static GT-positive BEV column the diagnostic reports:
  * nearest-anchor distance;
  * majority-semantic match;
  * 16-bin vertical occupancy IoU;
  * bottom/top absolute error;
  * degradation over 0--2, 2--4, 4--8 and >8 m anchor-distance bins.

It also evaluates fully causal deterministic copy baselines that copy the
nearest known static 3D column into New-FOV up to 2/4/8 m or without a distance
limit.  Future GT is used only for diagnostic target analysis and oracle
reporting; copy predictions use history, poses, frozen V18 and Static Memory.
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
    dynamic_annotations,
    match_sources_to_annotations,
)
from real_motion.nuscenes_adapter import gt_moving_support_for_horizon
from real_motion.prepared import load_nuscenes_window_raw
from real_motion.runtime_config import (
    add_config_args,
    load_runtime_config,
    make_prepare_config,
)
from real_motion.strong_w2det import StrongW2DetConfig
from real_motion.v19_innovation import (
    build_future_aligned_history_and_static_memory,
)
from real_motion.v19_scene_memory import protected_add_only
from real_motion.v19_static_novelty import (
    copy_static_anchor_columns,
    history_grid_footprint_bev,
    majority_semantic_per_column,
    nearest_static_anchor_map,
)
from tools.real_motion import eval_p0_f9_v18_se2 as base
from tools.real_motion import eval_p0_f9_v18_full_validation as full
from tools.real_motion.benchmark_p0_f9_v18_runtime import (
    _forecast_once,
    _prepare_record,
    _release_gpu_inputs,
    _stage_gpu_inputs,
)
from tools.real_motion.build_p0_f9_v19_innovation_cache import (
    _category_masks_for_future,
)
from tools.real_motion.diagnose_p0_f9_v19_innovation_decomposition import (
    CachedSource,
    _ann_map,
    _grid_spec,
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


PROTOCOL = "p0_f9_v19_new_fov_boundary_anchor_diagnostic_v1"
DYNAMIC_IDS = tuple(int(x) for x in DYNAMIC_CLASS_IDS)
COPY_GATES_M = (2.0, 4.0, 8.0, None)
DISTANCE_BINS = (
    ("0_to_2m", 0.0, 2.0),
    ("2_to_4m", 2.0, 4.0),
    ("4_to_8m", 4.0, 8.0),
    ("gt_8m", 8.0, float("inf")),
)


def _empty_anchor_stats():
    return {
        "target_columns": 0,
        "valid_anchor_columns": 0,
        "semantic_matches": 0,
        "vertical_iou_sum": 0.0,
        "bottom_abs_error_bins_sum": 0.0,
        "top_abs_error_bins_sum": 0.0,
        "target_positive_bins_sum": 0,
        "anchor_positive_bins_sum": 0,
        "distance_m_sum": 0.0,
    }


def _accumulate_anchor_rows(
    stats,
    *,
    target_rows,
    anchor_rows,
    target_semantic,
    anchor_semantic,
    distances_m,
):
    if len(target_rows) == 0:
        return
    tgt = np.asarray(target_rows, dtype=bool)
    anc = np.asarray(anchor_rows, dtype=bool)
    if tgt.shape != anc.shape or tgt.ndim != 2:
        raise ValueError("target/anchor rows must be matching [N,Z]")
    n = int(tgt.shape[0])
    inter = (tgt & anc).sum(axis=1).astype(np.float64)
    union = (tgt | anc).sum(axis=1).astype(np.float64)
    viou = inter / np.maximum(union, 1.0)

    z = int(tgt.shape[1])
    idx = np.arange(z, dtype=np.int64)[None, :]
    tgt_bottom = np.where(tgt, idx, z).min(axis=1)
    tgt_top = np.where(tgt, idx, -1).max(axis=1)
    anc_bottom = np.where(anc, idx, z).min(axis=1)
    anc_top = np.where(anc, idx, -1).max(axis=1)

    stats["target_columns"] += n
    stats["valid_anchor_columns"] += n
    stats["semantic_matches"] += int(
        (
            np.asarray(target_semantic, dtype=np.int64)
            == np.asarray(anchor_semantic, dtype=np.int64)
        ).sum()
    )
    stats["vertical_iou_sum"] += float(viou.sum())
    stats["bottom_abs_error_bins_sum"] += float(
        np.abs(tgt_bottom - anc_bottom).sum()
    )
    stats["top_abs_error_bins_sum"] += float(
        np.abs(tgt_top - anc_top).sum()
    )
    stats["target_positive_bins_sum"] += int(tgt.sum())
    stats["anchor_positive_bins_sum"] += int(anc.sum())
    stats["distance_m_sum"] += float(
        np.asarray(distances_m, dtype=np.float64).sum()
    )


def _finalize_anchor_stats(stats, *, voxel_z_m):
    n = int(stats["valid_anchor_columns"])
    target = int(stats["target_columns"])
    return {
        **stats,
        "anchor_coverage": float(n / max(target, 1)),
        "semantic_match_rate": float(
            stats["semantic_matches"] / max(n, 1)
        ),
        "mean_vertical_iou": float(
            stats["vertical_iou_sum"] / max(n, 1)
        ),
        "mean_bottom_abs_error_bins": float(
            stats["bottom_abs_error_bins_sum"] / max(n, 1)
        ),
        "mean_top_abs_error_bins": float(
            stats["top_abs_error_bins_sum"] / max(n, 1)
        ),
        "mean_bottom_abs_error_m": float(
            stats["bottom_abs_error_bins_sum"]
            / max(n, 1)
            * float(voxel_z_m)
        ),
        "mean_top_abs_error_m": float(
            stats["top_abs_error_bins_sum"]
            / max(n, 1)
            * float(voxel_z_m)
        ),
        "mean_target_positive_bins": float(
            stats["target_positive_bins_sum"] / max(n, 1)
        ),
        "mean_anchor_positive_bins": float(
            stats["anchor_positive_bins_sum"] / max(n, 1)
        ),
        "mean_anchor_distance_m": float(
            stats["distance_m_sum"] / max(n, 1)
        ),
    }


def _variant_name(gate):
    return (
        "copy_all"
        if gate is None
        else f"copy_{int(gate)}m"
    )


def _oracle(base_occ, gt, target):
    out = np.asarray(base_occ).copy()
    m = np.asarray(target, dtype=bool)
    out[m] = np.asarray(gt)[m]
    return out


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
                "precision": float(
                    tp / max(tp + fp, 1)
                ),
            }
        )
    return {
        "per_horizon": rows,
        "added_tp": int(tp_all),
        "added_fp": int(fp_all),
        "precision": float(
            tp_all / max(tp_all + fp_all, 1)
        ),
    }


def main():
    p = argparse.ArgumentParser()
    add_config_args(p)
    p.add_argument("--val-cache", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataroot", required=True)
    p.add_argument("--info-pkl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-windows", type=int, default=64)
    p.add_argument("--alignment-workers", type=int, default=6)
    p.add_argument("--match-max-distance-m", type=float, default=4.0)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    if int(a.alignment_workers) <= 0:
        raise ValueError("alignment-workers must be positive")

    pcfg = make_prepare_config(
        load_runtime_config(a.config, a.override)
    )
    vx, vy, vz = (
        float(x) for x in pcfg.grid.voxel_size
    )
    if abs(vx - vy) > 1e-8:
        raise RuntimeError(
            "boundary-anchor diagnostic expects square XY voxels"
        )

    _, records = base.load_cache(a.val_cache)
    if int(a.max_windows) > 0:
        records = records[: min(len(records), int(a.max_windows))]
    records = sorted(
        records,
        key=lambda r: str(window_from_record(r).scene_name),
    )
    if not records:
        raise RuntimeError("empty validation cache")

    device = torch.device(
        a.device
        if a.device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    ck, model, _ = full._load_model(
        a.checkpoint,
        CLEAN_PROTOCOL,
        device,
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
        fill_min_fraction=float(strong_cfg.fill_min_fraction),
    )
    metric_grid = _grid_spec(pcfg.grid)

    variant_names = [
        "v18_static",
        *[_variant_name(x) for x in COPY_GATES_M],
        "oracle_new_fov_static",
    ]
    raw_by_variant = {
        name: _new_raw() for name in variant_names
    }
    proposal_audit = {
        _variant_name(g): {
            "proposed_voxels": 0,
            "added_voxels": 0,
            "windows_with_additions": 0,
        }
        for g in COPY_GATES_M
    }

    overall_anchor = _empty_anchor_stats()
    by_distance = {
        name: _empty_anchor_stats()
        for name, _, _ in DISTANCE_BINS
    }
    by_horizon = {
        str(h): _empty_anchor_stats()
        for h in HORIZONS
    }
    total_target_columns = 0
    no_anchor_target_columns = 0
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
                model,
                state,
                pcfg,
                strong_cfg,
                device,
            )
        finally:
            _release_gpu_inputs(state)

        history_occ = np.asarray(
            raw["history_occ"],
            dtype=np.uint8,
        )
        history_obs = np.asarray(
            raw["history_observed"],
            dtype=bool,
        )
        history_poses = np.asarray(
            raw["history_poses"],
            dtype=np.float64,
        )
        future_poses = np.asarray(
            raw["future_poses"],
            dtype=np.float64,
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
                source.nusc,
                w.t0_token,
            ),
            max_distance_m=float(a.match_max_distance_m),
        )
        represented = {
            str(x)
            for x in source_tokens
            if x is not None
        }

        (
            _aligned_sem,
            _aligned_geo,
            history_coverage_all,
            static_render_all,
        ) = build_future_aligned_history_and_static_memory(
            history_occ,
            history_obs,
            history_poses,
            future_poses,
            grid=pcfg.grid,
            free_label=int(pcfg.free_label),
            dynamic_class_ids=DYNAMIC_IDS,
            workers=int(a.alignment_workers),
        )

        window_added = {
            _variant_name(g): 0
            for g in COPY_GATES_M
        }

        for fi, h in enumerate(HORIZONS):
            gt = np.asarray(
                raw["future_gt_occ"][fi],
                dtype=np.uint8,
            )
            pred_v18 = np.asarray(
                pred_all[fi],
                dtype=np.uint8,
            )
            masks, static_render, _ = _category_masks_for_future(
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
                history_tokens=tuple(w.history_tokens),
                ann_hist=ann_hist,
                ann0=ann0,
                t0_tokens=t0_tokens,
                source=source,
                pcfg=pcfg,
                future_component_cfg=future_component_cfg,
                metric_grid=metric_grid,
                match_max_distance_m=float(
                    a.match_max_distance_m
                ),
                history_coverage=history_coverage_all[fi],
                static_render=static_render_all[fi],
                moving_tokens=None,
            )
            explained = protected_add_only(
                pred_v18,
                static_render,
                free_label=int(pcfg.free_label),
            )
            free = explained == int(pcfg.free_label)
            footprint = history_grid_footprint_bev(
                history_poses,
                future_poses[fi],
                pcfg.grid,
            )
            new_fov = ~footprint
            new_fov_3d = np.broadcast_to(
                new_fov[..., None],
                gt.shape,
            )
            target = (
                np.asarray(
                    masks["never_seen_static"],
                    dtype=bool,
                )
                & new_fov_3d
                & free
            )
            target_bev = target.any(axis=2)

            (
                dist_cells,
                anchor_x,
                anchor_y,
                anchor_valid,
            ) = nearest_static_anchor_map(
                static_render,
                footprint,
                free_label=int(pcfg.free_label),
            )
            static_occ = (
                np.asarray(static_render)
                != int(pcfg.free_label)
            )
            anchor_sem_map = majority_semantic_per_column(
                static_occ,
                static_render,
                num_classes=17,
                ignore_label=255,
            )
            target_sem_map = majority_semantic_per_column(
                target,
                gt,
                num_classes=17,
                ignore_label=255,
            )

            total_target_columns += int(target_bev.sum())
            valid_target = target_bev & anchor_valid
            no_anchor_target_columns += int(
                (target_bev & ~anchor_valid).sum()
            )
            tx, ty = np.nonzero(valid_target)
            if len(tx):
                ax = anchor_x[tx, ty]
                ay = anchor_y[tx, ty]
                target_rows = target[tx, ty]
                anchor_rows = static_occ[ax, ay]
                target_sem = target_sem_map[tx, ty]
                anchor_sem = anchor_sem_map[ax, ay]
                distance_m = (
                    dist_cells[tx, ty].astype(np.float64)
                    * vx
                )

                _accumulate_anchor_rows(
                    overall_anchor,
                    target_rows=target_rows,
                    anchor_rows=anchor_rows,
                    target_semantic=target_sem,
                    anchor_semantic=anchor_sem,
                    distances_m=distance_m,
                )
                _accumulate_anchor_rows(
                    by_horizon[str(h)],
                    target_rows=target_rows,
                    anchor_rows=anchor_rows,
                    target_semantic=target_sem,
                    anchor_semantic=anchor_sem,
                    distances_m=distance_m,
                )
                for name, lo, hi in DISTANCE_BINS:
                    pick = (
                        (distance_m >= float(lo))
                        & (distance_m < float(hi))
                    )
                    if bool(pick.any()):
                        _accumulate_anchor_rows(
                            by_distance[name],
                            target_rows=target_rows[pick],
                            anchor_rows=anchor_rows[pick],
                            target_semantic=target_sem[pick],
                            anchor_semantic=anchor_sem[pick],
                            distances_m=distance_m[pick],
                        )

            moving, _, _ = gt_moving_support_for_horizon(
                source.nusc,
                str(w.t0_token),
                str(w.future_tokens[fi]),
                float(h),
                grid=pcfg.grid,
            )
            _update(
                raw_by_variant["v18_static"],
                fi,
                explained,
                gt,
                moving,
                int(pcfg.free_label),
            )

            for gate in COPY_GATES_M:
                name = _variant_name(gate)
                proposal = copy_static_anchor_columns(
                    static_render,
                    new_fov,
                    anchor_x,
                    anchor_y,
                    dist_cells,
                    anchor_valid,
                    free_label=int(pcfg.free_label),
                    voxel_size_xy_m=vx,
                    max_distance_m=gate,
                )
                proposal_audit[name][
                    "proposed_voxels"
                ] += int(
                    (proposal != int(pcfg.free_label)).sum()
                )
                pred = protected_add_only(
                    explained,
                    proposal,
                    free_label=int(pcfg.free_label),
                )
                nadd = int(
                    (
                        (pred != int(pcfg.free_label))
                        & free
                    ).sum()
                )
                proposal_audit[name]["added_voxels"] += nadd
                window_added[name] += nadd
                _update(
                    raw_by_variant[name],
                    fi,
                    pred,
                    gt,
                    moving,
                    int(pcfg.free_label),
                )

            oracle = _oracle(explained, gt, target)
            _update(
                raw_by_variant["oracle_new_fov_static"],
                fi,
                oracle,
                gt,
                moving,
                int(pcfg.free_label),
            )

        for name in window_added:
            proposal_audit[name][
                "windows_with_additions"
            ] += int(window_added[name] > 0)

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
                f"v19_boundary_anchor "
                f"{wi}/{len(records)} "
                f"rate={wi/elapsed:.3f} win/s",
                flush=True,
            )

    metrics = {
        k: _finalize(v)
        for k, v in raw_by_variant.items()
    }
    baseline = metrics["v18_static"]
    deltas = {
        k: _delta(v, baseline)
        for k, v in metrics.items()
        if k != "v18_static"
    }
    effective = {
        _variant_name(g): _effective_addition_quality(
            raw_by_variant["v18_static"],
            raw_by_variant[_variant_name(g)],
        )
        for g in COPY_GATES_M
    }

    overall = _finalize_anchor_stats(
        overall_anchor,
        voxel_z_m=vz,
    )
    # Preserve the real denominator even when no anchor exists in a window.
    overall["target_columns"] = int(total_target_columns)
    overall["anchor_coverage"] = float(
        overall_anchor["valid_anchor_columns"]
        / max(total_target_columns, 1)
    )
    overall["no_anchor_target_columns"] = int(
        no_anchor_target_columns
    )

    result = {
        "protocol": PROTOCOL,
        "analysis_only": True,
        "future_gt_used_for_prediction": False,
        "future_gt_used_for_target_and_oracle_only": True,
        "checkpoint": str(Path(a.checkpoint).resolve()),
        "checkpoint_epoch": int(ck.get("epoch", -1)),
        "num_windows": int(len(records)),
        "anchor_contract": (
            "nearest occupied deterministic Static-Memory BEV column inside "
            "the union of six historical geometric grid footprints"
        ),
        "copy_contract": (
            "copy the anchor's exact 3D static semantic column into New-FOV; "
            "optionally gate only by causal anchor distance; protected add-only "
            "composition into frozen V18 + Static Memory"
        ),
        "anchor_similarity": {
            "overall": overall,
            "by_distance": {
                name: _finalize_anchor_stats(
                    stats,
                    voxel_z_m=vz,
                )
                for name, stats in by_distance.items()
            },
            "per_horizon": {
                str(h): _finalize_anchor_stats(
                    by_horizon[str(h)],
                    voxel_z_m=vz,
                )
                for h in HORIZONS
            },
        },
        "metrics": metrics,
        "delta_vs_v18_static": deltas,
        "copy_proposal_audit": proposal_audit,
        "copy_effective_addition_quality": effective,
        "raw_counts": {
            k: {
                kk: np.asarray(vv).tolist()
                for kk, vv in raw.items()
            }
            for k, raw in raw_by_variant.items()
        },
    }
    op = Path(a.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print("\n=== V19 NEW-FOV BOUNDARY-ANCHOR DIAGNOSTIC ===")
    print(
        "anchor overall:",
        json.dumps(
            {
                "coverage": overall["anchor_coverage"],
                "mean_distance_m": overall[
                    "mean_anchor_distance_m"
                ],
                "semantic_match": overall[
                    "semantic_match_rate"
                ],
                "vertical_iou": overall[
                    "mean_vertical_iou"
                ],
                "bottom_err_m": overall[
                    "mean_bottom_abs_error_m"
                ],
                "top_err_m": overall[
                    "mean_top_abs_error_m"
                ],
            }
        ),
    )
    print("\nBY DISTANCE:")
    for name, _, _ in DISTANCE_BINS:
        s = result["anchor_similarity"]["by_distance"][name]
        print(
            f"{name:10s} "
            f"n={s['valid_anchor_columns']:8d} "
            f"sem={100*s['semantic_match_rate']:6.2f}% "
            f"vIoU={100*s['mean_vertical_iou']:6.2f}% "
            f"bErr={s['mean_bottom_abs_error_m']:.3f}m "
            f"tErr={s['mean_top_abs_error_m']:.3f}m"
        )

    print("\nCOPY BASELINES:")
    for gate in COPY_GATES_M:
        name = _variant_name(gate)
        d = deltas[name]
        q = effective[name]
        print(
            f"{name:10s} "
            f"dIoU={d['IoU']:+7.3f} "
            f"dmIoU={d['mIoU']:+7.3f} "
            f"dMovMicro={d['MovingMicro']:+7.3f} "
            f"addP={100*q['precision']:6.2f}% "
            f"added={proposal_audit[name]['added_voxels']}"
        )
    od = deltas["oracle_new_fov_static"]
    print(
        "oracle_new_fov_static "
        f"dIoU={od['IoU']:+.3f} "
        f"dmIoU={od['mIoU']:+.3f}"
    )
    print(f"saved {op}")


if __name__ == "__main__":
    main()
