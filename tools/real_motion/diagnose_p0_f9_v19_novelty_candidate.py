#!/usr/bin/env python3
"""V19 ancestor-free Novelty candidate/predictability diagnostic.

This diagnostic implements the revised Transport / Memory / Novelty contract:

Transport:
  known current-source motion / source-local refinement.
Memory:
  previously observed recoverable content.
Novelty:
  only future occupancy with no reliable history ancestor:
    * never_seen_static
    * future_birth_dynamic

It intentionally trains nothing.  The goal is to answer whether the two
Novelty targets have a useful *causal* candidate domain before another model is
introduced.

Static Novelty receives a deployable candidate derived only from the six
history LiDAR-observation masks warped into each future ego frame.  Birth
dynamic is audited separately because a future-born object can enter either an
unobserved region or a region that history observed as free; therefore it must
not be hard-masked to the unknown region.

Future GT occupancy / annotations are used only to define diagnostic targets
and perfect-add oracles.  No future GT enters a deployable prediction path.
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
from tools.real_motion.train_p0_f9_v18_se2_clean import (
    PROTOCOL as CLEAN_PROTOCOL,
)


PROTOCOL = "p0_f9_v19_ancestor_free_novelty_candidate_v1"
SEMANTIC_CLASSES = tuple(range(17))
DYNAMIC_IDS = tuple(int(x) for x in DYNAMIC_CLASS_IDS)
HORIZONS = tuple(0.5 * (i + 1) for i in range(6))


def _raw_state() -> dict[str, np.ndarray]:
    H = len(HORIZONS)
    return {
        "occ_inter": np.zeros(H, dtype=np.int64),
        "occ_union": np.zeros(H, dtype=np.int64),
        "sem_inter": np.zeros((H, len(SEMANTIC_CLASSES)), dtype=np.int64),
        "sem_union": np.zeros((H, len(SEMANTIC_CLASSES)), dtype=np.int64),
    }


def _update_raw(raw, hi, pred, gt, free_label):
    p = np.asarray(pred)
    g = np.asarray(gt)
    po = p != int(free_label)
    go = g != int(free_label)
    raw["occ_inter"][hi] += int((po & go).sum())
    raw["occ_union"][hi] += int((po | go).sum())
    for j, cid in enumerate(SEMANTIC_CLASSES):
        pp = p == int(cid)
        gg = g == int(cid)
        raw["sem_inter"][hi, j] += int((pp & gg).sum())
        raw["sem_union"][hi, j] += int((pp | gg).sum())


def _safe(inter, union):
    i = np.asarray(inter, dtype=np.float64)
    u = np.asarray(union, dtype=np.float64)
    out = np.full(i.shape, np.nan, dtype=np.float64)
    np.divide(i, u, out=out, where=u > 0)
    return 100.0 * out


def _metrics(raw):
    occ = _safe(raw["occ_inter"], raw["occ_union"])
    sem = _safe(raw["sem_inter"], raw["sem_union"])
    sem_h = np.nanmean(sem, axis=1)
    per = {}
    for hi, h in enumerate(HORIZONS):
        per[str(h)] = {
            "IoU": float(occ[hi]),
            "mIoU": float(sem_h[hi]),
        }
    return {
        "IoU": float(np.nanmean(occ)),
        "mIoU": float(np.nanmean(sem_h)),
        "per_horizon": per,
    }


def _delta(a, b):
    return {
        "IoU": float(a["IoU"]) - float(b["IoU"]),
        "mIoU": float(a["mIoU"]) - float(b["mIoU"]),
        "per_horizon": {
            str(h): {
                "IoU": float(a["per_horizon"][str(h)]["IoU"])
                - float(b["per_horizon"][str(h)]["IoU"]),
                "mIoU": float(a["per_horizon"][str(h)]["mIoU"])
                - float(b["per_horizon"][str(h)]["mIoU"]),
            }
            for h in HORIZONS
        },
    }


def _oracle(pred, gt, mask):
    out = np.asarray(pred).copy()
    m = np.asarray(mask, dtype=bool)
    out[m] = np.asarray(gt)[m]
    return out


def _empty_horizon_stats():
    return {
        "static_positive_voxels": 0,
        "static_positive_bev": 0,
        "birth_positive_voxels": 0,
        "birth_positive_bev": 0,
        "explained_free_voxels": 0,
        "explained_free_bev": 0,
        "unknown_free_voxels": 0,
        "unknown_any_bev": 0,
        "whole_unseen_bev": 0,
        "static_in_whole_unseen_voxels": 0,
        "static_in_whole_unseen_bev": 0,
        "birth_unknown_voxels": 0,
        "birth_seen_voxels": 0,
        "birth_whole_unseen_bev_voxels": 0,
        "birth_positive_bev_whole_unseen": 0,
    }


def _finalize_horizon_stats(row, grid_voxels_per_window, grid_bev_per_window, n):
    r = dict(row)
    static_v = int(r["static_positive_voxels"])
    static_b = int(r["static_positive_bev"])
    birth_v = int(r["birth_positive_voxels"])
    birth_b = int(r["birth_positive_bev"])
    unknown_v = int(r["unknown_free_voxels"])
    unknown_b = int(r["unknown_any_bev"])
    whole_b = int(r["whole_unseen_bev"])
    free_v = int(r["explained_free_voxels"])
    free_b = int(r["explained_free_bev"])

    r.update(
        {
            "static_voxel_prevalence_in_unknown_free": float(
                static_v / max(unknown_v, 1)
            ),
            "static_bev_prevalence_in_unknown_any": float(
                static_b / max(unknown_b, 1)
            ),
            "static_whole_unseen_voxel_coverage": float(
                r["static_in_whole_unseen_voxels"] / max(static_v, 1)
            ),
            "static_whole_unseen_bev_coverage": float(
                r["static_in_whole_unseen_bev"] / max(static_b, 1)
            ),
            "static_bev_prevalence_in_whole_unseen": float(
                r["static_in_whole_unseen_bev"] / max(whole_b, 1)
            ),
            "birth_voxel_prevalence_in_all_explained_free": float(
                birth_v / max(free_v, 1)
            ),
            "birth_bev_prevalence_in_all_explained_free": float(
                birth_b / max(free_b, 1)
            ),
            "birth_unknown_voxel_fraction": float(
                r["birth_unknown_voxels"] / max(birth_v, 1)
            ),
            "birth_seen_voxel_fraction": float(
                r["birth_seen_voxels"] / max(birth_v, 1)
            ),
            "birth_whole_unseen_bev_voxel_fraction": float(
                r["birth_whole_unseen_bev_voxels"] / max(birth_v, 1)
            ),
            "birth_whole_unseen_bev_coverage": float(
                r["birth_positive_bev_whole_unseen"] / max(birth_b, 1)
            ),
            "unknown_free_voxel_fraction_of_grid": float(
                unknown_v / max(int(grid_voxels_per_window) * int(n), 1)
            ),
            "unknown_any_bev_fraction_of_grid": float(
                unknown_b / max(int(grid_bev_per_window) * int(n), 1)
            ),
            "whole_unseen_bev_fraction_of_grid": float(
                whole_b / max(int(grid_bev_per_window) * int(n), 1)
            ),
        }
    )
    return r


def main():
    p = argparse.ArgumentParser()
    add_config_args(p)
    p.add_argument("--val-cache", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataroot", required=True)
    p.add_argument("--info-pkl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-windows", type=int, default=0)
    p.add_argument("--alignment-workers", type=int, default=6)
    p.add_argument("--match-max-distance-m", type=float, default=4.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--preserve-record-order", action="store_true")
    a = p.parse_args()

    if int(a.alignment_workers) <= 0:
        raise ValueError("alignment-workers must be positive")

    pcfg = make_prepare_config(
        load_runtime_config(a.config, a.override)
    )
    _, records = base.load_cache(a.val_cache)
    if int(a.max_windows) > 0:
        records = records[: min(len(records), int(a.max_windows))]
    if not bool(a.preserve_record_order):
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
        fill_min_fraction=float(
            strong_cfg.fill_min_fraction
        ),
    )
    metric_grid = _grid_spec(pcfg.grid)

    states = {
        "base_v18_static_memory": _raw_state(),
        "plus_never_seen_static": _raw_state(),
        "plus_future_birth_dynamic": _raw_state(),
        "plus_all_novelty": _raw_state(),
    }
    horizon_stats = {
        str(h): _empty_horizon_stats()
        for h in HORIZONS
    }
    static_class_hist = {
        str(cid): 0 for cid in SEMANTIC_CLASSES
    }
    birth_class_hist = {
        str(cid): 0 for cid in DYNAMIC_IDS
    }
    blocked = {
        "never_seen_static_after_memory": 0,
        "future_birth_dynamic_after_memory": 0,
    }
    scenes = set()
    started = time.perf_counter()

    grid_shape = tuple(int(x) for x in pcfg.grid.shape_hwd)
    grid_voxels = int(np.prod(grid_shape))
    grid_bev = int(grid_shape[0] * grid_shape[1])

    for wi, rec in enumerate(records, start=1):
        w = window_from_record(rec)
        scenes.add(str(w.scene_name))
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
                match_max_distance_m=float(a.match_max_distance_m),
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
            coverage = np.asarray(
                history_coverage_all[fi],
                dtype=bool,
            )

            static_raw = np.asarray(
                masks["never_seen_static"],
                dtype=bool,
            )
            birth_raw = np.asarray(
                masks["future_birth_dynamic"],
                dtype=bool,
            )
            static_pos = static_raw & free
            birth_pos = birth_raw & free
            blocked["never_seen_static_after_memory"] += int(
                (static_raw & ~free).sum()
            )
            blocked["future_birth_dynamic_after_memory"] += int(
                (birth_raw & ~free).sum()
            )

            # Inference-safe static candidates derived only from history
            # visibility + current explained state.
            unknown_free = (~coverage) & free
            unknown_any_bev = unknown_free.any(axis=2)
            whole_unseen_bev = (
                ~coverage.any(axis=2)
            ) & free.any(axis=2)

            static_bev = static_pos.any(axis=2)
            birth_bev = birth_pos.any(axis=2)
            free_bev = free.any(axis=2)

            hs = horizon_stats[str(h)]
            hs["static_positive_voxels"] += int(static_pos.sum())
            hs["static_positive_bev"] += int(static_bev.sum())
            hs["birth_positive_voxels"] += int(birth_pos.sum())
            hs["birth_positive_bev"] += int(birth_bev.sum())
            hs["explained_free_voxels"] += int(free.sum())
            hs["explained_free_bev"] += int(free_bev.sum())
            hs["unknown_free_voxels"] += int(unknown_free.sum())
            hs["unknown_any_bev"] += int(unknown_any_bev.sum())
            hs["whole_unseen_bev"] += int(whole_unseen_bev.sum())

            whole3d = np.broadcast_to(
                whole_unseen_bev[..., None],
                gt.shape,
            )
            hs["static_in_whole_unseen_voxels"] += int(
                (static_pos & whole3d).sum()
            )
            hs["static_in_whole_unseen_bev"] += int(
                (static_bev & whole_unseen_bev).sum()
            )
            hs["birth_unknown_voxels"] += int(
                (birth_pos & ~coverage).sum()
            )
            hs["birth_seen_voxels"] += int(
                (birth_pos & coverage).sum()
            )
            hs["birth_whole_unseen_bev_voxels"] += int(
                (birth_pos & whole3d).sum()
            )
            hs["birth_positive_bev_whole_unseen"] += int(
                (birth_bev & whole_unseen_bev).sum()
            )

            for cid in SEMANTIC_CLASSES:
                static_class_hist[str(cid)] += int(
                    (static_pos & (gt == int(cid))).sum()
                )
            for cid in DYNAMIC_IDS:
                birth_class_hist[str(cid)] += int(
                    (birth_pos & (gt == int(cid))).sum()
                )

            _update_raw(
                states["base_v18_static_memory"],
                fi,
                explained,
                gt,
                pcfg.free_label,
            )
            _update_raw(
                states["plus_never_seen_static"],
                fi,
                _oracle(explained, gt, static_pos),
                gt,
                pcfg.free_label,
            )
            _update_raw(
                states["plus_future_birth_dynamic"],
                fi,
                _oracle(explained, gt, birth_pos),
                gt,
                pcfg.free_label,
            )
            _update_raw(
                states["plus_all_novelty"],
                fi,
                _oracle(
                    explained,
                    gt,
                    static_pos | birth_pos,
                ),
                gt,
                pcfg.free_label,
            )

        if wi == 1 or wi % 25 == 0 or wi == len(records):
            elapsed = max(
                time.perf_counter() - started,
                1e-9,
            )
            print(
                f"v19_novelty_candidate {wi}/{len(records)} "
                f"rate={wi/elapsed:.3f} win/s",
                flush=True,
            )

    reports = {
        name: _metrics(st)
        for name, st in states.items()
    }
    baseline = reports["base_v18_static_memory"]
    finalized = {
        str(h): _finalize_horizon_stats(
            horizon_stats[str(h)],
            grid_voxels,
            grid_bev,
            len(records),
        )
        for h in HORIZONS
    }

    static_total = sum(
        finalized[str(h)]["static_positive_voxels"]
        for h in HORIZONS
    )
    birth_total = sum(
        finalized[str(h)]["birth_positive_voxels"]
        for h in HORIZONS
    )
    result = {
        "protocol": PROTOCOL,
        "analysis_only": True,
        "future_gt_used_for_prediction": False,
        "future_gt_used_for_target_and_oracle_only": True,
        "checkpoint": str(Path(a.checkpoint).resolve()),
        "checkpoint_epoch": int(ck.get("epoch", -1)),
        "num_windows": int(len(records)),
        "num_scenes": int(len(scenes)),
        "report_horizons_s": list(HORIZONS),
        "responsibility_contract": {
            "transport": (
                "known current-source motion and source-local refinement; "
                "source_shape_innovation is NOT Novelty"
            ),
            "memory": (
                "content recoverable from real historical observation"
            ),
            "novelty": [
                "never_seen_static",
                "future_birth_dynamic",
            ],
        },
        "static_candidate_contract": {
            "voxel_unknown_free": (
                "future-frame voxel is free after frozen V18 + deterministic "
                "Static Memory and has no aligned real LiDAR observation in "
                "any of the six history frames"
            ),
            "whole_unseen_bev": (
                "no z voxel in the future BEV column has any aligned history "
                "LiDAR observation; stricter but cheaper BEV candidate"
            ),
            "inference_safe": True,
        },
        "birth_candidate_contract": {
            "hard_unknown_mask_used": False,
            "reason": (
                "future-born dynamic occupancy may enter historically observed "
                "free space, so unknown-history support is audited but is not "
                "a valid hard inference gate"
            ),
        },
        "base_metrics": baseline,
        "oracle_deltas_vs_base": {
            "never_seen_static": _delta(
                reports["plus_never_seen_static"],
                baseline,
            ),
            "future_birth_dynamic": _delta(
                reports["plus_future_birth_dynamic"],
                baseline,
            ),
            "all_novelty": _delta(
                reports["plus_all_novelty"],
                baseline,
            ),
        },
        "per_horizon_candidate_stats": finalized,
        "class_histograms": {
            "never_seen_static": static_class_hist,
            "future_birth_dynamic": birth_class_hist,
        },
        "totals": {
            "never_seen_static_voxels": int(static_total),
            "future_birth_dynamic_voxels": int(birth_total),
            "blocked_after_v18_plus_static_memory": blocked,
        },
    }

    op = Path(a.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print("\n=== V19 ANCESTOR-FREE NOVELTY CANDIDATE DIAGNOSTIC ===")
    print(
        "base:",
        json.dumps(
            {
                k: baseline[k]
                for k in ("IoU", "mIoU")
            }
        ),
    )
    for name in (
        "never_seen_static",
        "future_birth_dynamic",
        "all_novelty",
    ):
        d = result["oracle_deltas_vs_base"][name]
        print(
            f"{name:24s} "
            f"d_IoU={d['IoU']:+7.3f} "
            f"d_mIoU={d['mIoU']:+7.3f}"
        )

    print("\nPER-HORIZON:")
    for h in HORIZONS:
        r = finalized[str(h)]
        print(
            f"{h:3.1f}s "
            f"static={r['static_positive_voxels']:9d} "
            f"unknown_vox_frac={100*r['unknown_free_voxel_fraction_of_grid']:6.2f}% "
            f"static_prev={100*r['static_voxel_prevalence_in_unknown_free']:7.4f}% "
            f"wholeBEV={100*r['whole_unseen_bev_fraction_of_grid']:6.2f}% "
            f"static_cov_whole={100*r['static_whole_unseen_bev_coverage']:6.2f}% "
            f"birth={r['birth_positive_voxels']:7d} "
            f"birth_unknown={100*r['birth_unknown_voxel_fraction']:6.2f}%"
        )

    print(
        "\nSTATIC CLASS HIST:",
        json.dumps(static_class_hist),
    )
    print(
        "BIRTH CLASS HIST:",
        json.dumps(birth_class_hist),
    )
    print(f"saved {op}")


if __name__ == "__main__":
    main()
