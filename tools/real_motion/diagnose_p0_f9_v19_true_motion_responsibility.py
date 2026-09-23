#!/usr/bin/env python3
"""True-motion responsibility decomposition for V19 dynamic addable errors.

This diagnostic answers the gating question before another Innovation retrain:

  Where does Moving headroom actually live after splitting dynamic addable
  occupancy by *instance motion status* rather than motion-capable semantics?

The existing V19 decomposition classifies causal responsibility:
  history_source_recoverable
  t0_unrepresented_dynamic
  current_source_transportable_miss
  future_birth_dynamic
  source_shape_innovation
  dynamic_other_ambiguous

This script further splits every linked component into:
  true_moving          same GT instance exists at t0/future and is Moving-eligible
  common_nonmoving     same GT instance exists at t0/future but is below threshold
  not_common_unscored  no t0/future common identity, therefore current Moving metric
                       does not score that instance as true-moving
  unresolved           component ancestry could not be linked reliably

Important:
- Future GT identity/motion is diagnostic-only.
- No future information enters any deployed prediction.
- Perfect-add oracles only fill GT occupied voxels where frozen V18 predicts free.
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
from real_motion.nuscenes_adapter import gt_moving_support_for_horizon
from real_motion.prepared import load_nuscenes_window_raw
from real_motion.rigid_transport import rasterize_rigid_component
from real_motion.runtime_config import (
    add_config_args,
    load_runtime_config,
    make_prepare_config,
)
from real_motion.runtime_fastpath import extract_instances_cropped_exact
from real_motion.strong_w2det import StrongW2DetConfig
from real_motion.v19_innovation_targets import (
    match_future_components_many_to_one,
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
    HORIZONS,
    REPORT,
    _ann_map,
    _delta,
    _grid_spec,
    _metrics,
    _raw_state,
    _same_class_history_evidence,
    _source_target,
    _update_raw,
)
from tools.real_motion.eval_p0_f9_v17_local_stwm import window_from_record
from tools.real_motion.train_p0_f9_v18_se2_clean import (
    PROTOCOL as CLEAN_PROTOCOL,
)


PROTOCOL = "p0_f9_v19_true_motion_responsibility_v1"
DYNAMIC_CATEGORIES = (
    "history_source_recoverable",
    "t0_unrepresented_dynamic",
    "current_source_transportable_miss",
    "future_birth_dynamic",
    "source_shape_innovation",
    "dynamic_other_ambiguous",
)
MOTION_STATUSES = (
    "true_moving",
    "common_nonmoving",
    "not_common_unscored",
    "unresolved",
)
_DYNAMIC = tuple(int(x) for x in DYNAMIC_CLASS_IDS)


def _motion_status(
    token: str | None,
    *,
    common_tokens: set[str],
    moving_tokens: set[str],
) -> str:
    if token is None:
        return "unresolved"
    tok = str(token)
    if tok in moving_tokens:
        return "true_moving"
    if tok in common_tokens:
        return "common_nonmoving"
    return "not_common_unscored"


def _new_split_masks(shape):
    return {
        cat: {
            status: np.zeros(shape, dtype=bool)
            for status in MOTION_STATUSES
        }
        for cat in DYNAMIC_CATEGORIES
    }


def _assign(
    masks,
    split_masks,
    *,
    category: str,
    status: str,
    voxels: np.ndarray,
):
    m = np.asarray(voxels, dtype=bool)
    masks[category] |= m
    split_masks[category][status] |= m


def _oracle_update(
    state,
    hi,
    pred,
    gt,
    moving,
    mask,
    free_label,
):
    oracle = np.asarray(pred).copy()
    m = np.asarray(mask, dtype=bool)
    oracle[m] = np.asarray(gt)[m]
    _update_raw(
        state,
        hi,
        oracle,
        gt,
        moving,
        int(free_label),
    )


def main():
    p = argparse.ArgumentParser()
    add_config_args(p)
    p.add_argument("--val-cache", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataroot", required=True)
    p.add_argument("--info-pkl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-windows", type=int, default=0)
    p.add_argument("--match-max-distance-m", type=float, default=4.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--preserve-record-order", action="store_true")
    a = p.parse_args()

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
        fill_min_fraction=float(strong_cfg.fill_min_fraction),
    )
    metric_grid = _grid_spec(pcfg.grid)

    states = {"base": _raw_state()}
    for cat in DYNAMIC_CATEGORIES:
        states[f"category/{cat}"] = _raw_state()
        for status in MOTION_STATUSES:
            states[f"split/{cat}/{status}"] = _raw_state()
    for status in MOTION_STATUSES:
        states[f"status/{status}"] = _raw_state()

    key_groups = {
        "true_moving_source_shape_innovation": (
            ("source_shape_innovation", "true_moving"),
        ),
        "true_moving_current_source_transportable_miss": (
            ("current_source_transportable_miss", "true_moving"),
        ),
        "true_moving_known_ancestor_dynamic_miss": (
            ("t0_unrepresented_dynamic", "true_moving"),
            ("current_source_transportable_miss", "true_moving"),
        ),
        "true_moving_memory_recoverable": (
            ("history_source_recoverable", "true_moving"),
        ),
        "all_true_moving_dynamic_addable": tuple(
            (cat, "true_moving")
            for cat in DYNAMIC_CATEGORIES
        ),
        "innovation_candidate_all_motion_status": (
            ("future_birth_dynamic", "true_moving"),
            ("future_birth_dynamic", "common_nonmoving"),
            ("future_birth_dynamic", "not_common_unscored"),
            ("source_shape_innovation", "true_moving"),
            ("source_shape_innovation", "common_nonmoving"),
            ("source_shape_innovation", "not_common_unscored"),
        ),
    }
    for name in key_groups:
        states[f"group/{name}"] = _raw_state()

    counts = {
        cat: {status: 0 for status in MOTION_STATUSES}
        for cat in DYNAMIC_CATEGORIES
    }
    moving_overlap_counts = {
        cat: {status: 0 for status in MOTION_STATUSES}
        for cat in DYNAMIC_CATEGORIES
    }
    per_horizon_counts = {
        str(h): {
            cat: {status: 0 for status in MOTION_STATUSES}
            for cat in DYNAMIC_CATEGORIES
        }
        for h in HORIZONS
    }
    total_dynamic_addable = 0
    total_dynamic_moving_overlap = 0
    instance_audit = {
        str(h): {
            "common_dynamic_instances": 0,
            "true_moving_instances": 0,
            "common_nonmoving_instances": 0,
            "future_only_dynamic_instances": 0,
        }
        for h in HORIZONS
    }

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
        history_poses = np.asarray(
            raw["history_poses"],
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
            max_distance_m=float(
                a.match_max_distance_m
            ),
        )
        represented = {
            str(x)
            for x in source_tokens
            if x is not None
        }

        for hi, h in enumerate(HORIZONS):
            fi = REPORT[h]
            ftok = str(w.future_tokens[fi])
            fpose = np.asarray(
                source.pose(ftok),
                dtype=np.float64,
            )
            gt = np.asarray(
                raw["future_gt_occ"][fi],
                dtype=np.uint8,
            )
            pred = np.asarray(
                pred_all[fi],
                dtype=np.uint8,
            )
            gt_occ = gt != int(pcfg.free_label)
            gt_dynamic = gt_occ & np.isin(
                gt,
                np.asarray(_DYNAMIC, dtype=gt.dtype),
            )
            dynamic_addable = (
                gt_dynamic
                & (pred == int(pcfg.free_label))
            )

            annh = _ann_map(source.nusc, ftok)
            common_tokens = set(ann0) & set(annh)
            moving, moving_records, _ = (
                gt_moving_support_for_horizon(
                    source.nusc,
                    str(w.t0_token),
                    ftok,
                    float(h),
                    grid=pcfg.grid,
                )
            )
            moving_tokens = {
                str(r["instance_token"])
                for r in moving_records
            }
            instance_audit[str(h)][
                "common_dynamic_instances"
            ] += int(len(common_tokens))
            instance_audit[str(h)][
                "true_moving_instances"
            ] += int(len(moving_tokens))
            instance_audit[str(h)][
                "common_nonmoving_instances"
            ] += int(
                len(common_tokens - moving_tokens)
            )
            instance_audit[str(h)][
                "future_only_dynamic_instances"
            ] += int(
                len(set(annh) - set(ann0))
            )

            masks = {
                cat: np.zeros(gt.shape, dtype=bool)
                for cat in DYNAMIC_CATEGORIES
            }
            split_masks = _new_split_masks(gt.shape)

            represented_transport_by_token = {}
            for src_i, comp in enumerate(
                state["current"]
            ):
                tok = source_tokens[src_i]
                if tok is None:
                    continue
                tok = str(tok)
                a0 = ann0.get(tok)
                ah = annh.get(tok)
                if a0 is None or ah is None:
                    continue
                target, dyaw = _source_target(
                    np.asarray(
                        comp["centroid_world"],
                        dtype=np.float64,
                    ),
                    a0,
                    ah,
                    int(comp["class_id"])
                    in set(YAW_ENABLED_CLASS_IDS),
                )
                rc = rasterize_rigid_component(
                    comp["voxel_indices"],
                    int(comp["class_id"]),
                    state["current_pose"],
                    fpose,
                    source_center_world=np.asarray(
                        comp["centroid_world"],
                        dtype=np.float64,
                    ),
                    target_center_world=target,
                    yaw_delta_rad=float(dyaw),
                    grid=pcfg.grid,
                )
                mm = represented_transport_by_token.setdefault(
                    tok,
                    np.zeros(gt.shape, dtype=bool),
                )
                idx = np.asarray(
                    rc.voxel_indices,
                    dtype=np.int64,
                )
                if len(idx):
                    mm[
                        idx[:, 0],
                        idx[:, 1],
                        idx[:, 2],
                    ] = True

            future_components = extract_instances_cropped_exact(
                gt,
                fpose,
                grid=pcfg.grid,
                cfg=future_component_cfg,
            )
            links = match_future_components_many_to_one(
                future_components,
                annh,
                max_distance_m=float(
                    a.match_max_distance_m
                ),
            )
            assigned = np.zeros(
                gt.shape,
                dtype=bool,
            )

            for comp, (tok, _nearest_d) in zip(
                future_components,
                links,
            ):
                idx = np.asarray(
                    comp["voxel_indices"],
                    dtype=np.int64,
                )
                if len(idx) == 0:
                    continue
                cm = np.zeros(
                    gt.shape,
                    dtype=bool,
                )
                cm[
                    idx[:, 0],
                    idx[:, 1],
                    idx[:, 2],
                ] = True
                ca = cm & dynamic_addable
                if not bool(ca.any()):
                    continue

                if tok is None:
                    _assign(
                        masks,
                        split_masks,
                        category="dynamic_other_ambiguous",
                        status="unresolved",
                        voxels=ca,
                    )
                    assigned |= ca
                    continue

                tok = str(tok)
                ah = annh.get(tok)
                if (
                    ah is None
                    or int(ah["class_id"])
                    != int(comp["class_id"])
                ):
                    raise RuntimeError(
                        "future component ancestry mismatch"
                    )
                status = _motion_status(
                    tok,
                    common_tokens=common_tokens,
                    moving_tokens=moving_tokens,
                )

                if tok in represented:
                    transport = (
                        represented_transport_by_token.get(
                            tok
                        )
                    )
                    if transport is None:
                        raise RuntimeError(
                            "represented future token lacks "
                            "GT transport support"
                        )
                    transportable = ca & transport
                    shape_new = ca & ~transport
                    _assign(
                        masks,
                        split_masks,
                        category=(
                            "current_source_transportable_miss"
                        ),
                        status=status,
                        voxels=transportable,
                    )
                    _assign(
                        masks,
                        split_masks,
                        category="source_shape_innovation",
                        status=status,
                        voxels=shape_new,
                    )
                    assigned |= ca
                    continue

                seen_pre_t0 = _same_class_history_evidence(
                    tok,
                    int(ah["class_id"]),
                    tuple(w.history_tokens[:-1]),
                    history_occ[:-1],
                    history_poses[:-1],
                    ann_hist[:-1],
                    metric_grid,
                )
                if seen_pre_t0:
                    category = "history_source_recoverable"
                elif tok in t0_tokens:
                    category = "t0_unrepresented_dynamic"
                else:
                    category = "future_birth_dynamic"
                _assign(
                    masks,
                    split_masks,
                    category=category,
                    status=status,
                    voxels=ca,
                )
                assigned |= ca

            residual = dynamic_addable & ~assigned
            if bool(residual.any()):
                _assign(
                    masks,
                    split_masks,
                    category="dynamic_other_ambiguous",
                    status="unresolved",
                    voxels=residual,
                )
                assigned |= residual

            if not np.array_equal(
                assigned,
                dynamic_addable,
            ):
                raise RuntimeError(
                    "true-motion dynamic decomposition "
                    "is not exhaustive"
                )

            # Structural invariants of the causal/Moving contracts.
            if any(
                bool(
                    split_masks["source_shape_innovation"][s].any()
                )
                for s in ("not_common_unscored", "unresolved")
            ):
                raise RuntimeError(
                    "represented source-shape innovation "
                    "must have common t0/future identity"
                )
            if any(
                bool(
                    split_masks[
                        "current_source_transportable_miss"
                    ][s].any()
                )
                for s in ("not_common_unscored", "unresolved")
            ):
                raise RuntimeError(
                    "represented transport miss must have "
                    "common t0/future identity"
                )
            if bool(
                split_masks["future_birth_dynamic"][
                    "true_moving"
                ].any()
                or split_masks["future_birth_dynamic"][
                    "common_nonmoving"
                ].any()
            ):
                raise RuntimeError(
                    "future birth unexpectedly has common "
                    "t0/future Moving identity"
                )

            total_dynamic_addable += int(
                dynamic_addable.sum()
            )
            total_dynamic_moving_overlap += int(
                (dynamic_addable & moving).sum()
            )

            _update_raw(
                states["base"],
                hi,
                pred,
                gt,
                moving,
                int(pcfg.free_label),
            )

            for cat in DYNAMIC_CATEGORIES:
                _oracle_update(
                    states[f"category/{cat}"],
                    hi,
                    pred,
                    gt,
                    moving,
                    masks[cat],
                    pcfg.free_label,
                )
                for status in MOTION_STATUSES:
                    sm = split_masks[cat][status]
                    n = int(sm.sum())
                    counts[cat][status] += n
                    per_horizon_counts[str(h)][cat][
                        status
                    ] += n
                    moving_overlap_counts[cat][
                        status
                    ] += int((sm & moving).sum())
                    _oracle_update(
                        states[
                            f"split/{cat}/{status}"
                        ],
                        hi,
                        pred,
                        gt,
                        moving,
                        sm,
                        pcfg.free_label,
                    )

            for status in MOTION_STATUSES:
                sm = np.zeros(
                    gt.shape,
                    dtype=bool,
                )
                for cat in DYNAMIC_CATEGORIES:
                    sm |= split_masks[cat][status]
                _oracle_update(
                    states[f"status/{status}"],
                    hi,
                    pred,
                    gt,
                    moving,
                    sm,
                    pcfg.free_label,
                )

            for name, members in key_groups.items():
                gm = np.zeros(
                    gt.shape,
                    dtype=bool,
                )
                for cat, status in members:
                    gm |= split_masks[cat][status]
                _oracle_update(
                    states[f"group/{name}"],
                    hi,
                    pred,
                    gt,
                    moving,
                    gm,
                    pcfg.free_label,
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
                f"v19_true_motion {wi}/{len(records)} "
                f"rate={wi/elapsed:.3f} win/s",
                flush=True,
            )

    reports = {
        name: _metrics(raw_state)
        for name, raw_state in states.items()
    }
    baseline = reports["base"]

    split_report = {}
    for cat in DYNAMIC_CATEGORIES:
        split_report[cat] = {}
        cat_total = sum(
            counts[cat][s]
            for s in MOTION_STATUSES
        )
        for status in MOTION_STATUSES:
            n = int(counts[cat][status])
            overlap = int(
                moving_overlap_counts[cat][status]
            )
            split_report[cat][status] = {
                "addable_voxels": n,
                "share_within_category": float(
                    n / max(cat_total, 1)
                ),
                "share_of_dynamic_addable": float(
                    n / max(total_dynamic_addable, 1)
                ),
                "moving_metric_overlap_voxels": overlap,
                "moving_metric_overlap_fraction": float(
                    overlap / max(n, 1)
                ),
                "perfect_add_metrics": reports[
                    f"split/{cat}/{status}"
                ],
                "delta_vs_base": _delta(
                    reports[
                        f"split/{cat}/{status}"
                    ],
                    baseline,
                ),
                "per_horizon_voxels": {
                    str(h): int(
                        per_horizon_counts[str(h)][
                            cat
                        ][status]
                    )
                    for h in HORIZONS
                },
            }

    category_report = {}
    for cat in DYNAMIC_CATEGORIES:
        n = int(
            sum(
                counts[cat][s]
                for s in MOTION_STATUSES
            )
        )
        category_report[cat] = {
            "addable_voxels": n,
            "share_of_dynamic_addable": float(
                n / max(total_dynamic_addable, 1)
            ),
            "perfect_add_metrics": reports[
                f"category/{cat}"
            ],
            "delta_vs_base": _delta(
                reports[f"category/{cat}"],
                baseline,
            ),
        }

    status_report = {}
    for status in MOTION_STATUSES:
        n = int(
            sum(
                counts[cat][status]
                for cat in DYNAMIC_CATEGORIES
            )
        )
        overlap = int(
            sum(
                moving_overlap_counts[cat][status]
                for cat in DYNAMIC_CATEGORIES
            )
        )
        status_report[status] = {
            "addable_voxels": n,
            "share_of_dynamic_addable": float(
                n / max(total_dynamic_addable, 1)
            ),
            "moving_metric_overlap_voxels": overlap,
            "moving_metric_overlap_fraction": float(
                overlap / max(n, 1)
            ),
            "perfect_add_metrics": reports[
                f"status/{status}"
            ],
            "delta_vs_base": _delta(
                reports[f"status/{status}"],
                baseline,
            ),
        }

    group_report = {}
    for name, members in key_groups.items():
        n = int(
            sum(
                counts[cat][status]
                for cat, status in members
            )
        )
        overlap = int(
            sum(
                moving_overlap_counts[cat][status]
                for cat, status in members
            )
        )
        group_report[name] = {
            "members": [
                {
                    "category": cat,
                    "motion_status": status,
                }
                for cat, status in members
            ],
            "addable_voxels": n,
            "share_of_dynamic_addable": float(
                n / max(total_dynamic_addable, 1)
            ),
            "moving_metric_overlap_voxels": overlap,
            "moving_metric_overlap_fraction": float(
                overlap / max(n, 1)
            ),
            "perfect_add_metrics": reports[
                f"group/{name}"
            ],
            "delta_vs_base": _delta(
                reports[f"group/{name}"],
                baseline,
            ),
        }

    result = {
        "protocol": PROTOCOL,
        "analysis_only": True,
        "future_gt_used_for_prediction": False,
        "future_gt_identity_and_motion_used_for_diagnostic": True,
        "checkpoint": str(
            Path(a.checkpoint).resolve()
        ),
        "checkpoint_epoch": int(
            ck.get("epoch", -1)
        ),
        "num_windows": int(len(records)),
        "report_horizons_s": list(HORIZONS),
        "motion_status_contract": {
            "true_moving": (
                "linked dynamic instance is common to t0/future "
                "and appears in gt_moving_support_for_horizon "
                "moving_records"
            ),
            "common_nonmoving": (
                "linked dynamic instance is common to t0/future "
                "but below the current Moving eligibility threshold"
            ),
            "not_common_unscored": (
                "linked future dynamic instance has no common "
                "t0/future identity, so current Moving metric does "
                "not score that instance as true-moving"
            ),
            "unresolved": (
                "future occupancy component lacks reliable "
                "annotation ancestry"
            ),
        },
        "base_metrics": baseline,
        "total_dynamic_addable_voxels": int(
            total_dynamic_addable
        ),
        "dynamic_addable_moving_metric_overlap_voxels": int(
            total_dynamic_moving_overlap
        ),
        "categories": category_report,
        "motion_statuses": status_report,
        "category_by_motion_status": split_report,
        "key_oracles": group_report,
        "instance_audit": instance_audit,
        "decision_targets": {
            "innovation_headroom": (
                "key_oracles.true_moving_source_shape_innovation"
            ),
            "transport_headroom": (
                "key_oracles."
                "true_moving_current_source_transportable_miss"
            ),
            "known_ancestor_dynamic_headroom": (
                "key_oracles."
                "true_moving_known_ancestor_dynamic_miss"
            ),
        },
    }

    op = Path(a.output)
    op.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    op.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print(
        "\n=== V19 TRUE-MOTION RESPONSIBILITY ==="
    )
    print(
        "base:",
        json.dumps(
            {
                k: baseline[k]
                for k in (
                    "IoU",
                    "mIoU",
                    "MovingMacro",
                    "MovingMicro",
                )
            }
        ),
    )
    print(
        f"dynamic_addable={total_dynamic_addable} "
        f"moving_metric_overlap={total_dynamic_moving_overlap}"
    )
    for name in (
        "true_moving_source_shape_innovation",
        "true_moving_current_source_transportable_miss",
        "true_moving_known_ancestor_dynamic_miss",
        "true_moving_memory_recoverable",
        "all_true_moving_dynamic_addable",
        "innovation_candidate_all_motion_status",
    ):
        row = group_report[name]
        d = row["delta_vs_base"]
        print(
            f"{name:48s} "
            f"vox={row['addable_voxels']:8d} "
            f"metric_overlap={100*row['moving_metric_overlap_fraction']:6.2f}% "
            f"d_mIoU={d['mIoU']:+7.3f} "
            f"d_MovMacro={d['MovingMacro']:+7.3f} "
            f"d_MovMicro={d['MovingMicro']:+7.3f}"
        )

    print("\nSOURCE SHAPE SPLIT:")
    for status in MOTION_STATUSES:
        row = split_report[
            "source_shape_innovation"
        ][status]
        d = row["delta_vs_base"]
        print(
            f"{status:24s} "
            f"vox={row['addable_voxels']:8d} "
            f"share={100*row['share_within_category']:6.2f}% "
            f"d_mIoU={d['mIoU']:+7.3f} "
            f"d_MovMicro={d['MovingMicro']:+7.3f}"
        )

    print("\nCURRENT TRANSPORT MISS SPLIT:")
    for status in MOTION_STATUSES:
        row = split_report[
            "current_source_transportable_miss"
        ][status]
        d = row["delta_vs_base"]
        print(
            f"{status:24s} "
            f"vox={row['addable_voxels']:8d} "
            f"share={100*row['share_within_category']:6.2f}% "
            f"d_mIoU={d['mIoU']:+7.3f} "
            f"d_MovMicro={d['MovingMicro']:+7.3f}"
        )

    print("\nFUTURE BIRTH:")
    for status in MOTION_STATUSES:
        row = split_report[
            "future_birth_dynamic"
        ][status]
        if row["addable_voxels"] == 0:
            continue
        d = row["delta_vs_base"]
        print(
            f"{status:24s} "
            f"vox={row['addable_voxels']:8d} "
            f"d_mIoU={d['mIoU']:+7.3f} "
            f"d_MovMicro={d['MovingMicro']:+7.3f}"
        )
    print(f"saved {op}")


if __name__ == "__main__":
    main()
