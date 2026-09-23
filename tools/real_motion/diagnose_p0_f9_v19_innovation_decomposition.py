#!/usr/bin/env python3
"""V19 innovation decomposition + perfect-add oracle diagnostic.

This analysis answers a gating question before any new V19 module is trained:
which future GT occupied voxels are missing from frozen Clean-E14 in an
add-only sense, and which causal mechanism could in principle supply them?

The disjoint addable categories are:
  * history_source_recoverable
  * t0_unrepresented_dynamic
  * current_source_transportable_miss
  * future_birth_dynamic
  * source_shape_innovation
  * dynamic_other_ambiguous
  * history_static_recoverable
  * history_static_seen_mismatch
  * never_seen_static
  * static_other_ambiguous

Dynamic GT occupancy is attributed through min-size-1 future occupancy
components.  For already represented sources, exact GT-motion transport of the
observed t0 source geometry separates known-ancestor transport/model misses
from genuinely new visible source geometry.

A voxel is "addable" only when GT is occupied and the frozen V18 prediction is
free.  Each perfect oracle therefore only fills currently-free voxels and never
changes an existing V18 prediction.  This is the exact upper bound relevant to
the proposed protected add-only V19 branches.

Future GT annotations/occupancy are diagnostic-only and never enter a
deployable prediction path.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from real_motion.geometry import quaternion_yaw, relative_transform, warp_mask
from real_motion.local_st_world_model_v18_se2 import YAW_ENABLED_CLASS_IDS
from real_motion.metrics.moving_miou_v2 import (
    BOX_MARGIN_M,
    Box3D,
    DYNAMIC_CLASS_IDS,
    GridSpec,
    rasterize_oriented_box,
)
from real_motion.motion_transport import (
    dynamic_annotations,
    match_sources_to_annotations,
)
from real_motion.nuscenes_adapter import (
    NuScenesWindowSource,
    category_to_dynamic_class,
    gt_moving_support_for_horizon,
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
from real_motion.v19_innovation_targets import (
    DECOMPOSITION_CATEGORIES,
    DECOMPOSITION_GROUPS,
    INNOVATION_POSITIVE_CATEGORIES,
    annotation_distance_bin,
    match_future_components_many_to_one,
)
from real_motion.v19_scene_memory import render_static_history_mosaic
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


PROTOCOL = "p0_f9_v19_innovation_decomposition_perfect_add_v2"
REPORT = {1.0: 1, 2.0: 3, 3.0: 5}
HORIZONS = tuple(REPORT)
SEMANTIC_CLASSES = tuple(range(17))
CATEGORIES = DECOMPOSITION_CATEGORIES
GROUPS = DECOMPOSITION_GROUPS
_DYNAMIC = tuple(int(x) for x in DYNAMIC_CLASS_IDS)
_DYNAMIC_SET = set(_DYNAMIC)


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


def _wrap(x: float) -> float:
    return float((float(x) + math.pi) % (2.0 * math.pi) - math.pi)


def _ego_yaw(T: np.ndarray) -> float:
    R = np.asarray(T, dtype=np.float64)[:3, :3]
    return math.atan2(float(R[1, 0]), float(R[0, 0]))


def _ann_map(nusc, token: str) -> dict[str, dict]:
    sample = nusc.get("sample", str(token))
    out = {}
    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        cid = category_to_dynamic_class(ann["category_name"])
        if cid is None:
            continue
        size = np.asarray(ann["size"], dtype=np.float64)
        out[str(ann["instance_token"])] = {
            "instance_token": str(ann["instance_token"]),
            "class_id": int(cid),
            "center_world": np.asarray(ann["translation"], dtype=np.float64),
            "yaw_world": float(quaternion_yaw(ann["rotation"])),
            "size_lwh": np.asarray(
                [size[1], size[0], size[2]], dtype=np.float64
            ),
        }
    return out


def _future_box(ann: dict, future_pose: np.ndarray) -> Box3D:
    W2F = np.linalg.inv(np.asarray(future_pose, dtype=np.float64))
    c = (W2F @ np.r_[ann["center_world"], 1.0])[:3]
    return Box3D(
        token=str(ann["instance_token"]),
        class_id=int(ann["class_id"]),
        center_xyz=tuple(float(x) for x in c),
        size_lwh=tuple(float(x) for x in ann["size_lwh"]),
        yaw=_wrap(float(ann["yaw_world"]) - _ego_yaw(future_pose)),
    )


def _source_target(source_center_world, ann0, annh, yaw_enabled):
    cs = np.asarray(source_center_world, dtype=np.float64)
    a0 = np.asarray(ann0["center_world"], dtype=np.float64)
    ah = np.asarray(annh["center_world"], dtype=np.float64)
    yaw = (
        _wrap(float(annh["yaw_world"]) - float(ann0["yaw_world"]))
        if yaw_enabled
        else 0.0
    )
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.asarray([[c, -s], [s, c]], dtype=np.float64)
    off = cs[:2] - a0[:2]
    ds = (ah[:2] - a0[:2]) + (R @ off - off)
    target = cs.copy()
    target[:2] = cs[:2] + ds
    return target, yaw


def _grid_spec(grid):
    return GridSpec(
        float(grid.x_min),
        float(grid.y_min),
        float(grid.z_min),
        tuple(float(x) for x in grid.voxel_size),
        tuple(int(x) for x in grid.shape_hwd),
    )


def _same_class_history_evidence(
    token: str,
    class_id: int,
    history_tokens,
    history_occ,
    history_poses,
    ann_maps,
    metric_grid,
) -> bool:
    """GT identity is used only to ask whether causal same-class occupancy existed."""
    for tok, sem, pose, amap in zip(
        history_tokens, history_occ, history_poses, ann_maps
    ):
        ann = amap.get(str(token))
        if ann is None or int(ann["class_id"]) != int(class_id):
            continue
        box = _future_box(ann, np.asarray(pose, dtype=np.float64))
        mask = rasterize_oriented_box(box, metric_grid, margin=0.5)
        if bool(((np.asarray(sem) == int(class_id)) & mask).any()):
            return True
    return False



def _raw_state():
    H = len(HORIZONS)
    return {
        "occ_inter": np.zeros(H, dtype=np.int64),
        "occ_union": np.zeros(H, dtype=np.int64),
        "sem_inter": np.zeros((H, len(SEMANTIC_CLASSES)), dtype=np.int64),
        "sem_union": np.zeros((H, len(SEMANTIC_CLASSES)), dtype=np.int64),
        "mov_inter": np.zeros((H, len(DYNAMIC_CLASS_IDS)), dtype=np.int64),
        "mov_union": np.zeros((H, len(DYNAMIC_CLASS_IDS)), dtype=np.int64),
    }


def _update_raw(raw, hi, pred, gt, moving, free_label):
    p = np.asarray(pred)
    g = np.asarray(gt)
    m = np.asarray(moving, dtype=bool)
    po, go = p != int(free_label), g != int(free_label)
    raw["occ_inter"][hi] += int((po & go).sum())
    raw["occ_union"][hi] += int((po | go).sum())
    for j, cid in enumerate(SEMANTIC_CLASSES):
        pp, gg = p == cid, g == cid
        raw["sem_inter"][hi, j] += int((pp & gg).sum())
        raw["sem_union"][hi, j] += int((pp | gg).sum())
    for j, cid in enumerate(DYNAMIC_CLASS_IDS):
        pp = (p == int(cid)) & m
        gg = (g == int(cid)) & m
        raw["mov_inter"][hi, j] += int((pp & gg).sum())
        raw["mov_union"][hi, j] += int((pp | gg).sum())


def _safe(inter, union):
    i = np.asarray(inter, dtype=np.float64)
    u = np.asarray(union, dtype=np.float64)
    out = np.full(i.shape, np.nan, dtype=np.float64)
    np.divide(i, u, out=out, where=u > 0)
    return 100.0 * out


def _metrics(raw):
    occ = _safe(raw["occ_inter"], raw["occ_union"])
    sem = _safe(raw["sem_inter"], raw["sem_union"])
    mov = _safe(raw["mov_inter"], raw["mov_union"])
    sem_h = np.nanmean(sem, axis=1)
    mov_macro = np.nanmean(mov, axis=1)
    mov_micro = _safe(
        raw["mov_inter"].sum(axis=1), raw["mov_union"].sum(axis=1)
    )
    per = {}
    for hi, h in enumerate(HORIZONS):
        per[str(h)] = {
            "IoU": float(occ[hi]),
            "mIoU": float(sem_h[hi]),
            "MovingMacro": float(mov_macro[hi]),
            "MovingMicro": float(mov_micro[hi]),
        }
    return {
        "IoU": float(np.nanmean(occ)),
        "mIoU": float(np.nanmean(sem_h)),
        "MovingMacro": float(np.nanmean(mov_macro)),
        "MovingMicro": float(np.nanmean(mov_micro)),
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
    p.add_argument("--match-max-distance-m", type=float, default=4.0)
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
    future_component_cfg = StrongW2DetConfig(
        free_label=int(pcfg.free_label),
        min_component_voxels=1,
        max_match_speed_mps=float(strong_cfg.max_match_speed_mps),
        connectivity=int(strong_cfg.connectivity),
        fill_kernel=tuple(strong_cfg.fill_kernel),
        fill_min_fraction=float(strong_cfg.fill_min_fraction),
    )
    metric_grid = _grid_spec(pcfg.grid)

    states = {"base": _raw_state()}
    for cat in CATEGORIES:
        states[f"base_plus_{cat}"] = _raw_state()
    for group in GROUPS:
        states[f"base_plus_group_{group}"] = _raw_state()
    states["base_plus_all_addable"] = _raw_state()

    counts = {
        h: {cat: 0 for cat in CATEGORIES}
        for h in HORIZONS
    }
    group_counts = {
        h: {group: 0 for group in GROUPS}
        for h in HORIZONS
    }
    ambiguous_audit = {
        "dynamic_other_voxels": 0,
        "dynamic_other_moving_voxels": 0,
        "dynamic_other_components": 0,
        "nearest_same_class_annotation_distance_bins": {
            "le_4m": 0,
            "4_to_6m": 0,
            "6_to_10m": 0,
            "gt_10m": 0,
            "no_same_class_annotation": 0,
        },
        "per_class_voxels": {str(cid): 0 for cid in _DYNAMIC},
    }
    total_addable = {h: 0 for h in HORIZONS}
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

        history_occ = np.asarray(raw["history_occ"], dtype=np.uint8)
        history_obs = np.asarray(raw["history_observed"], dtype=bool)
        history_poses = np.asarray(raw["history_poses"], dtype=np.float64)
        ann_hist = [_ann_map(source.nusc, tok) for tok in w.history_tokens]
        ann0 = ann_hist[-1]
        t0_tokens = set(ann0)

        # Strong current sources are already prepared in frozen source order.
        source_tokens = match_sources_to_annotations(
            state["current"],
            dynamic_annotations(source.nusc, w.t0_token),
            max_distance_m=float(a.match_max_distance_m),
        )
        represented = {str(x) for x in source_tokens if x is not None}

        for hi, h in enumerate(HORIZONS):
            fi = REPORT[h]
            ftok = str(w.future_tokens[fi])
            fpose = np.asarray(source.pose(ftok), dtype=np.float64)
            gt = np.asarray(
                source.load_semantics(w.scene_name, ftok), dtype=np.uint8
            )
            pred = np.asarray(pred_all[fi], dtype=np.uint8)
            gt_occ = gt != int(pcfg.free_label)
            addable = gt_occ & (pred == int(pcfg.free_label))
            total_addable[h] += int(addable.sum())

            masks = {
                cat: np.zeros(gt.shape, dtype=bool)
                for cat in CATEGORIES
            }
            gt_dynamic = gt_occ & np.isin(
                gt, np.asarray(_DYNAMIC, dtype=gt.dtype)
            )
            gt_static = gt_occ & ~gt_dynamic

            annh = _ann_map(source.nusc, ftok)

            # Per-instance GT-motion transport of the actually observed t0
            # source geometry.  If an addable GT voxel is covered here, it has
            # a reliable ancestor and is a transport/model miss, not innovation.
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
                    fpose,
                    source_center_world=np.asarray(
                        comp["centroid_world"], dtype=np.float64
                    ),
                    target_center_world=target,
                    yaw_delta_rad=float(dyaw),
                    grid=pcfg.grid,
                )
                mask = represented_transport_by_token.setdefault(
                    tok, np.zeros(gt.shape, dtype=bool)
                )
                idx = np.asarray(rc.voxel_indices, dtype=np.int64)
                if len(idx):
                    mask[idx[:, 0], idx[:, 1], idx[:, 2]] = True

            # Attribute every future dynamic GT voxel through occupancy
            # connected components first, rather than requiring every voxel to
            # lie inside a rasterized annotation box.  min_component_voxels=1
            # makes the decomposition exhaustive over dynamic GT occupancy.
            future_components = extract_instances_cropped_exact(
                gt,
                fpose,
                grid=pcfg.grid,
                cfg=future_component_cfg,
            )
            comp_links = match_future_components_many_to_one(
                future_components,
                annh,
                max_distance_m=float(a.match_max_distance_m),
            )
            dynamic_assigned = np.zeros(gt.shape, dtype=bool)
            unlinked_component_masks = []
            for comp, (tok, nearest_d) in zip(future_components, comp_links):
                idx = np.asarray(comp["voxel_indices"], dtype=np.int64)
                if len(idx) == 0:
                    continue
                comp_mask = np.zeros(gt.shape, dtype=bool)
                comp_mask[idx[:, 0], idx[:, 1], idx[:, 2]] = True
                comp_addable = comp_mask & addable & gt_dynamic
                if not bool(comp_addable.any()):
                    continue

                if tok is None:
                    masks["dynamic_other_ambiguous"] |= comp_addable
                    dynamic_assigned |= comp_addable
                    unlinked_component_masks.append(
                        (comp_addable, int(comp["class_id"]), float(nearest_d))
                    )
                    continue

                tok = str(tok)
                ah = annh.get(tok)
                if ah is None:
                    raise RuntimeError("linked future component token is missing")
                if int(ah["class_id"]) != int(comp["class_id"]):
                    raise RuntimeError("future component/annotation class mismatch")

                if tok in represented:
                    transport = represented_transport_by_token.get(tok)
                    if transport is None:
                        # A represented t0 source whose annotation vanishes
                        # from the future cannot be linked here; reaching this
                        # branch would indicate an ancestry bookkeeping bug.
                        raise RuntimeError(
                            "represented future token lacks GT transport support"
                        )
                    transportable = comp_addable & transport
                    shape_new = comp_addable & ~transport
                    masks["current_source_transportable_miss"] |= transportable
                    masks["source_shape_innovation"] |= shape_new
                    dynamic_assigned |= comp_addable
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
                    masks["history_source_recoverable"] |= comp_addable
                elif tok in t0_tokens:
                    masks["t0_unrepresented_dynamic"] |= comp_addable
                else:
                    masks["future_birth_dynamic"] |= comp_addable
                dynamic_assigned |= comp_addable

            # Any remaining dynamic addable voxel would mean the min-size-1
            # occupancy component extraction did not cover the semantic grid.
            residual_dynamic = addable & gt_dynamic & ~dynamic_assigned
            if bool(residual_dynamic.any()):
                masks["dynamic_other_ambiguous"] |= residual_dynamic

            # History static memory uses only lidar-observed non-dynamic voxels.
            static_render = render_static_history_mosaic(
                history_occ,
                history_obs,
                history_poses,
                fpose,
                grid=pcfg.grid,
                free_label=int(pcfg.free_label),
            )
            hist_static = (
                addable
                & gt_static
                & (static_render == gt)
            )
            masks["history_static_recoverable"] = hist_static

            # "Never seen" means no lidar-observed history voxel projects to
            # this future voxel.  It is stronger than simply being absent at t0.
            hist_coverage = np.zeros(gt.shape, dtype=bool)
            for obs, hp in zip(history_obs, history_poses):
                hist_coverage |= warp_mask(
                    obs,
                    relative_transform(hp, fpose),
                    grid=pcfg.grid,
                )
            masks["history_static_seen_mismatch"] = (
                addable
                & gt_static
                & hist_coverage
                & ~hist_static
            )
            masks["never_seen_static"] = (
                addable
                & gt_static
                & ~hist_coverage
                & ~hist_static
            )

            # Dynamic categories above are already mutually exclusive by
            # component ancestry.  Static categories are mutually exclusive by
            # visibility/mosaic state.
            masks["static_other_ambiguous"] = (
                addable
                & gt_static
                & ~masks["history_static_recoverable"]
                & ~masks["history_static_seen_mismatch"]
                & ~masks["never_seen_static"]
            )

            assigned = np.zeros(gt.shape, dtype=bool)
            for cat in CATEGORIES:
                overlap = assigned & masks[cat]
                if bool(overlap.any()):
                    raise RuntimeError(
                        f"innovation categories overlap at {cat}: "
                        f"{int(overlap.sum())} voxels"
                    )
                assigned |= masks[cat]
            if int(assigned.sum()) != int(addable.sum()):
                raise RuntimeError(
                    "innovation categories are not exhaustive: "
                    f"assigned={int(assigned.sum())} addable={int(addable.sum())}"
                )

            moving, _, _ = gt_moving_support_for_horizon(
                source.nusc,
                str(w.t0_token),
                ftok,
                float(h),
                grid=pcfg.grid,
            )
            dynamic_other = masks["dynamic_other_ambiguous"]
            ambiguous_audit["dynamic_other_voxels"] += int(
                dynamic_other.sum()
            )
            ambiguous_audit["dynamic_other_moving_voxels"] += int(
                (dynamic_other & moving).sum()
            )
            for comp_mask, cid, nearest_d in unlinked_component_masks:
                ambiguous_audit["dynamic_other_components"] += 1
                ambiguous_audit[
                    "nearest_same_class_annotation_distance_bins"
                ][annotation_distance_bin(float(nearest_d))] += int(
                    comp_mask.sum()
                )
                ambiguous_audit["per_class_voxels"][str(int(cid))] += int(
                    comp_mask.sum()
                )

            _update_raw(
                states["base"],
                hi,
                pred,
                gt,
                moving,
                int(pcfg.free_label),
            )
            for cat in CATEGORIES:
                counts[h][cat] += int(masks[cat].sum())
                oracle = pred.copy()
                oracle[masks[cat]] = gt[masks[cat]]
                _update_raw(
                    states[f"base_plus_{cat}"],
                    hi,
                    oracle,
                    gt,
                    moving,
                    int(pcfg.free_label),
                )
            for group, cats in GROUPS.items():
                group_mask = np.zeros(gt.shape, dtype=bool)
                for cat in cats:
                    group_mask |= masks[cat]
                group_counts[h][group] += int(group_mask.sum())
                oracle = pred.copy()
                oracle[group_mask] = gt[group_mask]
                _update_raw(
                    states[f"base_plus_group_{group}"],
                    hi,
                    oracle,
                    gt,
                    moving,
                    int(pcfg.free_label),
                )

            all_oracle = pred.copy()
            all_oracle[addable] = gt[addable]
            _update_raw(
                states["base_plus_all_addable"],
                hi,
                all_oracle,
                gt,
                moving,
                int(pcfg.free_label),
            )

        if wi == 1 or wi % 25 == 0 or wi == len(records):
            elapsed = max(time.perf_counter() - started, 1e-9)
            print(
                f"v19_decomposition {wi}/{len(records)} "
                f"rate={wi/elapsed:.3f} win/s",
                flush=True,
            )

    reports = {name: _metrics(s) for name, s in states.items()}
    baseline = reports["base"]
    category_report = {}
    for cat in CATEGORIES:
        category_report[cat] = {
            "addable_voxels": int(sum(counts[h][cat] for h in HORIZONS)),
            "share_of_addable": float(
                sum(counts[h][cat] for h in HORIZONS)
                / max(sum(total_addable.values()), 1)
            ),
            "perfect_add_metrics": reports[f"base_plus_{cat}"],
            "delta_vs_base": _delta(
                reports[f"base_plus_{cat}"], baseline
            ),
            "per_horizon_voxels": {
                str(h): int(counts[h][cat]) for h in HORIZONS
            },
        }

    group_report = {}
    for group, cats in GROUPS.items():
        group_report[group] = {
            "categories": list(cats),
            "addable_voxels": int(
                sum(group_counts[h][group] for h in HORIZONS)
            ),
            "share_of_addable": float(
                sum(group_counts[h][group] for h in HORIZONS)
                / max(sum(total_addable.values()), 1)
            ),
            "perfect_add_metrics": reports[f"base_plus_group_{group}"],
            "delta_vs_base": _delta(
                reports[f"base_plus_group_{group}"], baseline
            ),
            "per_horizon_voxels": {
                str(h): int(group_counts[h][group]) for h in HORIZONS
            },
        }

    ambiguous_total = int(ambiguous_audit["dynamic_other_voxels"])
    ambiguous_audit["moving_fraction"] = float(
        ambiguous_audit["dynamic_other_moving_voxels"]
        / max(ambiguous_total, 1)
    )

    result = {
        "protocol": PROTOCOL,
        "analysis_only": True,
        "future_gt_used_for_prediction": False,
        "future_gt_used_for_diagnostic_oracle": True,
        "checkpoint": str(Path(a.checkpoint).resolve()),
        "checkpoint_epoch": int(ck.get("epoch", -1)),
        "num_windows": int(len(records)),
        "report_horizons_s": list(HORIZONS),
        "category_contract": (
            "disjoint partition of GT-occupied voxels where frozen V18 predicts "
            "free; perfect oracles only add the GT label at those cells"
        ),
        "base_metrics": baseline,
        "all_addable_perfect_oracle": {
            "metrics": reports["base_plus_all_addable"],
            "delta_vs_base": _delta(
                reports["base_plus_all_addable"], baseline
            ),
        },
        "categories": category_report,
        "groups": group_report,
        "dynamic_ambiguous_audit": ambiguous_audit,
        "training_target_candidate": {
            "positive_categories": list(INNOVATION_POSITIVE_CATEGORIES),
            "excluded_memory_categories": list(
                GROUPS["memory_addressable"]
            ),
            "excluded_known_ancestor_model_miss": list(
                GROUPS["known_ancestor_model_miss"]
            ),
            "excluded_ambiguous_categories": list(GROUPS["ambiguous"]),
            "contract": (
                "first innovation training candidate uses only future-born "
                "dynamic occupancy, new visible geometry of represented "
                "sources, and never-seen static occupancy; all known-ancestor "
                "misses and unresolved ambiguity remain excluded"
            ),
        },
        "per_horizon_total_addable_voxels": {
            str(h): int(total_addable[h]) for h in HORIZONS
        },
    }
    op = Path(a.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\\n=== V19 INNOVATION DECOMPOSITION / PERFECT-ADD ORACLE ===")
    print("base:", json.dumps({
        k: baseline[k]
        for k in ("IoU", "mIoU", "MovingMacro", "MovingMicro")
    }))
    for cat in CATEGORIES:
        row = category_report[cat]
        print(
            f"{cat:28s} share={100*row['share_of_addable']:7.3f}% "
            f"d_mIoU={row['delta_vs_base']['mIoU']:+7.3f} "
            f"d_MovMicro={row['delta_vs_base']['MovingMicro']:+7.3f}"
        )
    print("\nGROUP ORACLES:")
    for group in GROUPS:
        row = group_report[group]
        print(
            f"{group:28s} share={100*row['share_of_addable']:7.3f}% "
            f"d_mIoU={row['delta_vs_base']['mIoU']:+7.3f} "
            f"d_MovMicro={row['delta_vs_base']['MovingMicro']:+7.3f}"
        )
    print(
        "DYNAMIC AMBIGUOUS AUDIT:",
        json.dumps(ambiguous_audit),
    )
    print(
        "ALL ADDABLE:",
        json.dumps(result["all_addable_perfect_oracle"]["delta_vs_base"]),
    )
    print(f"saved {op}")


if __name__ == "__main__":
    main()
