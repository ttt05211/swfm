#!/usr/bin/env python3
"""P0-F0.5: split P0-F0 C into source-association failure modes.

Important contract: the frozen Moving-mIoU v2 records already require the same
instance token at t0 and the future horizon. Therefore this audit does not call C
"future birth". Instead it checks whether a causal t0 occupancy component exists
and why the GT-supervision matcher failed to associate it.
"""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from real_motion.geometry import ego_compensate_sequence, quaternion_yaw
from real_motion.metrics.moving_miou_v2 import Box3D, GridSpec, rasterize_oriented_box
from real_motion.motion import decompose_masks
from real_motion.msp import (
    SOURCE_C,
    dynamic_t0_instances,
    extract_msp_candidates,
    match_candidates_to_instances,
    source_type_for_token,
)
from real_motion.msp_audit import (
    CATEGORIES,
    DISTANCE_BINS,
    C_NO_SAME_CLASS_CANDIDATE,
    C_DISTANCE_GATE,
    C_ONE_TO_ONE_CONFLICT,
    classify_unmatched_instance,
    distance_bin,
    mask_touches_xy_boundary,
)
from real_motion.nuscenes_adapter import NuScenesWindowSource, box3d_from_dict
from real_motion.prepared import (
    _align_observation_sequence,
    load_nuscenes_window_raw,
    prepare_nuscenes_window,
)
from real_motion.runtime_config import (
    add_config_args,
    load_runtime_config,
    make_prepare_config,
    save_resolved_config,
)
from tools.real_motion.p0_hybrid_scene_disjoint_validate import (
    DEFAULT_SEED,
    select_scene_disjoint_windows,
)

REPORT_INDICES = {1.0: 1, 2.0: 3, 3.0: 5}


def _ann_map(nusc, sample_token):
    sample = nusc.get("sample", sample_token)
    out = {}
    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        out[str(ann["instance_token"])] = ann
    return out


def _wrap_angle(x):
    return (float(x) + math.pi) % (2.0 * math.pi) - math.pi


def _ann_to_t0_box(ann, t0_pose, class_id):
    world_to_t0 = np.linalg.inv(np.asarray(t0_pose, dtype=np.float64))
    center_world = np.asarray(ann["translation"], dtype=np.float64)
    center_t0 = (world_to_t0 @ np.r_[center_world, 1.0])[:3]
    yaw_world = quaternion_yaw(ann["rotation"])
    yaw_t0_world = math.atan2(float(t0_pose[1, 0]), float(t0_pose[0, 0]))
    w, l, h = ann["size"]
    return Box3D(
        token=str(ann["instance_token"]),
        class_id=int(class_id),
        center_xyz=tuple(float(v) for v in center_t0),
        size_lwh=(float(l), float(w), float(h)),
        yaw=_wrap_angle(yaw_world - yaw_t0_world),
    )


def _grid_spec(grid):
    return GridSpec(
        x_min=float(grid.x_min),
        y_min=float(grid.y_min),
        z_min=float(grid.z_min),
        voxel_size=tuple(float(v) for v in grid.voxel_size),
        shape_hwd=tuple(int(v) for v in grid.shape_hwd),
    )


def _counter():
    return {
        "records": 0,
        "records_with_arrival_voxels": 0,
        "records_with_missed_voxels": 0,
        "arrival_voxels": 0,
        "missed_voxels": 0,
    }


def _add(dst, *, arrival_voxels, missed_voxels):
    dst["records"] += 1
    if arrival_voxels > 0:
        dst["records_with_arrival_voxels"] += 1
    if missed_voxels > 0:
        dst["records_with_missed_voxels"] += 1
    dst["arrival_voxels"] += int(arrival_voxels)
    dst["missed_voxels"] += int(missed_voxels)


def _finalize(table):
    out = {}
    total_miss = sum(int(v["missed_voxels"]) for v in table.values())
    total_records = sum(int(v["records"]) for v in table.values())
    for key, v in table.items():
        row = dict(v)
        row["share_of_C_missed_voxels"] = float(v["missed_voxels"] / max(total_miss, 1))
        row["share_of_C_records"] = float(v["records"] / max(total_records, 1))
        row["missed_fraction_within_arrival"] = float(
            v["missed_voxels"] / max(v["arrival_voxels"], 1)
        )
        out[key] = row
    return out


def _new_horizon_state():
    return {
        "mechanism": {k: _counter() for k in CATEGORIES},
        "distance_bin": {k: _counter() for k in DISTANCE_BINS},
        "occupancy_evidence": {
            "same_class_occ_in_exact_t0_box": _counter(),
            "same_class_occ_only_after_0p5m_margin": _counter(),
            "no_same_class_occ_in_0p5m_box": _counter(),
            "t0_box_outside_grid": _counter(),
        },
        "boundary": {
            "touches_xy_boundary": _counter(),
            "not_on_xy_boundary": _counter(),
        },
    }


def _merge_counter(dst, src):
    for k in dst:
        dst[k] += int(src[k])


def _merge_state(dst, src):
    for section in dst:
        for key in dst[section]:
            _merge_counter(dst[section][key], src[section][key])


def main():
    p = argparse.ArgumentParser()
    add_config_args(p)
    p.add_argument("--dataroot", required=True)
    p.add_argument("--info-pkl", required=True)
    p.add_argument("--max-windows", type=int, default=128)
    p.add_argument("--scene-seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--match-max-distance-m", type=float, default=4.0)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    if a.max_windows <= 0:
        raise ValueError("max-windows must be positive")
    if a.match_max_distance_m <= 0:
        raise ValueError("match-max-distance-m must be positive")

    cfg = load_runtime_config(a.config, a.override)
    pcfg = make_prepare_config(cfg)
    source = NuScenesWindowSource(a.dataroot, info_pkl=a.info_pkl, verbose=False)
    windows = select_scene_disjoint_windows(
        source,
        history=pcfg.history_frames,
        future=pcfg.future_frames,
        max_windows=a.max_windows,
        seed=a.scene_seed,
    )
    if not windows:
        raise RuntimeError("no scene-disjoint windows selected")
    if len({w.scene_name for w in windows}) != len(windows):
        raise AssertionError("C-source audit requires one window per scene")

    metric_grid = _grid_spec(pcfg.grid)
    per_h = {str(h): _new_horizon_state() for h in REPORT_INDICES}
    aggregate = _new_horizon_state()
    invariant = {
        "moving_records_checked": 0,
        "C_records_checked": 0,
        "C_records_missing_t0_annotation": 0,
        "C_records_that_are_true_births_under_record_contract": 0,
    }
    examples = []

    for wi, w in enumerate(windows):
        raw = load_nuscenes_window_raw(source, w, pcfg, include_gt=True)
        base = prepare_nuscenes_window(source, w, pcfg, include_gt=True, raw=raw)
        aligned_hist = ego_compensate_sequence(
            raw["history_occ"], raw["history_poses"], -1, pcfg.grid, pcfg.free_label
        )
        aligned_obs = _align_observation_sequence(
            raw["history_observed"], raw["history_poses"], -1, pcfg.grid
        )
        masks = decompose_masks(aligned_hist, pcfg.motion, history_observed=aligned_obs)
        candidates = extract_msp_candidates(
            aligned_hist, masks.moving, masks.uncertain, pcfg.grid, pcfg.kta
        )
        t0_pose = np.asarray(raw["history_poses"][-1], dtype=np.float64)
        instances = dynamic_t0_instances(source.nusc, w.t0_token, t0_pose)
        candidate_tokens, token_to_candidate = match_candidates_to_instances(
            candidates, instances, max_distance_m=float(a.match_max_distance_m)
        )
        instance_by_token = {str(x["instance_token"]): x for x in instances}
        ann0_map = _ann_map(source.nusc, w.t0_token)
        sem0 = np.asarray(raw["history_occ"][-1])

        for horizon, fi in REPORT_INDICES.items():
            state = per_h[str(horizon)]
            gt_h = np.asarray(base["future_gt_occ"])[fi]
            support_h = np.asarray(base["generation_support_occ"])[fi].astype(bool)
            for rec in base["moving_records"][fi]:
                invariant["moving_records_checked"] += 1
                token = str(rec["instance_token"])
                if source_type_for_token(token, token_to_candidate, candidates) != SOURCE_C:
                    continue
                invariant["C_records_checked"] += 1
                ann0 = ann0_map.get(token)
                inst0 = instance_by_token.get(token)
                present = ann0 is not None and inst0 is not None
                if not present:
                    invariant["C_records_missing_t0_annotation"] += 1

                center_xy = (
                    np.asarray(inst0["center_xy_t0_m"], dtype=np.float64)
                    if inst0 is not None else np.zeros(2, dtype=np.float64)
                )
                cls = int(rec["class_id"])
                detail = classify_unmatched_instance(
                    class_id=cls,
                    center_xy_t0_m=center_xy,
                    candidates=candidates,
                    candidate_tokens=candidate_tokens,
                    match_max_distance_m=float(a.match_max_distance_m),
                    t0_annotation_present=present,
                )

                future_box = box3d_from_dict(rec["boxh_future_ego"])
                arrival = rasterize_oriented_box(future_box, metric_grid, margin=0.0)
                arrival &= gt_h == cls
                arrival_n = int(arrival.sum())
                missed_n = int((arrival & ~support_h[..., None]).sum())

                _add(state["mechanism"][detail.category], arrival_voxels=arrival_n, missed_voxels=missed_n)
                _add(state["distance_bin"][distance_bin(detail.nearest_distance_m)], arrival_voxels=arrival_n, missed_voxels=missed_n)

                if present:
                    box0 = _ann_to_t0_box(ann0, t0_pose, cls)
                    exact = rasterize_oriented_box(box0, metric_grid, margin=0.0)
                    expanded = rasterize_oriented_box(box0, metric_grid, margin=0.5)
                    in_grid = bool(expanded.any())
                    exact_occ = int(((sem0 == cls) & exact).sum()) if in_grid else 0
                    expanded_occ = int(((sem0 == cls) & expanded).sum()) if in_grid else 0
                    if not in_grid:
                        occ_key = "t0_box_outside_grid"
                    elif exact_occ > 0:
                        occ_key = "same_class_occ_in_exact_t0_box"
                    elif expanded_occ > 0:
                        occ_key = "same_class_occ_only_after_0p5m_margin"
                    else:
                        occ_key = "no_same_class_occ_in_0p5m_box"
                    boundary_key = (
                        "touches_xy_boundary" if mask_touches_xy_boundary(expanded)
                        else "not_on_xy_boundary"
                    )
                else:
                    occ_key = "t0_box_outside_grid"
                    boundary_key = "not_on_xy_boundary"
                _add(state["occupancy_evidence"][occ_key], arrival_voxels=arrival_n, missed_voxels=missed_n)
                _add(state["boundary"][boundary_key], arrival_voxels=arrival_n, missed_voxels=missed_n)

                if len(examples) < 40 and missed_n > 0:
                    examples.append({
                        "scene": w.scene_name,
                        "horizon_s": float(horizon),
                        "instance_token": token,
                        "class_id": cls,
                        "mechanism": detail.category,
                        "nearest_distance_m": detail.nearest_distance_m,
                        "nearest_assigned_token": detail.nearest_assigned_token,
                        "occupancy_evidence": occ_key,
                        "touches_xy_boundary": boundary_key == "touches_xy_boundary",
                        "arrival_voxels": arrival_n,
                        "missed_voxels": missed_n,
                    })

        if wi % 8 == 0:
            print("processed", wi, w.scene_name, base["sample_id"], "candidates", len(candidates))

    for hstate in per_h.values():
        _merge_state(aggregate, hstate)

    # By construction gt_moving_support_for_horizon only emits common t0/future
    # dynamic instances. A non-zero missing-t0 count therefore indicates a code
    # or dataset-contract violation, not a legitimate birth category.
    invariant["C_records_that_are_true_births_under_record_contract"] = 0
    if invariant["C_records_missing_t0_annotation"] != 0:
        raise RuntimeError(
            "Moving-v2 common-instance contract violated: C record lacks t0 annotation"
        )

    def finalize_state(state):
        return {
            "mechanism": _finalize(state["mechanism"]),
            "distance_bin": _finalize(state["distance_bin"]),
            "occupancy_evidence": _finalize(state["occupancy_evidence"]),
            "boundary": _finalize(state["boundary"]),
        }

    mech = aggregate["mechanism"]
    potentially_recoverable = (
        mech[C_DISTANCE_GATE]["missed_voxels"]
        + mech[C_ONE_TO_ONE_CONFLICT]["missed_voxels"]
    )
    no_candidate = mech[C_NO_SAME_CLASS_CANDIDATE]["missed_voxels"]
    total_c_miss = sum(v["missed_voxels"] for v in mech.values())

    report = {
        "protocol": {
            "name": "p0_f05_c_source_audit_v1",
            "num_windows": len(windows),
            "num_unique_scenes": len({w.scene_name for w in windows}),
            "scene_seed": int(a.scene_seed),
            "match_max_distance_m": float(a.match_max_distance_m),
            "important_contract": (
                "Moving-v2 records are common t0/future instances; C is not a future-birth bucket"
            ),
        },
        "invariants": invariant,
        "per_horizon": {h: finalize_state(s) for h, s in per_h.items()},
        "aggregate": finalize_state(aggregate),
        "summary": {
            "C_missed_voxels": int(total_c_miss),
            "candidate_exists_but_label_association_failed_missed_voxels": int(potentially_recoverable),
            "candidate_exists_but_label_association_failed_share": float(
                potentially_recoverable / max(total_c_miss, 1)
            ),
            "no_same_class_candidate_missed_voxels": int(no_candidate),
            "no_same_class_candidate_share": float(no_candidate / max(total_c_miss, 1)),
            "interpretation": (
                "distance-gate/conflict rows still have a causal same-class MSP candidate; "
                "they are false-C for source availability and indicate supervision association limits"
            ),
        },
        "examples_first_40_missed_C": examples,
        "notes": [
            "Primary mechanism explains the GT-to-candidate association failure.",
            "Occupancy evidence is orthogonal and tests whether same-class Occ3D voxels exist inside the t0 GT box.",
            "Boundary statistics identify edge/FOV truncation rather than treating it as a birth.",
            "If distance-gate/conflict dominates, fix MSP supervision association before training or adding global queries.",
            "If no-same-class-candidate plus no-occupancy-evidence dominates, object-centric MSP has a genuine current-occupancy source limitation.",
        ],
    }

    op = Path(a.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(report, indent=2), encoding="utf-8")
    save_resolved_config(cfg, op.with_suffix(".resolved.yaml"))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
