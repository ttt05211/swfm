#!/usr/bin/env python3
"""P0-F0.6 (terminal diagnostic): history-seen vs never-seen for t0-outside C.

This is the last support diagnostic before MSP feasibility training. It only
answers whether P0-F0.5 C records whose t0 GT box lies outside the occupancy
grid had a causal same-class occupancy source in any earlier history frame.

GT instance identity/boxes are diagnostic-only. They are never MSP inputs.
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
from real_motion.msp_history_audit import (
    HISTORY_NEVER_SEEN,
    HISTORY_SEEN,
    HistoryFrameEvidence,
    summarize_history_source,
)
from real_motion.nuscenes_adapter import (
    NuScenesWindowSource,
    box3d_from_dict,
    category_to_dynamic_class,
)
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
SOURCE_CATEGORIES = (HISTORY_SEEN, HISTORY_NEVER_SEEN)


def _ann_map(nusc, sample_token):
    sample = nusc.get("sample", sample_token)
    out = {}
    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        out[str(ann["instance_token"])] = ann
    return out


def _wrap_angle(x):
    return (float(x) + math.pi) % (2.0 * math.pi) - math.pi


def _ann_to_ego_box(ann, ego_to_world, class_id):
    pose = np.asarray(ego_to_world, dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError("ego_to_world pose must be [4,4]")
    world_to_ego = np.linalg.inv(pose)
    center_world = np.asarray(ann["translation"], dtype=np.float64)
    center_ego = (world_to_ego @ np.r_[center_world, 1.0])[:3]
    yaw_world = quaternion_yaw(ann["rotation"])
    yaw_ego_world = math.atan2(float(pose[1, 0]), float(pose[0, 0]))
    w, l, h = ann["size"]
    return Box3D(
        token=str(ann["instance_token"]),
        class_id=int(class_id),
        center_xyz=tuple(float(v) for v in center_ego),
        size_lwh=(float(l), float(w), float(h)),
        yaw=_wrap_angle(yaw_world - yaw_ego_world),
    )


def _grid_spec(grid):
    return GridSpec(
        x_min=float(grid.x_min),
        y_min=float(grid.y_min),
        z_min=float(grid.z_min),
        voxel_size=tuple(float(v) for v in grid.voxel_size),
        shape_hwd=tuple(int(v) for v in grid.shape_hwd),
    )


def _box_occ_evidence(sem, ann, ego_to_world, class_id, metric_grid):
    """Return (in_grid, exact_same_class_count, margin_same_class_count)."""
    box = _ann_to_ego_box(ann, ego_to_world, class_id)
    exact = rasterize_oriented_box(box, metric_grid, margin=0.0)
    expanded = rasterize_oriented_box(box, metric_grid, margin=0.5)
    in_grid = bool(expanded.any())
    if not in_grid:
        return False, 0, 0
    sem = np.asarray(sem)
    if sem.shape != exact.shape:
        raise ValueError("semantic occupancy and rasterized GT box shape mismatch")
    exact_n = int(((sem == int(class_id)) & exact).sum())
    margin_n = int(((sem == int(class_id)) & expanded).sum())
    if margin_n < exact_n:
        raise RuntimeError("expanded GT box contains fewer class voxels than exact box")
    return True, exact_n, margin_n


def _counter():
    return {
        "records": 0,
        "records_with_missed_voxels": 0,
        "arrival_voxels": 0,
        "missed_voxels": 0,
    }


def _add(counter, arrival_n, missed_n):
    counter["records"] += 1
    counter["arrival_voxels"] += int(arrival_n)
    counter["missed_voxels"] += int(missed_n)
    if missed_n > 0:
        counter["records_with_missed_voxels"] += 1


def _new_state(history_ages):
    return {
        "source_category": {k: _counter() for k in SOURCE_CATEGORIES},
        "last_seen_age_s": {
            **{f"{float(a):.1f}": _counter() for a in history_ages},
            "never_seen": _counter(),
        },
        "seen_frame_count": {str(i): _counter() for i in range(len(history_ages) + 1)},
    }


def _merge_counter(dst, src):
    for k in dst:
        dst[k] += int(src[k])


def _merge_state(dst, src):
    for section in dst:
        for key in dst[section]:
            _merge_counter(dst[section][key], src[section][key])


def _finalize_table(table):
    total_miss = sum(int(v["missed_voxels"]) for v in table.values())
    total_records = sum(int(v["records"]) for v in table.values())
    out = {}
    for key, v in table.items():
        row = dict(v)
        row["share_of_target_missed_voxels"] = float(
            v["missed_voxels"] / max(total_miss, 1)
        )
        row["share_of_target_records"] = float(v["records"] / max(total_records, 1))
        out[key] = row
    return out


def _finalize_state(state):
    return {section: _finalize_table(table) for section, table in state.items()}


def _history_evidence_for_token(
    *,
    token,
    class_id,
    history_tokens,
    history_occ,
    history_poses,
    nusc,
    metric_grid,
    frame_dt_s,
):
    """Audit past frames only; t0 is explicitly excluded by the caller."""
    if not (len(history_tokens) == len(history_occ) == len(history_poses)):
        raise ValueError("history token/occupancy/pose lengths must match")
    if len(history_tokens) < 1:
        raise ValueError("at least one past history frame is required")
    if frame_dt_s <= 0:
        raise ValueError("frame_dt_s must be positive")

    rows = []
    total_past = len(history_tokens)
    for i, (sample_token, sem, pose) in enumerate(
        zip(history_tokens, history_occ, history_poses)
    ):
        # Caller passes oldest -> newest past frames. The newest has age=dt.
        age_s = float((total_past - i) * frame_dt_s)
        ann = _ann_map(nusc, sample_token).get(str(token))
        if ann is None:
            rows.append(HistoryFrameEvidence(age_s, False, False, 0, 0))
            continue
        mapped = category_to_dynamic_class(ann["category_name"])
        if mapped is None or int(mapped) != int(class_id):
            raise RuntimeError(
                f"history annotation class mismatch for instance {token}: "
                f"expected {class_id}, got {mapped}"
            )
        in_grid, exact_n, margin_n = _box_occ_evidence(
            sem, ann, pose, class_id, metric_grid
        )
        rows.append(
            HistoryFrameEvidence(
                age_s=age_s,
                annotation_present=True,
                box_in_grid=in_grid,
                exact_same_class_voxels=exact_n,
                margin_same_class_voxels=margin_n,
            )
        )
    return rows, summarize_history_source(rows)


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
    if pcfg.history_frames < 2:
        raise RuntimeError("P0-F0.6 requires at least one frame before t0")

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
        raise AssertionError("P0-F0.6 requires one window per scene")

    metric_grid = _grid_spec(pcfg.grid)
    past_count = pcfg.history_frames - 1
    history_ages = tuple(
        (past_count - i) * float(pcfg.frame_dt_s) for i in range(past_count)
    )
    per_h = {str(h): _new_state(history_ages) for h in REPORT_INDICES}
    aggregate = _new_state(history_ages)

    invariants = {
        "moving_records_checked": 0,
        "C_records_checked": 0,
        "C_t0_outside_records_checked": 0,
        "C_t0_outside_records_missing_t0_annotation": 0,
        "history_frames_per_record": int(past_count),
    }
    unique_target = set()
    unique_seen = set()
    unique_never = set()
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
        ann0_map = _ann_map(source.nusc, w.t0_token)
        sem0 = np.asarray(aligned_hist[-1])
        history_cache = {}

        for horizon, fi in REPORT_INDICES.items():
            state = per_h[str(horizon)]
            gt_h = np.asarray(base["future_gt_occ"])[fi]
            support_h = np.asarray(base["generation_support_occ"])[fi].astype(bool)
            for rec in base["moving_records"][fi]:
                invariants["moving_records_checked"] += 1
                token = str(rec["instance_token"])
                if source_type_for_token(token, token_to_candidate, candidates) != SOURCE_C:
                    continue
                invariants["C_records_checked"] += 1

                ann0 = ann0_map.get(token)
                if ann0 is None:
                    invariants["C_t0_outside_records_missing_t0_annotation"] += 1
                    continue
                cls = int(rec["class_id"])
                mapped0 = category_to_dynamic_class(ann0["category_name"])
                if mapped0 is None or int(mapped0) != cls:
                    raise RuntimeError(
                        f"t0 annotation class mismatch for instance {token}: "
                        f"expected {cls}, got {mapped0}"
                    )
                t0_in_grid, _, _ = _box_occ_evidence(
                    sem0, ann0, t0_pose, cls, metric_grid
                )
                if t0_in_grid:
                    continue

                invariants["C_t0_outside_records_checked"] += 1
                instance_key = (str(w.scene_name), token)
                unique_target.add(instance_key)

                if token not in history_cache:
                    rows, hist_summary = _history_evidence_for_token(
                        token=token,
                        class_id=cls,
                        history_tokens=tuple(w.history_tokens[:-1]),
                        history_occ=np.asarray(raw["history_occ"][:-1]),
                        history_poses=np.asarray(raw["history_poses"][:-1]),
                        nusc=source.nusc,
                        metric_grid=metric_grid,
                        frame_dt_s=float(pcfg.frame_dt_s),
                    )
                    history_cache[token] = (rows, hist_summary)
                rows, hist_summary = history_cache[token]

                if hist_summary.category == HISTORY_SEEN:
                    unique_seen.add(instance_key)
                elif hist_summary.category == HISTORY_NEVER_SEEN:
                    unique_never.add(instance_key)
                else:
                    raise RuntimeError(f"unknown historical-source category {hist_summary.category}")

                future_box = box3d_from_dict(rec["boxh_future_ego"])
                arrival = rasterize_oriented_box(future_box, metric_grid, margin=0.0)
                arrival &= gt_h == cls
                arrival_n = int(arrival.sum())
                missed_n = int((arrival & ~support_h[..., None]).sum())

                _add(state["source_category"][hist_summary.category], arrival_n, missed_n)
                last_key = (
                    "never_seen" if hist_summary.last_seen_age_s is None
                    else f"{float(hist_summary.last_seen_age_s):.1f}"
                )
                if last_key not in state["last_seen_age_s"]:
                    raise RuntimeError(f"unexpected last-seen age bin {last_key}")
                _add(state["last_seen_age_s"][last_key], arrival_n, missed_n)
                seen_count_key = str(int(hist_summary.seen_frame_count))
                _add(state["seen_frame_count"][seen_count_key], arrival_n, missed_n)

                if len(examples) < 40 and missed_n > 0:
                    examples.append({
                        "scene": w.scene_name,
                        "horizon_s": float(horizon),
                        "instance_token": token,
                        "class_id": cls,
                        "history_category": hist_summary.category,
                        "last_seen_age_s": hist_summary.last_seen_age_s,
                        "seen_frame_count": int(hist_summary.seen_frame_count),
                        "annotation_frame_count": int(hist_summary.annotation_frame_count),
                        "in_grid_frame_count": int(hist_summary.in_grid_frame_count),
                        "history_frames": [
                            {
                                "age_s": float(r.age_s),
                                "annotation_present": bool(r.annotation_present),
                                "box_in_grid": bool(r.box_in_grid),
                                "exact_same_class_voxels": int(r.exact_same_class_voxels),
                                "margin_same_class_voxels": int(r.margin_same_class_voxels),
                            }
                            for r in rows
                        ],
                        "arrival_voxels": arrival_n,
                        "missed_voxels": missed_n,
                    })

        if wi % 8 == 0:
            print(
                "processed", wi, w.scene_name, base["sample_id"],
                "candidates", len(candidates),
            )

    for hstate in per_h.values():
        _merge_state(aggregate, hstate)

    if invariants["C_t0_outside_records_missing_t0_annotation"] != 0:
        raise RuntimeError(
            "Moving-v2 common-instance contract violated: t0-outside C record lacks t0 annotation"
        )
    if invariants["C_t0_outside_records_checked"] <= 0:
        raise RuntimeError("no t0-outside C records found; protocol/input mismatch")
    if unique_seen & unique_never:
        raise RuntimeError("same target instance classified both history-seen and never-seen")
    if (unique_seen | unique_never) != unique_target:
        raise RuntimeError("historical-source classification did not cover every target instance")

    source_table = aggregate["source_category"]
    total_miss = sum(v["missed_voxels"] for v in source_table.values())
    seen_miss = source_table[HISTORY_SEEN]["missed_voxels"]
    never_miss = source_table[HISTORY_NEVER_SEEN]["missed_voxels"]

    report = {
        "protocol": {
            "name": "p0_f06_terminal_history_source_audit_v1",
            "terminal_diagnostic": True,
            "num_windows": len(windows),
            "num_unique_scenes": len({w.scene_name for w in windows}),
            "scene_seed": int(a.scene_seed),
            "target": "P0-F0 C records whose t0 GT box is outside the current occupancy grid",
            "history_source_definition": (
                "same instance has >=1 same-class occupancy voxel inside its past GT box "
                "with 0.5m margin in any pre-t0 history frame"
            ),
            "gt_usage": "instance identity/boxes are diagnostic-only and never MSP input",
            "next_step_frozen": (
                "after this audit, freeze current-vs-temporal-track candidate definition and "
                "run 1024-train/128-val MSP feasibility training; do not add another P0 diagnostic"
            ),
        },
        "invariants": invariants,
        "unique_instances": {
            "target": len(unique_target),
            "history_seen": len(unique_seen),
            "history_never_seen": len(unique_never),
        },
        "aggregate": _finalize_state(aggregate),
        "per_horizon": {h: _finalize_state(s) for h, s in per_h.items()},
        "summary": {
            "target_t0_outside_C_missed_voxels": int(total_miss),
            "history_seen_missed_voxels": int(seen_miss),
            "history_seen_share": float(seen_miss / max(total_miss, 1)),
            "history_never_seen_missed_voxels": int(never_miss),
            "history_never_seen_share": float(never_miss / max(total_miss, 1)),
            "decision_rule": (
                "This is the final diagnostic. If history-seen is substantial, retain last-seen "
                "temporal tracks as MSP candidates; otherwise use current candidates. In either "
                "case, treat never-seen as a causal input ceiling and proceed directly to MSP training."
            ),
        },
        "examples_first_40_missed_targets": examples,
    }

    op = Path(a.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(report, indent=2), encoding="utf-8")
    save_resolved_config(cfg, op.with_suffix(".resolved.yaml"))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
