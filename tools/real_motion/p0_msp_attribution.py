#!/usr/bin/env python3
"""P0-F0: attribute Hybrid support misses to causal source type A/B/C.

A = a future-moving GT instance is matched at t0 to an occupancy component that
    the real-motion decomposition marks OBSERVED_MOVING.
B = it is matched to a DORMANT motion-capable occupancy component.
C = no causal t0 occupancy component can be matched to that GT instance.

GT instance identity is used only for this diagnostic. Candidate components and
source states are derived from occupancy history exactly as they would be at
inference.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from real_motion.geometry import ego_compensate_sequence
from real_motion.metrics.moving_miou_v2 import GridSpec, rasterize_oriented_box
from real_motion.msp import (
    SOURCE_A, SOURCE_B, SOURCE_C,
    dynamic_t0_instances,
    extract_msp_candidates,
    match_candidates_to_instances,
    source_type_for_token,
)
from real_motion.nuscenes_adapter import NuScenesWindowSource, box3d_from_dict
from real_motion.prepared import (
    _align_observation_sequence,
    load_nuscenes_window_raw,
    prepare_nuscenes_window,
)
from real_motion.motion import decompose_masks
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
SOURCES = (SOURCE_A, SOURCE_B, SOURCE_C)


def _empty_row():
    return {
        "instances": 0,
        "instances_with_arrival_voxels": 0,
        "instances_with_any_miss": 0,
        "arrival_voxels": 0,
        "hit_voxels": 0,
        "missed_voxels": 0,
    }


def _grid_spec(grid):
    return GridSpec(
        x_min=float(grid.x_min), y_min=float(grid.y_min), z_min=float(grid.z_min),
        voxel_size=tuple(float(v) for v in grid.voxel_size),
        shape_hwd=tuple(int(v) for v in grid.shape_hwd),
    )


def _finalize(rows):
    out = {}
    for source, d in rows.items():
        total = int(d["arrival_voxels"])
        inst = int(d["instances_with_arrival_voxels"])
        out[source] = dict(d)
        out[source]["voxel_hit_ratio"] = float(d["hit_voxels"] / max(total, 1))
        out[source]["voxel_missed_ratio"] = float(d["missed_voxels"] / max(total, 1))
        out[source]["instance_any_miss_ratio"] = float(d["instances_with_any_miss"] / max(inst, 1))
    return out


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
        raise RuntimeError("no attribution windows selected")

    per_h = {
        str(h): {s: _empty_row() for s in SOURCES}
        for h in REPORT_INDICES
    }
    candidate_totals = {
        "observed_moving": 0,
        "dormant": 0,
        "matched": 0,
        "all": 0,
    }
    metric_grid = _grid_spec(pcfg.grid)

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
        candidate_totals["all"] += len(candidates)
        candidate_totals["matched"] += sum(t is not None for t in candidate_tokens)
        candidate_totals["observed_moving"] += sum(int(c.state) == 0 for c in candidates)
        candidate_totals["dormant"] += sum(int(c.state) == 1 for c in candidates)

        for horizon, fi in REPORT_INDICES.items():
            gt_h = np.asarray(base["future_gt_occ"])[fi]
            support_h = np.asarray(base["generation_support_occ"])[fi].astype(bool)
            records = base["moving_records"][fi]
            rows = per_h[str(horizon)]
            for rec in records:
                source_type = source_type_for_token(
                    rec["instance_token"], token_to_candidate, candidates
                )
                d = rows[source_type]
                d["instances"] += 1
                box = box3d_from_dict(rec["boxh_future_ego"])
                box_mask = rasterize_oriented_box(box, metric_grid, margin=0.0)
                arrival = box_mask & (gt_h == int(rec["class_id"]))
                n = int(arrival.sum())
                if n == 0:
                    continue
                d["instances_with_arrival_voxels"] += 1
                hit = arrival & support_h[..., None]
                hit_n = int(hit.sum())
                miss_n = n - hit_n
                d["arrival_voxels"] += n
                d["hit_voxels"] += hit_n
                d["missed_voxels"] += miss_n
                if miss_n > 0:
                    d["instances_with_any_miss"] += 1

        if wi % 8 == 0:
            print("processed", wi, w.scene_name, base["sample_id"], "candidates", len(candidates))

    report = {
        "protocol": {
            "name": "p0_f0_msp_source_attribution_v1",
            "num_windows": len(windows),
            "num_unique_scenes": len({w.scene_name for w in windows}),
            "scene_seed": int(a.scene_seed),
            "match_max_distance_m": float(a.match_max_distance_m),
            "source_definitions": {
                SOURCE_A: "future-moving GT instance matched to t0 OBSERVED_MOVING occupancy component",
                SOURCE_B: "future-moving GT instance matched to t0 DORMANT motion-capable occupancy component",
                SOURCE_C: "future-moving GT instance has no matched causal t0 occupancy component",
            },
            "gt_usage": "instance identity/boxes are diagnostic-only and never MSP input",
        },
        "candidate_summary": candidate_totals,
        "per_horizon": {h: _finalize(rows) for h, rows in per_h.items()},
        "notes": [
            "Only future-moving instances already eligible for the frozen Moving-mIoU v2 record set are attributed.",
            "Per-instance arrival voxels use the future GT box at margin=0 intersected with same-class future occupancy.",
            "Hybrid hit/miss is measured against the frozen v6 generation_support_occ.",
            "C is therefore not simply nuScenes birth count; it means the object-centric MSP has no matched causal occupancy source at t0.",
        ],
    }
    agg = {s: _empty_row() for s in SOURCES}
    for rows in per_h.values():
        for s in SOURCES:
            for key in agg[s]:
                agg[s][key] += rows[s][key]
    report["aggregate"] = _finalize(agg)

    op = Path(a.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(report, indent=2), encoding="utf-8")
    save_resolved_config(cfg, op.with_suffix(".resolved.yaml"))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
