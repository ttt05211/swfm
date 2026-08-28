#!/usr/bin/env python3
"""P0-B motion-detector audit: locate why causal WM candidates become dense.

This diagnostic deliberately runs *before* KTA, latent pooling, and window
planning.  For the same ego-aligned history it compares:

1. the current detector: voxel persistence + semantic component tracking;
2. a persistence-only counterfactual with component tracking disabled.

It also reports per-semantic-class state fractions and the largest current
components classified as MOVING by the component tracker.  No future GT is
used by the detector or by this audit.
"""
import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from real_motion.geometry import ego_compensate_sequence
from real_motion.motion import (
    decompose_masks,
    _frame_components,
    _centroid_metric,
)
from real_motion.nuscenes_adapter import NuScenesWindowSource
from real_motion.metrics.moving_miou_v2 import NUSCENES_LABELS
from real_motion.runtime_config import (
    add_config_args,
    load_runtime_config,
    make_prepare_config,
    save_resolved_config,
)

STATE_NAMES = ("confident_static", "moving", "uncertain")


def _ratio(n, d):
    return float(n / d) if d else float("nan")


def _empty_counts():
    return {k: 0 for k in STATE_NAMES} | {"occupied": 0}


def _accumulate_counts(dst, masks, current):
    occupied = current != 17
    dst["occupied"] += int(occupied.sum())
    dst["confident_static"] += int(masks.confident_static.sum())
    dst["moving"] += int(masks.moving.sum())
    dst["uncertain"] += int(masks.uncertain.sum())


def _finalize_counts(c):
    occ = c["occupied"]
    return {
        "occupied_voxels": int(occ),
        "confident_static_voxels": int(c["confident_static"]),
        "moving_voxels": int(c["moving"]),
        "uncertain_voxels": int(c["uncertain"]),
        "confident_static_over_occupied": _ratio(c["confident_static"], occ),
        "moving_over_occupied": _ratio(c["moving"], occ),
        "uncertain_over_occupied": _ratio(c["uncertain"], occ),
        "wm_candidate_over_occupied": _ratio(c["moving"] + c["uncertain"], occ),
    }


def _state_id(masks, shape):
    out = np.full(shape, -1, dtype=np.int8)
    out[masks.confident_static] = 0
    out[masks.moving] = 1
    out[masks.uncertain] = 2
    return out


def _component_rows(history_semantics, persistence, cfg, sample_id):
    """Replay the exact component-track decision while retaining diagnostics."""
    hist = np.asarray(history_semantics)
    cur = hist[-1]
    frame_comps = [
        _frame_components(x, cfg.free_label, cfg.min_component_bev_cells)
        for x in hist
    ]
    rows = []
    for cls, current_list in frame_comps[-1].items():
        for cells in current_list:
            cur_c = _centroid_metric(cells, cfg)
            ref = cur_c
            track = [cur_c]
            for ti in range(len(hist) - 2, -1, -1):
                candidates = frame_comps[ti].get(cls, [])
                if not candidates:
                    break
                centroids = [_centroid_metric(c, cfg) for c in candidates]
                distances = np.asarray([np.linalg.norm(c - ref) for c in centroids])
                j = int(distances.argmin())
                if float(distances[j]) > cfg.component_max_step_m:
                    break
                ref = centroids[j]
                track.append(ref)

            intervals = len(track) - 1
            speed = float("nan")
            if intervals > 0:
                speed = float(
                    np.linalg.norm(track[0] - track[-1])
                    / (intervals * cfg.history_dt_s)
                )

            cell_mask = np.zeros(cur.shape[:2], dtype=bool)
            cell_mask[cells[:, 0], cells[:, 1]] = True
            vox = (cur == cls) & cell_mask[:, :, None]
            mean_p = float(persistence[vox].mean()) if vox.any() else 0.0

            if len(track) >= cfg.min_track_frames and speed >= cfg.moving_speed_mps:
                state = "moving"
                reason = "component_speed>=moving_threshold"
            elif (
                len(track) >= cfg.min_track_frames
                and speed <= cfg.static_speed_mps
                and mean_p >= cfg.static_min_persistence
            ):
                state = "confident_static"
                reason = "slow_component_and_high_persistence"
            else:
                state = "uncertain"
                reason = "otherwise"

            rows.append({
                "sample_id": sample_id,
                "class_id": int(cls),
                "class_name": (
                    NUSCENES_LABELS[int(cls)]
                    if 0 <= int(cls) < len(NUSCENES_LABELS)
                    else str(int(cls))
                ),
                "state": state,
                "reason": reason,
                "bev_cells": int(len(cells)),
                "voxel_count": int(vox.sum()),
                "track_frames": int(len(track)),
                "estimated_speed_mps": speed,
                "mean_voxel_persistence": mean_p,
            })
    return rows


def main():
    p = argparse.ArgumentParser()
    add_config_args(p)
    p.add_argument("--dataroot", required=True)
    p.add_argument("--info-pkl", required=True)
    p.add_argument("--max-windows", type=int, default=16)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--top-components", type=int, default=30)
    p.add_argument("--output", required=True)
    a = p.parse_args()

    cfg = load_runtime_config(a.config, a.override)
    pcfg = make_prepare_config(cfg)
    tracked_cfg = pcfg.motion
    persistence_cfg = replace(tracked_cfg, use_component_tracks=False)

    source = NuScenesWindowSource(a.dataroot, info_pkl=a.info_pkl, verbose=False)

    totals = {
        "component_tracking": _empty_counts(),
        "persistence_only": _empty_counts(),
    }
    per_class = {
        c: {
            "component_tracking": _empty_counts(),
            "persistence_only": _empty_counts(),
            "transitions": np.zeros((3, 3), dtype=np.int64),
        }
        for c in range(17)
    }
    components = []
    num_windows = 0

    for i, w in enumerate(source.iter_windows(
        history=pcfg.history_frames,
        future=pcfg.future_frames,
        stride=a.stride,
        max_windows=a.max_windows,
    )):
        hist = np.stack([
            source.load_semantics(w.scene_name, token)
            for token in w.history_tokens
        ])
        poses = [source.pose(token) for token in w.history_tokens]
        aligned = ego_compensate_sequence(
            hist, poses, -1, pcfg.grid, pcfg.free_label
        )
        current = aligned[-1]

        tracked = decompose_masks(aligned, tracked_cfg)
        persistence_only = decompose_masks(aligned, persistence_cfg)

        _accumulate_counts(totals["component_tracking"], tracked, current)
        _accumulate_counts(totals["persistence_only"], persistence_only, current)

        tracked_state = _state_id(tracked, current.shape)
        persistence_state = _state_id(persistence_only, current.shape)

        for cls in range(17):
            cm = current == cls
            if not cm.any():
                continue
            for name, masks in (
                ("component_tracking", tracked),
                ("persistence_only", persistence_only),
            ):
                d = per_class[cls][name]
                d["occupied"] += int(cm.sum())
                d["confident_static"] += int((cm & masks.confident_static).sum())
                d["moving"] += int((cm & masks.moving).sum())
                d["uncertain"] += int((cm & masks.uncertain).sum())

            trans = per_class[cls]["transitions"]
            for ps in range(3):
                for ts in range(3):
                    trans[ps, ts] += int(
                        (cm & (persistence_state == ps) & (tracked_state == ts)).sum()
                    )

        components.extend(_component_rows(
            aligned,
            tracked.persistence,
            tracked_cfg,
            f"{w.scene_name}:{w.t0_token}",
        ))
        num_windows += 1
        if i % 8 == 0:
            print("audited", i, w.scene_name, w.t0_token)

    class_rows = []
    state_label = {0: "confident_static", 1: "moving", 2: "uncertain"}
    for cls in range(17):
        td = per_class[cls]
        occ = td["component_tracking"]["occupied"]
        if occ == 0:
            continue
        transitions = {}
        for ps in range(3):
            for ts in range(3):
                n = int(td["transitions"][ps, ts])
                if n:
                    transitions[f"{state_label[ps]}->{state_label[ts]}"] = {
                        "voxels": n,
                        "over_class_occupied": _ratio(n, occ),
                    }
        tracked_final = _finalize_counts(td["component_tracking"])
        persistence_final = _finalize_counts(td["persistence_only"])
        class_rows.append({
            "class_id": cls,
            "class_name": NUSCENES_LABELS[cls],
            "component_tracking": tracked_final,
            "persistence_only": persistence_final,
            "moving_delta_pp_component_minus_persistence": 100.0 * (
                tracked_final["moving_over_occupied"]
                - persistence_final["moving_over_occupied"]
            ),
            "transitions": transitions,
        })

    moving_components = [r for r in components if r["state"] == "moving"]
    moving_components.sort(key=lambda r: r["voxel_count"], reverse=True)

    moving_component_by_class = {}
    for r in moving_components:
        k = r["class_name"]
        d = moving_component_by_class.setdefault(k, {
            "class_id": r["class_id"],
            "num_components": 0,
            "total_voxels": 0,
            "total_bev_cells": 0,
            "max_component_voxels": 0,
            "max_estimated_speed_mps": float("nan"),
        })
        d["num_components"] += 1
        d["total_voxels"] += r["voxel_count"]
        d["total_bev_cells"] += r["bev_cells"]
        d["max_component_voxels"] = max(d["max_component_voxels"], r["voxel_count"])
        sp = r["estimated_speed_mps"]
        if np.isfinite(sp):
            old = d["max_estimated_speed_mps"]
            d["max_estimated_speed_mps"] = sp if not np.isfinite(old) else max(old, sp)

    tracked_total = _finalize_counts(totals["component_tracking"])
    persistence_total = _finalize_counts(totals["persistence_only"])
    report = {
        "protocol": "P0-B_motion_detector_audit_v1",
        "num_windows": num_windows,
        "causal": True,
        "uses_future_gt": False,
        "thresholds": {
            "static_min_persistence": tracked_cfg.static_min_persistence,
            "moving_max_persistence": tracked_cfg.moving_max_persistence,
            "component_moving_speed_mps": tracked_cfg.moving_speed_mps,
            "component_static_speed_mps": tracked_cfg.static_speed_mps,
            "component_max_step_m": tracked_cfg.component_max_step_m,
            "min_track_frames": tracked_cfg.min_track_frames,
        },
        "global_comparison": {
            "component_tracking": tracked_total,
            "persistence_only": persistence_total,
            "moving_delta_pp_component_minus_persistence": 100.0 * (
                tracked_total["moving_over_occupied"]
                - persistence_total["moving_over_occupied"]
            ),
            "wm_candidate_delta_pp_component_minus_persistence": 100.0 * (
                tracked_total["wm_candidate_over_occupied"]
                - persistence_total["wm_candidate_over_occupied"]
            ),
        },
        "per_class": class_rows,
        "moving_component_summary_by_class": sorted(
            moving_component_by_class.values(),
            key=lambda d: d["total_voxels"],
            reverse=True,
        ),
        "largest_moving_components": moving_components[:max(0, a.top_components)],
        "interpretation": (
            "If persistence-only is sparse but component-tracking is dense, the component "
            "promotion logic is the inflation source. Per-class transitions and largest "
            "moving components show which semantics/components cause it. If both are dense, "
            "inspect ego alignment and the voxel-persistence definition before changing KTA."
        ),
    }

    op = Path(a.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(report, indent=2, allow_nan=True), encoding="utf-8")
    save_resolved_config(cfg, op.with_suffix(".resolved.yaml"))

    print(json.dumps({
        "saved": str(op),
        "num_windows": num_windows,
        "component_tracking": tracked_total,
        "persistence_only": persistence_total,
        "moving_delta_pp_component_minus_persistence": report["global_comparison"]["moving_delta_pp_component_minus_persistence"],
        "top_moving_classes": report["moving_component_summary_by_class"][:8],
    }, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
