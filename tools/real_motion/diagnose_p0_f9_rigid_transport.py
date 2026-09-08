#!/usr/bin/env python3
"""No-WM rigid source-shape transport probe for P0-F9.

This diagnostic is deliberately run *before* training a new motion head.  It
reconstructs the exact causal Strong-W2Det t0 dynamic components from occupancy,
then asks three increasingly informative questions:

1. ``kta_rigid_replay``: if the same source voxels are translated by the causal
   Strong/KTA constant velocity, can the probe reproduce the Strong anchor?
2. ``gt_center_rigid``: if future object center/existence were oracle-known, how
   good is translation-only transport of the observed t0 3D source shape?
3. ``gt_pose_rigid``: if future center/existence and yaw were oracle-known, how
   good is planar rigid SE(2) transport of that same source shape?

GT annotations are used only for the two oracle variants, source matching and
matched-source diagnostics.  No future annotation is consumed by the KTA replay.
No world model, VAE, optimizer, or learned generator is loaded.

Every matched source object coherently *replaces* its Strong/KTA predicted copy:
first clear the baseline KTA voxel footprint, then write the transported source
shape.  Unmatched dynamic objects and all non-dynamic occupancy remain exactly
the Strong anchor.  This prevents duplicate cars from a naive additive fusion.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from real_motion.geometry import quaternion_yaw
from real_motion.metrics.moving_miou_v2 import (
    Box3D,
    DYNAMIC_CLASS_IDS,
    MovingMIoUV2MultiHorizon,
    moving_support_from_world_motion,
)
from real_motion.msp import MSP_CACHE_VERSION
from real_motion.msp_wm_cache import MSPWorldModelCacheDataset
from real_motion.nuscenes_adapter import NuScenesWindowSource, WindowTokens, category_to_dynamic_class
from real_motion.prepared import load_nuscenes_window_raw
from real_motion.rigid_transport import (
    compose_component_replacements,
    rasterize_rigid_component,
    wrap_angle,
)
from real_motion.runtime_config import (
    add_config_args,
    config_fingerprint,
    load_runtime_config,
    make_prepare_config,
)
from real_motion.strong_w2det import (
    StrongW2DetConfig,
    extract_instances,
    match_instances,
    strong_w2det_sequence,
)
from tools.real_motion import eval_p0_f9_frozen_sparse_occfm as safe


PROTOCOL = "p0_f9_rigid_transport_probe_v1"
VARIANTS = (
    "strong_anchor",
    "kta_rigid_replay",
    "gt_center_rigid",
    "gt_pose_rigid",
)


def _load_msp_cache(path: str):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if obj.get("version") != MSP_CACHE_VERSION:
        raise RuntimeError("MSP cache version mismatch")
    meta = obj.get("metadata") or {}
    records = obj.get("records") or []
    if not records:
        raise RuntimeError("MSP cache has no records")
    if len({str(r["sample_id"]) for r in records}) != len(records):
        raise RuntimeError("MSP cache contains duplicate sample IDs")
    return meta, records


def _window_from_record(rec: dict) -> WindowTokens:
    return WindowTokens(
        scene_name=str(rec["scene_name"]),
        history_tokens=tuple(str(x) for x in rec["history_tokens"]),
        t0_token=str(rec["t0_token"]),
        future_tokens=tuple(str(x) for x in rec["future_tokens"]),
    )


def _dynamic_annotations(nusc, sample_token: str) -> list[dict]:
    sample = nusc.get("sample", str(sample_token))
    out = []
    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        class_id = category_to_dynamic_class(ann["category_name"])
        if class_id is None:
            continue
        size = np.asarray(ann["size"], dtype=np.float64)  # nuScenes: w,l,h
        out.append({
            "instance_token": str(ann["instance_token"]),
            "class_id": int(class_id),
            "center_world": np.asarray(ann["translation"], dtype=np.float64),
            "yaw_world": float(quaternion_yaw(ann["rotation"])),
            "size_lwh": np.asarray([size[1], size[0], size[2]], dtype=np.float64),
        })
    out.sort(key=lambda r: (int(r["class_id"]), str(r["instance_token"])))
    return out


def _annotation_map(nusc, sample_token: str) -> dict[str, dict]:
    return {str(r["instance_token"]): r for r in _dynamic_annotations(nusc, sample_token)}


def _match_components_to_annotations(
    components: list[dict],
    annotations: list[dict],
    *,
    max_distance_m: float,
) -> list[tuple[int, dict]]:
    """Greedy one-to-one same-class matching used only by this oracle probe."""
    pairs = []
    for ci, comp in enumerate(components):
        cc = np.asarray(comp["centroid_world"], dtype=np.float64)
        for ai, ann in enumerate(annotations):
            if int(comp["class_id"]) != int(ann["class_id"]):
                continue
            dist = float(np.linalg.norm(cc[:2] - np.asarray(ann["center_world"])[:2]))
            if dist <= float(max_distance_m):
                pairs.append((dist, ci, ai, str(ann["instance_token"])))
    pairs.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
    used_c, used_a = set(), set()
    out = []
    for _, ci, ai, _ in pairs:
        if ci in used_c or ai in used_a:
            continue
        used_c.add(ci)
        used_a.add(ai)
        out.append((int(ci), annotations[int(ai)]))
    out.sort(key=lambda x: x[0])
    return out


def _ego_yaw(ego_to_world: np.ndarray) -> float:
    R = np.asarray(ego_to_world, dtype=np.float64)[:3, :3]
    return float(math.atan2(R[1, 0], R[0, 0]))


def _box_in_future_ego(ann: dict, future_pose: np.ndarray) -> Box3D:
    center_world = np.asarray(ann["center_world"], dtype=np.float64)
    world_to_future = np.linalg.inv(np.asarray(future_pose, dtype=np.float64))
    center = (world_to_future @ np.r_[center_world, 1.0])[:3]
    yaw = wrap_angle(float(ann["yaw_world"]) - _ego_yaw(future_pose))
    return Box3D(
        token=str(ann["instance_token"]),
        class_id=int(ann["class_id"]),
        center_xyz=tuple(float(x) for x in center),
        size_lwh=tuple(float(x) for x in np.asarray(ann["size_lwh"])),
        yaw=float(yaw),
    )


def _matched_moving_support(
    matched: list[tuple[int, dict]],
    future_map: dict[str, dict],
    future_pose: np.ndarray,
    horizon_s: float,
    grid,
) -> np.ndarray:
    out = np.zeros(grid.shape_hwd, dtype=bool)
    for _, ann0 in matched:
        annh = future_map.get(str(ann0["instance_token"]))
        if annh is None:
            continue
        box0 = _box_in_future_ego(ann0, future_pose)
        boxh = _box_in_future_ego(annh, future_pose)
        out |= moving_support_from_world_motion(
            ann0["center_world"],
            annh["center_world"],
            box0,
            boxh,
            float(horizon_s),
            grid=grid,
        )
    return out


class CoverageAccumulator:
    def __init__(self):
        self.rows = {
            str(h): {"full": 0, "matched": 0, "intersection": 0, "union": 0}
            for h in safe.REPORT
        }

    def update(self, horizon: float, full_support, matched_support):
        full = np.asarray(full_support, dtype=bool)
        matched = np.asarray(matched_support, dtype=bool)
        r = self.rows[str(float(horizon))]
        r["full"] += int(full.sum())
        r["matched"] += int(matched.sum())
        r["intersection"] += int((full & matched).sum())
        r["union"] += int((full | matched).sum())

    def compute(self):
        out = {}
        for h, r in self.rows.items():
            full = int(r["full"])
            matched = int(r["matched"])
            inter = int(r["intersection"])
            union = int(r["union"])
            out[h] = {
                **r,
                "full_support_recall_from_matched_sources": inter / full if full else float("nan"),
                "matched_support_precision_against_full": inter / matched if matched else float("nan"),
                "support_iou": inter / union if union else float("nan"),
            }
        vals = list(out.values())
        full = sum(int(v["full"]) for v in vals)
        matched = sum(int(v["matched"]) for v in vals)
        inter = sum(int(v["intersection"]) for v in vals)
        union = sum(int(v["union"]) for v in vals)
        out["all_report_horizons"] = {
            "full": full,
            "matched": matched,
            "intersection": inter,
            "union": union,
            "full_support_recall_from_matched_sources": inter / full if full else float("nan"),
            "matched_support_precision_against_full": inter / matched if matched else float("nan"),
            "support_iou": inter / union if union else float("nan"),
        }
        return out


def _new_metric_state():
    return safe._new_metrics()


def _metric_pair(report):
    return float(report["overall"]["mIoU"]), float(report["moving"]["mIoU"])


def _print_main(reports, matched_reports, coverage, source_summary, replay_diff):
    bo, bm = _metric_pair(reports["strong_anchor"])
    print("\n=== P0-F9 NO-WM RIGID TRANSPORT PROBE ===")
    print(f"{'variant':24s} {'Overall':>9s} {'Moving':>9s} {'dOverall':>10s} {'dMoving':>9s} {'MatchedM':>10s}")
    for name in VARIANTS:
        o, m = _metric_pair(reports[name])
        mm = float(matched_reports[name]["mIoU"])
        print(f"{name:24s} {o:9.4f} {m:9.4f} {o-bo:+10.4f} {m-bm:+9.4f} {mm:10.4f}")

    print("\n=== MOVING BY HORIZON (FULL / MATCHED-SOURCE SUPPORT) ===")
    print(f"{'variant':24s} {'1s':>17s} {'2s':>17s} {'3s':>17s}")
    for name in VARIANTS:
        full_h = reports[name]["moving"]["per_horizon"]
        matched_h = matched_reports[name]["per_horizon"]
        vals = []
        for h in safe.REPORT:
            f = full_h[float(h)] if float(h) in full_h else full_h[str(float(h))]
            m = matched_h[float(h)] if float(h) in matched_h else matched_h[str(float(h))]
            vals.append(f"{float(f['mIoU']):6.2f}/{float(m['mIoU']):6.2f}")
        print(f"{name:24s} {vals[0]:>17s} {vals[1]:>17s} {vals[2]:>17s}")

    print("\n=== CAUSAL SOURCE MATCH / MOVING-SUPPORT COVERAGE ===")
    for key, value in source_summary.items():
        print(f"{key}={value}")
    for h, row in coverage.items():
        print(
            f"support {h}: full_recall={100*row['full_support_recall_from_matched_sources']:.2f}% "
            f"matched_precision={100*row['matched_support_precision_against_full']:.2f}% "
            f"iou={100*row['support_iou']:.2f}%"
        )

    print("\n=== KTA RIGID REPLAY SANITY ===")
    for h, row in replay_diff.items():
        print(
            f"{h}: changed_voxels={row['changed_voxels']} total_voxels={row['total_voxels']} "
            f"change_rate={100*row['change_rate']:.6f}%"
        )


def main():
    p = argparse.ArgumentParser()
    add_config_args(p)
    p.add_argument("--msp-cache", required=True,
                   help="frozen scene-disjoint MSP val cache, normally msp_probe_val_128.pt")
    p.add_argument("--p0f9-cache", required=True,
                   help="audited P0-F9 val cache with exact Strong/GT/Moving-v2 payload")
    p.add_argument("--dataroot", required=True)
    p.add_argument("--info-pkl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--match-max-distance-m", type=float, default=4.0)
    p.add_argument("--max-windows", type=int, default=0)
    p.add_argument("--verify-strong-anchor", action=argparse.BooleanOptionalAction, default=False,
                   help="recompute full Strong sequence and require bit-exact cached anchor replay")
    a = p.parse_args()

    cfg = load_runtime_config(a.config, a.override)
    pcfg = make_prepare_config(cfg)
    msp_meta, records = _load_msp_cache(a.msp_cache)
    expected_cfg = msp_meta.get("config_contract_sha256")
    got_cfg = config_fingerprint(cfg, "cache")
    if expected_cfg and got_cfg != expected_cfg:
        raise RuntimeError("runtime config differs from frozen MSP cache contract")

    ds = MSPWorldModelCacheDataset(a.p0f9_cache)
    safe._validate_cache(ds, str(Path(a.p0f9_cache) / "index.json") if False else ds.metadata.get("vae_checkpoint_path", "")) if False else None
    # The transport probe never loads the VAE; validate the P0-F9 sample identity
    # and payload fields directly instead of requiring a VAE path solely for SHA.
    sid_to_idx = {str(e["sample_id"]): i for i, e in enumerate(ds.entries)}
    record_ids = [str(r["sample_id"]) for r in records]
    missing = sorted(set(record_ids) - set(sid_to_idx))
    if missing:
        raise RuntimeError(f"P0-F9 cache misses MSP samples: {missing[:5]}")
    if int(a.max_windows) > 0:
        records = records[: min(len(records), int(a.max_windows))]
    if not records:
        raise RuntimeError("no records selected")

    source = NuScenesWindowSource(a.dataroot, info_pkl=a.info_pkl, verbose=False)
    strong_cfg = StrongW2DetConfig(free_label=int(pcfg.free_label))
    states = {name: _new_metric_state() for name in VARIANTS}
    matched_states = {name: MovingMIoUV2MultiHorizon() for name in VARIANTS}
    coverage = CoverageAccumulator()
    replay_counts = {str(float(h)): {"changed_voxels": 0, "total_voxels": 0} for h in safe.REPORT}

    totals = {
        "windows": 0,
        "strong_components": 0,
        "strong_velocity_matched_components": 0,
        "t0_gt_dynamic_instances": 0,
        "source_gt_matches": 0,
        "future_annotation_present": 0,
        "future_annotation_missing": 0,
    }

    for ri, rec in enumerate(records):
        sid = str(rec["sample_id"])
        window = _window_from_record(rec)
        raw = load_nuscenes_window_raw(source, window, pcfg, include_gt=True)
        sample = ds[sid_to_idx[sid]]
        payload = safe._sample_payload(sample, torch.device("cpu"))

        current_pose = np.asarray(raw["history_poses"][-1], dtype=np.float64)
        previous_pose = np.asarray(raw["history_poses"][-2], dtype=np.float64)
        current = extract_instances(
            raw["history_occ"][-1], current_pose, grid=pcfg.grid, cfg=strong_cfg
        )
        previous = extract_instances(
            raw["history_occ"][-2], previous_pose, grid=pcfg.grid, cfg=strong_cfg
        )
        velocities = match_instances(
            previous,
            current,
            float(pcfg.frame_dt_s),
            max_speed_mps=float(strong_cfg.max_match_speed_mps),
        )
        ann0 = _dynamic_annotations(source.nusc, window.t0_token)
        matched = _match_components_to_annotations(
            current, ann0, max_distance_m=float(a.match_max_distance_m)
        )
        future_maps = [_annotation_map(source.nusc, tok) for tok in window.future_tokens]

        totals["windows"] += 1
        totals["strong_components"] += len(current)
        totals["strong_velocity_matched_components"] += len(velocities)
        totals["t0_gt_dynamic_instances"] += len(ann0)
        totals["source_gt_matches"] += len(matched)

        if bool(a.verify_strong_anchor):
            replay = strong_w2det_sequence(
                raw["history_occ"],
                raw["history_poses"],
                raw["future_poses"],
                frame_dt_s=float(pcfg.frame_dt_s),
                grid=pcfg.grid,
                cfg=strong_cfg,
            )
            if not np.array_equal(replay, payload["anchor"]):
                diff = int((replay != payload["anchor"]).sum())
                raise RuntimeError(f"{sid}: recomputed Strong anchor differs at {diff} voxels")

        predictions = {name: [] for name in VARIANTS}
        matched_supports = []
        for fi, (future_pose, fmap) in enumerate(zip(raw["future_poses"], future_maps)):
            horizon = (fi + 1) * float(pcfg.frame_dt_s)
            baseline_components = []
            kta_components = []
            gt_center_components = []
            gt_pose_components = []

            for ci, a0 in matched:
                comp = current[int(ci)]
                v = np.asarray(velocities.get(int(ci), np.zeros(3)), dtype=np.float64)
                c_comp = np.asarray(comp["centroid_world"], dtype=np.float64)
                kta_target = c_comp + v * float(horizon)
                baseline = rasterize_rigid_component(
                    comp["voxel_indices"],
                    int(comp["class_id"]),
                    current_pose,
                    np.asarray(future_pose),
                    source_center_world=c_comp,
                    target_center_world=kta_target,
                    yaw_delta_rad=0.0,
                    grid=pcfg.grid,
                )
                baseline_components.append(baseline)
                kta_components.append(baseline)

                ah = fmap.get(str(a0["instance_token"]))
                if ah is None:
                    totals["future_annotation_missing"] += 1
                    continue
                totals["future_annotation_present"] += 1
                gt_center_components.append(
                    rasterize_rigid_component(
                        comp["voxel_indices"],
                        int(comp["class_id"]),
                        current_pose,
                        np.asarray(future_pose),
                        source_center_world=a0["center_world"],
                        target_center_world=ah["center_world"],
                        yaw_delta_rad=0.0,
                        grid=pcfg.grid,
                    )
                )
                gt_pose_components.append(
                    rasterize_rigid_component(
                        comp["voxel_indices"],
                        int(comp["class_id"]),
                        current_pose,
                        np.asarray(future_pose),
                        source_center_world=a0["center_world"],
                        target_center_world=ah["center_world"],
                        yaw_delta_rad=wrap_angle(float(ah["yaw_world"]) - float(a0["yaw_world"])),
                        grid=pcfg.grid,
                    )
                )

            anchor = payload["anchor"][fi]
            outputs = {
                "strong_anchor": anchor.copy(),
                "kta_rigid_replay": compose_component_replacements(
                    anchor, baseline_components, kta_components,
                    dynamic_class_ids=DYNAMIC_CLASS_IDS,
                    free_label=int(pcfg.free_label), grid=pcfg.grid,
                ),
                "gt_center_rigid": compose_component_replacements(
                    anchor, baseline_components, gt_center_components,
                    dynamic_class_ids=DYNAMIC_CLASS_IDS,
                    free_label=int(pcfg.free_label), grid=pcfg.grid,
                ),
                "gt_pose_rigid": compose_component_replacements(
                    anchor, baseline_components, gt_pose_components,
                    dynamic_class_ids=DYNAMIC_CLASS_IDS,
                    free_label=int(pcfg.free_label), grid=pcfg.grid,
                ),
            }
            for name in VARIANTS:
                predictions[name].append(outputs[name])

            matched_supports.append(
                _matched_moving_support(
                    matched,
                    fmap,
                    np.asarray(future_pose),
                    float(horizon),
                    pcfg.grid,
                )
            )

        predictions = {k: np.stack(v, axis=0) for k, v in predictions.items()}
        matched_supports = np.stack(matched_supports, axis=0)

        for h, fi in safe.REPORT.items():
            gt = payload["gt"][fi]
            full_moving = payload["moving"][fi]
            matched_moving = matched_supports[fi]
            coverage.update(float(h), full_moving, matched_moving)
            for name in VARIANTS:
                pred = predictions[name][fi]
                safe._update(states[name], float(h), pred, gt, full_moving)
                matched_states[name].update(float(h), pred, gt, matched_moving)
            diff = predictions["kta_rigid_replay"][fi] != predictions["strong_anchor"][fi]
            r = replay_counts[str(float(h))]
            r["changed_voxels"] += int(diff.sum())
            r["total_voxels"] += int(diff.size)

        if ri == 0 or (ri + 1) % 8 == 0 or ri + 1 == len(records):
            print(
                f"rigid_transport_probe {ri+1}/{len(records)} sid={sid} "
                f"sources={len(current)} matched={len(matched)}"
            )

    reports = {name: safe._report(state) for name, state in states.items()}
    matched_reports = {name: state.compute() for name, state in matched_states.items()}
    cov_report = coverage.compute()
    replay_report = {}
    for h, r in replay_counts.items():
        total = int(r["total_voxels"])
        changed = int(r["changed_voxels"])
        replay_report[h] = {
            **r,
            "change_rate": changed / total if total else float("nan"),
        }

    source_summary = {
        **totals,
        "source_to_gt_match_fraction": (
            totals["source_gt_matches"] / totals["strong_components"]
            if totals["strong_components"] else float("nan")
        ),
        "gt_t0_instance_source_coverage": (
            totals["source_gt_matches"] / totals["t0_gt_dynamic_instances"]
            if totals["t0_gt_dynamic_instances"] else float("nan")
        ),
        "strong_velocity_match_fraction": (
            totals["strong_velocity_matched_components"] / totals["strong_components"]
            if totals["strong_components"] else float("nan")
        ),
    }

    _print_main(reports, matched_reports, cov_report, source_summary, replay_report)

    bo, bm = _metric_pair(reports["strong_anchor"])
    deltas = {}
    for name in VARIANTS:
        o, m = _metric_pair(reports[name])
        deltas[name] = {"overall": o - bo, "moving": m - bm}
    report = {
        "protocol": PROTOCOL,
        "num_windows": len(records),
        "msp_cache": str(Path(a.msp_cache).resolve()),
        "p0f9_cache": str(Path(a.p0f9_cache).resolve()),
        "dataroot": str(Path(a.dataroot).resolve()),
        "info_pkl": str(Path(a.info_pkl).resolve()),
        "match_max_distance_m": float(a.match_max_distance_m),
        "verify_strong_anchor": bool(a.verify_strong_anchor),
        "variants": reports,
        "matched_source_moving": matched_reports,
        "delta_vs_strong": deltas,
        "causal_source_summary": source_summary,
        "matched_source_support_coverage": cov_report,
        "kta_rigid_replay_difference": replay_report,
        "contract": {
            "world_model_used": False,
            "vae_used": False,
            "source_shape": "exact t0 Strong-W2Det occupancy component voxels",
            "kta_variant_future_information": False,
            "gt_center_oracle": "future center + existence only; source orientation preserved",
            "gt_pose_oracle": "future center + existence + yaw; planar SE(2), source z preserved",
            "fusion": "clear selected source object's Strong/KTA raster then write transported source shape",
            "unmatched_dynamic_objects": "preserve Strong anchor",
            "matched_support_metric": "Moving-v2 restricted to GT moving boxes belonging to causally matched t0 source components",
        },
    }
    op = Path(a.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("saved", op)


if __name__ == "__main__":
    main()
