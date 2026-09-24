#!/usr/bin/env python3
"""Long-horizon transportability diagnostic for 4D occupancy forecasting.

This is an offline *analysis*, not a forecasting method.  It asks how much of
future occupancy at 1--6 s can in principle be explained by transporting
geometry that is already present at t0.

For every 6-history + 12-future window:
  1. static/non-motion-capable t0 occupancy is transported only by the known
     ego-frame transform (same inverse warp + majority fill as Strong);
  2. t0 motion-capable semantic connected components are extracted by the
     frozen Strong source extractor;
  3. sources are matched to t0 GT instance annotations *only for this offline
     oracle diagnostic*;
  4. matched sources are transported by the GT box motion to each future
     horizon while preserving the exact observed t0 source voxels;
  5. the resulting transport oracle is compared with future GT occupancy.

Therefore, "unexplained" means "not covered by this exact observed-geometry
transport oracle".  It includes genuinely new/unseen content, geometry that was
unobserved at t0 and becomes visible later, source-extraction misses, and any
non-rigid/annotation mismatch.  It must not be described as a pure birth rate.

The script additionally reports GT-box-supported dynamic occupancy associated
with (a) represented t0 sources, (b) t0 dynamic instances not represented by a
source, and (c) future dynamic instances absent at t0.  Those support fractions
help separate new-object appearance from other transport-oracle failures.

No future GT is exposed to any deployable model path; future annotations are
used only to construct this diagnostic oracle.
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

from real_motion.geometry import relative_transform, quaternion_yaw
from real_motion.local_st_world_model_v18_se2 import YAW_ENABLED_CLASS_IDS
from real_motion.metrics.moving_miou_v2 import (
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
)
from real_motion.rigid_transport import rasterize_rigid_component
from real_motion.runtime_config import (
    add_config_args,
    load_runtime_config,
    make_prepare_config,
)
from real_motion.strong_w2det import (
    StrongW2DetConfig,
    extract_instances,
    inverse_warp,
    majority_fill,
)

PROTOCOL = "p0_f9_long_horizon_transportability_v1"


def _wrap(x: float) -> float:
    return float((float(x) + math.pi) % (2.0 * math.pi) - math.pi)


def _ego_yaw(T: np.ndarray) -> float:
    R = np.asarray(T, dtype=np.float64)[:3, :3]
    return math.atan2(float(R[1, 0]), float(R[0, 0]))


def _ann_map(nusc, sample_token: str) -> dict[str, dict]:
    sample = nusc.get("sample", str(sample_token))
    out: dict[str, dict] = {}
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
            # nuScenes stores [w,l,h]; internal metric uses [l,w,h].
            "size_lwh": np.asarray([size[1], size[0], size[2]], dtype=np.float64),
        }
    return out


def _box_future_ego(ann: dict, future_pose: np.ndarray) -> Box3D:
    T = np.linalg.inv(np.asarray(future_pose, dtype=np.float64))
    c = (T @ np.r_[np.asarray(ann["center_world"], dtype=np.float64), 1.0])[:3]
    return Box3D(
        token=str(ann["instance_token"]),
        class_id=int(ann["class_id"]),
        center_xyz=tuple(float(x) for x in c),
        size_lwh=tuple(float(x) for x in ann["size_lwh"]),
        yaw=_wrap(float(ann["yaw_world"]) - _ego_yaw(future_pose)),
    )


def _source_center_target(
    source_center_world: np.ndarray,
    ann0: dict,
    annh: dict,
    *,
    yaw_enabled: bool,
) -> tuple[np.ndarray, float]:
    """GT box motion expressed as a source-centred planar rigid transform."""
    cs = np.asarray(source_center_world, dtype=np.float64)
    a0 = np.asarray(ann0["center_world"], dtype=np.float64)
    ah = np.asarray(annh["center_world"], dtype=np.float64)
    dyaw = _wrap(float(annh["yaw_world"]) - float(ann0["yaw_world"])) if yaw_enabled else 0.0
    c, s = math.cos(dyaw), math.sin(dyaw)
    R = np.asarray([[c, -s], [s, c]], dtype=np.float64)
    off = cs[:2] - a0[:2]
    ds = (ah[:2] - a0[:2]) + (R @ off - off)
    target = cs.copy()
    target[:2] = cs[:2] + ds
    # Frozen V18 transport is planar; preserve observed source z.
    target[2] = cs[2]
    return target, dyaw


def _write_component(out: np.ndarray, comp) -> None:
    idx = np.asarray(comp.voxel_indices, dtype=np.int64)
    if len(idx):
        out[idx[:, 0], idx[:, 1], idx[:, 2]] = int(comp.class_id)


def _ratio(num: int, den: int) -> float:
    return float(num) / float(den) if den else float("nan")


def _new_semantic_counts():
    return {
        int(c): {"inter": 0, "union": 0}
        for c in range(17)
    }


def _update_semantic_counts(counts, pred, gt):
    for c in range(17):
        p = pred == c
        g = gt == c
        counts[c]["inter"] += int(np.logical_and(p, g).sum())
        counts[c]["union"] += int(np.logical_or(p, g).sum())


def _semantic_miou(counts) -> float:
    vals = []
    for c in range(17):
        u = int(counts[c]["union"])
        if u:
            vals.append(float(counts[c]["inter"]) / float(u))
    return 100.0 * float(np.mean(vals)) if vals else float("nan")


def _mask_stats_init():
    return {
        "gt_occupied": 0,
        "covered_occupied": 0,
        "semantic_correct_occupied": 0,
        "gt_static_occupied": 0,
        "covered_static_occupied": 0,
        "gt_dynamic_occupied": 0,
        "covered_dynamic_occupied": 0,
        "dynamic_in_represented_source_box": 0,
        "dynamic_in_unrepresented_t0_instance_box": 0,
        "dynamic_in_birth_instance_box": 0,
        "dynamic_box_support_overlap_voxels": 0,
        "future_dynamic_instances": 0,
        "represented_existing_instances": 0,
        "unrepresented_existing_instances": 0,
        "birth_instances": 0,
        "t0_sources": 0,
        "t0_sources_matched_to_gt": 0,
    }


def _finalize_row(raw, sem_counts):
    gt_occ = int(raw["gt_occupied"])
    gt_static = int(raw["gt_static_occupied"])
    gt_dyn = int(raw["gt_dynamic_occupied"])
    out = dict(raw)
    out.update({
        "transport_occupied_recall": _ratio(raw["covered_occupied"], gt_occ),
        "transport_unexplained_fraction": (
            1.0 - _ratio(raw["covered_occupied"], gt_occ) if gt_occ else float("nan")
        ),
        "transport_semantic_recall_on_gt_occupied": _ratio(
            raw["semantic_correct_occupied"], gt_occ
        ),
        "static_transport_recall": _ratio(raw["covered_static_occupied"], gt_static),
        "dynamic_transport_recall": _ratio(raw["covered_dynamic_occupied"], gt_dyn),
        "dynamic_gt_fraction_in_represented_source_box": _ratio(
            raw["dynamic_in_represented_source_box"], gt_dyn
        ),
        "dynamic_gt_fraction_in_unrepresented_t0_instance_box": _ratio(
            raw["dynamic_in_unrepresented_t0_instance_box"], gt_dyn
        ),
        "dynamic_gt_fraction_in_birth_instance_box": _ratio(
            raw["dynamic_in_birth_instance_box"], gt_dyn
        ),
        "transport_oracle_mIoU": _semantic_miou(sem_counts),
    })
    return out


def main():
    p = argparse.ArgumentParser()
    add_config_args(p)
    p.add_argument("--dataroot", required=True)
    p.add_argument("--info-pkl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--future-frames", type=int, default=12)
    p.add_argument(
        "--report-horizons",
        default="1,2,3,4,5,6",
        help="seconds, comma-separated; must align to the dataset frame interval",
    )
    p.add_argument("--match-max-distance-m", type=float, default=4.0)
    p.add_argument("--max-windows", type=int, default=0)
    a = p.parse_args()

    cfg = load_runtime_config(a.config, a.override)
    pcfg = make_prepare_config(cfg)
    frame_dt = float(pcfg.frame_dt_s)
    future_frames = int(a.future_frames)
    if future_frames <= 0:
        raise ValueError("--future-frames must be positive")

    horizons = tuple(float(x.strip()) for x in str(a.report_horizons).split(",") if x.strip())
    if not horizons:
        raise ValueError("--report-horizons is empty")
    horizon_to_idx = {}
    for h in horizons:
        k = h / frame_dt
        if h <= 0 or abs(k - round(k)) > 1e-8:
            raise ValueError(f"horizon {h} does not align to frame_dt={frame_dt}")
        idx = int(round(k)) - 1
        if idx < 0 or idx >= future_frames:
            raise ValueError(f"horizon {h} needs future index {idx}, future_frames={future_frames}")
        horizon_to_idx[h] = idx

    source = NuScenesWindowSource(a.dataroot, info_pkl=a.info_pkl, verbose=False)
    strong_cfg = StrongW2DetConfig(free_label=int(pcfg.free_label))
    dyn_ids = np.asarray(DYNAMIC_CLASS_IDS, dtype=np.uint8)
    metric_grid = GridSpec(
        pcfg.grid.x_min,
        pcfg.grid.y_min,
        pcfg.grid.z_min,
        pcfg.grid.voxel_size,
        pcfg.grid.shape_hwd,
    )

    maxw = int(a.max_windows) if int(a.max_windows) > 0 else None
    windows = source.iter_windows(
        history=6,
        future=future_frames,
        stride=1,
        max_windows=maxw,
    )

    rows = {h: _mask_stats_init() for h in horizons}
    sem_counts = {h: _new_semantic_counts() for h in horizons}
    nwin = 0
    scene_names = set()
    started = time.perf_counter()

    for w in windows:
        nwin += 1
        scene_names.add(str(w.scene_name))
        history_occ = [
            np.asarray(source.load_semantics(w.scene_name, tok), dtype=np.uint8)
            for tok in w.history_tokens
        ]
        history_poses = [
            np.asarray(source.pose(tok), dtype=np.float64)
            for tok in w.history_tokens
        ]
        t0_occ = history_occ[-1]
        t0_pose = history_poses[-1]

        current = extract_instances(
            t0_occ,
            t0_pose,
            grid=pcfg.grid,
            cfg=strong_cfg,
        )
        anns0_light = dynamic_annotations(source.nusc, w.t0_token)
        source_tokens = match_sources_to_annotations(
            current,
            anns0_light,
            max_distance_m=float(a.match_max_distance_m),
        )
        ann0 = _ann_map(source.nusc, w.t0_token)
        t0_tokens = set(ann0)
        represented_tokens = {str(x) for x in source_tokens if x is not None}

        static_src = t0_occ.copy()
        static_src[np.isin(static_src, dyn_ids)] = int(pcfg.free_label)

        for h in horizons:
            hi = horizon_to_idx[h]
            ftok = str(w.future_tokens[hi])
            gt = np.asarray(source.load_semantics(w.scene_name, ftok), dtype=np.uint8)
            fpose = np.asarray(source.pose(ftok), dtype=np.float64)
            annh = _ann_map(source.nusc, ftok)

            # Exact Strong static branch: t0 non-dynamic occupancy + ego transform.
            T = relative_transform(t0_pose, fpose)
            static_pred, known = inverse_warp(
                static_src,
                T,
                pcfg.grid,
                int(pcfg.free_label),
            )
            oracle = majority_fill(
                static_pred,
                ~known,
                kernel=strong_cfg.fill_kernel,
                min_fraction=strong_cfg.fill_min_fraction,
            )

            # Offline GT-motion transport of *observed source geometry*.
            for i, comp in enumerate(current):
                token = source_tokens[i]
                if token is None:
                    continue
                token = str(token)
                a0 = ann0.get(token)
                ah = annh.get(token)
                if a0 is None or ah is None:
                    continue
                target_center, dyaw = _source_center_target(
                    np.asarray(comp["centroid_world"], dtype=np.float64),
                    a0,
                    ah,
                    yaw_enabled=int(comp["class_id"]) in set(YAW_ENABLED_CLASS_IDS),
                )
                rc = rasterize_rigid_component(
                    comp["voxel_indices"],
                    int(comp["class_id"]),
                    t0_pose,
                    fpose,
                    source_center_world=np.asarray(comp["centroid_world"], dtype=np.float64),
                    target_center_world=target_center,
                    yaw_delta_rad=float(dyaw),
                    grid=pcfg.grid,
                )
                _write_component(oracle, rc)

            gt_occ = gt != int(pcfg.free_label)
            pr_occ = oracle != int(pcfg.free_label)
            gt_dyn = gt_occ & np.isin(gt, dyn_ids)
            gt_static = gt_occ & ~np.isin(gt, dyn_ids)

            r = rows[h]
            r["gt_occupied"] += int(gt_occ.sum())
            r["covered_occupied"] += int((gt_occ & pr_occ).sum())
            r["semantic_correct_occupied"] += int((gt_occ & (oracle == gt)).sum())
            r["gt_static_occupied"] += int(gt_static.sum())
            r["covered_static_occupied"] += int((gt_static & pr_occ).sum())
            r["gt_dynamic_occupied"] += int(gt_dyn.sum())
            r["covered_dynamic_occupied"] += int((gt_dyn & pr_occ).sum())
            r["t0_sources"] += int(len(current))
            r["t0_sources_matched_to_gt"] += int(len(represented_tokens))

            # Attribute future dynamic GT approximately with future GT boxes.
            masks = {
                "represented": np.zeros(gt.shape, dtype=bool),
                "unrepresented": np.zeros(gt.shape, dtype=bool),
                "birth": np.zeros(gt.shape, dtype=bool),
            }
            for tok, ah in annh.items():
                if tok in represented_tokens:
                    group = "represented"
                    r["represented_existing_instances"] += 1
                elif tok in t0_tokens:
                    group = "unrepresented"
                    r["unrepresented_existing_instances"] += 1
                else:
                    group = "birth"
                    r["birth_instances"] += 1
                r["future_dynamic_instances"] += 1
                box_mask = rasterize_oriented_box(
                    _box_future_ego(ah, fpose),
                    metric_grid,
                    margin=0.0,
                )
                # Restrict attribution to voxels with the same semantic class.
                masks[group] |= box_mask & (gt == int(ah["class_id"]))

            r["dynamic_in_represented_source_box"] += int((gt_dyn & masks["represented"]).sum())
            r["dynamic_in_unrepresented_t0_instance_box"] += int((gt_dyn & masks["unrepresented"]).sum())
            r["dynamic_in_birth_instance_box"] += int((gt_dyn & masks["birth"]).sum())
            overlap = (
                masks["represented"].astype(np.uint8)
                + masks["unrepresented"].astype(np.uint8)
                + masks["birth"].astype(np.uint8)
            ) > 1
            r["dynamic_box_support_overlap_voxels"] += int((gt_dyn & overlap).sum())

            _update_semantic_counts(sem_counts[h], oracle, gt)

        if nwin == 1 or nwin % 25 == 0:
            elapsed = max(time.perf_counter() - started, 1e-9)
            print(
                f"transportability {nwin} windows "
                f"scenes={len(scene_names)} rate={nwin/elapsed:.3f} win/s",
                flush=True,
            )

    if nwin == 0:
        raise RuntimeError(
            "no eligible 6-history + future windows; check dataroot/info-pkl/future-frames"
        )

    report = {
        "protocol": PROTOCOL,
        "analysis_only": True,
        "future_gt_used_for_prediction": False,
        "future_gt_used_for_offline_oracle": True,
        "history_frames": 6,
        "future_frames_required": future_frames,
        "frame_dt_s": frame_dt,
        "report_horizons_s": list(horizons),
        "num_windows": int(nwin),
        "num_scenes": int(len(scene_names)),
        "population_contract": (
            "same complete-window population requiring all requested future frames; "
            "horizon comparisons therefore do not change the window set"
        ),
        "source_contract": (
            "t0 motion-capable same-class 3D connected components from Strong; "
            "GT instance matching is offline diagnostic-only"
        ),
        "transport_oracle_contract": (
            "Strong static ego transport + GT-motion source-centred rigid transport "
            "of exact observed t0 source voxels"
        ),
        "unexplained_contract": (
            "GT occupied voxels not covered by the transport oracle; includes genuine "
            "new/unseen content, newly revealed geometry, extraction misses, and "
            "rigid-transport mismatch, so it is not a pure birth metric"
        ),
        "per_horizon": {
            str(h): _finalize_row(rows[h], sem_counts[h])
            for h in horizons
        },
    }
    op = Path(a.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== LONG-HORIZON TRANSPORTABILITY DIAGNOSTIC ===")
    print(f"windows={nwin} scenes={len(scene_names)}")
    print(
        f"{'horizon':>8s} {'occ_recall':>11s} {'unexpl':>10s} "
        f"{'static':>10s} {'dynamic':>10s} {'birth_dyn':>10s} {'mIoU':>9s}"
    )
    for h in horizons:
        x = report["per_horizon"][str(h)]
        print(
            f"{h:8.1f} "
            f"{100*x['transport_occupied_recall']:11.3f} "
            f"{100*x['transport_unexplained_fraction']:10.3f} "
            f"{100*x['static_transport_recall']:10.3f} "
            f"{100*x['dynamic_transport_recall']:10.3f} "
            f"{100*x['dynamic_gt_fraction_in_birth_instance_box']:10.3f} "
            f"{x['transport_oracle_mIoU']:9.3f}"
        )
    print(f"saved {op}")


if __name__ == "__main__":
    main()
