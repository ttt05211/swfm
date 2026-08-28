#!/usr/bin/env python3
"""P0-B follow-up: isolate alignment vs persistence inflation.

This diagnostic does not change the formal motion detector.  It compares
several causal counterfactuals on the same six-frame history:

- raw (unaligned), ego-pose aligned, and LIDAR_TOP-pose aligned same-class
  agreement with the current frame;
- strict exact-cell persistence (the current persistence-only behaviour);
- observed-only persistence, which does not count warp-created free holes in
  the denominator;
- radius-1 XY tolerant persistence, which measures sensitivity to one-cell
  rasterization/quantization shifts.

No future occupancy or future annotation is used.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from real_motion.geometry import ego_compensate_sequence, pose_matrix
from real_motion.nuscenes_adapter import NuScenesWindowSource
from real_motion.metrics.moving_miou_v2 import NUSCENES_LABELS
from real_motion.runtime_config import (
    add_config_args,
    load_runtime_config,
    make_prepare_config,
    save_resolved_config,
)


def _ratio(n, d):
    return float(n / d) if d else float("nan")


def _sample_lidar_to_world(source, token):
    """Return LIDAR_TOP -> world for one nuScenes keyframe."""
    nusc = source.nusc
    sample = nusc.get("sample", token)
    sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    ego = nusc.get("ego_pose", sd["ego_pose_token"])
    calib = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    ego_to_world = pose_matrix(ego["translation"], ego["rotation"])
    lidar_to_ego = pose_matrix(calib["translation"], calib["rotation"])
    return ego_to_world @ lidar_to_ego


def _shifted_equal(past, current, radius):
    """For each current voxel, same class exists within XY Chebyshev radius."""
    if radius <= 0:
        return past == current
    H, W, _ = current.shape
    out = np.zeros_like(current, dtype=bool)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            rolled = np.roll(past, shift=(-dy, -dx), axis=(0, 1))
            comp = rolled == current
            if dy > 0:
                comp[H - dy:, :, :] = False
            elif dy < 0:
                comp[: -dy, :, :] = False
            if dx > 0:
                comp[:, W - dx:, :] = False
            elif dx < 0:
                comp[:, : -dx, :] = False
            out |= comp
    return out


def _shifted_nonfree(past, free_label, radius):
    if radius <= 0:
        return past != free_label
    H, W, _ = past.shape
    out = np.zeros_like(past, dtype=bool)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            rolled = np.roll(past != free_label, shift=(-dy, -dx), axis=(0, 1))
            if dy > 0:
                rolled[H - dy:, :, :] = False
            elif dy < 0:
                rolled[: -dy, :, :] = False
            if dx > 0:
                rolled[:, W - dx:, :] = False
            elif dx < 0:
                rolled[:, : -dx, :] = False
            out |= rolled
    return out


def _persistence(hist, free_label, observed_only=False, radius=0):
    cur = hist[-1]
    matches = []
    observed = []
    for frame in hist:
        matches.append(_shifted_equal(frame, cur, radius))
        observed.append(_shifted_nonfree(frame, free_label, radius))
    matches = np.stack(matches, axis=0)
    observed = np.stack(observed, axis=0)
    if observed_only:
        denom = np.maximum(observed.sum(axis=0), 1)
        return ((matches & observed).sum(axis=0) / denom).astype(np.float32)
    return matches.mean(axis=0, dtype=np.float64).astype(np.float32)


def _empty_state():
    return {"occupied": 0, "static": 0, "moving": 0, "uncertain": 0}


def _acc_state(dst, current, persistence, free_label, static_thr, moving_thr, class_id=None):
    occ = current != free_label
    if class_id is not None:
        occ &= current == int(class_id)
    sta = occ & (persistence >= static_thr)
    mov = occ & (persistence <= moving_thr)
    unc = occ & ~(sta | mov)
    dst["occupied"] += int(occ.sum())
    dst["static"] += int(sta.sum())
    dst["moving"] += int(mov.sum())
    dst["uncertain"] += int(unc.sum())


def _final_state(c):
    occ = c["occupied"]
    return {
        "occupied_voxels": int(occ),
        "confident_static_over_occupied": _ratio(c["static"], occ),
        "moving_over_occupied": _ratio(c["moving"], occ),
        "uncertain_over_occupied": _ratio(c["uncertain"], occ),
        "wm_candidate_over_occupied": _ratio(c["moving"] + c["uncertain"], occ),
    }


def _empty_alignment(frames):
    return [
        {
            "current_occupied": 0,
            "same_exact": 0,
            "same_r1": 0,
            "nonfree_exact": 0,
            "nonfree_r1": 0,
        }
        for _ in range(frames - 1)
    ]


def _acc_alignment(dst, hist, free_label):
    cur = hist[-1]
    occ = cur != free_label
    for lag_idx, frame_idx in enumerate(range(len(hist) - 1)):
        past = hist[frame_idx]
        d = dst[lag_idx]
        d["current_occupied"] += int(occ.sum())
        d["same_exact"] += int((_shifted_equal(past, cur, 0) & occ).sum())
        d["same_r1"] += int((_shifted_equal(past, cur, 1) & occ).sum())
        d["nonfree_exact"] += int(((past != free_label) & occ).sum())
        d["nonfree_r1"] += int((_shifted_nonfree(past, free_label, 1) & occ).sum())


def _final_alignment(rows, history_dt_s):
    out = []
    Tpast = len(rows)
    for i, d in enumerate(rows):
        denom = d["current_occupied"]
        frames_back = Tpast - i
        out.append({
            "frames_back": int(frames_back),
            "seconds_back": float(frames_back * history_dt_s),
            "same_class_exact_over_current_occupied": _ratio(d["same_exact"], denom),
            "same_class_r1_over_current_occupied": _ratio(d["same_r1"], denom),
            "nonfree_exact_over_current_occupied": _ratio(d["nonfree_exact"], denom),
            "nonfree_r1_over_current_occupied": _ratio(d["nonfree_r1"], denom),
        })
    return out


def main():
    p = argparse.ArgumentParser()
    add_config_args(p)
    p.add_argument("--dataroot", required=True)
    p.add_argument("--info-pkl", required=True)
    p.add_argument("--max-windows", type=int, default=16)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--output", required=True)
    a = p.parse_args()

    cfg = load_runtime_config(a.config, a.override)
    pcfg = make_prepare_config(cfg)
    mcfg = pcfg.motion
    source = NuScenesWindowSource(a.dataroot, info_pkl=a.info_pkl, verbose=False)

    variant_names = (
        "ego_strict_exact",
        "ego_observed_only_exact",
        "ego_strict_r1",
        "ego_observed_only_r1",
        "lidar_strict_exact",
        "lidar_observed_only_exact",
        "lidar_strict_r1",
        "lidar_observed_only_r1",
    )
    totals = {k: _empty_state() for k in variant_names}
    per_class = {
        c: {k: _empty_state() for k in variant_names}
        for c in range(17)
    }
    alignment = {
        "raw_unaligned": _empty_alignment(pcfg.history_frames),
        "ego_pose_aligned": _empty_alignment(pcfg.history_frames),
        "lidar_pose_aligned": _empty_alignment(pcfg.history_frames),
    }

    num_windows = 0
    for i, w in enumerate(source.iter_windows(
        history=pcfg.history_frames,
        future=pcfg.future_frames,
        stride=a.stride,
        max_windows=a.max_windows,
    )):
        hist = np.stack([
            source.load_semantics(w.scene_name, tok) for tok in w.history_tokens
        ])
        ego_poses = [source.pose(tok) for tok in w.history_tokens]
        lidar_poses = [_sample_lidar_to_world(source, tok) for tok in w.history_tokens]
        ego_aligned = ego_compensate_sequence(
            hist, ego_poses, -1, pcfg.grid, pcfg.free_label
        )
        lidar_aligned = ego_compensate_sequence(
            hist, lidar_poses, -1, pcfg.grid, pcfg.free_label
        )

        _acc_alignment(alignment["raw_unaligned"], hist, pcfg.free_label)
        _acc_alignment(alignment["ego_pose_aligned"], ego_aligned, pcfg.free_label)
        _acc_alignment(alignment["lidar_pose_aligned"], lidar_aligned, pcfg.free_label)

        variants = {
            "ego_strict_exact": _persistence(ego_aligned, pcfg.free_label, False, 0),
            "ego_observed_only_exact": _persistence(ego_aligned, pcfg.free_label, True, 0),
            "ego_strict_r1": _persistence(ego_aligned, pcfg.free_label, False, 1),
            "ego_observed_only_r1": _persistence(ego_aligned, pcfg.free_label, True, 1),
            "lidar_strict_exact": _persistence(lidar_aligned, pcfg.free_label, False, 0),
            "lidar_observed_only_exact": _persistence(lidar_aligned, pcfg.free_label, True, 0),
            "lidar_strict_r1": _persistence(lidar_aligned, pcfg.free_label, False, 1),
            "lidar_observed_only_r1": _persistence(lidar_aligned, pcfg.free_label, True, 1),
        }
        # The current occupancy grid is identical before/after relative alignment.
        current = ego_aligned[-1]
        for name, pers in variants.items():
            _acc_state(
                totals[name], current, pers, pcfg.free_label,
                mcfg.static_min_persistence, mcfg.moving_max_persistence,
            )
            for cls in range(17):
                _acc_state(
                    per_class[cls][name], current, pers, pcfg.free_label,
                    mcfg.static_min_persistence, mcfg.moving_max_persistence,
                    class_id=cls,
                )

        num_windows += 1
        if i % 8 == 0:
            print("audited", i, w.scene_name, w.t0_token)

    global_variants = {k: _final_state(v) for k, v in totals.items()}
    class_rows = []
    for cls in range(17):
        occ = per_class[cls]["ego_strict_exact"]["occupied"]
        if occ == 0:
            continue
        class_rows.append({
            "class_id": cls,
            "class_name": NUSCENES_LABELS[cls],
            "variants": {k: _final_state(per_class[cls][k]) for k in variant_names},
        })

    alignment_out = {
        k: _final_alignment(v, mcfg.history_dt_s) for k, v in alignment.items()
    }

    report = {
        "protocol": "P0-B_alignment_persistence_audit_v1",
        "num_windows": num_windows,
        "causal": True,
        "uses_future_gt": False,
        "thresholds": {
            "static_min_persistence": mcfg.static_min_persistence,
            "moving_max_persistence": mcfg.moving_max_persistence,
            "radius1_xy_m": [
                float(mcfg.voxel_size_xy_m[0]),
                float(mcfg.voxel_size_xy_m[1]),
            ],
        },
        "alignment_quality": alignment_out,
        "global_persistence_variants": global_variants,
        "per_class": class_rows,
        "interpretation": {
            "alignment_test": (
                "If ego/lidar alignment does not improve same-class agreement over raw, "
                "inspect pose/frame conventions. Whichever aligned frame gives clearly "
                "higher agreement is the better occupancy-frame hypothesis."
            ),
            "free_hole_test": (
                "A large drop in moving ratio from strict_exact to observed_only_exact "
                "means warp-created/free cells in the fixed six-frame denominator are a "
                "major source of false motion."
            ),
            "quantization_test": (
                "A large drop from exact to radius-1 means one-cell rasterization shifts "
                "are a major source of false motion. Radius-1 here is diagnostic only."
            ),
        },
    }

    op = Path(a.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(report, indent=2, allow_nan=True), encoding="utf-8")
    save_resolved_config(cfg, op.with_suffix(".resolved.yaml"))

    print(json.dumps({
        "saved": str(op),
        "num_windows": num_windows,
        "global_persistence_variants": global_variants,
        "alignment_quality": alignment_out,
    }, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
