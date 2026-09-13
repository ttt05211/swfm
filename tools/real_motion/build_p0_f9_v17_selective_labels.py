#!/usr/bin/env python3
"""Build exact KTA-vs-V17 source utility labels for selective forecasting.

The label uses future GT only as supervision. For every source, it measures the
exact single-source marginal change in the frozen dataset-level Moving-mIoU-v2
when that source alone switches from KTA transport to V17 transport under the A1
CLEAR/WRITE compositor.

This cache is used twice:
  * train scenes: supervision for the tiny causal selector;
  * val scenes: GT oracle ranking and analysis only.

Selector inputs are copied verbatim from the causal 46-D V17 source features.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from real_motion.metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS, REPORT_HORIZONS_S
from real_motion.msp_wm_cache import MSPWorldModelCacheDataset
from real_motion.nuscenes_adapter import (
    NuScenesWindowSource,
    gt_moving_support_for_horizon,
)
from real_motion.rigid_transport import rasterize_rigid_component
from real_motion.runtime_config import add_config_args, load_runtime_config, make_prepare_config
from real_motion.selective_forecast import (
    SELECTIVE_LABEL_CACHE_VERSION,
    SELECTOR_INPUT_CONTRACT,
    UTILITY_CONTRACT,
    marginal_by_horizon_pp,
    marginal_utility_pp,
    moving_counts,
    moving_miou_from_counts,
    single_source_moving_delta_counts,
)
from real_motion.strong_w2det import (
    StrongW2DetConfig,
    extract_instances,
    match_instances,
    strong_w2det_sequence,
)
from tools.real_motion import eval_p0_f9_frozen_sparse_occfm as safe
from tools.real_motion.eval_p0_f9_v17_local_stwm import (
    load_cache,
    load_model,
    t0_xy_to_world_preserve_source_z,
    window_from_record,
)

PROTOCOL = "p0_f9_v17_selective_label_builder_v1"


def _select_records(records, p0f9_ids, *, max_windows: int, seed: int, allow_missing: bool):
    by_id = {str(r["sample_id"]): r for r in records}
    missing = [sid for sid in p0f9_ids if sid not in by_id]
    if missing and not allow_missing:
        raise RuntimeError(
            f"V17 cache misses {len(missing)}/{len(p0f9_ids)} P0-F9 samples; "
            f"first={missing[:5]}"
        )
    selected = [by_id[sid] for sid in p0f9_ids if sid in by_id]
    if int(max_windows) > 0 and len(selected) > int(max_windows):
        rng = random.Random(int(seed))
        rng.shuffle(selected)
        selected = selected[: int(max_windows)]
        selected.sort(key=lambda r: str(r["sample_id"]))
    return selected, missing


def _raw_eval_payload(source, w, pcfg, strong_cfg):
    history_occ = np.stack(
        [source.load_semantics(w.scene_name, tok) for tok in w.history_tokens],
        axis=0,
    ).astype(np.uint8, copy=False)
    history_poses = [
        np.asarray(source.pose(tok), dtype=np.float64) for tok in w.history_tokens
    ]
    future_poses = [
        np.asarray(source.pose(tok), dtype=np.float64) for tok in w.future_tokens
    ]
    gt = np.stack(
        [source.load_semantics(w.scene_name, tok) for tok in w.future_tokens],
        axis=0,
    ).astype(np.uint8, copy=False)
    anchor = strong_w2det_sequence(
        history_occ,
        history_poses,
        future_poses,
        frame_dt_s=float(pcfg.frame_dt_s),
        grid=pcfg.grid,
        cfg=strong_cfg,
    ).astype(np.uint8, copy=False)
    moving = np.zeros_like(gt, dtype=bool)
    for horizon, hi in safe.REPORT.items():
        sup, _, _ = gt_moving_support_for_horizon(
            source.nusc,
            w.t0_token,
            w.future_tokens[hi],
            float(horizon),
            grid=pcfg.grid,
        )
        moving[hi] = sup
    return {
        "gt": gt,
        "anchor": anchor,
        "moving": moving,
        "history_occ": history_occ,
        "history_poses": history_poses,
        "future_poses": future_poses,
        "source": "raw_occ3d_reconstructed",
    }


def _cached_eval_payload(sample):
    needed = (
        "eval_future_gt_occ",
        "eval_strong_anchor_occ",
        "eval_gt_moving_support",
    )
    if not all(k in sample for k in needed):
        return None
    return {
        "gt": sample["eval_future_gt_occ"].cpu().numpy(),
        "anchor": sample["eval_strong_anchor_occ"].cpu().numpy(),
        "moving": sample["eval_gt_moving_support"].cpu().numpy().astype(bool),
        "source": "p0f9_cached_eval_payload",
    }


def _assert_payload_equal(raw, cached, sid):
    for key in ("gt", "anchor", "moving"):
        if not np.array_equal(np.asarray(raw[key]), np.asarray(cached[key])):
            a = np.asarray(raw[key])
            b = np.asarray(cached[key])
            diff = int(np.count_nonzero(a != b))
            raise RuntimeError(
                f"{sid}: raw-reconstructed {key} differs from cached eval payload "
                f"at {diff} voxels"
            )


def _label_stats(values):
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return {}
    return {
        "count": int(x.size),
        "mean_pp": float(x.mean()),
        "std_pp": float(x.std()),
        "min_pp": float(x.min()),
        "p10_pp": float(np.quantile(x, 0.10)),
        "median_pp": float(np.median(x)),
        "p90_pp": float(np.quantile(x, 0.90)),
        "max_pp": float(x.max()),
        "positive_fraction": float(np.mean(x > 0.0)),
        "negative_fraction": float(np.mean(x < 0.0)),
        "zero_fraction": float(np.mean(x == 0.0)),
    }


def main():
    p = argparse.ArgumentParser()
    add_config_args(p)
    p.add_argument("--v17-cache", required=True)
    p.add_argument("--p0f9-cache", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataroot", required=True)
    p.add_argument("--info-pkl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-windows", type=int, default=0)
    p.add_argument("--selection-seed", type=int, default=20260913)
    p.add_argument("--allow-missing", action="store_true")
    p.add_argument(
        "--audit-raw-payload-windows",
        type=int,
        default=2,
        help="When cached eval payload exists, compare this many windows against exact raw reconstruction.",
    )
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    cfg = load_runtime_config(a.config, a.override)
    pcfg = make_prepare_config(cfg)
    cache_meta, all_records = load_cache(a.v17_cache)
    ds = MSPWorldModelCacheDataset(a.p0f9_cache)
    p0f9_ids = [str(e["sample_id"]) for e in ds.entries]
    records, missing = _select_records(
        all_records,
        p0f9_ids,
        max_windows=int(a.max_windows),
        seed=int(a.selection_seed),
        allow_missing=bool(a.allow_missing),
    )
    if not records:
        raise RuntimeError("no overlapping V17/P0-F9 windows")

    sid_to_idx = {str(e["sample_id"]): i for i, e in enumerate(ds.entries)}
    device = torch.device(
        a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    ck, model = load_model(a.checkpoint, device)
    if str(ck.get("variant")) != "RL":
        raise RuntimeError(f"expected V17-RL checkpoint, got variant={ck.get('variant')}")
    if not bool(ck.get("use_representation", False)):
        raise RuntimeError("selective labels require the V17 representation path")

    source = NuScenesWindowSource(a.dataroot, info_pkl=a.info_pkl, verbose=False)
    strong_cfg = StrongW2DetConfig(free_label=int(pcfg.free_label))

    # [3 report horizons, 8 Moving classes].
    anchor_inter = np.zeros((len(REPORT_HORIZONS_S), len(DYNAMIC_CLASS_IDS)), dtype=np.int64)
    anchor_union = np.zeros_like(anchor_inter)

    out_records = []
    source_total = 0
    started = time.perf_counter()

    for wi, rec in enumerate(records):
        sid = str(rec["sample_id"])
        w = window_from_record(rec)
        cached = None
        # Validation P0-F9 caches carry the compact eval payload; training
        # caches commonly do not. Raw reconstruction is therefore the canonical
        # path, with cached validation payload used only as an audited fast path.
        if bool(ds.metadata.get("include_eval_payload", False)):
            cached = _cached_eval_payload(ds[sid_to_idx[sid]])
        need_raw = cached is None or wi < int(a.audit_raw_payload_windows)
        raw = (
            _raw_eval_payload(source, w, pcfg, strong_cfg)
            if need_raw else None
        )
        if cached is not None and raw is not None:
            _assert_payload_equal(raw, cached, sid)
        payload = cached if cached is not None else raw
        if payload is None:
            raise RuntimeError(f"{sid}: no evaluation payload available")
        if raw is not None:
            history_occ = [np.asarray(x) for x in raw["history_occ"]]
            history_poses = raw["history_poses"]
            future_poses = raw["future_poses"]
        else:
            history_occ = [
                source.load_semantics(w.scene_name, tok) for tok in w.history_tokens
            ]
            history_poses = [
                np.asarray(source.pose(tok), dtype=np.float64)
                for tok in w.history_tokens
            ]
            future_poses = [
                np.asarray(source.pose(tok), dtype=np.float64)
                for tok in w.future_tokens
            ]
        current = extract_instances(
            history_occ[-1], history_poses[-1], grid=pcfg.grid, cfg=strong_cfg
        )
        previous = extract_instances(
            history_occ[-2], history_poses[-2], grid=pcfg.grid, cfg=strong_cfg
        )
        velocities = match_instances(
            previous,
            current,
            float(pcfg.frame_dt_s),
            max_speed_mps=strong_cfg.max_match_speed_mps,
        )
        if len(current) != int(rec["features"].shape[0]):
            raise RuntimeError(f"{sid}: Strong source count differs from V17 cache")
        if [int(c["class_id"]) for c in current] != [
            int(x) for x in rec["source_class_id"].tolist()
        ]:
            raise RuntimeError(f"{sid}: Strong source order/class differs from V17 cache")

        features = rec["features"].float().to(device)
        tube = rec["local_semantic_tube"].to(device)
        kta_disp = rec["kta_displacement_xy_m"].float().to(device)
        frame_motion = rec["frame_motion_features"].float().to(device)
        source_mask = rec["target_source_mask_tube"].to(device)
        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            pred = model(features, tube, kta_disp, frame_motion, source_mask)
        residual = pred["residual_xy_m"].float().cpu().numpy()

        n = len(current)
        d_inter = np.zeros(
            (n, len(REPORT_HORIZONS_S), len(DYNAMIC_CLASS_IDS)), dtype=np.int32
        )
        d_union = np.zeros_like(d_inter)
        t0_pose = history_poses[-1]

        for hs, (horizon, hi) in enumerate(safe.REPORT.items()):
            gt = payload["gt"][hi]
            anchor = payload["anchor"][hi]
            moving = payload["moving"][hi]
            ai, au = moving_counts(anchor, gt, moving)
            anchor_inter[hs] += ai
            anchor_union[hs] += au
            dt = (hi + 1) * float(pcfg.frame_dt_s)

            for i, comp in enumerate(current):
                src_center = np.asarray(comp["centroid_world"], dtype=np.float64)
                v = np.asarray(velocities.get(i, np.zeros(3)), dtype=np.float64)
                base = rasterize_rigid_component(
                    comp["voxel_indices"],
                    int(comp["class_id"]),
                    t0_pose,
                    future_poses[hi],
                    source_center_world=src_center,
                    target_center_world=src_center + v * dt,
                    yaw_delta_rad=0.0,
                    grid=pcfg.grid,
                )
                xy_pred = rec["anchors_xy_t0_m"][i, hi].numpy() + residual[i, hi]
                center_pred = t0_xy_to_world_preserve_source_z(
                    xy_pred, src_center, t0_pose
                )
                repl = rasterize_rigid_component(
                    comp["voxel_indices"],
                    int(comp["class_id"]),
                    t0_pose,
                    future_poses[hi],
                    source_center_world=src_center,
                    target_center_world=center_pred,
                    yaw_delta_rad=0.0,
                    grid=pcfg.grid,
                )
                di, du = single_source_moving_delta_counts(
                    anchor,
                    gt,
                    moving,
                    base,
                    repl,
                    free_label=int(pcfg.free_label),
                    grid=pcfg.grid,
                )
                d_inter[i, hs] = di.astype(np.int32)
                d_union[i, hs] = du.astype(np.int32)

        out_records.append({
            "sample_id": sid,
            "scene_name": str(rec["scene_name"]),
            "source_index": torch.arange(n, dtype=torch.int32),
            "source_class_id": rec["source_class_id"].to(torch.int16).cpu(),
            "source_voxel_count": rec["source_voxel_count"].to(torch.int32).cpu(),
            "features": rec["features"].float().cpu(),
            "v17_residual_xy_m": torch.from_numpy(residual.astype(np.float32)),
            "delta_inter": torch.from_numpy(d_inter),
            "delta_union": torch.from_numpy(d_union),
        })
        source_total += n

        if wi == 0 or (wi + 1) % 50 == 0 or wi + 1 == len(records):
            elapsed = max(time.perf_counter() - started, 1e-9)
            print(
                f"selective_labels {wi+1}/{len(records)} sources={source_total} "
                f"rate={(wi+1)/elapsed:.2f} win/s sid={sid}",
                flush=True,
            )

    anchor_moving = moving_miou_from_counts(anchor_inter, anchor_union)
    utility_values = []
    positive = 0
    for row in out_records:
        di = row["delta_inter"].numpy().astype(np.int64)
        du = row["delta_union"].numpy().astype(np.int64)
        util = np.zeros((len(di),), dtype=np.float32)
        by_h = np.zeros((len(di), len(REPORT_HORIZONS_S)), dtype=np.float32)
        for i in range(len(di)):
            util[i] = marginal_utility_pp(anchor_inter, anchor_union, di[i], du[i])
            by_h[i] = marginal_by_horizon_pp(
                anchor_inter, anchor_union, di[i], du[i]
            )
        row["utility_pp"] = torch.from_numpy(util)
        row["utility_by_horizon_pp"] = torch.from_numpy(by_h)
        utility_values.extend(util.tolist())
        positive += int((util > 0).sum())

    metadata = {
        "protocol": PROTOCOL,
        "utility_contract": UTILITY_CONTRACT,
        "selector_input_contract": SELECTOR_INPUT_CONTRACT,
        "v17_cache": str(Path(a.v17_cache).resolve()),
        "p0f9_cache": str(Path(a.p0f9_cache).resolve()),
        "checkpoint": str(Path(a.checkpoint).resolve()),
        "checkpoint_epoch": int(ck.get("epoch", -1)),
        "checkpoint_variant": str(ck.get("variant")),
        "num_windows": len(out_records),
        "num_sources": int(source_total),
        "num_p0f9_entries": len(p0f9_ids),
        "missing_p0f9_in_v17": len(missing),
        "selection_seed": int(a.selection_seed),
        "max_windows": int(a.max_windows),
        "moving_horizons_s": list(REPORT_HORIZONS_S),
        "dynamic_class_ids": [int(x) for x in DYNAMIC_CLASS_IDS],
        "anchor_inter": anchor_inter.tolist(),
        "anchor_union": anchor_union.tolist(),
        "anchor_moving_miou": float(anchor_moving),
        "label_stats": _label_stats(utility_values),
        "positive_sources": int(positive),
        "future_gt_used_for_labels_only": True,
        "selector_inputs_are_causal_only": True,
        "uses_gt_instance_matching": False,
        "eval_payload_source": (
            "cached_p0f9_when_available_else_raw_occ3d_reconstruction"
        ),
        "raw_payload_reconstruction_contract": (
            "raw_future_occ3d+strong_w2det_sequence+gt_moving_support_for_horizon_v1"
        ),
        "audit_raw_payload_windows": int(a.audit_raw_payload_windows),
        "p0f9_include_eval_payload": bool(ds.metadata.get("include_eval_payload", False)),
        "a1_single_source_clear_write": True,
        "cache_metadata": cache_meta,
    }
    payload = {
        "version": SELECTIVE_LABEL_CACHE_VERSION,
        "metadata": metadata,
        "records": out_records,
    }

    op = Path(a.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, op)
    op.with_suffix(".summary.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print("\n=== SELECTIVE EXPERT LABEL CACHE ===")
    print(json.dumps({
        "output": str(op),
        "windows": len(out_records),
        "sources": source_total,
        "anchor_moving_miou": anchor_moving,
        "label_stats": metadata["label_stats"],
    }, indent=2))


if __name__ == "__main__":
    main()
