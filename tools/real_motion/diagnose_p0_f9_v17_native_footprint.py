#!/usr/bin/env python3
"""B0 audit for the V17-RL native source-footprint proposal.

This is deliberately a *single-checkpoint, no-training* data/supervision audit.
It does not perform the earlier multi-checkpoint proxy matrix and does not run a
full occupancy evaluator.

For every Strong source in the frozen V17 validation cache it:
- reconstructs the exact t0 source voxels using the same Strong extractor;
- projects those voxels to an exact native-resolution XY footprint;
- aligns that footprint with the legacy 0.8 m source mask in the same local
  source-centered coordinates before comparing support;
- summarizes inclusion/omission by class and source-size tertile;
- runs the supplied RL checkpoint once and checks legacy-vs-native overlap loss
  values and residual-output gradients at each future horizon, using the same
  legacy overlap eligibility set for both masks.

Future occupancy is never read.  GT displacement labels already stored in the
frozen validation cache are used only for the loss/gradient audit.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from real_motion.local_st_world_model_v17 import soft_transport_overlap_loss
from real_motion.metrics.moving_miou_v2 import SPEED_THRESHOLD_MPS
from real_motion.nuscenes_adapter import NuScenesWindowSource
from real_motion.runtime_config import add_config_args, load_runtime_config, make_prepare_config
from real_motion.strong_w2det import StrongW2DetConfig, extract_instances
from tools.real_motion.eval_p0_f9_v17_local_stwm import load_cache, load_model, window_from_record

PROTOCOL = "p0_f9_v17_native_source_footprint_b0_audit_v1"
MASK_COMPARE_CONTRACT = "exact_strong_xy_or_pool_vs_legacy_local_mask_common_0p8m_coordinates_v1"
GRADIENT_CONTRACT = "same_legacy_overlap_eligibility_per_horizon_output_residual_gradient_v1"
FUTURE_HORIZONS_S = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)


def exact_source_masks(
    voxel_indices_xyz: np.ndarray,
    center_xy_m: np.ndarray,
    *,
    grid,
    patch_size_m: float,
    pooled_resolution_m: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return tight native footprint, exact pooled local mask, and crop coverage.

    ``pooled`` is built by OR-pooling the exact source footprint into the same
    source-centered 0.8 m cells used by the V17 legacy mask.  It therefore
    provides a valid common-coordinate comparison without pretending that the
    legacy 0.8 m mask has native 0.4 m precision.
    """
    idx = np.asarray(voxel_indices_xyz, dtype=np.int64)
    if idx.ndim != 2 or idx.shape[1] != 3 or len(idx) == 0:
        raise ValueError("voxel_indices_xyz must be non-empty [N,3]")
    xy = np.unique(idx[:, :2], axis=0)
    lo = xy.min(axis=0)
    hi = xy.max(axis=0)
    tight = np.zeros(tuple((hi - lo + 1).tolist()), dtype=np.uint8)
    q = xy - lo[None]
    tight[q[:, 0], q[:, 1]] = 1

    vx, vy = float(grid.voxel_size[0]), float(grid.voxel_size[1])
    if abs(vx - vy) > 1e-12:
        raise ValueError("B0 audit requires square native XY voxels")
    pool_f = float(pooled_resolution_m) / vx
    pool = int(round(pool_f))
    if pool <= 0 or abs(pool_f - pool) > 1e-9:
        raise ValueError("pooled resolution must be an integer multiple of native resolution")
    raw_f = float(patch_size_m) / vx
    raw = int(round(raw_f))
    if raw <= 0 or raw % 2 or abs(raw_f - raw) > 1e-9 or raw % pool:
        raise ValueError("patch size/resolution is incompatible with native grid")

    center = np.asarray(center_xy_m, dtype=np.float64)
    if center.shape != (2,):
        raise ValueError("center_xy_m must be [2]")
    cx = int(np.floor((center[0] - float(grid.x_min)) / vx))
    cy = int(np.floor((center[1] - float(grid.y_min)) / vy))
    x0, y0 = cx - raw // 2, cy - raw // 2
    local = xy - np.asarray([x0, y0], dtype=np.int64)[None]
    inside = (
        (local[:, 0] >= 0) & (local[:, 0] < raw) &
        (local[:, 1] >= 0) & (local[:, 1] < raw)
    )
    native_patch = np.zeros((raw, raw), dtype=np.uint8)
    if bool(inside.any()):
        qi = local[inside]
        native_patch[qi[:, 0], qi[:, 1]] = 1
    out_hw = raw // pool
    pooled = native_patch.reshape(out_hw, pool, out_hw, pool).any(axis=(1, 3)).astype(np.uint8)
    coverage = float(inside.sum()) / max(int(len(xy)), 1)
    return tight, pooled, coverage


def mask_counts(legacy: np.ndarray, exact_pooled: np.ndarray) -> dict[str, float | int]:
    a = np.asarray(legacy, dtype=bool)
    b = np.asarray(exact_pooled, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("legacy/exact pooled masks must share common coordinates")
    inter = int((a & b).sum())
    legacy_n = int(a.sum())
    exact_n = int(b.sum())
    union = int((a | b).sum())
    return {
        "intersection": inter,
        "legacy_cells": legacy_n,
        "exact_cells": exact_n,
        "legacy_extra_cells": legacy_n - inter,
        "legacy_missed_cells": exact_n - inter,
        "iou": inter / union if union else float("nan"),
        "precision": inter / legacy_n if legacy_n else float("nan"),
        "recall": inter / exact_n if exact_n else float("nan"),
    }


def _aggregate_rows(rows: list[dict]) -> dict:
    if not rows:
        return {"sources": 0}
    inter = sum(int(r["intersection"]) for r in rows)
    legacy = sum(int(r["legacy_cells"]) for r in rows)
    exact = sum(int(r["exact_cells"]) for r in rows)
    union = legacy + exact - inter
    ious = [float(r["iou"]) for r in rows if np.isfinite(float(r["iou"]))]
    return {
        "sources": len(rows),
        "legacy_present_fraction": float(np.mean([int(r["legacy_cells"]) > 0 for r in rows])),
        "mean_native_crop_coverage": float(np.mean([float(r["native_crop_coverage"]) for r in rows])),
        "intersection": inter,
        "legacy_cells": legacy,
        "exact_cells": exact,
        "legacy_extra_cells": legacy - inter,
        "legacy_missed_cells": exact - inter,
        "micro_iou": inter / union if union else float("nan"),
        "micro_precision": inter / legacy if legacy else float("nan"),
        "micro_recall": inter / exact if exact else float("nan"),
        "mean_source_iou": float(np.mean(ious)) if ious else float("nan"),
    }


def _size_edges(voxel_counts: Iterable[int]) -> tuple[int, int]:
    x = np.asarray(list(voxel_counts), dtype=np.int64)
    if x.size == 0:
        raise ValueError("cannot define size bins from an empty source set")
    try:
        q = np.quantile(x, [1.0 / 3.0, 2.0 / 3.0], method="nearest")
    except TypeError:  # NumPy < 1.22 compatibility
        q = np.quantile(x, [1.0 / 3.0, 2.0 / 3.0], interpolation="nearest")
    return int(q[0]), int(q[1])


def _size_name(n: int, edges: tuple[int, int]) -> str:
    q1, q2 = edges
    if int(n) <= q1:
        return "small"
    if int(n) <= q2:
        return "medium"
    return "large"


def _pad_tight_masks(masks: list[np.ndarray]) -> torch.Tensor:
    if not masks:
        return torch.zeros((0, 1, 1), dtype=torch.float32)
    H = max(int(m.shape[0]) for m in masks)
    W = max(int(m.shape[1]) for m in masks)
    out = torch.zeros((len(masks), H, W), dtype=torch.float32)
    for i, m in enumerate(masks):
        h, w = int(m.shape[0]), int(m.shape[1])
        x0 = (H - h) // 2
        y0 = (W - w) // 2
        out[i, x0:x0 + h, y0:y0 + w] = torch.from_numpy(np.asarray(m, dtype=np.float32))
    return out


def _horizon_overlap_gradient_audit(
    pred_cpu: torch.Tensor,
    target_cpu: torch.Tensor,
    valid_cpu: torch.Tensor,
    legacy_present_cpu: torch.Tensor,
    legacy_mask_cpu: torch.Tensor,
    native_masks: list[np.ndarray],
    *,
    device: torch.device,
    legacy_resolution_m: float,
    native_resolution_m: float,
    batch_size: int,
) -> list[dict]:
    N = int(pred_cpu.shape[0])
    reports = []
    for hi, seconds in enumerate(FUTURE_HORIZONS_S):
        accum = {
            "legacy_iou_sum": 0.0,
            "native_iou_sum": 0.0,
            "labels": 0,
            "legacy_grad_sq_sum": 0.0,
            "native_grad_sq_sum": 0.0,
            "legacy_nonzero": 0,
            "native_nonzero": 0,
            "gradient_pairs": 0,
            "legacy_finite": True,
            "native_finite": True,
        }
        for start in range(0, N, max(1, int(batch_size))):
            stop = min(N, start + max(1, int(batch_size)))
            eligibility = valid_cpu[start:stop, hi].bool() & legacy_present_cpu[start:stop].bool()
            n = int(eligibility.sum().item())
            if n == 0:
                continue
            horizon_valid = torch.zeros((stop - start, 6), dtype=torch.bool)
            horizon_valid[:, hi] = eligibility
            target = target_cpu[start:stop].to(device)

            def one(mask_cpu: torch.Tensor, resolution: float):
                pred = pred_cpu[start:stop].to(device).detach().requires_grad_(True)
                loss, stats = soft_transport_overlap_loss(
                    pred,
                    target,
                    mask_cpu.to(device),
                    horizon_valid.to(device),
                    patch_resolution_m=float(resolution),
                )
                grad = torch.autograd.grad(loss, pred, retain_graph=False, create_graph=False)[0]
                # soft_transport_overlap_loss is a mean over labels.  Multiply by
                # n so the per-pair gradient magnitude is independent of audit
                # chunk size before accumulating RMS/nonzero statistics.
                g = grad[:, hi][eligibility.to(device)] * float(n)
                finite = bool(torch.isfinite(g).all().item()) and bool(torch.isfinite(loss).item())
                pair_norm = torch.linalg.vector_norm(g.float(), dim=-1)
                return {
                    "iou_sum": float(stats["transport_soft_iou"]) * n,
                    "grad_sq_sum": float((g.float() ** 2).sum().detach().cpu()),
                    "nonzero": int((pair_norm > 1e-12).sum().item()),
                    "pairs": int(pair_norm.numel()),
                    "finite": finite,
                }

            legacy = one(legacy_mask_cpu[start:stop].float(), legacy_resolution_m)
            native_chunk = _pad_tight_masks(native_masks[start:stop])
            native = one(native_chunk, native_resolution_m)
            accum["legacy_iou_sum"] += legacy["iou_sum"]
            accum["native_iou_sum"] += native["iou_sum"]
            accum["labels"] += n
            accum["legacy_grad_sq_sum"] += legacy["grad_sq_sum"]
            accum["native_grad_sq_sum"] += native["grad_sq_sum"]
            accum["legacy_nonzero"] += legacy["nonzero"]
            accum["native_nonzero"] += native["nonzero"]
            accum["gradient_pairs"] += native["pairs"]
            accum["legacy_finite"] = bool(accum["legacy_finite"] and legacy["finite"])
            accum["native_finite"] = bool(accum["native_finite"] and native["finite"])

        labels = int(accum["labels"])
        pairs = max(int(accum["gradient_pairs"]), 1)
        reports.append({
            "horizon_s": float(seconds),
            "overlap_eligible_labels": labels,
            "legacy_soft_iou": accum["legacy_iou_sum"] / labels if labels else float("nan"),
            "native_soft_iou": accum["native_iou_sum"] / labels if labels else float("nan"),
            "legacy_per_pair_grad_rms": math.sqrt(accum["legacy_grad_sq_sum"] / (2 * pairs)) if labels else float("nan"),
            "native_per_pair_grad_rms": math.sqrt(accum["native_grad_sq_sum"] / (2 * pairs)) if labels else float("nan"),
            "legacy_nonzero_gradient_fraction": accum["legacy_nonzero"] / pairs if labels else float("nan"),
            "native_nonzero_gradient_fraction": accum["native_nonzero"] / pairs if labels else float("nan"),
            "legacy_gradient_finite": bool(accum["legacy_finite"]),
            "native_gradient_finite": bool(accum["native_finite"]),
        })
    return reports


def main():
    p = argparse.ArgumentParser()
    add_config_args(p)
    p.add_argument("--local-stwm-cache", required=True)
    p.add_argument("--checkpoint", required=True, help="expected V17-RL epoch-5 checkpoint")
    p.add_argument("--dataroot", required=True)
    p.add_argument("--info-pkl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-windows", type=int, default=0)
    p.add_argument("--gradient-batch-size", type=int, default=128)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    cfg = load_runtime_config(a.config, a.override)
    pcfg = make_prepare_config(cfg)
    meta, records = load_cache(a.local_stwm_cache)
    if a.max_windows > 0:
        records = records[: min(len(records), int(a.max_windows))]
    if not records:
        raise RuntimeError("no records selected")
    patch_size_m = float(meta.get("patch_size_m", 16.0))
    legacy_resolution_m = float(meta.get("patch_resolution_m", 0.8))
    native_resolution_m = float(pcfg.grid.voxel_size[0])
    if abs(float(pcfg.grid.voxel_size[1]) - native_resolution_m) > 1e-12:
        raise RuntimeError("non-square native XY grid is unsupported")

    device = torch.device(a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu")
    ck, model = load_model(a.checkpoint, device)
    if str(ck.get("variant")) != "RL" or not bool(ck.get("use_representation", False)):
        raise RuntimeError("B0 audit requires a V17-RL representation checkpoint")
    if float(ck.get("overlap_weight", 0.0)) <= 0:
        raise RuntimeError("B0 audit expects the existing RL overlap objective to be enabled")
    source = NuScenesWindowSource(a.dataroot, info_pkl=a.info_pkl, verbose=False)
    strong_cfg = StrongW2DetConfig(free_label=int(pcfg.free_label))

    rows = []
    pred_parts = []
    target_parts = []
    valid_parts = []
    target_disp_parts = []
    legacy_masks = []
    native_masks: list[np.ndarray] = []
    centroid_max_error_m = 0.0

    for wi, rec in enumerate(records):
        w = window_from_record(rec)
        history_occ = [source.load_semantics(w.scene_name, tok) for tok in w.history_tokens]
        history_poses = [np.asarray(source.pose(tok), dtype=np.float64) for tok in w.history_tokens]
        current = extract_instances(history_occ[-1], history_poses[-1], grid=pcfg.grid, cfg=strong_cfg)
        if len(current) != int(rec["features"].shape[0]):
            raise RuntimeError(f"{rec['sample_id']}: Strong source count differs from V17 cache")
        classes = [int(c["class_id"]) for c in current]
        cached_classes = [int(x) for x in rec["source_class_id"].tolist()]
        if classes != cached_classes:
            raise RuntimeError(f"{rec['sample_id']}: Strong source ordering differs from V17 cache")

        features = rec["features"].float().to(device)
        tube = rec["local_semantic_tube"].to(device)
        kta_disp = rec["kta_displacement_xy_m"].float().to(device)
        frame_motion = rec["frame_motion_features"].float().to(device)
        source_mask = rec["target_source_mask_tube"].to(device)
        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            out = model(features, tube, kta_disp, frame_motion, source_mask)
        pred_parts.append(out["residual_xy_m"].float().cpu())
        target_parts.append(rec["target_residual_xy_m"].float().cpu())
        valid_parts.append(rec["target_valid"].bool().cpu())
        target_disp_parts.append(rec["target_displacement_xy_m"].float().cpu())
        legacy_t0 = rec["target_source_mask_tube"][:, -1].to(torch.uint8).cpu()
        legacy_masks.append(legacy_t0)

        t0_pose = history_poses[-1]
        source_xy_cached = rec["source_centroid_xy_t0_m"].float().numpy()
        for i, comp in enumerate(current):
            cworld = np.asarray(comp["centroid_world"], dtype=np.float64)
            world_to_t0 = np.linalg.inv(t0_pose)
            ct0 = (world_to_t0 @ np.asarray([cworld[0], cworld[1], cworld[2], 1.0]))[:2]
            cerr = float(np.linalg.norm(ct0 - source_xy_cached[i].astype(np.float64)))
            centroid_max_error_m = max(centroid_max_error_m, cerr)
            if cerr > 0.05:
                raise RuntimeError(
                    f"{rec['sample_id']} source {i}: cached/source centroid mismatch {cerr:.4f} m"
                )
            tight, exact_pooled, crop_coverage = exact_source_masks(
                comp["voxel_indices"],
                source_xy_cached[i],
                grid=pcfg.grid,
                patch_size_m=patch_size_m,
                pooled_resolution_m=legacy_resolution_m,
            )
            legacy = legacy_t0[i].numpy()
            if legacy.shape != exact_pooled.shape:
                raise RuntimeError(
                    f"{rec['sample_id']} source {i}: legacy mask {legacy.shape} != exact pooled {exact_pooled.shape}"
                )
            counts = mask_counts(legacy, exact_pooled)
            counts.update({
                "sample_id": str(rec["sample_id"]),
                "source_index": int(i),
                "class_id": int(comp["class_id"]),
                "voxel_count": int(len(comp["voxel_indices"])),
                "native_xy_cells": int(tight.sum()),
                "native_height": int(tight.shape[0]),
                "native_width": int(tight.shape[1]),
                "native_crop_coverage": float(crop_coverage),
            })
            rows.append(counts)
            native_masks.append(tight)

        if wi == 0 or (wi + 1) % 16 == 0 or wi + 1 == len(records):
            print(f"b0_native_footprint {wi + 1}/{len(records)} sources={len(rows)}")

    pred_all = torch.cat(pred_parts, dim=0)
    target_all = torch.cat(target_parts, dim=0)
    valid_all = torch.cat(valid_parts, dim=0)
    target_disp_all = torch.cat(target_disp_parts, dim=0)
    legacy_mask_all = torch.cat(legacy_masks, dim=0)
    if len(rows) != int(pred_all.shape[0]) or len(native_masks) != int(pred_all.shape[0]):
        raise RuntimeError("source audit rows are not aligned with model predictions")
    legacy_present = legacy_mask_all.flatten(1).any(dim=1)

    size_edges = _size_edges(r["voxel_count"] for r in rows)
    by_class = {}
    for cid in sorted({int(r["class_id"]) for r in rows}):
        by_class[str(cid)] = _aggregate_rows([r for r in rows if int(r["class_id"]) == cid])
    by_size = {}
    for name in ("small", "medium", "large"):
        by_size[name] = _aggregate_rows(
            [r for r in rows if _size_name(int(r["voxel_count"]), size_edges) == name]
        )

    horizon_supervision = []
    for hi, seconds in enumerate(FUTURE_HORIZONS_S):
        dt = float(seconds)
        speed = torch.linalg.vector_norm(target_disp_all[:, hi], dim=-1) / dt
        tm = valid_all[:, hi] & (speed >= float(SPEED_THRESHOLD_MPS))
        horizon_supervision.append({
            "horizon_s": dt,
            "target_valid_labels": int(valid_all[:, hi].sum().item()),
            "true_moving_labels": int(tm.sum().item()),
            "legacy_overlap_eligible_labels": int((valid_all[:, hi] & legacy_present).sum().item()),
        })

    gradient_audit = _horizon_overlap_gradient_audit(
        pred_all,
        target_all,
        valid_all,
        legacy_present,
        legacy_mask_all,
        native_masks,
        device=device,
        legacy_resolution_m=legacy_resolution_m,
        native_resolution_m=native_resolution_m,
        batch_size=int(a.gradient_batch_size),
    )
    if any(not x["native_gradient_finite"] for x in gradient_audit):
        raise RuntimeError("native footprint overlap produced non-finite residual-output gradients")

    result = {
        "protocol": PROTOCOL,
        "mask_compare_contract": MASK_COMPARE_CONTRACT,
        "gradient_contract": GRADIENT_CONTRACT,
        "checkpoint": str(Path(a.checkpoint).resolve()),
        "checkpoint_epoch": int(ck.get("epoch", -1)),
        "checkpoint_variant": str(ck.get("variant")),
        "local_stwm_cache": str(Path(a.local_stwm_cache).resolve()),
        "num_windows": len(records),
        "num_sources": len(rows),
        "patch_size_m": patch_size_m,
        "legacy_mask_resolution_m": legacy_resolution_m,
        "native_footprint_resolution_m": native_resolution_m,
        "centroid_max_alignment_error_m": centroid_max_error_m,
        "size_bin_voxel_count_edges": {"q33": size_edges[0], "q67": size_edges[1]},
        "mask_alignment": {
            "overall": _aggregate_rows(rows),
            "by_class": by_class,
            "by_size": by_size,
        },
        "horizon_supervision": horizon_supervision,
        "overlap_gradient_audit": gradient_audit,
        "notes": {
            "no_future_occupancy_read": True,
            "single_checkpoint_only": True,
            "same_overlap_eligibility": "target_valid AND legacy_t0_mask_present",
            "size_bins": "descriptive source-voxel-count tertiles on the audited source set",
        },
    }

    print("\n=== B0 NATIVE SOURCE FOOTPRINT ALIGNMENT ===")
    o = result["mask_alignment"]["overall"]
    print(
        f"sources={o['sources']} legacy_present={100*o['legacy_present_fraction']:.2f}% "
        f"crop_coverage={100*o['mean_native_crop_coverage']:.3f}% "
        f"microIoU={100*o['micro_iou']:.3f}% precision={100*o['micro_precision']:.3f}% "
        f"recall={100*o['micro_recall']:.3f}% meanSourceIoU={100*o['mean_source_iou']:.3f}%"
    )
    print(f"size_edges_voxels={size_edges}")
    print("\nby_class:")
    for cid, r in by_class.items():
        print(
            f"  class={cid:>2s} n={r['sources']:4d} microIoU={100*r['micro_iou']:.3f}% "
            f"precision={100*r['micro_precision']:.3f}% recall={100*r['micro_recall']:.3f}%"
        )
    print("\nby_size:")
    for name, r in by_size.items():
        print(
            f"  {name:6s} n={r['sources']:4d} microIoU={100*r['micro_iou']:.3f}% "
            f"precision={100*r['micro_precision']:.3f}% recall={100*r['micro_recall']:.3f}%"
        )

    print("\n=== B0 HORIZON SUPERVISION / OVERLAP GRADIENT ===")
    for sup, grad in zip(horizon_supervision, gradient_audit):
        print(
            f"{sup['horizon_s']:.1f}s valid={sup['target_valid_labels']:5d} "
            f"trueMoving={sup['true_moving_labels']:5d} eligible={grad['overlap_eligible_labels']:5d} "
            f"legacyIoU={100*grad['legacy_soft_iou']:.3f}% nativeIoU={100*grad['native_soft_iou']:.3f}% "
            f"legacyGradRMS={grad['legacy_per_pair_grad_rms']:.6g} "
            f"nativeGradRMS={grad['native_per_pair_grad_rms']:.6g} "
            f"nativeNonzero={100*grad['native_nonzero_gradient_fraction']:.2f}% "
            f"finite={grad['native_gradient_finite']}"
        )

    op = Path(a.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"saved {op}")


if __name__ == "__main__":
    main()
