#!/usr/bin/env python3
"""Find a cheap V17 validation proxy for final Moving-mIoU checkpoint selection.

This diagnostic never reads nuScenes future occupancy and never performs the
full rigid-scene composition evaluator.  It uses only the frozen V17 validation
cache and checkpoint predictions to score true-moving source/horizon pairs.

Candidate proxies deliberately target the mismatch observed in V16/V17:
- source-footprint Soft-IoU under predicted-vs-GT displacement error;
- quantized hard footprint IoU;
- horizon/class macro versions aligned with Moving-mIoU aggregation;
- footprint-area weighted overlap;
- center-error hit rates at voxel-relevant thresholds;
- class/horizon macro ADE (sign-flipped so higher is always better).

Optional full evaluator JSONs provide the true Moving-mIoU values used only to
measure proxy correlation/selection regret.  They are never inputs to a proxy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from real_motion.local_st_world_model_v17 import (
    MODEL_PROTOCOL_V17,
    LocalSpatialTemporalWorldModelV17,
    config_from_mapping_v17,
)
from real_motion.metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS
from tools.real_motion.train_p0_f9_v17_local_stwm import (
    flatten_supervised,
    load_cache,
    make_dataset,
    true_moving_mask,
    unpack,
)

REPORT_INDEX = {1.0: 1, 2.0: 3, 3.0: 5}
PROTOCOL = "p0_f9_v17_checkpoint_proxy_diagnostic_v1"


def _lookup(mapping, key):
    if key in mapping:
        return mapping[key]
    for k in (str(key), str(float(key))):
        if k in mapping:
            return mapping[k]
    raise KeyError(key)


def _transport_iou_per_label(
    error_xy_m: torch.Tensor,
    footprint_mask: torch.Tensor,
    *,
    patch_resolution_m: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return continuous Soft-IoU, quantized hard IoU, and footprint-present mask.

    The coordinate transform intentionally mirrors ``soft_transport_overlap_loss``:
    GT footprint stays centered and the predicted copy is shifted by displacement
    error.  Hard IoU rounds that shift to the nearest local 0.8 m footprint cell.
    """
    if error_xy_m.ndim != 3 or error_xy_m.shape[-1] != 2:
        raise ValueError("error_xy_m must be [B,6,2]")
    if footprint_mask.ndim != 3 or footprint_mask.shape[0] != error_xy_m.shape[0]:
        raise ValueError("footprint_mask must be [B,H,W]")
    if patch_resolution_m <= 0:
        raise ValueError("patch_resolution_m must be positive")

    err = error_xy_m.float()
    mask = footprint_mask.float()
    B, Hf, _ = err.shape
    H, W = int(mask.shape[-2]), int(mask.shape[-1])
    canvas = F.pad(mask[:, None], (W, W, H, H))
    Hc, Wc = int(canvas.shape[-2]), int(canvas.shape[-1])
    inp = canvas[:, None].expand(B, Hf, 1, Hc, Wc).reshape(B * Hf, 1, Hc, Wc)

    ys = torch.linspace(-1.0, 1.0, Hc, device=err.device, dtype=err.dtype)
    xs = torch.linspace(-1.0, 1.0, Wc, device=err.device, dtype=err.dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    base = torch.stack((gx, gy), dim=-1)[None].expand(B * Hf, Hc, Wc, 2)

    def shifted_iou(shift_cells: torch.Tensor, mode: str) -> torch.Tensor:
        grid = base.clone()
        grid[..., 1] -= (2.0 * shift_cells[:, 0] / max(Hc - 1, 1))[:, None, None]
        grid[..., 0] -= (2.0 * shift_cells[:, 1] / max(Wc - 1, 1))[:, None, None]
        shifted = F.grid_sample(inp, grid, mode=mode, padding_mode="zeros", align_corners=True)
        inter = (shifted * inp).sum(dim=(1, 2, 3))
        union = (shifted + inp - shifted * inp).sum(dim=(1, 2, 3))
        return ((inter + 1e-6) / (union + 1e-6)).reshape(B, Hf)

    shift = err.reshape(B * Hf, 2) / float(patch_resolution_m)
    soft = shifted_iou(shift, "bilinear")
    hard = shifted_iou(torch.round(shift), "nearest")
    present = canvas.flatten(1).sum(dim=1) > 0
    return soft, hard, present


def _mean(values):
    a = np.asarray(values, dtype=np.float64)
    return float(np.mean(a)) if a.size else float("nan")


def _weighted_mean(values, weights):
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if not v.size or not w.size or float(w.sum()) <= 0:
        return float("nan")
    return float(np.sum(v * w) / np.sum(w))


def aggregate_proxy_samples(samples: list[dict]) -> dict:
    """Aggregate source/horizon samples into candidate checkpoint-selection scores."""
    if not samples:
        raise RuntimeError("no usable true-moving report-horizon samples")

    groups = {}
    horizons = {}
    for row in samples:
        key = (float(row["horizon"]), int(row["class_id"]))
        groups.setdefault(key, []).append(row)
        horizons.setdefault(float(row["horizon"]), []).append(row)

    def group_macro(field, transform=lambda x: x):
        vals = []
        for rows in groups.values():
            vals.append(_mean([transform(float(r[field])) for r in rows]))
        return _mean(vals)

    def horizon_macro(field, weighted=False):
        vals = []
        for rows in horizons.values():
            if weighted:
                vals.append(
                    _weighted_mean([float(r[field]) for r in rows], [float(r["area"]) for r in rows])
                )
            else:
                vals.append(_mean([float(r[field]) for r in rows]))
        return _mean(vals)

    metrics = {
        "tm_soft_iou_micro": _mean([r["soft_iou"] for r in samples]),
        "tm_soft_iou_horizon_macro": horizon_macro("soft_iou"),
        "tm_soft_iou_class_horizon_macro": group_macro("soft_iou"),
        "tm_soft_iou_area_horizon_macro": horizon_macro("soft_iou", weighted=True),
        "tm_hard_iou_class_horizon_macro": group_macro("hard_iou"),
        "tm_hit_0p4_class_horizon_macro": group_macro("error_m", lambda x: 1.0 if x <= 0.4 else 0.0),
        "tm_hit_0p8_class_horizon_macro": group_macro("error_m", lambda x: 1.0 if x <= 0.8 else 0.0),
        "tm_hit_1p2_class_horizon_macro": group_macro("error_m", lambda x: 1.0 if x <= 1.2 else 0.0),
        "neg_tm_ade_class_horizon_macro": -group_macro("error_m"),
    }
    return {
        **metrics,
        "usable_samples": int(len(samples)),
        "present_class_horizon_groups": int(len(groups)),
        "mean_footprint_area_cells": _mean([r["area"] for r in samples]),
    }


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = rank
        i = j
    return ranks


def _corr(x, y) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _truth_from_eval_jsons(paths, branch: str) -> dict[int, dict]:
    truth = {}
    for path in paths:
        p = Path(path)
        obj = json.loads(p.read_text())
        epoch = int(obj.get("checkpoint_epoch", -1))
        if epoch < 0:
            raise RuntimeError(f"{p}: missing checkpoint_epoch")
        report = obj["reports"][branch]
        moving = float(report["moving"]["mIoU"])
        occ = float(report["occupancy"]["IoU"]) if "occupancy" in report else float("nan")
        row = {"moving_mIoU": moving, "occupancy_IoU": occ, "path": str(p.resolve())}
        if epoch in truth:
            old = truth[epoch]
            if abs(float(old["moving_mIoU"]) - moving) > 1e-6:
                raise RuntimeError(
                    f"conflicting full-eval Moving-mIoU for epoch {epoch}: "
                    f"{old['moving_mIoU']} vs {moving}"
                )
            if np.isfinite(occ) and np.isfinite(float(old["occupancy_IoU"])) and abs(float(old["occupancy_IoU"]) - occ) > 1e-6:
                raise RuntimeError(f"conflicting occupancy IoU for epoch {epoch}")
        else:
            truth[epoch] = row
    return truth


def _checkpoint_metadata(path: Path) -> dict:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("protocol") != MODEL_PROTOCOL_V17:
        raise RuntimeError(f"{path}: checkpoint protocol mismatch: {ck.get('protocol')}")
    return {
        "path": path,
        "epoch": int(ck.get("epoch", -1)),
        "variant": str(ck.get("variant", "?")),
        "use_representation": bool(ck.get("use_representation", False)),
        "checkpoint": ck,
    }


def evaluate_checkpoint(meta: dict, loader, device, patch_resolution_m: float) -> dict:
    ck = meta["checkpoint"]
    cfg = config_from_mapping_v17(ck.get("model_config"))
    model = LocalSpatialTemporalWorldModelV17(cfg).to(device)
    model.load_state_dict(ck["state_dict"], strict=True)
    model.eval()
    use_rep = bool(meta["use_representation"])
    samples = []
    total_tm = total_present = 0

    with torch.no_grad():
        for raw in loader:
            b = unpack(raw, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = model(
                    b["features"],
                    b["local_semantic_tube"],
                    b["kta_displacement_xy_m"],
                    b["frame_motion_features"] if use_rep else None,
                    b["target_source_mask_tube"] if use_rep else None,
                )
            target = b["target_residual_xy_m"].float()
            err_xy = out["residual_xy_m"].float() - target
            error_m = torch.linalg.vector_norm(err_xy, dim=-1)
            moving = true_moving_mask(b["target_displacement_xy_m"].float(), b["target_valid"].bool())
            footprint = b["target_source_mask_tube"][:, -1].float()
            soft_iou, hard_iou, footprint_present = _transport_iou_per_label(
                err_xy, footprint, patch_resolution_m=patch_resolution_m
            )
            area = footprint.flatten(1).sum(dim=1)
            class_id = b["source_class_id"].long()
            dynamic_source = torch.zeros_like(class_id, dtype=torch.bool)
            for c in DYNAMIC_CLASS_IDS:
                dynamic_source |= class_id == int(c)

            for horizon, hi in REPORT_INDEX.items():
                tm = moving[:, hi] & dynamic_source
                total_tm += int(tm.sum().item())
                usable = tm & footprint_present
                total_present += int(usable.sum().item())
                ids = torch.nonzero(usable, as_tuple=False).flatten()
                for idx in ids.tolist():
                    samples.append(
                        {
                            "horizon": float(horizon),
                            "class_id": int(class_id[idx].item()),
                            "soft_iou": float(soft_iou[idx, hi].item()),
                            "hard_iou": float(hard_iou[idx, hi].item()),
                            "error_m": float(error_m[idx, hi].item()),
                            "area": float(area[idx].item()),
                        }
                    )

    result = aggregate_proxy_samples(samples)
    result.update(
        {
            "epoch": int(meta["epoch"]),
            "variant": str(meta["variant"]),
            "checkpoint": str(meta["path"].resolve()),
            "true_moving_report_labels": int(total_tm),
            "footprint_usable_fraction": total_present / max(total_tm, 1),
        }
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def compare_to_truth(rows: list[dict], truth: dict[int, dict]) -> list[dict]:
    matched = [r for r in rows if int(r["epoch"]) in truth]
    if len(matched) < 2:
        return []
    true_values = [float(truth[int(r["epoch"])]["moving_mIoU"]) for r in matched]
    true_best = int(matched[int(np.argmax(true_values))]["epoch"])
    true_best_value = float(max(true_values))

    proxy_names = [
        "tm_soft_iou_micro",
        "tm_soft_iou_horizon_macro",
        "tm_soft_iou_class_horizon_macro",
        "tm_soft_iou_area_horizon_macro",
        "tm_hard_iou_class_horizon_macro",
        "tm_hit_0p4_class_horizon_macro",
        "tm_hit_0p8_class_horizon_macro",
        "tm_hit_1p2_class_horizon_macro",
        "neg_tm_ade_class_horizon_macro",
    ]
    comparisons = []
    for name in proxy_names:
        x = [float(r[name]) for r in matched]
        selected_idx = int(np.argmax(x))
        selected_epoch = int(matched[selected_idx]["epoch"])
        selected_truth = float(truth[selected_epoch]["moving_mIoU"])
        comparisons.append(
            {
                "proxy": name,
                "n": len(matched),
                "pearson": _corr(x, true_values),
                "spearman": _corr(_rankdata(np.asarray(x)), _rankdata(np.asarray(true_values))),
                "selected_epoch": selected_epoch,
                "true_best_epoch": true_best,
                "selects_true_best": bool(selected_epoch == true_best),
                "selection_regret_moving": true_best_value - selected_truth,
            }
        )
    comparisons.sort(
        key=lambda r: (
            int(bool(r["selects_true_best"])),
            -1e9 if not np.isfinite(float(r["spearman"])) else float(r["spearman"]),
        ),
        reverse=True,
    )
    return comparisons


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--val-cache", required=True)
    p.add_argument("--checkpoint", nargs="+", required=True)
    p.add_argument("--eval-json", nargs="*", default=[])
    p.add_argument("--output", required=True)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="cuda")
    p.add_argument("--branch", default="local_stwm_center_always")
    p.add_argument("--keep-duplicate-epochs", action="store_true")
    a = p.parse_args()
    if a.batch_size <= 0 or a.num_workers < 0:
        raise ValueError("invalid loader arguments")

    cache_meta, records = load_cache(a.val_cache)
    flat = flatten_supervised(records)
    patch_resolution = float(cache_meta.get("patch_resolution_m", 0.8))
    dataset = make_dataset(flat)
    loader = DataLoader(
        dataset,
        batch_size=int(a.batch_size),
        shuffle=False,
        num_workers=int(a.num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    device = torch.device(a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu")

    metas = [_checkpoint_metadata(Path(x)) for x in a.checkpoint]
    variants = sorted(set(m["variant"] for m in metas))
    if len(variants) != 1:
        raise RuntimeError(
            f"checkpoint selection diagnostic must stay within one variant/run; got variants={variants}"
        )
    if not a.keep_duplicate_epochs:
        unique = {}
        for m in metas:
            epoch = int(m["epoch"])
            if epoch not in unique:
                unique[epoch] = m
        metas = [unique[e] for e in sorted(unique)]
    else:
        metas.sort(key=lambda m: (int(m["epoch"]), str(m["path"])))

    truth = _truth_from_eval_jsons(a.eval_json, a.branch) if a.eval_json else {}
    rows = []
    print("\n=== V17 CHECKPOINT PROXY DIAGNOSTIC ===")
    for i, meta in enumerate(metas, start=1):
        print(f"proxy {i}/{len(metas)} epoch={meta['epoch']} checkpoint={meta['path']}", flush=True)
        row = evaluate_checkpoint(meta, loader, device, patch_resolution)
        if int(row["epoch"]) in truth:
            row["full_moving_mIoU"] = float(truth[int(row["epoch"])]["moving_mIoU"])
            row["full_occupancy_IoU"] = float(truth[int(row["epoch"])]["occupancy_IoU"])
        rows.append(row)

    print(
        f"{'ep':>3s} {'Moving':>8s} {'softCH':>8s} {'hardCH':>8s} {'softArea':>8s} "
        f"{'hit0.4':>8s} {'hit0.8':>8s} {'-ADEch':>8s} {'cover':>7s}"
    )
    for r in rows:
        mv = float(r.get("full_moving_mIoU", float("nan")))
        print(
            f"{int(r['epoch']):3d} {mv:8.4f} {100*float(r['tm_soft_iou_class_horizon_macro']):8.3f} "
            f"{100*float(r['tm_hard_iou_class_horizon_macro']):8.3f} "
            f"{100*float(r['tm_soft_iou_area_horizon_macro']):8.3f} "
            f"{100*float(r['tm_hit_0p4_class_horizon_macro']):8.3f} "
            f"{100*float(r['tm_hit_0p8_class_horizon_macro']):8.3f} "
            f"{float(r['neg_tm_ade_class_horizon_macro']):8.4f} "
            f"{100*float(r['footprint_usable_fraction']):6.2f}%"
        )

    comparisons = compare_to_truth(rows, truth)
    if comparisons:
        print("\n=== PROXY vs FULL Moving-mIoU ===")
        print(f"{'proxy':40s} {'n':>3s} {'Pearson':>8s} {'Spearman':>9s} {'pick':>5s} {'true':>5s} {'regret':>8s}")
        for r in comparisons:
            print(
                f"{r['proxy']:40s} {int(r['n']):3d} {float(r['pearson']):8.4f} "
                f"{float(r['spearman']):9.4f} {int(r['selected_epoch']):5d} "
                f"{int(r['true_best_epoch']):5d} {float(r['selection_regret_moving']):8.4f}"
            )
        best = comparisons[0]
        print(
            f"\nrecommended_proxy={best['proxy']} selected_epoch={best['selected_epoch']} "
            f"true_best_epoch={best['true_best_epoch']} n_full_eval={best['n']}"
        )
        if int(best["n"]) < 4:
            print("WARNING: fewer than 4 unique full-eval epochs; ranking evidence is preliminary.")
    else:
        print("\nNo sufficient full-eval JSON truth was supplied; proxy values were computed but not validated.")

    result = {
        "protocol": PROTOCOL,
        "val_cache": str(Path(a.val_cache).resolve()),
        "variant": variants[0],
        "patch_resolution_m": patch_resolution,
        "report_horizons": list(REPORT_INDEX.keys()),
        "rows": rows,
        "truth_by_epoch": truth,
        "comparisons": comparisons,
        "proxy_selection_target": "full local_stwm_center_always Moving-mIoU",
        "proxy_uses_future_occupancy": False,
    }
    op = Path(a.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"saved {op}")


if __name__ == "__main__":
    main()
