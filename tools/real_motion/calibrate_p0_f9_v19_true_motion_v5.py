#!/usr/bin/env python3
"""Cache-only calibration/diagnosis for V19 true-motion endpoint v5.

Evaluates one checkpoint or all epoch checkpoints without re-running Clean-E14,
NuScenes I/O, history alignment, or occupancy metrics.

Important distinction:
- supervised/candidate metrics use the training candidate mask;
- deployment metrics DO NOT apply candidate_mask. Predictions outside the
  supervised candidate domain therefore count as false positives, matching
  inference more closely and exposing responsibility leakage.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from real_motion.v19_innovation_training import (
    dequantize_geometry_torch,
    unpack_vertical_occupancy_torch,
)
from real_motion.v19_innovation_v5 import (
    ResidualInnovationEndpointHead,
    decode_ordered_endpoints,
)
from tools.real_motion.train_p0_f9_v19_innovation import (
    PROTOCOL,
    _iter_batches,
    _load_index,
)


def _floats(text: str) -> tuple[float, ...]:
    vals = tuple(
        float(x.strip()) for x in str(text).split(",") if x.strip()
    )
    if not vals:
        raise ValueError("empty threshold list")
    return vals


def _f_beta(p: float, r: float, beta: float) -> float:
    b2 = float(beta) ** 2
    return (1.0 + b2) * p * r / max(b2 * p + r, 1e-12)


def _metrics(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return {
        "precision": float(p),
        "recall": float(r),
        "f0_5": float(_f_beta(p, r, 0.5)),
        "f1": float(_f_beta(p, r, 1.0)),
    }


def _checkpoint_paths(args) -> list[Path]:
    if args.checkpoint:
        return [Path(args.checkpoint)]
    root = Path(args.checkpoint_dir)
    rows = sorted(root.glob("epoch_*.pt"))
    if not rows:
        raise FileNotFoundError(
            f"no epoch_*.pt checkpoints under {root}"
        )
    return rows


def _evaluate_checkpoint(
    ckpt_path: Path,
    *,
    cache_root: Path,
    cache_index: dict,
    thresholds: tuple[float, ...],
    batch_size: int,
    device: torch.device,
) -> dict:
    ck = torch.load(
        ckpt_path,
        map_location="cpu",
        weights_only=False,
    )
    if ck.get("protocol") != PROTOCOL:
        raise RuntimeError(
            f"unexpected checkpoint protocol: {ck.get('protocol')}"
        )
    if ck.get("head_type") != "vertical_endpoints_v5":
        raise RuntimeError(
            f"expected vertical_endpoints_v5, got {ck.get('head_type')}"
        )

    arch = dict(ck["architecture"])
    z_bins = int(arch["vertical_bins"])
    model = ResidualInnovationEndpointHead(**arch).to(device)
    model.load_state_dict(
        ck["innovation_state_dict"],
        strict=True,
    )
    model.eval()

    # Thresholded BEV/3D stats. "deployment" intentionally does not mask cand.
    rows = {
        th: {
            "candidate_bev_tp": 0,
            "candidate_bev_fp": 0,
            "candidate_bev_fn": 0,
            "deployment_bev_tp": 0,
            "deployment_bev_fp": 0,
            "deployment_bev_fn": 0,
            "deployment_voxel_tp": 0,
            "deployment_voxel_fp": 0,
            "deployment_voxel_fn": 0,
            "pred_bev_inside_candidate": 0,
            "pred_bev_outside_candidate": 0,
            "predicted_voxels": 0,
            "target_voxels": 0,
        }
        for th in thresholds
    }

    sem_correct = 0
    sem_total = 0
    sem_target_hist = np.zeros(17, dtype=np.int64)
    sem_pred_hist = np.zeros(17, dtype=np.int64)

    bottom_exact = top_exact = geom_n = 0
    bottom_within1 = top_within1 = 0
    bottom_abs_error = top_abs_error = 0
    interval_tp = interval_fp = interval_fn = 0
    target_span_bins = pred_span_bins = 0
    contiguous_cols = 0
    positive_voxels = span_voxels = 0

    prob_summary = {
        "positive_sum": 0.0,
        "positive_n": 0,
        "negative_sum": 0.0,
        "negative_n": 0,
        "outside_candidate_sum": 0.0,
        "outside_candidate_n": 0,
    }

    with torch.inference_mode():
        for raw in _iter_batches(
            cache_root,
            cache_index,
            batch_size=int(batch_size),
            shuffle=False,
            seed=0,
        ):
            sem = raw["future_aligned_semantic"].to(
                device, non_blocking=True
            )
            geo = dequantize_geometry_torch(
                raw["future_aligned_geometry_q"].to(
                    device, non_blocking=True
                )
            )
            base = raw["base_explained"].to(
                device, non_blocking=True
            ).float()
            add_tgt = raw["add_target"].to(
                device, non_blocking=True
            ).bool()
            cand = raw["candidate_mask"].to(
                device, non_blocking=True
            ).bool()
            sem_tgt = raw["semantic_target"].to(
                device, non_blocking=True
            ).long()
            z_tgt = unpack_vertical_occupancy_torch(
                raw["vertical_bits"].to(
                    device, non_blocking=True
                ),
                z_bins,
            ).permute(0, 1, 3, 4, 2).bool()

            out = model(sem, geo, base)
            add_prob = torch.sigmoid(
                out["add_presence_logits"].float()
            )
            pos = add_tgt & cand
            target_3d = z_tgt & pos[..., None]

            # Presence probability scale diagnostics.
            if bool(pos.any()):
                prob_summary["positive_sum"] += float(
                    add_prob[pos].sum().item()
                )
                prob_summary["positive_n"] += int(pos.sum().item())
            neg = cand & ~pos
            if bool(neg.any()):
                prob_summary["negative_sum"] += float(
                    add_prob[neg].sum().item()
                )
                prob_summary["negative_n"] += int(neg.sum().item())
            outside = ~cand
            if bool(outside.any()):
                prob_summary["outside_candidate_sum"] += float(
                    add_prob[outside].sum().item()
                )
                prob_summary["outside_candidate_n"] += int(
                    outside.sum().item()
                )

            # Semantic and endpoint geometry, conditioned on true positive BEV.
            if bool(pos.any()):
                sem_pred = out["semantic_logits"].argmax(dim=2)
                sem_correct += int(
                    (sem_pred[pos] == sem_tgt[pos]).sum().item()
                )
                sem_total += int(pos.sum().item())
                st = sem_tgt[pos]
                sp = sem_pred[pos]
                sem_target_hist += np.bincount(
                    st.detach().cpu().numpy(),
                    minlength=17,
                )[:17]
                sem_pred_hist += np.bincount(
                    sp.detach().cpu().numpy(),
                    minlength=17,
                )[:17]

                z_true = z_tgt[pos]
                idx = torch.arange(
                    z_bins,
                    device=device,
                    dtype=torch.long,
                )[None, :]
                b_true = torch.where(
                    z_true,
                    idx,
                    torch.full_like(idx, z_bins),
                ).min(dim=1).values
                t_true = torch.where(
                    z_true,
                    idx,
                    torch.full_like(idx, -1),
                ).max(dim=1).values

                b_full, t_full = decode_ordered_endpoints(
                    out["bottom_logits"],
                    out["top_logits"],
                )
                b_pred = b_full[pos]
                t_pred = t_full[pos]

                db = (b_pred - b_true).abs()
                dt = (t_pred - t_true).abs()
                bottom_exact += int((db == 0).sum().item())
                top_exact += int((dt == 0).sum().item())
                bottom_within1 += int((db <= 1).sum().item())
                top_within1 += int((dt <= 1).sum().item())
                bottom_abs_error += int(db.sum().item())
                top_abs_error += int(dt.sum().item())
                geom_n += int(z_true.shape[0])

                z_pred = (
                    (idx >= b_pred[:, None])
                    & (idx <= t_pred[:, None])
                )
                interval_tp += int((z_pred & z_true).sum().item())
                interval_fp += int((z_pred & ~z_true).sum().item())
                interval_fn += int((~z_pred & z_true).sum().item())

                counts = z_true.sum(dim=1)
                true_span = t_true - b_true + 1
                pred_span = t_pred - b_pred + 1
                positive_voxels += int(counts.sum().item())
                span_voxels += int(true_span.sum().item())
                contiguous_cols += int(
                    (counts == true_span).sum().item()
                )
                target_span_bins += int(true_span.sum().item())
                pred_span_bins += int(pred_span.sum().item())
            else:
                b_full, t_full = decode_ordered_endpoints(
                    out["bottom_logits"],
                    out["top_logits"],
                )

            zidx = torch.arange(
                z_bins,
                device=device,
            ).view(1, 1, 1, 1, z_bins)
            interval_all = (
                (zidx >= b_full[..., None])
                & (zidx <= t_full[..., None])
            )

            for th in thresholds:
                pred_all = add_prob >= float(th)
                pred_cand = pred_all & cand

                rr = rows[th]
                rr["pred_bev_inside_candidate"] += int(
                    (pred_all & cand).sum().item()
                )
                rr["pred_bev_outside_candidate"] += int(
                    (pred_all & ~cand).sum().item()
                )

                rr["candidate_bev_tp"] += int(
                    (pred_cand & pos).sum().item()
                )
                rr["candidate_bev_fp"] += int(
                    (pred_cand & ~pos & cand).sum().item()
                )
                rr["candidate_bev_fn"] += int(
                    (~pred_cand & pos).sum().item()
                )

                # Deployment: no candidate mask is available at inference.
                rr["deployment_bev_tp"] += int(
                    (pred_all & pos).sum().item()
                )
                rr["deployment_bev_fp"] += int(
                    (pred_all & ~pos).sum().item()
                )
                rr["deployment_bev_fn"] += int(
                    (~pred_all & pos).sum().item()
                )

                pred_3d = interval_all & pred_all[..., None]
                rr["deployment_voxel_tp"] += int(
                    (pred_3d & target_3d).sum().item()
                )
                rr["deployment_voxel_fp"] += int(
                    (pred_3d & ~target_3d).sum().item()
                )
                rr["deployment_voxel_fn"] += int(
                    (~pred_3d & target_3d).sum().item()
                )
                rr["predicted_voxels"] += int(
                    pred_3d.sum().item()
                )
                rr["target_voxels"] += int(
                    target_3d.sum().item()
                )

    table = []
    for th in thresholds:
        rr = rows[th]
        cand_m = _metrics(
            rr["candidate_bev_tp"],
            rr["candidate_bev_fp"],
            rr["candidate_bev_fn"],
        )
        dep_bev = _metrics(
            rr["deployment_bev_tp"],
            rr["deployment_bev_fp"],
            rr["deployment_bev_fn"],
        )
        dep_vox = _metrics(
            rr["deployment_voxel_tp"],
            rr["deployment_voxel_fp"],
            rr["deployment_voxel_fn"],
        )
        inside = rr["pred_bev_inside_candidate"]
        outside_n = rr["pred_bev_outside_candidate"]
        table.append(
            {
                "add_threshold": float(th),
                "candidate_bev": cand_m,
                "deployment_bev": dep_bev,
                "deployment_voxel": dep_vox,
                "outside_candidate_fraction": float(
                    outside_n / max(inside + outside_n, 1)
                ),
                "predicted_bev_inside_candidate": int(inside),
                "predicted_bev_outside_candidate": int(outside_n),
                "predicted_voxels": int(rr["predicted_voxels"]),
                "target_voxels": int(rr["target_voxels"]),
                "prediction_to_target_voxel_ratio": float(
                    rr["predicted_voxels"]
                    / max(rr["target_voxels"], 1)
                ),
            }
        )

    interval_m = _metrics(
        interval_tp,
        interval_fp,
        interval_fn,
    )
    target_majority = (
        int(sem_target_hist.max())
        / max(int(sem_target_hist.sum()), 1)
    )
    best = max(
        table,
        key=lambda x: (
            x["deployment_voxel"]["f0_5"],
            x["deployment_voxel"]["precision"],
            x["deployment_voxel"]["recall"],
        ),
    )

    result = {
        "protocol": "p0_f9_v19_true_motion_v5_cache_calibration_v1",
        "checkpoint": str(ckpt_path.resolve()),
        "checkpoint_epoch": int(ck.get("epoch", -1)),
        "num_windows": int(cache_index["num_windows"]),
        "positive_mode": cache_index.get("positive_mode"),
        "presence_probability_means": {
            "positive": float(
                prob_summary["positive_sum"]
                / max(prob_summary["positive_n"], 1)
            ),
            "candidate_negative": float(
                prob_summary["negative_sum"]
                / max(prob_summary["negative_n"], 1)
            ),
            "outside_candidate": float(
                prob_summary["outside_candidate_sum"]
                / max(prob_summary["outside_candidate_n"], 1)
            ),
        },
        "semantic": {
            "accuracy": float(
                sem_correct / max(sem_total, 1)
            ),
            "majority_class_baseline": float(target_majority),
            "target_class_histogram": sem_target_hist.tolist(),
            "predicted_class_histogram": sem_pred_hist.tolist(),
        },
        "endpoint_geometry_on_gt_positive_bev": {
            "positive_columns": int(geom_n),
            "bottom_exact_accuracy": float(
                bottom_exact / max(geom_n, 1)
            ),
            "top_exact_accuracy": float(
                top_exact / max(geom_n, 1)
            ),
            "bottom_within_1bin_accuracy": float(
                bottom_within1 / max(geom_n, 1)
            ),
            "top_within_1bin_accuracy": float(
                top_within1 / max(geom_n, 1)
            ),
            "bottom_mae_bins": float(
                bottom_abs_error / max(geom_n, 1)
            ),
            "top_mae_bins": float(
                top_abs_error / max(geom_n, 1)
            ),
            "interval_voxel": interval_m,
            "target_contiguous_fraction": float(
                contiguous_cols / max(geom_n, 1)
            ),
            "target_fill_fraction_inside_minmax_span": float(
                positive_voxels / max(span_voxels, 1)
            ),
            "mean_target_span_bins": float(
                target_span_bins / max(geom_n, 1)
            ),
            "mean_predicted_span_bins": float(
                pred_span_bins / max(geom_n, 1)
            ),
        },
        "best_by_deployment_voxel_f0_5": best,
        "thresholds": table,
    }
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--val-cache", required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint")
    group.add_argument("--checkpoint-dir")
    p.add_argument("--output", required=True)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument(
        "--add-thresholds",
        default="0.02,0.05,0.075,0.10,0.125,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50",
    )
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    thresholds = _floats(a.add_thresholds)
    cache_root, cache_index = _load_index(a.val_cache)
    if cache_index.get("positive_mode") != "true_motion_shape":
        raise RuntimeError(
            "calibration requires true_motion_shape cache"
        )
    device = torch.device(
        a.device
        if a.device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )

    results = []
    for path in _checkpoint_paths(a):
        row = _evaluate_checkpoint(
            path,
            cache_root=cache_root,
            cache_index=cache_index,
            thresholds=thresholds,
            batch_size=int(a.batch_size),
            device=device,
        )
        results.append(row)
        best = row["best_by_deployment_voxel_f0_5"]
        geom = row["endpoint_geometry_on_gt_positive_bev"]
        sem = row["semantic"]
        means = row["presence_probability_means"]
        print(
            f"epoch={row['checkpoint_epoch']:2d} "
            f"prob(pos/neg/out)="
            f"{means['positive']:.3f}/"
            f"{means['candidate_negative']:.3f}/"
            f"{means['outside_candidate']:.3f} "
            f"sem={sem['accuracy']:.3f} "
            f"sem_majority={sem['majority_class_baseline']:.3f} "
            f"b/t exact={geom['bottom_exact_accuracy']:.3f}/"
            f"{geom['top_exact_accuracy']:.3f} "
            f"within1={geom['bottom_within_1bin_accuracy']:.3f}/"
            f"{geom['top_within_1bin_accuracy']:.3f} "
            f"span(gt/pred)={geom['mean_target_span_bins']:.2f}/"
            f"{geom['mean_predicted_span_bins']:.2f} "
            f"best_th={best['add_threshold']:.3f} "
            f"3D P/R/F0.5="
            f"{best['deployment_voxel']['precision']:.3f}/"
            f"{best['deployment_voxel']['recall']:.3f}/"
            f"{best['deployment_voxel']['f0_5']:.3f} "
            f"leak={best['outside_candidate_fraction']:.3f}"
        )

    global_best = max(
        results,
        key=lambda x: (
            x["best_by_deployment_voxel_f0_5"][
                "deployment_voxel"
            ]["f0_5"],
            x["best_by_deployment_voxel_f0_5"][
                "deployment_voxel"
            ]["precision"],
        ),
    )
    output = {
        "protocol": "p0_f9_v19_true_motion_v5_cache_calibration_sweep_v1",
        "num_checkpoints": len(results),
        "results": results,
        "global_best": {
            "checkpoint": global_best["checkpoint"],
            "checkpoint_epoch": global_best["checkpoint_epoch"],
            "operating_point": global_best[
                "best_by_deployment_voxel_f0_5"
            ],
        },
    }
    op = Path(a.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    gb = output["global_best"]
    bp = gb["operating_point"]
    print("\n=== GLOBAL BEST CACHE OPERATING POINT ===")
    print(
        f"epoch={gb['checkpoint_epoch']} "
        f"threshold={bp['add_threshold']:.3f} "
        f"BEV P/R="
        f"{bp['deployment_bev']['precision']:.3f}/"
        f"{bp['deployment_bev']['recall']:.3f} "
        f"3D P/R/F0.5="
        f"{bp['deployment_voxel']['precision']:.3f}/"
        f"{bp['deployment_voxel']['recall']:.3f}/"
        f"{bp['deployment_voxel']['f0_5']:.3f} "
        f"leak={bp['outside_candidate_fraction']:.3f}"
    )
    print(f"saved {op}")


if __name__ == "__main__":
    main()
