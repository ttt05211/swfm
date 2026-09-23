#!/usr/bin/env python3
"""Fast cache-only calibration sweep for a trained V19 Innovation checkpoint.

This avoids re-running Clean-E14, NuScenes I/O, geometry warps, or occupancy
metrics. It evaluates add/vertical thresholds directly on the frozen
Innovation validation cache and reports joint 3D innovation precision/recall.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from real_motion.v19_innovation import ResidualInnovationHead
from real_motion.v19_innovation_training import (
    dequantize_geometry_torch,
    unpack_vertical_occupancy_torch,
)
from tools.real_motion.train_p0_f9_v19_innovation import (
    PROTOCOL,
    _iter_batches,
    _load_index,
)


def _floats(text):
    vals = tuple(float(x.strip()) for x in str(text).split(",") if x.strip())
    if not vals:
        raise ValueError("empty threshold list")
    return vals


def _f_beta(p, r, beta):
    b2 = float(beta) ** 2
    return (1.0 + b2) * p * r / max(b2 * p + r, 1e-12)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--val-cache", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument(
        "--add-thresholds",
        default="0.3,0.4,0.5,0.6,0.7,0.8,0.9",
    )
    p.add_argument(
        "--vertical-thresholds",
        default="0.1,0.2,0.3,0.4,0.5",
    )
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    add_ts = _floats(a.add_thresholds)
    z_ts = _floats(a.vertical_thresholds)
    root, index = _load_index(a.val_cache)
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    if ck.get("protocol") != PROTOCOL:
        raise RuntimeError(
            f"unexpected checkpoint protocol: {ck.get('protocol')}"
        )
    arch = dict(ck["architecture"])
    z_bins = int(arch["vertical_bins"])
    device = torch.device(
        a.device
        if a.device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    model = ResidualInnovationHead(**arch).to(device)
    model.load_state_dict(ck["innovation_state_dict"], strict=True)
    model.eval()

    rows = {
        (at, zt): {
            "bev_tp": 0,
            "bev_fp": 0,
            "bev_fn": 0,
            "voxel_tp": 0,
            "voxel_fp": 0,
            "voxel_fn": 0,
            "predicted_voxels": 0,
            "target_voxels": 0,
        }
        for at in add_ts
        for zt in z_ts
    }
    semantic_correct = 0
    semantic_total = 0
    vertical_only = {
        zt: {"tp": 0, "fp": 0, "fn": 0}
        for zt in z_ts
    }
    vertical_positive_columns = 0
    vertical_contiguous_columns = 0
    vertical_span_voxels = 0
    vertical_positive_voxels = 0
    presence_leakage = {
        at: {"inside": 0, "outside": 0}
        for at in add_ts
    }

    with torch.inference_mode():
        for raw in _iter_batches(
            root,
            index,
            batch_size=int(a.batch_size),
            shuffle=False,
            seed=0,
        ):
            sem = raw["future_aligned_semantic"].to(device)
            geo = dequantize_geometry_torch(
                raw["future_aligned_geometry_q"].to(device)
            )
            base = raw["base_explained"].to(device).float()
            add_tgt = raw["add_target"].to(device).bool()
            cand = raw["candidate_mask"].to(device).bool()
            sem_tgt = raw["semantic_target"].to(device).long()
            z_tgt = unpack_vertical_occupancy_torch(
                raw["vertical_bits"].to(device), z_bins
            ).permute(0, 1, 3, 4, 2).bool()

            out = model(sem, geo, base)
            add_prob = torch.sigmoid(
                out["add_presence_logits"].float()
            )
            z_prob = torch.sigmoid(
                out["vertical_occupancy_logits"].float()
            ).permute(0, 1, 3, 4, 2)
            sem_pred = out["semantic_logits"].argmax(dim=2)
            pos = add_tgt & cand
            if bool(pos.any()):
                semantic_correct += int(
                    (sem_pred[pos] == sem_tgt[pos]).sum().item()
                )
                semantic_total += int(pos.sum().item())

            target_3d = z_tgt & pos[..., None]
            if bool(pos.any()):
                z_true_rows = z_tgt[pos]
                vertical_positive_columns += int(z_true_rows.shape[0])
                vertical_positive_voxels += int(z_true_rows.sum().item())
                idx = torch.arange(
                    z_bins, device=device, dtype=torch.long
                )[None]
                first = torch.where(
                    z_true_rows,
                    idx,
                    torch.full_like(idx, z_bins),
                ).min(dim=1).values
                last = torch.where(
                    z_true_rows,
                    idx,
                    torch.full_like(idx, -1),
                ).max(dim=1).values
                span = (last - first + 1).clamp_min(0)
                counts = z_true_rows.sum(dim=1)
                vertical_span_voxels += int(span.sum().item())
                vertical_contiguous_columns += int(
                    (counts == span).sum().item()
                )
                z_prob_rows = z_prob[pos]
                for zt in z_ts:
                    zp = z_prob_rows >= float(zt)
                    rr = vertical_only[zt]
                    rr["tp"] += int((zp & z_true_rows).sum().item())
                    rr["fp"] += int((zp & ~z_true_rows).sum().item())
                    rr["fn"] += int((~zp & z_true_rows).sum().item())

            for at in add_ts:
                add_pred_all = add_prob >= float(at)
                presence_leakage[at]["inside"] += int(
                    (add_pred_all & cand).sum().item()
                )
                presence_leakage[at]["outside"] += int(
                    (add_pred_all & ~cand).sum().item()
                )
                add_pred = add_pred_all & cand
                bev_tp = add_pred & pos
                bev_fp = add_pred & ~pos & cand
                bev_fn = ~add_pred & pos
                for zt in z_ts:
                    z_pred = z_prob >= float(zt)
                    pred_3d = z_pred & add_pred[..., None]
                    rr = rows[(at, zt)]
                    rr["bev_tp"] += int(bev_tp.sum().item())
                    rr["bev_fp"] += int(bev_fp.sum().item())
                    rr["bev_fn"] += int(bev_fn.sum().item())
                    rr["voxel_tp"] += int(
                        (pred_3d & target_3d).sum().item()
                    )
                    rr["voxel_fp"] += int(
                        (pred_3d & ~target_3d).sum().item()
                    )
                    rr["voxel_fn"] += int(
                        (~pred_3d & target_3d).sum().item()
                    )
                    rr["predicted_voxels"] += int(pred_3d.sum().item())
                    rr["target_voxels"] += int(target_3d.sum().item())

    table = []
    for (at, zt), rr in rows.items():
        bp = rr["bev_tp"] / max(rr["bev_tp"] + rr["bev_fp"], 1)
        br = rr["bev_tp"] / max(rr["bev_tp"] + rr["bev_fn"], 1)
        vp = rr["voxel_tp"] / max(
            rr["voxel_tp"] + rr["voxel_fp"], 1
        )
        vr = rr["voxel_tp"] / max(
            rr["voxel_tp"] + rr["voxel_fn"], 1
        )
        table.append(
            {
                "add_threshold": float(at),
                "vertical_threshold": float(zt),
                "bev_precision": float(bp),
                "bev_recall": float(br),
                "bev_f0_5": float(_f_beta(bp, br, 0.5)),
                "voxel_precision": float(vp),
                "voxel_recall": float(vr),
                "voxel_f0_5": float(_f_beta(vp, vr, 0.5)),
                "voxel_f1": float(_f_beta(vp, vr, 1.0)),
                "predicted_voxels": int(rr["predicted_voxels"]),
                "target_voxels": int(rr["target_voxels"]),
                "prediction_to_target_ratio": float(
                    rr["predicted_voxels"] / max(rr["target_voxels"], 1)
                ),
            }
        )

    vertical_table = []
    for zt, rr in vertical_only.items():
        vp = rr["tp"] / max(rr["tp"] + rr["fp"], 1)
        vr = rr["tp"] / max(rr["tp"] + rr["fn"], 1)
        vertical_table.append(
            {
                "vertical_threshold": float(zt),
                "precision_on_gt_positive_bev": float(vp),
                "recall_on_gt_positive_bev": float(vr),
                "f1_on_gt_positive_bev": float(_f_beta(vp, vr, 1.0)),
            }
        )
    leakage_table = []
    for at, rr in presence_leakage.items():
        total_pred = rr["inside"] + rr["outside"]
        leakage_table.append(
            {
                "add_threshold": float(at),
                "predicted_bev_inside_supervised_candidate": int(rr["inside"]),
                "predicted_bev_outside_supervised_candidate": int(rr["outside"]),
                "outside_fraction": float(
                    rr["outside"] / max(total_pred, 1)
                ),
            }
        )

    by_f05 = sorted(
        table,
        key=lambda x: (
            x["voxel_f0_5"],
            x["voxel_precision"],
            x["voxel_recall"],
        ),
        reverse=True,
    )
    result = {
        "protocol": "p0_f9_v19_innovation_cache_calibration_v1",
        "checkpoint": str(Path(a.checkpoint).resolve()),
        "checkpoint_epoch": int(ck.get("epoch", -1)),
        "objective_version": ck.get("objective_version"),
        "num_windows": int(index["num_windows"]),
        "positive_mode": index.get("positive_mode"),
        "semantic_accuracy_on_positive_bev": float(
            semantic_correct / max(semantic_total, 1)
        ),
        "vertical_target_geometry": {
            "positive_columns": int(vertical_positive_columns),
            "contiguous_fraction": float(
                vertical_contiguous_columns
                / max(vertical_positive_columns, 1)
            ),
            "mean_positive_bins": float(
                vertical_positive_voxels
                / max(vertical_positive_columns, 1)
            ),
            "mean_span_bins": float(
                vertical_span_voxels
                / max(vertical_positive_columns, 1)
            ),
            "fill_fraction_inside_minmax_span": float(
                vertical_positive_voxels
                / max(vertical_span_voxels, 1)
            ),
        },
        "vertical_only_on_gt_positive_bev": vertical_table,
        "presence_responsibility_leakage": leakage_table,
        "best_by_joint_voxel_f0_5": by_f05[0],
        "top10_by_joint_voxel_f0_5": by_f05[:10],
        "all": table,
    }
    op = Path(a.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("=== V19 INNOVATION CACHE CALIBRATION ===")
    print(
        "semantic_accuracy_on_positive_bev="
        f"{result['semantic_accuracy_on_positive_bev']:.4f}"
    )
    print(
        "vertical_target_geometry="
        + json.dumps(result["vertical_target_geometry"])
    )
    print("vertical_only:")
    for row in vertical_table:
        print(
            "  z={vertical_threshold:.3f} P/R={precision_on_gt_positive_bev:.3f}/"
            "{recall_on_gt_positive_bev:.3f} F1={f1_on_gt_positive_bev:.3f}"
            .format(**row)
        )
    print("presence_responsibility_leakage:")
    for row in leakage_table:
        print(
            "  add={add_threshold:.2f} outside_fraction={outside_fraction:.3f}"
            .format(**row)
        )
    for row in by_f05[:10]:
        print(
            "add={add_threshold:.2f} z={vertical_threshold:.2f} "
            "BEV P/R={bev_precision:.3f}/{bev_recall:.3f} "
            "3D P/R={voxel_precision:.3f}/{voxel_recall:.3f} "
            "F0.5={voxel_f0_5:.3f} pred/gt={prediction_to_target_ratio:.2f}"
            .format(**row)
        )
    print(f"saved {op}")


if __name__ == "__main__":
    main()
