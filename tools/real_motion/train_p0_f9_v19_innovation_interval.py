#!/usr/bin/env python3
"""V19 dynamic-Innovation v4: hard-negative BEV + compact vertical interval.

Reuses the existing dynamic-only Innovation caches.  No cache rebuild is
required.  The vertical target is represented by bottom z-bin plus contiguous
span length, matching the strong interval structure measured in the cached
targets.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from real_motion.v19_innovation import (
    InnovationLossWeights,
    ResidualInnovationIntervalHead,
    innovation_interval_loss,
)
from real_motion.v19_innovation_training import (
    dequantize_geometry_torch,
    unpack_vertical_occupancy_torch,
)
from tools.real_motion.train_p0_f9_v19_innovation import (
    PROTOCOL,
    _iter_batches,
    _load_index,
)

OBJECTIVE_VERSION = "hard_negative_presence_vertical_interval_v4"
HEAD_TYPE = "vertical_interval"


def _autocast(device, enabled):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _move(raw, device, vertical_bins):
    mv = lambda x: x.to(device, non_blocking=True)
    return {
        "semantic": mv(raw["future_aligned_semantic"]),
        "geometry": dequantize_geometry_torch(
            mv(raw["future_aligned_geometry_q"])
        ),
        "base": mv(raw["base_explained"]).float(),
        "add": mv(raw["add_target"]).bool(),
        "semantic_target": mv(raw["semantic_target"]).long(),
        "vertical": unpack_vertical_occupancy_torch(
            mv(raw["vertical_bits"]), vertical_bins
        ),
        "candidate": mv(raw["candidate_mask"]).bool(),
    }


def _f_beta(p, r, beta):
    b2 = float(beta) ** 2
    return (1.0 + b2) * p * r / max(b2 * p + r, 1e-12)


def _interval_rows(outputs, pos, vertical_target):
    z_true = vertical_target.permute(0, 1, 3, 4, 2)[pos].bool()
    Z = int(z_true.shape[1])
    idx = torch.arange(Z, device=z_true.device)[None]
    bottom_true = torch.where(
        z_true, idx, torch.full_like(idx, Z)
    ).min(dim=1).values
    top_true = torch.where(
        z_true, idx, torch.full_like(idx, -1)
    ).max(dim=1).values
    span_true = top_true - bottom_true

    bottom_pred = outputs["bottom_logits"].argmax(dim=2)[pos]
    span_pred = outputs["span_logits"].argmax(dim=2)[pos]
    top_pred = torch.clamp(
        bottom_pred + span_pred,
        max=Z - 1,
    )
    pred_z = (
        (idx >= bottom_pred[:, None])
        & (idx <= top_pred[:, None])
    )
    return (
        z_true,
        bottom_true,
        span_true,
        bottom_pred,
        span_pred,
        pred_z,
    )


def _evaluate(
    model,
    root,
    index,
    device,
    *,
    batch_size,
    vertical_bins,
    amp,
    weights,
    hard_negative_ratio,
):
    model.eval()
    sums = {
        "loss": 0.0,
        "add_bce": 0.0,
        "semantic_ce": 0.0,
        "bottom_ce": 0.0,
        "span_ce": 0.0,
        "vertical_interval_ce": 0.0,
    }
    nb = 0
    p_tp = p_fp = p_fn = 0
    sem_ok = sem_n = 0
    bottom_ok = span_ok = geom_n = 0
    z_tp = z_fp = z_fn = 0
    joint_tp = joint_fp = joint_fn = 0
    pred_inside = pred_outside = 0

    with torch.inference_mode():
        for raw in _iter_batches(
            root,
            index,
            batch_size=int(batch_size),
            shuffle=False,
            seed=0,
        ):
            b = _move(raw, device, vertical_bins)
            with _autocast(device, amp):
                out = model(
                    b["semantic"], b["geometry"], b["base"]
                )
                loss, stats = innovation_interval_loss(
                    out,
                    add_target=b["add"],
                    semantic_target=b["semantic_target"],
                    vertical_target=b["vertical"],
                    candidate_mask=b["candidate"],
                    weights=weights,
                    presence_hard_negative_ratio=float(
                        hard_negative_ratio
                    ),
                )
            for k in sums:
                sums[k] += float(stats[k])
            nb += 1

            cand = b["candidate"]
            pos = b["add"] & cand
            pred_all = (
                torch.sigmoid(out["add_presence_logits"].float()) >= 0.5
            )
            pred = pred_all & cand
            pred_inside += int(pred.sum().item())
            pred_outside += int((pred_all & ~cand).sum().item())
            p_tp += int((pred & pos).sum().item())
            p_fp += int((pred & ~pos & cand).sum().item())
            p_fn += int((~pred & pos).sum().item())

            if bool(pos.any()):
                sem = out["semantic_logits"].argmax(dim=2)
                sem_ok += int(
                    (sem[pos] == b["semantic_target"][pos]).sum().item()
                )
                sem_n += int(pos.sum().item())

                (
                    z_true,
                    bottom_true,
                    span_true,
                    bottom_pred,
                    span_pred,
                    pred_z_rows,
                ) = _interval_rows(out, pos, b["vertical"])
                bottom_ok += int(
                    (bottom_pred == bottom_true).sum().item()
                )
                span_ok += int(
                    (span_pred == span_true).sum().item()
                )
                geom_n += int(z_true.shape[0])
                z_tp += int((pred_z_rows & z_true).sum().item())
                z_fp += int((pred_z_rows & ~z_true).sum().item())
                z_fn += int((~pred_z_rows & z_true).sum().item())

            Z = int(vertical_bins)
            bottom_full = out["bottom_logits"].argmax(dim=2)
            span_full = out["span_logits"].argmax(dim=2)
            top_full = torch.clamp(
                bottom_full + span_full,
                max=Z - 1,
            )
            zidx = torch.arange(
                Z, device=device
            ).view(1, 1, 1, 1, Z)
            pred_3d = (
                (zidx >= bottom_full[..., None])
                & (zidx <= top_full[..., None])
                & pred[..., None]
            )
            tgt_3d = (
                b["vertical"].permute(0, 1, 3, 4, 2).bool()
                & pos[..., None]
            )
            joint_tp += int((pred_3d & tgt_3d).sum().item())
            joint_fp += int((pred_3d & ~tgt_3d).sum().item())
            joint_fn += int((~pred_3d & tgt_3d).sum().item())

    p_prec = p_tp / max(p_tp + p_fp, 1)
    p_rec = p_tp / max(p_tp + p_fn, 1)
    z_prec = z_tp / max(z_tp + z_fp, 1)
    z_rec = z_tp / max(z_tp + z_fn, 1)
    j_prec = joint_tp / max(joint_tp + joint_fp, 1)
    j_rec = joint_tp / max(joint_tp + joint_fn, 1)
    leakage = pred_outside / max(pred_inside + pred_outside, 1)
    return {
        **{k: v / max(nb, 1) for k, v in sums.items()},
        "presence_precision_at_0_5": float(p_prec),
        "presence_recall_at_0_5": float(p_rec),
        "presence_f0_5_at_0_5": float(_f_beta(p_prec, p_rec, 0.5)),
        "semantic_accuracy_on_positive_bev": float(
            sem_ok / max(sem_n, 1)
        ),
        "bottom_accuracy_on_positive_bev": float(
            bottom_ok / max(geom_n, 1)
        ),
        "span_accuracy_on_positive_bev": float(
            span_ok / max(geom_n, 1)
        ),
        "interval_voxel_precision_on_positive_bev": float(z_prec),
        "interval_voxel_recall_on_positive_bev": float(z_rec),
        "interval_voxel_f1_on_positive_bev": float(
            _f_beta(z_prec, z_rec, 1.0)
        ),
        "joint_voxel_precision_at_0_5": float(j_prec),
        "joint_voxel_recall_at_0_5": float(j_rec),
        "joint_voxel_f0_5_at_0_5": float(
            _f_beta(j_prec, j_rec, 0.5)
        ),
        "presence_outside_candidate_fraction_at_0_5": float(leakage),
    }


def _save(
    path,
    *,
    model,
    optimizer,
    epoch,
    global_step,
    arch,
    train_index,
    args,
    val_report,
):
    torch.save(
        {
            "protocol": PROTOCOL,
            "objective_version": OBJECTIVE_VERSION,
            "head_type": HEAD_TYPE,
            "training_mode": (
                "innovation_only_frozen_base_dynamic_interval_v4"
            ),
            "epoch": int(epoch),
            "global_step": int(global_step),
            "innovation_state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "architecture": arch,
            "positive_categories": list(
                train_index.get("positive_categories", [])
            ),
            "hard_negative_ratio": float(args.hard_negative_ratio),
            "base_checkpoint": train_index.get("base_checkpoint"),
            "base_checkpoint_epoch": train_index.get(
                "base_checkpoint_epoch"
            ),
            "explained_state": train_index.get("explained_state"),
            "val_report": val_report,
            "args": vars(args),
        },
        path,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-cache", required=True)
    p.add_argument("--val-cache", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--semantic-dim", type=int, default=8)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--semantic-weight", type=float, default=1.0)
    p.add_argument("--vertical-weight", type=float, default=1.0)
    p.add_argument("--hard-negative-ratio", type=float, default=4.0)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=20260923)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-amp", action="store_true")
    a = p.parse_args()

    if (
        a.epochs <= 0
        or a.batch_size <= 0
        or a.hard_negative_ratio <= 0
    ):
        raise ValueError("invalid training arguments")

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)

    train_root, train_index = _load_index(a.train_cache)
    val_root, val_index = _load_index(a.val_cache)
    if train_index.get("positive_mode") != "dynamic":
        raise RuntimeError("v4 requires positive-mode dynamic train cache")
    if val_index.get("positive_mode") != "dynamic":
        raise RuntimeError("v4 requires positive-mode dynamic val cache")
    overlap = sorted(
        set(train_index.get("scene_names", []))
        & set(val_index.get("scene_names", []))
    )
    if overlap:
        raise RuntimeError(
            f"innovation train/val scene overlap: {overlap[:5]}"
        )

    z = int(train_index["grid_shape_hwd"][2])
    arch = {
        "future_frames": int(train_index["future_frames"]),
        "history_frames": int(train_index["history_frames"]),
        "semantic_dim": int(a.semantic_dim),
        "hidden_dim": int(a.hidden_dim),
        "num_semantic_classes": 17,
        "vertical_bins": z,
    }
    device = torch.device(
        a.device
        if a.device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    amp = device.type == "cuda" and not bool(a.no_amp)
    model = ResidualInnovationIntervalHead(**arch).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(a.lr),
        weight_decay=float(a.weight_decay),
    )
    weights = InnovationLossWeights(
        semantic=float(a.semantic_weight),
        vertical=float(a.vertical_weight),
    )

    out_dir = Path(a.output_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(
            f"refusing non-empty output dir: {out_dir}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    initial_val = _evaluate(
        model,
        val_root,
        val_index,
        device,
        batch_size=a.batch_size,
        vertical_bins=z,
        amp=amp,
        weights=weights,
        hard_negative_ratio=a.hard_negative_ratio,
    )
    preflight = {
        "protocol": PROTOCOL,
        "objective_version": OBJECTIVE_VERSION,
        "head_type": HEAD_TYPE,
        "parameters": int(sum(x.numel() for x in model.parameters())),
        "clean_e14_parameters_in_optimizer": 0,
        "positive_categories": list(
            train_index.get("positive_categories", [])
        ),
        "hard_negative_ratio": float(a.hard_negative_ratio),
        "train_positive_bev_cells": int(
            train_index["totals"]["positive_bev_cells"]
        ),
        "train_candidate_bev_cells": int(
            train_index["totals"]["candidate_bev_cells"]
        ),
        "initial_val": initial_val,
    }
    (out_dir / "preflight.json").write_text(
        json.dumps(preflight, indent=2), encoding="utf-8"
    )
    print("=== V19 INNOVATION INTERVAL-V4 PREFLIGHT ===")
    print(json.dumps(preflight, indent=2))

    best_score = -1.0
    best_loss = float("inf")
    global_step = 0
    history = []
    for epoch in range(1, int(a.epochs) + 1):
        model.train()
        sums = {
            "loss": 0.0,
            "add_bce": 0.0,
            "semantic_ce": 0.0,
            "bottom_ce": 0.0,
            "span_ce": 0.0,
            "vertical_interval_ce": 0.0,
        }
        nb = 0
        for raw in _iter_batches(
            train_root,
            train_index,
            batch_size=int(a.batch_size),
            shuffle=True,
            seed=int(a.seed) + epoch,
        ):
            b = _move(raw, device, z)
            with _autocast(device, amp):
                out = model(
                    b["semantic"], b["geometry"], b["base"]
                )
                loss, stats = innovation_interval_loss(
                    out,
                    add_target=b["add"],
                    semantic_target=b["semantic_target"],
                    vertical_target=b["vertical"],
                    candidate_mask=b["candidate"],
                    weights=weights,
                    presence_hard_negative_ratio=float(
                        a.hard_negative_ratio
                    ),
                )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(
                    f"non-finite innovation loss at epoch {epoch}"
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(a.grad_clip)
            )
            optimizer.step()
            global_step += 1
            nb += 1
            for k in sums:
                sums[k] += float(stats[k])

        train_report = {
            k: v / max(nb, 1) for k, v in sums.items()
        }
        val_report = _evaluate(
            model,
            val_root,
            val_index,
            device,
            batch_size=a.batch_size,
            vertical_bins=z,
            amp=amp,
            weights=weights,
            hard_negative_ratio=a.hard_negative_ratio,
        )
        row = {
            "epoch": epoch,
            "global_step": global_step,
            "train": train_report,
            "val": val_report,
        }
        history.append(row)
        print(json.dumps(row), flush=True)

        _save(
            out_dir / f"epoch_{epoch:04d}.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            global_step=global_step,
            arch=arch,
            train_index=train_index,
            args=a,
            val_report=val_report,
        )
        score = float(val_report["joint_voxel_f0_5_at_0_5"])
        if score > best_score:
            best_score = score
            _save(
                out_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                arch=arch,
                train_index=train_index,
                args=a,
                val_report=val_report,
            )
        if float(val_report["loss"]) < best_loss:
            best_loss = float(val_report["loss"])
            _save(
                out_dir / "best_loss.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                arch=arch,
                train_index=train_index,
                args=a,
                val_report=val_report,
            )
        (out_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )

    print(f"saved interval-v4 innovation checkpoints to {out_dir}")


if __name__ == "__main__":
    main()
