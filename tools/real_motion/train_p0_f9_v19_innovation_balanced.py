#!/usr/bin/env python3
"""V19 Innovation trainer v2 with balanced vertical occupancy supervision.

Reuses the frozen-base V19 cache produced by
build_p0_f9_v19_innovation_cache.py. Clean-E14 is never optimized.
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
    ResidualInnovationHead,
    innovation_loss,
)
from real_motion.v19_innovation_targets import INNOVATION_POSITIVE_CATEGORIES
from real_motion.v19_innovation_training import (
    dequantize_geometry_torch,
    unpack_vertical_occupancy_torch,
)
from tools.real_motion.train_p0_f9_v19_innovation import (
    PROTOCOL,
    _iter_batches,
    _load_index,
)

OBJECTIVE_VERSION = "balanced_vertical_v2"


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


def _vertical_counts(root, index, vertical_bins):
    pos = 0
    total = 0
    for row in index["shards"]:
        obj = torch.load(
            root / row["file"], map_location="cpu", weights_only=False
        )
        bits = obj["vertical_bits"].to(torch.int64)
        add = obj["add_target"].bool()
        vals = bits[add]
        total += int(vals.numel()) * int(vertical_bins)
        for zi in range(int(vertical_bins)):
            pos += int(((vals >> zi) & 1).sum().item())
    return int(pos), int(total - pos)


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
    presence_positive_weight,
    vertical_positive_weight,
):
    model.eval()
    sums = {
        "loss": 0.0,
        "add_bce": 0.0,
        "semantic_ce": 0.0,
        "vertical_bce": 0.0,
    }
    nb = 0
    p_tp = p_fp = p_fn = 0
    z_tp = z_fp = z_fn = 0
    sem_ok = sem_n = 0

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
                loss, stats = innovation_loss(
                    out,
                    add_target=b["add"],
                    semantic_target=b["semantic_target"],
                    vertical_target=b["vertical"],
                    candidate_mask=b["candidate"],
                    weights=weights,
                    positive_weight=float(presence_positive_weight),
                    vertical_positive_weight=float(vertical_positive_weight),
                )
            for k in sums:
                sums[k] += float(stats[k])
            nb += 1

            cand = b["candidate"]
            tgt = b["add"] & cand
            pred = (
                torch.sigmoid(out["add_presence_logits"].float()) >= 0.5
            ) & cand
            p_tp += int((pred & tgt).sum().item())
            p_fp += int((pred & ~tgt & cand).sum().item())
            p_fn += int((~pred & tgt).sum().item())

            if bool(tgt.any()):
                sem = out["semantic_logits"].argmax(dim=2)
                sem_ok += int(
                    (sem[tgt] == b["semantic_target"][tgt]).sum().item()
                )
                sem_n += int(tgt.sum().item())

                z_pred = (
                    torch.sigmoid(
                        out["vertical_occupancy_logits"].float()
                    )
                    .permute(0, 1, 3, 4, 2)[tgt]
                    >= 0.5
                )
                z_tgt = (
                    b["vertical"]
                    .permute(0, 1, 3, 4, 2)[tgt]
                    .bool()
                )
                z_tp += int((z_pred & z_tgt).sum().item())
                z_fp += int((z_pred & ~z_tgt).sum().item())
                z_fn += int((~z_pred & z_tgt).sum().item())

    p_prec = p_tp / max(p_tp + p_fp, 1)
    p_rec = p_tp / max(p_tp + p_fn, 1)
    z_prec = z_tp / max(z_tp + z_fp, 1)
    z_rec = z_tp / max(z_tp + z_fn, 1)
    return {
        **{k: v / max(nb, 1) for k, v in sums.items()},
        "presence_precision_at_0_5": float(p_prec),
        "presence_recall_at_0_5": float(p_rec),
        "presence_f1_at_0_5": float(
            2 * p_prec * p_rec / max(p_prec + p_rec, 1e-12)
        ),
        "vertical_precision_at_0_5": float(z_prec),
        "vertical_recall_at_0_5": float(z_rec),
        "vertical_f1_at_0_5": float(
            2 * z_prec * z_rec / max(z_prec + z_rec, 1e-12)
        ),
        "semantic_accuracy_on_positive_bev": float(
            sem_ok / max(sem_n, 1)
        ),
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
    presence_positive_weight,
    vertical_positive_weight,
    val_report,
):
    torch.save(
        {
            "protocol": PROTOCOL,
            "objective_version": OBJECTIVE_VERSION,
            "training_mode": "innovation_only_frozen_base_balanced_vertical",
            "epoch": int(epoch),
            "global_step": int(global_step),
            "innovation_state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "architecture": arch,
            "positive_categories": list(INNOVATION_POSITIVE_CATEGORIES),
            "positive_weight": float(presence_positive_weight),
            "vertical_positive_weight": float(vertical_positive_weight),
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
    p.add_argument("--presence-positive-weight", type=float, default=0.0)
    p.add_argument("--vertical-positive-weight", type=float, default=0.0)
    p.add_argument("--max-presence-positive-weight", type=float, default=64.0)
    p.add_argument("--max-vertical-positive-weight", type=float, default=16.0)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=20260923)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-amp", action="store_true")
    a = p.parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)

    train_root, train_index = _load_index(a.train_cache)
    val_root, val_index = _load_index(a.val_cache)
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

    totals = train_index["totals"]
    pos_bev = int(totals["positive_bev_cells"])
    cand_bev = int(totals["candidate_bev_cells"])
    auto_presence = (cand_bev - pos_bev) / max(pos_bev, 1)
    presence_pw = (
        float(a.presence_positive_weight)
        if a.presence_positive_weight > 0
        else min(
            max(auto_presence, 1.0),
            float(a.max_presence_positive_weight),
        )
    )

    z_pos, z_neg = _vertical_counts(train_root, train_index, z)
    if z_pos <= 0 or z_neg <= 0:
        raise RuntimeError(
            f"invalid vertical labels: pos={z_pos} neg={z_neg}"
        )
    auto_vertical = z_neg / max(z_pos, 1)
    vertical_pw = (
        float(a.vertical_positive_weight)
        if a.vertical_positive_weight > 0
        else min(
            max(auto_vertical, 1.0),
            float(a.max_vertical_positive_weight),
        )
    )

    model = ResidualInnovationHead(**arch).to(device)
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
        presence_positive_weight=presence_pw,
        vertical_positive_weight=vertical_pw,
    )
    preflight = {
        "protocol": PROTOCOL,
        "objective_version": OBJECTIVE_VERSION,
        "parameters": int(sum(x.numel() for x in model.parameters())),
        "clean_e14_parameters_in_optimizer": 0,
        "presence_positive_weight": float(presence_pw),
        "auto_presence_positive_weight_uncapped": float(auto_presence),
        "vertical_positive_labels": int(z_pos),
        "vertical_negative_labels": int(z_neg),
        "vertical_positive_weight": float(vertical_pw),
        "auto_vertical_positive_weight_uncapped": float(auto_vertical),
        "initial_val": initial_val,
    }
    (out_dir / "preflight.json").write_text(
        json.dumps(preflight, indent=2), encoding="utf-8"
    )
    print("=== V19 INNOVATION BALANCED-VERTICAL PREFLIGHT ===")
    print(json.dumps(preflight, indent=2))

    best = float("inf")
    global_step = 0
    history = []
    for epoch in range(1, int(a.epochs) + 1):
        model.train()
        sums = {
            "loss": 0.0,
            "add_bce": 0.0,
            "semantic_ce": 0.0,
            "vertical_bce": 0.0,
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
                loss, stats = innovation_loss(
                    out,
                    add_target=b["add"],
                    semantic_target=b["semantic_target"],
                    vertical_target=b["vertical"],
                    candidate_mask=b["candidate"],
                    weights=weights,
                    positive_weight=float(presence_pw),
                    vertical_positive_weight=float(vertical_pw),
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
            presence_positive_weight=presence_pw,
            vertical_positive_weight=vertical_pw,
        )
        row = {
            "epoch": epoch,
            "global_step": global_step,
            "train": train_report,
            "val": val_report,
        }
        history.append(row)
        print(json.dumps(row), flush=True)

        ck = out_dir / f"epoch_{epoch:04d}.pt"
        _save(
            ck,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            global_step=global_step,
            arch=arch,
            train_index=train_index,
            args=a,
            presence_positive_weight=presence_pw,
            vertical_positive_weight=vertical_pw,
            val_report=val_report,
        )
        if float(val_report["loss"]) < best:
            best = float(val_report["loss"])
            _save(
                out_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                arch=arch,
                train_index=train_index,
                args=a,
                presence_positive_weight=presence_pw,
                vertical_positive_weight=vertical_pw,
                val_report=val_report,
            )

        (out_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )

    print(f"saved balanced innovation checkpoints to {out_dir}")


if __name__ == "__main__":
    main()
