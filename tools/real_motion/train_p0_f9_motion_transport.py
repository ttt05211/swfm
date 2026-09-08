#!/usr/bin/env python3
"""Train the first Strong-source learned motion-transport head.

The model is intentionally tiny and does not see future occupancy, boxes, VAE
latents, or world-model features.  It predicts center residuals relative to the
causal Strong/KTA anchor and future existence from occupancy-derived source
history only.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from real_motion.motion_transport import (
    FEATURE_DIM,
    FUTURE_FRAMES,
    MOTION_TRANSPORT_CACHE_VERSION,
    MotionTransportHead,
    motion_transport_loss,
)

PROTOCOL = "p0_f9_v12_learned_motion_transport_v1"


def load_cache(path: str):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if obj.get("version") != MOTION_TRANSPORT_CACHE_VERSION:
        raise RuntimeError(f"motion cache version mismatch: {path}")
    meta = obj.get("metadata") or {}
    if int(meta.get("feature_dim", -1)) != FEATURE_DIM:
        raise RuntimeError("feature dimension mismatch")
    records = obj.get("records") or []
    if not records:
        raise RuntimeError("motion cache has no records")
    return meta, records


def flatten_supervised(records):
    chunks = {k: [] for k in (
        "features", "target_residual_xy_m", "existence", "target_valid", "supervised_source"
    )}
    scene_ids = []
    for r in records:
        sup = r["supervised_source"].bool()
        if not bool(sup.any()):
            continue
        chunks["features"].append(r["features"][sup].float())
        chunks["target_residual_xy_m"].append(r["target_residual_xy_m"][sup].float())
        chunks["existence"].append(r["existence"][sup].float())
        chunks["target_valid"].append(r["target_valid"][sup].bool())
        chunks["supervised_source"].append(torch.ones(int(sup.sum()), dtype=torch.bool))
        scene_ids.extend([str(r["scene_name"])] * int(sup.sum()))
    if not chunks["features"]:
        raise RuntimeError("cache has no supervised Strong sources")
    out = {k: torch.cat(v, dim=0) for k, v in chunks.items()}
    out["scene_ids"] = scene_ids
    return out


def make_dataset(flat):
    return TensorDataset(
        flat["features"], flat["target_residual_xy_m"], flat["existence"],
        flat["target_valid"], flat["supervised_source"],
    )


def unpack(batch, device):
    f, residual, existence, valid, supervised = batch
    return {
        "features": f.to(device, non_blocking=True),
        "target_residual_xy_m": residual.to(device, non_blocking=True),
        "existence": existence.to(device, non_blocking=True),
        "target_valid": valid.to(device, non_blocking=True),
        "supervised_source": supervised.to(device, non_blocking=True),
    }


def eval_model(model, loader, device):
    model.eval()
    loss_sum = traj_sum = exist_sum = 0.0
    n_batches = 0
    kta_dist = []
    learned_dist = []
    kta_fde = []
    learned_fde = []
    exist_correct = exist_total = 0
    tp = fp = fn = 0
    with torch.no_grad():
        for raw in loader:
            b = unpack(raw, device)
            out = model(b["features"])
            loss, parts = motion_transport_loss(out, b)
            loss_sum += float(loss.item())
            traj_sum += float(parts["trajectory_smooth_l1"])
            exist_sum += float(parts["existence_bce"])
            n_batches += 1

            valid = b["target_valid"].bool()
            target = b["target_residual_xy_m"]
            kd = target.norm(dim=-1)
            ld = (out["residual_xy_m"] - target).norm(dim=-1)
            if bool(valid.any()):
                kta_dist.append(kd[valid].cpu())
                learned_dist.append(ld[valid].cpu())
            for i in range(valid.shape[0]):
                ids = torch.nonzero(valid[i], as_tuple=False).flatten()
                if ids.numel():
                    h = ids[-1]
                    kta_fde.append(float(kd[i, h].item()))
                    learned_fde.append(float(ld[i, h].item()))

            pred_exist = out["existence_logits"] >= 0.0
            gt_exist = b["existence"] > 0.5
            exist_correct += int((pred_exist == gt_exist).sum().item())
            exist_total += int(gt_exist.numel())
            tp += int((pred_exist & gt_exist).sum().item())
            fp += int((pred_exist & ~gt_exist).sum().item())
            fn += int((~pred_exist & gt_exist).sum().item())

    kta = torch.cat(kta_dist) if kta_dist else torch.empty(0)
    learned = torch.cat(learned_dist) if learned_dist else torch.empty(0)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "loss": loss_sum / max(n_batches, 1),
        "trajectory_smooth_l1": traj_sum / max(n_batches, 1),
        "existence_bce": exist_sum / max(n_batches, 1),
        "kta_ade_m": float(kta.mean()) if kta.numel() else float("nan"),
        "learned_ade_m": float(learned.mean()) if learned.numel() else float("nan"),
        "kta_fde_m": float(np.mean(kta_fde)) if kta_fde else float("nan"),
        "learned_fde_m": float(np.mean(learned_fde)) if learned_fde else float("nan"),
        "existence_accuracy": exist_correct / max(exist_total, 1),
        "existence_precision": precision,
        "existence_recall": recall,
        "existence_f1": 2 * precision * recall / max(precision + recall, 1e-12),
    }


def save_ckpt(path, model, optimizer, epoch, args, train_meta, val_meta, val_report):
    torch.save({
        "protocol": PROTOCOL,
        "epoch": int(epoch),
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "feature_dim": FEATURE_DIM,
        "future_frames": FUTURE_FRAMES,
        "hidden_dim": int(args.hidden_dim),
        "args": vars(args),
        "train_cache_metadata": train_meta,
        "val_cache_metadata": val_meta,
        "val_report": val_report,
    }, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-cache", required=True)
    p.add_argument("--val-cache", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=20260908)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    if a.epochs <= 0 or a.batch_size <= 0 or a.lr <= 0:
        raise ValueError("epochs/batch-size/lr must be positive")

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    device = torch.device(a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu")

    train_meta, train_records = load_cache(a.train_cache)
    val_meta, val_records = load_cache(a.val_cache)
    train = flatten_supervised(train_records)
    val = flatten_supervised(val_records)
    train_scenes = set(train["scene_ids"]); val_scenes = set(val["scene_ids"])
    overlap = sorted(train_scenes & val_scenes)
    if overlap:
        raise RuntimeError(f"train/val scene overlap: {overlap[:5]}")

    gen = torch.Generator().manual_seed(int(a.seed))
    train_loader = DataLoader(
        make_dataset(train), batch_size=int(a.batch_size), shuffle=True, generator=gen,
        num_workers=int(a.num_workers), pin_memory=(device.type == "cuda"), drop_last=False,
    )
    val_loader = DataLoader(
        make_dataset(val), batch_size=int(a.batch_size), shuffle=False,
        num_workers=int(a.num_workers), pin_memory=(device.type == "cuda"), drop_last=False,
    )

    model = MotionTransportHead(hidden_dim=int(a.hidden_dim)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(a.lr), weight_decay=float(a.weight_decay))
    # One fixed cosine schedule; no stage restart or motion-specific loss tuning.
    total_steps = max(1, int(a.epochs) * len(train_loader))
    step = 0
    best_ade = float("inf")
    out_dir = Path(a.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    history = []

    init_report = eval_model(model, val_loader, device)
    print(json.dumps({
        "protocol": PROTOCOL,
        "train_sources": len(train["features"]),
        "val_sources": len(val["features"]),
        "train_scenes": len(train_scenes), "val_scenes": len(val_scenes),
        "initial_val": init_report,
    }, indent=2))

    for epoch in range(1, int(a.epochs) + 1):
        model.train()
        running = 0.0
        for raw in train_loader:
            b = unpack(raw, device)
            out = model(b["features"])
            loss, _ = motion_transport_loss(out, b)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            step += 1
            # Cosine from lr to 10% of lr over the single 20-epoch budget.
            frac = min(step / total_steps, 1.0)
            scale = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * frac))
            for group in optimizer.param_groups:
                group["lr"] = float(a.lr) * scale
            running += float(loss.item())

        val_report = eval_model(model, val_loader, device)
        row = {
            "epoch": epoch, "train_loss": running / max(len(train_loader), 1),
            "lr": optimizer.param_groups[0]["lr"], **val_report,
        }
        history.append(row)
        print(json.dumps(row))
        save_ckpt(out_dir / "latest.pt", model, optimizer, epoch, a, train_meta, val_meta, val_report)
        if epoch in {1, 5, 10, int(a.epochs)}:
            save_ckpt(out_dir / f"epoch_{epoch:04d}.pt", model, optimizer, epoch, a, train_meta, val_meta, val_report)
        if float(val_report["learned_ade_m"]) < best_ade:
            best_ade = float(val_report["learned_ade_m"])
            save_ckpt(out_dir / "best.pt", model, optimizer, epoch, a, train_meta, val_meta, val_report)

    report = {
        "protocol": PROTOCOL,
        "best_learned_ade_m": best_ade,
        "initial_val": init_report,
        "history": history,
        "train_sources": len(train["features"]),
        "val_sources": len(val["features"]),
    }
    (out_dir / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
