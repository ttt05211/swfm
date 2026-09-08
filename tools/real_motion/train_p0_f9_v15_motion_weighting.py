#!/usr/bin/env python3
"""Controlled v15 M/MC training for learned rigid motion transport.

M  = v13 architecture/targets + exact true-motion observation weight 2.
MC = M + macro class rebalancing *within true-moving observations only*.

Everything else stays fixed: same Strong-source cache, tiny MLP, Smooth-L1,
existence BCE, optimizer, cosine schedule, seed contract and 20-epoch budget.
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

from real_motion.metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS, NUSCENES_LABELS
from real_motion.motion_transport_v2 import (
    FEATURE_DIM,
    FUTURE_FRAMES,
    MOTION_TRANSPORT_CACHE_VERSION,
    TARGET_CONTRACT,
    MotionTransportHead,
)
from real_motion.motion_transport_weighting import (
    DEFAULT_MOTION_WEIGHT,
    MODE_MOTION,
    MODE_MOTION_CLASS,
    WEIGHTING_MODES,
    compute_macro_class_weights,
    make_observation_weights,
    true_moving_mask_torch,
    weight_contract_summary,
    weighted_motion_transport_loss,
)

PROTOCOLS = {
    MODE_MOTION: "p0_f9_v15_motion_weighted_m",
    MODE_MOTION_CLASS: "p0_f9_v15_motion_class_balanced_mc",
}


def load_cache(path: str):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if obj.get("version") != MOTION_TRANSPORT_CACHE_VERSION:
        raise RuntimeError(f"motion cache version mismatch: {path}")
    meta = obj.get("metadata") or {}
    if meta.get("target_contract") != TARGET_CONTRACT:
        raise RuntimeError("motion cache target contract mismatch")
    if int(meta.get("feature_dim", -1)) != FEATURE_DIM:
        raise RuntimeError("feature dimension mismatch")
    records = obj.get("records") or []
    if not records:
        raise RuntimeError("motion cache has no records")
    return meta, records


def flatten_supervised(records):
    keys = (
        "features", "target_residual_xy_m", "existence", "target_valid",
        "supervised_source", "target_displacement_xy_m", "source_class_id",
    )
    chunks = {k: [] for k in keys}
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
        chunks["target_displacement_xy_m"].append(r["target_displacement_xy_m"][sup].float())
        chunks["source_class_id"].append(r["source_class_id"][sup].long())
        scene_ids.extend([str(r["scene_name"])] * int(sup.sum()))
    if not chunks["features"]:
        raise RuntimeError("cache has no supervised Strong sources")
    out = {k: torch.cat(v, dim=0) for k, v in chunks.items()}
    out["scene_ids"] = scene_ids
    out["true_moving"] = true_moving_mask_torch(
        out["target_displacement_xy_m"], out["target_valid"]
    )
    return out


def attach_weights(flat, *, mode: str, class_weights):
    flat = dict(flat)
    flat["observation_weights"] = make_observation_weights(
        flat["source_class_id"], flat["true_moving"], flat["target_valid"],
        mode=mode, class_weights=class_weights, motion_weight=DEFAULT_MOTION_WEIGHT,
    )
    return flat


def make_dataset(flat):
    return TensorDataset(
        flat["features"],
        flat["target_residual_xy_m"],
        flat["existence"],
        flat["target_valid"],
        flat["supervised_source"],
        flat["target_displacement_xy_m"],
        flat["source_class_id"],
        flat["true_moving"],
        flat["observation_weights"],
    )


def unpack(raw, device):
    f, residual, existence, valid, supervised, target_disp, class_id, true_moving, weights = raw
    return {
        "features": f.to(device, non_blocking=True),
        "target_residual_xy_m": residual.to(device, non_blocking=True),
        "existence": existence.to(device, non_blocking=True),
        "target_valid": valid.to(device, non_blocking=True),
        "supervised_source": supervised.to(device, non_blocking=True),
        "target_displacement_xy_m": target_disp.to(device, non_blocking=True),
        "source_class_id": class_id.to(device, non_blocking=True),
        "true_moving": true_moving.to(device, non_blocking=True),
        "observation_weights": weights.to(device, non_blocking=True),
    }


def _latest_errors(kd, ld, mask):
    ks, ls = [], []
    for i in range(mask.shape[0]):
        ids = torch.nonzero(mask[i], as_tuple=False).flatten()
        if ids.numel():
            h = ids[-1]
            ks.append(float(kd[i, h].item()))
            ls.append(float(ld[i, h].item()))
    return ks, ls


def eval_model(model, loader, device):
    model.eval()
    loss_sum = traj_sum = exist_sum = 0.0
    n_batches = 0
    all_k, all_l, mov_k, mov_l = [], [], [], []
    all_k_fde, all_l_fde, mov_k_fde, mov_l_fde = [], [], [], []
    class_rows = {int(c): [[], []] for c in DYNAMIC_CLASS_IDS}
    exist_correct = exist_total = tp = fp = fn = 0
    with torch.no_grad():
        for raw in loader:
            b = unpack(raw, device)
            out = model(b["features"])
            loss, parts = weighted_motion_transport_loss(out, b, b["observation_weights"])
            loss_sum += float(loss.item())
            traj_sum += float(parts["trajectory_smooth_l1_weighted"])
            exist_sum += float(parts["existence_bce"])
            n_batches += 1

            valid = b["target_valid"].bool()
            moving = b["true_moving"].bool()
            target = b["target_residual_xy_m"]
            kd = target.norm(dim=-1)
            ld = (out["residual_xy_m"] - target).norm(dim=-1)
            if bool(valid.any()):
                all_k.append(kd[valid].cpu()); all_l.append(ld[valid].cpu())
            if bool(moving.any()):
                mov_k.append(kd[moving].cpu()); mov_l.append(ld[moving].cpu())
            ks, ls = _latest_errors(kd, ld, valid)
            all_k_fde.extend(ks); all_l_fde.extend(ls)
            ks, ls = _latest_errors(kd, ld, moving)
            mov_k_fde.extend(ks); mov_l_fde.extend(ls)

            cls = b["source_class_id"].long()
            for c in DYNAMIC_CLASS_IDS:
                cm = moving & (cls[:, None] == int(c))
                if bool(cm.any()):
                    class_rows[int(c)][0].append(kd[cm].cpu())
                    class_rows[int(c)][1].append(ld[cm].cpu())

            pred_exist = out["existence_logits"] >= 0.0
            gt_exist = b["existence"] > 0.5
            exist_correct += int((pred_exist == gt_exist).sum().item())
            exist_total += int(gt_exist.numel())
            tp += int((pred_exist & gt_exist).sum().item())
            fp += int((pred_exist & ~gt_exist).sum().item())
            fn += int((~pred_exist & gt_exist).sum().item())

    def cat_mean(xs):
        return float(torch.cat(xs).mean().item()) if xs else float("nan")

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    per_class = {}
    for c in DYNAMIC_CLASS_IDS:
        klist, llist = class_rows[int(c)]
        if klist:
            km = float(torch.cat(klist).mean().item())
            lm = float(torch.cat(llist).mean().item())
            count = int(sum(x.numel() for x in klist))
        else:
            km = lm = float("nan"); count = 0
        per_class[NUSCENES_LABELS[int(c)]] = {
            "count": count,
            "kta_ade_m": km,
            "learned_ade_m": lm,
            "relative_reduction": ((km - lm) / km if count and km > 1e-12 else float("nan")),
        }

    return {
        "loss": loss_sum / max(n_batches, 1),
        "trajectory_smooth_l1_weighted": traj_sum / max(n_batches, 1),
        "existence_bce": exist_sum / max(n_batches, 1),
        "kta_ade_m": cat_mean(all_k),
        "learned_ade_m": cat_mean(all_l),
        "kta_fde_m": float(np.mean(all_k_fde)) if all_k_fde else float("nan"),
        "learned_fde_m": float(np.mean(all_l_fde)) if all_l_fde else float("nan"),
        "true_moving_kta_ade_m": cat_mean(mov_k),
        "true_moving_learned_ade_m": cat_mean(mov_l),
        "true_moving_kta_fde_m": float(np.mean(mov_k_fde)) if mov_k_fde else float("nan"),
        "true_moving_learned_fde_m": float(np.mean(mov_l_fde)) if mov_l_fde else float("nan"),
        "true_moving_per_class": per_class,
        "existence_accuracy": exist_correct / max(exist_total, 1),
        "existence_precision": precision,
        "existence_recall": recall,
        "existence_f1": 2 * precision * recall / max(precision + recall, 1e-12),
    }


def save_ckpt(path, model, optimizer, epoch, args, protocol, train_meta, val_meta,
              val_report, class_weights, train_weight_contract, val_weight_contract):
    torch.save({
        "protocol": protocol,
        "weighting_mode": args.mode,
        "motion_weight": DEFAULT_MOTION_WEIGHT,
        "epoch": int(epoch),
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "feature_dim": FEATURE_DIM,
        "future_frames": FUTURE_FRAMES,
        "hidden_dim": int(args.hidden_dim),
        "args": vars(args),
        "class_weights": class_weights,
        "train_weight_contract": train_weight_contract,
        "val_weight_contract": val_weight_contract,
        "train_cache_metadata": train_meta,
        "val_cache_metadata": val_meta,
        "val_report": val_report,
    }, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-cache", required=True)
    p.add_argument("--val-cache", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--mode", required=True, choices=WEIGHTING_MODES)
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
    protocol = PROTOCOLS[a.mode]

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

    class_weights = None
    if a.mode == MODE_MOTION_CLASS:
        class_weights = compute_macro_class_weights(train["source_class_id"], train["true_moving"])
    train = attach_weights(train, mode=a.mode, class_weights=class_weights)
    val = attach_weights(val, mode=a.mode, class_weights=class_weights)
    train_contract = weight_contract_summary(
        train["source_class_id"], train["true_moving"], train["target_valid"],
        train["observation_weights"], class_weights,
    )
    val_contract = weight_contract_summary(
        val["source_class_id"], val["true_moving"], val["target_valid"],
        val["observation_weights"], class_weights,
    )

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
    total_steps = max(1, int(a.epochs) * len(train_loader))
    step = 0
    best_all = float("inf")
    best_moving = float("inf")
    out_dir = Path(a.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    history = []

    init_report = eval_model(model, val_loader, device)
    header = {
        "protocol": protocol,
        "mode": a.mode,
        "motion_weight": DEFAULT_MOTION_WEIGHT,
        "train_sources": len(train["features"]),
        "val_sources": len(val["features"]),
        "train_scenes": len(train_scenes),
        "val_scenes": len(val_scenes),
        "class_weights": class_weights,
        "train_weight_contract": train_contract,
        "val_weight_contract": val_contract,
        "initial_val": init_report,
        "selection_contract": "best.pt uses all-valid learned ADE for direct v13 comparability; best_true_moving.pt is secondary diagnostic",
    }
    print(json.dumps(header, indent=2))

    for epoch in range(1, int(a.epochs) + 1):
        model.train(); running = 0.0
        for raw in train_loader:
            b = unpack(raw, device)
            out = model(b["features"])
            loss, _ = weighted_motion_transport_loss(out, b, b["observation_weights"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            step += 1
            frac = min(step / total_steps, 1.0)
            scale = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * frac))
            for group in optimizer.param_groups:
                group["lr"] = float(a.lr) * scale
            running += float(loss.item())

        val_report = eval_model(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": running / max(len(train_loader), 1),
            "lr": optimizer.param_groups[0]["lr"],
            **val_report,
        }
        history.append(row); print(json.dumps(row))
        common = (model, optimizer, epoch, a, protocol, train_meta, val_meta,
                  val_report, class_weights, train_contract, val_contract)
        save_ckpt(out_dir / "latest.pt", *common)
        if epoch in {1, 5, 10, int(a.epochs)}:
            save_ckpt(out_dir / f"epoch_{epoch:04d}.pt", *common)
        if float(val_report["learned_ade_m"]) < best_all:
            best_all = float(val_report["learned_ade_m"])
            save_ckpt(out_dir / "best.pt", *common)
        if float(val_report["true_moving_learned_ade_m"]) < best_moving:
            best_moving = float(val_report["true_moving_learned_ade_m"])
            save_ckpt(out_dir / "best_true_moving.pt", *common)

    report = {
        **header,
        "best_learned_ade_m": best_all,
        "best_true_moving_learned_ade_m": best_moving,
        "history": history,
    }
    (out_dir / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
