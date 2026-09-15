#!/usr/bin/env python3
"""Paired C/Y continuation from the frozen V17-RL epoch-5 checkpoint.

C: exact historical V17-RL objective/XY target, no yaw.
Y: source-centred SE(2) XY target + scalar yaw + differentiable SE(2) shape loss.

Both arms restore the same historical optimizer state, use the same deterministic
paired source order and keep the original cosine LR schedule for old parameters.
The Y arm appends only the fresh zero-initialized yaw-head parameters to AdamW.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from real_motion.local_st_world_model_v17 import (
    MODEL_PROTOCOL_V17,
    LocalSpatialTemporalWorldModelV17,
    config_from_mapping_v17,
)
from real_motion.local_st_world_model_v18_se2 import (
    MODEL_PROTOCOL_V18_SE2,
    SE2_CACHE_VERSION,
    SE2_SHAPE_CONTRACT,
    SE2_TARGET_CONTRACT,
    LocalSpatialTemporalWorldModelV18SE2,
    periodic_yaw_loss,
    soft_se2_transport_overlap_loss,
)
from real_motion.motion_transport import FEATURE_DIM, FUTURE_FRAMES
from tools.real_motion.train_p0_f9_v17_local_stwm import objective_loss as v17_objective_loss

PROTOCOL = "p0_f9_v18_se2_paired_continuation_v1"
ARMS = ("C", "Y")
EXPECTED_START_EPOCH = 5
EXPECTED_VARIANT = "RL"
EXPECTED_OVERLAP = 0.25


def load_se2_cache(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if obj.get("version") != SE2_CACHE_VERSION:
        raise RuntimeError(f"expected SE2 cache {SE2_CACHE_VERSION}: {path}")
    meta = obj.get("metadata") or {}
    if meta.get("se2_target_contract") != SE2_TARGET_CONTRACT:
        raise RuntimeError("SE2 target contract mismatch")
    records = obj.get("records") or []
    if not records:
        raise RuntimeError("SE2 cache has no records")
    return meta, records


def flatten_supervised(records):
    keys = (
        "features",
        "local_semantic_tube",
        "kta_displacement_xy_m",
        "target_residual_xy_m",
        "target_displacement_xy_m",
        "target_valid",
        "existence",
        "supervised_source",
        "source_class_id",
        "frame_motion_features",
        "target_source_mask_tube",
        "target_source_displacement_xy_m",
        "target_source_residual_xy_m",
        "target_yaw_rad",
        "yaw_label_valid",
        "se2_target_valid",
        "yaw_enabled",
    )
    chunks = {k: [] for k in keys}
    scene_ids = []
    for r in records:
        sup = r["supervised_source"].bool()
        if not bool(sup.any()):
            continue
        for k in keys:
            x = r[k]
            if k == "supervised_source":
                chunks[k].append(torch.ones(int(sup.sum()), dtype=torch.bool))
            elif k in {"local_semantic_tube", "target_source_mask_tube"}:
                chunks[k].append(x[sup].to(torch.uint8))
            elif k == "source_class_id":
                chunks[k].append(x[sup].long())
            elif k in {"target_valid", "yaw_label_valid", "se2_target_valid", "yaw_enabled"}:
                chunks[k].append(x[sup].bool())
            else:
                chunks[k].append(x[sup].float())
        scene_ids.extend([str(r["scene_name"])] * int(sup.sum()))
    if not chunks["features"]:
        raise RuntimeError("cache has no supervised sources")
    out = {k: torch.cat(v, dim=0) for k, v in chunks.items()}
    out["scene_ids"] = scene_ids
    return out


def make_dataset(flat):
    return TensorDataset(
        flat["features"],
        flat["local_semantic_tube"],
        flat["kta_displacement_xy_m"],
        flat["target_residual_xy_m"],
        flat["target_displacement_xy_m"],
        flat["target_valid"],
        flat["existence"],
        flat["supervised_source"],
        flat["source_class_id"],
        flat["frame_motion_features"],
        flat["target_source_mask_tube"],
        flat["target_source_displacement_xy_m"],
        flat["target_source_residual_xy_m"],
        flat["target_yaw_rad"],
        flat["yaw_label_valid"],
        flat["se2_target_valid"],
        flat["yaw_enabled"],
    )


def unpack(raw, device):
    (
        f, tube, kta, old_res, old_disp, old_valid, existence, supervised,
        class_id, frame_motion, source_mask, source_disp, source_res, yaw,
        yaw_valid, se2_valid, yaw_enabled,
    ) = raw
    move = lambda x: x.to(device, non_blocking=True)
    return {
        "features": move(f),
        "local_semantic_tube": move(tube),
        "kta_displacement_xy_m": move(kta),
        "target_residual_xy_m": move(old_res),
        "target_displacement_xy_m": move(old_disp),
        "target_valid": move(old_valid),
        "existence": move(existence),
        "supervised_source": move(supervised),
        "source_class_id": move(class_id),
        "frame_motion_features": move(frame_motion),
        "target_source_mask_tube": move(source_mask),
        "target_source_displacement_xy_m": move(source_disp),
        "target_source_residual_xy_m": move(source_res),
        "target_yaw_rad": move(yaw),
        "yaw_label_valid": move(yaw_valid),
        "se2_target_valid": move(se2_valid),
        "yaw_enabled": move(yaw_enabled),
    }


def autocast_context(device, enabled):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def forward_model(model, b, *, amp, device):
    with autocast_context(device, amp):
        return model(
            b["features"],
            b["local_semantic_tube"],
            b["kta_displacement_xy_m"],
            b["frame_motion_features"],
            b["target_source_mask_tube"],
        )


def _existence_loss(outputs, batch):
    logits = outputs["existence_logits"]
    labels = batch["existence"].to(logits.dtype)
    supervised = batch["supervised_source"].bool()[:, None].expand_as(labels)
    if bool(supervised.any()):
        return F.binary_cross_entropy_with_logits(logits[supervised], labels[supervised])
    return logits.sum() * 0.0


def se2_objective_loss(outputs, batch, *, yaw_weight, shape_weight, patch_resolution_m):
    pred_xy = outputs["residual_xy_m"]
    target_xy = batch["target_source_residual_xy_m"].to(pred_xy.dtype)
    valid = batch["se2_target_valid"].bool()
    if bool(valid.any()):
        trans = F.smooth_l1_loss(
            pred_xy[valid], target_xy[valid], reduction="mean", beta=1.0
        )
    else:
        trans = pred_xy.sum() * 0.0

    exist = _existence_loss(outputs, batch)
    yaw, yaw_stats = periodic_yaw_loss(
        outputs["yaw_delta_rad"].float(),
        batch["target_yaw_rad"].float(),
        batch["yaw_enabled"],
        batch["yaw_label_valid"] & valid,
    )

    pred_disp = (
        batch["kta_displacement_xy_m"].float()
        + outputs["residual_xy_m"].float()
    )
    shape, shape_stats = soft_se2_transport_overlap_loss(
        pred_disp,
        batch["target_source_displacement_xy_m"].float(),
        outputs["yaw_delta_rad"].float(),
        batch["target_yaw_rad"].float(),
        batch["target_source_mask_tube"][:, -1].float(),
        valid,
        batch["yaw_enabled"],
        batch["yaw_label_valid"],
        patch_resolution_m=float(patch_resolution_m),
    )
    total = trans + exist + float(yaw_weight) * yaw + float(shape_weight) * shape
    stats = {
        "objective_loss": float(total.detach().cpu()),
        "translation_smooth_l1": float(trans.detach().cpu()),
        "existence_bce": float(exist.detach().cpu()),
        "yaw_periodic_loss": float(yaw.detach().cpu()),
        "se2_shape_loss": float(shape.detach().cpu()),
        **yaw_stats,
        **shape_stats,
    }
    return total, stats


def _lr_scale(step, total_steps):
    frac = min(max(float(step) / max(int(total_steps), 1), 0.0), 1.0)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * frac))


def _parse_save_steps(text):
    vals = sorted({int(x.strip()) for x in str(text).split(",") if x.strip()})
    if not vals or vals[0] <= 0:
        raise ValueError("--save-steps must contain positive integers")
    return vals


def _load_start(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("protocol") != MODEL_PROTOCOL_V17:
        raise RuntimeError("start checkpoint is not standard V17")
    if str(ck.get("variant")) != EXPECTED_VARIANT:
        raise RuntimeError("SE2 continuation requires V17-RL")
    if int(ck.get("epoch", -1)) != EXPECTED_START_EPOCH:
        raise RuntimeError("SE2 continuation requires frozen epoch-5 checkpoint")
    if not bool(ck.get("use_representation", False)):
        raise RuntimeError("SE2 continuation requires V17 representation")
    if abs(float(ck.get("overlap_weight", -1.0)) - EXPECTED_OVERLAP) > 1e-12:
        raise RuntimeError("unexpected V17-RL overlap weight")
    if "optimizer" not in ck:
        raise RuntimeError("epoch-5 checkpoint lacks optimizer state")
    return ck


def _build_model_optimizer(arm, ck, device):
    cfg = config_from_mapping_v17(ck.get("model_config"))
    old_args = ck.get("args") or {}
    base_lr = float(old_args.get("lr", 5e-4))
    weight_decay = float(old_args.get("weight_decay", 1e-4))

    if arm == "C":
        model = LocalSpatialTemporalWorldModelV17(cfg).to(device)
        model.load_state_dict(ck["state_dict"], strict=True)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=base_lr, weight_decay=weight_decay
        )
        optimizer.load_state_dict(ck["optimizer"])
        return model, optimizer

    model = LocalSpatialTemporalWorldModelV18SE2(cfg).to(device)
    incompatible = model.load_state_dict(ck["state_dict"], strict=False)
    expected_missing = {"yaw_head.weight", "yaw_head.bias"}
    if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"unexpected V17->SE2 load mismatch: missing={incompatible.missing_keys} "
            f"unexpected={incompatible.unexpected_keys}"
        )

    old_params = [
        p for name, p in model.named_parameters()
        if not name.startswith("yaw_head.")
    ]
    optimizer = torch.optim.AdamW(
        old_params, lr=base_lr, weight_decay=weight_decay
    )
    optimizer.load_state_dict(ck["optimizer"])
    current_lr = float(optimizer.param_groups[0]["lr"])
    current_wd = float(optimizer.param_groups[0].get("weight_decay", weight_decay))
    optimizer.add_param_group(
        {
            "params": list(model.yaw_head.parameters()),
            "lr": current_lr,
            "weight_decay": current_wd,
        }
    )
    return model, optimizer


def _save(path, *, arm, model, optimizer, ck, args, global_step, val_meta):
    payload = {
        "protocol": PROTOCOL,
        "model_protocol": MODEL_PROTOCOL_V17 if arm == "C" else MODEL_PROTOCOL_V18_SE2,
        "arm": arm,
        "source_checkpoint": str(Path(args.start_checkpoint).resolve()),
        "source_epoch": int(ck["epoch"]),
        "continuation_step": int(global_step),
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_config": ck.get("model_config"),
        "variant": "RL" if arm == "C" else "RL-SE2",
        "use_representation": True,
        "overlap_weight": EXPECTED_OVERLAP,
        "yaw_weight": None if arm == "C" else float(args.yaw_weight),
        "shape_weight": EXPECTED_OVERLAP,
        "se2_target_contract": SE2_TARGET_CONTRACT,
        "se2_shape_contract": SE2_SHAPE_CONTRACT,
        "paired_seed": int(args.seed),
        "val_cache_metadata": val_meta,
        "args": vars(args),
    }
    torch.save(payload, path)


def _calibrate(model, loader, device, *, amp, patch_resolution_m, batches, target_fraction):
    vals = []
    model.eval()
    with torch.no_grad():
        for bi, raw in enumerate(loader):
            if bi >= int(batches):
                break
            b = unpack(raw, device)
            out = forward_model(model, b, amp=amp, device=device)
            _, stats = se2_objective_loss(
                out, b, yaw_weight=1.0, shape_weight=EXPECTED_OVERLAP,
                patch_resolution_m=patch_resolution_m,
            )
            vals.append(stats)
    if not vals:
        raise RuntimeError("calibration loader yielded no batches")
    trans = float(np.median([x["translation_smooth_l1"] for x in vals]))
    yaw = float(np.median([x["yaw_periodic_loss"] for x in vals]))
    shape = float(np.median([x["se2_shape_loss"] for x in vals]))
    recommended = float(target_fraction) * trans / max(yaw, 1e-8)
    report = {
        "calibration_batches": len(vals),
        "median_translation_smooth_l1": trans,
        "median_unit_yaw_periodic_loss": yaw,
        "median_se2_shape_loss": shape,
        "target_yaw_to_translation_loss_fraction": float(target_fraction),
        "recommended_yaw_weight": recommended,
        "note": "one-time magnitude calibration; not a sweep",
    }
    print("=== SE2 YAW-WEIGHT CALIBRATION ===")
    print(json.dumps(report, indent=2))
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-cache", required=True)
    p.add_argument("--val-cache", required=True)
    p.add_argument("--start-checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--save-steps", default="300,600")
    p.add_argument("--yaw-weight", type=float, default=None)
    p.add_argument("--seed", type=int, default=20260915)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--calibrate-only", action="store_true")
    p.add_argument("--calibration-batches", type=int, default=8)
    p.add_argument("--calibration-target-fraction", type=float, default=0.25)
    a = p.parse_args()

    if a.steps <= 0:
        raise ValueError("--steps must be positive")
    if a.arm == "Y" and not a.calibrate_only and a.yaw_weight is None:
        raise ValueError("Y training requires explicit --yaw-weight from calibration")
    if a.yaw_weight is not None and float(a.yaw_weight) < 0:
        raise ValueError("--yaw-weight must be non-negative")
    save_steps = _parse_save_steps(a.save_steps)
    if max(save_steps) > int(a.steps):
        raise ValueError("save step exceeds --steps")

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)

    device = torch.device(
        a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    amp = device.type == "cuda" and not bool(a.no_amp)
    ck = _load_start(a.start_checkpoint)
    train_meta, train_records = load_se2_cache(a.train_cache)
    val_meta, val_records = load_se2_cache(a.val_cache)
    train = flatten_supervised(train_records)
    val = flatten_supervised(val_records)
    overlap = sorted(set(train["scene_ids"]) & set(val["scene_ids"]))
    if overlap:
        raise RuntimeError(f"train/val scene overlap: {overlap[:5]}")

    old_args = ck.get("args") or {}
    batch_size = int(old_args.get("batch_size", 256))
    original_epochs = int(old_args.get("epochs", 20))
    base_lr = float(old_args.get("lr", 5e-4))
    generator = torch.Generator().manual_seed(int(a.seed))
    train_loader = DataLoader(
        make_dataset(train),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=int(a.num_workers),
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_loader = DataLoader(
        make_dataset(val),
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(a.num_workers),
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    steps_per_epoch = len(train_loader)
    original_total_steps = max(1, original_epochs * steps_per_epoch)
    original_start_step = int(ck["epoch"]) * steps_per_epoch

    model, optimizer = _build_model_optimizer(a.arm, ck, device)
    expected_lr = base_lr * _lr_scale(original_start_step, original_total_steps)
    got_lr = float(optimizer.param_groups[0]["lr"])
    if abs(got_lr - expected_lr) > max(1e-9, abs(expected_lr) * 1e-5):
        raise RuntimeError(
            f"resume LR mismatch: checkpoint={got_lr} expected={expected_lr}"
        )
    if a.arm == "Y":
        # Step-0 contract: inherited XY/existence are exactly the frozen V17
        # predictions and the new yaw output is exactly zero by construction.
        if not torch.equal(model.yaw_head.weight.detach(), torch.zeros_like(model.yaw_head.weight)):
            raise RuntimeError("fresh yaw head is not zero initialized")
        if not torch.equal(model.yaw_head.bias.detach(), torch.zeros_like(model.yaw_head.bias)):
            raise RuntimeError("fresh yaw bias is not zero initialized")

    patch_resolution = float(train_meta.get("patch_resolution_m", 0.8))
    print(json.dumps({
        "protocol": PROTOCOL,
        "arm": a.arm,
        "start_epoch": int(ck["epoch"]),
        "train_sources": len(train["features"]),
        "val_sources": len(val["features"]),
        "batch_size": batch_size,
        "steps_per_epoch": steps_per_epoch,
        "original_total_steps": original_total_steps,
        "original_start_step": original_start_step,
        "resume_lr": got_lr,
        "paired_seed": int(a.seed),
        "amp_bfloat16": amp,
    }, indent=2))

    if a.calibrate_only:
        if a.arm != "Y":
            raise ValueError("--calibrate-only is defined only for Y")
        _calibrate(
            model, train_loader, device, amp=amp,
            patch_resolution_m=patch_resolution,
            batches=int(a.calibration_batches),
            target_fraction=float(a.calibration_target_fraction),
        )
        return

    out_dir = Path(a.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.train()
    step = 0
    history = []
    while step < int(a.steps):
        for raw in train_loader:
            if step >= int(a.steps):
                break
            b = unpack(raw, device)
            out = forward_model(model, b, amp=amp, device=device)
            if a.arm == "C":
                # Exact historical V17 target and loss.
                loss, stats = v17_objective_loss(
                    out,
                    b,
                    overlap_weight=EXPECTED_OVERLAP,
                    patch_resolution_m=patch_resolution,
                )
            else:
                loss, stats = se2_objective_loss(
                    out,
                    b,
                    yaw_weight=float(a.yaw_weight),
                    shape_weight=EXPECTED_OVERLAP,
                    patch_resolution_m=patch_resolution,
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            step += 1

            absolute_step = original_start_step + step
            lr = base_lr * _lr_scale(absolute_step, original_total_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr

            if step == 1 or step % 50 == 0 or step in save_steps:
                row = {"step": step, "lr": lr, **stats}
                history.append(row)
                print(json.dumps(row), flush=True)

            if step in save_steps:
                _save(
                    out_dir / f"step_{step:06d}.pt",
                    arm=a.arm,
                    model=model,
                    optimizer=optimizer,
                    ck=ck,
                    args=a,
                    global_step=step,
                    val_meta=val_meta,
                )

    report = {
        "protocol": PROTOCOL,
        "arm": a.arm,
        "source_checkpoint": str(Path(a.start_checkpoint).resolve()),
        "steps": int(a.steps),
        "save_steps": save_steps,
        "history": history,
        "yaw_weight": None if a.arm == "C" else float(a.yaw_weight),
        "shape_weight": EXPECTED_OVERLAP,
        "se2_target_contract": SE2_TARGET_CONTRACT,
        "se2_shape_contract": SE2_SHAPE_CONTRACT,
    }
    (out_dir / "training_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print("=== V18 SE2 PAIRED CONTINUATION COMPLETE ===")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()