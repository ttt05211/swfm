#!/usr/bin/env python3
"""Balanced clean one-stage training for V18 SE(2).

This is the formal long-tail variant of clean one-stage V18 training. It keeps
exactly the same architecture, initialization, optimizer, loss, LR schedule,
validation loader and inference contract as train_p0_f9_v18_se2_clean.py. The
only treatment is the source sampler:

    uniform source shuffle
        -> class-aware + motion-density-aware weighted source sampling

Each epoch still draws exactly N supervised sources with replacement, so batch
count and schedule length remain matched to the clean baseline.
"""
from __future__ import annotations

import argparse
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
from torch.utils.data import DataLoader

from real_motion.local_st_world_model_v17 import LocalSTWMV17Config
from real_motion.local_st_world_model_v18_se2 import (
    MODEL_PROTOCOL_V18_SE2,
    SE2_SHAPE_CONTRACT,
    SE2_TARGET_CONTRACT,
    LocalSpatialTemporalWorldModelV18SE2,
)
from real_motion.long_tail_sampling import (
    PROTOCOL as SAMPLER_PROTOCOL,
    build_balanced_source_weights,
    make_balanced_source_sampler,
)
from real_motion.motion_transport import FEATURE_DIM, FUTURE_FRAMES
from tools.real_motion.train_p0_f9_v18_se2_clean import (
    PROTOCOL as CLEAN_PROTOCOL,
    _eval_objective,
    _lr_scale,
)
from tools.real_motion.train_p0_f9_v18_se2_pair import (
    EXPECTED_OVERLAP,
    flatten_supervised,
    forward_model,
    load_se2_cache,
    make_dataset,
    se2_objective_loss,
    unpack,
)

TRAINING_MODE = "clean_one_stage_balanced_source_sampling_v1"


def _save(path, *, model, optimizer, args, epoch, global_step, model_config,
          train_meta, val_meta, val_report, steps_per_epoch, sampler_report):
    torch.save(
        {
            "protocol": CLEAN_PROTOCOL,
            "model_protocol": MODEL_PROTOCOL_V18_SE2,
            "arm": "Y",
            "training_mode": TRAINING_MODE,
            "epoch": int(epoch),
            "global_step": int(global_step),
            "continuation_step": int(global_step),
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "feature_dim": FEATURE_DIM,
            "future_frames": FUTURE_FRAMES,
            "model_config": asdict(model_config),
            "variant": "CLEAN-SE2-BALANCED",
            "use_representation": True,
            "overlap_weight": EXPECTED_OVERLAP,
            "yaw_weight": float(args.yaw_weight),
            "shape_weight": EXPECTED_OVERLAP,
            "safe_weight": None,
            "safe_margin": None,
            "se2_target_contract": SE2_TARGET_CONTRACT,
            "se2_shape_contract": SE2_SHAPE_CONTRACT,
            "seed": int(args.seed),
            "steps_per_epoch": int(steps_per_epoch),
            "schedule_total_steps": int(args.schedule_epochs * steps_per_epoch),
            "sampler_protocol": SAMPLER_PROTOCOL,
            "sampler_report": sampler_report,
            "args": vars(args),
            "train_cache_metadata": train_meta,
            "val_cache_metadata": val_meta,
            "val_report": val_report,
        },
        path,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-cache", required=True)
    p.add_argument("--val-cache", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--schedule-epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--semantic-dim", type=int, default=32)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--decoder-blocks", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--yaw-weight", type=float, default=19.0)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=20260915)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-amp", action="store_true")
    a = p.parse_args()

    if min(a.epochs, a.schedule_epochs, a.batch_size) <= 0:
        raise ValueError("epochs/schedule-epochs/batch-size must be positive")
    if a.epochs > a.schedule_epochs:
        raise ValueError("balanced clean run may not exceed schedule horizon")
    if a.lr <= 0 or a.weight_decay < 0 or a.yaw_weight < 0:
        raise ValueError("invalid optimizer/loss arguments")

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)

    device = torch.device(
        a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    amp = device.type == "cuda" and not bool(a.no_amp)

    train_meta, train_records = load_se2_cache(a.train_cache)
    val_meta, val_records = load_se2_cache(a.val_cache)
    train = flatten_supervised(train_records)
    val = flatten_supervised(val_records)
    overlap = sorted(set(train["scene_ids"]) & set(val["scene_ids"]))
    if overlap:
        raise RuntimeError(f"train/val scene overlap: {overlap[:5]}")

    balanced = build_balanced_source_weights(
        train["source_class_id"],
        train["target_source_residual_xy_m"],
        train["se2_target_valid"],
    )

    tube_hw = int(train["local_semantic_tube"].shape[-1])
    patch_resolution = float(train_meta.get("patch_resolution_m", 0.8))
    mcfg = LocalSTWMV17Config(
        d_model=int(a.d_model),
        semantic_dim=int(a.semantic_dim),
        heads=int(a.heads),
        blocks=int(a.blocks),
        decoder_blocks=int(a.decoder_blocks),
        tube_hw=tube_hw,
        use_representation=True,
    )
    model = LocalSpatialTemporalWorldModelV18SE2(mcfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(a.lr), weight_decay=float(a.weight_decay)
    )

    train_dataset = make_dataset(train)
    sampler = make_balanced_source_sampler(balanced, seed=int(a.seed))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(a.batch_size),
        sampler=sampler,
        shuffle=False,
        num_workers=int(a.num_workers),
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_loader = DataLoader(
        make_dataset(val),
        batch_size=int(a.batch_size),
        shuffle=False,
        num_workers=int(a.num_workers),
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    if len(train_loader) != math.ceil(len(train_dataset) / int(a.batch_size)):
        raise RuntimeError("balanced sampler changed steps per epoch")
    steps_per_epoch = len(train_loader)
    total_steps = int(a.schedule_epochs) * steps_per_epoch

    raw0 = next(iter(val_loader))
    b0 = unpack(raw0, device)
    model.eval()
    with torch.no_grad():
        out0 = forward_model(model, b0, amp=amp, device=device)
    init_xy_max = float(out0["residual_xy_m"].abs().max().cpu())
    init_yaw_max = float(out0["yaw_delta_rad"].abs().max().cpu())
    if init_xy_max > 1e-7 or init_yaw_max > 1e-7:
        raise RuntimeError(
            f"fresh balanced model violates KTA/zero-yaw init: "
            f"xy={init_xy_max} yaw={init_yaw_max}"
        )

    initial_val = _eval_objective(
        model, val_loader, device, amp=amp,
        yaw_weight=float(a.yaw_weight), patch_resolution_m=patch_resolution,
    )
    preflight = {
        "protocol": CLEAN_PROTOCOL,
        "training_mode": TRAINING_MODE,
        "model_protocol": MODEL_PROTOCOL_V18_SE2,
        "parameters": sum(int(p.numel()) for p in model.parameters()),
        "model_config": asdict(mcfg),
        "train_sources": int(train["features"].shape[0]),
        "val_sources": int(val["features"].shape[0]),
        "train_scenes": len(set(train["scene_ids"])),
        "val_scenes": len(set(val["scene_ids"])),
        "steps_per_epoch": steps_per_epoch,
        "train_epochs": int(a.epochs),
        "schedule_epochs": int(a.schedule_epochs),
        "schedule_total_steps": total_steps,
        "yaw_weight": float(a.yaw_weight),
        "shape_weight": EXPECTED_OVERLAP,
        "initial_residual_max_abs": init_xy_max,
        "initial_yaw_max_abs": init_yaw_max,
        "initial_val": initial_val,
        "amp_bfloat16": bool(amp),
        "sampler": balanced.report,
        "matched_clean_baseline_contract": {
            "same_architecture": True,
            "same_random_initialization_rule": True,
            "same_optimizer": True,
            "same_loss": True,
            "same_lr_schedule": True,
            "same_num_samples_per_epoch": True,
            "only_treatment": "source_sampling_distribution",
        },
    }
    print("=== V18 BALANCED CLEAN PREFLIGHT ===")
    print(json.dumps(preflight, indent=2))

    out_dir = Path(a.output_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty output dir: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "preflight.json").write_text(json.dumps(preflight, indent=2), encoding="utf-8")
    (out_dir / "sampler_report.json").write_text(
        json.dumps(balanced.report, indent=2), encoding="utf-8"
    )

    history = []
    global_step = 0
    for epoch in range(1, int(a.epochs) + 1):
        model.train()
        sums = {k: 0.0 for k in (
            "objective_loss", "translation_smooth_l1", "existence_bce",
            "yaw_periodic_loss", "se2_shape_loss",
        )}
        nb = 0
        for raw in train_loader:
            b = unpack(raw, device)
            out = forward_model(model, b, amp=amp, device=device)
            loss, stats = se2_objective_loss(
                out, b, yaw_weight=float(a.yaw_weight),
                shape_weight=EXPECTED_OVERLAP,
                patch_resolution_m=patch_resolution, safe_weight=0.0,
            )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(
                    f"non-finite balanced objective epoch={epoch} step={global_step+1}"
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            global_step += 1
            lr = float(a.lr) * _lr_scale(global_step, total_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr

            sums["objective_loss"] += float(loss.detach().cpu())
            for k in (
                "translation_smooth_l1", "existence_bce",
                "yaw_periodic_loss", "se2_shape_loss",
            ):
                sums[k] += float(stats[k])
            nb += 1

        val_report = _eval_objective(
            model, val_loader, device, amp=amp,
            yaw_weight=float(a.yaw_weight), patch_resolution_m=patch_resolution,
        )
        row = {
            "epoch": epoch,
            "global_step": global_step,
            "lr": float(optimizer.param_groups[0]["lr"]),
            **{f"train_{k}": v / max(nb, 1) for k, v in sums.items()},
            **{f"val_{k}": v for k, v in val_report.items()},
        }
        history.append(row)
        print("BALANCED_CLEAN_EPOCH " + json.dumps(row), flush=True)

        _save(
            out_dir / f"epoch_{epoch:04d}.pt", model=model, optimizer=optimizer,
            args=a, epoch=epoch, global_step=global_step, model_config=mcfg,
            train_meta=train_meta, val_meta=val_meta, val_report=val_report,
            steps_per_epoch=steps_per_epoch, sampler_report=balanced.report,
        )
        _save(
            out_dir / "latest.pt", model=model, optimizer=optimizer,
            args=a, epoch=epoch, global_step=global_step, model_config=mcfg,
            train_meta=train_meta, val_meta=val_meta, val_report=val_report,
            steps_per_epoch=steps_per_epoch, sampler_report=balanced.report,
        )

    report = {
        "protocol": CLEAN_PROTOCOL,
        "training_mode": TRAINING_MODE,
        "preflight": preflight,
        "history": history,
        "final_epoch": int(a.epochs),
        "final_global_step": global_step,
    }
    (out_dir / "training_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print("=== V18 BALANCED CLEAN COMPLETE ===")
    print(json.dumps({
        "output_dir": str(out_dir.resolve()),
        "final_checkpoint": str((out_dir / f"epoch_{int(a.epochs):04d}.pt").resolve()),
        "final_global_step": global_step,
    }, indent=2))


if __name__ == "__main__":
    main()
