#!/usr/bin/env python3
"""Clean one-stage training for the final V18 SE(2) motion model.

Unlike the historical Y arm, this script does not warm-start from V17-RL and
never changes objectives mid-training.  From step 1 onward it optimizes exactly

    L = L_trans + L_exist + yaw_weight * L_yaw + 0.25 * L_shape_SE2

with the final V18 architecture, source-centred SE(2) targets, zero-initialized
XY/yaw output heads, source-level shuffled batches, and the historical cosine
LR schedule.  The purpose is to test whether the final method can be trained as
one coherent model rather than relying on the V17->V18 continuation recipe.
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
from real_motion.motion_transport import FEATURE_DIM, FUTURE_FRAMES
from tools.real_motion.train_p0_f9_v18_se2_pair import (
    EXPECTED_OVERLAP,
    flatten_supervised,
    forward_model,
    load_se2_cache,
    make_dataset,
    se2_objective_loss,
    unpack,
)

PROTOCOL = "p0_f9_v18_se2_clean_train_v1"


def _lr_scale(step: int, total_steps: int) -> float:
    frac = min(max(float(step) / max(int(total_steps), 1), 0.0), 1.0)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * frac))


def _eval_objective(model, loader, device, *, amp, yaw_weight, patch_resolution_m):
    model.eval()
    totals = {
        "objective_loss": 0.0,
        "translation_smooth_l1": 0.0,
        "existence_bce": 0.0,
        "yaw_periodic_loss": 0.0,
        "se2_shape_loss": 0.0,
    }
    yaw_num = yaw_den = 0.0
    soft_iou_num = soft_iou_den = 0.0
    ade_num = ade_den = 0.0
    fde_vals = []
    batches = 0
    with torch.no_grad():
        for raw in loader:
            b = unpack(raw, device)
            out = forward_model(model, b, amp=amp, device=device)
            loss, stats = se2_objective_loss(
                out,
                b,
                yaw_weight=float(yaw_weight),
                shape_weight=EXPECTED_OVERLAP,
                patch_resolution_m=float(patch_resolution_m),
                safe_weight=0.0,
            )
            totals["objective_loss"] += float(loss.detach().cpu())
            for k in (
                "translation_smooth_l1",
                "existence_bce",
                "yaw_periodic_loss",
                "se2_shape_loss",
            ):
                totals[k] += float(stats[k])
            n_yaw = int(stats.get("yaw_labels", 0))
            if n_yaw > 0 and np.isfinite(float(stats.get("yaw_mae_rad", float("nan")))):
                yaw_num += float(stats["yaw_mae_rad"]) * n_yaw
                yaw_den += n_yaw
            n_iou = int(stats.get("se2_transport_overlap_labels", 0))
            if n_iou > 0 and np.isfinite(float(stats.get("se2_transport_soft_iou", float("nan")))):
                soft_iou_num += float(stats["se2_transport_soft_iou"]) * n_iou
                soft_iou_den += n_iou

            valid = b["se2_target_valid"].bool()
            target = b["target_source_residual_xy_m"].to(out["residual_xy_m"].dtype)
            err = torch.linalg.vector_norm((out["residual_xy_m"] - target).float(), dim=-1)
            if bool(valid.any()):
                ade_num += float(err[valid].sum().cpu())
                ade_den += int(valid.sum().item())
            for i in range(valid.shape[0]):
                ids = torch.nonzero(valid[i], as_tuple=False).flatten()
                if ids.numel():
                    fde_vals.append(float(err[i, int(ids[-1])].cpu()))
            batches += 1

    out = {k: v / max(batches, 1) for k, v in totals.items()}
    out["yaw_mae_rad"] = yaw_num / yaw_den if yaw_den else float("nan")
    out["se2_transport_soft_iou"] = (
        soft_iou_num / soft_iou_den if soft_iou_den else float("nan")
    )
    out["learned_ade_m"] = ade_num / ade_den if ade_den else float("nan")
    out["learned_fde_m"] = float(np.mean(fde_vals)) if fde_vals else float("nan")
    return out


def _save(path, *, model, optimizer, args, epoch, global_step, model_config,
          train_meta, val_meta, val_report, steps_per_epoch):
    torch.save(
        {
            "protocol": PROTOCOL,
            "model_protocol": MODEL_PROTOCOL_V18_SE2,
            "arm": "Y",
            "training_mode": "clean_one_stage_from_scratch_v1",
            "epoch": int(epoch),
            "global_step": int(global_step),
            # Evaluator prints continuation_step; keep it meaningful as total
            # clean optimizer updates rather than pretending this is a continuation.
            "continuation_step": int(global_step),
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "feature_dim": FEATURE_DIM,
            "future_frames": FUTURE_FRAMES,
            "model_config": asdict(model_config),
            "variant": "CLEAN-SE2",
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
    p.add_argument(
        "--schedule-epochs",
        type=int,
        default=10,
        help="cosine schedule horizon; fixed to historical 10-epoch protocol by default",
    )
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
        raise ValueError("clean run may not exceed the declared cosine schedule horizon")
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

    gen = torch.Generator().manual_seed(int(a.seed))
    train_loader = DataLoader(
        make_dataset(train),
        batch_size=int(a.batch_size),
        shuffle=True,
        generator=gen,
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
    steps_per_epoch = len(train_loader)
    total_steps = int(a.schedule_epochs) * steps_per_epoch

    # Fresh V18 must exactly equal KTA + zero yaw before optimization.
    raw0 = next(iter(val_loader))
    b0 = unpack(raw0, device)
    model.eval()
    with torch.no_grad():
        out0 = forward_model(model, b0, amp=amp, device=device)
    init_xy_max = float(out0["residual_xy_m"].abs().max().cpu())
    init_yaw_max = float(out0["yaw_delta_rad"].abs().max().cpu())
    if init_xy_max > 1e-7 or init_yaw_max > 1e-7:
        raise RuntimeError(
            f"fresh clean model violates KTA/zero-yaw init: xy={init_xy_max} yaw={init_yaw_max}"
        )

    initial_val = _eval_objective(
        model,
        val_loader,
        device,
        amp=amp,
        yaw_weight=float(a.yaw_weight),
        patch_resolution_m=patch_resolution,
    )
    param_count = sum(int(p.numel()) for p in model.parameters())
    preflight = {
        "protocol": PROTOCOL,
        "training_mode": "clean_one_stage_from_scratch_v1",
        "model_protocol": MODEL_PROTOCOL_V18_SE2,
        "parameters": param_count,
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
    }
    print("=== V18 CLEAN ONE-STAGE PREFLIGHT ===")
    print(json.dumps(preflight, indent=2))

    out_dir = Path(a.output_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty output dir: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "preflight.json").write_text(
        json.dumps(preflight, indent=2), encoding="utf-8"
    )

    history = []
    global_step = 0
    for epoch in range(1, int(a.epochs) + 1):
        model.train()
        sums = {
            "objective_loss": 0.0,
            "translation_smooth_l1": 0.0,
            "existence_bce": 0.0,
            "yaw_periodic_loss": 0.0,
            "se2_shape_loss": 0.0,
        }
        nb = 0
        for raw in train_loader:
            b = unpack(raw, device)
            out = forward_model(model, b, amp=amp, device=device)
            loss, stats = se2_objective_loss(
                out,
                b,
                yaw_weight=float(a.yaw_weight),
                shape_weight=EXPECTED_OVERLAP,
                patch_resolution_m=patch_resolution,
                safe_weight=0.0,
            )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(
                    f"non-finite clean objective epoch={epoch} step={global_step+1}"
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
                "translation_smooth_l1",
                "existence_bce",
                "yaw_periodic_loss",
                "se2_shape_loss",
            ):
                sums[k] += float(stats[k])
            nb += 1

        val_report = _eval_objective(
            model,
            val_loader,
            device,
            amp=amp,
            yaw_weight=float(a.yaw_weight),
            patch_resolution_m=patch_resolution,
        )
        row = {
            "epoch": epoch,
            "global_step": global_step,
            "lr": float(optimizer.param_groups[0]["lr"]),
            **{f"train_{k}": v / max(nb, 1) for k, v in sums.items()},
            **{f"val_{k}": v for k, v in val_report.items()},
        }
        history.append(row)
        print("CLEAN_EPOCH " + json.dumps(row), flush=True)

        _save(
            out_dir / f"epoch_{epoch:04d}.pt",
            model=model,
            optimizer=optimizer,
            args=a,
            epoch=epoch,
            global_step=global_step,
            model_config=mcfg,
            train_meta=train_meta,
            val_meta=val_meta,
            val_report=val_report,
            steps_per_epoch=steps_per_epoch,
        )
        _save(
            out_dir / "latest.pt",
            model=model,
            optimizer=optimizer,
            args=a,
            epoch=epoch,
            global_step=global_step,
            model_config=mcfg,
            train_meta=train_meta,
            val_meta=val_meta,
            val_report=val_report,
            steps_per_epoch=steps_per_epoch,
        )

    report = {
        "protocol": PROTOCOL,
        "training_mode": "clean_one_stage_from_scratch_v1",
        "preflight": preflight,
        "history": history,
        "final_epoch": int(a.epochs),
        "final_global_step": global_step,
    }
    (out_dir / "training_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print("=== V18 CLEAN ONE-STAGE TRAINING COMPLETE ===")
    print(json.dumps({
        "output_dir": str(out_dir.resolve()),
        "final_checkpoint": str((out_dir / f"epoch_{int(a.epochs):04d}.pt").resolve()),
        "final_global_step": global_step,
    }, indent=2))


if __name__ == "__main__":
    main()
