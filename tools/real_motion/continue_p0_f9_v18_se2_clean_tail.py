#!/usr/bin/env python3
"""Continue a clean one-stage V18 SE(2) checkpoint with a fixed tail LR.

This is a diagnostic continuation only. It restores the exact model and AdamW
state from a clean checkpoint, keeps the final V18 objective unchanged, and
continues for a few extra epochs at the checkpoint's terminal LR (or an
explicit --tail-lr). No curriculum, Safe loss, target change, or optimizer
reinitialization is introduced.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from real_motion.local_st_world_model_v17 import config_from_mapping_v17
from real_motion.local_st_world_model_v18_se2 import LocalSpatialTemporalWorldModelV18SE2
from tools.real_motion.train_p0_f9_v18_se2_pair import (
    EXPECTED_OVERLAP,
    flatten_supervised,
    forward_model,
    load_se2_cache,
    make_dataset,
    se2_objective_loss,
    unpack,
)
from tools.real_motion.train_p0_f9_v18_se2_clean import (
    PROTOCOL,
    _eval_objective,
)


def _save(path, *, model, optimizer, ck, epoch, global_step, val_report, args,
          train_meta, val_meta, steps_per_epoch, tail_lr):
    payload = dict(ck)
    payload.update({
        "protocol": PROTOCOL,
        "arm": "Y",
        "training_mode": "clean_one_stage_from_scratch_v1_tail_continuation",
        "epoch": int(epoch),
        "global_step": int(global_step),
        "continuation_step": int(global_step),
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_config": asdict(model.v17_config),
        "yaw_weight": float(ck.get("yaw_weight", 19.0)),
        "shape_weight": EXPECTED_OVERLAP,
        "safe_weight": None,
        "safe_margin": None,
        "steps_per_epoch": int(steps_per_epoch),
        "tail_lr": float(tail_lr),
        "tail_source_checkpoint": str(Path(args.resume_checkpoint).resolve()),
        "tail_args": vars(args),
        "train_cache_metadata": train_meta,
        "val_cache_metadata": val_meta,
        "val_report": val_report,
    })
    torch.save(payload, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-cache", required=True)
    p.add_argument("--val-cache", required=True)
    p.add_argument("--resume-checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--target-epoch", type=int, default=15)
    p.add_argument("--tail-lr", type=float, default=-1.0,
                   help="<=0 preserves the LR stored in the resume optimizer")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-amp", action="store_true")
    a = p.parse_args()

    ck = torch.load(a.resume_checkpoint, map_location="cpu", weights_only=False)
    if ck.get("protocol") != PROTOCOL:
        raise RuntimeError(f"resume protocol mismatch: {ck.get('protocol')}")
    if str(ck.get("arm")) != "Y":
        raise RuntimeError("clean tail expects arm=Y")
    start_epoch = int(ck.get("epoch", -1))
    if start_epoch <= 0 or int(a.target_epoch) <= start_epoch:
        raise ValueError(f"target epoch must exceed resume epoch {start_epoch}")

    old_args = ck.get("args") or {}
    batch_size = int(old_args.get("batch_size", 256))
    seed = int(ck.get("seed", old_args.get("seed", 20260909)))
    yaw_weight = float(ck.get("yaw_weight", old_args.get("yaw_weight", 19.0)))

    random.seed(seed + start_epoch)
    np.random.seed(seed + start_epoch)
    torch.manual_seed(seed + start_epoch)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + start_epoch)

    device = torch.device(a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and not bool(a.no_amp)

    train_meta, train_records = load_se2_cache(a.train_cache)
    val_meta, val_records = load_se2_cache(a.val_cache)
    train = flatten_supervised(train_records)
    val = flatten_supervised(val_records)

    cfg = config_from_mapping_v17(ck.get("model_config"))
    model = LocalSpatialTemporalWorldModelV18SE2(cfg).to(device)
    model.load_state_dict(ck["state_dict"], strict=True)

    weight_decay = float(old_args.get("weight_decay", 1e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=weight_decay)
    optimizer.load_state_dict(ck["optimizer"])
    stored_lrs = [float(g["lr"]) for g in optimizer.param_groups]
    tail_lr = float(a.tail_lr) if float(a.tail_lr) > 0 else float(stored_lrs[0])
    for g in optimizer.param_groups:
        g["lr"] = tail_lr

    gen = torch.Generator().manual_seed(seed + start_epoch)
    train_loader = DataLoader(
        make_dataset(train), batch_size=batch_size, shuffle=True, generator=gen,
        num_workers=int(a.num_workers), pin_memory=device.type == "cuda", drop_last=False,
    )
    val_loader = DataLoader(
        make_dataset(val), batch_size=batch_size, shuffle=False,
        num_workers=int(a.num_workers), pin_memory=device.type == "cuda", drop_last=False,
    )
    steps_per_epoch = len(train_loader)
    expected_spe = int(ck.get("steps_per_epoch", steps_per_epoch))
    if expected_spe != steps_per_epoch:
        raise RuntimeError(f"steps/epoch mismatch: ckpt={expected_spe} now={steps_per_epoch}")
    global_step = int(ck.get("global_step", start_epoch * steps_per_epoch))
    patch_resolution = float(train_meta.get("patch_resolution_m", 0.8))

    out_dir = Path(a.output_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty output dir: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    preflight = {
        "protocol": PROTOCOL,
        "resume_checkpoint": str(Path(a.resume_checkpoint).resolve()),
        "resume_epoch": start_epoch,
        "target_epoch": int(a.target_epoch),
        "resume_global_step": global_step,
        "steps_per_epoch": steps_per_epoch,
        "batch_size": batch_size,
        "stored_optimizer_lrs": stored_lrs,
        "tail_lr": tail_lr,
        "yaw_weight": yaw_weight,
        "shape_weight": EXPECTED_OVERLAP,
        "amp_bfloat16": bool(amp),
        "note": "same clean V18 objective; restored AdamW state; fixed tail LR",
    }
    print("=== CLEAN V18 TAIL PREFLIGHT ===")
    print(json.dumps(preflight, indent=2))
    (out_dir / "preflight.json").write_text(json.dumps(preflight, indent=2), encoding="utf-8")

    history = []
    for epoch in range(start_epoch + 1, int(a.target_epoch) + 1):
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
                out, b, yaw_weight=yaw_weight, shape_weight=EXPECTED_OVERLAP,
                patch_resolution_m=patch_resolution, safe_weight=0.0,
            )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"non-finite tail objective epoch={epoch} step={global_step+1}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            global_step += 1
            for g in optimizer.param_groups:
                g["lr"] = tail_lr
            sums["objective_loss"] += float(loss.detach().cpu())
            for k in ("translation_smooth_l1", "existence_bce", "yaw_periodic_loss", "se2_shape_loss"):
                sums[k] += float(stats[k])
            nb += 1

        val_report = _eval_objective(
            model, val_loader, device, amp=amp, yaw_weight=yaw_weight,
            patch_resolution_m=patch_resolution,
        )
        row = {
            "epoch": epoch,
            "global_step": global_step,
            "lr": tail_lr,
            **{f"train_{k}": v / max(nb, 1) for k, v in sums.items()},
            **{f"val_{k}": v for k, v in val_report.items()},
        }
        history.append(row)
        print("CLEAN_TAIL_EPOCH " + json.dumps(row), flush=True)
        _save(
            out_dir / f"epoch_{epoch:04d}.pt", model=model, optimizer=optimizer,
            ck=ck, epoch=epoch, global_step=global_step, val_report=val_report,
            args=a, train_meta=train_meta, val_meta=val_meta,
            steps_per_epoch=steps_per_epoch, tail_lr=tail_lr,
        )
        _save(
            out_dir / "latest.pt", model=model, optimizer=optimizer,
            ck=ck, epoch=epoch, global_step=global_step, val_report=val_report,
            args=a, train_meta=train_meta, val_meta=val_meta,
            steps_per_epoch=steps_per_epoch, tail_lr=tail_lr,
        )

    report = {"preflight": preflight, "history": history}
    (out_dir / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("=== CLEAN V18 TAIL COMPLETE ===")
    print(json.dumps({"final_epoch": int(a.target_epoch), "output_dir": str(out_dir)}, indent=2))


if __name__ == "__main__":
    main()
