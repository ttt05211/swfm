#!/usr/bin/env python3
"""Train P0-F4 Strong-W2Det conditioned Top-2 Sparse World Model.

Frozen routing remains the P0-F2 Real-Motion MSP Top-2 plan. The expensive
future transition predicts 20x20 windows from a strong W2Det future anchor,
while each local transition sees a 40x40 crop of full historical occupancy
latent as additional context. The only objective is latent flow MSE, evaluated
only on the same causal MSP support that is authorized to write at inference.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UP = ROOT / "upstream_occfm"
sys.path[:0] = [str(ROOT), str(UP)]

import numpy as np
import torch
from torch.utils.data import DataLoader

from real_motion.checkpoint import load_shape_safe
from real_motion.context import crop_prediction_and_context
from real_motion.models.p0_f4 import make_p0_f4_model
from real_motion.msp_wm_cache import (
    MSP_WM_CACHE_VERSION_V2,
    MSPWorldModelCacheDataset,
    collate_msp_wm,
)
from real_motion.occfm_io import file_sha256
from real_motion.windows import WindowPlan, crop_windows

PRED_HW = (20, 20)
CONTEXT_HW = (40, 40)
FULL_HW = (50, 50)
LOSS_CONTRACT = "strong_anchor_to_gt_local_flow_full_history_context_no_auxiliary_losses"


def _validate_cache_pair(train_ds, val_ds):
    if train_ds.version != MSP_WM_CACHE_VERSION_V2 or val_ds.version != MSP_WM_CACHE_VERSION_V2:
        raise RuntimeError("P0-F4 requires the v2 strong-W2Det/full-history cache")
    tm, vm = train_ds.metadata, val_ds.metadata
    for name, meta in (("train", tm), ("val", vm)):
        if int(meta.get("topk", -1)) != 2:
            raise RuntimeError(f"{name} cache is not frozen Top-2")
        if list(meta.get("window_hw", [])) != list(PRED_HW):
            raise RuntimeError(f"{name} prediction window is not 20x20")
        if list(meta.get("context_hw", [])) != list(CONTEXT_HW):
            raise RuntimeError(f"{name} history context is not 40x40")
        if meta.get("vae_mode") != "mean":
            raise RuntimeError(f"{name} cache must use deterministic VAE mean latents")
        if meta.get("anchor_contract") != "strong_w2det_occ_only_v1":
            raise RuntimeError(f"{name} cache does not use the strong W2Det anchor")
        if meta.get("history_contract") != "full_native_occ_history_6f":
            raise RuntimeError(f"{name} cache does not use full history")
        if meta.get("loss_contract") != LOSS_CONTRACT:
            raise RuntimeError(f"{name} cache target contract mismatch")
    overlap = sorted(set(tm.get("scene_names", [])) & set(vm.get("scene_names", [])))
    if overlap:
        raise RuntimeError(f"train/val scene leakage ({len(overlap)}), e.g. {overlap[:3]}")
    if tm.get("msp_checkpoint_sha256") != vm.get("msp_checkpoint_sha256"):
        raise RuntimeError("train/val caches were routed by different MSP checkpoints")
    if tm.get("vae_checkpoint_sha256") != vm.get("vae_checkpoint_sha256"):
        raise RuntimeError("train/val caches use different VAE checkpoints")
    if float(tm.get("write_budget_ratio", -1)) != float(vm.get("write_budget_ratio", -2)):
        raise RuntimeError("train/val write-budget ratios differ")


def prepare_batch(batch, device):
    """Crop 20x20 prediction state, 40x40 context, and causal FM mask."""
    origins_cpu = batch["window_origins"].long()
    valid_cpu = batch["window_valid"].bool()
    B, K = valid_cpu.shape
    if K != 2:
        raise RuntimeError(f"P0-F4 expects K=2, got {K}")
    plan_cpu = WindowPlan(origins_cpu, valid_cpu, PRED_HW, FULL_HW)
    if not bool(plan_cpu.valid.any()):
        return None

    batch = {
        k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }
    plan = WindowPlan(
        plan_cpu.origins.to(device, non_blocking=True),
        plan_cpu.valid.to(device, non_blocking=True),
        PRED_HW,
        FULL_HW,
    )
    hist_local, hist_context, _ = crop_prediction_and_context(
        batch["full_history_latent"], plan, context_hw=CONTEXT_HW
    )
    anchor = crop_windows(batch["anchor_future_latent"], plan)
    target = crop_windows(batch["gt_future_latent"], plan)
    write = crop_windows(batch["msp_write_support_latent"], plan).bool()

    slot_valid = plan.valid.reshape(-1)
    write_flat = write.reshape(B * K, *write.shape[2:])
    effective = slot_valid & write_flat.reshape(B * K, -1).any(dim=1)
    if not bool(effective.any()):
        return None

    def flat(x):
        return x.reshape(B * K, *x.shape[2:])[effective]

    hist_local, hist_context, anchor, target = map(
        flat, (hist_local, hist_context, anchor, target)
    )
    loss_mask = write_flat[effective].unsqueeze(2)
    origins = plan.origins.reshape(B * K, 2)[effective]
    traj = batch["trajectory"]
    if tuple(traj.shape[1:]) != (12, 2):
        raise RuntimeError(f"trajectory batch must be [B,12,2], got {tuple(traj.shape)}")
    traj = traj[:, None].expand(B, K, 12, 2).reshape(B * K, 12, 2)[effective]
    return hist_local, hist_context, target, anchor, loss_mask, traj, origins


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total = 0.0
    count = 0
    cos = 0.0
    target_rms = 0.0
    pred_rms = 0.0
    active = 0.0
    skipped_batches = 0
    skipped_samples = 0
    for batch in loader:
        prepared = prepare_batch(batch, device)
        if prepared is None:
            skipped_batches += 1
            skipped_samples += len(batch.get("sample_id", []))
            continue
        hist, context, target, anchor, loss_mask, traj, origins = prepared
        loss, info = model.flow_loss(
            hist,
            target,
            anchor,
            history_context=context,
            loss_mask=loss_mask,
            trajectory=traj,
            window_origins=origins,
            t_override=0.5,
            source_noise=torch.zeros_like(anchor),
        )
        n = int(hist.shape[0])
        total += float(loss.item()) * n
        cos += float(info["cosine"]) * n
        target_rms += float(info["target_rms"]) * n
        pred_rms += float(info["pred_rms"]) * n
        active += float(info.get("loss_active_fraction", 1.0)) * n
        count += n
    model.train()
    if count <= 0:
        raise RuntimeError("validation contains no causal MSP write windows")
    d = float(count)
    return {
        "loss": total / d,
        "cosine": cos / d,
        "target_rms": target_rms / d,
        "pred_rms": pred_rms / d,
        "loss_active_fraction": active / d,
        "num_windows": count,
        "skipped_empty_batches": skipped_batches,
        "skipped_anchor_only_samples": skipped_samples,
    }


def _architecture(args):
    return {
        "protocol": "p0_f4_strong_w2det_full_context_v1",
        "window_hw": list(PRED_HW),
        "context_hw": list(CONTEXT_HW),
        "topk": 2,
        "sample_steps": int(args.sample_steps),
        "source_noise_std": float(args.source_noise_std),
        "prior_channels": 16,
        "context_channels": 16,
        "flow": "strong_w2det_anchor_to_full_gt_latent",
        "history": "full_history_latent",
        "loss": "masked_flow_mse_on_causal_msp_write_support",
    }


def _validate_resume_checkpoint(ck, args, train_ds, val_ds):
    arch = ck.get("architecture", {})
    if arch.get("protocol") != "p0_f4_strong_w2det_full_context_v1":
        raise RuntimeError("resume checkpoint is not P0-F4")
    if list(arch.get("window_hw", [])) != list(PRED_HW):
        raise RuntimeError("resume prediction window differs")
    if list(arch.get("context_hw", [])) != list(CONTEXT_HW):
        raise RuntimeError("resume context window differs")
    if arch.get("loss") != "masked_flow_mse_on_causal_msp_write_support":
        raise RuntimeError("resume loss contract differs")
    if int(arch.get("sample_steps", args.sample_steps)) != int(args.sample_steps):
        raise RuntimeError("resume sample_steps differs")
    if float(arch.get("source_noise_std", args.source_noise_std)) != float(args.source_noise_std):
        raise RuntimeError("resume source_noise_std differs")
    for ckmeta, dsmeta, prefix in (
        (ck.get("train_metadata", {}), train_ds.metadata, "train"),
        (ck.get("val_metadata", {}), val_ds.metadata, "val"),
    ):
        for key in ("msp_checkpoint_sha256", "vae_checkpoint_sha256", "anchor_contract"):
            if ckmeta.get(key) and ckmeta.get(key) != dsmeta.get(key):
                raise RuntimeError(f"resume {prefix} metadata mismatch for {key}")


def _payload(
    *, model_state, optimizer, step, best_val, history, train_ds, val_ds,
    args, upstream_ckpt, reuse, skipped_train_batches,
):
    return {
        "state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "step": int(step),
        "best_val_loss": float(best_val),
        "training_history": list(history),
        "train_metadata": train_ds.metadata,
        "val_metadata": val_ds.metadata,
        "architecture": _architecture(args),
        "upstream_checkpoint": str(Path(upstream_ckpt).resolve()),
        "upstream_checkpoint_sha256": file_sha256(upstream_ckpt),
        "upstream_reuse": reuse,
        "skipped_empty_train_batches": int(skipped_train_batches),
        "args": vars(args),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-cache", required=True)
    p.add_argument("--val-cache", required=True)
    p.add_argument("--upstream-ckpt", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--val-every", type=int, default=200)
    p.add_argument("--sample-steps", type=int, default=10)
    p.add_argument("--source-noise-std", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=20260830)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--resume-from", default=None)
    a = p.parse_args()
    if min(a.steps, a.batch_size, a.val_every) <= 0:
        raise ValueError("steps/batch-size/val-every must be positive")
    if a.source_noise_std != 0.0:
        raise ValueError("P0-F4 first run freezes source_noise_std=0")

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    device = torch.device(a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu")

    train_ds = MSPWorldModelCacheDataset(a.train_cache)
    val_ds = MSPWorldModelCacheDataset(a.val_cache)
    _validate_cache_pair(train_ds, val_ds)
    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=a.batch_size,
        shuffle=True,
        num_workers=a.num_workers,
        collate_fn=collate_msp_wm,
        drop_last=False,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=a.batch_size,
        shuffle=False,
        num_workers=a.num_workers,
        collate_fn=collate_msp_wm,
        drop_last=False,
        pin_memory=pin,
    )

    model = make_p0_f4_model(
        20, sample_steps=a.sample_steps, source_noise_std=a.source_noise_std
    ).to(device)
    reuse = load_shape_safe(model.transition, a.upstream_ckpt, verbose=True)
    if "traj_encoder.0.weight" not in set(reuse.get("loaded_keys", ())):
        raise RuntimeError("use the official OccFM-Fut epoch=000196 checkpoint")
    reuse_fraction = float(reuse.get("loaded", 0)) / max(float(reuse.get("target_total", 1)), 1.0)
    if reuse_fraction < 0.90:
        raise RuntimeError(f"unexpectedly low upstream reuse fraction {reuse_fraction:.3f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    use_amp = bool(a.amp and device.type == "cuda")
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    best_state = None
    history = []
    step = 0
    skipped_train_batches = 0
    if a.resume_from:
        ck = torch.load(a.resume_from, map_location="cpu", weights_only=False)
        _validate_resume_checkpoint(ck, a, train_ds, val_ds)
        model.load_state_dict(ck["state_dict"], strict=True)
        step = int(ck.get("step", 0))
        if step < 0 or step > a.steps:
            raise RuntimeError(f"resume step {step} incompatible with target {a.steps}")
        best_val = float(ck.get("best_val_loss", float("inf")))
        history = list(ck.get("training_history", []))
        skipped_train_batches = int(ck.get("skipped_empty_train_batches", 0))
        if np.isfinite(best_val):
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        if ck.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(ck["optimizer_state_dict"])
        print(f"resumed P0-F4 from {a.resume_from} at step={step} best_val={best_val:.6f}")

    iterator = iter(train_loader)
    model.train()
    while step < a.steps:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        prepared = prepare_batch(batch, device)
        if prepared is None:
            skipped_train_batches += 1
            if skipped_train_batches <= 3 or skipped_train_batches % 10 == 0:
                print(f"skip no-write batch: skipped={skipped_train_batches} optimizer_step={step}")
            continue
        hist, context, target, anchor, loss_mask, traj, origins = prepared
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            loss, info = model.flow_loss(
                hist,
                target,
                anchor,
                history_context=context,
                loss_mask=loss_mask,
                trajectory=traj,
                window_origins=origins,
            )
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}: {loss}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        step += 1

        if step == 1 or step % 20 == 0:
            print(
                f"step={step} loss={float(loss.item()):.6f} "
                f"cos={info['cosine']:.4f} target_rms={info['target_rms']:.5f} "
                f"pred_rms={info['pred_rms']:.5f} active={info.get('loss_active_fraction',1.0):.4f}"
            )
        if step % a.val_every == 0 or step == a.steps:
            val = validate(model, val_loader, device)
            row = {"step": step, "train": info, "val": val}
            history.append(row)
            print("validation", json.dumps(row))
            state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            torch.save(
                _payload(
                    model_state=state,
                    optimizer=optimizer,
                    step=step,
                    best_val=min(best_val, float(val["loss"])),
                    history=history,
                    train_ds=train_ds,
                    val_ds=val_ds,
                    args=a,
                    upstream_ckpt=a.upstream_ckpt,
                    reuse=reuse,
                    skipped_train_batches=skipped_train_batches,
                ),
                out / "latest.pt",
            )
            if float(val["loss"]) < best_val:
                best_val = float(val["loss"])
                best_state = state
                torch.save(
                    _payload(
                        model_state=best_state,
                        optimizer=optimizer,
                        step=step,
                        best_val=best_val,
                        history=history,
                        train_ds=train_ds,
                        val_ds=val_ds,
                        args=a,
                        upstream_ckpt=a.upstream_ckpt,
                        reuse=reuse,
                        skipped_train_batches=skipped_train_batches,
                    ),
                    out / "best.pt",
                )

    if best_state is None:
        raise RuntimeError("no validation checkpoint was produced")
    model.load_state_dict(best_state, strict=True)
    torch.save(
        _payload(
            model_state=best_state,
            optimizer=optimizer,
            step=step,
            best_val=best_val,
            history=history,
            train_ds=train_ds,
            val_ds=val_ds,
            args=a,
            upstream_ckpt=a.upstream_ckpt,
            reuse=reuse,
            skipped_train_batches=skipped_train_batches,
        ),
        out / "last.pt",
    )
    (out / "training_report.json").write_text(
        json.dumps({
            "protocol": "p0_f4_strong_w2det_full_context_v1",
            "best_val_loss": best_val,
            "history": history,
            "upstream_reuse_fraction": reuse_fraction,
            "skipped_empty_train_batches": skipped_train_batches,
            "resumed_from": a.resume_from,
            "decision": "Evaluate Strong W2Det vs trained Sparse WM vs same-support GT repair oracle.",
        }, indent=2),
        encoding="utf-8",
    )
    print("saved", out / "best.pt")


if __name__ == "__main__":
    main()
