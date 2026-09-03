#!/usr/bin/env python3
"""Train the P0-F8 v2 edit head on the GT-derived repair endpoint ceiling.

This is a diagnostic ceiling, not a causal forecasting method.  It bypasses
the World Model completely:

    cached repair_target_latent -> frozen VAE decoder -> edit head

Only the anchor-relative edit head is optimized.  The same v2 CE/Lovasz
populations and the same frozen edit sidecars used by causal P0-F8 are retained
so the result isolates decoder/action-head reachability from causal prediction.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
UP = ROOT / "upstream_occfm"
sys.path[:0] = [str(ROOT), str(UP)]

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from real_motion.edit_repair import EditTargetCache
from real_motion.models.p0_f8 import AnchorRelativeEditHead
from real_motion.msp_wm_cache import MSPWorldModelCacheDataset, collate_msp_wm
from real_motion.occfm_io import OccFMVAEAdapter, file_sha256, load_official_vae
from tools.real_motion import p0_f8_train_impl as base
from tools.real_motion import train_p0_f6_decoder_aware_wm as f6
from tools.real_motion.p0_f8_train_impl_v2 import (
    aggregate_edit_validation_infos,
    edit_loss_for_endpoint_v2,
)

TEACHER_PROTOCOL = "p0_f8_teacher_repair_endpoint_edit_head_v1"
ENDPOINT_SOURCE = "cached_gt_derived_repair_target_latent"


class TeacherEndpointEditModel(nn.Module):
    """A checkpointable container that intentionally has no transition model."""

    def __init__(self, *, keep_bias: float = 2.0):
        super().__init__()
        self.edit_head = AnchorRelativeEditHead(keep_bias=float(keep_bias))


def teacher_endpoint_from_batch(batch: dict, device: torch.device) -> torch.Tensor:
    """Return the exact cached repair endpoint; never infer it from history."""
    if "repair_target_latent" not in batch:
        raise KeyError("teacher ceiling requires repair_target_latent")
    endpoint = batch["repair_target_latent"]
    if not torch.is_tensor(endpoint) or endpoint.ndim != 5:
        raise ValueError("repair_target_latent batch must be [B,F,C,H,W]")
    if tuple(endpoint.shape[1:]) != (6, 16, 50, 50):
        raise ValueError(
            "repair_target_latent must use the frozen [6,16,50,50] contract"
        )
    return endpoint.to(device, non_blocking=True)


def _edit_record(loss: torch.Tensor, info: dict, grad_norm=None) -> dict:
    row = {
        "objective": float(loss.detach().item()),
        "edit_loss": float(loss.detach().item()),
        "edit_ce": float(info["ce"]),
        "result_lovasz": float(info["lovasz"]),
        "action_accuracy": info["accuracy"],
        "edit_accuracy": info["edit_accuracy"],
        "false_edit_rate": info["false_edit_rate"],
        "balanced_false_edit_rate": info["balanced_false_edit_rate"],
        "pool_false_edit_rate": info["pool_false_edit_rate"],
        "dynamic_keep_fraction_realized": info["dynamic_keep_fraction_realized"],
        "num_supervised_voxels": int(info["num_supervised_voxels"]),
        "num_lovasz_voxels": int(info["num_lovasz_voxels"]),
        "num_edits": int(info["num_edits"]),
        "num_keeps": int(info["num_keeps"]),
        "num_dynamic_keeps": int(info["num_dynamic_keeps"]),
        "num_background_keeps": int(info["num_background_keeps"]),
        "num_pool_keeps": int(info["num_pool_keeps"]),
        "num_pool_dynamic_keeps": int(info["num_pool_dynamic_keeps"]),
        "num_pool_background_keeps": int(info["num_pool_background_keeps"]),
        "num_moving_edits": int(info["num_moving_edits"]),
    }
    if grad_norm is not None:
        row["grad_norm_before_clip"] = float(torch.as_tensor(grad_norm).cpu())
    return row


@torch.no_grad()
def validate_teacher(
    model,
    loader,
    device,
    edit_cache,
    vae,
    use_amp,
    *,
    keep_ratio,
    keep_when_no_edit,
    lovasz_weight,
    seed,
) -> dict:
    model.eval()
    infos = []
    num_windows = 0
    skipped_empty_batches = 0
    for batch_idx, batch in enumerate(loader):
        endpoint = teacher_endpoint_from_batch(batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_amp,
        ):
            loss, info = edit_loss_for_endpoint_v2(
                model,
                endpoint,
                sample_ids=batch["sample_id"],
                edit_cache=edit_cache,
                vae=vae,
                keep_ratio=float(keep_ratio),
                keep_when_no_edit=int(keep_when_no_edit),
                lovasz_weight=float(lovasz_weight),
                deterministic=True,
                selection_seed=int(seed) + batch_idx,
            )
        num_windows += int(endpoint.shape[0])
        if int(info.get("num_supervised_voxels", 0)) <= 0:
            skipped_empty_batches += 1
        else:
            infos.append(info)
        del endpoint, loss
    model.train()
    agg = aggregate_edit_validation_infos(
        infos, lovasz_weight=float(lovasz_weight)
    )
    return {
        "objective": float(agg["edit_loss"]),
        **agg,
        "num_windows": int(num_windows),
        "skipped_empty_batches": int(skipped_empty_batches),
        "endpoint_source": ENDPOINT_SOURCE,
        "validation_aggregation": (
            "CE_by_balanced_voxels; Lovasz_by_full_pool_voxels; "
            "false_edit_by_full_pool_KEEP"
        ),
    }


def _architecture(
    args,
    *,
    train_windows: int,
    train_scenes: int,
    trainable_parameters: int,
) -> dict:
    return {
        "protocol": TEACHER_PROTOCOL,
        "diagnostic_only": True,
        "causal_deployment_eligible": False,
        "endpoint_source": ENDPOINT_SOURCE,
        "uses_future_gt": True,
        "transition_model": "none",
        "trainable_module": "AnchorRelativeEditHead_only",
        "num_trainable_parameters": int(trainable_parameters),
        "edit_actions": "KEEP_CLEAR_WRITE_8_dynamic_classes",
        "edit_sampling": {
            "all_edit_voxels": True,
            "keep_ratio": float(args.keep_ratio),
            "keep_when_no_edit": int(args.keep_when_no_edit),
            "keep_stratification": "50pct_dynamic_50pct_background_with_cross_fill",
            "ce_population": "all_EDIT_plus_stratified_balanced_KEEP",
        },
        "edit_loss": "balanced_action_CE_plus_full_pool_result_semantic_Lovasz",
        "lovasz_weight": float(args.lovasz_weight),
        "keep_bias": float(args.keep_bias),
        "learning_rate": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "checkpoint_selection": "population_correct_validation_edit_loss",
        "repair_endpoint_contract": f6.REPAIR_CONTRACT,
        "train_windows": int(train_windows),
        "train_unique_scenes": int(train_scenes),
    }


def _payload(
    *,
    model,
    optimizer,
    step,
    best_val,
    history,
    train_ds,
    val_ds,
    train_edit,
    val_edit,
    vae_path,
    args,
    skipped_train_batches,
) -> dict:
    scenes = train_ds.metadata.get("scene_names") or []
    return {
        "state_dict": copy.deepcopy(
            {key: value.detach().cpu() for key, value in model.state_dict().items()}
        ),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": int(step),
        "best_val_objective": float(best_val),
        "training_history": list(history),
        "architecture": _architecture(
            args,
            train_windows=len(train_ds),
            train_scenes=len(scenes),
            trainable_parameters=sum(p.numel() for p in model.parameters()),
        ),
        "train_metadata": train_ds.metadata,
        "val_metadata": val_ds.metadata,
        "train_edit_metadata": train_edit.metadata,
        "val_edit_metadata": val_edit.metadata,
        "vae_checkpoint": str(Path(vae_path).resolve()),
        "vae_checkpoint_sha256": file_sha256(vae_path),
        "skipped_empty_train_batches": int(skipped_train_batches),
        "args": vars(args),
    }


def _validate_resume(ck: dict, args, train_ds, val_ds, train_edit, val_edit) -> None:
    arch = ck.get("architecture") or {}
    if arch.get("protocol") != TEACHER_PROTOCOL:
        raise RuntimeError("resume checkpoint is not the P0-F8 teacher ceiling")
    if arch.get("endpoint_source") != ENDPOINT_SOURCE:
        raise RuntimeError("resume checkpoint used a different endpoint source")
    checks = (
        (float(arch.get("lovasz_weight", -1)), float(args.lovasz_weight), "lovasz-weight"),
        (
            float((arch.get("edit_sampling") or {}).get("keep_ratio", -1)),
            float(args.keep_ratio),
            "keep-ratio",
        ),
        (float(arch.get("keep_bias", -1)), float(args.keep_bias), "keep-bias"),
    )
    for got, expected, name in checks:
        if not math.isclose(got, expected, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"resume {name} differs")
    if int(arch.get("train_windows", -1)) != len(train_ds):
        raise RuntimeError("resume training cache size differs")
    train_edit.validate_source_cache(train_ds.root)
    val_edit.validate_source_cache(val_ds.root)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-cache", required=True)
    p.add_argument("--val-cache", required=True)
    p.add_argument("--train-edit-targets", required=True)
    p.add_argument("--val-edit-targets", required=True)
    p.add_argument("--vae-ckpt", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--min-train-windows", type=int, default=4000)
    p.add_argument("--val-every", type=int, default=200)
    p.add_argument("--keep-ratio", type=float, default=1.0)
    p.add_argument("--keep-when-no-edit", type=int, default=64)
    p.add_argument("--keep-bias", type=float, default=2.0)
    p.add_argument("--lovasz-weight", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=20260903)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--resume-from", default=None)
    args = p.parse_args()

    if min(args.steps, args.batch_size, args.val_every, args.min_train_windows) <= 0:
        raise ValueError("steps/batch-size/val-every/min-train-windows must be positive")
    if args.lr <= 0 or args.weight_decay < 0:
        raise ValueError("lr must be positive and weight-decay non-negative")
    if args.keep_ratio < 0 or args.keep_when_no_edit < 0 or args.lovasz_weight < 0:
        raise ValueError("keep/lovasz settings must be non-negative")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )

    train_ds = MSPWorldModelCacheDataset(args.train_cache)
    val_ds = MSPWorldModelCacheDataset(args.val_cache)
    f6._validate_cache_pair(train_ds, val_ds)
    base._validate_training_cache(
        train_ds, min_train_windows=int(args.min_train_windows)
    )
    train_edit = EditTargetCache(args.train_edit_targets)
    val_edit = EditTargetCache(args.val_edit_targets)
    base._validate_edit_pair(train_edit, val_edit, train_ds, val_ds)
    vae_path = f6._resolve_vae_path(args, train_ds, val_ds)

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_msp_wm,
        drop_last=False,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_msp_wm,
        drop_last=False,
        pin_memory=pin,
    )

    model = TeacherEndpointEditModel(keep_bias=float(args.keep_bias)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay)
    )
    vae_model, _ = load_official_vae(UP, vae_path, device)
    vae = OccFMVAEAdapter(vae_model)
    if any(p.requires_grad for p in vae_model.parameters()):
        raise RuntimeError("teacher ceiling VAE must remain frozen")
    use_amp = bool(args.amp and device.type == "cuda")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    step = 0
    best_val = float("inf")
    history = []
    skipped_train_batches = 0
    if args.resume_from:
        ck = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        _validate_resume(ck, args, train_ds, val_ds, train_edit, val_edit)
        model.load_state_dict(ck["state_dict"], strict=True)
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        step = int(ck.get("step", 0))
        if step < 0 or step > args.steps:
            raise RuntimeError(f"resume step {step} incompatible with target {args.steps}")
        best_val = float(ck.get("best_val_objective", float("inf")))
        history = list(ck.get("training_history") or [])
        skipped_train_batches = int(ck.get("skipped_empty_train_batches", 0))
        print(
            f"resumed teacher ceiling from {args.resume_from} "
            f"at step={step} best_val={best_val:.6f}"
        )

    print(json.dumps({
        "protocol": TEACHER_PROTOCOL,
        "diagnostic_only": True,
        "endpoint_source": ENDPOINT_SOURCE,
        "transition_model": "none",
        "trainable_parameters": sum(p.numel() for p in model.parameters()),
    }))
    iterator = iter(train_loader)
    model.train()
    latest_train = None
    while step < args.steps:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        endpoint = teacher_endpoint_from_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_amp,
        ):
            loss, info = edit_loss_for_endpoint_v2(
                model,
                endpoint,
                sample_ids=batch["sample_id"],
                edit_cache=train_edit,
                vae=vae,
                keep_ratio=float(args.keep_ratio),
                keep_when_no_edit=int(args.keep_when_no_edit),
                lovasz_weight=float(args.lovasz_weight),
                deterministic=False,
                selection_seed=int(args.seed) + int(step) * 1000003,
            )
        if int(info.get("num_supervised_voxels", 0)) <= 0:
            skipped_train_batches += 1
            continue
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite teacher edit loss at step {step}: {loss}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        step += 1
        latest_train = _edit_record(loss, info, grad_norm)
        if step == 1 or step % 20 == 0:
            print(
                f"step={step} edit={latest_train['edit_loss']:.6f} "
                f"ce={latest_train['edit_ce']:.4f} "
                f"lovasz={latest_train['result_lovasz']:.4f} "
                f"edit_acc={latest_train['edit_accuracy']:.4f} "
                f"pool_false_edit={latest_train['pool_false_edit_rate']:.4f} "
                f"E/K={latest_train['num_edits']}/{latest_train['num_keeps']}"
            )

        if step % args.val_every == 0 or step == args.steps:
            val = validate_teacher(
                model,
                val_loader,
                device,
                val_edit,
                vae,
                use_amp,
                keep_ratio=float(args.keep_ratio),
                keep_when_no_edit=int(args.keep_when_no_edit),
                lovasz_weight=float(args.lovasz_weight),
                seed=int(args.seed) + 90000000,
            )
            row = {"step": int(step), "train": latest_train, "val": val}
            history.append(row)
            print("validation", json.dumps(row))
            next_best = min(best_val, float(val["objective"]))
            payload = _payload(
                model=model,
                optimizer=optimizer,
                step=step,
                best_val=next_best,
                history=history,
                train_ds=train_ds,
                val_ds=val_ds,
                train_edit=train_edit,
                val_edit=val_edit,
                vae_path=vae_path,
                args=args,
                skipped_train_batches=skipped_train_batches,
            )
            torch.save(payload, out / "latest.pt")
            torch.save(payload, out / f"step_{step:04d}.pt")
            if float(val["objective"]) < best_val:
                best_val = float(val["objective"])
                torch.save(payload, out / "best.pt")

    if latest_train is None and not history:
        raise RuntimeError("no edit-valid teacher training batch was observed")
    final_payload = _payload(
        model=model,
        optimizer=optimizer,
        step=step,
        best_val=best_val,
        history=history,
        train_ds=train_ds,
        val_ds=val_ds,
        train_edit=train_edit,
        val_edit=val_edit,
        vae_path=vae_path,
        args=args,
        skipped_train_batches=skipped_train_batches,
    )
    torch.save(final_payload, out / "last.pt")
    report = {
        "protocol": TEACHER_PROTOCOL,
        "diagnostic_only": True,
        "endpoint_source": ENDPOINT_SOURCE,
        "steps": int(step),
        "best_val_objective": float(best_val),
        "trainable_parameters": int(sum(p.numel() for p in model.parameters())),
        "history": history,
    }
    (out / "training_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
