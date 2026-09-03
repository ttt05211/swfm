#!/usr/bin/env python3
"""Train a fresh P0-F8 edit head on frozen causal deployment endpoints.

This diagnostic closes the edit-head train/deployment endpoint gap without
changing the causal representation.  A completed P0-F8 v2 checkpoint supplies
the transition model.  For every training and validation batch we run the
exact deterministic ODE sampler used at deployment, freeze and detach its
endpoint, and optimize only a newly initialized anchor-relative edit head:

    history + Strong-W2Det anchor -> frozen causal ODE rollout
                                  -> frozen VAE sparse decode
                                  -> fresh trainable edit head

No repair target latent or future GT is an input to the rollout.  Future GT is
used only through the ordinary supervised edit labels.
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
from torch.utils.data import DataLoader

from real_motion.edit_repair import EditTargetCache
from real_motion.models.p0_f8 import AnchorRelativeEditHead, make_p0_f8_model
from real_motion.msp_wm_cache import MSPWorldModelCacheDataset, collate_msp_wm
from real_motion.occfm_io import OccFMVAEAdapter, file_sha256, load_official_vae
from tools.real_motion import p0_f8_train_impl as base
from tools.real_motion import train_p0_f6_decoder_aware_wm as f6
from tools.real_motion.p0_f8_train_impl_v2 import (
    F8_PROTOCOL,
    aggregate_edit_validation_infos,
    edit_loss_for_endpoint_v2,
)

PROBE_PROTOCOL = "p0_f8_frozen_causal_endpoint_edit_head_probe_v1"
ENDPOINT_SOURCE = "frozen_p0_f8_v2_deterministic_deployment_ode_rollout"


def validate_causal_checkpoint(ck: dict) -> dict:
    """Validate the source representation and return its frozen contract."""
    arch = ck.get("architecture") or {}
    if arch.get("protocol") != F8_PROTOCOL:
        raise RuntimeError("causal checkpoint is not P0-F8 v2")
    if arch.get("repair_endpoint_contract") != f6.REPAIR_CONTRACT:
        raise RuntimeError("causal checkpoint repair endpoint contract mismatch")
    if not isinstance(ck.get("state_dict"), dict) or not ck["state_dict"]:
        raise RuntimeError("causal checkpoint has no model state")
    step = int(ck.get("step", 0))
    if step <= 0:
        raise RuntimeError("causal checkpoint must contain a positive training step")
    sample_steps = int(arch.get("sample_steps", 0))
    if sample_steps <= 0:
        raise RuntimeError("causal checkpoint has invalid sample_steps")
    source_noise_std = float(arch.get("source_noise_std", float("nan")))
    if not math.isclose(source_noise_std, 0.0, rel_tol=0.0, abs_tol=0.0):
        raise RuntimeError("probe requires the frozen deterministic source_noise_std=0 contract")
    return {
        "protocol": arch["protocol"],
        "step": step,
        "sample_steps": sample_steps,
        "source_noise_std": source_noise_std,
        "keep_bias": float(arch.get("keep_bias", 2.0)),
    }


def reset_head_and_freeze_transition(model, *, keep_bias: float) -> None:
    """Discard the jointly trained head and make the transition strictly frozen."""
    model.edit_head = AnchorRelativeEditHead(keep_bias=float(keep_bias))
    model.transition.requires_grad_(False)
    model.transition.eval()
    model.edit_head.requires_grad_(True)
    model.edit_head.train()
    if any(p.requires_grad for p in model.transition.parameters()):
        raise RuntimeError("failed to freeze every causal transition parameter")
    if not any(p.requires_grad for p in model.edit_head.parameters()):
        raise RuntimeError("fresh edit head has no trainable parameters")


@torch.no_grad()
def causal_endpoint_from_prepared(model, prepared: dict) -> torch.Tensor:
    """Run the exact deployment sampler and scatter its detached endpoint."""
    model.transition.eval()
    endpoint_windows = model.sample(
        prepared["history"],
        prepared["anchor"],
        history_context=prepared["context"],
        trajectory=prepared["trajectory"],
        window_origins=prepared["origins"],
    )
    endpoint = f6.scatter_endpoint_to_full(endpoint_windows, prepared)
    if endpoint.requires_grad:
        raise RuntimeError("frozen causal endpoint unexpectedly retains gradients")
    return endpoint


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
        "num_ce_predicted_edits": int(info["num_ce_predicted_edits"]),
        "ce_predicted_edit_fraction": info["ce_predicted_edit_fraction"],
        "num_pool_predicted_edits": int(info["num_pool_predicted_edits"]),
        "pool_predicted_edit_fraction": info["pool_predicted_edit_fraction"],
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
def validate_probe(
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
    model.transition.eval()
    model.edit_head.eval()
    infos = []
    num_windows = 0
    skipped_empty_batches = 0
    skipped_no_route_batches = 0
    for batch_idx, batch in enumerate(loader):
        prepared = f6.prepare_batch(batch, device)
        if prepared is None:
            skipped_no_route_batches += 1
            continue
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_amp,
        ):
            endpoint = causal_endpoint_from_prepared(model, prepared)
            loss, info = edit_loss_for_endpoint_v2(
                model,
                endpoint,
                sample_ids=prepared["sample_ids"],
                edit_cache=edit_cache,
                vae=vae,
                keep_ratio=float(keep_ratio),
                keep_when_no_edit=int(keep_when_no_edit),
                lovasz_weight=float(lovasz_weight),
                deterministic=True,
                selection_seed=int(seed) + batch_idx,
            )
        num_windows += int(prepared["history"].shape[0])
        if int(info.get("num_supervised_voxels", 0)) <= 0:
            skipped_empty_batches += 1
        else:
            infos.append(info)
        del endpoint, loss
    model.edit_head.train()
    agg = aggregate_edit_validation_infos(
        infos, lovasz_weight=float(lovasz_weight)
    )
    return {
        "objective": float(agg["edit_loss"]),
        **agg,
        "num_windows": int(num_windows),
        "skipped_empty_batches": int(skipped_empty_batches),
        "skipped_no_route_batches": int(skipped_no_route_batches),
        "endpoint_source": ENDPOINT_SOURCE,
        "validation_aggregation": (
            "CE_by_balanced_voxels; Lovasz_by_full_pool_voxels; "
            "false_edit_by_full_pool_KEEP"
        ),
    }


def _validate_cache_provenance(ck: dict, train_ds, val_ds) -> None:
    source_train = ck.get("train_metadata") or {}
    source_val = ck.get("val_metadata") or {}
    for name, source, current in (
        ("train", source_train, train_ds.metadata),
        ("val", source_val, val_ds.metadata),
    ):
        for key in (
            "msp_checkpoint_sha256",
            "vae_checkpoint_sha256",
            "anchor_contract",
            "repair_endpoint_contract",
            "write_budget_ratio",
            "topk",
        ):
            if source.get(key) != current.get(key):
                raise RuntimeError(
                    f"causal checkpoint {name} cache differs for {key}"
                )
    arch = ck.get("architecture") or {}
    if int(arch.get("train_windows", -1)) != len(train_ds):
        raise RuntimeError("causal checkpoint training-cache size differs")


def _architecture(
    args,
    *,
    source_contract: dict,
    source_sha256: str,
    train_windows: int,
    train_scenes: int,
    trainable_parameters: int,
) -> dict:
    return {
        "protocol": PROBE_PROTOCOL,
        "diagnostic_only": True,
        "causal_deployment_eligible": True,
        "uses_future_gt_as_rollout_input": False,
        "endpoint_source": ENDPOINT_SOURCE,
        "source_causal_protocol": source_contract["protocol"],
        "source_causal_checkpoint_sha256": source_sha256,
        "source_causal_checkpoint_step": int(source_contract["step"]),
        "transition_trainable": False,
        "transition_mode": "eval",
        "source_edit_head_reused": False,
        "trainable_module": "fresh_AnchorRelativeEditHead_only",
        "num_trainable_parameters": int(trainable_parameters),
        "sample_steps": int(source_contract["sample_steps"]),
        "source_noise_std": float(source_contract["source_noise_std"]),
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
        "checkpoint_storage": "edit_head_only; no numbered step checkpoints",
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
    vae_sha256,
    causal_path,
    causal_sha256,
    source_contract,
    args,
    skipped_train_batches,
) -> dict:
    scenes = train_ds.metadata.get("scene_names") or []
    trainable = [p for p in model.edit_head.parameters() if p.requires_grad]
    return {
        "edit_head_state_dict": copy.deepcopy({
            key: value.detach().cpu()
            for key, value in model.edit_head.state_dict().items()
        }),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": int(step),
        "best_val_objective": float(best_val),
        "training_history": list(history),
        "architecture": _architecture(
            args,
            source_contract=source_contract,
            source_sha256=causal_sha256,
            train_windows=len(train_ds),
            train_scenes=len(scenes),
            trainable_parameters=sum(p.numel() for p in trainable),
        ),
        "train_metadata": train_ds.metadata,
        "val_metadata": val_ds.metadata,
        "train_edit_metadata": train_edit.metadata,
        "val_edit_metadata": val_edit.metadata,
        "causal_checkpoint": str(Path(causal_path).resolve()),
        "causal_checkpoint_sha256": causal_sha256,
        "vae_checkpoint": str(Path(vae_path).resolve()),
        "vae_checkpoint_sha256": vae_sha256,
        "skipped_empty_train_batches": int(skipped_train_batches),
        "args": vars(args),
    }


def validate_probe_checkpoint(
    ck: dict,
    *,
    causal_sha256: str,
    vae_sha256: str,
) -> dict:
    """Validate a head-only probe checkpoint before resume or evaluation."""
    arch = ck.get("architecture") or {}
    if arch.get("protocol") != PROBE_PROTOCOL:
        raise RuntimeError("checkpoint is not the frozen causal-endpoint probe")
    if arch.get("endpoint_source") != ENDPOINT_SOURCE:
        raise RuntimeError("probe checkpoint endpoint source mismatch")
    if arch.get("transition_trainable") is not False:
        raise RuntimeError("probe checkpoint does not attest a frozen transition")
    if arch.get("source_edit_head_reused") is not False:
        raise RuntimeError("probe checkpoint did not start from a fresh edit head")
    if ck.get("causal_checkpoint_sha256") != causal_sha256:
        raise RuntimeError("probe checkpoint belongs to a different causal checkpoint")
    if ck.get("vae_checkpoint_sha256") != vae_sha256:
        raise RuntimeError("probe checkpoint belongs to a different VAE")
    state = ck.get("edit_head_state_dict")
    if not isinstance(state, dict) or not state:
        raise RuntimeError("probe checkpoint has no edit-head state")
    return arch


def load_probe_head_into_model(
    model,
    probe_ck: dict,
    *,
    causal_sha256: str,
    vae_sha256: str,
) -> dict:
    """Strictly overlay only the trained probe head onto its source model."""
    arch = validate_probe_checkpoint(
        probe_ck,
        causal_sha256=causal_sha256,
        vae_sha256=vae_sha256,
    )
    model.edit_head.load_state_dict(probe_ck["edit_head_state_dict"], strict=True)
    return arch


def _validate_resume(
    ck: dict,
    args,
    train_ds,
    val_ds,
    *,
    causal_sha256: str,
    vae_sha256: str,
) -> None:
    arch = validate_probe_checkpoint(
        ck,
        causal_sha256=causal_sha256,
        vae_sha256=vae_sha256,
    )
    checks = (
        (float(arch.get("lovasz_weight", -1)), float(args.lovasz_weight), "lovasz-weight"),
        (
            float((arch.get("edit_sampling") or {}).get("keep_ratio", -1)),
            float(args.keep_ratio),
            "keep-ratio",
        ),
        (float(arch.get("keep_bias", -1)), float(args.keep_bias), "keep-bias"),
        (float(arch.get("learning_rate", -1)), float(args.lr), "learning-rate"),
    )
    for got, expected, name in checks:
        if not math.isclose(got, expected, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"resume {name} differs")
    if int(arch.get("train_windows", -1)) != len(train_ds):
        raise RuntimeError("resume training cache size differs")
    source_val = ck.get("val_metadata") or {}
    if source_val.get("vae_checkpoint_sha256") != val_ds.metadata.get(
        "vae_checkpoint_sha256"
    ):
        raise RuntimeError("resume validation cache differs")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-cache", required=True)
    p.add_argument("--val-cache", required=True)
    p.add_argument("--train-edit-targets", required=True)
    p.add_argument("--val-edit-targets", required=True)
    p.add_argument("--causal-ckpt", required=True)
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
    p.add_argument(
        "--collapse-check-step",
        type=int,
        default=200,
        help="Stop after saving if the full validation pool predicts only KEEP.",
    )
    args = p.parse_args()

    if min(args.steps, args.batch_size, args.val_every, args.min_train_windows) <= 0:
        raise ValueError("steps/batch-size/val-every/min-train-windows must be positive")
    if args.lr <= 0 or args.weight_decay < 0:
        raise ValueError("lr must be positive and weight-decay non-negative")
    if args.keep_ratio < 0 or args.keep_when_no_edit < 0 or args.lovasz_weight < 0:
        raise ValueError("keep/lovasz settings must be non-negative")
    if args.collapse_check_step < 0:
        raise ValueError("collapse-check-step must be non-negative")

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

    causal_sha256 = file_sha256(args.causal_ckpt)
    causal_ck = torch.load(args.causal_ckpt, map_location="cpu", weights_only=False)
    source_contract = validate_causal_checkpoint(causal_ck)
    _validate_cache_provenance(causal_ck, train_ds, val_ds)

    vae_path = f6._resolve_vae_path(args, train_ds, val_ds)
    vae_sha256 = file_sha256(vae_path)
    expected_vae = train_ds.metadata.get("vae_checkpoint_sha256")
    if expected_vae and vae_sha256 != expected_vae:
        raise RuntimeError("VAE checkpoint differs from routed caches")

    model = make_p0_f8_model(
        20,
        sample_steps=int(source_contract["sample_steps"]),
        source_noise_std=float(source_contract["source_noise_std"]),
        keep_bias=float(source_contract["keep_bias"]),
    )
    model.load_state_dict(causal_ck["state_dict"], strict=True)
    reset_head_and_freeze_transition(model, keep_bias=float(args.keep_bias))
    model = model.to(device)
    del causal_ck

    head_params = [p for p in model.edit_head.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        head_params, lr=float(args.lr), weight_decay=float(args.weight_decay)
    )
    vae_model, _ = load_official_vae(UP, vae_path, device)
    vae_model.eval()
    vae = OccFMVAEAdapter(vae_model)
    if any(p.requires_grad for p in vae_model.parameters()):
        raise RuntimeError("probe VAE must remain frozen")
    use_amp = bool(args.amp and device.type == "cuda")

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

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    occupied = [
        out / "latest.pt",
        out / "best.pt",
        out / "last.pt",
        out / "training_report.json",
    ]
    if not args.resume_from and any(path.exists() for path in occupied):
        raise RuntimeError("probe output directory already contains run artifacts")

    step = 0
    best_val = float("inf")
    history = []
    skipped_train_batches = 0
    termination = {"status": "COMPLETED", "reason": "target_steps_reached"}
    if args.resume_from:
        ck = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        _validate_resume(
            ck,
            args,
            train_ds,
            val_ds,
            causal_sha256=causal_sha256,
            vae_sha256=vae_sha256,
        )
        model.edit_head.load_state_dict(ck["edit_head_state_dict"], strict=True)
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        step = int(ck.get("step", 0))
        if step < 0 or step > args.steps:
            raise RuntimeError(f"resume step {step} incompatible with target {args.steps}")
        best_val = float(ck.get("best_val_objective", float("inf")))
        history = list(ck.get("training_history") or [])
        skipped_train_batches = int(ck.get("skipped_empty_train_batches", 0))
        print(
            f"resumed frozen causal-endpoint probe from {args.resume_from} "
            f"at step={step} best_val={best_val:.6f}"
        )

    print(json.dumps({
        "protocol": PROBE_PROTOCOL,
        "diagnostic_only": True,
        "endpoint_source": ENDPOINT_SOURCE,
        "source_causal_step": int(source_contract["step"]),
        "sample_steps": int(source_contract["sample_steps"]),
        "transition_trainable_parameters": sum(
            p.numel() for p in model.transition.parameters() if p.requires_grad
        ),
        "edit_head_trainable_parameters": sum(p.numel() for p in head_params),
    }))

    iterator = iter(train_loader)
    model.transition.eval()
    model.edit_head.train()
    latest_train = None
    while step < args.steps:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        prepared = f6.prepare_batch(batch, device)
        if prepared is None:
            skipped_train_batches += 1
            continue

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_amp,
        ):
            endpoint = causal_endpoint_from_prepared(model, prepared)
            loss, info = edit_loss_for_endpoint_v2(
                model,
                endpoint,
                sample_ids=prepared["sample_ids"],
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
            raise RuntimeError(f"non-finite probe edit loss at step {step}: {loss}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(head_params, 5.0)
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
                f"pred_edit={latest_train['pool_predicted_edit_fraction']:.4f} "
                f"E/K={latest_train['num_edits']}/{latest_train['num_keeps']}"
            )

        if step % args.val_every == 0 or step == args.steps:
            val = validate_probe(
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
            collapse_gate = base.assess_all_keep_collapse(
                val,
                step=step,
                check_step=int(args.collapse_check_step),
            )
            row = {
                "step": int(step),
                "train": latest_train,
                "val": val,
                "collapse_gate": collapse_gate,
            }
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
                vae_sha256=vae_sha256,
                causal_path=args.causal_ckpt,
                causal_sha256=causal_sha256,
                source_contract=source_contract,
                args=args,
                skipped_train_batches=skipped_train_batches,
            )
            torch.save(payload, out / "latest.pt")
            if float(val["objective"]) < best_val:
                best_val = float(val["objective"])
                torch.save(payload, out / "best.pt")
            if bool(collapse_gate["stop"]):
                termination = {
                    "status": "EARLY_STOPPED_ALL_KEEP_COLLAPSE",
                    "reason": str(collapse_gate["reason"]),
                    "step": int(step),
                    "gate": collapse_gate,
                }
                print("P0_F8_PROBE_ALL_KEEP_COLLAPSE", json.dumps(termination))
                break

    if latest_train is None and not history:
        raise RuntimeError("no edit-valid frozen-endpoint training batch was observed")
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
        vae_sha256=vae_sha256,
        causal_path=args.causal_ckpt,
        causal_sha256=causal_sha256,
        source_contract=source_contract,
        args=args,
        skipped_train_batches=skipped_train_batches,
    )
    torch.save(final_payload, out / "last.pt")
    report = {
        "protocol": PROBE_PROTOCOL,
        "diagnostic_only": True,
        "causal_deployment_eligible": True,
        "endpoint_source": ENDPOINT_SOURCE,
        "source_causal_checkpoint": str(Path(args.causal_ckpt).resolve()),
        "source_causal_checkpoint_sha256": causal_sha256,
        "source_causal_step": int(source_contract["step"]),
        "steps": int(step),
        "requested_steps": int(args.steps),
        "termination": termination,
        "best_val_objective": float(best_val),
        "trainable_parameters": int(sum(p.numel() for p in head_params)),
        "checkpoint_storage": "head-only best/latest/last; no numbered checkpoints",
        "history": history,
    }
    (out / "training_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
