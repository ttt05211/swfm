"""Implementation for P0-F8 anchor-relative edit World Model training.

The public entrypoint is ``train_p0_f8_anchor_relative_edit_wm.py``.  This module
keeps the implementation importable for tests while the entrypoint stays small.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
UP = ROOT / "upstream_occfm"
sys.path[:0] = [str(ROOT), str(UP)]

import numpy as np
import torch
from torch.utils.data import DataLoader

from real_motion.checkpoint import load_shape_safe
from real_motion.edit_repair import (
    DYNAMIC_IDS,
    EditTargetCache,
    anchor_relative_edit_loss,
    horizon_from_flat_indices,
    select_balanced_edit_supervision,
)
from real_motion.models.p0_f8 import make_p0_f8_model
from real_motion.msp_wm_cache import MSPWorldModelCacheDataset, collate_msp_wm
from real_motion.occfm_io import OccFMVAEAdapter, file_sha256, load_official_vae
from tools.real_motion import train_p0_f6_decoder_aware_wm as f6

F8_PROTOCOL = "p0_f8_anchor_relative_edit_wm_v1"
EDIT_PROTOCOL = "p0_f8_anchor_relative_edit_targets_v1"


def _validate_training_cache(train_ds, *, min_train_windows: int) -> None:
    if len(train_ds) < int(min_train_windows):
        raise RuntimeError(
            f"P0-F8 requires >= {min_train_windows} training windows; got {len(train_ds)}"
        )
    if train_ds.metadata.get("source_msp_mode") != "train":
        raise RuntimeError("P0-F8 train cache must originate from MSP train windows")
    if train_ds.metadata.get("source_msp_selection") != "scene_balanced_round_robin_v1":
        raise RuntimeError("P0-F8 train cache must be scene-balanced round-robin")
    scenes = train_ds.metadata.get("scene_names") or []
    if len(scenes) < 2:
        raise RuntimeError("P0-F8 train cache is not scene-diverse")


def _validate_edit_pair(train_edit, val_edit, train_ds, val_ds) -> None:
    train_edit.validate_source_cache(train_ds.root)
    val_edit.validate_source_cache(val_ds.root)
    for name, edit in (("train", train_edit), ("val", val_edit)):
        meta = edit.metadata
        if meta.get("protocol") != EDIT_PROTOCOL:
            raise RuntimeError(f"{name} edit sidecar protocol mismatch")
        if tuple(int(x) for x in meta.get("dynamic_class_ids", [])) != tuple(DYNAMIC_IDS):
            raise RuntimeError(f"{name} edit sidecar dynamic-class mapping mismatch")
        if meta.get("target_contract") != (
            "exact_strong_anchor_relative_keep_clear_write_inside_causal_msp_support"
        ):
            raise RuntimeError(f"{name} edit target contract mismatch")
    if set(train_edit.records) != {str(e["sample_id"]) for e in train_ds.entries}:
        raise RuntimeError("train edit sidecar sample set differs from train WM cache")
    if set(val_edit.records) != {str(e["sample_id"]) for e in val_ds.entries}:
        raise RuntimeError("val edit sidecar sample set differs from val WM cache")


def _build_optimizer(model, reuse, *, lr: float, backbone_lr_scale: float, weight_decay: float):
    if lr <= 0:
        raise ValueError("lr must be positive")
    if not 0.0 <= backbone_lr_scale <= 1.0:
        raise ValueError("backbone-lr-scale must be in [0,1]")
    loaded = set(reuse.get("loaded_keys", ()))
    pretrained, new_params = [], []
    pretrained_names, new_names = [], []
    for name, param in model.named_parameters():
        transition_key = name[len("transition."):] if name.startswith("transition.") else name
        if transition_key in loaded:
            if backbone_lr_scale == 0.0:
                param.requires_grad_(False)
            else:
                pretrained.append(param)
                pretrained_names.append(name)
        else:
            new_params.append(param)
            new_names.append(name)
    if not new_params:
        raise RuntimeError("P0-F8 found no new/unloaded trainable parameters")
    groups = []
    if pretrained:
        groups.append({
            "params": pretrained,
            "lr": float(lr) * float(backbone_lr_scale),
            "group_name": "pretrained_backbone",
        })
    groups.append({
        "params": new_params,
        "lr": float(lr),
        "group_name": "new_or_unloaded",
    })
    optimizer = torch.optim.AdamW(groups, lr=float(lr), weight_decay=float(weight_decay))
    return optimizer, {
        "base_lr": float(lr),
        "backbone_lr_scale": float(backbone_lr_scale),
        "backbone_lr": float(lr) * float(backbone_lr_scale),
        "num_pretrained_tensors": len(pretrained_names),
        "num_new_or_unloaded_tensors": len(new_names),
        "num_pretrained_parameters": int(sum(p.numel() for p in pretrained)),
        "num_new_or_unloaded_parameters": int(sum(p.numel() for p in new_params)),
        "new_or_unloaded_names": new_names,
    }


def calibrate_edit_lambda(
    fm_loss: torch.Tensor,
    edit_loss: torch.Tensor,
    model,
    *,
    target_ratio: float,
    lambda_min: float,
    lambda_max: float,
) -> tuple[float, dict]:
    """Balance the two objectives on their *shared WM transition* parameters.

    The edit head has no FM gradient. Including its private parameters in the
    edit norm would systematically shrink lambda and starve the World Model of
    the decoder-aware repair signal. Only the transition is shared by FM and
    edit objectives, so that is the correct calibration space.
    """
    if target_ratio <= 0:
        raise ValueError("edit grad ratio must be positive")
    if lambda_min <= 0 or lambda_max < lambda_min:
        raise ValueError("invalid edit lambda clamp")
    shared = [p for p in model.transition.parameters() if p.requires_grad]
    if not shared:
        raise RuntimeError("P0-F8 has no trainable shared transition parameters")
    g_fm = f6._grad_l2_norm(fm_loss, shared)
    g_edit = f6._grad_l2_norm(edit_loss, shared)
    head = [p for p in model.edit_head.parameters() if p.requires_grad]
    g_head = f6._grad_l2_norm(edit_loss, head) if head else 0.0
    if not math.isfinite(g_fm) or not math.isfinite(g_edit) or g_fm <= 0 or g_edit <= 0:
        raise RuntimeError(
            f"cannot calibrate P0-F8 edit lambda from shared gradients fm={g_fm} edit={g_edit}"
        )
    raw = float(target_ratio) * g_fm / g_edit
    lam = min(max(raw, float(lambda_min)), float(lambda_max))
    return lam, {
        "calibration_parameters": "shared_transition_only",
        "fm_shared_grad_norm": g_fm,
        "edit_shared_grad_norm_unweighted": g_edit,
        "edit_head_grad_norm_unweighted": g_head,
        "target_edit_to_fm_shared_grad_ratio": float(target_ratio),
        "raw_lambda": raw,
        "lambda": lam,
        "lambda_clamped": not math.isclose(raw, lam, rel_tol=0.0, abs_tol=1e-15),
        "realized_shared_grad_ratio": lam * g_edit / g_fm,
        "weighted_edit_head_to_fm_shared_grad_ratio": lam * g_head / g_fm,
    }


def edit_loss_for_endpoint(
    model,
    endpoint_full: torch.Tensor,
    *,
    sample_ids,
    edit_cache: EditTargetCache,
    vae: OccFMVAEAdapter,
    keep_ratio: float,
    keep_when_no_edit: int,
    lovasz_weight: float,
    deterministic: bool,
    selection_seed: int,
) -> tuple[torch.Tensor, dict]:
    records = edit_cache.get_batch(sample_ids)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(selection_seed))
    selected = [
        select_balanced_edit_supervision(
            rec,
            keep_ratio=float(keep_ratio),
            generator=generator,
            deterministic=bool(deterministic),
            keep_when_no_edit=int(keep_when_no_edit),
        )
        for rec in records
    ]
    indices = [item["flat_indices"].to(endpoint_full.device) for item in selected]
    sparse_semantic_logits = vae.decode_logits_at_flat_indices(endpoint_full, indices)

    action_rows = []
    action_targets = []
    anchor_rows = []
    result_rows = []
    edit_rows = []
    moving_edit_rows = []
    total_edits = 0
    total_keeps = 0
    for logits, sel in zip(sparse_semantic_logits, selected):
        idx = sel["flat_indices"]
        if idx.numel() == 0:
            continue
        horizons = horizon_from_flat_indices(idx).to(endpoint_full.device)
        anchor_slots = sel["anchor_slots"].to(endpoint_full.device)
        action_logits = model.edit_head(logits, anchor_slots, horizons)
        action_rows.append(action_logits)
        action_targets.append(sel["actions"].to(endpoint_full.device))
        anchor_rows.append(anchor_slots)
        result_rows.append(sel["result_slots"].to(endpoint_full.device))
        edit_rows.append(sel["is_edit"].to(endpoint_full.device))
        moving_edit_rows.append(sel["is_moving_edit"].to(endpoint_full.device))
        total_edits += int(sel["num_edits"])
        total_keeps += int(sel["num_keeps"])

    if not action_rows:
        zero = endpoint_full.sum() * 0.0
        return zero, {
            "num_supervised_voxels": 0,
            "num_edits": 0,
            "num_keeps": 0,
            "num_moving_edits": 0,
            "ce": 0.0,
            "lovasz": 0.0,
            "accuracy": float("nan"),
            "edit_accuracy": float("nan"),
            "false_edit_rate": float("nan"),
        }

    action_logits = torch.cat(action_rows, dim=0)
    actions = torch.cat(action_targets, dim=0)
    anchor_slots = torch.cat(anchor_rows, dim=0)
    result_slots = torch.cat(result_rows, dim=0)
    is_edit = torch.cat(edit_rows, dim=0)
    is_moving_edit = torch.cat(moving_edit_rows, dim=0)
    loss, info = anchor_relative_edit_loss(
        action_logits,
        actions,
        anchor_slots,
        result_slots,
        lovasz_weight=float(lovasz_weight),
    )
    info.update({
        "num_supervised_voxels": int(actions.numel()),
        "num_edits": int(total_edits),
        "num_keeps": int(total_keeps),
        "num_moving_edits": int(is_moving_edit.sum().item()),
        "edit_fraction": float(is_edit.float().mean().detach().cpu()),
        "moving_edit_fraction": float(
            is_moving_edit.float().sum().detach().cpu()
            / max(float(is_edit.float().sum().detach().cpu()), 1.0)
        ),
    })
    return loss, info


@torch.no_grad()
def validate(
    model,
    loader,
    device,
    edit_cache,
    vae,
    edit_lambda,
    use_amp,
    *,
    keep_ratio,
    keep_when_no_edit,
    lovasz_weight,
    seed,
):
    model.eval()
    fm_sum = 0.0
    fm_windows = 0
    edit_sum = 0.0
    edit_voxels = 0
    ce_sum = 0.0
    lovasz_sum = 0.0
    correct_sum = 0.0
    edit_acc_sum = 0.0
    edit_acc_weight = 0
    false_edit_sum = 0.0
    false_edit_weight = 0
    total_edits = 0
    total_keeps = 0
    total_moving_edits = 0
    cos_sum = 0.0
    skipped_batches = 0

    for batch_idx, batch in enumerate(loader):
        prepared = f6.prepare_batch(batch, device)
        if prepared is None:
            skipped_batches += 1
            continue
        # Decoder-aware validation needs autograd internally because the frozen
        # VAE adapter uses custom differentiable sparse decode paths.
        with torch.enable_grad():
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_amp,
            ):
                fm_loss, info = model.flow_loss(
                    prepared["history"],
                    prepared["target"],
                    prepared["anchor"],
                    history_context=prepared["context"],
                    trajectory=prepared["trajectory"],
                    window_origins=prepared["origins"],
                    t_override=0.5,
                    source_noise=torch.zeros_like(prepared["anchor"]),
                    return_endpoint=True,
                )
                endpoint = f6.scatter_endpoint_to_full(info["predicted_endpoint"], prepared)
                edit_loss, edit_info = edit_loss_for_endpoint(
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
        nwin = int(prepared["history"].shape[0])
        nvox = int(edit_info["num_supervised_voxels"])
        fm_sum += float(fm_loss.detach().item()) * nwin
        fm_windows += nwin
        cos_sum += float(info["cosine"]) * nwin
        if nvox > 0:
            edit_sum += float(edit_loss.detach().item()) * nvox
            ce_sum += float(edit_info["ce"]) * nvox
            lovasz_sum += float(edit_info["lovasz"]) * nvox
            correct_sum += float(edit_info["accuracy"]) * nvox
            ne = int(edit_info["num_edits"])
            nk = int(edit_info["num_keeps"])
            if ne > 0 and math.isfinite(float(edit_info["edit_accuracy"])):
                edit_acc_sum += float(edit_info["edit_accuracy"]) * ne
                edit_acc_weight += ne
            if nk > 0 and math.isfinite(float(edit_info["false_edit_rate"])):
                false_edit_sum += float(edit_info["false_edit_rate"]) * nk
                false_edit_weight += nk
            edit_voxels += nvox
            total_edits += ne
            total_keeps += nk
            total_moving_edits += int(edit_info["num_moving_edits"])
        del endpoint, edit_loss, fm_loss, info

    model.train()
    if fm_windows <= 0:
        raise RuntimeError("validation contains no valid routed Sparse-WM windows")
    if edit_voxels <= 0:
        raise RuntimeError("validation contains no P0-F8 edit supervision voxels")
    fm_avg = fm_sum / float(fm_windows)
    edit_avg = edit_sum / float(edit_voxels)
    return {
        "objective": fm_avg + float(edit_lambda) * edit_avg,
        "fm_loss": fm_avg,
        "edit_loss": edit_avg,
        "lambda_edit": float(edit_lambda),
        "weighted_edit_loss": float(edit_lambda) * edit_avg,
        "edit_ce": ce_sum / float(edit_voxels),
        "result_lovasz": lovasz_sum / float(edit_voxels),
        "action_accuracy": correct_sum / float(edit_voxels),
        "edit_accuracy": edit_acc_sum / float(edit_acc_weight) if edit_acc_weight else float("nan"),
        "false_edit_rate": false_edit_sum / float(false_edit_weight) if false_edit_weight else float("nan"),
        "num_supervised_voxels": edit_voxels,
        "num_edits": total_edits,
        "num_keeps": total_keeps,
        "num_moving_edits": total_moving_edits,
        "cosine": cos_sum / float(fm_windows),
        "num_windows": fm_windows,
        "skipped_empty_batches": skipped_batches,
    }


def _architecture(args, edit_lambda, optimizer_summary, train_ds):
    return {
        "protocol": F8_PROTOCOL,
        "window_hw": list(f6.PRED_HW),
        "context_hw": list(f6.CONTEXT_HW),
        "topk": 2,
        "sample_steps": int(args.sample_steps),
        "source_noise_std": float(args.source_noise_std),
        "flow": "strong_w2det_anchor_to_encoded_occ_repair_endpoint_uniform_fm",
        "base_loss": "uniform_full_window_flow_mse",
        "edit_actions": "KEEP_CLEAR_WRITE_8_dynamic_classes",
        "edit_head_input": (
            "frozen_decoder_dynamic_semantic_logprobs+exact_anchor_slot+horizon_embedding"
        ),
        "edit_sampling": {
            "all_edit_voxels": True,
            "keep_ratio": float(args.keep_ratio),
            "keep_when_no_edit": int(args.keep_when_no_edit),
            "keep_priority": (
                "exact_true-moving_correct_dynamic > correct_dynamic > background_near_edit"
            ),
        },
        "edit_loss": "balanced_action_CE_plus_result_semantic_Lovasz",
        "lovasz_weight": float(args.lovasz_weight),
        "edit_lambda": float(edit_lambda),
        "edit_grad_ratio": float(args.edit_grad_ratio),
        "edit_lambda_mode": (
            "explicit" if args.edit_lambda is not None
            else "first_batch_shared_transition_gradient_calibration"
        ),
        "keep_bias": float(args.keep_bias),
        "optimizer_groups": optimizer_summary,
        "repair_endpoint_contract": f6.REPAIR_CONTRACT,
        "deployment": (
            "exact_Strong_W2Det_outside_support; "
            "KEEP/CLEAR/WRITE_only_inside_causal_MSP_support"
        ),
        "checkpoint_selection": (
            "edit_aware_validation_objective_but_all_validation_steps_are_saved"
        ),
        "train_windows": len(train_ds),
        "train_unique_scenes": int(
            train_ds.metadata.get(
                "num_unique_scenes", len(train_ds.metadata.get("scene_names", []))
            )
        ),
        "train_selection": train_ds.metadata.get("source_msp_selection"),
    }


def _payload(
    *,
    model_state,
    optimizer,
    step,
    best_val,
    history,
    train_ds,
    val_ds,
    train_edit,
    val_edit,
    args,
    upstream_ckpt,
    reuse,
    optimizer_summary,
    skipped_train_batches,
    edit_lambda,
    calibration,
):
    return {
        "state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "step": int(step),
        "best_val_objective": float(best_val),
        "edit_lambda": float(edit_lambda),
        "edit_lambda_calibration": calibration,
        "training_history": list(history),
        "train_metadata": train_ds.metadata,
        "val_metadata": val_ds.metadata,
        "train_edit_metadata": train_edit.metadata,
        "val_edit_metadata": val_edit.metadata,
        "architecture": _architecture(args, edit_lambda, optimizer_summary, train_ds),
        "upstream_checkpoint": str(Path(upstream_ckpt).resolve()),
        "upstream_checkpoint_sha256": file_sha256(upstream_ckpt),
        "upstream_reuse": reuse,
        "optimizer_group_summary": optimizer_summary,
        "skipped_empty_train_batches": int(skipped_train_batches),
        "args": vars(args),
    }


def _validate_resume(ck, args, train_ds, val_ds, train_edit, val_edit):
    arch = ck.get("architecture", {})
    if arch.get("protocol") != F8_PROTOCOL:
        raise RuntimeError("resume checkpoint is not P0-F8")
    checks = (
        (float(arch.get("lovasz_weight", -1.0)), float(args.lovasz_weight), "lovasz-weight"),
        (
            float(arch.get("edit_sampling", {}).get("keep_ratio", -1.0)),
            float(args.keep_ratio),
            "keep-ratio",
        ),
        (
            float(arch.get("optimizer_groups", {}).get("backbone_lr_scale", -1.0)),
            float(args.backbone_lr_scale),
            "backbone-lr-scale",
        ),
        (float(arch.get("keep_bias", -999.0)), float(args.keep_bias), "keep-bias"),
    )
    for got, expected, name in checks:
        if not math.isclose(got, expected, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"resume {name} differs")
    if int(arch.get("sample_steps", -1)) != int(args.sample_steps):
        raise RuntimeError("resume sample-steps differs")
    if int(arch.get("train_windows", -1)) != len(train_ds):
        raise RuntimeError("resume training cache size differs")
    train_edit.validate_source_cache(train_ds.root)
    val_edit.validate_source_cache(val_ds.root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--val-cache", required=True)
    parser.add_argument("--train-edit-targets", required=True)
    parser.add_argument("--val-edit-targets", required=True)
    parser.add_argument("--upstream-ckpt", required=True)
    parser.add_argument("--vae-ckpt", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--backbone-lr-scale", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--min-train-windows", type=int, default=4000)
    parser.add_argument("--val-every", type=int, default=200)
    parser.add_argument("--sample-steps", type=int, default=10)
    parser.add_argument("--source-noise-std", type=float, default=0.0)
    parser.add_argument("--keep-ratio", type=float, default=1.0)
    parser.add_argument("--keep-when-no-edit", type=int, default=64)
    parser.add_argument("--keep-bias", type=float, default=2.0)
    parser.add_argument("--lovasz-weight", type=float, default=0.5)
    parser.add_argument("--edit-lambda", type=float, default=None)
    parser.add_argument("--edit-grad-ratio", type=float, default=0.5)
    parser.add_argument("--edit-lambda-min", type=float, default=1e-4)
    parser.add_argument("--edit-lambda-max", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume-from", default=None)
    args = parser.parse_args()

    if min(args.steps, args.batch_size, args.val_every, args.min_train_windows) <= 0:
        raise ValueError("steps/batch-size/val-every/min-train-windows must be positive")
    if args.source_noise_std != 0.0:
        raise ValueError("P0-F8 freezes source_noise_std=0")
    if not 0.0 <= args.backbone_lr_scale <= 1.0:
        raise ValueError("backbone-lr-scale must be in [0,1]")
    if args.keep_ratio < 0 or args.keep_when_no_edit < 0 or args.lovasz_weight < 0:
        raise ValueError("keep/lovasz settings must be non-negative")
    if args.edit_lambda is not None and args.edit_lambda <= 0:
        raise ValueError("explicit edit lambda must be positive")
    if args.edit_grad_ratio <= 0:
        raise ValueError("edit-grad-ratio must be positive")

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
    _validate_training_cache(train_ds, min_train_windows=int(args.min_train_windows))

    train_edit = EditTargetCache(args.train_edit_targets)
    val_edit = EditTargetCache(args.val_edit_targets)
    _validate_edit_pair(train_edit, val_edit, train_ds, val_ds)
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

    model = make_p0_f8_model(
        20,
        sample_steps=args.sample_steps,
        source_noise_std=args.source_noise_std,
        keep_bias=args.keep_bias,
    ).to(device)
    reuse = load_shape_safe(model.transition, args.upstream_ckpt, verbose=True)
    if "traj_encoder.0.weight" not in set(reuse.get("loaded_keys", ())):
        raise RuntimeError("use the official OccFM-Fut epoch=000196 checkpoint")
    reuse_fraction = float(reuse.get("loaded", 0)) / max(
        float(reuse.get("target_total", 1)), 1.0
    )
    if reuse_fraction < 0.90:
        raise RuntimeError(f"unexpectedly low upstream reuse fraction {reuse_fraction:.3f}")

    optimizer, optimizer_summary = _build_optimizer(
        model,
        reuse,
        lr=float(args.lr),
        backbone_lr_scale=float(args.backbone_lr_scale),
        weight_decay=float(args.weight_decay),
    )
    print("optimizer_groups", json.dumps(optimizer_summary))

    vae_model, _ = load_official_vae(UP, vae_path, device)
    vae = OccFMVAEAdapter(vae_model)
    if any(p.requires_grad for p in vae_model.parameters()):
        raise RuntimeError("P0-F8 VAE must remain frozen")

    use_amp = bool(args.amp and device.type == "cuda")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    history = []
    step = 0
    skipped_train_batches = 0
    edit_lambda = float(args.edit_lambda) if args.edit_lambda is not None else None
    calibration = None

    if args.resume_from:
        ck = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        _validate_resume(ck, args, train_ds, val_ds, train_edit, val_edit)
        model.load_state_dict(ck["state_dict"], strict=True)
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        step = int(ck.get("step", 0))
        if step < 0 or step > args.steps:
            raise RuntimeError(f"resume step {step} incompatible with target {args.steps}")
        best_val = float(ck.get("best_val_objective", float("inf")))
        history = list(ck.get("training_history", []))
        skipped_train_batches = int(ck.get("skipped_empty_train_batches", 0))
        edit_lambda = float(ck["edit_lambda"])
        calibration = ck.get("edit_lambda_calibration")
        print(
            f"resumed P0-F8 from {args.resume_from} at step={step} "
            f"best_val={best_val:.6f} lambda_edit={edit_lambda:.8g}"
        )

    iterator = iter(train_loader)
    model.train()
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
            fm_loss, info = model.flow_loss(
                prepared["history"],
                prepared["target"],
                prepared["anchor"],
                history_context=prepared["context"],
                trajectory=prepared["trajectory"],
                window_origins=prepared["origins"],
                return_endpoint=True,
            )
            endpoint = f6.scatter_endpoint_to_full(info["predicted_endpoint"], prepared)
            edit_loss, edit_info = edit_loss_for_endpoint(
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

        if int(edit_info["num_supervised_voxels"]) <= 0 and edit_lambda is None:
            print("skip edit-empty batch before lambda calibration")
            continue
        if not torch.isfinite(fm_loss) or not torch.isfinite(edit_loss):
            raise RuntimeError(
                f"non-finite P0-F8 loss at step {step}: fm={fm_loss} edit={edit_loss}"
            )

        if edit_lambda is None:
            edit_lambda, calibration = calibrate_edit_lambda(
                fm_loss,
                edit_loss,
                model,
                target_ratio=float(args.edit_grad_ratio),
                lambda_min=float(args.edit_lambda_min),
                lambda_max=float(args.edit_lambda_max),
            )
            calibration.update({
                "step_before_update": int(step),
                "fm_loss": float(fm_loss.detach().item()),
                "edit_loss": float(edit_loss.detach().item()),
                "edit_ce": float(edit_info["ce"]),
                "result_lovasz": float(edit_info["lovasz"]),
                "num_supervised_voxels": int(edit_info["num_supervised_voxels"]),
                "num_edits": int(edit_info["num_edits"]),
                "num_keeps": int(edit_info["num_keeps"]),
            })
            print("edit_lambda_calibration", json.dumps(calibration))

        total_loss = fm_loss + float(edit_lambda) * edit_loss
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 5.0
        )
        optimizer.step()
        step += 1

        train_info = {
            "objective": float(total_loss.detach().item()),
            "fm_loss": float(fm_loss.detach().item()),
            "edit_loss": float(edit_loss.detach().item()),
            "lambda_edit": float(edit_lambda),
            "weighted_edit_loss": float(edit_lambda) * float(edit_loss.detach().item()),
            "edit_ce": float(edit_info["ce"]),
            "result_lovasz": float(edit_info["lovasz"]),
            "action_accuracy": edit_info["accuracy"],
            "edit_accuracy": edit_info["edit_accuracy"],
            "false_edit_rate": edit_info["false_edit_rate"],
            "num_supervised_voxels": int(edit_info["num_supervised_voxels"]),
            "num_edits": int(edit_info["num_edits"]),
            "num_keeps": int(edit_info["num_keeps"]),
            "num_moving_edits": int(edit_info["num_moving_edits"]),
            "cosine": info["cosine"],
            "target_rms": info["target_rms"],
            "pred_rms": info["pred_rms"],
            "grad_norm_before_clip": float(torch.as_tensor(grad_norm).cpu()),
        }
        if step == 1 or step % 20 == 0:
            print(
                f"step={step} total={train_info['objective']:.6f} "
                f"fm={train_info['fm_loss']:.6f} edit={train_info['edit_loss']:.6f} "
                f"ce={train_info['edit_ce']:.4f} lovasz={train_info['result_lovasz']:.4f} "
                f"lambda={edit_lambda:.6g} edit_acc={train_info['edit_accuracy']:.4f} "
                f"false_edit={train_info['false_edit_rate']:.4f} "
                f"E/K={train_info['num_edits']}/{train_info['num_keeps']} "
                f"cos={train_info['cosine']:.4f}"
            )

        if step % args.val_every == 0 or step == args.steps:
            val = validate(
                model,
                val_loader,
                device,
                val_edit,
                vae,
                edit_lambda,
                use_amp,
                keep_ratio=float(args.keep_ratio),
                keep_when_no_edit=int(args.keep_when_no_edit),
                lovasz_weight=float(args.lovasz_weight),
                seed=int(args.seed) + 90000000,
            )
            row = {"step": step, "train": train_info, "val": val}
            history.append(row)
            print("validation", json.dumps(row))
            state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            next_best = min(best_val, float(val["objective"]))
            payload_args = dict(
                optimizer=optimizer,
                step=step,
                best_val=next_best,
                history=history,
                train_ds=train_ds,
                val_ds=val_ds,
                train_edit=train_edit,
                val_edit=val_edit,
                args=args,
                upstream_ckpt=args.upstream_ckpt,
                reuse=reuse,
                optimizer_summary=optimizer_summary,
                skipped_train_batches=skipped_train_batches,
                edit_lambda=edit_lambda,
                calibration=calibration,
            )
            payload = _payload(model_state=state, **payload_args)
            torch.save(payload, out / "latest.pt")
            torch.save(payload, out / f"step_{step:04d}.pt")
            if float(val["objective"]) < best_val:
                best_val = float(val["objective"])
                torch.save(payload, out / "best.pt")

    if edit_lambda is None:
        raise RuntimeError("no edit-valid P0-F8 training batch was observed")
    final_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
    torch.save(
        _payload(
            model_state=final_state,
            optimizer=optimizer,
            step=step,
            best_val=best_val,
            history=history,
            train_ds=train_ds,
            val_ds=val_ds,
            train_edit=train_edit,
            val_edit=val_edit,
            args=args,
            upstream_ckpt=args.upstream_ckpt,
            reuse=reuse,
            optimizer_summary=optimizer_summary,
            skipped_train_batches=skipped_train_batches,
            edit_lambda=edit_lambda,
            calibration=calibration,
        ),
        out / "last.pt",
    )
    report = {
        "protocol": F8_PROTOCOL,
        "steps": int(step),
        "best_val_objective": float(best_val),
        "edit_lambda": float(edit_lambda),
        "keep_ratio": float(args.keep_ratio),
        "lovasz_weight": float(args.lovasz_weight),
        "optimizer_groups": optimizer_summary,
        "train_windows": len(train_ds),
        "history": history,
    }
    (out / "training_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
