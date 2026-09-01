#!/usr/bin/env python3
"""Train P0-F7 Innovation-Weighted Anchor World Model.

P0-F7 keeps P0-F6 routing, endpoint, decoder-aware semantic supervision, and
inference contract fixed.  It tests three tightly scoped changes:

1) uniform FM MSE -> soft innovation-energy weighted FM MSE;
2) require a substantially larger scene-balanced training cache by default;
3) fine-tune upstream-loaded OccFM parameters at a smaller LR than parameters
   not reused from the official checkpoint.

Loss:
    L = L_weighted_FM + lambda_sem * L_sem

No new router, selector, writeback rule, or auxiliary loss is introduced.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
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
from real_motion.models.p0_f7 import make_p0_f7_model
from real_motion.msp_wm_cache import MSPWorldModelCacheDataset, collate_msp_wm
from real_motion.occfm_io import OccFMVAEAdapter, file_sha256, load_official_vae
from real_motion.semantic_repair import SemanticTargetCache
from tools.real_motion import train_p0_f6_decoder_aware_wm as f6

F7_PROTOCOL = "p0_f7_innovation_weighted_anchor_wm_v1"


def _validate_f7_training_cache(train_ds, *, min_train_windows: int) -> None:
    if len(train_ds) < int(min_train_windows):
        raise RuntimeError(
            f"P0-F7 requires >= {min_train_windows} training windows; got {len(train_ds)}. "
            "Build the scene-balanced 8k cache or explicitly lower --min-train-windows for a smoke test."
        )
    meta = train_ds.metadata
    if meta.get("source_msp_mode") != "train":
        raise RuntimeError("P0-F7 train cache must originate from MSP train windows")
    selection = str(meta.get("source_msp_selection", ""))
    if selection != "scene_balanced_round_robin_v1":
        raise RuntimeError(
            "P0-F7 expanded train cache must use scene_balanced_round_robin_v1; "
            f"got {selection!r}"
        )
    scenes = meta.get("scene_names") or []
    if len(scenes) < 2:
        raise RuntimeError("P0-F7 train cache is not scene-diverse")


def _build_optimizer(model, reuse, *, lr: float, backbone_lr_scale: float, weight_decay: float):
    if lr <= 0:
        raise ValueError("lr must be positive")
    if not 0.0 <= backbone_lr_scale <= 1.0:
        raise ValueError("backbone-lr-scale must be in [0,1]")

    loaded = set(reuse.get("loaded_keys", ()))
    pretrained = []
    new_or_unloaded = []
    pretrained_names = []
    new_names = []

    for name, p in model.named_parameters():
        # load_shape_safe is called on model.transition, so its loaded keys do
        # not contain the leading "transition." used by the wrapper model.
        transition_key = name[len("transition."):] if name.startswith("transition.") else name
        if transition_key in loaded:
            if backbone_lr_scale == 0.0:
                p.requires_grad_(False)
            else:
                pretrained.append(p)
                pretrained_names.append(name)
        else:
            new_or_unloaded.append(p)
            new_names.append(name)

    if not new_or_unloaded:
        raise RuntimeError("P0-F7 found no new/unloaded trainable parameters")
    groups = []
    if pretrained:
        groups.append({
            "params": pretrained,
            "lr": float(lr) * float(backbone_lr_scale),
            "group_name": "pretrained_backbone",
        })
    groups.append({
        "params": new_or_unloaded,
        "lr": float(lr),
        "group_name": "new_or_unloaded",
    })
    optimizer = torch.optim.AdamW(groups, lr=float(lr), weight_decay=float(weight_decay))
    summary = {
        "base_lr": float(lr),
        "backbone_lr_scale": float(backbone_lr_scale),
        "backbone_lr": float(lr) * float(backbone_lr_scale),
        "num_pretrained_tensors": len(pretrained_names),
        "num_new_or_unloaded_tensors": len(new_names),
        "num_pretrained_parameters": int(sum(p.numel() for p in pretrained)),
        "num_new_or_unloaded_parameters": int(sum(p.numel() for p in new_or_unloaded)),
        "new_or_unloaded_names": new_names,
    }
    return optimizer, summary


@torch.no_grad()
def validate(
    model,
    loader,
    device,
    semantic_cache,
    vae,
    semantic_lambda,
    use_amp,
    innovation_weight_alpha,
):
    model.eval()
    fm_sum = 0.0
    unweighted_fm_sum = 0.0
    fm_windows = 0
    sem_sum = 0.0
    sem_voxels = 0
    cos_sum = 0.0
    target_rms_sum = 0.0
    pred_rms_sum = 0.0
    weight_max_sum = 0.0
    focus_sum = 0.0
    sem_correct_weighted = 0.0
    skipped_batches = 0
    skipped_samples = 0

    for batch in loader:
        prepared = f6.prepare_batch(batch, device)
        if prepared is None:
            skipped_batches += 1
            skipped_samples += len(batch.get("sample_id", []))
            continue
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
                    innovation_weight_alpha=float(innovation_weight_alpha),
                )
                endpoint = f6.scatter_endpoint_to_full(info["predicted_endpoint"], prepared)
                sem_loss, sem_info = f6.semantic_loss_for_endpoint(
                    endpoint,
                    sample_ids=prepared["sample_ids"],
                    semantic_cache=semantic_cache,
                    vae=vae,
                )

        nwin = int(prepared["history"].shape[0])
        nvox = int(sem_info["num_supervised_voxels"])
        fm_sum += float(fm_loss.detach().item()) * nwin
        unweighted_fm_sum += float(info["unweighted_fm_loss"]) * nwin
        fm_windows += nwin
        cos_sum += float(info["cosine"]) * nwin
        target_rms_sum += float(info["target_rms"]) * nwin
        pred_rms_sum += float(info["pred_rms"]) * nwin
        weight_max_sum += float(info["innovation_weight_max"]) * nwin
        focus_sum += float(info["innovation_focus_mean"]) * nwin
        if nvox > 0:
            sem_sum += float(sem_loss.detach().item()) * nvox
            sem_voxels += nvox
            sem_correct_weighted += float(sem_info["accuracy"]) * nvox
        del endpoint, sem_loss, fm_loss, info

    model.train()
    if fm_windows <= 0:
        raise RuntimeError("validation contains no valid routed Sparse-WM windows")
    if sem_voxels <= 0:
        raise RuntimeError("validation contains no P0-F7 semantic supervision voxels")

    fm_avg = fm_sum / float(fm_windows)
    unweighted_fm_avg = unweighted_fm_sum / float(fm_windows)
    sem_avg = sem_sum / float(sem_voxels)
    total = fm_avg + float(semantic_lambda) * sem_avg
    return {
        "objective": total,
        "weighted_fm_loss": fm_avg,
        "unweighted_fm_loss": unweighted_fm_avg,
        "semantic_loss": sem_avg,
        "lambda_sem": float(semantic_lambda),
        "weighted_semantic_loss": float(semantic_lambda) * sem_avg,
        "semantic_accuracy": sem_correct_weighted / float(sem_voxels),
        "num_semantic_voxels": sem_voxels,
        "cosine": cos_sum / float(fm_windows),
        "target_rms": target_rms_sum / float(fm_windows),
        "pred_rms": pred_rms_sum / float(fm_windows),
        "innovation_weight_alpha": float(innovation_weight_alpha),
        "innovation_weight_max": weight_max_sum / float(fm_windows),
        "innovation_focus_mean": focus_sum / float(fm_windows),
        "num_windows": fm_windows,
        "skipped_empty_batches": skipped_batches,
        "skipped_anchor_only_samples": skipped_samples,
    }


def _architecture(args, semantic_lambda, optimizer_summary, train_ds):
    return {
        "protocol": F7_PROTOCOL,
        "window_hw": list(f6.PRED_HW),
        "context_hw": list(f6.CONTEXT_HW),
        "topk": 2,
        "sample_steps": int(args.sample_steps),
        "source_noise_std": float(args.source_noise_std),
        "prior_channels": 16,
        "context_channels": 16,
        "flow": "strong_w2det_anchor_to_encoded_occ_repair_endpoint",
        "history": "full_history_latent",
        "base_loss": "soft_innovation_energy_weighted_full_window_flow_mse",
        "innovation_weight": {
            "alpha": float(args.innovation_weight_alpha),
            "energy": "channel_rms_of_rescaled_Zrepair_minus_Zanchor",
            "focus": "energy_over_energy_plus_per_sample_mean_energy",
            "normalization": "unit_mean_per_sample",
            "hard_mask": False,
        },
        "semantic_loss": "9way_dynamic_repair_ce_on_union_gt_and_anchor_dynamic_voxels",
        "semantic_lambda": float(semantic_lambda),
        "semantic_grad_ratio": float(args.semantic_grad_ratio),
        "semantic_lambda_mode": (
            "explicit" if args.semantic_lambda is not None else "first_batch_gradient_calibration"
        ),
        "optimizer_groups": optimizer_summary,
        "repair_endpoint_contract": f6.REPAIR_CONTRACT,
        "checkpoint_selection": "decoder_aware_weighted_validation_objective",
        "train_windows": len(train_ds),
        "train_unique_scenes": int(train_ds.metadata.get("num_unique_scenes", len(train_ds.metadata.get("scene_names", [])))),
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
    train_sem,
    val_sem,
    args,
    upstream_ckpt,
    reuse,
    optimizer_summary,
    skipped_train_batches,
    semantic_lambda,
    calibration,
):
    return {
        "state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "step": int(step),
        "best_val_objective": float(best_val),
        "semantic_lambda": float(semantic_lambda),
        "semantic_lambda_calibration": calibration,
        "training_history": list(history),
        "train_metadata": train_ds.metadata,
        "val_metadata": val_ds.metadata,
        "train_semantic_metadata": train_sem.metadata,
        "val_semantic_metadata": val_sem.metadata,
        "architecture": _architecture(args, semantic_lambda, optimizer_summary, train_ds),
        "upstream_checkpoint": str(Path(upstream_ckpt).resolve()),
        "upstream_checkpoint_sha256": file_sha256(upstream_ckpt),
        "upstream_reuse": reuse,
        "optimizer_group_summary": optimizer_summary,
        "skipped_empty_train_batches": int(skipped_train_batches),
        "args": vars(args),
    }


def _validate_resume(ck, args, train_ds, val_ds, train_sem, val_sem):
    arch = ck.get("architecture", {})
    if arch.get("protocol") != F7_PROTOCOL:
        raise RuntimeError("resume checkpoint is not P0-F7")
    if not math.isclose(
        float(arch.get("innovation_weight", {}).get("alpha", -1.0)),
        float(args.innovation_weight_alpha),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("resume innovation-weight-alpha differs")
    if not math.isclose(
        float(arch.get("optimizer_groups", {}).get("backbone_lr_scale", -1.0)),
        float(args.backbone_lr_scale),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("resume backbone-lr-scale differs")
    if int(arch.get("sample_steps", -1)) != int(args.sample_steps):
        raise RuntimeError("resume sample-steps differs")
    if int(arch.get("train_windows", -1)) != len(train_ds):
        raise RuntimeError("resume training cache size differs")
    train_sem.validate_source_cache(train_ds.root)
    val_sem.validate_source_cache(val_ds.root)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-cache", required=True)
    p.add_argument("--val-cache", required=True)
    p.add_argument("--train-semantic-targets", required=True)
    p.add_argument("--val-semantic-targets", required=True)
    p.add_argument("--upstream-ckpt", required=True)
    p.add_argument("--vae-ckpt", default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5, help="LR for new/unloaded parameters")
    p.add_argument("--backbone-lr-scale", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--innovation-weight-alpha", type=float, default=4.0)
    p.add_argument("--min-train-windows", type=int, default=8000)
    p.add_argument("--val-every", type=int, default=400)
    p.add_argument("--sample-steps", type=int, default=10)
    p.add_argument("--source-noise-std", type=float, default=0.0)
    p.add_argument("--semantic-lambda", type=float, default=None)
    p.add_argument("--semantic-grad-ratio", type=float, default=0.5)
    p.add_argument("--semantic-lambda-min", type=float, default=1e-4)
    p.add_argument("--semantic-lambda-max", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=20260901)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--resume-from", default=None)
    a = p.parse_args()

    if min(a.steps, a.batch_size, a.val_every, a.min_train_windows) <= 0:
        raise ValueError("steps/batch-size/val-every/min-train-windows must be positive")
    if a.source_noise_std != 0.0:
        raise ValueError("P0-F7 freezes source_noise_std=0")
    if a.innovation_weight_alpha < 0:
        raise ValueError("innovation-weight-alpha must be non-negative")
    if not 0.0 <= a.backbone_lr_scale <= 1.0:
        raise ValueError("backbone-lr-scale must be in [0,1]")
    if a.semantic_lambda is not None and a.semantic_lambda <= 0:
        raise ValueError("explicit semantic lambda must be positive")
    if a.semantic_grad_ratio <= 0:
        raise ValueError("semantic-grad-ratio must be positive")

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    device = torch.device(a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu")

    train_ds = MSPWorldModelCacheDataset(a.train_cache)
    val_ds = MSPWorldModelCacheDataset(a.val_cache)
    f6._validate_cache_pair(train_ds, val_ds)
    _validate_f7_training_cache(train_ds, min_train_windows=int(a.min_train_windows))

    train_sem = SemanticTargetCache(a.train_semantic_targets)
    val_sem = SemanticTargetCache(a.val_semantic_targets)
    f6._validate_semantic_pair(train_sem, val_sem, train_ds, val_ds)
    vae_path = f6._resolve_vae_path(a, train_ds, val_ds)

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

    model = make_p0_f7_model(
        20,
        sample_steps=a.sample_steps,
        source_noise_std=a.source_noise_std,
    ).to(device)
    reuse = load_shape_safe(model.transition, a.upstream_ckpt, verbose=True)
    if "traj_encoder.0.weight" not in set(reuse.get("loaded_keys", ())):
        raise RuntimeError("use the official OccFM-Fut epoch=000196 checkpoint")
    reuse_fraction = float(reuse.get("loaded", 0)) / max(float(reuse.get("target_total", 1)), 1.0)
    if reuse_fraction < 0.90:
        raise RuntimeError(f"unexpectedly low upstream reuse fraction {reuse_fraction:.3f}")

    optimizer, optimizer_summary = _build_optimizer(
        model,
        reuse,
        lr=float(a.lr),
        backbone_lr_scale=float(a.backbone_lr_scale),
        weight_decay=float(a.weight_decay),
    )
    print("optimizer_groups", json.dumps(optimizer_summary))

    vae_model, _ = load_official_vae(UP, vae_path, device)
    vae = OccFMVAEAdapter(vae_model)
    if any(p.requires_grad for p in vae_model.parameters()):
        raise RuntimeError("P0-F7 VAE decoder must remain frozen")

    use_amp = bool(a.amp and device.type == "cuda")
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    history = []
    step = 0
    skipped_train_batches = 0
    semantic_lambda = float(a.semantic_lambda) if a.semantic_lambda is not None else None
    calibration = None

    if a.resume_from:
        ck = torch.load(a.resume_from, map_location="cpu", weights_only=False)
        _validate_resume(ck, a, train_ds, val_ds, train_sem, val_sem)
        model.load_state_dict(ck["state_dict"], strict=True)
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        step = int(ck.get("step", 0))
        if step < 0 or step > a.steps:
            raise RuntimeError(f"resume step {step} incompatible with target {a.steps}")
        best_val = float(ck.get("best_val_objective", float("inf")))
        history = list(ck.get("training_history", []))
        skipped_train_batches = int(ck.get("skipped_empty_train_batches", 0))
        semantic_lambda = float(ck["semantic_lambda"])
        calibration = ck.get("semantic_lambda_calibration")
        print(
            f"resumed P0-F7 from {a.resume_from} at step={step} "
            f"best_val={best_val:.6f} lambda_sem={semantic_lambda:.8g}"
        )

    iterator = iter(train_loader)
    model.train()
    while step < a.steps:
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
                innovation_weight_alpha=float(a.innovation_weight_alpha),
            )
            endpoint = f6.scatter_endpoint_to_full(info["predicted_endpoint"], prepared)
            sem_loss, sem_info = f6.semantic_loss_for_endpoint(
                endpoint,
                sample_ids=prepared["sample_ids"],
                semantic_cache=train_sem,
                vae=vae,
            )

        if int(sem_info["num_supervised_voxels"]) <= 0 and semantic_lambda is None:
            print("skip semantic-empty batch before lambda calibration")
            continue
        if not torch.isfinite(fm_loss) or not torch.isfinite(sem_loss):
            raise RuntimeError(
                f"non-finite P0-F7 loss at step {step}: fm={fm_loss} sem={sem_loss}"
            )

        if semantic_lambda is None:
            semantic_lambda, calibration = f6.calibrate_semantic_lambda(
                fm_loss,
                sem_loss,
                model,
                target_ratio=float(a.semantic_grad_ratio),
                lambda_min=float(a.semantic_lambda_min),
                lambda_max=float(a.semantic_lambda_max),
            )
            calibration.update({
                "step_before_update": int(step),
                "weighted_fm_loss": float(fm_loss.detach().item()),
                "unweighted_fm_loss": float(info["unweighted_fm_loss"]),
                "semantic_loss": float(sem_loss.detach().item()),
                "weighted_semantic_loss": float(semantic_lambda) * float(sem_loss.detach().item()),
                "num_semantic_voxels": int(sem_info["num_supervised_voxels"]),
                "innovation_weight_alpha": float(a.innovation_weight_alpha),
                "innovation_weight_max": float(info["innovation_weight_max"]),
            })
            print("semantic_lambda_calibration", json.dumps(calibration))

        total_loss = fm_loss + float(semantic_lambda) * sem_loss
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 5.0
        )
        optimizer.step()
        step += 1

        train_info = {
            "objective": float(total_loss.detach().item()),
            "weighted_fm_loss": float(fm_loss.detach().item()),
            "unweighted_fm_loss": float(info["unweighted_fm_loss"]),
            "semantic_loss": float(sem_loss.detach().item()),
            "lambda_sem": float(semantic_lambda),
            "weighted_semantic_loss": float(semantic_lambda) * float(sem_loss.detach().item()),
            "semantic_accuracy": sem_info["accuracy"],
            "num_semantic_voxels": int(sem_info["num_supervised_voxels"]),
            "cosine": info["cosine"],
            "target_rms": info["target_rms"],
            "pred_rms": info["pred_rms"],
            "innovation_weight_alpha": float(a.innovation_weight_alpha),
            "innovation_weight_max": float(info["innovation_weight_max"]),
            "innovation_focus_mean": float(info["innovation_focus_mean"]),
            "grad_norm_before_clip": float(torch.as_tensor(grad_norm).cpu()),
        }
        if step == 1 or step % 20 == 0:
            print(
                f"step={step} total={train_info['objective']:.6f} "
                f"wfm={train_info['weighted_fm_loss']:.6f} "
                f"ufm={train_info['unweighted_fm_loss']:.6f} "
                f"sem={train_info['semantic_loss']:.6f} "
                f"lambda={semantic_lambda:.6g} "
                f"sem_acc={train_info['semantic_accuracy']:.4f} "
                f"iw_max={train_info['innovation_weight_max']:.3f} "
                f"cos={train_info['cosine']:.4f}"
            )

        if step % a.val_every == 0 or step == a.steps:
            val = validate(
                model,
                val_loader,
                device,
                val_sem,
                vae,
                semantic_lambda,
                use_amp,
                float(a.innovation_weight_alpha),
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
                train_sem=train_sem,
                val_sem=val_sem,
                args=a,
                upstream_ckpt=a.upstream_ckpt,
                reuse=reuse,
                optimizer_summary=optimizer_summary,
                skipped_train_batches=skipped_train_batches,
                semantic_lambda=semantic_lambda,
                calibration=calibration,
            )
            torch.save(_payload(model_state=state, **payload_args), out / "latest.pt")
            if float(val["objective"]) < best_val:
                best_val = float(val["objective"])
                torch.save(_payload(model_state=state, **payload_args), out / "best.pt")

    if semantic_lambda is None:
        raise RuntimeError("no semantic-valid P0-F7 training batch was observed")

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
            train_sem=train_sem,
            val_sem=val_sem,
            args=a,
            upstream_ckpt=a.upstream_ckpt,
            reuse=reuse,
            optimizer_summary=optimizer_summary,
            skipped_train_batches=skipped_train_batches,
            semantic_lambda=semantic_lambda,
            calibration=calibration,
        ),
        out / "last.pt",
    )
    report = {
        "protocol": F7_PROTOCOL,
        "steps": int(step),
        "best_val_objective": float(best_val),
        "semantic_lambda": float(semantic_lambda),
        "innovation_weight_alpha": float(a.innovation_weight_alpha),
        "optimizer_groups": optimizer_summary,
        "train_windows": len(train_ds),
        "train_unique_scenes": int(train_ds.metadata.get("num_unique_scenes", len(train_ds.metadata.get("scene_names", [])))),
        "history": history,
    }
    (out / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
