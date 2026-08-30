#!/usr/bin/env python3
"""Train P0-F6 decoder-aware sparse innovation World Model.

Everything from P0-F5 is frozen except the objective. The same FM forward pass
returns a differentiable endpoint estimate; selected local endpoints are
scattered into the complete Strong-W2Det anchor latent, passed through the
*frozen* official VAE decoder, and supervised at sparse dynamic-repair voxels.

Loss:
    L = L_FM + lambda_sem * L_sem

If --semantic-lambda is omitted, lambda_sem is calibrated once on the first
semantic-valid batch so the weighted semantic gradient norm is approximately
--semantic-grad-ratio times the FM gradient norm. It is then frozen forever and
stored in every checkpoint.
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
from real_motion.context import crop_prediction_and_context
from real_motion.models.p0_f4 import make_p0_f4_model
from real_motion.msp_wm_cache import (
    MSP_WM_CACHE_VERSION_V3,
    MSPWorldModelCacheDataset,
    collate_msp_wm,
)
from real_motion.occfm_io import OccFMVAEAdapter, file_sha256, load_official_vae
from real_motion.semantic_repair import (
    DYNAMIC_IDS,
    SemanticTargetCache,
    sparse_dynamic_semantic_loss,
    sparse_union_targets,
)
from real_motion.windows import WindowPlan, crop_windows, scatter_windows

PRED_HW = (20, 20)
CONTEXT_HW = (40, 40)
FULL_HW = (50, 50)
LOSS_CONTRACT = "strong_anchor_to_occ_repair_endpoint_local_flow_full_history_context_no_auxiliary_losses"
REPAIR_CONTRACT = "strong_anchor_outside_support_gt_dynamic_inside_support_v1"
F6_PROTOCOL = "p0_f6_decoder_aware_sparse_innovation_v1"
SEMANTIC_PROTOCOL = "p0_f6_decoder_aware_sparse_dynamic_semantics_v1"


def _validate_cache_pair(train_ds, val_ds):
    if train_ds.version != MSP_WM_CACHE_VERSION_V3 or val_ds.version != MSP_WM_CACHE_VERSION_V3:
        raise RuntimeError("P0-F6 requires P0-F5/v3 repair-endpoint caches")
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
            raise RuntimeError(f"{name} cache does not use Strong W2Det")
        if meta.get("history_contract") != "full_native_occ_history_6f":
            raise RuntimeError(f"{name} cache does not use full history")
        if meta.get("repair_endpoint_contract") != REPAIR_CONTRACT:
            raise RuntimeError(f"{name} cache repair endpoint contract mismatch")
        if meta.get("loss_contract") != LOSS_CONTRACT:
            raise RuntimeError(f"{name} cache base FM contract mismatch")
        if meta.get("target") != "occupancy_sparse_repair_endpoint_vae_latent":
            raise RuntimeError(f"{name} cache is not the encoded occupancy repair endpoint")
        if meta.get("latent_loss_mask") != "none":
            raise RuntimeError(f"{name} cache unexpectedly requests a latent loss mask")
    overlap = sorted(set(tm.get("scene_names", [])) & set(vm.get("scene_names", [])))
    if overlap:
        raise RuntimeError(f"train/val scene leakage ({len(overlap)}), e.g. {overlap[:3]}")
    for key in ("msp_checkpoint_sha256", "vae_checkpoint_sha256"):
        if tm.get(key) != vm.get(key):
            raise RuntimeError(f"train/val cache mismatch for {key}")
    if float(tm.get("write_budget_ratio", -1)) != float(vm.get("write_budget_ratio", -2)):
        raise RuntimeError("train/val write-budget ratios differ")


def _validate_semantic_pair(train_sem, val_sem, train_cache, val_cache):
    train_sem.validate_source_cache(train_cache.root)
    val_sem.validate_source_cache(val_cache.root)
    for name, sem in (("train", train_sem), ("val", val_sem)):
        m = sem.metadata
        if m.get("protocol") != SEMANTIC_PROTOCOL:
            raise RuntimeError(f"{name} semantic sidecar protocol mismatch")
        if tuple(int(x) for x in m.get("dynamic_class_ids", [])) != tuple(DYNAMIC_IDS):
            raise RuntimeError(f"{name} semantic sidecar dynamic-class mapping mismatch")
        if m.get("vae_checkpoint_sha256") != train_cache.metadata.get("vae_checkpoint_sha256"):
            raise RuntimeError(f"{name} semantic sidecar VAE mismatch")
    if set(train_sem.records) != {str(e["sample_id"]) for e in train_cache.entries}:
        raise RuntimeError("train semantic sidecar sample set differs from train WM cache")
    if set(val_sem.records) != {str(e["sample_id"]) for e in val_cache.entries}:
        raise RuntimeError("val semantic sidecar sample set differs from val WM cache")


def prepare_batch(batch, device):
    """Crop valid Top-2 windows while retaining the full anchor for semantic decode."""
    origins_cpu = batch["window_origins"].long()
    valid_cpu = batch["window_valid"].bool()
    B, K = valid_cpu.shape
    if K != 2:
        raise RuntimeError(f"P0-F6 expects K=2, got {K}")
    plan_cpu = WindowPlan(origins_cpu, valid_cpu, PRED_HW, FULL_HW)
    if not bool(plan_cpu.valid.any()):
        return None

    moved = {
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
        moved["full_history_latent"], plan, context_hw=CONTEXT_HW
    )
    anchor_w = crop_windows(moved["anchor_future_latent"], plan)
    target_w = crop_windows(moved["repair_target_latent"], plan)
    effective = plan.valid.reshape(-1)

    def flat(x):
        return x.reshape(B * K, *x.shape[2:])[effective]

    hist_local, hist_context, anchor_w, target_w = map(
        flat, (hist_local, hist_context, anchor_w, target_w)
    )
    origins = plan.origins.reshape(B * K, 2)[effective]
    traj = moved["trajectory"]
    if tuple(traj.shape[1:]) != (12, 2):
        raise RuntimeError(f"trajectory batch must be [B,12,2], got {tuple(traj.shape)}")
    traj = traj[:, None].expand(B, K, 12, 2).reshape(B * K, 12, 2)[effective]
    return {
        "history": hist_local,
        "context": hist_context,
        "target": target_w,
        "anchor": anchor_w,
        "trajectory": traj,
        "origins": origins,
        "plan": plan,
        "effective": effective,
        "batch_size": B,
        "topk": K,
        "anchor_full": moved["anchor_future_latent"],
        "sample_ids": list(batch["sample_id"]),
    }


def scatter_endpoint_to_full(endpoint_windows: torch.Tensor, prepared: dict) -> torch.Tensor:
    """Use exactly the same overlap-average latent fusion as final inference."""
    B = int(prepared["batch_size"])
    K = int(prepared["topk"])
    effective = prepared["effective"]
    expected = int(effective.sum().item())
    if endpoint_windows.shape[0] != expected:
        raise ValueError(
            f"endpoint window count {endpoint_windows.shape[0]} != valid plan slots {expected}"
        )
    padded = endpoint_windows.new_zeros((B * K, *endpoint_windows.shape[1:]))
    dst = torch.nonzero(effective, as_tuple=False).flatten()
    padded = padded.index_copy(0, dst, endpoint_windows)
    windows = padded.reshape(B, K, *endpoint_windows.shape[1:])
    return scatter_windows(
        windows,
        prepared["plan"],
        base=prepared["anchor_full"],
    )


def semantic_loss_for_endpoint(
    endpoint_full: torch.Tensor,
    *,
    sample_ids,
    semantic_cache: SemanticTargetCache,
    vae: OccFMVAEAdapter,
) -> tuple[torch.Tensor, dict]:
    records = semantic_cache.get_batch(sample_ids)
    indices = []
    targets = []
    for rec in records:
        idx, tgt = sparse_union_targets(rec, endpoint_full.device)
        indices.append(idx)
        targets.append(tgt)
    sparse_logits = vae.decode_logits_at_flat_indices(endpoint_full, indices)
    return sparse_dynamic_semantic_loss(sparse_logits, targets)


def _grad_l2_norm(loss: torch.Tensor, params) -> float:
    grads = torch.autograd.grad(
        loss,
        params,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    total = None
    for g in grads:
        if g is None:
            continue
        term = g.detach().float().square().sum()
        total = term if total is None else total + term
    if total is None:
        return 0.0
    return float(total.sqrt().cpu())


def calibrate_semantic_lambda(
    fm_loss: torch.Tensor,
    sem_loss: torch.Tensor,
    model,
    *,
    target_ratio: float,
    lambda_min: float,
    lambda_max: float,
) -> tuple[float, dict]:
    if target_ratio <= 0:
        raise ValueError("semantic grad ratio must be positive")
    if lambda_min <= 0 or lambda_max < lambda_min:
        raise ValueError("invalid semantic lambda clamp")
    params = [p for p in model.parameters() if p.requires_grad]
    g_fm = _grad_l2_norm(fm_loss, params)
    g_sem = _grad_l2_norm(sem_loss, params)
    if not math.isfinite(g_fm) or not math.isfinite(g_sem) or g_fm <= 0 or g_sem <= 0:
        raise RuntimeError(
            f"cannot calibrate semantic lambda from gradient norms fm={g_fm} sem={g_sem}"
        )
    raw = float(target_ratio) * g_fm / g_sem
    lam = min(max(raw, float(lambda_min)), float(lambda_max))
    return lam, {
        "fm_grad_norm": g_fm,
        "semantic_grad_norm_unweighted": g_sem,
        "target_semantic_to_fm_grad_ratio": float(target_ratio),
        "raw_lambda": raw,
        "lambda": lam,
        "weighted_semantic_grad_norm": lam * g_sem,
        "realized_semantic_to_fm_grad_ratio": lam * g_sem / g_fm,
        "clamped": bool(lam != raw),
    }


@torch.no_grad()
def validate(model, loader, device, semantic_cache, vae, semantic_lambda, use_amp):
    model.eval()
    fm_sum = 0.0
    fm_windows = 0
    sem_sum = 0.0
    sem_voxels = 0
    cos_sum = 0.0
    target_rms_sum = 0.0
    pred_rms_sum = 0.0
    sem_correct_weighted = 0.0
    skipped_batches = 0
    skipped_samples = 0
    for batch in loader:
        prepared = prepare_batch(batch, device)
        if prepared is None:
            skipped_batches += 1
            skipped_samples += len(batch.get("sample_id", []))
            continue
        with torch.enable_grad():
            # Validation needs input-latent gradients disabled only after the
            # semantic forward. No optimizer graph is retained beyond this loop.
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
                endpoint = scatter_endpoint_to_full(info["predicted_endpoint"], prepared)
                sem_loss, sem_info = semantic_loss_for_endpoint(
                    endpoint,
                    sample_ids=prepared["sample_ids"],
                    semantic_cache=semantic_cache,
                    vae=vae,
                )
        nwin = int(prepared["history"].shape[0])
        nvox = int(sem_info["num_supervised_voxels"])
        fm_sum += float(fm_loss.detach().item()) * nwin
        fm_windows += nwin
        cos_sum += float(info["cosine"]) * nwin
        target_rms_sum += float(info["target_rms"]) * nwin
        pred_rms_sum += float(info["pred_rms"]) * nwin
        if nvox > 0:
            sem_sum += float(sem_loss.detach().item()) * nvox
            sem_voxels += nvox
            sem_correct_weighted += float(sem_info["accuracy"]) * nvox
        del endpoint, sem_loss, fm_loss, info
    model.train()
    if fm_windows <= 0:
        raise RuntimeError("validation contains no valid routed Sparse-WM windows")
    fm_avg = fm_sum / float(fm_windows)
    if sem_voxels <= 0:
        raise RuntimeError("validation contains no P0-F6 semantic supervision voxels")
    sem_avg = sem_sum / float(sem_voxels)
    total = fm_avg + float(semantic_lambda) * sem_avg
    return {
        "objective": total,
        "fm_loss": fm_avg,
        "semantic_loss": sem_avg,
        "lambda_sem": float(semantic_lambda),
        "weighted_semantic_loss": float(semantic_lambda) * sem_avg,
        "semantic_accuracy": sem_correct_weighted / float(sem_voxels),
        "num_semantic_voxels": sem_voxels,
        "cosine": cos_sum / float(fm_windows),
        "target_rms": target_rms_sum / float(fm_windows),
        "pred_rms": pred_rms_sum / float(fm_windows),
        "num_windows": fm_windows,
        "skipped_empty_batches": skipped_batches,
        "skipped_anchor_only_samples": skipped_samples,
    }


def _resolve_vae_path(args, train_ds, val_ds) -> Path:
    value = args.vae_ckpt or train_ds.metadata.get("vae_checkpoint")
    if not value:
        raise RuntimeError("VAE checkpoint path is required")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    sha = file_sha256(path)
    for name, ds in (("train", train_ds), ("val", val_ds)):
        expected = ds.metadata.get("vae_checkpoint_sha256")
        if expected and expected != sha:
            raise RuntimeError(f"{name} cache was built with a different VAE checkpoint")
    return path


def _architecture(args, semantic_lambda):
    return {
        "protocol": F6_PROTOCOL,
        "window_hw": list(PRED_HW),
        "context_hw": list(CONTEXT_HW),
        "topk": 2,
        "sample_steps": int(args.sample_steps),
        "source_noise_std": float(args.source_noise_std),
        "prior_channels": 16,
        "context_channels": 16,
        "flow": "strong_w2det_anchor_to_encoded_occ_repair_endpoint",
        "history": "full_history_latent",
        "base_loss": "plain_full_window_flow_mse_no_latent_mask",
        "semantic_loss": "9way_dynamic_repair_ce_on_union_gt_and_anchor_dynamic_voxels",
        "semantic_lambda": float(semantic_lambda),
        "semantic_grad_ratio": float(args.semantic_grad_ratio),
        "semantic_lambda_mode": (
            "explicit" if args.semantic_lambda is not None else "first_batch_gradient_calibration"
        ),
        "repair_endpoint_contract": REPAIR_CONTRACT,
        "checkpoint_selection": "decoder_aware_validation_objective",
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
        "architecture": _architecture(args, semantic_lambda),
        "upstream_checkpoint": str(Path(upstream_ckpt).resolve()),
        "upstream_checkpoint_sha256": file_sha256(upstream_ckpt),
        "upstream_reuse": reuse,
        "skipped_empty_train_batches": int(skipped_train_batches),
        "args": vars(args),
    }


def _validate_resume_checkpoint(ck, args, train_ds, val_ds, train_sem, val_sem):
    arch = ck.get("architecture", {})
    if arch.get("protocol") != F6_PROTOCOL:
        raise RuntimeError("resume checkpoint is not P0-F6")
    if list(arch.get("window_hw", [])) != list(PRED_HW):
        raise RuntimeError("resume prediction window differs")
    if list(arch.get("context_hw", [])) != list(CONTEXT_HW):
        raise RuntimeError("resume context window differs")
    if arch.get("repair_endpoint_contract") != REPAIR_CONTRACT:
        raise RuntimeError("resume repair endpoint contract differs")
    if int(arch.get("sample_steps", args.sample_steps)) != int(args.sample_steps):
        raise RuntimeError("resume sample_steps differs")
    for ckmeta, dsmeta, prefix in (
        (ck.get("train_metadata", {}), train_ds.metadata, "train"),
        (ck.get("val_metadata", {}), val_ds.metadata, "val"),
    ):
        for key in ("msp_checkpoint_sha256", "vae_checkpoint_sha256", "anchor_contract"):
            if ckmeta.get(key) and ckmeta.get(key) != dsmeta.get(key):
                raise RuntimeError(f"resume {prefix} metadata mismatch for {key}")
    for ckmeta, sem, prefix in (
        (ck.get("train_semantic_metadata", {}), train_sem, "train-semantic"),
        (ck.get("val_semantic_metadata", {}), val_sem, "val-semantic"),
    ):
        if ckmeta.get("source_wm_cache_index_sha256") != sem.metadata.get(
            "source_wm_cache_index_sha256"
        ):
            raise RuntimeError(f"resume {prefix} sidecar differs")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-cache", required=True)
    p.add_argument("--val-cache", required=True)
    p.add_argument("--train-semantic-targets", required=True)
    p.add_argument("--val-semantic-targets", required=True)
    p.add_argument("--upstream-ckpt", required=True)
    p.add_argument("--vae-ckpt", default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--val-every", type=int, default=200)
    p.add_argument("--sample-steps", type=int, default=10)
    p.add_argument("--source-noise-std", type=float, default=0.0)
    p.add_argument("--semantic-lambda", type=float, default=None)
    p.add_argument("--semantic-grad-ratio", type=float, default=0.5)
    p.add_argument("--semantic-lambda-min", type=float, default=1e-4)
    p.add_argument("--semantic-lambda-max", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=20260830)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--resume-from", default=None)
    a = p.parse_args()
    if min(a.steps, a.batch_size, a.val_every) <= 0:
        raise ValueError("steps/batch-size/val-every must be positive")
    if a.source_noise_std != 0.0:
        raise ValueError("P0-F6 freezes source_noise_std=0")
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
    _validate_cache_pair(train_ds, val_ds)
    train_sem = SemanticTargetCache(a.train_semantic_targets)
    val_sem = SemanticTargetCache(a.val_semantic_targets)
    _validate_semantic_pair(train_sem, val_sem, train_ds, val_ds)
    vae_path = _resolve_vae_path(a, train_ds, val_ds)

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
        20,
        sample_steps=a.sample_steps,
        source_noise_std=a.source_noise_std,
    ).to(device)
    reuse = load_shape_safe(model.transition, a.upstream_ckpt, verbose=True)
    if "traj_encoder.0.weight" not in set(reuse.get("loaded_keys", ())):
        raise RuntimeError("use the official OccFM-Fut epoch=000196 checkpoint")
    reuse_fraction = float(reuse.get("loaded", 0)) / max(
        float(reuse.get("target_total", 1)), 1.0
    )
    if reuse_fraction < 0.90:
        raise RuntimeError(f"unexpectedly low upstream reuse fraction {reuse_fraction:.3f}")

    vae_model, _ = load_official_vae(UP, vae_path, device)
    vae = OccFMVAEAdapter(vae_model)
    if any(p.requires_grad for p in vae_model.parameters()):
        raise RuntimeError("P0-F6 VAE decoder must remain frozen")

    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    use_amp = bool(a.amp and device.type == "cuda")
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    best_state = None
    history = []
    step = 0
    skipped_train_batches = 0
    semantic_lambda = float(a.semantic_lambda) if a.semantic_lambda is not None else None
    calibration = None

    if a.resume_from:
        ck = torch.load(a.resume_from, map_location="cpu", weights_only=False)
        _validate_resume_checkpoint(ck, a, train_ds, val_ds, train_sem, val_sem)
        model.load_state_dict(ck["state_dict"], strict=True)
        step = int(ck.get("step", 0))
        if step < 0 or step > a.steps:
            raise RuntimeError(f"resume step {step} incompatible with target {a.steps}")
        best_val = float(ck.get("best_val_objective", float("inf")))
        history = list(ck.get("training_history", []))
        skipped_train_batches = int(ck.get("skipped_empty_train_batches", 0))
        ck_lambda = ck.get("semantic_lambda")
        if ck_lambda is None:
            raise RuntimeError("P0-F6 resume checkpoint lacks frozen semantic lambda")
        if semantic_lambda is not None and not math.isclose(
            semantic_lambda, float(ck_lambda), rel_tol=0.0, abs_tol=1e-12
        ):
            raise RuntimeError("explicit semantic lambda differs from resume checkpoint")
        semantic_lambda = float(ck_lambda)
        calibration = ck.get("semantic_lambda_calibration")
        if np.isfinite(best_val):
            best_state = copy.deepcopy(
                {k: v.detach().cpu() for k, v in model.state_dict().items()}
            )
        if ck.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(ck["optimizer_state_dict"])
        print(
            f"resumed P0-F6 from {a.resume_from} at step={step} "
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
        prepared = prepare_batch(batch, device)
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
            endpoint = scatter_endpoint_to_full(info["predicted_endpoint"], prepared)
            sem_loss, sem_info = semantic_loss_for_endpoint(
                endpoint,
                sample_ids=prepared["sample_ids"],
                semantic_cache=train_sem,
                vae=vae,
            )

        if int(sem_info["num_supervised_voxels"]) <= 0:
            if semantic_lambda is None:
                print("skip semantic-empty batch before lambda calibration")
                continue
        if not torch.isfinite(fm_loss) or not torch.isfinite(sem_loss):
            raise RuntimeError(
                f"non-finite P0-F6 loss at step {step}: fm={fm_loss} sem={sem_loss}"
            )

        if semantic_lambda is None:
            semantic_lambda, calibration = calibrate_semantic_lambda(
                fm_loss,
                sem_loss,
                model,
                target_ratio=float(a.semantic_grad_ratio),
                lambda_min=float(a.semantic_lambda_min),
                lambda_max=float(a.semantic_lambda_max),
            )
            calibration.update({
                "step_before_update": int(step),
                "fm_loss": float(fm_loss.detach().item()),
                "semantic_loss": float(sem_loss.detach().item()),
                "weighted_semantic_loss": float(semantic_lambda) * float(sem_loss.detach().item()),
                "num_semantic_voxels": int(sem_info["num_supervised_voxels"]),
            })
            print("semantic_lambda_calibration", json.dumps(calibration))

        total_loss = fm_loss + float(semantic_lambda) * sem_loss
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        step += 1

        train_info = {
            "objective": float(total_loss.detach().item()),
            "fm_loss": float(fm_loss.detach().item()),
            "semantic_loss": float(sem_loss.detach().item()),
            "lambda_sem": float(semantic_lambda),
            "weighted_semantic_loss": float(semantic_lambda) * float(sem_loss.detach().item()),
            "semantic_accuracy": sem_info["accuracy"],
            "num_semantic_voxels": int(sem_info["num_supervised_voxels"]),
            "num_gt_dynamic_voxels": int(sem_info["num_gt_dynamic_voxels"]),
            "cosine": info["cosine"],
            "target_rms": info["target_rms"],
            "pred_rms": info["pred_rms"],
            "grad_norm_before_clip": float(torch.as_tensor(grad_norm).cpu()),
        }
        if step == 1 or step % 20 == 0:
            print(
                f"step={step} total={train_info['objective']:.6f} "
                f"fm={train_info['fm_loss']:.6f} sem={train_info['semantic_loss']:.6f} "
                f"lambda={semantic_lambda:.6g} wsem={train_info['weighted_semantic_loss']:.6f} "
                f"sem_acc={train_info['semantic_accuracy']:.4f} "
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
            )
            row = {"step": step, "train": train_info, "val": val}
            history.append(row)
            print("validation", json.dumps(row))
            state = copy.deepcopy(
                {k: v.detach().cpu() for k, v in model.state_dict().items()}
            )
            payload_args = dict(
                optimizer=optimizer,
                step=step,
                best_val=min(best_val, float(val["objective"])),
                history=history,
                train_ds=train_ds,
                val_ds=val_ds,
                train_sem=train_sem,
                val_sem=val_sem,
                args=a,
                upstream_ckpt=a.upstream_ckpt,
                reuse=reuse,
                skipped_train_batches=skipped_train_batches,
                semantic_lambda=semantic_lambda,
                calibration=calibration,
            )
            torch.save(
                _payload(model_state=state, **payload_args),
                out / "latest.pt",
            )
            if float(val["objective"]) < best_val:
                best_val = float(val["objective"])
                best_state = state
                torch.save(
                    _payload(model_state=best_state, **payload_args),
                    out / "best.pt",
                )

    if best_state is None:
        raise RuntimeError("no P0-F6 validation checkpoint was produced")
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
            train_sem=train_sem,
            val_sem=val_sem,
            args=a,
            upstream_ckpt=a.upstream_ckpt,
            reuse=reuse,
            skipped_train_batches=skipped_train_batches,
            semantic_lambda=semantic_lambda,
            calibration=calibration,
        ),
        out / "last.pt",
    )
    (out / "training_report.json").write_text(
        json.dumps({
            "protocol": F6_PROTOCOL,
            "best_val_objective": best_val,
            "semantic_lambda": semantic_lambda,
            "semantic_lambda_calibration": calibration,
            "history": history,
            "upstream_reuse_fraction": reuse_fraction,
            "skipped_empty_train_batches": skipped_train_batches,
            "resumed_from": a.resume_from,
            "decision": "Evaluate Strong W2Det vs trained P0-F6 vs same-support GT repair oracle.",
        }, indent=2),
        encoding="utf-8",
    )
    print("saved", out / "best.pt")


if __name__ == "__main__":
    main()
