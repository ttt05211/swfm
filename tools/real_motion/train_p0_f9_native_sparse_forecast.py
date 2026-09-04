#!/usr/bin/env python3
"""Stage-1 training for P0-F9 physics-conditioned native sparse forecasting.

P0-F9 trains the World Model on the released OccFM task (Gaussian noise ->
absolute future), while Strong-W2Det/KTA is a causal physics condition and exact
deployment fallback outside frozen MSP write support.

Audited v2 contracts:
- deterministic posterior *samples* are cached to match the released OccFM-Fut
  latent distribution (not VAE means);
- the inherited backbone keeps official HIST_LAST=4, while the new context
  branch still sees all six history frames;
- semantic supervision is equal-weighted over all six future horizons;
- resume/evaluation provenance is fail-closed rather than best-effort.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
UP = ROOT / "upstream_occfm"
sys.path[:0] = [str(ROOT), str(UP)]

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from real_motion.checkpoint import load_shape_safe, require_checkpoint_reuse
from real_motion.context import crop_prediction_and_context
from real_motion.edit_repair import DYNAMIC_IDS, EditTargetCache
from real_motion.model_ema import ModelEMA
from real_motion.models.p0_f9 import P0_F9_PROTOCOL, make_p0_f9_model
from real_motion.msp_wm_cache import (
    MSP_WM_CACHE_VERSION_V2,
    MSPWorldModelCacheDataset,
    collate_msp_wm,
)
from real_motion.native_forecast import (
    absolute_future_semantic_loss,
    class_weights_from_edit_cache,
    semantic_targets_for_sample,
)
from real_motion.occfm_io import OccFMVAEAdapter, file_sha256, load_official_vae
from real_motion.windows import WindowPlan, crop_windows, scatter_windows
from tools.real_motion.build_p0_f9_cache_fast import P0_F9_CACHE_PROTOCOL

PRED_HW = (20, 20)
CONTEXT_HW = (40, 40)
FULL_HW = (50, 50)
HIST_LAST = 4
OPTIMIZER_PROTOCOL = "p0_f9_loaded_vs_new_differential_lr_v2"
P0_F8_SEMANTIC_PROTOCOL = "p0_f8_anchor_relative_edit_targets_v1"
P0_F8_TARGET_CONTRACT = "exact_strong_anchor_relative_keep_clear_write_inside_causal_msp_support"


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_cache_pair(train_ds, val_ds, *, min_train_windows: int) -> None:
    if train_ds.version != MSP_WM_CACHE_VERSION_V2 or val_ds.version != MSP_WM_CACHE_VERSION_V2:
        raise RuntimeError("P0-F9 requires absolute-future v2 tensor caches")
    if len(train_ds) < int(min_train_windows):
        raise RuntimeError(f"P0-F9 requires >= {min_train_windows} train windows; got {len(train_ds)}")
    tm, vm = train_ds.metadata, val_ds.metadata
    for name, meta in (("train", tm), ("val", vm)):
        if meta.get("protocol") != P0_F9_CACHE_PROTOCOL:
            raise RuntimeError(f"{name} cache is not the audited P0-F9 native-future cache")
        if meta.get("target") != "absolute_gt_future_vae_latent":
            raise RuntimeError(f"{name} cache target is not absolute GT future")
        if meta.get("flow_source") != "gaussian_noise_not_anchor":
            raise RuntimeError(f"{name} cache flow-source contract mismatch")
        if meta.get("history_contract") != "full_native_occ_history_6f":
            raise RuntimeError(f"{name} cache history contract mismatch")
        if int(meta.get("native_backbone_hist_last", -1)) != HIST_LAST:
            raise RuntimeError(f"{name} cache does not record native HIST_LAST={HIST_LAST}")
        if meta.get("anchor_contract") != "strong_w2det_occ_only_v1":
            raise RuntimeError(f"{name} cache physics-anchor contract mismatch")
        if int(meta.get("topk", -1)) != 2 or list(meta.get("window_hw", [])) != [20, 20]:
            raise RuntimeError(f"{name} cache routing contract mismatch")
        if list(meta.get("context_hw", [])) != [40, 40]:
            raise RuntimeError(f"{name} cache context contract mismatch")
        if meta.get("vae_mode") != "sample":
            raise RuntimeError(
                f"{name} cache must use posterior samples to match released OccFM-Fut training latents"
            )
        if not meta.get("vae_sample_seed_contract"):
            raise RuntimeError(f"{name} cache lacks deterministic VAE sampling provenance")
    if tm.get("source_msp_mode") != "train":
        raise RuntimeError("P0-F9 train cache must come from the frozen MSP train split")
    if vm.get("source_msp_mode") != "val":
        raise RuntimeError("P0-F9 val cache must come from the frozen MSP val split")
    overlap = sorted(set(tm.get("scene_names", [])) & set(vm.get("scene_names", [])))
    if overlap:
        raise RuntimeError(f"train/val scene leakage ({len(overlap)}), e.g. {overlap[:3]}")
    for key in ("msp_checkpoint_sha256", "vae_checkpoint_sha256"):
        if tm.get(key) != vm.get(key):
            raise RuntimeError(f"train/val cache mismatch for {key}")
    if float(tm.get("write_budget_ratio", -1)) != float(vm.get("write_budget_ratio", -2)):
        raise RuntimeError("train/val write-budget ratios differ")


def _validate_semantic_sidecar(edit: EditTargetCache, ds, name: str) -> None:
    meta = edit.metadata
    if meta.get("protocol") != P0_F8_SEMANTIC_PROTOCOL:
        raise RuntimeError(f"{name} semantic sidecar protocol mismatch")
    if meta.get("target_contract") != P0_F8_TARGET_CONTRACT:
        raise RuntimeError(f"{name} semantic sidecar target contract mismatch")
    expected = ds.metadata.get("source_v3_cache_index_sha256")
    actual = meta.get("source_wm_cache_index_sha256")
    if not expected or actual != expected:
        raise RuntimeError(f"{name} semantic sidecar was not built from the exact routed source cache")
    ids = {str(e["sample_id"]) for e in ds.entries}
    if set(edit.records) != ids:
        raise RuntimeError(f"{name} semantic sidecar sample set differs from P0-F9 cache")
    dyn = tuple(int(x) for x in meta.get("dynamic_class_ids", []))
    if dyn != tuple(DYNAMIC_IDS):
        raise RuntimeError(f"{name} semantic sidecar dynamic-class mapping differs")


def _scene_balanced_sampler(ds, *, seed: int):
    scenes = [str(e["scene_name"]) for e in ds.entries]
    counts = Counter(scenes)
    weights = torch.tensor([1.0 / counts[s] for s in scenes], dtype=torch.double)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    return WeightedRandomSampler(weights, num_samples=len(ds), replacement=True, generator=gen)


def prepare_batch(batch, device):
    origins_cpu = batch["window_origins"].long()
    valid_cpu = batch["window_valid"].bool()
    B, K = valid_cpu.shape
    if K != 2:
        raise RuntimeError(f"P0-F9 expects frozen Top-2, got K={K}")
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
    physics = crop_windows(moved["anchor_future_latent"], plan)
    target = crop_windows(moved["gt_future_latent"], plan)
    effective = plan.valid.reshape(-1)

    def flat(x):
        return x.reshape(B * K, *x.shape[2:])[effective]

    hist_local, hist_context, physics, target = map(
        flat, (hist_local, hist_context, physics, target)
    )
    origins = plan.origins.reshape(B * K, 2)[effective]
    traj = moved["trajectory"]
    if tuple(traj.shape[1:]) != (12, 2):
        raise RuntimeError(f"trajectory must be [B,12,2], got {tuple(traj.shape)}")
    traj = traj[:, None].expand(B, K, 12, 2).reshape(B * K, 12, 2)[effective]
    return {
        "history": hist_local,
        "context": hist_context,
        "physics": physics,
        "target": target,
        "trajectory": traj,
        "origins": origins,
        "plan": plan,
        "effective": effective,
        "batch_size": B,
        "topk": K,
        "physics_full": moved["anchor_future_latent"],
        "sample_ids": list(batch["sample_id"]),
    }


def scatter_absolute_endpoint(endpoint_windows: torch.Tensor, prepared: dict) -> torch.Tensor:
    B = int(prepared["batch_size"])
    K = int(prepared["topk"])
    effective = prepared["effective"]
    if endpoint_windows.shape[0] != int(effective.sum().item()):
        raise ValueError("absolute endpoint window count differs from valid Top-2 slots")
    padded = endpoint_windows.new_zeros((B * K, *endpoint_windows.shape[1:]))
    dst = torch.nonzero(effective, as_tuple=False).flatten()
    padded = padded.index_copy(0, dst, endpoint_windows)
    return scatter_windows(
        padded.reshape(B, K, *endpoint_windows.shape[1:]),
        prepared["plan"],
        base=prepared["physics_full"],
    )


def semantic_loss_for_endpoint(
    endpoint_full,
    *,
    sample_ids,
    edit_cache,
    vae,
    class_weights,
    lovasz_weight,
):
    records = edit_cache.get_batch(sample_ids)
    indices = [semantic_targets_for_sample(rec)[0].to(endpoint_full.device) for rec in records]
    logits = vae.decode_logits_at_flat_indices(endpoint_full, indices)
    return absolute_future_semantic_loss(
        logits,
        records,
        class_weights=class_weights,
        lovasz_weight=float(lovasz_weight),
    )


def _fixed_noise_like(x: torch.Tensor, seed: int) -> torch.Tensor:
    gen = torch.Generator(device=x.device)
    gen.manual_seed(int(seed))
    return torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=gen)


def _lr_ratio(step: int, total_steps: int, warmup_fraction: float, min_ratio: float) -> float:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    warm = max(1, int(round(total_steps * float(warmup_fraction))))
    if step < warm:
        start = 1e-3
        return start + (1.0 - start) * float(step + 1) / float(warm)
    progress = min(max((step - warm) / max(total_steps - warm, 1), 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(min_ratio) + (1.0 - float(min_ratio)) * cosine


def _set_lr(optimizer, ratio: float) -> dict:
    out = {}
    for group in optimizer.param_groups:
        base_lr = float(group["base_lr"])
        group["lr"] = base_lr * float(ratio)
        out[str(group.get("group_name", "group"))] = float(group["lr"])
    return out


def _build_optimizer(model, reuse, *, wm_lr: float, new_lr: float, weight_decay: float):
    loaded = set(reuse.get("loaded_keys", ()))
    pretrained, new = [], []
    pretrained_names, new_names = [], []
    for name, p in model.transition.named_parameters():
        if not p.requires_grad:
            continue
        if name in loaded:
            pretrained.append(p)
            pretrained_names.append(name)
        else:
            new.append(p)
            new_names.append(name)
    if not pretrained:
        raise RuntimeError("P0-F9 found no loaded official OccFM transition parameters")
    if not new:
        raise RuntimeError("P0-F9 found no new physics/context parameters")
    groups = [
        {
            "params": pretrained,
            "lr": float(wm_lr),
            "base_lr": float(wm_lr),
            "group_name": "pretrained_occfm",
        },
        {
            "params": new,
            "lr": float(new_lr),
            "base_lr": float(new_lr),
            "group_name": "new_physics_context",
        },
    ]
    opt = torch.optim.AdamW(groups, weight_decay=float(weight_decay))
    return opt, {
        "protocol": OPTIMIZER_PROTOCOL,
        "wm_lr": float(wm_lr),
        "new_lr": float(new_lr),
        "weight_decay": float(weight_decay),
        "num_pretrained_tensors": len(pretrained_names),
        "num_new_tensors": len(new_names),
        "num_pretrained_parameters": int(sum(p.numel() for p in pretrained)),
        "num_new_parameters": int(sum(p.numel() for p in new)),
        "new_parameter_names": new_names,
    }


def _cpu_state(model):
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def _ema_payload(ema: ModelEMA):
    obj = ema.state_dict()
    return {
        "updates": int(obj["updates"]),
        "decay": float(obj["decay"]),
        "state_dict": {k: v.detach().cpu() for k, v in obj["state_dict"].items()},
    }


def _architecture(args):
    return {
        "protocol": P0_F9_PROTOCOL,
        "stage": 1,
        "window_hw": [20, 20],
        "context_hw": [40, 40],
        "topk": 2,
        "future_frames": 6,
        "native_backbone_hist_last": HIST_LAST,
        "flow": "gaussian_noise_to_absolute_gt_future",
        "latent_distribution": "deterministic_posterior_sample_matching_occfm_cache",
        "physics_prior": "strong_w2det_condition_and_fallback_not_flow_source",
        "physics_fusion": "zero_gated_mid_cross_attention_plus_zero_init_bias_free_token_condition",
        "sample_steps": int(args.sample_steps),
        "unconditional_probability": float(args.uncond_prob),
        "guidance_scale": float(args.guidance_scale),
        "semantic_population": "p0_f8_full_compact_pool_reused_as_absolute_future_targets",
        "semantic_horizon_weighting": "equal_over_six_future_frames",
        "loss": "weighted_dynamic_semantic_CE_plus_Lovasz_plus_small_FM_regularizer",
        "fm_weight": float(args.fm_weight),
        "lovasz_weight": float(args.lovasz_weight),
        "vae": "frozen_official_occfm_vae",
        "ema_decay": float(args.ema_decay),
        "scene_sampling": "inverse_scene_window_count_weighted_sampling",
    }


def _payload(
    model,
    ema,
    optimizer,
    *,
    step,
    best_val,
    history,
    args,
    train_ds,
    val_ds,
    train_sem,
    val_sem,
    reuse,
    optimizer_info,
    class_weights,
    skipped,
):
    return {
        "state_dict": _cpu_state(model),
        "ema": _ema_payload(ema),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": int(step),
        "best_val_objective": float(best_val),
        "training_history": list(history),
        "architecture": _architecture(args),
        "train_metadata": train_ds.metadata,
        "val_metadata": val_ds.metadata,
        "train_cache_index_sha256": file_sha256(train_ds.root / "index.json"),
        "val_cache_index_sha256": file_sha256(val_ds.root / "index.json"),
        "train_semantic_metadata": train_sem.metadata,
        "val_semantic_metadata": val_sem.metadata,
        "train_semantic_sha256": file_sha256(train_sem.path),
        "val_semantic_sha256": file_sha256(val_sem.path),
        "upstream_checkpoint": str(Path(args.upstream_ckpt).resolve()),
        "upstream_checkpoint_sha256": file_sha256(args.upstream_ckpt),
        "vae_checkpoint_sha256": file_sha256(args.vae_ckpt),
        "upstream_reuse": reuse,
        "optimizer_contract": optimizer_info,
        "semantic_class_weights": class_weights.detach().cpu(),
        "skipped_empty_train_batches": int(skipped),
        "args": vars(args),
    }


def _same_float(a, b, *, atol=1e-12):
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=float(atol))


def _validate_resume_checkpoint(
    ck,
    args,
    train_ds,
    val_ds,
    train_sem,
    val_sem,
    class_weights,
):
    arch = ck.get("architecture", {})
    if arch.get("protocol") != P0_F9_PROTOCOL or int(arch.get("stage", -1)) != 1:
        raise RuntimeError("resume checkpoint is not audited P0-F9 Stage-1")
    if int(arch.get("native_backbone_hist_last", -1)) != HIST_LAST:
        raise RuntimeError("resume checkpoint HIST_LAST contract differs")
    if ck.get("train_cache_index_sha256") != file_sha256(train_ds.root / "index.json"):
        raise RuntimeError("resume train cache differs from checkpoint")
    if ck.get("val_cache_index_sha256") != file_sha256(val_ds.root / "index.json"):
        raise RuntimeError("resume val cache differs from checkpoint")
    if ck.get("train_semantic_sha256") != file_sha256(train_sem.path):
        raise RuntimeError("resume train semantic sidecar differs from checkpoint")
    if ck.get("val_semantic_sha256") != file_sha256(val_sem.path):
        raise RuntimeError("resume val semantic sidecar differs from checkpoint")
    if ck.get("upstream_checkpoint_sha256") != file_sha256(args.upstream_ckpt):
        raise RuntimeError("resume upstream OccFM checkpoint differs")
    if ck.get("vae_checkpoint_sha256") != file_sha256(args.vae_ckpt):
        raise RuntimeError("resume VAE checkpoint differs")

    saved = ck.get("args", {})
    exact_keys = ("steps", "batch_size", "sample_steps", "seed", "min_train_windows")
    float_keys = (
        "wm_lr", "new_lr", "weight_decay", "warmup_fraction", "min_lr_ratio",
        "fm_weight", "lovasz_weight", "uncond_prob", "guidance_scale", "ema_decay",
    )
    for key in exact_keys:
        if key not in saved or int(saved[key]) != int(getattr(args, key)):
            raise RuntimeError(f"resume argument differs for {key}")
    for key in float_keys:
        if key not in saved or not _same_float(saved[key], getattr(args, key)):
            raise RuntimeError(f"resume argument differs for {key}")

    saved_weights = torch.as_tensor(ck.get("semantic_class_weights"), dtype=torch.float32)
    if not torch.allclose(saved_weights, class_weights, atol=0, rtol=0):
        raise RuntimeError("semantic class weights differ from resume checkpoint")
    step = int(ck.get("step", -1))
    if step < 0 or step > int(args.steps):
        raise RuntimeError(f"resume step {step} incompatible with target steps={args.steps}")


def validate(
    model,
    loader,
    device,
    edit_cache,
    vae,
    class_weights,
    *,
    fm_weight,
    lovasz_weight,
    use_amp,
    seed,
):
    was_training = model.training
    model.eval()
    fm_sum = 0.0
    fm_windows = 0
    sem_sum = 0.0
    sem_weight = 0
    ce_sum = 0.0
    lovasz_sum = 0.0
    acc_sum = 0.0
    acc_weight = 0
    dyn_acc_sum = 0.0
    dyn_acc_weight = 0
    bg_false_sum = 0.0
    bg_false_weight = 0
    cosine_sum = 0.0
    skipped = 0

    for batch_idx, batch in enumerate(loader):
        prepared = prepare_batch(batch, device)
        if prepared is None:
            skipped += 1
            continue
        source_noise = _fixed_noise_like(prepared["target"], int(seed) + batch_idx)
        with torch.enable_grad():
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_amp,
            ):
                fm_loss, info = model.flow_loss(
                    prepared["history"],
                    prepared["target"],
                    prepared["physics"],
                    history_context=prepared["context"],
                    trajectory=prepared["trajectory"],
                    window_origins=prepared["origins"],
                    t_override=0.5,
                    source_noise=source_noise,
                    return_endpoint=True,
                    force_conditioned=True,
                )
                endpoint = scatter_absolute_endpoint(info["predicted_endpoint"], prepared)
                sem_loss, sem_info = semantic_loss_for_endpoint(
                    endpoint,
                    sample_ids=prepared["sample_ids"],
                    edit_cache=edit_cache,
                    vae=vae,
                    class_weights=class_weights,
                    lovasz_weight=lovasz_weight,
                )
        nwin = int(prepared["history"].shape[0])
        nsample = int(prepared["batch_size"])
        nvox = int(sem_info["num_supervised_voxels"])
        ndyn = int(sem_info.get("num_dynamic_voxels", 0))
        nbg = int(sem_info.get("num_background_voxels", 0))
        fm_sum += float(fm_loss.detach().item()) * nwin
        fm_windows += nwin
        cosine_sum += float(info["cosine"]) * nwin
        if nvox > 0:
            # The semantic objective is already equal-horizon inside each batch;
            # aggregate batches by sample count rather than reintroducing voxel
            # population weighting here.
            sem_sum += float(sem_loss.detach().item()) * nsample
            ce_sum += float(sem_info["ce"]) * nsample
            lovasz_sum += float(sem_info["lovasz"]) * nsample
            sem_weight += nsample
            acc_sum += float(sem_info["accuracy"]) * nvox
            acc_weight += nvox
            if ndyn > 0 and math.isfinite(float(sem_info["dynamic_accuracy"])):
                dyn_acc_sum += float(sem_info["dynamic_accuracy"]) * ndyn
                dyn_acc_weight += ndyn
            if nbg > 0 and math.isfinite(float(sem_info["background_false_dynamic_rate"])):
                bg_false_sum += float(sem_info["background_false_dynamic_rate"]) * nbg
                bg_false_weight += nbg
        del endpoint, sem_loss, fm_loss
    model.train(was_training)
    if fm_windows <= 0 or sem_weight <= 0:
        raise RuntimeError("P0-F9 validation has no valid routed semantic windows")
    fm_avg = fm_sum / fm_windows
    sem_avg = sem_sum / sem_weight
    return {
        "objective": sem_avg + float(fm_weight) * fm_avg,
        "semantic_loss": sem_avg,
        "semantic_ce": ce_sum / sem_weight,
        "semantic_lovasz": lovasz_sum / sem_weight,
        "fm_loss": fm_avg,
        "fm_weight": float(fm_weight),
        "weighted_fm_loss": float(fm_weight) * fm_avg,
        "semantic_accuracy": acc_sum / max(acc_weight, 1),
        "dynamic_accuracy": dyn_acc_sum / max(dyn_acc_weight, 1),
        "background_false_dynamic_rate": bg_false_sum / max(bg_false_weight, 1),
        "fm_cosine": cosine_sum / fm_windows,
        "num_windows": fm_windows,
        "num_semantic_voxels": acc_weight,
        "num_dynamic_voxels": dyn_acc_weight,
        "num_background_voxels": bg_false_weight,
        "skipped_empty_batches": skipped,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-cache", required=True)
    p.add_argument("--val-cache", required=True)
    p.add_argument("--train-semantic-targets", required=True,
                   help="existing P0-F8 edit sidecar, reused only for absolute semantic targets")
    p.add_argument("--val-semantic-targets", required=True)
    p.add_argument("--upstream-ckpt", required=True)
    p.add_argument("--vae-ckpt", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--wm-lr", type=float, default=2e-5)
    p.add_argument("--new-lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--warmup-fraction", type=float, default=0.05)
    p.add_argument("--min-lr-ratio", type=float, default=0.2)
    p.add_argument("--fm-weight", type=float, default=0.1)
    p.add_argument("--lovasz-weight", type=float, default=1.0)
    p.add_argument("--sample-steps", type=int, default=10)
    p.add_argument("--uncond-prob", type=float, default=0.0,
                   help="Stage-1 default disables condition dropout; nonzero values are honored")
    p.add_argument("--guidance-scale", type=float, default=1.0)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--val-every", type=int, default=200)
    p.add_argument("--min-train-windows", type=int, default=4000)
    p.add_argument("--seed", type=int, default=20260904)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--resume-from", default=None)
    a = p.parse_args()
    if min(a.steps, a.batch_size, a.val_every, a.min_train_windows) <= 0:
        raise ValueError("steps/batch-size/val-every/min-train-windows must be positive")
    if not 0.0 <= a.warmup_fraction < 1.0 or not 0.0 < a.min_lr_ratio <= 1.0:
        raise ValueError("invalid LR schedule")
    if a.fm_weight < 0 or a.lovasz_weight < 0:
        raise ValueError("loss weights must be non-negative")
    if not 0.0 <= a.uncond_prob < 1.0:
        raise ValueError("uncond-prob must be in [0,1)")

    _seed_all(a.seed)
    device = torch.device(a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu")
    train_ds = MSPWorldModelCacheDataset(a.train_cache)
    val_ds = MSPWorldModelCacheDataset(a.val_cache)
    _validate_cache_pair(train_ds, val_ds, min_train_windows=a.min_train_windows)
    train_sem = EditTargetCache(a.train_semantic_targets)
    val_sem = EditTargetCache(a.val_semantic_targets)
    _validate_semantic_sidecar(train_sem, train_ds, "train")
    _validate_semantic_sidecar(val_sem, val_ds, "val")

    vae_sha = file_sha256(a.vae_ckpt)
    if vae_sha != train_ds.metadata.get("vae_checkpoint_sha256"):
        raise RuntimeError("VAE checkpoint differs from P0-F9 cache")

    pin = device.type == "cuda"
    sampler = _scene_balanced_sampler(train_ds, seed=a.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=a.batch_size,
        sampler=sampler,
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

    model = make_p0_f9_model(
        20,
        sample_steps=a.sample_steps,
        unconditional_probability=a.uncond_prob,
        guidance_scale=a.guidance_scale,
        hist_last=HIST_LAST,
    ).to(device)
    reuse = load_shape_safe(model.transition, a.upstream_ckpt, verbose=True)
    if "traj_encoder.0.weight" not in set(reuse.get("loaded_keys", ())):
        raise RuntimeError("P0-F9 requires the official OccFM-Fut epoch=000196 checkpoint")
    official_reuse_fraction = require_checkpoint_reuse(reuse, min_fraction=0.80)

    vae_model, _ = load_official_vae(UP, a.vae_ckpt, device)
    vae = OccFMVAEAdapter(vae_model)
    if any(p.requires_grad for p in vae_model.parameters()):
        raise RuntimeError("P0-F9 Stage-1 VAE must remain frozen")

    class_weights = class_weights_from_edit_cache(train_sem)
    optimizer, optimizer_info = _build_optimizer(
        model,
        reuse,
        wm_lr=a.wm_lr,
        new_lr=a.new_lr,
        weight_decay=a.weight_decay,
    )
    ema = ModelEMA(model, decay=a.ema_decay)
    use_amp = bool(a.amp and device.type == "cuda")
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    step = 0
    best_val = float("inf")
    history = []
    skipped = 0
    if a.resume_from:
        ck = torch.load(a.resume_from, map_location="cpu", weights_only=False)
        _validate_resume_checkpoint(
            ck, a, train_ds, val_ds, train_sem, val_sem, class_weights
        )
        model.load_state_dict(ck["state_dict"], strict=True)
        ema.load_state_dict(ck["ema"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        step = int(ck["step"])
        best_val = float(ck.get("best_val_objective", float("inf")))
        history = list(ck.get("training_history", []))
        skipped = int(ck.get("skipped_empty_train_batches", 0))
        print(f"resumed P0-F9 from {a.resume_from}: step={step} best={best_val:.6f}")

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
            skipped += 1
            continue

        lr_ratio = _lr_ratio(step, a.steps, a.warmup_fraction, a.min_lr_ratio)
        lrs = _set_lr(optimizer, lr_ratio)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_amp,
        ):
            fm_loss, info = model.flow_loss(
                prepared["history"],
                prepared["target"],
                prepared["physics"],
                history_context=prepared["context"],
                trajectory=prepared["trajectory"],
                window_origins=prepared["origins"],
                return_endpoint=True,
                # Do not force conditioned here: --uncond-prob must actually
                # control training behavior. Default 0.0 remains fully conditioned.
                force_conditioned=False,
            )
            endpoint = scatter_absolute_endpoint(info["predicted_endpoint"], prepared)
            sem_loss, sem_info = semantic_loss_for_endpoint(
                endpoint,
                sample_ids=prepared["sample_ids"],
                edit_cache=train_sem,
                vae=vae,
                class_weights=class_weights,
                lovasz_weight=a.lovasz_weight,
            )
            total_loss = sem_loss + float(a.fm_weight) * fm_loss
        if not torch.isfinite(total_loss):
            raise RuntimeError(
                f"non-finite P0-F9 loss step={step}: total={total_loss} sem={sem_loss} fm={fm_loss}"
            )
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        ema_decay_now = ema.update(model)
        step += 1

        authority = float(model.transition.physics_fusion.authority.cpu())
        train_info = {
            "objective": float(total_loss.detach().item()),
            "semantic_loss": float(sem_loss.detach().item()),
            "semantic_ce": float(sem_info["ce"]),
            "semantic_lovasz": float(sem_info["lovasz"]),
            "semantic_accuracy": float(sem_info["accuracy"]),
            "dynamic_accuracy": float(sem_info["dynamic_accuracy"]),
            "background_false_dynamic_rate": float(sem_info["background_false_dynamic_rate"]),
            "num_semantic_voxels": int(sem_info["num_supervised_voxels"]),
            "num_dynamic_voxels": int(sem_info.get("num_dynamic_voxels", 0)),
            "num_background_voxels": int(sem_info.get("num_background_voxels", 0)),
            "per_horizon_voxels": list(sem_info.get("per_horizon_voxels", [])),
            "fm_loss": float(fm_loss.detach().item()),
            "fm_weight": float(a.fm_weight),
            "fm_cosine": float(info["cosine"]),
            "conditioned_fraction": float(info["conditioned_fraction"]),
            "physics_authority": authority,
            "grad_norm_before_clip": float(torch.as_tensor(grad_norm).detach().cpu()),
            "ema_decay": float(ema_decay_now),
            "lr_ratio": float(lr_ratio),
            "lrs": lrs,
        }
        if step == 1 or step % 20 == 0:
            print(
                f"step={step} total={train_info['objective']:.6f} "
                f"sem={train_info['semantic_loss']:.6f} ce={train_info['semantic_ce']:.6f} "
                f"lov={train_info['semantic_lovasz']:.6f} fm={train_info['fm_loss']:.6f} "
                f"dyn_acc={train_info['dynamic_accuracy']:.4f} "
                f"bg_false={train_info['background_false_dynamic_rate']:.4f} "
                f"cond={train_info['conditioned_fraction']:.3f} "
                f"phys={authority:+.5f} lr={lrs}"
            )

        if step % a.val_every == 0 or step == a.steps:
            val = validate(
                ema.model,
                val_loader,
                device,
                val_sem,
                vae,
                class_weights,
                fm_weight=a.fm_weight,
                lovasz_weight=a.lovasz_weight,
                use_amp=use_amp,
                seed=a.seed + 100000,
            )
            row = {"step": step, "train": train_info, "val_ema": val}
            history.append(row)
            print("validation", json.dumps(row))
            current_best = min(best_val, float(val["objective"]))
            payload = _payload(
                model,
                ema,
                optimizer,
                step=step,
                best_val=current_best,
                history=history,
                args=a,
                train_ds=train_ds,
                val_ds=val_ds,
                train_sem=train_sem,
                val_sem=val_sem,
                reuse=reuse,
                optimizer_info=optimizer_info,
                class_weights=class_weights,
                skipped=skipped,
            )
            torch.save(payload, out / f"step_{step:04d}.pt")
            torch.save(payload, out / "latest.pt")
            if float(val["objective"]) < best_val:
                best_val = float(val["objective"])
                torch.save(payload, out / "best.pt")

    final_payload = _payload(
        model,
        ema,
        optimizer,
        step=step,
        best_val=best_val,
        history=history,
        args=a,
        train_ds=train_ds,
        val_ds=val_ds,
        train_sem=train_sem,
        val_sem=val_sem,
        reuse=reuse,
        optimizer_info=optimizer_info,
        class_weights=class_weights,
        skipped=skipped,
    )
    torch.save(final_payload, out / "last.pt")
    (out / "training_report.json").write_text(json.dumps({
        "protocol": P0_F9_PROTOCOL,
        "stage": 1,
        "best_val_objective": best_val,
        "official_transition_reuse_fraction": official_reuse_fraction,
        "optimizer_contract": optimizer_info,
        "semantic_class_weights": class_weights.tolist(),
        "history": history,
        "skipped_empty_train_batches": skipped,
        "decision": (
            "Evaluate EMA checkpoints with true Strong-W2Det vs P0-F9 Overall/Moving 1s/2s/3s; "
            "surrogate validation is not the deployment decision metric."
        ),
    }, indent=2), encoding="utf-8")
    print("saved", out / "last.pt")


if __name__ == "__main__":
    main()
