#!/usr/bin/env python3
"""Formal full-data P0-F9/M training from the released OccFM-Fut checkpoint.

This run is intentionally *not* a continuation of v7/U/M development
checkpoints.  It asks whether the final validated recipe scales when trained on
all eligible train windows from the official pretrained OccFM initialization.

Fixed method contract:
- released OccFM-Fut transition weights are loaded shape-safely;
- P0-F9 Top-2 sparse routing, Strong-W2Det physics condition/fallback and the
  original mean-context branch are retained;
- native Gaussian->absolute-future flow matching is the only objective family;
- true-motion cells use normalized lambda=2 weighting;
- no semantic CE/Lovasz, decoder loss, ordered context, selector or margin head.

The LR schedule is frozen against ``schedule_epochs`` (default 10).  A first run
may stop at epoch 5 and later resume to epoch 10 without restarting/redefining
the schedule.  Formal checkpoints are epoch milestones plus ``latest.pt`` and
``last.pt``; there is deliberately no FM-selected ``best.pt`` because deployment
Overall/Moving, not fixed-t FM alone, chooses the final model.
"""
from __future__ import annotations

import argparse
from collections import Counter
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
from torch.utils.data import DataLoader, Sampler

from real_motion.checkpoint import load_shape_safe, require_checkpoint_reuse
from real_motion.model_ema import ModelEMA
from real_motion.models.p0_f9 import P0_F9_PROTOCOL, make_p0_f9_model
from real_motion.motion_mask_sidecar import (
    MOTION_MASK_PROTOCOL,
    MotionMaskSidecar,
    effective_motion_weight_mass,
    normalized_motion_weighted_mse,
)
from real_motion.msp_wm_cache import MSPWorldModelCacheDataset, collate_msp_wm
from real_motion.native_forecast import crop_coherent_source_noise, deterministic_sample_seed
from real_motion.occfm_io import file_sha256
from tools.real_motion.build_p0_f9_cache_fast import P0_F9_CACHE_PROTOCOL
from tools.real_motion.build_p0_f9_full_native_cache import PROTOCOL as FULL_CACHE_BUILD_PROTOCOL
from tools.real_motion.train_p0_f9_native_sparse_forecast import (
    HIST_LAST,
    _build_optimizer,
    _cpu_state,
    _ema_payload,
    _lr_ratio,
    _seed_all,
    _set_lr,
    _validate_cache_pair,
    prepare_batch,
)
from tools.real_motion.train_p0_f9_v7_native_fm_only import (
    _validate_vae_provenance,
    validate_fm,
)
from tools.real_motion.train_p0_f9_v8_um_phase import (
    SOURCE_SPATIAL_CONTRACT,
    _crop_motion_mask,
    _flow_and_squared_error,
    _restore_rng_state,
    _rng_state,
    _validate_motion_sidecar,
)


PROTOCOL = "p0_f9_v9_full_data_m_official_init_v1"
SAMPLING_PROTOCOL = "scene_balanced_epoch_indexed_multinomial_full_population_v1"
SCHEDULE_PROTOCOL = "single_cosine_schedule_over_frozen_schedule_epochs_v1"
CHECKPOINT_PROTOCOL = "epoch_milestones_plus_latest_last_no_fm_best_v1"


class FixedIndexSampler(Sampler[int]):
    def __init__(self, indices):
        self.indices = [int(x) for x in indices]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def _epoch_seed(seed: int, epoch: int) -> int:
    return deterministic_sample_seed(
        f"full_epoch:{int(epoch)}", int(seed), stream="v9_full_scene_sampler"
    )


def scene_balanced_epoch_indices(ds, *, seed: int, epoch: int) -> list[int]:
    scenes = [str(e["scene_name"]) for e in ds.entries]
    counts = Counter(scenes)
    weights = torch.tensor([1.0 / float(counts[s]) for s in scenes], dtype=torch.double)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(_epoch_seed(seed, epoch))
    return torch.multinomial(
        weights,
        num_samples=len(ds),
        replacement=True,
        generator=gen,
    ).tolist()


def schedule_shape(num_samples: int, batch_size: int, schedule_epochs: int) -> dict:
    if min(int(num_samples), int(batch_size), int(schedule_epochs)) <= 0:
        raise ValueError("schedule shape arguments must be positive")
    steps_per_epoch = int(math.ceil(int(num_samples) / float(int(batch_size))))
    return {
        "num_samples": int(num_samples),
        "batch_size": int(batch_size),
        "steps_per_epoch": steps_per_epoch,
        "schedule_epochs": int(schedule_epochs),
        "total_schedule_steps": steps_per_epoch * int(schedule_epochs),
    }


def _validate_full_train_cache(train_ds) -> None:
    meta = train_ds.metadata
    checks = {
        "protocol": P0_F9_CACHE_PROTOCOL,
        "build_protocol": FULL_CACHE_BUILD_PROTOCOL,
        "direct_full_data_build": True,
        "all_eligible_windows": True,
        "source_msp_mode": "train",
        "target": "absolute_gt_future_vae_latent",
        "vae_mode": "sample",
        "topk": 2,
        "window_hw": [20, 20],
        "context_hw": [40, 40],
    }
    for key, expected in checks.items():
        if meta.get(key) != expected:
            raise RuntimeError(
                f"full train cache mismatch for {key}: {meta.get(key)!r} != {expected!r}"
            )
    declared = int(meta.get("native_eligible_window_count", -1))
    if declared != len(train_ds):
        raise RuntimeError(
            f"full train cache is not complete: declared={declared} actual={len(train_ds)}"
        )


def _architecture(a, train_ds) -> dict:
    return {
        "protocol": P0_F9_PROTOCOL,
        "stage": 1,
        "training_protocol": PROTOCOL,
        "initialization": "released_occfm_fut_shape_safe_pretrained_transition",
        "window_hw": [20, 20],
        "context_hw": [40, 40],
        "topk": 2,
        "future_frames": 6,
        "native_backbone_hist_last": HIST_LAST,
        "flow": "gaussian_noise_to_absolute_gt_future",
        "flow_source_spatial_contract": SOURCE_SPATIAL_CONTRACT,
        "latent_distribution": "deterministic_posterior_sample_matching_occfm_cache",
        "physics_prior": "strong_w2det_condition_and_fallback_not_flow_source",
        "physics_fusion": "zero_gated_mid_cross_attention_plus_zero_init_bias_free_token_condition",
        "history_context": "original_mean_context_branch_no_ordered_residual",
        "ordered_context": False,
        "training_objective": "normalized_true_motion_weighted_native_flow_matching_velocity_mse",
        "weighted_loss_formula": "sum((1+lambda*m)*squared_error)/sum(1+lambda*m)",
        "motion_mask_protocol": MOTION_MASK_PROTOCOL,
        "motion_lambda_applied": float(a.motion_weight_lambda),
        "semantic_auxiliary": False,
        "lovasz_auxiliary": False,
        "vae_decoder_in_training_graph": False,
        "sample_steps": int(a.sample_steps),
        "unconditional_probability": float(a.uncond_prob),
        "guidance_scale": float(a.guidance_scale),
        "ema_decay": float(a.ema_decay),
        "grad_norm_clip": float(a.grad_clip),
        "scene_sampling": SAMPLING_PROTOCOL,
        "full_train_samples": len(train_ds),
        "schedule_protocol": SCHEDULE_PROTOCOL,
        "schedule_epochs": int(a.schedule_epochs),
    }


def _payload(
    model,
    ema,
    optimizer,
    *,
    epoch: int,
    global_step: int,
    history: list,
    a,
    train_ds,
    val_ds,
    sidecar,
    reuse,
    optimizer_info,
    vae_sha: str,
    schedule: dict,
    skipped: int,
) -> dict:
    return {
        "state_dict": _cpu_state(model),
        "ema": _ema_payload(ema),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
        "step": int(global_step),
        "global_step": int(global_step),
        "training_history": list(history),
        "architecture": _architecture(a, train_ds),
        "train_metadata": train_ds.metadata,
        "val_metadata": val_ds.metadata,
        "train_cache_index_sha256": file_sha256(train_ds.root / "index.json"),
        "val_cache_index_sha256": file_sha256(val_ds.root / "index.json"),
        "motion_mask_sidecar": str(sidecar.path),
        "motion_mask_sidecar_sha256": file_sha256(sidecar.path),
        "motion_mask_metadata": sidecar.metadata,
        "upstream_checkpoint": str(Path(a.upstream_ckpt).resolve()),
        "upstream_checkpoint_sha256": file_sha256(a.upstream_ckpt),
        "vae_checkpoint": str(Path(a.vae_ckpt).resolve()),
        "vae_checkpoint_sha256": vae_sha,
        "upstream_reuse": reuse,
        "optimizer_contract": optimizer_info,
        "schedule": schedule,
        "rng_state": _rng_state(),
        "skipped_empty_train_batches": int(skipped),
        "checkpoint_protocol": CHECKPOINT_PROTOCOL,
        "args": vars(a),
    }


def _save_payload(out: Path, name: str, payload: dict) -> None:
    tmp = out / (name + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(out / name)


def _validate_resume(ck, a, train_ds, val_ds, sidecar, vae_sha: str, schedule: dict) -> int:
    arch = ck.get("architecture", {})
    if arch.get("training_protocol") != PROTOCOL:
        raise RuntimeError("resume checkpoint is not v9 full-data M")
    if arch.get("initialization") != "released_occfm_fut_shape_safe_pretrained_transition":
        raise RuntimeError("resume initialization contract differs")
    if bool(arch.get("ordered_context", True)):
        raise RuntimeError("resume unexpectedly contains ordered context")
    if float(arch.get("motion_lambda_applied", -1.0)) != float(a.motion_weight_lambda):
        raise RuntimeError("resume motion lambda differs")
    if ck.get("train_cache_index_sha256") != file_sha256(train_ds.root / "index.json"):
        raise RuntimeError("resume full train cache differs")
    if ck.get("val_cache_index_sha256") != file_sha256(val_ds.root / "index.json"):
        raise RuntimeError("resume val cache differs")
    if ck.get("motion_mask_sidecar_sha256") != file_sha256(sidecar.path):
        raise RuntimeError("resume motion-mask sidecar differs")
    if ck.get("upstream_checkpoint_sha256") != file_sha256(a.upstream_ckpt):
        raise RuntimeError("resume upstream OccFM checkpoint differs")
    if ck.get("vae_checkpoint_sha256") != vae_sha:
        raise RuntimeError("resume VAE provenance differs")
    if ck.get("schedule") != schedule:
        raise RuntimeError("resume schedule shape differs; do not redefine schedule_epochs")

    saved = ck.get("args") or {}
    for key in ("batch_size", "sample_steps", "seed", "schedule_epochs"):
        if int(saved.get(key, -1)) != int(getattr(a, key)):
            raise RuntimeError(f"resume argument differs for {key}")
    for key in (
        "wm_lr",
        "new_lr",
        "weight_decay",
        "warmup_fraction",
        "min_lr_ratio",
        "uncond_prob",
        "guidance_scale",
        "ema_decay",
        "grad_clip",
        "motion_weight_lambda",
    ):
        if not math.isclose(
            float(saved.get(key, float("nan"))),
            float(getattr(a, key)),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(f"resume argument differs for {key}")
    epoch = int(ck.get("epoch", -1))
    if epoch < 0 or epoch >= int(a.run_until_epoch):
        raise RuntimeError(
            f"resume epoch {epoch} incompatible with run-until={a.run_until_epoch}"
        )
    expected_step = epoch * int(schedule["steps_per_epoch"])
    if int(ck.get("global_step", -1)) != expected_step:
        raise RuntimeError("resume checkpoint is not an epoch-boundary full-data state")
    return epoch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-cache", required=True)
    p.add_argument("--val-cache", required=True)
    p.add_argument("--motion-mask-sidecar", required=True)
    p.add_argument("--upstream-ckpt", required=True)
    p.add_argument("--vae-ckpt", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--schedule-epochs", type=int, default=10)
    p.add_argument("--run-until-epoch", type=int, default=5)
    p.add_argument("--milestone-epochs", type=int, nargs="+", default=[1, 3, 5, 10])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--wm-lr", type=float, default=2e-5)
    p.add_argument("--new-lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--warmup-fraction", type=float, default=0.05)
    p.add_argument("--min-lr-ratio", type=float, default=0.2)
    p.add_argument("--sample-steps", type=int, default=10)
    p.add_argument("--uncond-prob", type=float, default=0.0)
    p.add_argument("--guidance-scale", type=float, default=1.0)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--motion-weight-lambda", type=float, default=2.0)
    p.add_argument("--min-train-windows", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20260906)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--resume-from", default=None)
    a = p.parse_args()

    if min(a.schedule_epochs, a.run_until_epoch, a.batch_size, a.min_train_windows) <= 0:
        raise ValueError("epoch/batch/min-train settings must be positive")
    if a.run_until_epoch > a.schedule_epochs:
        raise ValueError("run-until-epoch cannot exceed frozen schedule-epochs")
    if not 0.0 <= a.warmup_fraction < 1.0 or not 0.0 < a.min_lr_ratio <= 1.0:
        raise ValueError("invalid LR schedule")
    if a.motion_weight_lambda < 0 or a.grad_clip <= 0:
        raise ValueError("motion lambda must be non-negative and grad clip positive")
    if not 0.0 <= a.uncond_prob < 1.0:
        raise ValueError("uncond-prob must be in [0,1)")
    milestones = sorted(set(int(x) for x in a.milestone_epochs))
    if not milestones or any(x <= 0 or x > a.schedule_epochs for x in milestones):
        raise ValueError("invalid milestone epochs")

    _seed_all(a.seed)
    device = torch.device(a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("formal full-data P0-F9 training requires CUDA")
    use_amp = bool(a.amp)

    train_ds = MSPWorldModelCacheDataset(a.train_cache)
    val_ds = MSPWorldModelCacheDataset(a.val_cache)
    _validate_full_train_cache(train_ds)
    _validate_cache_pair(train_ds, val_ds, min_train_windows=int(a.min_train_windows))
    vae_sha = _validate_vae_provenance(train_ds, val_ds, a.vae_ckpt)
    sidecar = MotionMaskSidecar(a.motion_mask_sidecar)
    _validate_motion_sidecar(sidecar, train_ds)

    schedule = schedule_shape(len(train_ds), a.batch_size, a.schedule_epochs)
    if int(schedule["steps_per_epoch"]) <= 0:
        raise RuntimeError("invalid zero-step full-data epoch")

    val_loader = DataLoader(
        val_ds,
        batch_size=a.batch_size,
        shuffle=False,
        num_workers=a.num_workers,
        collate_fn=collate_msp_wm,
        drop_last=False,
        pin_memory=True,
    )

    model = make_p0_f9_model(
        20,
        sample_steps=a.sample_steps,
        unconditional_probability=a.uncond_prob,
        guidance_scale=a.guidance_scale,
        hist_last=HIST_LAST,
        ordered_context=False,
    ).to(device)
    reuse = load_shape_safe(model.transition, a.upstream_ckpt, verbose=True)
    if "traj_encoder.0.weight" not in set(reuse.get("loaded_keys", ())):
        raise RuntimeError("v9 full-data training requires released OccFM-Fut epoch=000196")
    official_reuse_fraction = require_checkpoint_reuse(reuse, min_fraction=0.80)

    optimizer, optimizer_info = _build_optimizer(
        model,
        reuse,
        wm_lr=a.wm_lr,
        new_lr=a.new_lr,
        weight_decay=a.weight_decay,
    )
    ema = ModelEMA(model, decay=a.ema_decay)
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    global_step = 0
    history = []
    skipped = 0
    if a.resume_from:
        ck = torch.load(a.resume_from, map_location="cpu", weights_only=False)
        start_epoch = _validate_resume(
            ck, a, train_ds, val_ds, sidecar, vae_sha, schedule
        )
        model.load_state_dict(ck["state_dict"], strict=True)
        ema.load_state_dict(ck["ema"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        _restore_rng_state(ck["rng_state"])
        global_step = int(ck["global_step"])
        history = list(ck.get("training_history") or [])
        skipped = int(ck.get("skipped_empty_train_batches", 0))
        print(
            f"resumed v9 full M from epoch={start_epoch} step={global_step} "
            f"to epoch={a.run_until_epoch}"
        )
    else:
        initial_val = validate_fm(
            ema.model,
            val_loader,
            device,
            use_amp=use_amp,
            seed=a.seed + 100000,
        )
        history.append({"epoch": 0, "global_step": 0, "train": None, "val_ema": initial_val})
        print("initial_validation", json.dumps(history[-1]))

    total_schedule_steps = int(schedule["total_schedule_steps"])
    model.train()
    for epoch in range(start_epoch + 1, int(a.run_until_epoch) + 1):
        indices = scene_balanced_epoch_indices(train_ds, seed=a.seed, epoch=epoch)
        sampled_ids = [str(train_ds.entries[i]["sample_id"]) for i in indices]
        sampled_scenes = [str(train_ds.entries[i]["scene_name"]) for i in indices]
        loader = DataLoader(
            train_ds,
            batch_size=a.batch_size,
            sampler=FixedIndexSampler(indices),
            num_workers=a.num_workers,
            collate_fn=collate_msp_wm,
            drop_last=False,
            pin_memory=True,
        )
        if len(loader) != int(schedule["steps_per_epoch"]):
            raise RuntimeError("epoch loader length differs from frozen schedule shape")

        epoch_obj_sum = 0.0
        epoch_uniform_sum = 0.0
        epoch_cos_sum = 0.0
        epoch_valid_batches = 0
        epoch_motion_sum = 0.0
        last_train = None

        for batch in loader:
            step_for_lr = int(global_step)
            lr_ratio = _lr_ratio(
                step_for_lr,
                total_schedule_steps,
                float(a.warmup_fraction),
                float(a.min_lr_ratio),
            )
            lrs = _set_lr(optimizer, lr_ratio)
            prepared = prepare_batch(batch, device)
            global_step += 1
            if prepared is None:
                skipped += 1
                continue

            motion_mask = _crop_motion_mask(sidecar, batch, prepared, device)
            global_noise = torch.randn_like(prepared["physics_full"])
            source_noise = crop_coherent_source_noise(
                global_noise, prepared["plan"], prepared["effective"]
            )

            optimizer.zero_grad(set_to_none=True)
            fm_loss, info, squared_error = _flow_and_squared_error(
                model,
                prepared,
                source_noise,
                None,
                use_amp=use_amp,
            )
            total_loss = normalized_motion_weighted_mse(
                squared_error,
                motion_mask,
                float(a.motion_weight_lambda),
            )
            if not torch.isfinite(total_loss):
                raise RuntimeError(
                    f"non-finite v9 full M loss epoch={epoch} step={global_step}: {total_loss}"
                )
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(a.grad_clip))
            optimizer.step()
            ema_decay_now = ema.update(model)

            motion_fraction = float(motion_mask.float().mean().detach().cpu())
            epoch_obj_sum += float(total_loss.detach().cpu())
            epoch_uniform_sum += float(fm_loss.detach().cpu())
            epoch_cos_sum += float(info["cosine"])
            epoch_motion_sum += motion_fraction
            epoch_valid_batches += 1
            last_train = {
                "objective": float(total_loss.detach().cpu()),
                "uniform_fm_loss": float(fm_loss.detach().cpu()),
                "fm_cosine": float(info["cosine"]),
                "pred_rms": float(info["pred_rms"]),
                "target_rms": float(info["target_rms"]),
                "motion_cell_fraction_batch": motion_fraction,
                "effective_motion_weight_mass_batch": effective_motion_weight_mass(
                    motion_fraction, float(a.motion_weight_lambda)
                ),
                "physics_authority": float(model.transition.physics_fusion.authority.cpu()),
                "grad_norm_before_clip": float(torch.as_tensor(grad_norm).detach().cpu()),
                "ema_decay": float(ema_decay_now),
                "lr_ratio": float(lr_ratio),
                "lrs": lrs,
            }
            if global_step == 1 or global_step % 100 == 0:
                print(
                    f"epoch={epoch} step={global_step}/{total_schedule_steps} "
                    f"obj={last_train['objective']:.6f} uniform={last_train['uniform_fm_loss']:.6f} "
                    f"cos={last_train['fm_cosine']:+.4f} motion={100*motion_fraction:.3f}% "
                    f"phys={last_train['physics_authority']:+.5f} "
                    f"grad={last_train['grad_norm_before_clip']:.4f} lr={lrs}"
                )

        expected_step = epoch * int(schedule["steps_per_epoch"])
        if global_step != expected_step:
            raise RuntimeError(
                f"global step/epoch mismatch: {global_step} != {expected_step}"
            )
        if epoch_valid_batches <= 0:
            raise RuntimeError("full-data epoch contained no valid routed training batches")

        val = validate_fm(
            ema.model,
            val_loader,
            device,
            use_amp=use_amp,
            seed=a.seed + 100000,
        )
        epoch_train = {
            "mean_weighted_objective": epoch_obj_sum / epoch_valid_batches,
            "mean_uniform_fm_loss": epoch_uniform_sum / epoch_valid_batches,
            "mean_fm_cosine": epoch_cos_sum / epoch_valid_batches,
            "mean_motion_cell_fraction": epoch_motion_sum / epoch_valid_batches,
            "valid_batches": epoch_valid_batches,
            "sampled_draws": len(indices),
            "unique_sample_ids": len(set(sampled_ids)),
            "unique_scenes": len(set(sampled_scenes)),
            "last_batch": last_train,
        }
        row = {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "train": epoch_train,
            "val_ema": val,
        }
        history.append(row)
        print("epoch_validation", json.dumps(row))

        payload = _payload(
            model,
            ema,
            optimizer,
            epoch=epoch,
            global_step=global_step,
            history=history,
            a=a,
            train_ds=train_ds,
            val_ds=val_ds,
            sidecar=sidecar,
            reuse=reuse,
            optimizer_info=optimizer_info,
            vae_sha=vae_sha,
            schedule=schedule,
            skipped=skipped,
        )
        _save_payload(out, "latest.pt", payload)
        if epoch in milestones:
            _save_payload(out, f"epoch_{epoch:04d}.pt", payload)
            print("saved_milestone", out / f"epoch_{epoch:04d}.pt")

        report = {
            "protocol": PROTOCOL,
            "initialization": "released_occfm_fut_checkpoint",
            "objective": "normalized_true_motion_weighted_native_flow_matching_velocity_mse",
            "motion_lambda": float(a.motion_weight_lambda),
            "full_train_samples": len(train_ds),
            "full_train_scenes": int(train_ds.metadata.get("num_unique_scenes", 0)),
            "train_routed_motion_fraction": sidecar.metadata.get("routed_top2_motion_fraction"),
            "effective_train_motion_weight_mass": effective_motion_weight_mass(
                float(sidecar.metadata.get("routed_top2_motion_fraction", 0.0)),
                float(a.motion_weight_lambda),
            ),
            "schedule": schedule,
            "run_until_epoch": int(a.run_until_epoch),
            "milestone_epochs": milestones,
            "official_transition_reuse_fraction": official_reuse_fraction,
            "optimizer_contract": optimizer_info,
            "history": history,
            "skipped_empty_train_batches": skipped,
            "checkpoint_policy": {
                "milestones": [f"epoch_{x:04d}.pt" for x in milestones if x <= epoch],
                "latest": True,
                "last": epoch == int(a.run_until_epoch),
                "best": False,
                "reason_no_best": "fixed-t FM does not select deployment Overall/Moving",
            },
            "decision": (
                "Run the unchanged NFE=10 deployment diagnostic at epoch 1/3/5. "
                "Continue to epoch10 only if deployment/physical metrics still improve; "
                "do not infer success from weighted or uniform FM loss alone."
            ),
        }
        (out / "training_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )

    final = torch.load(out / "latest.pt", map_location="cpu", weights_only=False)
    _save_payload(out, "last.pt", final)
    print("full_data_phase_complete", out / "last.pt")


if __name__ == "__main__":
    main()
