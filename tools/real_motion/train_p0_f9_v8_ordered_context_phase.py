#!/usr/bin/env python3
"""Small-budget M-control vs M+T ordered-history context experiment.

Both branches start from the exact same v8 M phase-400 EMA checkpoint and use
the same normalized true-motion-weighted native FM objective (lambda=2), data,
scene-balanced step-indexed batches, Gaussian source noise, sampled t, optimizer
LRs, EMA, physics condition, mean context path and deployment contract.

The only candidate change is an additional ordered-history residual:
    C = C_mean + Conv3x3_stride2(concat(H0,...,H5))
The ordered convolution is zero initialized.  MCTRL keeps it disabled/frozen;
MT enables/trains it.  Therefore both variants are functionally identical at
phase step 0.

Experimental checkpoints are milestone-only full resume states.  No
step_0000/best/latest/last files are written.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
UP = ROOT / "upstream_occfm"
sys.path[:0] = [str(ROOT), str(UP)]

import torch
from torch.utils.data import DataLoader

from real_motion.checkpoint import load_shape_safe, require_checkpoint_reuse
from real_motion.model_ema import ModelEMA
from real_motion.models.p0_f9 import P0_F9_PROTOCOL, make_p0_f9_model
from real_motion.models.transition_ordered_context import (
    ORDERED_CONTEXT_PROTOCOL,
    ordered_context_is_exact_zero,
)
from real_motion.motion_mask_sidecar import (
    MOTION_MASK_PROTOCOL,
    MotionMaskSidecar,
    effective_motion_weight_mass,
    normalized_motion_weighted_mse,
)
from real_motion.msp_wm_cache import MSPWorldModelCacheDataset, collate_msp_wm
from real_motion.occfm_io import file_sha256
from tools.real_motion.train_p0_f9_native_sparse_forecast import (
    HIST_LAST,
    _build_optimizer,
    _cpu_state,
    _ema_payload,
    _seed_all,
    _validate_cache_pair,
    prepare_batch,
)
from tools.real_motion.train_p0_f9_v7_native_fm_only import (
    _validate_vae_provenance,
    validate_fm,
)
from tools.real_motion.train_p0_f9_v8_um_phase import (
    LR_PROTOCOL,
    RANDOM_PROTOCOL,
    SAMPLING_PROTOCOL,
    SOURCE_SPATIAL_CONTRACT,
    PROTOCOL as UM_PROTOCOL,
    PhaseStepBatchSampler,
    _crop_motion_mask,
    _flow_and_squared_error,
    _restore_rng_state,
    _rng_state,
    _step_noise_and_t,
    _validate_motion_sidecar,
)


PROTOCOL = "p0_f9_v8_ordered_context_phase_v1"
VARIANTS = ("MCTRL", "MT")
ORDERED_STATE_KEYS = {
    "transition.ordered_context_proj.weight",
    "transition.ordered_context_proj.bias",
}


def _validate_parent(ck: dict, a, train_ds, val_ds, vae_sha: str) -> dict:
    arch = ck.get("architecture", {})
    if arch.get("protocol") != P0_F9_PROTOCOL or int(arch.get("stage", -1)) != 1:
        raise RuntimeError("parent is not audited P0-F9 Stage-1")
    if arch.get("training_protocol") != UM_PROTOCOL or arch.get("phase_variant") != "M":
        raise RuntimeError("ordered-context experiment requires the v8 M branch parent")
    if arch.get("training_objective") != "normalized_true_motion_weighted_native_flow_matching_velocity_mse":
        raise RuntimeError("parent objective is not normalized true-motion-weighted native FM")
    if float(arch.get("motion_lambda_applied", -1.0)) != float(a.motion_weight_lambda):
        raise RuntimeError("parent motion lambda differs from ordered-context phase")
    step = int(ck.get("phase_step", ck.get("step", -1)))
    if step != int(a.parent_step):
        raise RuntimeError(f"ordered-context parent step must be {a.parent_step}, got {step}")
    if not isinstance(ck.get("ema"), dict) or "state_dict" not in ck["ema"]:
        raise RuntimeError("parent checkpoint lacks EMA state")
    if ck.get("train_cache_index_sha256") != file_sha256(train_ds.root / "index.json"):
        raise RuntimeError("parent/train cache mismatch")
    if ck.get("val_cache_index_sha256") != file_sha256(val_ds.root / "index.json"):
        raise RuntimeError("parent/val cache mismatch")
    if ck.get("upstream_checkpoint_sha256") != file_sha256(a.upstream_ckpt):
        raise RuntimeError("parent/upstream OccFM checkpoint mismatch")
    if ck.get("vae_checkpoint_sha256") != vae_sha:
        raise RuntimeError("parent/VAE provenance mismatch")
    if int(arch.get("native_backbone_hist_last", -1)) != HIST_LAST:
        raise RuntimeError("parent HIST_LAST mismatch")
    if arch.get("flow_source_spatial_contract") != SOURCE_SPATIAL_CONTRACT:
        raise RuntimeError("parent source-noise contract mismatch")
    if bool(arch.get("semantic_auxiliary", True)) or bool(arch.get("vae_decoder_in_training_graph", True)):
        raise RuntimeError("parent unexpectedly used semantic/decoder supervision")
    return arch


def _lift_parent_ema(model, parent: dict) -> None:
    parent_state = parent["ema"]["state_dict"]
    state = model.state_dict()
    missing = set(state) - set(parent_state)
    extra = set(parent_state) - set(state)
    if missing != ORDERED_STATE_KEYS or extra:
        raise RuntimeError(
            f"ordered-context parent lift mismatch: missing={sorted(missing)} extra={sorted(extra)}"
        )
    for key, value in parent_state.items():
        state[key] = value
    model.load_state_dict(state, strict=True)
    if not ordered_context_is_exact_zero(model.transition):
        raise RuntimeError("new ordered-context residual is not exact-zero after parent lift")


def _ordered_stats(model) -> dict:
    tr = model.transition
    w = tr.ordered_context_proj.weight.detach().float()
    b = tr.ordered_context_proj.bias.detach().float()
    return {
        "enabled": bool(tr.ordered_context_enabled),
        "trainable_parameters": int(tr.ordered_context_trainable_parameters),
        "weight_rms": float(w.square().mean().sqrt().cpu()),
        "bias_rms": float(b.square().mean().sqrt().cpu()),
        "weight_max_abs": float(w.abs().max().cpu()),
    }


def _architecture(a) -> dict:
    enabled = a.variant == "MT"
    return {
        "protocol": P0_F9_PROTOCOL,
        "stage": 1,
        "training_protocol": PROTOCOL,
        "phase_variant": a.variant,
        "parent_training_protocol": UM_PROTOCOL,
        "parent_variant": "M",
        "parent_step": int(a.parent_step),
        "parent_weight_source": "ema",
        "window_hw": [20, 20],
        "context_hw": [40, 40],
        "topk": 2,
        "future_frames": 6,
        "native_backbone_hist_last": HIST_LAST,
        "flow": "gaussian_noise_to_absolute_gt_future",
        "flow_source_spatial_contract": SOURCE_SPATIAL_CONTRACT,
        "physics_prior": "strong_w2det_condition_and_fallback_not_flow_source",
        "physics_fusion": "unchanged_from_M400",
        "history_context": (
            "mean_context_plus_zero_init_ordered_history_residual"
            if enabled else "mean_context_only_ordered_module_dormant"
        ),
        "ordered_context_module": True,
        "ordered_context_enabled": enabled,
        "ordered_context_protocol": ORDERED_CONTEXT_PROTOCOL,
        "ordered_history_frames": 6,
        "ordered_context_in_channels": 96,
        "ordered_context_out_channels": 128,
        "ordered_context_kernel": 3,
        "ordered_context_stride": 2,
        "ordered_context_zero_initialized": True,
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
        "sampling_protocol": SAMPLING_PROTOCOL,
        "random_protocol": RANDOM_PROTOCOL,
        "learning_rate_protocol": LR_PROTOCOL,
        "constant_wm_lr": float(a.wm_lr),
        "constant_new_lr": float(a.new_lr),
    }


def _payload(model, ema, optimizer, *, phase_step, history, a, train_ds, val_ds,
             sidecar, reuse, optimizer_info, vae_sha, parent_sha) -> dict:
    return {
        "state_dict": _cpu_state(model),
        "ema": _ema_payload(ema),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": int(phase_step),
        "phase_step": int(phase_step),
        "training_history": list(history),
        "architecture": _architecture(a),
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
        "parent_checkpoint": str(Path(a.parent_checkpoint).resolve()),
        "parent_checkpoint_sha256": parent_sha,
        "parent_step": int(a.parent_step),
        "parent_weight_source": "ema",
        "upstream_reuse": reuse,
        "optimizer_contract": optimizer_info,
        "rng_state": _rng_state(),
        "sampling_progress": {
            "protocol": SAMPLING_PROTOCOL,
            "completed_phase_steps": int(phase_step),
            "next_phase_step": int(phase_step),
            "base_seed": int(a.seed),
        },
        "phase_limit": int(a.phase_limit),
        "ordered_context_stats": _ordered_stats(model),
        "args": vars(a),
    }


def _validate_resume(ck, a, train_ds, val_ds, sidecar, vae_sha, parent_sha) -> int:
    arch = ck.get("architecture", {})
    if arch.get("training_protocol") != PROTOCOL or arch.get("phase_variant") != a.variant:
        raise RuntimeError("resume checkpoint protocol/variant differs")
    if bool(arch.get("ordered_context_module", False)) is not True:
        raise RuntimeError("resume checkpoint lacks ordered-context module")
    if bool(arch.get("ordered_context_enabled", False)) != (a.variant == "MT"):
        raise RuntimeError("resume ordered-context enabled state differs")
    if float(arch.get("motion_lambda_applied", -1.0)) != float(a.motion_weight_lambda):
        raise RuntimeError("resume motion lambda differs")
    if ck.get("train_cache_index_sha256") != file_sha256(train_ds.root / "index.json"):
        raise RuntimeError("resume train cache differs")
    if ck.get("val_cache_index_sha256") != file_sha256(val_ds.root / "index.json"):
        raise RuntimeError("resume val cache differs")
    if ck.get("motion_mask_sidecar_sha256") != file_sha256(sidecar.path):
        raise RuntimeError("resume motion mask differs")
    if ck.get("upstream_checkpoint_sha256") != file_sha256(a.upstream_ckpt):
        raise RuntimeError("resume upstream differs")
    if ck.get("vae_checkpoint_sha256") != vae_sha:
        raise RuntimeError("resume VAE differs")
    if ck.get("parent_checkpoint_sha256") != parent_sha:
        raise RuntimeError("resume parent differs")
    if int(ck.get("phase_limit", -1)) != int(a.phase_limit):
        raise RuntimeError("resume phase limit differs")
    saved = ck.get("args") or {}
    for key in ("batch_size", "sample_steps", "seed", "parent_step"):
        if int(saved.get(key, -1)) != int(getattr(a, key)):
            raise RuntimeError(f"resume argument differs for {key}")
    for key in ("wm_lr", "new_lr", "weight_decay", "uncond_prob", "guidance_scale",
                "ema_decay", "grad_clip", "motion_weight_lambda"):
        if float(saved.get(key, float("nan"))) != float(getattr(a, key)):
            raise RuntimeError(f"resume argument differs for {key}")
    step = int(ck.get("phase_step", ck.get("step", -1)))
    if step < 0 or step >= int(a.phase_end_step):
        raise RuntimeError(f"resume step {step} incompatible with end {a.phase_end_step}")
    return step


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=VARIANTS, required=True)
    p.add_argument("--train-cache", required=True)
    p.add_argument("--val-cache", required=True)
    p.add_argument("--motion-mask-sidecar", required=True)
    p.add_argument("--parent-checkpoint", required=True)
    p.add_argument("--parent-step", type=int, default=400)
    p.add_argument("--upstream-ckpt", required=True)
    p.add_argument("--vae-ckpt", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--phase-end-step", type=int, default=200)
    p.add_argument("--phase-limit", type=int, default=400)
    p.add_argument("--milestones", type=int, nargs="+", default=[100, 200])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--wm-lr", type=float, default=4e-6)
    p.add_argument("--new-lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--sample-steps", type=int, default=10)
    p.add_argument("--uncond-prob", type=float, default=0.0)
    p.add_argument("--guidance-scale", type=float, default=1.0)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--motion-weight-lambda", type=float, default=2.0)
    p.add_argument("--min-train-windows", type=int, default=4000)
    p.add_argument("--seed", type=int, default=20260904)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--resume-from", default=None)
    a = p.parse_args()

    if min(a.phase_end_step, a.phase_limit, a.batch_size, a.min_train_windows) <= 0:
        raise ValueError("phase/batch/min-train settings must be positive")
    if a.phase_end_step > a.phase_limit:
        raise ValueError("phase-end-step cannot exceed phase-limit")
    milestones = sorted(set(int(x) for x in a.milestones))
    if not milestones or any(x <= 0 or x > a.phase_limit for x in milestones):
        raise ValueError("invalid milestones")
    if int(a.phase_end_step) not in milestones:
        raise ValueError("phase-end-step must be a saved milestone")
    if a.motion_weight_lambda < 0 or a.grad_clip <= 0:
        raise ValueError("invalid motion lambda / grad clip")

    _seed_all(a.seed)
    device = torch.device(a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("ordered-context phase requires CUDA")
    use_amp = bool(a.amp)

    train_ds = MSPWorldModelCacheDataset(a.train_cache)
    val_ds = MSPWorldModelCacheDataset(a.val_cache)
    _validate_cache_pair(train_ds, val_ds, min_train_windows=a.min_train_windows)
    vae_sha = _validate_vae_provenance(train_ds, val_ds, a.vae_ckpt)
    sidecar = MotionMaskSidecar(a.motion_mask_sidecar)
    _validate_motion_sidecar(sidecar, train_ds)

    parent = torch.load(a.parent_checkpoint, map_location="cpu", weights_only=False)
    _validate_parent(parent, a, train_ds, val_ds, vae_sha)
    parent_sha = file_sha256(a.parent_checkpoint)

    enabled = a.variant == "MT"
    model = make_p0_f9_model(
        20,
        sample_steps=a.sample_steps,
        unconditional_probability=a.uncond_prob,
        guidance_scale=a.guidance_scale,
        hist_last=HIST_LAST,
        ordered_context=True,
        ordered_context_enabled=enabled,
    ).to(device)
    reuse = load_shape_safe(model.transition, a.upstream_ckpt, verbose=True)
    if "traj_encoder.0.weight" not in set(reuse.get("loaded_keys", ())):
        raise RuntimeError("ordered-context phase requires official OccFM-Fut epoch=000196")
    official_reuse_fraction = require_checkpoint_reuse(reuse, min_fraction=0.80)
    _lift_parent_ema(model, parent)

    if not enabled:
        model.transition.ordered_context_proj.requires_grad_(False)
        model.transition.set_ordered_context_enabled(False)
    else:
        model.transition.ordered_context_proj.requires_grad_(True)
        model.transition.set_ordered_context_enabled(True)

    optimizer, optimizer_info = _build_optimizer(
        model, reuse, wm_lr=a.wm_lr, new_lr=a.new_lr, weight_decay=a.weight_decay
    )
    ema = ModelEMA(model, decay=a.ema_decay, updates=0)

    val_loader = DataLoader(
        val_ds,
        batch_size=a.batch_size,
        shuffle=False,
        num_workers=a.num_workers,
        collate_fn=collate_msp_wm,
        drop_last=False,
        pin_memory=True,
    )
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    phase_step = 0
    history = []
    if a.resume_from:
        ck = torch.load(a.resume_from, map_location="cpu", weights_only=False)
        phase_step = _validate_resume(ck, a, train_ds, val_ds, sidecar, vae_sha, parent_sha)
        model.load_state_dict(ck["state_dict"], strict=True)
        ema.load_state_dict(ck["ema"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        _restore_rng_state(ck["rng_state"])
        history = list(ck.get("training_history") or [])
        print(f"resumed {a.variant} ordered-context phase step={phase_step}->{a.phase_end_step}")
    else:
        initial_val = validate_fm(
            ema.model, val_loader, device, use_amp=use_amp, seed=a.seed + 100000
        )
        initial = {
            "phase_step": 0,
            "train": None,
            "val_ema": initial_val,
            "ordered_context": _ordered_stats(ema.model),
        }
        history.append(initial)
        print("initial_validation", json.dumps(initial))

    batch_sampler = PhaseStepBatchSampler(
        train_ds,
        batch_size=a.batch_size,
        start_step=phase_step,
        end_step=a.phase_end_step,
        seed=a.seed,
    )
    train_loader = DataLoader(
        train_ds,
        batch_sampler=batch_sampler,
        num_workers=a.num_workers,
        collate_fn=collate_msp_wm,
        pin_memory=True,
    )

    model.train()
    for current_step, batch in enumerate(train_loader, start=phase_step):
        if current_step != phase_step:
            raise RuntimeError("phase/data-loader progress desynchronized")
        prepared = prepare_batch(batch, device)
        if prepared is None:
            raise RuntimeError("deterministic ordered-context batch has no routed windows")
        motion_mask = _crop_motion_mask(sidecar, batch, prepared, device)
        source_noise, t_override = _step_noise_and_t(
            prepared, seed=a.seed, phase_step=phase_step
        )

        optimizer.zero_grad(set_to_none=True)
        fm_loss, info, squared_error = _flow_and_squared_error(
            model, prepared, source_noise, t_override, use_amp=use_amp
        )
        total_loss = normalized_motion_weighted_mse(
            squared_error, motion_mask, float(a.motion_weight_lambda)
        )
        if not torch.isfinite(total_loss):
            raise RuntimeError(f"non-finite {a.variant} loss at phase_step={phase_step}")
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(a.grad_clip))
        optimizer.step()
        ema_decay_now = ema.update(model)
        phase_step += 1

        motion_fraction = float(motion_mask.float().mean().detach().cpu())
        train_info = {
            "objective": float(total_loss.detach().cpu()),
            "uniform_fm_loss": float(fm_loss.detach().cpu()),
            "fm_cosine": float(info["cosine"]),
            "pred_rms": float(info["pred_rms"]),
            "target_rms": float(info["target_rms"]),
            "motion_cell_fraction_batch": motion_fraction,
            "effective_motion_weight_mass_batch": effective_motion_weight_mass(
                motion_fraction, float(a.motion_weight_lambda)
            ),
            "motion_lambda_applied": float(a.motion_weight_lambda),
            "physics_authority": float(model.transition.physics_fusion.authority.cpu()),
            "ordered_context": _ordered_stats(model),
            "grad_norm_before_clip": float(torch.as_tensor(grad_norm).detach().cpu()),
            "ema_decay": float(ema_decay_now),
            "lrs": {
                str(g.get("group_name", i)): float(g["lr"])
                for i, g in enumerate(optimizer.param_groups)
            },
            "sampling_phase_step": int(phase_step - 1),
        }
        if phase_step == 1 or phase_step % 20 == 0:
            oc = train_info["ordered_context"]
            print(
                f"variant={a.variant} phase_step={phase_step} obj={train_info['objective']:.6f} "
                f"uniform={train_info['uniform_fm_loss']:.6f} cos={train_info['fm_cosine']:+.4f} "
                f"ordered_w_rms={oc['weight_rms']:.6g} ordered_max={oc['weight_max_abs']:.6g} "
                f"grad={train_info['grad_norm_before_clip']:.4f} lrs={train_info['lrs']}"
            )

        if phase_step in milestones:
            val = validate_fm(
                ema.model, val_loader, device, use_amp=use_amp, seed=a.seed + 100000
            )
            row = {
                "phase_step": phase_step,
                "train": train_info,
                "val_ema": val,
                "ordered_context_ema": _ordered_stats(ema.model),
            }
            history.append(row)
            print("validation", json.dumps(row))
            payload = _payload(
                model,
                ema,
                optimizer,
                phase_step=phase_step,
                history=history,
                a=a,
                train_ds=train_ds,
                val_ds=val_ds,
                sidecar=sidecar,
                reuse=reuse,
                optimizer_info=optimizer_info,
                vae_sha=vae_sha,
                parent_sha=parent_sha,
            )
            ck_path = out / f"step_{phase_step:04d}.pt"
            torch.save(payload, ck_path)
            print("saved_milestone", ck_path)

    if phase_step != int(a.phase_end_step):
        raise RuntimeError(f"phase ended at {phase_step}, expected {a.phase_end_step}")
    final_ck = out / f"step_{phase_step:04d}.pt"
    if not final_ck.exists():
        raise RuntimeError("phase endpoint must be a saved milestone")

    report = {
        "protocol": PROTOCOL,
        "variant": a.variant,
        "parent_checkpoint": str(Path(a.parent_checkpoint).resolve()),
        "parent_checkpoint_sha256": parent_sha,
        "parent_step": int(a.parent_step),
        "parent_weight_source": "ema",
        "phase_end_step": int(phase_step),
        "phase_limit": int(a.phase_limit),
        "milestones": milestones,
        "objective": "normalized_true_motion_weighted_native_flow_matching_velocity_mse",
        "ordered_context_protocol": ORDERED_CONTEXT_PROTOCOL,
        "ordered_context_enabled": enabled,
        "ordered_context_final": _ordered_stats(model),
        "motion_mask_sidecar_sha256": file_sha256(sidecar.path),
        "train_routed_motion_fraction": sidecar.metadata.get("routed_top2_motion_fraction"),
        "motion_lambda": float(a.motion_weight_lambda),
        "effective_train_motion_weight_mass": effective_motion_weight_mass(
            float(sidecar.metadata.get("routed_top2_motion_fraction", 0.0)),
            float(a.motion_weight_lambda),
        ),
        "learning_rate_protocol": LR_PROTOCOL,
        "optimizer_contract": optimizer_info,
        "official_transition_reuse_fraction": official_reuse_fraction,
        "history": history,
        "checkpoint_policy": {
            "saved": [f"step_{x:04d}.pt" for x in milestones if x <= phase_step],
            "step_0000": False,
            "best": False,
            "latest": False,
            "last": False,
            "milestones_are_full_resume_state": True,
        },
        "decision": (
            "Compare MT against MCTRL at identical phase budget with the ordered-compatible "
            "deployment diagnostic. A useful ordered context should improve deployment Moving/Overall "
            "or WRITE precision/stale behavior, not merely increase dynamic volume."
        ),
    }
    (out / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("phase_complete", final_ck)


if __name__ == "__main__":
    main()
