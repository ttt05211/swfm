#!/usr/bin/env python3
"""Controlled P0-F9 v8 U/M phase training from the v7 step400 EMA parent.

U and M are intentionally identical except for one operation: the final native
flow-matching MSE reduction.

U: uniform native FM MSE.
M: normalized motion-weighted native FM MSE,
       w_i = 1 + lambda * motion_mask_i,
       L = sum(w_i * e_i^2) / sum(w_i), lambda=2 by default.

Architecture, physics/context conditions, train cache, scene-balanced sampling,
per-step Gaussian source, sampled FM time, optimizer groups/LRs, EMA, AMP and
all deployment contracts stay fixed.  The parent model is the v7 step400 EMA.
A fresh phase optimizer and fresh phase EMA are created at phase step 0.

Experimental checkpoint policy: only requested milestone files (default 200 and
400) are written.  There are deliberately no step_0000/best/latest/last files.
Each milestone contains the complete optimizer/EMA/RNG/sampling state so a later
400->800 continuation resumes the exact phase rather than restarting it.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
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
from real_motion.windows import crop_windows
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
    PROTOCOL as V7_PROTOCOL,
    _validate_vae_provenance,
    validate_fm,
)


PROTOCOL = "p0_f9_v8_um_phase_v1"
SOURCE_SPATIAL_CONTRACT = "one_global_gaussian_field_cropped_into_top2_windows"
SAMPLING_PROTOCOL = "scene_balanced_step_indexed_multinomial_v1"
RANDOM_PROTOCOL = "step_indexed_independent_batch_noise_t_streams_v1"
LR_PROTOCOL = "constant_phase_lr_from_v7_terminal_lr_no_cosine_restart_v1"
VARIANTS = ("U", "M")


def _phase_seed(base_seed: int, phase_step: int, stream: str) -> int:
    return deterministic_sample_seed(
        f"phase_step:{int(phase_step)}", int(base_seed), stream=f"v8_um_{stream}"
    )


class PhaseStepBatchSampler(Sampler[list[int]]):
    """Deterministic scene-balanced batches indexed directly by phase step.

    Each step owns an independent multinomial seed.  Therefore worker prefetch
    cannot change the future sample sequence, and a resume at step S starts at
    exactly the same batch S that an uninterrupted run would use.
    """

    def __init__(self, ds, *, batch_size: int, start_step: int, end_step: int, seed: int):
        self.batch_size = int(batch_size)
        self.start_step = int(start_step)
        self.end_step = int(end_step)
        self.seed = int(seed)
        if self.batch_size <= 0 or self.start_step < 0 or self.end_step < self.start_step:
            raise ValueError("invalid phase batch sampler arguments")
        scenes = [str(e["scene_name"]) for e in ds.entries]
        counts = Counter(scenes)
        self.weights = torch.tensor(
            [1.0 / float(counts[s]) for s in scenes], dtype=torch.double
        )

    def __iter__(self):
        for step in range(self.start_step, self.end_step):
            gen = torch.Generator(device="cpu")
            gen.manual_seed(_phase_seed(self.seed, step, "batch"))
            indices = torch.multinomial(
                self.weights,
                self.batch_size,
                replacement=True,
                generator=gen,
            )
            yield indices.tolist()

    def __len__(self):
        return self.end_step - self.start_step


def _rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(obj: dict) -> None:
    if not isinstance(obj, dict):
        raise RuntimeError("checkpoint lacks RNG state")
    random.setstate(obj["python"])
    np.random.set_state(obj["numpy"])
    torch.set_rng_state(obj["torch_cpu"])
    if torch.cuda.is_available() and obj.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(obj["torch_cuda"])


def _validate_motion_sidecar(sidecar: MotionMaskSidecar, train_ds) -> None:
    ids = [str(e["sample_id"]) for e in train_ds.entries]
    sidecar.validate_sample_ids(ids)
    meta = sidecar.metadata
    if meta.get("source_train_cache_index_sha256") != file_sha256(train_ds.root / "index.json"):
        raise RuntimeError("motion-mask sidecar was not built from this exact train cache")
    if meta.get("motion_mask_protocol") != MOTION_MASK_PROTOCOL:
        raise RuntimeError("motion-mask sidecar definition differs from v8 diagnostic")


def _validate_parent(ck: dict, a, train_ds, val_ds, vae_sha: str) -> dict:
    arch = ck.get("architecture", {})
    if arch.get("protocol") != P0_F9_PROTOCOL or int(arch.get("stage", -1)) != 1:
        raise RuntimeError("parent checkpoint is not audited P0-F9 Stage-1")
    if arch.get("training_protocol") != V7_PROTOCOL:
        raise RuntimeError("parent checkpoint is not the v7 native-FM-only control")
    if arch.get("training_objective") != "native_flow_matching_velocity_mse_only":
        raise RuntimeError("parent checkpoint did not use native FM-only")
    if bool(arch.get("semantic_auxiliary", True)) or bool(
        arch.get("vae_decoder_in_training_graph", True)
    ):
        raise RuntimeError("v8 U/M parent unexpectedly used semantic/decoder supervision")
    if int(ck.get("step", -1)) != int(a.parent_step):
        raise RuntimeError(
            f"v8 U/M requires parent step {a.parent_step}, got {ck.get('step')}"
        )
    if not isinstance(ck.get("ema"), dict) or "state_dict" not in ck["ema"]:
        raise RuntimeError("parent checkpoint lacks EMA state")
    if ck.get("train_cache_index_sha256") != file_sha256(train_ds.root / "index.json"):
        raise RuntimeError("parent/train cache mismatch")
    if ck.get("val_cache_index_sha256") != file_sha256(val_ds.root / "index.json"):
        raise RuntimeError("parent/val cache mismatch")
    if ck.get("upstream_checkpoint_sha256") != file_sha256(a.upstream_ckpt):
        raise RuntimeError("parent/upstream OccFM checkpoint mismatch")
    if ck.get("vae_checkpoint_sha256") != vae_sha:
        raise RuntimeError("parent/VAE latent provenance mismatch")
    if int(arch.get("native_backbone_hist_last", -1)) != HIST_LAST:
        raise RuntimeError("parent HIST_LAST contract mismatch")
    if arch.get("flow_source_spatial_contract") != SOURCE_SPATIAL_CONTRACT:
        raise RuntimeError("parent coherent-noise contract mismatch")
    return arch


def _architecture(a) -> dict:
    applied_lambda = 0.0 if a.variant == "U" else float(a.motion_weight_lambda)
    objective = (
        "native_flow_matching_velocity_mse_only"
        if a.variant == "U"
        else "normalized_true_motion_weighted_native_flow_matching_velocity_mse"
    )
    return {
        "protocol": P0_F9_PROTOCOL,
        "stage": 1,
        "training_protocol": PROTOCOL,
        "phase_variant": a.variant,
        "parent_training_protocol": V7_PROTOCOL,
        "parent_step": int(a.parent_step),
        "parent_weight_source": "ema",
        "window_hw": [20, 20],
        "context_hw": [40, 40],
        "topk": 2,
        "future_frames": 6,
        "native_backbone_hist_last": HIST_LAST,
        "flow": "gaussian_noise_to_absolute_gt_future",
        "flow_source_spatial_contract": SOURCE_SPATIAL_CONTRACT,
        "latent_distribution": "deterministic_posterior_sample_matching_occfm_cache",
        "physics_prior": "strong_w2det_condition_and_fallback_not_flow_source",
        "physics_fusion": "unchanged_from_v7",
        "history_context": "unchanged_v7_mean_context_branch",
        "training_objective": objective,
        "uniform_loss_formula": "mean((predicted_velocity-(target-source))**2)",
        "weighted_loss_formula": "sum((1+lambda*m)*squared_error)/sum(1+lambda*m)",
        "motion_mask_protocol": MOTION_MASK_PROTOCOL,
        "motion_lambda_configured": float(a.motion_weight_lambda),
        "motion_lambda_applied": applied_lambda,
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


def _crop_motion_mask(sidecar, batch, prepared, device) -> torch.Tensor:
    full = sidecar.get_batch(batch["sample_id"], device=device)  # [B,T,50,50]
    full = full[:, :, None].float()
    windows = crop_windows(full, prepared["plan"])
    B = int(prepared["batch_size"])
    K = int(prepared["topk"])
    flat = windows.reshape(B * K, *windows.shape[2:])[prepared["effective"]]
    mask = flat[:, :, 0].bool()
    expected = (
        int(prepared["target"].shape[0]),
        int(prepared["target"].shape[1]),
        int(prepared["target"].shape[-2]),
        int(prepared["target"].shape[-1]),
    )
    if tuple(mask.shape) != expected:
        raise RuntimeError(f"cropped motion mask shape {tuple(mask.shape)} != {expected}")
    return mask


def _step_noise_and_t(prepared, *, seed: int, phase_step: int):
    device = prepared["physics_full"].device
    noise_gen = torch.Generator(device=device)
    noise_gen.manual_seed(_phase_seed(seed, phase_step, "noise"))
    global_noise = torch.randn(
        prepared["physics_full"].shape,
        device=device,
        dtype=prepared["physics_full"].dtype,
        generator=noise_gen,
    )
    source_noise = crop_coherent_source_noise(
        global_noise, prepared["plan"], prepared["effective"]
    )
    t_gen = torch.Generator(device=device)
    t_gen.manual_seed(_phase_seed(seed, phase_step, "t"))
    nwin = int(prepared["history"].shape[0])
    t = torch.sigmoid(
        torch.randn(
            (nwin, 1, 1, 1, 1),
            device=device,
            dtype=prepared["target"].dtype,
            generator=t_gen,
        )
    )
    return source_noise, t


def _flow_and_squared_error(model, prepared, source_noise, t_override, *, use_amp: bool):
    captured = {}

    def hook(_module, _inputs, output):
        if not isinstance(output, dict) or "predicted_latent" not in output:
            raise RuntimeError("transition forward contract lacks predicted_latent")
        captured["predicted_latent"] = output["predicted_latent"]

    handle = model.transition.register_forward_hook(hook)
    try:
        with torch.autocast(
            device_type=prepared["history"].device.type,
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
                t_override=t_override,
                source_noise=source_noise,
                return_endpoint=False,
                force_conditioned=False,
            )
            if "predicted_latent" not in captured:
                raise RuntimeError("transition hook did not capture predicted velocity")
            hist_frames = int(prepared["history"].shape[1])
            pred = captured["predicted_latent"][:, hist_frames:]
            target_velocity = (
                prepared["target"] * float(model.rescale_factor)
                - source_noise.to(dtype=prepared["target"].dtype)
            )
            if pred.shape != target_velocity.shape:
                raise RuntimeError("captured velocity/target shape mismatch")
            squared_error = (pred - target_velocity).square()
    finally:
        handle.remove()
    return fm_loss, info, squared_error


def _capture_payload_rng() -> dict:
    # Capture after the milestone validation so a later resume restores the exact
    # complete phase state as it existed when the checkpoint was written.
    return _rng_state()


def _payload(
    model,
    ema,
    optimizer,
    *,
    phase_step: int,
    history: list,
    a,
    train_ds,
    val_ds,
    sidecar,
    reuse,
    optimizer_info,
    vae_sha: str,
    parent_sha: str,
) -> dict:
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
        "rng_state": _capture_payload_rng(),
        "sampling_progress": {
            "protocol": SAMPLING_PROTOCOL,
            "completed_phase_steps": int(phase_step),
            "next_phase_step": int(phase_step),
            "base_seed": int(a.seed),
        },
        "phase_limit": int(a.phase_limit),
        "args": vars(a),
    }


def _validate_resume(ck, a, train_ds, val_ds, sidecar, vae_sha, parent_sha) -> int:
    arch = ck.get("architecture", {})
    if arch.get("training_protocol") != PROTOCOL:
        raise RuntimeError("resume checkpoint is not P0-F9 v8 U/M phase training")
    if arch.get("phase_variant") != a.variant:
        raise RuntimeError("resume variant differs")
    expected_lambda = 0.0 if a.variant == "U" else float(a.motion_weight_lambda)
    if float(arch.get("motion_lambda_applied", -1.0)) != expected_lambda:
        raise RuntimeError("resume motion lambda differs")
    if arch.get("sampling_protocol") != SAMPLING_PROTOCOL:
        raise RuntimeError("resume sampling protocol differs")
    if arch.get("random_protocol") != RANDOM_PROTOCOL:
        raise RuntimeError("resume random protocol differs")
    if arch.get("learning_rate_protocol") != LR_PROTOCOL:
        raise RuntimeError("resume LR protocol differs")
    if ck.get("train_cache_index_sha256") != file_sha256(train_ds.root / "index.json"):
        raise RuntimeError("resume train cache differs")
    if ck.get("val_cache_index_sha256") != file_sha256(val_ds.root / "index.json"):
        raise RuntimeError("resume val cache differs")
    if ck.get("motion_mask_sidecar_sha256") != file_sha256(sidecar.path):
        raise RuntimeError("resume motion-mask sidecar differs")
    if ck.get("upstream_checkpoint_sha256") != file_sha256(a.upstream_ckpt):
        raise RuntimeError("resume upstream checkpoint differs")
    if ck.get("vae_checkpoint_sha256") != vae_sha:
        raise RuntimeError("resume VAE differs")
    if ck.get("parent_checkpoint_sha256") != parent_sha:
        raise RuntimeError("resume parent checkpoint differs")
    if int(ck.get("parent_step", -1)) != int(a.parent_step):
        raise RuntimeError("resume parent step differs")
    if int(ck.get("phase_limit", -1)) != int(a.phase_limit):
        raise RuntimeError("resume phase-limit differs; do not redefine the phase at continuation")
    saved = ck.get("args") or {}
    for key in ("batch_size", "sample_steps", "seed"):
        if int(saved.get(key, -1)) != int(getattr(a, key)):
            raise RuntimeError(f"resume argument differs for {key}")
    for key in (
        "wm_lr",
        "new_lr",
        "weight_decay",
        "uncond_prob",
        "guidance_scale",
        "ema_decay",
        "grad_clip",
        "motion_weight_lambda",
    ):
        if float(saved.get(key, float("nan"))) != float(getattr(a, key)):
            raise RuntimeError(f"resume argument differs for {key}")
    step = int(ck.get("phase_step", ck.get("step", -1)))
    if step < 0 or step >= int(a.phase_end_step):
        raise RuntimeError(
            f"resume phase step {step} incompatible with end={a.phase_end_step}"
        )
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
    p.add_argument("--phase-end-step", type=int, default=400)
    p.add_argument("--phase-limit", type=int, default=800)
    p.add_argument("--milestones", type=int, nargs="+", default=[200, 400])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    # v7 step400 terminal LRs; constant through the U/M phase so 400->800 does
    # not restart or re-amplify a cosine schedule.
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
        raise ValueError("invalid milestone list")
    if int(a.phase_end_step) not in milestones:
        raise ValueError("phase-end-step must be a saved milestone")
    if a.motion_weight_lambda < 0 or a.grad_clip <= 0:
        raise ValueError("motion lambda must be non-negative and grad clip positive")
    if not 0.0 <= a.uncond_prob < 1.0:
        raise ValueError("uncond-prob must be in [0,1)")

    _seed_all(a.seed)
    device = torch.device(a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("P0-F9 v8 U/M phase training requires CUDA")
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

    model = make_p0_f9_model(
        20,
        sample_steps=a.sample_steps,
        unconditional_probability=a.uncond_prob,
        guidance_scale=a.guidance_scale,
        hist_last=HIST_LAST,
    ).to(device)
    reuse = load_shape_safe(model.transition, a.upstream_ckpt, verbose=True)
    if "traj_encoder.0.weight" not in set(reuse.get("loaded_keys", ())):
        raise RuntimeError("v8 U/M requires the official OccFM-Fut epoch=000196 checkpoint")
    official_reuse_fraction = require_checkpoint_reuse(reuse, min_fraction=0.80)

    # Critical U/M contract: both variants start from the exact same v7 step400
    # EMA weights, not from v7 raw weights and not from a permanent backbone freeze.
    model.load_state_dict(parent["ema"]["state_dict"], strict=True)
    optimizer, optimizer_info = _build_optimizer(
        model,
        reuse,
        wm_lr=a.wm_lr,
        new_lr=a.new_lr,
        weight_decay=a.weight_decay,
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
        phase_step = _validate_resume(
            ck, a, train_ds, val_ds, sidecar, vae_sha, parent_sha
        )
        model.load_state_dict(ck["state_dict"], strict=True)
        ema.load_state_dict(ck["ema"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        _restore_rng_state(ck["rng_state"])
        history = list(ck.get("training_history") or [])
        print(
            f"resumed v8 {a.variant} phase from {a.resume_from}: "
            f"phase_step={phase_step} -> {a.phase_end_step}"
        )
    else:
        initial_val = validate_fm(
            ema.model,
            val_loader,
            device,
            use_amp=use_amp,
            seed=a.seed + 100000,
        )
        history.append({"phase_step": 0, "train": None, "val_ema": initial_val})
        print("initial_validation", json.dumps(history[-1]))

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
            raise RuntimeError(
                "deterministic U/M phase batch contains no routed windows; fail-close "
                "rather than changing sampling progress"
            )
        motion_mask = _crop_motion_mask(sidecar, batch, prepared, device)
        source_noise, t_override = _step_noise_and_t(
            prepared, seed=a.seed, phase_step=phase_step
        )

        optimizer.zero_grad(set_to_none=True)
        fm_loss, info, squared_error = _flow_and_squared_error(
            model,
            prepared,
            source_noise,
            t_override,
            use_amp=use_amp,
        )
        weighted_preview = normalized_motion_weighted_mse(
            squared_error, motion_mask, float(a.motion_weight_lambda)
        )
        if a.variant == "U":
            total_loss = fm_loss
        else:
            total_loss = weighted_preview
        if not torch.isfinite(total_loss):
            raise RuntimeError(
                f"non-finite v8 {a.variant} loss at phase_step={phase_step}: {total_loss}"
            )
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(a.grad_clip))
        optimizer.step()
        ema_decay_now = ema.update(model)
        phase_step += 1

        motion_fraction = float(motion_mask.float().mean().detach().cpu())
        train_info = {
            "objective": float(total_loss.detach().cpu()),
            "uniform_fm_loss": float(fm_loss.detach().cpu()),
            "lambda2_weighted_fm_loss": float(weighted_preview.detach().cpu()),
            "fm_cosine": float(info["cosine"]),
            "pred_rms": float(info["pred_rms"]),
            "target_rms": float(info["target_rms"]),
            "motion_cell_fraction_batch": motion_fraction,
            "effective_motion_weight_mass_batch": effective_motion_weight_mass(
                motion_fraction, float(a.motion_weight_lambda)
            ),
            "motion_lambda_applied": 0.0
            if a.variant == "U"
            else float(a.motion_weight_lambda),
            "physics_authority": float(model.transition.physics_fusion.authority.cpu()),
            "grad_norm_before_clip": float(torch.as_tensor(grad_norm).detach().cpu()),
            "ema_decay": float(ema_decay_now),
            "lrs": {
                str(g.get("group_name", i)): float(g["lr"])
                for i, g in enumerate(optimizer.param_groups)
            },
            "sampling_phase_step": int(phase_step - 1),
        }
        if phase_step == 1 or phase_step % 20 == 0:
            print(
                f"variant={a.variant} phase_step={phase_step} "
                f"obj={train_info['objective']:.6f} uniform={train_info['uniform_fm_loss']:.6f} "
                f"weighted={train_info['lambda2_weighted_fm_loss']:.6f} "
                f"motion={100.0*motion_fraction:.3f}% cos={train_info['fm_cosine']:+.4f} "
                f"grad={train_info['grad_norm_before_clip']:.4f} lrs={train_info['lrs']}"
            )

        if phase_step in milestones:
            val = validate_fm(
                ema.model,
                val_loader,
                device,
                use_amp=use_amp,
                seed=a.seed + 100000,
            )
            row = {"phase_step": phase_step, "train": train_info, "val_ema": val}
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
        raise RuntimeError(
            f"phase ended at {phase_step}, expected {a.phase_end_step}"
        )
    final_ck = out / f"step_{phase_step:04d}.pt"
    if not final_ck.exists():
        raise RuntimeError("phase end checkpoint missing; endpoint must be a milestone")

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
        "objective": _architecture(a)["training_objective"],
        "motion_mask_protocol": MOTION_MASK_PROTOCOL,
        "motion_mask_sidecar": str(sidecar.path),
        "motion_mask_sidecar_sha256": file_sha256(sidecar.path),
        "train_routed_motion_fraction": sidecar.metadata.get("routed_top2_motion_fraction"),
        "configured_motion_lambda": float(a.motion_weight_lambda),
        "effective_train_motion_weight_mass": effective_motion_weight_mass(
            float(sidecar.metadata.get("routed_top2_motion_fraction", 0.0)),
            0.0 if a.variant == "U" else float(a.motion_weight_lambda),
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
        "continuation": (
            "If U/M deployment results justify 800 steps, resume each branch from its own "
            "step_0400.pt with --phase-end-step 800 --milestones 800. Optimizer, EMA, RNG "
            "and step-indexed sampling progress are restored; constant phase LR is not restarted."
        ),
    }
    (out / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("phase_complete", final_ck)


if __name__ == "__main__":
    main()
