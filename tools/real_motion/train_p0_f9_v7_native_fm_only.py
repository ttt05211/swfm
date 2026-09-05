#!/usr/bin/env python3
"""P0-F9 v7 causal control: train the sparse world model with native FM MSE only.

This experiment keeps the audited P0-F9 architecture, data, Top-2 routing,
physics/context conditioning, optimizer grouping, coherent source noise, EMA and
sampling contract fixed.  The only intentional training-objective change from
P0-F9 Stage-1 is removal of decoded semantic CE/Lovasz supervision.

The native OccFM objective is already an MSE in velocity space:
    z_t = (1-t) z_0 + t z_1
    v*  = z_1 - z_0
    L   = mean((v_theta(z_t, t, c) - v*) ** 2)

No VAE decoder or semantic sidecar participates in the training graph.  The VAE
checkpoint argument is retained only to fail-close provenance against the cached
latents and to remain compatible with the existing deployment evaluator.
"""
from __future__ import annotations

import argparse
import json
import math
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
from real_motion.msp_wm_cache import MSPWorldModelCacheDataset, collate_msp_wm
from real_motion.native_forecast import crop_coherent_source_noise
from real_motion.occfm_io import file_sha256
from tools.real_motion.train_p0_f9_native_sparse_forecast import (
    HIST_LAST,
    _build_optimizer,
    _cpu_state,
    _ema_payload,
    _fixed_noise_like,
    _lr_ratio,
    _scene_balanced_sampler,
    _seed_all,
    _set_lr,
    _validate_cache_pair,
    prepare_batch,
)

PROTOCOL = "p0_f9_v7_native_fm_only_v1"
SOURCE_SPATIAL_CONTRACT = "one_global_gaussian_field_cropped_into_top2_windows"


def _validate_vae_provenance(train_ds, val_ds, vae_ckpt: str) -> str:
    sha = file_sha256(vae_ckpt)
    for name, ds in (("train", train_ds), ("val", val_ds)):
        expected = ds.metadata.get("vae_checkpoint_sha256")
        if not expected or sha != expected:
            raise RuntimeError(
                f"{name} cache VAE provenance differs from --vae-ckpt; "
                "v7 must use the exact cached latent representation"
            )
    return sha


def _architecture(args) -> dict:
    return {
        "protocol": P0_F9_PROTOCOL,
        "stage": 1,
        "training_protocol": PROTOCOL,
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
        "training_objective": "native_flow_matching_velocity_mse_only",
        "loss_formula": "mean((predicted_velocity-(target-source))**2)",
        "semantic_auxiliary": False,
        "lovasz_auxiliary": False,
        "vae_decoder_in_training_graph": False,
        "vae": "official_occfm_vae_latents_from_cache_provenance_only",
        "sample_steps": int(args.sample_steps),
        "unconditional_probability": float(args.uncond_prob),
        "guidance_scale": float(args.guidance_scale),
        "ema_decay": float(args.ema_decay),
        "grad_norm_clip": float(args.grad_clip),
        "scene_sampling": "inverse_scene_window_count_weighted_sampling",
    }


def _payload(
    model,
    ema,
    optimizer,
    *,
    step: int,
    best_val_fm: float,
    history: list,
    args,
    train_ds,
    val_ds,
    reuse,
    optimizer_info,
    vae_sha: str,
    skipped: int,
) -> dict:
    return {
        "state_dict": _cpu_state(model),
        "ema": _ema_payload(ema),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": int(step),
        "best_val_objective": float(best_val_fm),
        "best_val_fm_loss": float(best_val_fm),
        "training_history": list(history),
        "architecture": _architecture(args),
        "train_metadata": train_ds.metadata,
        "val_metadata": val_ds.metadata,
        "train_cache_index_sha256": file_sha256(train_ds.root / "index.json"),
        "val_cache_index_sha256": file_sha256(val_ds.root / "index.json"),
        "upstream_checkpoint": str(Path(args.upstream_ckpt).resolve()),
        "upstream_checkpoint_sha256": file_sha256(args.upstream_ckpt),
        "vae_checkpoint": str(Path(args.vae_ckpt).resolve()),
        "vae_checkpoint_sha256": vae_sha,
        "upstream_reuse": reuse,
        "optimizer_contract": optimizer_info,
        "skipped_empty_train_batches": int(skipped),
        "args": vars(args),
    }


def _same_float(a, b, *, atol=1e-12) -> bool:
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=float(atol))


def _validate_resume_checkpoint(ck, args, train_ds, val_ds, vae_sha: str) -> None:
    arch = ck.get("architecture", {})
    if arch.get("protocol") != P0_F9_PROTOCOL or int(arch.get("stage", -1)) != 1:
        raise RuntimeError("resume checkpoint is not audited P0-F9 Stage-1")
    if arch.get("training_protocol") != PROTOCOL:
        raise RuntimeError("resume checkpoint is not P0-F9 v7 native-FM-only")
    if arch.get("training_objective") != "native_flow_matching_velocity_mse_only":
        raise RuntimeError("resume checkpoint objective differs from v7 native FM")
    if bool(arch.get("semantic_auxiliary", True)):
        raise RuntimeError("resume checkpoint unexpectedly contains semantic auxiliary training")
    if bool(arch.get("vae_decoder_in_training_graph", True)):
        raise RuntimeError("resume checkpoint unexpectedly used the VAE decoder in training")
    if arch.get("flow_source_spatial_contract") != SOURCE_SPATIAL_CONTRACT:
        raise RuntimeError("resume source-noise spatial contract differs")
    if int(arch.get("native_backbone_hist_last", -1)) != HIST_LAST:
        raise RuntimeError("resume HIST_LAST contract differs")
    if ck.get("train_cache_index_sha256") != file_sha256(train_ds.root / "index.json"):
        raise RuntimeError("resume train cache differs")
    if ck.get("val_cache_index_sha256") != file_sha256(val_ds.root / "index.json"):
        raise RuntimeError("resume val cache differs")
    if ck.get("upstream_checkpoint_sha256") != file_sha256(args.upstream_ckpt):
        raise RuntimeError("resume upstream OccFM checkpoint differs")
    if ck.get("vae_checkpoint_sha256") != vae_sha:
        raise RuntimeError("resume VAE provenance differs")

    saved = ck.get("args", {})
    exact_keys = ("steps", "batch_size", "sample_steps", "seed", "min_train_windows", "val_every")
    float_keys = (
        "wm_lr",
        "new_lr",
        "weight_decay",
        "warmup_fraction",
        "min_lr_ratio",
        "uncond_prob",
        "guidance_scale",
        "ema_decay",
        "grad_clip",
    )
    for key in exact_keys:
        if key not in saved or int(saved[key]) != int(getattr(args, key)):
            raise RuntimeError(f"resume argument differs for {key}")
    for key in float_keys:
        if key not in saved or not _same_float(saved[key], getattr(args, key)):
            raise RuntimeError(f"resume argument differs for {key}")
    step = int(ck.get("step", -1))
    if step < 0 or step > int(args.steps):
        raise RuntimeError(f"resume step {step} incompatible with steps={args.steps}")


@torch.no_grad()
def validate_fm(model, loader, device, *, use_amp: bool, seed: int) -> dict:
    was_training = model.training
    model.eval()
    fm_sum = 0.0
    cosine_sum = 0.0
    pred_rms_sum = 0.0
    target_rms_sum = 0.0
    conditioned_sum = 0.0
    windows = 0
    skipped = 0

    for batch_idx, batch in enumerate(loader):
        prepared = prepare_batch(batch, device)
        if prepared is None:
            skipped += 1
            continue
        global_noise = _fixed_noise_like(prepared["physics_full"], int(seed) + batch_idx)
        source_noise = crop_coherent_source_noise(
            global_noise, prepared["plan"], prepared["effective"]
        )
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
                return_endpoint=False,
                force_conditioned=True,
            )
        nwin = int(prepared["history"].shape[0])
        fm_sum += float(fm_loss.item()) * nwin
        cosine_sum += float(info["cosine"]) * nwin
        pred_rms_sum += float(info["pred_rms"]) * nwin
        target_rms_sum += float(info["target_rms"]) * nwin
        conditioned_sum += float(info["conditioned_fraction"]) * nwin
        windows += nwin

    model.train(was_training)
    if windows <= 0:
        raise RuntimeError("P0-F9 v7 validation has no valid routed windows")
    return {
        "objective": fm_sum / windows,
        "fm_loss": fm_sum / windows,
        "fm_cosine": cosine_sum / windows,
        "pred_rms": pred_rms_sum / windows,
        "target_rms": target_rms_sum / windows,
        "conditioned_fraction": conditioned_sum / windows,
        "num_windows": windows,
        "skipped_empty_batches": skipped,
        "t_override": 0.5,
        "semantic_loss": None,
        "vae_decoder_used": False,
    }


def _save_checkpoint(
    out: Path,
    name: str,
    model,
    ema,
    optimizer,
    *,
    step,
    best_val_fm,
    history,
    args,
    train_ds,
    val_ds,
    reuse,
    optimizer_info,
    vae_sha,
    skipped,
) -> dict:
    payload = _payload(
        model,
        ema,
        optimizer,
        step=step,
        best_val_fm=best_val_fm,
        history=history,
        args=args,
        train_ds=train_ds,
        val_ds=val_ds,
        reuse=reuse,
        optimizer_info=optimizer_info,
        vae_sha=vae_sha,
        skipped=skipped,
    )
    torch.save(payload, out / name)
    return payload


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-cache", required=True)
    p.add_argument("--val-cache", required=True)
    p.add_argument("--upstream-ckpt", required=True)
    p.add_argument(
        "--vae-ckpt",
        required=True,
        help="provenance only: decoder is never loaded or used in the training graph",
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--steps", type=int, default=400)
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
    p.add_argument("--val-every", type=int, default=100)
    p.add_argument("--min-train-windows", type=int, default=4000)
    p.add_argument("--seed", type=int, default=20260904)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--resume-from", default=None)
    a = p.parse_args()

    if min(a.steps, a.batch_size, a.val_every, a.min_train_windows) <= 0:
        raise ValueError("steps/batch-size/val-every/min-train-windows must be positive")
    if a.grad_clip <= 0:
        raise ValueError("grad-clip must be positive")
    if not 0.0 <= a.warmup_fraction < 1.0 or not 0.0 < a.min_lr_ratio <= 1.0:
        raise ValueError("invalid LR schedule")
    if not 0.0 <= a.uncond_prob < 1.0:
        raise ValueError("uncond-prob must be in [0,1)")

    _seed_all(a.seed)
    device = torch.device(a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu")
    train_ds = MSPWorldModelCacheDataset(a.train_cache)
    val_ds = MSPWorldModelCacheDataset(a.val_cache)
    _validate_cache_pair(train_ds, val_ds, min_train_windows=a.min_train_windows)
    vae_sha = _validate_vae_provenance(train_ds, val_ds, a.vae_ckpt)

    sampler = _scene_balanced_sampler(train_ds, seed=a.seed)
    pin = device.type == "cuda"
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
        raise RuntimeError("P0-F9 v7 requires the official OccFM-Fut epoch=000196 checkpoint")
    official_reuse_fraction = require_checkpoint_reuse(reuse, min_fraction=0.80)

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
    best_val_fm = float("inf")
    history = []
    skipped = 0
    if a.resume_from:
        ck = torch.load(a.resume_from, map_location="cpu", weights_only=False)
        _validate_resume_checkpoint(ck, a, train_ds, val_ds, vae_sha)
        model.load_state_dict(ck["state_dict"], strict=True)
        ema.load_state_dict(ck["ema"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        step = int(ck["step"])
        best_val_fm = float(ck.get("best_val_fm_loss", ck.get("best_val_objective", float("inf"))))
        history = list(ck.get("training_history", []))
        skipped = int(ck.get("skipped_empty_train_batches", 0))
        print(f"resumed P0-F9 v7 from {a.resume_from}: step={step} best_fm={best_val_fm:.6f}")
    else:
        initial_val = validate_fm(
            ema.model,
            val_loader,
            device,
            use_amp=use_amp,
            seed=a.seed + 100000,
        )
        best_val_fm = float(initial_val["fm_loss"])
        row = {"step": 0, "train": None, "val_ema": initial_val}
        history.append(row)
        _save_checkpoint(
            out,
            "step_0000.pt",
            model,
            ema,
            optimizer,
            step=0,
            best_val_fm=best_val_fm,
            history=history,
            args=a,
            train_ds=train_ds,
            val_ds=val_ds,
            reuse=reuse,
            optimizer_info=optimizer_info,
            vae_sha=vae_sha,
            skipped=skipped,
        )
        _save_checkpoint(
            out,
            "best.pt",
            model,
            ema,
            optimizer,
            step=0,
            best_val_fm=best_val_fm,
            history=history,
            args=a,
            train_ds=train_ds,
            val_ds=val_ds,
            reuse=reuse,
            optimizer_info=optimizer_info,
            vae_sha=vae_sha,
            skipped=skipped,
        )
        print("initial_validation", json.dumps(row))

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

        # Same coherent source contract as the audited P0-F9 trainer: draw one
        # full 50x50 Gaussian field and crop the exact Top-2 windows from it.
        global_noise = torch.randn_like(prepared["physics_full"])
        source_noise = crop_coherent_source_noise(
            global_noise, prepared["plan"], prepared["effective"]
        )
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
                source_noise=source_noise,
                return_endpoint=False,
                force_conditioned=False,
            )
            total_loss = fm_loss
        if not torch.isfinite(total_loss):
            raise RuntimeError(f"non-finite P0-F9 v7 FM loss step={step}: {total_loss}")
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(a.grad_clip))
        optimizer.step()
        ema_decay_now = ema.update(model)
        step += 1

        authority = float(model.transition.physics_fusion.authority.cpu())
        train_info = {
            "objective": float(fm_loss.detach().item()),
            "fm_loss": float(fm_loss.detach().item()),
            "fm_cosine": float(info["cosine"]),
            "pred_rms": float(info["pred_rms"]),
            "target_rms": float(info["target_rms"]),
            "conditioned_fraction": float(info["conditioned_fraction"]),
            "semantic_loss": None,
            "vae_decoder_used": False,
            "physics_authority": authority,
            "grad_norm_before_clip": float(torch.as_tensor(grad_norm).detach().cpu()),
            "ema_decay": float(ema_decay_now),
            "lr_ratio": float(lr_ratio),
            "lrs": lrs,
        }
        if step == 1 or step % 20 == 0:
            print(
                f"step={step} fm={train_info['fm_loss']:.6f} "
                f"cos={train_info['fm_cosine']:+.4f} "
                f"pred_rms={train_info['pred_rms']:.4f} "
                f"target_rms={train_info['target_rms']:.4f} "
                f"cond={train_info['conditioned_fraction']:.3f} "
                f"phys={authority:+.5f} grad={train_info['grad_norm_before_clip']:.4f} "
                f"lr={lrs}"
            )

        if step % a.val_every == 0 or step == a.steps:
            val = validate_fm(
                ema.model,
                val_loader,
                device,
                use_amp=use_amp,
                seed=a.seed + 100000,
            )
            row = {"step": step, "train": train_info, "val_ema": val}
            history.append(row)
            print("validation", json.dumps(row))
            improved = float(val["fm_loss"]) < best_val_fm
            if improved:
                best_val_fm = float(val["fm_loss"])
            payload = _save_checkpoint(
                out,
                f"step_{step:04d}.pt",
                model,
                ema,
                optimizer,
                step=step,
                best_val_fm=best_val_fm,
                history=history,
                args=a,
                train_ds=train_ds,
                val_ds=val_ds,
                reuse=reuse,
                optimizer_info=optimizer_info,
                vae_sha=vae_sha,
                skipped=skipped,
            )
            torch.save(payload, out / "latest.pt")
            if improved:
                torch.save(payload, out / "best.pt")

    final_payload = _save_checkpoint(
        out,
        "last.pt",
        model,
        ema,
        optimizer,
        step=step,
        best_val_fm=best_val_fm,
        history=history,
        args=a,
        train_ds=train_ds,
        val_ds=val_ds,
        reuse=reuse,
        optimizer_info=optimizer_info,
        vae_sha=vae_sha,
        skipped=skipped,
    )
    torch.save(final_payload, out / "latest.pt")
    report = {
        "protocol": PROTOCOL,
        "stage": 1,
        "objective": "native_flow_matching_velocity_mse_only",
        "best_val_fm_loss": best_val_fm,
        "official_transition_reuse_fraction": official_reuse_fraction,
        "optimizer_contract": optimizer_info,
        "history": history,
        "skipped_empty_train_batches": skipped,
        "semantic_auxiliary": False,
        "vae_decoder_in_training_graph": False,
        "decision": (
            "Evaluate step_0000/0100/0200/0400 EMA checkpoints with the existing "
            "P0-F9 deployment evaluator. Primary causal checks: FM preservation, "
            "Overall/Moving, dynamic-volume flooding, stale/clear/write behavior."
        ),
    }
    (out / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("saved", out / "last.pt")


if __name__ == "__main__":
    main()
