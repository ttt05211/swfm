#!/usr/bin/env python3
"""Train the first real Top-2 MSP-routed Sparse World Model.

Frozen routing: one set of two 20x20 latent windows per sample, predicted by the
already-trained causal MSP. Trainable part: an OccFM-Fut-196 initialized local
operator flowing from the KTA/zero-motion anchor latent to the full GT latent.
Only latent flow MSE is used; there is no occupancy CE/Lovasz/ABE/router loss.

Samples for which the frozen MSP selects no valid window are legitimate
anchor-only cases. They contribute no Sparse-WM gradient and are skipped during
training/latent validation instead of being treated as an error. Real occupancy
evaluation still preserves the causal anchor for those samples.
"""
from __future__ import annotations

import argparse
import copy
import json
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
from real_motion.models import AnchorWindowCFM, MotionWindowFlowMatching
from real_motion.msp_wm_cache import MSPWorldModelCacheDataset, collate_msp_wm
from real_motion.occfm_io import file_sha256
from real_motion.windows import WindowPlan, crop_windows


def make_model(window=20, *, sample_steps=10, source_noise_std=0.0):
    tr = MotionWindowFlowMatching(
        in_channels=16,
        out_channels=16,
        model_channels=128,
        channel_multi=[2, 4],
        input_size=[window, window],
        trajectory_length=12,
        init_kernel_size=7,
        init_3d_conv_channels=64,
        attn_dim=32,
        temporal_attn_head=8,
        spatial_attn_head=8,
        prior_channels=16,
        full_input_size=(50, 50),
    )
    return AnchorWindowCFM(
        tr,
        rescale_factor=10.0,
        sample_steps=int(sample_steps),
        alpha_shift=3.0,
        source_noise_std=float(source_noise_std),
    )


def _validate_cache_pair(train_ds, val_ds):
    tm, vm = train_ds.metadata, val_ds.metadata
    for name, meta in (("train", tm), ("val", vm)):
        if int(meta.get("topk", -1)) != 2:
            raise RuntimeError(f"{name} cache is not frozen Top-2")
        if list(meta.get("window_hw", [])) != [20, 20]:
            raise RuntimeError(f"{name} cache window size is not 20x20")
        if meta.get("vae_mode") != "mean":
            raise RuntimeError(f"{name} cache must use deterministic VAE mean latents")
        if meta.get("loss_contract") != "anchor_to_gt_local_flow_no_auxiliary_losses":
            raise RuntimeError(f"{name} cache target contract mismatch")
    overlap = sorted(set(tm.get("scene_names", [])) & set(vm.get("scene_names", [])))
    if overlap:
        raise RuntimeError(f"train/val scene leakage ({len(overlap)}), e.g. {overlap[:3]}")
    if tm.get("msp_checkpoint_sha256") != vm.get("msp_checkpoint_sha256"):
        raise RuntimeError("train/val caches were routed by different MSP checkpoints")
    if tm.get("vae_checkpoint_sha256") != vm.get("vae_checkpoint_sha256"):
        raise RuntimeError("train/val caches use different VAE checkpoints")


def prepare_batch(batch, device):
    """Crop valid Top-2 windows; return None for a legitimate anchor-only batch."""
    origins_cpu = batch["window_origins"].long()
    valid_cpu = batch["window_valid"].bool()
    B, K = valid_cpu.shape
    if K != 2:
        raise RuntimeError(f"P0-F3 expects K=2, got {K}")
    plan_cpu = WindowPlan(origins_cpu, valid_cpu, (20, 20), (50, 50))
    if not bool(plan_cpu.valid.any()):
        return None
    batch = {
        k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }
    plan = WindowPlan(
        plan_cpu.origins.to(device, non_blocking=True),
        plan_cpu.valid.to(device, non_blocking=True),
        plan_cpu.window_hw,
        plan_cpu.full_hw,
    )
    hist = crop_windows(batch["moving_history_latent"], plan)
    anchor = crop_windows(batch["anchor_future_latent"], plan)
    target = crop_windows(batch["gt_future_latent"], plan)
    valid = plan.valid.reshape(-1)

    def flat(x):
        return x.reshape(B * K, *x.shape[2:])[valid]

    hist, anchor, target = map(flat, (hist, anchor, target))
    origins = plan.origins.reshape(B * K, 2)[valid]
    traj = batch["trajectory"]
    if tuple(traj.shape[1:]) != (12, 2):
        raise RuntimeError(f"trajectory batch must be [B,12,2], got {tuple(traj.shape)}")
    traj = traj[:, None].expand(B, K, 12, 2).reshape(B * K, 12, 2)[valid]
    return hist, target, anchor, traj, origins


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total = 0.0
    count = 0
    cos = 0.0
    target_rms = 0.0
    pred_rms = 0.0
    skipped_batches = 0
    skipped_samples = 0
    for batch in loader:
        prepared = prepare_batch(batch, device)
        if prepared is None:
            skipped_batches += 1
            skipped_samples += len(batch.get("sample_id", []))
            continue
        hist, target, anchor, traj, origins = prepared
        loss, info = model.flow_loss(
            hist, target, anchor,
            trajectory=traj,
            window_origins=origins,
            t_override=0.5,
            source_noise=torch.zeros_like(anchor),
        )
        n = int(hist.shape[0])
        total += float(loss.item()) * n
        cos += float(info["cosine"]) * n
        target_rms += float(info["target_rms"]) * n
        pred_rms += float(info["pred_rms"]) * n
        count += n
    model.train()
    if count <= 0:
        raise RuntimeError("validation contains no valid routed Sparse-WM windows")
    d = float(count)
    return {
        "loss": total / d,
        "cosine": cos / d,
        "target_rms": target_rms / d,
        "pred_rms": pred_rms / d,
        "num_windows": count,
        "skipped_empty_batches": skipped_batches,
        "skipped_anchor_only_samples": skipped_samples,
    }


def _architecture(args):
    return {
        "window_hw": [20, 20],
        "topk": 2,
        "sample_steps": int(args.sample_steps),
        "source_noise_std": float(args.source_noise_std),
        "prior_channels": 16,
        "flow": "anchor_to_full_gt_latent",
    }


def _validate_resume_checkpoint(ck, args, train_ds, val_ds):
    arch = ck.get("architecture", {})
    if list(arch.get("window_hw", [])) != [20, 20] or int(arch.get("topk", -1)) != 2:
        raise RuntimeError("resume checkpoint is not the frozen Top-2/20x20 P0-F3 model")
    if int(arch.get("sample_steps", args.sample_steps)) != int(args.sample_steps):
        raise RuntimeError("resume checkpoint sample_steps differs from current command")
    if float(arch.get("source_noise_std", args.source_noise_std)) != float(args.source_noise_std):
        raise RuntimeError("resume checkpoint source_noise_std differs from current command")
    ctm = ck.get("train_metadata", {})
    cvm = ck.get("val_metadata", {})
    for key in ("msp_checkpoint_sha256", "vae_checkpoint_sha256"):
        if ctm.get(key) and ctm.get(key) != train_ds.metadata.get(key):
            raise RuntimeError(f"resume train metadata mismatch for {key}")
        if cvm.get(key) and cvm.get(key) != val_ds.metadata.get(key):
            raise RuntimeError(f"resume val metadata mismatch for {key}")


def _checkpoint_payload(
    *, model_state, optimizer, step, best_val, history, train_ds, val_ds,
    args, upstream_ckpt, reuse, skipped_train_batches,
):
    return {
        "state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "step": int(step),
        "best_val_loss": float(best_val),
        "training_history": list(history),
        "train_metadata": train_ds.metadata,
        "val_metadata": val_ds.metadata,
        "architecture": _architecture(args),
        "upstream_checkpoint": str(Path(upstream_ckpt).resolve()),
        "upstream_checkpoint_sha256": file_sha256(upstream_ckpt),
        "upstream_reuse": reuse,
        "skipped_empty_train_batches": int(skipped_train_batches),
        "args": vars(args),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-cache", required=True)
    p.add_argument("--val-cache", required=True)
    p.add_argument("--upstream-ckpt", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--steps", type=int, default=3000,
                   help="absolute target optimizer step, including resumed steps")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--val-every", type=int, default=200)
    p.add_argument("--sample-steps", type=int, default=10)
    p.add_argument("--source-noise-std", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=20260829)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--resume-from", default=None,
                   help="P0-F3 best/latest checkpoint. Old checkpoints without optimizer state resume weights/history and reset AdamW moments.")
    a = p.parse_args()
    if min(a.steps, a.batch_size, a.val_every) <= 0:
        raise ValueError("steps/batch-size/val-every must be positive")
    if a.source_noise_std != 0.0:
        raise ValueError("P0-F3 first run freezes source_noise_std=0; do not tune it yet")

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    device = torch.device(a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu")

    train_ds = MSPWorldModelCacheDataset(a.train_cache)
    val_ds = MSPWorldModelCacheDataset(a.val_cache)
    _validate_cache_pair(train_ds, val_ds)
    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=a.batch_size, shuffle=True, num_workers=a.num_workers,
        collate_fn=collate_msp_wm, drop_last=False, pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds, batch_size=a.batch_size, shuffle=False, num_workers=a.num_workers,
        collate_fn=collate_msp_wm, drop_last=False, pin_memory=pin,
    )

    model = make_model(20, sample_steps=a.sample_steps, source_noise_std=a.source_noise_std).to(device)
    reuse = load_shape_safe(model.transition, a.upstream_ckpt, verbose=True)
    if "traj_encoder.0.weight" not in set(reuse.get("loaded_keys", ())):
        raise RuntimeError("use the official OccFM-Fut epoch=000196 checkpoint")
    reuse_fraction = float(reuse.get("loaded", 0)) / max(float(reuse.get("target_total", 1)), 1.0)
    if reuse_fraction < 0.90:
        raise RuntimeError(f"unexpectedly low upstream reuse fraction {reuse_fraction:.3f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    use_amp = bool(a.amp and device.type == "cuda")

    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    best_state = None
    history = []
    step = 0
    skipped_train_batches = 0

    if a.resume_from:
        ck = torch.load(a.resume_from, map_location="cpu", weights_only=False)
        _validate_resume_checkpoint(ck, a, train_ds, val_ds)
        model.load_state_dict(ck["state_dict"], strict=True)
        step = int(ck.get("step", 0))
        if step < 0 or step > a.steps:
            raise RuntimeError(f"resume step {step} is incompatible with target --steps {a.steps}")
        best_val = float(ck.get("best_val_loss", float("inf")))
        history = list(ck.get("training_history", []))
        skipped_train_batches = int(ck.get("skipped_empty_train_batches", 0))
        if np.isfinite(best_val):
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        opt_state = ck.get("optimizer_state_dict")
        if opt_state is not None:
            optimizer.load_state_dict(opt_state)
            print(f"resumed model+optimizer from {a.resume_from} at step={step} best_val={best_val:.6f}")
        else:
            print(
                f"resumed model weights/history from legacy checkpoint {a.resume_from} "
                f"at step={step} best_val={best_val:.6f}; optimizer moments reset"
            )

    iterator = iter(train_loader)
    model.train()
    last_info = None

    while step < a.steps:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        prepared = prepare_batch(batch, device)
        if prepared is None:
            skipped_train_batches += 1
            if skipped_train_batches <= 3 or skipped_train_batches % 10 == 0:
                print(
                    f"skip anchor-only batch: skipped={skipped_train_batches} "
                    f"optimizer_step={step}"
                )
            continue
        hist, target, anchor, traj, origins = prepared
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            loss, info = model.flow_loss(
                hist, target, anchor,
                trajectory=traj,
                window_origins=origins,
            )
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}: {loss}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        step += 1
        last_info = info
        if step == 1 or step % 20 == 0:
            print(
                f"step={step} loss={float(loss.item()):.6f} "
                f"cos={info['cosine']:.4f} target_rms={info['target_rms']:.5f} "
                f"pred_rms={info['pred_rms']:.5f}"
            )
        if step % a.val_every == 0 or step == a.steps:
            val = validate(model, val_loader, device)
            row = {"step": step, "train": info, "val": val}
            history.append(row)
            print("validation", json.dumps(row))

            current_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            torch.save(
                _checkpoint_payload(
                    model_state=current_state,
                    optimizer=optimizer,
                    step=step,
                    best_val=min(best_val, float(val["loss"])),
                    history=history,
                    train_ds=train_ds,
                    val_ds=val_ds,
                    args=a,
                    upstream_ckpt=a.upstream_ckpt,
                    reuse=reuse,
                    skipped_train_batches=skipped_train_batches,
                ),
                out / "latest.pt",
            )
            if val["loss"] < best_val:
                best_val = float(val["loss"])
                best_state = current_state
                torch.save(
                    _checkpoint_payload(
                        model_state=best_state,
                        optimizer=optimizer,
                        step=step,
                        best_val=best_val,
                        history=history,
                        train_ds=train_ds,
                        val_ds=val_ds,
                        args=a,
                        upstream_ckpt=a.upstream_ckpt,
                        reuse=reuse,
                        skipped_train_batches=skipped_train_batches,
                    ),
                    out / "best.pt",
                )

    if best_state is None:
        raise RuntimeError("no validation checkpoint was produced")
    if last_info is None and step < a.steps:
        raise RuntimeError("no optimizer step was completed")

    model.load_state_dict(best_state, strict=True)
    torch.save(
        _checkpoint_payload(
            model_state=best_state,
            optimizer=optimizer,
            step=step,
            best_val=best_val,
            history=history,
            train_ds=train_ds,
            val_ds=val_ds,
            args=a,
            upstream_ckpt=a.upstream_ckpt,
            reuse=reuse,
            skipped_train_batches=skipped_train_batches,
        ),
        out / "last.pt",
    )
    (out / "training_report.json").write_text(
        json.dumps({
            "protocol": "p0_f3_top2_anchor_sparse_wm_v1",
            "best_val_loss": best_val,
            "history": history,
            "upstream_reuse_fraction": reuse_fraction,
            "skipped_empty_train_batches": skipped_train_batches,
            "resumed_from": a.resume_from,
            "decision": "Evaluate real occupancy with eval_msp_sparse_wm.py; latent loss alone is not a GO signal.",
        }, indent=2),
        encoding="utf-8",
    )
    print("saved", out / "best.pt")


if __name__ == "__main__":
    main()
