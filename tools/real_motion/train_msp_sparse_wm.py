#!/usr/bin/env python3
"""Train the first real Top-2 MSP-routed Sparse World Model.

Frozen routing: one set of two 20x20 latent windows per sample, predicted by the
already-trained causal MSP.  Trainable part: an OccFM-Fut-196 initialized local
operator flowing from the KTA/zero-motion anchor latent to the full GT latent.
Only latent flow MSE is used; there is no occupancy CE/Lovasz/ABE/router loss.
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
    origins_cpu = batch["window_origins"].long()
    valid_cpu = batch["window_valid"].bool()
    B, K = valid_cpu.shape
    if K != 2:
        raise RuntimeError(f"P0-F3 expects K=2, got {K}")
    plan_cpu = WindowPlan(origins_cpu, valid_cpu, (20, 20), (50, 50))
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
    if not bool(valid.any()):
        raise RuntimeError("batch contains no valid Top-2 MSP windows")

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
    for batch in loader:
        hist, target, anchor, traj, origins = prepare_batch(batch, device)
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
    d = max(count, 1)
    return {
        "loss": total / d,
        "cosine": cos / d,
        "target_rms": target_rms / d,
        "pred_rms": pred_rms / d,
        "num_windows": count,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-cache", required=True)
    p.add_argument("--val-cache", required=True)
    p.add_argument("--upstream-ckpt", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--steps", type=int, default=3000)
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
    scaler = torch.amp.GradScaler("cuda", enabled=False)  # BF16 does not need scaling.

    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    best_state = None
    history = []
    step = 0
    iterator = iter(train_loader)
    model.train()

    while step < a.steps:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        hist, target, anchor, traj, origins = prepare_batch(batch, device)
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
            if val["loss"] < best_val:
                best_val = float(val["loss"])
                best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
                torch.save(
                    {
                        "state_dict": best_state,
                        "step": step,
                        "best_val_loss": best_val,
                        "training_history": history,
                        "train_metadata": train_ds.metadata,
                        "val_metadata": val_ds.metadata,
                        "architecture": {
                            "window_hw": [20, 20],
                            "topk": 2,
                            "sample_steps": int(a.sample_steps),
                            "source_noise_std": float(a.source_noise_std),
                            "prior_channels": 16,
                            "flow": "anchor_to_full_gt_latent",
                        },
                        "upstream_checkpoint": str(Path(a.upstream_ckpt).resolve()),
                        "upstream_checkpoint_sha256": file_sha256(a.upstream_ckpt),
                        "upstream_reuse": reuse,
                        "args": vars(a),
                    },
                    out / "best.pt",
                )

    if best_state is None:
        raise RuntimeError("no validation checkpoint was produced")
    model.load_state_dict(best_state, strict=True)
    torch.save(
        {
            "state_dict": best_state,
            "step": step,
            "best_val_loss": best_val,
            "training_history": history,
            "train_metadata": train_ds.metadata,
            "val_metadata": val_ds.metadata,
            "architecture": {
                "window_hw": [20, 20], "topk": 2,
                "sample_steps": int(a.sample_steps),
                "source_noise_std": float(a.source_noise_std),
                "prior_channels": 16,
                "flow": "anchor_to_full_gt_latent",
            },
            "upstream_checkpoint": str(Path(a.upstream_ckpt).resolve()),
            "upstream_checkpoint_sha256": file_sha256(a.upstream_ckpt),
            "upstream_reuse": reuse,
            "args": vars(a),
        },
        out / "last.pt",
    )
    (out / "training_report.json").write_text(
        json.dumps({
            "protocol": "p0_f3_top2_anchor_sparse_wm_v1",
            "best_val_loss": best_val,
            "history": history,
            "upstream_reuse_fraction": reuse_fraction,
            "decision": "Evaluate real occupancy with eval_msp_sparse_wm.py; latent loss alone is not a GO signal.",
        }, indent=2),
        encoding="utf-8",
    )
    print("saved", out / "best.pt")


if __name__ == "__main__":
    main()
