#!/usr/bin/env python3
"""DDP trainer for formal P0-F9 v9 full-data M.

This is the distributed counterpart of ``train_p0_f9_v9_full_m.py``.  It keeps
that scientific contract fixed while allowing one model to train across multiple
GPUs with a fixed *global* batch size.  For the intended two-GPU run,
``--batch-size 8`` means 4 samples per rank and global batch 8.

Important DDP details:
- scene-balanced samples are drawn once per epoch as one deterministic global
  sequence, then split rank-wise inside each global batch;
- the sequence is padded only to a complete global batch (at most B-1 draws);
- all-zero-route rank-local batches are forbidden; the epoch draw is
  deterministically retried if necessary;
- source noise and FM time are deterministic by global step + rank so resume is
  exact without rank-specific RNG snapshots;
- the motion-weighted FM numerator/denominator are reduced globally.  Each rank
  backpropagates ``world_size * local_numerator / global_denominator`` so DDP's
  gradient averaging is exactly the gradient of the global normalized loss;
- EMA is maintained on every rank after synchronized optimizer steps; validation,
  checkpointing and reports are rank-0 only;
- checkpoints store the unwrapped P0-F9 state_dict, so the existing deployment
  evaluators remain compatible and no ``module.`` prefix leaks into artifacts.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
UP = ROOT / "upstream_occfm"
sys.path[:0] = [str(ROOT), str(UP)]

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler

from real_motion.checkpoint import load_shape_safe, require_checkpoint_reuse
from real_motion.model_ema import ModelEMA
from real_motion.models.p0_f9 import make_p0_f9_model
from real_motion.motion_mask_sidecar import MotionMaskSidecar, effective_motion_weight_mass
from real_motion.msp_wm_cache import MSPWorldModelCacheDataset, collate_msp_wm
from real_motion.native_forecast import crop_coherent_source_noise, deterministic_sample_seed
from tools.real_motion.train_p0_f9_native_sparse_forecast import (
    HIST_LAST,
    _build_optimizer,
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
    _crop_motion_mask,
    _flow_and_squared_error,
    _validate_motion_sidecar,
)
from tools.real_motion import train_p0_f9_v9_full_m as base


PROTOCOL = "p0_f9_v9_full_data_m_official_init_ddp_v1"
DDP_PROTOCOL = "global_batch_preserving_rank_partition_global_weighted_loss_v1"
RANDOM_PROTOCOL = "global_step_rank_indexed_noise_and_t_v1"
SAMPLING_PROTOCOL = "scene_balanced_global_epoch_draw_rank_partition_v1"


class RankBatchSampler(Sampler[list[int]]):
    """Yield one rank's slice of every deterministic global batch."""

    def __init__(self, indices, *, global_batch_size: int, rank: int, world_size: int):
        self.indices = [int(x) for x in indices]
        self.global_batch_size = int(global_batch_size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        if self.global_batch_size <= 0 or self.world_size <= 0:
            raise ValueError("invalid global batch/world size")
        if self.global_batch_size % self.world_size:
            raise ValueError("global batch size must be divisible by world size")
        if len(self.indices) % self.global_batch_size:
            raise ValueError("global index sequence must contain complete global batches")
        self.local_batch_size = self.global_batch_size // self.world_size

    def __iter__(self):
        lb = self.local_batch_size
        gb = self.global_batch_size
        lo = self.rank * lb
        hi = lo + lb
        for start in range(0, len(self.indices), gb):
            yield self.indices[start + lo : start + hi]

    def __len__(self):
        return len(self.indices) // self.global_batch_size


class FlowDDPWrapper(nn.Module):
    """Give DDP a real forward while preserving the audited P0-F9 flow helper."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, prepared, source_noise, t_override, use_amp: bool):
        return _flow_and_squared_error(
            self.model,
            prepared,
            source_noise,
            t_override,
            use_amp=bool(use_amp),
        )


def _init_dist() -> tuple[int, int, int, torch.device]:
    required = ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    missing = [k for k in required if k not in os.environ]
    if missing:
        raise RuntimeError(
            "DDP trainer must be launched with torchrun; missing env " + ",".join(missing)
        )
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size < 2:
        raise RuntimeError("DDP trainer expects WORLD_SIZE>=2; use the single-GPU trainer otherwise")
    if not torch.cuda.is_available():
        raise RuntimeError("formal DDP full-data training requires CUDA")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return rank, world_size, local_rank, torch.device("cuda", local_rank)


def _rank0(rank: int) -> bool:
    return int(rank) == 0


def _all_reduce_sum(x: torch.Tensor) -> torch.Tensor:
    y = x.detach().clone()
    dist.all_reduce(y, op=dist.ReduceOp.SUM)
    return y


def _step_seed(seed: int, global_step: int, rank: int, stream: str) -> int:
    return deterministic_sample_seed(
        f"global_step:{int(global_step)}:rank:{int(rank)}",
        int(seed),
        stream=f"v9_ddp_{stream}",
    )


def _step_noise_and_t(prepared, *, seed: int, global_step: int, rank: int):
    device = prepared["physics_full"].device
    noise_gen = torch.Generator(device=device)
    noise_gen.manual_seed(_step_seed(seed, global_step, rank, "noise"))
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
    t_gen.manual_seed(_step_seed(seed, global_step, rank, "t"))
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


def _scene_weights(ds) -> torch.Tensor:
    scenes = [str(e["scene_name"]) for e in ds.entries]
    counts = Counter(scenes)
    return torch.tensor([1.0 / float(counts[s]) for s in scenes], dtype=torch.double)


def _draw_global_epoch_indices(
    ds,
    *,
    seed: int,
    epoch: int,
    global_batch_size: int,
    world_size: int,
    steps_per_epoch: int,
) -> tuple[list[int], int]:
    """Draw complete global batches and guarantee every rank has routed work."""
    zero_ids = set(str(x) for x in ds.metadata.get("zero_route_sample_ids", []))
    if "zero_route_sample_ids" not in ds.metadata:
        raise RuntimeError(
            "full native cache lacks zero_route_sample_ids metadata; rebuild it with the current "
            "build_p0_f9_full_native_cache.py before DDP training"
        )
    weights = _scene_weights(ds)
    total_draws = int(steps_per_epoch) * int(global_batch_size)
    local_bs = int(global_batch_size) // int(world_size)
    entries = ds.entries
    for attempt in range(32):
        gen = torch.Generator(device="cpu")
        draw_seed = deterministic_sample_seed(
            f"full_epoch:{int(epoch)}:attempt:{attempt}",
            int(seed),
            stream="v9_ddp_scene_sampler",
        )
        gen.manual_seed(draw_seed)
        indices = torch.multinomial(
            weights,
            num_samples=total_draws,
            replacement=True,
            generator=gen,
        ).tolist()
        bad = False
        for start in range(0, total_draws, int(global_batch_size)):
            for rank in range(int(world_size)):
                lo = start + rank * local_bs
                chunk = indices[lo : lo + local_bs]
                if all(str(entries[i]["sample_id"]) in zero_ids for i in chunk):
                    bad = True
                    break
            if bad:
                break
        if not bad:
            return indices, attempt
    raise RuntimeError("could not draw a DDP epoch without an all-zero-route rank-local batch")


def _global_weighted_loss(
    squared_error: torch.Tensor,
    motion_mask: torch.Tensor,
    motion_lambda: float,
    world_size: int,
) -> tuple[torch.Tensor, float, float, float]:
    """Return exact global normalized objective with DDP-correct local scaling."""
    if squared_error.ndim != 5 or motion_mask.ndim != 4:
        raise ValueError("unexpected FM error/motion-mask rank")
    weights = 1.0 + float(motion_lambda) * motion_mask.to(dtype=torch.float32)
    weights = weights[:, :, None].expand_as(squared_error)
    local_num = (squared_error.float() * weights).sum()
    local_den = weights.sum()
    global_den = _all_reduce_sum(local_den)
    if float(global_den.item()) <= 0:
        raise RuntimeError("global motion-weight denominator is zero")
    # DDP averages parameter gradients across ranks.  Multiplying each local
    # numerator by W/global_den makes the averaged gradient equal grad(sum N)/sum D.
    loss_for_backward = local_num * float(world_size) / global_den
    global_num = _all_reduce_sum(local_num)

    uniform_num = _all_reduce_sum(squared_error.detach().float().sum())
    uniform_den = _all_reduce_sum(
        torch.tensor(float(squared_error.numel()), device=squared_error.device)
    )
    global_motion = _all_reduce_sum(motion_mask.detach().float().sum())
    global_motion_den = _all_reduce_sum(
        torch.tensor(float(motion_mask.numel()), device=motion_mask.device)
    )
    return (
        loss_for_backward,
        float((global_num / global_den).item()),
        float((uniform_num / uniform_den.clamp_min(1.0)).item()),
        float((global_motion / global_motion_den.clamp_min(1.0)).item()),
    )


def _global_macro(value: float, count: int, device: torch.device) -> float:
    pair = torch.tensor([float(value) * int(count), float(count)], device=device, dtype=torch.float64)
    dist.all_reduce(pair, op=dist.ReduceOp.SUM)
    return float((pair[0] / pair[1].clamp_min(1.0)).item())


def _decorate_payload(payload: dict, *, world_size: int, global_batch: int, local_batch: int) -> dict:
    payload["architecture"]["training_protocol"] = PROTOCOL
    payload["architecture"]["distributed_training"] = DDP_PROTOCOL
    payload["architecture"]["random_protocol"] = RANDOM_PROTOCOL
    payload["architecture"]["scene_sampling"] = SAMPLING_PROTOCOL
    payload["ddp"] = {
        "protocol": DDP_PROTOCOL,
        "world_size": int(world_size),
        "global_batch_size": int(global_batch),
        "per_rank_batch_size": int(local_batch),
        "backend": "nccl",
        "rank0_checkpoint_only": True,
        "unwrapped_checkpoint_state_dict": True,
        "global_loss_reduction": "sum_numerator_sum_denominator_exact_with_ddp_gradient_scaling",
    }
    return payload


def _validate_ddp_resume(ck: dict, *, world_size: int, global_batch: int) -> None:
    meta = ck.get("ddp") or {}
    if meta.get("protocol") != DDP_PROTOCOL:
        raise RuntimeError("resume checkpoint is not the v9 DDP full-M contract")
    if int(meta.get("world_size", -1)) != int(world_size):
        raise RuntimeError("resume WORLD_SIZE differs; exact DDP continuation requires same world size")
    if int(meta.get("global_batch_size", -1)) != int(global_batch):
        raise RuntimeError("resume global batch size differs")


def _worker(a, rank: int, world_size: int, local_rank: int, device: torch.device) -> None:
    if a.batch_size % world_size:
        raise ValueError("--batch-size is GLOBAL batch size and must be divisible by WORLD_SIZE")
    local_batch = a.batch_size // world_size
    if local_batch <= 0:
        raise ValueError("per-rank batch size must be positive")
    if float(a.uncond_prob) != 0.0:
        raise ValueError("formal DDP contract fixes --uncond-prob 0 for exact step-indexed randomness")

    _seed_all(a.seed + rank)
    train_ds = MSPWorldModelCacheDataset(a.train_cache)
    val_ds = MSPWorldModelCacheDataset(a.val_cache)
    base._validate_full_train_cache(train_ds)
    _validate_cache_pair(train_ds, val_ds, min_train_windows=int(a.min_train_windows))
    vae_sha = _validate_vae_provenance(train_ds, val_ds, a.vae_ckpt)
    sidecar = MotionMaskSidecar(a.motion_mask_sidecar)
    _validate_motion_sidecar(sidecar, train_ds)

    schedule = base.schedule_shape(len(train_ds), a.batch_size, a.schedule_epochs)
    total_schedule_steps = int(schedule["total_schedule_steps"])
    steps_per_epoch = int(schedule["steps_per_epoch"])
    padded_draws = steps_per_epoch * int(a.batch_size)

    model = make_p0_f9_model(
        20,
        sample_steps=a.sample_steps,
        unconditional_probability=a.uncond_prob,
        guidance_scale=a.guidance_scale,
        hist_last=HIST_LAST,
        ordered_context=False,
    ).to(device)
    reuse = load_shape_safe(model.transition, a.upstream_ckpt, verbose=_rank0(rank))
    if "traj_encoder.0.weight" not in set(reuse.get("loaded_keys", ())):
        raise RuntimeError("v9 DDP requires released OccFM-Fut epoch=000196")
    official_reuse_fraction = require_checkpoint_reuse(reuse, min_fraction=0.80)
    optimizer, optimizer_info = _build_optimizer(
        model,
        reuse,
        wm_lr=a.wm_lr,
        new_lr=a.new_lr,
        weight_decay=a.weight_decay,
    )

    start_epoch = 0
    global_step = 0
    history = []
    skipped = 0
    resume_ck = None
    if a.resume_from:
        resume_ck = torch.load(a.resume_from, map_location="cpu", weights_only=False)
        start_epoch = base._validate_resume(
            resume_ck, a, train_ds, val_ds, sidecar, vae_sha, schedule
        )
        _validate_ddp_resume(resume_ck, world_size=world_size, global_batch=a.batch_size)
        model.load_state_dict(resume_ck["state_dict"], strict=True)
        optimizer.load_state_dict(resume_ck["optimizer_state_dict"])
        global_step = int(resume_ck["global_step"])
        history = list(resume_ck.get("training_history") or [])
        skipped = int(resume_ck.get("skipped_empty_train_batches", 0))

    ddp = DDP(
        FlowDDPWrapper(model),
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        find_unused_parameters=False,
    )
    ema = ModelEMA(model, decay=a.ema_decay)
    if resume_ck is not None:
        ema.load_state_dict(resume_ck["ema"])

    out = Path(a.output_dir)
    if _rank0(rank):
        out.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    val_loader = None
    if _rank0(rank):
        val_loader = DataLoader(
            val_ds,
            batch_size=a.batch_size,
            shuffle=False,
            num_workers=a.num_workers,
            collate_fn=collate_msp_wm,
            drop_last=False,
            pin_memory=True,
        )
        if resume_ck is None:
            initial_val = validate_fm(
                ema.model,
                val_loader,
                device,
                use_amp=bool(a.amp),
                seed=a.seed + 100000,
            )
            history.append({
                "epoch": 0,
                "global_step": 0,
                "train": None,
                "val_ema": initial_val,
            })
            print("initial_validation", json.dumps(history[-1]))
        else:
            print(
                f"resumed v9 DDP full M from epoch={start_epoch} step={global_step} "
                f"to epoch={a.run_until_epoch}"
            )
        print(json.dumps({
            "ddp": DDP_PROTOCOL,
            "world_size": world_size,
            "global_batch_size": a.batch_size,
            "per_rank_batch_size": local_batch,
            "train_samples": len(train_ds),
            "steps_per_epoch": steps_per_epoch,
            "padded_draws_per_epoch": padded_draws,
            "padding_draws": padded_draws - len(train_ds),
        }))
    dist.barrier()

    milestones = sorted(set(int(x) for x in a.milestone_epochs))
    model.train()
    for epoch in range(start_epoch + 1, int(a.run_until_epoch) + 1):
        indices, sampling_attempt = _draw_global_epoch_indices(
            train_ds,
            seed=a.seed,
            epoch=epoch,
            global_batch_size=a.batch_size,
            world_size=world_size,
            steps_per_epoch=steps_per_epoch,
        )
        batch_sampler = RankBatchSampler(
            indices,
            global_batch_size=a.batch_size,
            rank=rank,
            world_size=world_size,
        )
        loader = DataLoader(
            train_ds,
            batch_sampler=batch_sampler,
            num_workers=a.num_workers,
            collate_fn=collate_msp_wm,
            pin_memory=True,
        )
        if len(loader) != steps_per_epoch:
            raise RuntimeError("DDP epoch loader length differs from frozen schedule")

        epoch_obj_sum = 0.0
        epoch_uniform_sum = 0.0
        epoch_cos_sum = 0.0
        epoch_motion_sum = 0.0
        last_train = None

        for local_step, batch in enumerate(loader):
            if int(global_step) != (epoch - 1) * steps_per_epoch + local_step:
                raise RuntimeError("global step drift before DDP optimizer step")
            lr_ratio = _lr_ratio(
                int(global_step),
                total_schedule_steps,
                float(a.warmup_fraction),
                float(a.min_lr_ratio),
            )
            lrs = _set_lr(optimizer, lr_ratio)
            prepared = prepare_batch(batch, device)
            if prepared is None:
                # The sampler checks this from zero-route metadata before the epoch.
                raise RuntimeError(
                    f"rank {rank} unexpectedly received an all-zero-route local batch at step {global_step}"
                )
            motion_mask = _crop_motion_mask(sidecar, batch, prepared, device)
            source_noise, t_override = _step_noise_and_t(
                prepared,
                seed=a.seed,
                global_step=global_step,
                rank=rank,
            )

            optimizer.zero_grad(set_to_none=True)
            fm_loss, info, squared_error = ddp(
                prepared,
                source_noise,
                t_override,
                bool(a.amp),
            )
            total_loss, global_obj, global_uniform, global_motion = _global_weighted_loss(
                squared_error,
                motion_mask,
                float(a.motion_weight_lambda),
                world_size,
            )
            if not torch.isfinite(total_loss):
                raise RuntimeError(
                    f"non-finite v9 DDP loss epoch={epoch} global_step={global_step} rank={rank}"
                )
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(a.grad_clip))
            optimizer.step()
            ema_decay_now = ema.update(model)

            nwin = int(prepared["history"].shape[0])
            global_cos = _global_macro(float(info["cosine"]), nwin, device)
            global_step += 1
            epoch_obj_sum += global_obj
            epoch_uniform_sum += global_uniform
            epoch_cos_sum += global_cos
            epoch_motion_sum += global_motion
            last_train = {
                "objective": global_obj,
                "uniform_fm_loss": global_uniform,
                "fm_cosine": global_cos,
                "motion_cell_fraction_batch": global_motion,
                "effective_motion_weight_mass_batch": effective_motion_weight_mass(
                    global_motion, float(a.motion_weight_lambda)
                ),
                "physics_authority": float(model.transition.physics_fusion.authority.cpu()),
                "grad_norm_before_clip": float(torch.as_tensor(grad_norm).detach().cpu()),
                "ema_decay": float(ema_decay_now),
                "lr_ratio": float(lr_ratio),
                "lrs": lrs,
            }
            if _rank0(rank) and (global_step == 1 or global_step % 100 == 0):
                print(
                    f"epoch={epoch} step={global_step}/{total_schedule_steps} "
                    f"obj={global_obj:.6f} uniform={global_uniform:.6f} "
                    f"cos={global_cos:+.4f} motion={100*global_motion:.3f}% "
                    f"phys={last_train['physics_authority']:+.5f} "
                    f"grad={last_train['grad_norm_before_clip']:.4f} lr={lrs}"
                )

        expected_step = epoch * steps_per_epoch
        if global_step != expected_step:
            raise RuntimeError(f"DDP global step/epoch mismatch: {global_step} != {expected_step}")

        dist.barrier()
        if _rank0(rank):
            val = validate_fm(
                ema.model,
                val_loader,
                device,
                use_amp=bool(a.amp),
                seed=a.seed + 100000,
            )
            sampled_ids = [str(train_ds.entries[i]["sample_id"]) for i in indices]
            sampled_scenes = [str(train_ds.entries[i]["scene_name"]) for i in indices]
            epoch_train = {
                "mean_weighted_objective": epoch_obj_sum / steps_per_epoch,
                "mean_uniform_fm_loss": epoch_uniform_sum / steps_per_epoch,
                "mean_fm_cosine": epoch_cos_sum / steps_per_epoch,
                "mean_motion_cell_fraction": epoch_motion_sum / steps_per_epoch,
                "global_batches": steps_per_epoch,
                "global_sampled_draws": len(indices),
                "padding_draws": len(indices) - len(train_ds),
                "sampling_retry_attempt": sampling_attempt,
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

            payload = base._payload(
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
            payload = _decorate_payload(
                payload,
                world_size=world_size,
                global_batch=a.batch_size,
                local_batch=local_batch,
            )
            base._save_payload(out, "latest.pt", payload)
            if epoch in milestones:
                base._save_payload(out, f"epoch_{epoch:04d}.pt", payload)
                print("saved_milestone", out / f"epoch_{epoch:04d}.pt")

            report = {
                "protocol": PROTOCOL,
                "ddp": payload["ddp"],
                "initialization": "released_occfm_fut_checkpoint",
                "objective": "normalized_true_motion_weighted_native_flow_matching_velocity_mse",
                "motion_lambda": float(a.motion_weight_lambda),
                "full_train_samples": len(train_ds),
                "full_train_scenes": int(train_ds.metadata.get("num_unique_scenes", 0)),
                "train_routed_motion_fraction": sidecar.metadata.get("routed_top2_motion_fraction"),
                "schedule": schedule,
                "run_until_epoch": int(a.run_until_epoch),
                "milestone_epochs": milestones,
                "official_transition_reuse_fraction": official_reuse_fraction,
                "optimizer_contract": optimizer_info,
                "history": history,
                "checkpoint_policy": {
                    "rank0_only": True,
                    "milestones": [f"epoch_{x:04d}.pt" for x in milestones if x <= epoch],
                    "latest": True,
                    "last": epoch == int(a.run_until_epoch),
                    "best": False,
                },
                "decision": (
                    "Run unchanged NFE=10 deployment diagnostics at epoch 1/3/5. "
                    "Continue to epoch10 only if deployment and physical metrics still improve."
                ),
            }
            (out / "training_report.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8"
            )
        dist.barrier()

    if _rank0(rank):
        final = torch.load(out / "latest.pt", map_location="cpu", weights_only=False)
        base._save_payload(out, "last.pt", final)
        print("full_data_ddp_phase_complete", out / "last.pt")
    dist.barrier()


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
    p.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="GLOBAL batch size across all DDP ranks; two GPUs => 4 samples/rank at batch-size 8",
    )
    p.add_argument("--num-workers", type=int, default=4, help="DataLoader workers PER RANK")
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

    rank, world_size, local_rank, device = _init_dist()
    try:
        _worker(a, rank, world_size, local_rank, device)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
