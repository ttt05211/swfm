#!/usr/bin/env python3
"""Paired short continuation for V17-RL legacy vs exact-native footprints.

Both B-C and B-S start from the same frozen V17-RL epoch-5 checkpoint, restore
its AdamW state, use the same cache/source order, same paired shuffle seed,
same batch size, same residual/existence objectives, and the same remaining
cosine LR schedule.  The treatment changes only the overlap-supervision
footprint and its pre-declared gradient-scale calibration:

  B-C: legacy 0.8 m V17 t0 target-source mask, lambda_overlap = 0.25
  B-S: exact Strong-source 0.4 m XY footprint, lambda_overlap = 0.175

The native arm deliberately keeps the *legacy* target-source mask as a model
input.  Therefore this experiment changes the loss footprint, not the V17
representation.  Both arms also use the same legacy overlap eligibility set so
native coverage cannot silently add training labels.

This script is for the fixed-time paired experiment; it saves every continuation
epoch and does not select a winner by ADE, SoftCH, or any other proxy.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from real_motion.local_st_world_model_v17 import (
    MODEL_PROTOCOL_V17,
    LocalSpatialTemporalWorldModelV17,
    config_from_mapping_v17,
    soft_transport_overlap_loss,
)
from real_motion.motion_transport import FEATURE_DIM, FUTURE_FRAMES, motion_transport_loss
from real_motion.source_footprint import NATIVE_SOURCE_FOOTPRINT_CONTRACT
from tools.real_motion.train_p0_f9_v17_local_stwm import (
    _cat_mean,
    _latest_errors,
    flatten_supervised,
    forward_model,
    load_cache,
    true_moving_mask,
    unpack,
)
from tools.real_motion.train_p0_f9_v17_local_stwm_fast import _duration, _gpu_mem, _progress

PROTOCOL = "p0_f9_v17_native_footprint_paired_continuation_v1"
NATIVE_FIELD = "native_source_footprint_mask"
CONTROL_OVERLAP_WEIGHT = 0.25
NATIVE_OVERLAP_WEIGHT = 0.175
EXPECTED_START_EPOCH = 5
ARMS = ("B-C", "B-S")


def flatten_pair(records):
    flat = flatten_supervised(records)
    native = []
    for rec in records:
        if NATIVE_FIELD not in rec:
            raise RuntimeError(f"cache record {rec.get('sample_id', '?')} lacks {NATIVE_FIELD}")
        sup = rec["supervised_source"].bool()
        if bool(sup.any()):
            native.append(rec[NATIVE_FIELD][sup].to(torch.uint8))
    if not native:
        raise RuntimeError("native-footprint cache has no supervised sources")
    flat[NATIVE_FIELD] = torch.cat(native, dim=0)
    if int(flat[NATIVE_FIELD].shape[0]) != int(flat["features"].shape[0]):
        raise RuntimeError("native footprint/source flattening lost alignment")
    return flat


def make_pair_dataset(flat):
    base = (
        flat["features"], flat["local_semantic_tube"], flat["kta_displacement_xy_m"],
        flat["target_residual_xy_m"], flat["target_displacement_xy_m"], flat["existence"],
        flat["target_valid"], flat["supervised_source"], flat["source_class_id"],
        flat["frame_motion_features"], flat["target_source_mask_tube"],
    )
    return TensorDataset(*base, flat[NATIVE_FIELD])


def unpack_pair(raw, device):
    b = unpack(raw[:-1], device)
    b[NATIVE_FIELD] = raw[-1].to(device, non_blocking=True)
    return b


class PairCUDAPrefetcher:
    def __init__(self, loader, device: torch.device):
        if device.type != "cuda":
            raise ValueError("CUDA prefetcher requires a CUDA device")
        self._it = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self.next_batch = None
        self._preload()

    def _preload(self):
        try:
            raw = next(self._it)
        except StopIteration:
            self.next_batch = None
            return
        with torch.cuda.stream(self.stream):
            self.next_batch = unpack_pair(raw, self.device)

    def __iter__(self):
        return self

    def __next__(self):
        if self.next_batch is None:
            raise StopIteration
        current = torch.cuda.current_stream(self.device)
        current.wait_stream(self.stream)
        batch = self.next_batch
        for value in batch.values():
            if torch.is_tensor(value) and value.is_cuda:
                value.record_stream(current)
        self._preload()
        return batch


def _batch_iterator(loader, device, cuda_prefetch: bool):
    if cuda_prefetch and device.type == "cuda":
        return PairCUDAPrefetcher(loader, device)
    return (unpack_pair(raw, device) for raw in loader)


def _loader(dataset, *, batch_size: int, shuffle: bool, generator, num_workers: int,
            prefetch_factor: int, pin_memory: bool):
    kwargs = dict(
        dataset=dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        drop_last=False,
    )
    if generator is not None:
        kwargs["generator"] = generator
    if int(num_workers) > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(**kwargs)


def _footprint_contract(batch, arm: str, *, legacy_resolution_m: float, native_resolution_m: float):
    legacy = batch["target_source_mask_tube"][:, -1].float()
    legacy_present = legacy.flatten(1).any(dim=1)
    overlap_valid = batch["target_valid"].bool() & legacy_present[:, None]
    if arm == "B-C":
        return legacy, overlap_valid, float(legacy_resolution_m), CONTROL_OVERLAP_WEIGHT
    if arm == "B-S":
        native = batch[NATIVE_FIELD].float()
        native_present = native.flatten(1).any(dim=1)
        if bool((legacy_present & ~native_present).any()):
            raise RuntimeError("native footprint is empty for a legacy-overlap-eligible source")
        return native, overlap_valid, float(native_resolution_m), NATIVE_OVERLAP_WEIGHT
    raise ValueError(f"unknown arm {arm}")


def paired_objective(outputs, batch, arm: str, *, legacy_resolution_m: float, native_resolution_m: float):
    base, parts = motion_transport_loss(outputs, batch)
    footprint, overlap_valid, resolution, weight = _footprint_contract(
        batch,
        arm,
        legacy_resolution_m=legacy_resolution_m,
        native_resolution_m=native_resolution_m,
    )
    overlap, overlap_stats = soft_transport_overlap_loss(
        outputs["residual_xy_m"].float(),
        batch["target_residual_xy_m"].float(),
        footprint,
        overlap_valid,
        patch_resolution_m=resolution,
    )
    total = base + float(weight) * overlap
    result = dict(parts)
    result.update(overlap_stats)
    result.update({
        "transport_overlap_loss": float(overlap.detach().cpu()),
        "objective_loss": float(total.detach().cpu()),
        "overlap_weight": float(weight),
        "overlap_resolution_m": float(resolution),
        "overlap_eligible_labels": int(overlap_valid.sum().item()),
    })
    return total, result


def eval_pair(model, loader, device, *, arm: str, amp: bool,
              legacy_resolution_m: float, native_resolution_m: float):
    model.eval()
    objective_sum = base_sum = traj_sum = exist_sum = overlap_sum = 0.0
    soft_iou_num = soft_iou_den = 0.0
    n_batches = 0
    kta_all=[]; learned_all=[]; kta_moving=[]; learned_moving=[]
    kta_fde=[]; learned_fde=[]; kta_moving_fde=[]; learned_moving_fde=[]
    horizon = {1:[[],[]], 3:[[],[]], 5:[[],[]]}
    exist_correct=exist_total=tp=fp=fn=0
    eligible_total = 0
    with torch.no_grad():
        for raw in loader:
            b = unpack_pair(raw, device)
            out = forward_model(model, b, use_representation=True, amp=amp, device=device)
            loss, parts = paired_objective(
                out, b, arm,
                legacy_resolution_m=legacy_resolution_m,
                native_resolution_m=native_resolution_m,
            )
            objective_sum += float(loss)
            base_sum += float(parts["loss"])
            traj_sum += float(parts["trajectory_smooth_l1"])
            exist_sum += float(parts["existence_bce"])
            overlap_sum += float(parts["transport_overlap_loss"])
            eligible_total += int(parts["overlap_eligible_labels"])
            n_batches += 1
            labels = int(parts["transport_overlap_labels"])
            if labels > 0 and np.isfinite(float(parts["transport_soft_iou"])):
                soft_iou_num += float(parts["transport_soft_iou"]) * labels
                soft_iou_den += labels

            valid = b["target_valid"].bool()
            moving = true_moving_mask(b["target_displacement_xy_m"], valid)
            target = b["target_residual_xy_m"].to(out["residual_xy_m"].dtype)
            kd = torch.linalg.vector_norm(target.float(), dim=-1)
            ld = torch.linalg.vector_norm((out["residual_xy_m"] - target).float(), dim=-1)
            if bool(valid.any()):
                kta_all.append(kd[valid].cpu()); learned_all.append(ld[valid].cpu())
            if bool(moving.any()):
                kta_moving.append(kd[moving].cpu()); learned_moving.append(ld[moving].cpu())
            aa, zz = _latest_errors(kd, ld, valid); kta_fde.extend(aa); learned_fde.extend(zz)
            aa, zz = _latest_errors(kd, ld, moving); kta_moving_fde.extend(aa); learned_moving_fde.extend(zz)
            for hi in horizon:
                m = moving[:, hi]
                if bool(m.any()):
                    horizon[hi][0].append(kd[:, hi][m].cpu())
                    horizon[hi][1].append(ld[:, hi][m].cpu())

            pred_exist = out["existence_logits"] >= 0.0
            gt_exist = b["existence"] > 0.5
            exist_correct += int((pred_exist == gt_exist).sum())
            exist_total += int(gt_exist.numel())
            tp += int((pred_exist & gt_exist).sum())
            fp += int((pred_exist & ~gt_exist).sum())
            fn += int((~pred_exist & gt_exist).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    report = {
        "objective_loss": objective_sum / max(n_batches, 1),
        "base_motion_loss": base_sum / max(n_batches, 1),
        "trajectory_smooth_l1": traj_sum / max(n_batches, 1),
        "existence_bce": exist_sum / max(n_batches, 1),
        "transport_overlap_loss": overlap_sum / max(n_batches, 1),
        "transport_soft_iou": soft_iou_num / max(soft_iou_den, 1) if soft_iou_den else float("nan"),
        "overlap_eligible_labels": eligible_total,
        "kta_ade_m": _cat_mean(kta_all),
        "learned_ade_m": _cat_mean(learned_all),
        "kta_fde_m": float(np.mean(kta_fde)) if kta_fde else float("nan"),
        "learned_fde_m": float(np.mean(learned_fde)) if learned_fde else float("nan"),
        "true_moving_kta_ade_m": _cat_mean(kta_moving),
        "true_moving_learned_ade_m": _cat_mean(learned_moving),
        "true_moving_kta_fde_m": float(np.mean(kta_moving_fde)) if kta_moving_fde else float("nan"),
        "true_moving_learned_fde_m": float(np.mean(learned_moving_fde)) if learned_moving_fde else float("nan"),
        "existence_accuracy": exist_correct / max(exist_total, 1),
        "existence_precision": precision,
        "existence_recall": recall,
        "existence_f1": 2 * precision * recall / max(precision + recall, 1e-12),
    }
    for hi, seconds in ((1, 1.0), (3, 2.0), (5, 3.0)):
        report[f"true_moving_{seconds:g}s_kta_ade_m"] = _cat_mean(horizon[hi][0])
        report[f"true_moving_{seconds:g}s_learned_ade_m"] = _cat_mean(horizon[hi][1])
    return report


def _lr_scale(step: int, total_steps: int) -> float:
    frac = min(max(float(step) / max(int(total_steps), 1), 0.0), 1.0)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * frac))


def _optimizer_step_range(optimizer) -> tuple[int, int]:
    vals = []
    for state in optimizer.state.values():
        if "step" not in state:
            continue
        x = state["step"]
        vals.append(int(x.item()) if torch.is_tensor(x) else int(x))
    return (min(vals), max(vals)) if vals else (0, 0)


def save_pair_ckpt(path, model, optimizer, *, global_epoch: int, arm: str, resume_ckpt,
                   train_meta, val_meta, val_report, args, overlap_weight: float):
    torch.save({
        "protocol": MODEL_PROTOCOL_V17,
        "epoch": int(global_epoch),
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "feature_dim": FEATURE_DIM,
        "future_frames": FUTURE_FRAMES,
        "model_config": asdict(model.v17_config),
        "variant": "RL",
        "use_representation": True,
        "overlap_weight": float(overlap_weight),
        "args": vars(args),
        "train_cache_metadata": train_meta,
        "val_cache_metadata": val_meta,
        "val_report": val_report,
        "paired_continuation": {
            "protocol": PROTOCOL,
            "arm": arm,
            "resume_checkpoint": str(Path(resume_ckpt).resolve()),
            "start_epoch": EXPECTED_START_EPOCH,
            "legacy_overlap_weight": CONTROL_OVERLAP_WEIGHT,
            "native_overlap_weight": NATIVE_OVERLAP_WEIGHT,
            "same_legacy_overlap_eligibility": True,
            "representation_mask_unchanged": True,
        },
    }, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-cache", required=True, help="V17 train cache augmented with native footprints")
    p.add_argument("--val-cache", required=True, help="V17 val cache augmented with native footprints")
    p.add_argument("--resume-checkpoint", required=True, help="frozen V17-RL epoch-5 checkpoint")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--continuation-epochs", type=int, default=5)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--paired-shuffle-seed", type=int, default=20260910)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-cuda-prefetch", action="store_true")
    a = p.parse_args()
    if a.continuation_epochs <= 0 or a.num_workers < 0 or a.prefetch_factor <= 0 or a.log_every <= 0:
        raise ValueError("invalid continuation/runtime arguments")

    random.seed(a.paired_shuffle_seed)
    np.random.seed(a.paired_shuffle_seed)
    torch.manual_seed(a.paired_shuffle_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.paired_shuffle_seed)
    device = torch.device(a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and not bool(a.no_amp)
    cuda_prefetch = device.type == "cuda" and not bool(a.no_cuda_prefetch)

    train_meta, train_records = load_cache(a.train_cache)
    val_meta, val_records = load_cache(a.val_cache)
    for name, meta in (("train", train_meta), ("val", val_meta)):
        if meta.get("native_source_footprint_contract") != NATIVE_SOURCE_FOOTPRINT_CONTRACT:
            raise RuntimeError(f"{name} cache is not native-footprint augmented")
    train = flatten_pair(train_records)
    val = flatten_pair(val_records)
    scene_overlap = sorted(set(train["scene_ids"]) & set(val["scene_ids"]))
    if scene_overlap:
        raise RuntimeError(f"train/val scene overlap: {scene_overlap[:5]}")

    legacy_resolution = float(train_meta.get("patch_resolution_m", 0.8))
    native_resolution = float(train_meta["native_source_footprint_resolution_m"])
    if abs(native_resolution - float(val_meta["native_source_footprint_resolution_m"])) > 1e-12:
        raise RuntimeError("train/val native footprint resolutions differ")
    if int(train[NATIVE_FIELD].shape[-1]) != int(train_meta["native_source_footprint_hw"]):
        raise RuntimeError("train native footprint shape disagrees with metadata")
    if tuple(train[NATIVE_FIELD].shape[1:]) != tuple(val[NATIVE_FIELD].shape[1:]):
        raise RuntimeError("train/val native footprint shapes differ")

    ck = torch.load(a.resume_checkpoint, map_location="cpu", weights_only=False)
    if ck.get("protocol") != MODEL_PROTOCOL_V17 or str(ck.get("variant")) != "RL":
        raise RuntimeError("resume checkpoint must be a V17-RL checkpoint")
    if not bool(ck.get("use_representation", False)):
        raise RuntimeError("resume checkpoint must use the V17 representation")
    start_epoch = int(ck.get("epoch", -1))
    if start_epoch != EXPECTED_START_EPOCH:
        raise RuntimeError(f"paired protocol requires epoch {EXPECTED_START_EPOCH}, got {start_epoch}")
    if abs(float(ck.get("overlap_weight", -1.0)) - CONTROL_OVERLAP_WEIGHT) > 1e-12:
        raise RuntimeError("resume checkpoint is not the frozen RL lambda=0.25 checkpoint")
    ck_args = ck.get("args") or {}
    batch_size = int(ck_args.get("batch_size", 256))
    base_lr = float(ck_args.get("lr", 5e-4))
    weight_decay = float(ck_args.get("weight_decay", 1e-4))
    original_epochs = int(ck_args.get("epochs", 10))
    end_epoch = start_epoch + int(a.continuation_epochs)
    if end_epoch > original_epochs:
        raise RuntimeError(
            f"continuation would exceed original LR schedule: end={end_epoch}, original={original_epochs}"
        )

    gen = torch.Generator().manual_seed(int(a.paired_shuffle_seed))
    train_loader = _loader(
        make_pair_dataset(train), batch_size=batch_size, shuffle=True, generator=gen,
        num_workers=a.num_workers, prefetch_factor=a.prefetch_factor, pin_memory=device.type == "cuda",
    )
    val_loader = _loader(
        make_pair_dataset(val), batch_size=batch_size, shuffle=False, generator=None,
        num_workers=a.num_workers, prefetch_factor=a.prefetch_factor, pin_memory=device.type == "cuda",
    )

    cfg = config_from_mapping_v17(ck.get("model_config"))
    if not bool(cfg.use_representation):
        raise RuntimeError("checkpoint model_config disables the V17 representation")
    model = LocalSpatialTemporalWorldModelV17(cfg).to(device)
    model.load_state_dict(ck["state_dict"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    optimizer.load_state_dict(ck["optimizer"])

    batches = len(train_loader)
    start_step = start_epoch * batches
    total_steps = original_epochs * batches
    min_opt_step, max_opt_step = _optimizer_step_range(optimizer)
    if max_opt_step != start_step:
        raise RuntimeError(
            f"optimizer step mismatch: checkpoint max step={max_opt_step}, expected={start_step}; "
            "cache/source count or batch contract changed"
        )
    expected_lr = base_lr * _lr_scale(start_step, total_steps)
    actual_lr = float(optimizer.param_groups[0]["lr"])
    if not math.isclose(actual_lr, expected_lr, rel_tol=2e-6, abs_tol=1e-10):
        raise RuntimeError(f"resume LR mismatch: checkpoint={actual_lr} expected={expected_lr}")

    overlap_weight = CONTROL_OVERLAP_WEIGHT if a.arm == "B-C" else NATIVE_OVERLAP_WEIGHT
    out_dir = Path(a.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("paired initial validation ...", flush=True)
    init_started = time.perf_counter()
    init_report = eval_pair(
        model, val_loader, device, arm=a.arm, amp=amp,
        legacy_resolution_m=legacy_resolution, native_resolution_m=native_resolution,
    )
    init_seconds = time.perf_counter() - init_started
    prior_val = ck.get("val_report") or {}
    if "learned_ade_m" in prior_val and not math.isclose(
        float(init_report["learned_ade_m"]), float(prior_val["learned_ade_m"]),
        rel_tol=0.0, abs_tol=2e-4,
    ):
        raise RuntimeError(
            "resume-model/cache identity check failed: initial ADE does not reproduce checkpoint val ADE"
        )

    print(json.dumps({
        "protocol": PROTOCOL,
        "arm": a.arm,
        "start_epoch": start_epoch,
        "end_epoch": end_epoch,
        "resume_checkpoint": str(Path(a.resume_checkpoint).resolve()),
        "batch_size": batch_size,
        "train_sources": len(train["features"]),
        "val_sources": len(val["features"]),
        "paired_shuffle_seed": int(a.paired_shuffle_seed),
        "legacy_resolution_m": legacy_resolution,
        "native_resolution_m": native_resolution,
        "overlap_weight": overlap_weight,
        "optimizer_step_range": [min_opt_step, max_opt_step],
        "start_lr": actual_lr,
        "original_total_epochs": original_epochs,
        "amp_bfloat16": amp,
        "cuda_prefetch_one_batch": cuda_prefetch,
        "initial_val_seconds": init_seconds,
        "initial_val": init_report,
    }, indent=2), flush=True)

    history = []
    global_step = start_step
    run_started = time.perf_counter()
    for global_epoch in range(start_epoch + 1, end_epoch + 1):
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        epoch_started = time.perf_counter()
        running = torch.zeros((), device=device, dtype=torch.float32)
        seen = 0
        iterator = _batch_iterator(train_loader, device, cuda_prefetch=cuda_prefetch)
        for bi, b in enumerate(iterator, start=1):
            out = forward_model(model, b, use_representation=True, amp=amp, device=device)
            loss, _ = paired_objective(
                out, b, a.arm,
                legacy_resolution_m=legacy_resolution,
                native_resolution_m=native_resolution,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            global_step += 1
            next_lr = base_lr * _lr_scale(global_step, total_steps)
            for group in optimizer.param_groups:
                group["lr"] = next_lr
            running += loss.detach().float()
            seen += int(b["features"].shape[0])
            if bi == 1 or bi % int(a.log_every) == 0 or bi == batches:
                _progress(
                    epoch=global_epoch,
                    epochs=end_epoch,
                    batch=bi,
                    batches=batches,
                    samples=seen,
                    elapsed_s=time.perf_counter() - epoch_started,
                    loss=float(loss.detach().float().item()),
                    lr=float(optimizer.param_groups[0]["lr"]),
                    device=device,
                    final=(bi == batches),
                )

        train_seconds = time.perf_counter() - epoch_started
        train_loss = float((running / max(batches, 1)).item())
        print(f"paired validation global epoch {global_epoch} ...", flush=True)
        val_started = time.perf_counter()
        val_report = eval_pair(
            model, val_loader, device, arm=a.arm, amp=amp,
            legacy_resolution_m=legacy_resolution, native_resolution_m=native_resolution,
        )
        val_seconds = time.perf_counter() - val_started
        alloc, reserved, peak = _gpu_mem(device)
        row = {
            "global_epoch": int(global_epoch),
            "continuation_epoch": int(global_epoch - start_epoch),
            "train_objective_loss": train_loss,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_seconds": train_seconds,
            "validation_seconds": val_seconds,
            "train_sources_per_s": len(train["features"]) / max(train_seconds, 1e-9),
            "gpu_allocated_gb": alloc,
            "gpu_reserved_gb": reserved,
            "gpu_peak_allocated_gb": peak,
            **val_report,
        }
        history.append(row)
        print("=== PAIRED EPOCH SUMMARY ===", flush=True)
        print(json.dumps(row, indent=2), flush=True)
        save_pair_ckpt(
            out_dir / f"epoch_{global_epoch:04d}.pt",
            model,
            optimizer,
            global_epoch=global_epoch,
            arm=a.arm,
            resume_ckpt=a.resume_checkpoint,
            train_meta=train_meta,
            val_meta=val_meta,
            val_report=val_report,
            args=a,
            overlap_weight=overlap_weight,
        )
        save_pair_ckpt(
            out_dir / "latest.pt",
            model,
            optimizer,
            global_epoch=global_epoch,
            arm=a.arm,
            resume_ckpt=a.resume_checkpoint,
            train_meta=train_meta,
            val_meta=val_meta,
            val_report=val_report,
            args=a,
            overlap_weight=overlap_weight,
        )

    total_seconds = time.perf_counter() - run_started
    report = {
        "protocol": PROTOCOL,
        "arm": a.arm,
        "resume_checkpoint": str(Path(a.resume_checkpoint).resolve()),
        "start_epoch": start_epoch,
        "end_epoch": end_epoch,
        "overlap_weight": overlap_weight,
        "legacy_resolution_m": legacy_resolution,
        "native_resolution_m": native_resolution,
        "paired_shuffle_seed": int(a.paired_shuffle_seed),
        "batch_size": batch_size,
        "initial_val": init_report,
        "history": history,
        "total_seconds": total_seconds,
        "fixed_evaluation_epochs": [start_epoch, 8, end_epoch] if start_epoch < 8 < end_epoch else [start_epoch, end_epoch],
    }
    (out_dir / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("=== PAIRED CONTINUATION COMPLETE ===", flush=True)
    print(json.dumps({
        "arm": a.arm,
        "start_epoch": start_epoch,
        "end_epoch": end_epoch,
        "overlap_weight": overlap_weight,
        "total_time": _duration(total_seconds),
        "fixed_full_eval": [
            str(out_dir / "epoch_0008.pt") if end_epoch >= 8 else "",
            str(out_dir / f"epoch_{end_epoch:04d}.pt"),
        ],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
