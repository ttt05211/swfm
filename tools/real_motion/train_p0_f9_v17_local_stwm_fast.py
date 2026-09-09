#!/usr/bin/env python3
"""Runtime-optimized trainer for controlled V17 single-expert Local-STWM variants.

This entry point preserves the V17 optimization contract (same model, losses,
batch size, optimizer hyperparameters, shuffle seed and update schedule) while
improving data/runtime behavior that should not change the scientific variable:

- persistent DataLoader workers + configurable CPU prefetch;
- one-batch asynchronous CPU->GPU CUDA prefetch to use spare device memory and
  hide host-to-device copies;
- no per-step ``loss.item()`` synchronization (only at progress updates);
- dependency-free progress bar with elapsed/ETA, throughput, LR, losses and GPU
  allocated/reserved/peak memory;
- epoch-level train/validation timing in the saved training report.

For the controlled R/L/RL comparison, keep ``--batch-size 256``. Increasing the
batch size changes the number of optimizer updates per epoch and therefore is a
new optimization experiment rather than a pure runtime optimization.
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
from torch.utils.data import DataLoader

from real_motion.local_st_world_model_v17 import (
    MODEL_PROTOCOL_V17,
    LocalSTWMV17Config,
    LocalSpatialTemporalWorldModelV17,
)
from real_motion.motion_transport import FEATURE_DIM, FUTURE_FRAMES
from tools.real_motion.train_p0_f9_v17_local_stwm import (
    VARIANTS,
    eval_model,
    flatten_supervised,
    forward_model,
    load_cache,
    make_dataset,
    objective_loss,
    save_ckpt,
    unpack,
)


def _duration(seconds: float) -> str:
    seconds = max(0, int(round(float(seconds))))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _gpu_mem(device: torch.device) -> tuple[float, float, float]:
    if device.type != "cuda":
        return 0.0, 0.0, 0.0
    gb = 1024.0 ** 3
    return (
        torch.cuda.memory_allocated(device) / gb,
        torch.cuda.memory_reserved(device) / gb,
        torch.cuda.max_memory_allocated(device) / gb,
    )


def _progress(
    *, epoch: int, epochs: int, batch: int, batches: int, samples: int,
    elapsed_s: float, loss: float, lr: float, device: torch.device, final: bool = False,
) -> None:
    frac = batch / max(batches, 1)
    width = 28
    fill = min(width, int(round(frac * width)))
    bar = "=" * max(fill - 1, 0) + (">" if fill and fill < width else "")
    bar = (bar + "." * width)[:width]
    rate = samples / max(elapsed_s, 1e-9)
    eta = elapsed_s * (batches - batch) / max(batch, 1)
    alloc, reserv, peak = _gpu_mem(device)
    text = (
        f"\rE{epoch:02d}/{epochs:02d} [{bar}] {batch:4d}/{batches:<4d} "
        f"loss={loss:.4f} lr={lr:.2e} {rate:7.0f} src/s "
        f"elapsed={_duration(elapsed_s)} eta={_duration(eta)} "
        f"GPU={alloc:.1f}/{reserv:.1f}GB peak={peak:.1f}GB"
    )
    print(text, end="\n" if final else "", flush=True)


class CUDABatchPrefetcher:
    """One-batch asynchronous H2D prefetcher preserving batch/update semantics."""

    def __init__(self, loader, device: torch.device):
        if device.type != "cuda":
            raise ValueError("CUDA prefetch requires a CUDA device")
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
            self.next_batch = unpack(raw, self.device)

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


def _batch_iterator(loader, device: torch.device, *, cuda_prefetch: bool):
    if cuda_prefetch and device.type == "cuda":
        return CUDABatchPrefetcher(loader, device)
    return (unpack(raw, device) for raw in loader)


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-cache", required=True)
    p.add_argument("--val-cache", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--variant", choices=VARIANTS, required=True)
    p.add_argument("--epochs", type=int, default=10)
    # Keep 256 for R/L/RL scientific comparability. A larger value is accepted
    # for separate throughput/optimization experiments, but changes training.
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--semantic-dim", type=int, default=32)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--decoder-blocks", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--overlap-weight", type=float, default=0.25)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=20260909)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-cuda-prefetch", action="store_true")
    p.add_argument("--cudnn-benchmark", action="store_true",
                   help="allow cuDNN to benchmark fixed-shape convolutions; optional because it can change kernel selection")
    a = p.parse_args()
    if a.epochs <= 0 or a.batch_size <= 0 or a.lr <= 0 or a.overlap_weight < 0:
        raise ValueError("invalid optimization arguments")
    if a.num_workers < 0 or a.prefetch_factor <= 0 or a.log_every <= 0:
        raise ValueError("invalid runtime arguments")

    use_representation = a.variant in {"R", "RL"}
    effective_overlap = float(a.overlap_weight) if a.variant in {"L", "RL"} else 0.0

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    device = torch.device(a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and not bool(a.no_amp)
    cuda_prefetch = device.type == "cuda" and not bool(a.no_cuda_prefetch)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(a.cudnn_benchmark)

    train_meta, train_records = load_cache(a.train_cache)
    val_meta, val_records = load_cache(a.val_cache)
    train = flatten_supervised(train_records)
    val = flatten_supervised(val_records)
    overlap = sorted(set(train["scene_ids"]) & set(val["scene_ids"]))
    if overlap:
        raise RuntimeError(f"train/val scene overlap: {overlap[:5]}")
    tube_hw = int(train["local_semantic_tube"].shape[-1])
    patch_resolution = float(train_meta.get("patch_resolution_m", 0.8))
    if tuple(train["local_semantic_tube"].shape[1:]) != (6, tube_hw, tube_hw):
        raise RuntimeError("unexpected train tube shape")

    gen = torch.Generator().manual_seed(int(a.seed))
    train_loader = _loader(
        make_dataset(train), batch_size=a.batch_size, shuffle=True, generator=gen,
        num_workers=a.num_workers, prefetch_factor=a.prefetch_factor, pin_memory=device.type == "cuda",
    )
    val_loader = _loader(
        make_dataset(val), batch_size=a.batch_size, shuffle=False, generator=None,
        num_workers=a.num_workers, prefetch_factor=a.prefetch_factor, pin_memory=device.type == "cuda",
    )

    mcfg = LocalSTWMV17Config(
        d_model=a.d_model, semantic_dim=a.semantic_dim, heads=a.heads, blocks=a.blocks,
        decoder_blocks=a.decoder_blocks, tube_hw=tube_hw, use_representation=use_representation,
    )
    model = LocalSpatialTemporalWorldModelV17(mcfg).to(device)
    # Deliberately keep the same unfused AdamW implementation as R/L to isolate
    # RL as representation+loss rather than an optimizer-kernel experiment.
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    total_steps = max(1, a.epochs * len(train_loader))
    step = 0
    out_dir = Path(a.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_objective = float("inf")
    best_ade = float("inf")

    print("initial validation ...", flush=True)
    init_started = time.perf_counter()
    init_report = eval_model(
        model, val_loader, device, amp=amp, use_representation=use_representation,
        overlap_weight=effective_overlap, patch_resolution_m=patch_resolution,
    )
    init_seconds = time.perf_counter() - init_started
    if abs(float(init_report["learned_ade_m"]) - float(init_report["kta_ade_m"])) > 1e-5:
        raise RuntimeError("zero-init safety contract failed: fresh V17 is not KTA")
    param_count = sum(p.numel() for p in model.parameters())
    runtime = {
        "batch_size": int(a.batch_size),
        "num_workers": int(a.num_workers),
        "prefetch_factor": int(a.prefetch_factor),
        "persistent_workers": bool(a.num_workers > 0),
        "cuda_prefetch_one_batch": bool(cuda_prefetch),
        "cudnn_benchmark": bool(a.cudnn_benchmark),
        "log_every": int(a.log_every),
    }
    print(json.dumps({
        "protocol": MODEL_PROTOCOL_V17,
        "variant": a.variant,
        "use_representation": use_representation,
        "overlap_weight": effective_overlap,
        "model_config": asdict(mcfg),
        "parameters": param_count,
        "train_sources": len(train["features"]),
        "val_sources": len(val["features"]),
        "train_scenes": len(set(train["scene_ids"])),
        "val_scenes": len(set(val["scene_ids"])),
        "amp_bfloat16": amp,
        "runtime": runtime,
        "initial_val_seconds": init_seconds,
        "initial_val": init_report,
    }, indent=2), flush=True)

    run_started = time.perf_counter()
    for epoch in range(1, a.epochs + 1):
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        epoch_started = time.perf_counter()
        train_started = epoch_started
        running = torch.zeros((), device=device, dtype=torch.float32)
        last_loss = float("nan")
        seen = 0
        batches = len(train_loader)

        iterator = _batch_iterator(train_loader, device, cuda_prefetch=cuda_prefetch)
        for bi, b in enumerate(iterator, start=1):
            out = forward_model(model, b, use_representation=use_representation, amp=amp, device=device)
            loss, _ = objective_loss(
                out, b, overlap_weight=effective_overlap, patch_resolution_m=patch_resolution
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            step += 1
            frac = min(step / total_steps, 1.0)
            scale = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * frac))
            for group in optimizer.param_groups:
                group["lr"] = a.lr * scale
            running += loss.detach().float()
            seen += int(b["features"].shape[0])

            if bi == 1 or bi % int(a.log_every) == 0 or bi == batches:
                # Synchronize only at visible log points instead of every step.
                last_loss = float(loss.detach().float().item())
                _progress(
                    epoch=epoch, epochs=a.epochs, batch=bi, batches=batches, samples=seen,
                    elapsed_s=time.perf_counter() - train_started, loss=last_loss,
                    lr=float(optimizer.param_groups[0]["lr"]), device=device, final=(bi == batches),
                )

        train_seconds = time.perf_counter() - train_started
        train_loss = float((running / max(batches, 1)).item())
        print(f"validation epoch {epoch} ...", flush=True)
        val_started = time.perf_counter()
        val_report = eval_model(
            model, val_loader, device, amp=amp, use_representation=use_representation,
            overlap_weight=effective_overlap, patch_resolution_m=patch_resolution,
        )
        val_seconds = time.perf_counter() - val_started
        epoch_seconds = time.perf_counter() - epoch_started
        alloc, reserved, peak = _gpu_mem(device)
        row = {
            "epoch": epoch,
            "train_objective_loss": train_loss,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_seconds": train_seconds,
            "validation_seconds": val_seconds,
            "epoch_seconds": epoch_seconds,
            "train_sources_per_s": len(train["features"]) / max(train_seconds, 1e-9),
            "gpu_allocated_gb": alloc,
            "gpu_reserved_gb": reserved,
            "gpu_peak_allocated_gb": peak,
            **val_report,
        }
        history.append(row)
        print("=== EPOCH SUMMARY ===", flush=True)
        print(json.dumps(row, indent=2), flush=True)

        save_ckpt(
            out_dir / "latest.pt", model, optimizer, epoch, a, train_meta, val_meta, val_report,
            use_representation=use_representation, effective_overlap_weight=effective_overlap,
        )
        if epoch in {1, 5, 10, a.epochs}:
            save_ckpt(
                out_dir / f"epoch_{epoch:04d}.pt", model, optimizer, epoch, a, train_meta, val_meta, val_report,
                use_representation=use_representation, effective_overlap_weight=effective_overlap,
            )
        if float(val_report["objective_loss"]) < best_objective:
            best_objective = float(val_report["objective_loss"])
            save_ckpt(
                out_dir / "best.pt", model, optimizer, epoch, a, train_meta, val_meta, val_report,
                use_representation=use_representation, effective_overlap_weight=effective_overlap,
            )
        if float(val_report["learned_ade_m"]) < best_ade:
            best_ade = float(val_report["learned_ade_m"])
            save_ckpt(
                out_dir / "best_ade.pt", model, optimizer, epoch, a, train_meta, val_meta, val_report,
                use_representation=use_representation, effective_overlap_weight=effective_overlap,
            )

    total_seconds = time.perf_counter() - run_started
    report = {
        "protocol": MODEL_PROTOCOL_V17,
        "variant": a.variant,
        "model_config": asdict(mcfg),
        "parameters": param_count,
        "best_objective_loss": best_objective,
        "best_learned_ade_m": best_ade,
        "initial_val": init_report,
        "history": history,
        "train_sources": len(train["features"]),
        "val_sources": len(val["features"]),
        "runtime": runtime,
        "total_training_and_validation_seconds": total_seconds,
    }
    (out_dir / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("=== TRAINING COMPLETE ===", flush=True)
    print(json.dumps({
        "variant": a.variant,
        "epochs": a.epochs,
        "total_time": _duration(total_seconds),
        "best_objective_loss": best_objective,
        "best_learned_ade_m": best_ade,
        "best_checkpoint": str(out_dir / "best.pt"),
        "best_ade_checkpoint": str(out_dir / "best_ade.pt"),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
