#!/usr/bin/env python3
"""Single warmup->joint trainer for V20 unified transport completion."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from real_motion.runtime_config import add_config_args, load_runtime_config, make_prepare_config
from real_motion.strong_w2det import StrongW2DetConfig
from real_motion.v20_unified_loss import compute_training_loss
from real_motion.v20_unified_model import V20UnifiedTransportCompletion
from real_motion.v20_unified_training import (
    TrainerProgress,
    build_optimizer,
    checkpoint_payload,
    configure_training_phase,
    file_sha256,
    load_model_checkpoint,
    restore_training_checkpoint,
    save_checkpoint,
    scaler_step_succeeded,
    set_optimizer_phase_lrs,
    training_phase,
)
from tools.real_motion.eval_p0_f9_v20_unified import evaluate_model, load_clean_v18
from tools.real_motion.v20_unified_common import (
    CachedSource,
    ComponentLRU,
    first_stage_forward,
    hard_render_transport,
    lattice_from_dict,
    load_stage1_rows,
    load_v18_cache,
    prepare_unified_window,
    training_completion_inputs,
)

PROTOCOL = "p0_f9_v20_unified_transport_completion_train_v1"


def _autocast(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _transport_batch(record: dict, gpu: dict, device: torch.device) -> dict:
    names = (
        "target_source_residual_xy_m",
        "se2_target_valid",
        "existence",
        "supervised_source",
        "target_yaw_rad",
        "yaw_enabled",
        "yaw_label_valid",
        "target_source_displacement_xy_m",
    )
    missing = [name for name in names if name not in record]
    if missing:
        raise RuntimeError(f"V18 source cache lacks Clean-E14 targets: {missing}")
    batch = {name: record[name].to(device) for name in names}
    batch["kta_displacement_xy_m"] = gpu["kta"]
    batch["target_source_mask_tube"] = gpu["source_mask"]
    return batch


def _lr_scale(successful_updates: int, max_updates: int) -> float:
    fraction = min(max(float(successful_updates) / max(int(max_updates), 1), 0.0), 1.0)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * fraction))


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_config_args(parser)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--stage1-cache", required=True)
    parser.add_argument("--dev-cache", required=True)
    parser.add_argument("--dev-stage1-cache", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--info-pkl", required=True)
    parser.add_argument(
        "--dev-info-pkl",
        default="",
        help="Optional scene-disjoint dev info pickle; defaults to --info-pkl.",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--resume", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--batch-size", type=int, default=1, choices=[1])
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--warmup-updates", type=int, default=128)
    parser.add_argument("--max-updates", type=int, default=1024)
    parser.add_argument("--monitor-every", type=int, default=128)
    parser.add_argument("--monitor-windows", type=int, default=128)
    parser.add_argument("--new-lr", type=float, default=2.0e-4)
    parser.add_argument("--v18-lr", type=float, default=2.0e-5)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--clip-grad", type=float, default=5.0)
    parser.add_argument("--completion-weight", type=float, default=1.0)
    parser.add_argument("--patch-resolution-m", type=float, default=0.8)
    parser.add_argument("--alignment-workers", type=int, default=6)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one real-data update and one full-support dev window.",
    )
    args = parser.parse_args()
    if args.smoke:
        args.max_updates = 1
        args.grad_accum = 1
        args.monitor_every = 1
        args.monitor_windows = 1
    if min(
        int(args.grad_accum),
        int(args.max_updates),
        int(args.monitor_every),
        int(args.monitor_windows),
    ) <= 0:
        raise ValueError("update, accumulation and monitor counts must be positive")

    _seed_everything(int(args.seed))
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    amp = device.type == "cuda" and not bool(args.no_amp)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    pcfg = make_prepare_config(load_runtime_config(args.config, args.override))
    _, train_records = load_v18_cache(args.train_cache)
    _, dev_records = load_v18_cache(args.dev_cache)
    if not dev_records:
        raise RuntimeError("empty development population")
    stage1_index_path, stage1_index, stage1_rows = load_stage1_rows(args.stage1_cache)
    dev_stage1_index_path, dev_stage1_index, dev_stage1_rows = load_stage1_rows(
        args.dev_stage1_cache
    )
    if stage1_index["native_grid"] != dev_stage1_index["native_grid"]:
        raise RuntimeError("train/dev native grids differ")
    missing = [
        (str(r["scene_name"]), str(r["t0_token"]))
        for r in train_records
        if (str(r["scene_name"]), str(r["t0_token"])) not in stage1_rows
    ]
    if missing:
        raise RuntimeError(f"Stage1 train cache misses rows: {missing[:5]}")
    order = list(range(len(train_records)))
    random.Random(int(args.seed)).shuffle(order)
    if not order:
        raise RuntimeError("empty training population")

    if args.resume:
        model, resume_checkpoint = load_model_checkpoint(args.resume, map_location="cpu")
        if resume_checkpoint["base_checkpoint_sha256"] != file_sha256(args.base_checkpoint):
            raise RuntimeError("resume/base checkpoint hash mismatch")
    else:
        _, v18 = load_clean_v18(args.base_checkpoint, device)
        model = V20UnifiedTransportCompletion(
            v18,
            coarse_lattice=lattice_from_dict(stage1_index["coarse_lattice"]),
        )
        resume_checkpoint = None
    model.to(device)
    optimizer = build_optimizer(
        model,
        new_lr=float(args.new_lr),
        v18_lr=float(args.v18_lr),
        weight_decay=float(args.weight_decay),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    progress_state = TrainerProgress()
    window_cursor = 0
    if resume_checkpoint is not None:
        progress_state = restore_training_checkpoint(
            resume_checkpoint, model=model, optimizer=optimizer, scaler=scaler
        )
        window_cursor = int(resume_checkpoint.get("scheduler", {}).get("window_cursor", 0))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source = CachedSource(args.dataroot, info_pkl=args.info_pkl, verbose=False)
    dev_source = CachedSource(
        args.dataroot,
        info_pkl=args.dev_info_pkl or args.info_pkl,
        verbose=False,
    )
    component_cache = ComponentLRU(maxsize=1024)
    strong_cfg = StrongW2DetConfig(free_label=int(pcfg.free_label))
    tile_generator = torch.Generator().manual_seed(int(args.seed) + 41)
    if resume_checkpoint is not None:
        saved_tile_rng = resume_checkpoint.get("scheduler", {}).get("tile_generator_state")
        if saved_tile_rng is None:
            raise RuntimeError("resume checkpoint lacks tile sampler RNG state")
        tile_generator.set_state(saved_tile_rng)
    monitor_log: list[dict] = []
    started = time.perf_counter()

    while progress_state.successful_updates < int(args.max_updates):
        phase = training_phase(
            progress_state.successful_updates, int(args.warmup_updates)
        )
        progress_state.phase = phase
        adapter_enabled = configure_training_phase(model, phase)
        scale = _lr_scale(progress_state.successful_updates, int(args.max_updates))
        set_optimizer_phase_lrs(
            optimizer,
            phase=phase,
            new_lr=float(args.new_lr),
            v18_lr=float(args.v18_lr),
            scale=scale,
        )
        optimizer.zero_grad(set_to_none=True)
        group_stats: list[dict] = []
        for _ in range(int(args.grad_accum)):
            record = train_records[order[window_cursor % len(order)]]
            window_cursor += 1
            key = (str(record["scene_name"]), str(record["t0_token"]))
            prepared = prepare_unified_window(
                record,
                stage1_rows[key],
                source=source,
                pcfg=pcfg,
                strong_cfg=strong_cfg,
                component_cache=component_cache,
                device=device,
            )
            try:
                with _autocast(device, amp):
                    first = first_stage_forward(
                        model, prepared, adapter_enabled=adapter_enabled
                    )
                current_transport = hard_render_transport(
                    model,
                    prepared,
                    first["transport"],
                    pcfg=pcfg,
                    strong_cfg=strong_cfg,
                    device=device,
                )
                with _autocast(device, amp):
                    logits, targets, masks, completion_report = training_completion_inputs(
                        model,
                        prepared,
                        first,
                        current_transport,
                        native_grid=stage1_index["native_grid"],
                        generator=tile_generator,
                    )
                    loss, stats = compute_training_loss(
                        first["transport"],
                        _transport_batch(record, prepared.state["gpu"], device),
                        logits,
                        targets,
                        masks,
                        patch_resolution_m=float(args.patch_resolution_m),
                        completion_weight=float(args.completion_weight),
                        graph_anchor=model.completion_head.weight,
                    )
                    scaled_loss = loss / float(args.grad_accum)
                scaler.scale(scaled_loss).backward()
                stats["tiles"] = int(completion_report["tiles"])
                group_stats.append(stats)
            finally:
                prepared.release()

        progress_state.attempted_updates += 1
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(args.clip_grad)
        )
        finite = bool(torch.isfinite(torch.as_tensor(grad_norm)).item())
        succeeded = scaler_step_succeeded(scaler, optimizer) if finite else False
        if not finite:
            scaler.update(float(scaler.get_scale()) / 2.0 if scaler.is_enabled() else 1.0)
        optimizer.zero_grad(set_to_none=True)
        if not succeeded:
            print(
                f"update_attempt={progress_state.attempted_updates} overflow/nonfinite; "
                "successful counter unchanged",
                flush=True,
            )
            continue
        progress_state.successful_updates += 1
        progress_state.phase = training_phase(
            progress_state.successful_updates, int(args.warmup_updates)
        )
        mean_loss = float(np.mean([row["loss"] for row in group_stats]))
        mean_comp = float(np.mean([row["completion_loss"] for row in group_stats]))
        voxels = int(sum(row["completion_voxels"] for row in group_stats))
        print(
            f"update={progress_state.successful_updates}/{args.max_updates} "
            f"phase={phase} loss={mean_loss:.6f} completion={mean_comp:.6f} "
            f"voxels={voxels} grad_norm={float(grad_norm):.4f}",
            flush=True,
        )

        should_monitor = (
            progress_state.successful_updates % int(args.monitor_every) == 0
            or progress_state.successful_updates == int(args.max_updates)
        )
        if not should_monitor:
            continue
        payload = checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler_state={
                "window_cursor": window_cursor,
                "successful_updates": progress_state.successful_updates,
                "lr_scale": scale,
                "tile_generator_state": tile_generator.get_state(),
            },
            scaler=scaler,
            progress=progress_state,
            base_checkpoint=args.base_checkpoint,
            manifest_paths={
                "train_cache": args.train_cache,
                "train_stage1_index": stage1_index_path,
                "dev_cache": args.dev_cache,
                "dev_stage1_index": dev_stage1_index_path,
            },
            config={
                **vars(args),
                "warmup_updates": int(args.warmup_updates),
                "native_grid": stage1_index["native_grid"],
            },
            repository_root=ROOT,
        )
        checkpoint_path = out_dir / f"update_{progress_state.successful_updates:04d}.pt"
        save_checkpoint(checkpoint_path, payload)

        # Monitoring is full-support/no enrichment and uses an independent
        # frozen reference loaded from the declared Clean-E14 checkpoint.
        _, frozen_v18 = load_clean_v18(args.base_checkpoint, device)
        monitor = evaluate_model(
            model=model,
            frozen_v18=frozen_v18,
            records=dev_records[: int(args.monitor_windows)],
            stage1_rows=dev_stage1_rows,
            source=dev_source,
            pcfg=pcfg,
            native_grid=dev_stage1_index["native_grid"],
            device=device,
            amp=amp,
            alignment_workers=int(args.alignment_workers),
            progress=False,
        )
        monitor.update(
            {
                "successful_updates": progress_state.successful_updates,
                "attempted_updates": progress_state.attempted_updates,
                "phase": progress_state.phase,
                "checkpoint": str(checkpoint_path.resolve()),
            }
        )
        monitor_log.append(monitor)
        (out_dir / "monitor.json").write_text(
            json.dumps(monitor_log, indent=2, allow_nan=True), encoding="utf-8"
        )
        print(json.dumps(monitor, allow_nan=True), flush=True)

    summary = {
        "protocol": PROTOCOL,
        "successful_updates": progress_state.successful_updates,
        "attempted_updates": progress_state.attempted_updates,
        "elapsed_seconds": time.perf_counter() - started,
        "final_checkpoint": str(
            (out_dir / f"update_{progress_state.successful_updates:04d}.pt").resolve()
        ),
        "smoke": bool(args.smoke),
        "real_data_run": True,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
