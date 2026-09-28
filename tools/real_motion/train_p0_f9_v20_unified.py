#!/usr/bin/env python3
"""Single warmup->joint trainer for V20 unified transport completion."""
from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
ROOT_STR = str(ROOT)
while ROOT_STR in sys.path:
    sys.path.remove(ROOT_STR)
sys.path.insert(0, ROOT_STR)

import numpy as np
import torch

from real_motion.runtime_config import (
    add_config_args,
    config_fingerprint,
    load_runtime_config,
    make_prepare_config,
)
from real_motion.strong_w2det import StrongW2DetConfig
from real_motion.v20_unified_loss import (
    COMPLETION_OBJECTIVE_PROTOCOL,
    DEFAULT_COMPLETION_SEMANTIC_WEIGHT,
    DEFAULT_PRESENCE_FOCAL_GAMMA,
    compute_training_loss,
)
from real_motion.v20_unified_model import (
    V20UnifiedConfig,
    V20UnifiedTransportCompletion,
)
from real_motion.v20_unified_training import (
    TRAIN_PROTOCOL,
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
    verify_resume_inputs,
)
from tools.real_motion.eval_p0_f9_v20_unified import evaluate_model, load_clean_v18
from tools.real_motion.v20_unified_common import (
    CachedSource,
    ComponentLRU,
    align_v18_records_to_stage1,
    first_stage_forward_batch,
    hard_render_transport,
    lattice_from_dict,
    load_stage1_rows,
    load_v18_cache,
    move_prepared_to_device,
    prepare_unified_window,
    stage1_manifest_paths,
    training_completion_inputs_batch,
)

PROTOCOL = TRAIN_PROTOCOL


def _autocast(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
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


def _materialize_scalar_stats(rows: list[dict]) -> list[dict]:
    """Convert deferred scalar CUDA statistics after the update sync."""
    out = []
    for row in rows:
        converted = {}
        for name, value in row.items():
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    raise RuntimeError(f"training statistic {name!r} is not scalar")
                scalar = value.detach().item()
                converted[name] = (
                    int(scalar)
                    if value.dtype
                    in {
                        torch.uint8,
                        torch.int8,
                        torch.int16,
                        torch.int32,
                        torch.int64,
                    }
                    else float(scalar)
                )
            else:
                converted[name] = value
        out.append(converted)
    return out


def _completion_head_gradient_norms(
    model: V20UnifiedTransportCompletion,
) -> torch.Tensor:
    """Per-class L2 gradient norms, retained until the update synchronization."""
    weight = model.completion_head.weight.grad
    bias = model.completion_head.bias.grad
    if weight is None:
        return model.completion_head.weight.new_zeros(18, dtype=torch.float32)
    squared = weight.detach().float().flatten(1).square().sum(dim=1)
    if bias is not None:
        squared = squared + bias.detach().float().square()
    return squared.sqrt()


def _lr_scale(successful_updates: int, max_updates: int) -> float:
    fraction = min(max(float(successful_updates) / max(int(max_updates), 1), 0.0), 1.0)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * fraction))


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _population_sha256(records, order=None) -> str:
    indices = range(len(records)) if order is None else order
    rows = [
        (
            str(records[i]["scene_name"]),
            str(records[i]["t0_token"]),
            str(records[i].get("sample_id", "")),
        )
        for i in indices
    ]
    payload = json.dumps(rows, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_monitor_history(
    out_dir: Path,
    resume_checkpoint: dict | None,
    successful_updates: int,
) -> list[dict]:
    if resume_checkpoint is None:
        return []
    candidates = [out_dir / "monitor.json"]
    saved_out = (resume_checkpoint.get("config") or {}).get("out_dir")
    if saved_out:
        saved_path = Path(saved_out) / "monitor.json"
        if saved_path not in candidates:
            candidates.append(saved_path)
    histories = [list(resume_checkpoint.get("monitor_history") or [])]
    for path in candidates:
        if not path.exists():
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, list):
            raise RuntimeError(f"monitor history is not a list: {path}")
        has_future = any(
            int(row.get("successful_updates", -1)) > int(successful_updates)
            for row in value
        )
        if has_future and path.resolve() == (out_dir / "monitor.json").resolve():
            raise RuntimeError(
                f"{path} contains monitor rows newer than the resume checkpoint; "
                "use a fresh --out-dir to preserve the existing run"
            )
        histories.append(
            [
                row for row in value
                if int(row.get("successful_updates", -1)) <= int(successful_updates)
            ]
        )
    return max(histories, key=len)


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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        choices=[1, 2, 4],
        help=(
            "Windows processed in one GPU micro-batch. --grad-accum remains "
            "the total windows per optimizer update and must be divisible by it."
        ),
    )
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
    parser.add_argument(
        "--presence-focal-gamma",
        type=float,
        default=DEFAULT_PRESENCE_FOCAL_GAMMA,
        help=(
            "Focal gamma for occupied-vs-free energy derived from the coupled "
            "18-class logits."
        ),
    )
    parser.add_argument(
        "--completion-semantic-weight",
        type=float,
        default=DEFAULT_COMPLETION_SEMANTIC_WEIGHT,
        help=(
            "Weight of positive-only 17-class semantic CE inside the coupled "
            "completion objective."
        ),
    )
    parser.add_argument("--patch-resolution-m", type=float, default=0.8)
    parser.add_argument("--alignment-workers", type=int, default=6)
    parser.add_argument(
        "--tile-decode-batch-size",
        type=int,
        default=8,
        help=(
            "Number of unique completion tiles decoded together. Larger values "
            "trade GPU memory for speed without changing sampled tiles or loss."
        ),
    )
    parser.add_argument(
        "--no-completion-checkpoint",
        action="store_true",
        help=(
            "Disable activation recomputation in the completion decoder. This "
            "uses more GPU memory but is faster and does not change model outputs."
        ),
    )
    parser.add_argument(
        "--runtime-query-chunk",
        type=int,
        default=32,
        help=(
            "Maximum full-volume completion queries retained per evaluation "
            "chunk. Larger values trade GPU memory for monitor/eval speed."
        ),
    )
    parser.add_argument(
        "--keep-checkpoints",
        type=int,
        default=3,
        help="Keep only the newest N update_*.pt checkpoints; 0 keeps all.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one warmup + one joint real-data update with one-window monitors.",
    )
    parser.add_argument(
        "--screen1024",
        action="store_true",
        help=(
            "Use a deterministic fixed 1024-window train subset for 1024 "
            "successful updates with grad_accum=4."
        ),
    )
    parser.add_argument(
        "--io-locality-order",
        action="store_true",
        help="Group selected train windows by scene for better token-cache locality.",
    )
    parser.add_argument(
        "--no-cpu-prefetch",
        action="store_true",
        help=(
            "Disable one-microbatch-ahead CPU window preparation. Prefetching "
            "does not alter sample order or tensors and normally hides raw-data "
            "loading, Strong-W2Det and baseline rasterization behind GPU work."
        ),
    )
    args = parser.parse_args()
    if args.smoke and args.screen1024:
        raise ValueError("--smoke and --screen1024 are mutually exclusive")
    if args.smoke:
        args.warmup_updates = 1
        args.max_updates = 2
        args.grad_accum = int(args.batch_size)
        args.monitor_every = 1
        args.monitor_windows = 1
    if args.screen1024:
        args.max_updates = 1024
        args.grad_accum = 4
        args.io_locality_order = True
    if min(
        int(args.grad_accum),
        int(args.max_updates),
        int(args.monitor_every),
        int(args.monitor_windows),
        int(args.tile_decode_batch_size),
        int(args.runtime_query_chunk),
    ) <= 0:
        raise ValueError(
            "update, accumulation, monitor and tile batch counts must be positive"
        )
    if int(args.grad_accum) % int(args.batch_size) != 0:
        raise ValueError("--grad-accum must be divisible by --batch-size")
    if not math.isfinite(float(args.presence_focal_gamma)) or float(
        args.presence_focal_gamma
    ) < 0.0:
        raise ValueError("--presence-focal-gamma must be finite and non-negative")
    if not math.isfinite(float(args.completion_semantic_weight)) or float(
        args.completion_semantic_weight
    ) < 0.0:
        raise ValueError(
            "--completion-semantic-weight must be finite and non-negative"
        )

    _seed_everything(int(args.seed))
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    amp = device.type == "cuda" and not bool(args.no_amp)
    if device.type == "cuda":
        if amp and not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                "V20 formal AMP requires CUDA BF16 support; use --no-amp "
                "for an explicit FP32 run"
            )
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    runtime_cfg = load_runtime_config(args.config, args.override)
    pcfg = make_prepare_config(runtime_cfg)
    _, train_records = load_v18_cache(args.train_cache)
    _, dev_records = load_v18_cache(args.dev_cache)
    stage1_index_path, stage1_index, stage1_rows = load_stage1_rows(args.stage1_cache)
    dev_stage1_index_path, dev_stage1_index, dev_stage1_rows = load_stage1_rows(
        args.dev_stage1_cache
    )
    dev_records, dev_alignment = align_v18_records_to_stage1(
        dev_records,
        dev_stage1_rows,
        population_name="training dev selection",
    )
    print(json.dumps({"dev_population_alignment": dev_alignment}), flush=True)
    if not dev_records:
        raise RuntimeError("empty development population")
    if int(args.monitor_windows) > len(dev_records):
        raise RuntimeError(
            f"monitor-windows={args.monitor_windows} exceeds aligned dev "
            f"population={len(dev_records)}"
        )
    overlap = sorted(
        {str(r["scene_name"]) for r in train_records}
        & {str(r["scene_name"]) for r in dev_records}
    )
    if overlap:
        raise RuntimeError(f"train/dev scene overlap: {overlap[:8]}")
    if args.screen1024:
        if len(train_records) < 1024:
            raise RuntimeError("screen1024 requires at least 1024 train windows")
        select = list(range(len(train_records)))
        random.Random(int(args.seed) + 1024).shuffle(select)
        selected = sorted(select[:1024])
        train_records = [train_records[i] for i in selected]
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
    if bool(args.io_locality_order):
        by_scene: dict[str, list[int]] = {}
        for i, record in enumerate(train_records):
            by_scene.setdefault(str(record["scene_name"]), []).append(i)
        scenes = list(by_scene)
        random.Random(int(args.seed)).shuffle(scenes)
        order = [i for scene in scenes for i in by_scene[scene]]
    else:
        random.Random(int(args.seed)).shuffle(order)
    if not order:
        raise RuntimeError("empty training population")

    manifest_paths = {
        "train_cache": args.train_cache,
        "dev_cache": args.dev_cache,
        "train_info_pkl": args.info_pkl,
        "dev_info_pkl": args.dev_info_pkl or args.info_pkl,
    }
    manifest_paths.update(
        stage1_manifest_paths("train_stage1", stage1_index_path, stage1_index)
    )
    manifest_paths.update(
        stage1_manifest_paths(
            "dev_stage1", dev_stage1_index_path, dev_stage1_index
        )
    )
    resume_contract = {
        "runtime_config_sha256": config_fingerprint(runtime_cfg, kind="resume"),
        "seed": int(args.seed),
        "batch_size": int(args.batch_size),
        "grad_accum": int(args.grad_accum),
        "warmup_updates": int(args.warmup_updates),
        "max_updates": int(args.max_updates),
        "monitor_every": int(args.monitor_every),
        "monitor_windows": int(args.monitor_windows),
        "new_lr": float(args.new_lr),
        "v18_lr": float(args.v18_lr),
        "weight_decay": float(args.weight_decay),
        "clip_grad": float(args.clip_grad),
        "completion_weight": float(args.completion_weight),
        "completion_objective_protocol": COMPLETION_OBJECTIVE_PROTOCOL,
        "presence_focal_gamma": float(args.presence_focal_gamma),
        "completion_semantic_weight": float(args.completion_semantic_weight),
        "patch_resolution_m": float(args.patch_resolution_m),
        "alignment_workers": int(args.alignment_workers),
        "tile_decode_batch_size": int(args.tile_decode_batch_size),
        "runtime_query_chunk": int(args.runtime_query_chunk),
        "checkpoint_completion_tiles": not bool(args.no_completion_checkpoint),
        "amp": bool(amp),
        "amp_dtype": "bfloat16" if amp else "float32",
        "device": str(device),
        "smoke": bool(args.smoke),
        "screen1024": bool(args.screen1024),
        "io_locality_order": bool(args.io_locality_order),
        "cpu_prefetch": not bool(args.no_cpu_prefetch),
        "train_windows": len(train_records),
        "train_order_sha256": _population_sha256(train_records, order),
        "dev_monitor_population_sha256": _population_sha256(
            dev_records[: int(args.monitor_windows)]
        ),
        "dataroot": str(Path(args.dataroot).resolve()),
        "tile_draws_per_horizon": 16,
        "positive_tile_draws": 8,
    }
    if args.resume:
        model, resume_checkpoint = load_model_checkpoint(args.resume, map_location="cpu")
        verify_resume_inputs(
            resume_checkpoint,
            base_checkpoint=args.base_checkpoint,
            manifest_paths=manifest_paths,
            resume_contract=resume_contract,
            coarse_lattice=stage1_index["coarse_lattice"],
        )
        saved_native = (resume_checkpoint.get("config") or {}).get("native_grid")
        if saved_native is not None and saved_native != stage1_index["native_grid"]:
            raise RuntimeError("resume/current native grid mismatch")
    else:
        _, v18 = load_clean_v18(args.base_checkpoint, device)
        model = V20UnifiedTransportCompletion(
            v18,
            coarse_lattice=lattice_from_dict(stage1_index["coarse_lattice"]),
            config=V20UnifiedConfig(
                tile_decode_batch_size=int(args.tile_decode_batch_size),
                runtime_query_chunk=int(args.runtime_query_chunk),
                checkpoint_completion_tiles=not bool(
                    args.no_completion_checkpoint
                ),
            ),
        )
        resume_checkpoint = None
    model.to(device)
    optimizer = build_optimizer(
        model,
        new_lr=float(args.new_lr),
        v18_lr=float(args.v18_lr),
        weight_decay=float(args.weight_decay),
    )
    # BF16 has FP32-like exponent range and does not need dynamic loss scaling.
    # Keep a disabled scaler object solely for a uniform checkpoint schema.
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    progress_state = TrainerProgress()
    window_cursor = 0
    if resume_checkpoint is not None:
        progress_state = restore_training_checkpoint(
            resume_checkpoint, model=model, optimizer=optimizer, scaler=scaler
        )
        window_cursor = int(resume_checkpoint.get("scheduler", {}).get("window_cursor", 0))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_path = out_dir / "train_diagnostics.jsonl"
    natural_distribution_logged = bool(args.resume) or diagnostics_path.exists()
    print(
        json.dumps(
            {
                "completion_objective": COMPLETION_OBJECTIVE_PROTOCOL,
                "presence_focal_gamma": float(args.presence_focal_gamma),
                "completion_semantic_weight": float(
                    args.completion_semantic_weight
                ),
                "diagnostics": str(diagnostics_path.resolve()),
            }
        ),
        flush=True,
    )
    if resume_checkpoint is not None:
        future_checkpoints = [
            path
            for path in out_dir.glob("update_*.pt")
            if path.stem.startswith("update_")
            and path.stem[7:].isdigit()
            and int(path.stem[7:]) > progress_state.successful_updates
        ]
        if future_checkpoints:
            raise RuntimeError(
                "output directory contains checkpoints newer than --resume; "
                "use a fresh --out-dir to avoid deleting a later run"
            )
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
    monitor_log = _load_monitor_history(
        out_dir, resume_checkpoint, progress_state.successful_updates
    )
    frozen_reference_raw = (
        resume_checkpoint.get("frozen_reference_raw")
        if resume_checkpoint is not None
        else None
    )
    if frozen_reference_raw is None and monitor_log:
        frozen_reference_raw = (
            monitor_log[0].get("raw_metric_counts") or {}
        ).get("frozen_v18_reference")
    started = time.perf_counter()

    def prepare_cpu_batch(start_cursor: int) -> tuple[list, float]:
        batch_started = time.perf_counter()
        rows = []
        for offset in range(int(args.batch_size)):
            record = train_records[order[(start_cursor + offset) % len(order)]]
            key = (str(record["scene_name"]), str(record["t0_token"]))
            rows.append(
                prepare_unified_window(
                    record,
                    stage1_rows[key],
                    source=source,
                    pcfg=pcfg,
                    strong_cfg=strong_cfg,
                    component_cache=component_cache,
                    device=torch.device("cpu"),
                )
            )
        return rows, time.perf_counter() - batch_started

    prefetch_executor = (
        None
        if bool(args.no_cpu_prefetch)
        else ThreadPoolExecutor(max_workers=1, thread_name_prefix="v20-prepare")
    )
    prefetch_cursor = int(window_cursor)
    prefetched: Future | None = None

    def submit_prefetch() -> None:
        nonlocal prefetched, prefetch_cursor
        if prefetch_executor is None:
            return
        prefetched = prefetch_executor.submit(prepare_cpu_batch, prefetch_cursor)
        prefetch_cursor += int(args.batch_size)

    submit_prefetch()

    while progress_state.successful_updates < int(args.max_updates):
        if device.type == "cuda":
            # Exclude outstanding monitor/checkpoint work from the next update and
            # report the true per-update allocated-memory high-water mark.
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        update_started = time.perf_counter()
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
        natural_distribution_pieces: list[dict[str, torch.Tensor]] = []
        prepare_cpu_seconds = 0.0
        input_wait_seconds = 0.0
        render_cpu_seconds = 0.0
        cuda_stage_events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {
            "first": [],
            "completion_loss": [],
            "backward": [],
            "optimizer": [],
        }
        micro_steps = int(args.grad_accum) // int(args.batch_size)
        for micro_index in range(micro_steps):
            prepared_windows = []
            try:
                prepare_started = time.perf_counter()
                if prefetched is None:
                    prepared_windows, prepared_seconds = prepare_cpu_batch(
                        window_cursor
                    )
                else:
                    prepared_windows, prepared_seconds = prefetched.result()
                    prefetched = None
                window_cursor += int(args.batch_size)
                if (
                    micro_index + 1 < micro_steps
                    or progress_state.successful_updates + 1
                    < int(args.max_updates)
                ):
                    submit_prefetch()
                prepared_windows = [
                    move_prepared_to_device(prepared, device)
                    for prepared in prepared_windows
                ]
                prepare_cpu_seconds += prepared_seconds
                input_wait_seconds += time.perf_counter() - prepare_started
                first_events = None
                if device.type == "cuda":
                    first_events = (torch.cuda.Event(True), torch.cuda.Event(True))
                    first_events[0].record()
                with _autocast(device, amp):
                    first = first_stage_forward_batch(
                        model,
                        prepared_windows,
                        adapter_enabled=adapter_enabled,
                    )
                if first_events is not None:
                    first_events[1].record()
                    cuda_stage_events["first"].append(first_events)
                    # hard_render_transport immediately consumes detached CPU
                    # transport outputs, so the synchronization is inherent;
                    # make it explicit to keep renderer wall time unambiguous.
                    first_events[1].synchronize()
                source_counts = [
                    int(prepared.state["gpu"]["features"].shape[0])
                    for prepared in prepared_windows
                ]
                transport_windows = []
                current_windows = []
                source_start = 0
                render_started = time.perf_counter()
                for prepared, source_count in zip(
                    prepared_windows, source_counts
                ):
                    source_stop = source_start + source_count
                    transport = {
                        name: value[source_start:source_stop]
                        for name, value in first["transport"].items()
                    }
                    transport_windows.append(transport)
                    current_windows.append(
                        hard_render_transport(
                            model,
                            prepared,
                            transport,
                            pcfg=pcfg,
                            strong_cfg=strong_cfg,
                            device=device,
                        )
                    )
                    source_start = source_stop
                current_transport = torch.cat(current_windows, dim=0)
                render_cpu_seconds += time.perf_counter() - render_started
                completion_events = None
                if device.type == "cuda":
                    completion_events = (
                        torch.cuda.Event(True),
                        torch.cuda.Event(True),
                    )
                    completion_events[0].record()
                with _autocast(device, amp):
                    logits, targets, masks, completion_report = training_completion_inputs_batch(
                        model,
                        prepared_windows,
                        first,
                        current_transport,
                        native_grid=stage1_index["native_grid"],
                        generator=tile_generator,
                        collect_distribution_stats=not natural_distribution_logged,
                    )
                    if completion_report.get("distribution") is not None:
                        natural_distribution_pieces.append(
                            completion_report["distribution"]
                        )
                    losses = []
                    window_indices = completion_report["window_indices"]
                    for window, (prepared, transport) in enumerate(
                        zip(prepared_windows, transport_windows)
                    ):
                        selected = [
                            i
                            for i, owner in enumerate(window_indices)
                            if owner == window
                        ]
                        loss, stats = compute_training_loss(
                            transport,
                            _transport_batch(
                                prepared.record,
                                prepared.state["gpu"],
                                device,
                            ),
                            [logits[i] for i in selected],
                            [targets[i] for i in selected],
                            [masks[i] for i in selected],
                            patch_resolution_m=float(args.patch_resolution_m),
                            completion_weight=float(args.completion_weight),
                            presence_focal_gamma=float(
                                args.presence_focal_gamma
                            ),
                            completion_semantic_weight=float(
                                args.completion_semantic_weight
                            ),
                            graph_anchor=model.completion_head.weight,
                            materialize_stats=False,
                        )
                        stats["tiles"] = int(
                            completion_report["tiles_by_window"][window]
                        )
                        stats["unique_tiles"] = int(
                            completion_report["unique_tiles_by_window"][window]
                        )
                        losses.append(loss)
                        group_stats.append(stats)
                    scaled_loss = torch.stack(losses).mean() / float(micro_steps)
                if completion_events is not None:
                    completion_events[1].record()
                    cuda_stage_events["completion_loss"].append(
                        completion_events
                    )
                backward_events = None
                if device.type == "cuda":
                    backward_events = (
                        torch.cuda.Event(True),
                        torch.cuda.Event(True),
                    )
                    backward_events[0].record()
                scaler.scale(scaled_loss).backward()
                if backward_events is not None:
                    backward_events[1].record()
                    cuda_stage_events["backward"].append(backward_events)
            finally:
                for prepared in prepared_windows:
                    prepared.release()

        progress_state.attempted_updates += 1
        optimizer_events = None
        if device.type == "cuda":
            optimizer_events = (torch.cuda.Event(True), torch.cuda.Event(True))
            optimizer_events[0].record()
        scaler.unscale_(optimizer)
        completion_head_gradients = _completion_head_gradient_norms(model)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(args.clip_grad)
        )
        finite = bool(torch.isfinite(torch.as_tensor(grad_norm)).item())
        succeeded = scaler_step_succeeded(scaler, optimizer) if finite else False
        if not finite:
            scaler.update(float(scaler.get_scale()) / 2.0 if scaler.is_enabled() else 1.0)
        optimizer.zero_grad(set_to_none=True)
        if optimizer_events is not None:
            optimizer_events[1].record()
            cuda_stage_events["optimizer"].append(optimizer_events)
        if device.type == "cuda":
            # Optimizer kernels are asynchronous; synchronize before reporting
            # wall time and the update's peak allocated memory.
            torch.cuda.synchronize(device)
            peak_memory_mib = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        else:
            peak_memory_mib = 0.0
        update_seconds = time.perf_counter() - update_started
        group_stats = _materialize_scalar_stats(group_stats)
        completion_head_gradients_cpu = (
            completion_head_gradients.detach().float().cpu()
        )
        completion_head_gradient_by_class = [
            float(value) for value in completion_head_gradients_cpu.tolist()
        ]
        completion_head_nonfree_grad = float(
            completion_head_gradients_cpu[:17].norm()
        )
        completion_head_free_grad = float(completion_head_gradients_cpu[17])
        natural_distribution = None
        if natural_distribution_pieces:
            natural_counts = torch.stack(
                [row["natural_class_counts"] for row in natural_distribution_pieces]
            ).sum(dim=0).detach().cpu()
            natural_windows = int(
                torch.stack(
                    [row["natural_windows"] for row in natural_distribution_pieces]
                ).sum().detach().cpu()
            )
            natural_windows_without_positive = int(
                torch.stack(
                    [
                        row["natural_windows_without_positive"]
                        for row in natural_distribution_pieces
                    ]
                ).sum().detach().cpu()
            )
            natural_count_list = [int(value) for value in natural_counts.tolist()]
            natural_total = int(sum(natural_count_list))
            natural_occupied = int(sum(natural_count_list[:17]))
            natural_distribution = {
                "natural_support_voxels": natural_total,
                "natural_occupied_voxels": natural_occupied,
                "natural_occupied_fraction": float(
                    natural_occupied / max(natural_total, 1)
                ),
                "natural_class_counts": natural_count_list,
                "natural_windows": natural_windows,
                "natural_windows_without_positive": (
                    natural_windows_without_positive
                ),
            }
        cuda_ms = {
            name: sum(start.elapsed_time(end) for start, end in pairs)
            for name, pairs in cuda_stage_events.items()
        }
        timing_text = (
            f"prep_cpu={prepare_cpu_seconds * 1000.0:.1f},"
            f"input_wait={input_wait_seconds * 1000.0:.1f},"
            f"render_cpu={render_cpu_seconds * 1000.0:.1f},"
            f"first_gpu={cuda_ms['first']:.1f},"
            f"completion_loss_gpu={cuda_ms['completion_loss']:.1f},"
            f"backward_gpu={cuda_ms['backward']:.1f},"
            f"optimizer_gpu={cuda_ms['optimizer']:.1f}"
        )
        if not succeeded:
            reason = "amp_overflow" if finite else "nonfinite_grad_norm"
            print(
                f"update_attempt={progress_state.attempted_updates} {reason}; "
                f"seconds={update_seconds:.3f}; "
                f"peak_memory_mib={peak_memory_mib:.1f}; "
                f"timing_ms={timing_text}; "
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
        mean_presence = float(
            np.mean([row["completion_presence_focal"] for row in group_stats])
        )
        mean_semantic = float(
            np.mean([row["completion_semantic_ce"] for row in group_stats])
        )
        mean_presence_free = float(
            np.mean(
                [row["completion_presence_free_focal"] for row in group_stats]
            )
        )
        mean_presence_occupied = float(
            np.mean(
                [
                    row["completion_presence_occupied_focal"]
                    for row in group_stats
                ]
            )
        )
        voxels = int(sum(row["completion_voxels"] for row in group_stats))
        occupied_voxels = int(
            sum(row["completion_occupied_voxels"] for row in group_stats)
        )
        occupied_fraction = float(occupied_voxels / max(voxels, 1))
        class_counts = [
            int(
                sum(
                    row[f"completion_class_{class_id:02d}_voxels"]
                    for row in group_stats
                )
            )
            for class_id in range(18)
        ]
        tiles = int(sum(row["tiles"] for row in group_stats))
        unique_tiles = int(sum(row["unique_tiles"] for row in group_stats))
        diagnostic = {
            "successful_update": int(progress_state.successful_updates),
            "attempted_update": int(progress_state.attempted_updates),
            "phase": phase,
            "completion_objective": COMPLETION_OBJECTIVE_PROTOCOL,
            "completion_loss": mean_comp,
            "presence_focal_loss": mean_presence,
            "presence_free_focal_loss": mean_presence_free,
            "presence_occupied_focal_loss": mean_presence_occupied,
            "positive_semantic_ce": mean_semantic,
            "sampled_voxels": voxels,
            "sampled_occupied_voxels": occupied_voxels,
            "sampled_occupied_fraction": occupied_fraction,
            "sampled_windows_without_positive": int(
                sum(
                    int(row["completion_occupied_voxels"] == 0)
                    for row in group_stats
                )
            ),
            "sampled_class_counts": class_counts,
            "completion_head_gradient_norm_by_class": (
                completion_head_gradient_by_class
            ),
            "completion_head_nonfree_gradient_norm": (
                completion_head_nonfree_grad
            ),
            "completion_head_free_gradient_norm": completion_head_free_grad,
            "completion_head_nonfree_to_free_gradient_ratio": float(
                completion_head_nonfree_grad
                / max(completion_head_free_grad, 1.0e-12)
            ),
        }
        if natural_distribution is not None:
            diagnostic.update(natural_distribution)
            natural_distribution_logged = True
        with diagnostics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(diagnostic, allow_nan=False) + "\n")
        natural_text = (
            f"natural_occ={natural_distribution['natural_occupied_fraction']:.3%} "
            if natural_distribution is not None
            else ""
        )
        print(
            f"update={progress_state.successful_updates}/{args.max_updates} "
            f"phase={phase} loss={mean_loss:.6f} completion={mean_comp:.6f} "
            f"presence={mean_presence:.6f} semantic={mean_semantic:.6f} "
            f"sampled_occ={occupied_fraction:.3%} {natural_text}"
            f"windows={len(group_stats)} voxels={voxels} "
            f"tiles={tiles} unique_tiles={unique_tiles} "
            f"grad_norm={float(grad_norm):.4f} "
            f"head_grad_fg={completion_head_nonfree_grad:.4f} "
            f"head_grad_free={completion_head_free_grad:.4f} "
            f"seconds={update_seconds:.3f} "
            f"peak_memory_mib={peak_memory_mib:.1f} "
            f"timing_ms={timing_text}",
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
            manifest_paths=manifest_paths,
            config={
                **vars(args),
                "out_dir": str(out_dir.resolve()),
                "warmup_updates": int(args.warmup_updates),
                "completion_objective": COMPLETION_OBJECTIVE_PROTOCOL,
                "native_grid": stage1_index["native_grid"],
            },
            repository_root=ROOT,
            resume_contract=resume_contract,
            monitor_history=monitor_log,
            frozen_reference_raw=frozen_reference_raw,
        )
        checkpoint_path = out_dir / f"update_{progress_state.successful_updates:04d}.pt"
        save_checkpoint(checkpoint_path, payload)
        keep = int(args.keep_checkpoints)
        if keep > 0:
            checkpoints = sorted(out_dir.glob("update_*.pt"))
            for stale in checkpoints[:-keep]:
                stale.unlink(missing_ok=True)

        # Monitoring is full-support/no enrichment and uses an independent
        # frozen reference loaded from the declared Clean-E14 checkpoint.
        frozen_v18 = None
        if frozen_reference_raw is None:
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
            frozen_reference_raw=frozen_reference_raw,
        )
        if frozen_reference_raw is None:
            frozen_reference_raw = monitor["raw_metric_counts"][
                "frozen_v18_reference"
            ]
        monitor.update(
            {
                "successful_updates": progress_state.successful_updates,
                "attempted_updates": progress_state.attempted_updates,
                "phase": phase,
                "next_phase": progress_state.phase,
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
        "screen1024": bool(args.screen1024),
        "train_windows": len(train_records),
        "batch_size": int(args.batch_size),
        "windows_per_update": int(args.grad_accum),
        "keep_checkpoints": int(args.keep_checkpoints),
        "precision": "bfloat16" if amp else "float32",
        "tile_decode_batch_size": int(args.tile_decode_batch_size),
        "runtime_query_chunk": int(args.runtime_query_chunk),
        "checkpoint_completion_tiles": not bool(args.no_completion_checkpoint),
        "cpu_prefetch": not bool(args.no_cpu_prefetch),
        "completion_objective": COMPLETION_OBJECTIVE_PROTOCOL,
        "presence_focal_gamma": float(args.presence_focal_gamma),
        "completion_semantic_weight": float(args.completion_semantic_weight),
        "train_diagnostics": str(diagnostics_path.resolve()),
        "real_data_run": True,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    if prefetch_executor is not None:
        prefetch_executor.shutdown(wait=True, cancel_futures=True)


if __name__ == "__main__":
    main()
