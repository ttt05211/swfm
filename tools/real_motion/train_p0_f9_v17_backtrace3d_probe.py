#!/usr/bin/env python3
"""Paired 600-step V17 control vs KTA-backtrace-3D residual-branch probe.

This is a deliberately small decision experiment.  Both arms restart from the
same V17-RL epoch-5 checkpoint, restore the same AdamW state and cosine-LR
position, use the same deterministic 256-supervised-source batches, and optimize
exactly the historical RL objective:

    L = L_pos + L_exist + 0.25 * L_overlap

The only treatment change is a causal KTA-backtraced full-height 3D occupancy
branch injected as an additive future-query residual.  No scene CE and no MSP
routing are used.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from functools import lru_cache
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

from real_motion.local_st_world_model_v17 import (
    MODEL_PROTOCOL_V17,
    LocalSpatialTemporalWorldModelV17,
    config_from_mapping_v17,
)
from real_motion.nuscenes_adapter import NuScenesWindowSource
from real_motion.runtime_config import add_config_args, load_runtime_config, make_prepare_config
from real_motion.v17_backtrace3d_probe import (
    BACKTRACE3D_PROTOCOL,
    BACKTRACE3D_CROP_CONTRACT,
    BACKTRACE3D_FUSION_CONTRACT,
    Backtrace3DProbeConfig,
    LocalSpatialTemporalWorldModelV17Backtrace3D,
    build_kta_backtrace_3d_crops,
)
from tools.real_motion.train_p0_f9_v17_local_stwm import (
    load_cache,
    objective_loss,
)

PROTOCOL = "p0_f9_v17_backtrace3d_paired_probe_train_v1"
ARMS = ("control", "backtrace3d")
START_EPOCH = 5
EXPECTED_VARIANT = "RL"
OVERLAP_WEIGHT = 0.25


class CachedRawSource(NuScenesWindowSource):
    @lru_cache(maxsize=256)
    def load_occ3d(self, scene_name, token, require_lidar_mask=True):
        return super().load_occ3d(
            scene_name, token, require_lidar_mask=require_lidar_mask
        )

    @lru_cache(maxsize=2048)
    def pose(self, token):
        return super().pose(token)


def _optimizer_step_range(optimizer):
    vals = []
    for state in optimizer.state.values():
        if "step" in state:
            x = state["step"]
            vals.append(int(x.item()) if torch.is_tensor(x) else int(x))
    return (min(vals), max(vals)) if vals else (0, 0)


def _lr_scale(step, total_steps):
    frac = min(max(float(step) / max(int(total_steps), 1), 0.0), 1.0)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * frac))


def _gamma_ramp(local_step, warmup_steps):
    if int(warmup_steps) <= 1:
        return 1.0
    return min(
        max((int(local_step) - 1) / float(int(warmup_steps) - 1), 0.0),
        1.0,
    )


def _parse_save_steps(raw):
    out = set()
    for x in str(raw).split(","):
        x = x.strip()
        if x:
            out.add(int(x))
    if any(x <= 0 for x in out):
        raise ValueError("save steps must be positive")
    return out


def _reset_seeds(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _supervised_count(records):
    return sum(int(r["supervised_source"].bool().sum()) for r in records)


def _source_batch_iterator(records, *, batch_size: int, seed: int):
    """Deterministic window-shuffle -> original-source-order -> exact source packs."""
    pass_index = 0
    while True:
        order = list(range(len(records)))
        random.Random(int(seed) + 1000003 * pass_index).shuffle(order)
        pending = []
        for ridx in order:
            ids = torch.nonzero(
                records[ridx]["supervised_source"].bool(), as_tuple=False
            ).flatten().tolist()
            for sid in ids:
                pending.append((int(ridx), int(sid)))
                if len(pending) == int(batch_size):
                    yield pass_index, tuple(pending)
                    pending = []
        if pending:
            # The historical source-level loader also has a final short batch.
            yield pass_index, tuple(pending)
        pass_index += 1


_TENSOR_KEYS = (
    "features",
    "local_semantic_tube",
    "kta_displacement_xy_m",
    "target_residual_xy_m",
    "target_displacement_xy_m",
    "existence",
    "target_valid",
    "supervised_source",
    "source_class_id",
    "frame_motion_features",
    "target_source_mask_tube",
)


def _pack_v17_batch(records, refs, device):
    rows = {k: [] for k in _TENSOR_KEYS}
    for ridx, sid in refs:
        r = records[int(ridx)]
        for k in _TENSOR_KEYS:
            rows[k].append(r[k][int(sid)])
    batch = {}
    for k, vals in rows.items():
        x = torch.stack(vals, dim=0)
        if k in {"local_semantic_tube", "target_source_mask_tube"}:
            x = x.to(torch.uint8)
        elif k in {"target_valid", "supervised_source"}:
            x = x.bool()
        elif k == "source_class_id":
            x = x.long()
        else:
            x = x.float()
        batch[k] = x.to(device, non_blocking=True)
    return batch


def _pack_backtrace_crops(source, records, refs, *, grid, frame_dt_s):
    groups = {}
    for pos, (ridx, sid) in enumerate(refs):
        groups.setdefault(int(ridx), []).append((int(pos), int(sid)))
    n = len(refs)
    sem = torch.empty((n, 6, 64, 64, 16), dtype=torch.uint8)
    valid = torch.empty((n, 6, 64, 64, 16), dtype=torch.bool)
    source_mask = torch.empty((n, 64, 64), dtype=torch.bool)
    relative_times = torch.empty((n, 6), dtype=torch.float32)
    for ridx, items in groups.items():
        positions = [p for p, _ in items]
        source_ids = [s for _, s in items]
        crop = build_kta_backtrace_3d_crops(
            source,
            records[ridx],
            source_ids,
            grid=grid,
            frame_dt_s=float(frame_dt_s),
        )
        pos = torch.as_tensor(positions, dtype=torch.long)
        sem[pos] = crop["semantics"]
        valid[pos] = crop["valid"]
        source_mask[pos] = crop["source_mask"]
        relative_times[pos] = crop["relative_times"]
    return {
        "semantics": sem,
        "valid": valid,
        "source_mask": source_mask,
        "relative_times": relative_times,
    }


def _forward_base(model, batch, amp, device):
    with torch.autocast(
        device_type="cuda", dtype=torch.bfloat16,
        enabled=bool(amp and device.type == "cuda"),
    ):
        return LocalSpatialTemporalWorldModelV17.forward(
            model,
            batch["features"],
            batch["local_semantic_tube"],
            batch["kta_displacement_xy_m"],
            batch["frame_motion_features"],
            batch["target_source_mask_tube"],
        )


def _forward_treatment(model, batch, crops, gamma, amp, device):
    with torch.autocast(
        device_type="cuda", dtype=torch.bfloat16,
        enabled=bool(amp and device.type == "cuda"),
    ):
        return model(
            batch["features"],
            batch["local_semantic_tube"],
            batch["kta_displacement_xy_m"],
            batch["frame_motion_features"],
            batch["target_source_mask_tube"],
            backtrace_semantics=None if crops is None else crops["semantics"],
            backtrace_valid=None if crops is None else crops["valid"],
            backtrace_source_mask=None if crops is None else crops["source_mask"],
            backtrace_relative_times=None if crops is None else crops["relative_times"],
            branch_gamma=float(gamma),
        )


def _load_base_optimizer(model, ck, base_lr, weight_decay, arm):
    if arm == "control":
        params = list(model.parameters())
    else:
        params = [p for _, p in model.base_named_parameters()]
    opt = torch.optim.AdamW(params, lr=float(base_lr), weight_decay=float(weight_decay))
    opt.load_state_dict(ck["optimizer"])
    return opt


def _save_checkpoint(
    path, model, base_optimizer, branch_optimizer, ck, *, arm, local_step,
    global_step, gamma, args, train_meta, batch_contract,
):
    common = {
        "epoch": START_EPOCH,
        "state_dict": model.state_dict(),
        "feature_dim": ck.get("feature_dim"),
        "future_frames": ck.get("future_frames"),
        "model_config": ck.get("model_config"),
        "variant": EXPECTED_VARIANT,
        "use_representation": True,
        "overlap_weight": OVERLAP_WEIGHT,
        "args": ck.get("args") or {},
        "train_cache_metadata": train_meta,
        "fast_probe_continuation": {
            "protocol": PROTOCOL,
            "arm": arm,
            "resume_checkpoint": str(Path(args.resume_checkpoint).resolve()),
            "start_epoch": START_EPOCH,
            "local_step": int(local_step),
            "global_optimizer_step": int(global_step),
            "source_batch_contract": batch_contract,
            "paired_shuffle_seed": int(args.paired_shuffle_seed),
            "source_batch_size": int(args.batch_size),
            "branch_gamma": float(gamma),
            "branch_gamma_warmup_steps": int(args.branch_gamma_warmup_steps),
            "loss_contract": "L_pos+L_exist+0.25L_overlap",
            "scene_ce": False,
            "msp_routing": False,
        },
    }
    if arm == "control":
        common.update({
            "protocol": MODEL_PROTOCOL_V17,
            "optimizer": base_optimizer.state_dict(),
        })
    else:
        common.update({
            "protocol": BACKTRACE3D_PROTOCOL,
            "base_optimizer": base_optimizer.state_dict(),
            "branch_optimizer": branch_optimizer.state_dict(),
            "branch_config": asdict(model.backtrace3d_config),
            "backtrace3d_crop_contract": BACKTRACE3D_CROP_CONTRACT,
            "backtrace3d_fusion_contract": BACKTRACE3D_FUSION_CONTRACT,
            "eval_branch_gamma": 1.0,
        })
    torch.save(common, path)


def main():
    p = argparse.ArgumentParser()
    add_config_args(p)
    p.add_argument("--train-cache", required=True)
    p.add_argument("--resume-checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--dataroot", default="")
    p.add_argument("--info-pkl", default="")
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--save-steps", default="300,600")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--paired-shuffle-seed", type=int, default=20260910)
    p.add_argument("--branch-seed", type=int, default=20260912)
    p.add_argument("--branch-gamma-warmup-steps", type=int, default=20)
    p.add_argument("--branch-source-microbatch", type=int, default=12)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-amp", action="store_true")
    a = p.parse_args()

    if a.max_steps <= 0 or a.batch_size <= 0 or a.log_every <= 0:
        raise ValueError("invalid training length/batch/log settings")
    if a.branch_gamma_warmup_steps <= 0 or a.branch_source_microbatch <= 0:
        raise ValueError("invalid branch settings")
    if a.arm == "backtrace3d" and (not a.dataroot or not a.info_pkl):
        raise ValueError("backtrace3d arm requires --dataroot and --info-pkl")

    out = Path(a.output_dir)
    if out.exists():
        old = sorted(out.glob("*.pt"))
        if old:
            raise FileExistsError(
                "output directory already contains checkpoints: "
                + ", ".join(str(x) for x in old[:8])
            )
    out.mkdir(parents=True, exist_ok=True)
    save_steps = _parse_save_steps(a.save_steps)

    cfg_runtime = load_runtime_config(a.config, a.override)
    pcfg = make_prepare_config(cfg_runtime)
    train_meta, records = load_cache(a.train_cache)
    total_sources = _supervised_count(records)

    ck = torch.load(a.resume_checkpoint, map_location="cpu", weights_only=False)
    if ck.get("protocol") != MODEL_PROTOCOL_V17:
        raise RuntimeError("probe must restart from a standard V17 checkpoint")
    if int(ck.get("epoch", -1)) != START_EPOCH:
        raise RuntimeError(f"probe requires V17 epoch {START_EPOCH}")
    if str(ck.get("variant")) != EXPECTED_VARIANT:
        raise RuntimeError("probe requires the RL variant")
    if not bool(ck.get("use_representation", False)):
        raise RuntimeError("probe requires the V17 representation")
    if not math.isclose(float(ck.get("overlap_weight", -1)), OVERLAP_WEIGHT, abs_tol=1e-12):
        raise RuntimeError("probe requires overlap weight 0.25")

    ck_args = ck.get("args") or {}
    historical_batch = int(ck_args.get("batch_size", 256))
    if int(a.batch_size) != historical_batch:
        raise RuntimeError(
            f"paired probe batch must preserve checkpoint batch size {historical_batch}"
        )
    base_lr = float(ck_args.get("lr", 5e-4))
    weight_decay = float(ck_args.get("weight_decay", 1e-4))
    original_epochs = int(ck_args.get("epochs", 10))
    batches_per_epoch = int(math.ceil(total_sources / float(a.batch_size)))
    start_step = START_EPOCH * batches_per_epoch
    total_original_steps = original_epochs * batches_per_epoch

    device = torch.device(
        a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    amp = device.type == "cuda" and not bool(a.no_amp)

    _reset_seeds(a.branch_seed)
    v17_cfg = config_from_mapping_v17(ck.get("model_config"))
    if a.arm == "control":
        model = LocalSpatialTemporalWorldModelV17(v17_cfg).to(device)
        model.load_state_dict(ck["state_dict"], strict=True)
        branch_optimizer = None
    else:
        branch_cfg = Backtrace3DProbeConfig(
            branch_dim=int(v17_cfg.d_model),
            source_microbatch=int(a.branch_source_microbatch),
        )
        model = LocalSpatialTemporalWorldModelV17Backtrace3D(v17_cfg, branch_cfg).to(device)
        missing, unexpected = model.load_state_dict(ck["state_dict"], strict=False)
        expected_missing = {
            name for name, _ in model.branch_named_parameters()
        }
        if set(missing) != expected_missing or unexpected:
            raise RuntimeError(
                f"unexpected treatment checkpoint mismatch: missing={missing[:8]} "
                f"unexpected={unexpected[:8]}"
            )
        branch_optimizer = torch.optim.AdamW(
            [p for _, p in model.branch_named_parameters()],
            lr=base_lr,
            weight_decay=weight_decay,
        )

    for p0 in model.parameters():
        p0.requires_grad_(True)
    base_optimizer = _load_base_optimizer(
        model, ck, base_lr, weight_decay, a.arm
    )
    min_step, max_step = _optimizer_step_range(base_optimizer)
    if max_step != start_step:
        raise RuntimeError(
            f"restored optimizer step={max_step}, expected={start_step}; "
            "cache/source-count contract differs from epoch5"
        )
    expected_lr = base_lr * _lr_scale(start_step, total_original_steps)
    actual_lr = float(base_optimizer.param_groups[0]["lr"])
    if not math.isclose(actual_lr, expected_lr, rel_tol=2e-6, abs_tol=1e-10):
        raise RuntimeError(
            f"resume LR mismatch: checkpoint={actual_lr}, expected={expected_lr}"
        )
    if branch_optimizer is not None:
        for g in branch_optimizer.param_groups:
            g["lr"] = actual_lr

    raw_source = (
        CachedRawSource(a.dataroot, info_pkl=a.info_pkl, verbose=False)
        if a.arm == "backtrace3d" else None
    )
    batch_contract = (
        "window_shuffle_then_original_strong_source_order_exact_"
        f"{a.batch_size}_supervised_sources_v1"
    )

    preflight = {
        "protocol": PROTOCOL,
        "arm": a.arm,
        "resume_checkpoint": str(Path(a.resume_checkpoint).resolve()),
        "train_cache": str(Path(a.train_cache).resolve()),
        "start_epoch": START_EPOCH,
        "train_supervised_sources": total_sources,
        "batch_size": int(a.batch_size),
        "batches_per_epoch": batches_per_epoch,
        "optimizer_step_range": [min_step, max_step],
        "start_lr": actual_lr,
        "paired_shuffle_seed": int(a.paired_shuffle_seed),
        "branch_seed": int(a.branch_seed),
        "branch_gamma_warmup_steps": int(a.branch_gamma_warmup_steps),
        "batch_contract": batch_contract,
        "loss_contract": "L_pos+L_exist+0.25L_overlap",
        "scene_ce": False,
        "msp_routing": False,
        "backtrace3d_crop_contract": BACKTRACE3D_CROP_CONTRACT if a.arm == "backtrace3d" else None,
        "backtrace3d_fusion_contract": BACKTRACE3D_FUSION_CONTRACT if a.arm == "backtrace3d" else None,
    }

    batches = _source_batch_iterator(
        records, batch_size=int(a.batch_size), seed=int(a.paired_shuffle_seed)
    )
    first_pass, first_refs = next(batches)
    first_batch = _pack_v17_batch(records, first_refs, device)

    step0_identity = None
    if a.arm == "backtrace3d":
        model.eval()
        with torch.no_grad():
            ref = _forward_base(model, first_batch, amp, device)
            got = _forward_treatment(model, first_batch, None, 0.0, amp, device)
        dxy = float(
            (ref["residual_xy_m"].float() - got["residual_xy_m"].float()).abs().max().cpu()
        )
        de = float(
            (ref["existence_logits"].float() - got["existence_logits"].float()).abs().max().cpu()
        )
        step0_identity = {
            "max_abs_residual_xy": dxy,
            "max_abs_existence_logit": de,
            "bit_exact": bool(dxy == 0.0 and de == 0.0),
        }
        if not step0_identity["bit_exact"]:
            raise RuntimeError(f"treatment step0 is not exact V17: {step0_identity}")
    preflight["step0_identity"] = step0_identity
    (out / "preflight.json").write_text(json.dumps(preflight, indent=2), encoding="utf-8")
    print("=== V17 BACKTRACE3D PAIRED PREFLIGHT ===")
    print(json.dumps(preflight, indent=2), flush=True)

    # Recreate the iterator so both arms consume batch #1 as the first optimizer batch.
    batches = _source_batch_iterator(
        records, batch_size=int(a.batch_size), seed=int(a.paired_shuffle_seed)
    )

    model.train()
    global_step = start_step
    history = []
    started = time.perf_counter()
    for local_step in range(1, int(a.max_steps) + 1):
        pass_index, refs = next(batches)
        if len(refs) != int(a.batch_size):
            raise RuntimeError(
                "600-step probe unexpectedly reached a short tail batch; "
                "paired fixed-size contract would be broken"
            )
        batch = _pack_v17_batch(records, refs, device)
        gamma = (
            _gamma_ramp(local_step, int(a.branch_gamma_warmup_steps))
            if a.arm == "backtrace3d" else 0.0
        )

        crops = None
        crop_seconds = 0.0
        if a.arm == "backtrace3d" and gamma != 0.0:
            t_crop = time.perf_counter()
            crops = _pack_backtrace_crops(
                raw_source,
                records,
                refs,
                grid=pcfg.grid,
                frame_dt_s=float(pcfg.frame_dt_s),
            )
            crop_seconds = time.perf_counter() - t_crop

        base_optimizer.zero_grad(set_to_none=True)
        if branch_optimizer is not None:
            branch_optimizer.zero_grad(set_to_none=True)

        if a.arm == "control":
            pred = _forward_base(model, batch, amp, device)
        else:
            pred = _forward_treatment(model, batch, crops, gamma, amp, device)
        loss, parts = objective_loss(
            pred,
            batch,
            overlap_weight=OVERLAP_WEIGHT,
            patch_resolution_m=float(train_meta.get("patch_resolution_m", 0.8)),
        )
        if not bool(torch.isfinite(loss).detach().cpu()):
            raise FloatingPointError(f"non-finite loss at step {local_step}")
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        if not bool(torch.isfinite(torch.as_tensor(grad_norm)).detach().cpu()):
            raise FloatingPointError(f"non-finite gradient at step {local_step}")

        base_optimizer.step()
        if branch_optimizer is not None and gamma != 0.0:
            branch_optimizer.step()
        global_step += 1

        next_lr = base_lr * _lr_scale(global_step, total_original_steps)
        for g in base_optimizer.param_groups:
            g["lr"] = next_lr
        if branch_optimizer is not None:
            for g in branch_optimizer.param_groups:
                g["lr"] = next_lr

        if (
            local_step == 1
            or local_step % int(a.log_every) == 0
            or local_step in save_steps
            or local_step == int(a.max_steps)
        ):
            row = {
                "local_step": local_step,
                "global_optimizer_step": global_step,
                "pass_index": int(pass_index),
                "source_batch_sources": len(refs),
                "lr": next_lr,
                "branch_gamma": float(gamma),
                "objective_loss": float(loss.detach().cpu()),
                "trajectory_smooth_l1": float(parts["trajectory_smooth_l1"]),
                "existence_bce": float(parts["existence_bce"]),
                "transport_overlap_loss": float(parts["transport_overlap_loss"]),
                "crop_seconds": crop_seconds,
                "elapsed_s": time.perf_counter() - started,
            }
            if raw_source is not None:
                row["occ_cache"] = str(raw_source.load_occ3d.cache_info())
                row["pose_cache"] = str(raw_source.pose.cache_info())
            history.append(row)
            print("FAST_PROBE_STEP " + json.dumps(row), flush=True)

        if local_step in save_steps or local_step == int(a.max_steps):
            _save_checkpoint(
                out / f"step_{local_step:06d}.pt",
                model,
                base_optimizer,
                branch_optimizer,
                ck,
                arm=a.arm,
                local_step=local_step,
                global_step=global_step,
                gamma=gamma,
                args=a,
                train_meta=train_meta,
                batch_contract=batch_contract,
            )

    report = {
        **preflight,
        "local_steps": int(a.max_steps),
        "global_optimizer_step": global_step,
        "final_lr": float(base_optimizer.param_groups[0]["lr"]),
        "history": history,
        "elapsed_s": time.perf_counter() - started,
        "saved_steps": sorted(int(x) for x in save_steps if x <= int(a.max_steps)),
        "nonfinite_detected": False,
    }
    (out / "training_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print("=== V17 BACKTRACE3D PAIRED PROBE COMPLETE ===")
    print(json.dumps({
        "arm": a.arm,
        "local_steps": int(a.max_steps),
        "output_dir": str(out),
        "saved_steps": report["saved_steps"],
    }, indent=2))


if __name__ == "__main__":
    main()
