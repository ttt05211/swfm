#!/usr/bin/env python3
"""Paired C experiment: frozen-encoder V17-RL with/without full-scene CE.

C-C (control):
    L = L_pos + L_exist + 0.25 L_overlap

C-S (scene):
    L = L_pos + L_exist + 0.25 L_overlap + alpha(t) L_scene

Both arms start from the same historical V17-RL epoch-5 checkpoint, restore the
same AdamW state/LR schedule, freeze the same historical encoder, use the same
scene cache/order/source set, and run the same number of optimizer updates.
Only alpha(t) differs.  L_scene uses sparse computation but full-scene
normalization; GT moving masks are never used.

Run ``--calibrate-only`` once first.  It measures position, weighted-overlap,
combined-motion and unit scene-CE gradient norms on a fixed training subset and
prints a pre-declared alpha recommendation.  Paired training then requires that
exact alpha via ``--scene-alpha``; there is no dynamic weighting system.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Iterator

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from real_motion.local_st_world_model_v17 import (
    MODEL_PROTOCOL_V17,
    LocalSpatialTemporalWorldModelV17,
    config_from_mapping_v17,
)
from real_motion.local_stwm_scene_supervision import (
    SCENE_CACHE_VERSION,
    SCENE_FREEZE_CONTRACT,
    SCENE_LOSS_CONTRACT,
    SCENE_QUERY_CONTRACT,
    V17SceneCacheDataset,
    calibrate_scene_alpha_from_gradients,
    freeze_history_encoder_for_scene_continuation,
    grad_vector,
    sparse_full_scene_ce_ordered,
    v17_base_loss_tensors,
)
from real_motion.motion_transport import FUTURE_FRAMES
from real_motion.runtime_config import add_config_args, load_runtime_config, make_prepare_config
from tools.real_motion.train_p0_f9_v17_local_stwm import forward_model, load_cache

PROTOCOL = "p0_f9_v17_scene_supervision_paired_continuation_v1"
ARMS = ("C-C", "C-S")
EXPECTED_EPOCH = 5
EXPECTED_VARIANT = "RL"
EXPECTED_OVERLAP = 0.25

SOURCE_KEYS = (
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


def _optimizer_step_range(optimizer) -> tuple[int, int]:
    vals = []
    for state in optimizer.state.values():
        if "step" not in state:
            continue
        x = state["step"]
        vals.append(int(x.item()) if torch.is_tensor(x) else int(x))
    return (min(vals), max(vals)) if vals else (0, 0)


def _lr_scale(step: int, total_steps: int) -> float:
    frac = min(max(float(step) / max(int(total_steps), 1), 0.0), 1.0)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * frac))


def _record_map(v17_records, sample_ids: set[str]):
    out = {}
    for r in v17_records:
        sid = str(r["sample_id"])
        if sid in sample_ids:
            if sid in out:
                raise RuntimeError(f"duplicate V17 sample id {sid}")
            out[sid] = r
    missing = sorted(sample_ids - set(out))
    if missing:
        raise RuntimeError(f"scene cache samples missing from V17 train cache: {missing[:5]}")
    return out


def _combine_scene_and_v17(scene_row: dict, v17_row: dict) -> dict:
    sid = str(scene_row["sample_id"])
    if sid != str(v17_row["sample_id"]):
        raise RuntimeError("scene/V17 sample id mismatch")
    n = int(v17_row["features"].shape[0])
    if len(scene_row["source_voxel_indices_t0"]) != n:
        raise RuntimeError(f"{sid}: scene/V17 source count mismatch")
    if not torch.equal(scene_row["source_class_id"].long(), v17_row["source_class_id"].long()):
        raise RuntimeError(f"{sid}: scene/V17 source order mismatch")
    return {"scene": scene_row, "v17": v17_row}


def _shard_local_batches(
    scene_ds: V17SceneCacheDataset,
    v17_by_id: dict[str, dict],
    *,
    batch_size: int,
    seed: int,
    pass_index: int,
) -> Iterator[list[dict]]:
    """Deterministic paired shuffle while retaining shard locality for I/O."""
    by_shard: dict[str, list[int]] = {}
    for i, e in enumerate(scene_ds.entries):
        by_shard.setdefault(str(e["shard"]), []).append(i)
    rng = random.Random(int(seed) + 1000003 * int(pass_index))
    shards = sorted(by_shard)
    rng.shuffle(shards)
    pending: list[dict] = []
    for shard in shards:
        ids = list(by_shard[shard])
        rng.shuffle(ids)
        for i in ids:
            s = scene_ds[i]
            sid = str(s["sample_id"])
            pending.append(_combine_scene_and_v17(s, v17_by_id[sid]))
            if len(pending) >= int(batch_size):
                yield pending
                pending = []
    if pending:
        yield pending


def _pack_sources(samples: list[dict], device: torch.device) -> tuple[dict, list[slice]]:
    chunks = {k: [] for k in SOURCE_KEYS}
    slices = []
    start = 0
    for sample in samples:
        r = sample["v17"]
        n = int(r["features"].shape[0])
        slices.append(slice(start, start + n))
        start += n
        for k in SOURCE_KEYS:
            x = r[k]
            if k in {"local_semantic_tube", "target_source_mask_tube"}:
                chunks[k].append(x.to(torch.uint8))
            elif k == "source_class_id":
                chunks[k].append(x.long())
            elif k in {"target_valid", "supervised_source"}:
                chunks[k].append(x.bool())
            else:
                chunks[k].append(x.float())
    if start == 0:
        raise RuntimeError("scene batch contains no Strong source")
    out = {k: torch.cat(v, dim=0).to(device, non_blocking=True) for k, v in chunks.items()}
    return out, slices


def _scene_loss_batch(outputs, batch, samples, slices, *, pcfg, halo_voxels: int, eps: float, jitter: float):
    vals = []
    q = full = 0
    per_scene = []
    for sample, sl in zip(samples, slices):
        scene = sample["scene"]
        res = sparse_full_scene_ce_ordered(
            outputs["residual_xy_m"][sl].float(),
            batch["kta_displacement_xy_m"][sl].float(),
            [x.cpu().numpy().astype(np.int64, copy=False) for x in scene["source_voxel_indices_t0"]],
            scene["source_class_id"],
            scene["t0_ego_to_world"],
            scene["future_ego_to_world"],
            scene["strong_anchor_occ"],
            scene["future_gt_occ"],
            grid=pcfg.grid,
            free_label=int(pcfg.free_label),
            halo_voxels=int(halo_voxels),
            eps=float(eps),
            jitter_voxels=float(jitter),
        )
        vals.append(res.loss)
        q += int(res.query_voxels)
        full += int(res.full_voxels)
        per_scene.append(res)
    return torch.stack(vals).mean(), {
        "query_voxels": int(q),
        "full_voxels": int(full),
        "query_fraction": float(q) / max(float(full), 1.0),
        "scene_count": len(vals),
        "per_scene": per_scene,
    }


def _motion_params(model):
    names = []
    params = []
    for name, p in model.named_parameters():
        if not p.requires_grad or name.startswith("existence_head"):
            continue
        names.append(name)
        params.append(p)
    if not params:
        raise RuntimeError("no trainable residual-path parameters after freeze")
    return names, params


def _norm(g: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(g.float()).detach().cpu())


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    den = torch.linalg.vector_norm(a.float()) * torch.linalg.vector_norm(b.float())
    if float(den.detach().cpu()) <= 0:
        return float("nan")
    return float((torch.dot(a.float(), b.float()) / den).detach().cpu())


def calibrate(
    model,
    scene_ds,
    v17_by_id,
    device,
    *,
    pcfg,
    scene_batch_size: int,
    batches: int,
    paired_seed: int,
    halo_voxels: int,
    eps: float,
    jitter: float,
    target_ratio: float,
    max_alpha: float,
    amp: bool,
):
    model.train()
    _, params = _motion_params(model)
    rows = []
    iterator = _shard_local_batches(
        scene_ds, v17_by_id, batch_size=scene_batch_size, seed=paired_seed, pass_index=0
    )
    for bi, samples in enumerate(iterator, start=1):
        if bi > int(batches):
            break
        b, slices = _pack_sources(samples, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
            out = forward_model(model, b, use_representation=True, amp=False, device=device)
            base = v17_base_loss_tensors(
                out, b, overlap_weight=EXPECTED_OVERLAP,
                patch_resolution_m=float(scene_ds.metadata.get("legacy_patch_resolution_m", 0.8)),
            )
        # Scene renderer uses FP32 geometry/probabilities intentionally.
        scene_loss, scene_stats = _scene_loss_batch(
            out, b, samples, slices, pcfg=pcfg,
            halo_voxels=halo_voxels, eps=eps, jitter=jitter,
        )
        g_pos = grad_vector(base["position"], params, retain_graph=True)
        g_ov = grad_vector(base["weighted_overlap"], params, retain_graph=True)
        g_motion = grad_vector(base["motion"], params, retain_graph=True)
        g_scene = grad_vector(scene_loss, params, retain_graph=False)
        rows.append({
            "batch": bi,
            "scenes": len(samples),
            "sources": int(b["features"].shape[0]),
            "position_grad_l2": _norm(g_pos),
            "weighted_overlap_grad_l2": _norm(g_ov),
            "motion_grad_l2": _norm(g_motion),
            "scene_unit_grad_l2": _norm(g_scene),
            "position_overlap_cosine": _cos(g_pos, g_ov),
            "scene_motion_cosine": _cos(g_scene, g_motion),
            "scene_query_fraction": scene_stats["query_fraction"],
            "position_loss": float(base["position"].detach().cpu()),
            "weighted_overlap_loss": float(base["weighted_overlap"].detach().cpu()),
            "scene_full_ce": float(scene_loss.detach().cpu()),
        })
    if len(rows) != int(batches):
        raise RuntimeError(f"requested {batches} calibration batches, got {len(rows)}")
    cal = calibrate_scene_alpha_from_gradients(
        [r["motion_grad_l2"] for r in rows],
        [r["scene_unit_grad_l2"] for r in rows],
        target_ratio=target_ratio,
        max_alpha=max_alpha,
    )
    for key in (
        "position_grad_l2", "weighted_overlap_grad_l2", "motion_grad_l2", "scene_unit_grad_l2",
        "position_overlap_cosine", "scene_motion_cosine", "scene_query_fraction",
    ):
        vals = np.asarray([r[key] for r in rows], dtype=np.float64)
        cal[f"{key}_median"] = float(np.nanmedian(vals))
        cal[f"{key}_min"] = float(np.nanmin(vals))
        cal[f"{key}_max"] = float(np.nanmax(vals))
    cal["batches"] = rows
    return cal


def _freeze_contract_check(model, frozen_snapshot: dict[str, torch.Tensor]):
    for name, p in model.named_parameters():
        if p.requires_grad:
            continue
        before = frozen_snapshot[name]
        if not torch.equal(before, p.detach().cpu()):
            raise RuntimeError(f"frozen encoder parameter changed: {name}")


def _save_checkpoint(path, model, optimizer, ck, *, arm, local_step, global_step, alpha, alpha_now,
                     calibration, freeze_report, args):
    torch.save({
        "protocol": MODEL_PROTOCOL_V17,
        "epoch": EXPECTED_EPOCH,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "feature_dim": ck.get("feature_dim"),
        "future_frames": FUTURE_FRAMES,
        "model_config": ck.get("model_config"),
        "variant": EXPECTED_VARIANT,
        "use_representation": True,
        "overlap_weight": EXPECTED_OVERLAP,
        "scene_continuation": {
            "protocol": PROTOCOL,
            "arm": arm,
            "local_step": int(local_step),
            "global_optimizer_step": int(global_step),
            "scene_alpha_target": float(alpha),
            "scene_alpha_current": float(alpha_now),
            "scene_loss_contract": SCENE_LOSS_CONTRACT,
            "scene_query_contract": SCENE_QUERY_CONTRACT,
            "freeze_contract": SCENE_FREEZE_CONTRACT,
            "calibration": calibration,
            "freeze_report": freeze_report,
            "args": vars(args),
        },
    }, path)


def main():
    p = argparse.ArgumentParser()
    add_config_args(p)
    p.add_argument("--v17-train-cache", required=True)
    p.add_argument("--scene-cache", required=True)
    p.add_argument("--resume-checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--arm", choices=ARMS, default="C-C")
    p.add_argument("--scene-batch-size", type=int, default=4)
    p.add_argument("--continuation-steps", type=int, default=600)
    p.add_argument("--midpoint-step", type=int, default=300)
    p.add_argument("--paired-seed", type=int, default=20260910)
    p.add_argument("--halo-voxels", type=int, default=2)
    p.add_argument("--scene-eps", type=float, default=1e-4)
    p.add_argument("--scene-jitter-voxels", type=float, default=0.25)
    p.add_argument("--scene-alpha", type=float, default=-1.0,
                   help="fixed calibrated target alpha; required for training, ignored by --calibrate-only")
    p.add_argument("--scene-warmup-fraction", type=float, default=0.2)
    p.add_argument("--calibration-batches", type=int, default=8)
    p.add_argument("--calibration-target-ratio", type=float, default=0.25)
    p.add_argument("--max-calibrated-alpha", type=float, default=1e4)
    p.add_argument("--calibration-output", default="")
    p.add_argument("--calibrate-only", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--log-every", type=int, default=20)
    a = p.parse_args()
    if a.scene_batch_size <= 0 or a.continuation_steps <= 0 or a.midpoint_step <= 0:
        raise ValueError("invalid batch/step settings")
    if a.midpoint_step >= a.continuation_steps:
        raise ValueError("midpoint-step must be before continuation-steps")
    if a.halo_voxels < 0 or not 0 < a.scene_eps < 1/18 or a.scene_jitter_voxels < 0:
        raise ValueError("invalid scene renderer settings")
    if not 0 <= a.scene_warmup_fraction <= 1 or a.calibration_batches <= 0:
        raise ValueError("invalid warmup/calibration settings")
    if not a.calibrate_only and a.scene_alpha <= 0:
        raise ValueError("paired training requires a positive --scene-alpha from calibrate-only")

    cfg = load_runtime_config(a.config, a.override)
    pcfg = make_prepare_config(cfg)
    device = torch.device(a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and not bool(a.no_amp)

    train_meta, train_records = load_cache(a.v17_train_cache)
    scene_ds = V17SceneCacheDataset(a.scene_cache)
    if scene_ds.index.get("version") != SCENE_CACHE_VERSION:
        raise RuntimeError("scene cache version mismatch")
    ids = {str(e["sample_id"]) for e in scene_ds.entries}
    v17_by_id = _record_map(train_records, ids)
    # Keep legacy overlap resolution from V17 cache explicit in the scene cache
    # metadata for calibration/reporting without changing the scene loss domain.
    scene_ds.metadata.setdefault("legacy_patch_resolution_m", float(train_meta.get("patch_resolution_m", 0.8)))

    ck = torch.load(a.resume_checkpoint, map_location="cpu", weights_only=False)
    if ck.get("protocol") != MODEL_PROTOCOL_V17 or str(ck.get("variant")) != EXPECTED_VARIANT:
        raise RuntimeError("resume checkpoint must be V17-RL")
    if int(ck.get("epoch", -1)) != EXPECTED_EPOCH or not bool(ck.get("use_representation", False)):
        raise RuntimeError("C protocol requires the frozen V17-RL epoch-5 representation checkpoint")
    if not math.isclose(float(ck.get("overlap_weight", -1.0)), EXPECTED_OVERLAP, abs_tol=1e-12):
        raise RuntimeError("C protocol requires historical overlap weight 0.25")

    model = LocalSpatialTemporalWorldModelV17(config_from_mapping_v17(ck.get("model_config"))).to(device)
    model.load_state_dict(ck["state_dict"], strict=True)
    ck_args = ck.get("args") or {}
    base_lr = float(ck_args.get("lr", 5e-4))
    weight_decay = float(ck_args.get("weight_decay", 1e-4))
    original_epochs = int(ck_args.get("epochs", 10))
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    optimizer.load_state_dict(ck["optimizer"])
    freeze_report = freeze_history_encoder_for_scene_continuation(model)
    frozen_snapshot = {
        name: p.detach().cpu().clone() for name, p in model.named_parameters() if not p.requires_grad
    }
    min_step, max_step = _optimizer_step_range(optimizer)
    if max_step <= 0 or max_step % EXPECTED_EPOCH:
        raise RuntimeError(f"cannot infer historical steps/epoch from optimizer step={max_step}")
    steps_per_epoch = max_step // EXPECTED_EPOCH
    total_original_steps = steps_per_epoch * original_epochs
    if max_step + int(a.continuation_steps) > total_original_steps:
        raise RuntimeError(
            f"continuation exceeds historical 10-epoch LR schedule: start={max_step} "
            f"continuation={a.continuation_steps} total={total_original_steps}"
        )
    expected_lr = base_lr * _lr_scale(max_step, total_original_steps)
    actual_lr = float(optimizer.param_groups[0]["lr"])
    if not math.isclose(actual_lr, expected_lr, rel_tol=2e-6, abs_tol=1e-10):
        raise RuntimeError(f"resume LR mismatch: checkpoint={actual_lr} expected={expected_lr}")

    print("scene alpha calibration ...", flush=True)
    calibration = calibrate(
        model, scene_ds, v17_by_id, device,
        pcfg=pcfg,
        scene_batch_size=a.scene_batch_size,
        batches=a.calibration_batches,
        paired_seed=a.paired_seed,
        halo_voxels=a.halo_voxels,
        eps=a.scene_eps,
        jitter=a.scene_jitter_voxels,
        target_ratio=a.calibration_target_ratio,
        max_alpha=a.max_calibrated_alpha,
        amp=amp,
    )
    preflight = {
        "protocol": PROTOCOL,
        "checkpoint_epoch": EXPECTED_EPOCH,
        "optimizer_step_range": [min_step, max_step],
        "historical_steps_per_epoch": steps_per_epoch,
        "historical_total_steps": total_original_steps,
        "start_lr": actual_lr,
        "scene_windows": len(scene_ds),
        "scene_cache_metadata": scene_ds.metadata,
        "freeze_report": freeze_report,
        "calibration": calibration,
    }
    print("=== V17 C SCENE-LOSS CALIBRATION ===")
    print(json.dumps(preflight, indent=2))
    print(f"recommended_scene_alpha={calibration['alpha']:.12g}")
    if a.calibration_output:
        Path(a.calibration_output).parent.mkdir(parents=True, exist_ok=True)
        Path(a.calibration_output).write_text(json.dumps(preflight, indent=2), encoding="utf-8")
        print(f"saved {a.calibration_output}")
    if a.calibrate_only:
        return

    # Paired runs must use the pre-declared calibration, not independently tune.
    if not math.isclose(float(a.scene_alpha), float(calibration["alpha"]), rel_tol=5e-3, abs_tol=1e-12):
        raise RuntimeError(
            f"--scene-alpha {a.scene_alpha} differs from current fixed calibration "
            f"{calibration['alpha']}; use exactly the calibrate-only result"
        )

    out_dir = Path(a.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "preflight.json").write_text(json.dumps(preflight, indent=2), encoding="utf-8")

    global_step = max_step
    pass_index = 0
    iterator = iter(_shard_local_batches(
        scene_ds, v17_by_id, batch_size=a.scene_batch_size,
        seed=a.paired_seed, pass_index=pass_index,
    ))
    history = []
    warmup_steps = int(round(a.scene_warmup_fraction * a.continuation_steps))
    started = time.perf_counter()

    for local_step in range(1, int(a.continuation_steps) + 1):
        try:
            samples = next(iterator)
        except StopIteration:
            pass_index += 1
            iterator = iter(_shard_local_batches(
                scene_ds, v17_by_id, batch_size=a.scene_batch_size,
                seed=a.paired_seed, pass_index=pass_index,
            ))
            samples = next(iterator)
        b, slices = _pack_sources(samples, device)
        model.train()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
            out = forward_model(model, b, use_representation=True, amp=False, device=device)
            base = v17_base_loss_tensors(
                out, b, overlap_weight=EXPECTED_OVERLAP,
                patch_resolution_m=float(train_meta.get("patch_resolution_m", 0.8)),
            )
        scene_loss, scene_stats = _scene_loss_batch(
            out, b, samples, slices, pcfg=pcfg,
            halo_voxels=a.halo_voxels, eps=a.scene_eps, jitter=a.scene_jitter_voxels,
        )
        if warmup_steps <= 1:
            ramp = 1.0
        else:
            ramp = min(max((local_step - 1) / float(warmup_steps - 1), 0.0), 1.0)
        alpha_now = 0.0 if a.arm == "C-C" else float(a.scene_alpha) * ramp
        total = base["total"] + float(alpha_now) * scene_loss
        if not bool(torch.isfinite(total)):
            raise RuntimeError(f"non-finite C objective at step {local_step}")
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
        optimizer.step()
        global_step += 1
        next_lr = base_lr * _lr_scale(global_step, total_original_steps)
        for group in optimizer.param_groups:
            group["lr"] = next_lr

        if local_step == 1 or local_step % int(a.log_every) == 0 or local_step in {
            int(a.midpoint_step), int(a.continuation_steps)
        }:
            row = {
                "local_step": local_step,
                "global_optimizer_step": global_step,
                "pass_index": pass_index,
                "scenes": len(samples),
                "sources": int(b["features"].shape[0]),
                "lr": next_lr,
                "alpha": alpha_now,
                "base_total": float(base["total"].detach().cpu()),
                "position": float(base["position"].detach().cpu()),
                "existence": float(base["existence"].detach().cpu()),
                "weighted_overlap": float(base["weighted_overlap"].detach().cpu()),
                "scene_full_ce": float(scene_loss.detach().cpu()),
                "scene_query_fraction": scene_stats["query_fraction"],
                "total": float(total.detach().cpu()),
                "elapsed_s": time.perf_counter() - started,
            }
            history.append(row)
            print("C_STEP " + json.dumps(row), flush=True)

        if local_step in {int(a.midpoint_step), int(a.continuation_steps)}:
            _freeze_contract_check(model, frozen_snapshot)
            name = "midpoint.pt" if local_step == int(a.midpoint_step) else "endpoint.pt"
            _save_checkpoint(
                out_dir / name, model, optimizer, ck,
                arm=a.arm, local_step=local_step, global_step=global_step,
                alpha=float(a.scene_alpha), alpha_now=alpha_now,
                calibration=calibration, freeze_report=freeze_report, args=a,
            )
        if local_step % 100 == 0 or local_step == int(a.continuation_steps):
            _save_checkpoint(
                out_dir / "latest.pt", model, optimizer, ck,
                arm=a.arm, local_step=local_step, global_step=global_step,
                alpha=float(a.scene_alpha), alpha_now=alpha_now,
                calibration=calibration, freeze_report=freeze_report, args=a,
            )

    _freeze_contract_check(model, frozen_snapshot)
    report = {
        "protocol": PROTOCOL,
        "arm": a.arm,
        "resume_checkpoint": str(Path(a.resume_checkpoint).resolve()),
        "scene_alpha": float(a.scene_alpha),
        "scene_warmup_fraction": float(a.scene_warmup_fraction),
        "continuation_steps": int(a.continuation_steps),
        "midpoint_step": int(a.midpoint_step),
        "paired_seed": int(a.paired_seed),
        "calibration": calibration,
        "freeze_report": freeze_report,
        "history": history,
        "elapsed_s": time.perf_counter() - started,
    }
    (out_dir / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("=== V17 C TRAINING COMPLETE ===")
    print(json.dumps({
        "arm": a.arm,
        "midpoint": str(out_dir / "midpoint.pt"),
        "endpoint": str(out_dir / "endpoint.pt"),
        "elapsed_s": report["elapsed_s"],
    }, indent=2))


if __name__ == "__main__":
    main()
