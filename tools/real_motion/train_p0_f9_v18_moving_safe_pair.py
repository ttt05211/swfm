#!/usr/bin/env python3
"""Final paired continuation for scene-level Moving-aligned safety.

Both arms resume the exact frozen Y600 SE(2) checkpoint and optimizer state.

  control: existing V18-Y objective only
  safe:    existing V18-Y objective + alpha * scene Moving-Safe

The batch unit is a small group of complete windows so all Strong sources can be
A1-composed before the safety surrogate is measured.  The base Y loss is still
computed only on the same supervised sources used by the original Y training.

The safety coefficient is calibrated once from gradient norms on fixed batches;
there is no alpha sweep and no local-footprint safe term in this experiment.
"""
from __future__ import annotations

import argparse
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

from real_motion.local_st_world_model_v18_se2 import (
    LocalSpatialTemporalWorldModelV18SE2,
)
from real_motion.local_stwm_moving_safe import (
    MOVING_SAFE_LOSS_CONTRACT,
    load_moving_support_cache,
    scene_moving_safe_loss,
)
from real_motion.local_stwm_scene_supervision import V17SceneCacheDataset
from tools.real_motion.train_p0_f9_v18_se2_pair import (
    EXPECTED_OVERLAP,
    PROTOCOL as V18_PAIR_PROTOCOL,
    _lr_scale,
    forward_model,
    load_se2_cache,
    se2_objective_loss,
)
from real_motion.local_st_world_model_v17 import config_from_mapping_v17

PROTOCOL = "p0_f9_v18_scene_moving_safe_paired_continuation_v1"
ARMS = ("control", "safe")
EXPECTED_RESUME_ARM = "Y"
EXPECTED_RESUME_STEP = 600

SOURCE_KEYS = (
    "features",
    "local_semantic_tube",
    "kta_displacement_xy_m",
    "target_residual_xy_m",
    "target_displacement_xy_m",
    "target_valid",
    "existence",
    "supervised_source",
    "source_class_id",
    "frame_motion_features",
    "target_source_mask_tube",
    "target_source_displacement_xy_m",
    "target_source_residual_xy_m",
    "target_yaw_rad",
    "yaw_label_valid",
    "se2_target_valid",
    "yaw_enabled",
)


def _optimizer_step_range(optimizer):
    vals = []
    for state in optimizer.state.values():
        if "step" not in state:
            continue
        x = state["step"]
        vals.append(int(x.item()) if torch.is_tensor(x) else int(x))
    return (min(vals), max(vals)) if vals else (0, 0)


def _load_resume(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("protocol") != V18_PAIR_PROTOCOL:
        raise RuntimeError("resume checkpoint is not the paired V18 protocol")
    if str(ck.get("arm")) != EXPECTED_RESUME_ARM:
        raise RuntimeError("Moving-Safe must resume the Y arm")
    if int(ck.get("continuation_step", -1)) != EXPECTED_RESUME_STEP:
        raise RuntimeError(
            f"Moving-Safe requires Y{EXPECTED_RESUME_STEP}, got "
            f"{ck.get('continuation_step')}"
        )

    model = LocalSpatialTemporalWorldModelV18SE2(
        config_from_mapping_v17(ck.get("model_config"))
    ).to(device)
    model.load_state_dict(ck["state_dict"], strict=True)

    source_ck_path = ck.get("source_checkpoint")
    if not source_ck_path:
        raise RuntimeError("Y checkpoint does not record source_checkpoint")
    source_ck = torch.load(source_ck_path, map_location="cpu", weights_only=False)
    source_args = source_ck.get("args") or {}
    base_lr = float(source_args.get("lr", 5e-4))
    weight_decay = float(source_args.get("weight_decay", 1e-4))
    original_epochs = int(source_args.get("epochs", 20))

    old_params = [
        p for name, p in model.named_parameters()
        if not name.startswith("yaw_head.")
    ]
    optimizer = torch.optim.AdamW(
        old_params, lr=base_lr, weight_decay=weight_decay
    )
    optimizer.add_param_group(
        {
            "params": list(model.yaw_head.parameters()),
            "lr": base_lr,
            "weight_decay": weight_decay,
        }
    )
    optimizer.load_state_dict(ck["optimizer"])

    min_step, max_step = _optimizer_step_range(optimizer)
    inherited_start = int(max_step) - int(ck["continuation_step"])
    source_epoch = int(ck.get("source_epoch", 5))
    if inherited_start <= 0 or inherited_start % max(source_epoch, 1):
        raise RuntimeError(
            f"cannot reconstruct historical schedule from optimizer max step "
            f"{max_step} and Y continuation {ck['continuation_step']}"
        )
    steps_per_epoch = inherited_start // source_epoch
    total_steps = max(1, original_epochs * steps_per_epoch)
    expected_lr = base_lr * _lr_scale(max_step, total_steps)
    got_lr = float(optimizer.param_groups[0]["lr"])
    if not math.isclose(got_lr, expected_lr, rel_tol=2e-6, abs_tol=1e-10):
        raise RuntimeError(
            f"resume LR mismatch: checkpoint={got_lr} expected={expected_lr}"
        )
    return ck, model, optimizer, {
        "optimizer_step_range": [min_step, max_step],
        "historical_start_step": inherited_start,
        "steps_per_epoch": steps_per_epoch,
        "historical_total_steps": total_steps,
        "base_lr": base_lr,
        "resume_lr": got_lr,
    }


def _record_map(records, ids):
    out = {}
    for r in records:
        sid = str(r["sample_id"])
        if sid in ids:
            if sid in out:
                raise RuntimeError(f"duplicate SE2 sample {sid}")
            out[sid] = r
    missing = sorted(set(ids) - set(out))
    if missing:
        raise RuntimeError(
            f"scene cache samples missing from SE2 cache: {missing[:5]}"
        )
    return out


def _combine(scene, se2, support):
    sid = str(scene["sample_id"])
    if sid != str(se2["sample_id"]) or sid != str(support["sample_id"]):
        raise RuntimeError("scene/SE2/support sample mismatch")
    n = int(se2["features"].shape[0])
    if len(scene["source_voxel_indices_t0"]) != n:
        raise RuntimeError(f"{sid}: scene/SE2 source count mismatch")
    if not torch.equal(
        scene["source_class_id"].long(), se2["source_class_id"].long()
    ):
        raise RuntimeError(f"{sid}: scene/SE2 source order mismatch")
    return {"scene": scene, "se2": se2, "support": support}


def _shard_local_batches(
    scene_ds,
    se2_by_id,
    support_by_id,
    *,
    batch_size,
    seed,
    pass_index,
):
    by_shard = {}
    for i, e in enumerate(scene_ds.entries):
        by_shard.setdefault(str(e["shard"]), []).append(i)
    rng = random.Random(int(seed) + 1000003 * int(pass_index))
    shards = sorted(by_shard)
    rng.shuffle(shards)
    pending = []
    for shard in shards:
        ids = list(by_shard[shard])
        rng.shuffle(ids)
        for i in ids:
            scene = scene_ds[i]
            sid = str(scene["sample_id"])
            pending.append(
                _combine(scene, se2_by_id[sid], support_by_id[sid])
            )
            if len(pending) >= int(batch_size):
                yield pending
                pending = []
    if pending:
        yield pending


def _pack_sources(samples, device):
    chunks = {k: [] for k in SOURCE_KEYS}
    slices = []
    start = 0
    for sample in samples:
        r = sample["se2"]
        n = int(r["features"].shape[0])
        slices.append(slice(start, start + n))
        start += n
        for k in SOURCE_KEYS:
            x = r[k]
            if k in {"local_semantic_tube", "target_source_mask_tube"}:
                chunks[k].append(x.to(torch.uint8))
            elif k == "source_class_id":
                chunks[k].append(x.long())
            elif k in {
                "target_valid",
                "supervised_source",
                "yaw_label_valid",
                "se2_target_valid",
                "yaw_enabled",
            }:
                chunks[k].append(x.bool())
            else:
                chunks[k].append(x.float())
    if start == 0:
        raise RuntimeError("scene batch contains no Strong sources")
    out = {
        k: torch.cat(v, dim=0).to(device, non_blocking=True)
        for k, v in chunks.items()
    }
    return out, slices


def _supervised_base_loss(outputs, batch, *, yaw_weight, patch_resolution_m):
    m = batch["supervised_source"].bool()
    if not bool(m.any()):
        z = outputs["residual_xy_m"].sum() * 0.0
        return z, {"objective_loss": 0.0}
    sub_out = {k: v[m] for k, v in outputs.items()}
    sub_batch = {
        k: (v[m] if torch.is_tensor(v) and v.shape[0] == m.shape[0] else v)
        for k, v in batch.items()
    }
    return se2_objective_loss(
        sub_out,
        sub_batch,
        yaw_weight=float(yaw_weight),
        shape_weight=EXPECTED_OVERLAP,
        patch_resolution_m=float(patch_resolution_m),
        safe_weight=0.0,
    )


def _moving_safe(outputs, batch, samples, slices, *, grid, free_label, halo, jitter):
    return scene_moving_safe_loss(
        outputs["residual_xy_m"],
        outputs["yaw_delta_rad"],
        batch["kta_displacement_xy_m"],
        [
            [
                x.cpu().numpy().astype(np.int64, copy=False)
                for x in s["scene"]["source_voxel_indices_t0"]
            ]
            for s in samples
        ],
        [s["scene"]["source_class_id"] for s in samples],
        [s["scene"]["t0_ego_to_world"] for s in samples],
        [s["scene"]["future_ego_to_world"] for s in samples],
        [s["scene"]["strong_anchor_occ"] for s in samples],
        [s["scene"]["future_gt_occ"] for s in samples],
        [
            s["support"]["moving_support_flat_by_horizon"]
            for s in samples
        ],
        slices,
        grid=grid,
        free_label=int(free_label),
        halo_voxels=int(halo),
        jitter_voxels=float(jitter),
    )


def _grad_norm(loss, params, *, retain_graph):
    grads = torch.autograd.grad(
        loss, params, retain_graph=retain_graph, allow_unused=True
    )
    total = None
    for g in grads:
        if g is None:
            continue
        v = g.detach().float().pow(2).sum()
        total = v if total is None else total + v
    return float(torch.sqrt(total).cpu()) if total is not None else 0.0


def _calibrate(
    model,
    scene_ds,
    se2_by_id,
    support_by_id,
    device,
    *,
    grid,
    free_label,
    scene_batch_size,
    batches,
    seed,
    yaw_weight,
    patch_resolution_m,
    halo,
    jitter,
    target_ratio,
):
    params = [p for p in model.parameters() if p.requires_grad]
    rows = []
    iterator = _shard_local_batches(
        scene_ds,
        se2_by_id,
        support_by_id,
        batch_size=scene_batch_size,
        seed=seed,
        pass_index=0,
    )
    model.train()
    for bi, samples in enumerate(iterator, start=1):
        if bi > int(batches):
            break
        batch, slices = _pack_sources(samples, device)
        out = forward_model(model, batch, amp=device.type == "cuda", device=device)
        base, base_stats = _supervised_base_loss(
            out,
            batch,
            yaw_weight=yaw_weight,
            patch_resolution_m=patch_resolution_m,
        )
        safe = _moving_safe(
            out,
            batch,
            samples,
            slices,
            grid=grid,
            free_label=free_label,
            halo=halo,
            jitter=jitter,
        )
        base_norm = _grad_norm(base, params, retain_graph=True)
        safe_norm = _grad_norm(safe.loss, params, retain_graph=False)
        rows.append(
            {
                "batch": bi,
                "scenes": len(samples),
                "sources": int(batch["features"].shape[0]),
                "supervised_sources": int(
                    batch["supervised_source"].sum().item()
                ),
                "base_loss": float(base.detach().cpu()),
                "base_grad_norm": base_norm,
                "moving_safe_loss": float(safe.loss.detach().cpu()),
                "moving_safe_grad_norm": safe_norm,
                "pred_soft_moving_miou": float(
                    safe.pred_soft_moving_miou.detach().cpu()
                ),
                "kta_moving_miou": float(
                    safe.kta_moving_miou.detach().cpu()
                ),
                "safe_active": bool(safe.active),
                "support_voxels": int(safe.support_voxels),
            }
        )
    if len(rows) != int(batches):
        raise RuntimeError(
            f"requested {batches} calibration batches, got {len(rows)}"
        )
    active = [
        r for r in rows
        if r["safe_active"] and r["moving_safe_grad_norm"] > 1e-12
    ]
    if active:
        base_ref = float(np.median([r["base_grad_norm"] for r in active]))
        safe_ref = float(
            np.median([r["moving_safe_grad_norm"] for r in active])
        )
        alpha = float(target_ratio) * base_ref / safe_ref
    else:
        base_ref = float(np.median([r["base_grad_norm"] for r in rows]))
        safe_ref = 0.0
        alpha = 0.0
    return {
        "calibration_batches": len(rows),
        "active_batches": len(active),
        "active_batch_fraction": float(len(active)) / len(rows),
        "target_safe_to_base_gradient_fraction": float(target_ratio),
        "median_active_base_grad_norm": base_ref,
        "median_active_safe_grad_norm": safe_ref,
        "recommended_safe_alpha": alpha,
        "rows": rows,
        "note": (
            "gradient calibration uses only batches where the final-score "
            "one-sided safety constraint is active; no alpha sweep"
        ),
    }


def _save(path, *, model, optimizer, resume_ck, arm, local_step, alpha, args, calibration):
    payload = dict(resume_ck)
    payload["state_dict"] = model.state_dict()
    payload["optimizer"] = optimizer.state_dict()
    payload["arm"] = "Y"  # retain evaluator-compatible SE2 model contract
    payload["continuation_step"] = (
        int(resume_ck["continuation_step"]) + int(local_step)
    )
    payload["variant"] = (
        "RL-SE2-MOVING-SAFE" if arm == "safe" else "RL-SE2-SCENE-CONTROL"
    )
    payload["moving_safe_continuation"] = {
        "protocol": PROTOCOL,
        "arm": arm,
        "local_step": int(local_step),
        "safe_alpha": float(alpha) if arm == "safe" else 0.0,
        "moving_safe_loss_contract": MOVING_SAFE_LOSS_CONTRACT,
        "calibration": calibration,
        "args": vars(args),
    }
    torch.save(payload, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--se2-train-cache", required=True)
    p.add_argument("--scene-cache", required=True)
    p.add_argument("--moving-support-cache", required=True)
    p.add_argument("--resume-checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--arm", choices=ARMS, default="control")
    p.add_argument("--scene-batch-size", type=int, default=4)
    p.add_argument("--continuation-steps", type=int, default=300)
    p.add_argument("--midpoint-step", type=int, default=150)
    p.add_argument("--paired-seed", type=int, default=20260915)
    p.add_argument("--yaw-weight", type=float, default=19.0)
    p.add_argument("--safe-alpha", type=float, default=-1.0)
    p.add_argument("--calibrate-only", action="store_true")
    p.add_argument("--calibration-batches", type=int, default=8)
    p.add_argument("--calibration-target-ratio", type=float, default=0.25)
    p.add_argument("--halo-voxels", type=int, default=2)
    p.add_argument("--jitter-voxels", type=float, default=0.25)
    p.add_argument("--device", default="cuda")
    p.add_argument("--log-every", type=int, default=20)
    a = p.parse_args()

    if a.scene_batch_size <= 0 or a.continuation_steps <= 0:
        raise ValueError("invalid batch/continuation size")
    if a.midpoint_step <= 0 or a.midpoint_step >= a.continuation_steps:
        raise ValueError("midpoint must be inside continuation")
    if a.calibration_batches <= 0 or not 0 < a.calibration_target_ratio <= 1:
        raise ValueError("invalid calibration settings")
    if a.halo_voxels < 0 or a.jitter_voxels < 0:
        raise ValueError("invalid renderer settings")
    if a.yaw_weight < 0:
        raise ValueError("yaw weight must be non-negative")

    random.seed(a.paired_seed)
    np.random.seed(a.paired_seed)
    torch.manual_seed(a.paired_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.paired_seed)

    device = torch.device(
        a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    _, se2_records = load_se2_cache(a.se2_train_cache)
    scene_ds = V17SceneCacheDataset(a.scene_cache)
    support_meta, support_by_id = load_moving_support_cache(
        a.moving_support_cache
    )
    if Path(support_meta["source_scene_cache"]).resolve() != Path(
        a.scene_cache
    ).resolve():
        raise RuntimeError("moving-support cache was built from another scene cache")

    ids = {str(e["sample_id"]) for e in scene_ds.entries}
    if set(support_by_id) != ids:
        missing = sorted(ids - set(support_by_id))
        extra = sorted(set(support_by_id) - ids)
        raise RuntimeError(
            f"moving-support sample mismatch: missing={missing[:3]} "
            f"extra={extra[:3]}"
        )
    se2_by_id = _record_map(se2_records, ids)

    resume_ck, model, optimizer, schedule = _load_resume(
        a.resume_checkpoint, device
    )
    patch_resolution = 0.8
    # The SE2 cache metadata remains the authority for the historical local
    # footprint resolution.
    se2_meta, _ = load_se2_cache(a.se2_train_cache)
    patch_resolution = float(se2_meta.get("patch_resolution_m", 0.8))

    # The scene cache was already audited against the canonical runtime grid.
    from real_motion.geometry import OccupancyGrid
    grid = OccupancyGrid()
    free_label = int(scene_ds.metadata.get("free_label", 17))

    calibration = _calibrate(
        model,
        scene_ds,
        se2_by_id,
        support_by_id,
        device,
        grid=grid,
        free_label=free_label,
        scene_batch_size=int(a.scene_batch_size),
        batches=int(a.calibration_batches),
        seed=int(a.paired_seed),
        yaw_weight=float(a.yaw_weight),
        patch_resolution_m=patch_resolution,
        halo=int(a.halo_voxels),
        jitter=float(a.jitter_voxels),
        target_ratio=float(a.calibration_target_ratio),
    )
    preflight = {
        "protocol": PROTOCOL,
        "resume_checkpoint": str(Path(a.resume_checkpoint).resolve()),
        "resume_continuation_step": int(resume_ck["continuation_step"]),
        "schedule": schedule,
        "scene_windows": len(scene_ds),
        "scene_batch_size": int(a.scene_batch_size),
        "moving_support_cache": str(Path(a.moving_support_cache).resolve()),
        "moving_safe_loss_contract": MOVING_SAFE_LOSS_CONTRACT,
        "calibration": calibration,
    }
    print("=== V18 SCENE MOVING-SAFE CALIBRATION ===")
    print(json.dumps(preflight, indent=2))
    print(
        f"recommended_safe_alpha="
        f"{calibration['recommended_safe_alpha']:.12g}"
    )

    out_dir = Path(a.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "preflight.json").write_text(
        json.dumps(preflight, indent=2), encoding="utf-8"
    )
    if a.calibrate_only:
        return

    recommended = float(calibration["recommended_safe_alpha"])
    if a.arm == "safe":
        if recommended <= 0:
            raise RuntimeError(
                "scene Moving-Safe is inactive on all calibration batches; "
                "there is no calibrated safe experiment to run"
            )
        if a.safe_alpha <= 0:
            raise ValueError("safe arm requires positive --safe-alpha")
        if not math.isclose(
            float(a.safe_alpha), recommended, rel_tol=5e-3, abs_tol=1e-12
        ):
            raise RuntimeError(
                f"--safe-alpha {a.safe_alpha} differs from fixed calibration "
                f"{recommended}"
            )
    alpha = float(a.safe_alpha) if a.arm == "safe" else 0.0

    global_step = int(schedule["optimizer_step_range"][1])
    total_original_steps = int(schedule["historical_total_steps"])
    base_lr = float(schedule["base_lr"])
    pass_index = 0
    iterator = iter(
        _shard_local_batches(
            scene_ds,
            se2_by_id,
            support_by_id,
            batch_size=int(a.scene_batch_size),
            seed=int(a.paired_seed),
            pass_index=pass_index,
        )
    )
    history = []
    started = time.perf_counter()

    for local_step in range(1, int(a.continuation_steps) + 1):
        try:
            samples = next(iterator)
        except StopIteration:
            pass_index += 1
            iterator = iter(
                _shard_local_batches(
                    scene_ds,
                    se2_by_id,
                    support_by_id,
                    batch_size=int(a.scene_batch_size),
                    seed=int(a.paired_seed),
                    pass_index=pass_index,
                )
            )
            samples = next(iterator)

        batch, slices = _pack_sources(samples, device)
        model.train()
        out = forward_model(
            model, batch, amp=device.type == "cuda", device=device
        )
        base, base_stats = _supervised_base_loss(
            out,
            batch,
            yaw_weight=float(a.yaw_weight),
            patch_resolution_m=patch_resolution,
        )
        safe = _moving_safe(
            out,
            batch,
            samples,
            slices,
            grid=grid,
            free_label=free_label,
            halo=int(a.halo_voxels),
            jitter=float(a.jitter_voxels),
        )
        total = base + float(alpha) * safe.loss
        if not bool(torch.isfinite(total)):
            raise RuntimeError(f"non-finite objective at local step {local_step}")

        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        global_step += 1
        lr = base_lr * _lr_scale(global_step, total_original_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        if (
            local_step == 1
            or local_step % int(a.log_every) == 0
            or local_step
            in {int(a.midpoint_step), int(a.continuation_steps)}
        ):
            row = {
                "local_step": local_step,
                "pass_index": pass_index,
                "scenes": len(samples),
                "sources": int(batch["features"].shape[0]),
                "supervised_sources": int(
                    batch["supervised_source"].sum().item()
                ),
                "lr": lr,
                "base_objective": float(base.detach().cpu()),
                "scene_moving_safe_loss": float(safe.loss.detach().cpu()),
                "safe_alpha": float(alpha),
                "weighted_scene_safe": float(alpha)
                * float(safe.loss.detach().cpu()),
                "pred_soft_moving_miou": float(
                    safe.pred_soft_moving_miou.detach().cpu()
                ),
                "kta_moving_miou": float(
                    safe.kta_moving_miou.detach().cpu()
                ),
                "safe_active": bool(safe.active),
                "support_voxels": int(safe.support_voxels),
                "total": float(total.detach().cpu()),
                "elapsed_s": time.perf_counter() - started,
                "translation_smooth_l1": base_stats.get(
                    "translation_smooth_l1"
                ),
                "yaw_periodic_loss": base_stats.get("yaw_periodic_loss"),
                "se2_shape_loss": base_stats.get("se2_shape_loss"),
            }
            history.append(row)
            print("MOVSAFE_STEP " + json.dumps(row), flush=True)

        if local_step in {
            int(a.midpoint_step),
            int(a.continuation_steps),
        }:
            _save(
                out_dir / f"step_{local_step:06d}.pt",
                model=model,
                optimizer=optimizer,
                resume_ck=resume_ck,
                arm=a.arm,
                local_step=local_step,
                alpha=alpha,
                args=a,
                calibration=calibration,
            )

    report = {
        "protocol": PROTOCOL,
        "arm": a.arm,
        "resume_checkpoint": str(Path(a.resume_checkpoint).resolve()),
        "safe_alpha": alpha,
        "moving_safe_loss_contract": MOVING_SAFE_LOSS_CONTRACT,
        "calibration": calibration,
        "history": history,
        "elapsed_s": time.perf_counter() - started,
    }
    (out_dir / "training_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print("=== V18 SCENE MOVING-SAFE CONTINUATION COMPLETE ===")
    print(
        json.dumps(
            {
                "arm": a.arm,
                "midpoint": str(
                    out_dir / f"step_{int(a.midpoint_step):06d}.pt"
                ),
                "endpoint": str(
                    out_dir / f"step_{int(a.continuation_steps):06d}.pt"
                ),
                "elapsed_s": report["elapsed_s"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
