#!/usr/bin/env python3
"""Train Static New-FOV Novelty on frozen V18 + deterministic Static Memory."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from real_motion.v19_innovation_training import (
    dequantize_geometry_torch,
    unpack_vertical_occupancy_torch,
)
from real_motion.v19_static_novelty import (
    StaticNewFOVHead,
    static_new_fov_loss,
)
from tools.real_motion.build_p0_f9_v19_static_new_fov_cache import (
    PROTOCOL as CACHE_PROTOCOL,
)


PROTOCOL = "p0_f9_v19_static_new_fov_train_v1"
HEAD_TYPE = "static_new_fov_direct_z"


def _floats(text):
    vals = tuple(
        float(x.strip())
        for x in str(text).split(",")
        if x.strip()
    )
    if not vals:
        raise ValueError("empty threshold list")
    return vals


def _load_index(root):
    p = Path(root)
    idx = json.loads(
        (p / "index.json").read_text(encoding="utf-8")
    )
    if idx.get("protocol") != CACHE_PROTOCOL:
        raise RuntimeError(
            f"unexpected Static New-FOV cache protocol: "
            f"{idx.get('protocol')}"
        )
    if not idx.get("shards"):
        raise RuntimeError("Static New-FOV cache has no shards")
    return p, idx


def _iter_batches(
    root,
    index,
    *,
    batch_size,
    shuffle,
    seed,
):
    rng = np.random.default_rng(int(seed))
    shard_ids = np.arange(len(index["shards"]))
    if shuffle:
        rng.shuffle(shard_ids)
    keys = (
        "future_aligned_semantic",
        "future_aligned_geometry_q",
        "base_explained",
        "base_free_bits",
        "new_fov_mask",
        "occupancy_target_bits",
        "semantic_target",
    )
    for si in shard_ids.tolist():
        row = index["shards"][int(si)]
        obj = torch.load(
            root / row["file"],
            map_location="cpu",
            weights_only=False,
        )
        if obj.get("protocol") != CACHE_PROTOCOL:
            raise RuntimeError(
                f"shard protocol mismatch: {row['file']}"
            )
        n = int(row["count"])
        order = np.arange(n)
        if shuffle:
            rng.shuffle(order)
        for st in range(0, n, int(batch_size)):
            ids = torch.as_tensor(
                order[st : st + int(batch_size)],
                dtype=torch.long,
            )
            yield {
                k: obj[k].index_select(0, ids)
                for k in keys
            }


def _autocast(device, enabled):
    if enabled and device.type == "cuda":
        return torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        )
    return nullcontext()


def _move(raw, device, z):
    mv = lambda x: x.to(device, non_blocking=True)
    base_free = unpack_vertical_occupancy_torch(
        mv(raw["base_free_bits"]),
        int(z),
    ).bool()
    target = unpack_vertical_occupancy_torch(
        mv(raw["occupancy_target_bits"]),
        int(z),
    ).bool()
    new_fov = mv(raw["new_fov_mask"]).bool()
    candidate = new_fov.unsqueeze(2) & base_free
    return {
        "semantic": mv(raw["future_aligned_semantic"]),
        "geometry": dequantize_geometry_torch(
            mv(raw["future_aligned_geometry_q"])
        ),
        "base_explained": mv(
            raw["base_explained"]
        ).float(),
        "base_free": base_free,
        "new_fov": new_fov,
        "candidate": candidate,
        "target": target,
        "semantic_target": mv(
            raw["semantic_target"]
        ).long(),
    }


def _prf(tp, fp, fn):
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f = 2.0 * p * r / max(p + r, 1e-12)
    return float(p), float(r), float(f)


def _evaluate(
    model,
    root,
    index,
    device,
    *,
    batch_size,
    z,
    amp,
    positive_weight,
    semantic_weight,
    thresholds,
):
    model.eval()
    loss_sum = occ_sum = sem_sum = 0.0
    nb = 0
    sem_ok = sem_n = 0
    rows = {
        float(t): {
            "vox_tp": 0,
            "vox_fp": 0,
            "vox_fn": 0,
            "bev_tp": 0,
            "bev_fp": 0,
            "bev_fn": 0,
            "pred_voxels": 0,
            "target_voxels": 0,
        }
        for t in thresholds
    }

    with torch.inference_mode():
        for raw in _iter_batches(
            root,
            index,
            batch_size=batch_size,
            shuffle=False,
            seed=0,
        ):
            b = _move(raw, device, z)
            with _autocast(device, amp):
                out = model(
                    b["semantic"],
                    b["geometry"],
                    b["base_explained"],
                    b["new_fov"],
                )
                loss, stats = static_new_fov_loss(
                    out,
                    occupancy_target=b["target"],
                    candidate_voxels=b["candidate"],
                    semantic_target=b["semantic_target"],
                    occupancy_positive_weight=float(
                        positive_weight
                    ),
                    semantic_weight=float(semantic_weight),
                )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(
                    "non-finite Static New-FOV validation loss"
                )
            loss_sum += float(stats["loss"])
            occ_sum += float(stats["occupancy_bce"])
            sem_sum += float(stats["semantic_ce"])
            nb += 1

            pos_bev = b["target"].any(dim=2)
            if bool(pos_bev.any()):
                sem_pred = out["semantic_logits"].argmax(dim=2)
                sem_ok += int(
                    (
                        sem_pred[pos_bev]
                        == b["semantic_target"][pos_bev]
                    ).sum().item()
                )
                sem_n += int(pos_bev.sum().item())

            prob = torch.sigmoid(
                out["occupancy_logits"].float()
            )
            tgt = b["target"] & b["candidate"]
            tgt_bev = tgt.any(dim=2)
            for th in thresholds:
                pred = (
                    prob >= float(th)
                ) & b["candidate"]
                pred_bev = pred.any(dim=2)
                rr = rows[float(th)]
                rr["vox_tp"] += int(
                    (pred & tgt).sum().item()
                )
                rr["vox_fp"] += int(
                    (pred & ~tgt & b["candidate"]).sum().item()
                )
                rr["vox_fn"] += int(
                    (~pred & tgt).sum().item()
                )
                rr["bev_tp"] += int(
                    (pred_bev & tgt_bev).sum().item()
                )
                rr["bev_fp"] += int(
                    (pred_bev & ~tgt_bev & b["new_fov"]).sum().item()
                )
                rr["bev_fn"] += int(
                    (~pred_bev & tgt_bev).sum().item()
                )
                rr["pred_voxels"] += int(pred.sum().item())
                rr["target_voxels"] += int(tgt.sum().item())

    table = []
    for th, rr in rows.items():
        vp, vr, vf = _prf(
            rr["vox_tp"],
            rr["vox_fp"],
            rr["vox_fn"],
        )
        bp, br, bf = _prf(
            rr["bev_tp"],
            rr["bev_fp"],
            rr["bev_fn"],
        )
        table.append(
            {
                "threshold": float(th),
                "voxel_precision": vp,
                "voxel_recall": vr,
                "voxel_f1": vf,
                "bev_precision": bp,
                "bev_recall": br,
                "bev_f1": bf,
                "prediction_to_target_voxel_ratio": float(
                    rr["pred_voxels"]
                    / max(rr["target_voxels"], 1)
                ),
            }
        )
    best = max(
        table,
        key=lambda x: (
            x["voxel_f1"],
            x["voxel_precision"],
            x["voxel_recall"],
        ),
    )
    return {
        "loss": loss_sum / max(nb, 1),
        "occupancy_bce": occ_sum / max(nb, 1),
        "semantic_ce": sem_sum / max(nb, 1),
        "semantic_accuracy_on_positive_bev": float(
            sem_ok / max(sem_n, 1)
        ),
        "best_threshold": float(best["threshold"]),
        "best_voxel_precision": float(
            best["voxel_precision"]
        ),
        "best_voxel_recall": float(best["voxel_recall"]),
        "best_voxel_f1": float(best["voxel_f1"]),
        "best_bev_precision": float(best["bev_precision"]),
        "best_bev_recall": float(best["bev_recall"]),
        "best_bev_f1": float(best["bev_f1"]),
        "threshold_table": table,
    }


def _save(
    path,
    *,
    model,
    optimizer,
    epoch,
    global_step,
    arch,
    train_index,
    val_index,
    args,
    positive_weight,
    val_report,
):
    torch.save(
        {
            "protocol": PROTOCOL,
            "head_type": HEAD_TYPE,
            "epoch": int(epoch),
            "global_step": int(global_step),
            "model_state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "architecture": arch,
            "occupancy_positive_weight": float(
                positive_weight
            ),
            "semantic_weight": float(args.semantic_weight),
            "selected_threshold": float(
                val_report["best_threshold"]
            ),
            "base_checkpoint": train_index.get(
                "base_checkpoint"
            ),
            "base_checkpoint_epoch": train_index.get(
                "base_checkpoint_epoch"
            ),
            "target": train_index.get("target"),
            "candidate_contract": train_index.get(
                "candidate_contract"
            ),
            "train_cache": {
                k: train_index.get(k)
                for k in (
                    "protocol",
                    "num_windows",
                    "num_scenes",
                    "totals",
                    "occupancy_positive_fraction",
                    "occupancy_neg_pos_ratio",
                )
            },
            "val_cache": {
                k: val_index.get(k)
                for k in (
                    "protocol",
                    "num_windows",
                    "num_scenes",
                    "totals",
                    "occupancy_positive_fraction",
                    "occupancy_neg_pos_ratio",
                )
            },
            "val_report": val_report,
            "args": vars(args),
        },
        path,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-cache", required=True)
    p.add_argument("--val-cache", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--semantic-dim", type=int, default=8)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--semantic-weight", type=float, default=1.0)
    p.add_argument(
        "--occupancy-positive-weight",
        type=float,
        default=0.0,
        help=(
            "<=0 uses sqrt(cache neg/pos), capped by "
            "--max-positive-weight"
        ),
    )
    p.add_argument(
        "--max-positive-weight",
        type=float,
        default=8.0,
    )
    p.add_argument(
        "--val-thresholds",
        default="0.20,0.30,0.40,0.50,0.60,0.70",
    )
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=20260923)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-amp", action="store_true")
    a = p.parse_args()

    if min(
        int(a.epochs),
        int(a.batch_size),
        int(a.semantic_dim),
        int(a.hidden_dim),
    ) <= 0:
        raise ValueError("invalid training dimensions")
    if (
        float(a.lr) <= 0
        or float(a.weight_decay) < 0
        or float(a.semantic_weight) < 0
    ):
        raise ValueError("invalid optimizer/loss arguments")
    thresholds = _floats(a.val_thresholds)

    random.seed(int(a.seed))
    np.random.seed(int(a.seed))
    torch.manual_seed(int(a.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(a.seed))

    train_root, train_index = _load_index(a.train_cache)
    val_root, val_index = _load_index(a.val_cache)
    if (
        train_index["grid_shape_hwd"]
        != val_index["grid_shape_hwd"]
    ):
        raise RuntimeError("train/val grid mismatch")
    overlap = sorted(
        set(train_index.get("scene_names", []))
        & set(val_index.get("scene_names", []))
    )
    if overlap:
        raise RuntimeError(
            f"Static New-FOV train/val scene overlap: "
            f"{overlap[:5]}"
        )

    z = int(train_index["grid_shape_hwd"][2])
    arch = {
        "future_frames": int(train_index["future_frames"]),
        "history_frames": int(train_index["history_frames"]),
        "semantic_dim": int(a.semantic_dim),
        "hidden_dim": int(a.hidden_dim),
        "num_semantic_classes": 17,
        "vertical_bins": z,
    }
    device = torch.device(
        a.device
        if a.device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    amp = (
        device.type == "cuda"
        and not bool(a.no_amp)
    )

    model = StaticNewFOVHead(**arch).to(device)
    prior = float(
        train_index["occupancy_positive_fraction"]
    )
    model.set_occupancy_prior(prior)

    neg_pos = float(train_index["occupancy_neg_pos_ratio"])
    auto_pw = math.sqrt(max(neg_pos, 1.0))
    positive_weight = (
        float(a.occupancy_positive_weight)
        if float(a.occupancy_positive_weight) > 0
        else min(
            max(auto_pw, 1.0),
            float(a.max_positive_weight),
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(a.lr),
        weight_decay=float(a.weight_decay),
    )
    out_dir = Path(a.output_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(
            f"refusing non-empty output dir: {out_dir}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    initial_val = _evaluate(
        model,
        val_root,
        val_index,
        device,
        batch_size=int(a.batch_size),
        z=z,
        amp=amp,
        positive_weight=positive_weight,
        semantic_weight=float(a.semantic_weight),
        thresholds=thresholds,
    )
    preflight = {
        "protocol": PROTOCOL,
        "head_type": HEAD_TYPE,
        "parameters": int(
            sum(p.numel() for p in model.parameters())
        ),
        "clean_e14_parameters_in_optimizer": 0,
        "train_windows": int(train_index["num_windows"]),
        "val_windows": int(val_index["num_windows"]),
        "occupancy_prior": float(prior),
        "occupancy_neg_pos_ratio": float(neg_pos),
        "auto_positive_weight_uncapped": float(auto_pw),
        "occupancy_positive_weight": float(
            positive_weight
        ),
        "semantic_weight": float(a.semantic_weight),
        "architecture": arch,
        "initial_val": initial_val,
        "amp_bfloat16": bool(amp),
    }
    (out_dir / "preflight.json").write_text(
        json.dumps(preflight, indent=2),
        encoding="utf-8",
    )
    print("=== V19 STATIC NEW-FOV PREFLIGHT ===")
    print(json.dumps(preflight, indent=2))

    global_step = 0
    best_f1 = -1.0
    best_loss = float("inf")
    history = []
    for epoch in range(1, int(a.epochs) + 1):
        model.train()
        sums = {
            "loss": 0.0,
            "occupancy_bce": 0.0,
            "semantic_ce": 0.0,
        }
        nb = 0
        for raw in _iter_batches(
            train_root,
            train_index,
            batch_size=int(a.batch_size),
            shuffle=True,
            seed=int(a.seed) + epoch,
        ):
            b = _move(raw, device, z)
            with _autocast(device, amp):
                out = model(
                    b["semantic"],
                    b["geometry"],
                    b["base_explained"],
                    b["new_fov"],
                )
                loss, stats = static_new_fov_loss(
                    out,
                    occupancy_target=b["target"],
                    candidate_voxels=b["candidate"],
                    semantic_target=b["semantic_target"],
                    occupancy_positive_weight=positive_weight,
                    semantic_weight=float(a.semantic_weight),
                )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(
                    f"non-finite Static New-FOV loss "
                    f"epoch={epoch} step={global_step + 1}"
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(a.grad_clip),
            )
            optimizer.step()
            global_step += 1
            nb += 1
            for k in sums:
                sums[k] += float(stats[k])

        train_report = {
            k: v / max(nb, 1)
            for k, v in sums.items()
        }
        val_report = _evaluate(
            model,
            val_root,
            val_index,
            device,
            batch_size=int(a.batch_size),
            z=z,
            amp=amp,
            positive_weight=positive_weight,
            semantic_weight=float(a.semantic_weight),
            thresholds=thresholds,
        )
        row = {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "train": train_report,
            "val": val_report,
        }
        history.append(row)
        print(json.dumps(row), flush=True)

        _save(
            out_dir / f"epoch_{epoch:04d}.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            global_step=global_step,
            arch=arch,
            train_index=train_index,
            val_index=val_index,
            args=a,
            positive_weight=positive_weight,
            val_report=val_report,
        )
        score = float(val_report["best_voxel_f1"])
        if score > best_f1:
            best_f1 = score
            _save(
                out_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                arch=arch,
                train_index=train_index,
                val_index=val_index,
                args=a,
                positive_weight=positive_weight,
                val_report=val_report,
            )
        if float(val_report["loss"]) < best_loss:
            best_loss = float(val_report["loss"])
            _save(
                out_dir / "best_loss.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                arch=arch,
                train_index=train_index,
                val_index=val_index,
                args=a,
                positive_weight=positive_weight,
                val_report=val_report,
            )

        (out_dir / "history.json").write_text(
            json.dumps(history, indent=2),
            encoding="utf-8",
        )

    print(
        f"saved Static New-FOV checkpoints to {out_dir}"
    )


if __name__ == "__main__":
    main()
