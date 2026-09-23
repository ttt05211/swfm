#!/usr/bin/env python3
"""Train only the V19 Residual Innovation Head on a frozen explained state.

This trainer never instantiates an optimizer over Clean-E14/V18. The cache has
already frozen the Transport + deterministic Static Memory context, and only
ResidualInnovationHead parameters are optimized.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from real_motion.v19_innovation import (
    InnovationLossWeights,
    ResidualInnovationHead,
    innovation_loss,
)
from real_motion.v19_innovation_targets import INNOVATION_POSITIVE_CATEGORIES
from real_motion.v19_innovation_training import (
    INNOVATION_CACHE_PROTOCOL,
    dequantize_geometry_torch,
    unpack_vertical_occupancy_torch,
)

PROTOCOL = "p0_f9_v19_innovation_frozen_base_train_v1"


def _load_index(root: str | Path) -> tuple[Path, dict]:
    p = Path(root)
    index = json.loads((p / "index.json").read_text(encoding="utf-8"))
    if index.get("protocol") != INNOVATION_CACHE_PROTOCOL:
        raise RuntimeError(
            f"unexpected innovation cache protocol: {index.get('protocol')}"
        )
    if not index.get("shards"):
        raise RuntimeError("innovation cache has no shards")
    return p, index


def _iter_batches(
    root: Path,
    index: dict,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
):
    rng = np.random.default_rng(int(seed))
    shard_ids = np.arange(len(index["shards"]))
    if shuffle:
        rng.shuffle(shard_ids)
    tensor_keys = (
        "future_aligned_semantic",
        "future_aligned_geometry_q",
        "base_explained",
        "add_target",
        "semantic_target",
        "vertical_bits",
        "candidate_mask",
    )
    for si in shard_ids.tolist():
        row = index["shards"][int(si)]
        obj = torch.load(
            root / row["file"], map_location="cpu", weights_only=False
        )
        if obj.get("protocol") != INNOVATION_CACHE_PROTOCOL:
            raise RuntimeError(f"shard protocol mismatch: {row['file']}")
        n = int(row["count"])
        order = np.arange(n)
        if shuffle:
            rng.shuffle(order)
        for st in range(0, n, int(batch_size)):
            ids = torch.as_tensor(
                order[st : st + int(batch_size)], dtype=torch.long
            )
            yield {k: obj[k].index_select(0, ids) for k in tensor_keys}


def _autocast(device, enabled):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _move_batch(
    raw: dict,
    device: torch.device,
    vertical_bins: int,
) -> dict:
    move = lambda x: x.to(device, non_blocking=True)
    vertical_bits = move(raw["vertical_bits"])
    return {
        "semantic": move(raw["future_aligned_semantic"]),
        "geometry": dequantize_geometry_torch(
            move(raw["future_aligned_geometry_q"])
        ),
        "base_explained": move(raw["base_explained"]).to(torch.float32),
        "add_target": move(raw["add_target"]).bool(),
        "semantic_target": move(raw["semantic_target"]).long(),
        "vertical_target": unpack_vertical_occupancy_torch(
            vertical_bits, vertical_bins
        ),
        "candidate_mask": move(raw["candidate_mask"]).bool(),
    }


def _eval(
    model,
    root,
    index,
    device,
    *,
    batch_size,
    vertical_bins,
    amp,
    weights,
    positive_weight,
    threshold,
):
    model.eval()
    sums = {
        "loss": 0.0,
        "add_bce": 0.0,
        "semantic_ce": 0.0,
        "vertical_bce": 0.0,
    }
    batches = 0
    tp = fp = fn = 0
    pos_cells = cand_cells = 0
    sem_correct = sem_total = 0
    with torch.inference_mode():
        for raw in _iter_batches(
            root,
            index,
            batch_size=batch_size,
            shuffle=False,
            seed=0,
        ):
            b = _move_batch(raw, device, vertical_bins)
            with _autocast(device, amp):
                out = model(
                    b["semantic"],
                    b["geometry"],
                    b["base_explained"],
                )
                loss, stats = innovation_loss(
                    out,
                    add_target=b["add_target"],
                    semantic_target=b["semantic_target"],
                    vertical_target=b["vertical_target"],
                    candidate_mask=b["candidate_mask"],
                    weights=weights,
                    positive_weight=float(positive_weight),
                )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("non-finite innovation validation loss")
            for k in sums:
                sums[k] += float(stats[k])
            batches += 1
            cand = b["candidate_mask"]
            tgt = b["add_target"] & cand
            pred = (
                torch.sigmoid(out["add_presence_logits"].float())
                >= float(threshold)
            ) & cand
            tp += int((pred & tgt).sum().item())
            fp += int((pred & ~tgt & cand).sum().item())
            fn += int((~pred & tgt).sum().item())
            pos_cells += int(tgt.sum().item())
            cand_cells += int(cand.sum().item())
            if bool(tgt.any()):
                sem = out["semantic_logits"].argmax(dim=2)
                sem_correct += int(
                    (sem[tgt] == b["semantic_target"][tgt]).sum().item()
                )
                sem_total += int(tgt.sum().item())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        **{k: v / max(batches, 1) for k, v in sums.items()},
        "presence_precision": float(precision),
        "presence_recall": float(recall),
        "presence_f1": float(
            2 * precision * recall / max(precision + recall, 1e-12)
        ),
        "semantic_accuracy_on_positive_bev": float(
            sem_correct / max(sem_total, 1)
        ),
        "positive_bev_cells": int(pos_cells),
        "candidate_bev_cells": int(cand_cells),
    }


def _save(
    path,
    *,
    model,
    optimizer,
    epoch,
    global_step,
    args,
    arch,
    train_index,
    val_index,
    val_report,
    positive_weight,
):
    torch.save(
        {
            "protocol": PROTOCOL,
            "training_mode": (
                "innovation_only_frozen_clean_e14_transport_static_memory_v1"
            ),
            "epoch": int(epoch),
            "global_step": int(global_step),
            "innovation_state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "architecture": arch,
            "positive_categories": list(INNOVATION_POSITIVE_CATEGORIES),
            "positive_weight": float(positive_weight),
            "base_checkpoint": train_index.get("base_checkpoint"),
            "base_checkpoint_epoch": train_index.get("base_checkpoint_epoch"),
            "explained_state": train_index.get("explained_state"),
            "train_cache": {
                k: train_index.get(k)
                for k in (
                    "protocol",
                    "num_windows",
                    "num_scenes",
                    "grid_shape_hwd",
                    "totals",
                )
            },
            "val_cache": {
                k: val_index.get(k)
                for k in (
                    "protocol",
                    "num_windows",
                    "num_scenes",
                    "grid_shape_hwd",
                    "totals",
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
    p.add_argument("--vertical-weight", type=float, default=1.0)
    p.add_argument(
        "--positive-weight",
        type=float,
        default=0.0,
        help=(
            "<=0 uses cache-wide neg/pos ratio capped by "
            "--max-positive-weight"
        ),
    )
    p.add_argument("--max-positive-weight", type=float, default=64.0)
    p.add_argument("--presence-threshold", type=float, default=0.5)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=20260923)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-amp", action="store_true")
    a = p.parse_args()

    if min(a.epochs, a.batch_size, a.semantic_dim, a.hidden_dim) <= 0:
        raise ValueError(
            "epochs/batch/model dimensions must be positive"
        )
    if (
        a.lr <= 0
        or a.weight_decay < 0
        or a.semantic_weight < 0
        or a.vertical_weight < 0
    ):
        raise ValueError("invalid optimizer/loss arguments")

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)

    train_root, train_index = _load_index(a.train_cache)
    val_root, val_index = _load_index(a.val_cache)
    if train_index["grid_shape_hwd"] != val_index["grid_shape_hwd"]:
        raise RuntimeError("train/val innovation grid mismatch")
    overlap = sorted(
        set(train_index.get("scene_names", []))
        & set(val_index.get("scene_names", []))
    )
    if overlap:
        raise RuntimeError(
            f"innovation train/val scene overlap: {overlap[:5]}"
        )

    fh = int(train_index["future_frames"])
    hist = int(train_index["history_frames"])
    z = int(train_index["grid_shape_hwd"][2])
    arch = {
        "future_frames": fh,
        "history_frames": hist,
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
    amp = device.type == "cuda" and not bool(a.no_amp)
    model = ResidualInnovationHead(**arch).to(device)

    # Deliberately only Innovation parameters are optimized.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(a.lr),
        weight_decay=float(a.weight_decay),
    )
    weights = InnovationLossWeights(
        semantic=float(a.semantic_weight),
        vertical=float(a.vertical_weight),
    )

    tr_tot = train_index.get("totals", {})
    pos = int(tr_tot.get("positive_bev_cells", 0))
    cand = int(tr_tot.get("candidate_bev_cells", 0))
    if pos <= 0 or cand <= pos:
        raise RuntimeError(
            f"invalid innovation cache class counts: "
            f"pos={pos} candidate={cand}"
        )
    auto_pw = (cand - pos) / max(pos, 1)
    positive_weight = (
        float(a.positive_weight)
        if float(a.positive_weight) > 0
        else min(
            max(auto_pw, 1.0),
            float(a.max_positive_weight),
        )
    )

    out_dir = Path(a.output_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(
            f"refusing non-empty output dir: {out_dir}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    initial_val = _eval(
        model,
        val_root,
        val_index,
        device,
        batch_size=int(a.batch_size),
        vertical_bins=z,
        amp=amp,
        weights=weights,
        positive_weight=positive_weight,
        threshold=float(a.presence_threshold),
    )
    preflight = {
        "protocol": PROTOCOL,
        "training_mode": "innovation_only_frozen_base",
        "parameters": int(
            sum(p.numel() for p in model.parameters())
        ),
        "optimizer_parameters": int(
            sum(
                p.numel()
                for g in optimizer.param_groups
                for p in g["params"]
            )
        ),
        "clean_e14_parameters_in_optimizer": 0,
        "architecture": arch,
        "train_windows": int(train_index["num_windows"]),
        "val_windows": int(val_index["num_windows"]),
        "positive_weight": float(positive_weight),
        "auto_positive_weight_uncapped": float(auto_pw),
        "initial_val": initial_val,
        "amp_bfloat16": bool(amp),
        "base_checkpoint": train_index.get("base_checkpoint"),
    }
    (out_dir / "preflight.json").write_text(
        json.dumps(preflight, indent=2),
        encoding="utf-8",
    )
    print("=== V19 INNOVATION FROZEN-BASE PREFLIGHT ===")
    print(json.dumps(preflight, indent=2))

    global_step = 0
    history = []
    best_loss = float("inf")
    for epoch in range(1, int(a.epochs) + 1):
        model.train()
        sums = {
            "loss": 0.0,
            "add_bce": 0.0,
            "semantic_ce": 0.0,
            "vertical_bce": 0.0,
        }
        nb = 0
        for raw in _iter_batches(
            train_root,
            train_index,
            batch_size=int(a.batch_size),
            shuffle=True,
            seed=int(a.seed) + epoch,
        ):
            b = _move_batch(raw, device, z)
            with _autocast(device, amp):
                out = model(
                    b["semantic"],
                    b["geometry"],
                    b["base_explained"],
                )
                loss, stats = innovation_loss(
                    out,
                    add_target=b["add_target"],
                    semantic_target=b["semantic_target"],
                    vertical_target=b["vertical_target"],
                    candidate_mask=b["candidate_mask"],
                    weights=weights,
                    positive_weight=positive_weight,
                )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(
                    "non-finite innovation loss "
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
            k: v / max(nb, 1) for k, v in sums.items()
        }
        val_report = _eval(
            model,
            val_root,
            val_index,
            device,
            batch_size=int(a.batch_size),
            vertical_bins=z,
            amp=amp,
            weights=weights,
            positive_weight=positive_weight,
            threshold=float(a.presence_threshold),
        )
        row = {
            "epoch": epoch,
            "global_step": global_step,
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
            args=a,
            arch=arch,
            train_index=train_index,
            val_index=val_index,
            val_report=val_report,
            positive_weight=positive_weight,
        )
        if float(val_report["loss"]) < best_loss:
            best_loss = float(val_report["loss"])
            _save(
                out_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                args=a,
                arch=arch,
                train_index=train_index,
                val_index=val_index,
                val_report=val_report,
                positive_weight=positive_weight,
            )
        (out_dir / "history.json").write_text(
            json.dumps(history, indent=2),
            encoding="utf-8",
        )

    print(f"saved innovation checkpoints to {out_dir}")


if __name__ == "__main__":
    main()
