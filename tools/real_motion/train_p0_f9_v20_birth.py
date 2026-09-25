#!/usr/bin/env python3
"""Train V20 Stage-4 persistent Birth queries from strict BIRTH labels."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import replace
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

from real_motion.v20_birth import birth_targets_from_cache_row
from real_motion.v20_scene_model import BirthQueryHead
from real_motion.v20_training import birth_set_loss, checkpoint_payload, load_v20_checkpoint
from tools.real_motion.build_p0_f9_v20_history_cache import PROTOCOL as CACHE_PROTOCOL
from tools.real_motion.v20_birth_stats_from_cache import PROTOCOL as STATS_PROTOCOL

PROTOCOL = "p0_f9_v20_birth_train_v1"


def _load_index(path):
    root = Path(path)
    idx = json.loads((root / "index.json").read_text(encoding="utf-8"))
    if idx.get("protocol") != CACHE_PROTOCOL:
        raise RuntimeError("Birth training requires V20 Stage-1 cache")
    return root, idx


def _iter_rows(root, idx, shuffle, rng):
    order = list(range(len(idx["shards"])))
    if shuffle: rng.shuffle(order)
    for si in order:
        obj = torch.load(root / idx["shards"][si]["file"], map_location="cpu", weights_only=False)
        rows = list(obj["rows"])
        if shuffle: rng.shuffle(rows)
        yield from rows


def _unpack(bits, shape):
    arr = np.asarray(bits.cpu(), dtype=np.uint8)
    return np.unpackbits(arr, bitorder="little", count=int(np.prod(shape))).reshape(shape).astype(bool)


def _scene(model, row, device):
    shape = (6,) + tuple(int(x) for x in row["coarse_shape_xyz"])
    obs = _unpack(row["history_observed_bits"], shape)
    free = _unpack(row["history_observed_free_bits"], shape)
    sem = row["history_semantic_coarse"].to(device).unsqueeze(0)
    with torch.no_grad():
        return model.encode_history(
            sem,
            torch.from_numpy(obs).to(device).unsqueeze(0),
            torch.from_numpy(free).to(device).unsqueeze(0),
        )


def _shape_size_from_stats(stats, voxel_size, margin_voxels):
    p99 = stats.get("size_lwh_m", {}).get("p99")
    if p99 is None:
        raise RuntimeError("Birth stats lack size p99")
    l, w, h = [float(x) for x in p99]
    dims = [
        int(math.ceil(l / voxel_size)) + 2 * margin_voxels,
        int(math.ceil(w / voxel_size)) + 2 * margin_voxels,
        int(math.ceil(h / voxel_size)) + 2 * margin_voxels,
    ]
    mins = [12, 8, 6]
    return tuple(max(a, b) for a, b in zip(dims, mins))


def _run(model, root, idx, device, *, optimizer, rng, amp):
    train = optimizer is not None
    model.birth.train(train)
    losses = []
    matched = 0
    births = 0
    empty_windows = 0
    for row in _iter_rows(root, idx, train, rng):
        scene = _scene(model, row, device)
        target = birth_targets_from_cache_row(
            row,
            shape_size_xyz=model.cfg.birth_shape_size_xyz,
            shape_voxel_size_m=float(model.cfg.birth_shape_voxel_size_m),
            device=device,
        )
        births += int(target["class_id"].numel())
        empty_windows += int(target["class_id"].numel() == 0)
        with (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if amp and device.type == "cuda" else nullcontext()
        ):
            outputs = model.birth(scene)
            loss, stats = birth_set_loss(
                outputs,
                [target],
                no_object_class=int(model.cfg.dynamic_classes),
            )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("non-finite Birth loss")
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.birth.parameters(), 5.0)
            optimizer.step()
        losses.append(float(loss.detach().cpu()))
        matched += int(stats["matched_births"])
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "windows": len(losses),
        "birth_instances": int(births),
        "matched_births": int(matched),
        "empty_windows": int(empty_windows),
        "checkpoint_selection_metric": "NONE; use formal Birth matching + composed semantic metrics",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-cache", required=True)
    p.add_argument("--val-cache", required=True)
    p.add_argument("--birth-stats", required=True)
    p.add_argument("--dormant-checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--shape-voxel-size-m", type=float, default=0.4)
    p.add_argument("--shape-margin-voxels", type=int, default=2)
    p.add_argument("--seed", type=int, default=20260925)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-amp", action="store_true")
    a = p.parse_args()

    tr_root, tr_idx = _load_index(a.train_cache)
    va_root, va_idx = _load_index(a.val_cache)
    overlap = set(tr_idx["scene_names"]) & set(va_idx["scene_names"])
    if overlap:
        raise RuntimeError(f"Birth train/val scene overlap: {sorted(overlap)[:5]}")
    stats = json.loads(Path(a.birth_stats).read_text(encoding="utf-8"))
    if stats.get("protocol") != STATS_PROTOCOL:
        raise RuntimeError("unexpected Birth stats protocol")
    if int(stats["windows"]) != int(tr_idx["num_windows"]):
        raise RuntimeError("Birth stats do not correspond to the training cache")
    if float(stats["truncated_gt_fraction"]) > 0.0100001:
        raise RuntimeError(
            f"Birth Q truncates {stats['truncated_gt_fraction']:.4%} of GT; "
            "regenerate Q under the <=1% contract"
        )
    Q = int(stats["Q"])
    shape_voxel = float(a.shape_voxel_size_m)
    shape_size = _shape_size_from_stats(
        stats, shape_voxel, int(a.shape_margin_voxels)
    )

    model, parent = load_v20_checkpoint(a.dormant_checkpoint, map_location="cpu")
    if str(parent.get("stage")) != "dormant":
        raise RuntimeError("Birth training requires Stage-3 Dormant checkpoint")
    cfg = replace(
        model.cfg,
        birth_queries=Q,
        birth_shape_size_xyz=shape_size,
        birth_shape_voxel_size_m=shape_voxel,
    )
    # Preserve trained encoder/Static/Dormant and create a fresh, zero-contribution
    # Birth head whose architecture is now frozen by the training distribution.
    model.cfg = cfg
    model.birth = BirthQueryHead(model.encoder.output_dim, cfg)
    for p0 in model.parameters(): p0.requires_grad = False
    for p0 in model.birth.parameters(): p0.requires_grad = True

    device = torch.device(a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    model.birth.train()
    amp = device.type == "cuda" and not bool(a.no_amp)
    optimizer = torch.optim.AdamW(model.birth.parameters(), lr=float(a.lr), weight_decay=1e-4)
    rng = random.Random(int(a.seed))
    out = Path(a.output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing non-empty output dir: {out}")
    out.mkdir(parents=True, exist_ok=True)

    history = []
    parent_extra = dict(parent.get("extra") or {})
    for epoch in range(1, int(a.epochs) + 1):
        tr = _run(model, tr_root, tr_idx, device, optimizer=optimizer, rng=rng, amp=amp)
        model.birth.eval()
        with torch.no_grad():
            va = _run(
                model, va_root, va_idx, device,
                optimizer=None, rng=random.Random(int(a.seed)), amp=amp,
            )
        model.birth.train()
        row = {"epoch": epoch, "train": tr, "val": va}
        history.append(row); print(json.dumps(row))
        payload = checkpoint_payload(
            model,
            stage="birth",
            v18_checkpoint=str(parent["v18_checkpoint"]),
            thresholds={},
            extra={
                **parent_extra,
                "train_protocol": PROTOCOL,
                "epoch": epoch,
                "parent_dormant_checkpoint": str(Path(a.dormant_checkpoint).resolve()),
                "birth_stats": str(Path(a.birth_stats).resolve()),
                "birth_Q": Q,
                "birth_Q_truncated_gt_fraction": float(stats["truncated_gt_fraction"]),
                "birth_shape_size_xyz": list(shape_size),
                "birth_shape_voxel_size_m": shape_voxel,
                "history": history,
                "checkpoint_selection": (
                    "formal real-BIRTH matching/precision + composed semantic metrics on dev split"
                ),
            },
        )
        torch.save(payload, out / f"epoch_{epoch:04d}.pt")
        torch.save(payload, out / "latest.pt")
    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
