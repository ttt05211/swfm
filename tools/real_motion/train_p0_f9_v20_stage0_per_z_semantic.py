#!/usr/bin/env python3
"""Train V20 Stage-0 per-Z semantics on frozen V19 geometry/features."""
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

from real_motion.v19_innovation_training import (
    dequantize_geometry_torch,
    unpack_vertical_occupancy_torch,
)
from real_motion.v19_static_novelty_factorized import (
    ANCHOR_DISTANCE_MAX_M,
    FactorizedStaticNewFOVHead,
    dequantize_anchor_distance_torch,
)
from real_motion.v20_stage0_voxel_semantic import (
    PROTOCOL as HEAD_PROTOCOL,
    FrozenFactorizedFeatureAdapter,
    PerZSemanticHead,
    per_z_semantic_loss,
)
from tools.real_motion.build_p0_f9_v20_stage0_per_z_semantic_cache import (
    PROTOCOL as CACHE_PROTOCOL,
    IGNORE_LABEL,
)
from tools.real_motion.train_p0_f9_v19_factorized_static_new_fov import (
    HEAD_TYPE as V19_HEAD_TYPE,
    PROTOCOL as V19_TRAIN_PROTOCOL,
)

PROTOCOL = "p0_f9_v20_stage0_per_z_semantic_train_v1"


def _load_index(path):
    root = Path(path)
    idx = json.loads((root / "index.json").read_text(encoding="utf-8"))
    if idx.get("protocol") != CACHE_PROTOCOL:
        raise RuntimeError(f"unexpected Stage-0 cache protocol: {idx.get('protocol')}")
    return root, idx


def _iter_batches(root, idx, batch_size, shuffle, seed):
    rng = np.random.default_rng(int(seed))
    shard_order = np.arange(len(idx["shards"]))
    if shuffle:
        rng.shuffle(shard_order)
    keys = (
        "future_aligned_semantic",
        "future_aligned_geometry_q",
        "base_explained",
        "base_free_bits",
        "new_fov_mask",
        "anchor_semantic",
        "anchor_profile_bits",
        "anchor_distance_q",
        "vertical_target_bits",
        "voxel_semantic_target",
    )
    for si in shard_order.tolist():
        row = idx["shards"][si]
        obj = torch.load(root / row["file"], map_location="cpu", weights_only=False)
        if obj.get("protocol") != CACHE_PROTOCOL:
            raise RuntimeError(f"bad Stage-0 shard protocol: {row['file']}")
        n = int(row["count"])
        order = np.arange(n)
        if shuffle:
            rng.shuffle(order)
        for st in range(0, n, int(batch_size)):
            ids = torch.as_tensor(order[st:st + int(batch_size)], dtype=torch.long)
            yield {k: obj[k].index_select(0, ids) for k in keys}


def _move(raw, device, z, anchor_distance_max_m):
    mv = lambda x: x.to(device, non_blocking=True)
    return {
        "semantic": mv(raw["future_aligned_semantic"]),
        "geometry": dequantize_geometry_torch(mv(raw["future_aligned_geometry_q"])),
        "base_explained": mv(raw["base_explained"]).float(),
        "base_free": unpack_vertical_occupancy_torch(mv(raw["base_free_bits"]), z).bool(),
        "new_fov": mv(raw["new_fov_mask"]).bool(),
        "anchor_semantic": mv(raw["anchor_semantic"]).long(),
        "anchor_profile": unpack_vertical_occupancy_torch(
            mv(raw["anchor_profile_bits"]), z
        ),
        "anchor_distance_m": dequantize_anchor_distance_torch(
            mv(raw["anchor_distance_q"]), anchor_distance_max_m
        ),
        "vertical_target": unpack_vertical_occupancy_torch(
            mv(raw["vertical_target_bits"]), z
        ).bool(),
        "voxel_semantic_target": mv(raw["voxel_semantic_target"]).long(),
    }


def _autocast(device, enabled):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _flatten_stage0(feature, vertical, semantic_target):
    # frozen feature [B,F,C,H,W]
    B, Fh, C, H, W = feature.shape
    # cache vertical [B,F,Z,H,W]
    z = int(vertical.shape[2])
    feat = feature.reshape(B * Fh, C, H, W)
    vert = vertical.reshape(B * Fh, z, H, W)
    # cache target [B,F,H,W,Z] -> [BF,Z,H,W]
    target = semantic_target.permute(0, 1, 4, 2, 3).reshape(B * Fh, z, H, W)
    supervised = target.ne(int(IGNORE_LABEL)) & vert
    return feat, vert, target, supervised


def _semantic_stats(logits, target, supervised):
    masked = logits.detach().float().clone()
    from real_motion.v20_history_world import DYNAMIC_IDS
    dyn = torch.as_tensor(DYNAMIC_IDS, device=masked.device, dtype=torch.long)
    masked[:, dyn] = torch.finfo(masked.dtype).min
    pred = masked.argmax(1)
    valid = supervised.bool()
    conf = torch.zeros((17, 17), dtype=torch.int64, device="cpu")
    if bool(valid.any()):
        p = pred[valid].cpu()
        t = target[valid].cpu()
        code = t * 17 + p
        conf += torch.bincount(code, minlength=17 * 17).reshape(17, 17)
    return conf


def _finalize_conf(conf):
    rows = {}
    ious = []
    for cid in range(17):
        tp = int(conf[cid, cid])
        fp = int(conf[:, cid].sum()) - tp
        fn = int(conf[cid, :].sum()) - tp
        union = tp + fp + fn
        iou = float(tp / union) if union else float("nan")
        rows[str(cid)] = {"tp": tp, "fp": fp, "fn": fn, "iou": iou}
        if union:
            ious.append(iou)
    return {
        "per_class": rows,
        "mean_iou_on_supervised_voxels": float(np.mean(ious)) if ious else float("nan"),
    }


def _run_epoch(adapter, head, root, idx, device, *, batch_size, z, anchor_distance_max_m, amp, optimizer, seed):
    train = optimizer is not None
    head.train(train)
    adapter.eval()
    loss_sum = 0.0
    nb = 0
    conf = torch.zeros((17, 17), dtype=torch.int64)
    for raw in _iter_batches(root, idx, batch_size, train, seed):
        b = _move(raw, device, z, anchor_distance_max_m)
        with torch.no_grad(), _autocast(device, amp):
            frozen, feature = adapter(
                b["semantic"],
                b["geometry"],
                b["base_explained"],
                b["new_fov"],
                b["anchor_semantic"],
                b["anchor_profile"],
                b["anchor_distance_m"],
            )
        feat, vert, target, supervised = _flatten_stage0(
            feature, b["vertical_target"], b["voxel_semantic_target"]
        )
        with _autocast(device, amp):
            logits = head(feat, vert)
            loss = per_z_semantic_loss(logits, target, supervised)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("non-finite Stage-0 semantic loss")
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
            optimizer.step()
        loss_sum += float(loss.detach().cpu())
        nb += 1
        conf += _semantic_stats(logits, target, supervised)
    report = {"loss": float(loss_sum / max(nb, 1)), **_finalize_conf(conf)}
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-cache", required=True)
    p.add_argument("--val-cache", required=True)
    p.add_argument("--factorized-checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=20260925)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-amp", action="store_true")
    a = p.parse_args()

    random.seed(int(a.seed)); np.random.seed(int(a.seed)); torch.manual_seed(int(a.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(a.seed))

    train_root, train_idx = _load_index(a.train_cache)
    val_root, val_idx = _load_index(a.val_cache)
    overlap = set(train_idx.get("scene_names", [])) & set(val_idx.get("scene_names", []))
    if overlap:
        raise RuntimeError(f"Stage-0 train/val scene overlap: {sorted(overlap)[:5]}")
    z = int(train_idx["grid_shape_hwd"][2])
    if val_idx["grid_shape_hwd"] != train_idx["grid_shape_hwd"]:
        raise RuntimeError("Stage-0 train/val grid mismatch")

    ck = torch.load(a.factorized_checkpoint, map_location="cpu", weights_only=False)
    if ck.get("protocol") != V19_TRAIN_PROTOCOL or ck.get("head_type") != V19_HEAD_TYPE:
        raise RuntimeError("factorized checkpoint protocol mismatch")
    device = torch.device(a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and not bool(a.no_amp)
    factorized = FactorizedStaticNewFOVHead(**dict(ck["architecture"])).to(device)
    factorized.load_state_dict(ck["model_state_dict"], strict=True)
    adapter = FrozenFactorizedFeatureAdapter(factorized).to(device)
    head = PerZSemanticHead(
        bev_feature_channels=adapter.feature_channels,
        vertical_bins=z,
        hidden_dim=int(a.hidden_dim),
    ).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=float(a.lr), weight_decay=float(a.weight_decay))
    out = Path(a.output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing non-empty output dir: {out}")
    out.mkdir(parents=True, exist_ok=True)

    anchor_max = float(train_idx.get("anchor_distance_max_m", ANCHOR_DISTANCE_MAX_M))
    best = float("inf")
    history = []
    for epoch in range(1, int(a.epochs) + 1):
        tr = _run_epoch(
            adapter, head, train_root, train_idx, device,
            batch_size=int(a.batch_size), z=z, anchor_distance_max_m=anchor_max,
            amp=amp, optimizer=optimizer, seed=int(a.seed) + epoch,
        )
        va = _run_epoch(
            adapter, head, val_root, val_idx, device,
            batch_size=int(a.batch_size), z=z, anchor_distance_max_m=anchor_max,
            amp=amp, optimizer=None, seed=int(a.seed),
        )
        row = {"epoch": epoch, "train": tr, "val": va}
        history.append(row)
        print(json.dumps(row))
        payload = {
            "protocol": PROTOCOL,
            "head_protocol": HEAD_PROTOCOL,
            "epoch": epoch,
            "head_state_dict": head.state_dict(),
            "head_architecture": {
                "bev_feature_channels": adapter.feature_channels,
                "vertical_bins": z,
                "hidden_dim": int(a.hidden_dim),
                "num_classes": 17,
            },
            "factorized_checkpoint": str(Path(a.factorized_checkpoint).resolve()),
            "factorized_presence_threshold": float(ck["selected_presence_threshold"]),
            "factorized_vertical_threshold": float(ck["selected_vertical_threshold"]),
            "selection_metric": "validation_per_z_semantic_cross_entropy",
            "val": va,
            "history": history,
        }
        torch.save(payload, out / f"epoch_{epoch:04d}.pt")
        if float(va["loss"]) < best:
            best = float(va["loss"])
            torch.save(payload, out / "best.pt")
    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
