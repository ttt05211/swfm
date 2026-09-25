#!/usr/bin/env python3
"""Train V20 Stage-0 per-Z semantics on frozen V19 predicted geometry."""
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
    frozen_factorized_support,
    dense_targets_from_sparse,
    per_z_semantic_loss,
    validate_stage0_sidecar_pair,
)
from tools.real_motion.build_p0_f9_v19_factorized_static_new_fov_cache import (
    PROTOCOL as V19_CACHE_PROTOCOL,
)
from tools.real_motion.build_p0_f9_v20_stage0_per_z_semantic_cache import (
    PROTOCOL as LABEL_CACHE_PROTOCOL,
    IGNORE_LABEL,
)
from tools.real_motion.train_p0_f9_v19_factorized_static_new_fov import (
    HEAD_TYPE as V19_HEAD_TYPE,
    PROTOCOL as V19_TRAIN_PROTOCOL,
)

PROTOCOL = "p0_f9_v20_stage0_per_z_semantic_train_v2"

V19_INPUT_KEYS = (
    "future_aligned_semantic",
    "future_aligned_geometry_q",
    "base_explained",
    "base_free_bits",
    "new_fov_mask",
    "anchor_semantic",
    "anchor_profile_bits",
    "anchor_distance_q",
)


def _read_index(path):
    root = Path(path)
    idx = json.loads((root / "index.json").read_text(encoding="utf-8"))
    return root, idx


def _load_pair(v19_path, label_path):
    vroot, vidx = _read_index(v19_path)
    lroot, lidx = _read_index(label_path)
    if vidx.get("protocol") != V19_CACHE_PROTOCOL:
        raise RuntimeError(f"unexpected V19 cache protocol: {vidx.get('protocol')}")
    if lidx.get("protocol") != LABEL_CACHE_PROTOCOL:
        raise RuntimeError(
            f"unexpected Stage-0 label protocol: {lidx.get('protocol')}"
        )
    if not bool(lidx.get("label_only_sidecar")):
        raise RuntimeError("Stage-0 labels must use the label-only sidecar contract")
    for key in ("num_windows", "grid_shape_hwd"):
        if lidx.get(key) != vidx.get(key):
            raise RuntimeError(f"Stage-0 V19/label index mismatch: {key}")
    if list(lidx.get("scene_names", [])) != list(vidx.get("scene_names", [])):
        raise RuntimeError("Stage-0 V19/label scene list mismatch")
    if len(lidx.get("shards", [])) != len(vidx.get("shards", [])):
        raise RuntimeError("Stage-0 V19/label shard count mismatch")
    return vroot, vidx, lroot, lidx


def _iter_batches(
    vroot,
    vidx,
    lroot,
    lidx,
    batch_size,
    shuffle,
    seed,
):
    rng = np.random.default_rng(int(seed))
    shard_order = np.arange(len(vidx["shards"]))
    if shuffle:
        rng.shuffle(shard_order)
    for si in shard_order.tolist():
        vrow = vidx["shards"][int(si)]
        lrow = lidx["shards"][int(si)]
        if str(lrow.get("parent_v19_shard")) != str(vrow["file"]):
            raise RuntimeError(f"Stage-0 index shard pairing mismatch at {si}")
        vobj = torch.load(
            vroot / vrow["file"],
            map_location="cpu",
            weights_only=False,
        )
        lobj = torch.load(
            lroot / lrow["file"],
            map_location="cpu",
            weights_only=False,
        )
        if vobj.get("protocol") != V19_CACHE_PROTOCOL:
            raise RuntimeError(f"bad V19 shard protocol: {vrow['file']}")
        n = validate_stage0_sidecar_pair(
            vobj,
            lobj,
            expected_parent_shard=str(vrow["file"]),
        )
        if n != int(vrow["count"]) or n != int(lrow["count"]):
            raise RuntimeError("Stage-0 paired shard count mismatch")
        order = np.arange(n)
        if shuffle:
            rng.shuffle(order)
        for st in range(0, n, int(batch_size)):
            ids = torch.as_tensor(
                order[st : st + int(batch_size)],
                dtype=torch.long,
            )
            batch = {
                k: vobj[k].index_select(0, ids)
                for k in V19_INPUT_KEYS
            }
            # GT-derived V19 vertical_target_bits and sparse semantic labels are
            # supervision only.  They are attached after the model-input dict is
            # selected and are never passed into the frozen V19 forward call.
            batch["vertical_target_bits"] = vobj[
                "vertical_target_bits"
            ].index_select(0, ids)
            offsets = lobj["semantic_offsets"]
            chunks = []
            counts = []
            for rid in ids.tolist():
                lo = int(offsets[rid])
                hi = int(offsets[rid + 1])
                chunks.append(lobj["semantic_values"][lo:hi])
                counts.append(hi - lo)
            batch["semantic_values"] = (
                torch.cat(chunks, dim=0)
                if chunks
                else torch.empty(0, dtype=torch.uint8)
            )
            batch["semantic_counts"] = torch.as_tensor(
                counts, dtype=torch.int64
            )
            yield batch


def _move(raw, device, z, anchor_distance_max_m):
    mv = lambda x: x.to(device, non_blocking=True)
    return {
        "semantic": mv(raw["future_aligned_semantic"]),
        "geometry": dequantize_geometry_torch(
            mv(raw["future_aligned_geometry_q"])
        ),
        "base_explained": mv(raw["base_explained"]).float(),
        "base_free": unpack_vertical_occupancy_torch(
            mv(raw["base_free_bits"]), z
        ).bool(),
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
        "semantic_values": mv(raw["semantic_values"]),
        "semantic_counts": mv(raw["semantic_counts"]).to(torch.int64),
    }


def _autocast(device, enabled):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _flatten_stage0(feature, frozen_support, semantic_target):
    # frozen feature [B,F,C,H,W]
    B, Fh, C, H, W = feature.shape
    # frozen predicted support [B,F,Z,H,W]
    z = int(frozen_support.shape[2])
    feat = feature.reshape(B * Fh, C, H, W)
    support = frozen_support.reshape(B * Fh, z, H, W)
    # sparse labels are expanded on supervision-only [B,F,Z,H,W].
    if semantic_target.shape != (B, Fh, z, H, W):
        raise ValueError("expanded Stage-0 target shape mismatch")
    target = semantic_target.reshape(B * Fh, z, H, W)
    supervised = target.ne(int(IGNORE_LABEL)) & support
    return feat, support, target, supervised


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
        conf += torch.bincount(
            code, minlength=17 * 17
        ).reshape(17, 17)
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
        "mean_iou_on_supervised_voxels": (
            float(np.mean(ious)) if ious else float("nan")
        ),
    }


def _run_epoch(
    adapter,
    head,
    vroot,
    vidx,
    lroot,
    lidx,
    device,
    *,
    batch_size,
    z,
    anchor_distance_max_m,
    presence_threshold,
    vertical_threshold,
    amp,
    optimizer,
    seed,
):
    train = optimizer is not None
    head.train(train)
    adapter.eval()
    loss_sum = 0.0
    nb = 0
    support_voxels = 0
    supervised_voxels = 0
    conf = torch.zeros((17, 17), dtype=torch.int64)
    for raw in _iter_batches(
        vroot,
        vidx,
        lroot,
        lidx,
        batch_size,
        train,
        seed,
    ):
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
            support = frozen_factorized_support(
                frozen,
                new_fov_mask=b["new_fov"],
                base_free=b["base_free"],
                presence_threshold=presence_threshold,
                vertical_threshold=vertical_threshold,
            )
        semantic_target = dense_targets_from_sparse(
            b["vertical_target"],
            b["semantic_values"],
            b["semantic_counts"],
            ignore_label=int(IGNORE_LABEL),
        )
        feat, support_bf, target, supervised = _flatten_stage0(
            feature,
            support,
            semantic_target,
        )
        with _autocast(device, amp):
            logits = head(feat, support_bf)
            loss = per_z_semantic_loss(
                logits,
                target,
                supervised,
            )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("non-finite Stage-0 semantic loss")
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
            optimizer.step()
        loss_sum += float(loss.detach().cpu())
        nb += 1
        support_voxels += int(support_bf.sum().item())
        supervised_voxels += int(supervised.sum().item())
        conf += _semantic_stats(logits, target, supervised)
    return {
        "loss": float(loss_sum / max(nb, 1)),
        "frozen_v19_support_voxels": int(support_voxels),
        "supervised_support_voxels": int(supervised_voxels),
        **_finalize_conf(conf),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-v19-cache", required=True)
    p.add_argument("--train-label-cache", required=True)
    p.add_argument("--val-v19-cache", required=True)
    p.add_argument("--val-label-cache", required=True)
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

    random.seed(int(a.seed))
    np.random.seed(int(a.seed))
    torch.manual_seed(int(a.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(a.seed))

    train = _load_pair(a.train_v19_cache, a.train_label_cache)
    val = _load_pair(a.val_v19_cache, a.val_label_cache)
    train_root, train_idx, train_label_root, train_label_idx = train
    val_root, val_idx, val_label_root, val_label_idx = val
    overlap = set(train_idx.get("scene_names", [])) & set(
        val_idx.get("scene_names", [])
    )
    if overlap:
        raise RuntimeError(
            f"Stage-0 train/val scene overlap: {sorted(overlap)[:5]}"
        )
    z = int(train_idx["grid_shape_hwd"][2])
    if val_idx["grid_shape_hwd"] != train_idx["grid_shape_hwd"]:
        raise RuntimeError("Stage-0 train/val grid mismatch")

    ck = torch.load(
        a.factorized_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    if (
        ck.get("protocol") != V19_TRAIN_PROTOCOL
        or ck.get("head_type") != V19_HEAD_TYPE
    ):
        raise RuntimeError("factorized checkpoint protocol mismatch")
    pth = float(ck["selected_presence_threshold"])
    vth = float(ck["selected_vertical_threshold"])

    device = torch.device(
        a.device
        if a.device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    amp = device.type == "cuda" and not bool(a.no_amp)
    factorized = FactorizedStaticNewFOVHead(
        **dict(ck["architecture"])
    ).to(device)
    factorized.load_state_dict(
        ck["model_state_dict"],
        strict=True,
    )
    adapter = FrozenFactorizedFeatureAdapter(factorized).to(device)
    head = PerZSemanticHead(
        bev_feature_channels=adapter.feature_channels,
        vertical_bins=z,
        hidden_dim=int(a.hidden_dim),
    ).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=float(a.lr),
        weight_decay=float(a.weight_decay),
    )

    out = Path(a.output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing non-empty output dir: {out}")
    out.mkdir(parents=True, exist_ok=True)

    anchor_max = float(
        train_idx.get("anchor_distance_max_m", ANCHOR_DISTANCE_MAX_M)
    )
    history = []
    for epoch in range(1, int(a.epochs) + 1):
        tr = _run_epoch(
            adapter,
            head,
            train_root,
            train_idx,
            train_label_root,
            train_label_idx,
            device,
            batch_size=int(a.batch_size),
            z=z,
            anchor_distance_max_m=anchor_max,
            presence_threshold=pth,
            vertical_threshold=vth,
            amp=amp,
            optimizer=optimizer,
            seed=int(a.seed) + epoch,
        )
        va = _run_epoch(
            adapter,
            head,
            val_root,
            val_idx,
            val_label_root,
            val_label_idx,
            device,
            batch_size=int(a.batch_size),
            z=z,
            anchor_distance_max_m=anchor_max,
            presence_threshold=pth,
            vertical_threshold=vth,
            amp=amp,
            optimizer=None,
            seed=int(a.seed),
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
            "factorized_checkpoint": str(
                Path(a.factorized_checkpoint).resolve()
            ),
            "factorized_presence_threshold": pth,
            "factorized_vertical_threshold": vth,
            "geometry_input_contract": (
                "frozen V19 predicted presence/vertical support only; "
                "GT vertical_target is label-only"
            ),
            "train_v19_cache": str(Path(a.train_v19_cache).resolve()),
            "train_label_cache": str(Path(a.train_label_cache).resolve()),
            "val_v19_cache": str(Path(a.val_v19_cache).resolve()),
            "val_label_cache": str(Path(a.val_label_cache).resolve()),
            "selection_metric": (
                "none_in_training; select only by formal composed semantic "
                "mIoU on the scene-disjoint development set"
            ),
            "validation_ce_is_diagnostic_only": True,
            "val": va,
            "history": history,
        }
        torch.save(payload, out / f"epoch_{epoch:04d}.pt")
    (out / "history.json").write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
