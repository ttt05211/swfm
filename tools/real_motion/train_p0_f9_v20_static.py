#!/usr/bin/env python3
"""Train V20 Stage-2 shared 3D encoder + one canonical Static world."""
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
import torch.nn.functional as F

from real_motion.v20_history_world import (
    CanonicalLattice,
    DYNAMIC_IDS,
    canonical_tile_grid_sample_coordinates,
    native_sparse_to_canonical_indices,
)
from real_motion.v20_scene_model import V20HistoryWorldModel, V20SceneConfig
from real_motion.v20_training import checkpoint_payload
from tools.real_motion.build_p0_f9_v20_history_cache import PROTOCOL as CACHE_PROTOCOL

PROTOCOL = "p0_f9_v20_static_train_v1"
DYNAMIC_SET = set(int(x) for x in DYNAMIC_IDS)


def _load_index(root):
    root = Path(root)
    idx = json.loads((root / "index.json").read_text(encoding="utf-8"))
    if idx.get("protocol") != CACHE_PROTOCOL:
        raise RuntimeError(f"unexpected V20 cache protocol: {idx.get('protocol')}")
    return root, idx


def _lattice(d):
    return CanonicalLattice(
        tuple(float(x) for x in d["origin_xyz_m"]),
        tuple(float(x) for x in d["voxel_size_xyz_m"]),
        tuple(int(x) for x in d["shape_xyz"]),
    )


def _unpack(bits, shape):
    a = np.asarray(bits.cpu(), dtype=np.uint8)
    n = int(np.prod(shape))
    x = np.unpackbits(a, bitorder="little", count=n)
    return x.reshape(shape).astype(bool)


def _iter_rows(root, idx, shuffle, seed):
    order = list(range(len(idx["shards"])))
    rng = random.Random(int(seed))
    if shuffle:
        rng.shuffle(order)
    for si in order:
        obj = torch.load(root / idx["shards"][si]["file"], map_location="cpu", weights_only=False)
        if obj.get("protocol") != CACHE_PROTOCOL:
            raise RuntimeError("bad V20 cache shard")
        rows = list(obj["rows"])
        if shuffle:
            rng.shuffle(rows)
        yield from rows


def _row_history(row, device):
    shape = (6,) + tuple(int(x) for x in row["coarse_shape_xyz"])
    obs = _unpack(row["history_observed_bits"], shape)
    free = _unpack(row["history_observed_free_bits"], shape)
    sem = row["history_semantic_coarse"].to(device).unsqueeze(0)
    return (
        sem,
        torch.from_numpy(obs).to(device).unsqueeze(0),
        torch.from_numpy(free).to(device).unsqueeze(0),
        obs,
    )


def _aggregate_sparse_targets(row, lattice, native_origin, native_step):
    """Merge six future observed static/free labels in canonical coordinates.

    Disagreeing labels at one canonical cell are ignored instead of forcing a
    time-varying scene into one static world.
    """
    table = {}
    rel = np.asarray(row["future_ego_to_t0"], dtype=np.float64)
    for fi, sup in enumerate(row["static_supervision"]):
        native = np.asarray(sup["indices_xyz"], dtype=np.int64)
        labels = np.asarray(sup["semantic"], dtype=np.int64)
        if len(native) == 0:
            continue
        idx, valid = native_sparse_to_canonical_indices(
            lattice,
            native_indices_xyz=native,
            ego_to_canonical=rel[fi],
            native_origin_xyz_m=native_origin,
            native_voxel_size_xyz_m=native_step,
        )
        for cell, lab in zip(idx[valid], labels[valid]):
            key = (int(cell[0]), int(cell[1]), int(cell[2]))
            lab = int(lab)
            old = table.get(key)
            if old is None:
                table[key] = lab
            elif old != lab:
                table[key] = -1
    keys = [k for k, v in table.items() if v >= 0]
    if not keys:
        return np.zeros((0, 3), np.int64), np.zeros((0,), np.int64)
    return (
        np.asarray(keys, dtype=np.int64),
        np.asarray([table[k] for k in keys], dtype=np.int64),
    )


def _point_ce(logits, idx, labels, class_weights):
    if len(idx) == 0:
        return logits.sum() * 0.0
    ii = torch.as_tensor(idx, dtype=torch.long, device=logits.device)
    y = torch.as_tensor(labels, dtype=torch.long, device=logits.device)
    rows = logits[0, :, ii[:, 0], ii[:, 1], ii[:, 2]].transpose(0, 1).clone()
    dyn = torch.as_tensor(tuple(DYNAMIC_IDS), dtype=torch.long, device=logits.device)
    rows[:, dyn] = torch.finfo(rows.dtype).min
    return F.cross_entropy(rows, y, weight=class_weights)


def _class_weights(root, idx):
    hist = np.zeros(18, dtype=np.int64)
    for row in _iter_rows(root, idx, False, 0):
        for sup in row["static_supervision"]:
            y = np.asarray(sup["semantic"], dtype=np.int64)
            hist += np.bincount(y, minlength=18)[:18]
    allowed = [i for i in range(18) if i not in DYNAMIC_SET and hist[i] > 0]
    ref = float(np.median([hist[i] for i in allowed])) if allowed else 1.0
    w = np.zeros(18, dtype=np.float32)
    for i in allowed:
        w[i] = float(np.clip(math.sqrt(ref / max(float(hist[i]), 1.0)), 0.25, 8.0))
    return hist, torch.from_numpy(w)


def _tile_masks(obs_np, start, shape, high, coarse):
    sx, sy, sz = (int(x) for x in start)
    dx, dy, dz = (int(x) for x in shape)
    hi_step = np.asarray(high.voxel_size_xyz_m)
    co_step = np.asarray(coarse.voxel_size_xyz_m)
    factor = np.maximum(np.rint(co_step / hi_step).astype(np.int64), 1)
    x = np.arange(sx, sx + dx) // factor[0]
    y = np.arange(sy, sy + dy) // factor[1]
    z = np.arange(sz, sz + dz) // factor[2]
    x = np.clip(x, 0, coarse.shape_xyz[0] - 1)
    y = np.clip(y, 0, coarse.shape_xyz[1] - 1)
    z = np.clip(z, 0, coarse.shape_xyz[2] - 1)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    seen = obs_np[:, xx, yy, zz].any(axis=0)
    t0 = obs_np[-1, xx, yy, zz]
    missing = seen & ~t0
    return seen, missing


def _tile_loss(model, scene, high_idx, labels, obs_np, high, coarse, tile_size, weights):
    if len(high_idx) == 0:
        return scene.sum() * 0.0, np.zeros((17, 17), dtype=np.int64)
    tile_size = np.asarray(tile_size, dtype=np.int64)
    groups = {}
    for i, cell in enumerate(high_idx):
        key = tuple((cell // tile_size).tolist())
        groups.setdefault(key, []).append(i)
    loss = scene.sum() * 0.0
    conf = np.zeros((17, 17), dtype=np.int64)
    ng = 0
    for key, ids in groups.items():
        start = np.asarray(key, dtype=np.int64) * tile_size
        stop = np.minimum(start + tile_size, np.asarray(high.shape_xyz))
        tshape = tuple((stop - start).tolist())
        grid = canonical_tile_grid_sample_coordinates(
            high, coarse, start, tshape, device=scene.device, dtype=scene.dtype
        )
        seen, missing = _tile_masks(obs_np, start, tshape, high, coarse)
        query = np.ones(tshape, dtype=bool)
        logits = model.static.refine_tiles(
            scene,
            sample_grid=grid,
            query_mask=torch.from_numpy(query).to(scene.device).unsqueeze(0),
            seen_mask=torch.from_numpy(seen).to(scene.device).unsqueeze(0),
            t0_missing_mask=torch.from_numpy(missing).to(scene.device).unsqueeze(0),
        )
        local = high_idx[ids] - start[None]
        y = labels[ids]
        li = torch.as_tensor(local, dtype=torch.long, device=scene.device)
        yt = torch.as_tensor(y, dtype=torch.long, device=scene.device)
        rows = logits[0, :, li[:, 0], li[:, 1], li[:, 2]].transpose(0, 1).clone()
        dyn = torch.as_tensor(tuple(DYNAMIC_IDS), dtype=torch.long, device=scene.device)
        rows[:, dyn] = torch.finfo(rows.dtype).min
        loss = loss + F.cross_entropy(rows, yt, weight=weights)
        pred = rows.detach().float().argmax(-1).cpu().numpy()
        valid_sem = y < 17
        if np.any(valid_sem):
            code = y[valid_sem] * 17 + pred[valid_sem]
            conf += np.bincount(code, minlength=17 * 17).reshape(17, 17)
        ng += 1
    return loss / max(ng, 1), conf


def _miou(conf):
    vals = {}
    xs = []
    for cid in range(17):
        if cid in DYNAMIC_SET:
            continue
        tp = int(conf[cid, cid])
        u = int(conf[cid, :].sum() + conf[:, cid].sum() - tp)
        v = float(tp / u) if u else float("nan")
        vals[str(cid)] = v
        if u:
            xs.append(v)
    return float(np.mean(xs)) if xs else float("nan"), vals


def _autocast(device, enabled):
    if enabled and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def _epoch(model, root, idx, high, coarse, native_origin, native_step, device, weights, *, optimizer, tile_size, seed, amp):
    train = optimizer is not None
    model.train(train)
    # Dormant/Birth are not part of Stage 2.
    model.dormant.eval(); model.birth.eval()
    sums = {"loss": 0.0, "coarse": 0.0, "tile": 0.0}
    conf = np.zeros((17, 17), dtype=np.int64)
    n = 0
    for row in _iter_rows(root, idx, train, seed):
        sem, obs, free, obs_np = _row_history(row, device)
        coarse_idx, coarse_y = _aggregate_sparse_targets(row, coarse, native_origin, native_step)
        high_idx, high_y = _aggregate_sparse_targets(row, high, native_origin, native_step)
        with _autocast(device, amp):
            scene = model.encode_history(sem, obs, free)
            clogits = model.static.forward_coarse(scene)
            lc = _point_ce(clogits, coarse_idx, coarse_y, weights)
            lt, c = _tile_loss(
                model, scene, high_idx, high_y, obs_np, high, coarse,
                tile_size, weights,
            )
            loss = lc + lt
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("non-finite V20 Static loss")
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.encoder.parameters()) + list(model.static.parameters()), 5.0
            )
            optimizer.step()
        sums["loss"] += float(loss.detach().cpu())
        sums["coarse"] += float(lc.detach().cpu())
        sums["tile"] += float(lt.detach().cpu())
        conf += c
        n += 1
    miou, per = _miou(conf)
    return {
        **{k: float(v / max(n, 1)) for k, v in sums.items()},
        "static_supervised_semantic_miou": miou,
        "per_class_iou": per,
        "windows": n,
        "checkpoint_selection_metric": "NONE; run formal composed V20 Static evaluation",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-cache", required=True)
    p.add_argument("--val-cache", required=True)
    p.add_argument("--v18-checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--tile-size", default="32,32,16")
    p.add_argument("--seed", type=int, default=20260925)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-amp", action="store_true")
    a = p.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    tr_root, tr_idx = _load_index(a.train_cache)
    va_root, va_idx = _load_index(a.val_cache)
    overlap = set(tr_idx["scene_names"]) & set(va_idx["scene_names"])
    if overlap:
        raise RuntimeError(f"train/val scene overlap: {sorted(overlap)[:5]}")
    high = _lattice(tr_idx["highres_lattice"])
    coarse = _lattice(tr_idx["coarse_lattice"])
    if va_idx["highres_lattice"] != tr_idx["highres_lattice"]:
        raise RuntimeError("train/val Ωmax mismatch")

    _, weights_cpu = _class_weights(tr_root, tr_idx)
    cfg = V20SceneConfig()
    model = V20HistoryWorldModel(cfg)
    # Stage 2 optimizes only encoder + Static.
    for p0 in model.dormant.parameters(): p0.requires_grad = False
    for p0 in model.birth.parameters(): p0.requires_grad = False
    device = torch.device(a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu")
    model.to(device)
    weights = weights_cpu.to(device)
    optimizer = torch.optim.AdamW(
        [p0 for p0 in model.parameters() if p0.requires_grad],
        lr=float(a.lr), weight_decay=float(a.weight_decay)
    )
    amp = device.type == "cuda" and not bool(a.no_amp)
    tile_size = tuple(int(x) for x in a.tile_size.split(","))
    native_origin = (-40.0, -40.0, -1.0)
    native_step = (0.4, 0.4, 0.4)

    out = Path(a.output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing non-empty output dir: {out}")
    out.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, int(a.epochs) + 1):
        tr = _epoch(
            model, tr_root, tr_idx, high, coarse, native_origin, native_step,
            device, weights, optimizer=optimizer, tile_size=tile_size,
            seed=int(a.seed) + epoch, amp=amp,
        )
        with torch.no_grad():
            va = _epoch(
                model, va_root, va_idx, high, coarse, native_origin, native_step,
                device, weights, optimizer=None, tile_size=tile_size,
                seed=int(a.seed), amp=amp,
            )
        row = {"epoch": epoch, "train": tr, "val": va}
        history.append(row)
        print(json.dumps(row))
        payload = checkpoint_payload(
            model,
            stage="static",
            v18_checkpoint=str(Path(a.v18_checkpoint).resolve()),
            thresholds={},
            extra={
                "train_protocol": PROTOCOL,
                "epoch": epoch,
                "highres_lattice": tr_idx["highres_lattice"],
                "coarse_lattice": tr_idx["coarse_lattice"],
                "tile_size_xyz": list(tile_size),
                "class_weights": weights_cpu.tolist(),
                "selection": (
                    "No automatic best.pt. Select epoch only from formal composed "
                    "semantic mIoU on the scene-disjoint development set."
                ),
                "history": history,
            },
        )
        torch.save(payload, out / f"epoch_{epoch:04d}.pt")
        torch.save(payload, out / "latest.pt")
    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
