#!/usr/bin/env python3
"""Train V20 Stage-2 Static Repair v2.

Unlike the legacy Static-v1 trainer, this objective is mathematically aligned
with protected add-only deployment.  It uses full formal future-grid
supervision on V18-free locations, feeds the exact runtime future-union query
mask to the tile head, treats free/dynamic GT as no-add, retains every horizon
contribution after canonical mapping, uses ordinary CE, and aggregates loss by
supervised future voxels rather than equal-weighting tiles.
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from real_motion.v20_history_world import (
    CanonicalLattice,
    canonical_tile_grid_sample_coordinates,
    future_native_to_canonical_indices,
)
from real_motion.v20_scene_model import V20HistoryWorldModel, V20SceneConfig
from real_motion.v20_stage1_codec import (
    unpack_bool,
    unpack_history_semantic,
)
from real_motion.v20_static_repair import (
    FREE_LABEL,
    REPAIR_CACHE_PROTOCOL,
    REPAIR_STAGE,
    REPAIR_TRAIN_PROTOCOL,
    STATIC_POSITIVE_IDS,
    repair_confusion_summary,
    unpack_static_repair_supervision,
)
from real_motion.v20_training import checkpoint_payload
from tools.real_motion.build_p0_f9_v20_history_cache import (
    PROTOCOL as STAGE1_PROTOCOL,
)

PROTOCOL = REPAIR_TRAIN_PROTOCOL
_ALLOWED_IDS = tuple(STATIC_POSITIVE_IDS) + (FREE_LABEL,)
_GLOBAL_TO_LOCAL = np.full(18, -1, dtype=np.int64)
for _local, _global in enumerate(_ALLOWED_IDS):
    _GLOBAL_TO_LOCAL[int(_global)] = int(_local)
_FREE_LOCAL = int(_GLOBAL_TO_LOCAL[FREE_LABEL])


def _sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            x = f.read(1 << 20)
            if not x:
                break
            h.update(x)
    return h.hexdigest()


def _load_index(root, protocol):
    root = Path(root)
    path = root / "index.json"
    idx = json.loads(path.read_text(encoding="utf-8"))
    if idx.get("protocol") != protocol:
        raise RuntimeError(
            f"unexpected cache protocol at {root}: {idx.get('protocol')}"
        )
    return root, idx


def _lattice(d):
    return CanonicalLattice(
        tuple(float(x) for x in d["origin_xyz_m"]),
        tuple(float(x) for x in d["voxel_size_xyz_m"]),
        tuple(int(x) for x in d["shape_xyz"]),
    )


def _validate_pair(history_root, hidx, repair_root, ridx, v18_checkpoint):
    if str(Path(ridx["source_history_cache"]).resolve()) != str(
        Path(history_root).resolve()
    ):
        raise RuntimeError("Repair cache references a different Stage-1 history cache")
    got_hash = _sha256_file(Path(history_root) / "index.json")
    if str(ridx.get("source_history_index_sha256")) != got_hash:
        raise RuntimeError("Repair cache Stage-1 index hash mismatch")
    if str(Path(ridx["v18_checkpoint"]).resolve()) != str(
        Path(v18_checkpoint).resolve()
    ):
        raise RuntimeError("Repair cache references a different frozen V18 checkpoint")
    if ridx["highres_lattice"] != hidx["highres_lattice"]:
        raise RuntimeError("Repair/history high-resolution lattice mismatch")
    if ridx["coarse_lattice"] != hidx["coarse_lattice"]:
        raise RuntimeError("Repair/history coarse lattice mismatch")
    if ridx["native_grid"] != hidx["native_grid"]:
        raise RuntimeError("Repair/history native grid mismatch")

    hshards = {str(x["file"]): x for x in hidx["shards"]}
    total = 0
    for rsh in ridx["shards"]:
        hname = str(rsh["source_history_shard"])
        hs = hshards.get(hname)
        if hs is None:
            raise RuntimeError(f"Repair shard references missing history shard: {hname}")
        if int(rsh["count"]) > int(hs["count"]):
            raise RuntimeError(f"Repair shard has more rows than history shard: {hname}")
        total += int(rsh["count"])
    if total != int(ridx["num_windows"]):
        raise RuntimeError("Repair index shard counts do not sum to num_windows")


def _iter_paired_rows(history_root, hidx, repair_root, ridx, shuffle, seed, skip=0):
    order = list(range(len(ridx["shards"])))
    rng = random.Random(int(seed))
    if shuffle:
        rng.shuffle(order)
    remaining_skip = max(int(skip), 0)

    for si in order:
        rmeta = ridx["shards"][si]
        rname = str(rmeta["file"])
        hname = str(rmeta["source_history_shard"])
        robj = torch.load(
            repair_root / rname, map_location="cpu", weights_only=False
        )
        hobj = torch.load(
            history_root / hname, map_location="cpu", weights_only=False
        )
        if robj.get("protocol") != REPAIR_CACHE_PROTOCOL:
            raise RuntimeError(f"bad Repair-v2 shard: {rname}")
        if hobj.get("protocol") != STAGE1_PROTOCOL:
            raise RuntimeError(f"bad Stage-1 shard: {hname}")
        rrows = list(robj["rows"])
        hrows = list(hobj["rows"])[: len(rrows)]
        if len(rrows) != int(rmeta["count"]) or len(hrows) != len(rrows):
            raise RuntimeError("paired Repair/history shard row-count mismatch")

        indices = list(range(len(rrows)))
        if shuffle:
            rng.shuffle(indices)
        for i in indices:
            hr = hrows[i]
            rr = rrows[i]
            hk = (str(hr["scene_name"]), str(hr["t0_token"]))
            rk = (str(rr["scene_name"]), str(rr["t0_token"]))
            if hk != rk:
                raise RuntimeError(
                    f"paired Repair/history identity mismatch: {hk} != {rk}"
                )
            if remaining_skip:
                remaining_skip -= 1
                continue
            yield hr, rr


def _decode_history(row):
    shape = (6,) + tuple(int(x) for x in row["coarse_shape_xyz"])
    obs = unpack_bool(row["history_observed_bits"], shape)
    free = unpack_bool(row["history_observed_free_bits"], shape)
    sem = unpack_history_semantic(row, obs, free)
    return sem, obs, free


def _tile_entries_from_exact_targets(
    *,
    high,
    tile_size,
    render_index,
    v18_occupied,
    static_positive,
    positive_labels,
):
    """Prepare exact deployed-support loss terms for one window.

    Every V18-free future voxel contributes one target.  It is FREE by default;
    static-positive future voxels replace that one FREE target by their semantic
    label.  Contributions are accumulated after future->canonical mapping
    without merging or discarding horizon conflicts.
    """
    if int(render_index.out_of_bounds_voxels) != 0:
        raise RuntimeError(
            "Repair-v2 requires frozen Ωmax OOB=0; got "
            f"{render_index.out_of_bounds_voxels}"
        )
    shape = tuple(int(x) for x in high.shape_xyz)
    high_n = int(np.prod(shape))
    linear = np.asarray(render_index.linear_index, dtype=np.int32).reshape(-1)
    occ = np.asarray(v18_occupied, dtype=bool).reshape(-1)
    pos = np.asarray(static_positive, dtype=bool).reshape(-1)
    if linear.size != occ.size or pos.size != occ.size:
        raise RuntimeError("Repair-v2 native geometry/supervision size mismatch")
    if bool((pos & occ).any()):
        raise RuntimeError("Repair-v2 positive overlaps V18 occupied support")

    query = np.zeros(high_n, dtype=bool)
    query[linear] = True

    # Rigid equal-resolution mapping plus six horizons keeps multiplicity tiny;
    # uint16 is intentionally conservative and cannot overflow here in practice.
    support_count = np.zeros(high_n, dtype=np.uint16)
    support_linear = linear[~occ]
    np.add.at(support_count, support_linear, 1)
    expected_support = int((~occ).sum())
    if int(support_count.sum(dtype=np.int64)) != expected_support:
        raise RuntimeError("Repair-v2 support multiplicity accounting mismatch")

    pos_linear = linear[pos]
    if len(pos_linear) != len(positive_labels):
        raise RuntimeError("Repair-v2 positive label/mapping count mismatch")
    pos_xyz = (
        np.column_stack(np.unravel_index(pos_linear, shape)).astype(np.int64)
        if len(pos_linear)
        else np.empty((0, 3), dtype=np.int64)
    )

    tile = np.asarray(tile_size, dtype=np.int64)
    hshape = np.asarray(shape, dtype=np.int64)
    tgrid = tuple(np.ceil(hshape / tile).astype(np.int64).tolist())
    pos_by_tile = {}
    if len(pos_xyz):
        pxyz = pos_xyz // tile[None]
        plin = np.ravel_multi_index(pxyz.T, tgrid)
        order = np.argsort(plin, kind="stable")
        plin_s = plin[order]
        starts = np.r_[0, 1 + np.flatnonzero(plin_s[1:] != plin_s[:-1])]
        stops = np.r_[starts[1:], len(order)]
        for s, e in zip(starts, stops):
            ids = order[s:e]
            key = int(plin_s[s])
            pos_by_tile[key] = (
                pos_xyz[ids],
                np.asarray(positive_labels, dtype=np.uint8)[ids],
            )

    q3 = query.reshape(shape)
    c3 = support_count.reshape(shape)
    entries = []
    total_support = 0
    total_positive = 0
    for x in range(0, shape[0], int(tile[0])):
        for y in range(0, shape[1], int(tile[1])):
            for z in range(0, shape[2], int(tile[2])):
                start = np.asarray([x, y, z], dtype=np.int64)
                stop = np.minimum(start + tile, hshape)
                qtile = q3[
                    x:stop[0], y:stop[1], z:stop[2]
                ]
                if not bool(qtile.any()):
                    continue
                counts = c3[
                    x:stop[0], y:stop[1], z:stop[2]
                ]
                support_n = int(counts.sum(dtype=np.int64))
                if support_n == 0:
                    # Static cannot affect any future voxel represented by this
                    # tile because V18 already owns all of them.
                    continue
                key_xyz = start // tile
                key = int(np.ravel_multi_index(tuple(key_xyz), tgrid))
                pxyz, plabel = pos_by_tile.get(
                    key,
                    (
                        np.empty((0, 3), dtype=np.int64),
                        np.empty((0,), dtype=np.uint8),
                    ),
                )
                local = pxyz - start[None] if len(pxyz) else pxyz
                tshape = tuple((stop - start).tolist())
                entries.append({
                    "start": tuple(int(v) for v in start),
                    "shape": tshape,
                    "query": qtile.copy(),
                    "support_count": counts.copy(),
                    "positive_local": local.astype(np.int16, copy=False),
                    "positive_label": plabel.astype(np.uint8, copy=False),
                })
                total_support += support_n
                total_positive += int(len(plabel))

    if total_support != expected_support:
        raise RuntimeError(
            f"Repair-v2 tiled support mismatch: {total_support} != {expected_support}"
        )
    if total_positive != int(pos.sum()):
        raise RuntimeError("Repair-v2 tiled positive count mismatch")
    return entries, total_support, total_positive


def _prepare_pair(hr, rr, high, native, tile_size):
    started = time.perf_counter()
    sem, obs, free = _decode_history(hr)
    v18_occ, static_pos, positive_labels = unpack_static_repair_supervision(rr)
    expected_shape = tuple(int(x) for x in rr["shape_fxyz"])
    native_shape = tuple(int(x) for x in native["shape_xyz"])
    if expected_shape != (6,) + native_shape:
        raise RuntimeError("Repair-v2 cached native shape mismatch")
    render_index = future_native_to_canonical_indices(
        high,
        future_ego_to_canonical=np.asarray(
            hr["future_ego_to_t0"], dtype=np.float64
        ),
        native_shape_xyz=native_shape,
        native_origin_xyz_m=tuple(float(x) for x in native["origin_xyz_m"]),
        native_voxel_size_xyz_m=tuple(
            float(x) for x in native["voxel_size_xyz_m"]
        ),
    )
    entries, support_n, positive_n = _tile_entries_from_exact_targets(
        high=high,
        tile_size=tile_size,
        render_index=render_index,
        v18_occupied=v18_occ,
        static_positive=static_pos,
        positive_labels=positive_labels,
    )
    if support_n != int(rr["support_count"]):
        raise RuntimeError("Repair-v2 prepared support count differs from cache")
    if positive_n != int(rr["static_positive_count"]):
        raise RuntimeError("Repair-v2 prepared positive count differs from cache")
    return {
        "sem": sem,
        "obs": obs,
        "free": free,
        "entries": entries,
        "support_count": support_n,
        "positive_count": positive_n,
        "cpu_seconds": float(time.perf_counter() - started),
    }


def _iter_prepared(
    history_root,
    hidx,
    repair_root,
    ridx,
    *,
    shuffle,
    seed,
    high,
    native,
    tile_size,
    workers,
    prefetch,
    skip=0,
):
    raw = iter(
        _iter_paired_rows(
            history_root,
            hidx,
            repair_root,
            ridx,
            shuffle,
            seed,
            skip=skip,
        )
    )

    def prep(pair):
        return _prepare_pair(
            pair[0], pair[1], high, native, tile_size
        )

    nw = max(int(workers), 0)
    if nw == 0:
        for pair in raw:
            yield prep(pair)
        return

    cap = max(int(prefetch), nw)
    with ThreadPoolExecutor(
        max_workers=nw, thread_name_prefix="v20-repair-prep"
    ) as pool:
        pending = deque()
        exhausted = False
        for _ in range(cap):
            try:
                pending.append(pool.submit(prep, next(raw)))
            except StopIteration:
                exhausted = True
                break
        while pending:
            result = pending.popleft().result()
            if not exhausted:
                try:
                    pending.append(pool.submit(prep, next(raw)))
                except StopIteration:
                    exhausted = True
            yield result


def _tile_coarse_linear_map(start, shape, high, coarse, cache):
    key = (tuple(int(x) for x in start), tuple(int(x) for x in shape))
    got = cache.get(key)
    if got is not None:
        return got
    start = np.asarray(key[0], dtype=np.int64)
    tshape = np.asarray(key[1], dtype=np.int64)
    factor = np.maximum(
        np.rint(
            np.asarray(coarse.voxel_size_xyz_m)
            / np.asarray(high.voxel_size_xyz_m)
        ).astype(np.int64),
        1,
    )
    x = np.clip(
        np.arange(start[0], start[0] + tshape[0]) // factor[0],
        0, coarse.shape_xyz[0] - 1,
    )
    y = np.clip(
        np.arange(start[1], start[1] + tshape[1]) // factor[1],
        0, coarse.shape_xyz[1] - 1,
    )
    z = np.clip(
        np.arange(start[2], start[2] + tshape[2]) // factor[2],
        0, coarse.shape_xyz[2] - 1,
    )
    Y, Z = int(coarse.shape_xyz[1]), int(coarse.shape_xyz[2])
    lin = (
        x[:, None, None] * (Y * Z)
        + y[None, :, None] * Z
        + z[None, None, :]
    ).reshape(-1)
    cache[key] = lin
    return lin


def _autocast(device, enabled):
    if enabled and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def _assert_finite_async(x):
    finite = torch.isfinite(x.detach()).all()
    if x.device.type == "cuda" and hasattr(torch, "_assert_async"):
        torch._assert_async(finite, "non-finite V20 Static Repair loss")
    elif not bool(finite.item()):
        raise RuntimeError("non-finite V20 Static Repair loss")


def _repair_loss_and_confusion(
    model,
    scene,
    obs_np,
    entries,
    *,
    high,
    coarse,
    tile_batch_size,
    grid_cache,
    context_index_cache,
    allowed_ids,
    global_to_local,
):
    if not entries:
        z = torch.zeros((18, 18), dtype=torch.int64, device=scene.device)
        return scene.sum() * 0.0, z, 0, 0

    buckets = {}
    for e in entries:
        buckets.setdefault(tuple(e["shape"]), []).append(e)

    seen_coarse = np.asarray(obs_np, dtype=bool).any(axis=0).reshape(-1)
    t0_coarse = np.asarray(obs_np, dtype=bool)[-1].reshape(-1)
    bsz = max(int(tile_batch_size), 1)
    loss_num = scene.sum() * 0.0
    conf = torch.zeros((18, 18), dtype=torch.int64, device=scene.device)
    support_total = 0
    positive_total = 0
    free_local = int(_FREE_LOCAL)

    for tshape, bucket in buckets.items():
        for bi in range(0, len(bucket), bsz):
            chunk = bucket[bi:bi + bsz]
            grids = []
            qrows = []
            count_rows = []
            seen_rows = []
            missing_rows = []
            p_batch = []
            p_x = []
            p_y = []
            p_z = []
            p_label = []

            for local_b, entry in enumerate(chunk):
                start = tuple(int(x) for x in entry["start"])
                cache_key = (
                    start, tuple(tshape), str(scene.device), str(scene.dtype)
                )
                grid = grid_cache.get(cache_key)
                if grid is None:
                    grid = canonical_tile_grid_sample_coordinates(
                        high,
                        coarse,
                        start,
                        tshape,
                        device=scene.device,
                        dtype=scene.dtype,
                    )
                    grid_cache[cache_key] = grid
                grids.append(grid)
                qrows.append(entry["query"])
                count_rows.append(entry["support_count"])

                cmap = _tile_coarse_linear_map(
                    start, tshape, high, coarse, context_index_cache
                )
                seen = seen_coarse[cmap].reshape(tshape)
                t0_seen = t0_coarse[cmap].reshape(tshape)
                seen_rows.append(seen)
                missing_rows.append(seen & ~t0_seen)

                local = np.asarray(entry["positive_local"], dtype=np.int64)
                labels = np.asarray(entry["positive_label"], dtype=np.int64)
                if len(local):
                    n = len(local)
                    p_batch.append(np.full(n, local_b, dtype=np.int64))
                    p_x.append(local[:, 0])
                    p_y.append(local[:, 1])
                    p_z.append(local[:, 2])
                    p_label.append(labels)

            B = len(chunk)
            q = torch.from_numpy(np.stack(qrows, axis=0)).to(
                scene.device, non_blocking=True
            )
            counts = torch.from_numpy(
                np.stack(count_rows, axis=0)
            ).to(scene.device, non_blocking=True)
            seen_t = torch.from_numpy(np.stack(seen_rows, axis=0)).to(
                scene.device, non_blocking=True
            )
            missing_t = torch.from_numpy(
                np.stack(missing_rows, axis=0)
            ).to(scene.device, non_blocking=True)

            logits = model.static.refine_tiles(
                scene,
                sample_grid=torch.cat(grids, dim=0),
                query_mask=q,
                seen_mask=seen_t,
                t0_missing_mask=missing_t,
            )
            allowed_logits = logits.index_select(1, allowed_ids)
            logp = F.log_softmax(allowed_logits.float(), dim=1)
            free_lp = logp[:, free_local]
            count_f = counts.to(logp.dtype)
            loss_num = loss_num - (free_lp * count_f).sum()
            support_total += int(
                sum(int(e["support_count"].sum(dtype=np.int64)) for e in chunk)
            )

            pred_local = allowed_logits.detach().float().argmax(dim=1)
            pred_global = allowed_ids[pred_local]
            flat_count = counts.reshape(-1).to(torch.int64)
            flat_pred = pred_global.reshape(-1).to(torch.int64)
            active = flat_count > 0
            if bool(active.any()):
                codes = int(FREE_LABEL) * 18 + flat_pred[active]
                base_conf = torch.zeros(
                    18 * 18, dtype=torch.int64, device=scene.device
                )
                base_conf.scatter_add_(0, codes, flat_count[active])
                conf += base_conf.reshape(18, 18)

            if p_batch:
                bt = torch.from_numpy(np.concatenate(p_batch)).to(
                    scene.device, non_blocking=True
                )
                ix = torch.from_numpy(np.concatenate(p_x)).to(
                    scene.device, non_blocking=True
                )
                iy = torch.from_numpy(np.concatenate(p_y)).to(
                    scene.device, non_blocking=True
                )
                iz = torch.from_numpy(np.concatenate(p_z)).to(
                    scene.device, non_blocking=True
                )
                yg = torch.from_numpy(np.concatenate(p_label)).to(
                    scene.device, non_blocking=True
                ).long()
                yl = global_to_local[yg]
                if bool((yl < 0).any()):
                    raise RuntimeError(
                        "Repair-v2 positive target mapped outside static taxonomy"
                    )
                pos_cls_lp = logp[bt, yl, ix, iy, iz]
                pos_free_lp = free_lp[bt, ix, iy, iz]
                # Replace one implicit FREE contribution by its static semantic
                # target for every future-horizon positive voxel.
                loss_num = loss_num + (-pos_cls_lp + pos_free_lp).sum()
                positive_total += int(yg.numel())

                pp = pred_global[bt, ix, iy, iz].to(torch.int64)
                neg_codes = int(FREE_LABEL) * 18 + pp
                pos_codes = yg.to(torch.int64) * 18 + pp
                corr = torch.zeros(
                    18 * 18, dtype=torch.int64, device=scene.device
                )
                ones = torch.ones_like(pos_codes, dtype=torch.int64)
                corr.scatter_add_(0, neg_codes, -ones)
                corr.scatter_add_(0, pos_codes, ones)
                conf += corr.reshape(18, 18)

    if support_total <= 0:
        raise RuntimeError("Repair-v2 window has zero V18-free supervision")
    return loss_num / float(support_total), conf, support_total, positive_total


def _capture_rng_state():
    out = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        out["torch_cuda"] = torch.cuda.get_rng_state_all()
    return out


def _restore_rng_state(state):
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _atomic_torch_save(obj, path):
    path = Path(path)
    tmp = path.with_name("." + path.name + ".tmp")
    try:
        if tmp.exists():
            tmp.unlink()
        torch.save(obj, tmp)
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _epoch(
    model,
    history_root,
    hidx,
    repair_root,
    ridx,
    high,
    coarse,
    native,
    device,
    *,
    optimizer,
    tile_size,
    tile_batch_size,
    prep_workers,
    prefetch,
    progress_every,
    seed,
    amp,
    start_window=0,
    resume_accum=None,
    checkpoint_every=0,
    checkpoint_callback=None,
    geometry_caches=None,
):
    train = optimizer is not None
    model.train(train)
    model.dormant.eval()
    model.birth.eval()
    model.static.coarse_head.eval()

    resume_accum = dict(resume_accum or {})
    n = int(start_window)
    total = int(ridx["num_windows"])
    if n < 0 or n > total:
        raise ValueError("invalid Repair-v2 resume window")

    sums_gpu = torch.tensor(
        float(resume_accum.get("loss_sum_windows", 0.0)),
        dtype=torch.float32,
        device=device,
    )
    conf0 = np.asarray(
        resume_accum.get("confusion", np.zeros((18, 18), dtype=np.int64)),
        dtype=np.int64,
    )
    conf_gpu = torch.as_tensor(conf0, dtype=torch.int64, device=device).clone()
    support_seen = int(resume_accum.get("support_contributions", 0))
    positive_seen = int(resume_accum.get("positive_contributions", 0))
    cpu_work = float(resume_accum.get("cpu_work_seconds", 0.0))
    prior_elapsed = float(resume_accum.get("elapsed_seconds", 0.0))
    started = time.perf_counter()
    cpu_wait = 0.0

    geometry_caches = geometry_caches if geometry_caches is not None else {}
    grid_cache = geometry_caches.setdefault("grid", {})
    context_cache = geometry_caches.setdefault("context", {})

    allowed_ids = torch.as_tensor(
        _ALLOWED_IDS, dtype=torch.long, device=device
    )
    global_to_local = torch.as_tensor(
        _GLOBAL_TO_LOCAL, dtype=torch.long, device=device
    )

    prepared = iter(_iter_prepared(
        history_root,
        hidx,
        repair_root,
        ridx,
        shuffle=train,
        seed=seed,
        high=high,
        native=native,
        tile_size=tile_size,
        workers=prep_workers,
        prefetch=prefetch,
        skip=n,
    ))

    while n < total:
        tw = time.perf_counter()
        try:
            row = next(prepared)
        except StopIteration:
            break
        cpu_wait += time.perf_counter() - tw
        cpu_work += float(row["cpu_seconds"])

        sem = torch.from_numpy(row["sem"]).to(
            device, non_blocking=device.type == "cuda"
        ).unsqueeze(0)
        obs = torch.from_numpy(row["obs"]).to(
            device, non_blocking=device.type == "cuda"
        ).unsqueeze(0)
        free = torch.from_numpy(row["free"]).to(
            device, non_blocking=device.type == "cuda"
        ).unsqueeze(0)

        if train:
            optimizer.zero_grad(set_to_none=True)
        with _autocast(device, amp):
            scene = model.encode_history(sem, obs, free)
            loss, c, nsupport, npositive = _repair_loss_and_confusion(
                model,
                scene,
                row["obs"],
                row["entries"],
                high=high,
                coarse=coarse,
                tile_batch_size=tile_batch_size,
                grid_cache=grid_cache,
                context_index_cache=context_cache,
                allowed_ids=allowed_ids,
                global_to_local=global_to_local,
            )
        _assert_finite_async(loss)
        if train:
            loss.backward()
            params = [
                p for p in model.parameters()
                if p.requires_grad and p.grad is not None
            ]
            torch.nn.utils.clip_grad_norm_(
                params, 5.0, error_if_nonfinite=True, foreach=True
            )
            optimizer.step()

        sums_gpu.add_(loss.detach().float())
        conf_gpu.add_(c)
        support_seen += int(nsupport)
        positive_seen += int(npositive)
        n += 1

        report = (
            n == int(start_window) + 1
            or n % max(int(progress_every), 1) == 0
            or n == total
        )
        checkpoint = (
            train
            and checkpoint_callback is not None
            and int(checkpoint_every) > 0
            and n < total
            and n % int(checkpoint_every) == 0
        )
        if report or checkpoint:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            now = time.perf_counter()
            segment_n = max(n - int(start_window), 1)
            if report:
                summary = repair_confusion_summary(
                    conf_gpu.detach().cpu().numpy()
                )
                print(
                    f"v20_static_repair_{'train' if train else 'val'} "
                    f"{n}/{total} "
                    f"rate={segment_n/max(now-started,1e-9):.3f} win/s "
                    f"cpu_wait={cpu_wait/segment_n:.3f}s/win "
                    f"cpu_work={cpu_work/max(n,1):.3f}s/win "
                    f"loss={float(sums_gpu.item()/max(n,1)):.4f} "
                    f"addP={summary['added_precision']:.4f} "
                    f"addR={summary['added_recall']:.4f} "
                    f"repair_mIoU={summary['repair_static_mIoU']:.4f}",
                    flush=True,
                )
            if checkpoint:
                checkpoint_callback(
                    n,
                    {
                        "loss_sum_windows": float(sums_gpu.item()),
                        "confusion": conf_gpu.detach().cpu().numpy().tolist(),
                        "support_contributions": int(support_seen),
                        "positive_contributions": int(positive_seen),
                        "cpu_work_seconds": float(cpu_work),
                        "elapsed_seconds": float(
                            prior_elapsed + now - started
                        ),
                    },
                )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    conf = conf_gpu.detach().cpu().numpy()
    summary = repair_confusion_summary(conf)
    elapsed = prior_elapsed + time.perf_counter() - started
    return {
        "loss": float(sums_gpu.item() / max(n, 1)),
        "windows": int(n),
        "support_contributions": int(support_seen),
        "positive_contributions": int(positive_seen),
        "positive_fraction": float(
            positive_seen / max(support_seen, 1)
        ),
        "mean_cpu_work_seconds_per_window": float(
            cpu_work / max(n, 1)
        ),
        "mean_cpu_wait_seconds_per_resumed_window": float(
            cpu_wait / max(n - int(start_window), 1)
        ),
        "elapsed_seconds": float(elapsed),
        "repair_metrics": summary,
        "confusion": conf.tolist(),
        "formal_checkpoint_selection_metric": (
            "NONE; use composed V18+Static full-grid dev mIoU"
        ),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-history-cache", required=True)
    p.add_argument("--train-repair-cache", required=True)
    p.add_argument("--val-history-cache", required=True)
    p.add_argument("--val-repair-cache", required=True)
    p.add_argument("--v18-checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--tile-size", default="32,32,16")
    p.add_argument("--tile-batch-size", type=int, default=32)
    p.add_argument("--prep-workers", type=int, default=4)
    p.add_argument("--prefetch", type=int, default=16)
    p.add_argument("--progress-every", type=int, default=50)
    p.add_argument("--checkpoint-every-windows", type=int, default=2000)
    p.add_argument("--resume", default="")
    p.add_argument("--seed", type=int, default=20260926)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-amp", action="store_true")
    a = p.parse_args()

    if int(a.epochs) <= 0:
        raise ValueError("--epochs must be positive")
    if float(a.lr) <= 0 or float(a.weight_decay) < 0:
        raise ValueError("invalid optimizer hyperparameters")

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)

    trh_root, trh = _load_index(a.train_history_cache, STAGE1_PROTOCOL)
    trr_root, trr = _load_index(a.train_repair_cache, REPAIR_CACHE_PROTOCOL)
    vah_root, vah = _load_index(a.val_history_cache, STAGE1_PROTOCOL)
    var_root, var = _load_index(a.val_repair_cache, REPAIR_CACHE_PROTOCOL)
    _validate_pair(trh_root, trh, trr_root, trr, a.v18_checkpoint)
    _validate_pair(vah_root, vah, var_root, var, a.v18_checkpoint)

    overlap = set(trr["scene_names"]) & set(var["scene_names"])
    if overlap:
        raise RuntimeError(f"Repair-v2 train/val scene overlap: {sorted(overlap)[:5]}")
    for key in ("highres_lattice", "coarse_lattice", "native_grid"):
        if trr[key] != var[key]:
            raise RuntimeError(f"Repair-v2 train/val {key} mismatch")

    high = _lattice(trr["highres_lattice"])
    coarse = _lattice(trr["coarse_lattice"])
    native = dict(trr["native_grid"])
    tile_size = tuple(int(x) for x in str(a.tile_size).split(","))
    if len(tile_size) != 3 or min(tile_size) <= 0:
        raise ValueError("invalid --tile-size")

    base = torch.load(
        a.v18_checkpoint, map_location="cpu", weights_only=False
    )
    model_cfg = dict(base.get("model_config") or {})
    if "d_model" not in model_cfg:
        raise RuntimeError("frozen V18 checkpoint lacks model_config.d_model")
    cfg = V20SceneConfig(source_dim=int(model_cfg["d_model"]))
    model = V20HistoryWorldModel(cfg)

    # Repair-v2 optimizes exactly the deployed path: encoder + tile_refine.
    # The unused coarse auxiliary head and later Stage3/4 heads are frozen.
    for p0 in model.static.coarse_head.parameters():
        p0.requires_grad = False
    for p0 in model.dormant.parameters():
        p0.requires_grad = False
    for p0 in model.birth.parameters():
        p0.requires_grad = False

    device = torch.device(
        a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    model.to(device)
    amp = device.type == "cuda" and not bool(a.no_amp)

    params = [p0 for p0 in model.parameters() if p0.requires_grad]
    optimizer = torch.optim.AdamW(
        params,
        lr=float(a.lr),
        weight_decay=float(a.weight_decay),
        foreach=(device.type == "cuda"),
    )

    out = Path(a.output_dir)
    resume_path = Path(a.resume).resolve() if str(a.resume).strip() else None
    resume_ck = None
    if resume_path is None:
        if out.exists() and any(out.iterdir()):
            raise FileExistsError(f"refusing non-empty output dir: {out}")
        out.mkdir(parents=True, exist_ok=True)
        history = []
        start_epoch = 1
        resume_window = 0
        resume_accum = None
    else:
        if not resume_path.is_file():
            raise FileNotFoundError(resume_path)
        resume_ck = torch.load(
            resume_path, map_location="cpu", weights_only=False
        )
        if resume_ck.get("stage") != REPAIR_STAGE:
            raise RuntimeError("resume checkpoint is not Static Repair v2")
        if str(Path(resume_ck.get("v18_checkpoint", "")).resolve()) != str(
            Path(a.v18_checkpoint).resolve()
        ):
            raise RuntimeError("resume V18 checkpoint mismatch")
        extra = dict(resume_ck.get("extra") or {})
        contract = dict(extra.get("training_contract") or {})
        expected = {
            "train_history_cache": str(Path(a.train_history_cache).resolve()),
            "train_repair_cache": str(Path(a.train_repair_cache).resolve()),
            "val_history_cache": str(Path(a.val_history_cache).resolve()),
            "val_repair_cache": str(Path(a.val_repair_cache).resolve()),
            "seed": int(a.seed),
            "tile_size_xyz": list(tile_size),
            "lr": float(a.lr),
            "weight_decay": float(a.weight_decay),
        }
        for k, v in expected.items():
            if contract.get(k) != v:
                raise RuntimeError(
                    f"resume training contract mismatch for {k}: "
                    f"{contract.get(k)!r} != {v!r}"
                )
        model.load_state_dict(resume_ck["model"], strict=True)
        optimizer.load_state_dict(resume_ck["optimizer_state_dict"])
        _restore_rng_state(resume_ck.get("rng_state"))
        progress = dict(resume_ck.get("training_progress") or {})
        history = list(progress.get("history") or [])
        saved_epoch = int(progress.get("epoch", 0))
        if bool(progress.get("epoch_complete", False)):
            start_epoch = saved_epoch + 1
            resume_window = 0
            resume_accum = None
        else:
            start_epoch = saved_epoch
            resume_window = int(progress.get("completed_windows", 0))
            resume_accum = dict(progress.get("partial_epoch_state") or {})
        out.mkdir(parents=True, exist_ok=True)
        print(
            f"resumed V20 Static Repair v2: epoch={start_epoch} "
            f"window={resume_window}",
            flush=True,
        )

    if start_epoch > int(a.epochs):
        raise RuntimeError("resume checkpoint already exceeds requested epochs")

    geometry_caches = {}
    total_train = int(trr["num_windows"])

    def payload(epoch, *, complete, completed_windows, partial, hist):
        obj = checkpoint_payload(
            model,
            stage=REPAIR_STAGE,
            v18_checkpoint=str(Path(a.v18_checkpoint).resolve()),
            thresholds={},
            extra={
                "train_protocol": PROTOCOL,
                "epoch": int(epoch),
                "highres_lattice": trr["highres_lattice"],
                "coarse_lattice": trr["coarse_lattice"],
                "tile_size_xyz": list(tile_size),
                "class_weights": None,
                "loss_contract": (
                    "full_formal_future_grid AND frozen_v18_free; "
                    "static_gt=semantic; free_or_dynamic_gt=FREE; "
                    "ordinary_CE; exact_per_future_voxel_contributions; "
                    "no_tile_equal_weighting; runtime_query_mask"
                ),
                "training_contract": {
                    "train_history_cache": str(Path(a.train_history_cache).resolve()),
                    "train_repair_cache": str(Path(a.train_repair_cache).resolve()),
                    "val_history_cache": str(Path(a.val_history_cache).resolve()),
                    "val_repair_cache": str(Path(a.val_repair_cache).resolve()),
                    "seed": int(a.seed),
                    "tile_size_xyz": list(tile_size),
                    "lr": float(a.lr),
                    "weight_decay": float(a.weight_decay),
                },
                "selection": (
                    "Select only by formal composed V18+Static full-grid "
                    "semantic mIoU on scene-disjoint dev."
                ),
                "history": list(hist),
            },
        )
        obj["optimizer_state_dict"] = optimizer.state_dict()
        obj["rng_state"] = _capture_rng_state()
        obj["training_progress"] = {
            "epoch": int(epoch),
            "epoch_complete": bool(complete),
            "completed_windows": int(completed_windows),
            "total_windows": total_train,
            "partial_epoch_state": partial,
            "history": list(hist),
        }
        return obj

    for epoch in range(int(start_epoch), int(a.epochs) + 1):
        sw = int(resume_window) if epoch == int(start_epoch) else 0
        sa = resume_accum if epoch == int(start_epoch) else None

        def save_partial(completed_windows, partial_state):
            _atomic_torch_save(
                payload(
                    epoch,
                    complete=False,
                    completed_windows=completed_windows,
                    partial=partial_state,
                    hist=history,
                ),
                out / "resume_latest.pt",
            )

        train_report = _epoch(
            model,
            trh_root,
            trh,
            trr_root,
            trr,
            high,
            coarse,
            native,
            device,
            optimizer=optimizer,
            tile_size=tile_size,
            tile_batch_size=int(a.tile_batch_size),
            prep_workers=int(a.prep_workers),
            prefetch=int(a.prefetch),
            progress_every=int(a.progress_every),
            seed=int(a.seed) + epoch,
            amp=amp,
            start_window=sw,
            resume_accum=sa,
            checkpoint_every=int(a.checkpoint_every_windows),
            checkpoint_callback=save_partial,
            geometry_caches=geometry_caches,
        )
        with torch.inference_mode():
            val_report = _epoch(
                model,
                vah_root,
                vah,
                var_root,
                var,
                high,
                coarse,
                native,
                device,
                optimizer=None,
                tile_size=tile_size,
                tile_batch_size=int(a.tile_batch_size),
                prep_workers=int(a.prep_workers),
                prefetch=int(a.prefetch),
                progress_every=int(a.progress_every),
                seed=int(a.seed),
                amp=amp,
                geometry_caches=geometry_caches,
            )

        row = {
            "epoch": int(epoch),
            "train": train_report,
            "val": val_report,
        }
        history.append(row)
        print("REPAIR_EPOCH " + json.dumps(row), flush=True)
        obj = payload(
            epoch,
            complete=True,
            completed_windows=total_train,
            partial=None,
            hist=history,
        )
        _atomic_torch_save(obj, out / f"epoch_{epoch:04d}.pt")
        _atomic_torch_save(obj, out / "latest.pt")
        _atomic_torch_save(obj, out / "resume_latest.pt")
        resume_window = 0
        resume_accum = None

    (out / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
