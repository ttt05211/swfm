#!/usr/bin/env python3
"""Train V20 Stage-2 shared 3D encoder + one canonical Static world."""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
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
    DYNAMIC_IDS,
    canonical_tile_grid_sample_coordinates,
    native_sparse_to_canonical_indices,
    transform_points,
)
from real_motion.v20_scene_model import V20HistoryWorldModel, V20SceneConfig
from real_motion.v20_stage1_codec import (
    unpack_bool,
    unpack_history_semantic,
    unpack_static_indices_and_labels,
    unpack_static_labels,
)
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


def _row_history_cpu(row):
    shape = (6,) + tuple(int(x) for x in row["coarse_shape_xyz"])
    obs = unpack_bool(row["history_observed_bits"], shape)
    free = unpack_bool(row["history_observed_free_bits"], shape)
    sem = unpack_history_semantic(row, obs, free)
    return sem, obs, free


def _prepared_to_device(prepared, device):
    sem_np = prepared["sem"]
    obs_np = prepared["obs"]
    free_np = prepared["free"]
    non_blocking = device.type == "cuda"
    return (
        torch.from_numpy(sem_np).to(device, non_blocking=non_blocking).unsqueeze(0),
        torch.from_numpy(obs_np).to(device, non_blocking=non_blocking).unsqueeze(0),
        torch.from_numpy(free_np).to(device, non_blocking=non_blocking).unsqueeze(0),
        obs_np,
        prepared["coarse_idx"],
        prepared["coarse_y"],
        prepared["high_idx"],
        prepared["high_y"],
    )


def _aggregate_sparse_targets(
    row, lattice, native_shape, native_origin, native_step
):
    """Merge six future observed static/free labels in canonical coordinates.

    This is the exact vectorized equivalent of the former Python dict loop:
    a canonical cell is supervised iff every future observation mapped to that
    cell agrees on one label.  Output order follows first occurrence in the
    six-horizon/native-C-order stream for deterministic parity.
    """
    rel = np.asarray(row["future_ego_to_t0"], dtype=np.float64)
    linear_parts = []
    label_parts = []
    first_offset = 0
    first_parts = []
    shape = tuple(int(x) for x in lattice.shape_xyz)

    for fi, sup in enumerate(row["static_supervision"]):
        native, labels = unpack_static_indices_and_labels(sup, native_shape)
        if len(native) == 0:
            continue
        idx, valid = native_sparse_to_canonical_indices(
            lattice,
            native_indices_xyz=native,
            ego_to_canonical=rel[fi],
            native_origin_xyz_m=native_origin,
            native_voxel_size_xyz_m=native_step,
        )
        if not bool(valid.any()):
            continue
        cells = idx[valid]
        y = labels[valid].astype(np.int64, copy=False)
        lin = np.ravel_multi_index(cells.T, shape)
        linear_parts.append(lin)
        label_parts.append(y)
        first_parts.append(
            np.arange(first_offset, first_offset + len(lin), dtype=np.int64)
        )
        first_offset += len(lin)

    if not linear_parts:
        return np.zeros((0, 3), np.int64), np.zeros((0,), np.int64)

    lin = np.concatenate(linear_parts)
    y = np.concatenate(label_parts)
    source_order = np.concatenate(first_parts)

    order = np.argsort(lin, kind="stable")
    lin_s = lin[order]
    y_s = y[order]
    src_s = source_order[order]
    starts = np.r_[0, 1 + np.flatnonzero(lin_s[1:] != lin_s[:-1])]

    group_lin = lin_s[starts]
    group_first = src_s[starts]
    group_min = np.minimum.reduceat(y_s, starts)
    group_max = np.maximum.reduceat(y_s, starts)
    keep = group_min == group_max
    if not bool(keep.any()):
        return np.zeros((0, 3), np.int64), np.zeros((0,), np.int64)

    # Preserve the previous dict insertion order (first mapped occurrence).
    kept_order = np.argsort(group_first[keep], kind="stable")
    kept_lin = group_lin[keep][kept_order]
    kept_y = group_min[keep][kept_order]
    cells = np.column_stack(np.unravel_index(kept_lin, shape)).astype(
        np.int64, copy=False
    )
    return cells, kept_y.astype(np.int64, copy=False)


def _merge_linear_targets(linear_parts, label_parts, shape):
    if not linear_parts:
        return np.zeros((0, 3), np.int64), np.zeros((0,), np.int64)
    lin = np.concatenate(linear_parts)
    y = np.concatenate(label_parts)
    source_order = np.arange(len(lin), dtype=np.int64)
    order = np.argsort(lin, kind="stable")
    lin_s = lin[order]
    y_s = y[order]
    src_s = source_order[order]
    starts = np.r_[0, 1 + np.flatnonzero(lin_s[1:] != lin_s[:-1])]
    group_lin = lin_s[starts]
    group_first = src_s[starts]
    group_min = np.minimum.reduceat(y_s, starts)
    group_max = np.maximum.reduceat(y_s, starts)
    keep = group_min == group_max
    if not bool(keep.any()):
        return np.zeros((0, 3), np.int64), np.zeros((0,), np.int64)
    kept_order = np.argsort(group_first[keep], kind="stable")
    kept_lin = group_lin[keep][kept_order]
    kept_y = group_min[keep][kept_order]
    cells = np.column_stack(np.unravel_index(kept_lin, shape)).astype(
        np.int64, copy=False
    )
    return cells, kept_y.astype(np.int64, copy=False)


def _aggregate_sparse_targets_pair(
    row, coarse, high, native_shape, native_origin, native_step
):
    """Build coarse/high targets from one decode + rigid transform pass.

    This is algebraically identical to calling _aggregate_sparse_targets twice,
    but avoids decoding the same packed supervision and transforming the same
    native points once for each lattice.
    """
    rel = np.asarray(row["future_ego_to_t0"], dtype=np.float64)
    origin = np.asarray(native_origin, dtype=np.float64)
    step = np.asarray(native_step, dtype=np.float64)

    coarse_lin, coarse_y = [], []
    high_lin, high_y = [], []
    cshape = tuple(int(x) for x in coarse.shape_xyz)
    hshape = tuple(int(x) for x in high.shape_xyz)

    for fi, sup in enumerate(row["static_supervision"]):
        native, labels = unpack_static_indices_and_labels(sup, native_shape)
        if len(native) == 0:
            continue
        xyz = origin[None] + (
            native.astype(np.float64, copy=False) + 0.5
        ) * step[None]
        canon = transform_points(rel[fi], xyz)

        ci, cv = coarse.world_to_index(canon)
        if bool(cv.any()):
            coarse_lin.append(np.ravel_multi_index(ci[cv].T, cshape))
            coarse_y.append(labels[cv].astype(np.int64, copy=False))

        hi, hv = high.world_to_index(canon)
        if bool(hv.any()):
            high_lin.append(np.ravel_multi_index(hi[hv].T, hshape))
            high_y.append(labels[hv].astype(np.int64, copy=False))

    return (
        *_merge_linear_targets(coarse_lin, coarse_y, cshape),
        *_merge_linear_targets(high_lin, high_y, hshape),
    )


def _prepare_row_cpu(
    row, coarse, high, native_shape, native_origin, native_step
):
    started = time.perf_counter()
    sem, obs, free = _row_history_cpu(row)
    coarse_idx, coarse_y, high_idx, high_y = _aggregate_sparse_targets_pair(
        row, coarse, high, native_shape, native_origin, native_step
    )
    return {
        "sem": sem,
        "obs": obs,
        "free": free,
        "coarse_idx": coarse_idx,
        "coarse_y": coarse_y,
        "high_idx": high_idx,
        "high_y": high_y,
        "cpu_seconds": float(time.perf_counter() - started),
    }


def _iter_prepared_rows(
    root,
    idx,
    *,
    shuffle,
    seed,
    coarse,
    high,
    native_shape,
    native_origin,
    native_step,
    workers,
    prefetch,
    skip_windows=0,
):
    """Bounded ordered CPU prefetch without changing the training sample order."""
    raw = iter(_iter_rows(root, idx, shuffle, seed))
    skip = max(int(skip_windows), 0)
    for _ in range(skip):
        try:
            next(raw)
        except StopIteration:
            return

    def prepare(row):
        return _prepare_row_cpu(
            row,
            coarse,
            high,
            native_shape,
            native_origin,
            native_step,
        )

    nw = max(int(workers), 0)
    if nw == 0:
        for row in raw:
            yield prepare(row)
        return

    max_pending = max(int(prefetch), nw)
    with ThreadPoolExecutor(
        max_workers=nw, thread_name_prefix="v20-static-prep"
    ) as pool:
        pending = deque()
        exhausted = False
        for _ in range(max_pending):
            try:
                pending.append(pool.submit(prepare, next(raw)))
            except StopIteration:
                exhausted = True
                break

        while pending:
            fut = pending.popleft()
            result = fut.result()
            if not exhausted:
                try:
                    pending.append(pool.submit(prepare, next(raw)))
                except StopIteration:
                    exhausted = True
            yield result


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
            y = unpack_static_labels(sup).astype(np.int64, copy=False)
            hist += np.bincount(y, minlength=18)[:18]
    allowed = [i for i in range(18) if i not in DYNAMIC_SET and hist[i] > 0]
    ref = float(np.median([hist[i] for i in allowed])) if allowed else 1.0
    w = np.zeros(18, dtype=np.float32)
    for i in allowed:
        w[i] = float(np.clip(math.sqrt(ref / max(float(hist[i]), 1.0)), 0.25, 8.0))
    return hist, torch.from_numpy(w)


def _tile_coarse_linear_map(start, shape, high, coarse, cache):
    key = (tuple(int(x) for x in start), tuple(int(x) for x in shape))
    got = cache.get(key)
    if got is not None:
        return got
    sx, sy, sz = key[0]
    dx, dy, dz = key[1]
    hi_step = np.asarray(high.voxel_size_xyz_m)
    co_step = np.asarray(coarse.voxel_size_xyz_m)
    factor = np.maximum(np.rint(co_step / hi_step).astype(np.int64), 1)
    x = np.clip(
        np.arange(sx, sx + dx, dtype=np.int64) // factor[0],
        0, coarse.shape_xyz[0] - 1,
    )
    y = np.clip(
        np.arange(sy, sy + dy, dtype=np.int64) // factor[1],
        0, coarse.shape_xyz[1] - 1,
    )
    z = np.clip(
        np.arange(sz, sz + dz, dtype=np.int64) // factor[2],
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


def _equal_tile_weighted_ce(rows, targets, tile_ids, weights, num_tiles):
    """Exact vectorization of mean(weighted CE) per tile, then equal tile mean."""
    point = F.cross_entropy(
        rows, targets, weight=weights, reduction="none"
    )
    numer = torch.zeros(
        int(num_tiles), dtype=point.dtype, device=point.device
    )
    denom = torch.zeros_like(numer)
    numer.scatter_add_(0, tile_ids, point)
    denom.scatter_add_(
        0,
        tile_ids,
        weights[targets].to(point.dtype),
    )
    per_tile = numer / denom.clamp_min(torch.finfo(point.dtype).tiny)
    return per_tile.sum()



def _tile_loss(
    model,
    scene,
    high_idx,
    labels,
    obs_np,
    high,
    coarse,
    tile_size,
    weights,
    *,
    tile_batch_size,
    grid_cache,
    context_index_cache,
):
    if len(high_idx) == 0:
        z = torch.zeros((18, 18), dtype=torch.int64, device=scene.device)
        return scene.sum() * 0.0, z, 0

    tile_size = np.asarray(tile_size, dtype=np.int64)
    high_shape = np.asarray(high.shape_xyz, dtype=np.int64)
    tile_grid_shape = tuple(
        np.ceil(high_shape / tile_size).astype(np.int64).tolist()
    )

    tile_xyz = high_idx // tile_size[None]
    tile_lin = np.ravel_multi_index(tile_xyz.T, tile_grid_shape)
    order = np.argsort(tile_lin, kind="stable")
    lin_s = tile_lin[order]
    starts = np.r_[0, 1 + np.flatnonzero(lin_s[1:] != lin_s[:-1])]
    stops = np.r_[starts[1:], len(order)]

    entries = []
    for s, e in zip(starts, stops):
        ids = order[s:e]
        key = tile_xyz[ids[0]]
        start_xyz = key * tile_size
        stop_xyz = np.minimum(start_xyz + tile_size, high_shape)
        tshape = tuple((stop_xyz - start_xyz).tolist())
        entries.append((tuple(start_xyz.tolist()), tshape, ids))

    buckets = {}
    for entry in entries:
        buckets.setdefault(entry[1], []).append(entry)

    loss = scene.sum() * 0.0
    conf = torch.zeros((18, 18), dtype=torch.int64, device=scene.device)
    ng = 0
    bsz = max(int(tile_batch_size), 1)
    dyn = torch.as_tensor(
        tuple(DYNAMIC_IDS), dtype=torch.long, device=scene.device
    )

    # History context is reduced once per window, not once per tile.
    seen_coarse = obs_np.any(axis=0).reshape(-1)
    t0_coarse = obs_np[-1].reshape(-1)

    for tshape, bucket in buckets.items():
        for bi in range(0, len(bucket), bsz):
            chunk = bucket[bi:bi + bsz]
            grids = []
            seen_rows = []
            missing_rows = []
            batch_ids = []
            local_x = []
            local_y = []
            local_z = []
            target_rows = []
            tile_ids = []

            for local_b, (start_t, _, ids) in enumerate(chunk):
                cache_key = (start_t, tshape, str(scene.dtype))
                grid = grid_cache.get(cache_key)
                if grid is None:
                    grid = canonical_tile_grid_sample_coordinates(
                        high,
                        coarse,
                        start_t,
                        tshape,
                        device=scene.device,
                        dtype=scene.dtype,
                    )
                    grid_cache[cache_key] = grid
                grids.append(grid)

                cmap = _tile_coarse_linear_map(
                    start_t, tshape, high, coarse, context_index_cache
                )
                seen = seen_coarse[cmap].reshape(tshape)
                t0_seen = t0_coarse[cmap].reshape(tshape)
                seen_rows.append(seen)
                missing_rows.append(seen & ~t0_seen)

                start_xyz = np.asarray(start_t, dtype=np.int64)
                local = high_idx[ids] - start_xyz[None]
                npt = len(ids)
                batch_ids.append(np.full(npt, local_b, dtype=np.int64))
                local_x.append(local[:, 0])
                local_y.append(local[:, 1])
                local_z.append(local[:, 2])
                target_rows.append(labels[ids].astype(np.int64, copy=False))
                tile_ids.append(np.full(npt, local_b, dtype=np.int64))

            B = len(chunk)
            grid = torch.cat(grids, dim=0)
            seen_t = torch.from_numpy(np.stack(seen_rows, axis=0)).to(
                scene.device, non_blocking=True
            )
            missing_t = torch.from_numpy(
                np.stack(missing_rows, axis=0)
            ).to(scene.device, non_blocking=True)
            query_t = torch.ones(
                (B,) + tuple(tshape),
                dtype=torch.bool,
                device=scene.device,
            )

            logits = model.static.refine_tiles(
                scene.expand(B, -1, -1, -1, -1),
                sample_grid=grid,
                query_mask=query_t,
                seen_mask=seen_t,
                t0_missing_mask=missing_t,
            )

            bt = torch.from_numpy(np.concatenate(batch_ids)).to(
                scene.device, non_blocking=True
            )
            ix = torch.from_numpy(np.concatenate(local_x)).to(
                scene.device, non_blocking=True
            )
            iy = torch.from_numpy(np.concatenate(local_y)).to(
                scene.device, non_blocking=True
            )
            iz = torch.from_numpy(np.concatenate(local_z)).to(
                scene.device, non_blocking=True
            )
            yt = torch.from_numpy(np.concatenate(target_rows)).to(
                scene.device, non_blocking=True
            )
            gid = torch.from_numpy(np.concatenate(tile_ids)).to(
                scene.device, non_blocking=True
            )

            # One gather, CE and confusion update for the entire tile batch.
            rows = logits[bt, :, ix, iy, iz].clone()
            rows[:, dyn] = torch.finfo(rows.dtype).min
            loss = loss + _equal_tile_weighted_ce(
                rows, yt, gid, weights, B
            )

            with torch.no_grad():
                pred = rows.detach().float().argmax(-1)
                valid_sem = yt < 17
                codes = yt[valid_sem] * 18 + pred[valid_sem]
                if codes.numel():
                    conf += torch.bincount(
                        codes, minlength=18 * 18
                    ).reshape(18, 18)
            ng += B

    return loss / max(ng, 1), conf, ng


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


def _assert_finite_async(x):
    finite = torch.isfinite(x.detach()).all()
    if x.device.type == "cuda" and hasattr(torch, "_assert_async"):
        torch._assert_async(finite, "non-finite V20 Static loss")
    elif not bool(finite.item()):
        raise RuntimeError("non-finite V20 Static loss")


def _epoch(
    model, root, idx, high, coarse, native_shape, native_origin, native_step,
    device, weights, *, optimizer, tile_size, tile_batch_size,
    progress_every, prep_workers, prefetch, seed, amp,
    start_window=0, resume_accum=None, checkpoint_every=0,
    checkpoint_callback=None, geometry_caches=None
):
    train = optimizer is not None
    model.train(train)
    # Dormant/Birth are not part of Stage 2.
    model.dormant.eval(); model.birth.eval()
    resume_accum = dict(resume_accum or {})
    sums_gpu = {
        k: torch.tensor(
            float(resume_accum.get("sums", {}).get(k, 0.0)),
            dtype=torch.float32,
            device=device,
        )
        for k in ("loss", "coarse", "tile")
    }
    conf0 = np.asarray(
        resume_accum.get("confusion", np.zeros((18, 18), dtype=np.int64)),
        dtype=np.int64,
    )
    conf_gpu = torch.as_tensor(conf0, dtype=torch.int64, device=device).clone()
    n = int(start_window)
    total = int(idx["num_windows"])
    if n < 0 or n > total:
        raise ValueError(f"invalid start_window={n} for total={total}")
    started = time.perf_counter()
    cpu_wait_seconds = 0.0
    cpu_work_seconds = float(resume_accum.get("cpu_work_seconds", 0.0))
    prior_elapsed = float(resume_accum.get("elapsed_seconds", 0.0))
    tiles_total = int(resume_accum.get("tiles_total", 0))

    geometry_caches = geometry_caches if geometry_caches is not None else {}
    grid_cache = geometry_caches.setdefault("grid", {})
    context_index_cache = geometry_caches.setdefault("context_index", {})

    prepared_iter = iter(_iter_prepared_rows(
        root,
        idx,
        shuffle=train,
        seed=seed,
        coarse=coarse,
        high=high,
        native_shape=native_shape,
        native_origin=native_origin,
        native_step=native_step,
        workers=prep_workers,
        prefetch=prefetch,
        skip_windows=n,
    ))

    while n < total:
        wait_started = time.perf_counter()
        try:
            prepared = next(prepared_iter)
        except StopIteration:
            break
        cpu_wait_seconds += time.perf_counter() - wait_started
        cpu_work_seconds += float(prepared["cpu_seconds"])
        (
            sem,
            obs,
            free,
            obs_np,
            coarse_idx,
            coarse_y,
            high_idx,
            high_y,
        ) = _prepared_to_device(prepared, device)

        if train:
            optimizer.zero_grad(set_to_none=True)
        with _autocast(device, amp):
            scene = model.encode_history(sem, obs, free)
            clogits = model.static.forward_coarse(scene)
            lc = _point_ce(clogits, coarse_idx, coarse_y, weights)
            lt, c, nt = _tile_loss(
                model,
                scene,
                high_idx,
                high_y,
                obs_np,
                high,
                coarse,
                tile_size,
                weights,
                tile_batch_size=tile_batch_size,
                grid_cache=grid_cache,
                context_index_cache=context_index_cache,
            )
            loss = lc + lt
        _assert_finite_async(loss)
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.encoder.parameters()) + list(model.static.parameters()),
                5.0,
                error_if_nonfinite=True,
                foreach=True,
            )
            optimizer.step()
        sums_gpu["loss"].add_(loss.detach().float())
        sums_gpu["coarse"].add_(lc.detach().float())
        sums_gpu["tile"].add_(lt.detach().float())
        conf_gpu.add_(c)
        tiles_total += int(nt)
        n += 1
        should_report = (
            n == int(start_window) + 1
            or n % max(int(progress_every), 1) == 0
            or n == total
        )
        should_checkpoint = (
            train
            and checkpoint_callback is not None
            and int(checkpoint_every) > 0
            and n < total
            and n % int(checkpoint_every) == 0
        )
        if should_report or should_checkpoint:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            now = time.perf_counter()
            segment_n = max(n - int(start_window), 1)
            phase = "train" if train else "val"
            loss_mean = float(sums_gpu["loss"].item() / max(n, 1))
            alloc_gib = (
                torch.cuda.memory_allocated(device) / 1024**3
                if device.type == "cuda" else 0.0
            )
            if should_report:
                print(
                    f"v20_static_{phase} {n}/{total} "
                    f"rate={segment_n/max(now-started,1e-9):.3f} win/s "
                    f"cpu_wait={cpu_wait_seconds/segment_n:.3f}s/win "
                    f"cpu_work={cpu_work_seconds/max(n,1):.3f}s/win "
                    f"tiles={tiles_total/max(n,1):.1f}/win "
                    f"gpu_mem={alloc_gib:.2f}GiB "
                    f"loss={loss_mean:.4f}",
                    flush=True,
                )
            if should_checkpoint:
                checkpoint_callback(
                    n,
                    {
                        "sums": {
                            k: float(v.item()) for k, v in sums_gpu.items()
                        },
                        "confusion": conf_gpu.detach().cpu().numpy().tolist(),
                        "tiles_total": int(tiles_total),
                        "cpu_work_seconds": float(cpu_work_seconds),
                        "elapsed_seconds": float(
                            prior_elapsed + (now - started)
                        ),
                    },
                )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    conf = conf_gpu.cpu().numpy()
    miou, per = _miou(conf)
    sums = {k: float(v.item() / max(n, 1)) for k, v in sums_gpu.items()}
    return {
        **sums,
        "static_supervised_semantic_miou": miou,
        "per_class_iou": per,
        "windows": n,
        "mean_cpu_work_seconds_per_window": float(
            cpu_work_seconds / max(n, 1)
        ),
        "mean_cpu_wait_seconds_per_resumed_window": float(
            cpu_wait_seconds / max(n - int(start_window), 1)
        ),
        "mean_tiles_per_window": float(tiles_total / max(n, 1)),
        "epoch_elapsed_seconds": float(
            prior_elapsed + time.perf_counter() - started
        ),
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
    p.add_argument(
        "--tile-batch-size",
        type=int,
        default=8,
        help="Number of equal-shape fine tiles refined in one GPU batch.",
    )
    p.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Print Stage-2 progress every N windows.",
    )
    p.add_argument(
        "--prep-workers",
        type=int,
        default=4,
        help="CPU workers for ordered history/target preparation.",
    )
    p.add_argument(
        "--prefetch",
        type=int,
        default=32,
        help="Maximum prepared/pending windows kept ahead of the GPU.",
    )
    p.add_argument(
        "--checkpoint-every-windows",
        type=int,
        default=500,
        help="Atomically refresh resume_latest.pt every N train windows; 0 disables.",
    )
    p.add_argument(
        "--resume",
        default="",
        help="Resume exactly from a V20 Static resume/epoch checkpoint.",
    )
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

    resume_path = Path(a.resume).resolve() if str(a.resume).strip() else None
    resume_ck = None
    if resume_path is not None:
        print(f"startup: loading resume checkpoint {resume_path}", flush=True)
        if not resume_path.is_file():
            raise FileNotFoundError(resume_path)
        resume_ck = torch.load(
            resume_path, map_location="cpu", weights_only=False
        )
        if resume_ck.get("protocol") != "p0_f9_v20_checkpoint_v1":
            raise RuntimeError("resume checkpoint protocol mismatch")
        if resume_ck.get("stage") != "static":
            raise RuntimeError("resume checkpoint is not V20 Static")
        saved_weights = (
            dict(resume_ck.get("extra") or {}).get("class_weights")
        )
        if saved_weights is None or len(saved_weights) != 18:
            raise RuntimeError(
                "resume checkpoint lacks valid saved class_weights"
            )
        weights_cpu = torch.as_tensor(
            saved_weights, dtype=torch.float32
        )
        print("startup: reused class_weights from resume checkpoint", flush=True)
    else:
        print(
            "startup: computing class_weights from train cache "
            "(one-time full train scan)",
            flush=True,
        )
        _, weights_cpu = _class_weights(tr_root, tr_idx)
        print("startup: class_weights ready", flush=True)

    print("startup: loading frozen V18 checkpoint", flush=True)
    v18_obj = torch.load(a.v18_checkpoint, map_location="cpu", weights_only=False)
    v18_model_cfg = dict(v18_obj.get("model_config") or {})
    if "d_model" not in v18_model_cfg:
        raise RuntimeError("Clean-E14 checkpoint lacks model_config.d_model")
    cfg = V20SceneConfig(source_dim=int(v18_model_cfg["d_model"]))
    model = V20HistoryWorldModel(cfg)
    # Stage 2 optimizes only encoder + Static.
    for p0 in model.dormant.parameters(): p0.requires_grad = False
    for p0 in model.birth.parameters(): p0.requires_grad = False
    device = torch.device(
        a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    model.to(device)
    weights = weights_cpu.to(device)
    optimizer = torch.optim.AdamW(
        [p0 for p0 in model.parameters() if p0.requires_grad],
        lr=float(a.lr),
        weight_decay=float(a.weight_decay),
        foreach=(device.type == "cuda"),
    )
    amp = device.type == "cuda" and not bool(a.no_amp)
    tile_size = tuple(int(x) for x in a.tile_size.split(","))
    native = dict(tr_idx["native_grid"])
    if dict(va_idx["native_grid"]) != native:
        raise RuntimeError("train/val native grid mismatch")
    native_shape = tuple(int(x) for x in native["shape_xyz"])
    native_origin = tuple(float(x) for x in native["origin_xyz_m"])
    native_step = tuple(float(x) for x in native["voxel_size_xyz_m"])
    print(
        f"startup: model ready on {device}; "
        f"prep_workers={int(a.prep_workers)} prefetch={int(a.prefetch)} "
        f"tile_batch={int(a.tile_batch_size)}",
        flush=True,
    )

    out = Path(a.output_dir)
    if resume_path is None:
        if out.exists() and any(out.iterdir()):
            raise FileExistsError(f"refusing non-empty output dir: {out}")
        out.mkdir(parents=True, exist_ok=True)
        history = []
        start_epoch = 1
        resume_window = 0
        resume_accum = None
    else:
        out.mkdir(parents=True, exist_ok=True)
        if resume_ck is None:
            raise RuntimeError("internal error: resume checkpoint not preloaded")
        ck = resume_ck
        if str(Path(ck.get("v18_checkpoint", "")).resolve()) != str(
            Path(a.v18_checkpoint).resolve()
        ):
            raise RuntimeError("resume V18 checkpoint mismatch")
        extra = dict(ck.get("extra") or {})
        contract = dict(extra.get("training_contract") or {})
        expected = {
            "train_cache": str(Path(a.train_cache).resolve()),
            "val_cache": str(Path(a.val_cache).resolve()),
            "seed": int(a.seed),
            "tile_size_xyz": list(tile_size),
        }
        for key, value in expected.items():
            if contract.get(key) != value:
                raise RuntimeError(
                    f"resume contract mismatch for {key}: "
                    f"{contract.get(key)!r} != {value!r}"
                )
        model.load_state_dict(ck["model"], strict=True)
        if ck.get("optimizer_state_dict") is None:
            raise RuntimeError(
                "resume checkpoint lacks optimizer_state_dict; "
                "use a checkpoint produced by the new Static trainer"
            )
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        _restore_rng_state(ck.get("rng_state"))
        progress = dict(ck.get("training_progress") or {})
        history = list(progress.get("history") or [])
        completed_epoch = bool(progress.get("epoch_complete", False))
        saved_epoch = int(progress.get("epoch", 0))
        if completed_epoch:
            start_epoch = saved_epoch + 1
            resume_window = 0
            resume_accum = None
        else:
            start_epoch = saved_epoch
            resume_window = int(progress.get("completed_windows", 0))
            resume_accum = dict(progress.get("partial_epoch_state") or {})
        print(
            f"resumed V20 Static from {resume_path}: "
            f"epoch={start_epoch} window={resume_window}",
            flush=True,
        )

    if start_epoch > int(a.epochs):
        raise RuntimeError(
            f"resume already completed epoch {start_epoch-1}; "
            f"target --epochs={a.epochs}"
        )

    geometry_caches = {}
    total_train_windows = int(tr_idx["num_windows"])

    def make_payload(epoch, *, epoch_complete, completed_windows, partial_state, hist):
        payload = checkpoint_payload(
            model,
            stage="static",
            v18_checkpoint=str(Path(a.v18_checkpoint).resolve()),
            thresholds={},
            extra={
                "train_protocol": PROTOCOL,
                "epoch": int(epoch),
                "highres_lattice": tr_idx["highres_lattice"],
                "coarse_lattice": tr_idx["coarse_lattice"],
                "tile_size_xyz": list(tile_size),
                "tile_batch_size": int(a.tile_batch_size),
                "class_weights": weights_cpu.tolist(),
                "training_contract": {
                    "train_cache": str(Path(a.train_cache).resolve()),
                    "val_cache": str(Path(a.val_cache).resolve()),
                    "seed": int(a.seed),
                    "tile_size_xyz": list(tile_size),
                    "lr": float(a.lr),
                    "weight_decay": float(a.weight_decay),
                },
                "selection": (
                    "No automatic best.pt. Select epoch only from formal composed "
                    "semantic mIoU on the scene-disjoint development set."
                ),
                "history": list(hist),
            },
        )
        payload["optimizer_state_dict"] = optimizer.state_dict()
        payload["rng_state"] = _capture_rng_state()
        payload["training_progress"] = {
            "epoch": int(epoch),
            "epoch_complete": bool(epoch_complete),
            "completed_windows": int(completed_windows),
            "total_windows": int(total_train_windows),
            "partial_epoch_state": partial_state,
            "history": list(hist),
        }
        return payload

    for epoch in range(int(start_epoch), int(a.epochs) + 1):
        this_start = int(resume_window) if epoch == int(start_epoch) else 0
        this_accum = resume_accum if epoch == int(start_epoch) else None

        def save_partial(completed_windows, partial_state):
            payload = make_payload(
                epoch,
                epoch_complete=False,
                completed_windows=completed_windows,
                partial_state=partial_state,
                hist=history,
            )
            _atomic_torch_save(payload, out / "resume_latest.pt")

        tr = _epoch(
            model, tr_root, tr_idx, high, coarse, native_shape, native_origin, native_step,
            device,
            weights,
            optimizer=optimizer,
            tile_size=tile_size,
            tile_batch_size=int(a.tile_batch_size),
            progress_every=int(a.progress_every),
            prep_workers=int(a.prep_workers),
            prefetch=int(a.prefetch),
            seed=int(a.seed) + epoch,
            amp=amp,
            start_window=this_start,
            resume_accum=this_accum,
            checkpoint_every=int(a.checkpoint_every_windows),
            checkpoint_callback=save_partial,
            geometry_caches=geometry_caches,
        )
        with torch.inference_mode():
            va = _epoch(
                model, va_root, va_idx, high, coarse, native_shape, native_origin, native_step,
                device,
                weights,
                optimizer=None,
                tile_size=tile_size,
                tile_batch_size=int(a.tile_batch_size),
                progress_every=int(a.progress_every),
                prep_workers=int(a.prep_workers),
                prefetch=int(a.prefetch),
                seed=int(a.seed),
                amp=amp,
                geometry_caches=geometry_caches,
            )
        row = {"epoch": epoch, "train": tr, "val": va}
        history.append(row)
        print(json.dumps(row))
        payload = make_payload(
            epoch,
            epoch_complete=True,
            completed_windows=total_train_windows,
            partial_state=None,
            hist=history,
        )
        _atomic_torch_save(payload, out / f"epoch_{epoch:04d}.pt")
        _atomic_torch_save(payload, out / "latest.pt")
        _atomic_torch_save(payload, out / "resume_latest.pt")
        resume_window = 0
        resume_accum = None
    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
