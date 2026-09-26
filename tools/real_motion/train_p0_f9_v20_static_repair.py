#!/usr/bin/env python3
"""Train V20 Stage-2 Static Repair v2 with deployment-aligned supervision.

Key contract:
* historical evidence comes from the frozen Stage1 aligned-history cache;
* frozen V18 defines where Static can act (V18 predicts free);
* supervision covers the same full future occupancy grid as formal evaluation;
* dynamic GT is a no-add/free target for Static;
* the exact runtime M_query is supplied as the query input channel;
* six future horizons contribute independently, including conflicting labels
  that map to the same canonical cell;
* loss is ordinary per-future-voxel CE with no class/tile reweighting.
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import hashlib
import json
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
    FREE_LABEL,
    canonical_tile_grid_sample_coordinates,
    grid_centers_xyz,
)
from real_motion.v20_scene_model import V20HistoryWorldModel, V20SceneConfig
from real_motion.v20_stage1_codec import unpack_bool, unpack_history_semantic
from real_motion.v20_static_repair import (
    STATIC_ALLOWED_IDS,
    SUPPORT_CACHE_PROTOCOL,
    TRAIN_PROTOCOL,
    full_grid_metrics_from_confusion,
    repair_diagnostics_from_confusion,
    repair_target_from_gt,
    unpack_v18_free_support,
    unpack_v18_prediction,
)
from real_motion.v20_training import checkpoint_payload
from tools.real_motion.build_p0_f9_v20_history_cache import (
    PROTOCOL as STAGE1_PROTOCOL,
)
from tools.real_motion.eval_p0_f9_v19_factorized_static_new_fov import CachedSource


def _load_index(root, expected):
    root = Path(root)
    idx = json.loads((root / "index.json").read_text(encoding="utf-8"))
    if idx.get("protocol") != expected:
        raise RuntimeError(
            f"{root}: protocol={idx.get('protocol')!r}, expected {expected!r}"
        )
    return root, idx


def _lattice(d):
    return CanonicalLattice(
        tuple(float(x) for x in d["origin_xyz_m"]),
        tuple(float(x) for x in d["voxel_size_xyz_m"]),
        tuple(int(x) for x in d["shape_xyz"]),
    )


def _stage1_rows(root, shard_name):
    obj = torch.load(
        Path(root) / str(shard_name),
        map_location="cpu",
        weights_only=False,
    )
    if obj.get("protocol") != STAGE1_PROTOCOL:
        raise RuntimeError(f"bad Stage1 shard: {shard_name}")
    return list(obj["rows"])


def _repair_rows(root, shard_name):
    obj = torch.load(
        Path(root) / str(shard_name),
        map_location="cpu",
        weights_only=False,
    )
    if obj.get("protocol") != SUPPORT_CACHE_PROTOCOL:
        raise RuntimeError(f"bad repair-support shard: {shard_name}")
    return list(obj["rows"])


def _row_identity(row):
    return (str(row["scene_name"]), str(row["t0_token"]))


def _iter_paired_rows_canonical(stage_root, repair_root, repair_idx):
    """Yield Stage1/support pairs in the frozen cache population order."""
    for rmeta in repair_idx["shards"]:
        rrows = _repair_rows(repair_root, rmeta["file"])
        srows = _stage1_rows(stage_root, rmeta["source_stage1_shard"])
        if len(rrows) > len(srows):
            raise RuntimeError(
                f"repair shard longer than Stage1 source shard: {rmeta['file']}"
            )
        srows = srows[:len(rrows)]
        for sr, rr in zip(srows, rrows):
            rkey = _row_identity(rr)
            skey = _row_identity(sr)
            if rkey != skey:
                raise RuntimeError(
                    f"Stage1/repair identity mismatch: {skey} != {rkey}"
                )
            yield sr, rr


def _fixed_identity_subset(stage_root, repair_root, repair_idx, count):
    """Freeze the first N identities before any epoch shuffling."""
    n = int(count)
    if n <= 0:
        return ()
    out = []
    for sr, rr in _iter_paired_rows_canonical(
        stage_root, repair_root, repair_idx
    ):
        out.append(_row_identity(rr))
        if len(out) == n:
            break
    if len(out) != n:
        raise RuntimeError(
            f"requested fixed subset of {n} windows but cache has {len(out)}"
        )
    if len(set(out)) != len(out):
        raise RuntimeError("fixed Static Repair subset contains duplicate identities")
    return tuple(out)


def _identity_fingerprint(identities):
    h = hashlib.sha256()
    for scene, token in identities:
        h.update(str(scene).encode("utf-8"))
        h.update(b"\0")
        h.update(str(token).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _iter_paired_rows(
    stage_root,
    repair_root,
    repair_idx,
    *,
    shuffle,
    seed,
    max_windows=0,
    skip_windows=0,
    fixed_identities=None,
):
    rng = random.Random(int(seed))
    max_windows = int(max_windows)
    skip_windows = int(skip_windows)

    if fixed_identities is not None:
        fixed = tuple(tuple(x) for x in fixed_identities)
        wanted = set(fixed)
        if len(wanted) != len(fixed):
            raise RuntimeError("fixed identity subset contains duplicates")
        found = {}
        for sr, rr in _iter_paired_rows_canonical(
            stage_root, repair_root, repair_idx
        ):
            key = _row_identity(rr)
            if key in wanted:
                found[key] = (sr, rr)
                if len(found) == len(wanted):
                    break
        missing = [key for key in fixed if key not in found]
        if missing:
            raise RuntimeError(
                f"fixed Static Repair subset misses identities: {missing[:5]}"
            )
        pairs = [found[key] for key in fixed]
        if shuffle:
            rng.shuffle(pairs)
        if skip_windows > len(pairs):
            raise RuntimeError("resume skip exceeds fixed Static Repair subset")
        pairs = pairs[skip_windows:]
        if max_windows > 0:
            pairs = pairs[:max_windows]
        yield from pairs
        return

    order = list(range(len(repair_idx["shards"])))
    if shuffle:
        rng.shuffle(order)

    yielded = 0
    skipped = 0
    for rsi in order:
        rmeta = repair_idx["shards"][rsi]
        rrows = _repair_rows(repair_root, rmeta["file"])
        srows = _stage1_rows(stage_root, rmeta["source_stage1_shard"])
        if len(rrows) > len(srows):
            raise RuntimeError(
                f"repair shard longer than Stage1 source shard: {rmeta['file']}"
            )
        srows = srows[:len(rrows)]

        row_order = list(range(len(rrows)))
        if shuffle:
            rng.shuffle(row_order)
        for j in row_order:
            rr = rrows[j]
            sr = srows[j]
            rkey = _row_identity(rr)
            skey = _row_identity(sr)
            if rkey != skey:
                raise RuntimeError(
                    f"Stage1/repair identity mismatch: {skey} != {rkey}"
                )
            if skipped < skip_windows:
                skipped += 1
                continue
            if max_windows > 0 and yielded >= max_windows:
                return
            yield sr, rr
            yielded += 1

def _decode_history(row):
    shape = (6,) + tuple(int(x) for x in row["coarse_shape_xyz"])
    obs = unpack_bool(row["history_observed_bits"], shape)
    free = unpack_bool(row["history_observed_free_bits"], shape)
    sem = unpack_history_semantic(row, obs, free)
    return sem, obs, free


def _prepare_pair_cpu(
    srow,
    rrow,
    source,
    native_shape,
    free_label,
    *,
    need_metrics,
):
    started = time.perf_counter()
    sem, obs, obsfree = _decode_history(srow)
    support = unpack_v18_free_support(
        rrow["v18_free_bits"], native_shape
    )
    gt = np.stack([
        source.load_semantics(str(rrow["scene_name"]), str(tok))
        for tok in rrow["future_tokens"]
    ]).astype(np.uint8, copy=False)
    if gt.shape != (6,) + tuple(native_shape):
        raise RuntimeError(
            f"future GT shape mismatch: {gt.shape}"
        )
    target = repair_target_from_gt(gt, free_label=int(free_label))
    # Stage1 already stores the authoritative future->t0 transforms used
    # by V20. Reusing them removes seven pose lookups and duplicate SE(3)
    # algebra per training window.
    future_rel = np.asarray(
        srow["future_ego_to_t0"], dtype=np.float64
    )
    if future_rel.shape != (6, 4, 4):
        raise RuntimeError("Stage1 future_ego_to_t0 shape mismatch")

    base_full_conf = None
    base_support_conf = None
    if need_metrics:
        # Evaluation-only work. Dense frozen-V18 reconstruction and full-grid
        # confusion do not contribute to the training CE gradient.
        base_pred = unpack_v18_prediction(
            rrow, native_shape, free_label=int(free_label)
        )
        if not np.array_equal(
            base_pred == int(free_label), support
        ):
            raise RuntimeError("reconstructed V18 prediction/support mismatch")
        base_full_conf = np.zeros((6, 18, 18), dtype=np.int64)
        base_support_conf = np.zeros((6, 18, 18), dtype=np.int64)
        for hi in range(6):
            gh = gt[hi].reshape(-1).astype(np.int64, copy=False)
            bh = base_pred[hi].reshape(-1).astype(np.int64, copy=False)
            sh = support[hi].reshape(-1)
            base_full_conf[hi] = np.bincount(
                gh * 18 + bh, minlength=18 * 18
            ).reshape(18, 18)
            gs = gh[sh]
            base_support_conf[hi] = np.bincount(
                gs * 18 + int(free_label), minlength=18 * 18
            ).reshape(18, 18)

    cached_counts = np.asarray(
        rrow["v18_free_count_by_horizon"], dtype=np.int64
    )
    got_counts = support.reshape(6, -1).sum(axis=1).astype(np.int64)
    if not np.array_equal(cached_counts, got_counts):
        raise RuntimeError("V18-free packed support count mismatch")
    out = {
        "sem": sem,
        "obs": obs,
        "obsfree": obsfree,
        "support": support,
        "target": target,
        "future_rel": future_rel,
        "cpu_seconds": float(time.perf_counter() - started),
    }
    if need_metrics:
        out["gt"] = gt
        out["base_full_conf"] = base_full_conf
        out["base_support_conf"] = base_support_conf
    return out


def _iter_prepared(
    stage_root,
    repair_root,
    repair_idx,
    source,
    *,
    native_shape,
    free_label,
    shuffle,
    seed,
    max_windows,
    skip_windows,
    workers,
    prefetch,
    fixed_identities=None,
    need_metrics=True,
):
    raw = iter(_iter_paired_rows(
        stage_root,
        repair_root,
        repair_idx,
        shuffle=shuffle,
        seed=seed,
        max_windows=max_windows,
        skip_windows=skip_windows,
        fixed_identities=fixed_identities,
    ))

    def prepare(pair):
        return _prepare_pair_cpu(
            pair[0],
            pair[1],
            source,
            native_shape,
            free_label,
            need_metrics=bool(need_metrics),
        )

    nw = max(int(workers), 0)
    if nw == 0:
        for pair in raw:
            yield prepare(pair)
        return

    pending = deque()
    capacity = max(int(prefetch), nw)
    with ThreadPoolExecutor(
        max_workers=nw, thread_name_prefix="v20-static-repair"
    ) as pool:
        exhausted = False
        for _ in range(capacity):
            try:
                pending.append(pool.submit(prepare, next(raw)))
            except StopIteration:
                exhausted = True
                break
        while pending:
            fut = pending.popleft()
            item = fut.result()
            if not exhausted:
                try:
                    pending.append(pool.submit(prepare, next(raw)))
                except StopIteration:
                    exhausted = True
            yield item


class _Geometry:
    def __init__(self, high, coarse, native_shape, native_origin, native_step, tile_size, device):
        self.high = high
        self.coarse = coarse
        self.high_shape = tuple(int(x) for x in high.shape_xyz)
        self.coarse_shape = tuple(int(x) for x in coarse.shape_xyz)
        self.native_shape = tuple(int(x) for x in native_shape)
        self.tile = tuple(int(x) for x in tile_size)
        self.device = device

        xyz = grid_centers_xyz(
            self.native_shape, native_origin, native_step
        ).reshape(-1, 3)
        self.native_xyz = torch.from_numpy(
            np.array(xyz, dtype=np.float64, copy=True)
        ).to(device)
        self.high_origin = torch.as_tensor(
            high.origin_xyz_m, dtype=torch.float64, device=device
        )
        self.high_step = torch.as_tensor(
            high.voxel_size_xyz_m, dtype=torch.float64, device=device
        )
        self.high_shape_t = torch.as_tensor(
            self.high_shape, dtype=torch.long, device=device
        )
        hi = np.asarray(high.voxel_size_xyz_m, dtype=np.float64)
        co = np.asarray(coarse.voxel_size_xyz_m, dtype=np.float64)
        factor = np.rint(co / hi).astype(np.int64)
        if not np.allclose(factor * hi, co):
            raise RuntimeError("coarse/high lattice ratio is not integral")
        self.factor = tuple(int(x) for x in factor)
        self.grid_cache = {}
        self.coarse_index_cache = {}
        self.high_numel = int(np.prod(self.high_shape))
        self.allowed_ids = torch.as_tensor(
            STATIC_ALLOWED_IDS, dtype=torch.long, device=device
        )
        self.global_to_local = torch.full(
            (18,), -1, dtype=torch.long, device=device
        )
        self.global_to_local[self.allowed_ids] = torch.arange(
            len(STATIC_ALLOWED_IDS), device=device
        )

    def future_linear_and_query(self, future_rel):
        T = torch.as_tensor(
            future_rel, dtype=torch.float64, device=self.device
        )
        if T.shape != (6, 4, 4):
            raise ValueError("future_rel must be [6,4,4]")
        canon = torch.einsum(
            "fij,nj->fni", T[:, :3, :3], self.native_xyz
        ) + T[:, None, :3, 3]
        idx = torch.floor(
            (canon - self.high_origin[None, None])
            / self.high_step[None, None]
        ).to(torch.long)
        valid = ((idx >= 0) & (idx < self.high_shape_t)).all(dim=-1)
        ok = valid.all()
        if self.device.type == "cuda" and hasattr(torch, "_assert_async"):
            torch._assert_async(
                ok, "V20 repair geometry violated frozen Omega-max"
            )
        elif not bool(ok.item()):
            raise RuntimeError("repair geometry escaped frozen Omega-max")

        Y, Z = self.high_shape[1], self.high_shape[2]
        linear = (
            idx[..., 0] * (Y * Z)
            + idx[..., 1] * Z
            + idx[..., 2]
        )
        q = torch.zeros(
            int(np.prod(self.high_shape)),
            dtype=torch.bool,
            device=self.device,
        )
        q[linear.reshape(-1)] = True
        q = q.reshape(self.high_shape)
        return linear, q

    def active_tiles(self, q):
        tx, ty, tz = self.tile
        pooled = F.max_pool3d(
            q[None, None].to(torch.float32),
            kernel_size=(tx, ty, tz),
            stride=(tx, ty, tz),
            ceil_mode=True,
        )
        return torch.nonzero(
            pooled[0, 0] > 0, as_tuple=False
        ).cpu().numpy()

    def tile_grid(self, start, shape, dtype):
        key = (
            tuple(int(x) for x in start),
            tuple(int(x) for x in shape),
            str(dtype),
        )
        got = self.grid_cache.get(key)
        if got is not None:
            return got
        grid = canonical_tile_grid_sample_coordinates(
            self.high,
            self.coarse,
            start,
            shape,
            device=self.device,
            dtype=dtype,
        )
        self.grid_cache[key] = grid
        return grid

    def tile_coarse_linear(self, start, shape):
        key = (
            tuple(int(x) for x in start),
            tuple(int(x) for x in shape),
        )
        got = self.coarse_index_cache.get(key)
        if got is not None:
            return got
        start = tuple(int(x) for x in start)
        shape = tuple(int(x) for x in shape)
        fx, fy, fz = self.factor
        x = torch.arange(
            start[0], start[0] + shape[0],
            dtype=torch.long, device=self.device,
        ) // int(fx)
        y = torch.arange(
            start[1], start[1] + shape[1],
            dtype=torch.long, device=self.device,
        ) // int(fy)
        z = torch.arange(
            start[2], start[2] + shape[2],
            dtype=torch.long, device=self.device,
        ) // int(fz)
        Y, Z = self.coarse_shape[1], self.coarse_shape[2]
        linear = (
            x[:, None, None] * (Y * Z)
            + y[None, :, None] * Z
            + z[None, None, :]
        ).reshape(-1)
        self.coarse_index_cache[key] = linear
        return linear


def _high_context(obs, geom, device):
    obs_t = torch.from_numpy(np.asarray(obs, dtype=bool)).to(
        device, non_blocking=True
    )
    seen = obs_t.any(dim=0)
    t0 = obs_t[-1]
    fx, fy, fz = geom.factor
    seen_h = seen.repeat_interleave(fx, 0).repeat_interleave(
        fy, 1
    ).repeat_interleave(fz, 2)
    t0_h = t0.repeat_interleave(fx, 0).repeat_interleave(
        fy, 1
    ).repeat_interleave(fz, 2)
    X, Y, Z = geom.high_shape
    seen_h = seen_h[:X, :Y, :Z]
    t0_h = t0_h[:X, :Y, :Z]
    return seen_h, seen_h & ~t0_h


def _decode_query_logits(
    model,
    scene,
    q,
    seen_h,
    missing_h,
    geom,
    *,
    tile_batch_size,
):
    allowed = torch.as_tensor(
        STATIC_ALLOWED_IDS, dtype=torch.long, device=scene.device
    )
    world = torch.empty(
        (len(STATIC_ALLOWED_IDS),) + geom.high_shape,
        dtype=scene.dtype,
        device=scene.device,
    )

    active = geom.active_tiles(q)
    buckets = {}
    high_shape = np.asarray(geom.high_shape, dtype=np.int64)
    tile = np.asarray(geom.tile, dtype=np.int64)
    for tc in active:
        start = tc.astype(np.int64) * tile
        stop = np.minimum(start + tile, high_shape)
        shape = tuple((stop - start).tolist())
        buckets.setdefault(shape, []).append(tuple(start.tolist()))

    bsz = max(int(tile_batch_size), 1)
    ntiles = 0
    for tshape, starts in buckets.items():
        for bi in range(0, len(starts), bsz):
            chunk = starts[bi:bi + bsz]
            grids = []
            qrows = []
            srows = []
            mrows = []
            stops = []
            for start_t in chunk:
                start = np.asarray(start_t, dtype=np.int64)
                stop = np.minimum(start + tile, high_shape)
                stops.append(stop)
                grids.append(
                    geom.tile_grid(start_t, tshape, scene.dtype)
                )
                sl = (
                    slice(start[0], stop[0]),
                    slice(start[1], stop[1]),
                    slice(start[2], stop[2]),
                )
                qrows.append(q[sl])
                srows.append(seen_h[sl])
                mrows.append(missing_h[sl])

            logits = model.static.refine_tiles(
                scene,
                sample_grid=torch.cat(grids, dim=0),
                query_mask=torch.stack(qrows, dim=0),
                seen_mask=torch.stack(srows, dim=0),
                t0_missing_mask=torch.stack(mrows, dim=0),
            ).index_select(1, allowed)

            for b, (start_t, stop) in enumerate(zip(chunk, stops)):
                start = np.asarray(start_t, dtype=np.int64)
                world[
                    :,
                    start[0]:stop[0],
                    start[1]:stop[1],
                    start[2]:stop[2],
                ] = logits[b]
            ntiles += len(chunk)
    return world, int(ntiles)


def _decode_query_logits_sparse(
    model,
    scene,
    q,
    obs,
    geom,
    *,
    tile_batch_size,
):
    """Decode only canonical query cells, preserving exact tiled semantics.

    The legacy path materialized an [C,X,Y,Z] world over the entire Ωmax
    lattice even though loss only reads M_query. This path keeps the exact same
    tile refinement calls but stores logits only for query cells.
    """
    q_flat = q.reshape(-1)
    query_linear = torch.nonzero(q_flat, as_tuple=False).reshape(-1)
    qcount = int(query_linear.numel())
    if qcount <= 0:
        raise RuntimeError("Static Repair query union is empty")

    row_map = torch.full(
        (geom.high_numel,),
        -1,
        dtype=torch.int32,
        device=scene.device,
    )
    row_map[query_linear] = torch.arange(
        qcount, dtype=torch.int32, device=scene.device
    )
    query_logits = torch.empty(
        (qcount, len(STATIC_ALLOWED_IDS)),
        dtype=scene.dtype,
        device=scene.device,
    )
    filled = torch.zeros(qcount, dtype=torch.bool, device=scene.device)

    obs_t = torch.from_numpy(np.asarray(obs, dtype=bool)).to(
        scene.device, non_blocking=True
    )
    seen_flat = obs_t.any(dim=0).reshape(-1)
    t0_flat = obs_t[-1].reshape(-1)

    active = geom.active_tiles(q)
    buckets = {}
    high_shape = np.asarray(geom.high_shape, dtype=np.int64)
    tile = np.asarray(geom.tile, dtype=np.int64)
    for tc in active:
        start = tc.astype(np.int64) * tile
        stop = np.minimum(start + tile, high_shape)
        tshape = tuple((stop - start).tolist())
        buckets.setdefault(tshape, []).append(tuple(start.tolist()))

    row_map_3d = row_map.reshape(geom.high_shape)
    bsz = max(int(tile_batch_size), 1)
    ntiles = 0
    for tshape, starts in buckets.items():
        for bi in range(0, len(starts), bsz):
            chunk = starts[bi:bi + bsz]
            grids, qrows, srows, mrows, rrows = [], [], [], [], []
            for start_t in chunk:
                start = np.asarray(start_t, dtype=np.int64)
                stop = np.minimum(start + tile, high_shape)
                sl = (
                    slice(start[0], stop[0]),
                    slice(start[1], stop[1]),
                    slice(start[2], stop[2]),
                )
                grids.append(
                    geom.tile_grid(start_t, tshape, scene.dtype)
                )
                qtile = q[sl]
                qrows.append(qtile)
                rrows.append(row_map_3d[sl])

                cmap = geom.tile_coarse_linear(start_t, tshape)
                seen = seen_flat[cmap].reshape(tshape)
                t0_seen = t0_flat[cmap].reshape(tshape)
                srows.append(seen)
                mrows.append(seen & ~t0_seen)

            qstack = torch.stack(qrows, dim=0)
            logits = model.static.refine_tiles(
                scene,
                sample_grid=torch.cat(grids, dim=0),
                query_mask=qstack,
                seen_mask=torch.stack(srows, dim=0),
                t0_missing_mask=torch.stack(mrows, dim=0),
            ).index_select(1, geom.allowed_ids)

            # [B,C,X,Y,Z] -> [B,X,Y,Z,C], then retain only M_query cells.
            selected_logits = logits.permute(0, 2, 3, 4, 1)[qstack]
            selected_rows = torch.stack(rrows, dim=0)[qstack].long()
            # Under BF16 autocast tile_refine may return BF16 while scene (and
            # therefore the sparse destination buffer) remains FP32. The
            # legacy dense slice assignment performed this cast implicitly;
            # index_copy_ requires exact dtype equality, so make the same
            # conversion explicit here.
            if selected_logits.dtype != query_logits.dtype:
                selected_logits = selected_logits.to(query_logits.dtype)
            query_logits.index_copy_(0, selected_rows, selected_logits)
            filled[selected_rows] = True
            ntiles += len(chunk)

    complete = filled.all()
    if scene.device.type == "cuda" and hasattr(torch, "_assert_async"):
        torch._assert_async(
            complete, "Static Repair sparse query decode missed query cells"
        )
    elif not bool(complete.item()):
        raise RuntimeError("Static Repair sparse query decode missed query cells")
    return query_logits, row_map, int(ntiles)


def _repair_sparse_loss_only(
    query_logits,
    query_row_map,
    linear,
    support,
    target,
    *,
    geom,
    device,
):
    """Exact Repair-v2 CE with repeated canonical rows aggregated first.

    The formal objective is still one equally weighted CE term for every
    future voxel/horizon in frozen-V18-free support. Multiple future voxels
    can map to the same canonical query row, so evaluating CE independently
    repeats the same logit vector millions of times. Count those repeated
    (query-row, class) pairs first, then evaluate the algebraically identical
    weighted negative log-likelihood once per canonical query row.
    """
    support_t = torch.from_numpy(
        np.asarray(support, dtype=bool)
    ).to(device, non_blocking=True)
    target_t = torch.from_numpy(
        np.asarray(target, dtype=np.uint8)
    ).to(device, non_blocking=True)

    nclass = int(query_logits.shape[1])
    qcount = int(query_logits.shape[0])
    pair_keys = []
    total_support = 0

    for hi in range(6):
        s = support_t[hi].reshape(-1)
        lin = linear[hi][s]
        qrow = query_row_map[lin].long()
        valid_rows = (qrow >= 0).all()
        if device.type == "cuda" and hasattr(torch, "_assert_async"):
            torch._assert_async(
                valid_rows,
                "Static Repair support escaped decoded query union",
            )
        elif not bool(valid_rows.item()):
            raise RuntimeError(
                "Static Repair support escaped decoded query union"
            )

        y = target_t[hi].reshape(-1)[s].long()
        yl = geom.global_to_local[y]
        valid_labels = (yl >= 0).all()
        if device.type == "cuda" and hasattr(torch, "_assert_async"):
            torch._assert_async(
                valid_labels,
                "dynamic label entered Static Repair target",
            )
        elif not bool(valid_labels.item()):
            raise RuntimeError("dynamic label entered Static Repair target")

        pair_keys.append(qrow * nclass + yl)
        total_support += int(qrow.numel())

    if total_support <= 0:
        return query_logits.sum() * 0.0

    keys = torch.cat(pair_keys, dim=0)
    # float32 integer counts are exact at the collision multiplicities present
    # here and avoid an 8-byte int64 count volume followed by another copy.
    count_dtype = (
        torch.float32
        if query_logits.dtype in {torch.float16, torch.bfloat16}
        else query_logits.dtype
    )
    counts_flat = torch.zeros(
        qcount * nclass,
        dtype=count_dtype,
        device=device,
    )
    counts_flat.scatter_add_(
        0,
        keys,
        torch.ones(keys.shape, dtype=count_dtype, device=device),
    )
    counts = counts_flat.view(qcount, nclass)

    logits = query_logits
    if logits.dtype != count_dtype:
        logits = logits.to(count_dtype)

    # Sum_{examples} CE(logits[row], y)
    # = Sum_row [N_row * logsumexp(logits_row)
    #            - Sum_class N_{row,class} * logits_{row,class}]
    row_count = counts.sum(dim=1)
    loss_sum = (
        row_count * torch.logsumexp(logits, dim=1)
        - (counts * logits).sum(dim=1)
    ).sum()
    return loss_sum / float(total_support)

def _repair_sparse_loss_and_confusion(
    query_logits,
    query_row_map,
    linear,
    support,
    target,
    *,
    gt,
    base_full_conf,
    base_support_conf,
    geom,
    device,
):
    """Exact Repair-v2 loss/metrics without dense Ωmax logits/full-grid bincounts."""
    support_t = torch.from_numpy(
        np.asarray(support, dtype=bool)
    ).to(device, non_blocking=True)
    # Keep labels compact during host->device transfer; cast only selected rows.
    target_t = torch.from_numpy(
        np.asarray(target, dtype=np.uint8)
    ).to(device, non_blocking=True)
    gt_t = torch.from_numpy(
        np.asarray(gt, dtype=np.uint8)
    ).to(device, non_blocking=True)

    base_conf = torch.as_tensor(
        np.asarray(base_full_conf, dtype=np.int64),
        dtype=torch.int64,
        device=device,
    )
    base_support = torch.as_tensor(
        np.asarray(base_support_conf, dtype=np.int64),
        dtype=torch.int64,
        device=device,
    )
    final_conf = base_conf.clone()
    conf = torch.zeros((18, 18), dtype=torch.int64, device=device)
    loss_sum = query_logits.sum() * 0.0

    for hi in range(6):
        s = support_t[hi].reshape(-1)
        lin = linear[hi][s]
        qrow = query_row_map[lin].long()
        valid_rows = (qrow >= 0).all()
        if device.type == "cuda" and hasattr(torch, "_assert_async"):
            torch._assert_async(
                valid_rows,
                "Static Repair support escaped decoded query union",
            )
        elif not bool(valid_rows.item()):
            raise RuntimeError(
                "Static Repair support escaped decoded query union"
            )

        rows = query_logits[qrow]
        y = target_t[hi].reshape(-1)[s].long()
        yl = geom.global_to_local[y]
        valid_labels = (yl >= 0).all()
        if device.type == "cuda" and hasattr(torch, "_assert_async"):
            torch._assert_async(
                valid_labels,
                "dynamic label entered Static Repair target",
            )
        elif not bool(valid_labels.item()):
            raise RuntimeError("dynamic label entered Static Repair target")

        loss_sum = loss_sum + F.cross_entropy(
            rows, yl, reduction="sum"
        )
        with torch.no_grad():
            pred = geom.allowed_ids[
                rows.detach().float().argmax(dim=1)
            ]
            conf += torch.bincount(
                y * 18 + pred,
                minlength=18 * 18,
            ).reshape(18, 18)

            # Exact formal final confusion = frozen V18 full confusion
            # - V18-free contribution + repaired contribution.
            gy = gt_t[hi].reshape(-1)[s].long()
            repaired = torch.bincount(
                gy * 18 + pred,
                minlength=18 * 18,
            ).reshape(18, 18)
            final_conf[hi] = (
                base_conf[hi] - base_support[hi] + repaired
            )

    denom = support_t.sum().clamp_min(1)
    loss = loss_sum / denom.to(loss_sum.dtype)
    return loss, conf, base_conf, final_conf


def _repair_loss_and_confusion(
    world_allowed,
    linear,
    support,
    target,
    *,
    gt=None,
    base_pred=None,
    device,
):
    support_t = torch.from_numpy(
        np.asarray(support, dtype=bool)
    ).to(device, non_blocking=True)
    target_t = torch.from_numpy(
        np.asarray(target, dtype=np.uint8)
    ).to(device, non_blocking=True).long()

    allowed = torch.as_tensor(
        STATIC_ALLOWED_IDS, dtype=torch.long, device=device
    )
    global_to_local = torch.full(
        (18,), -1, dtype=torch.long, device=device
    )
    global_to_local[allowed] = torch.arange(
        len(STATIC_ALLOWED_IDS), device=device
    )

    flat_world = world_allowed.reshape(
        len(STATIC_ALLOWED_IDS), -1
    )
    loss_sum = world_allowed.sum() * 0.0
    conf = torch.zeros((18, 18), dtype=torch.int64, device=device)
    base_conf = torch.zeros((6, 18, 18), dtype=torch.int64, device=device)
    final_conf = torch.zeros((6, 18, 18), dtype=torch.int64, device=device)
    gt_t = (
        torch.from_numpy(np.asarray(gt, dtype=np.uint8)).to(
            device, non_blocking=True
        ).long()
        if gt is not None else None
    )
    base_t = (
        torch.from_numpy(np.asarray(base_pred, dtype=np.uint8)).to(
            device, non_blocking=True
        ).long()
        if base_pred is not None else None
    )
    if (gt_t is None) != (base_t is None):
        raise ValueError("gt/base_pred must be supplied together")

    for hi in range(6):
        s = support_t[hi].reshape(-1)
        y = target_t[hi].reshape(-1)[s]
        lin = linear[hi][s]
        rows = flat_world[:, lin].transpose(0, 1)
        yl = global_to_local[y]
        if device.type == "cuda" and hasattr(torch, "_assert_async"):
            torch._assert_async(
                (yl >= 0).all(),
                "dynamic label entered Static Repair target",
            )
        elif not bool((yl >= 0).all().item()):
            raise RuntimeError("dynamic label entered Static Repair target")
        loss_sum = loss_sum + F.cross_entropy(
            rows, yl, reduction="sum"
        )
        with torch.no_grad():
            pred = allowed[rows.detach().float().argmax(dim=1)]
            conf += torch.bincount(
                y * 18 + pred,
                minlength=18 * 18,
            ).reshape(18, 18)
            if gt_t is not None:
                gt_h = gt_t[hi].reshape(-1)
                base_h = base_t[hi].reshape(-1)
                final_h = base_h.clone()
                final_h[s] = pred
                base_conf[hi] = torch.bincount(
                    gt_h * 18 + base_h,
                    minlength=18 * 18,
                ).reshape(18, 18)
                final_conf[hi] = torch.bincount(
                    gt_h * 18 + final_h,
                    minlength=18 * 18,
                ).reshape(18, 18)

    denom = support_t.sum().clamp_min(1)
    loss = loss_sum / denom.to(loss_sum.dtype)
    if gt_t is None:
        # Keep the focused contract-test/diagnostic API compact.
        return loss, conf
    return loss, conf, base_conf, final_conf


def _autocast(device, enabled):
    if enabled and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def _capture_rng():
    out = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        out["torch_cuda"] = torch.cuda.get_rng_state_all()
    return out


def _restore_rng(state):
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _atomic_save(obj, path):
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


class _CudaStageProfiler:
    """Low-overhead CUDA-event timing, flushed only at existing sync points."""

    STAGES = (
        "h2d",
        "encoder",
        "geometry",
        "tile_decode",
        "loss",
        "backward",
        "optim",
    )

    def __init__(self, device, *, enabled=False, warmup_windows=5):
        self.enabled = bool(enabled) and device.type == "cuda"
        self.warmup_windows = max(int(warmup_windows), 0)
        self.window_index = 0
        self.pending = []
        self.total_ms = {name: 0.0 for name in self.STAGES}
        self.count = {name: 0 for name in self.STAGES}
        self.last_interval_ms = {}

    def begin_window(self):
        active = self.enabled and self.window_index >= self.warmup_windows
        self.window_index += 1
        return {} if active else None

    def start(self, events, name):
        if events is None:
            return
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        events[name] = [ev, None]

    def stop(self, events, name):
        if events is None:
            return
        pair = events.get(name)
        if pair is None:
            raise RuntimeError(f"profiler stage {name!r} was not started")
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        pair[1] = ev

    def finish_window(self, events):
        if events:
            self.pending.append(events)

    def flush_after_sync(self):
        if not self.enabled:
            self.last_interval_ms = {}
            return {}
        interval_sum = {name: 0.0 for name in self.STAGES}
        interval_count = {name: 0 for name in self.STAGES}
        for events in self.pending:
            for name, pair in events.items():
                if pair[1] is None:
                    raise RuntimeError(
                        f"profiler stage {name!r} was not stopped"
                    )
                ms = float(pair[0].elapsed_time(pair[1]))
                self.total_ms[name] += ms
                self.count[name] += 1
                interval_sum[name] += ms
                interval_count[name] += 1
        self.pending.clear()
        self.last_interval_ms = {
            name: (
                float(interval_sum[name] / interval_count[name])
                if interval_count[name] > 0 else None
            )
            for name in self.STAGES
        }
        return self.summary()

    def interval_summary(self):
        return dict(self.last_interval_ms)

    def summary(self):
        return {
            name: (
                float(self.total_ms[name] / self.count[name])
                if self.count[name] > 0 else None
            )
            for name in self.STAGES
        }


def _epoch(
    model,
    stage_root,
    repair_root,
    repair_idx,
    source,
    geom,
    *,
    device,
    optimizer,
    tile_batch_size,
    prep_workers,
    prefetch,
    progress_every,
    seed,
    amp,
    max_windows=0,
    start_window=0,
    resume_state=None,
    checkpoint_every=0,
    checkpoint_callback=None,
    fixed_identities=None,
    profile_gpu_stages=False,
    profile_warmup_windows=5,
):
    train = optimizer is not None
    model.train(train)
    model.dormant.eval()
    model.birth.eval()
    model.static.coarse_head.eval()
    trainable_params = (
        [p for p in model.parameters() if p.requires_grad]
        if train else []
    )

    total = (
        min(int(repair_idx["num_windows"]), int(max_windows))
        if int(max_windows) > 0
        else int(repair_idx["num_windows"])
    )
    if start_window < 0 or start_window > total:
        raise ValueError("invalid resume window")

    saved = dict(resume_state or {})
    compute_metrics = not train
    if compute_metrics:
        conf0 = np.asarray(
            saved.get("confusion", np.zeros((18, 18), dtype=np.int64)),
            dtype=np.int64,
        )
        conf_gpu = torch.as_tensor(
            conf0, dtype=torch.int64, device=device
        ).clone()
        base_full_gpu = torch.as_tensor(
            np.asarray(
                saved.get(
                    "base_full_confusion",
                    np.zeros((6, 18, 18), dtype=np.int64),
                ),
                dtype=np.int64,
            ),
            dtype=torch.int64,
            device=device,
        ).clone()
        final_full_gpu = torch.as_tensor(
            np.asarray(
                saved.get(
                    "final_full_confusion",
                    np.zeros((6, 18, 18), dtype=np.int64),
                ),
                dtype=np.int64,
            ),
            dtype=torch.int64,
            device=device,
        ).clone()
    else:
        conf_gpu = None
        base_full_gpu = None
        final_full_gpu = None
    loss_sum_scalar = torch.tensor(
        float(saved.get("loss_sum", 0.0)),
        dtype=torch.float64,
        device=device,
    )
    cpu_work = float(saved.get("cpu_work_seconds", 0.0))
    prior_elapsed = float(saved.get("elapsed_seconds", 0.0))
    tiles_total = int(saved.get("tiles_total", 0))

    prepared = iter(_iter_prepared(
        stage_root,
        repair_root,
        repair_idx,
        source,
        native_shape=geom.native_shape,
        free_label=FREE_LABEL,
        shuffle=train,
        seed=seed,
        max_windows=total,
        skip_windows=start_window,
        workers=prep_workers,
        prefetch=prefetch,
        fixed_identities=fixed_identities,
        need_metrics=compute_metrics,
    ))

    n = int(start_window)
    started = time.perf_counter()
    cpu_wait = 0.0
    profiler = _CudaStageProfiler(
        device,
        enabled=bool(profile_gpu_stages),
        warmup_windows=int(profile_warmup_windows),
    )
    last_report_n = int(start_window)
    last_report_time = started
    last_report_cpu_wait = 0.0
    last_report_cpu_work = float(cpu_work)
    last_report_tiles = int(tiles_total)

    while n < total:
        tw = time.perf_counter()
        try:
            item = next(prepared)
        except StopIteration:
            break
        cpu_wait += time.perf_counter() - tw
        cpu_work += float(item["cpu_seconds"])

        prof_events = profiler.begin_window()
        profiler.start(prof_events, "h2d")
        sem = torch.from_numpy(item["sem"]).to(
            device, non_blocking=True
        ).unsqueeze(0)
        obs = torch.from_numpy(item["obs"]).to(
            device, non_blocking=True
        ).unsqueeze(0)
        obsfree = torch.from_numpy(item["obsfree"]).to(
            device, non_blocking=True
        ).unsqueeze(0)
        profiler.stop(prof_events, "h2d")

        if train:
            optimizer.zero_grad(set_to_none=True)

        with _autocast(device, amp):
            profiler.start(prof_events, "encoder")
            scene = model.encode_history(sem, obs, obsfree)
            profiler.stop(prof_events, "encoder")

            profiler.start(prof_events, "geometry")
            linear, q = geom.future_linear_and_query(item["future_rel"])
            profiler.stop(prof_events, "geometry")

            profiler.start(prof_events, "tile_decode")
            query_logits, query_row_map, ntiles = (
                _decode_query_logits_sparse(
                    model,
                    scene,
                    q,
                    item["obs"],
                    geom,
                    tile_batch_size=tile_batch_size,
                )
            )
            profiler.stop(prof_events, "tile_decode")

            profiler.start(prof_events, "loss")
            if compute_metrics:
                loss, conf, base_full, final_full = (
                    _repair_sparse_loss_and_confusion(
                        query_logits,
                        query_row_map,
                        linear,
                        item["support"],
                        item["target"],
                        gt=item["gt"],
                        base_full_conf=item["base_full_conf"],
                        base_support_conf=item["base_support_conf"],
                        geom=geom,
                        device=device,
                    )
                )
            else:
                loss = _repair_sparse_loss_only(
                    query_logits,
                    query_row_map,
                    linear,
                    item["support"],
                    item["target"],
                    geom=geom,
                    device=device,
                )
                conf = base_full = final_full = None
            profiler.stop(prof_events, "loss")

        finite = torch.isfinite(loss.detach()).all()
        if device.type == "cuda" and hasattr(torch, "_assert_async"):
            torch._assert_async(finite, "non-finite Static Repair loss")
        elif not bool(finite.item()):
            raise RuntimeError("non-finite Static Repair loss")

        if train:
            profiler.start(prof_events, "backward")
            loss.backward()
            profiler.stop(prof_events, "backward")
            profiler.start(prof_events, "optim")
            torch.nn.utils.clip_grad_norm_(
                trainable_params,
                5.0,
                error_if_nonfinite=True,
                foreach=(device.type == "cuda"),
            )
            optimizer.step()
            profiler.stop(prof_events, "optim")

        profiler.finish_window(prof_events)
        loss_sum_scalar.add_(loss.detach().double())
        if compute_metrics:
            conf_gpu.add_(conf)
            base_full_gpu.add_(base_full)
            final_full_gpu.add_(final_full)
        tiles_total += int(ntiles)
        n += 1

        report = (
            n == int(start_window) + 1
            or n % max(int(progress_every), 1) == 0
            or n == total
        )
        save_now = (
            train
            and checkpoint_callback is not None
            and int(checkpoint_every) > 0
            and n < total
            and n % int(checkpoint_every) == 0
        )
        if report or save_now:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            now = time.perf_counter()
            seg = max(n - int(start_window), 1)
            gpu_stage_ms = profiler.flush_after_sync()
            gpu_recent_ms = profiler.interval_summary()
            if compute_metrics:
                diag = repair_diagnostics_from_confusion(
                    conf_gpu.detach().cpu().numpy()
                )
                base_metrics = full_grid_metrics_from_confusion(
                    base_full_gpu.detach().cpu().numpy()
                )
                final_metrics = full_grid_metrics_from_confusion(
                    final_full_gpu.detach().cpu().numpy()
                )
            else:
                diag = {}
                base_metrics = {}
                final_metrics = {}
            if report:
                phase = "train" if train else "val"
                metric_text = (
                    f"addP={diag['addition_precision']:.4f} "
                    f"addR={diag['static_positive_recall']:.4f} "
                    f"base_mIoU={base_metrics['mIoU']:.2f} "
                    f"repair_mIoU={final_metrics['mIoU']:.2f} "
                    f"delta={final_metrics['mIoU']-base_metrics['mIoU']:+.2f}"
                    if compute_metrics
                    else "metrics=deferred_to_val"
                )
                gpu_text = ""
                if gpu_stage_ms:
                    parts = [
                        f"{name}:{value:.2f}"
                        for name, value in gpu_stage_ms.items()
                        if value is not None
                    ]
                    if parts:
                        gpu_text = " gpu_ms=" + ",".join(parts)
                if gpu_recent_ms:
                    parts = [
                        f"{name}:{value:.2f}"
                        for name, value in gpu_recent_ms.items()
                        if value is not None
                    ]
                    if parts:
                        gpu_text += " gpu_recent_ms=" + ",".join(parts)
                recent_n = max(n - last_report_n, 1)
                recent_elapsed = max(now - last_report_time, 1e-9)
                recent_cpu_wait = (
                    cpu_wait - last_report_cpu_wait
                ) / recent_n
                recent_cpu_work = (
                    cpu_work - last_report_cpu_work
                ) / recent_n
                recent_tiles = (
                    tiles_total - last_report_tiles
                ) / recent_n
                print(
                    f"v20_static_repair_{phase} {n}/{total} "
                    f"rate={seg/max(now-started,1e-9):.3f} win/s "
                    f"recent_rate={recent_n/recent_elapsed:.3f} win/s "
                    f"cpu_wait={cpu_wait/seg:.3f}s/win "
                    f"recent_cpu_wait={recent_cpu_wait:.3f}s/win "
                    f"cpu_work={cpu_work/max(n,1):.3f}s/win "
                    f"recent_cpu_work={recent_cpu_work:.3f}s/win "
                    f"tiles={tiles_total/max(n,1):.1f}/win "
                    f"recent_tiles={recent_tiles:.1f}/win "
                    f"loss={float(loss_sum_scalar.item()/max(n,1)):.5f} "
                    f"{metric_text}{gpu_text}",
                    flush=True,
                )
                last_report_n = n
                last_report_time = now
                last_report_cpu_wait = cpu_wait
                last_report_cpu_work = cpu_work
                last_report_tiles = tiles_total
            if save_now:
                checkpoint_callback(
                    n,
                    {
                        "loss_sum": float(loss_sum_scalar.item()),
                        **({
                            "confusion": (
                                conf_gpu.detach().cpu().numpy().tolist()
                            ),
                            "base_full_confusion": (
                                base_full_gpu.detach().cpu().numpy().tolist()
                            ),
                            "final_full_confusion": (
                                final_full_gpu.detach().cpu().numpy().tolist()
                            ),
                        } if compute_metrics else {}),
                        "cpu_work_seconds": float(cpu_work),
                        "tiles_total": int(tiles_total),
                        "elapsed_seconds": float(
                            prior_elapsed + now - started
                        ),
                    },
                )

        del query_logits, query_row_map, linear, q, scene

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    gpu_stage_ms = profiler.flush_after_sync()
    if compute_metrics:
        conf_np = conf_gpu.cpu().numpy()
        base_full_np = base_full_gpu.cpu().numpy()
        final_full_np = final_full_gpu.cpu().numpy()
        diag = repair_diagnostics_from_confusion(conf_np)
        base_metrics = full_grid_metrics_from_confusion(base_full_np)
        final_metrics = full_grid_metrics_from_confusion(final_full_np)
        full_delta = {
            "IoU": float(final_metrics["IoU"] - base_metrics["IoU"]),
            "mIoU": float(final_metrics["mIoU"] - base_metrics["mIoU"]),
            "main_1_2_3s": {
                "IoU": float(
                    final_metrics["main_1_2_3s"]["IoU"]
                    - base_metrics["main_1_2_3s"]["IoU"]
                ),
                "mIoU": float(
                    final_metrics["main_1_2_3s"]["mIoU"]
                    - base_metrics["main_1_2_3s"]["mIoU"]
                ),
            },
        }
        conf_list = conf_np.tolist()
    else:
        diag = {}
        base_metrics = {}
        final_metrics = {}
        full_delta = {}
        conf_list = None
    return {
        "metrics_computed": bool(compute_metrics),
        "loss": float(loss_sum_scalar.item() / max(n, 1)),
        "windows": int(n),
        "mean_tiles_per_window": float(
            tiles_total / max(n, 1)
        ),
        "mean_cpu_work_seconds_per_window": float(
            cpu_work / max(n, 1)
        ),
        "epoch_elapsed_seconds": float(
            prior_elapsed + time.perf_counter() - started
        ),
        "repair_diagnostics": diag,
        "full_grid_v18_metrics": base_metrics,
        "full_grid_v18_plus_static_metrics": final_metrics,
        "full_grid_delta": full_delta,
        "confusion_18x18": conf_list,
        "gpu_stage_ms_per_window": gpu_stage_ms,
        "checkpoint_selection_metric": (
            "NONE; use formal composed dev mIoU"
        ),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage1-train-cache", required=True)
    p.add_argument("--stage1-val-cache", required=True)
    p.add_argument("--repair-train-cache", required=True)
    p.add_argument("--repair-val-cache", required=True)
    p.add_argument("--v18-checkpoint", required=True)
    p.add_argument("--dataroot", required=True)
    p.add_argument("--train-info-pkl", required=True)
    p.add_argument("--val-info-pkl", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--tile-size", default="32,32,16")
    p.add_argument("--tile-batch-size", type=int, default=128)
    p.add_argument(
        "--val-tile-batch-size",
        type=int,
        default=256,
        help="Larger inference-only tile batch for validation.",
    )
    p.add_argument("--prep-workers", type=int, default=4)
    p.add_argument("--prefetch", type=int, default=16)
    p.add_argument("--progress-every", type=int, default=50)
    p.add_argument("--checkpoint-every-windows", type=int, default=2000)
    p.add_argument("--max-train-windows", type=int, default=0)
    p.add_argument("--max-val-windows", type=int, default=0)
    p.add_argument(
        "--overfit-windows",
        type=int,
        default=0,
        help=(
            "Diagnostic only: train and validate on the same first N train "
            "windows. Checkpoints are explicitly marked non-formal."
        ),
    )
    p.add_argument("--resume", default="")
    p.add_argument("--seed", type=int, default=20260926)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument(
        "--profile-gpu-stages",
        action="store_true",
        help=(
            "Use CUDA events to report mean h2d/encoder/geometry/tile/loss/"
            "backward/optimizer milliseconds at normal progress sync points."
        ),
    )
    p.add_argument(
        "--profile-warmup-windows",
        type=int,
        default=5,
        help="Windows excluded from CUDA stage timing warmup.",
    )
    a = p.parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)

    st_root, st_idx = _load_index(
        a.stage1_train_cache, STAGE1_PROTOCOL
    )
    sv_root, sv_idx = _load_index(
        a.stage1_val_cache, STAGE1_PROTOCOL
    )
    rt_root, rt_idx = _load_index(
        a.repair_train_cache, SUPPORT_CACHE_PROTOCOL
    )
    rv_root, rv_idx = _load_index(
        a.repair_val_cache, SUPPORT_CACHE_PROTOCOL
    )

    for name, sroot, ridx in (
        ("train", st_root, rt_idx),
        ("val", sv_root, rv_idx),
    ):
        if str(Path(ridx["stage1_cache"]).resolve()) != str(
            Path(sroot).resolve()
        ):
            raise RuntimeError(
                f"{name} repair cache references different Stage1 cache"
            )
        if str(Path(ridx["base_checkpoint"]).resolve()) != str(
            Path(a.v18_checkpoint).resolve()
        ):
            raise RuntimeError(
                f"{name} repair cache references different V18 checkpoint"
            )
        if bool(ridx.get("future_gt_used", True)):
            raise RuntimeError("repair support cache must not use future GT")
        if bool(ridx.get("future_lidar_mask_used", True)):
            raise RuntimeError(
                "repair support cache must not use future lidar mask"
            )
        if not bool(
            ridx.get("contains_lossless_v18_semantic_prediction", False)
        ):
            raise RuntimeError(
                "repair cache lacks lossless frozen-V18 semantic prediction"
            )

    overlap = set(st_idx["scene_names"]) & set(sv_idx["scene_names"])
    if overlap:
        raise RuntimeError(f"train/val scene overlap: {sorted(overlap)[:5]}")

    overfit = int(a.overfit_windows)
    fixed_overfit_identities = None
    fixed_overfit_population_sha256 = ""
    if overfit > 0:
        fixed_overfit_identities = _fixed_identity_subset(
            st_root, rt_root, rt_idx, overfit
        )
        fixed_overfit_population_sha256 = _identity_fingerprint(
            fixed_overfit_identities
        )
        # Validation is intentionally the exact same frozen N identities.
        sv_root, sv_idx = st_root, st_idx
        rv_root, rv_idx = rt_root, rt_idx
        a.max_train_windows = overfit
        a.max_val_windows = overfit
        val_info = a.train_info_pkl
    else:
        val_info = a.val_info_pkl

    train_support_truncated = bool(
        rt_idx.get(
            "truncated_population",
            int(rt_idx["num_windows"]) < int(st_idx["num_windows"]),
        )
    )
    val_support_truncated = bool(
        rv_idx.get(
            "truncated_population",
            int(rv_idx["num_windows"]) < int(sv_idx["num_windows"]),
        )
    )
    diagnostic_reasons = []
    if overfit > 0:
        diagnostic_reasons.append("overfit_windows")
    if int(a.max_train_windows) > 0:
        diagnostic_reasons.append("max_train_windows")
    if int(a.max_val_windows) > 0:
        diagnostic_reasons.append("max_val_windows")
    if train_support_truncated:
        diagnostic_reasons.append("truncated_train_support_cache")
    if val_support_truncated:
        diagnostic_reasons.append("truncated_val_support_cache")
    diagnostic_only = bool(diagnostic_reasons)

    if st_idx["highres_lattice"] != sv_idx["highres_lattice"]:
        raise RuntimeError("train/val high-resolution lattice mismatch")
    if st_idx["coarse_lattice"] != sv_idx["coarse_lattice"]:
        raise RuntimeError("train/val coarse lattice mismatch")
    if dict(st_idx["native_grid"]) != dict(sv_idx["native_grid"]):
        raise RuntimeError("train/val native grid mismatch")

    device = torch.device(
        a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    amp = device.type == "cuda" and not bool(a.no_amp)

    v18_obj = torch.load(
        a.v18_checkpoint, map_location="cpu", weights_only=False
    )
    v18_cfg = dict(v18_obj.get("model_config") or {})
    if "d_model" not in v18_cfg:
        raise RuntimeError("V18 checkpoint lacks model_config.d_model")

    cfg = V20SceneConfig(source_dim=int(v18_cfg["d_model"]))
    model = V20HistoryWorldModel(cfg)
    for p0 in model.dormant.parameters():
        p0.requires_grad = False
    for p0 in model.birth.parameters():
        p0.requires_grad = False
    for p0 in model.static.coarse_head.parameters():
        p0.requires_grad = False
    model.to(device)

    optimizer = torch.optim.AdamW(
        [p0 for p0 in model.parameters() if p0.requires_grad],
        lr=float(a.lr),
        weight_decay=float(a.weight_decay),
        foreach=(device.type == "cuda"),
    )

    high = _lattice(st_idx["highres_lattice"])
    coarse = _lattice(st_idx["coarse_lattice"])
    native = dict(st_idx["native_grid"])
    native_shape = tuple(int(x) for x in native["shape_xyz"])
    native_origin = tuple(float(x) for x in native["origin_xyz_m"])
    native_step = tuple(float(x) for x in native["voxel_size_xyz_m"])
    if tuple(int(x) for x in rt_idx["native_shape_xyz"]) != native_shape:
        raise RuntimeError("repair/native shape mismatch")
    tile_size = tuple(int(x) for x in a.tile_size.split(","))
    if len(tile_size) != 3 or min(tile_size) <= 0:
        raise ValueError("tile-size must be positive xyz triple")

    geom = _Geometry(
        high,
        coarse,
        native_shape,
        native_origin,
        native_step,
        tile_size,
        device,
    )
    train_source = CachedSource(
        a.dataroot, info_pkl=a.train_info_pkl, verbose=False
    )
    val_source = (
        train_source
        if overfit > 0
        else CachedSource(
            a.dataroot, info_pkl=val_info, verbose=False
        )
    )

    out = Path(a.output_dir)
    resume_path = (
        Path(a.resume).resolve() if str(a.resume).strip() else None
    )
    history = []
    start_epoch = 1
    resume_window = 0
    resume_partial = None

    if resume_path is None:
        if out.exists() and any(out.iterdir()):
            raise FileExistsError(
                f"refusing non-empty output dir: {out}"
            )
        out.mkdir(parents=True, exist_ok=True)
    else:
        if not resume_path.is_file():
            raise FileNotFoundError(resume_path)
        out.mkdir(parents=True, exist_ok=True)
        ck = torch.load(
            resume_path, map_location="cpu", weights_only=False
        )
        extra = dict(ck.get("extra") or {})
        if extra.get("train_protocol") != TRAIN_PROTOCOL:
            raise RuntimeError(
                "resume checkpoint is not Static Repair v2"
            )
        contract = dict(extra.get("training_contract") or {})
        expected = {
            "stage1_train_cache": str(Path(a.stage1_train_cache).resolve()),
            "repair_train_cache": str(Path(a.repair_train_cache).resolve()),
            "seed": int(a.seed),
            "tile_size_xyz": list(tile_size),
            "overfit_windows": int(overfit),
            "max_train_windows": int(a.max_train_windows),
            "max_val_windows": int(a.max_val_windows),
            "diagnostic_only": bool(diagnostic_only),
            "fixed_overfit_population_sha256": (
                fixed_overfit_population_sha256
            ),
        }
        for key, value in expected.items():
            if contract.get(key) != value:
                raise RuntimeError(
                    f"resume contract mismatch {key}: "
                    f"{contract.get(key)!r} != {value!r}"
                )
        model.load_state_dict(ck["model"], strict=True)
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        _restore_rng(ck.get("rng_state"))
        progress = dict(ck.get("training_progress") or {})
        history = list(progress.get("history") or [])
        saved_epoch = int(progress.get("epoch", 0))
        if bool(progress.get("epoch_complete", False)):
            start_epoch = saved_epoch + 1
        else:
            start_epoch = saved_epoch
            resume_window = int(progress.get("completed_windows", 0))
            resume_partial = dict(
                progress.get("partial_epoch_state") or {}
            )
        print(
            f"resumed Static Repair v2: epoch={start_epoch} "
            f"window={resume_window}",
            flush=True,
        )

    if start_epoch > int(a.epochs):
        raise RuntimeError("requested epochs already completed")

    train_total = (
        min(int(rt_idx["num_windows"]), int(a.max_train_windows))
        if int(a.max_train_windows) > 0
        else int(rt_idx["num_windows"])
    )

    def make_payload(
        epoch,
        *,
        epoch_complete,
        completed_windows,
        partial,
        hist,
    ):
        payload = checkpoint_payload(
            model,
            stage="static",
            v18_checkpoint=str(Path(a.v18_checkpoint).resolve()),
            thresholds={},
            extra={
                "train_protocol": TRAIN_PROTOCOL,
                "static_role": "protected_add_only_repair",
                "supervision_domain": (
                    "formal_full_future_grid_intersection_v18_free"
                ),
                "future_lidar_mask_used_for_supervision": False,
                "dynamic_gt_target": "free_no_add",
                "query_mask_contract": "exact_runtime_M_query",
                "loss": "ordinary_per_future_voxel_cross_entropy",
                "class_weighting": "none",
                "tile_weighting": "none",
                "horizon_conflicts": (
                    "preserved_as_independent_future_voxel_contributions"
                ),
                "overfit_diagnostic_only": bool(overfit > 0),
                "diagnostic_only": bool(diagnostic_only),
                "diagnostic_reasons": list(diagnostic_reasons),
                "checkpoint_eligible_for_formal_selection": bool(
                    not diagnostic_only
                ),
                "fixed_overfit_population_sha256": (
                    fixed_overfit_population_sha256
                ),
                "highres_lattice": st_idx["highres_lattice"],
                "coarse_lattice": st_idx["coarse_lattice"],
                "tile_size_xyz": list(tile_size),
                "tile_batch_size": int(a.tile_batch_size),
                "training_contract": {
                    "stage1_train_cache": str(
                        Path(a.stage1_train_cache).resolve()
                    ),
                    "repair_train_cache": str(
                        Path(a.repair_train_cache).resolve()
                    ),
                    "stage1_val_cache": str(
                        Path(a.stage1_val_cache).resolve()
                    ),
                    "repair_val_cache": str(
                        Path(a.repair_val_cache).resolve()
                    ),
                    "seed": int(a.seed),
                    "tile_size_xyz": list(tile_size),
                    "lr": float(a.lr),
                    "weight_decay": float(a.weight_decay),
                    "overfit_windows": int(overfit),
                    "max_train_windows": int(a.max_train_windows),
                    "max_val_windows": int(a.max_val_windows),
                    "diagnostic_only": bool(diagnostic_only),
                    "fixed_overfit_population_sha256": (
                        fixed_overfit_population_sha256
                    ),
                },
                "history": list(hist),
                "selection": (
                    "Select only by formal composed scene-disjoint dev mIoU; "
                    "repair diagnostics are safety/capacity diagnostics."
                ),
            },
        )
        payload["optimizer_state_dict"] = optimizer.state_dict()
        payload["rng_state"] = _capture_rng()
        payload["training_progress"] = {
            "epoch": int(epoch),
            "epoch_complete": bool(epoch_complete),
            "completed_windows": int(completed_windows),
            "total_windows": int(train_total),
            "partial_epoch_state": partial,
            "history": list(hist),
        }
        return payload

    print(json.dumps({
        "protocol": TRAIN_PROTOCOL,
        "device": str(device),
        "amp_bfloat16": bool(amp),
        "train_windows": int(train_total),
        "tile_batch_size": int(a.tile_batch_size),
        "val_tile_batch_size": int(a.val_tile_batch_size),
        "val_windows": (
            min(int(rv_idx["num_windows"]), int(a.max_val_windows))
            if int(a.max_val_windows) > 0
            else int(rv_idx["num_windows"])
        ),
        "overfit_diagnostic_only": bool(overfit > 0),
        "diagnostic_only": bool(diagnostic_only),
        "diagnostic_reasons": list(diagnostic_reasons),
        "fixed_overfit_population_sha256": (
            fixed_overfit_population_sha256 or None
        ),
        "profile_gpu_stages": bool(a.profile_gpu_stages),
        "profile_warmup_windows": int(a.profile_warmup_windows),
        "support": "formal full grid AND frozen V18 free",
        "target": "static semantic; free/dynamic -> no-add",
        "class_weighting": "none",
        "tile_weighting": "none",
    }, indent=2), flush=True)

    for epoch in range(start_epoch, int(a.epochs) + 1):
        this_start = resume_window if epoch == start_epoch else 0
        this_partial = (
            resume_partial if epoch == start_epoch else None
        )

        def save_partial(done, state):
            _atomic_save(
                make_payload(
                    epoch,
                    epoch_complete=False,
                    completed_windows=done,
                    partial=state,
                    hist=history,
                ),
                out / "resume_latest.pt",
            )

        tr = _epoch(
            model,
            st_root,
            rt_root,
            rt_idx,
            train_source,
            geom,
            device=device,
            optimizer=optimizer,
            tile_batch_size=int(a.tile_batch_size),
            prep_workers=int(a.prep_workers),
            prefetch=int(a.prefetch),
            progress_every=int(a.progress_every),
            seed=int(a.seed) + epoch,
            amp=amp,
            max_windows=int(a.max_train_windows),
            start_window=int(this_start),
            resume_state=this_partial,
            checkpoint_every=int(a.checkpoint_every_windows),
            checkpoint_callback=save_partial,
            fixed_identities=fixed_overfit_identities,
            profile_gpu_stages=bool(a.profile_gpu_stages),
            profile_warmup_windows=int(a.profile_warmup_windows),
        )
        with torch.inference_mode():
            va = _epoch(
                model,
                sv_root,
                rv_root,
                rv_idx,
                val_source,
                geom,
                device=device,
                optimizer=None,
                tile_batch_size=int(a.val_tile_batch_size),
                prep_workers=int(a.prep_workers),
                prefetch=int(a.prefetch),
                progress_every=int(a.progress_every),
                seed=int(a.seed),
                amp=amp,
                max_windows=int(a.max_val_windows),
                fixed_identities=fixed_overfit_identities,
                profile_gpu_stages=bool(a.profile_gpu_stages),
                profile_warmup_windows=int(a.profile_warmup_windows),
            )

        row = {"epoch": int(epoch), "train": tr, "val": va}
        history.append(row)
        print(json.dumps(row), flush=True)

        payload = make_payload(
            epoch,
            epoch_complete=True,
            completed_windows=train_total,
            partial=None,
            hist=history,
        )
        _atomic_save(payload, out / f"epoch_{epoch:04d}.pt")
        _atomic_save(payload, out / "latest.pt")
        _atomic_save(payload, out / "resume_latest.pt")
        resume_window = 0
        resume_partial = None

    (out / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
