#!/usr/bin/env python3
"""Train V20 Factorized Static Repair.

This is a new Stage-2 protocol that preserves the V18-free deployment support
and full future-grid supervision of Static Repair v2, but factorizes the final
decision into:
  presence: add / no-add
  semantic: one of the nine frozen static classes, only when GT is positive.

Loss:
  L_presence = 0.5 * mean_pos BCE + 0.5 * mean_neg BCE
  L_semantic = CE on GT static-positive contributions only
  L = L_presence + L_semantic

All future voxel/horizon contributions remain equally represented. Repeated
future-to-canonical mappings are aggregated before loss evaluation.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
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

from real_motion.v20_history_world import FREE_LABEL
from real_motion.v20_scene_model import V20HistoryWorldModel, V20SceneConfig
from real_motion.v20_static_repair import (
    STATIC_SEMANTIC_IDS,
    SUPPORT_CACHE_PROTOCOL,
    full_grid_metrics_from_confusion,
    repair_diagnostics_from_confusion,
)
from real_motion.v20_training import checkpoint_payload
from tools.real_motion.build_p0_f9_v20_history_cache import (
    PROTOCOL as STAGE1_PROTOCOL,
)
from tools.real_motion.eval_p0_f9_v19_factorized_static_new_fov import CachedSource
from tools.real_motion.train_p0_f9_v20_static_repair import (
    _Geometry,
    _atomic_save,
    _capture_rng,
    _fixed_identity_subset,
    _identity_fingerprint,
    _iter_prepared,
    _lattice,
    _load_index,
    _restore_rng,
)

PROTOCOL = "p0_f9_v20_static_repair_factorized_train_v1"


def _autocast(device, enabled):
    if enabled and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def _decode_query_factorized_sparse(
    model,
    scene,
    q,
    obs,
    geom,
    *,
    tile_batch_size,
    tile_batch_pad_multiple=1,
):
    if str(model.cfg.static_head_type) != "factorized":
        raise RuntimeError("factorized decoder requires factorized Static head")

    q_flat = q.reshape(-1)
    query_linear = torch.nonzero(q_flat, as_tuple=False).reshape(-1)
    qcount = int(query_linear.numel())
    if qcount <= 0:
        raise RuntimeError("Static Repair query union is empty")

    row_map = torch.full(
        (geom.high_numel,), -1, dtype=torch.int32, device=scene.device
    )
    row_map[query_linear] = torch.arange(
        qcount, dtype=torch.int32, device=scene.device
    )
    presence = torch.empty(qcount, dtype=scene.dtype, device=scene.device)
    semantic = torch.empty(
        (qcount, len(STATIC_SEMANTIC_IDS)),
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
                grids.append(geom.tile_grid(start_t, tshape, scene.dtype))
                qtile = q[sl]
                qrows.append(qtile)
                rrows.append(row_map_3d[sl])
                cmap = geom.tile_coarse_linear(start_t, tshape)
                seen = seen_flat[cmap].reshape(tshape)
                t0_seen = t0_flat[cmap].reshape(tshape)
                srows.append(seen)
                mrows.append(seen & ~t0_seen)

            qstack = torch.stack(qrows, dim=0)
            grid_batch = torch.cat(grids, dim=0)
            seen_batch = torch.stack(srows, dim=0)
            missing_batch = torch.stack(mrows, dim=0)

            real_b = int(qstack.shape[0])
            pad_multiple = max(int(tile_batch_pad_multiple), 1)
            padded_b = real_b
            if pad_multiple > 1 and real_b >= pad_multiple:
                padded_b = (
                    (real_b + pad_multiple - 1) // pad_multiple
                ) * pad_multiple
                padded_b = min(padded_b, bsz)
            if padded_b > real_b:
                pad = padded_b - real_b
                grid_batch = torch.cat(
                    (
                        grid_batch,
                        grid_batch[-1:].expand(
                            pad, *grid_batch.shape[1:]
                        ),
                    ),
                    dim=0,
                )
                zero = torch.zeros(
                    (pad,) + tuple(qstack.shape[1:]),
                    dtype=qstack.dtype,
                    device=qstack.device,
                )
                q_model = torch.cat((qstack, zero), dim=0)
                seen_model = torch.cat((seen_batch, zero), dim=0)
                missing_model = torch.cat((missing_batch, zero), dim=0)
            else:
                q_model = qstack
                seen_model = seen_batch
                missing_model = missing_batch

            p_logits, s_logits = model.static.refine_tiles_factorized(
                scene,
                sample_grid=grid_batch,
                query_mask=q_model,
                seen_mask=seen_model,
                t0_missing_mask=missing_model,
            )
            p_logits = p_logits[:real_b, 0]
            s_logits = s_logits[:real_b]
            p_sel = p_logits[qstack]
            s_sel = s_logits.permute(0, 2, 3, 4, 1)[qstack]
            rows = torch.stack(rrows, dim=0)[qstack].long()
            if p_sel.dtype != presence.dtype:
                p_sel = p_sel.to(presence.dtype)
            if s_sel.dtype != semantic.dtype:
                s_sel = s_sel.to(semantic.dtype)
            presence.index_copy_(0, rows, p_sel)
            semantic.index_copy_(0, rows, s_sel)
            filled[rows] = True
            ntiles += len(chunk)

    complete = filled.all()
    if scene.device.type == "cuda" and hasattr(torch, "_assert_async"):
        torch._assert_async(
            complete, "Factorized Static decode missed query cells"
        )
    elif not bool(complete.item()):
        raise RuntimeError("Factorized Static decode missed query cells")
    return presence, semantic, row_map, int(ntiles)


def _aggregate_targets(
    row_map,
    linear,
    support,
    target,
    *,
    qcount,
    device,
):
    support_t = torch.from_numpy(
        np.asarray(support, dtype=bool)
    ).to(device, non_blocking=True)
    target_t = torch.from_numpy(
        np.asarray(target, dtype=np.uint8)
    ).to(device, non_blocking=True)

    pos = torch.zeros(qcount, dtype=torch.float32, device=device)
    neg = torch.zeros(qcount, dtype=torch.float32, device=device)
    sem = torch.zeros(
        (qcount, len(STATIC_SEMANTIC_IDS)),
        dtype=torch.float32,
        device=device,
    )
    lut = torch.full((18,), -1, dtype=torch.long, device=device)
    static_ids = torch.as_tensor(
        STATIC_SEMANTIC_IDS, dtype=torch.long, device=device
    )
    lut[static_ids] = torch.arange(len(STATIC_SEMANTIC_IDS), device=device)

    for hi in range(6):
        s = support_t[hi].reshape(-1)
        qrow = row_map[linear[hi][s]].long()
        if device.type == "cuda" and hasattr(torch, "_assert_async"):
            torch._assert_async(
                (qrow >= 0).all(),
                "Factorized Static support escaped query union",
            )
        elif not bool((qrow >= 0).all().item()):
            raise RuntimeError(
                "Factorized Static support escaped query union"
            )

        y = target_t[hi].reshape(-1)[s].long()
        is_pos = y != int(FREE_LABEL)
        if bool(is_pos.any()):
            qr = qrow[is_pos]
            yl = lut[y[is_pos]]
            if bool((yl < 0).any()):
                raise RuntimeError(
                    "Factorized Static positive target is not a static class"
                )
            pos.scatter_add_(
                0, qr, torch.ones(qr.shape, device=device)
            )
            keys = qr * len(STATIC_SEMANTIC_IDS) + yl
            sem.view(-1).scatter_add_(
                0, keys, torch.ones(keys.shape, device=device)
            )
        if bool((~is_pos).any()):
            qr = qrow[~is_pos]
            neg.scatter_add_(
                0, qr, torch.ones(qr.shape, device=device)
            )
    return pos, neg, sem


def _factorized_loss(
    presence,
    semantic,
    row_map,
    linear,
    support,
    target,
    *,
    device,
):
    pos, neg, sem = _aggregate_targets(
        row_map,
        linear,
        support,
        target,
        qcount=int(presence.shape[0]),
        device=device,
    )
    p = presence.float()
    s = semantic.float()

    pos_total = pos.sum().clamp_min(1.0)
    neg_total = neg.sum().clamp_min(1.0)
    sem_total = sem.sum().clamp_min(1.0)

    pos_bce = (pos * F.softplus(-p)).sum() / pos_total
    neg_bce = (neg * F.softplus(p)).sum() / neg_total
    presence_loss = 0.5 * (pos_bce + neg_bce)

    row_count = sem.sum(dim=1)
    semantic_loss = (
        row_count * torch.logsumexp(s, dim=1)
        - (sem * s).sum(dim=1)
    ).sum() / sem_total
    total = presence_loss + semantic_loss
    return total, {
        "presence_loss": presence_loss.detach(),
        "semantic_loss": semantic_loss.detach(),
        "positive_bce": pos_bce.detach(),
        "negative_bce": neg_bce.detach(),
        "positive_contributions": pos.sum().detach(),
        "negative_contributions": neg.sum().detach(),
    }


@torch.no_grad()
def _factorized_confusions(
    presence,
    semantic,
    row_map,
    linear,
    support,
    target,
    gt,
    *,
    base_full_conf,
    base_support_conf,
    device,
    threshold=0.5,
):
    support_t = torch.from_numpy(
        np.asarray(support, dtype=bool)
    ).to(device, non_blocking=True)
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
    static_ids = torch.as_tensor(
        STATIC_SEMANTIC_IDS, dtype=torch.long, device=device
    )
    logit_thr = float(np.log(float(threshold) / (1.0 - float(threshold))))

    for hi in range(6):
        s = support_t[hi].reshape(-1)
        qrow = row_map[linear[hi][s]].long()
        y = target_t[hi].reshape(-1)[s].long()
        sem_cls = static_ids[semantic[qrow].float().argmax(dim=1)]
        active = presence[qrow].float() >= logit_thr
        pred = torch.where(
            active,
            sem_cls,
            torch.full_like(sem_cls, int(FREE_LABEL)),
        )
        conf += torch.bincount(
            y * 18 + pred, minlength=18 * 18
        ).reshape(18, 18)

        gy = gt_t[hi].reshape(-1)[s].long()
        repaired = torch.bincount(
            gy * 18 + pred, minlength=18 * 18
        ).reshape(18, 18)
        final_conf[hi] = (
            base_conf[hi] - base_support[hi] + repaired
        )
    return conf, base_conf, final_conf


def _run_epoch(
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
    tile_batch_pad_multiple,
    workers,
    prefetch,
    seed,
    amp,
    max_windows,
    fixed_identities,
    progress_every,
    presence_threshold,
):
    train = optimizer is not None
    model.train(train)
    model.dormant.eval()
    model.birth.eval()
    model.static.coarse_head.eval()

    total = (
        min(int(repair_idx["num_windows"]), int(max_windows))
        if int(max_windows) > 0
        else int(repair_idx["num_windows"])
    )
    prepared = _iter_prepared(
        stage_root,
        repair_root,
        repair_idx,
        source,
        native_shape=geom.native_shape,
        free_label=FREE_LABEL,
        shuffle=train,
        seed=int(seed),
        max_windows=total,
        skip_windows=0,
        workers=int(workers),
        prefetch=int(prefetch),
        fixed_identities=fixed_identities,
        need_metrics=not train,
    )

    loss_sum = 0.0
    presence_sum = 0.0
    semantic_sum = 0.0
    pos_bce_sum = 0.0
    neg_bce_sum = 0.0
    pos_contrib = 0.0
    neg_contrib = 0.0
    tiles = 0

    conf = np.zeros((18, 18), dtype=np.int64)
    base_full = np.zeros((6, 18, 18), dtype=np.int64)
    final_full = np.zeros((6, 18, 18), dtype=np.int64)
    started = time.perf_counter()

    for wi, item in enumerate(prepared, start=1):
        sem = torch.from_numpy(item["sem"]).to(
            device, non_blocking=True
        ).unsqueeze(0)
        obs = torch.from_numpy(item["obs"]).to(
            device, non_blocking=True
        ).unsqueeze(0)
        obsfree = torch.from_numpy(item["obsfree"]).to(
            device, non_blocking=True
        ).unsqueeze(0)
        if train:
            optimizer.zero_grad(set_to_none=True)

        with _autocast(device, amp):
            scene = model.encode_history(sem, obs, obsfree)
            linear, q = geom.future_linear_and_query(item["future_rel"])
            p_logits, s_logits, row_map, ntiles = (
                _decode_query_factorized_sparse(
                    model,
                    scene,
                    q,
                    item["obs"],
                    geom,
                    tile_batch_size=int(tile_batch_size),
                    tile_batch_pad_multiple=int(tile_batch_pad_multiple),
                )
            )
            loss, parts = _factorized_loss(
                p_logits,
                s_logits,
                row_map,
                linear,
                item["support"],
                item["target"],
                device=device,
            )
        if not bool(torch.isfinite(loss.detach()).all()):
            raise RuntimeError("non-finite Factorized Static loss")

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 5.0
            )
            optimizer.step()
        else:
            c, b, f = _factorized_confusions(
                p_logits,
                s_logits,
                row_map,
                linear,
                item["support"],
                item["target"],
                item["gt"],
                base_full_conf=item["base_full_conf"],
                base_support_conf=item["base_support_conf"],
                device=device,
                threshold=float(presence_threshold),
            )
            conf += c.cpu().numpy()
            base_full += b.cpu().numpy()
            final_full += f.cpu().numpy()

        loss_sum += float(loss.detach().cpu())
        presence_sum += float(parts["presence_loss"].cpu())
        semantic_sum += float(parts["semantic_loss"].cpu())
        pos_bce_sum += float(parts["positive_bce"].cpu())
        neg_bce_sum += float(parts["negative_bce"].cpu())
        pos_contrib += float(parts["positive_contributions"].cpu())
        neg_contrib += float(parts["negative_contributions"].cpu())
        tiles += int(ntiles)

        if (
            wi == 1
            or (int(progress_every) > 0 and wi % int(progress_every) == 0)
            or wi == total
        ):
            elapsed = max(time.perf_counter() - started, 1e-9)
            print(
                f"v20_static_factorized_{'train' if train else 'val'} "
                f"{wi}/{total} rate={wi/elapsed:.3f} win/s "
                f"loss={loss_sum/wi:.5f} "
                f"presence={presence_sum/wi:.5f} "
                f"semantic={semantic_sum/wi:.5f} "
                f"tiles={tiles/wi:.1f}/win",
                flush=True,
            )

        del scene, linear, q, p_logits, s_logits, row_map

    out = {
        "loss": float(loss_sum / max(total, 1)),
        "presence_loss": float(presence_sum / max(total, 1)),
        "semantic_loss": float(semantic_sum / max(total, 1)),
        "positive_bce": float(pos_bce_sum / max(total, 1)),
        "negative_bce": float(neg_bce_sum / max(total, 1)),
        "positive_contributions": int(pos_contrib),
        "negative_contributions": int(neg_contrib),
        "positive_fraction": float(
            pos_contrib / max(pos_contrib + neg_contrib, 1.0)
        ),
        "windows": int(total),
        "mean_tiles_per_window": float(tiles / max(total, 1)),
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    if not train:
        diag = repair_diagnostics_from_confusion(conf)
        bm = full_grid_metrics_from_confusion(base_full)
        fm = full_grid_metrics_from_confusion(final_full)
        out.update({
            "repair_diagnostics": diag,
            "full_grid_v18_metrics": bm,
            "full_grid_v18_plus_static_metrics": fm,
            "full_grid_delta": {
                "IoU": float(fm["IoU"] - bm["IoU"]),
                "mIoU": float(fm["mIoU"] - bm["mIoU"]),
                "main_1_2_3s": {
                    "IoU": float(
                        fm["main_1_2_3s"]["IoU"]
                        - bm["main_1_2_3s"]["IoU"]
                    ),
                    "mIoU": float(
                        fm["main_1_2_3s"]["mIoU"]
                        - bm["main_1_2_3s"]["mIoU"]
                    ),
                },
            },
        })
    return out


def _summ(row):
    v = row["val"]
    d = v["repair_diagnostics"]
    delta = v["full_grid_delta"]
    return (
        f"epoch={row['epoch']} "
        f"val_loss={v['loss']:.5f} "
        f"addP={d['addition_precision']:.4f} "
        f"addR={d['static_positive_recall']:.4f} "
        f"semAcc={d['semantic_accuracy_on_static_positive']:.4f} "
        f"dmIoU={delta['mIoU']:+.4f} "
        f"d123={delta['main_1_2_3s']['mIoU']:+.4f}"
    )


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
    p.add_argument("--presence-bias-init", type=float, default=-4.0)
    p.add_argument("--presence-threshold", type=float, default=0.5)
    p.add_argument("--tile-size", default="32,32,16")
    p.add_argument("--tile-batch-size", type=int, default=256)
    p.add_argument("--val-tile-batch-size", type=int, default=256)
    p.add_argument("--tile-batch-pad-multiple", type=int, default=16)
    p.add_argument("--prep-workers", type=int, default=8)
    p.add_argument("--prefetch", type=int, default=32)
    p.add_argument("--progress-every", type=int, default=50)
    p.add_argument("--max-train-windows", type=int, default=0)
    p.add_argument("--max-val-windows", type=int, default=0)
    p.add_argument("--overfit-windows", type=int, default=0)
    p.add_argument("--seed", type=int, default=20260927)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-amp", action="store_true")
    a = p.parse_args()

    if not (0.0 < float(a.presence_threshold) < 1.0):
        raise ValueError("presence threshold must be in (0,1)")

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)

    st_root, st_idx = _load_index(a.stage1_train_cache, STAGE1_PROTOCOL)
    sv_root, sv_idx = _load_index(a.stage1_val_cache, STAGE1_PROTOCOL)
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
        if str(Path(ridx["stage1_cache"]).resolve()) != str(sroot.resolve()):
            raise RuntimeError(
                f"{name} repair cache references different Stage1 cache"
            )
        if str(Path(ridx["base_checkpoint"]).resolve()) != str(
            Path(a.v18_checkpoint).resolve()
        ):
            raise RuntimeError(
                f"{name} repair cache references different V18 checkpoint"
            )

    if set(st_idx["scene_names"]) & set(sv_idx["scene_names"]):
        raise RuntimeError("factorized Static train/val scene overlap")

    overfit = int(a.overfit_windows)
    fixed = None
    fingerprint = ""
    if overfit > 0:
        fixed = _fixed_identity_subset(
            st_root, rt_root, rt_idx, overfit
        )
        fingerprint = _identity_fingerprint(fixed)
        sv_root, sv_idx = st_root, st_idx
        rv_root, rv_idx = rt_root, rt_idx
        a.max_train_windows = overfit
        a.max_val_windows = overfit
        val_info = a.train_info_pkl
    else:
        val_info = a.val_info_pkl

    diagnostic_reasons = []
    if overfit > 0:
        diagnostic_reasons.append("overfit_windows")
    if int(a.max_train_windows) > 0:
        diagnostic_reasons.append("max_train_windows")
    if int(a.max_val_windows) > 0:
        diagnostic_reasons.append("max_val_windows")
    if bool(rt_idx.get("truncated_population", False)):
        diagnostic_reasons.append("truncated_train_support_cache")
    if bool(rv_idx.get("truncated_population", False)):
        diagnostic_reasons.append("truncated_val_support_cache")
    diagnostic_only = bool(diagnostic_reasons)

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

    cfg = V20SceneConfig(
        source_dim=int(v18_cfg["d_model"]),
        static_head_type="factorized",
        static_presence_bias_init=float(a.presence_bias_init),
    )
    model = V20HistoryWorldModel(cfg)
    for p0 in model.dormant.parameters():
        p0.requires_grad = False
    for p0 in model.birth.parameters():
        p0.requires_grad = False
    for p0 in model.static.coarse_head.parameters():
        p0.requires_grad = False
    model.to(device)
    if device.type == "cuda":
        model.encoder.set_channels_last_3d(True)
        model.static.set_tile_channels_last_3d(True)

    optimizer = torch.optim.AdamW(
        [p0 for p0 in model.parameters() if p0.requires_grad],
        lr=float(a.lr),
        weight_decay=float(a.weight_decay),
        foreach=(device.type == "cuda"),
    )

    high = _lattice(st_idx["highres_lattice"])
    coarse = _lattice(st_idx["coarse_lattice"])
    native = dict(st_idx["native_grid"])
    tile_size = tuple(int(x) for x in a.tile_size.split(","))
    geom = _Geometry(
        high,
        coarse,
        tuple(int(x) for x in native["shape_xyz"]),
        tuple(float(x) for x in native["origin_xyz_m"]),
        tuple(float(x) for x in native["voxel_size_xyz_m"]),
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
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing non-empty output dir: {out}")
    out.mkdir(parents=True, exist_ok=True)

    print(json.dumps({
        "protocol": PROTOCOL,
        "device": str(device),
        "amp_bfloat16": bool(amp),
        "static_head_type": "factorized",
        "presence_bias_init": float(a.presence_bias_init),
        "presence_threshold": float(a.presence_threshold),
        "loss": (
            "0.5*mean_positive_BCE + 0.5*mean_negative_BCE "
            "+ static_positive_semantic_CE"
        ),
        "train_windows": (
            min(int(rt_idx["num_windows"]), int(a.max_train_windows))
            if int(a.max_train_windows) > 0
            else int(rt_idx["num_windows"])
        ),
        "val_windows": (
            min(int(rv_idx["num_windows"]), int(a.max_val_windows))
            if int(a.max_val_windows) > 0
            else int(rv_idx["num_windows"])
        ),
        "diagnostic_only": bool(diagnostic_only),
        "diagnostic_reasons": diagnostic_reasons,
    }, indent=2), flush=True)

    history = []
    best_miou = -float("inf")
    for epoch in range(1, int(a.epochs) + 1):
        tr = _run_epoch(
            model,
            st_root,
            rt_root,
            rt_idx,
            train_source,
            geom,
            device=device,
            optimizer=optimizer,
            tile_batch_size=int(a.tile_batch_size),
            tile_batch_pad_multiple=int(a.tile_batch_pad_multiple),
            workers=int(a.prep_workers),
            prefetch=int(a.prefetch),
            seed=int(a.seed) + epoch,
            amp=amp,
            max_windows=int(a.max_train_windows),
            fixed_identities=fixed,
            progress_every=int(a.progress_every),
            presence_threshold=float(a.presence_threshold),
        )
        with torch.inference_mode():
            va = _run_epoch(
                model,
                sv_root,
                rv_root,
                rv_idx,
                val_source,
                geom,
                device=device,
                optimizer=None,
                tile_batch_size=int(a.val_tile_batch_size),
                tile_batch_pad_multiple=int(a.tile_batch_pad_multiple),
                workers=int(a.prep_workers),
                prefetch=int(a.prefetch),
                seed=int(a.seed),
                amp=amp,
                max_windows=int(a.max_val_windows),
                fixed_identities=fixed,
                progress_every=int(a.progress_every),
                presence_threshold=float(a.presence_threshold),
            )
        row = {"epoch": int(epoch), "train": tr, "val": va}
        history.append(row)
        print(json.dumps(row), flush=True)
        print(_summ(row), flush=True)

        payload = checkpoint_payload(
            model,
            stage="static",
            v18_checkpoint=str(Path(a.v18_checkpoint).resolve()),
            thresholds={"static_presence": float(a.presence_threshold)},
            extra={
                "train_protocol": PROTOCOL,
                "static_role": "protected_add_only_repair_factorized",
                "static_head_type": "factorized",
                "presence_bias_init": float(a.presence_bias_init),
                "presence_threshold": float(a.presence_threshold),
                "presence_objective": (
                    "0.5*mean_positive_BCE + 0.5*mean_negative_BCE"
                ),
                "semantic_objective": "CE only on GT static-positive contributions",
                "supervision_domain": (
                    "formal_full_future_grid_intersection_v18_free"
                ),
                "future_lidar_mask_used_for_supervision": False,
                "dynamic_gt_target": "no-add",
                "class_weighting": "none",
                "diagnostic_only": bool(diagnostic_only),
                "diagnostic_reasons": list(diagnostic_reasons),
                "checkpoint_eligible_for_formal_selection": bool(
                    not diagnostic_only
                ),
                "fixed_overfit_population_sha256": fingerprint or None,
                "highres_lattice": st_idx["highres_lattice"],
                "coarse_lattice": st_idx["coarse_lattice"],
                "tile_size_xyz": list(tile_size),
                "tile_batch_pad_multiple": int(a.tile_batch_pad_multiple),
                "channels_last_3d": bool(device.type == "cuda"),
                "epoch": int(epoch),
                "history": history,
                "checkpoint_selection": (
                    "formal composed scene-disjoint dev semantic mIoU"
                ),
            },
        )
        _atomic_save(payload, out / f"epoch_{epoch:04d}.pt")
        _atomic_save(payload, out / "latest.pt")

        miou = float(va["full_grid_v18_plus_static_metrics"]["mIoU"])
        if miou > best_miou:
            best_miou = miou
            _atomic_save(payload, out / "best_dev_miou.pt")

    (out / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
