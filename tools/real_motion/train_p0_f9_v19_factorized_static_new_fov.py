#!/usr/bin/env python3
"""Train factorized Static New-FOV completion on a frozen V18+Memory context."""
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
from real_motion.v19_static_novelty_factorized import (
    ANCHOR_DISTANCE_MAX_M,
    FactorizedStaticNewFOVHead,
    dequantize_anchor_distance_torch,
    factorized_static_new_fov_loss,
)
from tools.real_motion.build_p0_f9_v19_factorized_static_new_fov_cache import (
    PROTOCOL as CACHE_PROTOCOL,
)


PROTOCOL = "p0_f9_v19_factorized_static_new_fov_train_v1"
HEAD_TYPE = "factorized_static_new_fov"


def _parse_floats(text):
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
    idx = json.loads((p / "index.json").read_text(encoding="utf-8"))
    if idx.get("protocol") != CACHE_PROTOCOL:
        raise RuntimeError(
            f"unexpected factorized cache protocol: {idx.get('protocol')}"
        )
    if not idx.get("shards"):
        raise RuntimeError("factorized cache has no shards")
    return p, idx


def _iter_batches(root, index, *, batch_size, shuffle, seed):
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
        "presence_target",
        "vertical_target_bits",
        "semantic_target",
        "anchor_semantic",
        "anchor_profile_bits",
        "anchor_distance_q",
        "gt_occupied_bits",
    )
    for si in shard_ids.tolist():
        row = index["shards"][int(si)]
        obj = torch.load(
            root / row["file"],
            map_location="cpu",
            weights_only=False,
        )
        if obj.get("protocol") != CACHE_PROTOCOL:
            raise RuntimeError(f"shard protocol mismatch: {row['file']}")
        n = int(row["count"])
        order = np.arange(n)
        if shuffle:
            rng.shuffle(order)
        for st in range(0, n, int(batch_size)):
            ids = torch.as_tensor(
                order[st : st + int(batch_size)],
                dtype=torch.long,
            )
            yield {k: obj[k].index_select(0, ids) for k in keys}


def _autocast(device, enabled):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _move(raw, device, z, anchor_distance_max_m):
    mv = lambda x: x.to(device, non_blocking=True)
    base_free = unpack_vertical_occupancy_torch(
        mv(raw["base_free_bits"]),
        z,
    ).bool()
    vertical = unpack_vertical_occupancy_torch(
        mv(raw["vertical_target_bits"]),
        z,
    ).bool()
    anchor_profile = unpack_vertical_occupancy_torch(
        mv(raw["anchor_profile_bits"]),
        z,
    )
    gt_occupied = unpack_vertical_occupancy_torch(
        mv(raw["gt_occupied_bits"]),
        z,
    ).bool()
    new_fov = mv(raw["new_fov_mask"]).bool()
    return {
        "semantic": mv(raw["future_aligned_semantic"]),
        "geometry": dequantize_geometry_torch(
            mv(raw["future_aligned_geometry_q"])
        ),
        "base_explained": mv(raw["base_explained"]).float(),
        "base_free": base_free,
        "new_fov": new_fov,
        "candidate_bev": new_fov & base_free.any(dim=2),
        "presence_target": mv(raw["presence_target"]).bool(),
        "vertical_target": vertical,
        "semantic_target": mv(raw["semantic_target"]).long(),
        "anchor_semantic": mv(raw["anchor_semantic"]).long(),
        "anchor_profile": anchor_profile,
        "anchor_distance_m": dequantize_anchor_distance_torch(
            mv(raw["anchor_distance_q"]),
            anchor_distance_max_m,
        ),
        "gt_occupied": gt_occupied,
    }


def _prf(tp, fp, fn):
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2.0 * p * r / max(p + r, 1e-12)
    f05 = 1.25 * p * r / max(0.25 * p + r, 1e-12)
    return float(p), float(r), float(f1), float(f05)


def _evaluate(
    model,
    root,
    index,
    device,
    *,
    batch_size,
    z,
    amp,
    presence_weight,
    vertical_weight_pos,
    semantic_weight,
    vertical_loss_weight,
    presence_thresholds,
    vertical_thresholds,
    anchor_distance_max_m,
):
    model.eval()
    sums = {
        "loss": 0.0,
        "presence_bce": 0.0,
        "semantic_ce": 0.0,
        "vertical_bce": 0.0,
    }
    nb = 0
    sem_ok = sem_n = 0

    # Conditional vertical quality assumes GT-positive BEV, isolating geometry
    # from the presence task.
    vertical_rows = {
        float(t): {"tp": 0, "fp": 0, "fn": 0}
        for t in vertical_thresholds
    }
    presence_rows = {
        float(t): {"tp": 0, "fp": 0, "fn": 0}
        for t in presence_thresholds
    }
    joint = {
        (float(pt), float(vt)): {
            "gt_tp": 0,
            "gt_fp": 0,
            "target_tp": 0,
            "target_fp": 0,
            "target_fn": 0,
            "predicted_voxels": 0,
        }
        for pt in presence_thresholds
        for vt in vertical_thresholds
    }

    with torch.inference_mode():
        for raw in _iter_batches(
            root,
            index,
            batch_size=batch_size,
            shuffle=False,
            seed=0,
        ):
            b = _move(raw, device, z, anchor_distance_max_m)
            with _autocast(device, amp):
                out = model(
                    b["semantic"],
                    b["geometry"],
                    b["base_explained"],
                    b["new_fov"],
                    b["anchor_semantic"],
                    b["anchor_profile"],
                    b["anchor_distance_m"],
                )
                loss, stats = factorized_static_new_fov_loss(
                    out,
                    presence_target=b["presence_target"],
                    candidate_bev=b["candidate_bev"],
                    semantic_target=b["semantic_target"],
                    vertical_target=b["vertical_target"],
                    base_free=b["base_free"],
                    presence_positive_weight=presence_weight,
                    vertical_positive_weight=vertical_weight_pos,
                    semantic_weight=semantic_weight,
                    vertical_weight=vertical_loss_weight,
                )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("non-finite factorized validation loss")
            for k in sums:
                sums[k] += float(stats[k])
            nb += 1

            tgt_bev = b["presence_target"] & b["candidate_bev"]
            if bool(tgt_bev.any()):
                sem_pred = out["semantic_logits"].argmax(dim=2)
                sem_ok += int(
                    (
                        sem_pred[tgt_bev]
                        == b["semantic_target"][tgt_bev]
                    ).sum().item()
                )
                sem_n += int(tgt_bev.sum().item())

            pres_prob = torch.sigmoid(out["presence_logits"].float())
            vert_prob = torch.sigmoid(out["vertical_logits"].float())

            for pt in presence_thresholds:
                pred_bev = (
                    pres_prob >= float(pt)
                ) & b["candidate_bev"]
                rr = presence_rows[float(pt)]
                rr["tp"] += int((pred_bev & tgt_bev).sum().item())
                rr["fp"] += int(
                    (pred_bev & ~tgt_bev & b["candidate_bev"]).sum().item()
                )
                rr["fn"] += int((~pred_bev & tgt_bev).sum().item())

            gt_pos_vox = tgt_bev.unsqueeze(2) & b["base_free"]
            for vt in vertical_thresholds:
                vpred = (
                    vert_prob >= float(vt)
                ) & gt_pos_vox
                vtgt = b["vertical_target"] & gt_pos_vox
                rr = vertical_rows[float(vt)]
                rr["tp"] += int((vpred & vtgt).sum().item())
                rr["fp"] += int((vpred & ~vtgt & gt_pos_vox).sum().item())
                rr["fn"] += int((~vpred & vtgt).sum().item())

            for pt in presence_thresholds:
                active = (
                    pres_prob >= float(pt)
                ) & b["candidate_bev"]
                for vt in vertical_thresholds:
                    pred = (
                        active.unsqueeze(2)
                        & (vert_prob >= float(vt))
                        & b["base_free"]
                    )
                    rr = joint[(float(pt), float(vt))]
                    gt_occ = b["gt_occupied"]
                    target = b["vertical_target"]
                    rr["gt_tp"] += int((pred & gt_occ).sum().item())
                    rr["gt_fp"] += int((pred & ~gt_occ).sum().item())
                    rr["target_tp"] += int((pred & target).sum().item())
                    rr["target_fp"] += int((pred & ~target).sum().item())
                    rr["target_fn"] += int((~pred & target).sum().item())
                    rr["predicted_voxels"] += int(pred.sum().item())

    presence_table = []
    for th, rr in presence_rows.items():
        p, r, f1, f05 = _prf(rr["tp"], rr["fp"], rr["fn"])
        presence_table.append(
            {
                "threshold": th,
                "precision": p,
                "recall": r,
                "f1": f1,
                "f0_5": f05,
            }
        )
    vertical_table = []
    for th, rr in vertical_rows.items():
        p, r, f1, f05 = _prf(rr["tp"], rr["fp"], rr["fn"])
        vertical_table.append(
            {
                "threshold": th,
                "precision": p,
                "recall": r,
                "f1": f1,
                "f0_5": f05,
            }
        )

    base_i = int(index["totals"]["baseline_occ_inter"])
    base_u = int(index["totals"]["baseline_occ_union"])
    base_iou = base_i / max(base_u, 1)
    joint_table = []
    for (pt, vt), rr in joint.items():
        new_iou = (
            (base_i + rr["gt_tp"])
            / max(base_u + rr["gt_fp"], 1)
        )
        tp = rr["target_tp"]
        fp = rr["target_fp"]
        fn = rr["target_fn"]
        p, r, f1, f05 = _prf(tp, fp, fn)
        joint_table.append(
            {
                "presence_threshold": pt,
                "vertical_threshold": vt,
                "estimated_occ_iou": float(100.0 * new_iou),
                "estimated_delta_occ_iou": float(
                    100.0 * (new_iou - base_iou)
                ),
                "target_voxel_precision": p,
                "target_voxel_recall": r,
                "target_voxel_f1": f1,
                "target_voxel_f0_5": f05,
                "predicted_voxels": int(rr["predicted_voxels"]),
                "gt_occupied_added_tp": int(rr["gt_tp"]),
                "gt_free_added_fp": int(rr["gt_fp"]),
            }
        )
    selected = max(
        joint_table,
        key=lambda x: (
            x["estimated_delta_occ_iou"],
            x["target_voxel_f0_5"],
            x["target_voxel_precision"],
        ),
    )
    best_presence = max(
        presence_table,
        key=lambda x: (x["f1"], x["precision"]),
    )
    best_vertical = max(
        vertical_table,
        key=lambda x: (x["f1"], x["precision"]),
    )
    return {
        **{k: v / max(nb, 1) for k, v in sums.items()},
        "semantic_accuracy_on_positive_bev": float(
            sem_ok / max(sem_n, 1)
        ),
        "best_presence": best_presence,
        "best_conditional_vertical": best_vertical,
        "selected": selected,
        "presence_threshold_table": presence_table,
        "conditional_vertical_threshold_table": vertical_table,
        "joint_threshold_table": joint_table,
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
    presence_weight,
    vertical_weight_pos,
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
            "presence_positive_weight": float(presence_weight),
            "vertical_positive_weight": float(vertical_weight_pos),
            "semantic_weight": float(args.semantic_weight),
            "vertical_loss_weight": float(args.vertical_weight),
            "selected_presence_threshold": float(
                val_report["selected"]["presence_threshold"]
            ),
            "selected_vertical_threshold": float(
                val_report["selected"]["vertical_threshold"]
            ),
            "base_checkpoint": train_index.get("base_checkpoint"),
            "base_checkpoint_epoch": train_index.get("base_checkpoint_epoch"),
            "target": train_index.get("target"),
            "representation": train_index.get("representation"),
            "anchor_condition": train_index.get("anchor_condition"),
            "train_cache": {
                k: train_index.get(k)
                for k in (
                    "protocol",
                    "num_windows",
                    "num_scenes",
                    "presence_positive_fraction",
                    "vertical_positive_fraction_on_positive_columns",
                    "baseline_occ_iou",
                    "totals",
                )
            },
            "val_cache": {
                k: val_index.get(k)
                for k in (
                    "protocol",
                    "num_windows",
                    "num_scenes",
                    "presence_positive_fraction",
                    "vertical_positive_fraction_on_positive_columns",
                    "baseline_occ_iou",
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
    p.add_argument("--anchor-semantic-dim", type=int, default=4)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--semantic-weight", type=float, default=1.0)
    p.add_argument("--vertical-weight", type=float, default=1.0)
    p.add_argument("--presence-positive-weight", type=float, default=0.0)
    p.add_argument("--vertical-positive-weight", type=float, default=0.0)
    p.add_argument("--max-positive-weight", type=float, default=4.0)
    p.add_argument(
        "--presence-thresholds",
        default="0.30,0.40,0.50,0.60,0.70,0.80",
    )
    p.add_argument(
        "--vertical-thresholds",
        default="0.20,0.30,0.40,0.50,0.60,0.70",
    )
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=20260924)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-amp", action="store_true")
    a = p.parse_args()

    if min(
        int(a.epochs),
        int(a.batch_size),
        int(a.semantic_dim),
        int(a.anchor_semantic_dim),
        int(a.hidden_dim),
    ) <= 0:
        raise ValueError("invalid training dimensions")
    if (
        float(a.lr) <= 0
        or float(a.weight_decay) < 0
        or float(a.semantic_weight) < 0
        or float(a.vertical_weight) < 0
    ):
        raise ValueError("invalid optimizer/loss arguments")

    presence_thresholds = _parse_floats(a.presence_thresholds)
    vertical_thresholds = _parse_floats(a.vertical_thresholds)

    random.seed(int(a.seed))
    np.random.seed(int(a.seed))
    torch.manual_seed(int(a.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(a.seed))

    train_root, train_index = _load_index(a.train_cache)
    val_root, val_index = _load_index(a.val_cache)
    if train_index["grid_shape_hwd"] != val_index["grid_shape_hwd"]:
        raise RuntimeError("factorized train/val grid mismatch")
    overlap = sorted(
        set(train_index.get("scene_names", []))
        & set(val_index.get("scene_names", []))
    )
    if overlap:
        raise RuntimeError(
            f"factorized train/val scene overlap: {overlap[:5]}"
        )

    z = int(train_index["grid_shape_hwd"][2])
    anchor_distance_max_m = float(
        train_index.get("anchor_distance_max_m", ANCHOR_DISTANCE_MAX_M)
    )
    if abs(
        anchor_distance_max_m
        - float(val_index.get("anchor_distance_max_m", ANCHOR_DISTANCE_MAX_M))
    ) > 1e-9:
        raise RuntimeError("anchor distance quantization mismatch")

    arch = {
        "future_frames": int(train_index["future_frames"]),
        "history_frames": int(train_index["history_frames"]),
        "semantic_dim": int(a.semantic_dim),
        "anchor_semantic_dim": int(a.anchor_semantic_dim),
        "hidden_dim": int(a.hidden_dim),
        "num_semantic_classes": 17,
        "vertical_bins": z,
        "anchor_distance_max_m": anchor_distance_max_m,
    }
    device = torch.device(
        a.device
        if a.device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    amp = device.type == "cuda" and not bool(a.no_amp)
    model = FactorizedStaticNewFOVHead(**arch).to(device)

    presence_prior = float(train_index["presence_positive_fraction"])
    vertical_prior = float(
        train_index["vertical_positive_fraction_on_positive_columns"]
    )
    model.set_output_priors(
        presence_probability=presence_prior,
        vertical_probability=vertical_prior,
    )

    pres_ratio = float(train_index["presence_neg_pos_ratio"])
    vert_ratio = float(
        train_index["vertical_neg_pos_ratio_on_positive_columns"]
    )
    auto_pres = math.sqrt(max(pres_ratio, 1.0))
    auto_vert = math.sqrt(max(vert_ratio, 1.0))
    presence_weight = (
        float(a.presence_positive_weight)
        if float(a.presence_positive_weight) > 0
        else min(max(auto_pres, 1.0), float(a.max_positive_weight))
    )
    vertical_weight_pos = (
        float(a.vertical_positive_weight)
        if float(a.vertical_positive_weight) > 0
        else min(max(auto_vert, 1.0), float(a.max_positive_weight))
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(a.lr),
        weight_decay=float(a.weight_decay),
    )
    out_dir = Path(a.output_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty output dir: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    initial_val = _evaluate(
        model,
        val_root,
        val_index,
        device,
        batch_size=int(a.batch_size),
        z=z,
        amp=amp,
        presence_weight=presence_weight,
        vertical_weight_pos=vertical_weight_pos,
        semantic_weight=float(a.semantic_weight),
        vertical_loss_weight=float(a.vertical_weight),
        presence_thresholds=presence_thresholds,
        vertical_thresholds=vertical_thresholds,
        anchor_distance_max_m=anchor_distance_max_m,
    )
    preflight = {
        "protocol": PROTOCOL,
        "head_type": HEAD_TYPE,
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "clean_e14_parameters_in_optimizer": 0,
        "train_windows": int(train_index["num_windows"]),
        "val_windows": int(val_index["num_windows"]),
        "presence_prior": presence_prior,
        "vertical_prior_on_positive_columns": vertical_prior,
        "presence_positive_weight": float(presence_weight),
        "vertical_positive_weight": float(vertical_weight_pos),
        "architecture": arch,
        "initial_selected": initial_val["selected"],
        "amp_bfloat16": bool(amp),
    }
    (out_dir / "preflight.json").write_text(
        json.dumps(preflight, indent=2),
        encoding="utf-8",
    )
    print("=== V19 FACTORIZED STATIC NEW-FOV PREFLIGHT ===")
    print(json.dumps(preflight, indent=2))

    global_step = 0
    best_delta_iou = -float("inf")
    best_loss = float("inf")
    history = []
    for epoch in range(1, int(a.epochs) + 1):
        model.train()
        sums = {
            "loss": 0.0,
            "presence_bce": 0.0,
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
            b = _move(raw, device, z, anchor_distance_max_m)
            with _autocast(device, amp):
                out = model(
                    b["semantic"],
                    b["geometry"],
                    b["base_explained"],
                    b["new_fov"],
                    b["anchor_semantic"],
                    b["anchor_profile"],
                    b["anchor_distance_m"],
                )
                loss, stats = factorized_static_new_fov_loss(
                    out,
                    presence_target=b["presence_target"],
                    candidate_bev=b["candidate_bev"],
                    semantic_target=b["semantic_target"],
                    vertical_target=b["vertical_target"],
                    base_free=b["base_free"],
                    presence_positive_weight=presence_weight,
                    vertical_positive_weight=vertical_weight_pos,
                    semantic_weight=float(a.semantic_weight),
                    vertical_weight=float(a.vertical_weight),
                )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(
                    f"non-finite factorized loss epoch={epoch} "
                    f"step={global_step + 1}"
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
            presence_weight=presence_weight,
            vertical_weight_pos=vertical_weight_pos,
            semantic_weight=float(a.semantic_weight),
            vertical_loss_weight=float(a.vertical_weight),
            presence_thresholds=presence_thresholds,
            vertical_thresholds=vertical_thresholds,
            anchor_distance_max_m=anchor_distance_max_m,
        )
        selected = val_report["selected"]
        row = {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "train": train_report,
            "val": val_report,
        }
        history.append(row)

        bp = val_report["best_presence"]
        bv = val_report["best_conditional_vertical"]
        print(
            f"E{epoch:02d} "
            f"loss={val_report['loss']:.4f} "
            f"sem={val_report['semantic_accuracy_on_positive_bev']:.3f} "
            f"BEV={bp['precision']:.3f}/{bp['recall']:.3f}/{bp['f1']:.3f}"
            f"@{bp['threshold']:.2f} "
            f"condZ={bv['precision']:.3f}/{bv['recall']:.3f}/{bv['f1']:.3f}"
            f"@{bv['threshold']:.2f} "
            f"sel=({selected['presence_threshold']:.2f},"
            f"{selected['vertical_threshold']:.2f}) "
            f"dIoU~={selected['estimated_delta_occ_iou']:+.3f} "
            f"voxP/R={selected['target_voxel_precision']:.3f}/"
            f"{selected['target_voxel_recall']:.3f}",
            flush=True,
        )

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
            presence_weight=presence_weight,
            vertical_weight_pos=vertical_weight_pos,
            val_report=val_report,
        )
        score = float(selected["estimated_delta_occ_iou"])
        if score > best_delta_iou:
            best_delta_iou = score
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
                presence_weight=presence_weight,
                vertical_weight_pos=vertical_weight_pos,
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
                presence_weight=presence_weight,
                vertical_weight_pos=vertical_weight_pos,
                val_report=val_report,
            )
        (out_dir / "history.json").write_text(
            json.dumps(history, indent=2),
            encoding="utf-8",
        )

    print(f"saved factorized Static New-FOV checkpoints to {out_dir}")


if __name__ == "__main__":
    main()
