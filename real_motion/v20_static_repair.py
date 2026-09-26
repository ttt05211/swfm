"""V20 Stage-2 Static Repair supervision contract.

The deployed Static branch is protected-add-only on top of frozen V18.
Training therefore supervises only positions where frozen V18 predicts free,
over the same full future occupancy grid used by formal evaluation.

Target semantics on that support:
* static occupied GT -> its static semantic label;
* free GT -> free/no-add;
* dynamic GT -> free/no-add (owned by Dormant/Birth, never Static).

Future lidar observation masks are deliberately absent from this contract.
"""
from __future__ import annotations

import hashlib
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch

from .metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS
from .v20_history_world import FREE_LABEL
from .v20_stage1_codec import pack_bool, unpack_bool

SUPPORT_CACHE_PROTOCOL = "p0_f9_v20_static_repair_support_v2"
TRAIN_PROTOCOL = "p0_f9_v20_static_repair_train_v2"

DYNAMIC_IDS = tuple(int(x) for x in DYNAMIC_CLASS_IDS)
DYNAMIC_SET = frozenset(DYNAMIC_IDS)
STATIC_ALLOWED_IDS = tuple(i for i in range(18) if i not in DYNAMIC_SET)
STATIC_SEMANTIC_IDS = tuple(i for i in range(17) if i not in DYNAMIC_SET)


def population_fingerprint(rows: Iterable[Mapping[str, object]]) -> str:
    h = hashlib.sha256()
    for row in rows:
        h.update(str(row["scene_name"]).encode("utf-8"))
        h.update(b"\0")
        h.update(str(row["t0_token"]).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def pack_v18_free_support(mask: np.ndarray) -> torch.Tensor:
    arr = np.asarray(mask, dtype=bool)
    if arr.ndim != 4 or arr.shape[0] != 6:
        raise ValueError("V18-free support must be [6,X,Y,Z]")
    return pack_bool(arr)


def unpack_v18_free_support(
    bits: torch.Tensor | np.ndarray,
    native_shape_xyz: Sequence[int],
) -> np.ndarray:
    shape = (6,) + tuple(int(x) for x in native_shape_xyz)
    return unpack_bool(bits, shape)


def _pack_5bit(values: np.ndarray) -> torch.Tensor:
    vals = np.asarray(values, dtype=np.uint8).reshape(-1)
    if vals.size and int(vals.max()) > 31:
        raise ValueError("5-bit pack received value >31")
    if vals.size == 0:
        return torch.empty(0, dtype=torch.uint8)
    shifts = np.arange(5, dtype=np.uint8)
    bits = ((vals[:, None] >> shifts[None]) & 1).astype(np.uint8)
    return torch.from_numpy(
        np.packbits(bits.reshape(-1), bitorder="little").copy()
    )


def _unpack_5bit(packed: torch.Tensor | np.ndarray, count: int) -> np.ndarray:
    count = int(count)
    if count == 0:
        return np.empty(0, dtype=np.uint8)
    arr = np.asarray(
        packed.cpu() if isinstance(packed, torch.Tensor) else packed,
        dtype=np.uint8,
    )
    bits = np.unpackbits(
        arr.reshape(-1), bitorder="little", count=count * 5
    ).reshape(count, 5)
    weights = (1 << np.arange(5, dtype=np.uint8))[None]
    return (bits * weights).sum(
        axis=1, dtype=np.uint16
    ).astype(np.uint8)


def pack_v18_prediction(
    pred: np.ndarray,
    *,
    free_label: int = FREE_LABEL,
) -> dict[str, object]:
    y = np.asarray(pred, dtype=np.uint8)
    if y.ndim != 4 or y.shape[0] != 6:
        raise ValueError("V18 prediction must be [6,X,Y,Z]")
    if y.size and int(y.max()) >= 18:
        raise ValueError("V18 prediction outside frozen taxonomy")
    free = y == int(free_label)
    occupied_labels = y[~free]
    return {
        "v18_free_bits": pack_v18_free_support(free),
        "v18_occupied_semantic_5bit": _pack_5bit(occupied_labels),
        "v18_occupied_semantic_count": int(occupied_labels.size),
        "v18_free_count_by_horizon": torch.from_numpy(
            free.reshape(6, -1).sum(axis=1).astype(np.int64)
        ),
    }


def unpack_v18_prediction(
    row: Mapping[str, object],
    native_shape_xyz: Sequence[int],
    *,
    free_label: int = FREE_LABEL,
) -> np.ndarray:
    free = unpack_v18_free_support(
        row["v18_free_bits"], native_shape_xyz
    )
    labels = _unpack_5bit(
        row["v18_occupied_semantic_5bit"],
        int(row["v18_occupied_semantic_count"]),
    )
    occupied = ~free
    if int(occupied.sum()) != int(labels.size):
        raise RuntimeError("V18 occupied semantic count mismatch")
    out = np.full(free.shape, int(free_label), dtype=np.uint8)
    out[occupied] = labels
    return out


def repair_target_from_gt(
    gt: np.ndarray,
    *,
    free_label: int = FREE_LABEL,
) -> np.ndarray:
    """Convert formal future GT into Static add/no-add semantic targets."""
    y = np.asarray(gt, dtype=np.uint8)
    if y.ndim != 4 or y.shape[0] != 6:
        raise ValueError("future GT must be [6,X,Y,Z]")
    if y.size and int(y.max()) >= 18:
        raise ValueError("future GT contains label outside frozen 18-class taxonomy")
    out = y.copy()
    dyn = np.isin(out, np.asarray(DYNAMIC_IDS, dtype=np.uint8))
    out[dyn] = int(free_label)
    return out


def repair_confusion(
    target: np.ndarray,
    pred: np.ndarray,
    support: np.ndarray,
    *,
    free_label: int = FREE_LABEL,
) -> np.ndarray:
    """18x18 confusion on the actual Static decision support.

    The GT-free row is retained.  Static semantic mIoU later excludes free
    from the class average but free->semantic errors remain in semantic unions.
    """
    y = np.asarray(target)
    p = np.asarray(pred)
    s = np.asarray(support, dtype=bool)
    if y.shape != p.shape or y.shape != s.shape:
        raise ValueError("repair target/pred/support shape mismatch")
    yy = y[s].astype(np.int64, copy=False)
    pp = p[s].astype(np.int64, copy=False)
    code = yy * 18 + pp
    return np.bincount(code, minlength=18 * 18).reshape(18, 18)


def repair_diagnostics_from_confusion(
    conf: np.ndarray,
    *,
    free_label: int = FREE_LABEL,
) -> dict[str, object]:
    c = np.asarray(conf, dtype=np.int64)
    if c.shape != (18, 18):
        raise ValueError("repair confusion must be 18x18")
    free = int(free_label)
    static_ids = tuple(int(x) for x in STATIC_SEMANTIC_IDS)

    pred_occ = int(c[:, :free].sum())
    gt_static = int(c[list(static_ids), :].sum())
    add_tp = int(c[list(static_ids), :free].sum())
    add_fp = int(c[free, :free].sum())
    semantic_correct = int(sum(c[i, i] for i in static_ids))

    per = {}
    vals = []
    for cid in static_ids:
        tp = int(c[cid, cid])
        union = int(c[cid, :].sum() + c[:, cid].sum() - tp)
        iou = float(tp / union) if union else float("nan")
        per[str(cid)] = iou
        if union:
            vals.append(iou)

    return {
        "support_voxels": int(c.sum()),
        "predicted_add_voxels": pred_occ,
        "target_static_positive_voxels": gt_static,
        "added_tp": add_tp,
        "added_fp": add_fp,
        "addition_precision": float(add_tp / max(add_tp + add_fp, 1)),
        "static_positive_recall": float(add_tp / max(gt_static, 1)),
        "semantic_accuracy_on_static_positive": float(
            semantic_correct / max(add_tp, 1)
        ),
        "repair_support_semantic_miou": (
            float(np.mean(vals)) if vals else float("nan")
        ),
        "per_static_class_iou": per,
    }


def full_grid_metrics_from_confusion(conf_by_horizon: np.ndarray) -> dict[str, object]:
    """Formal full-grid IoU/mIoU summary from six 18x18 confusions."""
    c = np.asarray(conf_by_horizon, dtype=np.int64)
    if c.shape != (6, 18, 18):
        raise ValueError("full-grid confusion must be [6,18,18]")
    occ, miou = [], []
    per = {}
    for hi in range(6):
        ch = c[hi]
        oi = int(ch[:17, :17].sum())
        ou = int(ch.sum() - ch[17, 17])
        ov = float(100.0 * oi / ou) if ou else float("nan")
        vals = []
        for cid in range(17):
            tp = int(ch[cid, cid])
            union = int(
                ch[cid, :].sum() + ch[:, cid].sum() - tp
            )
            if union:
                vals.append(float(tp / union))
        mv = float(100.0 * np.mean(vals)) if vals else float("nan")
        occ.append(ov)
        miou.append(mv)
        per[str(0.5 * (hi + 1))] = {"IoU": ov, "mIoU": mv}
    main = (1, 3, 5)
    return {
        "IoU": float(np.nanmean(occ)),
        "mIoU": float(np.nanmean(miou)),
        "main_1_2_3s": {
            "IoU": float(np.nanmean([occ[i] for i in main])),
            "mIoU": float(np.nanmean([miou[i] for i in main])),
        },
        "per_horizon": per,
    }
