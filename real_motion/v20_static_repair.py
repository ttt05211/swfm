"""V20 Stage-2 Static Repair v2 supervision contracts.

The v1 Static objective was future-LiDAR sparse while formal evaluation was
full-grid.  Repair-v2 makes training match deployment:

* supervision domain is the full formal future occupancy grid;
* Static has decision authority only where frozen V18 predicts FREE;
* static GT occupied -> its semantic class;
* GT free or dynamic -> FREE/no-add;
* six horizon contributions are retained independently after canonical mapping;
* no class reweighting is part of this protocol.

The cache stores only two compact native-grid masks plus packed positive static
labels. Full-grid FREE/no-add targets are implicit, which keeps the cache
lossless without materializing dense semantic targets.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from .metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS
from .motion_transport import FUTURE_FRAMES
from .v20_stage1_codec import (
    _pack_nibbles,
    _unpack_nibbles,
    pack_bool,
    unpack_bool,
)

FREE_LABEL = 17
REPAIR_CACHE_PROTOCOL = "p0_f9_v20_static_repair_cache_v2"
REPAIR_TRAIN_PROTOCOL = "p0_f9_v20_static_repair_train_v2"
REPAIR_STAGE = "static_repair_v2"

DYNAMIC_IDS = tuple(int(x) for x in DYNAMIC_CLASS_IDS)
DYNAMIC_SET = frozenset(DYNAMIC_IDS)
STATIC_POSITIVE_IDS = tuple(
    i for i in range(FREE_LABEL) if i not in DYNAMIC_SET
)
if len(STATIC_POSITIVE_IDS) > 16:
    raise RuntimeError("Static positive taxonomy no longer fits 4 bits")

_STATIC_POS_TO_CODE = np.full(18, 255, dtype=np.uint8)
for _code, _label in enumerate(STATIC_POSITIVE_IDS):
    _STATIC_POS_TO_CODE[_label] = _code
_CODE_TO_STATIC_POS = np.asarray(STATIC_POSITIVE_IDS, dtype=np.uint8)


def pack_static_repair_supervision(
    v18_prediction: np.ndarray,
    future_gt: np.ndarray,
    *,
    free_label: int = FREE_LABEL,
) -> dict[str, object]:
    """Pack exact full-grid composition-aware Static supervision.

    V18-occupied voxels are outside Static's deployed decision support.
    Everywhere else the target defaults to FREE/no-add; only static occupied GT
    needs an explicit semantic label. Dynamic GT is intentionally FREE/no-add
    for this branch and remains the responsibility of V18/Dormant/Birth.
    """
    pred = np.asarray(v18_prediction, dtype=np.uint8)
    gt = np.asarray(future_gt, dtype=np.uint8)
    if pred.shape != gt.shape or pred.ndim != 4:
        raise ValueError("V18 prediction/GT must match [F,X,Y,Z]")
    if pred.shape[0] != FUTURE_FRAMES:
        raise ValueError("Static Repair v2 requires six future frames")
    if pred.size and (int(pred.min()) < 0 or int(pred.max()) > int(free_label)):
        raise ValueError("V18 prediction label outside semantic taxonomy")
    if gt.size and (int(gt.min()) < 0 or int(gt.max()) > int(free_label)):
        raise ValueError("future GT label outside semantic taxonomy")

    v18_occupied = pred != int(free_label)
    dynamic_gt = np.isin(gt, np.asarray(DYNAMIC_IDS, dtype=np.uint8))
    static_positive = (
        (~v18_occupied)
        & (gt != int(free_label))
        & (~dynamic_gt)
    )
    labels = gt[static_positive]
    codes = (
        _STATIC_POS_TO_CODE[labels]
        if labels.size
        else np.empty(0, dtype=np.uint8)
    )
    if codes.size and bool((codes == 255).any()):
        bad = np.unique(labels[codes == 255]).tolist()
        raise RuntimeError(
            f"non-static label entered Static Repair positive targets: {bad}"
        )

    support = ~v18_occupied
    static_positive_count = int(static_positive.sum())
    support_count = int(support.sum())
    dynamic_noadd_count = int((support & dynamic_gt).sum())
    free_noadd_count = int((support & (gt == int(free_label))).sum())
    if static_positive_count + dynamic_noadd_count + free_noadd_count != support_count:
        raise RuntimeError("Static Repair support partition is not exhaustive")

    return {
        "shape_fxyz": tuple(int(x) for x in pred.shape),
        "v18_occupied_bits": pack_bool(v18_occupied),
        "static_positive_bits": pack_bool(static_positive),
        "static_positive_semantic_4bit": _pack_nibbles(codes),
        "static_positive_count": static_positive_count,
        "v18_occupied_count": int(v18_occupied.sum()),
        "support_count": support_count,
        "free_noadd_count": free_noadd_count,
        "dynamic_noadd_count": dynamic_noadd_count,
    }


def unpack_static_repair_supervision(
    row: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return V18 occupied mask, static-positive mask and positive labels."""
    shape = tuple(int(x) for x in row["shape_fxyz"])
    if len(shape) != 4 or shape[0] != FUTURE_FRAMES:
        raise RuntimeError("invalid Static Repair cached shape")
    v18_occupied = unpack_bool(row["v18_occupied_bits"], shape)
    static_positive = unpack_bool(row["static_positive_bits"], shape)
    if bool((static_positive & v18_occupied).any()):
        raise RuntimeError("Static Repair positive overlaps V18-occupied support")
    count = int(row["static_positive_count"])
    if count != int(static_positive.sum()):
        raise RuntimeError("Static Repair positive count/mask mismatch")
    codes = _unpack_nibbles(
        row["static_positive_semantic_4bit"], count
    )
    if codes.size and int(codes.max()) >= len(_CODE_TO_STATIC_POS):
        raise RuntimeError("invalid Static Repair semantic code")
    labels = (
        _CODE_TO_STATIC_POS[codes]
        if codes.size
        else np.empty(0, dtype=np.uint8)
    )
    if int((~v18_occupied).sum()) != int(row["support_count"]):
        raise RuntimeError("Static Repair support count mismatch")
    return v18_occupied, static_positive, labels


def repair_confusion_summary(confusion: np.ndarray) -> dict[str, float | int]:
    """Composition-aware diagnostics from exact repair-target confusion.

    Rows are repair GT (static semantic or FREE/no-add), columns are Static
    predictions. Dynamic semantic output columns are structurally impossible.
    FREE-row false positives are retained in every static-class IoU union.
    """
    conf = np.asarray(confusion, dtype=np.int64)
    if conf.shape != (18, 18):
        raise ValueError("repair confusion must be 18x18")
    static_ids = np.asarray(STATIC_POSITIVE_IDS, dtype=np.int64)
    tp_sem = int(sum(int(conf[c, c]) for c in STATIC_POSITIVE_IDS))
    gt_pos = int(conf[static_ids, :].sum())
    added_on_pos = int(conf[np.ix_(static_ids, static_ids)].sum())
    added_on_noadd = int(conf[int(FREE_LABEL), static_ids].sum())
    added = added_on_pos + added_on_noadd
    vals = []
    per = {}
    for cid in STATIC_POSITIVE_IDS:
        tp = int(conf[cid, cid])
        union = int(conf[cid, :].sum() + conf[:, cid].sum() - tp)
        value = float(tp / union) if union else float("nan")
        per[str(cid)] = value
        if union:
            vals.append(value)
    return {
        "repair_static_mIoU": float(np.mean(vals)) if vals else float("nan"),
        "added_precision": float(added_on_pos / max(added, 1)),
        "added_recall": float(added_on_pos / max(gt_pos, 1)),
        "semantic_accuracy_on_positive_adds": float(
            tp_sem / max(added_on_pos, 1)
        ),
        "added_voxels": added,
        "added_positive": added_on_pos,
        "added_noadd_fp": added_on_noadd,
        "target_static_positive": gt_pos,
        "semantic_correct_positive": tp_sem,
        "per_class_iou": per,
    }
