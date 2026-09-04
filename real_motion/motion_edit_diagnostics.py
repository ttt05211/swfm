"""Voxel-level physical motion-edit diagnostics for anchor-relative forecasting.

These metrics intentionally separate the *physical action* implied by a future
occupancy proposal from final mIoU.  Relative to a Strong-W2Det anchor, GT
future occupancy defines three motion events inside a frozen write support:

- CLEAR: anchor dynamic -> GT non-dynamic (the old occupied location must vanish)
- KEEP:  anchor dynamic -> GT dynamic (dynamic presence must remain)
- WRITE: anchor non-dynamic -> GT dynamic (new dynamic occupancy must appear)

A proposal is then scored on whether it clears, preserves, or writes the correct
voxels.  This directly exposes stale/ghost dynamic occupancy, wrong clears,
missed writes, and false writes.  It is voxel-level because the Occ3D semantic
payload has no persistent instance identity; therefore these statistics are a
physical ghost/double-occupancy proxy, not an instance-level duplicate-car
metric.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


_COUNT_KEYS = (
    "support_voxels",
    "required_clear",
    "required_keep",
    "required_keep_same_class",
    "required_relabel",
    "required_write",
    "stable_non_dynamic",
    "correct_clear",
    "stale_dynamic",
    "correct_keep_presence",
    "wrong_clear",
    "correct_keep_class",
    "correct_write_presence",
    "missed_write",
    "correct_write_class",
    "false_write",
    "predicted_clear",
    "predicted_write",
    "correct_predicted_clear",
    "correct_predicted_write",
    "proposal_dynamic",
    "gt_dynamic",
    "proposal_dynamic_true_presence",
)


def _safe_ratio(num: int, den: int) -> float:
    return float(num) / float(den) if int(den) > 0 else float("nan")


def motion_edit_counts(
    anchor_occ: np.ndarray,
    proposal_occ: np.ndarray,
    gt_occ: np.ndarray,
    write_support_bev: np.ndarray,
    *,
    dynamic_class_ids: Iterable[int],
) -> dict[str, int]:
    """Return physical CLEAR/KEEP/WRITE counts for one horizon or time stack.

    Occupancy tensors are [X,Y,Z] or [T,X,Y,Z].  ``write_support_bev`` is
    [X,Y] or [T,X,Y] and is expanded through height.  Only supported voxels are
    counted, because outside support the sparse WM has no deployment authority.
    """
    anchor = np.asarray(anchor_occ)
    proposal = np.asarray(proposal_occ)
    gt = np.asarray(gt_occ)
    write = np.asarray(write_support_bev, dtype=bool)
    if anchor.shape != proposal.shape or anchor.shape != gt.shape or anchor.ndim not in (3, 4):
        raise ValueError("anchor/proposal/gt must match [X,Y,Z] or [T,X,Y,Z]")
    if write.shape != anchor.shape[:-1]:
        raise ValueError("write_support_bev must match occupancy leading BEV dimensions")
    dynamic = np.asarray(tuple(int(x) for x in dynamic_class_ids), dtype=np.int64)
    if dynamic.size == 0:
        raise ValueError("dynamic_class_ids must be non-empty")

    support = np.broadcast_to(write[..., None], anchor.shape)
    a_dyn = np.isin(anchor, dynamic)
    p_dyn = np.isin(proposal, dynamic)
    g_dyn = np.isin(gt, dynamic)

    req_clear = support & a_dyn & ~g_dyn
    req_keep = support & a_dyn & g_dyn
    req_keep_same = req_keep & (anchor == gt)
    req_relabel = req_keep & (anchor != gt)
    req_write = support & ~a_dyn & g_dyn
    stable_non = support & ~a_dyn & ~g_dyn

    pred_clear = support & a_dyn & ~p_dyn
    pred_write = support & ~a_dyn & p_dyn

    counts = {
        "support_voxels": int(support.sum()),
        "required_clear": int(req_clear.sum()),
        "required_keep": int(req_keep.sum()),
        "required_keep_same_class": int(req_keep_same.sum()),
        "required_relabel": int(req_relabel.sum()),
        "required_write": int(req_write.sum()),
        "stable_non_dynamic": int(stable_non.sum()),
        "correct_clear": int((req_clear & ~p_dyn).sum()),
        "stale_dynamic": int((req_clear & p_dyn).sum()),
        "correct_keep_presence": int((req_keep & p_dyn).sum()),
        "wrong_clear": int((req_keep & ~p_dyn).sum()),
        "correct_keep_class": int((req_keep & p_dyn & (proposal == gt)).sum()),
        "correct_write_presence": int((req_write & p_dyn).sum()),
        "missed_write": int((req_write & ~p_dyn).sum()),
        "correct_write_class": int((req_write & p_dyn & (proposal == gt)).sum()),
        "false_write": int((stable_non & p_dyn).sum()),
        "predicted_clear": int(pred_clear.sum()),
        "predicted_write": int(pred_write.sum()),
        "correct_predicted_clear": int((pred_clear & ~g_dyn).sum()),
        "correct_predicted_write": int((pred_write & g_dyn).sum()),
        "proposal_dynamic": int((support & p_dyn).sum()),
        "gt_dynamic": int((support & g_dyn).sum()),
        "proposal_dynamic_true_presence": int((support & p_dyn & g_dyn).sum()),
    }
    return counts


def summarize_motion_edit_counts(counts: dict[str, int]) -> dict:
    """Convert accumulated counts into interpretable rates.

    ``stale_dynamic_rate`` is the direct old-position ghost proxy: among voxels
    that GT says must be cleared, how often the proposal still predicts dynamic.
    It does not claim instance-level duplication because no instance association
    is available in the semantic occupancy payload.
    """
    c = {k: int(counts.get(k, 0)) for k in _COUNT_KEYS}
    return {
        "counts": c,
        "clear_recall": _safe_ratio(c["correct_clear"], c["required_clear"]),
        "stale_dynamic_rate": _safe_ratio(c["stale_dynamic"], c["required_clear"]),
        "clear_precision": _safe_ratio(c["correct_predicted_clear"], c["predicted_clear"]),
        "keep_presence_recall": _safe_ratio(c["correct_keep_presence"], c["required_keep"]),
        "wrong_clear_rate": _safe_ratio(c["wrong_clear"], c["required_keep"]),
        "keep_class_accuracy": _safe_ratio(c["correct_keep_class"], c["required_keep"]),
        "write_recall": _safe_ratio(c["correct_write_presence"], c["required_write"]),
        "missed_write_rate": _safe_ratio(c["missed_write"], c["required_write"]),
        "write_class_accuracy": _safe_ratio(c["correct_write_class"], c["required_write"]),
        "write_precision": _safe_ratio(c["correct_predicted_write"], c["predicted_write"]),
        "false_write_rate_on_stable_non_dynamic": _safe_ratio(
            c["false_write"], c["stable_non_dynamic"]
        ),
        "proposal_dynamic_precision": _safe_ratio(
            c["proposal_dynamic_true_presence"], c["proposal_dynamic"]
        ),
        "proposal_dynamic_recall": _safe_ratio(
            c["proposal_dynamic_true_presence"], c["gt_dynamic"]
        ),
        "dynamic_volume_ratio_proposal_over_gt": _safe_ratio(
            c["proposal_dynamic"], c["gt_dynamic"]
        ),
        "predicted_clear_to_write_ratio": _safe_ratio(c["predicted_clear"], c["predicted_write"]),
        "gt_clear_to_write_ratio": _safe_ratio(c["required_clear"], c["required_write"]),
    }


@dataclass
class MotionEditAccumulator:
    dynamic_class_ids: tuple[int, ...]
    counts: dict[str, int] = field(default_factory=lambda: {k: 0 for k in _COUNT_KEYS})

    def update(
        self,
        anchor_occ: np.ndarray,
        proposal_occ: np.ndarray,
        gt_occ: np.ndarray,
        write_support_bev: np.ndarray,
    ) -> None:
        row = motion_edit_counts(
            anchor_occ,
            proposal_occ,
            gt_occ,
            write_support_bev,
            dynamic_class_ids=self.dynamic_class_ids,
        )
        for key in _COUNT_KEYS:
            self.counts[key] += int(row[key])

    def compute(self) -> dict:
        return summarize_motion_edit_counts(self.counts)
