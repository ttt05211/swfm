"""Training contracts for the V19 residual Innovation Head.

This module deliberately contains only supervision/cache utilities.  The
scientific category contract lives in :mod:`real_motion.v19_innovation_targets`
and the network itself lives in :mod:`real_motion.v19_innovation`.

The key rule is responsibility masking: occupancy that belongs to memory,
known-ancestor transport/model misses, or unresolved ambiguity is *ignored*
for add-presence supervision.  It is never converted into an innovation
negative merely because it is not an innovation positive.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import torch

from .v19_innovation_targets import (
    AMBIGUOUS_CATEGORIES,
    INNOVATION_POSITIVE_CATEGORIES,
    KNOWN_ANCESTOR_MODEL_MISS_CATEGORIES,
    MEMORY_ADDRESSABLE_CATEGORIES,
)


INNOVATION_CACHE_PROTOCOL = "p0_f9_v19_innovation_training_cache_v1"
GEOMETRY_QUANTIZATION_LEVELS = 255.0

EXCLUDED_FROM_INNOVATION_SUPERVISION = (
    *MEMORY_ADDRESSABLE_CATEGORIES,
    *KNOWN_ANCESTOR_MODEL_MISS_CATEGORIES,
    *AMBIGUOUS_CATEGORIES,
)


def quantize_geometry(x: np.ndarray) -> np.ndarray:
    """Quantize normalized [0,1] geometry channels to uint8."""
    a = np.asarray(x, dtype=np.float32)
    if bool((a < -1e-6).any()) or bool((a > 1.0 + 1e-6).any()):
        raise ValueError("innovation geometry must be normalized to [0,1]")
    return np.rint(np.clip(a, 0.0, 1.0) * GEOMETRY_QUANTIZATION_LEVELS).astype(
        np.uint8
    )


def dequantize_geometry_torch(x: torch.Tensor) -> torch.Tensor:
    """Dequantize uint8 cache geometry to float32 [0,1]."""
    return x.to(torch.float32) / GEOMETRY_QUANTIZATION_LEVELS


def pack_vertical_occupancy(mask: np.ndarray) -> np.ndarray:
    """Pack [F,H,W,Z] boolean occupancy into uint16 [F,H,W]."""
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 4:
        raise ValueError("vertical mask must be [F,H,W,Z]")
    z = int(m.shape[-1])
    if z <= 0 or z > 16:
        raise ValueError("uint16 packing supports 1..16 vertical bins")
    out = np.zeros(m.shape[:-1], dtype=np.uint16)
    for zi in range(z):
        out |= (m[..., zi].astype(np.uint16) << np.uint16(zi))
    return out


def unpack_vertical_occupancy_torch(bits: torch.Tensor, vertical_bins: int) -> torch.Tensor:
    """Unpack [B,F,H,W] uint16/int tensor to [B,F,Z,H,W] float32."""
    z = int(vertical_bins)
    if z <= 0 or z > 16:
        raise ValueError("uint16 packing supports 1..16 vertical bins")
    x = bits.to(torch.int64)
    shifts = torch.arange(z, device=x.device, dtype=torch.int64)
    rows = ((x.unsqueeze(2) >> shifts.view(1, 1, z, 1, 1)) & 1)
    return rows.to(torch.float32)


def _union_masks(
    category_masks: Mapping[str, np.ndarray],
    names: tuple[str, ...],
    shape: tuple[int, ...],
) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    for name in names:
        if name not in category_masks:
            raise KeyError(f"missing innovation category mask: {name}")
        m = np.asarray(category_masks[name], dtype=bool)
        if m.shape != shape:
            raise ValueError(f"category mask shape mismatch for {name}")
        out |= m
    return out


def build_innovation_bev_supervision(
    gt_occ: np.ndarray,
    explained_occ: np.ndarray,
    category_masks: Mapping[str, np.ndarray],
    *,
    free_label: int,
) -> dict[str, np.ndarray | int]:
    """Build one future-frame BEV target with explicit ignore semantics.

    Args:
        gt_occ: [H,W,Z] GT semantic occupancy.
        explained_occ: [H,W,Z] frozen Transport+Memory prediction before
            Innovation. Innovation is actionable only where this tensor is free.
        category_masks: disjoint V19 decomposition masks for the same frame.

    Returns:
        add_target [H,W] uint8;
        semantic_target [H,W] uint8 (topmost positive class, free elsewhere);
        vertical_target [H,W,Z] uint8;
        candidate_mask [H,W] uint8;
        ignore_mask [H,W] uint8;
        plus audit counts.

    Any BEV column containing excluded responsibility is ignored wholesale.
    This prevents the column-level add/semantic heads from learning that
    known-ancestor or ambiguous occupied voxels are innovation negatives.
    """
    gt = np.asarray(gt_occ, dtype=np.uint8)
    base = np.asarray(explained_occ, dtype=np.uint8)
    if gt.shape != base.shape or gt.ndim != 3:
        raise ValueError("gt/explained occupancy must share [H,W,Z] shape")

    positive_raw = _union_masks(
        category_masks,
        tuple(INNOVATION_POSITIVE_CATEGORIES),
        gt.shape,
    )
    excluded = _union_masks(
        category_masks,
        tuple(EXCLUDED_FROM_INNOVATION_SUPERVISION),
        gt.shape,
    )

    positive = positive_raw & (base == int(free_label))
    blocked_positive_voxels = int((positive_raw & ~positive).sum())

    excluded_bev = excluded.any(axis=2)
    positive_bev_raw = positive.any(axis=2)
    mixed_bev = excluded_bev & positive_bev_raw
    free_capacity = (base == int(free_label)).any(axis=2)

    candidate = free_capacity & ~excluded_bev
    add_target = positive_bev_raw & candidate

    positive = positive & add_target[..., None]
    vertical_target = positive.astype(np.uint8)
    semantic_target = np.full(gt.shape[:2], int(free_label), dtype=np.uint8)
    if bool(add_target.any()):
        rev = positive[:, :, ::-1]
        z_top = gt.shape[2] - 1 - np.argmax(rev, axis=2)
        ix, iy = np.nonzero(add_target)
        semantic_target[ix, iy] = gt[ix, iy, z_top[ix, iy]]
        if bool((semantic_target[ix, iy] == int(free_label)).any()):
            raise RuntimeError("positive innovation column received free semantic target")

    return {
        "add_target": add_target.astype(np.uint8),
        "semantic_target": semantic_target,
        "vertical_target": vertical_target,
        "candidate_mask": candidate.astype(np.uint8),
        "ignore_mask": (~candidate).astype(np.uint8),
        "positive_voxels_raw": int(positive_raw.sum()),
        "positive_voxels_actionable": int(positive.sum()),
        "blocked_positive_voxels": blocked_positive_voxels,
        "positive_bev_cells": int(add_target.sum()),
        "candidate_bev_cells": int(candidate.sum()),
        "mixed_responsibility_bev_cells": int(mixed_bev.sum()),
        "excluded_bev_cells": int(excluded_bev.sum()),
    }
