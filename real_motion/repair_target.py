"""Occupancy-space sparse repair targets for anchor-preserving Sparse WM training.

P0-F5 defines the learned future endpoint in semantic occupancy space first,
then encodes that endpoint with the frozen VAE. This avoids treating a hard
latent mask as if it were equivalent to a local semantic edit after a
convolutional encoder.
"""
from __future__ import annotations

from typing import Iterable
import numpy as np


def build_dynamic_repair_endpoint(
    anchor_occ: np.ndarray,
    gt_occ: np.ndarray,
    write_support_bev: np.ndarray,
    *,
    dynamic_class_ids: Iterable[int],
    free_label: int = 17,
) -> np.ndarray:
    """Return an anchor-preserving semantic repair endpoint.

    Args:
        anchor_occ: [T,X,Y,Z] strong causal future anchor semantics.
        gt_occ: [T,X,Y,Z] future GT semantics, used only to construct training
            supervision / oracle endpoint.
        write_support_bev: [T,X,Y] causal MSP support in future ego frames.
        dynamic_class_ids: semantic classes whose occupancy may be repaired.
        free_label: semantic free-space label.

    Contract:
        - outside write support, output is bit-exact anchor occupancy;
        - inside support, anchor dynamic voxels are cleared;
        - inside support, GT dynamic voxels are inserted;
        - static / non-dynamic anchor semantics are never replaced by GT.

    This is the exact deployment semantics used by the same-support GT repair
    oracle, but only the encoded endpoint enters Sparse-WM training.
    """
    anchor = np.asarray(anchor_occ)
    gt = np.asarray(gt_occ)
    write = np.asarray(write_support_bev, dtype=bool)
    if anchor.shape != gt.shape or anchor.ndim != 4:
        raise ValueError("anchor_occ and gt_occ must match [T,X,Y,Z]")
    if write.shape != anchor.shape[:3]:
        raise ValueError("write_support_bev must match anchor [T,X,Y]")

    dynamic = np.asarray(tuple(int(c) for c in dynamic_class_ids), dtype=np.int64)
    if dynamic.size == 0:
        raise ValueError("dynamic_class_ids must be non-empty")

    out = anchor.copy()
    write3d = write[..., None]
    anchor_dynamic = np.isin(anchor, dynamic)
    gt_dynamic = np.isin(gt, dynamic)

    out[write3d & anchor_dynamic] = int(free_label)
    out[write3d & gt_dynamic] = gt[write3d & gt_dynamic]

    # Strong preservation assertion: causal support is the only semantic write
    # authority. This is cheap compared with cache construction and catches any
    # accidental GT leakage immediately.
    outside = ~write3d
    if not np.array_equal(out[outside], anchor[outside]):
        raise AssertionError("repair endpoint modified occupancy outside causal support")
    return out
