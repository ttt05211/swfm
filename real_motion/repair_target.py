"""Occupancy-space sparse repair targets for anchor-preserving Sparse WM training.

P0-F5 defines the learned future endpoint in semantic occupancy space first,
then encodes that endpoint with the frozen VAE. This avoids treating a hard
latent mask as if it were equivalent to a local semantic edit after a
convolutional encoder.
"""
from __future__ import annotations

from typing import Iterable
import numpy as np


def apply_dynamic_repair(
    anchor_occ: np.ndarray,
    proposal_occ: np.ndarray,
    write_support_bev: np.ndarray,
    *,
    dynamic_class_ids: Iterable[int],
    free_label: int = 17,
) -> np.ndarray:
    """Apply dynamic-only proposal semantics inside a causal BEV write support.

    ``anchor_occ`` and ``proposal_occ`` may be either one horizon [X,Y,Z] or a
    temporal stack [T,X,Y,Z]. ``write_support_bev`` must match the leading BEV
    dimensions ([X,Y] or [T,X,Y]). Outside support the output is bit-exact
    anchor occupancy. Inside support, anchor dynamic voxels are cleared and
    proposal dynamic voxels are inserted; static/non-dynamic anchor semantics
    are never copied from the proposal.
    """
    anchor = np.asarray(anchor_occ)
    proposal = np.asarray(proposal_occ)
    write = np.asarray(write_support_bev, dtype=bool)
    if anchor.shape != proposal.shape or anchor.ndim not in (3, 4):
        raise ValueError("anchor_occ and proposal_occ must match [X,Y,Z] or [T,X,Y,Z]")
    if write.shape != anchor.shape[:-1]:
        raise ValueError("write_support_bev must match anchor leading BEV dimensions")

    dynamic = np.asarray(tuple(int(c) for c in dynamic_class_ids), dtype=np.int64)
    if dynamic.size == 0:
        raise ValueError("dynamic_class_ids must be non-empty")

    out = anchor.copy()
    write3d = write[..., None]
    anchor_dynamic = np.isin(anchor, dynamic)
    proposal_dynamic = np.isin(proposal, dynamic)
    out[write3d & anchor_dynamic] = int(free_label)
    out[write3d & proposal_dynamic] = proposal[write3d & proposal_dynamic]

    outside = ~write3d
    if not np.array_equal(out[outside], anchor[outside]):
        raise AssertionError("dynamic repair modified occupancy outside causal support")
    return out


def build_dynamic_repair_endpoint(
    anchor_occ: np.ndarray,
    gt_occ: np.ndarray,
    write_support_bev: np.ndarray,
    *,
    dynamic_class_ids: Iterable[int],
    free_label: int = 17,
) -> np.ndarray:
    """Return the P0-F5 occupancy-space training endpoint.

    Future GT is used only as the dynamic proposal inside the already-causal MSP
    support. The resulting endpoint is exactly Strong W2Det outside support and
    is then encoded by the frozen VAE to define the flow target.
    """
    return apply_dynamic_repair(
        anchor_occ,
        gt_occ,
        write_support_bev,
        dynamic_class_ids=dynamic_class_ids,
        free_label=free_label,
    )
