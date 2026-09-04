"""Occupancy-space sparse repair targets and deployment fusion operators.

P0-F5 defines the learned future endpoint in semantic occupancy space first,
then encodes that endpoint with the frozen VAE. This avoids treating a hard
latent mask as if it were equivalent to a local semantic edit after a
convolutional encoder.

P0-F9 failure analysis also uses two explicitly non-destructive fusion rules to
test whether learned motion evidence is useful when it is not allowed to erase
Strong-W2Det dynamic predictions.
"""
from __future__ import annotations

from typing import Iterable
import numpy as np


def _fusion_inputs(
    anchor_occ: np.ndarray,
    proposal_occ: np.ndarray,
    write_support_bev: np.ndarray,
    dynamic_class_ids: Iterable[int],
):
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
    return anchor, proposal, write, dynamic


def _assert_outside_support_unchanged(out: np.ndarray, anchor: np.ndarray, write: np.ndarray) -> None:
    # Boolean indexing with [T,X,Y] (or [X,Y]) preserves the final Z dimension.
    outside = ~write
    if not np.array_equal(out[outside], anchor[outside]):
        raise AssertionError("dynamic fusion modified occupancy outside causal support")


def apply_dynamic_repair(
    anchor_occ: np.ndarray,
    proposal_occ: np.ndarray,
    write_support_bev: np.ndarray,
    *,
    dynamic_class_ids: Iterable[int],
    free_label: int = 17,
) -> np.ndarray:
    """Destructive takeover fusion used by the frozen P0-F4/P0-F5 oracle.

    Contract:
    - outside support, output is bit-exact anchor occupancy;
    - inside support, anchor dynamic voxels are first cleared;
    - proposal dynamic voxels are then written and may overwrite the anchor
      semantic at the same voxel;
    - proposal non-dynamic semantics are never copied into the result.

    This function is intentionally kept unchanged because existing P0-F5..P0-F9
    cache/oracle provenance depends on its exact semantics.
    """
    anchor, proposal, write, dynamic = _fusion_inputs(
        anchor_occ, proposal_occ, write_support_bev, dynamic_class_ids
    )
    out = anchor.copy()
    write3d = write[..., None]
    anchor_dynamic = np.isin(anchor, dynamic)
    proposal_dynamic = np.isin(proposal, dynamic)
    out[write3d & anchor_dynamic] = int(free_label)
    out[write3d & proposal_dynamic] = proposal[write3d & proposal_dynamic]
    _assert_outside_support_unchanged(out, anchor, write)
    return out


def apply_dynamic_write_only(
    anchor_occ: np.ndarray,
    proposal_occ: np.ndarray,
    write_support_bev: np.ndarray,
    *,
    dynamic_class_ids: Iterable[int],
) -> np.ndarray:
    """Conservative innovation fusion: add only *new* proposal dynamics.

    Inside the causal support, a proposal dynamic class may overwrite an anchor
    voxel only when the anchor voxel is currently non-dynamic. Existing anchor
    dynamic voxels, including their semantic class, are preserved bit-exactly.
    Proposal background/non-dynamic semantics are ignored.

    This tests the strongest anchor-preservation hypothesis: the learned model is
    allowed to contribute missing dynamic evidence but has no authority to erase
    or relabel a Strong-W2Det dynamic prediction.
    """
    anchor, proposal, write, dynamic = _fusion_inputs(
        anchor_occ, proposal_occ, write_support_bev, dynamic_class_ids
    )
    out = anchor.copy()
    write3d = write[..., None]
    anchor_dynamic = np.isin(anchor, dynamic)
    proposal_dynamic = np.isin(proposal, dynamic)
    add = write3d & (~anchor_dynamic) & proposal_dynamic
    out[add] = proposal[add]

    if not np.array_equal(out[write3d & anchor_dynamic], anchor[write3d & anchor_dynamic]):
        raise AssertionError("write-only fusion changed an existing anchor dynamic voxel")
    _assert_outside_support_unchanged(out, anchor, write)
    return out


def apply_dynamic_union(
    anchor_occ: np.ndarray,
    proposal_occ: np.ndarray,
    write_support_bev: np.ndarray,
    *,
    dynamic_class_ids: Iterable[int],
) -> np.ndarray:
    """Non-destructive dynamic union with proposal priority on dynamic classes.

    Inside support, every proposal dynamic voxel is written. Therefore a proposal
    may add a missing dynamic voxel or relabel an existing dynamic voxel. However,
    proposal background/non-dynamic semantics never clear an anchor dynamic
    prediction. Dynamic *presence* from the Strong anchor is thus monotonic.

    This is less conservative than ``apply_dynamic_write_only`` while still
    removing the destructive CLEAR authority that dominated P0-F9 Moving loss.
    """
    anchor, proposal, write, dynamic = _fusion_inputs(
        anchor_occ, proposal_occ, write_support_bev, dynamic_class_ids
    )
    out = anchor.copy()
    write3d = write[..., None]
    anchor_dynamic = np.isin(anchor, dynamic)
    proposal_dynamic = np.isin(proposal, dynamic)
    write_dynamic = write3d & proposal_dynamic
    out[write_dynamic] = proposal[write_dynamic]

    out_dynamic = np.isin(out, dynamic)
    if bool((write3d & anchor_dynamic & (~out_dynamic)).any()):
        raise AssertionError("dynamic-union fusion erased anchor dynamic presence")
    _assert_outside_support_unchanged(out, anchor, write)
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
