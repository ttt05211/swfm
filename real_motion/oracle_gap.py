"""Controlled occupancy-space oracle interventions for P0-F9 gap attribution.

These helpers do not train a model.  They modify a decoded dynamic proposal only
inside an explicit BEV support so CLEAR/KEEP, WRITE and semantic-label errors can
be corrected one family at a time.  A separate GT edit-support helper expands
support only where Strong-W2Det and GT dynamic occupancy disagree, allowing the
routing/support ceiling to be measured with a GT proposal on both sides.
"""
from __future__ import annotations

from typing import Iterable
import numpy as np


def _inputs(anchor, proposal, gt, support_bev, dynamic_class_ids):
    a = np.asarray(anchor)
    p = np.asarray(proposal)
    g = np.asarray(gt)
    s = np.asarray(support_bev, dtype=bool)
    if a.shape != p.shape or a.shape != g.shape or a.ndim not in (3, 4):
        raise ValueError("anchor/proposal/gt must match [X,Y,Z] or [T,X,Y,Z]")
    if s.shape != a.shape[:-1]:
        raise ValueError("support_bev must match occupancy leading BEV dimensions")
    dyn = np.asarray(tuple(int(x) for x in dynamic_class_ids), dtype=np.int64)
    if dyn.size == 0:
        raise ValueError("dynamic_class_ids must be non-empty")
    s3 = np.broadcast_to(s[..., None], a.shape)
    return a, p, g, s, s3, dyn


def oracle_clear_keep_presence(
    anchor, proposal, gt, support_bev, *, dynamic_class_ids: Iterable[int], free_label: int = 17
):
    """Perfect anchor-dynamic departure/persistence decisions, not semantics.

    Required CLEAR voxels are made non-dynamic.  Required KEEP voxels that the
    proposal wrongly cleared are restored with the anchor dynamic class, so this
    oracle fixes presence without granting the GT class on already occupied
    source voxels.
    """
    a, p, g, _, s3, dyn = _inputs(anchor, proposal, gt, support_bev, dynamic_class_ids)
    out = p.copy()
    a_dyn = np.isin(a, dyn)
    p_dyn = np.isin(p, dyn)
    g_dyn = np.isin(g, dyn)
    req_clear = s3 & a_dyn & ~g_dyn
    req_keep_missing = s3 & a_dyn & g_dyn & ~p_dyn
    out[req_clear] = int(free_label)
    out[req_keep_missing] = a[req_keep_missing]
    return out


def oracle_write_presence(
    anchor, proposal, gt, support_bev, *, dynamic_class_ids: Iterable[int], free_label: int = 17
):
    """Perfect WRITE/no-WRITE decisions on anchor-non-dynamic voxels.

    Required future dynamics receive the GT dynamic class; false writes on
    stable non-dynamic voxels are removed.  CLEAR/KEEP behavior on anchor-dynamic
    voxels is left untouched.
    """
    a, p, g, _, s3, dyn = _inputs(anchor, proposal, gt, support_bev, dynamic_class_ids)
    out = p.copy()
    a_dyn = np.isin(a, dyn)
    g_dyn = np.isin(g, dyn)
    req_write = s3 & ~a_dyn & g_dyn
    stable_non = s3 & ~a_dyn & ~g_dyn
    out[req_write] = g[req_write]
    out[stable_non] = int(free_label)
    return out


def oracle_event_presence(
    anchor, proposal, gt, support_bev, *, dynamic_class_ids: Iterable[int], free_label: int = 17
):
    """Joint perfect CLEAR/KEEP presence plus WRITE decisions inside support."""
    out = oracle_clear_keep_presence(
        anchor,
        proposal,
        gt,
        support_bev,
        dynamic_class_ids=dynamic_class_ids,
        free_label=free_label,
    )
    return oracle_write_presence(
        anchor,
        out,
        gt,
        support_bev,
        dynamic_class_ids=dynamic_class_ids,
        free_label=free_label,
    )


def oracle_dynamic_semantics(anchor, proposal, gt, support_bev, *, dynamic_class_ids: Iterable[int]):
    """Correct dynamic class only where proposal and GT already agree on presence."""
    _, p, g, _, s3, dyn = _inputs(anchor, proposal, gt, support_bev, dynamic_class_ids)
    out = p.copy()
    p_dyn = np.isin(p, dyn)
    g_dyn = np.isin(g, dyn)
    same_presence = s3 & p_dyn & g_dyn
    out[same_presence] = g[same_presence]
    return out


def gt_edit_support_bev(anchor, gt, current_support_bev, *, dynamic_class_ids: Iterable[int]):
    """Expand support to the exact BEV cells requiring any dynamic edit.

    The expansion covers dynamic appearance/disappearance and dynamic-class
    relabeling.  Current support is always retained, so comparison of GT proposal
    under current vs expanded support isolates support/routing headroom.
    """
    a = np.asarray(anchor)
    g = np.asarray(gt)
    s = np.asarray(current_support_bev, dtype=bool)
    if a.shape != g.shape or a.ndim not in (3, 4) or s.shape != a.shape[:-1]:
        raise ValueError("shape mismatch for GT edit support")
    dyn = np.asarray(tuple(int(x) for x in dynamic_class_ids), dtype=np.int64)
    a_dyn = np.isin(a, dyn)
    g_dyn = np.isin(g, dyn)
    need = (a_dyn != g_dyn) | (a_dyn & g_dyn & (a != g))
    return s | need.any(axis=-1)


def support_coverage_counts(anchor, gt, support_bev, *, dynamic_class_ids: Iterable[int]):
    """Count how much GT-required dynamic editing is covered by current support."""
    a = np.asarray(anchor)
    g = np.asarray(gt)
    s = np.asarray(support_bev, dtype=bool)
    if a.shape != g.shape or a.ndim not in (3, 4) or s.shape != a.shape[:-1]:
        raise ValueError("shape mismatch for support coverage")
    dyn = np.asarray(tuple(int(x) for x in dynamic_class_ids), dtype=np.int64)
    s3 = np.broadcast_to(s[..., None], a.shape)
    a_dyn = np.isin(a, dyn)
    g_dyn = np.isin(g, dyn)
    clear = a_dyn & ~g_dyn
    write = ~a_dyn & g_dyn
    relabel = a_dyn & g_dyn & (a != g)
    event = clear | write | relabel

    def pair(mask):
        total = int(mask.sum())
        covered = int((mask & s3).sum())
        return {"total": total, "covered": covered}

    return {
        "clear": pair(clear),
        "write": pair(write),
        "relabel": pair(relabel),
        "event": pair(event),
        "support_bev_cells": int(s.sum()),
        "total_bev_cells": int(s.size),
    }
