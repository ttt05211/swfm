"""P0-F8 v2 edit supervision aligned with deployment statistics.

This module fixes two training/deployment mismatches in the first P0-F8
implementation without changing the deployed KEEP/CLEAR/WRITE action space:

1. balanced CE still uses KEEP:EDIT control, but the KEEP budget is explicitly
   split between hard dynamic KEEP and background KEEP so background false
   writes cannot disappear from training;
2. Lovasz is computed on the complete compact edit sidecar pool (all EDIT,
   every dynamic KEEP, and the bounded background KEEP pool) rather than on the
   artificial 1:1 CE sample.

The existing edit sidecar format is reused; no expensive WM latent cache rebuild
is required.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .edit_repair import (
    KEEP,
    action_probs_to_result_probs,
    lovasz_softmax_flat,
    validate_edit_record,
)

DEFAULT_DYNAMIC_KEEP_FRACTION = 0.5


def _pick_positions(
    positions: torch.Tensor,
    priority: torch.Tensor,
    count: int,
    *,
    generator: torch.Generator | None,
    deterministic: bool,
) -> torch.Tensor:
    """Pick positions with descending hard-negative priority."""
    positions = positions.long().cpu().reshape(-1)
    priority = priority.long().cpu().reshape(-1)
    if positions.numel() != priority.numel():
        raise ValueError("positions/priority length mismatch")
    if count <= 0 or positions.numel() == 0:
        return torch.empty(0, dtype=torch.long)

    remaining = min(int(count), int(positions.numel()))
    selected = []
    # Current v1 sidecars use priority 2=true-moving dynamic, 1=other dynamic,
    # 0=background. Keep the helper future-proof for any larger priority value.
    levels = sorted({int(x) for x in priority.tolist()}, reverse=True)
    for level in levels:
        local = torch.nonzero(priority == int(level), as_tuple=False).flatten()
        if local.numel() == 0:
            continue
        take = min(remaining, int(local.numel()))
        if deterministic or take == int(local.numel()):
            chosen_local = local[:take]
        else:
            order = torch.randperm(int(local.numel()), generator=generator)[:take]
            chosen_local = local.index_select(0, order)
        selected.append(positions.index_select(0, chosen_local))
        remaining -= take
        if remaining <= 0:
            break

    if not selected:
        return torch.empty(0, dtype=torch.long)
    return torch.cat(selected, dim=0)


def _remaining_positions(all_positions: torch.Tensor, picked: torch.Tensor) -> torch.Tensor:
    all_positions = all_positions.long().cpu().reshape(-1)
    picked = picked.long().cpu().reshape(-1)
    if picked.numel() == 0:
        return all_positions
    keep = torch.ones(int(all_positions.numel()), dtype=torch.bool)
    # Positions are indices into the KEEP pool, so a direct boolean lookup is
    # safe and avoids O(N^2) membership checks.
    max_pos = int(all_positions.max()) if all_positions.numel() else -1
    mark = torch.zeros(max_pos + 1, dtype=torch.bool)
    mark[picked] = True
    keep = ~mark.index_select(0, all_positions)
    return all_positions[keep]


def full_edit_supervision(record: dict) -> dict:
    """Return the complete compact sidecar pool in a stable EDIT-then-KEEP order."""
    validate_edit_record(record)
    eidx = record["edit_flat_indices"].long()
    ea = record["edit_anchor_slots"].long()
    er = record["edit_result_slots"].long()
    eact = record["edit_actions"].long()
    emov = record["edit_moving"].bool()
    kidx = record["keep_flat_indices"].long()
    ka = record["keep_anchor_slots"].long()
    nk = int(kidx.numel())
    ne = int(eidx.numel())
    return {
        "flat_indices": torch.cat([eidx, kidx], dim=0),
        "actions": torch.cat([
            eact,
            torch.full((nk,), KEEP, dtype=torch.long),
        ], dim=0),
        "anchor_slots": torch.cat([ea, ka], dim=0),
        "result_slots": torch.cat([er, ka], dim=0),
        "is_edit": torch.cat([
            torch.ones(ne, dtype=torch.bool),
            torch.zeros(nk, dtype=torch.bool),
        ], dim=0),
        "is_moving_edit": torch.cat([
            emov,
            torch.zeros(nk, dtype=torch.bool),
        ], dim=0),
        "num_edits": ne,
        "num_keeps": nk,
        "num_dynamic_keeps": int((ka > 0).sum().item()),
        "num_background_keeps": int((ka == 0).sum().item()),
    }


def select_stratified_balanced_edit_supervision(
    record: dict,
    *,
    keep_ratio: float = 1.0,
    dynamic_keep_fraction: float = DEFAULT_DYNAMIC_KEEP_FRACTION,
    generator: torch.Generator | None = None,
    deterministic: bool = False,
    keep_when_no_edit: int = 64,
) -> dict:
    """Keep every EDIT while forcing both dynamic and background KEEP exposure.

    ``dynamic_keep_fraction`` applies *inside* the bounded KEEP budget. With the
    frozen default 0.5 and KEEP:EDIT=1:1, half of sampled KEEP voxels are drawn
    from correct dynamic anchors and half from background/non-dynamic anchors
    whenever both strata have enough candidates. Any shortfall is filled from
    the other stratum so the requested total KEEP budget is preserved.
    """
    validate_edit_record(record)
    if keep_ratio < 0:
        raise ValueError("keep_ratio must be non-negative")
    if not 0.0 <= float(dynamic_keep_fraction) <= 1.0:
        raise ValueError("dynamic_keep_fraction must be in [0,1]")
    if keep_when_no_edit < 0:
        raise ValueError("keep_when_no_edit must be non-negative")

    ne = int(record["edit_flat_indices"].numel())
    nk_total = int(record["keep_flat_indices"].numel())
    target_keep = int(math.ceil(float(ne) * float(keep_ratio)))
    if ne == 0:
        target_keep = int(keep_when_no_edit)
    target_keep = min(target_keep, nk_total)

    k_anchor = record["keep_anchor_slots"].long().cpu()
    k_priority = record["keep_priority"].long().cpu()
    all_keep_pos = torch.arange(nk_total, dtype=torch.long)
    dynamic_pos = all_keep_pos[k_anchor > 0]
    background_pos = all_keep_pos[k_anchor == 0]

    desired_dynamic = int(math.floor(target_keep * float(dynamic_keep_fraction) + 0.5))
    desired_dynamic = min(desired_dynamic, target_keep)
    desired_background = target_keep - desired_dynamic

    dyn_pick = _pick_positions(
        dynamic_pos,
        k_priority.index_select(0, dynamic_pos) if dynamic_pos.numel() else k_priority[:0],
        desired_dynamic,
        generator=generator,
        deterministic=deterministic,
    )
    bg_pick = _pick_positions(
        background_pos,
        k_priority.index_select(0, background_pos) if background_pos.numel() else k_priority[:0],
        desired_background,
        generator=generator,
        deterministic=deterministic,
    )

    picked = torch.cat([dyn_pick, bg_pick], dim=0)
    shortage = target_keep - int(picked.numel())
    if shortage > 0:
        remain = _remaining_positions(all_keep_pos, picked)
        fill = _pick_positions(
            remain,
            k_priority.index_select(0, remain) if remain.numel() else k_priority[:0],
            shortage,
            generator=generator,
            deterministic=deterministic,
        )
        picked = torch.cat([picked, fill], dim=0)

    if int(picked.numel()) != target_keep:
        raise RuntimeError(
            f"stratified KEEP sampler produced {picked.numel()} != target {target_keep}"
        )
    if torch.unique(picked).numel() != picked.numel():
        raise RuntimeError("stratified KEEP sampler produced duplicate positions")

    eidx = record["edit_flat_indices"].long()
    ea = record["edit_anchor_slots"].long()
    er = record["edit_result_slots"].long()
    eact = record["edit_actions"].long()
    emov = record["edit_moving"].bool()
    kidx = record["keep_flat_indices"].long().index_select(0, picked)
    ka = record["keep_anchor_slots"].long().index_select(0, picked)
    pool_positions = torch.cat([
        torch.arange(ne, dtype=torch.long),
        ne + picked,
    ], dim=0)

    return {
        "flat_indices": torch.cat([eidx, kidx], dim=0),
        "actions": torch.cat([
            eact,
            torch.full((target_keep,), KEEP, dtype=torch.long),
        ], dim=0),
        "anchor_slots": torch.cat([ea, ka], dim=0),
        "result_slots": torch.cat([er, ka], dim=0),
        "is_edit": torch.cat([
            torch.ones(ne, dtype=torch.bool),
            torch.zeros(target_keep, dtype=torch.bool),
        ], dim=0),
        "is_moving_edit": torch.cat([
            emov,
            torch.zeros(target_keep, dtype=torch.bool),
        ], dim=0),
        "pool_positions": pool_positions,
        "num_edits": ne,
        "num_keeps": target_keep,
        "num_dynamic_keeps": int((ka > 0).sum().item()),
        "num_background_keeps": int((ka == 0).sum().item()),
        "dynamic_keep_fraction": float(dynamic_keep_fraction),
    }


def split_population_edit_loss(
    pool_action_logits: torch.Tensor,
    pool_anchor_slots: torch.Tensor,
    pool_result_slots: torch.Tensor,
    pool_actions: torch.Tensor,
    ce_pool_positions: torch.Tensor,
    ce_actions: torch.Tensor,
    *,
    lovasz_weight: float = 0.5,
) -> tuple[torch.Tensor, dict]:
    """Balanced action CE + deployment-like full-pool Lovasz.

    The CE population is explicitly re-sampled for class balance. Lovasz sees
    the complete compact sidecar population instead, so the IoU surrogate is
    not optimized on the artificial 1:1 class distribution.
    """
    if lovasz_weight < 0:
        raise ValueError("lovasz_weight must be non-negative")
    if pool_action_logits.ndim != 2:
        raise ValueError("pool_action_logits must be [N,A]")
    device = pool_action_logits.device
    pool_anchor_slots = pool_anchor_slots.to(device=device, dtype=torch.long).reshape(-1)
    pool_result_slots = pool_result_slots.to(device=device, dtype=torch.long).reshape(-1)
    pool_actions = pool_actions.to(device=device, dtype=torch.long).reshape(-1)
    ce_pool_positions = ce_pool_positions.to(device=device, dtype=torch.long).reshape(-1)
    ce_actions = ce_actions.to(device=device, dtype=torch.long).reshape(-1)

    n = int(pool_action_logits.shape[0])
    if not (pool_anchor_slots.numel() == pool_result_slots.numel() == pool_actions.numel() == n):
        raise ValueError("full-pool target lengths must match action logits")
    if ce_pool_positions.numel() != ce_actions.numel():
        raise ValueError("balanced CE positions/actions length mismatch")
    if ce_pool_positions.numel() == 0:
        zero = pool_action_logits.sum() * 0.0
        return zero, {
            "ce": 0.0,
            "lovasz": 0.0,
            "accuracy": float("nan"),
            "edit_accuracy": float("nan"),
            "false_edit_rate": float("nan"),
            "balanced_false_edit_rate": float("nan"),
            "pool_false_edit_rate": float("nan"),
        }
    if int(ce_pool_positions.min()) < 0 or int(ce_pool_positions.max()) >= n:
        raise ValueError("balanced CE pool position out of range")

    ce_logits = pool_action_logits.index_select(0, ce_pool_positions)
    ce = F.cross_entropy(ce_logits.float(), ce_actions, reduction="mean")
    result_probs = action_probs_to_result_probs(pool_action_logits, pool_anchor_slots)
    lovasz = lovasz_softmax_flat(result_probs, pool_result_slots)
    loss = ce + float(lovasz_weight) * lovasz

    with torch.no_grad():
        ce_pred = ce_logits.argmax(dim=-1)
        ce_edit = ce_actions != KEEP
        ce_keep = ~ce_edit
        accuracy = (ce_pred == ce_actions).float().mean()
        edit_accuracy = (
            (ce_pred[ce_edit] == ce_actions[ce_edit]).float().mean()
            if bool(ce_edit.any()) else torch.tensor(float("nan"), device=device)
        )
        balanced_false = (
            (ce_pred[ce_keep] != KEEP).float().mean()
            if bool(ce_keep.any()) else torch.tensor(float("nan"), device=device)
        )
        pool_pred = pool_action_logits.argmax(dim=-1)
        pool_keep = pool_actions == KEEP
        pool_false = (
            (pool_pred[pool_keep] != KEEP).float().mean()
            if bool(pool_keep.any()) else torch.tensor(float("nan"), device=device)
        )

    return loss, {
        "ce": float(ce.detach().cpu()),
        "lovasz": float(lovasz.detach().cpu()),
        "accuracy": float(accuracy.detach().cpu()),
        "edit_accuracy": float(edit_accuracy.detach().cpu()),
        # Preserve the historical key, but make it deployment-like by measuring
        # every KEEP in the compact pool rather than only the balanced CE subset.
        "false_edit_rate": float(pool_false.detach().cpu()),
        "balanced_false_edit_rate": float(balanced_false.detach().cpu()),
        "pool_false_edit_rate": float(pool_false.detach().cpu()),
    }
