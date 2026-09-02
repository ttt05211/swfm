"""P0-F8 v2 training alignment patch.

The original P0-F8 implementation is retained as the stable training/runtime
skeleton. This module replaces only the edit-supervision population logic:

- balanced CE uses all EDIT plus a 50/50 split of dynamic/background KEEP;
- Lovasz uses the complete compact sidecar pool instead of the artificial 1:1
  CE sample;
- reported false-edit rate is measured on the complete KEEP pool.

All model, cache, optimizer, AMP, checkpoint, resume, and validation mechanics
remain inherited from ``p0_f8_train_impl``.
"""
from __future__ import annotations

import json

import torch

from real_motion.edit_repair import EditTargetCache, horizon_from_flat_indices
from real_motion.edit_repair_v2 import (
    DEFAULT_DYNAMIC_KEEP_FRACTION,
    full_edit_supervision,
    select_stratified_balanced_edit_supervision,
    split_population_edit_loss,
)
from real_motion.occfm_io import OccFMVAEAdapter
from tools.real_motion import p0_f8_train_impl as base

F8_PROTOCOL = "p0_f8_anchor_relative_edit_wm_v2"

# Keep a handle to the unpatched architecture helper so repeated imports/tests
# cannot accidentally wrap the wrapper.
_BASE_ARCHITECTURE = base._architecture


def edit_loss_for_endpoint_v2(
    model,
    endpoint_full: torch.Tensor,
    *,
    sample_ids,
    edit_cache: EditTargetCache,
    vae: OccFMVAEAdapter,
    keep_ratio: float,
    keep_when_no_edit: int,
    lovasz_weight: float,
    deterministic: bool,
    selection_seed: int,
) -> tuple[torch.Tensor, dict]:
    records = edit_cache.get_batch(sample_ids)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(selection_seed))

    pools = [full_edit_supervision(rec) for rec in records]
    balanced = [
        select_stratified_balanced_edit_supervision(
            rec,
            keep_ratio=float(keep_ratio),
            dynamic_keep_fraction=DEFAULT_DYNAMIC_KEEP_FRACTION,
            generator=generator,
            deterministic=bool(deterministic),
            keep_when_no_edit=int(keep_when_no_edit),
        )
        for rec in records
    ]
    indices = [item["flat_indices"].to(endpoint_full.device) for item in pools]
    sparse_semantic_logits = vae.decode_logits_at_flat_indices(endpoint_full, indices)

    pool_action_rows = []
    pool_anchor_rows = []
    pool_result_rows = []
    pool_target_rows = []
    ce_position_rows = []
    ce_action_rows = []
    total_edits = 0
    total_keeps = 0
    total_dynamic_keeps = 0
    total_background_keeps = 0
    total_moving_edits = 0
    total_lovasz_voxels = 0
    pool_offset = 0

    for logits, pool, sel in zip(sparse_semantic_logits, pools, balanced):
        idx = pool["flat_indices"]
        if idx.numel() == 0:
            continue
        horizons = horizon_from_flat_indices(idx).to(endpoint_full.device)
        anchor_slots = pool["anchor_slots"].to(endpoint_full.device)
        action_logits = model.edit_head(logits, anchor_slots, horizons)

        pool_action_rows.append(action_logits)
        pool_anchor_rows.append(anchor_slots)
        pool_result_rows.append(pool["result_slots"].to(endpoint_full.device))
        pool_target_rows.append(pool["actions"].to(endpoint_full.device))

        ce_position_rows.append(
            sel["pool_positions"].to(endpoint_full.device) + int(pool_offset)
        )
        ce_action_rows.append(sel["actions"].to(endpoint_full.device))
        pool_offset += int(action_logits.shape[0])

        total_edits += int(sel["num_edits"])
        total_keeps += int(sel["num_keeps"])
        total_dynamic_keeps += int(sel["num_dynamic_keeps"])
        total_background_keeps += int(sel["num_background_keeps"])
        total_moving_edits += int(sel["is_moving_edit"].sum().item())
        total_lovasz_voxels += int(pool["flat_indices"].numel())

    if not pool_action_rows or not ce_position_rows:
        zero = endpoint_full.sum() * 0.0
        return zero, {
            "num_supervised_voxels": 0,
            "num_lovasz_voxels": 0,
            "num_edits": 0,
            "num_keeps": 0,
            "num_dynamic_keeps": 0,
            "num_background_keeps": 0,
            "num_moving_edits": 0,
            "ce": 0.0,
            "lovasz": 0.0,
            "accuracy": float("nan"),
            "edit_accuracy": float("nan"),
            "false_edit_rate": float("nan"),
            "balanced_false_edit_rate": float("nan"),
            "pool_false_edit_rate": float("nan"),
        }

    pool_action_logits = torch.cat(pool_action_rows, dim=0)
    pool_anchor_slots = torch.cat(pool_anchor_rows, dim=0)
    pool_result_slots = torch.cat(pool_result_rows, dim=0)
    pool_actions = torch.cat(pool_target_rows, dim=0)
    ce_pool_positions = torch.cat(ce_position_rows, dim=0)
    ce_actions = torch.cat(ce_action_rows, dim=0)

    loss, info = split_population_edit_loss(
        pool_action_logits,
        pool_anchor_slots,
        pool_result_slots,
        pool_actions,
        ce_pool_positions,
        ce_actions,
        lovasz_weight=float(lovasz_weight),
    )
    num_ce = int(ce_actions.numel())
    info.update({
        "num_supervised_voxels": num_ce,
        "num_lovasz_voxels": int(total_lovasz_voxels),
        "num_edits": int(total_edits),
        "num_keeps": int(total_keeps),
        "num_dynamic_keeps": int(total_dynamic_keeps),
        "num_background_keeps": int(total_background_keeps),
        "num_moving_edits": int(total_moving_edits),
        "edit_fraction": float(total_edits / max(num_ce, 1)),
        "moving_edit_fraction": float(total_moving_edits / max(total_edits, 1)),
        "dynamic_keep_fraction_realized": float(
            total_dynamic_keeps / max(total_keeps, 1)
        ),
    })
    return loss, info


def _architecture_v2(args, edit_lambda, optimizer_summary, train_ds):
    arch = _BASE_ARCHITECTURE(args, edit_lambda, optimizer_summary, train_ds)
    arch["protocol"] = F8_PROTOCOL
    sampling = dict(arch.get("edit_sampling") or {})
    sampling.update({
        "keep_stratification": "dynamic_keep_and_background_keep",
        "dynamic_keep_fraction": DEFAULT_DYNAMIC_KEEP_FRACTION,
        "background_keep_fraction": 1.0 - DEFAULT_DYNAMIC_KEEP_FRACTION,
        "shortfall_policy": "fill_from_other_keep_stratum",
        "ce_population": "all_EDIT_plus_stratified_balanced_KEEP",
    })
    arch["edit_sampling"] = sampling
    arch["lovasz_population"] = (
        "complete_compact_sidecar_pool: all_EDIT + all_dynamic_KEEP + bounded_background_KEEP"
    )
    arch["false_edit_metric"] = "complete_compact_KEEP_pool"
    arch["edit_loss"] = (
        "stratified_balanced_action_CE_plus_full_sidecar_result_semantic_Lovasz"
    )
    return arch


def _install_v2_patch() -> None:
    # ``base.main`` resolves these names from its own module globals at runtime,
    # so replacing them here changes only the intended P0-F8 supervision logic.
    base.F8_PROTOCOL = F8_PROTOCOL
    base.edit_loss_for_endpoint = edit_loss_for_endpoint_v2
    base._architecture = _architecture_v2


def main():
    _install_v2_patch()
    print(json.dumps({
        "p0_f8_protocol": F8_PROTOCOL,
        "balanced_ce": "all_EDIT + 50% dynamic KEEP + 50% background KEEP",
        "lovasz_population": "complete compact sidecar pool",
        "false_edit_metric": "complete KEEP pool",
    }))
    base.main()


__all__ = [
    "F8_PROTOCOL",
    "edit_loss_for_endpoint_v2",
    "main",
]
