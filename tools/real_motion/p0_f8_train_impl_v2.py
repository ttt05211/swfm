"""P0-F8 v2 training alignment patch.

The original P0-F8 implementation is retained as the stable training/runtime
skeleton. This module replaces the edit-supervision population logic and the
validation aggregation that depends on those populations:

- balanced CE uses all EDIT plus a 50/50 split of dynamic/background KEEP;
- Lovasz uses the complete compact sidecar pool instead of the artificial 1:1
  CE sample;
- reported false-edit rate is measured on the complete compact KEEP pool;
- validation aggregates CE, Lovasz, and false-edit with their own population
  sizes, so checkpoint selection is no longer biased by sampled CE counts;
- v2-only sampling/population statistics are persisted into training history.

Model, cache, optimizer, AMP, checkpoint format, resume rules, and deployment
KEEP/CLEAR/WRITE semantics remain inherited from ``p0_f8_train_impl``.
"""
from __future__ import annotations

import json
import math

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

# Keep handles to the unpatched helpers so repeated patch installation cannot
# wrap an already wrapped function.
_BASE_ARCHITECTURE = base._architecture
_BASE_PAYLOAD = base._payload

_LAST_TRAIN_V2_INFO: dict = {}


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
    total_pool_keeps = 0
    total_pool_dynamic_keeps = 0
    total_pool_background_keeps = 0
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
        total_pool_keeps += int(pool["num_keeps"])
        total_pool_dynamic_keeps += int(pool["num_dynamic_keeps"])
        total_pool_background_keeps += int(pool["num_background_keeps"])
        total_moving_edits += int(sel["is_moving_edit"].sum().item())
        total_lovasz_voxels += int(pool["flat_indices"].numel())

    if not pool_action_rows or not ce_position_rows:
        zero = endpoint_full.sum() * 0.0
        info = {
            "num_supervised_voxels": 0,
            "num_lovasz_voxels": 0,
            "num_edits": 0,
            "num_keeps": 0,
            "num_dynamic_keeps": 0,
            "num_background_keeps": 0,
            "num_pool_keeps": 0,
            "num_pool_dynamic_keeps": 0,
            "num_pool_background_keeps": 0,
            "num_moving_edits": 0,
            "ce": 0.0,
            "lovasz": 0.0,
            "accuracy": float("nan"),
            "edit_accuracy": float("nan"),
            "false_edit_rate": float("nan"),
            "balanced_false_edit_rate": float("nan"),
            "pool_false_edit_rate": float("nan"),
            "num_ce_predicted_edits": 0,
            "ce_predicted_edit_fraction": float("nan"),
            "num_pool_predicted_edits": 0,
            "pool_predicted_edit_fraction": float("nan"),
            "dynamic_keep_fraction_realized": float("nan"),
        }
        if not deterministic:
            _remember_train_v2_info(info)
        return zero, info

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
        "num_pool_keeps": int(total_pool_keeps),
        "num_pool_dynamic_keeps": int(total_pool_dynamic_keeps),
        "num_pool_background_keeps": int(total_pool_background_keeps),
        "num_moving_edits": int(total_moving_edits),
        "edit_fraction": float(total_edits / max(num_ce, 1)),
        "moving_edit_fraction": float(total_moving_edits / max(total_edits, 1)),
        "dynamic_keep_fraction_realized": float(
            total_dynamic_keeps / max(total_keeps, 1)
        ),
    })
    if not deterministic:
        _remember_train_v2_info(info)
    return loss, info


def _remember_train_v2_info(info: dict) -> None:
    """Keep the latest v2-only training statistics for checkpoint/report history."""
    global _LAST_TRAIN_V2_INFO
    keys = (
        "balanced_false_edit_rate",
        "pool_false_edit_rate",
        "dynamic_keep_fraction_realized",
        "num_lovasz_voxels",
        "num_dynamic_keeps",
        "num_background_keeps",
        "num_pool_keeps",
        "num_pool_dynamic_keeps",
        "num_pool_background_keeps",
        "num_ce_predicted_edits",
        "ce_predicted_edit_fraction",
        "num_pool_predicted_edits",
        "pool_predicted_edit_fraction",
    )
    _LAST_TRAIN_V2_INFO = {k: info.get(k) for k in keys}


def aggregate_edit_validation_infos(
    infos: list[dict], *, lovasz_weight: float
) -> dict:
    """Aggregate v2 validation statistics with population-correct weights.

    CE/action accuracy use the balanced CE population. Lovasz uses the full
    compact sidecar population. Pool false-edit uses the complete compact KEEP
    pool, while the diagnostic balanced false-edit rate uses sampled KEEP only.
    """
    if lovasz_weight < 0:
        raise ValueError("lovasz_weight must be non-negative")
    if not infos:
        raise RuntimeError("validation contains no P0-F8 edit supervision voxels")

    ce_sum = 0.0
    ce_weight = 0
    lovasz_sum = 0.0
    lovasz_population = 0
    accuracy_sum = 0.0
    accuracy_weight = 0
    edit_acc_sum = 0.0
    edit_acc_weight = 0
    balanced_false_sum = 0.0
    balanced_false_weight = 0
    pool_false_sum = 0.0
    pool_false_weight = 0
    total_ce_predicted_edits = 0
    total_pool_predicted_edits = 0

    total_edits = 0
    total_keeps = 0
    total_dynamic_keeps = 0
    total_background_keeps = 0
    total_pool_keeps = 0
    total_pool_dynamic_keeps = 0
    total_pool_background_keeps = 0
    total_moving_edits = 0

    for info in infos:
        nce = int(info.get("num_supervised_voxels", 0))
        if nce <= 0:
            continue
        nlov = int(info.get("num_lovasz_voxels", nce))
        ne = int(info.get("num_edits", 0))
        nk = int(info.get("num_keeps", 0))
        npool_keep = int(info.get("num_pool_keeps", nk))

        ce = float(info["ce"])
        lovasz = float(info["lovasz"])
        accuracy = float(info["accuracy"])
        if math.isfinite(ce):
            ce_sum += ce * nce
            ce_weight += nce
        if nlov > 0 and math.isfinite(lovasz):
            lovasz_sum += lovasz * nlov
            lovasz_population += nlov
        if math.isfinite(accuracy):
            accuracy_sum += accuracy * nce
            accuracy_weight += nce

        edit_acc = float(info.get("edit_accuracy", float("nan")))
        if ne > 0 and math.isfinite(edit_acc):
            edit_acc_sum += edit_acc * ne
            edit_acc_weight += ne

        balanced_false = float(
            info.get("balanced_false_edit_rate", info.get("false_edit_rate", float("nan")))
        )
        if nk > 0 and math.isfinite(balanced_false):
            balanced_false_sum += balanced_false * nk
            balanced_false_weight += nk

        pool_false = float(
            info.get("pool_false_edit_rate", info.get("false_edit_rate", float("nan")))
        )
        if npool_keep > 0 and math.isfinite(pool_false):
            pool_false_sum += pool_false * npool_keep
            pool_false_weight += npool_keep

        total_edits += ne
        total_keeps += nk
        total_dynamic_keeps += int(info.get("num_dynamic_keeps", 0))
        total_background_keeps += int(info.get("num_background_keeps", 0))
        total_pool_keeps += npool_keep
        total_pool_dynamic_keeps += int(
            info.get("num_pool_dynamic_keeps", info.get("num_dynamic_keeps", 0))
        )
        total_pool_background_keeps += int(
            info.get("num_pool_background_keeps", info.get("num_background_keeps", 0))
        )
        total_moving_edits += int(info.get("num_moving_edits", 0))
        total_ce_predicted_edits += int(info.get("num_ce_predicted_edits", 0))
        total_pool_predicted_edits += int(info.get("num_pool_predicted_edits", 0))

    if ce_weight <= 0:
        raise RuntimeError("validation contains no P0-F8 balanced CE voxels")
    if lovasz_population <= 0:
        raise RuntimeError("validation contains no P0-F8 Lovasz voxels")

    ce_avg = ce_sum / float(ce_weight)
    lovasz_avg = lovasz_sum / float(lovasz_population)
    edit_avg = ce_avg + float(lovasz_weight) * lovasz_avg
    pool_false_avg = (
        pool_false_sum / float(pool_false_weight)
        if pool_false_weight else float("nan")
    )

    return {
        "edit_loss": edit_avg,
        "edit_ce": ce_avg,
        "result_lovasz": lovasz_avg,
        "action_accuracy": (
            accuracy_sum / float(accuracy_weight)
            if accuracy_weight else float("nan")
        ),
        "edit_accuracy": (
            edit_acc_sum / float(edit_acc_weight)
            if edit_acc_weight else float("nan")
        ),
        # Historical alias now intentionally means the deployment-like pool metric.
        "false_edit_rate": pool_false_avg,
        "balanced_false_edit_rate": (
            balanced_false_sum / float(balanced_false_weight)
            if balanced_false_weight else float("nan")
        ),
        "pool_false_edit_rate": pool_false_avg,
        "num_ce_predicted_edits": int(total_ce_predicted_edits),
        "ce_predicted_edit_fraction": float(
            total_ce_predicted_edits / max(ce_weight, 1)
        ),
        "num_pool_predicted_edits": int(total_pool_predicted_edits),
        "pool_predicted_edit_fraction": float(
            total_pool_predicted_edits / max(lovasz_population, 1)
        ),
        "num_supervised_voxels": int(ce_weight),
        "num_lovasz_voxels": int(lovasz_population),
        "num_edits": int(total_edits),
        "num_keeps": int(total_keeps),
        "num_dynamic_keeps": int(total_dynamic_keeps),
        "num_background_keeps": int(total_background_keeps),
        "num_pool_keeps": int(total_pool_keeps),
        "num_pool_dynamic_keeps": int(total_pool_dynamic_keeps),
        "num_pool_background_keeps": int(total_pool_background_keeps),
        "num_moving_edits": int(total_moving_edits),
        "dynamic_keep_fraction_realized": float(
            total_dynamic_keeps / max(total_keeps, 1)
        ),
        "pool_dynamic_keep_fraction": float(
            total_pool_dynamic_keeps / max(total_pool_keeps, 1)
        ),
    }


@torch.no_grad()
def validate_v2(
    model,
    loader,
    device,
    edit_cache,
    vae,
    edit_lambda,
    use_amp,
    *,
    keep_ratio,
    keep_when_no_edit,
    lovasz_weight,
    seed,
):
    """Validate with separate CE/Lovasz/KEEP population aggregation."""
    model.eval()
    fm_sum = 0.0
    fm_windows = 0
    cos_sum = 0.0
    skipped_batches = 0
    edit_infos: list[dict] = []

    for batch_idx, batch in enumerate(loader):
        prepared = base.f6.prepare_batch(batch, device)
        if prepared is None:
            skipped_batches += 1
            continue
        # Sparse decoder validation needs autograd internally even though model
        # parameters are not updated here.
        with torch.enable_grad():
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_amp,
            ):
                fm_loss, flow_info = model.flow_loss(
                    prepared["history"],
                    prepared["target"],
                    prepared["anchor"],
                    history_context=prepared["context"],
                    trajectory=prepared["trajectory"],
                    window_origins=prepared["origins"],
                    t_override=0.5,
                    source_noise=torch.zeros_like(prepared["anchor"]),
                    return_endpoint=True,
                )
                endpoint = base.f6.scatter_endpoint_to_full(
                    flow_info["predicted_endpoint"], prepared
                )
                edit_loss, edit_info = edit_loss_for_endpoint_v2(
                    model,
                    endpoint,
                    sample_ids=prepared["sample_ids"],
                    edit_cache=edit_cache,
                    vae=vae,
                    keep_ratio=float(keep_ratio),
                    keep_when_no_edit=int(keep_when_no_edit),
                    lovasz_weight=float(lovasz_weight),
                    deterministic=True,
                    selection_seed=int(seed) + batch_idx,
                )

        nwin = int(prepared["history"].shape[0])
        fm_sum += float(fm_loss.detach().item()) * nwin
        fm_windows += nwin
        cos_sum += float(flow_info["cosine"]) * nwin
        if int(edit_info.get("num_supervised_voxels", 0)) > 0:
            edit_infos.append(edit_info)
        del endpoint, edit_loss, fm_loss, flow_info

    model.train()
    if fm_windows <= 0:
        raise RuntimeError("validation contains no valid routed Sparse-WM windows")

    edit_agg = aggregate_edit_validation_infos(
        edit_infos, lovasz_weight=float(lovasz_weight)
    )
    fm_avg = fm_sum / float(fm_windows)
    edit_avg = float(edit_agg["edit_loss"])
    return {
        "objective": fm_avg + float(edit_lambda) * edit_avg,
        "fm_loss": fm_avg,
        "lambda_edit": float(edit_lambda),
        "weighted_edit_loss": float(edit_lambda) * edit_avg,
        **edit_agg,
        "cosine": cos_sum / float(fm_windows),
        "num_windows": fm_windows,
        "skipped_empty_batches": skipped_batches,
        "validation_aggregation": (
            "CE_by_balanced_voxels; Lovasz_by_full_pool_voxels; "
            "false_edit_by_full_pool_KEEP"
        ),
    }


def _merge_latest_train_v2_fields(history: list, fields: dict) -> None:
    """Persist v2-only train stats in the validation-step training record."""
    if not history or not fields:
        return
    row = history[-1]
    train = row.get("train") if isinstance(row, dict) else None
    if isinstance(train, dict):
        train.update(fields)


def _payload_v2(**kwargs):
    history = kwargs.get("history")
    if isinstance(history, list):
        _merge_latest_train_v2_fields(history, _LAST_TRAIN_V2_INFO)
    return _BASE_PAYLOAD(**kwargs)


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
    arch["validation_aggregation"] = (
        "CE_weighted_by_balanced_voxels; Lovasz_weighted_by_full_pool_voxels; "
        "false_edit_weighted_by_full_pool_KEEP"
    )
    arch["all_keep_collapse_gate"] = {
        "protocol": "all_keep_validation_gate_v1",
        "check_step": int(args.collapse_check_step),
        "population": "complete_compact_sidecar_pool",
        "failure": "zero_predicted_non_KEEP_actions",
    }
    return arch


def _install_v2_patch() -> None:
    # ``base.main`` resolves these names from its own module globals at runtime.
    base.F8_PROTOCOL = F8_PROTOCOL
    base.edit_loss_for_endpoint = edit_loss_for_endpoint_v2
    base.validate = validate_v2
    base._payload = _payload_v2
    base._architecture = _architecture_v2


def main():
    _install_v2_patch()
    print(json.dumps({
        "p0_f8_protocol": F8_PROTOCOL,
        "balanced_ce": "all_EDIT + 50% dynamic KEEP + 50% background KEEP",
        "lovasz_population": "complete compact sidecar pool",
        "false_edit_metric": "complete KEEP pool",
        "validation_aggregation": (
            "CE/balanced voxels + Lovasz/full-pool voxels + false-edit/full-pool KEEP"
        ),
    }))
    base.main()


__all__ = [
    "F8_PROTOCOL",
    "aggregate_edit_validation_infos",
    "edit_loss_for_endpoint_v2",
    "validate_v2",
    "main",
]
