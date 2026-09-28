"""Losses for V20 unified transport completion."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

from .local_st_world_model_v18_se2 import (
    periodic_yaw_loss,
    soft_se2_transport_overlap_loss,
)
from .v20_history_world import FREE_LABEL


COMPLETION_OBJECTIVE_PROTOCOL = (
    "coupled_energy_focal_presence_positive_semantic_v1"
)
DEFAULT_PRESENCE_FOCAL_GAMMA = 2.0
DEFAULT_COMPLETION_SEMANTIC_WEIGHT = 1.0


@dataclass(frozen=True)
class CompletionLoss:
    mean: torch.Tensor
    loss_sum: torch.Tensor
    count: int | torch.Tensor


@dataclass(frozen=True)
class CoupledCompletionLoss:
    """Coupled occupancy/semantic objective and its sufficient statistics."""

    mean: torch.Tensor
    loss_sum: torch.Tensor
    count: int | torch.Tensor
    presence_mean: torch.Tensor
    semantic_mean: torch.Tensor
    presence_free_mean: torch.Tensor
    presence_occupied_mean: torch.Tensor
    free_count: int | torch.Tensor
    occupied_count: int | torch.Tensor
    class_counts: tuple[int, ...] | torch.Tensor


def completion_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor,
) -> CompletionLoss:
    """FP32 18-way CE with an explicit sum/count statistic."""
    loss_sum, count_tensor = _completion_cross_entropy_tensors(
        logits, target, loss_mask
    )
    count = int(count_tensor.detach().cpu())
    mean = loss_sum / count_tensor.clamp_min(1).to(loss_sum.dtype)
    return CompletionLoss(mean=mean, loss_sum=loss_sum, count=count)


def _completion_cross_entropy_tensors(
    logits: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return graph-connected CE sum/count without synchronizing the device."""
    if logits.shape[:-1] != target.shape or target.shape != loss_mask.shape:
        raise ValueError("completion logits/target/mask shape mismatch")
    if logits.shape[-1] != 18:
        raise ValueError("completion requires exactly 18 classes")
    mask = loss_mask.bool()
    values = F.cross_entropy(
        logits[mask].float(), target[mask].long(), reduction="none"
    )
    return values.sum(dtype=torch.float32), mask.sum(dtype=torch.int64)


def _weighted_completion_cross_entropy_tensors(
    rows: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """One CE launch for unique equal-shaped tiles with draw multiplicity."""
    logits = torch.stack([row[0] for row in rows], dim=0)
    targets = torch.stack([row[1] for row in rows], dim=0)
    masks = torch.stack([row[2] for row in rows], dim=0).bool()
    values = F.cross_entropy(
        logits[masks].float(), targets[masks].long(), reduction="none"
    )
    multiplicity = torch.as_tensor(
        [row[3] for row in rows], device=masks.device, dtype=torch.int64
    )
    voxel_weight = multiplicity.view(
        len(rows), *([1] * (masks.ndim - 1))
    ).expand_as(masks)[masks]
    return (
        (values * voxel_weight.to(values.dtype)).sum(dtype=torch.float32),
        voxel_weight.sum(dtype=torch.int64),
    )


def completion_tiles_cross_entropy(
    logits: Sequence[torch.Tensor],
    targets: Sequence[torch.Tensor],
    loss_masks: Sequence[torch.Tensor],
    *,
    graph_anchor: torch.Tensor | None = None,
    materialize_stats: bool = True,
) -> CompletionLoss:
    """Combine repeated tile draws with exact multiplicity and one sync."""
    if not (len(logits) == len(targets) == len(loss_masks)):
        raise ValueError("tile loss inputs must have equal length")
    unique: dict[tuple[int, int, int], list] = {}
    for a, b, c in zip(logits, targets, loss_masks):
        identity = (id(a), id(b), id(c))
        previous = unique.get(identity)
        if previous is None:
            unique[identity] = [a, b, c, 1]
        else:
            previous[3] = int(previous[3]) + 1
    groups: dict[
        tuple[tuple[int, ...], tuple[int, ...]],
        list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]],
    ] = {}
    for a, b, c, multiplicity in unique.values():
        key = (tuple(a.shape), tuple(b.shape))
        groups.setdefault(key, []).append((a, b, c, int(multiplicity)))
    # Boundary tiles have a handful of shapes.  One batched CE per shape avoids
    # launching up to 96 tiny cross-entropy kernels while retaining repeated
    # unique tiles and applying the original draw multiplicity as a voxel
    # weight preserves the exact numerator/denominator without duplicating
    # logits, targets and masks in memory.
    pieces = [
        _weighted_completion_cross_entropy_tensors(rows)
        for rows in groups.values()
    ]
    if pieces:
        loss_sum = torch.stack([piece[0] for piece in pieces]).sum()
        count_tensor = torch.stack([piece[1] for piece in pieces]).sum()
    elif graph_anchor is not None:
        loss_sum = graph_anchor.sum() * 0.0
        count_tensor = torch.zeros((), dtype=torch.int64, device=loss_sum.device)
    else:
        loss_sum = torch.zeros((), dtype=torch.float32, requires_grad=True)
        count_tensor = torch.zeros((), dtype=torch.int64)
    count = (
        int(count_tensor.detach().cpu())
        if bool(materialize_stats)
        else count_tensor.detach()
    )
    mean = loss_sum / count_tensor.clamp_min(1).to(loss_sum.dtype)
    return CompletionLoss(mean=mean, loss_sum=loss_sum, count=count)


def _coupled_completion_terms(
    logits: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor,
    voxel_weight: torch.Tensor,
    *,
    presence_focal_gamma: float,
) -> dict[str, torch.Tensor]:
    """Return additive terms for the coupled 18-way completion objective.

    ``presence_logit`` is derived from the same semantic logits used at
    inference.  This is deliberately not an independent binary head:

        logsumexp(non-free logits) - free logit

    The positive-only semantic term then trains the conditional class
    distribution without letting the overwhelmingly frequent free label
    dilute its normalization.
    """
    if logits.shape[:-1] != target.shape or target.shape != loss_mask.shape:
        raise ValueError("completion logits/target/mask shape mismatch")
    if logits.shape[-1] != 18 or FREE_LABEL != 17:
        raise ValueError("completion requires 17 occupied classes plus free")
    gamma = float(presence_focal_gamma)
    if not math.isfinite(gamma) or gamma < 0.0:
        raise ValueError("presence_focal_gamma must be finite and non-negative")

    mask = loss_mask.bool()
    rows = logits[mask].float()
    labels = target[mask].long()
    weights = voxel_weight[mask].to(dtype=torch.float32)
    if weights.shape != labels.shape:
        raise ValueError("completion voxel weights must match masked targets")

    occupied = labels != FREE_LABEL
    occupied_float = occupied.to(dtype=torch.float32)
    free_float = (~occupied).to(dtype=torch.float32)
    presence_logit = torch.logsumexp(rows[:, :FREE_LABEL], dim=-1) - rows[:, FREE_LABEL]
    presence_bce = F.binary_cross_entropy_with_logits(
        presence_logit, occupied_float, reduction="none"
    )
    if gamma == 0.0:
        presence_focal = presence_bce
    else:
        # exp(-BCE) is p_t and is stable for either binary target.
        presence_focal = (1.0 - torch.exp(-presence_bce)).pow(gamma) * presence_bce

    # Materialize semantic softmax only for scarce positive rows.  Besides
    # matching the intended normalization this avoids a second 17-way softmax
    # over the overwhelmingly free sampled population.
    semantic_nll = F.cross_entropy(
        rows[occupied, :FREE_LABEL], labels[occupied], reduction="none"
    )

    integer_weights = voxel_weight[mask].to(dtype=torch.int64)
    class_counts = torch.zeros(
        18, dtype=torch.int64, device=labels.device
    ).scatter_add(0, labels, integer_weights)
    return {
        "presence_sum": (presence_focal * weights).sum(dtype=torch.float32),
        "presence_free_sum": (
            presence_focal * weights * free_float
        ).sum(dtype=torch.float32),
        "presence_occupied_sum": (
            presence_focal * weights * occupied_float
        ).sum(dtype=torch.float32),
        "semantic_sum": (
            semantic_nll * weights[occupied]
        ).sum(dtype=torch.float32),
        "count": integer_weights.sum(dtype=torch.int64),
        "free_count": (integer_weights * (~occupied).to(torch.int64)).sum(
            dtype=torch.int64
        ),
        "occupied_count": (integer_weights * occupied.to(torch.int64)).sum(
            dtype=torch.int64
        ),
        "class_counts": class_counts,
    }


def _finalize_coupled_completion(
    pieces: Sequence[Mapping[str, torch.Tensor]],
    *,
    graph_anchor: torch.Tensor | None,
    semantic_weight: float,
    materialize_stats: bool,
) -> CoupledCompletionLoss:
    weight = float(semantic_weight)
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError("completion_semantic_weight must be finite and non-negative")
    if pieces:
        combined = {
            name: torch.stack([piece[name] for piece in pieces]).sum(dim=0)
            for name in pieces[0]
        }
    else:
        if graph_anchor is None:
            zero = torch.zeros((), dtype=torch.float32, requires_grad=True)
        else:
            zero = graph_anchor.sum() * 0.0
        combined = {
            "presence_sum": zero,
            "presence_free_sum": zero,
            "presence_occupied_sum": zero,
            "semantic_sum": zero,
            "count": torch.zeros((), dtype=torch.int64, device=zero.device),
            "free_count": torch.zeros((), dtype=torch.int64, device=zero.device),
            "occupied_count": torch.zeros((), dtype=torch.int64, device=zero.device),
            "class_counts": torch.zeros(18, dtype=torch.int64, device=zero.device),
        }

    count_tensor = combined["count"]
    free_count_tensor = combined["free_count"]
    occupied_count_tensor = combined["occupied_count"]
    presence_mean = combined["presence_sum"] / count_tensor.clamp_min(1).to(
        torch.float32
    )
    semantic_mean = combined["semantic_sum"] / occupied_count_tensor.clamp_min(1).to(
        torch.float32
    )
    presence_free_mean = combined["presence_free_sum"] / free_count_tensor.clamp_min(
        1
    ).to(torch.float32)
    presence_occupied_mean = combined[
        "presence_occupied_sum"
    ] / occupied_count_tensor.clamp_min(1).to(torch.float32)
    mean = presence_mean + weight * semantic_mean
    # Preserve the historical mean == loss_sum / count invariant even though
    # the two objective components intentionally use different denominators.
    loss_sum = mean * count_tensor.to(mean.dtype)

    if materialize_stats:
        count: int | torch.Tensor = int(count_tensor.detach().cpu())
        free_count: int | torch.Tensor = int(free_count_tensor.detach().cpu())
        occupied_count: int | torch.Tensor = int(
            occupied_count_tensor.detach().cpu()
        )
        class_counts: tuple[int, ...] | torch.Tensor = tuple(
            int(value) for value in combined["class_counts"].detach().cpu().tolist()
        )
    else:
        count = count_tensor.detach()
        free_count = free_count_tensor.detach()
        occupied_count = occupied_count_tensor.detach()
        class_counts = combined["class_counts"].detach()
    return CoupledCompletionLoss(
        mean=mean,
        loss_sum=loss_sum,
        count=count,
        presence_mean=presence_mean,
        semantic_mean=semantic_mean,
        presence_free_mean=presence_free_mean,
        presence_occupied_mean=presence_occupied_mean,
        free_count=free_count,
        occupied_count=occupied_count,
        class_counts=class_counts,
    )


def completion_coupled_focal_semantic(
    logits: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    presence_focal_gamma: float = DEFAULT_PRESENCE_FOCAL_GAMMA,
    semantic_weight: float = DEFAULT_COMPLETION_SEMANTIC_WEIGHT,
) -> CoupledCompletionLoss:
    """Materialized coupled objective for one dense tensor (mainly tests)."""
    piece = _coupled_completion_terms(
        logits,
        target,
        loss_mask,
        torch.ones_like(target, dtype=torch.int64),
        presence_focal_gamma=float(presence_focal_gamma),
    )
    return _finalize_coupled_completion(
        [piece],
        graph_anchor=logits,
        semantic_weight=float(semantic_weight),
        materialize_stats=True,
    )


def completion_tiles_coupled_focal_semantic(
    logits: Sequence[torch.Tensor],
    targets: Sequence[torch.Tensor],
    loss_masks: Sequence[torch.Tensor],
    *,
    presence_focal_gamma: float = DEFAULT_PRESENCE_FOCAL_GAMMA,
    semantic_weight: float = DEFAULT_COMPLETION_SEMANTIC_WEIGHT,
    graph_anchor: torch.Tensor | None = None,
    materialize_stats: bool = True,
) -> CoupledCompletionLoss:
    """Coupled completion loss with exact repeated-tile multiplicity."""
    if not (len(logits) == len(targets) == len(loss_masks)):
        raise ValueError("tile loss inputs must have equal length")
    unique: dict[tuple[int, int, int], list] = {}
    for a, b, c in zip(logits, targets, loss_masks):
        identity = (id(a), id(b), id(c))
        previous = unique.get(identity)
        if previous is None:
            unique[identity] = [a, b, c, 1]
        else:
            previous[3] = int(previous[3]) + 1
    groups: dict[
        tuple[tuple[int, ...], tuple[int, ...]],
        list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]],
    ] = {}
    for a, b, c, multiplicity in unique.values():
        key = (tuple(a.shape), tuple(b.shape))
        groups.setdefault(key, []).append((a, b, c, int(multiplicity)))

    pieces = []
    for rows in groups.values():
        stacked_logits = torch.stack([row[0] for row in rows], dim=0)
        stacked_targets = torch.stack([row[1] for row in rows], dim=0)
        stacked_masks = torch.stack([row[2] for row in rows], dim=0).bool()
        multiplicity = torch.as_tensor(
            [row[3] for row in rows],
            device=stacked_masks.device,
            dtype=torch.int64,
        )
        voxel_weight = multiplicity.view(
            len(rows), *([1] * (stacked_masks.ndim - 1))
        ).expand_as(stacked_masks)
        pieces.append(
            _coupled_completion_terms(
                stacked_logits,
                stacked_targets,
                stacked_masks,
                voxel_weight,
                presence_focal_gamma=float(presence_focal_gamma),
            )
        )
    return _finalize_coupled_completion(
        pieces,
        graph_anchor=graph_anchor,
        semantic_weight=float(semantic_weight),
        materialize_stats=bool(materialize_stats),
    )


def exact_v18_clean_e14_loss(
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    *,
    patch_resolution_m: float,
    materialize_stats: bool = True,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """The unchanged Clean-E14 objective: trans + exist + 19*yaw + .25*shape."""
    pred_xy = outputs["residual_xy_m"]
    target_xy = batch["target_source_residual_xy_m"].to(pred_xy.dtype)
    valid = batch["se2_target_valid"].bool()
    if bool(materialize_stats):
        trans = (
            F.smooth_l1_loss(
                pred_xy[valid], target_xy[valid], reduction="mean", beta=1.0
            )
            if bool(valid.any())
            else pred_xy.sum() * 0.0
        )
    else:
        trans_values = F.smooth_l1_loss(
            pred_xy, target_xy, reduction="none", beta=1.0
        ).sum(dim=-1)
        trans_count = valid.sum(dtype=torch.int64)
        trans = trans_values.masked_select(valid).sum() / (
            trans_count.clamp_min(1).to(trans_values.dtype) * pred_xy.shape[-1]
        )

    existence_logits = outputs["existence_logits"]
    existence = batch["existence"].to(existence_logits.dtype)
    supervised = batch["supervised_source"].bool()[:, None].expand_as(existence)
    if bool(materialize_stats):
        exist = (
            F.binary_cross_entropy_with_logits(
                existence_logits[supervised], existence[supervised]
            )
            if bool(supervised.any())
            else existence_logits.sum() * 0.0
        )
    else:
        exist_values = F.binary_cross_entropy_with_logits(
            existence_logits, existence, reduction="none"
        )
        exist_count = supervised.sum(dtype=torch.int64)
        exist = exist_values.masked_select(supervised).sum() / exist_count.clamp_min(
            1
        ).to(exist_values.dtype)

    yaw, yaw_stats = periodic_yaw_loss(
        outputs["yaw_delta_rad"].float(),
        batch["target_yaw_rad"].float(),
        batch["yaw_enabled"],
        batch["yaw_label_valid"].bool() & valid,
        materialize_stats=bool(materialize_stats),
    )
    pred_disp = batch["kta_displacement_xy_m"].float() + pred_xy.float()
    shape, shape_stats = soft_se2_transport_overlap_loss(
        pred_disp,
        batch["target_source_displacement_xy_m"].float(),
        outputs["yaw_delta_rad"].float(),
        batch["target_yaw_rad"].float(),
        batch["target_source_mask_tube"][:, -1].float(),
        valid,
        batch["yaw_enabled"],
        batch["yaw_label_valid"],
        patch_resolution_m=float(patch_resolution_m),
        materialize_stats=bool(materialize_stats),
    )
    total = trans + exist + 19.0 * yaw + 0.25 * shape
    if not bool(materialize_stats):
        return total, {
            "transport_loss": total.detach(),
            "translation_smooth_l1": trans.detach(),
            "existence_bce": exist.detach(),
            "yaw_periodic_loss": yaw.detach(),
            "se2_shape_loss": shape.detach(),
            **yaw_stats,
            **shape_stats,
        }
    return total, {
        "transport_loss": float(total.detach().cpu()),
        "translation_smooth_l1": float(trans.detach().cpu()),
        "existence_bce": float(exist.detach().cpu()),
        "yaw_periodic_loss": float(yaw.detach().cpu()),
        "se2_shape_loss": float(shape.detach().cpu()),
        **yaw_stats,
        **shape_stats,
    }


def compute_training_loss(
    transport_outputs: Mapping[str, torch.Tensor],
    transport_batch: Mapping[str, torch.Tensor],
    completion_logits: Sequence[torch.Tensor],
    completion_targets: Sequence[torch.Tensor],
    completion_masks: Sequence[torch.Tensor],
    *,
    patch_resolution_m: float,
    completion_weight: float = 1.0,
    presence_focal_gamma: float = DEFAULT_PRESENCE_FOCAL_GAMMA,
    completion_semantic_weight: float = DEFAULT_COMPLETION_SEMANTIC_WEIGHT,
    graph_anchor: torch.Tensor | None = None,
    materialize_stats: bool = True,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    trans, stats = exact_v18_clean_e14_loss(
        transport_outputs,
        transport_batch,
        patch_resolution_m=patch_resolution_m,
        materialize_stats=bool(materialize_stats),
    )
    comp = completion_tiles_coupled_focal_semantic(
        completion_logits,
        completion_targets,
        completion_masks,
        presence_focal_gamma=float(presence_focal_gamma),
        semantic_weight=float(completion_semantic_weight),
        graph_anchor=graph_anchor,
        materialize_stats=bool(materialize_stats),
    )
    total = trans + float(completion_weight) * comp.mean
    count_tensor = torch.as_tensor(comp.count, device=comp.mean.device)
    occupied_count_tensor = torch.as_tensor(
        comp.occupied_count, device=comp.mean.device
    )
    occupied_fraction = occupied_count_tensor.to(torch.float32) / count_tensor.clamp_min(
        1
    ).to(torch.float32)
    class_stats = {
        f"completion_class_{class_id:02d}_voxels": comp.class_counts[class_id]
        for class_id in range(18)
    }
    if not bool(materialize_stats):
        return total, {
            "loss": total.detach(),
            **stats,
            "completion_loss": comp.mean.detach(),
            "completion_loss_sum": comp.loss_sum.detach(),
            "completion_presence_focal": comp.presence_mean.detach(),
            "completion_semantic_ce": comp.semantic_mean.detach(),
            "completion_presence_free_focal": comp.presence_free_mean.detach(),
            "completion_presence_occupied_focal": (
                comp.presence_occupied_mean.detach()
            ),
            "completion_voxels": comp.count,
            "completion_free_voxels": comp.free_count,
            "completion_occupied_voxels": comp.occupied_count,
            "completion_occupied_fraction": occupied_fraction.detach(),
            "completion_weight": float(completion_weight),
            "completion_semantic_weight": float(completion_semantic_weight),
            "presence_focal_gamma": float(presence_focal_gamma),
            **class_stats,
        }
    return total, {
        "loss": float(total.detach().cpu()),
        **stats,
        "completion_loss": float(comp.mean.detach().cpu()),
        "completion_loss_sum": float(comp.loss_sum.detach().cpu()),
        "completion_presence_focal": float(comp.presence_mean.detach().cpu()),
        "completion_semantic_ce": float(comp.semantic_mean.detach().cpu()),
        "completion_presence_free_focal": float(
            comp.presence_free_mean.detach().cpu()
        ),
        "completion_presence_occupied_focal": float(
            comp.presence_occupied_mean.detach().cpu()
        ),
        "completion_voxels": comp.count,
        "completion_free_voxels": comp.free_count,
        "completion_occupied_voxels": comp.occupied_count,
        "completion_occupied_fraction": float(occupied_fraction.detach().cpu()),
        "completion_weight": float(completion_weight),
        "completion_semantic_weight": float(completion_semantic_weight),
        "presence_focal_gamma": float(presence_focal_gamma),
        **class_stats,
    }
