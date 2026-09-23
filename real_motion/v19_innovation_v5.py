"""V19 Innovation v5: structured absolute vertical endpoints.

This module keeps the proven future-aligned temporal trunk and changes only the
vertical geometry parameterization/loss:

- bottom z-bin: absolute 16-way classification;
- top z-bin: absolute 16-way classification;
- Gaussian-smoothed endpoint CE gives neighboring bins partial credit;
- differentiable soft interval IoU couples both endpoints;
- constrained joint decoding enforces top >= bottom without a post-hoc max.

The design intentionally avoids stacking ordinal, EMD, Gaussian smoothing and
GIoU simultaneously. Gaussian endpoint CE already supplies ordinal locality;
soft interval IoU supplies the joint interval constraint with dense gradients.
"""
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn.functional as F

from .v19_innovation import (
    InnovationLossWeights,
    ResidualInnovationIntervalHead,
)


class ResidualInnovationEndpointHead(ResidualInnovationIntervalHead):
    """Same temporal trunk as v4, but second geometry head predicts absolute top."""

    def forward(
        self,
        future_aligned_semantic: torch.Tensor,
        future_aligned_geometry: torch.Tensor,
        base_explained: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        out = super().forward(
            future_aligned_semantic,
            future_aligned_geometry,
            base_explained,
        )
        return {
            "add_presence_logits": out["add_presence_logits"],
            "semantic_logits": out["semantic_logits"],
            "bottom_logits": out["bottom_logits"],
            "top_logits": out["span_logits"],
        }


def gaussian_endpoint_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    sigma: float = 0.75,
) -> torch.Tensor:
    """Cross entropy against a normalized Gaussian over ordered z bins."""
    if logits.ndim != 2:
        raise ValueError("endpoint logits must be [N,Z]")
    if target.ndim != 1 or target.shape[0] != logits.shape[0]:
        raise ValueError("endpoint target must be [N]")
    if float(sigma) <= 0:
        return F.cross_entropy(logits, target.long())

    Z = int(logits.shape[1])
    bins = torch.arange(
        Z,
        device=logits.device,
        dtype=logits.dtype,
    )[None, :]
    center = target.to(logits.dtype)[:, None]
    soft = torch.exp(
        -0.5 * ((bins - center) / float(sigma)) ** 2
    )
    soft = soft / soft.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return -(soft * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def soft_interval_iou_loss(
    bottom_logits: torch.Tensor,
    top_logits: torch.Tensor,
    bottom_target: torch.Tensor,
    top_target: torch.Tensor,
) -> torch.Tensor:
    """Differentiable 1D interval IoU from endpoint distributions.

    P(z is inside) = P(bottom <= z) * P(top >= z).  Unlike hard-box IoU/GIoU,
    this remains differentiable even when the current endpoint modes do not
    overlap the target interval.
    """
    if bottom_logits.shape != top_logits.shape or bottom_logits.ndim != 2:
        raise ValueError("endpoint logits must share [N,Z] shape")
    Z = int(bottom_logits.shape[1])
    pb = F.softmax(bottom_logits, dim=1)
    pt = F.softmax(top_logits, dim=1)

    p_bottom_le_z = torch.cumsum(pb, dim=1)
    p_top_ge_z = torch.flip(
        torch.cumsum(torch.flip(pt, dims=(1,)), dim=1),
        dims=(1,),
    )
    p_occ = p_bottom_le_z * p_top_ge_z

    z = torch.arange(Z, device=bottom_logits.device)[None, :]
    target_occ = (
        (z >= bottom_target[:, None])
        & (z <= top_target[:, None])
    ).to(p_occ.dtype)

    inter = (p_occ * target_occ).sum(dim=1)
    union = (
        p_occ + target_occ - p_occ * target_occ
    ).sum(dim=1)
    iou = (inter + 1e-6) / (union + 1e-6)
    return (1.0 - iou).mean()


def decode_ordered_endpoints(
    bottom_logits: torch.Tensor,
    top_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Joint MAP decode over valid endpoint pairs b<=t in O(Z), not O(Z^2)."""
    if bottom_logits.shape != top_logits.shape:
        raise ValueError("bottom/top logits shape mismatch")
    if bottom_logits.ndim != 5:
        raise ValueError("endpoint logits must be [B,F,Z,H,W]")
    Z = int(bottom_logits.shape[2])

    # For each candidate bottom b, find the best top among t>=b.
    rev = torch.flip(top_logits, dims=(2,))
    rev_val, rev_idx = torch.cummax(rev, dim=2)
    suffix_val = torch.flip(rev_val, dims=(2,))
    suffix_rev_idx = torch.flip(rev_idx, dims=(2,))
    suffix_top = (Z - 1) - suffix_rev_idx

    pair_score = bottom_logits + suffix_val
    bottom = pair_score.argmax(dim=2)
    top = torch.gather(
        suffix_top,
        2,
        bottom.unsqueeze(2),
    ).squeeze(2)
    if bool((top < bottom).any()):
        raise RuntimeError("ordered endpoint decoder violated top>=bottom")
    return bottom, top


def innovation_endpoint_loss(
    outputs: dict[str, torch.Tensor],
    *,
    add_target: torch.Tensor,
    semantic_target: torch.Tensor,
    vertical_target: torch.Tensor,
    candidate_mask: torch.Tensor,
    weights: InnovationLossWeights = InnovationLossWeights(),
    presence_hard_negative_ratio: float = 4.0,
    endpoint_sigma: float = 0.75,
    interval_iou_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Hard-negative presence + semantic + structured endpoint geometry."""
    add_logits = outputs["add_presence_logits"]
    sem_logits = outputs["semantic_logits"]
    bottom_logits = outputs["bottom_logits"]
    top_logits = outputs["top_logits"]
    pos = add_target.bool()
    cand = candidate_mask.bool()

    if add_logits.shape != pos.shape or cand.shape != pos.shape:
        raise ValueError("presence/candidate shape mismatch")
    if semantic_target.shape != pos.shape:
        raise ValueError("semantic_target shape mismatch")
    if vertical_target.ndim != 5:
        raise ValueError("vertical_target must be [B,F,Z,H,W]")
    Z = int(vertical_target.shape[2])
    if bottom_logits.shape[2] != Z or top_logits.shape[2] != Z:
        raise ValueError("endpoint logits/target bin mismatch")

    ratio = float(presence_hard_negative_ratio)
    if ratio <= 0:
        raise ValueError("presence_hard_negative_ratio must be positive")

    pos_rows = add_logits[pos & cand]
    neg_rows = add_logits[cand & ~pos]
    pos_loss = F.softplus(-pos_rows)
    neg_loss_all = F.softplus(neg_rows)
    npos = int(pos_rows.numel())
    nneg = int(neg_rows.numel())
    hard_negative_count = 0
    if npos > 0 and nneg > 0:
        hard_negative_count = min(
            nneg,
            max(1, int(math.ceil(ratio * npos))),
        )
        neg_loss = torch.topk(
            neg_loss_all,
            k=hard_negative_count,
            largest=True,
            sorted=False,
        ).values
        add = torch.cat((pos_loss, neg_loss), dim=0).mean()
    elif npos > 0:
        add = pos_loss.mean()
    elif nneg > 0:
        add = neg_loss_all.mean()
        hard_negative_count = nneg
    else:
        add = add_logits.sum() * 0.0

    if bool(pos.any()):
        sem_rows = sem_logits.permute(0, 1, 3, 4, 2)[pos]
        sem = F.cross_entropy(
            sem_rows,
            semantic_target[pos].long(),
        )

        z_rows = vertical_target.permute(
            0, 1, 3, 4, 2
        )[pos].bool()
        if not bool(z_rows.any(dim=1).all()):
            raise RuntimeError(
                "positive BEV cell without vertical target"
            )
        idx = torch.arange(Z, device=z_rows.device)[None]
        bottom_target = torch.where(
            z_rows,
            idx,
            torch.full_like(idx, Z),
        ).min(dim=1).values
        top_target = torch.where(
            z_rows,
            idx,
            torch.full_like(idx, -1),
        ).max(dim=1).values

        b_rows = bottom_logits.permute(
            0, 1, 3, 4, 2
        )[pos]
        t_rows = top_logits.permute(
            0, 1, 3, 4, 2
        )[pos]
        bottom = gaussian_endpoint_cross_entropy(
            b_rows,
            bottom_target,
            sigma=float(endpoint_sigma),
        )
        top = gaussian_endpoint_cross_entropy(
            t_rows,
            top_target,
            sigma=float(endpoint_sigma),
        )
        interval_iou = soft_interval_iou_loss(
            b_rows,
            t_rows,
            bottom_target,
            top_target,
        )
        vertical = (
            0.5 * (bottom + top)
            + float(interval_iou_weight) * interval_iou
        )
    else:
        sem = sem_logits.sum() * 0.0
        bottom = bottom_logits.sum() * 0.0
        top = top_logits.sum() * 0.0
        interval_iou = (bottom + top) * 0.0
        vertical = interval_iou

    total = (
        add
        + float(weights.semantic) * sem
        + float(weights.vertical) * vertical
    )
    return total, {
        "loss": float(total.detach().cpu()),
        "add_bce": float(add.detach().cpu()),
        "semantic_ce": float(sem.detach().cpu()),
        "bottom_soft_ce": float(bottom.detach().cpu()),
        "top_soft_ce": float(top.detach().cpu()),
        "interval_iou_loss": float(interval_iou.detach().cpu()),
        "vertical_endpoint_loss": float(vertical.detach().cpu()),
        "positive_bev_cells": int(pos.sum().item()),
        "candidate_bev_cells": int(cand.sum().item()),
        "hard_negative_bev_cells": int(hard_negative_count),
    }
