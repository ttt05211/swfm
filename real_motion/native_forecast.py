"""Deployment-aligned semantic supervision utilities for P0-F9."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .edit_repair import DYNAMIC_IDS, NUM_RESULT_CLASSES, lovasz_softmax_flat
from .edit_repair_v2 import full_edit_supervision

NUM_SEMANTIC_CLASSES = 18
NON_DYNAMIC_IDS = tuple(i for i in range(NUM_SEMANTIC_CLASSES) if i not in set(DYNAMIC_IDS))


def collapse_occ_logits_to_dynamic(logits: torch.Tensor) -> torch.Tensor:
    """Collapse official 18-way Occ3D logits to background + 8 dynamic classes.

    Background is the union of every non-motion-capable semantic class. Using
    logsumexp makes the resulting softmax exactly equal to probability
    marginalization of the original 18-way distribution.
    """
    if logits.ndim != 2 or logits.shape[-1] != NUM_SEMANTIC_CLASSES:
        raise ValueError(f"semantic logits must be [N,{NUM_SEMANTIC_CLASSES}]")
    bg = torch.logsumexp(logits.float()[:, list(NON_DYNAMIC_IDS)], dim=-1, keepdim=True)
    dyn = logits.float()[:, list(DYNAMIC_IDS)]
    out = torch.cat([bg, dyn], dim=-1)
    if out.shape[-1] != NUM_RESULT_CLASSES:
        raise RuntimeError("collapsed semantic class count mismatch")
    return out


def class_weights_from_edit_cache(
    edit_cache,
    *,
    min_weight: float = 0.5,
    max_weight: float = 2.0,
) -> torch.Tensor:
    """Mild inverse-sqrt class balancing from the compact deployment pool."""
    if min_weight <= 0 or max_weight < min_weight:
        raise ValueError("invalid class-weight clamp")
    counts = torch.zeros(NUM_RESULT_CLASSES, dtype=torch.float64)
    for rec in edit_cache.records.values():
        pool = full_edit_supervision(rec)
        labels = pool["result_slots"].long()
        if labels.numel():
            counts += torch.bincount(labels, minlength=NUM_RESULT_CLASSES).double()
    if float(counts.sum()) <= 0:
        raise RuntimeError("semantic target cache contains no supervision voxels")
    # Missing classes should not receive infinite/huge weights; they simply do
    # not contribute until observed in a batch.
    safe = counts.clamp_min(1.0)
    weights = safe.rsqrt()
    present = counts > 0
    norm = weights[present].mean() if bool(present.any()) else weights.mean()
    weights = weights / norm.clamp_min(1e-12)
    weights = weights.clamp(float(min_weight), float(max_weight)).float()
    return weights


def semantic_targets_for_sample(record: dict) -> tuple[torch.Tensor, torch.Tensor]:
    pool = full_edit_supervision(record)
    return pool["flat_indices"].long(), pool["result_slots"].long()


def absolute_future_semantic_loss(
    sparse_logits_per_sample: list[torch.Tensor],
    records: list[dict],
    *,
    class_weights: torch.Tensor | None = None,
    lovasz_weight: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    if lovasz_weight < 0:
        raise ValueError("lovasz_weight must be non-negative")
    if len(sparse_logits_per_sample) != len(records):
        raise ValueError("one semantic-logit tensor is required per target record")

    logits_rows = []
    target_rows = []
    for logits, rec in zip(sparse_logits_per_sample, records):
        _, target = semantic_targets_for_sample(rec)
        if int(logits.shape[0]) != int(target.numel()):
            raise ValueError("sparse decoder logits/targets length mismatch")
        if target.numel() == 0:
            continue
        logits_rows.append(collapse_occ_logits_to_dynamic(logits))
        target_rows.append(target.to(device=logits.device, dtype=torch.long))

    if not logits_rows:
        if sparse_logits_per_sample:
            zero = sparse_logits_per_sample[0].sum() * 0.0
        else:
            zero = torch.tensor(0.0)
        return zero, {
            "ce": 0.0,
            "lovasz": 0.0,
            "accuracy": float("nan"),
            "dynamic_accuracy": float("nan"),
            "background_false_dynamic_rate": float("nan"),
            "num_supervised_voxels": 0,
        }

    logits = torch.cat(logits_rows, dim=0)
    target = torch.cat(target_rows, dim=0)
    weight = None
    if class_weights is not None:
        weight = class_weights.to(device=logits.device, dtype=torch.float32)
        if tuple(weight.shape) != (NUM_RESULT_CLASSES,):
            raise ValueError("class_weights must have background+8 dynamic entries")
    ce = F.cross_entropy(logits.float(), target, weight=weight)
    probs = F.softmax(logits.float(), dim=-1)
    lovasz = lovasz_softmax_flat(probs, target)
    loss = ce + float(lovasz_weight) * lovasz

    with torch.no_grad():
        pred = logits.argmax(dim=-1)
        acc = (pred == target).float().mean()
        dyn = target > 0
        bg = ~dyn
        dyn_acc = (
            (pred[dyn] == target[dyn]).float().mean()
            if bool(dyn.any()) else torch.tensor(float("nan"), device=logits.device)
        )
        bg_false = (
            (pred[bg] > 0).float().mean()
            if bool(bg.any()) else torch.tensor(float("nan"), device=logits.device)
        )
    return loss, {
        "ce": float(ce.detach().cpu()),
        "lovasz": float(lovasz.detach().cpu()),
        "accuracy": float(acc.detach().cpu()),
        "dynamic_accuracy": float(dyn_acc.detach().cpu()),
        "background_false_dynamic_rate": float(bg_false.detach().cpu()),
        "num_supervised_voxels": int(target.numel()),
    }
