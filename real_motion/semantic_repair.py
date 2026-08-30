"""Sparse decoder-aware semantic supervision for P0-F6.

The final P0-F4/P0-F5 write contract only cares whether a voxel should contain
one of the eight motion-capable semantic classes.  P0-F6 therefore collapses
all non-dynamic semantics (including free) into one background class and keeps
the eight dynamic classes distinct.  Supervision is restricted to the causal
MSP write support and only to voxels where either the Strong-W2Det anchor decode
or GT contains a dynamic semantic.  This directly trains keep/remove/create
without letting empty vertical columns dominate the loss.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS
from .occfm_io import file_sha256

P0_F6_SEMANTIC_CACHE_VERSION = "p0_f6_sparse_dynamic_semantic_targets_v1"
DYNAMIC_IDS = tuple(int(x) for x in DYNAMIC_CLASS_IDS)
DYNAMIC_TO_SLOT = {cid: i + 1 for i, cid in enumerate(DYNAMIC_IDS)}
SLOT_TO_DYNAMIC = {v: k for k, v in DYNAMIC_TO_SLOT.items()}
NUM_REPAIR_CLASSES = 1 + len(DYNAMIC_IDS)
OCC_SHAPE = (6, 200, 200, 16)


def latent_support_to_occ_bev_np(write_support_latent: torch.Tensor) -> np.ndarray:
    """Exact aligned 50x50 -> 200x200 nearest/block expansion."""
    x = torch.as_tensor(write_support_latent).bool()
    if tuple(x.shape) != (6, 50, 50):
        raise ValueError(f"write support must be [6,50,50], got {tuple(x.shape)}")
    return (
        x.repeat_interleave(4, dim=-2)
        .repeat_interleave(4, dim=-1)
        .cpu()
        .numpy()
        .astype(bool)
    )


def build_sparse_semantic_record(
    *,
    sample_id: str,
    scene_name: str,
    gt_future_occ,
    anchor_decoded_occ,
    write_support_latent: torch.Tensor,
) -> dict:
    """Create sparse keep/remove/create supervision for one 6-frame sample."""
    gt = np.asarray(gt_future_occ)
    anchor = np.asarray(anchor_decoded_occ)
    if tuple(gt.shape) != OCC_SHAPE or tuple(anchor.shape) != OCC_SHAPE:
        raise ValueError(
            f"GT/anchor decode must be {OCC_SHAPE}, got {gt.shape}/{anchor.shape}"
        )
    support_bev = latent_support_to_occ_bev_np(write_support_latent)
    support = np.broadcast_to(support_bev[..., None], OCC_SHAPE)

    gt_slots = np.zeros(OCC_SHAPE, dtype=np.uint8)
    for cid, slot in DYNAMIC_TO_SLOT.items():
        gt_slots[gt == int(cid)] = np.uint8(slot)
    gt_dynamic = (gt_slots > 0) & support
    anchor_dynamic = np.isin(anchor, np.asarray(DYNAMIC_IDS)) & support

    gt_flat = np.flatnonzero(gt_dynamic.reshape(-1)).astype(np.int32, copy=False)
    gt_slot = gt_slots.reshape(-1)[gt_flat.astype(np.int64)].astype(np.uint8, copy=False)
    anchor_flat = np.flatnonzero(anchor_dynamic.reshape(-1)).astype(np.int32, copy=False)

    rec = {
        "sample_id": str(sample_id),
        "scene_name": str(scene_name),
        "gt_dynamic_flat_indices": torch.from_numpy(gt_flat.copy()),
        "gt_dynamic_slots": torch.from_numpy(gt_slot.copy()),
        "anchor_dynamic_flat_indices": torch.from_numpy(anchor_flat.copy()),
    }
    validate_sparse_semantic_record(rec)
    return rec


def validate_sparse_semantic_record(record: dict) -> bool:
    required = (
        "sample_id",
        "scene_name",
        "gt_dynamic_flat_indices",
        "gt_dynamic_slots",
        "anchor_dynamic_flat_indices",
    )
    missing = [k for k in required if k not in record]
    if missing:
        raise KeyError(f"P0-F6 semantic record missing {missing}")
    gi = record["gt_dynamic_flat_indices"]
    gs = record["gt_dynamic_slots"]
    ai = record["anchor_dynamic_flat_indices"]
    if not all(torch.is_tensor(x) and x.ndim == 1 for x in (gi, gs, ai)):
        raise ValueError("P0-F6 sparse semantic arrays must be 1D tensors")
    if gi.numel() != gs.numel():
        raise ValueError("GT sparse indices/slots length mismatch")
    flat_size = int(np.prod(OCC_SHAPE))
    for name, idx in (("gt", gi), ("anchor", ai)):
        if idx.numel():
            ii = idx.long()
            if int(ii.min()) < 0 or int(ii.max()) >= flat_size:
                raise ValueError(f"{name} sparse semantic index out of range")
            if torch.unique(ii).numel() != ii.numel():
                raise ValueError(f"{name} sparse semantic indices contain duplicates")
    if gs.numel():
        slots = gs.long()
        if int(slots.min()) < 1 or int(slots.max()) >= NUM_REPAIR_CLASSES:
            raise ValueError("GT sparse semantic slot must be in [1,8]")
    return True


def sparse_union_targets(record: dict, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return union(anchor-dynamic, GT-dynamic) indices and 9-way targets.

    Target 0 means background/non-dynamic; targets 1..8 map to the eight dynamic
    semantic classes. GT dynamic semantics overwrite background at their exact
    sparse coordinates, so anchor-only coordinates explicitly learn removal.
    """
    validate_sparse_semantic_record(record)
    gi = record["gt_dynamic_flat_indices"].to(device=device, dtype=torch.long)
    gs = record["gt_dynamic_slots"].to(device=device, dtype=torch.long)
    ai = record["anchor_dynamic_flat_indices"].to(device=device, dtype=torch.long)
    if gi.numel() == 0 and ai.numel() == 0:
        return (
            torch.empty(0, device=device, dtype=torch.long),
            torch.empty(0, device=device, dtype=torch.long),
        )
    union = torch.unique(torch.cat([gi, ai], dim=0), sorted=True)
    targets = torch.zeros(union.numel(), device=device, dtype=torch.long)
    if gi.numel():
        pos = torch.searchsorted(union, gi)
        if not torch.equal(union[pos], gi):
            raise AssertionError("GT sparse coordinates are not contained in union")
        targets[pos] = gs
    return union, targets


def collapse_dynamic_logits(logits: torch.Tensor) -> torch.Tensor:
    """Collapse 18 Occ3D classes to [background + 8 dynamic] logits."""
    if logits.ndim != 2:
        raise ValueError("semantic logits must be [N,C]")
    num_classes = int(logits.shape[-1])
    if num_classes <= max(DYNAMIC_IDS):
        raise ValueError(f"semantic logits have only {num_classes} classes")
    dynamic_set = set(DYNAMIC_IDS)
    bg_ids = [i for i in range(num_classes) if i not in dynamic_set]
    bg = torch.logsumexp(logits[:, bg_ids].float(), dim=-1, keepdim=True)
    dyn = logits[:, list(DYNAMIC_IDS)].float()
    return torch.cat([bg, dyn], dim=-1)


def sparse_dynamic_semantic_loss(
    sparse_logits: list[torch.Tensor],
    sparse_targets: list[torch.Tensor],
) -> tuple[torch.Tensor, dict]:
    """Mean 9-way CE over the sparse union supervision set."""
    if len(sparse_logits) != len(sparse_targets):
        raise ValueError("semantic logits/targets batch length mismatch")
    logits_rows = []
    target_rows = []
    for logits, targets in zip(sparse_logits, sparse_targets):
        if logits.shape[0] != targets.numel():
            raise ValueError("semantic sparse logit/target length mismatch")
        if targets.numel() == 0:
            continue
        logits_rows.append(collapse_dynamic_logits(logits))
        target_rows.append(targets.long())
    if not logits_rows:
        # Keep a differentiable zero if every selected sample has no semantic
        # repair voxels. The caller may skip such a batch during lambda calibration.
        if sparse_logits:
            zero = sum((x.sum() * 0.0 for x in sparse_logits), start=sparse_logits[0].sum() * 0.0)
        else:
            zero = torch.tensor(0.0)
        return zero, {
            "num_supervised_voxels": 0,
            "num_gt_dynamic_voxels": 0,
            "accuracy": float("nan"),
        }
    logits_all = torch.cat(logits_rows, dim=0)
    targets_all = torch.cat(target_rows, dim=0)
    loss = F.cross_entropy(logits_all, targets_all, reduction="mean")
    with torch.no_grad():
        pred = logits_all.argmax(dim=-1)
        acc = (pred == targets_all).float().mean()
    return loss, {
        "num_supervised_voxels": int(targets_all.numel()),
        "num_gt_dynamic_voxels": int((targets_all > 0).sum().item()),
        "accuracy": float(acc.cpu()),
    }


class SemanticTargetCache:
    """In-memory sparse sidecar keyed by the frozen P0-F5 sample_id."""

    def __init__(self, path):
        self.path = Path(path)
        obj = torch.load(self.path, map_location="cpu", weights_only=False)
        if obj.get("version") != P0_F6_SEMANTIC_CACHE_VERSION:
            raise ValueError(f"unsupported P0-F6 semantic cache {obj.get('version')}")
        self.metadata = obj.get("metadata") or {}
        records = obj.get("records") or []
        if not records:
            raise RuntimeError("empty P0-F6 semantic target cache")
        self.records = {}
        for rec in records:
            validate_sparse_semantic_record(rec)
            sid = str(rec["sample_id"])
            if sid in self.records:
                raise RuntimeError(f"duplicate semantic target sample_id {sid}")
            self.records[sid] = rec

    def __len__(self):
        return len(self.records)

    def get_batch(self, sample_ids) -> list[dict]:
        out = []
        for sid in sample_ids:
            key = str(sid)
            if key not in self.records:
                raise KeyError(f"semantic target cache missing {key}")
            out.append(self.records[key])
        return out

    def validate_source_cache(self, wm_cache_root) -> None:
        index_path = Path(wm_cache_root) / "index.json"
        expected = self.metadata.get("source_wm_cache_index_sha256")
        if not expected:
            raise RuntimeError("semantic target cache lacks source cache SHA256")
        actual = file_sha256(index_path)
        if actual != expected:
            raise RuntimeError(
                "P0-F6 semantic sidecar was not built from this exact P0-F5 cache"
            )
        ids = self.metadata.get("source_sample_ids")
        if ids is not None and set(map(str, ids)) != set(self.records):
            raise RuntimeError("semantic sidecar metadata/sample IDs disagree")
