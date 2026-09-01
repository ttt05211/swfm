"""Anchor-relative sparse edit supervision for P0-F8.

P0-F8 predicts actions relative to the exact Strong-W2Det occupancy anchor:
KEEP, CLEAR, or WRITE one of the eight motion-capable semantic classes.  The
training sidecar stores every dynamic EDIT voxel and a compact pool of hard KEEP
voxels.  Balanced sampling keeps all edits while subsampling KEEP examples so
majority-class anchor preservation cannot swamp repair gradients.

Lovasz supervision is applied to the *resulting* 9-way dynamic semantic
probability after marginalizing the action probabilities through the anchor,
rather than to action IDs themselves.  This makes the IoU surrogate match the
occupancy that would exist after applying the predicted edit.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS
from .occfm_io import file_sha256
from .semantic_repair import collapse_dynamic_logits, latent_support_to_occ_bev_np

P0_F8_EDIT_CACHE_VERSION = "p0_f8_anchor_relative_edit_targets_v1"
DYNAMIC_IDS = tuple(int(x) for x in DYNAMIC_CLASS_IDS)
DYNAMIC_TO_SLOT = {cid: i + 1 for i, cid in enumerate(DYNAMIC_IDS)}
SLOT_TO_DYNAMIC = {v: k for k, v in DYNAMIC_TO_SLOT.items()}
NUM_RESULT_CLASSES = 1 + len(DYNAMIC_IDS)  # background + 8 dynamic

KEEP = 0
CLEAR = 1
WRITE_OFFSET = 2
NUM_ACTIONS = WRITE_OFFSET + len(DYNAMIC_IDS)  # KEEP, CLEAR, WRITE x8
OCC_SHAPE = (6, 200, 200, 16)
BEV_SHAPE = OCC_SHAPE[:-1]


def _semantic_slots(occ: np.ndarray) -> np.ndarray:
    slots = np.zeros(occ.shape, dtype=np.uint8)
    for cid, slot in DYNAMIC_TO_SLOT.items():
        slots[occ == int(cid)] = np.uint8(slot)
    return slots


def _near_bev(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """Small deterministic BEV dilation without scipy."""
    if mask.ndim != 3:
        raise ValueError("BEV mask must be [T,X,Y]")
    out = mask.copy()
    if radius <= 0:
        return out
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            src_x0 = max(0, -dx)
            src_x1 = mask.shape[1] - max(0, dx)
            src_y0 = max(0, -dy)
            src_y1 = mask.shape[2] - max(0, dy)
            dst_x0 = max(0, dx)
            dst_x1 = mask.shape[1] - max(0, -dx)
            dst_y0 = max(0, dy)
            dst_y1 = mask.shape[2] - max(0, -dy)
            out[:, dst_x0:dst_x1, dst_y0:dst_y1] |= mask[:, src_x0:src_x1, src_y0:src_y1]
    return out


def _stable_subsample(indices: np.ndarray, limit: int, key: str) -> np.ndarray:
    if limit <= 0 or indices.size <= limit:
        return indices
    digest = hashlib.sha256(str(key).encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little", signed=False)
    rng = np.random.default_rng(seed)
    pick = rng.choice(indices.size, size=int(limit), replace=False)
    pick.sort()
    return indices[pick]


def _moving_flags_for_flat(flat_indices: np.ndarray, moving_support_bev: np.ndarray) -> np.ndarray:
    if flat_indices.size == 0:
        return np.zeros((0,), dtype=np.bool_)
    stride = OCC_SHAPE[1] * OCC_SHAPE[2] * OCC_SHAPE[3]
    rem = flat_indices.astype(np.int64, copy=False) % stride
    t = flat_indices.astype(np.int64, copy=False) // stride
    x = rem // (OCC_SHAPE[2] * OCC_SHAPE[3])
    rem = rem % (OCC_SHAPE[2] * OCC_SHAPE[3])
    y = rem // OCC_SHAPE[3]
    return moving_support_bev[t, x, y].astype(np.bool_, copy=False)


def build_anchor_relative_edit_record(
    *,
    sample_id: str,
    scene_name: str,
    gt_future_occ,
    strong_anchor_occ,
    write_support_latent: torch.Tensor,
    moving_support_bev,
    easy_keep_limit: int = 4096,
) -> dict:
    """Build exact deployment-aligned KEEP/CLEAR/WRITE supervision.

    All edit voxels inside the causal MSP support are retained.  KEEP examples
    contain every correct dynamic anchor voxel plus a compact deterministic pool
    of background KEEP voxels near edits (then, if needed, elsewhere in support).
    """
    gt = np.asarray(gt_future_occ)
    anchor = np.asarray(strong_anchor_occ)
    moving = np.asarray(moving_support_bev, dtype=bool)
    if tuple(gt.shape) != OCC_SHAPE or tuple(anchor.shape) != OCC_SHAPE:
        raise ValueError(f"GT/anchor must be {OCC_SHAPE}, got {gt.shape}/{anchor.shape}")
    if tuple(moving.shape) != BEV_SHAPE:
        raise ValueError(f"moving support must be {BEV_SHAPE}, got {moving.shape}")

    support_bev = latent_support_to_occ_bev_np(write_support_latent)
    support = np.broadcast_to(support_bev[..., None], OCC_SHAPE)
    a_slot = _semantic_slots(anchor)
    g_slot = _semantic_slots(gt)

    union = support & ((a_slot > 0) | (g_slot > 0))
    keep_dynamic = union & (a_slot == g_slot)
    edit = union & (a_slot != g_slot)

    edit_flat = np.flatnonzero(edit.reshape(-1)).astype(np.int32, copy=False)
    edit_a = a_slot.reshape(-1)[edit_flat.astype(np.int64)].astype(np.uint8, copy=False)
    edit_g = g_slot.reshape(-1)[edit_flat.astype(np.int64)].astype(np.uint8, copy=False)
    edit_actions = np.where(
        edit_g == 0,
        np.uint8(CLEAR),
        (WRITE_OFFSET + edit_g.astype(np.int16) - 1).astype(np.uint8),
    ).astype(np.uint8, copy=False)

    hard_keep_flat = np.flatnonzero(keep_dynamic.reshape(-1)).astype(np.int32, copy=False)
    hard_keep_a = a_slot.reshape(-1)[hard_keep_flat.astype(np.int64)].astype(np.uint8, copy=False)

    # Background KEEP examples are useful because inference evaluates every voxel
    # inside the write support. Prefer locations near an actual edit so these are
    # hard negatives, then fill from the remaining support if the pool is small.
    edit_bev = edit.any(axis=-1)
    near = _near_bev(edit_bev, radius=1)
    bg_keep = support & (a_slot == 0) & (g_slot == 0)
    near_bg = np.flatnonzero((bg_keep & near[..., None]).reshape(-1)).astype(np.int32, copy=False)
    near_bg = _stable_subsample(near_bg, int(easy_keep_limit), f"{sample_id}:near")
    remaining = max(0, int(easy_keep_limit) - int(near_bg.size))
    if remaining > 0:
        far_bg_mask = bg_keep & ~near[..., None]
        far_bg = np.flatnonzero(far_bg_mask.reshape(-1)).astype(np.int32, copy=False)
        far_bg = _stable_subsample(far_bg, remaining, f"{sample_id}:far")
        easy_keep_flat = np.concatenate([near_bg, far_bg]).astype(np.int32, copy=False)
    else:
        easy_keep_flat = near_bg

    keep_flat = np.concatenate([hard_keep_flat, easy_keep_flat]).astype(np.int32, copy=False)
    keep_a = np.concatenate([
        hard_keep_a,
        np.zeros(easy_keep_flat.size, dtype=np.uint8),
    ])
    # priority 2: true-moving correct dynamic anchor; 1: other correct dynamic;
    # 0: sampled background KEEP.
    hard_moving = _moving_flags_for_flat(hard_keep_flat, moving)
    keep_priority = np.concatenate([
        np.where(hard_moving, 2, 1).astype(np.uint8),
        np.zeros(easy_keep_flat.size, dtype=np.uint8),
    ])

    rec = {
        "sample_id": str(sample_id),
        "scene_name": str(scene_name),
        "edit_flat_indices": torch.from_numpy(edit_flat.copy()),
        "edit_actions": torch.from_numpy(edit_actions.copy()),
        "edit_anchor_slots": torch.from_numpy(edit_a.copy()),
        "edit_result_slots": torch.from_numpy(edit_g.copy()),
        "edit_moving": torch.from_numpy(_moving_flags_for_flat(edit_flat, moving).copy()),
        "keep_flat_indices": torch.from_numpy(keep_flat.copy()),
        "keep_anchor_slots": torch.from_numpy(keep_a.copy()),
        "keep_priority": torch.from_numpy(keep_priority.copy()),
    }
    validate_edit_record(rec)
    return rec


def validate_edit_record(record: dict) -> bool:
    required = (
        "sample_id",
        "scene_name",
        "edit_flat_indices",
        "edit_actions",
        "edit_anchor_slots",
        "edit_result_slots",
        "edit_moving",
        "keep_flat_indices",
        "keep_anchor_slots",
        "keep_priority",
    )
    missing = [k for k in required if k not in record]
    if missing:
        raise KeyError(f"P0-F8 edit record missing {missing}")
    for k in required[2:]:
        if not torch.is_tensor(record[k]) or record[k].ndim != 1:
            raise ValueError(f"P0-F8 {k} must be a 1D tensor")
    ne = int(record["edit_flat_indices"].numel())
    nk = int(record["keep_flat_indices"].numel())
    for k in ("edit_actions", "edit_anchor_slots", "edit_result_slots", "edit_moving"):
        if int(record[k].numel()) != ne:
            raise ValueError(f"P0-F8 {k} length mismatch")
    for k in ("keep_anchor_slots", "keep_priority"):
        if int(record[k].numel()) != nk:
            raise ValueError(f"P0-F8 {k} length mismatch")
    flat_size = int(np.prod(OCC_SHAPE))
    for k in ("edit_flat_indices", "keep_flat_indices"):
        idx = record[k].long()
        if idx.numel():
            if int(idx.min()) < 0 or int(idx.max()) >= flat_size:
                raise ValueError(f"P0-F8 {k} out of range")
            if torch.unique(idx).numel() != idx.numel():
                raise ValueError(f"P0-F8 {k} contains duplicates")
    if ne:
        a = record["edit_actions"].long()
        if int(a.min()) < CLEAR or int(a.max()) >= NUM_ACTIONS:
            raise ValueError("P0-F8 edit actions must be CLEAR or WRITE")
        rs = record["edit_result_slots"].long()
        if int(rs.min()) < 0 or int(rs.max()) >= NUM_RESULT_CLASSES:
            raise ValueError("P0-F8 result slots out of range")
    for k in ("edit_anchor_slots", "keep_anchor_slots"):
        x = record[k].long()
        if x.numel() and (int(x.min()) < 0 or int(x.max()) >= NUM_RESULT_CLASSES):
            raise ValueError(f"P0-F8 {k} out of range")
    p = record["keep_priority"].long()
    if p.numel() and (int(p.min()) < 0 or int(p.max()) > 2):
        raise ValueError("P0-F8 keep priority must be 0..2")
    return True


def _pick_keep_indices(
    priority: torch.Tensor,
    count: int,
    *,
    generator: torch.Generator | None,
    deterministic: bool,
) -> torch.Tensor:
    if count <= 0 or priority.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    selected = []
    remaining = int(count)
    for level in (2, 1, 0):
        pool = torch.nonzero(priority.long() == level, as_tuple=False).flatten()
        if pool.numel() == 0:
            continue
        take = min(remaining, int(pool.numel()))
        if deterministic or take == int(pool.numel()):
            chosen = pool[:take]
        else:
            order = torch.randperm(int(pool.numel()), generator=generator)[:take]
            chosen = pool.index_select(0, order)
        selected.append(chosen)
        remaining -= take
        if remaining <= 0:
            break
    if not selected:
        return torch.empty(0, dtype=torch.long)
    return torch.cat(selected, dim=0)


def select_balanced_edit_supervision(
    record: dict,
    *,
    keep_ratio: float = 1.0,
    generator: torch.Generator | None = None,
    deterministic: bool = False,
    keep_when_no_edit: int = 64,
) -> dict:
    """Keep every EDIT and sample hard KEEP examples at a bounded ratio."""
    validate_edit_record(record)
    if keep_ratio < 0:
        raise ValueError("keep_ratio must be non-negative")
    ne = int(record["edit_flat_indices"].numel())
    target_keep = int(math.ceil(float(ne) * float(keep_ratio)))
    if ne == 0:
        target_keep = int(keep_when_no_edit)
    target_keep = min(target_keep, int(record["keep_flat_indices"].numel()))
    pick = _pick_keep_indices(
        record["keep_priority"].cpu(),
        target_keep,
        generator=generator,
        deterministic=deterministic,
    )

    eidx = record["edit_flat_indices"].long()
    ea = record["edit_anchor_slots"].long()
    er = record["edit_result_slots"].long()
    eact = record["edit_actions"].long()
    emov = record["edit_moving"].bool()
    kidx = record["keep_flat_indices"].long().index_select(0, pick)
    ka = record["keep_anchor_slots"].long().index_select(0, pick)

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
        "num_edits": ne,
        "num_keeps": target_keep,
    }


def horizon_from_flat_indices(flat_indices: torch.Tensor) -> torch.Tensor:
    stride = OCC_SHAPE[1] * OCC_SHAPE[2] * OCC_SHAPE[3]
    return torch.as_tensor(flat_indices, dtype=torch.long) // int(stride)


def action_probs_to_result_probs(
    action_logits: torch.Tensor,
    anchor_slots: torch.Tensor,
) -> torch.Tensor:
    """Marginalize KEEP/CLEAR/WRITE into background+8 dynamic result classes."""
    if action_logits.ndim != 2 or action_logits.shape[-1] != NUM_ACTIONS:
        raise ValueError(f"action logits must be [N,{NUM_ACTIONS}]")
    anchor = anchor_slots.to(device=action_logits.device, dtype=torch.long).reshape(-1)
    if anchor.numel() != action_logits.shape[0]:
        raise ValueError("anchor slot length mismatch")
    if anchor.numel() and (int(anchor.min()) < 0 or int(anchor.max()) >= NUM_RESULT_CLASSES):
        raise ValueError("anchor slots out of range")
    p = F.softmax(action_logits.float(), dim=-1)
    result = p.new_zeros((p.shape[0], NUM_RESULT_CLASSES))
    # CLEAR always maps to background.
    result[:, 0] = result[:, 0] + p[:, CLEAR]
    # WRITE slot s maps to result slot s.
    for slot in range(1, NUM_RESULT_CLASSES):
        action = WRITE_OFFSET + slot - 1
        result[:, slot] = result[:, slot] + p[:, action]
    # KEEP maps to the current anchor semantic slot.
    result.scatter_add_(1, anchor[:, None], p[:, KEEP:KEEP + 1])
    return result


def lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    p = int(gt_sorted.numel())
    if p == 0:
        return gt_sorted
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1.0 - gt_sorted.float()).cumsum(0)
    jaccard = 1.0 - intersection / union.clamp_min(1e-12)
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[:-1]
    return jaccard


def lovasz_softmax_flat(probas: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Multiclass Lovasz-Softmax over present classes on flattened samples."""
    if probas.ndim != 2:
        raise ValueError("Lovasz probabilities must be [N,C]")
    labels = labels.to(device=probas.device, dtype=torch.long).reshape(-1)
    if labels.numel() != probas.shape[0]:
        raise ValueError("Lovasz labels length mismatch")
    if labels.numel() == 0:
        return probas.sum() * 0.0
    losses = []
    for c in range(int(probas.shape[1])):
        fg = (labels == c).float()
        if float(fg.sum().detach().cpu()) <= 0:
            continue
        errors = (fg - probas[:, c].float()).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg.index_select(0, perm)
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg_sorted)))
    if not losses:
        return probas.sum() * 0.0
    return torch.stack(losses).mean()


def anchor_relative_edit_loss(
    action_logits: torch.Tensor,
    actions: torch.Tensor,
    anchor_slots: torch.Tensor,
    result_slots: torch.Tensor,
    *,
    lovasz_weight: float = 0.5,
) -> tuple[torch.Tensor, dict]:
    if lovasz_weight < 0:
        raise ValueError("lovasz_weight must be non-negative")
    actions = actions.to(device=action_logits.device, dtype=torch.long)
    anchor_slots = anchor_slots.to(device=action_logits.device, dtype=torch.long)
    result_slots = result_slots.to(device=action_logits.device, dtype=torch.long)
    if actions.numel() == 0:
        zero = action_logits.sum() * 0.0
        return zero, {
            "ce": 0.0,
            "lovasz": 0.0,
            "accuracy": float("nan"),
            "edit_accuracy": float("nan"),
            "false_edit_rate": float("nan"),
        }
    ce = F.cross_entropy(action_logits.float(), actions, reduction="mean")
    result_probs = action_probs_to_result_probs(action_logits, anchor_slots)
    lovasz = lovasz_softmax_flat(result_probs, result_slots)
    loss = ce + float(lovasz_weight) * lovasz
    with torch.no_grad():
        pred = action_logits.argmax(dim=-1)
        edit = actions != KEEP
        keep = ~edit
        acc = (pred == actions).float().mean()
        edit_acc = (pred[edit] == actions[edit]).float().mean() if bool(edit.any()) else torch.tensor(float("nan"))
        false_edit = (pred[keep] != KEEP).float().mean() if bool(keep.any()) else torch.tensor(float("nan"))
    return loss, {
        "ce": float(ce.detach().cpu()),
        "lovasz": float(lovasz.detach().cpu()),
        "accuracy": float(acc.detach().cpu()),
        "edit_accuracy": float(edit_acc.detach().cpu()),
        "false_edit_rate": float(false_edit.detach().cpu()),
    }


def apply_anchor_relative_actions(
    anchor_occ: np.ndarray,
    flat_indices: np.ndarray,
    actions: np.ndarray,
    *,
    free_label: int = 17,
) -> np.ndarray:
    """Apply discrete P0-F8 actions to an exact Strong-W2Det anchor copy."""
    anchor = np.asarray(anchor_occ)
    if tuple(anchor.shape) != OCC_SHAPE:
        raise ValueError(f"anchor must be {OCC_SHAPE}")
    idx = np.asarray(flat_indices, dtype=np.int64).reshape(-1)
    act = np.asarray(actions, dtype=np.int64).reshape(-1)
    if idx.size != act.size:
        raise ValueError("action index/value length mismatch")
    out = anchor.copy()
    flat = out.reshape(-1)
    if idx.size == 0:
        return out
    if idx.min() < 0 or idx.max() >= flat.size:
        raise ValueError("action index out of range")
    dynamic = np.asarray(DYNAMIC_IDS, dtype=np.int64)
    clear = act == CLEAR
    if clear.any():
        ci = idx[clear]
        current = flat[ci]
        can_clear = np.isin(current, dynamic)
        flat[ci[can_clear]] = int(free_label)
    for slot in range(1, NUM_RESULT_CLASSES):
        action = WRITE_OFFSET + slot - 1
        mask = act == action
        if mask.any():
            flat[idx[mask]] = int(SLOT_TO_DYNAMIC[slot])
    # KEEP and invalid CLEAR-on-static are exact no-ops by construction.
    return out


class EditTargetCache:
    def __init__(self, path):
        self.path = Path(path)
        obj = torch.load(self.path, map_location="cpu", weights_only=False)
        if obj.get("version") != P0_F8_EDIT_CACHE_VERSION:
            raise ValueError(f"unsupported P0-F8 edit cache {obj.get('version')}")
        self.metadata = obj.get("metadata") or {}
        records = obj.get("records") or []
        if not records:
            raise RuntimeError("empty P0-F8 edit target cache")
        self.records = {}
        for rec in records:
            validate_edit_record(rec)
            sid = str(rec["sample_id"])
            if sid in self.records:
                raise RuntimeError(f"duplicate P0-F8 edit sample_id {sid}")
            self.records[sid] = rec

    def __len__(self):
        return len(self.records)

    def get_batch(self, sample_ids) -> list[dict]:
        out = []
        for sid in sample_ids:
            key = str(sid)
            if key not in self.records:
                raise KeyError(f"P0-F8 edit target cache missing {key}")
            out.append(self.records[key])
        return out

    def validate_source_cache(self, wm_cache_root) -> None:
        index_path = Path(wm_cache_root) / "index.json"
        expected = self.metadata.get("source_wm_cache_index_sha256")
        if not expected:
            raise RuntimeError("P0-F8 edit cache lacks source cache SHA256")
        actual = file_sha256(index_path)
        if actual != expected:
            raise RuntimeError("P0-F8 edit sidecar was not built from this exact WM cache")
        ids = self.metadata.get("source_sample_ids")
        if ids is not None and set(map(str, ids)) != set(self.records):
            raise RuntimeError("P0-F8 edit sidecar metadata/sample IDs disagree")
