"""V20 supervision, set matching and checkpoint contracts."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .local_st_world_model import SEMANTIC_CLASSES
from .v20_history_world import DYNAMIC_IDS, FREE_LABEL, V20_PROTOCOL
from .v20_scene_model import V20HistoryWorldModel, V20SceneConfig

CHECKPOINT_PROTOCOL = "p0_f9_v20_checkpoint_v1"

_DYNAMIC_TO_LOCAL = {int(cid): i for i, cid in enumerate(DYNAMIC_IDS)}


def dynamic_global_to_local(class_id: torch.Tensor) -> torch.Tensor:
    """Map frozen Occ3D dynamic semantic IDs to Birth-local [0,K) IDs."""
    out = torch.full_like(class_id.long(), -1)
    for cid, local in _DYNAMIC_TO_LOCAL.items():
        out[class_id.long() == int(cid)] = int(local)
    if bool((out < 0).any()):
        bad = torch.unique(class_id[out < 0]).detach().cpu().tolist()
        raise ValueError(f"Birth target contains non-dynamic class IDs: {bad}")
    return out


def dynamic_local_to_global(class_id: torch.Tensor) -> torch.Tensor:
    """Inverse map for Birth rendering/evaluation; no-object is not accepted."""
    lut = torch.as_tensor(DYNAMIC_IDS, dtype=torch.long, device=class_id.device)
    x = class_id.long()
    if bool(((x < 0) | (x >= len(DYNAMIC_IDS))).any()):
        raise ValueError("Birth local class index outside dynamic taxonomy")
    return lut[x]


def static_semantic_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    *,
    class_weights: torch.Tensor | None = None,
    dynamic_class_ids: Sequence[int] = DYNAMIC_IDS,
) -> torch.Tensor:
    """Semantic CE over future-observed static/free voxels only."""
    if logits.ndim != 5:
        raise ValueError("static logits must be [B,C,X,Y,Z]")
    if target.shape != valid.shape or logits.shape[0] != target.shape[0] or logits.shape[2:] != target.shape[1:]:
        raise ValueError("static target/logit shape mismatch")
    mask = valid.bool()
    if not bool(mask.any()):
        return logits.sum() * 0.0
    rows = logits.permute(0, 2, 3, 4, 1)[mask].clone()
    # Static branch is never allowed to select a dynamic semantic class.
    if dynamic_class_ids:
        rows[:, torch.as_tensor(tuple(dynamic_class_ids), device=rows.device)] = torch.finfo(rows.dtype).min
    return F.cross_entropy(rows, target.long()[mask], weight=class_weights)


def decode_static_logits(logits: torch.Tensor) -> torch.Tensor:
    """Argmax static/free semantics while structurally excluding dynamic IDs."""
    if logits.ndim < 2 or logits.shape[1] != SEMANTIC_CLASSES:
        raise ValueError("static logits must have semantic class dimension at dim=1")
    masked = logits.clone()
    dyn = torch.as_tensor(DYNAMIC_IDS, dtype=torch.long, device=masked.device)
    masked[:, dyn] = torch.finfo(masked.dtype).min
    return masked.argmax(dim=1)


def dormant_source_loss(
    outputs: Mapping[str, torch.Tensor],
    *,
    target_xy_m: torch.Tensor,
    target_yaw_rad: torch.Tensor,
    target_exists: torch.Tensor,
    yaw_valid: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    xy = outputs["residual_xy_m"]
    yaw = outputs["yaw_delta_rad"]
    exist = outputs["existence_logits"]
    if xy.shape != target_xy_m.shape or exist.shape != target_exists.shape:
        raise ValueError("Dormant target shape mismatch")
    e = F.binary_cross_entropy_with_logits(exist, target_exists.to(exist.dtype))
    mask = target_exists.bool()
    trans = F.smooth_l1_loss(xy[mask], target_xy_m.to(xy.dtype)[mask]) if bool(mask.any()) else xy.sum() * 0.0
    ym = mask & yaw_valid.bool()
    if bool(ym.any()):
        delta = torch.atan2(torch.sin(yaw - target_yaw_rad), torch.cos(yaw - target_yaw_rad))
        yl = (1.0 - torch.cos(delta[ym])).mean()
    else:
        yl = yaw.sum() * 0.0
    total = e + trans + 0.25 * yl
    return total, {
        "loss": float(total.detach().cpu()),
        "exist_bce": float(e.detach().cpu()),
        "translation": float(trans.detach().cpu()),
        "yaw": float(yl.detach().cpu()),
    }


def choose_birth_query_count(
    births_per_window: Sequence[int],
    *,
    max_truncation_fraction: float = 0.01,
    minimum_q: int = 1,
) -> dict[str, float | int]:
    """Choose the smallest Q whose GT truncation fraction is within budget."""
    arr = np.asarray(list(births_per_window), dtype=np.int64)
    if arr.ndim != 1 or len(arr) == 0 or bool((arr < 0).any()):
        raise ValueError("births_per_window must be non-negative non-empty")
    total = int(arr.sum())
    max_n = int(arr.max())
    chosen = max(int(minimum_q), 1)
    for q in range(chosen, max(max_n, chosen) + 1):
        truncated = int(np.maximum(arr - q, 0).sum())
        frac = float(truncated / max(total, 1))
        if frac <= float(max_truncation_fraction):
            chosen = q
            break
        chosen = q
    truncated = int(np.maximum(arr - chosen, 0).sum())
    return {
        "Q": int(chosen),
        "windows": int(len(arr)),
        "birth_instances": total,
        "truncated_birth_instances": truncated,
        "truncated_gt_fraction": float(truncated / max(total, 1)),
        "windows_0": int((arr == 0).sum()),
        "windows_1": int((arr == 1).sum()),
        "windows_2": int((arr == 2).sum()),
        "windows_ge3": int((arr >= 3).sum()),
    }


def _pair_cost(
    pred_class_logits: torch.Tensor,
    pred_exist_logits: torch.Tensor,
    pred_traj: torch.Tensor,
    target_class: torch.Tensor,
    target_exist: torch.Tensor,
    target_traj: torch.Tensor,
) -> torch.Tensor:
    """[Q,N] DETR-style matching cost for persistent six-frame birth tracks."""
    Q = pred_class_logits.shape[0]
    N = target_class.shape[0]
    if N == 0:
        return pred_class_logits.new_empty((Q, 0))
    prob = pred_class_logits.softmax(-1)
    target_local = dynamic_global_to_local(target_class)
    cls = -prob[:, target_local.long()]  # Q,N
    pe = torch.sigmoid(pred_exist_logits)[:, None, :]
    te = target_exist.to(pe.dtype)[None, :, :]
    exist = (pe - te).abs().mean(dim=-1)
    ptr = pred_traj[:, None, :, :2]
    ttr = target_traj.to(ptr.dtype)[None, :, :, :2]
    mask = te.unsqueeze(-1)
    denom = mask.sum(dim=(-2, -1)).clamp_min(1.0)
    traj = ((ptr - ttr).abs() * mask).sum(dim=(-2, -1)) / denom
    return cls + exist + 0.25 * traj


def hungarian_birth_match(
    outputs: Mapping[str, torch.Tensor],
    targets: Sequence[Mapping[str, torch.Tensor]],
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """CPU Hungarian indices per batch; GT identities are targets only."""
    from scipy.optimize import linear_sum_assignment

    matches = []
    for b, tgt in enumerate(targets):
        n = int(tgt["class_id"].numel())
        if n == 0:
            z = torch.empty(0, dtype=torch.long, device=outputs["class_logits"].device)
            matches.append((z, z))
            continue
        cost = _pair_cost(
            outputs["class_logits"][b],
            outputs["existence_logits"][b],
            outputs["trajectory_xy_yaw"][b],
            tgt["class_id"],
            tgt["existence"],
            tgt["trajectory_xy_yaw"],
        )
        qidx, tidx = linear_sum_assignment(cost.detach().float().cpu().numpy())
        matches.append((
            torch.as_tensor(qidx, dtype=torch.long, device=cost.device),
            torch.as_tensor(tidx, dtype=torch.long, device=cost.device),
        ))
    return matches


def birth_set_loss(
    outputs: Mapping[str, torch.Tensor],
    targets: Sequence[Mapping[str, torch.Tensor]],
    *,
    no_object_class: int,
    shape_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Set loss with one query tied to one object through all future horizons."""
    B, Q, _ = outputs["class_logits"].shape
    if len(targets) != B:
        raise ValueError("target batch mismatch")
    matches = hungarian_birth_match(outputs, targets)
    cls_tgt = torch.full(
        (B, Q), int(no_object_class), dtype=torch.long, device=outputs["class_logits"].device
    )
    exist_loss = outputs["existence_logits"].sum() * 0.0
    traj_loss = outputs["trajectory_xy_yaw"].sum() * 0.0
    shape_loss = outputs["shape_logits"].sum() * 0.0
    matched = 0
    for b, (qi, ti) in enumerate(matches):
        if qi.numel() == 0:
            continue
        tgt = targets[b]
        cls_tgt[b, qi] = dynamic_global_to_local(tgt["class_id"].long()[ti])
        ex = tgt["existence"].to(outputs["existence_logits"].dtype)[ti]
        exist_loss = exist_loss + F.binary_cross_entropy_with_logits(
            outputs["existence_logits"][b, qi], ex
        )
        mask = ex.bool().unsqueeze(-1)
        ptr = outputs["trajectory_xy_yaw"][b, qi]
        ttr = tgt["trajectory_xy_yaw"].to(ptr.dtype)[ti]
        if bool(mask.any()):
            traj_loss = traj_loss + F.smooth_l1_loss(ptr[mask.expand_as(ptr)], ttr[mask.expand_as(ttr)])
        if "shape" in tgt:
            shape_loss = shape_loss + F.binary_cross_entropy_with_logits(
                outputs["shape_logits"][b, qi],
                tgt["shape"].to(outputs["shape_logits"].dtype)[ti],
            )
        matched += int(qi.numel())
    denom = max(B, 1)
    cls_loss = F.cross_entropy(outputs["class_logits"].reshape(B * Q, -1), cls_tgt.reshape(-1))
    total = cls_loss + exist_loss / denom + traj_loss / denom + float(shape_weight) * shape_loss / denom
    return total, {
        "loss": float(total.detach().cpu()),
        "class_ce": float(cls_loss.detach().cpu()),
        "matched_births": int(matched),
    }


def checkpoint_payload(
    model: V20HistoryWorldModel,
    *,
    stage: str,
    v18_checkpoint: str,
    thresholds: Mapping[str, float],
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Self-describing V20 checkpoint; V18 weights remain external/read-only."""
    return {
        "protocol": CHECKPOINT_PROTOCOL,
        "v20_protocol": V20_PROTOCOL,
        "stage": str(stage),
        "scene_config": asdict(model.cfg),
        "v18_checkpoint": str(v18_checkpoint),
        "thresholds": dict(thresholds),
        "model": model.state_dict(),
        "extra": dict(extra or {}),
    }


def load_v20_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[V20HistoryWorldModel, dict[str, object]]:
    obj = torch.load(Path(path), map_location=map_location, weights_only=False)
    if obj.get("protocol") != CHECKPOINT_PROTOCOL:
        raise RuntimeError(f"unexpected V20 checkpoint protocol: {obj.get('protocol')}")
    cfg = V20SceneConfig(**obj["scene_config"])
    model = V20HistoryWorldModel(cfg)
    model.load_state_dict(obj["model"], strict=True)
    return model, obj
