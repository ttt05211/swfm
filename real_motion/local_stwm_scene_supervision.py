"""Sparse computation of full-scene supervision for V17 Local-STWM.

This module intentionally stays on the V17-STWM research line.  It borrows the
useful *loss contract* from the MT-V1 renderer (constant hard-KTA loss outside a
small differentiable query domain, full-scene normalization) without importing
or coupling to the MT-V1-STPN branch.

The hard evaluation compositor for C is the already-audited A1 contract:

    legacy CLEAR of all Strong/KTA source destinations
    + replacement WRITE in original Strong source order.

The differentiable surrogate mirrors that contract.  For each horizon it only
constructs probabilities on

    KTA source destinations U predicted-source AABBs (+ interpolation halo),

while all other voxels remain the fixed Strong/KTA anchor.  GT support is never
used to choose the query domain.  The CE is nevertheless divided by the full
scene voxel count, so sparse computation does not introduce ROI reweighting.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import json
import threading
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .geometry import OccupancyGrid
from .local_st_world_model_v17 import soft_transport_overlap_loss
from .metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS
from .motion_transport import FUTURE_FRAMES

SCENE_CACHE_VERSION = "p0_f9_v17_scene_supervision_cache_v1"
SCENE_LOSS_CONTRACT = "a1_source_order_sparse_compute_full_scene_ce_v1"
SCENE_QUERY_CONTRACT = "kta_dest_union_predicted_aabb_plus_halo_no_gt_support_v1"
SCENE_FREEZE_CONTRACT = "freeze_history_encoder_train_future_decoder_heads_v1"
SCENE_CLASS_COUNT = 18

_SHARD_LOCK = threading.RLock()


def _transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    p = np.asarray(pts, dtype=np.float64)
    M = np.asarray(T, dtype=np.float64)
    return p @ M[:3, :3].T + M[:3, 3]


def _grid_origin_step(grid: OccupancyGrid) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray([grid.x_min, grid.y_min, grid.z_min], dtype=np.float64),
        np.asarray(grid.voxel_size, dtype=np.float64),
    )


def _flat_from_indices(idx: np.ndarray, grid: OccupancyGrid) -> np.ndarray:
    q = np.asarray(idx, dtype=np.int64)
    if q.size == 0:
        return np.zeros((0,), dtype=np.int64)
    _, Y, Z = [int(x) for x in grid.shape_hwd]
    return (q[:, 0] * Y + q[:, 1]) * Z + q[:, 2]


def _indices_from_flat(flat: torch.Tensor, grid: OccupancyGrid) -> torch.Tensor:
    _, Y, Z = [int(x) for x in grid.shape_hwd]
    f = flat.long()
    x = f // (Y * Z)
    r = f % (Y * Z)
    y = r // Z
    z = r % Z
    return torch.stack((x, y, z), dim=1)


def hard_shifted_source_flats(
    voxel_indices_t0: np.ndarray,
    displacement_xy_t0_m: Sequence[float],
    t0_ego_to_world: np.ndarray,
    future_ego_to_world: np.ndarray,
    *,
    grid: OccupancyGrid = OccupancyGrid(),
) -> np.ndarray:
    """Hard future destination cells for one t0 source under planar translation."""
    idx = np.asarray(voxel_indices_t0, dtype=np.int64)
    if idx.ndim != 2 or idx.shape[1] != 3:
        raise ValueError("voxel_indices_t0 must be [N,3]")
    if not len(idx):
        return np.zeros((0,), dtype=np.int64)
    origin, step = _grid_origin_step(grid)
    p0 = origin[None] + (idx.astype(np.float64) + 0.5) * step[None]
    d = np.asarray(displacement_xy_t0_m, dtype=np.float64)
    if d.shape != (2,):
        raise ValueError("displacement_xy_t0_m must be [2]")
    p0[:, :2] += d[None]
    pw = _transform_points(t0_ego_to_world, p0)
    pf = _transform_points(np.linalg.inv(np.asarray(future_ego_to_world, dtype=np.float64)), pw)
    out = np.floor((pf - origin[None]) / step[None]).astype(np.int64)
    shape = np.asarray(grid.shape_hwd, dtype=np.int64)
    ok = ((out >= 0) & (out < shape[None])).all(axis=1)
    return np.unique(_flat_from_indices(out[ok], grid))


def _predicted_aabb_flats(
    voxel_indices_t0: np.ndarray,
    displacement_xy_t0_m: np.ndarray,
    t0_ego_to_world: np.ndarray,
    future_ego_to_world: np.ndarray,
    *,
    grid: OccupancyGrid,
    halo_voxels: int,
) -> np.ndarray:
    """Conservative query cells for one predicted translated source.

    The bounds depend only on the prediction and causal source geometry.  GT is
    deliberately absent from this function.
    """
    idx = np.asarray(voxel_indices_t0, dtype=np.int64)
    if idx.ndim != 2 or idx.shape[1] != 3 or not len(idx):
        return np.zeros((0,), dtype=np.int64)
    origin, step = _grid_origin_step(grid)
    lo = idx.min(axis=0)
    hi = idx.max(axis=0) + 1
    mins = origin + lo * step
    maxs = origin + hi * step
    corners = np.asarray(
        [[x, y, z] for x in (mins[0], maxs[0]) for y in (mins[1], maxs[1]) for z in (mins[2], maxs[2])],
        dtype=np.float64,
    )
    d = np.asarray(displacement_xy_t0_m, dtype=np.float64)
    corners[:, :2] += d[None]
    pw = _transform_points(t0_ego_to_world, corners)
    pf = _transform_points(np.linalg.inv(np.asarray(future_ego_to_world, dtype=np.float64)), pw)
    center_idx = (pf - origin[None]) / step[None] - 0.5
    ilo = np.floor(center_idx.min(axis=0)).astype(np.int64) - int(halo_voxels)
    ihi = np.ceil(center_idx.max(axis=0)).astype(np.int64) + int(halo_voxels)
    shape = np.asarray(grid.shape_hwd, dtype=np.int64)
    ilo = np.maximum(ilo, 0)
    ihi = np.minimum(ihi, shape - 1)
    if bool((ilo > ihi).any()):
        return np.zeros((0,), dtype=np.int64)
    xx, yy, zz = np.meshgrid(
        np.arange(ilo[0], ihi[0] + 1),
        np.arange(ilo[1], ihi[1] + 1),
        np.arange(ilo[2], ihi[2] + 1),
        indexing="ij",
    )
    return _flat_from_indices(np.stack((xx.ravel(), yy.ravel(), zz.ravel()), axis=1), grid)


def _source_template(voxel_indices_t0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    idx = np.asarray(voxel_indices_t0, dtype=np.int64)
    if idx.ndim != 2 or idx.shape[1] != 3 or not len(idx):
        raise ValueError("source template requires non-empty [N,3] indices")
    lo = idx.min(axis=0) - 1
    hi = idx.max(axis=0) + 1
    arr = np.zeros(tuple((hi - lo + 1).astype(int)), dtype=np.float32)
    q = idx - lo[None]
    arr[q[:, 0], q[:, 1], q[:, 2]] = 1.0
    return lo, arr


def _sample_source_alpha(
    voxel_indices_t0: np.ndarray,
    displacement_xy_t0_m: torch.Tensor,
    t0_ego_to_world: np.ndarray,
    future_ego_to_world: np.ndarray,
    flat_indices: torch.Tensor,
    *,
    grid: OccupancyGrid,
    jitter_voxels: float = 0.25,
) -> torch.Tensor:
    """Inverse-sample one exact t0 source on requested future cells."""
    if flat_indices.numel() == 0:
        return displacement_xy_t0_m.sum() * 0.0 + torch.zeros(
            (0,), device=displacement_xy_t0_m.device, dtype=torch.float32
        )
    lo, arr = _source_template(voxel_indices_t0)
    device = displacement_xy_t0_m.device
    dtype = torch.float32
    # grid_sample 5D input is [N,C,D,H,W] = [Z,Y,X].
    tmpl = torch.as_tensor(arr, device=device, dtype=dtype).permute(2, 1, 0)[None, None]
    qidx = _indices_from_flat(flat_indices, grid).to(device=device)
    origin = torch.tensor([grid.x_min, grid.y_min, grid.z_min], device=device, dtype=dtype)
    step = torch.tensor(grid.voxel_size, device=device, dtype=dtype)
    pf = origin[None] + (qidx.to(dtype) + 0.5) * step[None]
    T = np.linalg.inv(np.asarray(t0_ego_to_world, dtype=np.float64)) @ np.asarray(
        future_ego_to_world, dtype=np.float64
    )
    TT = torch.as_tensor(T, device=device, dtype=dtype)
    p0 = pf @ TT[:3, :3].T + TT[:3, 3]
    d = displacement_xy_t0_m.to(dtype)
    p0 = p0 - torch.stack((d[0], d[1], d[0] * 0.0))[None]

    local_center = (p0 - origin[None]) / step[None] - 0.5 - torch.as_tensor(
        lo, device=device, dtype=dtype
    )[None]
    dims_xyz = torch.tensor(arr.shape, device=device, dtype=dtype)
    norm = 2.0 * (local_center + 0.5) / dims_xyz[None] - 1.0

    def sample(n: torch.Tensor) -> torch.Tensor:
        # Last grid dimension is input W,H,D => X,Y,Z, which is exactly norm xyz.
        g = n.reshape(1, 1, 1, -1, 3)
        return F.grid_sample(
            tmpl, g, mode="bilinear", padding_mode="zeros", align_corners=False
        ).reshape(-1)

    j = float(jitter_voxels)
    if j <= 0:
        return sample(norm).clamp(0.0, 1.0)
    offsets = torch.tensor(
        [[j, j, 0.0], [j, -j, 0.0], [-j, j, 0.0], [-j, -j, 0.0]],
        device=device,
        dtype=dtype,
    )
    offsets = 2.0 * offsets / dims_xyz[None]
    return torch.stack([sample(norm + off[None]) for off in offsets], dim=0).mean(0).clamp(0.0, 1.0)


@dataclass
class SceneCELossResult:
    loss: torch.Tensor
    per_horizon: list[dict]
    query_voxels: int
    full_voxels: int


def sparse_full_scene_ce_ordered(
    pred_residual_xy_m: torch.Tensor,
    kta_displacement_xy_m: torch.Tensor,
    source_voxel_indices: Sequence[np.ndarray | torch.Tensor],
    source_class_ids: Sequence[int] | torch.Tensor,
    t0_ego_to_world: np.ndarray | torch.Tensor,
    future_ego_to_world: np.ndarray | torch.Tensor,
    strong_anchor_occ: np.ndarray | torch.Tensor,
    future_gt_occ: np.ndarray | torch.Tensor,
    *,
    grid: OccupancyGrid = OccupancyGrid(),
    free_label: int = 17,
    class_count: int = SCENE_CLASS_COUNT,
    halo_voxels: int = 2,
    eps: float = 1e-4,
    jitter_voxels: float = 0.25,
) -> SceneCELossResult:
    """Full-scene CE with sparse differentiable A1-style source composition.

    Outside the query domain the scene remains the constant Strong/KTA anchor.
    Inside it, all KTA source destinations are cleared with the legacy dynamic
    CLEAR rule and predicted sources are alpha-composited in original source
    order.  The loss is normalized by the complete X*Y*Z scene, not |query|.
    """
    if not 0.0 < float(eps) < 1.0 / float(class_count):
        raise ValueError("eps must be in (0,1/class_count)")
    pred = pred_residual_xy_m
    kta = kta_displacement_xy_m.to(pred.device, dtype=pred.dtype)
    if pred.ndim != 3 or pred.shape[-1] != 2 or pred.shape != kta.shape:
        raise ValueError("pred/kta displacement tensors must match [N,6,2]")
    Nsrc, H, _ = pred.shape
    if H != FUTURE_FRAMES or len(source_voxel_indices) != Nsrc:
        raise ValueError("source count/future-frame mismatch")
    classes = torch.as_tensor(source_class_ids, dtype=torch.long).cpu().numpy()
    if classes.shape != (Nsrc,):
        raise ValueError("source_class_ids must be [N]")
    anchor = np.asarray(strong_anchor_occ, dtype=np.int64)
    gt = np.asarray(future_gt_occ, dtype=np.int64)
    if anchor.shape != gt.shape or tuple(anchor.shape) != (FUTURE_FRAMES, *tuple(grid.shape_hwd)):
        raise ValueError("scene occupancy shape mismatch")
    if anchor.min() < 0 or anchor.max() >= class_count or gt.min() < 0 or gt.max() >= class_count:
        raise ValueError("scene labels outside class range")
    t0_pose = np.asarray(t0_ego_to_world, dtype=np.float64)
    future_poses = np.asarray(future_ego_to_world, dtype=np.float64)
    if t0_pose.shape != (4, 4) or future_poses.shape != (FUTURE_FRAMES, 4, 4):
        raise ValueError("pose shape mismatch")

    device = pred.device
    dyn = np.asarray(tuple(int(x) for x in DYNAMIC_CLASS_IDS), dtype=np.int64)
    full_count = int(np.prod(grid.shape_hwd))
    corr_ce = -math.log(max(1.0 - class_count * float(eps), float(eps)))
    wrong_ce = -math.log(float(eps))
    losses: list[torch.Tensor] = []
    rows: list[dict] = []
    total_query = 0

    for h in range(FUTURE_FRAMES):
        kta_flats = []
        pred_aabbs = []
        for i in range(Nsrc):
            vox = np.asarray(source_voxel_indices[i], dtype=np.int64)
            kta_np = kta[i, h].detach().float().cpu().numpy()
            pred_np = (kta[i, h] + pred[i, h]).detach().float().cpu().numpy()
            kf = hard_shifted_source_flats(
                vox, kta_np, t0_pose, future_poses[h], grid=grid
            )
            pa = _predicted_aabb_flats(
                vox,
                pred_np,
                t0_pose,
                future_poses[h],
                grid=grid,
                halo_voxels=int(halo_voxels),
            )
            kta_flats.append(kf)
            pred_aabbs.append(pa)
        parts = [x for x in (*kta_flats, *pred_aabbs) if len(x)]
        U = np.unique(np.concatenate(parts)) if parts else np.zeros((0,), dtype=np.int64)
        total_query += int(len(U))

        a_flat = anchor[h].reshape(-1)
        g_flat = gt[h].reshape(-1)
        n_match = int((a_flat == g_flat).sum())
        base_full_sum = n_match * corr_ce + (full_count - n_match) * wrong_ce
        if not len(U):
            losses.append(pred[:, h].sum() * 0.0 + float(base_full_sum / full_count))
            rows.append({
                "horizon_s": 0.5 * (h + 1),
                "query_voxels": 0,
                "query_fraction": 0.0,
                "full_ce": float(base_full_sum / full_count),
                "variable_ce_contribution": 0.0,
            })
            continue

        anchor_u = a_flat[U]
        gt_u_np = g_flat[U]
        base_labels = anchor_u.copy()
        clear_parts = [x for x in kta_flats if len(x)]
        clear_u = np.unique(np.concatenate(clear_parts)) if clear_parts else np.zeros((0,), dtype=np.int64)
        if len(clear_u):
            pos = np.searchsorted(U, clear_u)
            good = (pos < len(U)) & (U[np.minimum(pos, len(U) - 1)] == clear_u)
            pos = pos[good]
            clear_dyn = np.isin(base_labels[pos], dyn)
            base_labels[pos[clear_dyn]] = int(free_label)

        uf = torch.as_tensor(U, device=device, dtype=torch.long)
        P = F.one_hot(
            torch.as_tensor(base_labels, device=device, dtype=torch.long),
            num_classes=int(class_count),
        ).to(torch.float32)

        # Original Strong source order is the compositor order.  No sorting here.
        for i in range(Nsrc):
            local_u = pred_aabbs[i]
            if not len(local_u):
                continue
            pos_np = np.searchsorted(U, local_u)
            good = (pos_np < len(U)) & (U[np.minimum(pos_np, len(U) - 1)] == local_u)
            local_u = local_u[good]
            pos_np = pos_np[good]
            if not len(local_u):
                continue
            pos_t = torch.as_tensor(pos_np, device=device, dtype=torch.long)
            flats_t = torch.as_tensor(local_u, device=device, dtype=torch.long)
            total_disp = kta[i, h].float() + pred[i, h].float()
            alpha = _sample_source_alpha(
                np.asarray(source_voxel_indices[i], dtype=np.int64),
                total_disp,
                t0_pose,
                future_poses[h],
                flats_t,
                grid=grid,
                jitter_voxels=float(jitter_voxels),
            )
            cls = F.one_hot(
                torch.tensor(int(classes[i]), device=device), num_classes=int(class_count)
            ).to(torch.float32)
            old = P.index_select(0, pos_t)
            new = (1.0 - alpha[:, None]) * old + alpha[:, None] * cls[None]
            P = torch.index_copy(P, 0, pos_t, new)

        qprob = (1.0 - class_count * float(eps)) * P + float(eps)
        gt_u = torch.as_tensor(gt_u_np, device=device, dtype=torch.long)
        new_ce = -torch.log(
            qprob.gather(1, gt_u[:, None]).squeeze(1).clamp_min(float(eps))
        )
        old_ce_np = np.where(anchor_u == gt_u_np, corr_ce, wrong_ce)
        old_sum = float(old_ce_np.sum())
        total = pred[:, h].sum() * 0.0 + float(base_full_sum - old_sum) + new_ce.sum()
        loss_h = total / float(full_count)
        losses.append(loss_h)
        rows.append({
            "horizon_s": 0.5 * (h + 1),
            "query_voxels": int(len(U)),
            "query_fraction": float(len(U)) / float(full_count),
            "full_ce": float(loss_h.detach().cpu()),
            "variable_ce_contribution": float(new_ce.detach().sum().cpu()) / float(full_count),
        })

    loss = torch.stack(losses).mean() if losses else pred.sum() * 0.0
    return SceneCELossResult(
        loss=loss,
        per_horizon=rows,
        query_voxels=int(total_query),
        full_voxels=int(full_count * FUTURE_FRAMES),
    )


def v17_base_loss_tensors(
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    *,
    overlap_weight: float = 0.25,
    patch_resolution_m: float = 0.8,
) -> dict[str, torch.Tensor | int | float]:
    """Tensor-valued form of the frozen V17-RL objective for C training."""
    pred = outputs["residual_xy_m"]
    target = batch["target_residual_xy_m"].to(pred.dtype)
    valid = batch["target_valid"].bool()
    if bool(valid.any()):
        pos = F.smooth_l1_loss(pred[valid], target[valid], reduction="mean", beta=1.0)
    else:
        pos = pred.sum() * 0.0

    logits = outputs["existence_logits"]
    labels = batch["existence"].to(logits.dtype)
    supervised = batch["supervised_source"].bool().unsqueeze(-1).expand_as(labels)
    if bool(supervised.any()):
        exist = F.binary_cross_entropy_with_logits(logits[supervised], labels[supervised])
    else:
        exist = logits.sum() * 0.0

    overlap, stats = soft_transport_overlap_loss(
        pred.float(),
        target.float(),
        batch["target_source_mask_tube"][:, -1].float(),
        valid,
        patch_resolution_m=float(patch_resolution_m),
    )
    motion = pos + float(overlap_weight) * overlap
    total = motion + exist
    return {
        "position": pos,
        "existence": exist,
        "overlap": overlap,
        "weighted_overlap": float(overlap_weight) * overlap,
        "motion": motion,
        "total": total,
        "overlap_labels": int(stats["transport_overlap_labels"]),
        "soft_iou": float(stats["transport_soft_iou"]),
    }


def freeze_history_encoder_for_scene_continuation(model: torch.nn.Module) -> dict:
    """Freeze history encoder; keep future decoder/query/heads trainable.

    ``kinematic_proj`` is shared by encoder and future-query paths, therefore it
    is frozen as part of the history representation rather than silently updated
    through the decoder branch.
    """
    trainable_prefixes = (
        "future_query",
        "future_time_embedding",
        "kta_future_proj",
        "decoder",
        "residual_head",
        "existence_head",
    )
    trainable = frozen = 0
    trainable_names = []
    frozen_names = []
    for name, p in model.named_parameters():
        keep = name.startswith(trainable_prefixes)
        p.requires_grad_(keep)
        n = int(p.numel())
        if keep:
            trainable += n
            trainable_names.append(name)
        else:
            frozen += n
            frozen_names.append(name)
    if trainable == 0 or frozen == 0:
        raise RuntimeError("scene continuation freeze contract produced empty parameter side")
    return {
        "contract": SCENE_FREEZE_CONTRACT,
        "trainable_parameters": trainable,
        "frozen_parameters": frozen,
        "trainable_names": trainable_names,
        "frozen_names": frozen_names,
    }


def grad_vector(loss: torch.Tensor, params: Sequence[torch.nn.Parameter], *, retain_graph: bool = True) -> torch.Tensor:
    grads = torch.autograd.grad(loss, params, retain_graph=retain_graph, allow_unused=True)
    chunks = []
    for p, g in zip(params, grads):
        chunks.append(torch.zeros_like(p).reshape(-1) if g is None else g.reshape(-1))
    return torch.cat(chunks) if chunks else loss.new_zeros((0,))


def calibrate_scene_alpha_from_gradients(
    motion_norms: Sequence[float],
    scene_norms: Sequence[float],
    *,
    target_ratio: float = 0.25,
    max_alpha: float = 1.0e4,
) -> dict:
    """Fixed one-time alpha calibration; no dynamic weighting during training."""
    m = np.asarray(motion_norms, dtype=np.float64)
    s = np.asarray(scene_norms, dtype=np.float64)
    if m.size == 0 or s.size == 0 or m.size != s.size:
        raise ValueError("calibration norm arrays must be non-empty and aligned")
    if not (np.isfinite(m).all() and np.isfinite(s).all()) or bool((m <= 0).any()) or bool((s <= 0).any()):
        raise ValueError("calibration gradient norms must be finite and positive")
    if not 0.0 < float(target_ratio) <= 1.0 or float(max_alpha) <= 0:
        raise ValueError("invalid alpha calibration settings")
    motion_ref = float(np.median(m))
    scene_ref = float(np.median(s))
    raw = float(target_ratio) * motion_ref / scene_ref
    alpha = min(raw, float(max_alpha))
    return {
        "target_scene_to_motion_grad_ratio": float(target_ratio),
        "motion_grad_median": motion_ref,
        "scene_unit_grad_median": scene_ref,
        "alpha_raw": raw,
        "alpha": alpha,
        "alpha_clipped": bool(alpha < raw),
        "max_alpha": float(max_alpha),
    }


class V17SceneCacheDataset(Dataset):
    """Small sharded cache used only by the C scene-supervision experiment."""
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.index = json.loads((self.root / "index.json").read_text(encoding="utf-8"))
        if self.index.get("version") != SCENE_CACHE_VERSION:
            raise RuntimeError("V17 scene cache version mismatch")
        self.metadata = self.index.get("metadata") or {}
        self.entries = self.index.get("entries") or []
        if not self.entries:
            raise RuntimeError("empty V17 scene cache")
        self._shard_name = None
        self._shard = None

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index: int):
        e = self.entries[int(index)]
        with _SHARD_LOCK:
            if e["shard"] != self._shard_name:
                self._shard = torch.load(self.root / e["shard"], map_location="cpu", weights_only=False)
                self._shard_name = e["shard"]
            row = self._shard[int(e["index"])]
        return row
