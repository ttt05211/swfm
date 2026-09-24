"""Differentiable scene-level Moving-mIoU safety for V18 SE(2) transport.

This is the task-aligned replacement for the earlier local-footprint safe loss.
It mirrors the frozen hard evaluation contract as closely as a differentiable
surrogate can:

  Strong/KTA anchor
    -> legacy A1 CLEAR at all KTA source destinations
    -> predicted SE(2) source WRITE in original Strong order
    -> evaluate only on frozen GT true-moving support
    -> class-wise semantic IoU
    -> mean classes, then mean report horizons (1s/2s/3s).

The only deliberate mismatch to evaluation is soft trilinear source occupancy
instead of hard voxel rasterization so gradients can reach displacement/yaw.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import json
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .geometry import OccupancyGrid
from .local_st_world_model_v18_se2 import YAW_ENABLED_CLASS_IDS
from .local_stwm_scene_supervision import hard_shifted_source_flats
from .metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS
from .motion_transport import FUTURE_FRAMES

MOVING_SUPPORT_CACHE_VERSION = "p0_f9_v18_moving_support_cache_v1"
MOVING_SAFE_LOSS_CONTRACT = (
    "a1_se2_source_order_soft_true_moving_miou_vs_strong_kta_v1"
)
REPORT_FRAME_INDICES = (1, 3, 5)
REPORT_HORIZONS_S = (1.0, 2.0, 3.0)


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


def _source_center_t0_ego(
    voxel_indices_t0: np.ndarray,
    grid: OccupancyGrid,
) -> np.ndarray:
    idx = np.asarray(voxel_indices_t0, dtype=np.float64)
    if idx.ndim != 2 or idx.shape[1] != 3 or not len(idx):
        raise ValueError("source center requires non-empty [N,3] indices")
    origin, step = _grid_origin_step(grid)
    return origin + (idx.mean(axis=0) + 0.5) * step


def _predicted_se2_aabb_flats(
    voxel_indices_t0: np.ndarray,
    displacement_xy_t0_m: np.ndarray,
    yaw_delta_rad: float,
    t0_ego_to_world: np.ndarray,
    future_ego_to_world: np.ndarray,
    *,
    grid: OccupancyGrid,
    halo_voxels: int,
) -> np.ndarray:
    """Conservative future-grid AABB for a detached predicted SE(2) source."""
    idx = np.asarray(voxel_indices_t0, dtype=np.int64)
    if idx.ndim != 2 or idx.shape[1] != 3 or not len(idx):
        return np.zeros((0,), dtype=np.int64)
    origin, step = _grid_origin_step(grid)
    lo = idx.min(axis=0)
    hi = idx.max(axis=0) + 1
    mins = origin + lo * step
    maxs = origin + hi * step
    corners = np.asarray(
        [
            [x, y, z]
            for x in (mins[0], maxs[0])
            for y in (mins[1], maxs[1])
            for z in (mins[2], maxs[2])
        ],
        dtype=np.float64,
    )
    center = _source_center_t0_ego(idx, grid)
    d = np.asarray(displacement_xy_t0_m, dtype=np.float64)
    if d.shape != (2,):
        raise ValueError("displacement must be [2]")
    theta = float(yaw_delta_rad)
    c, s = math.cos(theta), math.sin(theta)
    rel = corners[:, :2] - center[None, :2]
    rot = np.empty_like(rel)
    rot[:, 0] = c * rel[:, 0] - s * rel[:, 1]
    rot[:, 1] = s * rel[:, 0] + c * rel[:, 1]
    moved = corners.copy()
    moved[:, :2] = center[None, :2] + d[None] + rot

    pw = _transform_points(t0_ego_to_world, moved)
    pf = _transform_points(
        np.linalg.inv(np.asarray(future_ego_to_world, dtype=np.float64)), pw
    )
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
    return np.unique(
        _flat_from_indices(
            np.stack((xx.ravel(), yy.ravel(), zz.ravel()), axis=1), grid
        )
    )


def _sample_source_alpha_se2(
    voxel_indices_t0: np.ndarray,
    displacement_xy_t0_m: torch.Tensor,
    yaw_delta_rad: torch.Tensor,
    t0_ego_to_world: np.ndarray,
    future_ego_to_world: np.ndarray,
    flat_indices: torch.Tensor,
    *,
    grid: OccupancyGrid,
    jitter_voxels: float = 0.25,
) -> torch.Tensor:
    """Inverse-sample one exact t0 source under source-centred SE(2).

    Query voxels are future-ego cells.  They are mapped into the t0 ego frame,
    then through the inverse object transform

        q = c_s + R(-yaw) [q' - c_s - d],

    before trilinear sampling from the exact observed 3D source template.
    """
    if flat_indices.numel() == 0:
        return (
            displacement_xy_t0_m.sum() * 0.0
            + yaw_delta_rad.sum() * 0.0
            + torch.zeros(
                (0,),
                device=displacement_xy_t0_m.device,
                dtype=torch.float32,
            )
        )

    lo, arr = _source_template(voxel_indices_t0)
    device = displacement_xy_t0_m.device
    dtype = torch.float32
    tmpl = torch.as_tensor(arr, device=device, dtype=dtype).permute(
        2, 1, 0
    )[None, None]
    qidx = _indices_from_flat(flat_indices, grid).to(device=device)
    origin = torch.tensor(
        [grid.x_min, grid.y_min, grid.z_min], device=device, dtype=dtype
    )
    step = torch.tensor(grid.voxel_size, device=device, dtype=dtype)
    pf = origin[None] + (qidx.to(dtype) + 0.5) * step[None]

    # future ego -> world -> t0 ego
    T = np.linalg.inv(np.asarray(t0_ego_to_world, dtype=np.float64)) @ np.asarray(
        future_ego_to_world, dtype=np.float64
    )
    TT = torch.as_tensor(T, device=device, dtype=dtype)
    qprime = pf @ TT[:3, :3].T + TT[:3, 3]

    center_np = _source_center_t0_ego(voxel_indices_t0, grid)
    center = torch.as_tensor(center_np, device=device, dtype=dtype)
    d = displacement_xy_t0_m.to(dtype)
    theta = yaw_delta_rad.to(dtype)
    relx = qprime[:, 0] - center[0] - d[0]
    rely = qprime[:, 1] - center[1] - d[1]
    c = torch.cos(theta)
    s = torch.sin(theta)

    src = qprime.clone()
    src[:, 0] = center[0] + c * relx + s * rely
    src[:, 1] = center[1] - s * relx + c * rely

    local_center = (
        (src - origin[None]) / step[None]
        - 0.5
        - torch.as_tensor(lo, device=device, dtype=dtype)[None]
    )
    dims_xyz = torch.tensor(arr.shape, device=device, dtype=dtype)
    norm = 2.0 * (local_center + 0.5) / dims_xyz[None] - 1.0

    def sample(n: torch.Tensor) -> torch.Tensor:
        g = n.reshape(1, 1, 1, -1, 3)
        return F.grid_sample(
            tmpl,
            g,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
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
    return (
        torch.stack([sample(norm + off[None]) for off in offsets], dim=0)
        .mean(0)
        .clamp(0.0, 1.0)
    )


def _intersect_sorted(query: np.ndarray, candidate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return candidate flats inside sorted query plus their query positions."""
    if not len(query) or not len(candidate):
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    pos = np.searchsorted(query, candidate)
    good = (pos < len(query)) & (
        query[np.minimum(pos, len(query) - 1)] == candidate
    )
    return candidate[good], pos[good]


@dataclass
class MovingSafeResult:
    loss: torch.Tensor
    pred_soft_moving_miou: torch.Tensor
    kta_moving_miou: torch.Tensor
    active: bool
    per_horizon: list[dict]
    support_voxels: int


def scene_moving_safe_loss(
    pred_residual_xy_m: torch.Tensor,
    pred_yaw_rad: torch.Tensor,
    kta_displacement_xy_m: torch.Tensor,
    source_voxel_indices: Sequence[Sequence[np.ndarray | torch.Tensor]],
    source_class_ids: Sequence[Sequence[int] | torch.Tensor],
    t0_ego_to_world: Sequence[np.ndarray | torch.Tensor],
    future_ego_to_world: Sequence[np.ndarray | torch.Tensor],
    strong_anchor_occ: Sequence[np.ndarray | torch.Tensor],
    future_gt_occ: Sequence[np.ndarray | torch.Tensor],
    moving_support_flats: Sequence[Mapping[int, np.ndarray | torch.Tensor]],
    source_slices: Sequence[slice],
    *,
    grid: OccupancyGrid = OccupancyGrid(),
    free_label: int = 17,
    halo_voxels: int = 2,
    jitter_voxels: float = 0.25,
    eps: float = 1e-6,
) -> MovingSafeResult:
    """Batch soft Moving-mIoU safety after differentiable A1 composition.

    The KTA reference is the frozen Strong/KTA anchor itself.  The prediction is
    formed by A1-style CLEAR + ordered SE(2) WRITE.  Intersections/unions are
    accumulated over the whole mini-batch before class/horizon averaging, which
    matches the frozen metric's dataset-level aggregation as closely as a
    stochastic training loss can.
    """
    pred = pred_residual_xy_m.float()
    yaw = pred_yaw_rad.float()
    kta = kta_displacement_xy_m.to(pred.device, dtype=torch.float32)
    if pred.ndim != 3 or pred.shape[-1] != 2 or pred.shape != kta.shape:
        raise ValueError("pred/kta tensors must match [N,6,2]")
    if yaw.shape != pred.shape[:2]:
        raise ValueError("pred_yaw_rad must be [N,6]")
    if pred.shape[1] != FUTURE_FRAMES:
        raise ValueError("future-frame mismatch")

    B = len(source_slices)
    fields = (
        source_voxel_indices,
        source_class_ids,
        t0_ego_to_world,
        future_ego_to_world,
        strong_anchor_occ,
        future_gt_occ,
        moving_support_flats,
    )
    if any(len(x) != B for x in fields):
        raise ValueError("scene-batch metadata length mismatch")

    device = pred.device
    dyn_classes = tuple(int(c) for c in DYNAMIC_CLASS_IDS)
    dyn_np = np.asarray(dyn_classes, dtype=np.int64)
    class_count = 18

    pred_inter = {
        hi: [pred.sum() * 0.0 for _ in dyn_classes] for hi in REPORT_FRAME_INDICES
    }
    pred_union = {
        hi: [pred.sum() * 0.0 for _ in dyn_classes] for hi in REPORT_FRAME_INDICES
    }
    kta_inter = {hi: [0.0 for _ in dyn_classes] for hi in REPORT_FRAME_INDICES}
    kta_union = {hi: [0.0 for _ in dyn_classes] for hi in REPORT_FRAME_INDICES}
    support_total = 0

    for bi, sl in enumerate(source_slices):
        nsrc = int(sl.stop - sl.start)
        voxels = [
            np.asarray(v, dtype=np.int64) for v in source_voxel_indices[bi]
        ]
        classes = np.asarray(source_class_ids[bi], dtype=np.int64)
        if len(voxels) != nsrc or classes.shape != (nsrc,):
            raise ValueError("scene/source count mismatch")

        t0_pose = np.asarray(t0_ego_to_world[bi], dtype=np.float64)
        future_poses = np.asarray(future_ego_to_world[bi], dtype=np.float64)
        anchor = np.asarray(strong_anchor_occ[bi], dtype=np.int64)
        gt = np.asarray(future_gt_occ[bi], dtype=np.int64)
        if tuple(anchor.shape) != (FUTURE_FRAMES, *tuple(grid.shape_hwd)):
            raise ValueError("anchor shape mismatch")
        if gt.shape != anchor.shape:
            raise ValueError("GT shape mismatch")

        for hi in REPORT_FRAME_INDICES:
            raw_u = moving_support_flats[bi].get(int(hi))
            if raw_u is None:
                raise KeyError(f"moving support missing horizon index {hi}")
            if torch.is_tensor(raw_u):
                U = raw_u.detach().cpu().numpy().astype(np.int64, copy=False)
            else:
                U = np.asarray(raw_u, dtype=np.int64)
            U = np.unique(U)
            if not len(U):
                continue
            support_total += int(len(U))

            a_flat = anchor[hi].reshape(-1)
            g_flat = gt[hi].reshape(-1)
            anchor_u = a_flat[U]
            gt_u_np = g_flat[U]

            # Frozen KTA reference is the hard Strong/KTA anchor.
            for ci, cls in enumerate(dyn_classes):
                kp = anchor_u == int(cls)
                kg = gt_u_np == int(cls)
                kta_inter[hi][ci] += float(np.logical_and(kp, kg).sum())
                kta_union[hi][ci] += float(np.logical_or(kp, kg).sum())

            # Candidate starts from anchor, applies the exact legacy A1 CLEAR
            # contract, then ordered differentiable SE(2) source writes.
            base_labels = anchor_u.copy()
            kta_flats = []
            for i in range(nsrc):
                kf = hard_shifted_source_flats(
                    voxels[i],
                    kta[sl][i, hi].detach().cpu().numpy(),
                    t0_pose,
                    future_poses[hi],
                    grid=grid,
                )
                kta_flats.append(kf)
            clear_parts = [x for x in kta_flats if len(x)]
            if clear_parts:
                clear_u = np.unique(np.concatenate(clear_parts))
                _, pos = _intersect_sorted(U, clear_u)
                if len(pos):
                    clear_dyn = np.isin(base_labels[pos], dyn_np)
                    base_labels[pos[clear_dyn]] = int(free_label)

            P = F.one_hot(
                torch.as_tensor(base_labels, device=device, dtype=torch.long),
                num_classes=class_count,
            ).to(torch.float32)

            local_pred = pred[sl]
            local_yaw = yaw[sl]
            local_kta = kta[sl]
            for i in range(nsrc):
                total_disp = local_kta[i, hi] + local_pred[i, hi]
                use_yaw = int(classes[i]) in YAW_ENABLED_CLASS_IDS
                yaw_i = (
                    local_yaw[i, hi]
                    if use_yaw
                    else local_yaw[i, hi] * 0.0
                )
                aabb = _predicted_se2_aabb_flats(
                    voxels[i],
                    total_disp.detach().cpu().numpy(),
                    float(yaw_i.detach().cpu()),
                    t0_pose,
                    future_poses[hi],
                    grid=grid,
                    halo_voxels=int(halo_voxels),
                )
                local_u, pos_np = _intersect_sorted(U, aabb)
                if not len(local_u):
                    continue
                pos_t = torch.as_tensor(pos_np, device=device, dtype=torch.long)
                flats_t = torch.as_tensor(local_u, device=device, dtype=torch.long)
                alpha = _sample_source_alpha_se2(
                    voxels[i],
                    total_disp,
                    yaw_i,
                    t0_pose,
                    future_poses[hi],
                    flats_t,
                    grid=grid,
                    jitter_voxels=float(jitter_voxels),
                )
                cls = F.one_hot(
                    torch.tensor(int(classes[i]), device=device),
                    num_classes=class_count,
                ).to(torch.float32)
                old = P.index_select(0, pos_t)
                new = (1.0 - alpha[:, None]) * old + alpha[:, None] * cls[None]
                P = torch.index_copy(P, 0, pos_t, new)

            gt_u = torch.as_tensor(gt_u_np, device=device, dtype=torch.long)
            for ci, cls in enumerate(dyn_classes):
                pp = P[:, int(cls)]
                gg = (gt_u == int(cls)).to(pp.dtype)
                pred_inter[hi][ci] = pred_inter[hi][ci] + (pp * gg).sum()
                pred_union[hi][ci] = pred_union[hi][ci] + (
                    pp + gg - pp * gg
                ).sum()

    horizon_rows = []
    pred_h_scores = []
    kta_h_scores = []
    for hi, horizon_s in zip(REPORT_FRAME_INDICES, REPORT_HORIZONS_S):
        pvals = []
        kvals = []
        used = []
        for ci, cls in enumerate(dyn_classes):
            pu = pred_union[hi][ci]
            ku = float(kta_union[hi][ci])
            # One shared class set keeps the pred/KTA comparison paired.
            union_present = bool(float(pu.detach().cpu()) > float(eps) or ku > 0.0)
            if not union_present:
                continue
            pi = pred_inter[hi][ci]
            pvals.append((pi + float(eps)) / (pu + float(eps)))
            kvals.append(
                pred.new_tensor(
                    (float(kta_inter[hi][ci]) + float(eps))
                    / (ku + float(eps))
                )
            )
            used.append(int(cls))
        if not pvals:
            continue
        ps = torch.stack(pvals).mean()
        ks = torch.stack(kvals).mean().detach()
        pred_h_scores.append(ps)
        kta_h_scores.append(ks)
        horizon_rows.append(
            {
                "horizon_s": float(horizon_s),
                "pred_soft_moving_miou": float(ps.detach().cpu()),
                "kta_moving_miou": float(ks.detach().cpu()),
                "classes": used,
            }
        )

    if not pred_h_scores:
        z = pred.sum() * 0.0 + yaw.sum() * 0.0
        nan = z.detach() * 0.0 + float("nan")
        return MovingSafeResult(
            loss=z,
            pred_soft_moving_miou=nan,
            kta_moving_miou=nan,
            active=False,
            per_horizon=[],
            support_voxels=0,
        )

    pred_score = torch.stack(pred_h_scores).mean()
    kta_score = torch.stack(kta_h_scores).mean().detach()
    loss = F.relu(kta_score - pred_score)
    return MovingSafeResult(
        loss=loss,
        pred_soft_moving_miou=pred_score,
        kta_moving_miou=kta_score,
        active=bool(float(loss.detach().cpu()) > 0.0),
        per_horizon=horizon_rows,
        support_voxels=int(support_total),
    )


def load_moving_support_cache(path: str | Path) -> tuple[dict, dict[str, dict]]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if obj.get("version") != MOVING_SUPPORT_CACHE_VERSION:
        raise RuntimeError(
            f"moving-support cache version mismatch: {obj.get('version')}"
        )
    meta = obj.get("metadata") or {}
    if meta.get("moving_safe_loss_contract") != MOVING_SAFE_LOSS_CONTRACT:
        raise RuntimeError("moving-support loss contract mismatch")
    records = obj.get("records") or {}
    if not records:
        raise RuntimeError("empty moving-support cache")
    return meta, records


def moving_support_cache_summary(path: str | Path) -> dict:
    meta, records = load_moving_support_cache(path)
    counts = {str(h): 0 for h in REPORT_FRAME_INDICES}
    for r in records.values():
        for h in REPORT_FRAME_INDICES:
            x = r["moving_support_flat_by_horizon"][int(h)]
            counts[str(h)] += int(x.numel() if torch.is_tensor(x) else len(x))
    return {
        "path": str(Path(path).resolve()),
        "num_windows": len(records),
        "support_voxels_by_frame_index": counts,
        "metadata": meta,
    }
