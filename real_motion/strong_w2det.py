"""Strong occupancy-only W2Det/KTA anchor used by P0-F4.

This ports the proven pre-SWFM contract: static dynamic-free occupancy is inverse
warped by future ego motion, while every motion-capable semantic component is
extracted directly from t-1/t0 occupancy, matched by same-class mutual nearest
neighbor, assigned a causal backward-difference velocity, and propagated with a
constant-velocity model.  Unmatched and too-small dynamic components keep zero
object velocity and therefore move only with the ego-frame transform.

No future semantic labels, GT boxes, or annotation velocities are consumed.
Occ3D arrays use official [X,Y,Z] axis order.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np
from scipy.ndimage import label, generate_binary_structure, uniform_filter

from .geometry import OccupancyGrid, relative_transform
from .metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS


@dataclass(frozen=True)
class StrongW2DetConfig:
    free_label: int = 17
    min_component_voxels: int = 6
    max_match_speed_mps: float = 25.0
    connectivity: int = 2
    fill_kernel: tuple[int, int, int] = (5, 5, 1)
    fill_min_fraction: float = 0.3


def _transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ np.asarray(T, dtype=np.float64)[:3, :3].T + np.asarray(T, dtype=np.float64)[:3, 3]


def _metric_to_voxel(pts: np.ndarray, grid: OccupancyGrid) -> np.ndarray:
    origin = np.asarray([grid.x_min, grid.y_min, grid.z_min], dtype=np.float64)
    step = np.asarray(grid.voxel_size, dtype=np.float64)
    return np.floor((np.asarray(pts, dtype=np.float64) - origin) / step).astype(np.int64)


def _in_grid(idx: np.ndarray, grid: OccupancyGrid) -> np.ndarray:
    shape = np.asarray(grid.shape_hwd, dtype=np.int64)
    return ((idx >= 0) & (idx < shape[None])).all(axis=1)


@lru_cache(maxsize=16)
def _voxel_centers_cached(
    shape: tuple[int, int, int],
    origin: tuple[float, float, float],
    step: tuple[float, float, float],
) -> np.ndarray:
    X, Y, Z = shape
    xs = origin[0] + (np.arange(X, dtype=np.float64) + 0.5) * step[0]
    ys = origin[1] + (np.arange(Y, dtype=np.float64) + 0.5) * step[1]
    zs = origin[2] + (np.arange(Z, dtype=np.float64) + 0.5) * step[2]
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1)


def _voxel_centers(grid: OccupancyGrid) -> np.ndarray:
    return _voxel_centers_cached(
        tuple(int(v) for v in grid.shape_hwd),
        (float(grid.x_min), float(grid.y_min), float(grid.z_min)),
        tuple(float(v) for v in grid.voxel_size),
    )


def inverse_warp(
    semantics: np.ndarray,
    src_to_dst: np.ndarray,
    grid: OccupancyGrid,
    free_label: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Old W1 inverse nearest-neighbor warp and geometry-known mask."""
    sem = np.asarray(semantics)
    if tuple(sem.shape) != tuple(grid.shape_hwd):
        raise ValueError("semantic grid shape mismatch")
    dst_pts = _voxel_centers(grid)
    dst_to_src = np.linalg.inv(np.asarray(src_to_dst, dtype=np.float64))
    src_pts = _transform_points(dst_to_src, dst_pts)
    idx = _metric_to_voxel(src_pts, grid)
    known = _in_grid(idx, grid)
    flat = np.full(len(dst_pts), int(free_label), dtype=sem.dtype)
    q = idx[known]
    flat[known] = sem[q[:, 0], q[:, 1], q[:, 2]]
    return flat.reshape(grid.shape_hwd), known.reshape(grid.shape_hwd)


def majority_fill(
    semantics: np.ndarray,
    unknown_mask: np.ndarray,
    *,
    kernel: tuple[int, int, int] = (5, 5, 1),
    min_fraction: float = 0.3,
) -> np.ndarray:
    """Functional equivalent of the old W1 5x5x1 majority fill."""
    sem = np.asarray(semantics)
    unknown = np.asarray(unknown_mask, dtype=bool)
    if not bool(unknown.any()):
        return sem.copy()
    out = sem.copy()
    known = ~unknown
    denom = uniform_filter(known.astype(np.float32), size=kernel, mode="constant")
    denom = np.maximum(denom, 1e-6)
    best_score = np.zeros_like(denom, dtype=np.float32)
    best_label = np.zeros_like(sem)
    for cls in np.unique(sem[known]):
        score = uniform_filter(((sem == cls) & known).astype(np.float32), size=kernel, mode="constant") / denom
        upd = score > best_score
        best_score[upd] = score[upd]
        best_label[upd] = cls
    fill = unknown & (best_score >= float(min_fraction))
    out[fill] = best_label[fill]
    return out


def extract_instances(
    semantics: np.ndarray,
    ego_to_world: np.ndarray,
    *,
    grid: OccupancyGrid,
    cfg: StrongW2DetConfig,
) -> list[dict]:
    """3D same-class connected components from occupancy only."""
    sem = np.asarray(semantics)
    structure = generate_binary_structure(3, int(cfg.connectivity))
    out: list[dict] = []
    origin = np.asarray([grid.x_min, grid.y_min, grid.z_min], dtype=np.float64)
    step = np.asarray(grid.voxel_size, dtype=np.float64)
    for cls in DYNAMIC_CLASS_IDS:
        comp_map, n = label(sem == int(cls), structure=structure)
        for comp_id in range(1, int(n) + 1):
            idx = np.argwhere(comp_map == comp_id)
            if len(idx) < int(cfg.min_component_voxels):
                continue
            pts_ego = origin + (idx.astype(np.float64) + 0.5) * step
            centroid_world = _transform_points(ego_to_world, pts_ego.mean(axis=0, keepdims=True))[0]
            out.append({
                "class_id": int(cls),
                "voxel_indices": idx.astype(np.int64),
                "centroid_world": centroid_world,
                "voxel_count": int(len(idx)),
            })
    return out


def match_instances(
    previous: Sequence[dict],
    current: Sequence[dict],
    dt_s: float,
    *,
    max_speed_mps: float,
) -> dict[int, np.ndarray]:
    """Same-class mutual-nearest matching with the old 25 m/s gate."""
    dt = max(float(dt_s), 1e-3)
    gate = float(max_speed_mps) * dt
    velocities: dict[int, np.ndarray] = {}
    by_class: dict[int, list[int]] = {}
    for j, row in enumerate(current):
        by_class.setdefault(int(row["class_id"]), []).append(j)
    for cls, cur_ids in by_class.items():
        prev_ids = [i for i, row in enumerate(previous) if int(row["class_id"]) == cls]
        if not prev_ids:
            continue
        P = np.stack([np.asarray(previous[i]["centroid_world"], dtype=np.float64) for i in prev_ids])
        C = np.stack([np.asarray(current[j]["centroid_world"], dtype=np.float64) for j in cur_ids])
        dist = np.linalg.norm(C[:, None, :] - P[None, :, :], axis=-1)
        nearest_prev = dist.argmin(axis=1)
        nearest_cur = dist.argmin(axis=0)
        for local_cur, cur_id in enumerate(cur_ids):
            p = int(nearest_prev[local_cur])
            if int(nearest_cur[p]) != int(local_cur):
                continue
            if float(dist[local_cur, p]) > gate:
                continue
            v = (C[local_cur] - P[p]) / dt
            v = np.asarray(v, dtype=np.float64)
            v[2] = 0.0
            velocities[int(cur_id)] = v
    return velocities


def w2det_predict(
    current_semantics: np.ndarray,
    previous_semantics: np.ndarray | None,
    current_ego_to_world: np.ndarray,
    previous_ego_to_world: np.ndarray,
    future_ego_to_world: np.ndarray,
    *,
    dt_future_s: float,
    dt_previous_s: float,
    grid: OccupancyGrid = OccupancyGrid(),
    cfg: StrongW2DetConfig = StrongW2DetConfig(),
) -> np.ndarray:
    """Strong W2Det: W1 static transport + causal constant-velocity instances."""
    sem0 = np.asarray(current_semantics)
    dyn = np.isin(sem0, np.asarray(DYNAMIC_CLASS_IDS, dtype=sem0.dtype))
    static_src = sem0.copy()
    static_src[dyn] = int(cfg.free_label)
    t_future_from_current = relative_transform(current_ego_to_world, future_ego_to_world)
    static_dst, known = inverse_warp(static_src, t_future_from_current, grid, cfg.free_label)
    out = majority_fill(
        static_dst,
        ~known,
        kernel=cfg.fill_kernel,
        min_fraction=cfg.fill_min_fraction,
    )
    if not bool(dyn.any()):
        return out

    current = extract_instances(sem0, current_ego_to_world, grid=grid, cfg=cfg)
    previous = (
        extract_instances(previous_semantics, previous_ego_to_world, grid=grid, cfg=cfg)
        if previous_semantics is not None else []
    )
    velocities = match_instances(
        previous, current, dt_previous_s, max_speed_mps=cfg.max_match_speed_mps
    )

    origin = np.asarray([grid.x_min, grid.y_min, grid.z_min], dtype=np.float64)
    step = np.asarray(grid.voxel_size, dtype=np.float64)
    world_to_future = np.linalg.inv(np.asarray(future_ego_to_world, dtype=np.float64))
    covered = np.zeros_like(dyn, dtype=bool)
    moved_points: list[np.ndarray] = []
    moved_labels: list[np.ndarray] = []
    for j, inst in enumerate(current):
        idx = inst["voxel_indices"]
        covered[idx[:, 0], idx[:, 1], idx[:, 2]] = True
        pts_ego = origin + (idx.astype(np.float64) + 0.5) * step
        pts_world = _transform_points(current_ego_to_world, pts_ego)
        v = velocities.get(j)
        if v is not None:
            pts_world = pts_world + v[None] * float(dt_future_s)
        moved_points.append(pts_world)
        moved_labels.append(np.full(len(idx), int(inst["class_id"]), dtype=sem0.dtype))

    # Components below the size threshold keep zero object velocity.
    rest = dyn & ~covered
    if bool(rest.any()):
        ridx = np.argwhere(rest)
        pts_ego = origin + (ridx.astype(np.float64) + 0.5) * step
        moved_points.append(_transform_points(current_ego_to_world, pts_ego))
        moved_labels.append(sem0[ridx[:, 0], ridx[:, 1], ridx[:, 2]])

    if moved_points:
        pts_future = _transform_points(world_to_future, np.concatenate(moved_points, axis=0))
        labels = np.concatenate(moved_labels, axis=0)
        idx = _metric_to_voxel(pts_future, grid)
        valid = _in_grid(idx, grid)
        idx = idx[valid]
        labels = labels[valid]
        out[idx[:, 0], idx[:, 1], idx[:, 2]] = labels
    return out


def strong_w2det_sequence(
    history_semantics: np.ndarray,
    history_ego_to_world: Sequence[np.ndarray],
    future_ego_to_world: Sequence[np.ndarray],
    *,
    frame_dt_s: float = 0.5,
    grid: OccupancyGrid = OccupancyGrid(),
    cfg: StrongW2DetConfig = StrongW2DetConfig(),
) -> np.ndarray:
    """Generate all future strong anchors from the final two causal history frames."""
    hist = np.asarray(history_semantics)
    if hist.ndim != 4 or hist.shape[0] < 2:
        raise ValueError("history_semantics must be [T,X,Y,Z] with T>=2")
    if len(history_ego_to_world) != hist.shape[0]:
        raise ValueError("history pose count mismatch")
    current_pose = np.asarray(history_ego_to_world[-1], dtype=np.float64)
    previous_pose = np.asarray(history_ego_to_world[-2], dtype=np.float64)
    outputs = []
    for i, future_pose in enumerate(future_ego_to_world):
        outputs.append(w2det_predict(
            hist[-1], hist[-2], current_pose, previous_pose, np.asarray(future_pose),
            dt_future_s=(i + 1) * float(frame_dt_s),
            dt_previous_s=float(frame_dt_s),
            grid=grid,
            cfg=cfg,
        ))
    return np.stack(outputs, axis=0)
