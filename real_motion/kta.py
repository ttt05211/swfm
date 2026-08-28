"""Causal occupancy-level constant-motion KTA baseline.

This is intentionally simple and replaceable. It uses only historical
occupancy after ego compensation, matches current BEV components to the previous
frame, estimates a constant planar velocity, and extrapolates current voxels.
Future GT is never consumed.

Occ3D arrays follow official ``[X,Y,Z]`` order.
"""
from dataclasses import dataclass
from typing import Sequence
import numpy as np

from .geometry import OccupancyGrid


@dataclass(frozen=True)
class KTAConfig:
    free_label: int = 17
    history_dt_s: float = 0.5
    max_match_distance_m: float = 6.0
    min_component_cells: int = 1


@dataclass
class MotionComponent:
    class_id: int
    bev_cells: np.ndarray       # [N,2] x_index,y_index
    voxel_indices: np.ndarray   # [M,3] x_index,y_index,z_index
    centroid_xy_m: np.ndarray   # [2] x,y
    velocity_xy_mps: np.ndarray # [2] x,y
    matched: bool


def _components_8(mask_xy: np.ndarray):
    """Small dependency-free 8-connected component extraction on [X,Y]."""
    m = np.asarray(mask_xy, dtype=bool)
    X, Y = m.shape
    seen = np.zeros_like(m)
    comps = []
    for x0, y0 in np.argwhere(m):
        if seen[x0, y0]:
            continue
        stack = [(int(x0), int(y0))]
        seen[x0, y0] = True
        cells = []
        while stack:
            x, y = stack.pop()
            cells.append((x, y))
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    xx, yy = x + dx, y + dy
                    if 0 <= xx < X and 0 <= yy < Y and m[xx, yy] and not seen[xx, yy]:
                        seen[xx, yy] = True
                        stack.append((xx, yy))
        comps.append(np.asarray(cells, dtype=np.int64))
    return comps


def _centroid_xy(cells_xy: np.ndarray, grid: OccupancyGrid):
    vx, vy, _ = grid.voxel_size
    x = grid.x_min + (cells_xy[:, 0] + 0.5) * vx
    y = grid.y_min + (cells_xy[:, 1] + 0.5) * vy
    return np.array([x.mean(), y.mean()], dtype=np.float64)


def _extract_class_components(sem: np.ndarray, support: np.ndarray | None,
                              grid: OccupancyGrid, free_label: int, min_cells: int):
    classes = np.unique(sem if support is None else sem[support])
    out = []
    for cls in classes:
        cls = int(cls)
        if cls == free_label:
            continue
        voxel_mask = sem == cls
        if support is not None:
            voxel_mask &= support
        bev = voxel_mask.any(axis=2)
        for cells in _components_8(bev):
            if len(cells) < min_cells:
                continue
            cell_selector = np.zeros(bev.shape, dtype=bool)
            cell_selector[cells[:, 0], cells[:, 1]] = True
            vox = np.argwhere(voxel_mask & cell_selector[:, :, None])
            out.append((cls, cells, vox, _centroid_xy(cells, grid)))
    return out


def estimate_components(
    aligned_history: np.ndarray,
    current_candidate_mask: np.ndarray,
    grid: OccupancyGrid = OccupancyGrid(),
    cfg: KTAConfig = KTAConfig(),
):
    """Estimate current component velocities from the final two aligned frames."""
    hist = np.asarray(aligned_history)
    if hist.ndim != 4 or hist.shape[0] < 2:
        raise ValueError("aligned_history must be [T,X,Y,Z] with T>=2")
    if current_candidate_mask.shape != hist.shape[1:]:
        raise ValueError("candidate mask shape mismatch")

    prev, cur = hist[-2], hist[-1]
    current = _extract_class_components(cur, current_candidate_mask, grid, cfg.free_label,
                                        cfg.min_component_cells)
    previous = _extract_class_components(prev, None, grid, cfg.free_label,
                                         cfg.min_component_cells)
    prev_by_class = {}
    for item in previous:
        prev_by_class.setdefault(item[0], []).append(item)

    components = []
    for cls, cells, vox, centroid in current:
        best = None
        for p in prev_by_class.get(cls, []):
            dist = float(np.linalg.norm(centroid - p[3]))
            if best is None or dist < best[0]:
                best = (dist, p)
        if best is not None and best[0] <= cfg.max_match_distance_m:
            velocity = (centroid - best[1][3]) / cfg.history_dt_s
            matched = True
        else:
            velocity = np.zeros(2, dtype=np.float64)
            matched = False
        components.append(MotionComponent(cls, cells, vox, centroid, velocity, matched))
    return components


def extrapolate_components(
    current_semantics: np.ndarray,
    components: Sequence[MotionComponent],
    horizons_s: Sequence[float],
    grid: OccupancyGrid = OccupancyGrid(),
    free_label: int = 17,
):
    """Translate current component voxels under constant planar velocity.

    Returns:
        semantics: [F,X,Y,Z] in the same reference ego grid as current_semantics.
        bev_support: [F,X,Y]
    """
    X, Y, Z = grid.shape_hwd
    vx, vy, _ = grid.voxel_size
    outputs = []
    supports = []
    ordered = sorted(components, key=lambda c: (-len(c.voxel_indices), c.class_id))
    for horizon in horizons_s:
        out = np.full((X, Y, Z), free_label, dtype=current_semantics.dtype)
        sup = np.zeros((X, Y), dtype=bool)
        for comp in ordered:
            dx = int(np.rint(comp.velocity_xy_mps[0] * float(horizon) / vx))
            dy = int(np.rint(comp.velocity_xy_mps[1] * float(horizon) / vy))
            vox = comp.voxel_indices
            xx = vox[:, 0] + dx
            yy = vox[:, 1] + dy
            zz = vox[:, 2]
            valid = (xx >= 0) & (xx < X) & (yy >= 0) & (yy < Y) & (zz >= 0) & (zz < Z)
            if not valid.any():
                continue
            xx, yy, zz = xx[valid], yy[valid], zz[valid]
            free_dst = out[xx, yy, zz] == free_label
            xx, yy, zz = xx[free_dst], yy[free_dst], zz[free_dst]
            out[xx, yy, zz] = comp.class_id
            sup[xx, yy] = True
        outputs.append(out)
        supports.append(sup)
    return np.stack(outputs, axis=0), np.stack(supports, axis=0)


def causal_kta(
    aligned_history: np.ndarray,
    current_candidate_mask: np.ndarray,
    horizons_s: Sequence[float],
    grid: OccupancyGrid = OccupancyGrid(),
    cfg: KTAConfig = KTAConfig(),
):
    comps = estimate_components(aligned_history, current_candidate_mask, grid, cfg)
    sem, support = extrapolate_components(aligned_history[-1], comps, horizons_s, grid, cfg.free_label)
    return sem, support, comps
