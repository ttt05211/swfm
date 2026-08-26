"""Causal occupancy-level constant-motion KTA baseline.

This is intentionally simple and replaceable. It uses only historical
occupancy after ego compensation, matches current BEV components to the previous
frame, estimates a constant planar velocity, and extrapolates current voxels.
Future GT is never consumed.
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
    bev_cells: np.ndarray       # [N,2] y,x
    voxel_indices: np.ndarray   # [M,3] y,x,z
    centroid_xy_m: np.ndarray   # [2] x,y
    velocity_xy_mps: np.ndarray # [2] x,y
    matched: bool


def _components_8(mask_hw: np.ndarray):
    """Small dependency-free 8-connected component extraction."""
    m = np.asarray(mask_hw, dtype=bool)
    H, W = m.shape
    seen = np.zeros_like(m)
    comps = []
    for y0, x0 in np.argwhere(m):
        if seen[y0, x0]:
            continue
        stack = [(int(y0), int(x0))]
        seen[y0, x0] = True
        cells = []
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < H and 0 <= xx < W and m[yy, xx] and not seen[yy, xx]:
                        seen[yy, xx] = True
                        stack.append((yy, xx))
        comps.append(np.asarray(cells, dtype=np.int64))
    return comps


def _centroid_xy(cells_yx: np.ndarray, grid: OccupancyGrid):
    vx, vy, _ = grid.voxel_size
    y = grid.y_min + (cells_yx[:, 0] + 0.5) * vy
    x = grid.x_min + (cells_yx[:, 1] + 0.5) * vx
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
        raise ValueError("aligned_history must be [T,H,W,D] with T>=2")
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
        semantics: [F,H,W,D] in the same reference ego grid as current_semantics.
        bev_support: [F,H,W]
    """
    H, W, D = grid.shape_hwd
    vx, vy, _ = grid.voxel_size
    outputs = []
    supports = []
    # Large components first gives deterministic collision behavior.
    ordered = sorted(components, key=lambda c: (-len(c.voxel_indices), c.class_id))
    for horizon in horizons_s:
        out = np.full((H, W, D), free_label, dtype=current_semantics.dtype)
        sup = np.zeros((H, W), dtype=bool)
        for comp in ordered:
            dx = int(np.rint(comp.velocity_xy_mps[0] * float(horizon) / vx))
            dy = int(np.rint(comp.velocity_xy_mps[1] * float(horizon) / vy))
            vox = comp.voxel_indices
            yy = vox[:, 0] + dy
            xx = vox[:, 1] + dx
            zz = vox[:, 2]
            valid = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W) & (zz >= 0) & (zz < D)
            if not valid.any():
                continue
            yy, xx, zz = yy[valid], xx[valid], zz[valid]
            # Only write into still-free destination cells. Ordering is deterministic.
            free_dst = out[yy, xx, zz] == free_label
            yy, xx, zz = yy[free_dst], xx[free_dst], zz[free_dst]
            out[yy, xx, zz] = comp.class_id
            sup[yy, xx] = True
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
