"""Causal swept support for already-moving occupancy components.

The swept mask is not a future prediction.  It is only a cheap reachable
corridor from the current object footprint to the constant-velocity KTA
endpoint.  Learned prediction is still responsible for deciding the actual
future occupancy inside this support.
"""
from __future__ import annotations

from typing import Sequence
import numpy as np

from .geometry import OccupancyGrid, relative_transform, warp_mask
from .kta import MotionComponent


def swept_component_voxel_masks(
    components: Sequence[MotionComponent],
    horizons_s: Sequence[float],
    grid: OccupancyGrid = OccupancyGrid(),
) -> np.ndarray:
    """Rasterize current->KTA-endpoint corridors in the current ego frame.

    Each component keeps its observed 3-D footprint.  For a horizon, the
    footprint is translated along every integer cell on the straight line from
    zero displacement to the KTA constant-velocity endpoint.  Components are
    swept independently, so unrelated objects are never connected by a large
    rectangle.

    Returns:
        Boolean array [F,X,Y,Z] in the current (t0) ego frame.
    """
    horizons = [float(h) for h in horizons_s]
    X, Y, Z = grid.shape_hwd
    vx, vy, _ = grid.voxel_size
    out = np.zeros((len(horizons), X, Y, Z), dtype=bool)

    for fi, horizon in enumerate(horizons):
        for comp in components:
            vox = np.asarray(comp.voxel_indices, dtype=np.int64)
            if vox.size == 0:
                continue
            dx = int(np.rint(float(comp.velocity_xy_mps[0]) * horizon / vx))
            dy = int(np.rint(float(comp.velocity_xy_mps[1]) * horizon / vy))
            steps = max(abs(dx), abs(dy), 1)
            # linspace + rint is a compact Bresenham-like integer sweep.  The
            # set removes duplicate shifts caused by rounding at shallow slopes.
            shifts = {
                (
                    int(np.rint(dx * step / steps)),
                    int(np.rint(dy * step / steps)),
                )
                for step in range(steps + 1)
            }
            for sx, sy in shifts:
                xx = vox[:, 0] + sx
                yy = vox[:, 1] + sy
                zz = vox[:, 2]
                valid = (
                    (xx >= 0) & (xx < X) &
                    (yy >= 0) & (yy < Y) &
                    (zz >= 0) & (zz < Z)
                )
                if valid.any():
                    out[fi, xx[valid], yy[valid], zz[valid]] = True
    return out


def swept_support_in_future_ego(
    components: Sequence[MotionComponent],
    horizons_s: Sequence[float],
    t0_ego_to_world: np.ndarray,
    future_ego_to_world: Sequence[np.ndarray],
    grid: OccupancyGrid = OccupancyGrid(),
) -> np.ndarray:
    """Return swept BEV support [F,X,Y] in each horizon's ego frame."""
    if len(horizons_s) != len(future_ego_to_world):
        raise ValueError("horizons and future poses length mismatch")
    swept_t0 = swept_component_voxel_masks(components, horizons_s, grid)
    future = []
    for mask_t0, pose_h in zip(swept_t0, future_ego_to_world):
        rel = relative_transform(t0_ego_to_world, pose_h)
        warped = warp_mask(mask_t0, rel, grid=grid)
        future.append(warped.any(axis=2))
    return np.stack(future, axis=0)
