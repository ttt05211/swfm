"""Rigid transport helpers for the no-WM motion-transport probe.

The functions in this module are intentionally small and occupancy-native.  A
causal source component is represented by the exact occupied voxels observed at
t0.  The component is moved in WORLD coordinates by an object-centric SE(2)
transform, then rasterized into a requested future ego grid.

This is not a learned model.  It is used to answer a narrower question before a
new training branch is introduced: if object motion were known, how much of the
remaining Moving-mIoU gap could be closed by transporting the observed 3D source
shape instead of asking a world model to regenerate it?
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np

from .geometry import OccupancyGrid


@dataclass(frozen=True)
class RasterizedRigidComponent:
    """One transported semantic component in a future ego grid."""

    class_id: int
    voxel_indices: np.ndarray  # [N,3] unique destination indices
    source_voxel_count: int

    def mask(self, grid: OccupancyGrid = OccupancyGrid()) -> np.ndarray:
        out = np.zeros(grid.shape_hwd, dtype=bool)
        idx = np.asarray(self.voxel_indices, dtype=np.int64)
        if len(idx):
            out[idx[:, 0], idx[:, 1], idx[:, 2]] = True
        return out


def wrap_angle(angle: float) -> float:
    """Wrap radians into [-pi, pi)."""
    return float((float(angle) + math.pi) % (2.0 * math.pi) - math.pi)


def _indices_to_ego_xyz(indices_xyz: np.ndarray, grid: OccupancyGrid) -> np.ndarray:
    idx = np.asarray(indices_xyz, dtype=np.int64)
    if idx.ndim != 2 or idx.shape[1] != 3:
        raise ValueError("voxel_indices must be [N,3]")
    origin = np.asarray([grid.x_min, grid.y_min, grid.z_min], dtype=np.float64)
    step = np.asarray(grid.voxel_size, dtype=np.float64)
    return origin[None] + (idx.astype(np.float64) + 0.5) * step[None]


def _transform_points(T: np.ndarray, pts_xyz: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts_xyz, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must be [N,3]")
    mat = np.asarray(T, dtype=np.float64)
    if mat.shape != (4, 4):
        raise ValueError("transform must be 4x4")
    return pts @ mat[:3, :3].T + mat[:3, 3]


def _metric_to_indices(points_xyz: np.ndarray, grid: OccupancyGrid) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points_xyz, dtype=np.float64)
    origin = np.asarray([grid.x_min, grid.y_min, grid.z_min], dtype=np.float64)
    step = np.asarray(grid.voxel_size, dtype=np.float64)
    idx = np.floor((pts - origin[None]) / step[None]).astype(np.int64)
    shape = np.asarray(grid.shape_hwd, dtype=np.int64)
    valid = ((idx >= 0) & (idx < shape[None])).all(axis=1)
    return idx, valid


def _deduplicate_indices(indices_xyz: np.ndarray, grid: OccupancyGrid) -> np.ndarray:
    idx = np.asarray(indices_xyz, dtype=np.int64)
    if len(idx) == 0:
        return idx.reshape(0, 3)
    X, Y, Z = [int(x) for x in grid.shape_hwd]
    flat = (idx[:, 0] * Y + idx[:, 1]) * Z + idx[:, 2]
    _, first = np.unique(flat, return_index=True)
    return idx[np.sort(first)]


def rasterize_rigid_component(
    voxel_indices: np.ndarray,
    class_id: int,
    current_ego_to_world: np.ndarray,
    future_ego_to_world: np.ndarray,
    *,
    source_center_world: Sequence[float],
    target_center_world: Sequence[float],
    yaw_delta_rad: float = 0.0,
    translate_z: bool = False,
    grid: OccupancyGrid = OccupancyGrid(),
) -> RasterizedRigidComponent:
    """Move one t0 component by an object-centric rigid transform.

    ``voxel_indices`` are the exact t0 occupied voxels in the current ego grid.
    They are first mapped to WORLD.  In XY they are rotated around
    ``source_center_world`` and translated so that the pivot reaches
    ``target_center_world``.  Z is preserved by default because the proposed
    learned transport contract is planar ``(dx, dy, dyaw)``; ``translate_z`` is
    exposed only for diagnostic use.
    """
    src_idx = np.asarray(voxel_indices, dtype=np.int64)
    if src_idx.ndim != 2 or src_idx.shape[1] != 3:
        raise ValueError("voxel_indices must be [N,3]")
    if len(src_idx) == 0:
        return RasterizedRigidComponent(int(class_id), np.zeros((0, 3), dtype=np.int64), 0)

    source_center = np.asarray(source_center_world, dtype=np.float64)
    target_center = np.asarray(target_center_world, dtype=np.float64)
    if source_center.shape != (3,) or target_center.shape != (3,):
        raise ValueError("source/target center must be xyz")

    pts_ego = _indices_to_ego_xyz(src_idx, grid)
    pts_world = _transform_points(current_ego_to_world, pts_ego)

    theta = float(yaw_delta_rad)
    c, s = math.cos(theta), math.sin(theta)
    rel = pts_world[:, :2] - source_center[None, :2]
    rot = np.empty_like(rel)
    rot[:, 0] = c * rel[:, 0] - s * rel[:, 1]
    rot[:, 1] = s * rel[:, 0] + c * rel[:, 1]
    moved_world = pts_world.copy()
    moved_world[:, :2] = target_center[None, :2] + rot
    if bool(translate_z):
        moved_world[:, 2] += float(target_center[2] - source_center[2])

    world_to_future = np.linalg.inv(np.asarray(future_ego_to_world, dtype=np.float64))
    moved_future = _transform_points(world_to_future, moved_world)
    dst_idx, valid = _metric_to_indices(moved_future, grid)
    dst_idx = _deduplicate_indices(dst_idx[valid], grid)
    return RasterizedRigidComponent(int(class_id), dst_idx, int(len(src_idx)))


def compose_component_replacements(
    anchor_occ: np.ndarray,
    baseline_components: Iterable[RasterizedRigidComponent],
    replacement_components: Iterable[RasterizedRigidComponent],
    *,
    dynamic_class_ids: Sequence[int],
    free_label: int = 17,
    grid: OccupancyGrid = OccupancyGrid(),
) -> np.ndarray:
    """Coherently replace selected Strong/KTA object predictions.

    ``anchor_occ`` is the frozen Strong-W2Det future occupancy.  For each selected
    source object we first clear the voxels where the Strong/KTA copy of that
    object would land, then write the replacement rigidly transported source
    shape.  All clears are performed before any writes so one object's write is
    never erased by a later object's baseline mask.

    Dynamic occupancy outside the selected source-object masks remains exactly
    the Strong anchor.  This makes the probe a controlled replacement rather
    than a second independent predictor pasted on top of KTA.
    """
    anchor = np.asarray(anchor_occ)
    if tuple(anchor.shape) != tuple(grid.shape_hwd):
        raise ValueError("anchor_occ shape differs from occupancy grid")
    out = anchor.copy()
    dyn_ids = np.asarray(tuple(int(x) for x in dynamic_class_ids), dtype=np.int64)
    if dyn_ids.size == 0:
        raise ValueError("dynamic_class_ids cannot be empty")

    clear = np.zeros(grid.shape_hwd, dtype=bool)
    for comp in baseline_components:
        clear |= comp.mask(grid)
    out[clear & np.isin(out, dyn_ids)] = int(free_label)

    replacements = list(replacement_components)
    replacements.sort(key=lambda c: (-int(c.source_voxel_count), int(c.class_id)))
    for comp in replacements:
        idx = np.asarray(comp.voxel_indices, dtype=np.int64)
        if len(idx):
            out[idx[:, 0], idx[:, 1], idx[:, 2]] = int(comp.class_id)
    return out
