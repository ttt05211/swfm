"""Rigid geometry for ego-aligned Occ3D occupancy processing.

Official Occ3D array convention
-------------------------------
The released ``semantics``/``mask_lidar`` arrays are indexed as ``[X,Y,Z]``:
array axis 0 maps to metric x, axis 1 to metric y, and axis 2 to metric z.
Occ3D's official ``voxel2points`` helper uses exactly this mapping.

For backward API compatibility the dataclass field is still named
``shape_hwd`` throughout this repository, but its value must be interpreted as
``(X,Y,Z)``.  The default nuScenes volume is 200x200x16 with range
[-40,-40,-1, 40,40,5.4] and 0.4 m voxels.
"""
from dataclasses import dataclass
from typing import Sequence
import math
import numpy as np


@dataclass(frozen=True)
class OccupancyGrid:
    x_min: float = -40.0
    y_min: float = -40.0
    z_min: float = -1.0
    voxel_size: tuple = (0.4, 0.4, 0.4)
    # Legacy field name; order is (X,Y,Z), matching official Occ3D arrays.
    shape_hwd: tuple = (200, 200, 16)

    @property
    def x_max(self):
        return self.x_min + self.shape_hwd[0] * self.voxel_size[0]

    @property
    def y_max(self):
        return self.y_min + self.shape_hwd[1] * self.voxel_size[1]

    @property
    def z_max(self):
        return self.z_min + self.shape_hwd[2] * self.voxel_size[2]


def quaternion_wxyz_to_matrix(q: Sequence[float]) -> np.ndarray:
    """Quaternion [w,x,y,z] -> 3x3 rotation matrix."""
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (4,):
        raise ValueError("quaternion must be [w,x,y,z]")
    n = float(np.dot(q, q))
    if n < 1e-16:
        raise ValueError("zero quaternion")
    q = q / math.sqrt(n)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def quaternion_yaw(q: Sequence[float]) -> float:
    r = quaternion_wxyz_to_matrix(q)
    return float(math.atan2(r[1, 0], r[0, 0]))


def pose_matrix(translation_xyz: Sequence[float], rotation_wxyz: Sequence[float]) -> np.ndarray:
    t = np.asarray(translation_xyz, dtype=np.float64)
    if t.shape != (3,):
        raise ValueError("translation must be xyz")
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = quaternion_wxyz_to_matrix(rotation_wxyz)
    out[:3, 3] = t
    return out


def relative_transform(src_ego_to_world: np.ndarray, dst_ego_to_world: np.ndarray) -> np.ndarray:
    """Transform points expressed in src ego coordinates into dst ego coordinates."""
    return np.linalg.inv(np.asarray(dst_ego_to_world)) @ np.asarray(src_ego_to_world)


def _occupied_indices_to_xyz(indices_xyz: np.ndarray, grid: OccupancyGrid) -> np.ndarray:
    ix, iy, iz = indices_xyz.T
    vx, vy, vz = grid.voxel_size
    x = grid.x_min + (ix.astype(np.float64) + 0.5) * vx
    y = grid.y_min + (iy.astype(np.float64) + 0.5) * vy
    z = grid.z_min + (iz.astype(np.float64) + 0.5) * vz
    return np.stack([x, y, z], axis=1)


def _xyz_to_indices(xyz: np.ndarray, grid: OccupancyGrid):
    vx, vy, vz = grid.voxel_size
    ix = np.floor((xyz[:, 0] - grid.x_min) / vx).astype(np.int64)
    iy = np.floor((xyz[:, 1] - grid.y_min) / vy).astype(np.int64)
    iz = np.floor((xyz[:, 2] - grid.z_min) / vz).astype(np.int64)
    X, Y, Z = grid.shape_hwd
    valid = (ix >= 0) & (ix < X) & (iy >= 0) & (iy < Y) & (iz >= 0) & (iz < Z)
    return ix, iy, iz, valid


def warp_semantic_grid(
    semantics: np.ndarray,
    src_to_dst: np.ndarray,
    grid: OccupancyGrid = OccupancyGrid(),
    free_label: int = 17,
) -> np.ndarray:
    """Nearest-cell rigid warp for official Occ3D ``[X,Y,Z]`` arrays.

    In a collision, the transformed source voxel whose center is closest to the
    destination cell center wins. This avoids order-dependent writes.
    """
    sem = np.asarray(semantics)
    if tuple(sem.shape) != tuple(grid.shape_hwd):
        raise ValueError(f"semantic grid {sem.shape} != configured {grid.shape_hwd}")
    occ_idx = np.argwhere(sem != free_label)
    out = np.full_like(sem, free_label)
    if len(occ_idx) == 0:
        return out

    xyz = _occupied_indices_to_xyz(occ_idx, grid)
    xyz_h = np.concatenate([xyz, np.ones((len(xyz), 1), dtype=np.float64)], axis=1)
    dst_xyz = (np.asarray(src_to_dst, dtype=np.float64) @ xyz_h.T).T[:, :3]
    ix, iy, iz, valid = _xyz_to_indices(dst_xyz, grid)
    if not valid.any():
        return out

    src_idx = occ_idx[valid]
    vals = sem[tuple(src_idx.T)]
    ix, iy, iz = ix[valid], iy[valid], iz[valid]
    dst_xyz = dst_xyz[valid]

    vx, vy, vz = grid.voxel_size
    centers = np.stack([
        grid.x_min + (ix + 0.5) * vx,
        grid.y_min + (iy + 0.5) * vy,
        grid.z_min + (iz + 0.5) * vz,
    ], axis=1)
    dist2 = np.sum((dst_xyz - centers) ** 2, axis=1)
    X, Y, Z = grid.shape_hwd
    flat = (ix * Y + iy) * Z + iz

    order = np.lexsort((dist2, flat))
    flat_sorted = flat[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = flat_sorted[1:] != flat_sorted[:-1]
    chosen = order[first]
    out[ix[chosen], iy[chosen], iz[chosen]] = vals[chosen]
    return out


def warp_mask(mask: np.ndarray, src_to_dst: np.ndarray, grid: OccupancyGrid = OccupancyGrid()) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    if tuple(m.shape) != tuple(grid.shape_hwd):
        raise ValueError("mask shape mismatch")
    synthetic = np.zeros(grid.shape_hwd, dtype=np.uint8)
    synthetic[m] = 1
    warped = warp_semantic_grid(synthetic, src_to_dst, grid=grid, free_label=0)
    return warped == 1


def ego_compensate_sequence(
    semantics: Sequence[np.ndarray],
    ego_to_world: Sequence[np.ndarray],
    reference_index: int = -1,
    grid: OccupancyGrid = OccupancyGrid(),
    free_label: int = 17,
) -> np.ndarray:
    if len(semantics) != len(ego_to_world):
        raise ValueError("semantics and poses length mismatch")
    ref = ego_to_world[reference_index]
    aligned = []
    for sem, pose in zip(semantics, ego_to_world):
        aligned.append(warp_semantic_grid(sem, relative_transform(pose, ref), grid, free_label))
    return np.stack(aligned, axis=0)


def transport_current_to_future(
    current_semantics: np.ndarray,
    current_ego_to_world: np.ndarray,
    future_ego_to_world: Sequence[np.ndarray],
    grid: OccupancyGrid = OccupancyGrid(),
    free_label: int = 17,
) -> np.ndarray:
    return np.stack([
        warp_semantic_grid(
            current_semantics,
            relative_transform(current_ego_to_world, pose_h),
            grid=grid,
            free_label=free_label,
        )
        for pose_h in future_ego_to_world
    ], axis=0)


def transport_mask_to_future(
    current_mask: np.ndarray,
    current_ego_to_world: np.ndarray,
    future_ego_to_world: Sequence[np.ndarray],
    grid: OccupancyGrid = OccupancyGrid(),
) -> np.ndarray:
    return np.stack([
        warp_mask(current_mask, relative_transform(current_ego_to_world, pose_h), grid=grid)
        for pose_h in future_ego_to_world
    ], axis=0)
