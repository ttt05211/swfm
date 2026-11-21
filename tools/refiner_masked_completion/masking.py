"""One world-fixed forward cut, sampled from window identity only."""
import hashlib
import json

import numpy as np

from real_motion.metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS
from tools.joint_occ_experiments.soft_geometry import grid_arrays


MASK_RECIPE = dict(version='t0_forward_static_holdout_v1', probability=0.5,
                   depths_m=[8, 12, 16], shared_all_horizons=True)


def depth_for(record, seed, *, force=False):
    identity = [MASK_RECIPE['version'], seed, str(record['scene_name']), str(record['t0_token'])]
    digest = hashlib.sha256(json.dumps(identity).encode()).digest()
    if force or digest[0] < 128:
        return MASK_RECIPE['depths_m'][digest[1] % 3]
    return None


def axes(grid, *, coarse=False):
    origin, step, shape = grid_arrays(grid)
    spacing = step * ([2, 2, 1] if coarse else [1, 1, 1])
    size = shape // ([2, 2, 1] if coarse else [1, 1, 1])
    return np.ix_(*(o + (np.arange(n) + 0.5) * s for o, n, s in zip(origin, size, spacing)))


def coordinate(axis, matrix, indices):
    return matrix[axis, 3] + sum(matrix[axis, j] * values for j, values in enumerate(indices))


def forward_cut(grid, target_to_t0, depth_m, *, coarse=False):
    origin, step, shape = grid_arrays(grid)
    end = origin[0] + step[0] * shape[0]
    if not 0 < depth_m < end - origin[0]:
        raise ValueError('cut depth must be inside the native grid')
    return coordinate(0, target_to_t0, axes(grid, coarse=coarse)) >= end - depth_m


def mask_history(raw, history, grid, depth_m):
    """Mask static encoder inputs AND static gather weights before future alignment."""
    poses = np.asarray(raw['history_poses'], dtype=np.float64)
    to_t0 = np.linalg.inv(poses[-1]) @ poses
    native = np.stack([forward_cut(grid, matrix, depth_m) for matrix in to_t0])
    coarse = np.stack([forward_cut(grid, matrix, depth_m, coarse=True) for matrix in to_t0])
    labels = np.asarray(raw['history_occ'])
    hidden = native & ~np.isin(labels, DYNAMIC_CLASS_IDS)
    observed = np.asarray(raw['history_observed']) & ~hidden
    # The original prior reads t0 semantic labels directly. Moving sources
    # are provided separately by the original transport using the clean state.
    prior_labels = labels.copy()
    prior_labels[-1][native[-1]] = 17
    memory = history[0]['memory_reads']
    masked_memory = dict(memory, static_valid=np.where(coarse, 0, memory['static_valid']))
    return dict(raw, history_occ=prior_labels), observed, (dict(history[0], memory_reads=masked_memory), *history[1:])


def regions(grid, histories, future_pose, depth_m):
    """Common target-free masks at 3s, derived from physical grid footprints."""
    idx = axes(grid)
    origin, step, shape = grid_arrays(grid)
    end = origin + step * shape
    transforms = np.linalg.inv(np.asarray(histories, dtype=np.float64)) @ future_pose

    def inside(matrix, dimensions):
        mask = np.ones(tuple(shape), dtype=bool)
        for j in range(dimensions):
            value = coordinate(j, matrix, idx)
            mask &= (value >= origin[j]) & (value < end[j])
        return mask

    outside = ~np.logical_or.reduce([inside(matrix, 2) for matrix in transforms])
    artificial = inside(transforms[-1], 3) & forward_cut(grid, transforms[-1], depth_m)
    return dict(natural_front_3s=outside & (idx[0] >= 0), artificial_cut_3s=artificial)
