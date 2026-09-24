import math
import numpy as np

from real_motion.geometry import (
    OccupancyGrid,
    warp_mask,
    warp_semantic_and_mask,
    warp_semantic_grid,
)


def _transform(theta=0.13, tx=0.7, ty=-0.35):
    c, s = math.cos(theta), math.sin(theta)
    T = np.eye(4, dtype=np.float64)
    T[0, 0] = c
    T[0, 1] = -s
    T[1, 0] = s
    T[1, 1] = c
    T[0, 3] = tx
    T[1, 3] = ty
    return T


def test_fast_warp_mask_matches_historical_semantic_route():
    grid = OccupancyGrid(
        x_min=-4.0,
        y_min=-4.0,
        z_min=-1.0,
        voxel_size=(0.4, 0.4, 0.4),
        shape_hwd=(20, 20, 5),
    )
    rng = np.random.default_rng(7)
    mask = rng.random(grid.shape_hwd) < 0.17
    synthetic = np.zeros(grid.shape_hwd, dtype=np.uint8)
    synthetic[mask] = 1
    ref = warp_semantic_grid(
        synthetic,
        _transform(),
        grid=grid,
        free_label=0,
    ) == 1
    got = warp_mask(mask, _transform(), grid=grid)
    assert np.array_equal(got, ref)


def test_fused_semantic_mask_matches_two_historical_warps():
    grid = OccupancyGrid(
        x_min=-4.0,
        y_min=-4.0,
        z_min=-1.0,
        voxel_size=(0.4, 0.4, 0.4),
        shape_hwd=(20, 20, 5),
    )
    free = 17
    rng = np.random.default_rng(11)
    observed = rng.random(grid.shape_hwd) < 0.25
    semantics = np.full(grid.shape_hwd, free, dtype=np.uint8)
    occupied = observed & (rng.random(grid.shape_hwd) < 0.22)
    semantics[occupied] = rng.integers(0, 17, size=int(occupied.sum()), dtype=np.uint8)

    masked = semantics.copy()
    masked[~observed] = free
    ref_sem = warp_semantic_grid(
        masked,
        _transform(),
        grid=grid,
        free_label=free,
    )
    synthetic = np.zeros(grid.shape_hwd, dtype=np.uint8)
    synthetic[observed] = 1
    ref_mask = warp_semantic_grid(
        synthetic,
        _transform(),
        grid=grid,
        free_label=0,
    ) == 1

    got_sem, got_mask = warp_semantic_and_mask(
        semantics,
        observed,
        _transform(),
        grid=grid,
        free_label=free,
    )
    assert np.array_equal(got_sem, ref_sem)
    assert np.array_equal(got_mask, ref_mask)
