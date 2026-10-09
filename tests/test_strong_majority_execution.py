from concurrent.futures import ThreadPoolExecutor
import shutil
import numpy as np
import pytest
import torch

from real_motion.strong_majority_execution import (
    ParallelNativeMajority, selected_execution, strong_majority_execution,
)
from real_motion.v18_execution_trial import majority_fill_native_exact
from real_motion.strong_w2det import majority_fill


@pytest.fixture(scope='module')
def native(tmp_path_factory):
    from real_motion.native_column_cpu import prepare_native
    if not any(shutil.which(name) for name in ('cl', 'c++', 'g++', 'clang++')):
        pytest.skip('C++ compiler unavailable; no package installation')
    prepare_native(tmp_path_factory.mktemp('majority-build'))


@pytest.mark.parametrize('workers', [1, 2, 4, 6])
@pytest.mark.parametrize('shape', [(1, 1, 1), (3, 5, 2), (23, 25, 4)])
def test_full_grid_including_halo_boundaries_and_noncontiguous(native, workers, shape):
    rng = np.random.default_rng(9401)
    with ParallelNativeMajority(workers, profile=True) as parallel:
        for probability in (0., .02, .3, .7, .99, 1.):
            sem = rng.integers(0, 18, shape, dtype=np.uint8)[::-1]
            unknown = (rng.random(shape) < probability)[::-1]
            before_sem, before_unknown = sem.copy(), unknown.copy()
            expected = majority_fill(sem, unknown)
            with strong_majority_execution(parallel):
                actual = majority_fill_native_exact(sem, unknown)
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(actual, majority_fill_native_exact(sem, unknown))
            np.testing.assert_array_equal(sem, before_sem)
            np.testing.assert_array_equal(unknown, before_unknown)
            assert actual.dtype == np.uint8 and not np.shares_memory(actual, sem)
        assert parallel.stats()['calls'] == 6


@pytest.mark.parametrize('workers', [2, 4, 6])
def test_ties_and_exact_three_tenths_use_full_grid_scipy_replay(native, workers):
    with ParallelNativeMajority(workers, profile=True) as parallel:
        # Patches cross both internal split boundaries and the global edge.
        for center in (0, 2, 4, 7):
            for labels in ([1]*3+[2]*3+[3]*2+[4]*2, [11]*4+[13]*4):
                sem = np.full((9, 9, 2), 17, np.uint8)
                unknown = np.ones(sem.shape, bool)
                positions = [(x, y, 0) for x in range(max(0, center-2), min(9, center+3))
                             for y in range(2, 7) if (x, y) != (center, 4)]
                for coordinate, label in zip(positions, labels):
                    sem[coordinate] = label; unknown[coordinate] = False
                # Extra known class outside the local block must not change
                # the original global class order or ambiguous replay.
                sem[8, 8, 1] = 0; unknown[8, 8, 1] = False
                with strong_majority_execution(parallel):
                    actual = majority_fill_native_exact(sem, unknown)
                np.testing.assert_array_equal(actual, majority_fill(sem, unknown))
        assert parallel.stats()['ambiguous_cells'] > 0


def test_context_is_scoped_restored_and_not_inherited_by_workers():
    with ParallelNativeMajority(2) as parallel:
        assert selected_execution() is None
        with strong_majority_execution(parallel):
            assert selected_execution() is parallel
            with ThreadPoolExecutor(1) as pool:
                assert pool.submit(selected_execution).result() is None
            with pytest.raises(RuntimeError):
                with strong_majority_execution():
                    assert selected_execution() is None
                    raise RuntimeError('restore')
            assert selected_execution() is parallel
        assert selected_execution() is None


def test_reject_bad_contract_and_worker_errors_restore_scope(native):
    for value in (0, 9, True, 1.5):
        with pytest.raises(ValueError): ParallelNativeMajority(value)
    with pytest.raises(TypeError):
        with strong_majority_execution('parallel'): pass
    with ParallelNativeMajority(2) as parallel:
        for sem, unknown, kwargs, exception in (
            (np.zeros((3, 4), np.uint8), np.zeros((3, 4), bool), {}, ValueError),
            (np.zeros((3, 4, 1), np.int64), np.zeros((3, 4, 1), bool), {}, TypeError),
            (np.full((3, 4, 1), 18, np.uint8), np.zeros((3, 4, 1), bool), {}, ValueError),
            (np.zeros((3, 4, 1), np.uint8), np.zeros((3, 4, 1), bool), {'min_fraction': .4}, ValueError),
            (np.zeros((3, 4, 1), np.uint8), np.zeros((3, 4, 1), bool), {'kernel': (3, 3, 1)}, ValueError),
        ):
            with pytest.raises(exception):
                with strong_majority_execution(parallel):
                    majority_fill_native_exact(sem, unknown, **kwargs)
            assert selected_execution() is None
        # A rejected call must not poison the pool or the next prediction.
        sem = np.full((3, 4, 1), 17, np.uint8)
        np.testing.assert_array_equal(parallel(sem, np.zeros(sem.shape, bool)), sem)
    with pytest.raises(RuntimeError, match='closed'):
        parallel(sem, np.zeros(sem.shape, bool))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='actual CUDA required')
def test_strong_all_six_anchors_components_and_clear_stay_exact(native):
    from tools.real_motion.benchmark_p0_f9_v18_runtime import _strong_all_horizons
    from real_motion.geometry import OccupancyGrid
    from real_motion.runtime_fastpath import baseline_clear_flat_indices
    from real_motion.strong_w2det import StrongW2DetConfig, extract_instances, _transform_points
    from test_strong_warp_execution import transforms
    grid = OccupancyGrid(x_min=-4., y_min=-4., z_min=-1., shape_hwd=(21, 19, 5))
    sem = np.full(grid.shape_hwd, 17, np.uint8)
    sem[:, :, 0] = 11; sem[3:8, 6:12, 1:3] = 4; sem[9:12, 4:7, 2] = 2
    cfg = StrongW2DetConfig(); current = extract_instances(sem, np.eye(4), grid=grid, cfg=cfg)
    origin = np.array([grid.x_min, grid.y_min, grid.z_min]); step = np.array(grid.voxel_size)
    points = [_transform_points(np.eye(4), origin+(row['voxel_indices']+.5)*step) for row in current]
    velocities = {i: np.array([.31, -.11, 0.]) for i in range(len(current))}
    def run():
        return _strong_all_horizons(sem, np.eye(4), transforms()[:6], current, velocities, points,
            frame_dt_s=.5, grid=grid, cfg=cfg, runtime_device='cuda', majority_backend='native')
    original = run()
    with ParallelNativeMajority(4) as parallel, strong_majority_execution(parallel):
        actual = run()
    for old, new in zip(original[0], actual[0]): np.testing.assert_array_equal(old, new)
    for old, new in zip(original[1], actual[1]):
        assert len(old) == len(new)
        for x, y in zip(old, new):
            assert x.class_id == y.class_id and x.source_voxel_count == y.source_voxel_count
            np.testing.assert_array_equal(x.voxel_indices, y.voxel_indices)
        np.testing.assert_array_equal(baseline_clear_flat_indices(old, grid=grid),
                                      baseline_clear_flat_indices(new, grid=grid))
