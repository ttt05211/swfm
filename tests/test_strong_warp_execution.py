from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest
import torch
from real_motion.geometry import OccupancyGrid, quaternion_wxyz_to_matrix
from real_motion.runtime_fastpath import inverse_warp_sequence_cuda_exact
from real_motion.strong_w2det import inverse_warp, StrongW2DetConfig, extract_instances, _transform_points
from real_motion.strong_warp_execution import selected_backend, strong_warp_execution


def transforms():
    rows = []
    for translation, quaternion in [((0, 0, 0), (1, 0, 0, 0)),
            ((.2, -.2, .2), (1, 0, 0, 0)), ((40, -40, 2), (1, 0, 0, 0)),
            ((200, 200, 200), (1, 0, 0, 0)), ((.17, -.11, .031), (.99, .01, -.02, .03)),
            ((-.391, .511, -.013), (.97, -.02, .03, -.09)), ((.200001, -.199999, .0), (1, 0, 0, 0))]:
        pose = np.eye(4); pose[:3, :3] = quaternion_wxyz_to_matrix(quaternion)
        pose[:3, 3] = translation; rows.append(pose)
    return rows


def test_opt_in_context_restored_and_does_not_change_worker_defaults():
    assert selected_backend() == 'reference'
    with strong_warp_execution('buffered'):
        assert selected_backend() == 'buffered'
        with ThreadPoolExecutor(1) as pool:
            assert pool.submit(selected_backend).result() == 'reference'
        with pytest.raises(RuntimeError):
            with strong_warp_execution('reference'):
                raise RuntimeError('deliberate')
        assert selected_backend() == 'buffered'
    assert selected_backend() == 'reference'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='actual CUDA required')
@pytest.mark.parametrize('dtype', [np.uint8, np.int64])
@pytest.mark.parametrize('shape', [(1, 1, 1), (21, 19, 5)])
def test_buffered_equals_original_and_float64_reference(dtype, shape):
    grid = OccupancyGrid(x_min=-4., y_min=-4., z_min=-1., shape_hwd=shape)
    sem = np.random.default_rng(20261009).integers(0, 18, shape).astype(dtype)[::-1]
    before = sem.copy(); poses = transforms()
    args = dict(grid=grid, free_label=17, device='cuda')
    with strong_warp_execution('reference'):
        original = inverse_warp_sequence_cuda_exact(sem, poses, **args)
    with strong_warp_execution('buffered'):
        buffered = inverse_warp_sequence_cuda_exact(sem, poses, **args)
        assert inverse_warp_sequence_cuda_exact(sem, [], **args) == []
    assert len(buffered) == len(original) == 7  # second bounded group exercised
    for pose, expected, actual in zip(poses, original, buffered):
        reference = inverse_warp(sem, pose, grid, 17)
        for ref, old, new in zip(reference, expected, actual):
            np.testing.assert_array_equal(new, old)
            np.testing.assert_array_equal(new, ref)
            assert new.dtype == ref.dtype
    np.testing.assert_array_equal(sem, before)
    assert not np.shares_memory(buffered[0][0], buffered[1][0])
    assert not np.shares_memory(buffered[0][1], buffered[1][1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='actual CUDA required')
def test_full_strong_anchors_components_and_clear_unchanged():
    from tools.real_motion.benchmark_p0_f9_v18_runtime import _strong_all_horizons
    from real_motion.runtime_fastpath import baseline_clear_flat_indices
    grid = OccupancyGrid(x_min=-4., y_min=-4., z_min=-1., shape_hwd=(21, 19, 5))
    sem = np.full(grid.shape_hwd, 17, np.uint8)
    sem[:, :, 0] = 11; sem[3:8, 6:12, 1:3] = 4; sem[9:12, 4:7, 2] = 2
    cfg = StrongW2DetConfig(); current = extract_instances(sem, np.eye(4), grid=grid, cfg=cfg)
    origin = np.array([grid.x_min, grid.y_min, grid.z_min]); step = np.array(grid.voxel_size)
    points = [_transform_points(np.eye(4), origin+(row['voxel_indices']+.5)*step) for row in current]
    velocities = {i: np.array([.31, -.11, 0.]) for i in range(len(current))}
    rows = []
    for backend in ('reference', 'buffered'):
        with strong_warp_execution(backend):
            rows.append(_strong_all_horizons(sem, np.eye(4), transforms()[:6], current, velocities, points,
                frame_dt_s=.5, grid=grid, cfg=cfg, runtime_device='cuda', majority_backend='dense_cuda'))
    for old, new in zip(rows[0][0], rows[1][0]): np.testing.assert_array_equal(old, new)
    for old, new in zip(rows[0][1], rows[1][1]):
        assert len(old) == len(new)
        for x, y in zip(old, new):
            assert x.class_id == y.class_id and x.source_voxel_count == y.source_voxel_count
            np.testing.assert_array_equal(x.voxel_indices, y.voxel_indices)
        np.testing.assert_array_equal(baseline_clear_flat_indices(old, grid=grid),
                                      baseline_clear_flat_indices(new, grid=grid))
