"""Full-grid/large-source candidate equality, including original warm artifacts."""
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.ndimage import binary_dilation
from real_motion.geometry import OccupancyGrid
from real_motion.causal_column_completion import ColumnConfig
from real_motion.rigid_transport import RasterizedRigidComponent, rigid_source_points_world
from tools.real_motion import causal_column_common as col
from tools.real_motion import joint_column_common as common


def candidate_fixture(sources=32, narrow=False, legacy=True):
    grid = OccupancyGrid(x_min=0, y_min=0, z_min=0, voxel_size=(.4, .4, .4), shape_hwd=(200, 200, 16))
    cfg = ColumnConfig(z_bins=16)
    m = np.full(grid.shape_hwd, 17, np.uint8)
    m[:105, :, 2] = 11; m[:105, 130:, 2] = 13
    b = m.copy(); b[96:110, :, 2] = 17; b[20:40, 20:100, 2] = 17
    footprint = np.zeros((200, 200), bool); footprint[:4 if narrow else 100] = True
    own = np.full(grid.shape_hwd, -1, np.int32); fall = b.copy()
    current, registrations, components, targets = [], [], [], []
    for i in range(sources):
        x, y = 10+5*(i % 8), 20+15*(i//8)
        indices = np.array([[a, c, d] for a in range(x, x+6) for c in range(y, y+4) for d in range(2, 5)])
        center = rigid_source_points_world(indices, np.eye(4), grid=grid).mean(0)
        current.append(dict(class_id=4, centroid_world=center, voxel_indices=indices))
        registrations.append([(np.eye(4), indices)]*4)
        future = indices+np.array([2, 1, 0]); at = tuple(future.T)
        components.append(RasterizedRigidComponent(4, future, len(future)))
        own[at] = i; b[at] = 4; targets.append(center+np.array([.8, .4, 0]))
    prep = SimpleNamespace(raw=dict(history_occ=np.stack([m]*4), history_observed=np.ones((4, *grid.shape_hwd), bool),
        history_poses=[np.eye(4)]*4, future_poses=[np.eye(4)]*6, future_gt_occ=[m]*6),
        baseline=[b]*6, owners=[own]*6, fallbacks=[fall]*6, memory=[m]*6, footprints=[footprint]*6,
        state=dict(current=current, current_pose=np.eye(4), world_to_future=[np.eye(4)]*6),
        components=[components]*6, targets=[targets]*6, yaws=[[.13]*sources]*6, registrations=registrations,
        cpu_pipeline_optimized=True)
    prep.fixed_candidate_geometry = col.fixed_candidate_geometry(prep.memory, prep.footprints, grid, cfg)
    if legacy:
        for geometry in prep.fixed_candidate_geometry:
            geometry.pop('generation_xy'); geometry.pop('static_xy')
    return prep, grid, cfg


@pytest.mark.parametrize('sources', [0, 8, 32])
@pytest.mark.parametrize('narrow', [False, True])
@pytest.mark.parametrize('legacy', [False, True])
def test_full_grid_candidate_population_labels_rng_and_context_exact(sources, narrow, legacy):
    prep, grid, cfg = candidate_fixture(sources, narrow, legacy)
    prep.cpu_kernels_optimized = False
    previous = common.build_online_column_candidates(prep, cfg, grid)
    prep.cpu_kernels_optimized = True
    current = common.build_online_column_candidates(prep, cfg, grid, defer_context=True)
    for (h, old, labels), (nh, wrapped, new_labels) in zip(previous, current):
        new = wrapped.subset(np.arange(len(wrapped)))
        assert h == nh and np.array_equal(labels, new_labels)
        assert all(np.array_equal(v, getattr(new, k)) for k, v in vars(old).items())
    x, y = np.random.default_rng(19), np.random.default_rng(19)
    prep.cpu_kernels_optimized = False
    old = common.select_online_columns(prep, cfg, grid, x, candidates=previous)
    prep.cpu_kernels_optimized = True
    new = common.select_online_columns(prep, cfg, grid, y, candidates=current)
    assert x.bit_generator.state == y.bit_generator.state
    for a, b in zip(old, new):
        assert a[0] == b[0] and all(np.array_equal(v, getattr(b[1], k)) for k, v in vars(a[1]).items())
        assert np.array_equal(a[2], b[2]) and np.array_equal(a[3], b[3])


def test_manual_dilation_exact_cross_stencil_all_borders_and_small_grids():
    rng = np.random.default_rng(90)
    for shape in ((1, 1), (1, 7), (4, 7), (200, 200)):
        for _ in range(20):
            xy = np.column_stack([rng.integers(0, n, 50) for n in shape])
            mask = np.zeros(shape, bool); mask[tuple(xy.T)] = True
            expected = np.argwhere(binary_dilation(mask))
            assert np.array_equal(col.padded_support_xy(xy, shape, optimize=True), expected)


@pytest.mark.parametrize('case', ['small', 'huge_xy', 'huge_actor', 'uint64'])
def test_packed_uniqueness_matches_reference_with_duplicates_and_int64_overflow_fallback(case):
    prep, grid, cfg = candidate_fixture(0)
    plan = col.candidate_plan(prep, 0, grid, cfg).subset(np.arange(100))
    plan.xy = plan.xy.astype(np.int64)
    if case == 'huge_xy': plan.xy[:, 0] += 2**62
    if case == 'huge_actor':
        plan.kind[:] = 1; plan.actor = np.arange(100, dtype=np.int64)+2**62
    if case == 'uint64':
        plan.kind[:] = 1; plan.actor = np.arange(100, dtype=np.uint64)+np.uint64(2**53)
        plan.xy = plan.xy.astype(np.uint64)
    plan.validate(); plan.validate(packed_keys=True)
    plan.xy[-1] = plan.xy[0]; plan.actor[-1] = plan.actor[0]
    for flag in (False, True):
        with pytest.raises(ValueError, match='duplicate'): plan.validate(packed_keys=flag)
