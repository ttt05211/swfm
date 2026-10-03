"""Optional compiled backend: byte equality, actual updates and ABI safety."""
import copy
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from scipy.ndimage import binary_dilation

from real_motion import native_column_cpu as native
from real_motion.causal_column_sampling import ColumnFeatureSampler
from tools.real_motion import causal_column_common as col
from tools.real_motion import joint_column_common as common
from test_column_cpu_kernels import candidate_fixture
from test_causal_column_sampling import fixture as sampling_fixture


@pytest.fixture(scope='module')
def compiled():
    try: native._compiler()
    except RuntimeError:
        if os.environ.get('SWFM_COLUMN_CPU_BACKEND') == 'native': raise
        pytest.skip('optional C++ compiler absent')
    native.prepare_native()
    return native._loaded


@pytest.fixture(autouse=True)
def backend(compiled, monkeypatch):
    monkeypatch.setenv('SWFM_COLUMN_CPU_BACKEND', 'native')


@pytest.mark.parametrize('sources', [0, 8, 32])
@pytest.mark.parametrize('legacy', [False, True])
@pytest.mark.parametrize('narrow', [False, True])
def test_complete_population_context_labels_draws_exact(monkeypatch, sources, legacy, narrow):
    prep, grid, cfg = candidate_fixture(sources, narrow, legacy)
    monkeypatch.setenv('SWFM_COLUMN_CPU_BACKEND', 'numpy')
    old = common.build_online_column_candidates(prep, cfg, grid)
    monkeypatch.setenv('SWFM_COLUMN_CPU_BACKEND', 'native')
    new = common.build_online_column_candidates(prep, cfg, grid, defer_context=True)
    for (h, a, y), (hh, wrapped, yy) in zip(old, new):
        b = wrapped.subset(np.arange(len(wrapped)))
        assert h == hh and np.array_equal(y, yy)
        assert all(np.array_equal(v, getattr(b, k)) for k, v in vars(a).items())
    x, y = np.random.default_rng(29), np.random.default_rng(29)
    monkeypatch.setenv('SWFM_COLUMN_CPU_BACKEND', 'numpy')
    a = common.select_online_columns(prep, cfg, grid, x, candidates=old)
    monkeypatch.setenv('SWFM_COLUMN_CPU_BACKEND', 'native')
    b = common.select_online_columns(prep, cfg, grid, y, candidates=new)
    assert x.bit_generator.state == y.bit_generator.state
    for aa, bb in zip(a, b):
        assert all(np.array_equal(v, getattr(bb[1], k)) for k, v in vars(aa[1]).items())
        assert np.array_equal(aa[2], bb[2]) and np.array_equal(aa[3], bb[3])


@pytest.mark.parametrize('history', [4, 6])
@pytest.mark.parametrize('limit', [0, 64])
@pytest.mark.parametrize('seed', [0, 3])
def test_dense_sparse_patches_padding_rotation_membership_exact(monkeypatch, history, limit, seed):
    prep, grid, cfg, plan = sampling_fixture(seed)
    for k in ('history_occ', 'history_observed', 'history_poses'): prep.raw[k] = prep.raw[k][-history:]
    prep.registrations = [r[-history:] for r in prep.registrations]
    before = copy.deepcopy(prep.raw)
    monkeypatch.setenv('SWFM_COLUMN_CPU_BACKEND', 'numpy')
    old = ColumnFeatureSampler(prep, 3, plan, grid, cfg, col.pose_motion, workers=4, max_cache_mib=limit).sample(plan, None)
    monkeypatch.setenv('SWFM_COLUMN_CPU_BACKEND', 'native')
    new = ColumnFeatureSampler(prep, 3, plan, grid, cfg, col.pose_motion, workers=4, max_cache_mib=limit).sample(plan, None)
    assert all(np.array_equal(v, new[k]) for k, v in old.items())
    assert all(np.array_equal(v, before[k]) for k, v in prep.raw.items())


@pytest.mark.parametrize('history', [4, 6])
def test_multi_update_adamw_parameters_gradients_rng_exact(history):
    from test_local_cpu_pipeline import test_optimized_full_updates_loss_parameters_optimizer_rng_and_source_gradient_exact
    test_optimized_full_updates_loss_parameters_optimizer_rng_and_source_gradient_exact(history, 4)


def test_actual_numpy_native_three_updates_identical(monkeypatch):
    import torch
    from test_joint_causal_columns import fixture, optimizers, provider_for
    from tools.real_motion.joint_column_full_common import train_full_batch
    prep, grid, joint, control, rec = fixture()
    torch.nn.init.normal_(joint.columns.refinement.weight, std=.01)
    ref = copy.deepcopy(joint)
    p, q = provider_for(prep, grid, joint), provider_for(prep, grid, ref)
    opt, _ = optimizers(joint, control); ropt, _ = optimizers(ref, control)
    x, y = np.random.default_rng(67), np.random.default_rng(67)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for update in (1, 2, 3):
            monkeypatch.setenv('SWFM_COLUMN_CPU_BACKEND', 'numpy')
            a = train_full_batch(ref, ropt, q, None, [(rec, None)]*4, y, update, 100, sampling_pool=pool, probe=True)
            monkeypatch.setenv('SWFM_COLUMN_CPU_BACKEND', 'native')
            b = train_full_batch(joint, opt, p, None, [(rec, None)]*4, x, update, 100, sampling_pool=pool, probe=True)
            assert a['loss'] == b['loss'] and a['source_query_gradient_norm'] == b['source_query_gradient_norm'] > 0
            assert x.bit_generator.state == y.bit_generator.state
            assert all(torch.equal(v, ref.state_dict()[k]) for k, v in joint.state_dict().items())
            for k, state in opt.state_dict()['state'].items():
                assert all(torch.equal(v, ropt.state_dict()['state'][k][name]) for name, v in state.items())


@pytest.mark.parametrize('history', [4, 6])
def test_cuda_native_live_pose_gradients_actual_updates(history):
    import torch
    if not torch.cuda.is_available(): pytest.skip('requires real CUDA')
    from test_local_cpu_pipeline import test_cuda_live_readback_parallel_warm_updates_equal_reference
    test_cuda_live_readback_parallel_warm_updates_equal_reference(history)


def test_support_borders_empty_and_concurrent_workspaces(compiled):
    rng = np.random.default_rng(51)
    jobs = []
    for shape in ((1, 1, 1), (1, 7, 3), (4, 7, 4), (200, 200, 16)):
        for _ in range(8): jobs.append((rng.integers(0, np.prod(shape), 200, dtype=np.int64), shape))
    def check(job):
        flat, shape = job
        xy, lo, hi = compiled.support(flat, shape)
        ijk = np.column_stack(np.unravel_index(flat, shape))
        mask = np.zeros(shape[:2], bool); mask[tuple(ijk[:, :2].T)] = True
        assert np.array_equal(xy, np.argwhere(binary_dilation(mask)))
        assert lo == ijk[:, 2].min() and hi == ijk[:, 2].max()
    with ThreadPoolExecutor(max_workers=4) as pool: list(pool.map(check, jobs))
    xy, lo, hi = compiled.support(np.empty(0, np.int64), (2, 3, 4))
    assert xy.shape == (0, 2) and (lo, hi) == (4, -1)
    with pytest.raises(ValueError): compiled.support(np.array([-1], np.int64), (2, 3, 4))
    with pytest.raises(ValueError): compiled.support(np.array([24], np.int64), (2, 3, 4))


def test_gather_sorted_and_table_flags_invalid_coords(compiled):
    shape = (3, 4, 5); h = np.arange(60, dtype=np.uint8).reshape(shape) % 18
    obs = (h % 3).astype(np.uint8)  # preserve raw uint8 visibility bits, not bool conversion
    indices = np.column_stack(np.unravel_index(np.arange(60), shape)).astype(np.int64)
    indices = np.concatenate((indices, [[-1, 0, 0], [3, 0, 0], [0, 4, 0], [0, 0, 5]]))
    for owned in (np.empty(0, np.int64), np.array([0, 7, 59], np.int64), np.arange(60, dtype=np.int64)):
        bits = np.isin(np.arange(60), owned)
        for table in (None, (0, bits)):
            labels, flags = compiled.gather(indices, h, obs, owned, table)
            assert np.array_equal(labels[:60], h.ravel()) and (labels[60:] == 18).all()
            assert np.array_equal(flags[:60], obs.ravel() | bits.astype(np.uint8)*2)
            assert (flags[60:] == 0).all()
    with pytest.raises(TypeError): compiled.gather(indices.astype(np.float64), h, obs)
    with pytest.raises(ValueError): compiled.gather(indices, h, obs[:, :, :1])
    with pytest.raises(ValueError): compiled.expand(h.reshape(3, 20), h.reshape(3, 20), np.array([3], np.int64), np.array([4], np.uint8), True)


def test_random_all_classes_ownership_fallback_and_target_precedence(compiled):
    from types import SimpleNamespace
    rng = np.random.default_rng(72); shape = (9, 7, 5)
    for _ in range(20):
        b = rng.integers(0, 18, shape, dtype=np.uint8)
        own = rng.integers(-1, 4, shape, dtype=np.int32)
        restored = rng.integers(0, 18, shape, dtype=np.uint8)
        gt = rng.integers(0, 18, shape, dtype=np.uint8)
        xy = np.column_stack((rng.integers(0, 9, 40), rng.integers(0, 7, 40))).astype(np.int64)
        cls = rng.integers(0, 17, 40, dtype=np.uint8); allowed = rng.random((40, 5)) > .4
        for kind, actor in ((0, -3), (1, -2), (1, 0), (1, 3)):
            flat = (xy[:, :1]*7+xy[:, 1:])*5+np.arange(5)
            base = b.ravel()[flat]; fall = base.copy()
            legal = np.zeros((40, 5, 3), bool); legal[..., 0] = True
            legal[..., 1] = (base == 17)&allowed
            if kind:
                ours = own.ravel()[flat] == actor if actor >= 0 else base == cls[:, None]
                fallback = restored.ravel()[flat] if actor >= 0 else np.full_like(base, 17)
                fall[ours] = fallback[ours]
                legal[..., 2] = ours&(base == cls[:, None])&(fall != base)
            result = compiled.rows(xy, cls, allowed, b, own, restored, kind, actor)
            assert all(np.array_equal(a, c) for a, c in zip(result, (flat, base, fall, legal, legal[..., 1:].any((1, 2)))))
            # Explicit synthetic overlap checks ADD then REMOVE precedence.
            legal[..., 1] |= rng.random((40, 5)) > .8
            plan = SimpleNamespace(flat=flat, classes=cls, base=base, fallback=fall, legal=legal)
            expected = np.zeros((40, 5), np.int64); g = gt.ravel()[flat]
            expected[legal[..., 1]&(g == cls[:, None])] = 1
            expected[legal[..., 2]&(g == fall)&(g != base)] = 2
            assert np.array_equal(compiled.targets(plan, gt), expected)
    workspace = (np.empty(shape[:2], np.uint8), np.empty((63, 2), np.int64), np.empty(2, np.int64))
    workspace[0].flags.writeable = False
    with pytest.raises(ValueError, match='writable'): compiled.support(np.array([1], np.int64), shape, workspace)


def test_backend_fail_closed_and_artifact_report(monkeypatch, compiled):
    monkeypatch.setenv('SWFM_COLUMN_CPU_BACKEND', 'typo')
    with pytest.raises(ValueError): native.get_native()
    monkeypatch.setenv('SWFM_COLUMN_CPU_BACKEND', 'native')
    monkeypatch.setattr(native, '_loaded', None)
    with pytest.raises(RuntimeError, match='before prepare_native'): native.get_native()
    info = compiled.info()
    assert info['calls']['support'] > 0 and len(info['library_sha256']) == 64
    assert info['floating_point_geometry'] == 'unchanged_numpy_float64'
