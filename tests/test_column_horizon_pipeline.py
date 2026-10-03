"""Horizon scheduling must preserve population, RNG, live gradients and resume."""
import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event, Lock, get_ident

import numpy as np
import pytest
import torch

from real_motion.column_cpu_pipeline import HorizonCpuPool, sampling_worker_budget
from real_motion.joint_causal_columns import JointCausalColumns
from tools.real_motion import joint_column_common as common
from tools.real_motion import joint_column_full_common as full
from test_joint_causal_columns import fixture, optimizers, provider_for
from test_native_column_cpu import compiled


def assert_state_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left: assert_state_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right): assert_state_equal(a, b)
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right)
    else:
        assert left == right


def test_shared_capacity_and_feature_priority_avoid_queued_candidate_backlog():
    assert sampling_worker_budget(8, horizons=True) == 6
    assert sampling_worker_budget(8) == 4
    assert sampling_worker_budget(128, 128, horizons=True) == 8
    gate = Event(); started = Event(); order = []
    with HorizonCpuPool(1) as pool:
        def blocking():
            started.set(); gate.wait(5)
        active = pool.submit(blocking)
        assert started.wait(2)
        backlog = [pool.submit(lambda i=i: order.append(('candidate', i))) for i in range(12)]
        feature = pool.submit_features(lambda: order.append(('feature', 0)))
        gate.set(); feature.result(timeout=2); active.result(timeout=2)
        assert order[0] == ('feature', 0)
        for job in backlog: job.result(timeout=2)
    with HorizonCpuPool(6) as pool:
        assert pool.candidate_workers == pool.feature_workers == pool.workers == 6
        assert pool.shared_worker_pool
        gate.clear(); starts = [Event() for _ in range(6)]
        def occupied(event):
            event.set(); return gate.wait(5)
        blocked = [pool.submit(occupied, event) for event in starts]
        try:
            assert all(event.wait(2) for event in starts)  # all SIX, not three, can do candidates
        finally:
            gate.set()
        assert all(job.result(timeout=2) for job in blocked)
    for workers in (0, 9):
        with pytest.raises(ValueError): HorizonCpuPool(workers)


@pytest.mark.parametrize('workers', [1, 2, 6, 8])
def test_fifo_parent_index_precedes_dependent_jobs_even_with_one_worker(workers):
    with HorizonCpuPool(workers) as pool:
        parents = []
        jobs = []
        for window in range(4):
            index = pool.submit_features(lambda value: (value, get_ident()), window)
            parents.append(index)
            jobs.extend(pool.submit_features(lambda p, h: (p.result()[0], h), index, h) for h in range(6))
        assert [job.result(timeout=5) for job in jobs] == [(w, h) for w in range(4) for h in range(6)]


def test_shutdown_cancels_pending_only_and_running_task_finishes():
    gate = Event(); started = Event(); ran = []
    pool = HorizonCpuPool(1)
    def active():
        started.set(); gate.wait(5); return 'committed CPU result'
    job = pool.submit(active)
    try:
        assert started.wait(2)
        a = pool.submit(lambda: ran.append('candidate'))
        b = pool.submit_features(lambda: ran.append('feature'))
        pool.shutdown(wait=False, cancel_futures=True)
        assert a.cancelled() and b.cancelled() and not job.cancelled()
        with pytest.raises(RuntimeError, match='shutdown'): pool.submit(lambda: None)
    finally:
        gate.set(); pool.shutdown(wait=True)
    assert job.result(timeout=2) == 'committed CPU result' and not ran


def test_worker_exception_does_not_kill_shared_worker_or_drop_remaining_jobs():
    with HorizonCpuPool(1) as pool:
        def bad(): raise ValueError('CPU job error')
        job = pool.submit_features(bad)
        good = pool.submit(lambda: 42)
        with pytest.raises(ValueError, match='CPU job error'): job.result(timeout=2)
        assert good.result(timeout=2) == 42


def test_draw_only_does_not_materialize_and_keeps_original_rng(monkeypatch):
    prep, grid, joint, _, _ = fixture()
    plans = common.build_online_column_candidates(prep, joint.columns.config, grid, defer_context=True)
    a, b = np.random.default_rng(8), np.random.default_rng(8)
    old = common.select_online_columns(prep, joint.columns.config, grid, a, candidates=plans)
    original = type(plans[0][1]).subset
    def forbidden(*args): raise AssertionError('draw touched voxel fields')
    monkeypatch.setattr(type(plans[0][1]), 'subset', forbidden)
    draws = common.draw_online_column_indices(prep, joint.columns.config, grid, b, candidates=plans)
    assert a.bit_generator.state == b.bit_generator.state
    monkeypatch.setattr(type(plans[0][1]), 'subset', original)
    with ThreadPoolExecutor(max_workers=6) as pool:
        new = list(pool.map(common.materialize_online_column, draws))
    for x, y in zip(old, new):
        assert x[0] == y[0]
        assert all(np.array_equal(v, getattr(y[1], k)) for k, v in vars(x[1]).items())
        assert np.array_equal(x[2], y[2]) and np.array_equal(x[3], y[3])


@pytest.mark.parametrize('history', [4, 6])
@pytest.mark.parametrize('workers', [1, 6, 8])
def test_horizon_and_window_actual_adamw_steps_rng_and_gradient_link_exact(monkeypatch, history, workers):
    # Run the actual NumPy candidate/feature path to test scheduling everywhere,
    # including Windows without a compiler. Native two-pass is tested below.
    prep, grid, original, control, rec = fixture()
    mc = replace(original.transport.v17_config, history_frames=history)
    joint = JointCausalColumns(mc, original.columns.config)
    torch.nn.init.normal_(joint.columns.refinement.weight, std=.01)
    serial = copy.deepcopy(joint)
    prep.raw = {**prep.raw, **{k: prep.raw[k][-history:] for k in ('history_occ', 'history_observed', 'history_poses')}}
    prep.registrations = [r[-history:] for r in prep.registrations]
    opt, _ = optimizers(joint, control); refopt, _ = optimizers(serial, control)
    p, q = provider_for(prep, grid, joint), provider_for(prep, grid, serial)
    rows = [(rec, None)] * 4
    rng, ref_rng = np.random.default_rng(74), np.random.default_rng(74)
    caller = get_ident(); seen = []; lock = Lock()
    old_draw = common.sample_queries
    def draw(*args, **kwargs):
        assert get_ident() == caller
        return old_draw(*args, **kwargs)
    monkeypatch.setattr(common, 'sample_queries', draw)
    old_materialize = full.materialize_online_column
    def materialize(*args):
        assert get_ident() != caller
        with lock: seen.append(get_ident())
        return old_materialize(*args)
    monkeypatch.setattr(full, 'materialize_online_column', materialize)
    old_gather = joint.columns.source_features_for
    def gather(*args, **kwargs):
        assert get_ident() == caller
        return old_gather(*args, **kwargs)
    joint.columns.source_features_for = gather
    monkeypatch.setattr(full, 'bundle_enabled', lambda: True)
    with HorizonCpuPool(workers) as pool:
        for update in (20695, 20696, 20697):
            monkeypatch.setattr(full, 'horizon_pipeline_enabled', lambda: False)
            ref = full.train_full_batch(serial, refopt, q, None, rows, ref_rng, update, 77172, probe=True)
            monkeypatch.setattr(full, 'horizon_pipeline_enabled', lambda: True)
            stats = full.train_full_batch(joint, opt, p, None, rows, rng, update, 77172, probe=True,
                                          sampling_pool=pool, sampling_workers=workers)
            for key in ('loss', 'motion_loss', 'column_loss', 'grad_norm', 'column_grad_norm',
                        'source_query_gradient_norm', 'sampled_columns', 'full_candidate_columns'):
                assert stats[key] == ref[key]
            assert stats['source_query_gradient_norm'] > 0
            assert stats['online_candidate_horizon_jobs'] == 24
            assert stats['online_history_index_jobs'] == 4
            assert stats['online_feature_horizon_jobs'] <= 24
            assert stats['cpu_task_granularity'] == 'horizon'
            assert_state_equal(opt.state_dict(), refopt.state_dict())
            assert_state_equal(joint.state_dict(), serial.state_dict())
            assert rng.bit_generator.state == ref_rng.bit_generator.state
    assert seen


@pytest.mark.parametrize('workers', [1, 6, 8])
def test_native_horizon_pool_actual_adamw_steps_are_exact(compiled, monkeypatch, workers):
    monkeypatch.setenv('SWFM_COLUMN_CPU_BACKEND', 'native')
    monkeypatch.setenv('SWFM_COLUMN_CPU_BUNDLE', '1')
    test_horizon_and_window_actual_adamw_steps_rng_and_gradient_link_exact(monkeypatch, 4, workers)
    assert compiled.info()['calls']['compact_materialize'] > 0


def test_worker_error_propagates_before_any_optimizer_update(monkeypatch):
    prep, grid, joint, control, rec = fixture()
    opt, _ = optimizers(joint, control); provider = provider_for(prep, grid, joint)
    before = copy.deepcopy(joint.state_dict())
    monkeypatch.setattr(full, 'bundle_enabled', lambda: True)
    monkeypatch.setattr(full, 'horizon_pipeline_enabled', lambda: True)
    old = common.OnlineColumnCandidateBuilder.build
    def broken(self, h):
        if h == 5: raise RuntimeError('horizon candidate failure')
        return old(self, h)
    monkeypatch.setattr(common.OnlineColumnCandidateBuilder, 'build', broken)
    with HorizonCpuPool(6) as pool, pytest.raises(RuntimeError, match='candidate failure'):
        full.train_full_batch(joint, opt, provider, None, [(rec, None)] * 4,
                              np.random.default_rng(5), 1, 20, sampling_pool=pool)
    assert_state_equal(before, joint.state_dict())
    assert not opt.state_dict()['state']


def test_empty_column_windows_keep_motion_training_and_do_not_wait_on_missing_index(monkeypatch):
    prep, grid, joint, control, rec = fixture()
    opt, _ = optimizers(joint, control); provider = provider_for(prep, grid, joint)
    monkeypatch.setattr(full, 'bundle_enabled', lambda: True)
    monkeypatch.setattr(full, 'horizon_pipeline_enabled', lambda: True)
    monkeypatch.setattr(full, 'draw_online_column_indices', lambda *args, **kwargs: [])
    with HorizonCpuPool(1) as pool:
        stats = full.train_full_batch(joint, opt, provider, None, [(rec, None)] * 2,
                                      np.random.default_rng(5), 1, 20, sampling_pool=pool, sampling_workers=1)
    assert stats['sampled_columns'] == stats['online_history_index_jobs'] == 0
    assert stats['optimizer_updated'] and stats['motion_loss'] > 0


def test_native_compact_horizon_order_fields_draws_and_features_exact(monkeypatch, compiled):
    from test_column_cpu_kernels import candidate_fixture
    prep, grid, cfg = candidate_fixture(8)
    monkeypatch.setenv('SWFM_COLUMN_CPU_BACKEND', 'native')
    monkeypatch.setenv('SWFM_COLUMN_CPU_BUNDLE', '1')
    prep.cpu_bundle_optimized = prep.cpu_kernels_optimized = True
    original = common.build_online_column_candidates(prep, cfg, grid, defer_context=True)
    builder = common.OnlineColumnCandidateBuilder(prep, cfg, grid, defer_context=True)
    with HorizonCpuPool(6) as pool:
        # Submit reverse order; return/order and RNG must still be canonical.
        jobs = {h: pool.submit(builder.build, h) for h in reversed(range(6))}
        new = [jobs[h].result(timeout=20) for h in range(6)]
        a, b = np.random.default_rng(93), np.random.default_rng(93)
        old_rows = common.select_online_columns(prep, cfg, grid, a, candidates=original)
        draws = common.draw_online_column_indices(prep, cfg, grid, b, candidates=new)
        assert a.bit_generator.state == b.bit_generator.state
        index = common.online_column_history_index(prep, draws, grid)
        rows = [job.result() for job in [pool.submit_features(common.materialize_online_column, draw) for draw in draws]]
        for old, row in zip(old_rows, rows):
            assert all(np.array_equal(v, getattr(row[1], k)) for k, v in vars(old[1]).items())
            assert np.array_equal(old[2], row[2]) and np.array_equal(old[3], row[3])
            reference = common.sample_online_column(prep, old, grid, cfg)
            actual = common.sample_online_column(prep, row, grid, cfg, history_index=index)
            assert all(np.array_equal(v, actual[k]) for k, v in reference.items())
    assert compiled.info()['calls']['compact_materialize'] > 0
