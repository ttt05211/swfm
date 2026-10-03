"""Fused integer buckets retain every candidate and exactly the caller RNG."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

from real_motion.causal_column_completion import sample_queries
from test_native_column_cpu import compiled


def reference_buckets(kinds, actors, positive):
    populations = (np.flatnonzero(kinds == 0), np.flatnonzero((kinds == 1)&(actors < 0)),
                   np.flatnonzero((kinds == 1)&(actors >= 0)))
    return tuple(bucket for rows in populations for bucket in (rows[positive[rows]], rows[~positive[rows]]))


@pytest.mark.parametrize('n', [0, 1, 19, 100003])
@pytest.mark.parametrize('mode', ['mixed', 'free', 'positive'])
def test_native_strata_order_complete_coverage_empty_and_all_positive_exact(compiled, n, mode):
    rng = np.random.default_rng(n)
    kinds = rng.integers(0, 2, n, dtype=np.uint8)
    actors = rng.integers(-3, 20, n, dtype=np.int32)
    positive = rng.random(n) < .13 if mode == 'mixed' else np.full(n, mode == 'positive', bool)
    before = [a.copy() for a in (kinds, actors, positive)]
    buckets = compiled.sampling_strata(kinds, actors, positive)
    expected = reference_buckets(kinds, actors, positive)
    assert all(np.array_equal(a, b) for a, b in zip(buckets, expected))
    assert sum(map(len, buckets)) == n
    assert np.array_equal(np.sort(np.concatenate(buckets)), np.arange(n))
    assert all(np.array_equal(a, b) for a, b in zip(before, (kinds, actors, positive)))
    assert all(bucket.base is buckets[0].base for bucket in buckets)  # one packed buffer


@pytest.mark.parametrize('budget', [2, 3, 20, 24, 500])
def test_prepared_buckets_preserve_repeated_rng_choices_and_importance(compiled, monkeypatch, budget):
    monkeypatch.setenv('SWFM_COLUMN_CPU_BACKEND', 'numpy')
    rng = np.random.default_rng(83); n = 17531
    kinds = rng.integers(0, 2, n, dtype=np.uint8); actors = rng.integers(-2, 40, n, dtype=np.int32)
    positive = rng.random(n) < .027
    reference = SimpleNamespace(kind=kinds, actor=actors, positive_rows=positive)
    cached = SimpleNamespace(**vars(reference), sampling_buckets=compiled.sampling_strata(kinds, actors, positive))
    x, y = np.random.default_rng(20694), np.random.default_rng(20694)
    for _ in range(20):
        a = sample_queries(reference, None, budget, x, optimize=True)
        b = sample_queries(cached, None, budget, y, optimize=True)
        assert all(np.array_equal(v, w) for v, w in zip(a, b))
        assert x.bit_generator.state == y.bit_generator.state


def test_cached_caller_draw_does_not_scan_full_population(compiled, monkeypatch):
    kinds = np.array([0, 0, 1, 1, 1, 1], np.uint8)
    actors = np.array([-3, -3, -2, -2, 0, 0], np.int32)
    positive = np.array([1, 0, 1, 0, 1, 0], bool)
    plan = SimpleNamespace(kind=kinds, actor=actors, positive_rows=positive,
        sampling_buckets=compiled.sampling_strata(kinds, actors, positive))
    monkeypatch.setenv('SWFM_COLUMN_CPU_BACKEND', 'numpy')
    def forbidden(*args): raise AssertionError('caller rescanned population')
    monkeypatch.setattr(np, 'flatnonzero', forbidden)
    ids, weight = sample_queries(plan, None, 2, np.random.default_rng(6), optimize=True)
    assert ids.tolist() == [0, 1, 2, 3, 4, 5] and weight.tolist() == [1]*6


def test_native_strata_validation_strides_and_concurrent_private_buffers(compiled):
    kinds = np.array([0, 1]*100, np.uint8); actors = np.arange(-100, 100, dtype=np.int32)
    positive = actors % 3 == 0
    def check(_):
        out = compiled.sampling_strata(kinds[::2], actors[::2], positive[::2])
        assert all(np.array_equal(a, b) for a, b in zip(out, reference_buckets(kinds[::2], actors[::2], positive[::2])))
    with ThreadPoolExecutor(max_workers=6) as pool: list(pool.map(check, range(24)))
    with pytest.raises(TypeError): compiled.sampling_strata(kinds.astype(np.int64), actors, positive)
    with pytest.raises(ValueError): compiled.sampling_strata(kinds[:, None], actors, positive)
    with pytest.raises(ValueError): compiled.sampling_strata(kinds, actors[:-1], positive)
    with pytest.raises(ValueError): compiled.sampling_strata(np.array([2], np.uint8), actors[:1], positive[:1])
    with pytest.raises(ValueError): compiled.sampling_strata(kinds[:1], actors[:1], np.array([2], np.uint8))
