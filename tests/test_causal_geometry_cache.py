"""Bounded causal-only geometry reuse, never stale learned geometry or GT."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch
import copy
import numpy as np
import pytest
from real_motion.causal_geometry_cache import CausalGeometryCache, causal_input_digest


def raw_fixture():
    return dict(history_occ=np.full((6, 8, 8, 2), 17, np.uint8),
        history_observed=np.ones((6, 8, 8, 2), bool),
        history_poses=np.stack([np.eye(4)]*6), future_poses=np.stack([np.eye(4)]*6),
        future_gt_occ=np.zeros((6, 8, 8, 2), np.uint8))


def test_memory_and_fresh_disk_hits_preserve_arrays_and_objects(tmp_path):
    raw = raw_fixture(); calls = []
    def build():
        calls.append(1)
        return dict(memory=raw['history_occ'].copy(), object=SimpleNamespace(points=np.eye(4)))
    cache = CausalGeometryCache(tmp_path, 'provenance', ram_bytes=4096, reserve_bytes=0)
    a, hit = cache.get_or_build(('s', 't'), raw, build); assert not hit
    b, hit = cache.get_or_build(('s', 't'), raw, build); assert hit and a is b
    fresh = CausalGeometryCache(tmp_path, 'provenance', ram_bytes=0, reserve_bytes=0)
    c, hit = fresh.get_or_build(('s', 't'), raw, build)
    assert hit and len(calls) == 1 and np.array_equal(c['memory'], a['memory'])
    assert np.array_equal(c['object'].points, a['object'].points)
    assert cache.stats()['ram_mib']*2**20 <= 4096


def test_future_gt_not_hashed_but_all_causal_inputs_and_identity_are(tmp_path):
    raw = raw_fixture(); cache = CausalGeometryCache(tmp_path, 'p', reserve_bytes=0)
    build = lambda: {'v': np.arange(10)}
    cache.get_or_build(('s', 't'), raw, build)
    changed = {**raw, 'future_gt_occ': 'must never read'}
    assert causal_input_digest(changed) == causal_input_digest(raw)
    assert cache.get_or_build(('s', 't'), changed, build)[1]
    for key in ('history_occ', 'history_observed', 'history_poses', 'future_poses'):
        changed = copy.deepcopy(raw); changed[key].flat[0] = 0
        assert not cache.get_or_build(('s', 't'), changed, build)[1]
    assert not cache.get_or_build(('different_scene', 't'), raw, build)[1]
    other = CausalGeometryCache(tmp_path, 'different_frozen_config', reserve_bytes=0)
    assert not other.get_or_build(('s', 't'), raw, build)[1]


def test_corrupt_cache_fails_closed_not_silent_recompute(tmp_path):
    raw = raw_fixture(); cache = CausalGeometryCache(tmp_path, 'p', ram_bytes=0, reserve_bytes=0)
    cache.get_or_build(('s', 't'), raw, lambda: {'v': np.zeros(2)})
    path = next(cache.root.glob('*.cgc')); blob = path.read_bytes()
    path.write_bytes(blob[:-1]+bytes([blob[-1]^1]))
    with pytest.raises(RuntimeError, match='corrupt'):
        cache.get_or_build(('s', 't'), raw, lambda: pytest.fail('must fail closed'))


def test_wrong_namespace_artifact_with_valid_checksum_fails_closed(tmp_path):
    raw = raw_fixture(); a = CausalGeometryCache(tmp_path, 'a', ram_bytes=0, reserve_bytes=0)
    a.get_or_build(('s', 't'), raw, lambda: {'v': np.zeros(2)})
    b = CausalGeometryCache(tmp_path, 'b', ram_bytes=0, reserve_bytes=0)
    path = next(a.root.glob('*.cgc'))
    (b.root/path.name).write_bytes(path.read_bytes())
    with pytest.raises(RuntimeError, match='provenance mismatch'):
        b.get_or_build(('s', 't'), raw, lambda: pytest.fail('must fail closed'))


def test_budget_shared_across_namespaces_never_evicts_and_ram_bounded(tmp_path):
    raw = raw_fixture(); a = CausalGeometryCache(tmp_path, 'a', ram_bytes=64, reserve_bytes=0)
    a.get_or_build(('s', 't'), raw, lambda: {'v': np.arange(64)})
    original = {p: p.read_bytes() for p in tmp_path.glob('*/*.cgc')}
    used = a.disk_used
    b = CausalGeometryCache(tmp_path, 'b', max_bytes=used+1, ram_bytes=64, reserve_bytes=0)
    value, hit = b.get_or_build(('s', 't'), raw, lambda: {'v': np.arange(64)})
    assert not hit and value['v'].shape == (64,)
    assert b.stats()['skipped_writes'] == 1 and b.disk_used == used
    assert b.stats()['ram_mib'] == 0 and {p: p.read_bytes() for p in original} == original
    tiny = CausalGeometryCache(tmp_path/'tiny', 'p', max_bytes=1, ram_bytes=16, reserve_bytes=0)
    for i in range(5): tiny.get_or_build(('s', i), raw, lambda: {'v': np.zeros(2)})
    assert tiny.disk_used == 0 and tiny.stats()['ram_mib']*2**20 <= 16


def test_low_free_space_skips_write_without_changing_result(tmp_path):
    raw = raw_fixture(); cache = CausalGeometryCache(tmp_path, 'p', ram_bytes=0)
    with patch('real_motion.causal_geometry_cache.shutil.disk_usage', return_value=SimpleNamespace(free=1)):
        result, hit = cache.get_or_build(('s', 't'), raw, lambda: {'v': np.array([1, 2])})
    assert not hit and result['v'].tolist() == [1, 2]
    assert cache.stats()['skipped_writes'] == 1 and not list(cache.root.glob('*.cgc'))


def test_concurrent_workers_atomic_files_and_quota(tmp_path):
    raw = raw_fixture(); cache = CausalGeometryCache(tmp_path, 'p', max_bytes=1800, ram_bytes=0, reserve_bytes=0)
    def load(i): return cache.get_or_build(('s', str(i)), raw, lambda: {'v': np.arange(30)+i})[0]
    with ThreadPoolExecutor(max_workers=4) as pool: rows = list(pool.map(load, range(12)))
    assert all(np.array_equal(v['v'], np.arange(30)+i) for i, v in enumerate(rows))
    assert cache.disk_used <= 1800 and not list(cache.root.glob('*.tmp.*'))
    fresh = CausalGeometryCache(tmp_path, 'p', max_bytes=1800, ram_bytes=0, reserve_bytes=0)
    for path in fresh.root.glob('*.cgc'): assert path.stat().st_size > 40
    for i in range(12): assert np.array_equal(fresh.get_or_build(('s', str(i)), raw, lambda: {'v': np.arange(30)+i})[0]['v'], rows[i]['v'])
