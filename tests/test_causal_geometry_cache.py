"""Bounded causal-only geometry reuse, never stale learned geometry or GT."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch
from threading import Event
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


def test_deferred_cold_geometry_never_persists_until_completed(tmp_path):
    raw = raw_fixture(); cache = CausalGeometryCache(tmp_path, 'p', reserve_bytes=0)
    light, hit = cache.get_or_build(('s', 't'), raw, lambda: {'history': np.arange(5)}, defer_write=True)
    assert not hit and not cache.rows and not list(cache.root.glob('*.cgc'))
    full = {**light, 'prepared_state': {'anchors': np.arange(12)}}
    assert cache.store(('s', 't'), raw, full, asynchronous=True)
    cache.close()
    fresh = CausalGeometryCache(tmp_path, 'p', reserve_bytes=0)
    actual, hit = fresh.get_or_build(('s', 't'), raw, lambda: pytest.fail('completed geometry must hit'))
    assert hit and np.array_equal(actual['prepared_state']['anchors'], full['prepared_state']['anchors'])


def test_background_compression_does_not_hold_reader_lock_and_queue_is_bounded(tmp_path):
    import real_motion.causal_geometry_cache as module
    raw = raw_fixture(); cache = CausalGeometryCache(tmp_path, 'p', reserve_bytes=0)
    entered, release = Event(), Event(); original = module.zlib.compress
    def slow(*args, **kwargs):
        entered.set(); assert release.wait(5)
        return original(*args, **kwargs)
    try:
        with patch.object(module.zlib, 'compress', side_effect=slow):
            cache.store(('s', '0'), raw, {'v': np.arange(5)}, asynchronous=True)
            assert entered.wait(2)
            with ThreadPoolExecutor(max_workers=1) as pool:
                result = pool.submit(cache.get_or_build, ('s', '0'), raw, lambda: pytest.fail('RAM hit required')).result(timeout=1)
            assert result[1]
            for i in range(1, 11): cache.store(('s', str(i)), raw, {'v': np.arange(5)}, asynchronous=True)
            assert cache.stats()['pending_writes'] == 8 and cache.stats()['skipped_writes'] == 3
            release.set(); cache.close()
    finally:
        release.set(); cache.close()
    assert cache.stats()['pending_writes'] == 0 and cache.stats()['writes'] == 8
    assert not list(cache.root.glob('*.tmp.*'))


def test_background_writer_failures_are_reported_on_drain(tmp_path):
    cache = CausalGeometryCache(tmp_path, 'p', reserve_bytes=0)
    with patch.object(cache, '_store', side_effect=ValueError('bad geometry')):
        cache.store(('s', 't'), raw_fixture(), {'v': np.arange(5)}, asynchronous=True)
        with pytest.raises(RuntimeError, match='writer failed'): cache.close()


def test_ram_budget_counts_whole_backing_allocation(tmp_path):
    cache = CausalGeometryCache(tmp_path, 'p', max_bytes=0, ram_bytes=24, reserve_bytes=0)
    backing = np.zeros(100, np.uint8)
    cache.get_or_build(('s', 't'), raw_fixture(), lambda: {'current': backing[:2], 'previous': backing[2:4]})
    assert cache.stats()['ram_mib'] == 0
