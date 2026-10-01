"""Speed changes must preserve causal reads, ordering and bounded RAM."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event, get_ident
from types import SimpleNamespace
import numpy as np
import pytest
from real_motion.column_runtime_pipeline import CachedColumnSource, prefetch_raw_columns


class Source:
    def __init__(self): self.calls = []; self.nusc = object()
    def load_occ3d(self, scene, token, require_lidar_mask=True):
        self.calls.append(('occ', token, require_lidar_mask))
        return np.full((4, 4, 2), int(token), np.uint8), np.ones((4, 4, 2), bool)
    def load_semantics(self, scene, token):
        self.calls.append(('sem', token)); return np.full((4, 4, 2), int(token), np.uint8)


def test_cache_readonly_budget_eviction_and_mask_contract():
    source = Source(); cache = CachedColumnSource(source, 64/2**20)
    sem, observed = cache.load_occ3d('s', '1')
    assert cache.nusc is source.nusc and cache.bytes == 64
    assert cache.load_semantics('s', '1') is sem and cache.hits == 1
    assert cache.load_occ3d('s', '1')[0] is sem
    with pytest.raises(ValueError): sem[0, 0, 0] = 17
    with pytest.raises(ValueError): observed[0, 0, 0] = False
    cache.load_occ3d('s', '1', require_lidar_mask=False)
    assert source.calls[-1] == ('occ', '1', False)
    cache.load_occ3d('s', '2'); assert cache.bytes <= cache.limit
    cache.load_occ3d('s', '1'); assert source.calls.count(('occ', '1', True)) == 2
    cache.load_semantics('s', '3'); assert source.calls[-1] == ('sem', '3')
    assert cache.bytes <= cache.limit


@pytest.mark.parametrize('budget', [0, 1/2**20])
def test_oversized_frame_not_cached(budget):
    source = Source(); cache = CachedColumnSource(source, budget)
    cache.load_occ3d('s', '1'); cache.load_occ3d('s', '1')
    assert cache.bytes == 0 and len(source.calls) == 2


def test_single_flight_and_distinct_io_not_serialized():
    entered, release = Event(), Event(); source = Source()
    original = source.load_occ3d
    def blocked(scene, token, require_lidar_mask=True):
        if token == '1': entered.set(); assert release.wait(5)
        return original(scene, token, require_lidar_mask)
    source.load_occ3d = blocked; cache = CachedColumnSource(source)
    with ThreadPoolExecutor(3) as pool:
        a = pool.submit(cache.load_occ3d, 's', '1'); assert entered.wait(5)
        b = pool.submit(cache.load_occ3d, 's', '1')
        c = pool.submit(cache.load_occ3d, 's', '2')
        try: assert c.result(timeout=5)[0][0, 0, 0] == 2
        finally: release.set()
        assert a.result()[0] is b.result()[0]
    assert source.calls.count(('occ', '1', True)) == 1 and not cache.pending


def test_cache_error_allows_retry():
    source = Source(); original = source.load_semantics
    source.load_semantics = lambda *_: (_ for _ in ()).throw(RuntimeError('bad frame'))
    cache = CachedColumnSource(source)
    with pytest.raises(RuntimeError, match='bad frame'): cache.load_semantics('s', '1')
    assert not cache.pending
    source.load_semantics = original
    assert cache.load_semantics('s', '1')[0, 0, 0] == 1


def test_prefetch_bounded_order_flags_worker_and_fallback():
    calls = []; caller = get_ident(); next_ready = Event()
    def loader(source, record, *, include_gt):
        calls.append((record, include_gt, get_ident()))
        if record == 1: next_ready.set()
        return record*10
    provider = SimpleNamespace(load_raw_columns=loader)
    it = prefetch_raw_columns(provider, None, range(5), include_gt=False)
    assert next(it) == (0, 0); assert next_ready.wait(5)
    assert len(calls) == 2  # no third window until caller advances
    assert list(it) == [(i, i*10) for i in range(1, 5)]
    assert [r for r, _, _ in calls] == list(range(5))
    assert all(not gt and tid != caller for _, gt, tid in calls)
    assert list(prefetch_raw_columns(SimpleNamespace(), None, range(3))) == [(i, None) for i in range(3)]
    assert list(prefetch_raw_columns(provider, None, [])) == []


def test_prefetch_error_propagates():
    def loader(source, record, *, include_gt):
        if record == 1: raise RuntimeError('missing window')
        return record
    it = prefetch_raw_columns(SimpleNamespace(load_raw_columns=loader), None, range(3))
    assert next(it) == (0, 0)
    with pytest.raises(RuntimeError, match='missing window'): next(it)


def test_prefetched_four_way_report_equals_legacy_bit_exact():
    import copy
    import torch
    from unittest.mock import patch
    from test_causal_columns import fake_provider, moving_fixture
    from real_motion.causal_column_model import CausalColumnModel
    from tools.real_motion import causal_column_common as common
    provider, prep, cfg, calls = fake_provider(); model = CausalColumnModel(cfg)
    source = SimpleNamespace(nusc=None); records = [dict(scene_name='dev', t0_token='t0')]
    with patch.object(common, 'gt_moving_support_sequence', return_value=moving_fixture(prep)):
        before = common.evaluate_columns(provider, source, records, model, (.5, .5, None))
        old_prepare = provider.prepare_columns
        read_threads = []
        def load(source, record, *, include_gt):
            read_threads.append(get_ident()); assert include_gt
            return copy.deepcopy(prep.raw)
        def prepare(source, record, *, include_gt, raw_window):
            row = old_prepare(source, record, include_gt=include_gt)
            assert np.array_equal(raw_window['history_occ'], row.raw['history_occ'])
            row.raw = raw_window
            return row
        provider.load_raw_columns = load; provider.prepare_columns = prepare
        after = common.evaluate_columns(provider, source, records, model, (.5, .5, None))
    from tools.real_motion.static_evidence_selector_common import finite_json
    # Missing toy classes have NaN IoU on both passes; compare persisted form.
    assert finite_json(before) == finite_json(after) and read_threads[0] != get_ident()
