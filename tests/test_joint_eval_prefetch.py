"""CPU window concurrency must not alter inference order, bytes or metrics."""
import copy
import json
from pathlib import Path
from threading import Barrier, Event, Lock, get_ident
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from real_motion.column_runtime_pipeline import prefetch_raw_columns
from real_motion.native_column_cpu import backend_name, prepare_native
from tools.real_motion import causal_column_common as col
from tools.real_motion import joint_evaluation_reporting as reporting
from tools.real_motion.joint_column_full_common import EvaluationJointColumnProvider
from tools.real_motion.static_evidence_selector_common import finite_json
from test_joint_checkpoint_selection import _eval_fixture
from test_causal_columns import moving_fixture

if backend_name() == 'native': prepare_native()


@pytest.mark.parametrize('workers', (2, 4))
def test_cpu_windows_are_concurrent_ordered_and_bounded(workers):
    barrier = Barrier(2); caller = get_ident(); ids = []; lock = Lock()
    def load(source, record, *, include_gt):
        assert not include_gt and get_ident() != caller
        with lock: ids.append(get_ident())
        barrier.wait(5)
        return record*10
    provider = SimpleNamespace(load_raw_columns=load, raw_prefetch_workers=workers, raw_prefetch_depth=4)
    assert list(prefetch_raw_columns(provider, None, range(6), include_gt=False)) == [(i, i*10) for i in range(6)]
    assert 2 <= len(set(ids)) <= workers


def test_no_unbounded_prefill_and_close_releases_workers():
    release = Event(); calls = []; lock = Lock()
    def load(source, record, *, include_gt):
        with lock: calls.append(record)
        if record: assert release.wait(5)
        return record
    provider = SimpleNamespace(load_raw_columns=load, raw_prefetch_workers=4, raw_prefetch_depth=4)
    iterator = prefetch_raw_columns(provider, None, range(100))
    try:
        assert next(iterator) == (0, 0)
        assert max(calls) <= 4 and len(calls) <= 5  # CURRENT + at most four NEXT
    finally:
        release.set(); iterator.close()


@pytest.mark.parametrize('workers,depth', ((0, 1), (4, 3), (1, 5), (True, 1)))
def test_invalid_prefill_budgets_rejected(workers, depth):
    provider = SimpleNamespace(load_raw_columns=lambda *a, **k: None,
        raw_prefetch_workers=workers, raw_prefetch_depth=depth)
    with pytest.raises(ValueError, match='prefetch'): list(prefetch_raw_columns(provider, None, [1]))


def test_worker_failure_is_delivered_in_record_order():
    def load(source, record, *, include_gt):
        if record == 1: raise RuntimeError('missing window 1')
        return record
    provider = SimpleNamespace(load_raw_columns=load, raw_prefetch_workers=4, raw_prefetch_depth=4)
    iterator = prefetch_raw_columns(provider, None, range(5))
    assert next(iterator) == (0, 0)
    with pytest.raises(RuntimeError, match='window 1'): next(iterator)


@pytest.mark.parametrize('workers', (2, 4))
def test_parallel_prefill_exact_reports_and_all_network_work_on_caller(workers):
    prep, grid, joint, records, source, owner = _eval_fixture(); caller = get_ident()
    with torch.no_grad(): joint.columns.generation.bias.fill_(1.5)
    actual = joint.columns.forward
    def forward(**kwargs):
        assert get_ident() == caller
        return actual(**kwargs)
    with patch.object(col, 'gt_moving_support_sequence', return_value=moving_fixture(prep)), \
            patch.object(joint.columns, 'forward', side_effect=forward):
        expected = col.evaluate_columns(owner(joint), source, records, joint.columns, (.5, .5, None),
            dev64_keys=[('dev', 't1')], diagnostic_thresholds=(.75, .95, .5))
        provider = owner(joint); provider.raw_prefetch_workers = workers; provider.raw_prefetch_depth = 4
        joint.columns.column_inference_optimized = True; joint.columns.column_inference_verify_remaining = 3
        result = col.evaluate_columns(provider, source, records, joint.columns, (.5, .5, None),
            dev64_keys=[('dev', 't1')], diagnostic_thresholds=(.75, .95, .5))
    assert finite_json(result) == finite_json(expected)


def test_nested_geometry_worker_budget_and_worker_timings():
    provider = EvaluationJointColumnProvider.__new__(EvaluationJointColumnProvider)
    provider.workers, provider.raw_prefetch_workers = 8, 4
    provider.pcfg = provider.strong = None
    provider.joint = SimpleNamespace(columns=SimpleNamespace(config=None))
    def build(raw, record, pcfg, strong, workers, config, *, profile):
        assert workers == 2
        profile['strong_state'] = .1
        return {'fixed': True}
    with patch.object(col.FrozenColumns, 'load_raw_columns', return_value={}), \
            patch('tools.real_motion.joint_column_full_common.build_fixed_geometry', side_effect=build):
        raw = provider.load_raw_columns(None, {}, include_gt=True)
    assert raw['_column_causal_preparation'] == {'fixed': True}
    assert raw['_evaluation_raw_prepare_seconds']['geometry_workers'] == 2
    assert raw['_evaluation_raw_prepare_seconds']['total'] >= 0


def result_fixture():
    metrics = dict(IoU=52.5, mIoU=44., MovingMacro=20., MovingMicro=30.)
    metrics['per_horizon'] = {str(h): {k: v for k, v in metrics.items() if k != 'per_horizon'} for h in (1., 2., 3.)}
    variant = dict(metrics=metrics, quality={'added': 4}, scene_delta={'positive': 1, 'negative': 0, 'zero': 0, 'by_scene': {'dev': 1}})
    all_rows = dict(windows=2, scenes=1, baseline=metrics, reference_metrics={'frozen_E14': metrics},
        variants={k: variant for k in ('generation', 'refine', 'joint')})
    return dict(status='complete', reports={'all': all_rows}, population='dev512', attempted_updates=99,
        cursor_epoch=19, cursor_batch=0, history_frames=4, thresholds=[.5, .5, None], threshold_source='fixed',
        seconds=2., snapshot='unchanged.pt')


def test_readonly_summary_has_all_metrics_and_aggregates_existing_timings(tmp_path):
    path = tmp_path/'evaluation.json'; path.write_text(json.dumps(result_fixture()))
    log = tmp_path/'progress.jsonl'
    rows = [dict(event='evaluation', seconds=1.+i, compute_seconds=.5, input_wait_seconds=.5+i,
        prepare_seconds={'raw_and_v18': .01}, reference_seconds=.02,
        prediction_seconds_by_horizon={'1.0': {'queries': 8, 'patch_gather_seconds': .03}}) for i in range(2)]
    log.write_text('\n'.join(json.dumps(r) for r in rows))
    originals = {p: p.read_bytes() for p in (path, log)}
    text = reporting.read_report(tmp_path)
    assert 'joint: IoU=52.500000 mIoU=44.000000 MovingMacro=20.000000 MovingMicro=30.000000' in text
    assert '1.0s joint: IoU=52.500000' in text and 'dIoU_vs_E14=+0.000000' in text
    assert 'input_wait_fraction=66.67%' in text and 'logged_windows=2' in text
    assert 'CURRENT transport' in text and 'by_scene' not in text
    assert all(p.read_bytes() == data for p, data in originals.items())


def test_unsupported_metric_prints_na_and_incomplete_progress_not_silently_accepted(tmp_path):
    result = copy.deepcopy(result_fixture()); result['reports']['all']['reference_metrics']['frozen_E14']['MovingMacro'] = None
    assert 'MovingMacro=N/A' in reporting.summary_text(result)
    path = tmp_path/'progress.jsonl'; path.write_text('{incomplete')
    with pytest.raises(ValueError, match='corrupt progress'): reporting.summarize_progress(path)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='actual CUDA required')
def test_actual_cuda_parallel_prefill_preserves_probabilities_and_report():
    prep, grid, joint, records, source, owner = _eval_fixture(); joint.cuda()
    provider = owner(joint); provider.device = torch.device('cuda')
    def prepare(source, record, *, include_gt, raw_window=None):
        row = copy.deepcopy(prep); row.outputs = joint.motion(record, provider.device); return row
    provider.prepare_columns = prepare
    with patch.object(col, 'gt_moving_support_sequence', return_value=moving_fixture(prep)):
        expected = col.evaluate_columns(provider, source, records, joint.columns, (.5, .5, None), diagnostic_thresholds=None)
        provider.raw_prefetch_workers = provider.raw_prefetch_depth = 4
        joint.columns.column_inference_optimized = True; joint.columns.column_inference_verify_remaining = 3
        actual = col.evaluate_columns(provider, source, records, joint.columns, (.5, .5, None), diagnostic_thresholds=None)
    assert finite_json(actual) == finite_json(expected)
