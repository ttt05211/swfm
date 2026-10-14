"""Execution-only noise parity, ordered bounded prefetch and real old resume."""
import copy
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from tools.stc_noise_execution import pipeline, run as fast, workflow as fast_workflow
from tools.stc_noise_adaptation import augmentation, data, workflow, run
from tools.joint_occ_experiments import run as legacy
from test_stc_noise_adaptation import empirical_value, profile_files
from test_perception_robustness import history_fixture, real_fixture


def test_dead_field_skip_same_noise_counts_and_random_draws_concurrently():
    raw, rec, grid = history_fixture(); noise = augmentation.NoiseModel(empirical_value())
    raw['history_poses'][0, 0, 3] = 1.
    refs = [augmentation.corrupt(raw, dict(rec, t0_token=str(i)), grid, noise, 21, force=True) for i in range(12)]
    before = augmentation.uniform_field, data.corrupt
    from concurrent.futures import ThreadPoolExecutor
    with pipeline.noise_execution(), ThreadPoolExecutor(max_workers=4) as pool:
        pairs = list(pool.map(lambda i: data.corrupt(raw, dict(rec, t0_token=str(i)), grid, noise, 21, force=True), range(12)))
    assert before == (augmentation.uniform_field, data.corrupt)
    for (ref, audit), (new, report) in zip(refs, pairs):
        for k in ref: np.testing.assert_array_equal(ref[k], new[k])
        assert audit == report
    # Unselected windows retain same identity and RNG choice too.
    refs = [augmentation.corrupt(raw, dict(rec, t0_token=str(i)), grid, noise, 21) for i in range(12)]
    with pipeline.noise_execution():
        for i, (ref, audit) in enumerate(refs):
            new, report = data.corrupt(raw, dict(rec, t0_token=str(i)), grid, noise, 21)
            np.testing.assert_array_equal(ref['history_occ'], new['history_occ']); assert audit == report


def test_ordered_CPU_lookahead_runs_during_consumer_and_closes(monkeypatch):
    source = object(); p = SimpleNamespace(ccr_history_cache_source=source)
    original = dict(history_occ=np.full((4, 2, 2, 1), 17, np.uint8), history_observed=np.zeros((4, 2, 2, 1), bool),
                    _joint_prepare_timing={'input_wait': .1})
    closed = []; started = []; next_started = threading.Event(); caller = threading.get_ident()
    def base(p, src, groups, *a, **kw):
        try:
            for g in groups: yield [({'n': i}, original, (), ()) for i in g]
        finally: closed.append(True)
    def augment(p, row, *a):
        assert threading.get_ident() != caller and row[1]['history_observed'].all()
        started.append(row[0]['n'])
        if row[0]['n'] >= 2: next_started.set()
        time.sleep(.015); return row
    monkeypatch.setattr(data, 'augment_row', augment)
    batches = pipeline.prefetch_adapter(base, 'empirical', None, 21)(p, source, [[i] for i in range(40)], None, workers=4)
    try:
        first = next(batches); assert first[0][0]['n'] == 0
        assert next_started.wait(1.)  # started future windows BEFORE consumer asks for next batch
        result = [first[0][0]['n']]
        for _ in range(6): result += [next(batches)[0][0]['n']]
        assert result == list(range(7))
    finally:
        batches.close()
    assert closed == [True] and not original['history_observed'].any()
    assert not any(t.name.startswith(('empirical-window', 'empirical-prefetch')) for t in threading.enumerate())
    assert first[0][1]['_joint_prepare_timing']['input_wait'] == 0
    assert 'simulation_and_rebuild_wait' in first[0][1]['_joint_prepare_timing']


def test_prefetch_error_propagation_and_clean_eval_passthrough(monkeypatch):
    source = object(); p = SimpleNamespace(ccr_history_cache_source=source)
    row = ({}, dict(history_occ=np.zeros((4, 2, 2, 1), np.uint8), history_observed=np.ones((4, 2, 2, 1), bool)), (), ())
    def base(*a, **kw): yield [row]
    def broken(*a): raise ValueError('bad geometry preserved')
    monkeypatch.setattr(data, 'augment_row', broken)
    iterator = pipeline.prefetch_adapter(base, 'empirical', None, 21)(p, source, None, None)
    with pytest.raises(ValueError, match='bad geometry'): next(iterator)
    assert list(pipeline.prefetch_adapter(base, 'empirical', None, 21)(p, object(), None, None))[0][0] is row
    assert not any(t.name.startswith('empirical-') for t in threading.enumerate())


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_reuse_prepared_Strong_exact_all_six_and_no_duplicate_CC(device, monkeypatch):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    model, _, p, original = real_fixture(device)
    hist = {k: copy.deepcopy(original[1][k]) for k in ('history_occ', 'history_observed', 'history_poses')}
    hist['history_occ'][:, 2:4, 2:4, 1:3] = 4; hist['history_occ'][:, :, :, 0] = 11
    hist['history_observed'][:] = True
    row = data.historical_row(p, original[0], hist, original[2][0])
    raw = dict(row[1], future_poses=original[1]['future_poses'], future_gt_occ=object(), _robust_background_pending=True)
    row = row[0], raw, row[2], row[3]
    reference = data.finish_backgrounds(p, row)
    def forbidden(*a, **kw): raise AssertionError('duplicate extraction/match')
    with monkeypatch.context() as m:
        m.setattr(pipeline.integration.runtime, 'extract_instances_cropped_exact', forbidden)
        m.setattr(pipeline.integration.runtime, 'match_instances', forbidden)
        optimized = pipeline.finish_backgrounds(p, row)
    a = reference[1]['_column_causal_preparation']['prepared_state']['column_backgrounds']
    b = optimized[1]['_column_causal_preparation']['prepared_state']['column_backgrounds']
    assert len(a) == len(b) == 6
    for x, y in zip(a, b): np.testing.assert_array_equal(x, y)
    reports = []; checked = pipeline.CheckedStrong(reports.append)
    checked(p, row); checked(p, row)
    assert len(reports) == 1 and reports[0]['pass_gate']
    assert pipeline.finish_backgrounds(p, optimized) is optimized
    assert row[1]['_robust_background_pending']  # input and cached geometry were NOT mutated


@pytest.mark.parametrize('arm', ['clean', 'empirical'])
def test_original_checkpoint_resume_through_execution_overlay(tmp_path, monkeypatch, arm):
    import test_stc_noise_adaptation as original_test
    # First update via ORIGINAL runner; second restores that exact old .pt
    # through the execution overlay (not merely new -> new resume).
    calls = []
    def dispatch(*args, **kw):
        target = run.main if not calls else fast.main
        calls.append(target); return target(*args, **kw)
    monkeypatch.setattr(original_test, 'run', SimpleNamespace(main=dispatch, joint_data=run.joint_data))
    original_test.test_real_warm_cli_stop_resume_scoped_hooks_and_frozen_body(tmp_path, monkeypatch, arm)
    assert calls == [run.main, fast.main]
    ledger = json.loads((tmp_path/'experiment/execution_speed_resume.json').read_text())
    assert ledger['source_update'] == 1 and not ledger['Adam_RNG_schedule_reset']
    assert not ledger['scientific_contract_modified'] and not ledger['CUDA_in_worker']


def test_saved_workflow_resume_skips_completed_clean_and_preserves_metadata(tmp_path, monkeypatch):
    gtpath = tmp_path/'source/GT.pt'; gtpath.parent.mkdir(); gtpath.write_bytes(b'GT')
    noise = profile_files(tmp_path/'noise', legacy.sha256(gtpath))
    keys = [['DEV', str(i)] for i in range(64)]
    monkeypatch.setattr(workflow, 'selection', lambda *a: (keys, []))
    paths = {n: tmp_path/n for n in ('config', 'train-cache', 'dev-cache', 'population-manifest',
                                    'base-checkpoint', 'train-info', 'dev-info')}
    for path in paths.values(): path.write_text('metadata')
    root = tmp_path/'experiment'
    args = [v for k, path in paths.items() for v in ('--'+k, str(path))]
    args += ['--dataroot', str(tmp_path), '--ccr-history-cache', str(tmp_path), '--ccr-val-history-cache', str(tmp_path),
        '--stc-root', str(tmp_path), '--out-dir', str(root), '--gt-checkpoint', str(gtpath), '--noise-model', str(noise)]
    def complete(arm):
        return dict(status='complete', updates=500,
            contract=dict(protocol=run.PROTOCOL, augmentation=arm, total_updates=500, eval_keys=keys),
            evaluation=dict(windows=64, evaluation_keys=keys, variants={'gt': {
                f'{m}_{r}': dict(mIoU=20., IoU=30., MovingMicro=10.)
                for m in ('occ_gt', 'stc_gt') for r in ('transport', 'semantic_refine')}}))
    def first(command, check):
        d = Path(command[command.index('--out-dir')+1]); d.mkdir(); (d/'train').mkdir()
        (d/'train/last.pt').write_bytes(b'saved Adam cursor')
        (d/'initialization_audit.json').write_text('{}')
        result = complete(d.name) if d.name == 'clean' else dict(status='stopped_at_complete_update')
        (d/'results.json').write_text(json.dumps(result)); return SimpleNamespace(returncode=0)
    workflow.main(argv=args+['--max-updates', '8'], launch=first)
    before = (root/'workflow.json').read_bytes(), (root/'clean/results.json').read_bytes()
    calls = []
    def resumed(command, check):
        calls.append(command); assert 'tools.stc_noise_execution.run' in command and '--resume' in command
        d = Path(command[command.index('--out-dir')+1]); assert d.name == 'empirical'
        (d/'results.json').write_text(json.dumps(complete('empirical'))); return SimpleNamespace(returncode=0)
    assert fast_workflow.main(argv=['--out-dir', str(root)], launch=resumed) == 0
    assert len(calls) == 1 and before == ((root/'workflow.json').read_bytes(), (root/'clean/results.json').read_bytes())
    fast_workflow.main(argv=['--out-dir', str(root)], launch=resumed); assert len(calls) == 1
