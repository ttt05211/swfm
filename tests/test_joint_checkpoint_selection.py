"""Retained-epoch lineage, byte-exact inference and synchronized eval recovery."""
import copy
import json
from pathlib import Path
from threading import Event, get_ident
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from real_motion.native_column_cpu import backend_name, prepare_native
from real_motion.joint_causal_columns import JointCausalColumns
from real_motion.local_st_world_model_v17 import LocalSTWMV17Config
from real_motion.causal_column_model import CausalColumnModel
from tools.real_motion import causal_column_common as col
from tools.real_motion import joint_checkpoint_evaluation as group
from tools.real_motion.joint_checkpoint_selection import audit_runs, weight_fingerprint, shortlist, lineage
from tools.real_motion.static_evidence_selector_common import finite_json
from test_joint_causal_columns import fixture, provider_for
from test_causal_columns import moving_fixture
from test_joint_causal_columns_full import full_cli_fixture

# Shell native preflight is another process; tests must initialize themselves.
if backend_name() == 'native': prepare_native()


def _epoch(e):
    values = dict(mIoU=50.-abs(e-3), IoU=60., MovingMicro=30.-abs(e-2), MovingMacro=20.)
    return dict(epoch=e, attempted_updates=e*100,
        dev64_fixed_gate={'variants': {'joint': {'metrics': values}}})


def _audit_fixture(tmp_path):
    old = tmp_path/'full15_history4_old'/'model'; new = tmp_path/'full20_history4_new'/'model'
    old.mkdir(parents=True); new.mkdir(parents=True)
    contract = dict(model_configs={'motion': {'history_frames': 4}}, seed=43,
        train_keys=[['train', 'a']], dev_keys=[['dev', 'b']], window_batch_size=4, source_budget=128,
        dev_manifest_fingerprint='frozen', arguments={'resume': None})
    (old/'execution_contract.json').write_text(json.dumps(contract))
    latest = copy.deepcopy(contract); latest['arguments']['resume'] = str(old/'last.pt')
    (new/'execution_contract.json').write_text(json.dumps(latest))
    history = [_epoch(e) for e in (1, 2, 3, 15, 16, 17, 18, 19, 20)]
    (old/'epoch_history.json').write_text(json.dumps(history[:4]))
    (new/'epoch_history.json').write_text(json.dumps(history))
    def checkpoint(directory, epoch, role='epoch_snapshot', name=None):
        path = directory/(name or f'epoch_{epoch:04d}.pt')
        torch.save({**contract, 'checkpoint_role': role, 'cursor_epoch': epoch, 'cursor_batch': 0,
            'prior_completed': True, 'attempted_updates': epoch*100,
            'state_dict': {'weight': torch.tensor([float(epoch)])}}, path)
        return path
    for e in (2, 3, 15): checkpoint(old, e)
    checkpoint(old, 15, 'resume_last', 'last.pt')
    for e in (16, 17, 18, 19, 20): checkpoint(new, e)
    checkpoint(new, 20, 'calibrated_candidate', 'candidate.pt')
    return old, new, checkpoint


def test_lineage_uses_resume_not_directory_name_and_reports_missing_weights(tmp_path):
    old, new, _ = _audit_fixture(tmp_path)
    stranger = tmp_path/'full20_history6_latest'/'model'; stranger.mkdir(parents=True)
    (stranger/'execution_contract.json').write_text(json.dumps({'model_configs': {'motion': {'history_frames': 6}}}))
    original = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    audit = audit_runs(new.parent, tmp_path, 8)
    assert audit['lineage'] == [str(new), str(old)]
    assert audit['missing_weight_epochs'] == [1]
    assert len(audit['selected']) == 8
    assert {2, 3, 15, 20} <= {r['epoch'] for r in audit['selected']}
    last = next(r for r in audit['epochs'] if r['epoch'] == 20)
    assert last['checkpoint']['role'] == 'epoch_snapshot' and len(last['aliases']) == 2
    assert any(r['history_frames'] == 6 and not r['in_lineage'] for r in audit['directories'])
    assert all(p.read_bytes() == value for p, value in original.items())


@pytest.mark.parametrize('problem', ('cycle', 'history', 'outside'))
def test_unsafe_lineages_fail_closed(tmp_path, problem):
    old, new, _ = _audit_fixture(tmp_path)
    path = old/'execution_contract.json'; contract = json.loads(path.read_text())
    if problem == 'cycle': contract['arguments']['resume'] = str(new/'last.pt')
    elif problem == 'history': contract['model_configs']['motion']['history_frames'] = 6
    else: contract['arguments']['resume'] = str(tmp_path.parent/'outside'/'model'/'last.pt')
    path.write_text(json.dumps(contract))
    with pytest.raises(RuntimeError): lineage(new.parent, tmp_path)


def test_conflicting_weights_same_epoch_update_rejected(tmp_path):
    _, new, _ = _audit_fixture(tmp_path)
    path = new/'candidate.pt'; ck = torch.load(path, weights_only=False); ck['state_dict']['weight'] += .1
    torch.save(ck, path)
    with pytest.raises(RuntimeError, match='different weights'): audit_runs(new, tmp_path)


def test_weight_hash_handles_scalars_bfloat16_and_detects_values():
    a = {'scalar': torch.tensor(1., dtype=torch.bfloat16), 'empty': torch.empty(0)}
    assert weight_fingerprint(a) == weight_fingerprint(copy.deepcopy(a))
    assert weight_fingerprint(a) != weight_fingerprint({**a, 'scalar': a['scalar']+1})


def _eval_fixture(frames=4):
    prep, grid, joint, _, record = fixture()
    mc = LocalSTWMV17Config(d_model=8, semantic_dim=4, heads=2, blocks=1, decoder_blocks=1, history_frames=frames)
    joint = JointCausalColumns(mc, joint.columns.config).eval()
    for key in ('history_occ', 'history_observed', 'history_poses'): prep.raw[key] = prep.raw[key][-frames:]
    prep.registrations = [r[-frames:] for r in prep.registrations]
    records = [{**copy.deepcopy(record), 'scene_name': 'dev', 't0_token': f't{i}'} for i in range(3)]
    source = SimpleNamespace(nusc=None)
    def owner(model):
        provider = provider_for(prep, grid, model); provider.workers = 2
        provider.load_raw_columns = lambda source, record, include_gt: copy.deepcopy(prep.raw)
        provider.reference_enabled = True; provider.frozen_metric_counts = {}
        calls = []
        def references(prepared, record, *, skip_frozen=False):
            calls.append(skip_frozen)
            return {} if skip_frozen else {'frozen_E14': prep.baseline}
        provider.reference_predictions = references; provider.reference_calls = calls
        return provider
    return prep, grid, joint, records, source, owner


@pytest.mark.parametrize('frames', (4, 6))
@pytest.mark.parametrize('empty', (False, True))
def test_cpu_prefetch_probabilities_are_exact_and_network_stays_on_caller(frames, empty):
    prep, grid, joint, records, source, owner = _eval_fixture(frames)
    provider = owner(joint); prepared = provider.prepare_columns(source, records[0], include_gt=True)
    plan = col.candidate_plan(prepared, 1, grid, joint.columns.config)
    if empty: plan = plan.subset(np.empty(0, np.int64))
    model = joint.columns; caller = get_ident(); original = model.forward
    def forward(**kwargs):
        assert get_ident() == caller
        return original(**kwargs)
    with patch.object(model, 'forward', side_effect=forward):
        before = col.predict_probabilities(model, prepared, 1, plan, grid, torch.device('cpu'), 7, optimized=False)
        model.column_inference_verify_remaining = 1
        after = col.predict_probabilities(model, prepared, 1, plan, grid, torch.device('cpu'), 7, optimized=True)
    assert np.array_equal(before, after) and model.column_inference_verify_remaining == 0


@pytest.mark.parametrize('optimized', (False, True))
def test_unused_nonfinite_logits_cannot_escape_deferred_checks(optimized):
    prep, grid, joint, records, source, owner = _eval_fixture()
    prepared = owner(joint).prepare_columns(source, records[0], include_gt=True)
    plan = col.candidate_plan(prepared, 1, grid, joint.columns.config)
    model = CausalColumnModel(joint.columns.config, history_frames=4)
    actual = model.forward
    def broken(**kwargs):
        g, r = actual(**kwargs); g[:] = float('nan'); return g, r
    with patch.object(model, 'forward', side_effect=broken), pytest.raises(RuntimeError, match='nonfinite'):
        col.predict_probabilities(model, prepared, 1, plan, grid, torch.device('cpu'), 7, optimized=optimized)


def test_shared_group_is_exact_recomputes_models_and_shares_only_fixed_work():
    prep, grid, joint, records, source, owner = _eval_fixture()
    other = copy.deepcopy(joint)
    with torch.no_grad():
        joint.columns.generation.bias.fill_(1.5); other.columns.generation.bias.fill_(-2.)
    expected = {}
    with patch.object(col, 'gt_moving_support_sequence', return_value=moving_fixture(prep)):
        for name, model in (('a', joint), ('b', other)):
            expected[name] = col.evaluate_columns(owner(model), source, records, model.columns, (.5, .5, None),
                batch_size=7, diagnostic_thresholds=None, dev64_keys=[('dev', 't1')])
    jobs = {'a': (owner(joint), joint.columns), 'b': (owner(other), other.columns)}
    with patch.object(col, 'gt_moving_support_sequence', return_value=moving_fixture(prep)) as moving, \
            patch.object(group, 'window_from_record', return_value=prep.window):
        result, performance = group.evaluate_group(jobs, source, records, batch_size=7, dev64_keys=[('dev', 't1')])
    assert finite_json(result) == finite_json(expected)
    assert result['a']['all']['variants']['joint']['quality'] != result['b']['all']['variants']['joint']['quality']
    assert moving.call_count == 3 and performance['model_windows'] == 6
    assert jobs['a'][0].reference_calls == [False]*3
    assert jobs['b'][0].reference_calls == [True]*3


@pytest.mark.parametrize('stop_inside_model', (False, True))
def test_group_interrupt_resume_keeps_counts_scores_order_and_source_bytes(stop_inside_model):
    prep, grid, joint, records, source, owner = _eval_fixture()
    baseline_jobs = {'a': (owner(joint), joint.columns), 'b': (owner(copy.deepcopy(joint)), copy.deepcopy(joint.columns))}
    saved = {}; stop = Event()
    def save(wi, states, totals):
        saved.clear(); saved.update(cursor=wi, states=json.loads(json.dumps(states)))
    def progress(row):
        if row['window'] == 1 and row['event'] == ('evaluation' if stop_inside_model else 'group_window_complete'):
            stop.set()
    with patch.object(col, 'gt_moving_support_sequence', return_value=moving_fixture(prep)), \
            patch.object(group, 'window_from_record', return_value=prep.window):
        expected, _ = group.evaluate_group(baseline_jobs, source, records, batch_size=7)
        jobs = {'a': (owner(joint), joint.columns), 'b': (owner(copy.deepcopy(joint)), copy.deepcopy(joint.columns))}
        with pytest.raises(InterruptedError):
            group.evaluate_group(jobs, source, records, batch_size=7, stop_event=stop, progress=progress,
                save_state=save, checkpoint_every=2)
        assert saved['cursor'] == (0 if stop_inside_model else 1)
        restored = {'a': (owner(joint), joint.columns), 'b': (owner(copy.deepcopy(joint)), copy.deepcopy(joint.columns))}
        result, _ = group.evaluate_group(restored, source, records, batch_size=7,
            start_window=saved['cursor'], saved_states=saved['states'])
        assert finite_json(result) == finite_json(expected)


def test_group_can_finalize_all_windows_already_saved_without_replay():
    prep, grid, joint, records, source, owner = _eval_fixture()
    saved = {}
    def save(wi, states, totals): saved.update(cursor=wi, states=json.loads(json.dumps(states)))
    with patch.object(col, 'gt_moving_support_sequence', return_value=moving_fixture(prep)), \
            patch.object(group, 'window_from_record', return_value=prep.window):
        expected, _ = group.evaluate_group({'a': (owner(joint), joint.columns)}, source, records, save_state=save)
        got, _ = group.evaluate_group({'a': (owner(joint), joint.columns)}, source, records,
            start_window=saved['cursor'], saved_states=saved['states'])
    assert finite_json(got) == finite_json(expected)


def test_rank_reports_does_not_silently_relax_moving_guard():
    def report(miou, moving):
        return {'all': {'baseline': {'mIoU': 40.}, 'reference_metrics': {'frozen_E14': {'MovingMicro': 30.}},
            'variants': {'joint': {'metrics': {'mIoU': miou, 'IoU': 50., 'MovingMicro': moving}}}}}
    rows = group.rank_reports({'old': report(41., 30.), 'new': report(42., 29.)})
    assert rows['overall_mIoU_best'] == 'new' and rows['moving_safe_best'] == 'old'
    assert group.rank_reports({'bad': report(43., 29.)})['moving_safe_best'] is None


def test_rejected_train_fastpaths_default_off(monkeypatch):
    from real_motion.local_supervision_fastpath import enabled, static_roi_enabled
    monkeypatch.delenv('SWFM_LOCAL_FAST_SUPERVISION', raising=False)
    monkeypatch.delenv('SWFM_LOCAL_STATIC_ROI', raising=False)
    assert not enabled() and not static_roi_enabled()


def test_output_lease_rejects_concurrent_resume_and_releases_on_error(tmp_path):
    from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
    with pytest.raises(ValueError):
        with evaluation_lock(tmp_path):
            with pytest.raises(RuntimeError, match='another evaluator'):
                with evaluation_lock(tmp_path): pass
            raise ValueError('simulated evaluator failure')
    with evaluation_lock(tmp_path): pass


def test_real_checkpoint_cli_snapshots_and_resumes_without_source_writes(tmp_path, full_cli_fixture):
    from tools.real_motion import eval_p0_f9_joint_checkpoints as cli
    from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import sha256
    run, files, _ = full_cli_fixture
    prep, grid, dev, manifest, keys, _ = run.eval_data
    trained = tmp_path/'full2_history4'/'model'
    assert run(trained, 2, history_frames=4) in (None, 0)
    originals = {p: p.read_bytes() for p in trained.iterdir() if p.is_file()}
    class Provider:
        def __init__(self, checkpoint, digest, pcfg, device, workers, joint, control):
            self.device, self.pcfg, self.workers = device, pcfg, workers
            self.joint, self.model, self.control = joint, joint.transport, control
            self.reference_enabled = True; self.frozen_metric_counts = {}
        def load_raw_columns(self, source, record, *, include_gt):
            raw = copy.deepcopy(prep.raw)
            for key in ('history_occ', 'history_observed', 'history_poses'): raw[key] = raw[key][-4:]
            return raw
        def prepare_columns(self, source, record, *, include_gt, raw_window=None):
            result = copy.deepcopy(prep); result.raw = raw_window
            result.registrations = [r[-4:] for r in result.registrations]
            result.window.scene_name, result.window.t0_token = record['scene_name'], record['t0_token']
            result.outputs = self.joint.motion(record, self.device); return result
        def reference_predictions(self, prepared, record, *, skip_frozen=False):
            return {} if skip_frozen else {'frozen_E14': prepared.baseline}
    def evaluate(destination, resume=False, event=None):
        argv = ['eval', '--run-dir', str(trained.parent), '--runs-root', str(tmp_path),
            '--out-dir', str(destination), '--shortlist-size', '4', '--device', 'cpu', '--cpu-workers', '1']
        if resume: argv.append('--resume')
        with patch('sys.argv', argv), patch.object(cli, 'CLEAN_SHA256', 'a'*64), \
                patch.object(cli, 'make_prepare_config', return_value=SimpleNamespace(grid=grid)), \
                patch.object(cli, 'load_manifest', return_value=(json.loads(json.dumps(manifest)), keys[:3], None)), \
                patch.object(cli, 'load_cache', return_value=({}, list(reversed(dev)))), \
                patch.object(cli, 'evaluation_keys', return_value=tuple(keys[:3])), \
                patch.object(cli, 'sha256', side_effect=lambda p: 'a'*64 if Path(p) == files['base-checkpoint'] else sha256(p)), \
                patch.object(cli, 'EvaluationJointColumnProvider', Provider), \
                patch.object(cli, 'NuScenesWindowSource', return_value=SimpleNamespace(nusc=None)), \
                patch.object(col, 'gt_moving_support_sequence', return_value=moving_fixture(prep)), \
                patch.object(group, 'window_from_record', return_value=prep.window):
            return cli.main(event)
    expected = tmp_path/'uninterrupted'
    assert evaluate(expected) == 0
    cancelled = tmp_path/'interrupted'; stop = Event(); stop.set()
    assert evaluate(cancelled, event=stop) == 130
    assert not (cancelled/'comparison.json').exists() and not (cancelled/'summary.txt').exists()
    assert json.loads((cancelled/'evaluation_state.json').read_text())['completed_windows'] == 0
    saved = cancelled/'evaluation_state.json'; original_state = saved.read_bytes()
    corrupted = json.loads(original_state); corrupted['completed_windows'] = 1
    saved.write_text(json.dumps(corrupted))
    with pytest.raises(RuntimeError, match='resume identity/state'): evaluate(cancelled, resume=True)
    saved.write_bytes(original_state)
    assert evaluate(cancelled, resume=True) == 0
    before = json.loads((expected/'comparison.json').read_text()); after = json.loads((cancelled/'comparison.json').read_text())
    assert before['reports'] == after['reports'] and before['ranking'] == after['ranking']
    assert all(p.read_bytes() == value for p, value in originals.items())
    assert before['execution']['thresholds'] == [.5, .5, None]
    assert before['windows'] == 3 and set(before['reports']) == {'epoch_0001', 'epoch_0002'}
    with pytest.raises(SystemExit): evaluate(cancelled, resume=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='actual CUDA required')
def test_actual_cuda_prefetch_probabilities_and_four_way_reports():
    prep, grid, joint, records, source, owner = _eval_fixture()
    joint = joint.cuda(); provider = owner(joint); provider.device = torch.device('cuda')
    # The toy provider emits its live motion on the network's real device.
    def prepare(source, record, *, include_gt, raw_window=None, outputs=None):
        result = copy.deepcopy(prep); result.outputs = joint.motion(record, provider.device); return result
    provider.prepare_columns = prepare
    with patch.object(col, 'gt_moving_support_sequence', return_value=moving_fixture(prep)):
        before = col.evaluate_columns(provider, source, records, joint.columns, (.5, .5, None), diagnostic_thresholds=None)
        joint.columns.column_inference_optimized = True; joint.columns.column_inference_verify_remaining = 3
        after = col.evaluate_columns(provider, source, records, joint.columns, (.5, .5, None), diagnostic_thresholds=None)
    assert finite_json(before) == finite_json(after)
