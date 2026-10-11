"""Narrow execution: original window network order, grouped renderer only."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from test_joint_occ_ego import fixture
from test_joint_occ_batching import empty_row
from test_joint_occ_speed_probe import saved_fixture
from tools.joint_occ_experiments.engine import train_arm as reference_step
from tools.joint_occ_experiments.horizons import engine, probe, run as wrapper
from tools.joint_occ_experiments.performance.probe import assert_close_state, legacy_fingerprint


@pytest.mark.parametrize('empty', [False, True])
def test_two_adam_updates_preserve_original_batch4_losses_links_state(empty):
    torch.manual_seed(12)
    model, optimizer, provider, row = fixture()
    candidate, adam = copy.deepcopy((model, optimizer))
    rows = [copy.deepcopy(row) for _ in range(4)]
    for i, item in enumerate(rows): item[0]['features'] = item[0]['features']+i*.02
    if empty: rows[:2] = [empty_row(row, empty_evidence=True), empty_row(row)]
    for update in (1, 2):
        expected = reference_step(model, optimizer, provider, rows, 'gt', update=update, total_updates=20, probe=True)
        actual = engine.train_arm(candidate, adam, provider, rows, 'gt', update=update, total_updates=20, probe=True)
        for key in ('CE', 'Lovasz', 'source_geometry', 'existence', 'ego_geometry', 'grad_norm'):
            assert actual[key] == pytest.approx(expected[key], rel=2e-4, abs=2e-5)
        for key, value in actual['gradient_links'].items():
            assert value == pytest.approx(expected['gradient_links'][key], rel=2e-4, abs=2e-5)
        assert_close_state(model.state_dict(), candidate.state_dict())
        assert_close_state(optimizer.state_dict(), adam.state_dict())
    assert actual['windows'] == 4


def test_original_forward_backward_interleaving_and_no_network_batching(monkeypatch):
    model, optimizer, provider, row = fixture()
    events = []
    old_forward = engine.forward_window
    old_backward = engine.RenderGradientBridge.backward_shared
    def forward(*args, **kwargs):
        events.append(('forward', len(args[2][0]['features'])))
        return old_forward(*args, **kwargs)
    def backward(self, auxiliary):
        old_backward(self, auxiliary)
        events.append(('backward', None))
    def forbidden(*args, **kwargs): raise AssertionError('packed network must NEVER run')
    from tools.joint_occ_experiments.batching import engine as packed
    monkeypatch.setattr(packed, 'forward_batch', forbidden)
    monkeypatch.setattr(engine, 'forward_window', forward)
    monkeypatch.setattr(engine.RenderGradientBridge, 'backward_shared', backward)
    engine.train_arm(model, optimizer, provider, [row]*4, 'gt', update=1, total_updates=20)
    assert events == [('forward', 1), ('backward', None)]*4


def test_readonly_two_route_probe_reuses_no_packed_forward_and_restores_state(monkeypatch):
    model, optimizer, provider, row, saved = saved_fixture()
    saved['contract'].update(window_batch=4, source_budget=128)
    immutable = copy.deepcopy(saved)
    from tools.joint_occ_experiments.batching import engine as packed
    def forbidden(*args, **kwargs): raise AssertionError('no packed network')
    monkeypatch.setattr(packed, 'forward_batch', forbidden)
    previous = probe.audit.BACKENDS, probe.audit.train_arm, probe.audit.execution_fingerprint
    result = probe.measure(model, optimizer, provider, saved, [[row, row], [row]], progress=lambda *a, **k: None)
    assert tuple(result['seconds_per_window']) == ('current_fused', 'sequential_h2')
    assert result['projection_workers'] == 0
    assert all(r['numerical_parity'] for r in result['warmup'].values())
    assert all(r['numerical_parity'] for r in result['final_state_parity'].values())
    assert not result['candidates']['sequential_h2']['pass_gate'] # CPU memory gate is false.
    assert (probe.audit.BACKENDS, probe.audit.train_arm, probe.audit.execution_fingerprint) == previous
    assert_close_state(model.state_dict(), saved['models']['gt'])
    assert_close_state(optimizer.state_dict(), saved['optimizers']['gt'])
    for key in ('models', 'optimizers', 'contract', 'epoch', 'batch', 'updates', 'reports'):
        assert_close_state(saved[key], immutable[key])
    assert torch.equal(saved['rng']['torch'], immutable['rng']['torch'])
    np.testing.assert_array_equal(saved['rng']['numpy'][1], immutable['rng']['numpy'][1])
    assert saved['rng']['python'] == immutable['rng']['python']
    assert torch.equal(wrapper.legacy.rng_state()['torch'], saved['rng']['torch'])


def test_failed_probe_restores_adapter_globals(monkeypatch):
    previous = probe.audit.BACKENDS, probe.audit.train_arm, probe.audit.execution_fingerprint
    def fail(*args, **kwargs): raise RuntimeError('diagnostic failure')
    monkeypatch.setattr(probe.audit, 'measure', fail)
    with pytest.raises(RuntimeError, match='diagnostic failure'):
        probe.measure(None, None, None, None, None)
    assert (probe.audit.BACKENDS, probe.audit.train_arm, probe.audit.execution_fingerprint) == previous


def test_resume_import_and_same_execution_preserve_old_source(monkeypatch, tmp_path):
    model, optimizer, provider, row, saved = saved_fixture()
    root = Path(wrapper.__file__).resolve().parents[3]
    saved['contract'].update(implementation=legacy_fingerprint(root), window_batch=4, source_budget=128)
    original = tmp_path/'original'; original.mkdir()
    torch.save(saved, original/'last.pt')
    before = (original/'last.pt').read_bytes()
    fingerprint = probe.execution_fingerprint(root)
    proof = tmp_path/'proof.json'
    receipt = dict(status='complete', source_snapshot_sha256=wrapper.legacy.sha256(original/'last.pt'),
        candidates=dict(sequential_h2=dict(execution_fingerprint=fingerprint, pass_gate=True,
            numerical_parity=True, baseline_repeat_parity=True, faster_each_pass=True,
            memory_headroom_ge_10percent=True, speedup=1.2)))
    proof.write_text(json.dumps(receipt), encoding='utf-8')
    previous = wrapper.legacy.train_arm, wrapper.legacy.restore_checkpoint, wrapper.legacy.stable_json_fingerprint, wrapper.legacy.parser
    def fake_main(event, argv):
        a = wrapper.legacy.parser().parse_args(argv)
        state = torch.load(a.resume, weights_only=False)
        current = {**state['contract'], 'implementation': wrapper.legacy.stable_json_fingerprint(
            {'tools/joint_occ_experiments/run.py': 'unchanged_original_code'})}
        assert current['implementation'] == fingerprint
        cursor = wrapper.legacy.restore_checkpoint(state, {'gt': model}, {'gt': optimizer}, current)
        assert cursor[:3] == (state['epoch'], state['batch'], state['updates'])
        assert torch.equal(wrapper.legacy.rng_state()['torch'], state['rng']['torch'])
        wrapper.legacy.train_arm(model, optimizer, provider, [row, row], 'gt', update=state['updates']+1, total_updates=7)
        destination = Path(a.out_dir); destination.mkdir(exist_ok=True)
        torch.save(wrapper.legacy.checkpoint({'gt': model}, {'gt': optimizer}, current, 0,
            state['batch']+1, state['updates']+1, {}), destination/'last.pt')
        return 0
    monkeypatch.setattr(wrapper.legacy, 'main', fake_main)
    out = tmp_path/'horizons'
    args = []
    for name in ('config', 'train-cache', 'dev-cache', 'population-manifest', 'base-checkpoint',
                 'dataroot', 'train-info', 'dev-info', 'ccr-history-cache', 'ccr-val-history-cache'):
        args += ['--'+name, str(tmp_path/name)]
    args += ['--mode', 'gt', '--out-dir', str(out), '--resume', str(original/'last.pt'), '--horizon-proof', str(proof)]
    bad = copy.deepcopy(receipt); bad['candidates']['sequential_h2']['baseline_repeat_parity'] = False
    proof.write_text(json.dumps(bad), encoding='utf-8')
    with pytest.raises(RuntimeError): wrapper.main(argv=args)
    assert not out.exists()
    proof.write_text(json.dumps(receipt), encoding='utf-8')
    assert wrapper.main(argv=args) == 0
    assert (original/'last.pt').read_bytes() == before
    assert json.loads((out/'horizon_import.json').read_text())['source_updates'] == saved['updates']
    args[args.index('--resume')+1] = str(out/'last.pt')
    # Ordinary same-directory resume no longer needs a probe receipt.
    del args[-2:]
    assert wrapper.main(argv=args) == 0
    restored = torch.load(out/'last.pt', weights_only=False)
    assert restored['updates'] == saved['updates']+2
    assert (wrapper.legacy.train_arm, wrapper.legacy.restore_checkpoint,
            wrapper.legacy.stable_json_fingerprint, wrapper.legacy.parser) == previous


def test_packed_receipt_cannot_authorize_horizon_only_resume():
    receipt = dict(status='complete', source_snapshot_sha256='source', candidates=dict(packed_h2=dict(pass_gate=True)))
    with pytest.raises(RuntimeError): wrapper.validate_receipt(receipt, 'sequential_h2', 'source', 'code')
