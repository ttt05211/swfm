import copy
import json
from pathlib import Path
import threading
import pytest
import torch
from test_full_gt_training import fixture
from tools.joint_occ_experiments import run as legacy
from tools.shared_history_experiments.full_gt.model import PROTOCOL
from tools.shared_history_experiments.refiner_batch_readout import probe
from tools.shared_history_experiments.refiner_batch_readout.optimized import SERVER_BACKENDS


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_actual_b4_probe_restores_model_adam_rng_and_preserves_failures(device):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    torch.set_num_threads(1); torch.manual_seed(19)
    model, adam, provider, row = fixture()
    model.to(device); provider.device = torch.device(device)
    with torch.no_grad(): model.columns.readout[-1].weight.normal_(std=.02)
    from tools.shared_history_experiments.full_gt import engine
    engine.train_arm(model, adam, provider, [row], 'gt', update=1, total_updates=6)
    before = copy.deepcopy(model.state_dict()); before_adam = copy.deepcopy(adam.state_dict())
    model.eval().requires_grad_(False)
    saved = dict(models={'gt': before}, optimizers={'gt': before_adam}, rng=legacy.rng_state(), updates=1,
                 contract=dict(total_updates=6))
    caller_rng = legacy.rng_state()
    report = probe.measure(model, adam, provider, saved, [[row]*4, [row]*4],
                           progress=lambda *a, **k: None)
    assert set(report['seconds_per_window']) == set(SERVER_BACKENDS)
    assert all(len(v) == 2 for v in report['timed_passes'].values())
    assert report['scientific_updates_published'] == 0 and not report['automatic_promotion']
    assert report['strict_FAIL_retained'] and report['approximate_execution_test_authorized']
    assert all(not c['quality_verified'] for c in report['candidates'].values())
    assert all(report['warmup'][n]['loss']['pass_gate'] for n in SERVER_BACKENDS)
    assert all(report['warmup'][n]['full6_semantic_changed_voxels'] == 0 for n in SERVER_BACKENDS)
    assert not model.training and all(not p.requires_grad for p in model.parameters())
    for k, v in before.items(): torch.testing.assert_close(model.state_dict()[k], v, rtol=0, atol=0)
    from tools.joint_occ_experiments.performance.probe import state_parity
    assert state_parity(before_adam, adam.state_dict())['numerical_parity']
    assert torch.equal(legacy.rng_state()['torch'], caller_rng['torch'])
    if device == 'cpu': assert all(not c['pass_gate'] for c in report['candidates'].values())


def test_source_gate_auto_selection_and_rejection(tmp_path):
    contract = dict(protocol=PROTOCOL, modes=['gt'], epochs=2, train_keys=[['TRAIN', str(i)] for i in range(20430)],
                    eval_keys=[['DEV', str(i)] for i in range(512)], window_batch=4, source_budget=128,
                    total_updates=10000, model=dict(objective=dict(ce=1., lovasz=1., motion=0., existence=0., ego=0.)))
    saved = dict(protocol=PROTOCOL, contract=contract, epoch=0, batch=123, updates=123)
    probe.validate_source(saved)
    for change in (dict(protocol='OLD'), dict(updates=10001)):
        with pytest.raises(RuntimeError): probe.validate_source({**saved, **change})
    root = tmp_path/'outputs/p0_f9_shared_history'
    for name, data in (('full_gt_ce_lovasz_2ep_valid', saved), ('full_gt_ce_lovasz_2ep_wrong', dict(protocol='OLD'))):
        folder = root/name/'train'; folder.mkdir(parents=True)
        torch.save(data, folder/'last.pt')
    chosen = probe.resolve_source('AUTO_FULL_GT', tmp_path)
    assert chosen == (root/'full_gt_ce_lovasz_2ep_valid/train/last.pt').resolve()


def test_interruption_restores_hooks_and_source_state():
    model, adam, provider, row = fixture()
    before = copy.deepcopy(model.state_dict())
    saved = dict(models={'gt': before}, optimizers={'gt': copy.deepcopy(adam.state_dict())},
                 rng=legacy.rng_state(), updates=0, contract=dict(total_updates=4))
    event = threading.Event(); event.set()
    with pytest.raises(InterruptedError): probe.measure(model, adam, provider, saved, [[row]*4], stop_event=event)
    assert 'forward' not in model.columns.__dict__ and not model.columns._forward_hooks
    for k, v in before.items(): torch.testing.assert_close(model.state_dict()[k], v, rtol=0, atol=0)


def test_candidate_oom_is_recorded_without_fallback_or_source_change(monkeypatch):
    model, adam, provider, row = fixture()
    before = copy.deepcopy(model.state_dict())
    saved = dict(models={'gt': before}, optimizers={'gt': copy.deepcopy(adam.state_dict())},
                 rng=legacy.rng_state(), updates=0, contract=dict(total_updates=4))
    original = probe.replay_update
    def run(*args, **kwargs):
        if args[4] == 'packed_reference': raise torch.cuda.OutOfMemoryError('CUDA out of memory')
        return original(*args, **kwargs)
    monkeypatch.setattr(probe, 'replay_update', run)
    report = probe.measure(model, adam, provider, saved, [[row]], progress=lambda *a, **k: None)
    assert report['seconds_per_window']['packed_reference'] is None
    assert not report['candidates']['packed_reference']['pass_gate']
    assert report['failed']['packed_reference']['stage'] == 'warmup'
    assert report['seconds_per_window']['packed_slice'] is not None
    for k, v in before.items(): torch.testing.assert_close(model.state_dict()[k], v, rtol=0, atol=0)


def test_real_cli_full_gt_interim_source_readonly_no_dev_eval(tmp_path, monkeypatch):
    from test_full_gt_training import test_real_cli_full_population_epoch_snapshots_resume_and_eval_readonly
    test_real_cli_full_population_epoch_snapshots_resume_and_eval_readonly(tmp_path, monkeypatch)
    source = tmp_path/'pilot/epoch_0001.pt'
    receipts = {p: p.read_bytes() for p in (source, tmp_path/'pilot/last.pt', tmp_path/'pilot/summary.txt')}
    out = tmp_path/'speed'
    args = [v for name, path in (
        ('config', tmp_path/'config.yaml'), ('base-checkpoint', tmp_path/'base.pt'),
        ('train-cache', tmp_path/'train-cache'), ('dev-cache', tmp_path/'dev-cache'),
        ('population-manifest', tmp_path/'population-manifest'), ('dataroot', tmp_path),
        ('train-info', tmp_path/'train-info'), ('dev-info', tmp_path/'dev-info'),
        ('ccr-history-cache', tmp_path), ('ccr-val-history-cache', tmp_path),
        ('out-dir', out), ('resume', source)) for v in ('--'+name, str(path))]
    assert probe.main(argv=args+['--probe-steps', '2']) == 0
    report = json.loads((out/'speed.json').read_text())
    assert set(report['seconds_per_window']) == set(SERVER_BACKENDS)
    assert report['source_checkpoint'] == str(source) and report['source_update'] == 2
    assert report['source_data_contract']['objective']['ego'] == 0
    assert all(k[0] != 'DEV' for k in json.loads((out/'population.json').read_text()))
    assert not (out/'last.pt').exists()
    for p, value in receipts.items(): assert p.read_bytes() == value
