import copy
import json
import threading
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from test_full_gt_training import fixture
from test_scene_ego_head import causal_background
from tools.scene_ego_head import integration
from tools.scene_ego_head.model import SceneEgoHead, SceneEgoConfig, prepare_scene
from tools.scene_ego_head.frozen_gt import bank, training, engine, run
from tools.shared_history_experiments.full_gt.model import PROTOCOL
from tools.real_motion.surface_ego_ablation_common import tensor_state_fingerprint


def small_bank(body, row, grid, count=6):
    poses = row[1]['history_poses']
    from real_motion.ego_navigation import ego_history_features
    raw = {k: row[1][k] for k in ('history_occ', 'history_observed', 'history_poses')}
    times = ego_history_features(poses)[:, -1]*1.5
    values = bank.materialize(body, raw, times, grid, pool_side=2)
    head = SceneEgoHead(SceneEgoConfig(body.transport.history_encoder.width, 32, 4, 2))
    e = values['history'][None]
    scene = prepare_scene(body.transport.history_encoder(torch.as_tensor(raw['history_occ']),
        torch.as_tensor(raw['history_observed'])), observed=raw['history_observed'], history_poses=poses,
        ego_history=e[0], grid=grid, pool_side=2)
    target = head(e, torch.full((1, 6), 2), scene)['prior'][0].detach()+torch.tensor([.25, -.1, .03])
    return {**{k: v[None].repeat(count, *([1]*(v.ndim))) for k, v in values.items()},
        'commands': torch.full((count, 6), 2), 'target': target[None].repeat(count, 1, 1)}


def test_final_source_gate():
    c = dict(protocol=PROTOCOL, modes=['gt'], epochs=2, train_keys=[['TRAIN', str(i)] for i in range(20430)],
        eval_keys=[['DEV', str(i)] for i in range(512)], window_batch=4, source_budget=128, total_updates=100,
        model=dict(objective=dict(ce=1., lovasz=1., motion=0., existence=0., ego=0.)))
    good = dict(protocol=PROTOCOL, contract=c, updates=100, epoch=2, batch=0)
    run.validate_completed_source(good)
    for change in (dict(epoch=1), dict(updates=99), dict(batch=1)):
        with pytest.raises(RuntimeError, match='FINISH'): run.validate_completed_source({**good, **change})


def test_history_bank_no_future_features_and_prefix_resume(tmp_path, monkeypatch):
    torch.set_num_threads(1); body, _, p, row = fixture(); body.eval().requires_grad_(False)
    before = tensor_state_fingerprint(body.state_dict())
    raw = copy.deepcopy(row[1]); raw['future_poses'] = object(); raw['future_gt_occ'] = object()
    raw['_column_causal_preparation'] = object()
    times = np.arange(4)*.5
    a = bank.materialize(body, row[1], times, p.pcfg.grid, pool_side=2)
    b = bank.materialize(body, raw, times, p.pcfg.grid, pool_side=2)
    for k in a: torch.testing.assert_close(a[k], b[k], rtol=0, atol=0)
    records = [{**row[0], 't0_token': str(i)} for i in range(5)]
    monkeypatch.setattr(bank, 'navigation_commands', lambda *a: np.full(6, 2, np.int64))
    source = SimpleNamespace(nusc=None, pose=lambda _: np.eye(4))
    event = threading.Event(); calls = []
    def reader(record):
        calls.append(record['t0_token']); return raw, times
    original = bank.legacy.write_json
    def stopped(path, value):
        original(path, value)
        if path.name == 'manifest.json' and len(value['chunks']) == 1: event.set()
    # Stop AFTER publication of one full 2-window chunk, not during extraction.
    monkeypatch.setattr(bank.legacy, 'write_json', stopped)
    with pytest.raises(InterruptedError):
        bank.build(body, records, source, p.pcfg.grid, tmp_path/'bank', {'source': 'fixture'},
                   pool_side=2, workers=1, stop_event=event, reader=reader, chunk_size=2)
    assert json.loads((tmp_path/'bank/manifest.json').read_text())['chunks'][0]['windows'] == 2
    monkeypatch.setattr(bank.legacy, 'write_json', original); event.clear()
    values, manifest = bank.build(body, records, source, p.pcfg.grid, tmp_path/'bank', {'source': 'fixture'},
        pool_side=2, workers=1, reader=reader, chunk_size=2)
    assert manifest['complete'] and len(values['target']) == 5
    with pytest.raises(RuntimeError, match='changed'):
        bank.build(body, records, source, p.pcfg.grid, tmp_path/'bank', {'source': 'DIFFERENT'},
                   pool_side=2, reader=reader, chunk_size=2)
    assert tensor_state_fingerprint(body.state_dict()) == before and all(v.grad is None for v in body.parameters())


def test_head_only_updates_exact_recovery(tmp_path, monkeypatch):
    torch.set_num_threads(1); body, _, p, row = fixture(); body.eval().requires_grad_(False)
    before = tensor_state_fingerprint(body.state_dict()); values = small_bank(body, row, p.pcfg.grid)
    torch.manual_seed(21); initial = SceneEgoHead(SceneEgoConfig(16, 32, 4, 2))
    contract = dict(schedule=dict(epochs=3, batch_size=2, seed=21, lr=.0003), source='frozen fixture')
    full, resumed = tmp_path/'full', tmp_path/'resumed'; full.mkdir(); resumed.mkdir()
    a, b = copy.deepcopy(initial), copy.deepcopy(initial)
    training.train(a, values, full, contract)
    event = threading.Event(); step = torch.optim.AdamW.step; updates = []
    def stop(adam, *args, **kwargs):
        v = step(adam, *args, **kwargs); updates.append(1)
        if len(updates) == 2: event.set()
        return v
    monkeypatch.setattr(torch.optim.AdamW, 'step', stop)
    with pytest.raises(InterruptedError): training.train(b, values, resumed, contract, stop_event=event)
    monkeypatch.setattr(torch.optim.AdamW, 'step', step); event.clear()
    result = training.train(copy.deepcopy(initial), values, resumed, contract, resume=True)
    aa = torch.load(full/'head_last.pt', weights_only=False); bb = torch.load(resumed/'head_last.pt', weights_only=False)
    assert result['updates'] == 9 and tensor_state_fingerprint(aa['head']) == tensor_state_fingerprint(bb['head'])
    from tools.joint_occ_experiments.performance.probe import state_parity
    assert state_parity(aa['adam'], bb['adam'])['numerical_parity']
    assert tensor_state_fingerprint(aa['head']) != tensor_state_fingerprint(initial.state_dict())
    assert tensor_state_fingerprint(body.state_dict()) == before and all(v.grad is None for v in body.parameters())


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_internal_forward_future_poison_and_eval_resume(monkeypatch, device):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    torch.set_num_threads(1); body, _, p, row = fixture(); body.to(device); p.device = torch.device(device)
    body.eval().requires_grad_(False); head = SceneEgoHead(SceneEgoConfig(16, 32, 4, 2)).to(device).eval()
    monkeypatch.setattr(integration, 'rebuild_pred_backgrounds', causal_background)
    with torch.no_grad(): head.readout[-1].bias[0] = .05
    fw = engine.predict(body, head, p, row)
    raw = copy.deepcopy(row[1]); raw['future_poses'] = object(); raw['future_gt_occ'] = object()
    raw['_column_causal_preparation'] = object()
    poisoned = (row[0], raw, (row[2][0], row[2][1], object()), row[3])
    other = engine.predict(body, head, p, poisoned)
    for k in ('poses', 'prior', 'memory', 'logits'): torch.testing.assert_close(fw[k], other[k], rtol=0, atol=0)
    monkeypatch.setattr(engine.gt, 'gt_moving_support_sequence', lambda *a, **k: [])
    monkeypatch.setattr(engine.gt, 'moving_support_masks', lambda _, shape: np.ones((6, *shape), bool))
    next_row = ({**row[0], 't0_token': 'next'}, *row[1:]); states = []
    engine.evaluate(body, head, p, SimpleNamespace(nusc=None), [[row]], save=states.append)
    resumed, full = [], []
    result = engine.evaluate(body, head, p, SimpleNamespace(nusc=None), [[row], [next_row]],
                             saved=states[-1], save=resumed.append)
    engine.evaluate(body, head, p, SimpleNamespace(nusc=None), [[row], [next_row]], save=full.append)
    assert full[-1] == resumed[-1] and set(result['variants']) == {'internal'}
    assert set(result['trajectory']) == {'head', 'prior'} and not result['planner_used']


def test_same_pose_gate_reports_background_difference(monkeypatch):
    body, _, p, row = fixture(); body.eval()
    cached = row[1]['_column_causal_preparation']['prepared_state']['column_backgrounds']
    monkeypatch.setattr(integration, 'rebuild_pred_backgrounds', lambda *a: [np.array(x, copy=True) for x in cached])
    assert engine.entrance_audit(body, p, row)['pass_gate']
    def wrong(*args):
        values = [np.array(x, copy=True) for x in cached]; values[0].flat[0] = (int(values[0].flat[0])+1)%18
        return values
    monkeypatch.setattr(integration, 'rebuild_pred_backgrounds', wrong)
    assert not engine.entrance_audit(body, p, row)['pass_gate']


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_deterministic_entrance_preserves_amp_and_restores_settings(monkeypatch, device):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    torch.set_num_threads(1); body, _, p, row = fixture()
    body.to(device).eval(); p.device = torch.device(device)
    with torch.no_grad(): body.columns.readout[-1].weight.normal_(std=.03)
    cached = row[1]['_column_causal_preparation']['prepared_state']['column_backgrounds']
    monkeypatch.setattr(integration, 'rebuild_pred_backgrounds', lambda *a: [np.array(x, copy=True) for x in cached])
    settings = lambda: (torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(), torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic, torch.get_float32_matmul_precision(), torch.autocast)
    before = settings(); weights = tensor_state_fingerprint(body.state_dict())
    a, b = engine.entrance_audit(body, p, row), engine.entrance_audit(body, p, row)
    assert a == b and a['pass_gate'] and a['semantic_argmax_changed_voxels'] == 0
    assert a['original_precision_preserved'] and not a['deployment_execution_changed']
    assert settings() == before and tensor_state_fingerprint(body.state_dict()) == weights
    def failed(*args):
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
        raise RuntimeError('unsupported deterministic operation')
    monkeypatch.setattr(engine, '_entrance_audit', failed)
    with pytest.raises(RuntimeError, match='unsupported'): engine.entrance_audit(body, p, row)
    assert settings() == before


def test_real_cli_completed_body_train_head_and_internal_only(tmp_path, monkeypatch):
    from test_full_gt_training import test_real_cli_full_population_epoch_snapshots_resume_and_eval_readonly
    test_real_cli_full_population_epoch_snapshots_resume_and_eval_readonly(tmp_path, monkeypatch)
    from tools.joint_occ_experiments import run as legacy
    source = tmp_path/'pilot/epoch_0002.pt'; out = tmp_path/'ego'
    receipts = {p: p.read_bytes() for p in (source, tmp_path/'pilot/last.pt', tmp_path/'pilot/summary.txt')}
    _, _, _, row = fixture()
    monkeypatch.setattr(integration, 'rebuild_pred_backgrounds', causal_background)
    monkeypatch.setattr(engine, 'entrance_audit', lambda *a: dict(pass_gate=True, diagnostic_GT_poses_only=True))
    monkeypatch.setattr(engine.gt, 'gt_moving_support_sequence', lambda *a, **k: [])
    monkeypatch.setattr(engine.gt, 'moving_support_masks', lambda _, shape: np.ones((6, *shape), bool))
    def build(body, records, source, grid, directory, contract, **kwargs):
        assert len(records) == 20430 and not any(v.requires_grad for v in body.parameters())
        values = small_bank(body, row, grid)
        return values, dict(tensor_mib=1.)  # Simulated dataset; real head training and forecasts below.
    monkeypatch.setattr(bank, 'build', build)
    args = [v for n, path in (
        ('config', tmp_path/'config.yaml'), ('base-checkpoint', tmp_path/'base.pt'), ('train-cache', tmp_path/'train-cache'),
        ('dev-cache', tmp_path/'dev-cache'), ('population-manifest', tmp_path/'population-manifest'), ('dataroot', tmp_path),
        ('train-info', tmp_path/'train-info'), ('dev-info', tmp_path/'dev-info'), ('ccr-history-cache', tmp_path),
        ('ccr-val-history-cache', tmp_path), ('out-dir', out), ('resume', source)) for v in ('--'+n, str(path))]
    originals = legacy.load_cache, legacy.summarize, engine.gt.history_inputs
    assert run.main(argv=args+['--head-epochs', '1', '--head-pool-side', '2']) == 0
    result = json.loads((out/'results.json').read_text())
    assert result['evaluation']['windows'] == 512 and result['head_training']['updates'] == 1
    assert (out/'head_epoch1.pt').is_file() and not (out/'last.pt').exists()
    freeze = json.loads((out/'body_freeze_audit.json').read_text())
    assert freeze['unchanged'] and freeze['trainable_body_parameters'] == freeze['body_gradient_parameters'] == 0
    exported = (out/'head_epoch1.pt').read_bytes()
    assert run.main(argv=args+['--head-epochs', '1', '--head-pool-side', '2', '--resume-stage']) == 0
    assert (out/'head_epoch1.pt').read_bytes() == exported
    assert (legacy.load_cache, legacy.summarize, engine.gt.history_inputs) == originals
    for p, contents in receipts.items(): assert p.read_bytes() == contents
