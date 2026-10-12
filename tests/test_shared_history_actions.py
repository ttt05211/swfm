"""Dual action causality, deletion gradients, parity and safe recovery."""
import copy
import json
import threading
from dataclasses import asdict
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from test_shared_history_pilot import fixture as old_fixture
from tools.shared_history_experiments.actions import engine, run
from tools.shared_history_experiments.actions.model import SharedJointBoth, SharedRepairBoth, PROTOCOL
from tools.shared_history_experiments.model import SharedRepair
from tools.shared_history_experiments.speed.engine import forward_batch, PackedSceneRenderer, WideGroupedScene
from tools.joint_occ_experiments import run as legacy
from tools.joint_occ_experiments.soft_inference.fast import predict
from tools.joint_occ_experiments.soft_inference.engine import soft_predictions
from tools.joint_occ_experiments.fast_autograd import compose_sparse


def fixture(device='cpu'):
    old, _, p, row = old_fixture()
    p.device = torch.device(device)
    model = SharedJointBoth(old.transport.config, width=16, z_bins=4).to(p.device)
    p.joint = model
    adam = torch.optim.AdamW([
        dict(params=model.transport.parameters(), lr=5e-4, initial_lr=5e-4),
        dict(params=model.columns.parameters(), lr=3e-4, initial_lr=3e-4),
        dict(params=model.ego.parameters(), lr=3e-4, initial_lr=3e-4)])
    return model, adam, p, row


def test_dual_head_small_nonzero_remove_and_fallback_condition():
    old, new = SharedRepair(128), SharedRepairBoth(128)
    assert sum(p.numel() for p in new.parameters()) - sum(p.numel() for p in old.parameters()) == 609
    model, _, p, row = fixture()
    fw = engine.forward_window(model, p, row)
    assert torch.all((fw['actions'][..., 1] > 0) & (fw['actions'][..., 1] < .1))
    assert model.configs()['repair']['fallback_condition']
    assert model.columns.context.in_features == 32
    assert model.columns.readout[-1].out_features == 2


def test_future_occupancy_not_input_and_packed_fallback_matches_sequential():
    model, _, p, row = fixture(); model.eval()
    fw = engine.forward_window(model, p, row)
    raw = copy.deepcopy(row[1]); raw['future_gt_occ'] = 'POISON'
    raw['future_annotations'] = 'POISON'
    other = engine.forward_window(model, p, (row[0], raw, row[2], row[3]))
    torch.testing.assert_close(fw['actions'], other['actions'], rtol=0, atol=0)
    packed = forward_batch(model, p, [row, row])
    for value in packed:
        torch.testing.assert_close(value['actions'], fw['actions'], rtol=2e-5, atol=2e-6)


def test_remove_transfers_mass_to_fallback_and_has_correct_gradient():
    background = torch.zeros(18, 2)
    background[4, 0] = 1.; background[11, 1] = 1.
    r = torch.tensor([.25, .6], requires_grad=True)
    result = compose_sparse(background,
        [('remove', torch.tensor([0]), 4, torch.tensor([11])),
         ('remove', torch.tensor([1]), 11, torch.tensor([17]))], [r[:1], r[1:]])
    torch.testing.assert_close(result.sum(0), torch.ones(2))
    assert result[4, 0] == .75 and result[11, 0] == .25
    assert torch.isclose(result[17, 1], torch.tensor(.6))
    (-result[11, 0] - result[17, 1]).backward()
    torch.testing.assert_close(r.grad, -torch.ones(2))


def test_full_occ_updates_delete_head_and_exact_adam_rng_resume():
    model, adam, p, row = fixture()
    before = model.columns.readout[-1].weight[1].detach().clone()
    stats = engine.train_arm(model, adam, p, [row], 'gt', update=1, total_updates=3, probe=True)
    assert stats['gradient_links']['occupancy_to_shared_history'] > 0
    assert not torch.equal(model.columns.readout[-1].weight[1], before)
    contract = {'epochs': 2, 'action': 'add_remove'}
    old_protocol = legacy.PROTOCOL
    try:
        legacy.PROTOCOL = PROTOCOL
        saved = copy.deepcopy(legacy.checkpoint({'gt': model}, {'gt': adam}, contract, 0, 1, 1, {}))
        engine.train_arm(model, adam, p, [row], 'gt', update=2, total_updates=3)
        expected = copy.deepcopy(model.state_dict())
        expected_adam = copy.deepcopy(adam.state_dict())
        legacy.restore_checkpoint(saved, {'gt': model}, {'gt': adam}, contract)
        engine.train_arm(model, adam, p, [row], 'gt', update=2, total_updates=3)
        for key in expected: torch.testing.assert_close(model.state_dict()[key], expected[key], rtol=0, atol=0)
        for key, state in expected_adam['state'].items():
            for name, value in state.items(): torch.testing.assert_close(adam.state_dict()['state'][key][name], value, rtol=0, atol=0)
        saved['protocol'] = 'p0_f9_shared_history_gt_soft_addonly_20percent_v1_shared'
        with pytest.raises(RuntimeError, match='differs'):
            legacy.restore_checkpoint(saved, {'gt': model}, {'gt': adam}, contract)
    finally:
        legacy.PROTOCOL = old_protocol


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_six_frame_fast_matches_training_and_scene_packed_with_learned_remove(device):
    if device == 'cuda' and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        pytest.skip('CUDA/BF16 unavailable')
    model, adam, p, row = fixture(device)
    engine.train_arm(model, adam, p, [row], 'gt', update=1, total_updates=3)
    model.eval()
    with torch.no_grad():
        fw = engine.forward_window(model, p, row)
        expected = soft_predictions(p, fw, ('transport', 'add_only', 'add_remove'), range(6))
        for route in expected:
            actual = predict(p, fw, route, range(6), group_size=3)
            for a, b in zip(actual, expected[route]): np.testing.assert_array_equal(a, b)
        renderer = PackedSceneRenderer([fw, fw], p.pcfg.grid, p.device)
        actual = renderer.render_group([0, 1, 2], [fw, fw], [fw, fw])
        scene = WideGroupedScene(fw['prep'], p.pcfg.grid, p.device)
        keys = {k: fw[k] for k in ('poses', 'targets', 'yaws', 'mapped', 'actions')}
        wanted = scene.render_group([0, 1, 2], **keys, evidence=fw['evidence'], plan=fw['plan'])
        torch.testing.assert_close(actual, torch.cat([wanted, wanted]), rtol=2e-5, atol=2e-6)


def test_evaluation_integer_resume_preserves_weights_and_source_state(monkeypatch):
    model, _, p, row = fixture()
    monkeypatch.setattr(engine, 'gt_moving_support_sequence', lambda *a, **k: [])
    monkeypatch.setattr(engine, 'moving_support_masks', lambda _, shape: np.ones((6, *shape), bool))
    before = copy.deepcopy(model.state_dict()); states = []
    first = engine.evaluate({'gt': model}, p, SimpleNamespace(nusc=None), [[row]], save=lambda v: states.append(copy.deepcopy(v)))
    saved = json.loads(json.dumps(states[-1])); original = copy.deepcopy(saved)
    resumed_states, full_states = [], []
    second = engine.evaluate({'gt': model}, p, SimpleNamespace(nusc=None), [[row], [row]], saved=saved,
        save=lambda v: resumed_states.append(copy.deepcopy(v)))
    engine.evaluate({'gt': model}, p, SimpleNamespace(nusc=None), [[row], [row]],
        save=lambda v: full_states.append(copy.deepcopy(v)))
    # Compare exact integer counts, not dict equality on absent-class NaNs.
    assert resumed_states[-1]['metrics'] == full_states[-1]['metrics'] and second['windows'] == 2
    assert saved == original and first['main_readout'] == 'soft_add_remove'
    for key, value in before.items(): torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)


def test_real_cli_population_epoch_snapshots_resume_readonly_eval_and_old_rejection(tmp_path, monkeypatch):
    model, _, p, row = fixture()
    config = tmp_path/'config.yaml'; config.write_text('test')
    base = tmp_path/'base.pt'; torch.save({'model_config': asdict(model.transport.config)}, base)
    paths = {n: tmp_path/n for n in ('train-cache', 'dev-cache', 'train-info', 'dev-info', 'population-manifest')}
    for path in paths.values(): path.write_text('test')
    train = [{**row[0], 't0_token': f'train{i}'} for i in range(20430)]
    dev = [{**row[0], 'scene_name': 'DEV', 't0_token': f'dev{i}'} for i in range(4369)]
    keys = [('DEV', r['t0_token']) for r in dev[:512]]
    monkeypatch.setattr(legacy, 'require_cuda', lambda _: torch.device('cpu'))
    monkeypatch.setattr(legacy, 'load_cache', lambda path: ({}, train if path == str(paths['train-cache']) else dev))
    monkeypatch.setattr(legacy, 'load_manifest', lambda _: ({'parent_keys': keys}, keys[:64], None))
    monkeypatch.setattr(legacy, 'load_runtime_config', lambda *a: None)
    monkeypatch.setattr(legacy, 'make_prepare_config', lambda _: p.pcfg)
    monkeypatch.setattr(legacy, 'PilotProvider', lambda *a: p)
    monkeypatch.setattr(legacy, 'NuScenesWindowSource', lambda root, *, info_pkl: SimpleNamespace(nusc=None))
    class Disposable:
        kernels = pool = None
        def __init__(self, *a): pass
        def close(self): pass
    monkeypatch.setattr('real_motion.canonical_repair_execution.CanonicalCpuExecution', Disposable)
    monkeypatch.setattr('real_motion.strong_majority_execution.ParallelNativeMajority', Disposable)
    def caches(provider, *a):
        provider.ccr_history_cache = provider.ccr_val_history_cache = SimpleNamespace(stats=lambda: {}, close=lambda: None)
        return {'train': {'namespace': 'train'}, 'val': {'namespace': 'val'}}
    monkeypatch.setattr(legacy, 'attach_caches', caches)
    def groups(records, *a):
        assert len(records) == 4086
        return [[records[0]], [records[1]]]
    monkeypatch.setattr(legacy, 'epoch_groups', groups)
    monkeypatch.setattr(legacy, 'prefetched_batches', lambda provider, source, groups, descriptors, **k:
        ([(record, row[1], row[2], row[3]) for record in group] for group in groups))
    def step(m, adam, provider, rows, mode, **kw):
        adam.zero_grad(set_to_none=True)
        loss = sum(v.square().sum() for v in m.parameters())*.0001*torch.rand(())
        loss.backward(); adam.step()
        return dict(windows=len(rows), seconds=.01, stages_seconds={}, CE=float(loss.detach()), gradient_links={})
    monkeypatch.setattr(engine, 'train_arm', step)
    monkeypatch.setattr(engine, 'evaluate', lambda *a, **k: dict(windows=512, variants={}, trajectory={}))
    out = tmp_path/'pilot'
    args = ['--config', str(config), '--base-checkpoint', str(base), '--dataroot', str(tmp_path),
        '--ccr-history-cache', str(tmp_path), '--ccr-val-history-cache', str(tmp_path), '--out-dir', str(out)]
    args += [value for name, path in paths.items() for value in ('--'+name, str(path))]
    assert run.main(threading.Event(), args+['--max-updates', '2']) == 0
    saved = torch.load(out/'last.pt', weights_only=False)
    assert saved['protocol'] == PROTOCOL and saved['updates'] == 2
    assert len(saved['contract']['train_keys']) == 4086
    assert (out/'epoch_0001.pt').is_file()
    assert run.main(threading.Event(), args+['--resume', str(out/'last.pt')]) == 0
    assert (out/'epoch_0002.pt').is_file()
    last_bytes, summary_bytes = (out/'last.pt').read_bytes(), (out/'summary.txt').read_bytes()
    eval_args = args.copy(); eval_args[eval_args.index(str(out))] = str(out/'eval')
    assert run.main(threading.Event(), eval_args+['--resume', str(out/'epoch_0001.pt'), '--evaluate-only']) == 0
    assert (out/'last.pt').read_bytes() == last_bytes and (out/'summary.txt').read_bytes() == summary_bytes
    saved['protocol'] = 'OLD_ADD_ONLY'
    wrong = out/'old.pt'; torch.save(saved, wrong)
    with pytest.raises(RuntimeError, match='differs'): run.main(threading.Event(), args+['--resume', str(wrong)])
