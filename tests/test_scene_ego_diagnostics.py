"""Actual small-grid VJPs/forecasts/read-only driver, never real-data claims."""
import copy
import json
import threading
from dataclasses import asdict
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from test_scene_ego_head import refiner_fixture, causal_background
from test_shared_semantic_refiner import fixture as gt_fixture
from test_scene_ego_experiment import planner_files
from tools.scene_ego_head import integration
from tools.scene_ego_head.diagnostics import run, engine
from tools.scene_ego_head.experiment.planner import PlannerCache
from tools.joint_occ_experiments import run as legacy


def test_ego_checks_only_history_and_prior_same_population(monkeypatch):
    model, p, row = refiner_fixture()
    raw = copy.deepcopy(row[1]); raw['future_poses'] = 'POISON'; raw['future_gt_occ'] = 'POISON'
    raw['_column_causal_preparation'] = 'POISON GT BACKGROUND'
    row = (row[0], raw, row[2], row[3])
    monkeypatch.setattr(model.transport, 'forward', lambda *a: pytest.fail('STWM not needed'))
    monkeypatch.setattr(model.columns, 'forward', lambda *a: pytest.fail('dense Refiner not needed'))
    monkeypatch.setattr(integration, 'rebuild_pred_backgrounds', lambda *a: pytest.fail('no Strong in ego-only'))
    before = run.model_digest(model); states = []
    result = engine.ego_checks(model, p, [[row]], save=lambda x: states.append(copy.deepcopy(x)))
    assert result['windows'] == 1 and not result['dense_forecasting']
    assert result['head_minus_prior']['ADE_m'] == 0.  # zero residual initial head
    resumed = []; full = []
    engine.ego_checks(model, p, [[row], [row]], saved=states[-1], save=lambda x: resumed.append(copy.deepcopy(x)))
    engine.ego_checks(model, p, [[row], [row]], save=lambda x: full.append(copy.deepcopy(x)))
    assert resumed[-1] == full[-1] and run.model_digest(model) == before


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_gradient_vjps_no_weight_or_rng_changes(monkeypatch, device):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    model, p, row = refiner_fixture(device)
    monkeypatch.setattr(integration, 'rebuild_pred_backgrounds', causal_background)
    model.eval().requires_grad_(False)
    with torch.no_grad(): model.ego.readout[-1].weight.normal_(std=.003)
    digest = run.model_digest(model); rng = torch.get_rng_state().clone()
    result = engine.gradient_window(model, p, row)
    assert result['gradients']['pose_R10']['occupancy_norm'] > 0
    assert result['gradients']['ego_parameters']['ego_norm'] > 0
    assert len(result['per_horizon_pose_R10']) == 6
    assert run.model_digest(model) == digest and torch.equal(torch.get_rng_state(), rng)
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    assert all(p.grad is None for p in model.parameters())


def test_external_gt_body_no_gt_pose_background_or_internal_head(monkeypatch):
    model, _, p, row = gt_fixture()
    monkeypatch.setattr(integration, 'rebuild_pred_backgrounds', causal_background)
    monkeypatch.setattr(model.ego, 'forward', lambda *a: pytest.fail('GT auxiliary ego not used'))
    poses = np.stack(row[1]['future_poses'])
    expected = engine.external_forward(model, p, row, poses)
    raw = copy.deepcopy(row[1]); raw['future_poses'] = 'POISON'; raw['future_gt_occ'] = 'POISON'
    raw['_column_causal_preparation'] = 'POISON'
    actual = engine.external_forward(model, p, (row[0], raw, (row[2][0], 'POISON NAV', 'POISON LABEL'), row[3]), poses)
    for key in ('poses', 'prior', 'memory', 'logits'): torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)


def test_matched_two_models_population_and_atomic_resume(tmp_path, monkeypatch):
    gt, _, p, row = gt_fixture(); internal, _, _ = refiner_fixture()
    monkeypatch.setattr(integration, 'rebuild_pred_backgrounds', causal_background)
    monkeypatch.setattr(engine, 'gt_moving_support_sequence', lambda *a, **k: [])
    monkeypatch.setattr(engine, 'moving_support_masks', lambda _, shape: np.ones((6, *shape), bool))
    planner = PlannerCache(planner_files(tmp_path, row))
    key = (row[0]['scene_name'], row[0]['t0_token']); planner.prepare([key, ('missing', 'none')])
    missing = ({**row[0], 'scene_name': 'missing', 't0_token': 'none'}, *row[1:])
    before = (run.model_digest(gt), run.model_digest(internal)); states = []; event = threading.Event()
    original = engine.forward_pred_window
    def stopping(*a, **kw):
        value = original(*a, **kw); event.set(); return value
    monkeypatch.setattr(engine, 'forward_pred_window', stopping)
    with pytest.raises(InterruptedError):
        engine.matched_external(gt, internal, p, SimpleNamespace(nusc=None), [[row], [missing]], planner,
            stop_event=event, save=lambda x: states.append(copy.deepcopy(x)))
    assert len(states[-1]['errors']) == states[-1]['seen'] == 1
    monkeypatch.setattr(engine, 'forward_pred_window', original); event.clear(); full = []; resumed = []
    result = engine.matched_external(gt, internal, p, SimpleNamespace(nusc=None), [[row], [missing]], planner,
        saved=states[-1], save=lambda x: resumed.append(copy.deepcopy(x)))
    engine.matched_external(gt, internal, p, SimpleNamespace(nusc=None), [[row], [missing]], planner,
        save=lambda x: full.append(copy.deepcopy(x)))
    assert result['windows'] == 1 and not result['GT_pose_reference_run']
    assert resumed[-1]['counts'] == full[-1]['counts']
    assert set(result['variants']) == {'GT500_external', 'Pred500_internal'}
    assert before == (run.model_digest(gt), run.model_digest(internal))


@pytest.mark.parametrize('task', ['ego_checks', 'gt_external'])
def test_real_readonly_cli_source_provenance_resume_no_train(tmp_path, monkeypatch, task):
    model, p, row = refiner_fixture()
    paths = {name: tmp_path/name for name in ('config', 'train-cache', 'dev-cache', 'train-info', 'dev-info', 'population-manifest')}
    for path in paths.values(): path.write_text('fixture')
    base = tmp_path/'base.pt'; torch.save({'model_config': asdict(model.transport.config)}, base)
    train = [{**row[0], 't0_token': f'train{i}'} for i in range(20430)]
    dev = [{**row[0], 'scene_name': 'DEV', 't0_token': f'dev{i}'} for i in range(4369)]
    keys = [('DEV', r['t0_token']) for r in dev[:512]]
    train_keys = legacy.select_scene_balanced_round_robin(legacy.record_keys(train), 4086)
    c = dict(protocol=run.EGO_PROTOCOL, modes=['pred'], total_updates=500, model=model.configs(),
        train_keys=[list(k) for k in train_keys], eval_keys=[list(k) for k in keys], window_batch=4, source_budget=128,
        seed=20261011, weights=[1., 1., 1.], lr=[.0005, .0003, .0003], schedule='whole_cycle_cosine_floor_0.01',
        execution=dict(torch_version=str(torch.__version__), cuda_version=torch.version.cuda, motion_precision='FP32', device_type='cpu'),
        cache_namespace=dict(train='train', val='val'), data={name.replace('-', '_'): legacy.sha256(path)
            for name, path in paths.items() if name in ('train-cache', 'dev-cache', 'train-info', 'dev-info')})
    c['data']['base_checkpoint'] = legacy.sha256(base)
    (tmp_path/'original_run').mkdir()
    source = tmp_path/'original_run'/'original.pt'
    ego_source = tmp_path/'original_run'/'internal.pt'
    torch.save(dict(protocol=run.EGO_PROTOCOL, contract=c, epoch=1, batch=0, updates=500,
        models={'pred': model.state_dict()}, optimizers={'pred': {'POISON_SOURCE_ADAM': True}}, rng='POISON_SOURCE_RNG'), ego_source)
    if task == 'gt_external':
        gt, _, _, _ = gt_fixture()
        c = {**c, 'model': gt.configs(), 'protocol': run.GT_PROTOCOL, 'modes': ['gt']}
    torch.save(dict(protocol=run.EGO_PROTOCOL, contract=c, epoch=1, batch=0, updates=500,
        models={'pred': model.state_dict()}, optimizers={'pred': {'POISON_SOURCE_ADAM': True}}, rng='POISON_SOURCE_RNG'), source)
    if task == 'gt_external':
        torch.save(dict(protocol=run.GT_PROTOCOL, contract=c, epoch=1, batch=0, updates=500,
            models={'gt': gt.state_dict()}, optimizers={'gt': {'POISON_ADAM': True}}, rng='POISON_SOURCE_RNG'), source)
    source_bytes = source.read_bytes()
    monkeypatch.setattr(legacy, 'require_cuda', lambda _: torch.device('cpu'))
    monkeypatch.setattr(legacy, 'load_cache', lambda path: ({}, train if path == str(paths['train-cache']) else dev))
    monkeypatch.setattr(legacy, 'load_manifest', lambda _: ({'parent_keys': keys}, keys[:64], None))
    monkeypatch.setattr(legacy, 'load_runtime_config', lambda *a: None)
    monkeypatch.setattr(legacy, 'make_prepare_config', lambda _: p.pcfg)
    def provider(*args): p.joint = args[-2]; return p
    monkeypatch.setattr(legacy, 'PilotProvider', provider)
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
    monkeypatch.setattr(legacy, 'epoch_groups', lambda records, *a: [[records[i]] for i in range(600)])
    monkeypatch.setattr(legacy, 'prefetched_batches', lambda provider, source, groups, descriptors, **k:
        ([(r, row[1], row[2], row[3]) for r in group] for group in groups))
    monkeypatch.setattr(integration, 'rebuild_pred_backgrounds', causal_background)
    monkeypatch.setattr(engine, 'gt_moving_support_sequence', lambda *a, **k: [])
    monkeypatch.setattr(engine, 'moving_support_masks', lambda _, shape: np.ones((6, *shape), bool))
    if task == 'gt_external':
        class FakePlanner:
            def __init__(self, path): pass
            def prepare(self, requested):
                self.requested = [tuple(k) for k in requested]; self.rows = {k: {} for k in self.requested[:2]}
                self.metadata = dict(requested=len(requested), covered=2, missing_keys=[list(k) for k in self.requested[2:]])
            def poses(self, record, current): return np.stack(row[1]['future_poses'])
        monkeypatch.setattr(run, 'PlannerCache', FakePlanner)
    out = tmp_path/'readonly'
    args = ['--base-checkpoint', str(base), '--dataroot', str(tmp_path), '--resume', str(source), '--ego-checkpoint', str(ego_source),
        '--ccr-history-cache', str(tmp_path), '--ccr-val-history-cache', str(tmp_path), '--out-dir', str(out),
        '--mode', 'pred' if task == 'ego_checks' else 'gt', '--epochs', '1', '--evaluate-only', '--task', task]
    if task == 'gt_external': args += ['--plan-cache', str(tmp_path)]
    args += [v for n, path in paths.items() for v in ('--'+n, str(path))]
    old_restore, old_train = legacy.restore_checkpoint, legacy.train_arm
    assert run.main(threading.Event(), args) == 0
    result = json.loads((out/'results.json').read_text())
    if task == 'ego_checks':
        assert result['evaluation']['windows'] == 512 and len(result['evaluation']['gradient_rows']) == 4
    else:
        assert result['evaluation']['windows'] == 2
        assert set(result['evaluation']['variants']) == {'GT500_external', 'Pred500_internal'}
    assert source.read_bytes() == source_bytes and not (out/'last.pt').exists()
    assert legacy.restore_checkpoint is old_restore and legacy.train_arm is old_train
    counts = (out/'eval_state_update500.json').read_bytes()
    assert run.main(threading.Event(), args+['--diagnostic-resume']) == 0
    assert (out/'eval_state_update500.json').read_bytes() == counts and source.read_bytes() == source_bytes
    assert legacy.restore_checkpoint is old_restore
