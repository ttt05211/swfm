"""Small real joint steps and matched pose interventions; not nuScenes scores."""
import copy
import json
import threading
from dataclasses import asdict
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from test_scene_ego_head import refiner_fixture, causal_background
from tools.scene_ego_head import integration
from tools.scene_ego_head.experiment import engine, run
from tools.scene_ego_head.experiment.planner import PlannerCache
from tools.joint_occ_experiments import run as legacy
from real_motion.stc_camera_protocol import PLAN_TAG, PLAN_SHA


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_real_joint_step_no_source_aux_and_restore(monkeypatch, device):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    model, p, row = refiner_fixture(device)
    monkeypatch.setattr(integration, 'rebuild_pred_backgrounds', causal_background)
    adam = torch.optim.AdamW([dict(params=model.parameters(), lr=3e-4, initial_lr=3e-4)])
    before = copy.deepcopy(model.state_dict())
    first = engine.train_arm(model, adam, p, [row], 'pred', update=1, total_updates=500, probe=True)
    assert not first['source_auxiliary_computed']
    assert first['gradient_links']['occupancy_to_scene_ego'] > 0
    # Both scene-ego and semantic residual last layers start at zero. Their
    # upstream paths open after the first real Adam update, not before it.
    assert not torch.equal(before['ego.readout.2.weight'], model.state_dict()['ego.readout.2.weight'])
    assert not torch.equal(before['columns.readout.4.weight'], model.state_dict()['columns.readout.4.weight'])
    saved_model, saved_adam, rng = copy.deepcopy(model.state_dict()), copy.deepcopy(adam.state_dict()), legacy.rng_state()
    second = engine.train_arm(model, adam, p, [row], 'pred', update=2, total_updates=500, probe=True)
    assert second['gradient_links']['occupancy_to_shared_history'] > 0
    assert not torch.equal(before['ego.attention.in_proj_weight'], model.state_dict()['ego.attention.in_proj_weight'])
    if device == 'cpu':
        expected, expected_adam = copy.deepcopy(model.state_dict()), copy.deepcopy(adam.state_dict())
        model.load_state_dict(saved_model); adam.load_state_dict(saved_adam); legacy.restore_rng(rng)
        engine.train_arm(model, adam, p, [row], 'pred', update=2, total_updates=500)
        for key, value in expected.items(): torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
        for key, fields in expected_adam['state'].items():
            for name, value in fields.items(): torch.testing.assert_close(adam.state_dict()['state'][key][name], value, rtol=0, atol=0)


def test_conditioned_same_encoder_source_outputs_and_no_gt_background(monkeypatch):
    model, p, row = refiner_fixture(); model.eval()
    monkeypatch.setattr(integration, 'rebuild_pred_backgrounds', causal_background)
    calls = []; hook = model.transport.history_encoder.register_forward_hook(lambda *a: calls.append(1))
    with torch.no_grad():
        pred = integration.forward_pred_window(model, p, row)
        same = integration.forecast_conditioned(model, p, row, pred, pred['poses'].cpu().numpy())
    hook.remove(); assert len(calls) == 1
    for key in ('prior', 'memory', 'logits'): torch.testing.assert_close(pred[key], same[key], atol=0, rtol=0)
    assert same['output'] is pred['output'] and same['field'] is pred['field']
    assert 'column_backgrounds' not in row[3][0]['state']


def planner_files(tmp_path, row):
    root = tmp_path/'planner'; (root/'records').mkdir(parents=True)
    key = row[0]['scene_name']+'__'+row[0]['t0_token']
    manifest = dict(format='stochocc_transport_cache_v2', come_protocol_setting='camera_pred',
        come_protocol_tag=PLAN_TAG, future_ego='predicted', come_trajectory_sha256=PLAN_SHA, hist_last=4,
        predicted_yaw_semantics='each_future_json_yaw_offset_relative_to_current_pose_non_cumulative', num_windows=1, keys=[key])
    (root/'manifest.json').write_text(json.dumps(manifest))
    np.savez(root/'records'/(key+'.npz'), format='stochocc_transport_cache_v2', protocol_tag=PLAN_TAG,
        scene=row[0]['scene_name'], history_tokens=row[0]['history_tokens'][-4:],
        future_tokens=row[0]['future_tokens'], future_e2g=np.stack(row[1]['future_poses']),
        history_semantic=np.array(['POISON CAMERA']), future_semantic=np.array(['POISON LABEL']))
    return root


def test_planner_selective_read_identity_and_same_population(tmp_path):
    _, _, row = refiner_fixture()
    root = planner_files(tmp_path, row); planner = PlannerCache(root)
    key = (row[0]['scene_name'], row[0]['t0_token'])
    planner.prepare([key, ('missing', 'no_planner')])
    assert planner.metadata['covered'] == 1 and len(planner.metadata['missing_keys']) == 1
    np.testing.assert_array_equal(planner.poses(row[0], row[1]['history_poses'][-1]), row[1]['future_poses'])
    bad = copy.deepcopy(row[0]); bad['future_tokens'] = tuple(reversed(bad['future_tokens']))
    with pytest.raises(ValueError, match='identity'): planner.poses(bad, row[1]['history_poses'][-1])
    changed = np.asarray(row[1]['history_poses'][-1]).copy(); changed[2, 3] += 1
    with pytest.raises(ValueError, match='z/tilt'): planner.poses(row[0], changed)


def test_evaluation_all_three_atomic_resume_and_same_windows(tmp_path, monkeypatch):
    model, p, row = refiner_fixture(); model.eval()
    monkeypatch.setattr(integration, 'rebuild_pred_backgrounds', causal_background)
    monkeypatch.setattr(engine, 'gt_moving_support_sequence', lambda *a, **k: [])
    monkeypatch.setattr(engine, 'moving_support_masks', lambda _, shape: np.ones((6, *shape), bool))
    key = (row[0]['scene_name'], row[0]['t0_token'])
    planner = PlannerCache(planner_files(tmp_path, row)); planner.prepare([key, ('missing', 'no_planner')])
    missing = ({**row[0], 'scene_name': 'missing', 't0_token': 'no_planner'}, *row[1:])
    states = []; event = threading.Event(); first_forward = engine.forward_pred_window
    def stopping(*args, **kwargs):
        value = first_forward(*args, **kwargs); event.set(); return value
    monkeypatch.setattr(engine, 'forward_pred_window', stopping)
    with pytest.raises(InterruptedError):
        engine.evaluate({'pred': model}, p, SimpleNamespace(nusc=None), [[row], [missing]], planner=planner,
            stop_event=event, save=lambda s: states.append(copy.deepcopy(s)))
    assert states[-1]['windows'] == states[-1]['seen'] == 1
    assert set(states[-1]['metrics']) == set(engine.ROUTES)
    event.clear(); monkeypatch.setattr(engine, 'forward_pred_window', first_forward)
    resumed = engine.evaluate({'pred': model}, p, SimpleNamespace(nusc=None), [[row], [missing]], planner=planner,
        saved=states[-1], save=lambda s: states.append(copy.deepcopy(s)))
    whole = []
    result = engine.evaluate({'pred': model}, p, SimpleNamespace(nusc=None), [[row], [missing]], planner=planner,
        save=lambda s: whole.append(copy.deepcopy(s)))
    assert resumed['windows'] == result['windows'] == 1
    assert states[-1]['metrics'] == whole[-1]['metrics']
    broken = copy.deepcopy(states[-1]); broken['windows'] = 2
    with pytest.raises(ValueError): engine.restore_metrics(broken, planner.metadata)


@pytest.mark.parametrize('runner_mode', ['planner', 'internal_only'])
def test_actual_cli_safe_partial_resume_own_protocol_and_hook_restore(tmp_path, monkeypatch, runner_mode):
    from tools.scene_ego_head.experiment.internal_only import run as internal_run
    runner = run if runner_mode == 'planner' else internal_run
    model, p, row = refiner_fixture()
    paths = {name: tmp_path/name for name in ('config', 'train-cache', 'dev-cache', 'train-info', 'dev-info', 'population-manifest')}
    for path in paths.values(): path.write_text('fixture')
    base = tmp_path/'base.pt'; torch.save({'model_config': asdict(model.transport.config)}, base)
    train = [{**row[0], 't0_token': f'train{i}'} for i in range(20430)]
    dev = [{**row[0], 'scene_name': 'DEV', 't0_token': f'dev{i}'} for i in range(4369)]
    keys = [('DEV', r['t0_token']) for r in dev[:512]]
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
    def groups(records, *a):
        assert len(records) == 4086
        return [[records[i]] for i in range(600)]
    monkeypatch.setattr(legacy, 'epoch_groups', groups)
    monkeypatch.setattr(legacy, 'prefetched_batches', lambda provider, source, groups, descriptors, **k:
        ([(record, row[1], row[2], row[3]) for record in group] for group in groups))
    original_class = integration.SceneEgoRefiner
    class Tiny(original_class):
        def __init__(self, cfg):
            super().__init__(cfg, feature_width=16, z_bins=4, memory_width=4, width=8, chunk_size=32, ego_width=32, pool_side=2)
    monkeypatch.setattr(runner, 'SceneEgoRefiner', Tiny)
    class FakePlanner:
        def __init__(self, root): self.metadata = {'covered': 512, 'missing_keys': []}
        def prepare(self, requested): self.metadata['requested'] = len(requested)
    if runner_mode == 'planner': monkeypatch.setattr(run, 'PlannerCache', FakePlanner)
    monkeypatch.setattr(integration, 'rebuild_pred_backgrounds', causal_background)
    out = tmp_path/'joint'
    args = ['--base-checkpoint', str(base), '--dataroot', str(tmp_path),
        '--ccr-history-cache', str(tmp_path), '--ccr-val-history-cache', str(tmp_path), '--out-dir', str(out)]
    if runner_mode == 'planner': args += ['--plan-cache', str(tmp_path)]
    args += [value for name, path in paths.items() for value in ('--'+name, str(path))]
    original_train = legacy.train_arm
    assert runner.main(threading.Event(), args+['--max-updates', '2']) == 0
    state = torch.load(out/'last.pt', weights_only=False)
    assert state['updates'] == 2 and state['contract']['total_updates'] == 500
    assert state['contract']['modes'] == ['pred'] and state['protocol'] == runner.PROTOCOL
    if runner_mode == 'internal_only': assert state['contract']['model']['experiment']['planner_used'] is False
    assert legacy.train_arm is original_train
    assert runner.main(threading.Event(), args+['--resume', str(out/'last.pt'), '--max-updates', '3']) == 0
    assert torch.load(out/'last.pt', weights_only=False)['updates'] == 3
    # Read-only interim evaluation must neither train nor write source state.
    source_bytes = (out/'last.pt').read_bytes(); source_summary = (out/'summary.txt').read_bytes()
    fake_eval = lambda *a, **k: dict(windows=512, population=dict(requested=512, missing_keys=[]),
        variants={'internal': {}}, trajectory={}, same_model_internal_minus_external={})
    if runner_mode == 'planner': monkeypatch.setattr(engine, 'evaluate', fake_eval)
    else: monkeypatch.setattr(runner, 'evaluate', fake_eval)
    eval_args = args.copy(); eval_args[eval_args.index(str(out))] = str(tmp_path/'evaluation')
    assert runner.main(threading.Event(), eval_args+['--resume', str(out/'last.pt'), '--evaluate-only']) == 0
    assert (out/'last.pt').read_bytes() == source_bytes and (out/'summary.txt').read_bytes() == source_summary
    bad = copy.deepcopy(state); bad['protocol'] = 'old_gt'
    wrong = out/'wrong.pt'; torch.save(bad, wrong)
    preserved = (out/'last.pt').read_bytes()
    with pytest.raises(RuntimeError, match='differs'): runner.main(threading.Event(), args+['--resume', str(wrong)])
    assert (out/'last.pt').read_bytes() == preserved
    assert legacy.train_arm is original_train


def test_internal_only_no_planner_or_gt_pose_forward_and_eval_resume(monkeypatch):
    from tools.scene_ego_head.experiment.internal_only import run as internal
    from tools.shared_history_experiments.refine import engine as quality
    model, p, row = refiner_fixture()
    monkeypatch.setattr(integration, 'rebuild_pred_backgrounds', causal_background)
    monkeypatch.setattr(integration, 'forecast_conditioned', lambda *a, **k: pytest.fail('pose intervention must NOT run'))
    monkeypatch.setattr(quality, 'gt_moving_support_sequence', lambda *a, **k: [])
    monkeypatch.setattr(quality, 'moving_support_masks', lambda _, shape: np.ones((6, *shape), bool))
    raw = copy.deepcopy(row[1]); raw['future_poses'] = 'POISON GT FUTURE POSES'
    raw['_column_causal_preparation'] = 'POISON GT BACKGROUNDS'
    row = (row[0], raw, row[2], row[3])
    before = quality.forward_window, quality.EVAL_PROTOCOL
    states = []
    result = internal.evaluate({'pred': model}, p, SimpleNamespace(nusc=None), [[row]],
        save=lambda value: states.append(copy.deepcopy(value)))
    assert result['evaluation_routes'] == ['internal'] and not result['planner_used']
    assert set(result['variants']) == {'internal'} and result['windows'] == 1
    assert (quality.forward_window, quality.EVAL_PROTOCOL) == before
    resumed = []; full = []
    internal.evaluate({'pred': model}, p, SimpleNamespace(nusc=None), [[row], [row]], saved=states[-1],
        save=lambda value: resumed.append(copy.deepcopy(value)))
    internal.evaluate({'pred': model}, p, SimpleNamespace(nusc=None), [[row], [row]],
        save=lambda value: full.append(copy.deepcopy(value)))
    assert resumed[-1]['metrics'] == full[-1]['metrics']
    with pytest.raises(RuntimeError, match='differs'):
        internal.evaluate({'pred': model}, p, None, [], saved={**states[-1], 'protocol': 'old_gt'})
    assert (quality.forward_window, quality.EVAL_PROTOCOL) == before
