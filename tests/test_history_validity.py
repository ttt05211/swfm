"""Real small-network policy/gradient/CLI tests, not nuScenes quality tests."""
import copy
import json
import threading
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from test_perception_robustness import real_fixture
from tools.history_validity import data, evaluation, run, workflow
from tools.joint_occ_experiments import run as legacy
from tools.shared_history_experiments.full_gt import engine as gt
from tools.shared_history_experiments.refine.history import ResidualHistoryInputs


def masked_row(provider, row):
    raw = dict(row[1], history_observed=np.array(row[1]['history_observed'], copy=True))
    raw['history_observed'][:, :4] = False
    return row[0], raw, row[2], ResidualHistoryInputs(provider, ram_mib=0).build(raw)


def test_policy_changes_only_known_and_preserves_true_invalid():
    _, _, p, row = real_fixture(); row = masked_row(p, row)
    raw = dict(row[1], history_occ=np.array(row[1]['history_occ'], copy=True))
    raw['history_occ'][0, 0, 0, 0] = 255
    raw['future_gt_occ'] = object(); raw['future_poses'] = object()
    original = np.array(raw['history_observed'], copy=True)
    assert data.policy_raw(raw, 'lidar_known') is raw
    dense = data.policy_raw(raw, 'dense_valid')
    assert dense['history_occ'] is raw['history_occ'] and dense['future_gt_occ'] is raw['future_gt_occ']
    assert dense['future_poses'] is raw['future_poses']
    assert dense['_column_causal_preparation'] is raw['_column_causal_preparation']
    assert not dense['history_observed'][0, 0, 0, 0] and dense['history_observed'][1:].all()
    np.testing.assert_array_equal(raw['history_observed'], original)
    with pytest.raises(ValueError): data.policy_raw(raw, 'bad')
    with pytest.raises(ValueError): data.policy_raw(dict(raw, history_observed=original.astype(np.uint8)), 'dense_valid')


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_actual_history_coverage_prior_and_training(device):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    model, adam, p, row = real_fixture(device); torch.set_num_threads(1)
    row = masked_row(p, row)
    reader = data.history_inputs('dense_valid')(p, ram_mib=1)
    dense_history = reader.get(row[0], row[1])
    dense = row[0], data.policy_raw(row[1], 'dense_valid'), row[2], dense_history
    assert dense_history is reader.get(row[0], row[1])
    assert dense_history[0]['state']['current'] is row[3][0]['state']['current']
    assert dense_history[0]['registrations'] is row[3][0]['registrations']
    assert dense_history[0]['memory_reads']['static_valid'].sum() > row[3][0]['memory_reads']['static_valid'].sum()
    # Future targets/poses MUST NOT be read while building historical coverage.
    poisoned = dict(row[1], future_gt_occ=object(), future_poses=object())
    rebuilt = reader.build(poisoned)
    np.testing.assert_array_equal(rebuilt[0]['memory_reads']['static_valid'], dense_history[0]['memory_reads']['static_valid'])
    with torch.no_grad():
        a, b = gt.forward_window(model, p, row), gt.forward_window(model, p, dense)
        assert not torch.equal(a['field'], b['field'])
        # Source motion can depend on F even at init; this fixture initializes
        # residual XY/yaw to zero, isolating the unchanged prior construction.
        torch.testing.assert_close(a['prior'], b['prior'], rtol=1e-5, atol=1e-6)
    del a, b
    for update in (1, 2):
        result = gt.train_arm(model, adam, p, [dense], 'gt', update=update, total_updates=4, probe=True)
    assert not result['auxiliary_computed'] and result['CE'] > 0
    assert result['gradient_links']['occupancy_to_shared_history'] > 0
    assert result['gradient_links']['occupancy_to_xy'] > 0
    assert not list(model.ego.parameters())


def test_prefetch_both_train_and_eval_no_source_or_cache_mutation():
    _, _, p, row = real_fixture(); row = masked_row(p, row)
    reader = data.history_inputs('dense_valid')(p, ram_mib=0)
    closed = []
    def base(provider, source, groups, descriptors, **kwargs):
        try: yield [(row[0], row[1], row[2], descriptors.get(row[0], row[1]))]
        finally: closed.append(source)
    wrapped = data.prefetch_adapter(base, 'dense_valid')
    for source in ('TRAIN', 'DEV'):
        result = next(iter(wrapped(p, source, None, reader)))
        assert result[0][1]['history_observed'].all()
        assert result[0][1]['_column_causal_preparation'] is row[1]['_column_causal_preparation']
    assert closed == ['TRAIN', 'DEV'] and not row[1]['history_observed'].all()


def test_real_occ_stc_eval_policy_bound_resume_and_no_weight_updates(tmp_path, monkeypatch):
    model, _, p, row = real_fixture(); torch.set_num_threads(1)
    row = masked_row(p, row)
    for token in row[0]['history_tokens']:
        path = tmp_path/row[0]['scene_name']/token/'labels.npz'; path.parent.mkdir(parents=True, exist_ok=True)
        # Poison released visibility fields: evaluator must ignore them.
        np.savez(path, semantics=row[1]['history_occ'][0], mask_camera=np.zeros(1), mask_lidar=np.zeros(1))
    monkeypatch.setattr(gt, 'gt_moving_support_sequence', lambda *a, **k: [])
    monkeypatch.setattr(gt, 'moving_support_masks', lambda _, shape: np.ones((6, *shape), bool))
    before = copy.deepcopy(model.state_dict()); states = []
    raw = data.policy_raw(row[1], 'dense_valid')
    dense = row[0], raw, row[2], data.history_inputs('dense_valid')(p, ram_mib=0).build(raw)
    evaluate = evaluation.evaluator(tmp_path, 'dense_valid')
    result = evaluate({'gt': model}, p, SimpleNamespace(nusc=None), [[dense]], save=lambda s: states.append(copy.deepcopy(s)))
    assert result['windows'] == 1 and not result['metric_visibility_mask']
    assert result['history_known_policy'] == 'dense_valid' and result['STC_known_policy'] == 'dense_valid'
    assert set(result['variants']['gt']) == set(evaluation.ROUTES)
    saved = copy.deepcopy(states[-1]); continued, full = [], []
    evaluate({'gt': model}, p, SimpleNamespace(nusc=None), [[dense], [dense]], saved=saved,
             save=lambda s: continued.append(copy.deepcopy(s)))
    evaluate({'gt': model}, p, SimpleNamespace(nusc=None), [[dense], [dense]], save=lambda s: full.append(copy.deepcopy(s)))
    assert continued[-1] == full[-1] and saved == states[-1]
    with pytest.raises(RuntimeError, match='policy differs'):
        evaluation.evaluator(tmp_path, 'lidar_known')({'gt': model}, p, None, [], saved=saved)
    for k, v in before.items(): torch.testing.assert_close(model.state_dict()[k], v, rtol=0, atol=0)


def metadata_args(tmp_path):
    paths = {n: tmp_path/n for n in ('config', 'train-cache', 'dev-cache', 'population-manifest',
                                    'base-checkpoint', 'train-info', 'dev-info')}
    for path in paths.values(): path.write_text('fixture')
    args = [v for n, p in paths.items() for v in ('--'+n, str(p))]
    args += ['--dataroot', str(tmp_path), '--ccr-history-cache', str(tmp_path),
             '--ccr-val-history-cache', str(tmp_path), '--stc-root', str(tmp_path)]
    return paths, args


def test_workflow_smoke_resume_complete_and_initializer_guard(tmp_path, monkeypatch):
    _, args = metadata_args(tmp_path); root = tmp_path/'workflow'
    args += ['--out-dir', str(root)]
    monkeypatch.setattr(workflow, 'stc_inventory', lambda a: [])
    calls = []
    def launch(command, check):
        calls.append(command); policy = command[command.index('--known-policy')+1]
        directory = Path(command[command.index('--out-dir')+1]); directory.mkdir(exist_ok=True)
        if '--max-updates' in command:
            (directory/'last.pt').write_bytes(b'safe smoke cursor')
            result = dict(status='stopped_at_complete_update')
        else:
            if (directory/'last.pt').exists(): assert '--resume' in command
            c = dict(protocol=run.PROTOCOL, epochs=1, total_updates=1030, implementation=policy, model='same')
            torch.save(dict(contract=c, updates=1030, epoch=1, batch=0, history_known_policy=policy,
                            random_initializer='same_init'), directory/'last.pt')
            m = dict(mIoU=20., IoU=30., MovingMicro=10.)
            result = dict(status='complete', epoch=1, updates=1030, contract=c,
                evaluation=dict(windows=64, history_known_policy=policy, variants={'gt': {n: m for n in evaluation.ROUTES}}))
        (directory/'results.json').write_text(json.dumps(result))
        return SimpleNamespace(returncode=0)
    assert workflow.main(argv=args+['--max-updates', '8'], launch=launch) == 0 and len(calls) == 2
    assert workflow.main(argv=args+['--resume-workflow'], launch=launch) == 0 and len(calls) == 4
    assert workflow.main(argv=args+['--resume-workflow'], launch=launch) == 0 and len(calls) == 4
    assert json.loads((root/'comparison.json').read_text())['no_automatic_promotion']
    saved = torch.load(root/'dense_valid'/'last.pt', weights_only=False)
    saved['random_initializer'] = 'wrong'; torch.save(saved, root/'dense_valid'/'last.pt')
    with pytest.raises(RuntimeError, match='initializers differ'): workflow.comparison(root)
    with pytest.raises(ValueError): workflow.main(argv=args+['--epochs', '3'], launch=launch)
    direct = list(args); direct[direct.index('--out-dir')+1] = str(tmp_path/'direct')
    assert workflow.main(argv=direct, launch=launch) == 0
    assert all('--max-updates' not in c and '--resume' not in c and c[c.index('--epochs')+1] == '1'
               for c in calls[-2:])


@pytest.mark.parametrize('policy', data.POLICIES)
def test_actual_cli_safe_resume_policy_and_original_binding_guards(tmp_path, monkeypatch, policy):
    model, _, p, row = real_fixture(); row = masked_row(p, row); torch.set_num_threads(1)
    paths, args = metadata_args(tmp_path)
    torch.save({'model_config': asdict(model.transport.config)}, paths['base-checkpoint'])
    train = [{**row[0], 't0_token': f'train{i}'} for i in range(20430)]
    dev = [{**row[0], 'scene_name': 'DEV', 't0_token': f'dev{i}'} for i in range(4369)]
    keys = [('DEV', r['t0_token']) for r in dev[:512]]
    monkeypatch.setattr(run, 'stc_inventory', lambda a: [])
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
        def __init__(self, *args): pass
        def close(self): pass
    monkeypatch.setattr('real_motion.canonical_repair_execution.CanonicalCpuExecution', Disposable)
    monkeypatch.setattr('real_motion.strong_majority_execution.ParallelNativeMajority', Disposable)
    def caches(provider, *args):
        provider.ccr_history_cache = provider.ccr_val_history_cache = SimpleNamespace(stats=lambda: {}, close=lambda: None)
        return {'train': {'namespace': 'train'}, 'val': {'namespace': 'val'}}
    monkeypatch.setattr(legacy, 'attach_caches', caches)
    def groups(records, *args):
        assert len(records) == 4086
        return [[records[0]], [records[1]]]
    monkeypatch.setattr(legacy, 'epoch_groups', groups)
    def prefetch(provider, source, groups, descriptors, **kwargs):
        for group in groups:
            yield [(record, row[1], row[2], descriptors.get(record, row[1])) for record in group]
    monkeypatch.setattr(legacy, 'prefetched_batches', prefetch)
    body = run.GTBody
    monkeypatch.setattr(run, 'GTBody', lambda cfg: body(cfg, feature_width=16, z_bins=4, memory_width=4, width=8, chunk_size=32))
    metrics = {n: dict(mIoU=20., IoU=30., MovingMicro=10.) for n in evaluation.ROUTES}
    monkeypatch.setattr(run, 'evaluator', lambda root, known, **kw: lambda *a, **k:
        dict(windows=64, history_known_policy=known, variants={'gt': metrics}, trajectory={}))
    out = tmp_path/'train'; args += ['--out-dir', str(out), '--known-policy', policy]
    original_parser, original_ego = legacy.parser, run.data.ego_inputs
    assert run.main(threading.Event(), args+['--max-updates', '1']) == 0
    initial = torch.load(out/'last.pt', weights_only=False)
    assert initial['updates'] == 1 and initial['epoch'] == 0 and initial['batch'] == 1
    assert initial['history_known_policy'] == policy
    assert run.main(threading.Event(), args+['--resume', str(out/'last.pt')]) == 0
    saved = torch.load(out/'last.pt', weights_only=False)
    assert saved['updates'] == 2 and saved['epoch'] == 1 and saved['batch'] == 0
    assert saved['contract']['epochs'] == 1 and saved['contract']['total_updates'] == 2
    assert saved['random_initializer'] == initial['random_initializer']
    assert (out/'epoch_0001.pt').is_file() and not (out/'epoch_0002.pt').exists()
    fraction = saved['reports']['training'][0]['gt']['history_known']['fractions'][0]
    assert fraction == (1. if policy == 'dense_valid' else float(row[1]['history_observed'].mean()))
    assert legacy.parser is original_parser and run.data.ego_inputs is original_ego
    old = copy.deepcopy(saved)
    old['protocol'] = 'shared_gt_history_known_train20pct_2epochs_v1'
    old['contract']['epochs'] = 2
    torch.save(old, out/'old_two_epoch.pt')
    with pytest.raises(RuntimeError, match='differs'):
        run.main(argv=args+['--resume', str(out/'old_two_epoch.pt')])
    other = 'lidar_known' if policy == 'dense_valid' else 'dense_valid'
    wrong = [other if v == policy else v for v in args]
    with pytest.raises(RuntimeError, match='differs'):
        run.main(argv=wrong+['--resume', str(out/'last.pt')])
    assert legacy.parser is original_parser and run.data.ego_inputs is original_ego
