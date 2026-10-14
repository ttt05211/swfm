"""Small real geometry/Adam tests; NOT real STC quality or L40S throughput."""
import copy
import json
import random
import threading
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from test_full_gt_training import fixture
from tools.perception_robustness import augmentation as aug, data, evaluation, run, workflow
from tools.shared_history_experiments.full_gt import engine as gt
from tools.joint_occ_experiments import run as legacy


def real_fixture(device='cpu'):
    model, adam, p, row = fixture()
    from real_motion.strong_w2det import StrongW2DetConfig
    p.strong = StrongW2DetConfig(free_label=17)
    if device != 'cpu':
        model.to(device); p.device = torch.device(device)
        adam = torch.optim.AdamW([dict(params=model.parameters(), lr=1e-3, initial_lr=1e-3)])
    return model, adam, p, row


def history_fixture():
    # Use a larger occupancy grid only for pure NumPy augmentation tests.
    from real_motion.geometry import OccupancyGrid
    grid = OccupancyGrid(x_min=-16, y_min=-16, z_min=-2, voxel_size=(1., 1., 1.), shape_hwd=(32, 32, 8))
    sem = np.full((4, 32, 32, 8), 17, np.uint8)
    sem[:, :, :, 2] = 11; sem[:, 14:18, 14:18, 3:5] = 4
    known = np.ones_like(sem, bool); known[:, 0] = False
    poses = np.repeat(np.eye(4)[None], 4, axis=0)
    raw = dict(history_occ=sem, history_observed=known, history_poses=poses,
               future_gt_occ=object(), future_poses=object(), _column_causal_preparation=object())
    return raw, dict(scene_name='TRAIN', t0_token='token'), grid


@pytest.mark.parametrize('family', ['dropout', 'semantic_false_occupancy', 'geometry'])
def test_corruption_is_causal_deterministic_bounded_and_known_preserved(family):
    raw, record, grid = history_fixture(); original = copy.deepcopy(raw['history_occ'])
    np_before = np.random.get_state(); py_before = random.getstate(); torch_before = torch.get_rng_state().clone()
    a, audit = aug.corrupt(raw, record, 0, 21, grid, force_family=family)
    b, other = aug.corrupt(raw, record, 0, 21, grid, force_family=family)
    np.testing.assert_array_equal(a['history_occ'], b['history_occ']); assert audit == other
    assert set(a) == {'history_occ', 'history_observed', 'history_poses'}
    np.testing.assert_array_equal(a['history_observed'], raw['history_observed'])
    np.testing.assert_array_equal(raw['history_occ'], original)
    assert not np.shares_memory(a['history_occ'], raw['history_occ'])
    assert sum(audit['changed_voxels']) > 0 and not audit['clean_geometry_reusable']
    occupied = ((original < 17) & raw['history_observed']).sum((1, 2, 3))
    assert np.all(np.asarray(audit['changed_voxels']) <= np.maximum(1, np.floor(.15*occupied)))
    assert a['history_occ'].min() >= 0 and a['history_occ'].max() <= 17
    np.testing.assert_array_equal(a['history_occ'][~raw['history_observed']], original[~raw['history_observed']])
    assert py_before == random.getstate() and torch.equal(torch_before, torch.get_rng_state())
    now = np.random.get_state(); assert np_before[0] == now[0]
    np.testing.assert_array_equal(np_before[1], now[1]); assert np_before[2:] == now[2:]
    # One stable box is same in world even when ego axes rotate.
    pose = np.eye(4); angle = .7
    pose[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    at = aug.patch_cells(np.array([0., 0., .5]), np.array([4., 2., 2.]), pose, pose, grid)
    assert len(at) > 0


def test_zero_probability_invalid_and_epoch_rng():
    raw, record, grid = history_fixture()
    result, audit = aug.corrupt(raw, record, 0, 21, grid, recipe=aug.Recipe(probability=0))
    np.testing.assert_array_equal(result['history_occ'], raw['history_occ'])
    assert not audit['selected'] and audit['clean_geometry_reusable']
    assert aug.local_rng(record, 0, 21).random() != aug.local_rng(record, 1, 21).random()
    with pytest.raises(ValueError): aug.corrupt(raw, record, 0, 21, grid, force_family='bad')
    with pytest.raises(ValueError): aug.corrupt({**raw, 'history_observed': np.zeros((1,), bool)}, record, 0, 21, grid)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_changed_inputs_cannot_reuse_clean_source_cache_real_geometry_and_backward(monkeypatch, device):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    model, adam, p, original = real_fixture(device); torch.set_num_threads(1)
    historical = {k: copy.deepcopy(original[1][k]) for k in ('history_occ', 'history_observed', 'history_poses')}
    historical['history_occ'][:, 2:4, 2:4, 1:3] = 4
    historical['history_observed'][:] = True
    first = data.historical_row(p, original[0], historical, original[2][0])
    assert len(first[0]['source_class_id']) > 0
    fresh_raw = {**first[1], 'future_poses': original[1]['future_poses'], 'future_gt_occ': original[1]['future_gt_occ']}
    # Clean geometry is intentionally unusable; B MUST reconstruct from semantics.
    fresh_raw['_column_causal_preparation'] = object()
    row = (first[0], fresh_raw, first[2], object())
    def changed(raw, *args):
        value = {k: copy.deepcopy(raw[k]) for k in ('history_occ', 'history_observed', 'history_poses')}
        value['history_occ'][value['history_occ'] == 4] = 17
        return value, dict(selected=True, family='dropout', changed_voxels=[8]*4, clean_geometry_reusable=False)
    monkeypatch.setattr(data, 'corrupt', changed)
    changed_row = data.augment_row(p, row, 0, 21)
    assert not len(changed_row[0]['source_class_id'])
    assert len(row[0]['source_class_id']) > 0 and isinstance(row[1]['_column_causal_preparation'], object)
    assert changed_row[1]['future_gt_occ'] is row[1]['future_gt_occ']
    ready = data.finish_backgrounds(p, changed_row)
    assert len(ready[1]['_column_causal_preparation']['prepared_state']['column_backgrounds']) == 6
    result = gt.train_arm(model, adam, p, [ready], 'gt', update=1, total_updates=4)
    assert result['CE'] > 0 and not result['auxiliary_computed']
    clean_finished = data.finish_backgrounds(p, original)
    assert clean_finished is original


def test_prefetch_train_only_epoch_and_resume(monkeypatch):
    source = object(); p = SimpleNamespace(ccr_history_cache_source=source)
    row = ({}, {'_joint_prepare_timing': {'input_wait': .1}}, (), ())
    calls = []
    monkeypatch.setattr(data, 'augment_row', lambda p, row, epoch, seed: calls.append(epoch) or row)
    base = lambda *a, **k: iter([[row]])
    wrapped = data.prefetch_adapter(base, 'robust', 21, initial_epoch=1)
    list(wrapped(p, object(), None, None)); assert calls == []
    list(wrapped(p, source, None, None)); list(wrapped(p, source, None, None))
    assert calls == [1, 2]


def test_real_occ_stc_eval_and_integer_resume(tmp_path, monkeypatch):
    model, _, p, row = real_fixture(); torch.set_num_threads(1)
    # Correct identity-addressed synthetic STC; no camera/lidar masks in inputs.
    for token in row[0]['history_tokens'][-4:]:
        path = tmp_path/row[0]['scene_name']/token/'labels.npz'; path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, semantics=row[1]['history_occ'][0])
    monkeypatch.setattr(gt, 'gt_moving_support_sequence', lambda *a, **k: [])
    monkeypatch.setattr(gt, 'moving_support_masks', lambda _, shape: np.ones((6, *shape), bool))
    before = copy.deepcopy(model.state_dict()); states = []
    evaluate = evaluation.evaluator(tmp_path)
    report = evaluate({'gt': model}, p, SimpleNamespace(nusc=None), [[row]], save=lambda s: states.append(copy.deepcopy(s)))
    assert report['windows'] == 1 and set(report['variants']['gt']) == set(evaluation.ROUTES)
    saved = copy.deepcopy(states[-1]); resume = []
    evaluate({'gt': model}, p, SimpleNamespace(nusc=None), [[row], [row]], saved=saved,
             save=lambda s: resume.append(copy.deepcopy(s)))
    full = []
    evaluate({'gt': model}, p, SimpleNamespace(nusc=None), [[row], [row]], save=lambda s: full.append(copy.deepcopy(s)))
    assert resume[-1] == full[-1] and saved == states[-1]
    for k, v in before.items(): torch.testing.assert_close(model.state_dict()[k], v, rtol=0, atol=0)
    bad = copy.deepcopy(saved); bad['keys'][0][1] = 'other'
    with pytest.raises(RuntimeError, match='prefix'): evaluate({'gt': model}, p, None, [[row]], saved=bad)


def test_workflow_two_arms_fixed_budget_resume_and_no_promotion(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow, 'stc_inventory', lambda _: [])
    paths = {n: tmp_path/n for n in ('config', 'train-cache', 'dev-cache', 'population-manifest',
                                      'base-checkpoint', 'train-info', 'dev-info')}
    for path in paths.values(): path.write_text('synthetic metadata')
    out = tmp_path/'experiment'
    args = ['--dataroot', str(tmp_path), '--ccr-history-cache', str(tmp_path), '--ccr-val-history-cache', str(tmp_path),
            '--stc-root', str(tmp_path), '--out-dir', str(out)]
    args += [v for k, path in paths.items() for v in ('--'+k, str(path))]
    calls = []
    def child(command, check):
        calls.append(command); directory = __import__('pathlib').Path(command[command.index('--out-dir')+1])
        directory.mkdir(exist_ok=True)
        contract = dict(protocol=run.PROTOCOL, total_updates=2060, implementation=command[command.index('--augmentation')+1])
        m = dict(mIoU=20., IoU=30., MovingMicro=10.)
        report = dict(status='complete', epoch=2, updates=2060, contract=contract,
                      evaluation=dict(windows=64, variants={'gt': {n: m for n in evaluation.ROUTES}}))
        (directory/'results.json').write_text(json.dumps(report))
        return SimpleNamespace(returncode=0)
    assert workflow.main(argv=args, launch=child) == 0 and len(calls) == 2
    assert [c[c.index('--augmentation')+1] for c in calls] == ['clean', 'robust']
    assert all(c[c.index('--epochs')+1] == '2' for c in calls)
    assert workflow.main(argv=args+['--resume-workflow'], launch=child) == 0 and len(calls) == 2
    result = json.loads((out/'comparison.json').read_text()); assert result['no_automatic_promotion']
    with pytest.raises(RuntimeError, match='NEW output'): workflow.main(argv=args, launch=child)
    with pytest.raises(ValueError): workflow.main(argv=args+['--epochs', '3'], launch=child)


@pytest.mark.parametrize('arm', ['clean', 'robust'])
def test_real_cli_epoch_checkpoint_resume_and_scoped_bindings(tmp_path, monkeypatch, arm):
    from dataclasses import asdict
    model, _, p, row = real_fixture()
    paths = {name: tmp_path/name for name in ('config', 'train-cache', 'dev-cache', 'population-manifest',
                                             'base-checkpoint', 'train-info', 'dev-info')}
    for path in paths.values(): path.write_text('metadata')
    torch.save({'model_config': asdict(model.transport.config)}, paths['base-checkpoint'])
    train = [{**row[0], 't0_token': f'train{i}'} for i in range(20430)]
    dev = [{**row[0], 'scene_name': 'DEV', 't0_token': f'dev{i}'} for i in range(4369)]
    keys = [('DEV', r['t0_token']) for r in dev[:512]]
    stc = tmp_path/'stc'
    for token in row[0]['history_tokens']:
        path = stc/'DEV'/token/'labels.npz'; path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, semantics=row[1]['history_occ'][0])
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
        for group in groups: yield [(record, row[1], row[2], row[3]) for record in group]
    monkeypatch.setattr(legacy, 'prefetched_batches', prefetch)
    body_class = run.GTBody
    monkeypatch.setattr(run, 'GTBody', lambda cfg: body_class(cfg, feature_width=16, z_bins=4,
                                              memory_width=4, width=8, chunk_size=32))
    # Augmentation rebuilding itself is tested above with actual geometry. This
    # CLI fixture tests original Adam/RNG/cursor/epoch flow with small tensors.
    monkeypatch.setattr(data, 'augment_row', lambda provider, row, epoch, seed: row)
    values = {name: dict(mIoU=20., IoU=30., MovingMicro=10.) for name in evaluation.ROUTES}
    monkeypatch.setattr(run, 'evaluator', lambda root, **kw: lambda *args, **kwargs:
        dict(windows=64, variants={'gt': values}, trajectory={}))
    out = tmp_path/'train'
    args = ['--dataroot', str(tmp_path), '--ccr-history-cache', str(tmp_path), '--ccr-val-history-cache', str(tmp_path),
            '--stc-root', str(stc), '--out-dir', str(out), '--augmentation', arm]
    args += [v for name, path in paths.items() for v in ('--'+name, str(path))]
    original_parser, original_inputs, original_forward = legacy.parser, run.data.ego_inputs, gt.forward_window
    assert run.main(threading.Event(), args+['--max-updates', '1']) == 0
    saved = torch.load(out/'last.pt', weights_only=False)
    assert saved['updates'] == 1 and saved['batch'] == 1 and saved['epoch'] == 0
    assert len(saved['contract']['train_keys']) == 4086 and saved['contract']['epochs'] == 2
    assert saved['contract']['model']['ego']['parameters'] == 0
    assert run.main(threading.Event(), args+['--resume', str(out/'last.pt')]) == 0
    saved = torch.load(out/'last.pt', weights_only=False)
    assert saved['updates'] == 4 and saved['epoch'] == 2 and saved['batch'] == 0
    assert (out/'epoch_0001.pt').is_file() and (out/'epoch_0002.pt').is_file()
    assert legacy.parser is original_parser and run.data.ego_inputs is original_inputs and gt.forward_window is original_forward
    saved['protocol'] = 'old_experiment'; torch.save(saved, out/'wrong.pt')
    with pytest.raises(RuntimeError, match='differs'): run.main(argv=args+['--resume', str(out/'wrong.pt')])
    assert legacy.parser is original_parser and run.data.ego_inputs is original_inputs


def test_eight_update_smoke_covers_both_then_same_workflow_resumes(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow, 'stc_inventory', lambda _: [])
    paths = {n: tmp_path/n for n in ('config', 'train-cache', 'dev-cache', 'population-manifest',
                                      'base-checkpoint', 'train-info', 'dev-info')}
    for path in paths.values(): path.write_text('metadata')
    out = tmp_path/'experiment'
    args = ['--dataroot', str(tmp_path), '--ccr-history-cache', str(tmp_path), '--ccr-val-history-cache', str(tmp_path),
            '--stc-root', str(tmp_path), '--out-dir', str(out)]
    args += [v for k, path in paths.items() for v in ('--'+k, str(path))]
    calls = []
    def child(command, check):
        calls.append(command); directory = __import__('pathlib').Path(command[command.index('--out-dir')+1])
        directory.mkdir(exist_ok=True)
        if '--max-updates' in command:
            (directory/'last.pt').write_bytes(b'synthetic cursor')
            result = dict(status='stopped_at_complete_update')
        else:
            assert '--resume' in command
            m = dict(mIoU=20., IoU=30., MovingMicro=10.)
            result = dict(status='complete', epoch=2, updates=2060,
                contract=dict(protocol=run.PROTOCOL, total_updates=2060, implementation=directory.name),
                evaluation=dict(windows=64, variants={'gt': {n: m for n in evaluation.ROUTES}}))
        (directory/'results.json').write_text(json.dumps(result))
        return SimpleNamespace(returncode=0)
    assert workflow.main(argv=args+['--max-updates', '8'], launch=child) == 0
    assert len(calls) == 2
    assert workflow.main(argv=args+['--resume-workflow'], launch=child) == 0
    assert len(calls) == 4 and all('--resume' in c for c in calls[2:])
