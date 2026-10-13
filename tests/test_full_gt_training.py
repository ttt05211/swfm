"""Small real tensor/Adam checks plus simulated full-population workflow."""
import copy
import json
import threading
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from test_shared_semantic_refiner import fixture as old_fixture
from tools.shared_history_experiments.full_gt.model import GTBody, PROTOCOL
from tools.shared_history_experiments.full_gt import engine, workflow
from tools.joint_occ_experiments import run as legacy


def fixture():
    old, _, provider, row = old_fixture()
    model = GTBody(old.transport.config, feature_width=16, z_bins=4,
                   memory_width=4, width=8, chunk_size=32)
    provider.joint = model
    adam = torch.optim.AdamW([dict(params=model.transport.parameters(), lr=1e-3, initial_lr=1e-3),
        dict(params=model.columns.parameters(), lr=1e-3, initial_lr=1e-3),
        dict(params=model.ego.parameters(), lr=1e-3, initial_lr=1e-3)])
    return model, adam, provider, row


def test_no_ego_commands_or_aux_targets_and_occ_gradients():
    torch.set_num_threads(1)
    model, adam, provider, row = fixture()
    assert not list(model.ego.parameters())
    before = copy.deepcopy(model.state_dict())
    row = (row[0], row[1], (row[2][0], object(), object()), row[3])
    record = dict(row[0])
    # No direct source motion/existence target reads; geometric input remains.
    record['future_displacement_xy_m'] = object()
    record['future_yaw_delta_rad'] = object()
    row = (record, *row[1:])
    fw = engine.forward_window(model, provider, row)
    assert 'ego' not in fw and fw['logits'].shape[0] == 6
    for update in (1, 2):
        result = engine.train_arm(model, adam, provider, [row], 'gt',
                                  update=update, total_updates=4, probe=True)
    assert set(result).isdisjoint({'ego_geometry', 'source_geometry', 'existence'})
    assert not result['auxiliary_computed']
    assert result['CE'] > 0 and result['Lovasz'] > 0
    assert result['gradient_links']['occupancy_to_xy'] > 0
    assert result['gradient_links']['occupancy_to_memory'] > 0
    assert not torch.equal(before['columns.readout.4.weight'], model.state_dict()['columns.readout.4.weight'])
    for key in before:
        if 'existence' in key:
            torch.testing.assert_close(before[key], model.state_dict()[key], atol=0, rtol=0)


def test_history_reader_does_not_read_future_cmd_or_pose():
    _, _, _, row = fixture()
    raw = dict(row[1], future_poses=object(), future_gt_occ=object())
    record = dict(row[0], future_tokens=object())
    timestamps = {token: int(i*500000) for i, token in enumerate(record['history_tokens'])}
    source = SimpleNamespace(nusc=SimpleNamespace(get=lambda _, token: {'timestamp': timestamps[token]}))
    e, commands, target = engine.history_inputs(source, record, raw)
    assert np.isfinite(e).all() and commands is None and target is None


def test_eval_integer_resume_no_ego_metrics_or_source_mutation(monkeypatch):
    model, _, provider, row = fixture()
    monkeypatch.setattr(engine, 'gt_moving_support_sequence', lambda *a, **k: [])
    monkeypatch.setattr(engine, 'moving_support_masks', lambda _, shape: np.ones((6, *shape), bool))
    before = copy.deepcopy(model.state_dict())
    states = []
    report = engine.evaluate({'gt': model}, provider, SimpleNamespace(nusc=None), [[row]],
                            save=lambda s: states.append(copy.deepcopy(s)))
    assert report['trajectory'] == {} and 'errors' not in states[-1]
    saved = copy.deepcopy(states[-1]); original = copy.deepcopy(saved)
    resumed, full = [], []
    engine.evaluate({'gt': model}, provider, SimpleNamespace(nusc=None), [[row], [row]], saved=saved,
                    save=lambda s: resumed.append(copy.deepcopy(s)))
    engine.evaluate({'gt': model}, provider, SimpleNamespace(nusc=None), [[row], [row]],
                    save=lambda s: full.append(copy.deepcopy(s)))
    assert resumed[-1] == full[-1] and saved == original
    bad = copy.deepcopy(saved); bad['keys'][0][1] = 'WRONG'
    with pytest.raises(RuntimeError, match='prefix identity'):
        engine.evaluate({'gt': model}, provider, SimpleNamespace(nusc=None), [[row]], saved=bad)
    for k, v in before.items(): torch.testing.assert_close(model.state_dict()[k], v, rtol=0, atol=0)


def test_workflow_full_population_epoch_evaluations_resume(tmp_path, monkeypatch):
    paths = {n: tmp_path/n for n in ('config', 'train-cache', 'dev-cache', 'population-manifest',
        'base-checkpoint', 'train-info', 'dev-info')}
    for path in paths.values(): path.write_text('fixture')
    root = tmp_path/'experiment'
    args = [v for n, p in paths.items() for v in ('--'+n, str(p))]
    args += ['--dataroot', str(tmp_path), '--ccr-history-cache', str(tmp_path),
             '--ccr-val-history-cache', str(tmp_path), '--out-dir', str(root)]
    records = [dict(scene_name='TRAIN', t0_token=str(i)) for i in range(20430)]
    monkeypatch.setattr(legacy, 'load_cache', lambda _: ({}, records))
    def groups(records, *a):
        assert len(records) == 20430
        return [[], []]
    monkeypatch.setattr(legacy, 'epoch_groups', groups)
    calls = []
    def child(command, **_):
        calls.append(command)
        assert command[command.index('--train-fraction')+1] == '1.0'
        assert command[command.index('--epochs')+1] == '2'
        directory = Path(command[command.index('--out-dir')+1])
        directory.mkdir(parents=True, exist_ok=True)
        if '--evaluate-only' in command:
            result(directory, 1)
        else:
            epoch = 1 if '--max-updates' in command else 2
            torch.save(dict(protocol=PROTOCOL, epoch=epoch, updates=epoch*2), directory/'last.pt')
            torch.save({}, directory/f'epoch_{epoch:04d}.pt')
            if epoch == 2: result(directory, 2)
        return SimpleNamespace(returncode=0)
    def result(directory, epoch):
        m = dict(mIoU=40., IoU=52., MovingMicro=28.)
        legacy.write_json(directory/'results.json', dict(status='complete' if epoch == 2 else 'interim_read_only_evaluation',
            contract=dict(protocol=PROTOCOL), epoch=epoch, updates=epoch*2,
            evaluation=dict(windows=512, variants={'gt': {'transport': m, 'semantic_refine': m}})))
    assert workflow.main(argv=args, launch=child) == 0
    assert len(calls) == 3
    assert '--resume' not in calls[0] and '--max-updates' in calls[0]
    assert '--evaluate-only' in calls[1] and '--resume' in calls[2]
    assert json.loads((root/'comparison.json').read_text())['status'] == 'complete'
    assert workflow.main(argv=args+['--resume-workflow'], launch=child) == 0
    assert len(calls) == 3
    with pytest.raises(RuntimeError, match='NEW output'):
        workflow.main(argv=args, launch=child)
    with pytest.raises(ValueError):
        workflow.main(argv=args+['--train-fraction', '.2'], launch=child)


def test_real_cli_full_population_epoch_snapshots_resume_and_eval_readonly(tmp_path, monkeypatch):
    from tools.shared_history_experiments.full_gt import run
    from tools.shared_history_experiments.full_gt.model import PROTOCOL
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
        assert len(records) == 20430
        return [[records[0]], [records[1]]]
    monkeypatch.setattr(legacy, 'epoch_groups', groups)
    monkeypatch.setattr(legacy, 'prefetched_batches', lambda provider, source, groups, descriptors, **k:
        ([(record, row[1], row[2], row[3]) for record in group] for group in groups))
    original_model = run.ResidualJoint
    monkeypatch.setattr(run, 'ResidualJoint', lambda cfg: original_model(cfg, feature_width=16, z_bins=4,
        memory_width=4, width=8, chunk_size=32))
    # Actual renderer/network/CE/Lovasz/backward runs; dataset only is simulated.
    monkeypatch.setattr(engine, 'evaluate', lambda *a, **k: dict(windows=512, variants={}, trajectory={}))
    out = tmp_path/'pilot'
    args = ['--config', str(config), '--base-checkpoint', str(base), '--dataroot', str(tmp_path),
        '--ccr-history-cache', str(tmp_path), '--ccr-val-history-cache', str(tmp_path), '--out-dir', str(out)]
    args += [value for name, path in paths.items() for value in ('--'+name, str(path))]
    assert run.main(threading.Event(), args+['--max-updates', '2']) == 0
    saved = torch.load(out/'last.pt', weights_only=False)
    assert saved['protocol'] == PROTOCOL and saved['epoch'] == 1 and saved['updates'] == 2
    assert saved['contract']['model']['repair']['action_heads'] is False
    assert saved['contract']['model']['ego']['parameters'] == 0
    assert saved['contract']['model']['objective'] == dict(ce=1., lovasz=1., motion=0., existence=0., ego=0.)
    assert len(saved['contract']['train_keys']) == 20430
    assert not any(key.startswith('ego.') for key in saved['models']['gt'])
    assert (out/'epoch_0001.pt').is_file()
    assert run.main(threading.Event(), args+['--resume', str(out/'last.pt')]) == 0
    assert (out/'epoch_0002.pt').is_file()
    last, summary = (out/'last.pt').read_bytes(), (out/'summary.txt').read_bytes()
    evaluation = args.copy(); evaluation[evaluation.index(str(out))] = str(out/'eval')
    assert run.main(threading.Event(), evaluation+['--resume', str(out/'epoch_0001.pt'), '--evaluate-only']) == 0
    assert (out/'last.pt').read_bytes() == last and (out/'summary.txt').read_bytes() == summary
    saved['protocol'] = 'OLD_CCR'
    wrong = out/'old.pt'; torch.save(saved, wrong)
    with pytest.raises(RuntimeError, match='differs'): run.main(threading.Event(), args+['--resume', str(wrong)])
