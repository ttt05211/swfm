import copy
import json
import threading
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from test_shared_semantic_refiner import fixture
from tools.shared_history_experiments.refine import engine
from tools.shared_history_experiments.loss_controls import run, workflow
from tools.joint_occ_experiments import run as legacy


@pytest.mark.parametrize('arm', list(run.ARMS))
def test_real_training_loss_switches_and_occupancy_motion_links(arm):
    torch.set_num_threads(1)
    model, adam, provider, row = fixture()
    before = copy.deepcopy(model.state_dict())
    auxiliary = engine.source_auxiliary
    first = run.train_control(engine.train_arm, auxiliary, arm, model, adam, provider, [row],
        'gt', update=1, total_updates=500, probe=True)
    result = run.train_control(engine.train_arm, auxiliary, arm, model, adam, provider, [row],
        'gt', update=2, total_updates=500, probe=True)
    assert engine.source_auxiliary is auxiliary
    assert result['CE'] > 0 and result['Lovasz'] > 0 and result['ego_geometry'] >= 0
    assert result['source_geometry'] == (0 if arm == 'no_motion' else result['raw_source_geometry'])
    assert result['existence'] == (0 if arm == 'no_existence' else result['raw_existence'])
    assert result['gradient_links']['occupancy_to_xy'] > 0
    assert result['gradient_links']['occupancy_to_memory'] > 0
    assert not torch.equal(before['columns.readout.4.weight'], model.state_dict()['columns.readout.4.weight'])
    existence_names = [k for k in before if 'existence' in k]
    assert existence_names
    if arm == 'no_existence':
        for name in existence_names:
            torch.testing.assert_close(before[name], model.state_dict()[name], atol=0, rtol=0)
    else:
        assert any(not torch.equal(before[k], model.state_dict()[k]) for k in existence_names)
    assert (first['motion_weight'], first['existence_weight']) == run.ARMS[arm]


def test_fixed_groups_budget():
    groups = [[i] for i in range(1031)]
    assert run.capped_groups(lambda *a: groups, [], 21, 0, 4, 128) == groups[:500]
    with pytest.raises(ValueError): run.capped_groups(lambda *a: groups, [], 21, 1, 4, 128)
    with pytest.raises(RuntimeError): run.capped_groups(lambda *a: [], [], 21, 0, 4, 128)


def test_workflow_three_arms_same_args_stop_resume_skip_completed(tmp_path):
    paths = {n: tmp_path/n for n in ('config', 'train-cache', 'dev-cache', 'population-manifest',
        'base-checkpoint', 'train-info', 'dev-info')}
    for path in paths.values(): path.write_text('fixture')
    root = tmp_path/'experiment'
    args = [value for name, path in paths.items() for value in ('--'+name, str(path))]
    args += ['--dataroot', str(tmp_path), '--ccr-history-cache', str(tmp_path),
        '--ccr-val-history-cache', str(tmp_path), '--out-dir', str(root)]
    calls = []
    def child(command, **_):
        calls.append(command)
        arm = command[command.index('--loss-arm')+1]
        directory = Path(command[command.index('--out-dir')+1]); directory.mkdir(parents=True, exist_ok=True)
        m = dict(mIoU=40., IoU=52., MovingMicro=28.)
        legacy.write_json(directory/'results.json', dict(status='complete', updates=500,
            contract=dict(protocol=run.protocol(arm)), evaluation=dict(windows=512,
            variants={'gt': {'transport': m, 'semantic_refine': m}})))
        return SimpleNamespace(returncode=0)
    assert workflow.main(argv=args, launch=child) == 0
    assert [c[c.index('--loss-arm')+1] for c in calls] == list(run.ARMS)
    assert all('--resume' not in c for c in calls)
    assert workflow.main(argv=args+['--resume-workflow'], launch=child) == 0
    assert len(calls) == 3
    assert json.loads((root/'comparison.json').read_text())['status'] == 'complete'
    with pytest.raises(RuntimeError): workflow.main(argv=args, launch=child)


def test_bad_complete_receipt_not_reused(tmp_path):
    path = tmp_path/'results.json'
    path.write_text(json.dumps(dict(status='complete', updates=499)))
    with pytest.raises(RuntimeError): workflow.complete(path, 'full')


def test_runner_wiring_budget_fingerprint_and_restore(tmp_path, monkeypatch):
    args = [value for name in ('config', 'train-cache', 'dev-cache', 'population-manifest',
        'base-checkpoint', 'dataroot', 'train-info', 'dev-info', 'ccr-history-cache', 'ccr-val-history-cache')
        for value in ('--'+name, str(tmp_path/name))]
    args += ['--out-dir', str(tmp_path/'output'), '--mode', 'gt', '--epochs', '1', '--loss-arm', 'no_motion']
    original = (run.refiner.PROTOCOL, run.refiner.ResidualJoint, run.refiner.engine.train_arm,
                legacy.epoch_groups, legacy.stable_json_fingerprint)
    monkeypatch.setattr(legacy, 'epoch_groups', lambda *a: [[i] for i in range(1031)])
    def child(event, argv):
        assert run.refiner.PROTOCOL == run.protocol('no_motion')
        assert len(legacy.epoch_groups([], 20261011, 0, 4, 128)) == 500
        assert legacy.stable_json_fingerprint({'tools/joint_occ_experiments/run.py': 'test'}) != original[-1]({'tools/joint_occ_experiments/run.py': 'test'})
        return 0
    monkeypatch.setattr(run.refiner, 'main', child)
    assert run.main(argv=args) == 0
    assert (run.refiner.PROTOCOL, run.refiner.ResidualJoint, run.refiner.engine.train_arm) == original[:3]
    assert len(legacy.epoch_groups([], 21, 0, 4, 128)) == 1031
    with pytest.raises(ValueError): run.main(argv=args+['--epochs', '2'])
    with pytest.raises(ValueError): run.main(argv=args+['--lovasz-weight', '0'])


def test_auxiliary_hook_restored_after_failure():
    auxiliary = engine.source_auxiliary
    def fail(*args, **kwargs): raise RuntimeError('failed update')
    with pytest.raises(RuntimeError, match='failed update'):
        run.train_control(fail, auxiliary, 'no_motion')
    assert engine.source_auxiliary is auxiliary


def test_partial_arm_resume_uses_own_checkpoint(tmp_path):
    paths = {n: tmp_path/n for n in ('config', 'train-cache', 'dev-cache', 'population-manifest',
        'base-checkpoint', 'train-info', 'dev-info')}
    for path in paths.values(): path.write_text('fixture')
    root = tmp_path/'experiment'
    args = [value for name, path in paths.items() for value in ('--'+name, str(path))]
    args += ['--dataroot', str(tmp_path), '--ccr-history-cache', str(tmp_path),
        '--ccr-val-history-cache', str(tmp_path), '--out-dir', str(root)]
    event = threading.Event()
    calls = []
    def stop(command, **_):
        calls.append(command)
        directory = root/'full'; directory.mkdir()
        torch.save({'updates': 64}, directory/'last.pt')
        legacy.write_json(directory/'results.json', dict(status='stopped_at_complete_update'))
        event.set()
        return SimpleNamespace(returncode=0)
    assert workflow.main(event, args, launch=stop) == 0
    def finish(command, **_):
        calls.append(command)
        arm = command[command.index('--loss-arm')+1]
        if arm == 'full':
            assert Path(command[command.index('--resume')+1]) == root/'full/last.pt'
        else: assert '--resume' not in command
        directory = root/arm; directory.mkdir(exist_ok=True)
        m = dict(mIoU=40., IoU=52., MovingMicro=28.)
        legacy.write_json(directory/'results.json', dict(status='complete', updates=500,
            contract=dict(protocol=run.protocol(arm)), evaluation=dict(windows=512,
            variants={'gt': {'transport': m, 'semantic_refine': m}})))
        return SimpleNamespace(returncode=0)
    assert workflow.main(argv=args+['--resume-workflow'], launch=finish) == 0
    assert len(calls) == 4
