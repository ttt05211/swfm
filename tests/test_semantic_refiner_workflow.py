import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from tools.joint_occ_experiments import run as legacy
from tools.shared_history_experiments.refine.model import PROTOCOL
from tools.shared_history_experiments.refine.experiment import probe, workflow


def test_probe_gate_and_aggregate(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'max_memory_reserved', lambda _: 200)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda _: 100)
    monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda _: SimpleNamespace(total_memory=1000))
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda _: (800, 1000))
    row = dict(windows=4, seconds=2., CE=.1, Lovasz=.2, source_geometry=.3,
        existence=.1, ego_geometry=.2, grad_norm=1., stages_seconds={'backward': 1.})
    links = dict(occupancy_to_memory=.1, occupancy_to_shared_history=.1, occupancy_to_xy=.1)
    report = probe.aggregate([row]*8, links, torch.device('cuda'), 2, 8)
    assert report['pass_gate'] and report['seconds_per_window'] == .5
    assert report['host_stages_seconds_per_window']['backward'] == .25
    assert not probe.aggregate([row]*7, links, torch.device('cuda'), 2, 8)['pass_gate']
    assert not probe.aggregate([row]*8, {**links, 'occupancy_to_xy': 0.}, torch.device('cuda'), 2, 8)['pass_gate']
    assert not probe.aggregate([{**row, 'CE': float('nan')}]*8, links, torch.device('cuda'), 2, 8)['pass_gate']
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda _: (50, 1000))
    assert not probe.aggregate([row]*8, links, torch.device('cuda'), 2, 8)['pass_gate']


@pytest.mark.parametrize('probe_fail', [False, True])
def test_workflow_probe_epoch1_epoch2_resume_no_repeat(tmp_path, monkeypatch, probe_fail):
    paths = {n: tmp_path/n for n in ('config', 'train-cache', 'dev-cache', 'population-manifest',
        'base-checkpoint', 'train-info', 'dev-info')}
    for path in paths.values(): path.write_text('fixture')
    root = tmp_path/'experiment'
    args = [value for name, path in paths.items() for value in ('--'+name, str(path))]
    args += ['--dataroot', str(tmp_path), '--ccr-history-cache', str(tmp_path),
        '--ccr-val-history-cache', str(tmp_path), '--out-dir', str(root)]
    records = [dict(scene_name='TRAIN', t0_token=str(i)) for i in range(20430)]
    monkeypatch.setattr(legacy, 'load_cache', lambda _: ({}, records))
    monkeypatch.setattr(legacy, 'epoch_groups', lambda *a: [[], []])
    calls = []
    def child(command, **_):
        calls.append(command)
        directory = Path(command[command.index('--out-dir')+1]); directory.mkdir(parents=True, exist_ok=True)
        if command[command.index('-m')+1].endswith('.probe'):
            legacy.write_json(directory/'probe.json', dict(pass_gate=not probe_fail, scientific_updates_published=0))
        elif '--evaluate-only' in command:
            result(directory, 1)
        else:
            epoch = 1 if '--max-updates' in command else 2
            torch.save(dict(protocol=PROTOCOL, epoch=epoch, updates=epoch*2), directory/'last.pt')
            torch.save({}, directory/f'epoch_{epoch:04d}.pt')
            if epoch == 2: result(directory, 2)
        return SimpleNamespace(returncode=0)
    def result(directory, epoch):
        legacy.write_json(directory/'results.json', dict(status='complete' if epoch == 2 else 'interim_read_only_evaluation',
            contract=dict(protocol=PROTOCOL), epoch=epoch, updates=epoch*2,
            evaluation=dict(windows=512, variants={'gt': {'transport': dict(mIoU=39., IoU=51., MovingMicro=27.),
                'semantic_refine': dict(mIoU=40., IoU=52., MovingMicro=28.)}})))
    if probe_fail:
        with pytest.raises(RuntimeError, match='probe FAIL'): workflow.main(argv=args, launch=child)
        assert len(calls) == 1 and not (root/'train').exists()
        return
    assert workflow.main(argv=args+['--probe-only'], launch=child) == 0
    assert len(calls) == 1 and not (root/'train').exists()
    assert workflow.main(argv=args+['--resume-workflow'], launch=child) == 0
    assert len(calls) == 4
    assert '--max-updates' in calls[1] and '--resume' not in calls[1]  # FRESH random after probe
    assert '--evaluate-only' in calls[2] and '--resume' in calls[3]
    assert json.loads((root/'comparison.json').read_text())['status'] == 'complete'
    assert workflow.main(argv=args+['--resume-workflow'], launch=child) == 0
    assert len(calls) == 4  # completed evaluation/training never repeated
    with pytest.raises(RuntimeError, match='NEW output'): workflow.main(argv=args, launch=child)
