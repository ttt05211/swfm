"""One new training arm; epoch1/2 eval and old baseline read-only reuse."""
import copy
import json
import threading
from types import SimpleNamespace
import pytest
import torch
from tools.shared_history_experiments.actions.validation import workflow


def simulation(tmp_path, monkeypatch):
    paths = {n: tmp_path/n for n in ('config', 'train-cache', 'dev-cache', 'population-manifest',
        'base-checkpoint', 'train-info', 'dev-info')}
    for path in paths.values(): path.write_text('fixture')
    baseline, root = tmp_path/'shared_20pct_2ep_old', tmp_path/'experiment'
    baseline.mkdir(); torch.save(dict(epoch=1, updates=2), baseline/'epoch_0001.pt')
    args = [x for n, path in paths.items() for x in ('--'+n, str(path))]
    args += ['--dataroot', str(tmp_path), '--ccr-history-cache', str(tmp_path),
        '--ccr-val-history-cache', str(tmp_path), '--out-dir', str(root), '--epochs', '2', '--mode', 'gt']
    records = [{'scene_name': 'TRAIN', 't0_token': str(i)} for i in range(20430)]
    keys = [('DEV', str(i)) for i in range(512)]
    monkeypatch.setattr(workflow.legacy, 'load_cache', lambda _: ({}, records))
    monkeypatch.setattr(workflow.legacy, 'load_manifest', lambda _: ({'parent_keys': keys}, keys[:64], None))
    monkeypatch.setattr(workflow.legacy, 'epoch_groups', lambda *a: [['batch1'], ['batch2']])
    selected = workflow.legacy.select_scene_balanced_round_robin(workflow.legacy.record_keys(records), 4086)
    contract = {**{k: [] for k in workflow.CHECKS}, 'epochs': 2, 'protocol': workflow.BASE_PROTOCOL,
        'seed': 20261011, 'window_batch': 4, 'source_budget': 128, 'total_updates': 4,
        'train_keys': [list(k) for k in selected], 'eval_keys': [list(k) for k in keys],
        'lr': [5e-4, 3e-4, 3e-4], 'weights': [1., 1., 1.],
        'data': {n.replace('-', '_'): workflow.legacy.sha256(path) for n, path in paths.items()
                 if n in ('train-cache', 'dev-cache', 'train-info', 'dev-info', 'base-checkpoint')}}
    def report(epoch, old):
        metric = lambda score: dict(mIoU=score, IoU=score+10, MovingMicro=score+2)
        return dict(status='complete' if epoch == 2 else 'interim_read_only_evaluation', epoch=epoch,
            updates=epoch*2, contract=copy.deepcopy(contract), evaluation=dict(windows=512, variants={'gt': {
                'soft_transport': metric(30), 'soft_add_only': metric(31), 'soft_add_remove': metric(31 if old else 32)}}))
    (baseline/'results.json').write_text(json.dumps(report(2, True)))
    calls = []
    def launch(command, **kwargs):
        def value(flag): return command[command.index(flag)+1]
        module, out = command[command.index('-m')+1], workflow.Path(value('--out-dir'))
        old = module == 'tools.shared_history_experiments.run'
        evaluation = '--evaluate-only' in command
        assert not old or evaluation  # NO duplicate ADD-only training
        limit = int(value('--max-updates')) if '--max-updates' in command else None
        calls.append((old, evaluation, limit, '--resume' in command))
        assert value('--epochs') == '2' and value('--eval-population') == 'dev512'
        out.mkdir(parents=True, exist_ok=True)
        epoch = 1 if evaluation or limit else 2
        if not evaluation:
            torch.save(dict(updates=epoch*2, epoch=epoch), out/'last.pt')
            torch.save(dict(updates=epoch*2, epoch=epoch), out/f'epoch_{epoch:04d}.pt')
        result = report(epoch, old)
        if not old: result['contract']['protocol'] = 'NEW_REMOVE'
        if not evaluation and limit: result['status'] = 'stopped_at_complete_update'; result.pop('evaluation')
        (out/'results.json').write_text(json.dumps(result))
        return SimpleNamespace(returncode=0)
    return root, baseline, args, calls, launch


def test_single_new_arm_epoch1_eval_then_resume_epoch2_reuses_final_baseline(tmp_path, monkeypatch):
    root, baseline, args, calls, launch = simulation(tmp_path, monkeypatch)
    before = {p.name: p.read_bytes() for p in baseline.iterdir()}
    assert workflow.main(argv=args, launch=launch) == 0
    assert calls == [(False, False, 2, False), (False, True, None, True),
                     (True, True, None, True), (False, False, None, True)]
    result = workflow.load_json(root/'comparison.json')
    assert result['new_training_arms'] == ['add_remove']
    assert result['status'] == 'complete' and [r['epoch'] for r in result['epochs']] == [1, 2]
    assert result['epochs'][0]['same_model_REMOVE_effect']['mIoU'] == 1
    assert workflow.main(argv=args+['--resume-workflow'], launch=launch) == 0
    assert len(calls) == 4  # no duplicate training/eval
    assert before == {p.name: p.read_bytes() for p in baseline.iterdir()}
    with pytest.raises(RuntimeError, match='NEW output'): workflow.main(argv=args, launch=launch)


def test_interrupt_epoch_boundary_then_pending_eval_resume_no_restart(tmp_path, monkeypatch):
    root, baseline, args, calls, launch = simulation(tmp_path, monkeypatch)
    event = threading.Event()
    def stop_first(command, **kwargs):
        result = launch(command, **kwargs); event.set(); return result
    assert workflow.main(event, args, launch=stop_first) == 0
    saved = (root/'add_remove'/'epoch_0001.pt').read_bytes()
    assert len(calls) == 1
    assert workflow.main(argv=args+['--resume-workflow'], launch=launch) == 0
    assert calls[1] == (False, True, None, True)
    assert (root/'add_remove'/'epoch_0001.pt').read_bytes() == saved


def test_partial_epoch_eval_resume_keeps_training_last_readonly(tmp_path, monkeypatch):
    root, _, args, calls, launch = simulation(tmp_path, monkeypatch)
    event = threading.Event()
    def partial_eval(command, **kwargs):
        result = launch(command, **kwargs)
        if '--evaluate-only' in command:
            out = workflow.Path(command[command.index('--out-dir')+1])
            data = workflow.load_json(out/'results.json'); data['status'] = 'stopped_at_complete_eval_window'
            (out/'results.json').write_text(json.dumps(data)); event.set()
        return result
    workflow.main(event, args, launch=partial_eval)
    last = (root/'add_remove'/'last.pt').read_bytes()
    def check_eval(command, **kwargs):
        if '--evaluate-only' in command:
            assert (root/'add_remove'/'last.pt').read_bytes() == last
        return launch(command, **kwargs)
    assert workflow.main(argv=args+['--resume-workflow'], launch=check_eval) == 0
    assert calls[:3] == [(False, False, 2, False), (False, True, None, True), (False, True, None, True)]


def test_wrong_old_baseline_population_rejected_before_training(tmp_path, monkeypatch):
    root, baseline, args, calls, launch = simulation(tmp_path, monkeypatch)
    path = baseline/'results.json'; data = workflow.load_json(path)
    data['contract']['train_keys'] = [['wrong', 'token']]
    path.write_text(json.dumps(data))
    with pytest.raises(RuntimeError, match='no new training'): workflow.main(argv=args, launch=launch)
    assert calls == []


def test_comparison_rejects_different_population_and_non512(tmp_path, monkeypatch):
    root, baseline, args, _, launch = simulation(tmp_path, monkeypatch)
    workflow.main(argv=args, launch=launch)
    path = root/'add_remove'/'eval_epoch1'/'results.json'; data = workflow.load_json(path)
    wrong = copy.deepcopy(data); wrong['contract']['eval_keys'] = []
    path.write_text(json.dumps(wrong))
    with pytest.raises(RuntimeError, match='contracts differ'): workflow.build_comparison(root, baseline)
    data['evaluation']['windows'] = 64; path.write_text(json.dumps(data))
    with pytest.raises(RuntimeError, match='DEV512'): workflow.build_comparison(root, baseline)


def test_parent_only_stop_forwarded_once_to_child(monkeypatch):
    event = threading.Event(); event.set()
    calls = []
    class Child:
        ticks = 0
        def wait(self, timeout):
            self.ticks += 1
            if self.ticks <= 2: raise workflow.subprocess.TimeoutExpired('child', timeout)
            return 0
        def send_signal(self, sig): calls.append(sig)
    monkeypatch.setattr(workflow.subprocess, 'Popen', lambda _: Child())
    assert workflow.launch_child(['unused'], event).returncode == 0
    assert calls == [workflow.signal.SIGINT]
