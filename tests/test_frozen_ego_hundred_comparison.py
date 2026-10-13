import copy
import json
from pathlib import Path
import threading
import torch
import pytest
from test_full_gt_training import fixture
from test_frozen_gt_scene_ego import small_bank
from tools.scene_ego_head.frozen_gt import training, comparison, run, reuse, engine
from tools.scene_ego_head.model import SceneEgoHead, SceneEgoConfig
from tools.real_motion.surface_ego_ablation_common import tensor_state_fingerprint


def test_hundred_random_snapshots_and_resume_exact(tmp_path, monkeypatch):
    torch.set_num_threads(1); body, _, p, row = fixture()
    values = small_bank(body, row, p.pcfg.grid)
    torch.manual_seed(21); initial = SceneEgoHead(SceneEgoConfig(16, 32, 4, 2))
    c = dict(schedule=dict(epochs=100, batch_size=64, seed=21, lr=.0003), save_epochs=[50, 80, 100])
    full = tmp_path/'full'; full.mkdir()
    result = training.train(copy.deepcopy(initial), values, full, c)
    assert result['updates'] == 100 and result['complete_epochs'] == 100
    assert sorted(p.name for p in full.glob('head_epoch*.pt')) == ['head_epoch100.pt', 'head_epoch50.pt', 'head_epoch80.pt']
    for epoch in (50, 80, 100):
        v = torch.load(full/f'head_epoch{epoch}.pt', weights_only=False)
        assert v['epochs'] == v['updates'] == epoch and v['contract']['schedule']['epochs'] == 100
        assert v['final_epoch_only'] == (epoch == 100)
    stopped = tmp_path/'stopped'; stopped.mkdir(); event = threading.Event()
    original = training.save_resume_checkpoint
    def stop_after_boundary(path, payload):
        original(path, payload)
        if path.name == 'head_last.pt' and payload['epoch'] == 50: event.set()
    monkeypatch.setattr(training, 'save_resume_checkpoint', stop_after_boundary)
    with pytest.raises(InterruptedError): training.train(copy.deepcopy(initial), values, stopped, c, stop_event=event)
    monkeypatch.setattr(training, 'save_resume_checkpoint', original); event.clear()
    training.train(copy.deepcopy(initial), values, stopped, c, resume=True)
    for epoch in (50, 80, 100):
        a = torch.load(full/f'head_epoch{epoch}.pt', weights_only=False)
        b = torch.load(stopped/f'head_epoch{epoch}.pt', weights_only=False)
        assert a['state_sha256'] == b['state_sha256']


def test_real_cli_four_frozen_heads_same_dev512(tmp_path, monkeypatch):
    from test_frozen_ego_history_reuse import test_all_val_cli_reuses_tokens_fresh_head_resume_and_old_files_unchanged
    test_all_val_cli_reuses_tokens_fresh_head_resume_and_old_files_unchanged(tmp_path, monkeypatch)
    baseline = tmp_path/'new10/head_epoch10.pt'
    v = torch.load(baseline, weights_only=False)
    # Simulated TRAIN population count; actual snapshot tensors are genuine.
    v['updates'] = 3200; torch.save(v, baseline)
    baseline_bytes = baseline.read_bytes()
    body, _, p, row = fixture(); values = small_bank(body, row, p.pcfg.grid)
    def imported(directory, c, **kwargs):
        assert c['schedule']['epochs'] == 100 and c['save_epochs'] == [50, 80, 100]
        return values, dict(tensor_mib=1., windows=6, fresh_windows=0)
    monkeypatch.setattr(reuse, 'load_readonly', imported)
    def trained(head, values, directory, c, **kwargs):
        assert c['schedule']['epochs'] == 100
        for epoch in (50, 80, 100):
            with torch.no_grad(): head.readout[-1].bias.fill_(epoch*.0001)
            torch.save(dict(protocol=training.PROTOCOL, contract=c, head=head.state_dict(), epochs=epoch,
                updates=epoch*320, final_epoch_only=epoch == 100,
                state_sha256=tensor_state_fingerprint(head.state_dict())), directory/f'head_epoch{epoch}.pt')
        return dict(complete_epochs=100, updates=32000, train_windows=20430, report={})
    monkeypatch.setattr(training, 'train', trained)
    calls = []
    def evaluated(body, head, provider, source, rows, **kwargs):
        keys = [[r[0]['scene_name'], r[0]['t0_token']] for group in rows for r in group]
        assert len(keys) == 512 and keys[-1] == ['DEV', 'dev511']
        calls.append(keys)
        state = dict(keys=keys)
        kwargs['save'](state)
        if kwargs['saved']: assert kwargs['saved'] == state
        m = dict(mIoU=20., IoU=30., MovingMicro=22., per_horizon={})
        t = dict(ADE_m=.8, FDE_3s_m=1.7, yaw_mean_deg=[0]*6)
        return dict(windows=512, population_keys=keys,
            variants={'internal': dict(transport=m, semantic_refine=m)}, trajectory={'head': t})
    monkeypatch.setattr(engine, 'evaluate', evaluated)
    source = tmp_path/'pilot/epoch_0002.pt'; source_bytes = source.read_bytes(); out = tmp_path/'new100'
    args = [v for n, path in (('config', tmp_path/'config.yaml'), ('base-checkpoint', tmp_path/'base.pt'),
        ('train-cache', tmp_path/'train-cache'), ('dev-cache', tmp_path/'dev-cache'),
        ('population-manifest', tmp_path/'population-manifest'), ('dataroot', tmp_path),
        ('train-info', tmp_path/'train-info'), ('dev-info', tmp_path/'dev-info'),
        ('ccr-history-cache', tmp_path), ('ccr-val-history-cache', tmp_path), ('out-dir', out),
        ('resume', source), ('reuse-stage-dir', tmp_path/'old'), ('comparison-head10', baseline))
        for v in ('--'+n, str(path))]
    args += ['--head-epochs', '100', '--head-pool-side', '2', '--save-head-epochs', '50', '80', '100']
    assert run.main(argv=args) == 0
    report = json.loads((out/'head_comparison.json').read_text())
    assert report['status'] == 'complete' and list(report['reports']) == ['10', '50', '80', '100']
    assert len(calls) == 4 and all(k == calls[0] for k in calls)
    assert run.main(argv=args+['--resume-stage']) == 0
    assert baseline.read_bytes() == baseline_bytes and source.read_bytes() == source_bytes
    assert not (out/'last.pt').exists() and all((out/f'dev512_head{n}_state.json').is_file() for n in (10,50,80,100))
    bad = copy.deepcopy(v); bad['epochs'] = 9
    torch.save(bad, tmp_path/'bad_head.pt')
    with pytest.raises(RuntimeError, match='differs'):
        comparison.checked_head(tmp_path/'bad_head.pt', v['contract'], 10)
