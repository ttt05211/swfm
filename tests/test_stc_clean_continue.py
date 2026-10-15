"""Continuation must preserve Adam/RNG, skip first500 groups and resume itself."""
import copy
import json
import threading
import numpy as np
import pytest
import torch
from tools.stc_clean_continue import run
from tools.joint_occ_experiments import run as legacy


def test_next500_and_source_selection(tmp_path):
    groups = [[i] for i in range(1100)]
    assert run.next_groups(lambda *a: groups, [], 21, 0, 4, 128) == groups[500:1000]
    with pytest.raises(RuntimeError, match='never repeat'):
        run.next_groups(lambda *a: groups[:999], [], 21, 0, 4, 128)
    with pytest.raises(RuntimeError, match='found 0'): run.select_source(None, tmp_path)
    def completed(name):
        p = tmp_path/name; p.mkdir()
        (p/'results.json').write_text(json.dumps(dict(status='complete', updates=500,
            contract=dict(protocol=run.previous.PROTOCOL))))
        return p.resolve()
    a = completed('stc_clean_full500_a'); assert run.select_source(None, tmp_path) == a
    completed('stc_clean_full500_b')
    with pytest.raises(RuntimeError, match='found 2'): run.select_source(None, tmp_path)
    assert run.select_source(a, tmp_path) == a


def test_import_keeps_Adam_and_RNG():
    model = torch.nn.Linear(2, 1); adam = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model(torch.ones(1, 2)).sum().backward(); adam.step(); adam.zero_grad(set_to_none=True)
    saved = dict(models={'gt': copy.deepcopy(model.state_dict())},
        optimizers={'gt': copy.deepcopy(adam.state_dict())}, rng=legacy.rng_state())
    expected_torch = torch.rand(3); expected_numpy = np.random.rand(3)
    target = torch.nn.Linear(2, 1); target_adam = torch.optim.AdamW(target.parameters(), lr=9.)
    assert run.import_training_state(saved, {'gt': target}, {'gt': target_adam})[:3] == (0, 0, 0)
    for p, q in zip(model.parameters(), target.parameters()): torch.testing.assert_close(p, q, atol=0, rtol=0)
    for key, value in saved['optimizers']['gt']['state'].items():
        for name, expected in value.items():
            torch.testing.assert_close(target_adam.state_dict()['state'][key][name], expected, atol=0, rtol=0)
    torch.testing.assert_close(torch.rand(3), expected_torch, atol=0, rtol=0)
    np.testing.assert_array_equal(np.random.rand(3), expected_numpy)


def test_actual_continuation_cli_and_resume(tmp_path, monkeypatch):
    # Reuse the tested clean-full tiny actual CLI fixture, no nuScenes/GPU run.
    from test_stc_clean_full import test_actual_cli_original_source_clean_full_and_resume
    test_actual_cli_original_source_clean_full_and_resume(tmp_path, monkeypatch)
    parent = tmp_path/'new_clean_full'; path = parent/'train/last.pt'
    source = torch.load(path, weights_only=False)
    # Mark a synthetic completed stage; retain the real two-update Adam state
    # to verify it is imported rather than reset. No real500 quality claim.
    source.update(epoch=1, batch=0, updates=500); torch.save(source, path)
    metrics = {f'{m}_{r}': dict(mIoU=20., IoU=30., MovingMicro=10.)
               for m in ('occ_gt', 'stc_gt') for r in ('transport', 'semantic_refine')}
    report = dict(status='complete', updates=500, contract=source['contract'],
        evaluation=dict(windows=64, evaluation_keys=source['contract']['eval_keys'], variants={'gt': metrics}))
    (parent/'results.json').write_text(json.dumps(report))
    before = {str(p): p.read_bytes() for p in parent.rglob('*') if p.is_file()}
    monkeypatch.setattr(legacy, 'epoch_groups', lambda records, *a: [[records[i]] for i in range(1100)])
    original_train = run.gt.train_arm; imported = []
    def train(model, optimizer, provider, rows, mode, **kw):
        imported.append((rows[0][0]['t0_token'], copy.deepcopy(optimizer.state_dict())))
        return original_train(model, optimizer, provider, rows, mode, **kw)
    monkeypatch.setattr(run.gt, 'train_arm', train)
    out = tmp_path/'continuation'; args = ['--source-dir', str(parent), '--out-dir', str(out)]
    assert run.main(threading.Event(), args+['--max-updates', '1']) == 0
    expected_before = source['optimizers']['gt']
    assert imported[0][1]['param_groups'] == expected_before['param_groups']
    for key, values in expected_before['state'].items():
        for name, value in values.items(): torch.testing.assert_close(imported[0][1]['state'][key][name], value, atol=0, rtol=0)
    audit = json.loads((out/'import_audit.json').read_text())
    assert audit['source_model_sha256'] == audit['stage_initial_model_sha256'] and audit['next_source_group'] == 501
    first = torch.load(out/'train/last.pt', weights_only=False)
    assert first['updates'] == 1 and first['contract']['cumulative_budget'] == 1000
    assert first['contract']['source_completed_updates'] == 500 and first['contract']['total_updates'] == 500
    assert run.main(threading.Event(), args+['--resume-experiment', '--max-updates', '2']) == 0
    second = torch.load(out/'train/last.pt', weights_only=False)
    assert second['updates'] == 2 and second['batch'] == 2
    assert [r[0] for r in imported] == ['train500', 'train501']
    final = json.loads((out/'results.json').read_text())
    assert final['additional_updates'] == 2 and final['cumulative_adaptation_updates'] == 502
    assert before == {str(p): p.read_bytes() for p in parent.rglob('*') if p.is_file()}
    text = run.summary(dict(status='complete', updates=500, evaluation={'variants': {'gt': metrics}}), report)
    assert 'additional500_minus_source500' in text and 'NOT one planned1000' in text
    with pytest.raises(RuntimeError, match='NEW output'): run.main(argv=args)
