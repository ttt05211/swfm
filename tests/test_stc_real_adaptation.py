"""Actual STC frame files/chain/gradient/Adam resume; no simulated noise."""
import copy
from dataclasses import asdict
import json
from types import SimpleNamespace
import threading
import numpy as np
import pytest
import torch
from tools.stc_real_adaptation import data, evaluation, run
from tools.final_semantic_evaluation.sources import NuscInputs
from tools.final_semantic_evaluation.predictor import Predictor
from tools.shared_history_experiments.full_gt import engine as gt
from test_perception_robustness import real_fixture


def fake_inputs(row, tmp_path, monkeypatch):
    rec = row[0]; times = {t: i for i, t in enumerate((*rec['history_tokens'], *rec['future_tokens']))}
    inputs = object.__new__(NuscInputs)
    from collections import OrderedDict
    inputs.stc = tmp_path/'stc'; inputs.cache = OrderedDict(); inputs.bytes = 0; inputs.limit = 0; inputs.inventory = {}
    samples = {}
    tokens = (*rec['history_tokens'], *rec['future_tokens'])
    for i, token in enumerate(tokens):
        samples[token] = dict(scene_token='scene', timestamp=i*500000,
            prev=tokens[i-1] if i else '', next=tokens[i+1] if i+1 < len(tokens) else '')
    nusc = SimpleNamespace(get=lambda kind, t: samples[t] if kind == 'sample' else {'name': rec['scene_name']})
    inputs.source = SimpleNamespace(dataroot=str(tmp_path/'nusc'), nusc=nusc, pose=lambda t:
        row[1]['history_poses'][times[t]] if times[t] < 4 else row[1]['future_poses'][times[t]-4])
    # Actual released-style STC files intentionally have poisoned GT masks.
    stc = row[1]['history_occ'].copy(); stc[:, 2:4, 2:4, 1:3] = 4
    for i, token in enumerate(tokens):
        path = inputs.path('occ', rec['scene_name'], token); path.parent.mkdir(parents=True, exist_ok=True)
        sem = row[1]['history_occ'][i] if i < 4 else row[1]['future_gt_occ'][i-4]
        np.savez(path, semantics=sem, mask_lidar=np.ones_like(sem, bool))
        if i < 4:
            path = inputs.path('stc', rec['scene_name'], token); path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, semantics=stc[i], mask_lidar=np.array(['DO_NOT_READ_GT_MASK']))
    # Production loader enforces 200x200x16; this only relaxes fixture shape, NOT data validity or training code.
    monkeypatch.setattr('tools.final_semantic_evaluation.sources.semantics', lambda x: np.array(x, np.uint8, copy=True))
    return inputs, stc


def test_split_schedule_and_continuous_token_chain(monkeypatch):
    records = [dict(scene_name='FIT'+str(i%18), t0_token=str(i), features=[0]) for i in range(90)]
    records += [dict(scene_name='DEV', t0_token='d'+str(i), features=[0]) for i in range(64)]
    split = dict(fit_scenes=['FIT'+str(i) for i in range(18)], remaining_scenes=['DEV']+[str(i) for i in range(131)])
    train, dev = run.selection(records, split, [[r['scene_name'], r['t0_token']] for r in records[-64:]])
    assert len(train) == 90 and len(dev) == 64
    groups, positions = data.schedule(train, 21)
    assert len(groups) == len(positions) == 500 and all(len(g) <= 4 for g in groups)
    assert positions[-1]['cycle'] > 0
    assert data.schedule(train, 21)[1] == positions
    bad = dict(split, remaining_scenes=['FIT0']+split['remaining_scenes'][1:])
    with pytest.raises(RuntimeError): run.selection(records, bad, [])
    tokens = [str(i) for i in range(10)]
    samples = {t: dict(scene_token='scene', timestamp=i*500000, prev=tokens[i-1] if i else '',
        next=tokens[i+1] if i < 9 else '') for i, t in enumerate(tokens)}
    nusc = SimpleNamespace(get=lambda kind, t: samples[t] if kind == 'sample' else {'name': 'FIT0'})
    rec = dict(scene_name='FIT0', history_tokens=tokens[:4], t0_token=tokens[3], future_tokens=tokens[4:])
    assert data.validate_chain(nusc, rec)['actual_future_times_s'] == [.5, 1., 1.5, 2., 2.5, 3.]
    samples['3']['next'] = '5'
    with pytest.raises(ValueError, match='non-contiguous'): data.validate_chain(nusc, rec)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_real_STC_histories_GT_targets_full_gradient_and_eval_resume(tmp_path, monkeypatch, device):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    torch.set_num_threads(1)
    model, _, p, row = real_fixture(device)
    with torch.no_grad(): model.columns.readout[4].weight.normal_(0., .01)
    original = copy.deepcopy(model).eval().requires_grad_(False)
    inputs, stc = fake_inputs(row, tmp_path, monkeypatch)
    predictor = object.__new__(Predictor); predictor.provider = p
    from tools.shared_history_experiments.refine.history import ResidualHistoryInputs
    predictor.history = ResidualHistoryInputs(p, ram_mib=0)
    reader = data.HistoryRows(inputs, predictor, ram_mib=1)
    rec = row[0]
    assert data.validate_chain(inputs.source.nusc, rec)
    prepared = reader.conditioned(rec, target=True)
    np.testing.assert_array_equal(prepared[1]['history_occ'], stc)
    np.testing.assert_array_equal(prepared[1]['future_gt_occ'], row[1]['future_gt_occ'])
    assert prepared[1]['history_observed'].all()
    assert 'future_gt_occ' not in reader.prepare(rec)[1] and 'future_poses' not in reader.prepare(rec)[1]
    assert reader.stats()['hits'] >= 2
    for v in inputs.history(data.window(rec), 'stc')[1]['history_occ']:
        assert v.dtype == np.uint8
    model.train().requires_grad_(True)
    adam = torch.optim.AdamW([dict(params=model.transport.parameters(), lr=1e-5, initial_lr=1e-5),
                             dict(params=model.columns.parameters(), lr=1e-4, initial_lr=1e-4)])
    before = copy.deepcopy(model.state_dict())
    stats = gt.train_arm(model, adam, p, [data.ready(p, prepared)], 'gt', update=1, total_updates=500, probe=True)
    assert stats['gradient_links']['occupancy_to_shared_history'] > 0
    assert stats['gradient_links']['occupancy_to_xy'] > 0 and not stats['auxiliary_computed']
    for prefix in ('transport.history_encoder.', 'transport.decoder.', 'columns.'):
        assert any(not torch.equal(v, model.state_dict()[k]) for k, v in before.items() if k.startswith(prefix))
    cp = run.checkpoint(model, adam, {'protocol': run.PROTOCOL}, 1, [stats])
    torch.save(cp, tmp_path/'last.pt'); restored = torch.load(tmp_path/'last.pt', weights_only=False)
    other = copy.deepcopy(model); other.load_state_dict(restored['model'])
    # Check checkpoint contains real optimizer/RNG and no noise/old ego weights.
    assert restored['update'] == 1 and restored['adam']['state'] and restored['rng']['torch'].numel() > 0
    monkeypatch.setattr(inputs, 'moving', lambda *a: [np.zeros_like(row[1]['future_gt_occ'][0], bool) for _ in range(3)])
    baseline = copy.deepcopy(original.state_dict()); saves = []
    report = evaluation.evaluate({'original': original, 'adapted': model}, reader, [rec], save=lambda x: saves.append(copy.deepcopy(x)))
    assert report['windows'] == 1 and set(report['metrics']['original']) == set(evaluation.ROUTES)
    saved = copy.deepcopy(saves[-1]); check = evaluation.evaluate({'original': original, 'adapted': model}, reader, [rec], saved=saved)
    assert check['metrics']['adapted']['stc_gt_semantic_refine']['mIoU'] == report['metrics']['adapted']['stc_gt_semantic_refine']['mIoU']
    for k, v in original.state_dict().items(): torch.testing.assert_close(v, baseline[k], rtol=0, atol=0)
    bad = copy.deepcopy(saved); bad['keys'][0][1] = 'bad'
    with pytest.raises(RuntimeError, match='prefix'): evaluation.evaluate({'original': original, 'adapted': model}, reader, [rec], saved=bad)


def test_actual_new_runner_fresh500_schedule_checkpoint_and_resume(tmp_path, monkeypatch):
    from tools.shared_history_experiments.full_gt.model import PROTOCOL as GT_PROTOCOL
    from tools.stc_noise_adaptation import augmentation
    m, _, p, row = real_fixture(); torch.set_num_threads(1)
    with torch.no_grad(): m.columns.readout[4].weight.normal_(0., .01)
    inputs, _ = fake_inputs(row, tmp_path, monkeypatch)
    metadata = {k: tmp_path/k for k in ('config', 'dev_cache', 'dev_info')}
    for path in metadata.values(): path.write_text('metadata')
    parent = tmp_path/'reference'; (parent/'empirical/frozen').mkdir(parents=True); (parent/'profile').mkdir()
    source_contract = dict(protocol=GT_PROTOCOL, modes=['gt'], epochs=2, total_updates=10292,
        train_keys=[['TRAIN', str(i)] for i in range(20430)], eval_keys=[['DEV', str(i)] for i in range(512)],
        window_batch=4, source_budget=128, model=m.configs())
    source = dict(protocol=GT_PROTOCOL, contract=source_contract, epoch=2, batch=0, updates=10292,
                  models={'gt': m.state_dict()})
    source_path = parent/'empirical/frozen/GTepoch2.pt'; torch.save(source, source_path)
    fields = dict({k: str(v) for k, v in metadata.items()}, device='cuda', seed=20261011, dataroot=str(tmp_path),
                  stc_root=str(inputs.stc), frame_cache_mib=1, cpu_workers=1, override=[])
    hashes = {k: run.legacy.sha256(v) for k, v in metadata.items()}
    (parent/'workflow.json').write_text(json.dumps(dict(args=fields, data=hashes)))
    (parent/'empirical/results.json').write_text(json.dumps(dict(status='complete', updates=500,
        contract=dict(source_GT_sha256=run.legacy.sha256(source_path), eval_keys=[['DEV', str(i)] for i in range(64)]))))
    (parent/'profile/evaluation.json').write_text(json.dumps(dict(split=dict(fit_scenes=['FIT'+str(i) for i in range(18)],
                                                                         remaining_scenes=['DEV'+str(i) for i in range(132)]))))
    # No fitted noise_model.json exists; any generator call is a test failure.
    monkeypatch.setattr(augmentation, 'corrupt', lambda *a, **kw: pytest.fail('NO simulated noise allowed'))
    before = {str(x): x.read_bytes() for x in parent.rglob('*') if x.is_file()}
    monkeypatch.setattr(run.legacy, 'require_cuda', lambda _: torch.device('cpu'))
    monkeypatch.setattr(run.legacy, 'load_cache', lambda _: ({}, [row[0]]*4369))
    dev = [dict(row[0], scene_name='DEV', t0_token='dev'+str(i)) for i in range(64)]
    monkeypatch.setattr(run, 'selection', lambda *a: ([row[0]], dev))
    monkeypatch.setattr(run, 'validate_chain', lambda *a: {'actual_future_times_s': [.5, 1., 1.5, 2., 2.5, 3.]})
    monkeypatch.setattr(inputs, 'preflight', lambda *a: [])
    monkeypatch.setattr(run, 'NuscInputs', lambda *a, **kw: inputs)
    monkeypatch.setattr(run.legacy, 'load_runtime_config', lambda *a: None)
    monkeypatch.setattr(run.legacy, 'make_prepare_config', lambda _: p.pcfg)
    def predictor(body, *a, **kw):
        p.joint = body; obj = object.__new__(Predictor); obj.provider = p
        from tools.shared_history_experiments.refine.history import ResidualHistoryInputs
        obj.history = ResidualHistoryInputs(p, ram_mib=0); obj.close = lambda: None
        return obj
    monkeypatch.setattr(run, 'Predictor', predictor)
    out = tmp_path/'real'
    args = ['--reference-dir', str(parent), '--out-dir', str(out)]
    assert run.main(argv=args+['--max-updates', '1']) == 0
    cp = torch.load(out/'last.pt', weights_only=False)
    assert cp['update'] == 1 and cp['contract']['updates'] == 500
    assert cp['contract']['synthetic_noise'] is False and cp['contract']['extra_augmentation'] is False
    assert cp['contract']['objective'] == dict(CE=1., Lovasz=1., motion=0., existence=0., ego=0.)
    assert len(cp['adam']['state']) > 0 and json.loads((out/'results.json').read_text())['original_weights_unchanged']
    assert run.main(argv=args+['--resume', '--max-updates', '2']) == 0
    later = torch.load(out/'last.pt', weights_only=False)
    assert later['update'] == len(later['reports']) == 2
    assert all(v['step'].item() == 2 for v in later['adam']['state'].values())
    assert before == {str(x): x.read_bytes() for x in parent.rglob('*') if x.is_file()}
    assert not (out/'noise_model.json').exists()
    with pytest.raises(RuntimeError, match='NEW output'): run.main(argv=args)
