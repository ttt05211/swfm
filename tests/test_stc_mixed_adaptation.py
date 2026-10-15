"""Expanded scene isolation, exact 2+2 replay, real reads and resume."""
import copy
import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from tools.stc_mixed_adaptation import data, run
from tools.stc_real_adaptation import data as real_data
from tools.final_semantic_evaluation.predictor import Predictor
from test_perception_robustness import real_fixture
from test_stc_real_adaptation import fake_inputs


def test_expanded_STC_excludes_ENTIRE_eval_scenes_and_GT_is_unique_TRAIN():
    val = [dict(scene_name='V'+str(i%150), t0_token='v'+str(i), features=[]) for i in range(1500)]
    gt = [dict(scene_name='TRAIN'+str(i%100), t0_token='g'+str(i), features=[]) for i in range(2043)]
    keys = [['V'+str(i), 'v'+str(i)] for i in range(64)]
    stc, clean, dev = data.selection(val, gt, keys, 21)
    assert len({r['scene_name'] for r in stc}) == 86
    assert len(clean) == len({r['t0_token'] for r in clean}) == 1000
    assert {r['scene_name'] for r in stc}.isdisjoint({r['scene_name'] for r in dev})
    groups = data.schedule(stc, clean, 21)
    assert len(groups) == 500
    assert all([k for k, _ in g] == ['stc', 'occ', 'stc', 'occ'] for g in groups)
    consumed = [(k, r['t0_token']) for g in groups for k, r in g]
    assert len({t for k, t in consumed if k == 'occ'}) == 1000
    assert len({t for k, t in consumed if k == 'stc'}) == len(stc)
    assert [[(k, r['t0_token']) for k, r in g] for g in data.schedule(stc, clean, 21)] == [[(k, r['t0_token']) for k, r in g] for g in groups]
    bad = [dict(r, scene_name='V0') for r in gt]
    with pytest.raises(RuntimeError, match='overlap'): data.selection(val, bad, keys, 21)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_real_B4_has_both_modalities_and_full_occ_gradients(tmp_path, monkeypatch, device):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    torch.set_num_threads(1)
    model, _, p, row = real_fixture(device)
    with torch.no_grad(): model.columns.readout[4].weight.normal_(0., .01)
    inputs, stc = fake_inputs(row, tmp_path, monkeypatch)
    # Force a visible distinction: STC never consumes this GT known mask.
    for token in row[0]['history_tokens']:
        path = inputs.path('occ', row[0]['scene_name'], token)
        with np.load(path) as f: sem = f['semantics'].copy()
        np.savez(path, semantics=sem, mask_lidar=np.zeros_like(sem, bool))
    predictor = object.__new__(Predictor); predictor.provider = p
    from tools.shared_history_experiments.refine.history import ResidualHistoryInputs
    predictor.history = ResidualHistoryInputs(p, ram_mib=0)
    # Fixture shares metadata only; production passes disjoint TRAIN and VAL sources.
    reader = data.MixedRows(inputs, inputs, predictor, [])
    event = __import__('threading').Event()
    iterator = data.batches(reader, [[('stc', row[0]), ('occ', row[0]), ('stc', row[0]), ('occ', row[0])]], event)
    rows, _ = next(iterator); iterator.close()
    assert len(rows) == 4
    for i in (0, 2):
        np.testing.assert_array_equal(rows[i][1]['history_occ'], stc)
        assert rows[i][1]['history_observed'].all()
    for i in (1, 3):
        np.testing.assert_array_equal(rows[i][1]['history_occ'], row[1]['history_occ'])
        assert not rows[i][1]['history_observed'].any()
    for r in rows: np.testing.assert_array_equal(r[1]['future_gt_occ'], row[1]['future_gt_occ'])
    adam = torch.optim.AdamW([dict(params=model.transport.parameters(), lr=1e-5, initial_lr=1e-5),
                             dict(params=model.columns.parameters(), lr=1e-4, initial_lr=1e-4)])
    before = copy.deepcopy(model.state_dict())
    stats = run.gt.train_arm(model, adam, p, [real_data.ready(p, r) for r in rows], 'gt', update=1, total_updates=500, probe=True)
    assert stats['windows'] == 4 and stats['gradient_links']['occupancy_to_xy'] > 0
    assert not stats['auxiliary_computed']
    for prefix in ('transport.history_encoder.', 'transport.decoder.', 'columns.'):
        assert any(not torch.equal(v, model.state_dict()[k]) for k, v in before.items() if k.startswith(prefix))


def test_actual_mixed_runner_fresh_fixed500_and_Adam_resume(tmp_path, monkeypatch):
    from tools.shared_history_experiments.full_gt.model import PROTOCOL as GT_PROTOCOL
    m, _, p, row = real_fixture(); torch.set_num_threads(1)
    with torch.no_grad(): m.columns.readout[4].weight.normal_(0., .01)
    inputs, _ = fake_inputs(row, tmp_path, monkeypatch)
    metadata = {k: tmp_path/k for k in ('config', 'train_cache', 'train_info', 'dev_cache', 'dev_info')}
    for path in metadata.values(): path.write_text('metadata')
    parent = tmp_path/'paired'; (parent/'empirical/frozen').mkdir(parents=True)
    c = dict(protocol=GT_PROTOCOL, modes=['gt'], epochs=2, total_updates=10292,
        train_keys=[['TRAIN', str(i)] for i in range(20430)], eval_keys=[['DEV', str(i)] for i in range(512)],
        window_batch=4, source_budget=128, model=m.configs())
    source_path = parent/'empirical/frozen/GTepoch2.pt'
    torch.save(dict(protocol=GT_PROTOCOL, contract=c, epoch=2, batch=0, updates=10292, models={'gt': m.state_dict()}), source_path)
    fields = dict({k: str(v) for k, v in metadata.items()}, device='cuda', seed=20261011, dataroot=str(tmp_path),
                  stc_root=str(inputs.stc), cpu_workers=1, override=[])
    (parent/'workflow.json').write_text(json.dumps(dict(args=fields, data={k: run.legacy.sha256(v) for k, v in metadata.items()})))
    (parent/'empirical/results.json').write_text(json.dumps(dict(status='complete',
        contract=dict(source_GT_sha256=run.legacy.sha256(source_path), eval_keys=[['DEV', str(i)] for i in range(64)]))))
    before = {str(x): x.read_bytes() for x in parent.rglob('*') if x.is_file()}
    monkeypatch.setattr(run.legacy, 'require_cuda', lambda _: torch.device('cpu'))
    monkeypatch.setattr(run.legacy, 'load_cache', lambda path: ({}, [row[0]]*(20430 if path == fields['train_cache'] else 4369)))
    dev = [dict(row[0], scene_name='DEV', t0_token='dev'+str(i)) for i in range(64)]
    # Actual sampling isolation is covered above; runner test uses one tiny replay.
    clean = [dict(row[0], scene_name='CLEAN', t0_token='clean'+str(i)) for i in range(1000)]
    monkeypatch.setattr(run, 'selection', lambda *a: ([row[0]], clean, dev))
    monkeypatch.setattr(run, 'validate_chain', lambda *a: {})
    monkeypatch.setattr(inputs, 'preflight', lambda *a: [])
    monkeypatch.setattr(run, 'NuscInputs', lambda *a, **kw: inputs)
    monkeypatch.setattr(run.legacy, 'load_runtime_config', lambda *a: None)
    monkeypatch.setattr(run.legacy, 'make_prepare_config', lambda _: p.pcfg)
    class Reader:
        def __init__(self, *a): pass
        def conditioned(self, record, kind='stc', target=False): return row
        def stats(self): return {}
    monkeypatch.setattr(run, 'MixedRows', Reader)
    def predictor(body, *a, **kw):
        p.joint = body; return SimpleNamespace(provider=p, close=lambda: None)
    monkeypatch.setattr(run, 'Predictor', predictor)
    out = tmp_path/'mixed'; args = ['--reference-dir', str(parent), '--out-dir', str(out)]
    assert run.main(argv=args+['--max-updates', '1']) == 0
    first = torch.load(out/'last.pt', weights_only=False)
    assert first['protocol'] == run.PROTOCOL and first['contract']['updates'] == 500
    assert first['contract']['batch'] == dict(windows=4, STC=2, GT=2, source_truncation=False)
    assert first['reports'][0]['STC_windows'] == first['reports'][0]['GT_windows'] == 2
    assert run.main(argv=args+['--resume', '--max-updates', '2']) == 0
    later = torch.load(out/'last.pt', weights_only=False)
    assert later['update'] == len(later['reports']) == 2
    assert all(v['step'].item() == 2 for v in later['adam']['state'].values())
    assert before == {str(x): x.read_bytes() for x in parent.rglob('*') if x.is_file()}
    assert json.loads((out/'results.json').read_text())['original_weights_unchanged']
    assert not (parent/'profile').exists()  # No fitted noise or old18 split needed.
    with pytest.raises(RuntimeError, match='NEW output'): run.main(argv=args)
