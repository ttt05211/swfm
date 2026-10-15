"""Fixed equal mean, provenance/known policies and single-card evaluation recovery."""
import copy
import json
from pathlib import Path
from threading import Event
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from tools.dense_checkpoint_average import run, weights
from tools.real_motion.surface_ego_ablation_common import tensor_state_fingerprint
from test_shared_source_track import fixture


def make_bundle(tmp_path, body):
    parent, stage = tmp_path/'original', tmp_path/'epoch3'; parent.mkdir(); stage.mkdir()
    files = {}
    for key in ('config', 'dev_cache', 'dev_info'):
        path = tmp_path/key; path.write_text('fixture'); files[key] = str(path)
    original = dict(protocol=weights.ORIGINAL, epochs=2, total_updates=10292, source_policy='shared_pinned',
        model=body.configs(), eval_keys=[['VAL', f't{i}'] for i in range(64)], train_keys=[['TRAIN', str(i)] for i in range(20430)],
        training_known='dense_valid', OCC_eval_known='original_lidar_known', STC_eval_known='dense_valid',
        global_batch=4, source_budget=128, objective=dict(CE=1., Lovasz=1., motion=0., existence=0., ego=0.),
        data={k: run.base.sha256(p) for k, p in files.items()}, implementation={}, STC_inventory=[])
    baseline = copy.deepcopy(body.state_dict())
    def checkpoint(path, step, contract, delta):
        state = copy.deepcopy(baseline)
        for name in dict(body.named_parameters()): state[name] += delta
        torch.save(dict(protocol=contract['protocol'], contract=contract, models={'gt': state}, updates=step,
                        epoch=2, batch=0, rank_rng=[{}, {}], evaluations={}), path)
    checkpoint(parent/'update_010000.pt', 10000, original, 0.)
    checkpoint(parent/'epoch_0002.pt', 10292, original, .02)
    continuation = dict(protocol=weights.CONTINUATION, source_checkpoint=str(parent/'epoch_0002.pt'),
        source_updates=10292, source_sha256=run.base.sha256(parent/'epoch_0002.pt'), source_contract=original,
        model=body.configs(), training_known='dense_valid', source_policy='shared_pinned',
        eval_keys=original['eval_keys'], OCC_eval_known='dense_valid', STC_eval_known='dense_valid', implementation={})
    checkpoint(stage/'update_012000.pt', 12000, continuation, .04)
    checkpoint(stage/'update_014000.pt', 14000, continuation, .06)
    run.base.write_json(stage/'contract.json', continuation)
    def report(folder, step, c):
        m = dict(mIoU=20., IoU=30., MovingMicro=10.)
        value = dict(windows=64, evaluation_keys=c['eval_keys'], variants={'gt': {r: m for r in weights.ROUTES}})
        run.base.write_json(folder/f'eval_update_{step:06d}.json', dict(contract=c, evaluation=value))
    report(parent, 10000, original); report(stage, 10292, continuation)
    report(stage, 12000, continuation); report(stage, 14000, continuation)
    return parent, stage, continuation, files, baseline


def test_parameter_mean_reuses_safe_old_arithmetic_and_rejects_lineage(tmp_path):
    p, _, _ = fixture()
    try:
        parent, stage, c, _, baseline = make_bundle(tmp_path, p.body)
        sources = weights.load_sources(stage, parent, c)
        before = {s['path']: Path(s['path']).read_bytes() for s in sources}
        payload = weights.averaged_payload(sources, p.body, dict(test=True))
        assert payload['coefficients'] == [.25]*4 and payload['resume_allowed'] is False
        assert not any(k in payload for k in ('optimizers', 'rank_rng'))
        for key in dict(p.body.named_parameters()):
            expected = sum(s['state'][key].double() for s in sources)/4
            torch.testing.assert_close(payload['models']['gt'][key], expected.to(baseline[key].dtype), atol=0, rtol=0)
        for key in set(baseline)-set(dict(p.body.named_parameters())):
            torch.testing.assert_close(payload['models']['gt'][key], baseline[key], atol=0, rtol=0)
        c['source_sha256'] = 'wrong'
        with pytest.raises(RuntimeError, match='initializer'): weights.load_sources(stage, parent, c)
        for path, value in before.items(): assert Path(path).read_bytes() == value
    finally: p.close()


def test_reports_label_old_known_and_exact_population(tmp_path):
    p, _, _ = fixture()
    try:
        parent, stage, c, _, _ = make_bundle(tmp_path, p.body)
        sources = weights.load_sources(stage, parent, c)
        r = weights.historical_report(sources[0]); assert r['known']['occ'] == 'original_lidar_known'
        assert weights.historical_report(sources[2])['known']['occ'] == 'dense_valid'
        assert weights.select_directories(stage, None, tmp_path)[1] == parent
        path = parent/'eval_update_010000.json'; value = json.loads(path.read_text())
        value['evaluation']['evaluation_keys'][0][1] = 'WRONG'; run.base.write_json(path, value)
        with pytest.raises(RuntimeError, match='population'): weights.historical_report(sources[0])
    finally: p.close()


def test_wide_recipe_exact_steps_equal_mean_and_no_late_weights(tmp_path):
    p, _, _ = fixture()
    try:
        parent, stage, c, _, _ = make_bundle(tmp_path, p.body)
        template = torch.load(parent/'update_010000.pt', weights_only=False)
        for step, delta in ((4000, .01), (6000, .02), (8000, .03)):
            value = copy.deepcopy(template); value['updates'] = step
            for key in dict(p.body.named_parameters()): value['models']['gt'][key] += delta
            torch.save(value, parent/f'update_{step:06d}.pt')
        sources = weights.load_sources(stage, parent, c, steps=weights.RECIPES['wide'])
        assert [s['step'] for s in sources] == [4000, 6000, 8000, 10292]
        assert all(Path(s['path']).parent == parent for s in sources)
        payload = weights.averaged_payload(sources, p.body, dict(test=True))
        assert payload['source_updates'] == [4000, 6000, 8000, 10292]
        for key in dict(p.body.named_parameters()):
            expected = (sum(s['state'][key].double() for s in sources)/4).float()
            torch.testing.assert_close(payload['models']['gt'][key], expected, atol=0, rtol=0)
        text = run.comparison_text(dict(status='complete', contract=dict(source_steps=payload['source_updates']),
                                       historical={}, matched={}))
        assert '4000,6000,8000,10292(epoch2)' in text and '12000' not in text
        with pytest.raises(ValueError, match='explicitly fixed'):
            weights.load_sources(stage, parent, c, steps=(4000, 6000, 8000, 10000))
    finally: p.close()


def test_single_card_cli_reuses_dense_reports_only_and_resumes_real_engine(tmp_path, monkeypatch):
    torch.set_num_threads(1); p, rec, raw = fixture(); event = Event()
    try:
        parent, stage, c, files, _ = make_bundle(tmp_path, p.body)
        before = {str(path): path.read_bytes() for folder in (parent, stage) for path in folder.glob('*.pt')}
        records = [dict(rec, t0_token=f't{i}') for i in range(64)]
        class Inputs:
            def __init__(self, *a, **kw): pass
            def preflight(self, jobs, kind): return []
            def history(self, job, kind):
                value = dict(raw, history_observed=np.zeros_like(raw['history_occ'], bool))
                return dict(rec, t0_token=job.t0_token, history_tokens=[*rec['history_tokens'][:-1], job.t0_token]), value, np.arange(4)*.5
            def poses(self, job): return np.tile(np.eye(4), (6, 1, 1))
            def targets(self, job, indices): return [raw['history_occ'][-1] for _ in indices]
            def moving(self, *a): return [np.ones(p.provider.pcfg.grid.shape_hwd, bool)]*3
        calls = []
        class Runner:
            # Actual small SharedTrackPredictor/Refiner, not a mock prediction array.
            def __init__(self, body, pcfg, device, workers):
                self.real = run.SharedTrackPredictorOriginal(body, pcfg, device, workers=1)
                self.provider = self.real.provider
            def prepare(self, record, raw, **kwargs):
                assert raw['history_observed'].all()
                calls.append(record['t0_token'])
                if len(calls) == 8: event.set()
                return self.real.prepare(record, raw, **kwargs)
            def forward(self, *a, **kw): return self.real.forward(*a, **kw)
            def dense(self, *a): return self.real.dense(*a)
            def close(self): self.real.close()
        monkeypatch.setattr(run, 'SharedTrackPredictorOriginal', run.SharedTrackPredictor, raising=False)
        monkeypatch.setattr(run, 'SharedTrackPredictor', Runner)
        monkeypatch.setattr(run, 'NuscInputs', Inputs)
        monkeypatch.setattr(run.base, 'load_cache', lambda _: ({}, records))
        monkeypatch.setattr(run.base, 'load_runtime_config', lambda *a: None)
        monkeypatch.setattr(run.base, 'make_prepare_config', lambda _: p.provider.pcfg)
        out = tmp_path/'average'
        args = ['--epoch3-dir', str(stage), '--out-dir', str(out), '--device', 'cpu',
                '--dataroot', str(tmp_path), '--stc-root', str(tmp_path/'stc')]
        for key, path in files.items(): args += ['--'+key.replace('_', '-'), path]
        assert run.main(event, args) == 130
        assert len(json.loads((out/'10000_occ_state.json').read_text())['keys']) == 8
        event.clear(); assert run.main(event, args+['--resume']) == 0
        result = json.loads((out/'results.json').read_text())
        assert result['status'] == 'complete' and set(result['matched']) == {'10000', '10292', '12000', '14000', 'mean4'}
        assert result['historical']['10000']['known']['occ'] == 'original_lidar_known'
        assert result['historical']['10292']['known']['occ'] == 'dense_valid'
        assert result['matched']['10000']['origins']['occ_gt_semantic_refine'].startswith('fresh')
        assert result['matched']['10000']['origins']['stc_gt_semantic_refine'].startswith('reused')
        assert result['matched']['10292']['origins']['occ_gt_semantic_refine'].startswith('reused')
        assert len(calls) == 64*3  #10000 OCC corrected; mean4 OCC+STC, no other forecasts.
        assert (out/'comparison.csv').exists() and 'MATCHED DENSE-VALID' in (out/'summary.txt').read_text()
        old_calls = len(calls); assert run.main(event, args+['--resume']) == 0
        assert len(calls) == old_calls
        for path, value in before.items(): assert Path(path).read_bytes() == value
        assert not p.body.training and not any(t.requires_grad for t in p.body.parameters())
    finally: p.close()
