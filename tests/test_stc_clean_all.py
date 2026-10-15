"""Frozen clean1000 loading, full population, GT/STC routing and exact eval resume."""
import copy
import json
import threading
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from tools.stc_clean_all import run
from test_perception_robustness import real_fixture


def checkpoint_fixture():
    model, _, _, _ = real_fixture(); spec = model.configs()
    c = dict(protocol=run.SOURCE_PROTOCOL, model=spec, cumulative_budget=1000, source_completed_updates=500,
        augmentation='clean', fully_unfrozen=True, training_known='dense_valid',
        objective=dict(CE=1., Lovasz=1., motion=0., existence=0., ego=0.))
    saved = dict(protocol=run.SOURCE_PROTOCOL, contract=c, updates=500, epoch=1, batch=0,
        models={'gt': copy.deepcopy(model.state_dict())}, optimizers={'gt': {'untouched': 'sentinel'}})
    report = dict(status='complete', updates=500, cumulative_adaptation_updates=1000, contract=c)
    return saved, report


def test_load_exact_clean1000_no_optimizer_or_gradients(tmp_path):
    saved, report = checkpoint_fixture(); path = tmp_path/'last.pt'; torch.save(saved, path); before = path.read_bytes()
    result, body = run.load_frozen(path, report, 'cpu')
    assert not body.training and not any(p.requires_grad for p in body.parameters())
    assert result['optimizers'] == saved['optimizers']
    for key, value in saved['models']['gt'].items(): torch.testing.assert_close(body.state_dict()[key], value, atol=0, rtol=0)
    assert before == path.read_bytes()
    broken = copy.deepcopy(report); broken['cumulative_adaptation_updates'] = 500
    with pytest.raises(RuntimeError, match='only completed'): run.load_frozen(path, broken, 'cpu')


def test_all4369_no_planner_or_population_substitution(tmp_path):
    records = [dict(scene_name=f'scene{i%150}', t0_token=f't{i}', history_tokens=['a', 'b', 'c', f't{i}'],
                    future_tokens=[f'f{j}' for j in range(6)]) for i in range(4369)]
    jobs, keys = run.select_jobs(records)
    assert len(jobs) == len(keys) == 4369 and len({k[0] for k in keys}) == 150
    with pytest.raises(RuntimeError, match='VAL4369'): run.select_jobs(records[:64])
    with pytest.raises(RuntimeError, match='duplicate'): run.select_jobs(records[:-1]+[records[0]])
    with pytest.raises(RuntimeError, match='found 0'): run.select_source(None, tmp_path)
    parent = tmp_path/'stc_clean_continue1000_a'; parent.mkdir()
    (parent/'results.json').write_text(json.dumps(dict(status='complete', cumulative_adaptation_updates=1000,
                                                     contract=dict(protocol=run.SOURCE_PROTOCOL))))
    assert run.select_source(None, tmp_path) == parent.resolve()


class TinyInputs:
    def __init__(self, *args, **kwargs): self.events = []; self.stop = None
    def preflight(self, jobs, kind): return [kind, len(jobs)]
    def history(self, job, kind):
        self.events.append(('history', kind, job.t0_token))
        labels = np.full((4, 2, 2, 1), 11 if kind == 'occ' else 12, np.uint8)
        raw = dict(history_occ=labels, history_observed=np.full(labels.shape, kind == 'stc', bool),
                   history_poses=np.repeat(np.eye(4)[None], 4, axis=0))
        return dict(t0_token=job.t0_token), raw, np.arange(4)*.5
    def poses(self, job): return np.repeat(np.eye(4)[None], 6, axis=0)
    def targets(self, job, indices):
        assert self.events[-1][0] == 'dense'
        self.events.append(('target', job.t0_token))
        if self.stop: self.stop.set()
        return [np.full((2, 2, 1), 11, np.uint8) for _ in indices]
    def moving(self, job, indices, horizons, *args): return [np.ones((2, 2, 1), bool) for _ in indices]


class TinyPredictor:
    inputs = None
    def __init__(self, *args, **kw):
        self.provider = SimpleNamespace(pcfg=SimpleNamespace(grid=SimpleNamespace(shape_hwd=(2, 2, 1))), workers=1)
    def prepare(self, rec, raw, times):
        assert set(raw) == {'history_occ', 'history_observed', 'history_poses'}
        return raw
    def forward(self, row, *, poses):
        assert poses.shape == (6, 4, 4)
        self.inputs.events.append(('forward',))
        return row['history_occ'][-1]
    def dense(self, value):
        self.inputs.events.append(('dense',))
        frames = np.repeat(value[None], 6, axis=0)
        return dict(transport=frames, semantic_refine=frames.copy())
    def close(self): pass


def test_same_engine_modal_inputs_targets_after_prediction_and_resume():
    inputs = TinyInputs(); TinyPredictor.inputs = inputs; predictor = TinyPredictor()
    jobs = [SimpleNamespace(t0_token=f't{i}') for i in range(2)]; keys = [['s', f't{i}'] for i in range(2)]
    event = threading.Event(); inputs.stop = event; states = []
    with pytest.raises(InterruptedError):
        run.evaluate(predictor, run.ModalInputs(inputs, 'stc'), jobs, 'gt', keys, {'kind': 'stc'},
            stop_event=event, save=states.append)
    assert states[-1]['keys'] == keys[:1]
    event.clear(); inputs.stop = None
    report = run.evaluate(predictor, run.ModalInputs(inputs, 'stc'), jobs, 'gt', keys, {'kind': 'stc'},
        saved=states[-1], stop_event=event)
    assert report['windows'] == 2 and report['reports']['gt/semantic_refine']['average']['mIoU'] == 0
    assert [x[1] for x in inputs.events if x[0] == 'history'] == ['stc', 'stc']
    run.evaluate(predictor, run.ModalInputs(inputs, 'occ'), jobs, 'gt', keys, {'kind': 'occ'})
    assert [x[1] for x in inputs.events if x[0] == 'history'][-2:] == ['occ', 'occ']
    with pytest.raises(RuntimeError, match='identity'):
        run.evaluate(predictor, run.ModalInputs(inputs, 'occ'), jobs, 'gt', keys, {'kind': 'occ'}, saved=states[-1])


def test_actual_read_only_cli_and_completed_resume(tmp_path, monkeypatch):
    saved, report = checkpoint_fixture()
    parent = tmp_path/'continuation'; (parent/'train').mkdir(parents=True)
    prior500 = tmp_path/'source500'; prior500.mkdir()
    fields = {k: str(tmp_path/k) for k in ('config', 'dev_cache', 'dev_info')}
    for path in fields.values():
        from pathlib import Path
        Path(path).write_text('metadata')
    fields.update(dataroot=str(tmp_path), stc_root=str(tmp_path/'STC'), override=[])
    workflow = tmp_path/'workflow.json'; workflow.write_text(json.dumps(dict(args=fields)))
    (prior500/'reference.json').write_text(json.dumps(dict(reference={str(workflow): run.legacy.sha256(workflow)})))
    saved['contract'].update(source500=dict(source_checkpoint=str(prior500/'train/last.pt')),
        data={k: run.legacy.sha256(fields[k]) for k in ('dev_cache', 'dev_info')}, train_keys=[['TRAIN', 't']])
    torch.save(saved, parent/'train/last.pt'); (parent/'results.json').write_text(json.dumps(report))
    before = {str(p): p.read_bytes() for p in parent.rglob('*') if p.is_file()}
    jobs = [SimpleNamespace(t0_token=f't{i}') for i in range(2)]; keys = [['VAL', f't{i}'] for i in range(2)]
    monkeypatch.setattr(run.legacy, 'load_cache', lambda _: ({}, []))
    monkeypatch.setattr(run, 'select_jobs', lambda _: (jobs, keys))  # only tiny test; production requires4369
    inputs = TinyInputs(); TinyPredictor.inputs = inputs
    monkeypatch.setattr(run, 'NuscInputs', lambda *a, **kw: inputs)
    monkeypatch.setattr(run, 'Predictor', TinyPredictor)
    monkeypatch.setattr(run.legacy, 'load_runtime_config', lambda *a: None)
    monkeypatch.setattr(run.legacy, 'make_prepare_config', lambda _: None)
    def forbidden(*a, **kw): pytest.fail('evaluation constructed an optimizer')
    monkeypatch.setattr(torch.optim, 'AdamW', forbidden)
    out = tmp_path/'all_eval'; args = ['--source-dir', str(parent), '--out-dir', str(out), '--device', 'cpu']
    assert run.main(argv=args) == 0
    result = json.loads((out/'evaluation.json').read_text())
    assert result['status'] == 'complete' and result['scientific_updates'] == 0
    assert set(result['reports']) == {m+'/'+r for m in ('occ_gt', 'stc_gt') for r in ('transport', 'semantic_refine')}
    assert result['reports']['occ_gt/semantic_refine']['average']['mIoU'] == 100
    count = len(inputs.events)
    assert run.main(argv=args+['--resume']) == 0
    assert len(inputs.events) == count  # completed integer state skips all forwards
    assert before == {str(p): p.read_bytes() for p in parent.rglob('*') if p.is_file()}
