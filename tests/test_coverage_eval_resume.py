"""Real row preparation: STC count differs from GT; preserve frozen OCC prefix."""
import json
from pathlib import Path
import threading
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from tools.coverage_projection import eval_inputs, evaluate, evaluate_resume, run


def record(index=0, sources=3):
    return dict(scene_name='scene'+str(index), t0_token='t'+str(index),
        history_tokens=[f'{index}_past{h}' for h in range(3)]+['t'+str(index)],
        future_tokens=[f'{index}_future{h}' for h in range(6)], features=[None]*sources)


class Inputs:
    def __init__(self, *a, **kw): self.label_reads = 0
    def history(self, w, kind):
        raw = dict(history_occ=np.full((4, 4, 4, 2), 4 if kind == 'occ' else 17, np.uint8),
            history_observed=np.ones((4, 4, 4, 2), bool), history_poses=np.stack([np.eye(4)]*4))
        rec = dict(scene_name=w.scene_name, t0_token=w.t0_token, history_tokens=w.history_tokens, future_tokens=w.future_tokens)
        return rec, raw, np.arange(4)*.5
    def poses(self, w): return np.stack([np.eye(4)]*6)
    def targets(self, w, horizons):
        self.label_reads += 1
        return [np.full((4, 4, 2), 17, np.uint8) for _ in horizons]


class Predictor:
    def __init__(self, *a, **kw): self.provider = None
    def prepare(self, rec, raw, *, times):
        n = 3 if raw['history_occ'][0, 0, 0, 0] == 4 else 1
        return dict(rec, source_class_id=np.full(n, 4)), raw, (times, None, None), None
    def close(self): pass


def test_STC_owns_its_components_and_TRAIN_guard_remains():
    inputs, predictor, metadata = Inputs(), Predictor(), record()
    with pytest.raises(ValueError, match='TRAIN grouping metadata'):
        run.prepare_row(inputs, predictor, metadata, kind='stc')
    value, poses, labels = eval_inputs.prepare_evaluation_row(inputs, predictor, metadata, kind='stc')
    assert len(value[0]['source_class_id']) == 1 and len(metadata['features']) == 3
    assert labels is None and inputs.label_reads == 0
    assert 'features' not in value[0] and poses.shape == (6, 4, 4)
    old = run.prepare_row(inputs, predictor, metadata, kind='occ')
    new = eval_inputs.prepare_evaluation_row(inputs, predictor, metadata, kind='occ')
    np.testing.assert_array_equal(old[0][1]['history_occ'], new[0][1]['history_occ'])
    np.testing.assert_array_equal(old[1], new[1])
    with pytest.raises(ValueError, match='TRAIN grouping metadata'):
        run.prepare_row(inputs, predictor, record(sources=4), labels=True, kind='occ')
    with pytest.raises(ValueError, match='labels follow six'):
        eval_inputs.prepare_evaluation_row(inputs, predictor, metadata, labels=True, kind='stc')


def test_actual_STC_zero_sources_forward_remains_valid():
    from test_coverage_projection import tiny_scene
    from tools.coverage_projection.adapter import forward_window
    body, predictor, old_row, poses, _ = tiny_scene()
    try:
        metadata = dict(old_row[0], features=[None]*len(old_row[0]['source_class_id']))
        raw = {k: np.array(old_row[1][k], copy=True) for k in ('history_occ', 'history_observed', 'history_poses')}
        raw['history_occ'].fill(17); raw['history_observed'].fill(True)
        source = SimpleNamespace(history=lambda w, kind: ({k:metadata[k] for k in eval_inputs.FIELDS}, raw, np.arange(4)*.5),
            poses=lambda w: poses)
        with pytest.raises(ValueError, match='TRAIN grouping metadata'):
            run.prepare_row(source, predictor, metadata, kind='stc')
        row, future_poses, labels = eval_inputs.prepare_evaluation_row(source, predictor, metadata, kind='stc')
        assert len(row[0]['source_class_id']) == 0 and labels is None
        with torch.no_grad(): fw = forward_window(body.eval(), predictor.provider, row, poses=future_poses)
        assert fw['logits'].shape[0] == 6 and bool(torch.isfinite(fw['logits']).all())
    finally: predictor.close()


def test_resume_real_row_path_from_failed_STC_keeps_OCC64_and_source_hashes(tmp_path, monkeypatch):
    shape = (4, 4, 2); volume = int(np.prod(shape)); out = tmp_path/'run'; out.mkdir()
    control_dir = tmp_path/'control'; control_dir.mkdir()
    config = tmp_path/'config'; config.write_text('config')
    dev = tmp_path/'dev'; dev.write_text('dev'); info = tmp_path/'info'; info.write_text('info')
    report_path = control_dir/'results.json'; report_path.write_text('{}')
    weights = torch.nn.Linear(1, 1).state_dict()
    torch.save(dict(contract=dict(model={}), models={'gt':weights}), control_dir/'epoch_0001.pt')
    records = [record(i) for i in range(4369)]; keys = [[r['scene_name'], r['t0_token']] for r in records[:64]]
    cc = dict(eval_keys=keys, overrides=[], data=dict(dev_info=run.sha(info), dev_cache=run.sha(dev)))
    pinned = {name:run.sha(run.ROOT/name) for name in ('tools/coverage_projection/run.py',
        'tools/coverage_projection/evaluate.py', 'tools/coverage_projection/model.py', 'tools/coverage_projection/geometry.py')}
    c = dict(implementations=pinned, control_dir=str(control_dir), control_sha256=run.sha(control_dir/'epoch_0001.pt'),
        control_results_sha256=run.sha(report_path), eval_keys=keys, train_data=dict(config=run.sha(config)))
    checkpoint = out/'epoch_0001.pt'
    blob = dict(protocol=run.PROTOCOL, updates=1031, contract=dict(c, model={}), models={'gt':weights})
    torch.save(blob, checkpoint); original_sha = (run.sha(checkpoint), run.sha(control_dir/'epoch_0001.pt'))
    grid = SimpleNamespace(x_min=0, y_min=0, z_min=0, voxel_size=[1, 1, 1], shape_hwd=shape)
    gt = np.full(shape, 17, np.uint8); gt[0, 0, 0] = 11
    calls, label_reads = [], []
    class Source(Inputs):
        def preflight(self, *a): return []
        def targets(self, w, hs):
            assert len(calls) > len(label_reads)
            label_reads.append(w.t0_token); return [gt for _ in hs]
    def body(spec, *, candidate):
        m = torch.nn.Linear(1, 1); m.candidate = candidate; return m
    def forward(net, provider, row, **kw):
        kind = 'occ' if row[1]['history_occ'][0, 0, 0, 0] == 4 else 'stc'
        assert len(row[0]['source_class_id']) == (3 if kind == 'occ' else 1)
        calls.append(('candidate' if net.candidate else 'control', kind, row[0]['t0_token']))
        logits = torch.zeros(6, 18, volume); logits[:, 17] = 2.; prior = logits.clone()
        if net.candidate: logits[:, 11, 0] = 3.
        return dict(shape=shape, prior=prior, logits=logits, prior_coverage=torch.ones(6, 1, volume))
    from tools.joint_occ_experiments import run as base
    monkeypatch.setattr(base, 'load_cache', lambda _: ({}, records))
    for module in (evaluate, evaluate_resume):
        monkeypatch.setattr(module, 'reference', lambda _:dict(contract=cc))
        monkeypatch.setattr(module, 'load_runtime_config', lambda *a: {})
        monkeypatch.setattr(module, 'make_prepare_config', lambda _: SimpleNamespace(grid=grid))
        monkeypatch.setattr(module, 'DenseInputs', Source); monkeypatch.setattr(module, 'RecoveryPredictor', Predictor)
        monkeypatch.setattr(module, 'from_spec', body)
        monkeypatch.setattr(module, 'original_forward', forward); monkeypatch.setattr(module, 'forward_window', forward)
    args = SimpleNamespace(device='cpu', out_dir=out, dev_cache=dev, dev_info=info, config=config,
        dataroot=tmp_path, stc_root=tmp_path, cpu_workers=1)
    with pytest.raises(ValueError, match='TRAIN grouping metadata'): evaluate.run(args, threading.Event())
    old_state = json.loads((out/'evaluation_state.json').read_text())
    assert old_state['progress'] == {'control/occ':64} and len(calls) == 64
    visual_hash = run.sha(out/'visuals/occ_01_control.npz')
    evaluate_resume.run(args, threading.Event())
    state = json.loads((out/'evaluation_state.json').read_text())
    assert all(n == 64 for n in state['progress'].values()) and len(state['progress']) == 4
    assert state['counts']['control/occ'] == old_state['counts']['control/occ']
    assert state['receipt'] == old_state['receipt']
    assert len(calls) == len(set(calls)) == len(label_reads) == 256
    assert run.sha(out/'visuals/occ_01_control.npz') == visual_hash
    assert (run.sha(checkpoint), run.sha(control_dir/'epoch_0001.pt')) == original_sha
    for name, digest in pinned.items(): assert run.sha(run.ROOT/name) == digest
    for kind in state['counts'].values():
        for route in kind.values(): assert np.asarray(route['confusion']).sum() == 3*64*volume
    result = json.loads((out/'evaluation.json').read_text())
    assert result['reports']['candidate/stc']['semantic_refine']['average']['mIoU'] == 100
    assert result['delta']['occ']['semantic_refine']['IoU'] == 100
    # The repair still rejects changed pinned code. Never waive source hashes.
    blob['contract']['implementations']['tools/coverage_projection/run.py'] = 'bad-digest'
    torch.save(blob, checkpoint)
    with pytest.raises(ValueError, match='implementation changed'):
        evaluate_resume.run(args, threading.Event())


def test_unique_interrupted_run_selection_and_eval_only_entry(tmp_path):
    keys = [['s',str(i)] for i in range(64)]
    def make(name, complete=False):
        root = tmp_path/name; root.mkdir(); (root/'epoch_0001.pt').write_bytes(b'presence only')
        (root/'contract.json').write_text(json.dumps(dict(protocol=run.PROTOCOL, eval_keys=keys)))
        (root/'progress.json').write_text(json.dumps(dict(status='complete', updates=1031)))
        progress = {k:64 for k in ('control/occ','control/stc','candidate/occ','candidate/stc')} if complete else {'control/occ':64}
        (root/'evaluation_state.json').write_text(json.dumps(dict(receipt=dict(keys=keys), progress=progress)))
        return root.resolve()
    chosen = make('coverage_projection_dual_fix_one'); make('coverage_projection_already_complete', complete=True)
    assert eval_inputs.select_interrupted_run(tmp_path) == chosen
    make('coverage_projection_dual_fix_another')
    with pytest.raises(ValueError, match='no date/score selection'): eval_inputs.select_interrupted_run(tmp_path)
    script = (run.ROOT/'tools/coverage_projection/eval_only.sh').read_text()
    assert 'torch.distributed.run' not in script and 'tools.coverage_projection.run ' not in script
    assert 'tools.coverage_projection.evaluate_resume' in script and 'tools.coverage_projection.visuals' in script
