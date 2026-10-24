"""Fixed calibration, scene exclusion and read-only complete-window recovery."""
import copy
import json
from threading import Event
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from tools.fpn_stc_bias448 import run


def test_population_excludes_whole_original64_scenes_not_just_keys():
    from test_fpn_unet_eval import records
    data = records(); original = [[r['scene_name'], r['t0_token']] for r in data[1000:1064]]
    jobs, keys = run.select_jobs(data, original, [['TRAIN', 'token']])
    assert len(jobs) == len(keys) == len(set(map(tuple, keys))) == 448
    assert len({k[0] for k in keys}) == 86
    assert not {k[0] for k in keys} & {k[0] for k in original}
    assert run.select_jobs(list(reversed(data)), original, [])[1] == keys
    for bad, used, train in ((data[:-1], original, []), (data, original[:-1]+original[:1], []),
            (data, original, [['VAL001', 'train']])):
        with pytest.raises(RuntimeError): run.select_jobs(bad, used, train)
    # Unique windows but two from the same scene are NOT allowed as original64.
    bad = original[:-1]+[[data[1150]['scene_name'], data[1150]['t0_token']]]
    with pytest.raises(RuntimeError): run.select_jobs(data, bad, [])


def test_all_six_readouts_exact_fixed_bias_and_ties_without_mutation():
    torch.manual_seed(21); logits = torch.randn(6, 18, 32)
    logits[:, :, 0] = -10.; logits[:, 3, 0] = 0.; logits[:, 17, 0] = 1.
    before = logits.clone(); expected = logits.clone(); expected[:, 17] -= 1.
    predictions = run.predict(logits, (4, 4, 2))
    for name, value in zip(run.NAMES, (before, expected)):
        np.testing.assert_array_equal(predictions[name], value.argmax(1).reshape(6, 4, 4, 2))
    assert (predictions[run.NAMES[0]][:, 0, 0, 0] == 17).all()
    assert (predictions[run.NAMES[1]][:, 0, 0, 0] == 3).all()
    torch.testing.assert_close(logits, before, rtol=0, atol=0)
    with pytest.raises(ValueError): run.predict(logits.bfloat16(), (4, 4, 2))
    with pytest.raises(ValueError): run.predict(logits[:3], (4, 4, 2))


def test_count_restore_and_negative_results_are_not_hidden():
    keys = [['scene', 'a'], ['scene', 'b']]; contract = dict(keys=keys)
    counts, completed = run.restore(None, contract, keys, 8)
    for n in run.NAMES:
        counts[n]['confusion'][:, 4, 4 if n == run.NAMES[0] else 5] = 8
        counts[n]['moving'][:] = counts[n]['confusion']
    saved = dict(keys=keys[:1], contract=contract,
        counts={n: {k: v.tolist() for k, v in c.items()} for n, c in counts.items()})
    restored, completed = run.restore(saved, contract, keys, 8)
    reports = {n: run.metrics.report(v, run.HORIZONS) for n, v in restored.items()}
    text = run.summary(dict(keys=completed, contract=contract, status='complete', reports=reports))
    assert '"mIoU": -100.0' in text and '"MovingMicro": -100.0' in text
    assert 'Labeled target-domain OUTPUT CALIBRATION' in text and 'NOT FPS' in text
    for field in ('keys', 'counts'):
        bad = copy.deepcopy(saved)
        if field == 'keys': bad['keys'] = keys[1:]
        else: bad['counts'][run.NAMES[0]]['moving'][0][4][4] += 1
        with pytest.raises(RuntimeError): run.restore(bad, contract, keys, 8)


def test_entrypoint_one_forward_labels_after_readouts_and_resume(tmp_path, monkeypatch):
    from test_fpn_unet_eval import fixture, bundle
    from tools.fpn_unet_full.run import PROTOCOL as FULL
    p, record, raw = fixture(); event = Event(); calls = []; readouts = []
    try:
        folder, paths = bundle(tmp_path, p)
        value = torch.load(folder/'epoch_0001.pt', weights_only=False); c = value['contract']
        c.update(protocol=FULL, epochs=2, total_updates=10292, train_keys=[['TRAIN', str(i)] for i in range(20430)])
        value.update(protocol=FULL, epoch=2, updates=10292)
        source = folder/'epoch_0002.pt'; torch.save(value, source)
        (folder/'results.json').write_text(json.dumps(dict(status='complete', epoch=2, contract=c)))
        before = source.read_bytes(); monkeypatch.setattr(run, 'SOURCE_SHA256', run.base.sha256(source))
        keys = [['NEW', str(i)] for i in range(3)]
        jobs = [SimpleNamespace(scene_name=k[0], t0_token=k[1]) for k in keys]
        monkeypatch.setattr(run, 'select_jobs', lambda *a: (jobs, keys))
        monkeypatch.setattr(run.base, 'load_cache', lambda *a: ({}, []))
        monkeypatch.setattr(run.base, 'load_runtime_config', lambda *a: None)
        monkeypatch.setattr(run.base, 'make_prepare_config', lambda *a: p.provider.pcfg)
        shape = p.provider.pcfg.grid.shape_hwd
        logits = torch.full((6, 18, int(np.prod(shape))), -10.)
        logits[:, 4] = 0.; logits[:, 17] = .5
        original_predict = run.predict
        def predict(*a):
            result = original_predict(*a); readouts.append(tuple(result)); return result
        monkeypatch.setattr(run, 'predict', predict)
        class Inputs:
            def __init__(self, *a, **kw): pass
            def preflight(self, *a): return []
            def history(self, job, kind):
                assert kind == 'stc'; calls.append(job.t0_token)
                if len(calls) == 1: event.set()
                return record[0], raw, np.arange(4)*.5
            def poses(self, job): return np.tile(np.eye(4), (6, 1, 1))
            def targets(self, job, indices):
                assert len(readouts) == len(calls) and readouts[-1] == run.NAMES
                return [np.full(shape, 4, np.uint8)]*3
            def moving(self, *a): return [np.ones(shape, bool)]*3
        class Predictor:
            def __init__(self, body, pcfg, device, workers): self.provider = SimpleNamespace(workers=workers)
            def prepare(self, *a, **kw): return None
            def forward(self, *a, **kw):
                assert self.kind == 'stc'; return dict(logits=logits, shape=shape)
            def close(self): pass
        monkeypatch.setattr(run, 'DenseInputs', Inputs); monkeypatch.setattr(run, 'AllPredictor', Predictor)
        out = tmp_path/'new'; argv = ['--source-dir', str(folder), '--out-dir', str(out), '--device', 'cpu',
            '--dataroot', str(tmp_path), '--stc-root', str(tmp_path/'stc')]
        for k, path in paths.items(): argv += ['--'+k.replace('_', '-'), str(path)]
        assert run.main(argv, event) == 130 and calls == ['0']
        event.clear(); assert run.main(argv+['--resume'], event) == 0 and calls == ['0', '1', '2']
        assert run.main(argv+['--resume'], event) == 0 and len(calls) == 3
        result = json.loads((out/'evaluation.json').read_text())
        assert result['status'] == 'complete' and result['keys'] == keys
        assert result['reports'][run.NAMES[1]]['average']['MovingMicro'] == 100.
        assert source.read_bytes() == before and (out/'frozen_epoch2.pt').read_bytes() == before
        torch.testing.assert_close(logits[:, 17], torch.full_like(logits[:, 17], .5), rtol=0, atol=0)
        monkeypatch.setattr(run, 'SOURCE_SHA256', '0'*64)
        with pytest.raises(RuntimeError, match='cedcf7c'): run.main(argv+['--resume'], event)
    finally: p.close()
