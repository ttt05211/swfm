"""Full population, identical fixed readout and complete-window recovery."""
import json
from threading import Event
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from tools.fpn_stc_bias_all import run
from tools.fpn_stc_bias448 import run as screen


def test_same_original_full4369_not_new448_or_planner_trimmed():
    from test_fpn_unet_eval import records
    data = records(); original = [[r['scene_name'], r['t0_token']] for r in data[1000:1064]]
    jobs, keys = run.select_jobs(data, original, [['TRAIN', 'token']])
    assert len(jobs) == len(keys) == len(set(map(tuple, keys))) == 4369
    assert len({k[0] for k in keys}) == 150 and keys[:64] == original
    assert keys[64:] == sorted(keys[64:])
    assert run.select_jobs(list(reversed(data)), original, [])[1] == keys
    assert run.predict is screen.predict and run.restore is screen.restore
    for bad, train in ((data[:-1], []), (data, [['VAL001', 'train']])):
        with pytest.raises(RuntimeError): run.select_jobs(bad, original, train)


def test_frozen_full_runner_one_forward_and_resume_no_duplicate(tmp_path, monkeypatch):
    from test_fpn_unet_eval import fixture, bundle
    from tools.fpn_unet_full.run import PROTOCOL as FULL
    p, record, raw = fixture(); event = Event(); calls = []; forwards = []; readouts = []
    try:
        folder, paths = bundle(tmp_path, p)
        value = torch.load(folder/'epoch_0001.pt', weights_only=False); c = value['contract']
        c.update(protocol=FULL, epochs=2, total_updates=10292, train_keys=[['TRAIN', str(i)] for i in range(20430)])
        value.update(protocol=FULL, epoch=2, updates=10292)
        source = folder/'epoch_0002.pt'; torch.save(value, source)
        (folder/'results.json').write_text(json.dumps(dict(status='complete', epoch=2, contract=c)))
        before = source.read_bytes(); monkeypatch.setattr(run, 'SOURCE_SHA256', run.base.sha256(source))
        keys = [['VAL', str(i)] for i in range(3)]
        jobs = [SimpleNamespace(scene_name=k[0], t0_token=k[1]) for k in keys]
        monkeypatch.setattr(run, 'select_jobs', lambda *a: (jobs, keys))
        monkeypatch.setattr(run.base, 'load_cache', lambda *a: ({}, []))
        monkeypatch.setattr(run.base, 'load_runtime_config', lambda *a: None)
        monkeypatch.setattr(run.base, 'make_prepare_config', lambda *a: p.provider.pcfg)
        shape = p.provider.pcfg.grid.shape_hwd
        logits = torch.full((6, 18, int(np.prod(shape))), -10.)
        logits[:, 4] = 0.; logits[:, 17] = .5
        actual = run.predict
        def predict(*a):
            result = actual(*a); readouts.append(tuple(result)); return result
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
                assert len(readouts) == len(forwards) == len(calls) and readouts[-1] == run.NAMES
                return [np.full(shape, 4, np.uint8)]*3
            def moving(self, *a): return [np.ones(shape, bool)]*3
        class Predictor:
            def __init__(self, body, pcfg, device, workers): self.provider = SimpleNamespace(workers=workers)
            def prepare(self, *a, **kw): return None
            def forward(self, *a, **kw):
                assert self.kind == 'stc'; forwards.append(1); return dict(logits=logits, shape=shape)
            def close(self): pass
        monkeypatch.setattr(run, 'DenseInputs', Inputs); monkeypatch.setattr(run, 'AllPredictor', Predictor)
        out = tmp_path/'full'; argv = ['--source-dir', str(folder), '--out-dir', str(out), '--device', 'cpu',
            '--dataroot', str(tmp_path), '--stc-root', str(tmp_path/'stc')]
        for k, path in paths.items(): argv += ['--'+k.replace('_', '-'), str(path)]
        assert run.main(argv, event) == 130 and calls == ['0']
        event.clear(); assert run.main(argv+['--resume'], event) == 0 and calls == ['0', '1', '2']
        assert run.main(argv+['--resume'], event) == 0 and len(forwards) == 3
        result = json.loads((out/'evaluation.json').read_text())
        assert result['status'] == 'complete' and result['keys'] == keys
        assert result['reports'][run.NAMES[1]]['average']['MovingMicro'] == 100.
        assert result['contract']['calibration']['calibration_scenes_included']
        progress = json.loads((out/'progress.json').read_text())
        assert progress == dict(status='complete', windows=3, total=3)
        text = (out/'summary.txt').read_text()
        assert 'ALL4369' in text and 'NOT an independent test' in text and 'NOT FPS' in text
        assert source.read_bytes() == before and (out/'frozen_epoch2.pt').read_bytes() == before
        torch.testing.assert_close(logits[:, 17], torch.full_like(logits[:, 17], .5), rtol=0, atol=0)
        monkeypatch.setattr(run, 'SOURCE_SHA256', '0'*64)
        with pytest.raises(RuntimeError, match='cedcf7c'): run.main(argv+['--resume'], event)
    finally: p.close()
