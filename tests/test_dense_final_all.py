"""Fixed epoch2 only; complete population; same-GPU sharding and integer merge."""
import copy
import json
from pathlib import Path
from threading import Event
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from tools.dense_final_all import run, parallel
from tools.final_semantic_evaluation import metrics
from test_shared_source_track import fixture
from test_dense_checkpoint_average import make_bundle


def records():
    return [dict(scene_name=f'VAL{i%150:03d}', t0_token=f't{i}',
                 history_tokens=['h0', 'h1', 'h2', f't{i}'], future_tokens=[f'f{h}' for h in range(6)]) for i in range(4369)]


def test_complete_population_and_only_epoch2_frozen(tmp_path):
    jobs, keys = run.select_jobs(records()); assert len(jobs) == len(keys) == 4369
    with pytest.raises(RuntimeError, match='ALL_VAL'): run.select_jobs(records()[:-1])
    p, _, _ = fixture()
    try:
        parent, _, _, _, _ = make_bundle(tmp_path, p.body); path = parent/'epoch_0002.pt'; before = path.read_bytes()
        c, model = run.load_frozen(path, 'cpu')
        assert not model.training and not any(p.requires_grad for p in model.parameters())
        assert c['training_known'] == 'dense_valid' and path.read_bytes() == before
        with pytest.raises(RuntimeError, match='completed original'): run.load_frozen(parent/'update_010000.pt', 'cpu')
    finally: p.close()


def states_for(keys, sha, voxels, modality, source_contract=None):
    states = []
    for rank in range(2):
        subset = keys[rank::2]; counts = metrics.empty(3)
        # Unequal shard sizes: global IoU MUST NOT be the average of two shard IoUs.
        counts['confusion'][:, 4, 4 if rank == 0 else 17] = len(subset)*voxels
        counts['moving'][:] = counts['confusion']
        c = dict(protocol=run.PROTOCOL, source_sha256=sha, keys=subset, source_contract=source_contract,
            shard=[rank, 2], windows=len(subset), parent_windows=4369,
            known=dict(occ='dense_valid', stc='dense_valid'))
        states.append(dict(contract=dict(base=c, modality=modality), keys=subset, ego_errors=[],
            counts={n: {k: v.tolist() for k, v in counts.items()} for n in ('gt/transport', 'gt/semantic_refine')}))
    return states


def test_merge_integer_counts_before_iou_and_reject_wrong_shard():
    _, keys = run.select_jobs(records()); states = states_for(keys, 'SHA', 8, 'occ')
    result, _ = parallel.merge_states(states, keys, 'SHA', 8, 'occ')
    assert result['occ_gt/semantic_refine']['average']['mIoU'] == pytest.approx(100*2185/4369)
    assert result['occ_gt/semantic_refine']['average']['mIoU'] != pytest.approx(50., abs=1e-6)
    bad = copy.deepcopy(states); bad[1]['keys'] = bad[0]['keys']
    with pytest.raises(RuntimeError, match='identity'): parallel.merge_states(bad, keys, 'SHA', 8, 'occ')
    bad = copy.deepcopy(states); bad[0]['counts']['gt/transport']['confusion'][0][4][4] -= 1
    with pytest.raises(RuntimeError, match='incomplete'): parallel.merge_states(bad, keys, 'SHA', 8, 'occ')


def test_coordinator_launches_both_workers_on_same_visible_gpu_and_resume(tmp_path, monkeypatch):
    p, _, _ = fixture()
    try:
        parent, _, _, files, _ = make_bundle(tmp_path, p.body)
        source = parent/'epoch_0002.pt'; saved = torch.load(source, weights_only=False); c = saved['contract']
        _, keys = run.select_jobs(records()); voxel_count = int(np.prod(p.provider.pcfg.grid.shape_hwd)); calls = []
        monkeypatch.setattr(parallel.torch.cuda, 'device_count', lambda: 1)
        monkeypatch.setattr(parallel.base, 'load_cache', lambda _: ({}, records()))
        monkeypatch.setattr(parallel.base, 'load_runtime_config', lambda *a: None)
        monkeypatch.setattr(parallel.base, 'make_prepare_config', lambda _: p.provider.pcfg)
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
        class Child:
            def __init__(self, cmd, **kw):
                self.pid = 1000+len(calls); self.returncode = 0; calls.append((cmd, kw['env']['CUDA_VISIBLE_DEVICES']))
                rank = int(cmd[cmd.index('--shard-index')+1]); out = Path(cmd[cmd.index('--out-dir')+1]); out.mkdir(exist_ok=True)
                for kind in ('occ', 'stc'):
                    state = states_for(keys, parallel.base.sha256(source), voxel_count, kind, c)[rank]
                    parallel.base.write_json(out/f'{kind}_state.json', state)
                parallel.base.write_json(out/'contract.json', state['contract']['base'])
                parallel.base.write_json(out/'evaluation.json', dict(status='complete', contract=state['contract']['base']))
            def poll(self): return self.returncode
            def wait(self): return self.returncode
            def terminate(self): self.returncode = 130
        monkeypatch.setattr(parallel.subprocess, 'Popen', Child)
        out = tmp_path/'parallel'
        args = ['--source-dir', str(parent), '--out-dir', str(out), '--dataroot', str(tmp_path), '--stc-root', str(tmp_path/'stc')]
        for key, path in files.items(): args += ['--'+key.replace('_', '-'), path]
        before = source.read_bytes(); assert parallel.main(argv=args) == 0
        assert [v for _, v in calls] == ['0', '0']
        result = json.loads((out/'evaluation.json').read_text())
        assert result['status'] == 'complete' and result['contract']['physical_GPUs'] == 1
        assert len(result['reports']) == 4 and source.read_bytes() == before
        assert parallel.main(argv=args+['--resume']) == 0
        assert all('--resume' in cmd for cmd, _ in calls[2:])
        assert (out/'occ_merged_counts.json').is_file() and (out/'stc_merged_counts.json').is_file()
    finally: p.close()


def test_worker_real_metric_engine_partial_stc_resume_no_repeat_occ(tmp_path, monkeypatch):
    p, rec, raw = fixture(); event = Event(); calls = []
    try:
        parent, _, _, files, _ = make_bundle(tmp_path, p.body)
        jobs, keys = run.select_jobs(records()); jobs, keys = jobs[:2], keys[:2]
        monkeypatch.setattr(run, 'select_jobs', lambda _: (jobs, keys))  # Tiny workflow, not a real full-VAL test.
        monkeypatch.setattr(run.base, 'load_cache', lambda _: ({}, []))
        monkeypatch.setattr(run.base, 'load_runtime_config', lambda *a: None)
        monkeypatch.setattr(run.base, 'make_prepare_config', lambda _: p.provider.pcfg)
        class Inputs:
            def __init__(self, *a, **kw): pass
            def preflight(self, *a): return []
            def history(self, job, kind):
                calls.append((kind, job.t0_token))
                if len(calls) == 3: event.set()
                return rec, dict(raw, history_observed=np.zeros_like(raw['history_occ'], bool)), np.arange(4)*.5
            def poses(self, job): return np.tile(np.eye(4), (6, 1, 1))
            def targets(self, job, indices): return [np.full(p.provider.pcfg.grid.shape_hwd, 4, np.uint8)]*3
            def moving(self, *a): return [np.ones(p.provider.pcfg.grid.shape_hwd, bool)]*3
        class Predictor:
            def __init__(self, body, pcfg, device, workers): self.provider = SimpleNamespace(pcfg=pcfg, workers=workers)
            def prepare(self, record, raw, **kw): assert raw['history_observed'].all(); return None
            def forward(self, *a, **kw): return None
            def dense(self, *a):
                return {n: np.full((6, *p.provider.pcfg.grid.shape_hwd), 4, np.uint8)
                        for n in ('transport', 'semantic_refine')}
            def close(self): pass
        monkeypatch.setattr(run, 'NuscInputs', Inputs); monkeypatch.setattr(run, 'SharedTrackPredictor', Predictor)
        out = tmp_path/'worker'; args = ['--source-dir', str(parent), '--out-dir', str(out), '--device', 'cpu',
            '--dataroot', str(tmp_path), '--stc-root', str(tmp_path/'stc')]
        for key, path in files.items(): args += ['--'+key.replace('_', '-'), path]
        original = (parent/'epoch_0002.pt').read_bytes()
        assert run.main(event, args) == 130 and len(calls) == 3
        assert len(json.loads((out/'occ_state.json').read_text())['keys']) == 2
        assert len(json.loads((out/'stc_state.json').read_text())['keys']) == 1
        event.clear(); assert run.main(event, args+['--resume']) == 0 and len(calls) == 4
        result = json.loads((out/'evaluation.json').read_text())
        assert result['status'] == 'complete' and len(result['reports']) == 4
        assert run.main(event, args+['--resume']) == 0 and len(calls) == 4
        assert (parent/'epoch_0002.pt').read_bytes() == original
    finally: p.close()
