"""Fixed epoch2 only; complete population; same-GPU sharding and integer merge."""
import copy
import json
from pathlib import Path
from threading import Event
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from tools.source_scene_ego_all import run, parallel
from tools.final_semantic_evaluation import metrics
from tools.source_scene_ego_all import common


def fixture():
    from test_source_scene_ego_conditioned import fixture as previous
    body,old,row=previous()
    pcfg=old.provider.pcfg;old.close()
    predictor=common.EgoPredictor(body,pcfg,'cpu',1)
    raw={k:row[1][k] for k in ('history_occ','history_observed','history_poses')}
    return predictor,row[0],raw


def make_bundle(root,body):
    files={k:str(root/k) for k in ('config','dev_info','dev_cache')}
    for p in files.values():Path(p).write_text('fixture')
    c=dict(protocol=common.TRAIN_PROTOCOL,arm='ego_query_pure_memory_cross_first_full',model=body.configs(),
        epochs=2,total_updates=10292,epochs_updates=[5145,5147],registration='none',source_policy='shared_pinned',
        known='TRAIN_OCC_STC_dense_valid',objective=dict(CE=1.,Lovasz=1.,motion=0.,existence=0.,ego=0.),
        train_keys=[['TRAIN',str(i)] for i in range(20430)],eval_keys=[['VAL',str(i)] for i in range(64)],
        data={k:run.base.sha256(p) for k,p in files.items()},implementation={},overrides=[])
    parent=root/'source';parent.mkdir()
    for step in (10000,10292):
        torch.save(dict(protocol=common.TRAIN_PROTOCOL,contract=c,epoch=2 if step==10292 else 1,
            batch=0 if step==10292 else step-5145,updates=step,models={'gt':body.state_dict()}),
            parent/('epoch_0002.pt' if step==10292 else 'step_0010000.pt'))
    return parent,None,None,files,None


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
        assert c['known'] == 'TRAIN_OCC_STC_dense_valid' and path.read_bytes() == before
        with pytest.raises(RuntimeError, match='completed NEW'): run.load_frozen(parent/'step_0010000.pt', 'cpu')
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
        monkeypatch.setattr(run, 'NuscInputs', Inputs); monkeypatch.setattr(run, 'EgoPredictor', Predictor)
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


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_actual_NEW_ego_predictor_matches_candidate_adapter_and_no_legacy_forward(device,monkeypatch):
    if device=='cuda' and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):pytest.skip('BF16 CUDA')
    from test_source_scene_ego_conditioned import fixture as candidate_fixture
    from tools.source_scene_ego_conditioned.adapter import forward_window
    from tools.scene_ego_head.frozen_gt.engine import deterministic_entrance
    body,old,row=candidate_fixture(device)
    p=common.EgoPredictor(body.eval().requires_grad_(False),old.provider.pcfg,device,1)
    try:
        monkeypatch.setattr(common.DirectPredictor,'forward',lambda *a,**k:pytest.fail('old no-ego forward called'))
        with torch.no_grad():body.transport.ego_future_proj[-1].weight.normal_(std=.1)
        raw={k:row[1][k] for k in ('history_occ','history_observed','history_poses')}
        raw['history_observed']=np.zeros_like(raw['history_occ'],bool)
        inputs=common.DenseModalInputs(SimpleNamespace(history=lambda *a:(row[0],raw,np.arange(4)*.5)),'occ')
        rec,dense,times=inputs.history(None)
        assert dense['history_observed'].all() and not raw['history_observed'].any()
        prepared=p.prepare(rec,dense,times=times)
        with torch.no_grad(),deterministic_entrance():
            actual=p.forward(prepared,poses=row[1]['future_poses'])
            expected=forward_window(body,p.provider,prepared,poses=row[1]['future_poses'],fresh=True)
        for f in ('ego_features','targets','prior','memory','logits'):
            torch.testing.assert_close(actual[f],expected[f],rtol=0,atol=0)
        assert not any(v.grad is not None for v in body.parameters())
    finally:p.close();old.close()


def test_reject_old_or_pilot_and_ambiguous_auto_selection(tmp_path):
    p,_,_=fixture()
    try:
        parent,_,_,_,_=make_bundle(tmp_path,p.body)
        saved=torch.load(parent/'epoch_0002.pt',weights_only=False)
        with pytest.raises(RuntimeError):common.validate_source(dict(saved,protocol='old_or_pilot'))
        bad=copy.deepcopy(saved);bad['contract']['model']['memory']['future_query_content']=True
        with pytest.raises(RuntimeError):common.validate_source(bad)
        for suffix in ('one','two'):
            d=tmp_path/('scene_ego_query_full_2ep_'+suffix);d.mkdir()
            torch.save(saved,d/'epoch_0002.pt')
            (d/'results.json').write_text(json.dumps(dict(status='complete',contract=saved['contract'])))
        with pytest.raises(RuntimeError,match='unique'):common.select_source(None,tmp_path)
        assert common.select_source(parent,tmp_path)==parent/'epoch_0002.pt'
    finally:p.close()


def test_saved_progress_matches_complete_prefix_and_handles_inflight_write(tmp_path):
    keys=[['V',str(i)] for i in range(6)]
    for rank in range(2):
        d=tmp_path/f'shard_{rank:02d}';d.mkdir()
        (d/'occ_state.json').write_text(json.dumps(dict(keys=keys[rank::2][:2])))
    assert parallel.saved_progress(tmp_path,keys)==dict(occ=4,stc=0)
    (tmp_path/'shard_00/occ_state.json').write_text('{')
    assert parallel.saved_progress(tmp_path,keys)['occ'] is None
    (tmp_path/'shard_00/occ_state.json').write_text(json.dumps(dict(keys=[['wrong','window']])))
    with pytest.raises(RuntimeError,match='population'):parallel.saved_progress(tmp_path,keys)
