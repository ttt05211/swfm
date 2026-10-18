"""Fixed recipe arithmetic, lineage, actual query adapter, frozen workflow and recovery."""
import copy
import json
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from tools.source_scene_integrated_average import run, weights
from tools.source_scene_integrated_full.model import make_model


def fixture():
    from test_source_scene_stage2 import fixture as old
    previous, runner, row = old()
    body = make_model(replace(previous.transport.config, decoder_blocks=2), feature_width=16,
        z_bins=4, memory_width=4, width=8, chunk_size=32)
    return body, runner, row


def contract(body):
    return dict(protocol=weights.TRAIN_PROTOCOL, arm='integrated22_full', model=body.configs(),
        source_policy='shared_pinned', registration='none', known='TRAIN_OCC_STC_dense_valid',
        epochs=2, total_updates=10292, epochs_updates=[5145, 5147],
        objective=dict(CE=1., Lovasz=1., motion=0., existence=0., ego=0.),
        precision='original_BF16_geometry_FP64_H6', train_keys=[['TRAIN', str(i)] for i in range(20430)],
        eval_keys=[['DEV', str(i)] for i in range(64)], overrides=[], data={}, implementation={}, STC_inventory=[])


def source_files(folder, body, c):
    folder.mkdir()
    reports = {}
    for i, step in enumerate(weights.STEPS):
        state = {k:v.clone() for k,v in body.state_dict().items()}
        for k in dict(body.named_parameters()): state[k] += i*.001
        torch.save(dict(protocol=weights.TRAIN_PROTOCOL, contract=c, updates=step,
            epoch=2 if step == 10292 else 1, batch=0 if step == 10292 else step-5145,
            models={'gt':state}, optimizers={'gt':{'POISON':'must not be copied'}}, rank_rng=['POISON']),
            folder/('epoch_0002.pt' if step == 10292 else f'step_{step:07d}.pt'))
        reports[str(step)] = dict(windows=64, keys=c['eval_keys'], variants={
            r:dict(mIoU=40.+i, IoU=50.+i, MovingMicro=20.+i) for r in weights.ROUTES})
    (folder/'contract.json').write_text(json.dumps(c))
    (folder/'results.json').write_text(json.dumps(dict(status='complete', epoch=2, updates=10292, contract=c, evaluations=reports)))


def test_actual_parameter_mean_buffers_and_load(tmp_path):
    body, runner, _ = fixture()
    try:
        folder = tmp_path/'source'; c = contract(body); source_files(folder, body, c)
        actual_c, sources, controls, header, _ = weights.load_sources(folder)
        assert actual_c == c and set(controls) == set(map(str, weights.STEPS))
        payload = weights.mean_payload(sources, body, {'own':'contract'})
        assert payload['resume_allowed'] is False
        assert 'optimizers' not in payload and 'rank_rng' not in payload
        assert tuple(payload['source_updates']) == weights.STEPS
        names = dict(body.named_parameters())
        for k, v in payload['models']['gt'].items():
            expected = (sum(s['state'][k].double() for s in sources)/3).to(v.dtype) if k in names else sources[0]['state'][k]
            torch.testing.assert_close(v, expected, rtol=0, atol=0)
        restored = run.checkpoint_model(header)
        restored.load_state_dict(payload['models']['gt'], strict=True)
        wrong = copy.deepcopy(sources); wrong[0]['state'][next(iter(names))].flatten()[0] = float('nan')
        with pytest.raises(RuntimeError): weights.mean_payload(wrong, body, {})
        with pytest.raises(RuntimeError): weights.mean_payload(sources[::-1], body, {})
        small = torch.nn.Linear(1, 1); small.register_buffer('counter', torch.tensor(2))
        a = copy.deepcopy(small.state_dict()); b = copy.deepcopy(a); b['counter'] += 1
        with pytest.raises(RuntimeError, match='buffers'): weights.average_named_parameters([a,b], small)
    finally: runner.close()


def test_reject_wrong_architecture_cursor_population_and_ambiguous_source(tmp_path):
    body, runner, _ = fixture()
    try:
        c = contract(body); weights.validate_contract(c)
        for field, value in [('registration','ICP'), ('known','lidar_known'), ('epochs',1), ('total_updates',1031)]:
            with pytest.raises(RuntimeError): weights.validate_contract(dict(c, **{field:value}))
        bad = copy.deepcopy(c); bad['model']['kta_conditioning'] = False
        with pytest.raises(RuntimeError): weights.validate_contract(bad)
        with pytest.raises(RuntimeError): weights.validate_contract(dict(c, eval_keys=c['eval_keys'][:-1]))
        folder = tmp_path/'scene_integrated22_full_2ep_one'; source_files(folder, body, c)
        assert weights.select_source(None, tmp_path) == folder
        second = tmp_path/'scene_integrated22_full_2ep_two'; second.mkdir()
        (second/'results.json').write_bytes((folder/'results.json').read_bytes())
        with pytest.raises(RuntimeError, match='ONE'): weights.select_source(None, tmp_path)
        p = folder/'step_0008000.pt'; v = torch.load(p, weights_only=False); v['batch'] += 1; torch.save(v,p)
        with pytest.raises(RuntimeError, match='cursor'): weights.load_sources(folder)
    finally: runner.close()


def test_real_query_forward_fresh_no_ICP_no_labels_and_dense_policy(monkeypatch):
    from real_motion import shared_source_track
    body, runner, row = fixture()
    predictor = run.QueryPredictor(body.eval(), runner.provider.pcfg, 'cpu', 1)
    try:
        monkeypatch.setattr(shared_source_track, 'register_history_shape', lambda *a, **k: pytest.fail('ICP called'))
        raw = {k:row[1][k] for k in ('history_occ','history_observed','history_poses')}
        raw['history_observed'] = np.zeros_like(raw['history_occ'], dtype=bool)
        source = SimpleNamespace(history=lambda *a: (row[0],raw,np.arange(4)*.5))
        rec, dense, times = run.DenseInputs(source, 'occ').history(None)
        assert dense['history_observed'].all() and not raw['history_observed'].any()
        prepared = predictor.prepare(rec,dense,times=times)
        body.columns.readout[-1].weight.data.normal_(std=.01)
        poses = row[1]['future_poses']
        with torch.no_grad():
            expected = run.forward_window(body,predictor.provider,prepared,poses=poses,fresh=True)
            poisoned = prepared[0],dict(prepared[1], future_gt_occ=object(), future_poses=object()),*prepared[2:]
            actual = predictor.forward(poisoned,poses=poses)
        torch.testing.assert_close(actual['logits'],expected['logits'],rtol=0,atol=0)
        assert actual['logits'].shape[0] == 6 and not any(p.grad is not None for p in body.parameters())
    finally: predictor.close(); runner.close()


def test_frozen_runner_resume_controls_not_recomputed_and_sources_unchanged(tmp_path,monkeypatch):
    body, old_runner, _ = fixture()
    try:
        files = {n:tmp_path/n for n in ('config','dev_cache','dev_info')}
        for p in files.values(): p.write_text('fixture')
        c = contract(body); c['data'] = {k:run.base.sha256(p) for k,p in files.items()}
        folder = tmp_path/'source'; source_files(folder,body,c)
        _,sources,_,_,_ = weights.load_sources(folder); hashes = {s['path']:s['sha256'] for s in sources}
        monkeypatch.setattr(run.base,'load_cache',lambda *a: ({},[dict(scene_name='DEV',t0_token=str(i)) for i in range(64)]))
        monkeypatch.setattr(run,'window',lambda r:r)
        monkeypatch.setattr(run.base,'load_runtime_config',lambda *a:{})
        monkeypatch.setattr(run.base,'make_prepare_config',lambda *a:None)
        monkeypatch.setattr(run,'NuscInputs',lambda *a,**k:SimpleNamespace(preflight=lambda *a:[]))
        monkeypatch.setattr(run,'QueryPredictor',lambda *a:SimpleNamespace(close=lambda:None))
        calls=[]; interrupted=[True]
        def evaluate(predictor,inputs,jobs,task,keys,contract,**kw):
            calls.append(inputs.kind)
            kw['save'](dict(keys=keys[:8],contract=contract))
            if inputs.kind == 'stc' and interrupted[0]: raise InterruptedError()
            if inputs.kind == 'stc': assert kw['saved']['keys'] == keys[:8]
            return dict(status='complete',reports={'gt/'+r:dict(average=dict(mIoU=43.,IoU=53.,MovingMicro=23.),per_horizon={})
                for r in ('transport','semantic_refine')})
        monkeypatch.setattr(run,'evaluate',evaluate)
        out = tmp_path/'mean'
        argv=['--source-dir',str(folder),'--out-dir',str(out),'--device','cpu','--dataroot',str(tmp_path),'--stc-root',str(tmp_path)]
        for k,p in files.items(): argv += ['--'+k.replace('_','-'),str(p)]
        assert run.main(argv) == 2
        assert calls == ['occ','stc']
        interrupted[0]=False; run.main(argv+['--resume'])
        v=json.loads((out/'results.json').read_text())
        assert v['status']=='complete' and list(v['comparison'])==['8000','10000','10292','mean3']
        assert calls==['occ','stc','stc']
        run.main(argv+['--resume']); assert calls==['occ','stc','stc']
        text=(out/'summary.txt').read_text(); assert 'mean3_minus_10292' in text and '43.000000 53.000000 23.000000' in text
        for p,h in hashes.items(): assert run.base.sha256(p)==h
        payload=torch.load(out/'mean3.pt',weights_only=False)
        assert payload['protocol']==weights.PROTOCOL and 'optimizers' not in payload
        with pytest.raises(FileExistsError): run.main(argv)
        with pytest.raises(RuntimeError,match='outside'): run.main([*argv,'--out-dir',str(folder/'forbidden')])
    finally: old_runner.close()


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_actual_engine_integer_counts_metrics_and_completed_resume(device, monkeypatch):
    if device == 'cuda' and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()): pytest.skip('BF16 CUDA')
    from tools.shared_history_experiments.full_gt.engine import Metrics
    from tools.final_semantic_evaluation.engine import evaluate
    body, old_runner, row = fixture(); body.to(device).eval().requires_grad_(False)
    predictor = run.QueryPredictor(body,old_runner.provider.pcfg,device,1)
    try:
        body.columns.readout[-1].weight.data.normal_(std=.01)
        raw={k:row[1][k] for k in ('history_occ','history_observed','history_poses')}
        poses=row[1]['future_poses']; targets=row[1]['future_gt_occ']
        source=SimpleNamespace(history=lambda *a:(row[0],raw,np.arange(4)*.5),poses=lambda *a:poses,
            targets=lambda job,idx:[targets[i] for i in idx],
            moving=lambda *a:[np.ones_like(targets[0],dtype=bool) for _ in range(3)])
        state=[]; keys=[['DEV','one']]
        result=evaluate(predictor,source,[None],'gt',keys,{},save=lambda v:state.append(copy.deepcopy(v)))
        assert len(state[-1]['keys'])==1
        prepared=predictor.prepare(row[0],raw,times=np.arange(4)*.5)
        dense=predictor.dense(predictor.forward(prepared,poses=poses))
        for route,prediction in dense.items():
            metric=Metrics()
            for i,h in enumerate((1,3,5)): metric.update(i,prediction[h],targets[h],np.ones_like(targets[h],dtype=bool))
            expected=metric.compute()
            for m in weights.METRICS:
                assert result['reports']['gt/'+route]['average'][m]==pytest.approx(expected[m])
        monkeypatch.setattr(predictor,'forward',lambda *a,**k:pytest.fail('completed resume re-ran inference'))
        again=evaluate(predictor,source,[None],'gt',keys,{},saved=state[-1])
        assert result['reports']==again['reports']
        bad=copy.deepcopy(state[-1]); bad['counts']['gt/transport']['confusion'][0][17][17]+=1
        with pytest.raises(RuntimeError,match='totals'): evaluate(predictor,source,[None],'gt',keys,{},saved=bad)
    finally: predictor.close(); old_runner.close()
