"""Fixed other-scene selection, exact artifact pair and real frozen paired workflow."""
import copy
import json
import threading
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from tools.causal_forward_compare256 import run,artifacts


def population():
    rows=[dict(scene_name='V'+str(i%150),t0_token=str(i),history_tokens=('older1','older2','a','b','c',str(i)),
        future_tokens=tuple('123456')) for i in range(4369)]
    return rows,dict(eval_keys=[[r['scene_name'],r['t0_token']] for r in rows[:64]],train_keys=[['TRAIN','one']])


def test_selection_deterministic_unique256_excludes_ALL_old64_scenes():
    rows,c=population();selected,info=run.select_records(rows,c)
    assert len(selected)==len(set(run.base.record_keys(selected)))==256
    assert not {r['scene_name'] for r in selected}&{k[0] for k in c['eval_keys']}
    assert run.select_records(rows[::-1],c)==(selected,info) and info['independent_test'] is False
    assert info['scenes']==86 and info['planner_trimmed'] is False
    with pytest.raises(RuntimeError):run.select_records(rows[:-1]+[rows[0]],c)
    with pytest.raises(RuntimeError):run.select_records(rows,dict(c,train_keys=[['V149','t']]))


def test_existing_artifacts_load_exact_epoch_and_mean_without_regeneration(tmp_path,monkeypatch):
    from test_causal_forward_average import body_and_contract,source_files
    from tools.causal_forward_average import weights
    body,c=body_and_contract();source=tmp_path/'full';source_files(source,body,c)
    _,states,_,_,report_sha=weights.load_sources(source)
    mc=dict(protocol=weights.PROTOCOL,source_dir=str(source),source_contract=c,keys=c['eval_keys'],
        sources=[{k:s[k] for k in ('step','path','sha256')} for s in states],coefficients=[1/3]*3,source_report_sha256=report_sha,
        implementation={'tools/causal_forward_average/run.py':run.sha('tools/causal_forward_average/run.py')})
    mean=tmp_path/'adaln_forward_mean3_one';mean.mkdir()
    torch.save(weights.mean_payload(states,body,mc),mean/'mean3.pt')
    (mean/'mean_receipt.json').write_text(json.dumps(dict(sha256=run.sha(mean/'mean3.pt'))))
    (mean/'results.json').write_text(json.dumps(dict(status='complete',contract=mc)))
    hashes={str(p):run.sha(p) for d in (source,mean) for p in d.iterdir() if p.is_file()}
    monkeypatch.setattr(weights,'mean_payload',lambda *a:pytest.fail('mean regenerated'))
    monkeypatch.setattr(torch.optim,'AdamW',lambda *a,**kw:pytest.fail('optimizer created'))
    bodies,loaded,receipt=artifacts.load_pair(mean,source)
    assert loaded==c and not any(p.requires_grad for b in bodies.values() for p in b.parameters())
    expected=torch.load(mean/'mean3.pt',weights_only=False)['models']['gt']
    for k,v in expected.items():
        torch.testing.assert_close(bodies['mean3'].state_dict()[k],v,rtol=0,atol=0)
        torch.testing.assert_close(bodies['epoch2'].state_dict()[k],states[-1]['state'][k],rtol=0,atol=0)
    assert receipt['scientific_updates']==0 and artifacts.select_mean(None,tmp_path)==mean
    with pytest.raises(RuntimeError,match='parent'):artifacts.load_pair(mean,tmp_path/'other')
    for p,digest in hashes.items():assert run.sha(p)==digest
    second=tmp_path/'adaln_forward_mean3_two';second.mkdir();(second/'results.json').write_bytes((mean/'results.json').read_bytes())
    with pytest.raises(RuntimeError,match='ONE'):artifacts.select_mean(None,tmp_path)
    with (mean/'mean3.pt').open('ab') as f:f.write(b'tamper')
    with pytest.raises(RuntimeError,match='receipt'):artifacts.load_pair(mean)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_shared_history_geometry_matches_own_preparation_and_both_models_encode(device):
    if device=='cuda' and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):pytest.skip('BF16 CUDA')
    from test_fpn_adaln_zero import fixture
    from tools.causal_forward_full import model
    previous,base,row=fixture(device)
    one=model.make_model(previous.transport.config,feature_width=16,z_bins=4,memory_width=4).to(device).eval().requires_grad_(False)
    two=copy.deepcopy(one)
    with torch.no_grad():two.transport.residual_head.bias.add_(.1)
    predictors={n:run.ForwardMeanPredictor(b,base.provider.pcfg,device,1) for n,b in (('epoch2',one),('mean3',two))}
    calls={n:0 for n in predictors};handles=[]
    for name,p in predictors.items():
        def hook(m,a,o,name=name):calls[name]+=1
        handles.append(p.body.transport.history_encoder.register_forward_hook(hook))
    try:
        raw={k:row[1][k] for k in ('history_occ','history_observed','history_poses')}
        legacy=dict(row[0],history_tokens=('old1','old2')+tuple(row[0]['history_tokens']))
        shared=predictors['epoch2'].prepare(legacy,raw,times=np.arange(4)*.5)
        own=predictors['mean3'].prepare(legacy,raw,times=np.arange(4)*.5)
        poses=row[1]['future_poses']
        a=predictors['epoch2'].forward(shared,poses=poses)
        b=predictors['mean3'].forward(shared,poses=poses)
        expected=predictors['mean3'].forward(own,poses=poses)
        assert calls==dict(epoch2=1,mean3=2)
        assert not torch.equal(a['output']['residual_xy_m'],b['output']['residual_xy_m'])
        if device=='cpu':torch.testing.assert_close(b['logits'],expected['logits'],rtol=0,atol=0)
        else:assert torch.isfinite(b['logits']).all() and torch.isfinite(expected['logits']).all()
        for key in ('kta_displacement_xy_m','anchors_xy_t0_m'):np.testing.assert_array_equal(shared[0][key],own[0][key])
    finally:
        for h in handles:h.remove()
        for p in predictors.values():p.close()
        base.close()


def test_actual_cli_paired256_stop_resume_predictions_before_labels_no_repeated_complete_windows(tmp_path,monkeypatch):
    rows,c=population();c['overrides']=[]
    bodies={n:torch.nn.Linear(1,1) for n in run.MODELS};mean=tmp_path/'mean';mean.mkdir();full=tmp_path/'full';full.mkdir()
    files={n:tmp_path/(n+'.fixture') for n in ('config','dev_cache','dev_info')}
    for p in files.values():p.write_text('fixture')
    c['data']={n:run.sha(p) for n,p in files.items()}
    source={}
    for n in run.MODELS:
        p=tmp_path/(n+'.pt');p.write_bytes(b'not_loaded_fixture');source[n]=dict(path=str(p),sha256=run.sha(p))
    sources=dict(full_dir=str(full),mean_dir=str(mean),checkpoints=source,source_checkpoints=[],
        full_report=source['epoch2'],scientific_updates=0)
    monkeypatch.setattr(run,'select_mean',lambda *a:mean)
    monkeypatch.setattr(run,'load_pair',lambda *a:(bodies,c,sources))
    monkeypatch.setattr(run.base,'load_cache',lambda *a:({},rows))
    monkeypatch.setattr(run.base,'load_runtime_config',lambda *a:None)
    shape=(2,2,2);pcfg=SimpleNamespace(grid=SimpleNamespace(shape_hwd=shape))
    monkeypatch.setattr(run.base,'make_prepare_config',lambda *a:pcfg)
    monkeypatch.setattr(run,'updated_record',lambda *a:None)
    monkeypatch.setattr(torch.optim,'AdamW',lambda *a,**kw:pytest.fail('optimizer created'))
    event=threading.Event();calls=dict(prepare=0,forward=0,labels=0)
    class Inputs:
        def __init__(self,*a,**kw):pass
        def preflight(self,*a):return []
        def poses(self,*a):return np.repeat(np.eye(4)[None],6,0)
        def history(self,*a):return {},{},np.arange(4)*.5
        def targets(self,*a):
            assert calls['forward']==4*(calls['labels']+1);calls['labels']+=1
            return np.full((3,*shape),4,np.uint8)
        def moving(self,*a):return np.ones((3,*shape),bool)
    class Predictor:
        def __init__(self,body,pcfg,device,workers):self.provider=SimpleNamespace(pcfg=pcfg,workers=workers)
        def prepare(self,*a,**kw):calls['prepare']+=1;return {},{}
        def forward(self,*a,**kw):
            calls['forward']+=1
            if calls['forward']==4:event.set()
            return {}
        def dense(self,*a):return {r:np.full((6,*shape),4,np.uint8) for r in ('transport','semantic_refine')}
        def close(self):pass
    monkeypatch.setattr(run,'DenseInputs',Inputs);monkeypatch.setattr(run,'ForwardMeanPredictor',Predictor)
    out=tmp_path/'out';args=['--out-dir',str(out),'--dataroot',str(tmp_path),'--stc-root',str(tmp_path),'--device','cpu']
    for name,p in files.items():args+=['--'+name.replace('_','-'),str(p)]
    assert run.main(args,event=event)==130 and calls==dict(prepare=2,forward=4,labels=1)
    state=json.loads((out/'state.json').read_text());assert len(state['keys'])==1
    bad=copy.deepcopy(state);bad['counts'][run.NAMES[0]]['confusion'][0][4][4]-=1
    with pytest.raises(RuntimeError):run.restore(bad,state['contract'],8)
    event.clear();assert run.main(args+['--resume'],event=event)==0
    value=json.loads((out/'results.json').read_text());assert value['status']=='complete' and len(value['keys'])==256
    assert calls==dict(prepare=512,forward=1024,labels=256)
    assert all(report['average']['mIoU']==100 for report in value['reports'].values())
    assert run.main(args+['--resume'],event=event)==0 and calls==dict(prepare=512,forward=1024,labels=256)
    for item in source.values():assert run.sha(item['path'])==item['sha256']
