"""Current full lineage, epoch0 source, strict mean and frozen workflow recovery."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from tools.causal_forward_average import run,weights
from tools.causal_forward_full import model


def body_and_contract():
    from real_motion.local_st_world_model_v17 import LocalSTWMV17Config
    body=model.make_model(LocalSTWMV17Config(history_frames=4,decoder_blocks=2),
        experiment=dict(FREE_bias=0.,history_age=False,candidate_implementation={}))
    c=dict(protocol=weights.TRAIN_PROTOCOL,model=body.configs(),parameters=2816122,world_size=2,
        source_policy=model.SOURCE_POLICY,registration='none',known='TRAIN_OCC_STC_dense_valid',
        epochs=2,total_updates=10292,epochs_updates=[5145,5147],
        objective=dict(CE=1.,Lovasz=1.,motion=0.,existence=0.,ego=0.),precision='BF16_network_FP64_geometry_full6',
        train_keys=[['TRAIN',str(i)] for i in range(20430)],eval_keys=[['DEV',str(i)] for i in range(64)],
        overrides=[],data={},implementation={},STC_inventory=[])
    return body,c


def source_files(folder,body,c):
    folder.mkdir();reports={};names=dict(body.named_parameters())
    for i,step in enumerate(weights.STEPS):
        state={k:(v.detach().clone()+i*.001 if k in names else v.detach().clone()) for k,v in body.state_dict().items()}
        epoch,batch=weights.cursor(step,c['epochs_updates'])
        torch.save(dict(protocol=weights.TRAIN_PROTOCOL,contract=c,epoch=epoch,batch=batch,updates=step,
            models={'gt':state},optimizers={'gt':{'POISON':'do not copy Adam'}},rank_rng=['POISON']),
            folder/('epoch_0002.pt' if step==10292 else f'step_{step:07d}.pt'))
        reports[str(step)]=dict(windows=64,keys=c['eval_keys'],variants={
            r:dict(mIoU=40.+i,IoU=50.+i,MovingMicro=20.+i) for r in weights.ROUTES})
    (folder/'contract.json').write_text(json.dumps(c))
    (folder/'results.json').write_text(json.dumps(dict(status='complete',epoch=2,updates=10292,contract=c,evaluations=reports)))


def test_current_exact_mean3_epoch0_cursor_buffers_and_no_optimizer(tmp_path):
    body,c=body_and_contract();folder=tmp_path/'source';source_files(folder,body,c)
    actual,sources,controls,header,_=weights.load_sources(folder)
    assert actual==c and weights.cursor(4000,[5145,5147])==(0,4000)
    assert weights.cursor(10000,[5145,5147])==(1,4855) and weights.cursor(10292,[5145,5147])==(2,0)
    value=weights.mean_payload(sources,body,{'own':'contract'})
    assert not value['resume_allowed'] and value['source_updates']==[4000,10000,10292]
    assert not {'optimizers','rank_rng','epoch','batch'} & set(value)
    for k,v in value['models']['gt'].items():
        expected=(sum(s['state'][k].double() for s in sources)/3).to(v.dtype) if k in dict(body.named_parameters()) else sources[0]['state'][k]
        torch.testing.assert_close(v,expected,rtol=0,atol=0)
    restored=run.checkpoint_model(header);restored.load_state_dict(value['models']['gt'],strict=True)
    with pytest.raises(RuntimeError):weights.mean_payload(sources[::-1],body,{})
    bad=copy.deepcopy(sources);bad[0]['state'][next(iter(dict(body.named_parameters())))].flatten()[0]=float('nan')
    with pytest.raises(RuntimeError):weights.mean_payload(bad,body,{})
    small=torch.nn.Linear(1,1);small.register_buffer('counter',torch.tensor(1))
    a=small.state_dict();b=copy.deepcopy(a);b['counter']+=1
    with pytest.raises(RuntimeError,match='buffers'):weights.average_named_parameters([a,b],small)


def test_reject_other_versions_cursor_and_ambiguous_completed_source(tmp_path):
    body,c=body_and_contract();weights.validate_contract(c)
    for field,value in (('protocol','old'),('known','lidar_known'),('epochs',1),('parameters',1194146)):
        with pytest.raises(RuntimeError):weights.validate_contract(dict(c,**{field:value}))
    bad=copy.deepcopy(c);bad['model']['experiment']['history_age']=True
    with pytest.raises(RuntimeError):weights.validate_contract(bad)
    first=tmp_path/'adaln_forward_full_2ep_first';source_files(first,body,c)
    assert weights.select_source(None,tmp_path)==first
    second=tmp_path/'adaln_forward_full_2ep_second';second.mkdir();(second/'results.json').write_bytes((first/'results.json').read_bytes())
    with pytest.raises(RuntimeError,match='ONE'):weights.select_source(None,tmp_path)
    assert weights.select_source(first,tmp_path)==first
    path=first/'step_0004000.pt';saved=torch.load(path,weights_only=False);saved['epoch']=1;torch.save(saved,path)
    with pytest.raises(RuntimeError,match='cursor'):weights.load_sources(first)


def test_actual_frozen_workflow_resume_mean_only_controls_read_and_sources_unchanged(tmp_path,monkeypatch):
    body,c=body_and_contract();files={n:tmp_path/n for n in ('config','dev_cache','dev_info')}
    for p in files.values():p.write_text('fixture')
    c['data']={k:weights.base.sha256(p) for k,p in files.items()}
    folder=tmp_path/'source';source_files(folder,body,c)
    hashes={str(p):weights.base.sha256(p) for p in folder.iterdir() if p.is_file()}
    runner=run.runner
    monkeypatch.setattr(runner.base,'load_cache',lambda *a:({},[dict(scene_name='DEV',t0_token=str(i)) for i in range(64)]))
    monkeypatch.setattr(runner,'window',lambda r:r)
    monkeypatch.setattr(runner.base,'load_runtime_config',lambda *a:{})
    monkeypatch.setattr(runner.base,'make_prepare_config',lambda *a:None)
    monkeypatch.setattr(runner,'NuscInputs',lambda *a,**k:SimpleNamespace(preflight=lambda *a:[]))
    monkeypatch.setattr(runner,'QueryPredictor',lambda *a:SimpleNamespace(close=lambda:None))
    monkeypatch.setattr(torch.optim,'AdamW',lambda *a,**k:pytest.fail('optimizer created'))
    calls=[];stopped=[True]
    def evaluate(predictor,inputs,jobs,task,keys,contract,**kw):
        calls.append(inputs.kind);kw['save'](dict(keys=keys[:8],contract=contract))
        if inputs.kind=='stc' and stopped[0]:raise InterruptedError()
        if inputs.kind=='stc':assert kw['saved']['keys']==keys[:8]
        return dict(status='complete',reports={'gt/'+r:dict(average=dict(mIoU=39.,IoU=49.,MovingMicro=19.),per_horizon={})
            for r in ('transport','semantic_refine')})
    monkeypatch.setattr(runner,'evaluate',evaluate)
    out=tmp_path/'mean';args=['--source-dir',str(folder),'--out-dir',str(out),'--device','cpu','--dataroot',str(tmp_path),'--stc-root',str(tmp_path)]
    for k,p in files.items():args+=['--'+k.replace('_','-'),str(p)]
    assert run.main(args)==2 and calls==['occ','stc']
    stopped[0]=False;run.main(args+['--resume'])
    result=json.loads((out/'results.json').read_text());assert result['status']=='complete'
    assert list(result['comparison'])==['4000','10000','10292','mean3'] and calls==['occ','stc','stc']
    run.main(args+['--resume']);assert calls==['occ','stc','stc']
    assert 'mean3_minus_10292' in (out/'summary.txt').read_text() and '-3.0' in (out/'summary.txt').read_text()
    payload=torch.load(out/'mean3.pt',weights_only=False);assert payload['protocol']==weights.PROTOCOL and 'optimizers' not in payload
    for path,digest in hashes.items():assert weights.base.sha256(path)==digest
    for dep in ('tools/causal_forward_average/run.py','tools/causal_forward_average/weights.py','tools/causal_forward_recovery/adapter.py'):
        assert dep in {str(Path(k).as_posix()) for k in result['contract']['implementation']}
    with pytest.raises(FileExistsError):run.main(args)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_real_current_forward_predictor_six_metadata_H6_no_labels_or_age(device):
    if device=='cuda' and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):pytest.skip('BF16 CUDA')
    from test_fpn_adaln_zero import fixture
    previous,old,row=fixture(device)
    body=model.make_model(previous.transport.config,feature_width=16,z_bins=4,memory_width=4).to(device).eval().requires_grad_(False)
    predictor=run.ForwardMeanPredictor(body,old.provider.pcfg,device,1)
    try:
        raw={k:row[1][k] for k in ('history_occ','history_observed','history_poses')}
        legacy=dict(row[0],history_tokens=('unused_old1','unused_old2')+tuple(row[0]['history_tokens']))
        prepared=predictor.prepare(legacy,raw,times=np.arange(4)*.5)
        poison=prepared[0],dict(prepared[1],future_gt_occ=object(),future_poses=object()),prepared[2],prepared[3]
        before={k:v.clone() for k,v in body.state_dict().items()}
        fw=predictor.forward(poison,poses=row[1]['future_poses'])
        dense=predictor.dense(fw)
        assert set(dense)=={'transport','semantic_refine'} and all(v.shape[0]==6 for v in dense.values())
        assert not fw['logits'].requires_grad and len(legacy['history_tokens'])==6
        for k,v in before.items():torch.testing.assert_close(body.state_dict()[k],v,rtol=0,atol=0)
    finally:predictor.close();old.close()
