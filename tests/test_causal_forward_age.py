"""Actual current AdaLN geometry plus completed-C CLI/full-window stop/resume."""
import copy
import json
import threading
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from tools.causal_forward_age import run
from tools.causal_forward_train import model
from tools.causal_forward_recovery.adapter import RecoveryPredictor,four_frame_record
from tools.fpn_unet_stc_age import processing


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_current_AdaLN_forward_pair_reuses_motion_prior_and_does_not_mutate_history(device,monkeypatch):
    if device=='cuda' and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):pytest.skip('BF16 CUDA')
    from test_fpn_adaln_zero import fixture
    body,old,row = fixture(device);body.eval().requires_grad_(False)
    with torch.no_grad():body.columns.head.weight.normal_(std=.02)
    predictor = RecoveryPredictor(body,old.provider.pcfg,device,workers=1)
    try:
        raw = {k:row[1][k] for k in ('history_occ','history_observed','history_poses')}
        record = dict(row[0],history_tokens=('older_a','older_b')+tuple(row[0]['history_tokens']))
        prepared = predictor.prepare(record,raw,times=np.arange(4)*.5)
        with torch.no_grad():fw = predictor.infer(prepared,row[1]['future_poses'])
        original = {k:v.clone() for k,v in body.state_dict().items()}
        reads = {k:v.copy() for k,v in prepared[3][0]['memory_reads'].items()}
        tensors = {k:fw[k].clone() for k in ('prior','targets','yaws','logits','memory')}
        forbid = lambda *a:pytest.fail('age recomputed Encoder/Decoder/motion')
        handles = [body.transport.history_encoder.register_forward_hook(forbid),body.transport.residual_head.register_forward_hook(forbid)]
        try:
            poison = prepared[0],dict(prepared[1],future_gt_occ=object()),prepared[2],prepared[3]
            aged,weights = run.candidate(body,predictor.provider,poison,fw)
            assert aged.shape==fw['logits'].shape and torch.isfinite(aged).all() and not aged.requires_grad
            np.testing.assert_array_equal(weights,[.125,.25,.5,1.])
            # The unchanged-weight path must reconstruct the original readout;
            # CPU checks byte equality, CUDA retains its original execution.
            monkeypatch.setattr(processing,'weight_history',lambda h,t:(h,np.ones(4,np.float32)))
            repeated,_ = run.candidate(body,predictor.provider,poison,fw)
            if device=='cpu':torch.testing.assert_close(repeated,fw['logits'],rtol=0,atol=0)
            else:assert torch.isfinite(repeated).all()
        finally:
            for handle in handles:handle.remove()
        for k,v in tensors.items():torch.testing.assert_close(fw[k],v,rtol=0,atol=0)
        for k,v in reads.items():np.testing.assert_array_equal(prepared[3][0]['memory_reads'][k],v)
        for k,v in original.items():torch.testing.assert_close(body.state_dict()[k],v,rtol=0,atol=0)
        assert all(p.grad is None for p in body.parameters()) and len(record['history_tokens'])==6
    finally:predictor.close();old.close()


def test_current_metrics_match_original_C_and_retain_negative_deltas():
    from tools.causal_forward_recovery.run import gt
    old = gt.Metrics();counts = run.metrics.empty(3)
    truth = np.array([4,4,17,17,11,13,4,17],np.uint8).reshape(2,2,2)
    pred = np.array([4,17,4,17,11,11,4,17],np.uint8).reshape(2,2,2)
    moving = truth==4
    for h in range(3):
        old.update(h,pred,truth,moving);run.metrics.add(counts,h,pred,truth,moving)
    a,b = old.compute(),run.metrics.report(counts,run.HORIZONS)
    # Exact integer statistics, equivalent metric formula; multiplication order
    # in the two existing float64 reporters can differ by one final ULP.
    c = counts['confusion'];diag = np.diagonal(c,axis1=1,axis2=2)
    np.testing.assert_array_equal(old.si,diag[:,:17])
    np.testing.assert_array_equal(old.su,(c.sum(1)+c.sum(2)-diag)[:,:17])
    for k in ('mIoU','IoU','MovingMicro','MovingMacro'):
        assert a[k]==pytest.approx(b['average'][k],rel=1e-15,abs=0)
    assert run.change(dict(mIoU=1,IoU=2,MovingMicro=3,MovingMacro=None),
        dict(mIoU=2,IoU=2,MovingMicro=4,MovingMacro=1))==dict(mIoU=-1,IoU=0,MovingMicro=-1,MovingMacro=None)


def test_actual_cli_strict_completed_checkpoint_six_metadata_same64_stop_resume_no_training(tmp_path,monkeypatch):
    from test_fpn_adaln_zero import fixture
    from test_causal_forward_train import fake_control
    old,base,_ = fixture();pcfg = base.provider.pcfg;base.close()
    body = model.make_model(old.transport.config,feature_width=16,z_bins=4,memory_width=4)
    control = tmp_path/'control.json';control.write_text('{}')
    paths = {}
    for name in ('config','dev_info','dev_cache'):
        path = tmp_path/(name+'.fixture');path.write_text('test metadata');paths[name]=path
    records = [dict(scene_name='VAL',t0_token=str(i),history_tokens=('old1','old2','a','b','c',str(i)),future_tokens=tuple('123456')) for i in range(64)]
    stc = tmp_path/'stc';inventory = []
    for token in sorted({'a','b','c',*map(str,range(64))}):
        path = stc/'VAL'/token/'labels.npz';path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'fixture')
        s = path.stat();inventory.append((str(path.resolve()),s.st_size,s.st_mtime_ns))
    experiment = dict(control_results=str(control),control_results_sha256=run.sha(control),candidate_implementation={})
    body.experiment = experiment
    from tools.causal_forward_train.run import PROTOCOL as TRAIN_PROTOCOL
    c = fake_control()['contract'];c.update(protocol=TRAIN_PROTOCOL,world_size=1,
        source_policy=model.SOURCE_POLICY,model=body.configs(),parameters=sum(p.numel() for p in body.parameters()),
        implementation={'tools/causal_forward_train/data.py':run.sha('tools/causal_forward_train/data.py')},
        data={n:run.sha(p) for n,p in paths.items()},eval_keys=[['VAL',str(i)] for i in range(64)],
        STC_inventory=[list(row) for row in sorted(inventory)])
    # All loader metadata are real current C values. Only dataset/geometry I/O
    # are mocked; actual current CPU/CUDA geometry is covered above.
    source = tmp_path/'C_train/epoch_0001.pt';source.parent.mkdir()
    torch.save(dict(protocol=c['protocol'],contract=c,epoch=1,batch=0,updates=1031,
        models={'gt':body.state_dict()},optimizers={'gt':dict(unchanged='Adam')},rank_rng=[dict(marker=21)]),source)
    source_sha = run.sha(source);stop = threading.Event();calls = dict(infer=0,age=0,labels=0)
    shape = (2,2,2);pcfg = SimpleNamespace(grid=SimpleNamespace(shape_hwd=shape))
    monkeypatch.setattr(run,'reference',lambda *a:(control,{}))
    monkeypatch.setattr(run.base,'load_runtime_config',lambda *a:None)
    monkeypatch.setattr(run.base,'make_prepare_config',lambda *a:pcfg)
    monkeypatch.setattr(run.base,'load_cache',lambda *a:({},records))
    monkeypatch.setattr(run,'updated_record',lambda *a:None)
    monkeypatch.setattr(torch.optim,'AdamW',lambda *a,**k:pytest.fail('optimizer created'))
    class Inputs:
        def __init__(self,*a,**kw):pass
        def path(self,kind,scene,token):return stc/scene/token/'labels.npz'
        def stat(self,path):s=path.stat();return s.st_size,s.st_mtime_ns
        def preflight(self,*a):return ['test inventory']
        def history(self,job,kind):
            raw = dict(history_occ=np.full((4,*shape),4,np.uint8),history_observed=np.ones((4,*shape),bool),
                history_poses=np.repeat(np.eye(4)[None],4,0))
            return {},raw,np.arange(4)*.5
        def poses(self,*a):return np.repeat(np.eye(4)[None],6,0)
        def targets(self,job,*a):
            assert calls['infer']==calls['age']==2*(int(job.t0_token)+1)
            calls['labels']+=1;return np.full((3,*shape),4,np.uint8)
        def moving(self,*a):return np.ones((3,*shape),bool)
    class Predictor:
        def __init__(self,body,pcfg,device,workers):self.provider=SimpleNamespace(pcfg=pcfg,workers=workers)
        def prepare(self,rec,raw,**kw):return four_frame_record(rec,raw),raw
        def infer(self,*a):
            calls['infer']+=1;v=torch.zeros(6,18,8);v[:,4]=1
            return dict(logits=v,prior=v,shape=shape)
        def close(self):pass
    def age(*args):
        calls['age']+=1
        if calls['age']==2:stop.set()
        v=torch.zeros(6,18,8);v[:,17]=1
        return v,np.array([.125,.25,.5,1.],np.float32)
    monkeypatch.setattr(run,'DenseInputs',Inputs);monkeypatch.setattr(run,'RecoveryPredictor',Predictor);monkeypatch.setattr(run,'candidate',age)
    out = tmp_path/'out';args=['--checkpoint',str(source),'--out-dir',str(out),'--dataroot',str(tmp_path),
        '--stc-root',str(stc),'--device','cpu']
    for name,path in paths.items():args+=['--'+name.replace('_','-'),str(path)]
    assert run.main(args,event=stop)==130 and calls==dict(infer=2,age=2,labels=1)
    state = json.loads((out/'state.json').read_text());assert len(state['keys'])==1
    assert run.restore(state,state['contract'],8)[1]==[['VAL','0']]
    for field in ('count','weights','contract'):
        bad=copy.deepcopy(state)
        if field=='count':bad['counts'][run.NAMES[0]]['confusion'][0][4][4]-=1
        elif field=='weights':bad['weights']={}
        else:bad['contract']['FREE_bias']=-1
        with pytest.raises(RuntimeError):run.restore(bad,state['contract'],8)
    stop.clear();assert run.main(args+['--resume'],event=stop)==0
    result = json.loads((out/'results.json').read_text());assert result['status']=='complete' and len(result['keys'])==64
    assert calls==dict(infer=128,age=128,labels=64) and run.sha(source)==source_sha
    for kind in ('occ','stc'):
        assert result['age_on_minus_off'][kind]['average']['mIoU']==-100
        assert result['reports'][kind+'/age_off/semantic_refine']['average']['mIoU']==100
    assert 'FREE bias OFF' in (out/'summary.txt').read_text() and '-100.0' in (out/'summary.txt').read_text()
    assert run.main(args+['--resume'],event=stop)==0 and calls==dict(infer=128,age=128,labels=64)
    assert run.sha(source)==source_sha


def test_incomplete_window_never_commits_counts_or_reads_labels(monkeypatch):
    body=torch.nn.Linear(1,1);shape=(2,2,2)
    predictor=SimpleNamespace(provider=SimpleNamespace(pcfg=SimpleNamespace(grid=SimpleNamespace(shape_hwd=shape)),workers=1),
        prepare=lambda *a,**k:(None,{}),infer=lambda *a:dict(prior=torch.zeros(6,18,8),logits=torch.zeros(6,18,8),shape=shape))
    inputs=SimpleNamespace(history=lambda *a:({},None,None),poses=lambda *a:None,
        targets=lambda *a:pytest.fail('labels read before all predictions'),moving=lambda *a:None)
    monkeypatch.setattr(run,'updated_record',lambda *a:None)
    monkeypatch.setattr(run,'candidate',lambda *a:(torch.full((6,18,8),float('nan')),np.ones(4)))
    records=[dict(scene_name='VAL',t0_token='t0',history_tokens=('a','b','c','t0'),future_tokens=tuple('123456'))]
    states=[]
    with pytest.raises(RuntimeError,match='finite'):
        run.evaluate(body,predictor,inputs,records,dict(keys=[['VAL','t0']]),save=states.append)
    assert states[-1]['keys']==[] and states[-1]['weights']=={}
    assert all(np.asarray(v).sum()==0 for fields in states[-1]['counts'].values() for v in fields.values())
