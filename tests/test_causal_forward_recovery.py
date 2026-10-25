"""Real six-token metadata regression and ZERO-update recovery of final weights."""
import copy
import json
import threading
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from tools.causal_forward_recovery import adapter,run
from tools.causal_forward_train import data,model
from tools.causal_forward_train.control import sha


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_legacy_six_tokens_same_actual_four_inputs_geometry_and_logits(device):
    if device=='cuda' and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):pytest.skip('BF16 CUDA')
    from test_fpn_adaln_zero import fixture
    body,base,row = fixture(device)
    raw = {k:row[1][k] for k in data.HISTORICAL}
    legacy = dict(row[0],history_tokens=('unused_older_1','unused_older_2')+tuple(row[0]['history_tokens']))
    predictor = adapter.RecoveryPredictor(body,base.provider.pcfg,device,workers=1)
    try:
        with pytest.raises(ValueError,match='four histories'):data.ForwardPredictor.prepare(predictor,legacy,raw,times=np.arange(4)*.5)
        old = data.ForwardPredictor.prepare(predictor,row[0],raw,times=np.arange(4)*.5)
        new = predictor.prepare(legacy,raw,times=np.arange(4)*.5)
        assert len(legacy['history_tokens'])==6 and new[0]['history_tokens']==old[0]['history_tokens']
        np.testing.assert_array_equal(new[1]['history_occ'],raw['history_occ'])
        for key in ('kta_displacement_xy_m','anchors_xy_t0_m','source_class_id'):
            np.testing.assert_array_equal(new[0][key],old[0][key])
        for field in ('shared_reads','memory_reads'):
            for key,value in new[3][0][field].items():np.testing.assert_array_equal(value,old[3][0][field][key])
        a,b = (predictor.infer(r,row[1]['future_poses']) for r in (old,new))
        assert a['logits'].shape[0]==b['logits'].shape[0]==6
        if device=='cpu':torch.testing.assert_close(a['logits'],b['logits'],rtol=0,atol=0)
        else:assert torch.isfinite(b['logits']).all()
        with pytest.raises(ValueError,match='labels_shape'):
            adapter.four_frame_record(legacy,dict(raw,history_occ=np.concatenate((raw['history_occ'][:2],raw['history_occ']))))
        with pytest.raises(ValueError):adapter.four_frame_record(dict(legacy,t0_token='wrong'),raw)
        with pytest.raises(ValueError):adapter.four_frame_record(dict(legacy,future_tokens=legacy['future_tokens'][:-1]),raw)
    finally:predictor.close();base.close()


def test_completed_checkpoint_strict_loader_keeps_all_training_source_hashes(tmp_path,monkeypatch):
    from test_causal_forward_train import fake_control
    from test_fpn_adaln_zero import fixture
    old,base,_ = fixture()
    try:
        body = model.make_model(old.transport.config,feature_width=16,z_bins=4,memory_width=4)
        c = fake_control()['contract'];c.update(protocol=run.TRAIN_PROTOCOL,world_size=1,source_policy=model.SOURCE_POLICY,
            model=body.configs(),parameters=sum(p.numel() for p in body.parameters()),
            implementation={'tools/causal_forward_train/data.py':sha('tools/causal_forward_train/data.py')})
        saved = dict(protocol=run.TRAIN_PROTOCOL,contract=c,epoch=1,batch=0,updates=c['total_updates'],
            models={'gt':body.state_dict()},optimizers={'gt':dict(unchanged='Adam evidence')},rank_rng=[dict(marker=21)])
        checkpoint = tmp_path/'epoch_0001.pt';torch.save(saved,checkpoint);before = sha(checkpoint)
        monkeypatch.setattr(torch.optim,'AdamW',lambda *a,**k:pytest.fail('optimizer created'))
        restored,contract,receipt = run.load_source(checkpoint,'cpu')
        assert receipt['scientific_updates']==0 and receipt['updates']==1031 and sha(checkpoint)==before
        for k,v in body.state_dict().items():torch.testing.assert_close(restored.state_dict()[k],v,rtol=0,atol=0)
        assert not any(p.requires_grad for p in restored.parameters())
        for bad in (dict(saved,epoch=0),dict(saved,batch=1),dict(saved,updates=1030)):
            with pytest.raises(RuntimeError):run.validate_checkpoint(bad)
        wrong = copy.deepcopy(saved);wrong['contract']['implementation']['tools/causal_forward_train/data.py']='wrong'
        torch.save(wrong,checkpoint)
        with pytest.raises(RuntimeError,match='trained source changed'):run.load_source(checkpoint,'cpu')
    finally:base.close()


def test_evaluation_full_six_before_labels_and_complete_window_resume(monkeypatch):
    from real_motion.geometry import OccupancyGrid
    grid = OccupancyGrid(x_min=0,y_min=0,z_min=0,voxel_size=(1,1,1),shape_hwd=(2,2,2))
    records = [dict(scene_name='VAL',t0_token=str(i),history_tokens=('a','b','c','d','e',str(i)),future_tokens=tuple('123456')) for i in range(2)]
    contract = dict(protocol=run.PROTOCOL,keys=[['VAL',str(i)] for i in range(2)])
    body = torch.nn.Linear(1,1);stop = threading.Event();calls = dict(infer=0,labels=0)
    raw = dict(history_occ=np.full((4,2,2,2),4,np.uint8),history_observed=np.ones((4,2,2,2),bool),
        history_poses=np.repeat(np.eye(4)[None],4,0))
    class Predictor:
        provider = SimpleNamespace(pcfg=SimpleNamespace(grid=grid),workers=1)
        def prepare(self,record,raw,**kw):return adapter.four_frame_record(record,raw),raw
        def infer(self,*a):
            calls['infer']+=1
            if calls['infer']==2:stop.set()
            logits = torch.zeros(6,18,8);logits[:,4]=1.
            return dict(logits=logits,prior=logits,shape=(2,2,2))
    class Inputs:
        def history(self,*a):return {},raw,np.arange(4)*.5
        def poses(self,*a):return np.repeat(np.eye(4)[None],6,0)
        def targets(self,job,*a):
            assert calls['infer']==2*(int(job.t0_token)+1);calls['labels']+=1
            return np.full((3,2,2,2),4,np.uint8)
        def moving(self,*a):return np.ones((3,2,2,2),bool)
    monkeypatch.setattr(run,'updated_record',lambda *a:None)
    states=[]
    with pytest.raises(InterruptedError):run.evaluate(body,Predictor(),Inputs(),records,contract,event=stop,save=lambda x:states.append(copy.deepcopy(x)))
    assert len(states[-1]['keys'])==1 and calls==dict(infer=2,labels=1)
    stop.clear();result = run.evaluate(body,Predictor(),Inputs(),records,contract,event=stop,saved=states[-1],save=lambda x:states.append(copy.deepcopy(x)))
    assert result['windows']==2 and calls==dict(infer=4,labels=2)
    run.evaluate(body,Predictor(),Inputs(),records,contract,saved=states[-1])
    assert calls==dict(infer=4,labels=2)
    for m in result['variants'].values():assert m['mIoU']==100. and m['IoU']==100.
    with pytest.raises(RuntimeError):run.restore(states[-1],dict(contract,keys=contract['keys'][::-1]))


def test_finish_writes_only_new_view_and_preserves_failed_training_results(tmp_path,monkeypatch):
    workflow = tmp_path/'old';out = tmp_path/'recovery';out.mkdir()
    for folder in ('frozen_B','pair_audit','C_train'):
        p = workflow/folder;p.mkdir(parents=True)
        (p/'results.json').write_text(json.dumps(dict(status='running' if folder=='C_train' else 'complete')))
    before = {str(p):sha(p) for p in workflow.rglob('*') if p.is_file()}
    c = dict(model=dict(experiment=dict(control_results=str(tmp_path/'control/results.json'))))
    receipt = dict(updates=1031,scientific_updates=0)
    def combine(*a):
        value = json.loads((out/'C_train/results.json').read_text())
        assert value['status']=='complete' and value['updates']==1031 and value['recovery']==receipt
        (out/'results.json').write_text(json.dumps(dict(status='complete')))
        return 'A/B/C raw+FREE summary\n'
    monkeypatch.setattr(run,'combine',combine)
    text = run.finish(workflow,out,c,dict(windows=64),receipt)
    assert 'ZERO training/optimizer steps' in text
    for path,digest in before.items():assert sha(path)==digest
    assert (out/'recovery_receipt.json').is_file()
