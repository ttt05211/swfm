"""ALL population, age-only routing and complete-window recovery checks."""
import copy
import json
from threading import Event
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from tools.fpn_unet_all import run


def test_all_population_no_planner_or_STC_trimming():
    from test_fpn_unet_eval import records
    data=records(); original=[[r['scene_name'],r['t0_token']] for r in data[1000:1064]]
    jobs,keys=run.select_jobs(data,original,[['TRAIN','token']])
    assert len(jobs)==len(keys)==len(set(map(tuple,keys)))==4369 and keys[:64]==original
    assert run.select_jobs(list(reversed(data)),original,[['TRAIN','token']])[1]==keys
    assert keys[64:]==sorted(keys[64:])
    with pytest.raises(RuntimeError):run.select_jobs(data[:-1],original,[])
    with pytest.raises(RuntimeError):run.select_jobs(data,original,[['VAL001','train']])
    with pytest.raises(RuntimeError):run.select_jobs(data,original[:-1]+[original[0]],[])


def test_real_forward_age_matches_previous_screen_without_double_refiner(monkeypatch):
    from test_fpn_unet import fixture
    from tools.fpn_unet_stc_age.processing import candidate
    body,old,row=fixture();pcfg=old.provider.pcfg;old.close()
    body.eval().requires_grad_(False)
    with torch.no_grad():body.columns.head.weight.normal_(std=.02)
    predictor=run.AllPredictor(body,pcfg,'cpu',workers=1)
    try:
        raw={k:row[1][k] for k in ('history_occ','history_observed','history_poses')}
        prepared=predictor.prepare(row[0],raw,times=np.arange(4)*.5)
        before=copy.deepcopy(prepared[3][0]['memory_reads'])
        model_before={k:v.clone() for k,v in body.state_dict().items()}
        baseline=predictor.forward(prepared,poses=row[1]['future_poses'])
        expected,weights=candidate(body,predictor.provider,prepared,baseline)
        calls=[];actual=body.columns.forward
        monkeypatch.setattr(body.columns,'forward',lambda *a,**k:(calls.append(1),actual(*a,**k))[1])
        predictor.kind='stc';aged=predictor.forward(prepared,poses=row[1]['future_poses'])
        assert calls==[1]  # no baseline + candidate redundant U-Net pass
        torch.testing.assert_close(aged['logits'],expected,rtol=0,atol=0)
        torch.testing.assert_close(aged['prior'],baseline['prior'],rtol=0,atol=0)
        torch.testing.assert_close(aged['targets'],baseline['targets'],rtol=0,atol=0)
        np.testing.assert_array_equal(weights,[.125,.25,.5,1.])
        for k,v in before.items():np.testing.assert_array_equal(prepared[3][0]['memory_reads'][k],v)
        for k,v in model_before.items():torch.testing.assert_close(body.state_dict()[k],v,rtol=0,atol=0)
        predictor.kind='occ';restored=predictor.forward(prepared,poses=row[1]['future_poses'])
        torch.testing.assert_close(restored['logits'],baseline['logits'],rtol=0,atol=0)
        assert not any(p.grad is not None for p in body.parameters())
    finally:predictor.close()


def test_full_checkpoint_and_entrypoint_resume_no_duplicate_windows(tmp_path,monkeypatch):
    from test_fpn_unet_eval import fixture,bundle
    from tools.fpn_unet_full.run import PROTOCOL as FULL
    p,record,raw=fixture();event=Event();calls=[]
    try:
        folder,paths=bundle(tmp_path,p)
        value=torch.load(folder/'epoch_0001.pt',weights_only=False)
        c=value['contract'];c.update(protocol=FULL,epochs=2,total_updates=10292,
            train_keys=[['TRAIN',str(i)] for i in range(20430)])
        value.update(protocol=FULL,epoch=2,updates=10292)
        source=folder/'epoch_0002.pt';torch.save(value,source)
        (folder/'results.json').write_text(json.dumps(dict(status='complete',epoch=2,updates=10292,contract=c)))
        before=source.read_bytes()
        _,loaded=run.load_frozen(source,'cpu')
        assert not loaded.training and not any(v.requires_grad for v in loaded.parameters())
        del loaded
        # Exercise the real runner/metric accumulator on tiny 64+1 fixtures.
        # The full4369 selector has a separate non-mocked test above.
        keys=[['VAL',str(i)] for i in range(65)]
        jobs=[SimpleNamespace(scene_name=k[0],t0_token=k[1]) for k in keys]
        monkeypatch.setattr(run,'select_jobs',lambda *a:(jobs,keys))
        monkeypatch.setattr(run.base,'load_cache',lambda *a:({},[]))
        monkeypatch.setattr(run.base,'load_runtime_config',lambda *a:None)
        monkeypatch.setattr(run.base,'make_prepare_config',lambda *a:p.provider.pcfg)
        class Inputs:
            def __init__(self,*a,**k):pass
            def preflight(self,*a):return []
            def history(self,job,kind):
                calls.append((kind,job.t0_token))
                if len(calls)==66:event.set()
                return record[0],raw,np.arange(4)*.5
            def poses(self,job):return np.tile(np.eye(4),(6,1,1))
            def targets(self,job,indices):return [np.full(p.provider.pcfg.grid.shape_hwd,4,np.uint8)]*3
            def moving(self,*a):return [np.ones(p.provider.pcfg.grid.shape_hwd,bool)]*3
        class Predictor:
            def __init__(self,body,pcfg,device,workers):self.provider=SimpleNamespace(pcfg=pcfg,workers=workers)
            def prepare(self,*a,**k):return None
            def forward(self,*a,**k):return None
            def dense(self,*a):return {n:np.full((6,*p.provider.pcfg.grid.shape_hwd),4,np.uint8) for n in ('transport','semantic_refine')}
            def close(self):pass
        monkeypatch.setattr(run,'DenseInputs',Inputs);monkeypatch.setattr(run,'AllPredictor',Predictor)
        argv=['--source-dir',str(folder),'--out-dir',str(tmp_path/'new'),'--device','cpu',
            '--dataroot',str(tmp_path),'--stc-root',str(tmp_path/'stc')]
        for k,path in paths.items():argv+=['--'+k.replace('_','-'),str(path)]
        assert run.main(argv,event)==130 and len(calls)==66
        event.clear();assert run.main(argv+['--resume'],event)==0 and len(calls)==130
        assert run.main(argv+['--resume'],event)==0 and len(calls)==130
        result=json.loads((tmp_path/'new/evaluation.json').read_text())
        assert result['status']=='complete' and len(result['reports']['all4369'])==4
        assert result['reports']['all4369']['stc_gt_age_semantic_refine']['average']['mIoU']==100.
        assert result['progress']==dict(occ=65,stc=65) and source.read_bytes()==before
        text=(tmp_path/'new/summary.txt').read_text()
        assert 'STC=fixed history Memory age' in text and '1.0s' in text and 'NOT FPS' in text
    finally:p.close()
