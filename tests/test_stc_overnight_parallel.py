import copy
import hashlib
import json
import threading
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from tools.stc_overnight_parallel import common as c,run
from tools.stc_overnight import verify_all as old


def counts(keys,names,volume=4):
    out={n:old.metrics.empty(3) for n in names}
    for i,key in enumerate(keys):
        gt=np.full((2,2,1),17,np.uint8);gt[0,0,0]=1
        for j,n in enumerate(names):
            pred=gt.copy()
            if j%2:pred[0,1,0]=1
            for h in range(3):old.metrics.add(out[n],h,pred,gt)
    return c.serialized(out)


def legacy(keys):
    return dict(protocol='frozen_same_body_STC_KTA_recency_v1',keys=keys,population='dev64',portable_subset=True,
        model='clean1000',source_policy='baseline',source_sha256='body',
        manifest_sha256=hashlib.sha256(b'manifest').hexdigest(),metadata={},
        known='STC dense-valid; NO GT visibility',metric_mask='none',future_GT_ego_conditioned=True,
        posterior_alignment='fixed',history_support='fixed',
        implementation={n:old.base.sha256(Path(old.__file__).parent/n) for n in ['verify_all.py','replay.py','bundle.py','calibration.py']})


def test_integer_merge_imports_prefix_once_and_rejects_wrong_shards():
    keys=[[f'scene{i}',str(i)] for i in range(9)];contract=legacy(keys);names=c.routes(contract)
    imported=dict(keys=keys[:3],counts=counts(keys[:3],names));job=dict(legacy_contract=contract,voxel_count=4)
    states=[]
    for rank in range(2):
        ks=keys[3:][rank::2]
        states.append(dict(job_sha256=old.base.stable_json_fingerprint(job),rank=rank,keys=ks,counts=counts(ks,names)))
    total,seen=c.merge(job,imported,states,complete=True)
    assert seen==keys
    for n in names:
        for field in total[n]:np.testing.assert_array_equal(total[n][field],np.asarray(counts(keys,names)[n][field]))
    wrong=copy.deepcopy(states);wrong[1]['keys'][0]=wrong[0]['keys'][0]
    with pytest.raises(ValueError):c.merge(job,imported,wrong,complete=True)
    wrong=copy.deepcopy(states);wrong[0]['counts'][names[0]]['confusion']=np.asarray(wrong[0]['counts'][names[0]]['confusion'],float).tolist()
    with pytest.raises(ValueError,match='integer'):c.merge(job,imported,wrong)


def test_stop_match_is_exact_module_and_output(tmp_path):
    source=tmp_path/'old'
    assert c.matching_command(['python','-m','tools.stc_overnight.verify_all','--out',str(source)],source,tmp_path)
    assert c.matching_command(['python','-m','tools.stc_overnight.verify_all','--out=old'],source,tmp_path)
    assert not c.matching_command(['python','-m','tools.stc_overnight_parallel.run','--out',str(source)],source,tmp_path)
    assert not c.matching_command(['python','-m','tools.stc_overnight.verify_all','--out',str(tmp_path/'other')],source,tmp_path)


def test_h6_forecast_keeps_all_original_and_supported_routes():
    from test_shared_source_track import fixture
    torch.set_num_threads(1);p,rec,raw=fixture()
    try:
        torch.nn.init.normal_(p.body.columns.readout[-1].weight,std=.01)
        row=p.prepare(rec,raw,times=np.arange(4)*.5);poses=np.tile(np.eye(4),(6,1,1));poses[:,0,3]=np.arange(6)*.05
        bias=[.1]*17+[0.]
        expected=old.forecast(p,row,poses,bias,True)
        actual=run.forecast(p,row,poses,bias,True,6)
        assert actual.keys()==expected.keys()
        for name in expected:np.testing.assert_array_equal(expected[name],actual[name])
    finally:p.close()


def fixture_source(tmp_path):
    source=tmp_path/'source';source.mkdir();keys=[[f'scene{i}',str(i)] for i in range(64)]
    contract=legacy(keys);names=c.routes(contract)
    c.write(source/'contract.json',contract)
    c.write(source/'unlabeled_posterior_fit.json',dict(source_sha256='body',bias=[0.]*18))
    state=dict(contract=contract,keys=keys[:16],counts=counts(keys[:16],names),
        calibration_sha256=old.base.sha256(source/'unlabeled_posterior_fit.json'))
    c.write(source/'state.json',state)
    return source,contract


def test_parent_two_same_gpu_processes_import_resume_and_no_source_writes(tmp_path,monkeypatch):
    source,contract=fixture_source(tmp_path);before=(source/'state.json').read_bytes();out=tmp_path/'parallel';children=[]
    class FakeBundle:
        def __init__(self,*args,**kwargs):
            self.pcfg=SimpleNamespace(grid=SimpleNamespace(shape_hwd=(2,2,1)));self.zip=SimpleNamespace(read=lambda n:b'manifest')
        def load_model(self,*args):return torch.nn.Linear(1,1),dict(source_sha256='body'),'baseline'
        def close(self):pass
    class Child:
        def __init__(self,argv,**kwargs):
            self.rank=int(argv[-1]);self.returncode=None;self.pid=1000+self.rank;children.append((argv,kwargs))
        def poll(self):
            if self.returncode is None:
                job=c.read(out/'contract.json');ks=contract['keys'][16:][self.rank::2]
                path=out/f'shard_{self.rank:02d}';path.mkdir(exist_ok=True)
                c.write(path/'state.json',dict(job_sha256=old.base.stable_json_fingerprint(job),rank=self.rank,
                    keys=ks,counts=counts(ks,c.routes(contract)),status='complete'))
                self.returncode=0
            return self.returncode
        def wait(self):return self.poll()
        def terminate(self):self.returncode=130
    monkeypatch.setattr(run,'Bundle',FakeBundle);monkeypatch.setattr(run.subprocess,'Popen',Child)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    argv=['--source-dir',str(source),'--out-dir',str(out),'--bundle',str(tmp_path/'bundle.zip'),'--device','cpu']
    assert run.main(argv)==0
    r=c.read(out/'results.json');assert r['windows']==64 and r['imported_complete_windows']==16
    assert r['new_complete_windows']==48 and len(children)==2
    assert all(kwargs['env']['CUDA_VISIBLE_DEVICES']=='0' for _,kwargs in children)
    assert all(v is False for v in r['full_target_verified_by_route'].values())
    assert (source/'state.json').read_bytes()==before
    expected=counts(contract['keys'],c.routes(contract));state=c.read(out/'state.json')
    assert state['counts']==expected
    assert run.main(argv+['--resume'])==0 and c.read(out/'state.json')['counts']==expected
    fit=out/'source_snapshot/unlabeled_posterior_fit.json';fit.write_text('{}')
    with pytest.raises(ValueError,match='snapshot changed'):run.main(argv+['--resume'])


def test_worker_labels_only_after_both_forecasts_and_completed_resume_skips(tmp_path,monkeypatch):
    source,contract=fixture_source(tmp_path);out=tmp_path/'parallel';out.mkdir()
    _,hashes=c.snapshot(source,out,contract,4)
    job=dict(legacy_contract=contract,source_snapshot_sha256=hashes,imported_windows=62,voxel_count=4,
        device='cpu',cpu_workers=1,horizon_group=6,data_paths=dict(bundle='unused'))
    c.write(out/'contract.json',job);events=[]
    class FakeBundle:
        def __init__(self,*args,**kwargs):
            self.pcfg=SimpleNamespace(grid=SimpleNamespace(shape_hwd=(2,2,1)))
            self.records=[dict(role='reference64',scene_name=s,t0_token=t,future_poses=np.tile(np.eye(4),(6,1,1))) for s,t in contract['keys']]
        def load_model(self,*args):events.append('model');return torch.nn.Linear(1,1),dict(source_sha256='body'),'baseline'
        def history(self,r):return r,{},np.arange(4)*.5
        def targets(self,r):
            assert events[-2:]==['forecast','forecast'];events.append('labels');return np.zeros((3,2,2,1),np.uint8)
        def close(self):pass
    class P:
        def __init__(self,*args,**kwargs):self.provider=None
        def prepare(self,r,*args,**kwargs):return r
        def close(self):pass
    def forecast(p,row,poses,bias,support,group):
        assert group==6;events.append('forecast')
        names=['transport','semantic_refine']+(['calibrated_semantic_refine'] if bias is not None else [])
        if support:names+=['supported_'+n for n in list(names)]
        return {n:np.zeros((6,2,2,1),np.uint8) for n in names}
    monkeypatch.setattr(run,'Bundle',FakeBundle);monkeypatch.setattr(run,'Predictor',P)
    monkeypatch.setattr(run,'forecast',forecast);monkeypatch.setattr(run.old,'candidate',lambda row,*args:(row,{}))
    argv=['--source-dir',str(source),'--out-dir',str(out),'--worker-rank','0']
    assert run.main(argv)==0;assert events.count('labels')==1
    assert run.main(argv)==0;assert events.count('model')==1 and events.count('labels')==1
