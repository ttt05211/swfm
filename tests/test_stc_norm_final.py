import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace
import threading
import numpy as np
import pytest
import torch
from tools.stc_norm_final import common as c,run,probe
from tools.stc_overnight_parallel.common import serialized
from tools.stc_overnight.bundle import Bundle


def counts(keys,volume=4):
    result={n:c.metrics.empty(3) for n in c.ROUTES}
    for _ in keys:
        gt=np.full((2,2,1),17,np.uint8);gt[0,0,0]=4
        for n in c.ROUTES:
            for h in range(3):c.metrics.add(result[n],h,gt,gt)
    return serialized(result)


@pytest.mark.parametrize('workers',(1,2,4))
def test_uneven_shards_sum_integer_counts_and_reject_duplicate_or_partial(workers):
    keys=[[f'scene{i}',str(i)] for i in range(11)];job=dict(keys=keys,processes=workers,voxel_count=4)
    states=[]
    for rank in range(workers):
        part=keys[rank::workers]
        states.append(dict(job_sha256=c.base.stable_json_fingerprint(job),rank=rank,keys=part,counts=counts(part),status='complete'))
    result,seen=c.merge(job,states,True);assert seen==keys
    assert serialized(result)==counts(keys)
    wrong=copy.deepcopy(states);wrong[0]['keys']=wrong[0]['keys'][:-1]
    with pytest.raises(ValueError):c.merge(job,wrong,True)
    if workers>1:
        wrong=copy.deepcopy(states);wrong[1]['keys'][0]=wrong[0]['keys'][0]
        with pytest.raises(ValueError):c.merge(job,wrong,True)


def test_frozen_patch_only_changes_named_norm192_and_restore_is_exact():
    from tools.shared_history_experiments.refine.model import SemanticRefiner
    body=torch.nn.Module();body.columns=SemanticRefiner();patch=c.read(c.ROOT/'norm192.json')
    before={n:p.detach().clone() for n,p in body.named_parameters()}
    params,original,adapted=c.norm_state(body,patch,c.BODY_SHA)
    c.assign(params,adapted)
    assert sum(p.numel() for p in params.values())==192
    assert any(not torch.equal(p,before[n]) for n,p in params.items())
    assert all(torch.equal(p,before[n]) for n,p in body.named_parameters() if n not in params)
    c.assign(params,original);assert all(torch.equal(p,before[n]) for n,p in body.named_parameters())
    wrong=copy.deepcopy(patch);wrong['state']['columns.empty']=[0.]*16
    with pytest.raises(ValueError,match='ABI'):c.norm_state(body,wrong,c.BODY_SHA)
    with pytest.raises(ValueError,match='fixed'):c.norm_state(body,patch,'wrongbody')


def test_paired_forecast_matches_existing_route_and_keeps_source_unchanged():
    from test_shared_source_track import fixture
    p,rec,raw=fixture();torch.set_num_threads(1)
    try:
        row=p.prepare(rec,raw,times=np.arange(4)*.5);row,_=run.old.candidate(row,p.provider,np.arange(4)*.5)
        parameters={n:v for n,v in p.body.columns.named_parameters() if '.block.0.' in n}
        original={n:v.detach().clone() for n,v in parameters.items()}
        adapted={n:v+.01 for n,v in original.items()};patch=parameters,original,adapted
        poses=np.tile(np.eye(4),(6,1,1));bias=[.2]*17+[0.]
        expected=run.old.forecast(p,row,poses,bias,True)
        actual=run.forecast(p,row,poses,patch,bias,6)
        np.testing.assert_array_equal(actual['shared/transport'],expected['supported_transport'])
        np.testing.assert_array_equal(actual['previous/semantic_refine'],expected['supported_calibrated_semantic_refine'])
        assert all(torch.equal(parameters[n],v) for n,v in original.items())
        assert all(v.shape==(6,*raw['history_occ'].shape[1:]) for v in actual.values())
    finally:p.close()


@pytest.mark.parametrize('workers',(2,4))
def test_parent_launches_all_workers_on_same_gpu_and_complete_resume_no_relaunch(tmp_path,monkeypatch,workers):
    keys=[[f'scene{i}',str(i)] for i in range(64)];out=tmp_path/'out';children=[]
    class B:
        def __init__(self,*args,**kwargs):
            self.pcfg=SimpleNamespace(grid=SimpleNamespace(shape_hwd=(2,2,1)))
            self.records=[dict(role='reference64',scene_name=s,t0_token=t) for s,t in keys]
            self.zip=SimpleNamespace(read=lambda _:b'manifest')
        def load_model(self,*args):return torch.nn.Linear(1,1),dict(source_sha256=c.BODY_SHA),'baseline'
        def close(self):pass
    class Child:
        def __init__(self,argv,**kwargs):
            self.rank=int(argv[-1]);self.returncode=None;children.append(kwargs)
        def poll(self):
            if self.returncode is None:
                job=c.read(out/'contract.json');part=job['keys'][self.rank::workers]
                folder=out/f'shard_{self.rank:02d}';folder.mkdir(exist_ok=True)
                c.write(folder/'state.json',dict(job_sha256=c.base.stable_json_fingerprint(job),rank=self.rank,
                    keys=part,counts=counts(part),status='complete',measured_windows=len(part),measured_seconds=1.))
                self.returncode=0
            return self.returncode
        def wait(self):return self.poll()
        def terminate(self):self.returncode=130
    monkeypatch.setattr(run,'Bundle',B);monkeypatch.setattr(run,'norm_state',lambda *args:None)
    monkeypatch.setattr(run.subprocess,'Popen',Child);monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    args=['--out',str(out),'--bundle','unused','--portable-subset','--processes',str(workers),'--device','cpu']
    assert run.main(args)==0;assert len(children)==workers
    assert all(child['env']['CUDA_VISIBLE_DEVICES']=='0' for child in children)
    r=c.read(out/'results.json');assert r['windows']==64 and r['contract']['physical_GPUs']==0
    assert run.main(args+['--resume'])==0;assert len(children)==workers
    p=out/'frozen/norm192.json';p.write_text('{}')
    with pytest.raises(ValueError,match='asset changed'):run.main(args+['--resume'])


def test_probe_command_is_same_method_same_windows_one_gpu_and_four_processes():
    a=SimpleNamespace(bundle='b',dev_cache='c',dev_info='i',dataroot='d',stc_root='s',horizon_group=6,
        cpu_workers=2,device='cuda',windows=32,portable_subset=False)
    argv=probe.command(a,Path('out'),4)
    assert argv[argv.index('--processes')+1]=='4'
    assert argv[argv.index('--limit')+1]=='32'
    assert '--probe' in argv and '--worker-gpus' not in argv


def test_speed_choice_never_selects_on_quality_and_keeps_oom_or_noisy4_off():
    rows=[dict(processes=w,exit_code=0,steady_windows_per_second=v) for w,v in [(2,2.),(4,3.),(4,3.),(2,2.)]]
    assert probe.recommendation(rows)[0]==4
    rows[2]['steady_windows_per_second']=1.9;assert probe.recommendation(rows)[0]==2
    rows[2]['exit_code']=1;assert probe.recommendation(rows)[0]==2
    rows[0]['exit_code']=1;assert probe.recommendation(rows)[0] is None


def test_worker_reads_target_after_both_readouts_and_complete_resume_skips(tmp_path,monkeypatch):
    out=tmp_path/'out';out.mkdir();(out/'frozen').mkdir();events=[]
    for name in ('norm192.json','posterior_fit.json'):(out/'frozen'/name).write_bytes((c.ROOT/name).read_bytes())
    job=dict(keys=[['scene','token']],processes=1,voxel_count=4,device='cpu',source_policy='baseline',
        portable_subset=True,paths=dict(bundle='unused'),cpu_workers=1,horizon_group=6,probe=False,
        implementation=run.implementation(),assets={n:c.base.sha256(out/'frozen'/n) for n in ('norm192.json','posterior_fit.json')})
    c.write(out/'contract.json',job)
    class B:
        def __init__(self,*args,**kwargs):self.records=[dict(role='reference64',scene_name='scene',t0_token='token',future_poses=np.tile(np.eye(4),(6,1,1)))];self.pcfg=None
        def load_model(self,*args):events.append('load');return torch.nn.Linear(1,1),dict(source_sha256=c.BODY_SHA),'baseline'
        def history(self,r):return r,{},np.arange(4)*.5
        def targets(self,r):assert events[-1]=='both_readouts';events.append('labels');return np.zeros((3,2,2,1),np.uint8)
        def close(self):pass
    class P:
        def __init__(self,body,*args,**kwargs):self.body=body;self.provider=None
        def prepare(self,r,*args,**kwargs):return r
        def close(self):pass
    def forecast(*args):events.append('both_readouts');return {n:np.zeros((6,2,2,1),np.uint8) for n in c.ROUTES}
    monkeypatch.setattr(run,'Bundle',B);monkeypatch.setattr(run,'Predictor',P);monkeypatch.setattr(run,'norm_state',lambda *args:({}, {}, {}))
    monkeypatch.setattr(run,'forecast',forecast);monkeypatch.setattr(run.old,'candidate',lambda row,*args:(row,{}))
    assert run.worker(out,0,threading.Event())==0;assert events==['load','both_readouts','labels']
    assert run.worker(out,0,threading.Event())==0;assert events==['load','both_readouts','labels']
