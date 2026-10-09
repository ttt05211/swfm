"""Bounded ordering/recovery plus real spawned CPU/CUDA predictions, no mock FPS."""
from copy import deepcopy
import pickle
from threading import Event

import numpy as np
import pytest
import torch

from real_motion.waymo_i2world import file_sha256, fingerprint
from real_motion.waymo_i2world_10hz import WaymoI2World10HzSource
from tools.real_motion.waymo_parallel_execution import SpawnPool,evaluate_parallel,paired_parallel_speed
from tools.real_motion.waymo_zero_shot_common import evaluate_windows,restore
from test_waymo_fast_execution import model_fixture
from test_waymo_i2world_10hz import metadata,predict,fixture


def spawned_fixture(tmp_path,device):
    from tools.real_motion.joint_surface_checkpoint_selection import evaluation_payload,AVERAGE_EPOCHS
    source,pcfg,model=model_fixture(tmp_path,shape=(40,40,4))
    infos,poses=metadata(lengths=(12,9))
    for name,data in (('waymo_infos_val.pkl',infos),('cam_infos_vali.pkl',poses)):
        with (source.root/name).open('wb') as handle: pickle.dump(data,handle)
    source=WaymoI2World10HzSource.from_files(source.root,shape=source.shape,cache_mib=1)
    inventory=source.preflight(source.windows)
    checkpoint=tmp_path/'mean.pt'
    prior=dict(positive_weights=model.columns.positive_weight.cpu().tolist())
    payload=evaluation_payload(model.state_dict(),model.configs(),{},prior,
        [dict(epoch=e) for e in AVERAGE_EPOCHS],average=True)
    torch.save(payload,checkpoint)
    spec=dict(waymo_root=str(source.root),raw_free_label=23,frame_cache_mib=1,shape=source.shape,
        windows=len(source.windows),manifest_fingerprint=source.manifest_fingerprint,data=source.metadata,
        inventory=inventory,checkpoint=str(checkpoint),checkpoint_sha256=file_sha256(checkpoint),
        pcfg=pcfg,device=device,workers=2,backend='v2',graphs=device=='cuda',geometry_mib=8,
        surface_chunk=256,parallel_majority=True,history_prefetch=True,implementation={})
    return source,spec


def fake_rows(source,indices):
    from real_motion.waymo_i2world import WaymoMetrics
    rows=[]
    for index in indices:
        window=source.windows[index]; record,raw=source.prediction_inputs(window)
        a,b,edits,_=predict(record,raw,verify=False); targets=source.metric_targets(window)
        counts={}
        for key,dense in (('transport',a),('joint',b)):
            m=WaymoMetrics(); m.add(dense,targets); counts[key]=m.counts.tolist()
        rows.append(dict(index=index,anchor=window.anchor,t0=record['t0_token'],counts=counts,edits=edits,
            stages=dict(window=.1),model_stages=dict(head=.01),exactness_passed=True,pid=1,geometry_cache={}))
    return rows


@pytest.mark.parametrize('bad',['wrong_index','bad_count','negative_time','remove','unverified'])
def test_parallel_bad_chunk_never_partially_commits(tmp_path,bad):
    source=fixture(tmp_path); rows=fake_rows(source,[0,1]); contract=dict(windows=len(source.windows))
    if bad=='wrong_index': rows[1]['index']=0
    if bad=='bad_count': rows[1]['counts']['joint'][0][0][0]+=1
    if bad=='negative_time': rows[1]['stages']['window']=-.1
    if bad=='remove': rows[1]['edits']['removed']=1
    if bad=='unverified': rows[1]['exactness_passed']=False
    class Fake:
        def batches(self,*a,**kw): yield rows
    saved=[]
    with pytest.raises(RuntimeError):
        evaluate_parallel(Fake(),list(range(len(source.windows))),contract,shape=source.shape,save=saved.append)
    assert saved[-1]['completed_windows']==0
    restore(saved[-1],contract,voxel_count=int(np.prod(source.shape)))


def test_parallel_ordered_chunk_stop_resume_same_integer_state(tmp_path):
    source=fixture(tmp_path); contract=dict(windows=len(source.windows)); saved=[]; stopped=Event()
    rows=fake_rows(source,list(range(len(source.windows))))
    class Fake:
        def batches(self,indices,chunk,stop_event=None):
            for b in range(0,len(indices),chunk):
                if stop_event and stop_event.is_set(): return
                yield [rows[i] for i in indices[b:b+chunk]]
    def progress(row):
        if row['window']==3: stopped.set()
    first=evaluate_parallel(Fake(),list(range(len(rows))),contract,shape=source.shape,
        save=saved.append,progress=progress,stop_event=stopped,chunk=4)
    assert first['completed_windows']==4 and first['status']=='stopped'
    prefix=deepcopy(saved[-1])
    result=evaluate_parallel(Fake(),list(range(len(rows))),contract,shape=source.shape,
        saved=prefix,save=saved.append,chunk=4)
    plain=[]; reference=evaluate_windows(source,source.windows,predict,contract,save=plain.append)
    assert result['status']=='complete' and result['reports']==reference['reports']
    assert saved[-1]['counts']==plain[-1]['counts'] and prefix['completed_windows']==4


def test_actual_two_spawned_cpu_models_same_six_bytes_counts_and_bounded_jobs(tmp_path):
    source,spec=spawned_fixture(tmp_path,'cpu')
    d=paired_parallel_speed(spec,list(range(6)),processes=2,chunk=2,repeats=1)
    assert d['counts_exact'] and d['probability_and_six_dense_bytes_exact']
    assert len(d['worker_pids']['fast_v2_parallel'])==2
    pool=SpawnPool(spec,2); saved=[]
    try:
        result=evaluate_parallel(pool,list(range(len(source.windows))),dict(windows=len(source.windows)),
            shape=source.shape,save=saved.append,chunk=4)
        assert result['status']=='complete' and result['completed_windows']==21
        restore(saved[-1],dict(windows=len(source.windows)),voxel_count=int(np.prod(source.shape)))
    finally: pool.close()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='real CUDA required')
def test_actual_two_cuda_processes_with_graphs_same_motion_probability_six_and_counts(tmp_path):
    source,spec=spawned_fixture(tmp_path,'cuda')
    d=paired_parallel_speed(spec,list(range(6)),processes=2,chunk=2,repeats=1)
    assert d['counts_exact'] and len(d['worker_pids']['fast_v2_parallel'])==2
    assert d['probability_and_six_dense_bytes_exact']
    assert file_sha256(spec['checkpoint'])==spec['checkpoint_sha256']


def test_parallel_cli_speed_only_prefix_preserved_then_resume_without_checkpoint_search(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    from real_motion.prepared import PrepareConfig
    from real_motion.geometry import OccupancyGrid
    from tools.real_motion import eval_p0_f9_joint_surface_waymo_10hz_parallel as cli
    from tools.real_motion.eval_p0_f9_joint_surface_waymo_10hz_fast import IMPLEMENTATION_FILES as v1_files
    from tools.real_motion.waymo_zero_shot_common import write_json
    source=fixture(tmp_path/'data'); rows=fake_rows(source,list(range(len(source.windows))))
    ckpt=tmp_path/'weights'/'mean.pt'; ckpt.parent.mkdir(); ckpt.write_bytes(b'FROZEN')
    cfg=tmp_path/'config.yaml'; cfg.write_bytes(b'CONFIG'); stopped=Event(); calls=[]
    monkeypatch.setattr(cli.WaymoI2World10HzSource,'from_files',lambda *a,**kw:source)
    monkeypatch.setattr(cli.torch.cuda,'is_available',lambda:True); monkeypatch.setattr(cli,'SHAPE',source.shape)
    monkeypatch.setattr(cli,'load_runtime_config',lambda p:None)
    monkeypatch.setattr(cli,'make_prepare_config',lambda c:PrepareConfig(
        grid=OccupancyGrid(-40,-40,-1,(.4,)*3,source.shape)))
    monkeypatch.setattr(cli,'load_evaluation_model',lambda *a,**kw:(
        dict(source_epochs=list(cli.AVERAGE_EPOCHS),averaging=True),
        SimpleNamespace(transport=SimpleNamespace(config=SimpleNamespace(history_frames=4)))))
    class FakePool:
        def __init__(self,*a,**kw): pass
        def reset(self,*a,**kw): pass
        def batches(self,indices,chunk,stop_event=None):
            for b in range(0,len(indices),chunk):
                if stop_event is not None and stop_event.is_set(): return
                chosen=indices[b:b+chunk]; calls.extend(chosen)
                if len(calls)==4: stopped.set()
                yield [rows[i] for i in chosen]
        def close(self): pass
    monkeypatch.setattr(cli,'SpawnPool',FakePool)
    base=['--waymo-root',str(source.root),'--checkpoint',str(ckpt),'--config',str(cfg),
        '--expected-scenes','2','--speed-windows','0','--no-graphs','--parallel-chunk','4']
    part=tmp_path/'part'; assert cli.main(stopped,argv=[*base,'--out-dir',str(part)])==0
    old=json.loads((part/'contract.json').read_text()); old['fast_execution']={'protocol':'V1'}
    old['implementation']={k:v for k,v in old['implementation'].items() if k in v1_files}
    state=json.loads((part/'state.json').read_text()); state.pop('fingerprint')
    state['contract_fingerprint']=fingerprint(old); state['fingerprint']=fingerprint(state)
    v1=tmp_path/'v1'; v1.mkdir(); write_json(v1/'contract.json',old); write_json(v1/'state.json',state)
    source_bytes={p.name:p.read_bytes() for p in v1.iterdir()}; out=tmp_path/'parallel'
    def speed(*a,**kw):
        return dict(windows=4,seconds_per_window=dict(fast_v1_serial=.5,fast_v2_parallel=.25),speedup=2.)
    monkeypatch.setattr(cli,'paired_parallel_speed',speed)
    assert cli.main(argv=[*base,'--out-dir',str(out),'--continue-from-dir',str(v1),
                         '--speed-only','--speed-windows','4'])==0
    assert len(calls)==4 and json.loads((out/'state.json').read_text())['completed_windows']==4
    assert not (out/'waymo_validation.json').exists()
    with monkeypatch.context() as guard:
        guard.setattr(cli.original,'resolve_checkpoint',lambda *a:pytest.fail('saved checkpoint required'))
        assert cli.main(argv=[*base[:2],*base[4:],'--out-dir',str(out),'--resume'])==0
    result=json.loads((out/'waymo_validation.json').read_text())
    assert result['completed_windows']==17 and result['reused_prefix_windows']==4
    assert sorted(calls)==list(range(17)) and len(calls)==17
    assert all((v1/k).read_bytes()==v for k,v in source_bytes.items()) and ckpt.read_bytes()==b'FROZEN'
    with pytest.raises(RuntimeError,match='contract'):
        cli.main(argv=[*base,'--out-dir',str(out),'--resume','--parallel-chunk','8'])
