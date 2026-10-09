"""Geometry/eager/native parity, bounded cache, causality and continuation."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from threading import Event

import numpy as np
import pytest
import torch

from real_motion.canonical_causal_repair import CanonicalEvidence, FEATURE_DIM, STATIC, build_canonical_evidence
from real_motion.geometry import OccupancyGrid
from real_motion.prepared import PrepareConfig
from real_motion.strong_w2det import StrongW2DetConfig
from real_motion.surface_canonical_repair import SurfaceAtlas
from real_motion.waymo_geometry_execution import (FrameGeometryCache, ChunkedSurfaceAtlas,
    GeometryPrefetchSource, build_cached_evidence)
from real_motion.waymo_i2world import fingerprint
from real_motion.waymo_i2world_10hz import PROTOCOL
from tools.real_motion.waymo_fast_execution import (FastWaymoSurfaceProvider, FastSurfaceBlockExecution,
    migrate_state, paired_speed, read_continuation)
from tools.real_motion.waymo_zero_shot_common import evaluate_windows, restore, write_json
from test_waymo_i2world_10hz import fixture, predict


def model_fixture(tmp_path, device='cpu', shape=(32,32,4)):
    from real_motion.local_st_world_model_v17 import LocalSTWMV17Config
    from real_motion.joint_surface_ccr import JointSurfaceCCR
    source=fixture(tmp_path/'data', shape=shape, lengths=(12,9))
    pcfg=PrepareConfig(grid=OccupancyGrid(-6.4,-6.4,-1,(.4,)*3,shape))
    torch.manual_seed(71)
    model=JointSurfaceCCR(LocalSTWMV17Config(history_frames=4,d_model=16,semantic_dim=4,
        blocks=1,decoder_blocks=1),width=16,z_bins=4).to(device).eval().requires_grad_(False)
    return source,pcfg,model


def test_cache_is_content_pose_visibility_keyed_readonly_bounded_and_evicted(tmp_path):
    source=fixture(tmp_path,shape=(16,16,4),lengths=(10,)); source.preflight(source.windows)
    grid=OccupancyGrid(-3.2,-3.2,-1,(.4,)*3,source.shape)
    cache=FrameGeometryCache(grid,StrongW2DetConfig(free_label=17),ram_mib=1,max_entries=2)
    record,raw=source.prediction_inputs(source.windows[4])
    a=cache.get('SAME',raw['history_occ'][0],raw['history_observed'][0],raw['history_poses'][0])
    b=cache.get('SAME',raw['history_occ'][0],raw['history_observed'][0],raw['history_poses'][0])
    assert a is b and cache.hits==1
    with pytest.raises(ValueError): a.static_world[0,0]=10
    with pytest.raises(TypeError): a.components[0]['class_id']=6
    pose=raw['history_poses'][0].copy(); pose[0,3]+=1
    c=cache.get('SAME',raw['history_occ'][0],raw['history_observed'][0],pose)
    assert c is not a and not np.array_equal(c.static_world,a.static_world)
    occ=raw['history_occ'][0].copy(); occ[0,0,0]=17
    d=cache.get('SAME',occ,raw['history_observed'][0],raw['history_poses'][0])
    assert d is not a and cache.evictions>=1 and len(cache.entries)==2
    assert cache.bytes<=cache.limit and cache.stats()['persistent_writes']==0
    zero=FrameGeometryCache(grid,StrongW2DetConfig(free_label=17),ram_mib=0)
    zero.window(record,raw); assert zero.bytes==0 and not zero.entries


def test_prefetch_cache_build_does_not_lock_logging_and_singleflight(tmp_path,monkeypatch):
    source=fixture(tmp_path); record,raw=source.prediction_inputs(source.windows[0])
    cache=FrameGeometryCache(OccupancyGrid(-40,-40,-1,(.4,)*3,source.shape),
                             StrongW2DetConfig(free_label=17),ram_mib=8)
    started=Event(); release=Event(); build=cache._build
    def held(*args):
        started.set(); assert release.wait(5)
        return build(*args)
    monkeypatch.setattr(cache,'_build',held)
    def get(): return cache.get('single',raw['history_occ'][0],raw['history_observed'][0],raw['history_poses'][0])
    with ThreadPoolExecutor(max_workers=3) as pool:
        first=pool.submit(get); assert started.wait(5)
        second=pool.submit(get)
        try:
            stats=pool.submit(cache.stats).result(timeout=2)
            assert stats['pending_frames']==1  # logging must not force prefetch completion
        finally: release.set()
        assert first.result() is second.result()
    assert cache.misses==1 and cache.hits==1 and not cache.inflight


@pytest.mark.parametrize('chunk',[256,4096,65536])
@pytest.mark.parametrize('parallel',[False,True])
def test_surface_chunked_fit_descriptor_bytes_exact_with_ties_empty_and_stacked(chunk,parallel):
    rng=np.random.default_rng(2); n=9803
    points=rng.uniform(-8,8,(n,3)); points[:,2]*=.025
    points[20:35]=points[19]  # distance ties/duplicate layers must retain cKDTree order
    classes=np.where(np.arange(n)%3,11,13).astype(np.uint8)
    actors=np.full(n,STATIC,np.int32); actors[::103]=0
    seen=rng.random((n,4))>.3; seen[::7]=False
    evidence=CanonicalEvidence(np.zeros((n,FEATURE_DIM),np.float32),np.zeros((n,4),np.uint8),
        actors,classes,points,seen,{})
    pose=np.eye(4); pose[:2,:2]=[[np.cos(.2),-np.sin(.2)],[np.sin(.2),np.cos(.2)]]
    pose[:3,3]=[1e5,-2e5,.7]; grid=OccupancyGrid()
    for take in (np.arange(n),np.flatnonzero(classes==11),np.empty(0,np.int64)):
        e=replace(evidence,features=evidence.features[take],labels=evidence.labels[take],
            actor=actors[take],classes=classes[take],world=points[take],presence=seen[take])
        ref=SurfaceAtlas(e.world,e.classes,e.presence,e.actor,pose,grid).describe(e)
        atlas=ChunkedSurfaceAtlas(e.world,e.classes,e.presence,e.actor,pose,grid); atlas.chunk_rows=chunk
        with ThreadPoolExecutor(max_workers=3) as pool:
            atlas.fit_pool=pool if parallel else None
            actual=atlas.describe(e)
        assert np.array_equal(ref,actual)


def test_prefetch_order_future_access_only_after_predict_and_exact_resume(tmp_path):
    source=fixture(tmp_path,shape=(16,16,4),lengths=(12,8)); source.preflight(source.windows)
    grid=OccupancyGrid(-3.2,-3.2,-1,(.4,)*3,source.shape)
    cache=FrameGeometryCache(grid,StrongW2DetConfig(free_label=17),ram_mib=8)
    stream=GeometryPrefetchSource(source,source.windows,cache)
    calls=[]; raw_targets=source.metric_targets
    def targets(w):
        assert calls[-1]==w.anchor
        return raw_targets(w)
    source.metric_targets=targets
    def pred(record,raw,**kw):
        assert len(raw['_waymo_frame_geometry'])==4 and raw['future_gt_occ'] is None
        calls.append(len(calls))
        return predict(record,raw,**kw)
    contract=dict(protocol=PROTOCOL,windows=len(source.windows)); event=Event(); saved={}
    def progress(row):
        if row['window']==5: event.set()
    try:
        part=evaluate_windows(stream,source.windows,pred,contract,stop_event=event,progress=progress,
                              save=lambda s:saved.update(s))
        assert part['completed_windows']==5 and part['status']=='stopped'
    finally: stream.close()
    stream=GeometryPrefetchSource(source,source.windows,cache)
    try:
        result=evaluate_windows(stream,source.windows,pred,contract,saved=saved)
    finally: stream.close()
    source.metric_targets=raw_targets
    full=evaluate_windows(source,source.windows,predict,contract)
    assert result['reports']==full['reports'] and len(calls)==len(source.windows)
    assert cache.hits>cache.misses and cache.stats()['entries']<=32


@pytest.mark.parametrize('mode',['numpy','native_parallel'])
def test_real_model_all_six_probability_evidence_inputs_registrations_exact(tmp_path,mode):
    from tools.real_motion.eval_p0_f9_joint_surface_waymo import WaymoSurfaceProvider
    from tools.real_motion.joint_surface_long_rollout_common import SurfaceBlockExecution,verify_first_block
    from tools.real_motion import joint_long_rollout_common as rollout
    from real_motion.waymo_i2world import WaymoMetrics
    source,pcfg,model=model_fixture(tmp_path); source.preflight(source.windows)
    original=WaymoSurfaceProvider(model,pcfg,'cpu',2)
    fast=FastWaymoSurfaceProvider(model,pcfg,'cpu',2,geometry_mib=8)
    ref=SurfaceBlockExecution(original,mode=mode,workers=2,query_workers=2,graphs=False)
    opt=FastSurfaceBlockExecution(fast,mode=mode,workers=2,query_workers=2,graphs=False,surface_chunk=256)
    versions={k:v.clone() for k,v in model.state_dict().items()}
    old_counts,fast_counts=WaymoMetrics(),WaymoMetrics()
    threads=torch.get_num_threads(); torch.set_num_threads(1)
    try:
        for i in (0,4,5,11,12,20):
            record,raw=source.prediction_inputs(source.windows[i])
            a=original.prepare_columns(None,record,include_gt=False,raw_window=raw)
            record,raw=source.prediction_inputs(source.windows[i])
            b=fast.prepare_columns(None,record,include_gt=False,raw_window=raw)
            rollout.assert_four_inputs_equal(a.state['rec'],b.state['rec'])
            assert a.source_audit==b.source_audit
            for ar,br in zip(a.registrations,b.registrations):
                for x,y in zip(ar,br):
                    assert (x is None)==(y is None)
                    if x is not None: assert all(np.array_equal(u,v) for u,v in zip(x,y))
            plain=build_canonical_evidence(a,pcfg.grid)
            cached=build_cached_evidence(b,pcfg.grid,kernels=opt.cpu.kernels,executor=opt.cpu.pool)
            assert plain.audit==cached.audit
            for field in ('features','labels','world','actor','classes','presence'):
                assert np.array_equal(getattr(plain,field),getattr(cached,field)),field
            ad,_,_,ap=ref.predict(a); bd,_,_,bp=opt.predict(b)
            assert np.array_equal(ap,bp) and all(np.array_equal(x,y) for x,y in zip(ad,bd))
            verify_first_block(fast,b.state['rec'],b,bd,bp,opt)
            targets=source.metric_targets(source.windows[i])
            old_counts.add(ad,targets); fast_counts.add(bd,targets)
        assert np.array_equal(old_counts.counts,fast_counts.counts) and old_counts.report()==fast_counts.report()
        assert all(torch.equal(versions[k],v) for k,v in model.state_dict().items())
    finally:
        ref.close(); opt.close(); torch.set_num_threads(threads)


def test_migration_reads_original_state_and_rejects_any_semantic_source_change(tmp_path):
    source=fixture(tmp_path/'data'); contract=dict(protocol=PROTOCOL,windows=len(source.windows),
        thresholds=[.5,None],checkpoint_sha256='FROZEN',cpu_workers=2,implementation={'old.py':'ABC'},data={'cadence':.1})
    event=Event(); saved={}
    evaluate_windows(source,source.windows,predict,contract,stop_event=event,
        progress=lambda r:event.set() if r['window']==4 else None,save=lambda v:saved.update(v))
    original=tmp_path/'original'; original.mkdir()
    write_json(original/'contract.json',contract); write_json(original/'state.json',saved)
    before={p.name:p.read_bytes() for p in original.iterdir()}
    old,state,receipt=read_continuation(original)
    new={**contract,'cpu_workers':4,'implementation':{**contract['implementation'],'fast.py':'XYZ'},
         'fast_execution':{'cache_mib':1},'execution_migration':receipt}
    converted=migrate_state(old,state,new,shape=source.shape)
    assert restore(converted,new,voxel_count=np.prod(source.shape))['completed_windows']==4
    continued=evaluate_windows(source,source.windows,predict,new,saved=converted)
    complete=evaluate_windows(source,source.windows,predict,contract)
    assert continued['reports']==complete['reports']
    assert all((original/n).read_bytes()==v for n,v in before.items())
    for change in ({'protocol':'2Hz'},{'thresholds':[.3,None]},{'checkpoint_sha256':'DIFFERENT'},
                   {'data':{'cadence':.5}},{'windows':10},{'implementation':{'old.py':'WRONG','fast.py':'XYZ'}}):
        with pytest.raises(RuntimeError): migrate_state(old,state,{**new,**change},shape=source.shape)
    broken=deepcopy(state); broken['counts']['joint'][0][0][0]+=1
    with pytest.raises(RuntimeError): migrate_state(old,broken,new,shape=source.shape)


def test_paired_speed_runs_six_frames_without_any_gt_or_model_writes(tmp_path,monkeypatch):
    source,pcfg,model=model_fixture(tmp_path)
    monkeypatch.setattr(source,'metric_targets',lambda w:pytest.fail('speed loaded future GT'))
    before={k:v.clone() for k,v in model.state_dict().items()}
    threads=torch.get_num_threads(); torch.set_num_threads(1)
    try:
        result=paired_speed(source,source.windows[4:8],model,pcfg,'cpu',workers=2,graphs=False,
            parallel_majority=False,surface_chunk=256,geometry_mib=8,repeats=2)
        assert result['probability_and_six_dense_bytes_exact'] and result['no_saved_scientific_updates']
        assert result['full_population_equality_checks']==4 and result['speedup']>0
        assert all(len(r)==2 for r in result['repeat_seconds_per_window'].values())
        assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())
    finally: torch.set_num_threads(threads)


def test_old_two_and_ten_hz_hash_files_not_edited():
    # Frozen source checksums are also enforced by the migration bridge. This
    # regression explicitly keeps new fast dependencies out of the old contract.
    from tools.real_motion.eval_p0_f9_joint_surface_waymo import IMPLEMENTATION_FILES as two
    from tools.real_motion.eval_p0_f9_joint_surface_waymo_10hz import IMPLEMENTATION_FILES as ten
    from tools.real_motion.eval_p0_f9_joint_surface_waymo_10hz_fast import IMPLEMENTATION_FILES as fast
    assert set(two)<set(ten)<set(fast)
    assert 'real_motion/waymo_geometry_execution.py' not in ten
    assert 'tools/real_motion/waymo_fast_execution.py' not in ten
    assert 'tools/real_motion/eval_p0_f9_joint_surface_waymo_10hz_fast.py' not in ten


def test_fast_cli_migration_then_same_contract_resume_is_readonly(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from tools.real_motion import eval_p0_f9_joint_surface_waymo_10hz_fast as cli
    from tools.real_motion.eval_p0_f9_joint_surface_waymo_10hz import IMPLEMENTATION_FILES as old_files
    source=fixture(tmp_path/'data'); ckpt=tmp_path/'weights'/'mean.pt'; ckpt.parent.mkdir()
    ckpt.write_bytes(b'FROZEN WEIGHTS'); config=tmp_path/'config.yaml'; config.write_bytes(b'CONFIG')
    monkeypatch.setattr(cli.WaymoI2World10HzSource,'from_files',lambda *a,**kw:source)
    monkeypatch.setattr(cli.torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(cli,'SHAPE',source.shape)
    monkeypatch.setattr(cli,'load_runtime_config',lambda p:None)
    pcfg=PrepareConfig(grid=OccupancyGrid(-40,-40,-1,(.4,)*3,source.shape))
    monkeypatch.setattr(cli,'make_prepare_config',lambda c:pcfg)
    monkeypatch.setattr(cli,'load_evaluation_model',lambda *a,**kw:(
        dict(source_epochs=list(cli.AVERAGE_EPOCHS),averaging=True),
        SimpleNamespace(transport=SimpleNamespace(config=SimpleNamespace(history_frames=4)))))
    class Provider:
        def __init__(self,joint,pcfg,device,workers,**kw):
            self.geometry=FrameGeometryCache(pcfg.grid,StrongW2DetConfig(free_label=17),ram_mib=1)
            self.fast_prepare_stages={}
        def prepare_columns(self,s,record,*,include_gt,raw_window):
            assert not include_gt and raw_window['future_gt_occ'] is None
            p=predict(record,raw_window,verify=False)
            return SimpleNamespace(baseline=p[0],state=dict(rec=record),raw=raw_window)
    calls=[]; event=Event()
    class Execution:
        def __init__(self,*a,**kw): pass
        def predict(self,prep):
            calls.append(prep.state['rec']['t0_token'])
            if len(calls)==3: event.set()
            return prep.baseline,dict(added=0,removed=0),{},None
        def close(self): pass
    monkeypatch.setattr(cli,'FastWaymoSurfaceProvider',Provider)
    monkeypatch.setattr(cli,'FastSurfaceBlockExecution',Execution)
    checked=[]; monkeypatch.setattr(cli,'verify_first_block',lambda *a:checked.append(True))
    base=['--waymo-root',str(source.root),'--checkpoint',str(ckpt),'--config',str(config),
          '--expected-scenes','2','--speed-windows','0','--no-graphs','--geometry-cache-mib','1']
    part=tmp_path/'part'; assert cli.main(event,argv=[*base,'--out-dir',str(part)])==0
    original_dir=tmp_path/'old_original'; original_dir.mkdir()
    old=json.loads((part/'contract.json').read_text()); old.pop('fast_execution')
    old['cpu_workers']=2; old['implementation']={k:v for k,v in old['implementation'].items() if k in old_files}
    state=json.loads((part/'state.json').read_text()); state.pop('fingerprint')
    state['contract_fingerprint']=fingerprint(old); state['fingerprint']=fingerprint(state)
    write_json(original_dir/'contract.json',old); write_json(original_dir/'state.json',state)
    old_bytes={p.name:p.read_bytes() for p in original_dir.iterdir()}
    out=tmp_path/'continued'
    assert cli.main(argv=[*base,'--out-dir',str(out),'--continue-from-dir',str(original_dir)])==0
    result=json.loads((out/'waymo_validation.json').read_text())
    assert result['status']=='complete' and result['reused_prefix_windows']==3
    assert result['new_windows_this_invocation']==14 and len(calls)==17 and len(checked)==2
    assert all((original_dir/k).read_bytes()==v for k,v in old_bytes.items())
    assert ckpt.read_bytes()==b'FROZEN WEIGHTS'
    assert cli.main(argv=[*base,'--out-dir',str(out),'--resume'])==0 and len(calls)==17
    with pytest.raises(RuntimeError,match='contract'):
        cli.main(argv=[*base,'--out-dir',str(out),'--resume','--geometry-cache-mib','2'])


@pytest.mark.skipif(not torch.cuda.is_available(),reason='real CUDA unavailable')
def test_real_cuda_probability_and_six_dense_parity(tmp_path):
    from tools.real_motion.joint_surface_long_rollout_common import verify_first_block
    source,pcfg,model=model_fixture(tmp_path,device='cuda',shape=(80,80,4))
    provider=FastWaymoSurfaceProvider(model,pcfg,'cuda',2,geometry_mib=8)
    execution=FastSurfaceBlockExecution(provider,workers=2,query_workers=2,graphs=True,surface_chunk=256)
    try:
        for window in source.windows[4:7]:
            record,raw=source.prediction_inputs(window)
            prep=provider.prepare_columns(None,record,include_gt=False,raw_window=raw)
            dense,_,_,scores=execution.predict(prep)
            verify_first_block(provider,prep.state['rec'],prep,dense,scores,execution)
        assert execution.head.counts['graph_replays']>0
    finally: execution.close()
