"""Byte parity of independent tube scatter/surface arithmetic and continuation."""
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest

from real_motion.geometry import OccupancyGrid, warp_semantic_grid
from real_motion.local_st_world_model import top_surface_semantic
from real_motion.canonical_causal_repair import CanonicalEvidence, FEATURE_DIM, STATIC
from real_motion.surface_canonical_repair import SurfaceAtlas
from real_motion.strong_w2det import StrongW2DetConfig
from real_motion.waymo_native_execution import prepare_waymo_native
from real_motion.waymo_geometry_execution_v2 import TubeFrameCache, aligned_bev, NativeSurfaceAtlas
from test_waymo_fast_execution import model_fixture


@pytest.mark.parametrize('points',[0,3,6,18,511,1000])
@pytest.mark.parametrize('yaw',[False,True])
def test_presorted_icp_identical_to_independent_reference(points,yaw):
    from real_motion.waymo_geometry_execution_v2 import registration_reference,register_presorted
    from real_motion.source_evidence_audit import register_history_shape
    rng=np.random.default_rng(points); p=rng.normal(size=(points,3)); p[:,2]*=.1
    for dx in (.1,2.5,50.):
        q=p+np.array([dx,.2,0.])
        ref=register_history_shape(p,q,allow_yaw=yaw)
        actual=register_presorted(p,p[np.lexsort(p.T[::-1])],registration_reference(q),allow_yaw=yaw)
        assert actual.points.tobytes()==ref.points.tobytes()
        assert (actual.accepted,actual.yaw_rad,actual.median_error_m,actual.inlier_fraction)==(
            ref.accepted,ref.yaw_rad,ref.median_error_m,ref.inlier_fraction)


@pytest.fixture(scope='module')
def native(): return prepare_waymo_native()


def test_native_warp_collision_stable_order_and_validation(native):
    flat=np.array([4,2,4,2,4],np.int64); distance=np.array([1,.2,.3,.2,.3],np.float64)
    labels=np.array([1,2,3,4,5],np.uint8)
    out=native.warp(flat,distance,labels,(2,3,2)).ravel()
    assert out[4]==3 and out[2]==2 and np.count_nonzero(out!=17)==2
    with pytest.raises(ValueError): native.warp(flat-10,distance,labels,(2,3,2))
    with pytest.raises(ValueError): native.warp(flat,distance*np.nan,labels,(2,3,2))


@pytest.mark.parametrize('seed',range(6))
def test_tube_bev_fp64_warp_all_classes_collisions_large_translation(native,seed):
    rng=np.random.default_rng(seed); grid=OccupancyGrid(-6,-5,-1,(.4,)*3,(32,32,8))
    sem=rng.integers(0,18,grid.shape_hwd,dtype=np.uint8)
    if seed==0: sem[:]=17
    pose=np.eye(4); cache=TubeFrameCache(grid,StrongW2DetConfig(free_label=17),ram_mib=8)
    f=cache.get('frame',sem,np.ones_like(sem,bool),pose)
    theta=.131*seed; transform=np.eye(4)
    transform[:2,:2]=[[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]
    transform[:3,3]=[.2*seed,-.1*seed,.2]
    ref=top_surface_semantic(warp_semantic_grid(sem,transform,grid),grid=grid)
    assert np.array_equal(ref,aligned_bev(f,transform,grid,native))
    assert not f.warp_homogeneous.flags.writeable and not f.warp_labels.flags.writeable
    assert cache.bytes<=cache.limit


@pytest.mark.parametrize('seed',range(8))
def test_native_surface_fit_bytes_empty_ties_stacked_and_sparse(native,seed):
    rng=np.random.default_rng(seed); n=6403
    points=rng.uniform(-8,8,(n,3)); points[:,2]*=.025
    points[20:35]=points[19]
    classes=np.where(np.arange(n)%3,11,13).astype(np.uint8)
    actors=np.full(n,STATIC,np.int32); actors[::103]=0
    seen=rng.random((n,4))>.3; seen[::7]=False
    e=CanonicalEvidence(np.zeros((n,FEATURE_DIM),np.float32),np.zeros((n,4),np.uint8),actors,classes,points,seen,{})
    pose=np.eye(4); theta=.2
    pose[:2,:2]=[[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]
    if seed%2: pose[:3,3]=[1e5,-2e5,.7]
    grid=OccupancyGrid()
    for ids in (np.arange(n),np.flatnonzero(classes==11),np.array([1,2]),np.empty(0,np.int64)):
        a=replace(e,features=e.features[ids],labels=e.labels[ids],actor=actors[ids],classes=classes[ids],world=points[ids],presence=seen[ids])
        ref=SurfaceAtlas(a.world,a.classes,a.presence,a.actor,pose,grid).describe(a)
        atlas=NativeSurfaceAtlas(a.world,a.classes,a.presence,a.actor,pose,grid)
        atlas.chunk_rows=256; atlas.native=native
        with ThreadPoolExecutor(3) as pool:
            atlas.fit_pool=pool; actual=atlas.describe(a)
        assert ref.tobytes()==actual.tobytes(), (seed,np.max(np.abs(ref-actual)),np.argwhere(ref!=actual)[:8])


@pytest.mark.parametrize('mode',['numpy','native_parallel'])
def test_real_model_state_tubes_surface_six_dense_and_integer_counts(tmp_path,native,mode):
    import torch
    from tools.real_motion.waymo_fast_execution import FastWaymoSurfaceProvider,FastSurfaceBlockExecution
    from tools.real_motion.waymo_fast_execution_v2 import FastV2WaymoProvider,FastV2SurfaceExecution
    from tools.real_motion.joint_surface_long_rollout_common import verify_first_block
    from real_motion.waymo_i2world import WaymoMetrics
    source,pcfg,model=model_fixture(tmp_path); source.preflight(source.windows)
    v1=FastWaymoSurfaceProvider(model,pcfg,'cpu',2,geometry_mib=8)
    v2=FastV2WaymoProvider(model,pcfg,'cpu',2,geometry_mib=8)
    one=FastSurfaceBlockExecution(v1,mode=mode,workers=2,graphs=False,surface_chunk=256)
    two=FastV2SurfaceExecution(v2,mode=mode,workers=2,graphs=False,surface_chunk=256)
    counts=[WaymoMetrics(),WaymoMetrics()]; before={k:v.clone() for k,v in model.state_dict().items()}
    threads=torch.get_num_threads(); torch.set_num_threads(1)
    try:
        for i in (0,1,4,5,11,12,20):
            r,raw=source.prediction_inputs(source.windows[i]); a=v1.prepare_columns(None,r,include_gt=False,raw_window=raw)
            r,raw=source.prediction_inputs(source.windows[i]); b=v2.prepare_columns(None,r,include_gt=False,raw_window=raw)
            for k in ('features','local_semantic_tube','kta_displacement_xy_m','frame_motion_features',
                      'target_source_mask_tube','source_class_id','anchors_xy_t0_m','source_centroid_xy_t0_m'):
                assert a.state['rec'][k].numpy().tobytes()==b.state['rec'][k].numpy().tobytes(),k
            for k in ('source_z_t0','anchors','source_world_points','source_rel_xy','baseline_clear_by_hi',
                      'baseline_clear_flat_by_hi','world_to_future'):
                assert all(np.array_equal(x,y) for x,y in zip(a.state[k],b.state[k])),k
            assert a.source_audit==b.source_audit
            for ar,br in zip(a.registrations,b.registrations):
                for x,y in zip(ar,br):
                    if x is None: assert y is None
                    else:
                        assert y is not None
                        assert all(u.shape==v.shape and u.dtype==v.dtype and u.tobytes()==v.tobytes()
                                   for u,v in zip(x,y))
            ad,_,_,ap=one.predict(a); bd,_,_,bp=two.predict(b)
            assert np.array_equal(ap,bp) and all(np.array_equal(x,y) for x,y in zip(ad,bd))
            verify_first_block(v2,b.state['rec'],b,bd,bp,two)
            targets=source.metric_targets(source.windows[i]); counts[0].add(ad,targets); counts[1].add(bd,targets)
            assert b.raw['future_gt_occ'] is None and b.state['rec']['features'].shape[0]>0
        assert np.array_equal(counts[0].counts,counts[1].counts)
        assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())
        assert v2.geometry.bytes<=v2.geometry.limit
    finally:
        one.close(); two.close(); v2.close(); torch.set_num_threads(threads)


def test_v2_paired_speed_full_path_no_gt_or_saved_updates(tmp_path,native,monkeypatch):
    import torch
    from tools.real_motion.waymo_fast_execution_v2 import paired_speed_v2
    source,pcfg,model=model_fixture(tmp_path)
    monkeypatch.setattr(source,'metric_targets',lambda w:pytest.fail('speed reads GT'))
    threads=torch.get_num_threads(); torch.set_num_threads(1)
    try:
        d=paired_speed_v2(source,source.windows[4:8],model,pcfg,'cpu',workers=2,graphs=False,
            parallel_majority=False,surface_chunk=256,geometry_mib=8)
        assert d['motion_inputs_bytes_exact'] and d['probability_and_six_dense_bytes_exact']
        assert set(d['seconds_per_window'])=={'fast_v1','fast_v2'} and d['speedup']>0
    finally: torch.set_num_threads(threads)


def test_v1_to_v2_cli_integer_migration_resume_and_changed_settings_rejected(tmp_path,native,monkeypatch):
    import json
    from threading import Event
    from types import SimpleNamespace
    from tools.real_motion import eval_p0_f9_joint_surface_waymo_10hz_fast_v2 as cli
    from tools.real_motion.eval_p0_f9_joint_surface_waymo_10hz_fast import IMPLEMENTATION_FILES as v1_files
    from tools.real_motion.waymo_zero_shot_common import write_json
    from real_motion.waymo_i2world import fingerprint
    from test_waymo_i2world_10hz import fixture,predict
    source=fixture(tmp_path/'data'); ckpt=tmp_path/'weights'/'mean.pt'; ckpt.parent.mkdir()
    ckpt.write_bytes(b'FROZEN'); config=tmp_path/'config.yaml'; config.write_bytes(b'CONFIG')
    monkeypatch.setattr(cli.WaymoI2World10HzSource,'from_files',lambda *a,**kw:source)
    monkeypatch.setattr(cli.torch.cuda,'is_available',lambda:True); monkeypatch.setattr(cli,'SHAPE',source.shape)
    monkeypatch.setattr(cli,'load_runtime_config',lambda p:None)
    from real_motion.prepared import PrepareConfig
    pcfg=PrepareConfig(grid=OccupancyGrid(-40,-40,-1,(.4,)*3,source.shape))
    monkeypatch.setattr(cli,'make_prepare_config',lambda c:pcfg)
    monkeypatch.setattr(cli,'load_evaluation_model',lambda *a,**kw:(
        dict(source_epochs=list(cli.AVERAGE_EPOCHS),averaging=True),
        SimpleNamespace(transport=SimpleNamespace(config=SimpleNamespace(history_frames=4)))))
    closed=[]; calls=[]; event=Event()
    class Provider:
        def __init__(self,*a,**kw):
            self.geometry=TubeFrameCache(pcfg.grid,StrongW2DetConfig(free_label=17),ram_mib=1)
            self.fast_prepare_stages={}
        def prepare_columns(self,s,record,*,include_gt,raw_window):
            assert not include_gt and raw_window['future_gt_occ'] is None
            outputs=predict(record,raw_window,verify=False)
            return SimpleNamespace(baseline=outputs[0],state=dict(rec=record),raw=raw_window)
        def close(self): closed.append(True)
    class Execution:
        def __init__(self,*a,**kw): pass
        def predict(self,prep):
            calls.append(prep.state['rec']['t0_token'])
            if len(calls)==3: event.set()
            return prep.baseline,dict(added=0,removed=0),{},None
        def close(self): pass
    monkeypatch.setattr(cli,'FastWaymoSurfaceProvider',Provider)
    monkeypatch.setattr(cli,'FastSurfaceBlockExecution',Execution)
    monkeypatch.setattr(cli,'verify_first_block',lambda *a:None)
    base=['--waymo-root',str(source.root),'--checkpoint',str(ckpt),'--config',str(config),
        '--expected-scenes','2','--speed-windows','0','--no-graphs','--geometry-cache-mib','1']
    part=tmp_path/'part'; assert cli.main(event,argv=[*base,'--out-dir',str(part)])==0
    old=json.loads((part/'contract.json').read_text()); old['fast_execution']={'protocol':'V1'}
    old['implementation']={k:v for k,v in old['implementation'].items() if k in v1_files}
    state=json.loads((part/'state.json').read_text()); state.pop('fingerprint')
    state['contract_fingerprint']=fingerprint(old); state['fingerprint']=fingerprint(state)
    stopped=tmp_path/'v1'; stopped.mkdir(); write_json(stopped/'contract.json',old); write_json(stopped/'state.json',state)
    before={p.name:p.read_bytes() for p in stopped.iterdir()}
    out=tmp_path/'v2'
    speed_calls=[]
    def speed(source,windows,*args,**kwargs):
        speed_calls.append(len(windows))
        return dict(seconds_per_window=dict(fast_v1=.5,fast_v2=.4),speedup=1.25,windows=len(windows))
    monkeypatch.setattr(cli,'paired_speed',speed)
    with monkeypatch.context() as guard:
        guard.setattr(source,'metric_targets',lambda *a:pytest.fail('speed-only reads future GT'))
        assert cli.main(argv=[*base,'--out-dir',str(out),'--continue-from-dir',str(stopped),
                            '--speed-only','--speed-windows','4'])==0
    assert speed_calls==[4] and len(calls)==3
    assert json.loads((out/'state.json').read_text())['completed_windows']==3
    assert not (out/'waymo_validation.json').exists()
    with monkeypatch.context() as guard:
        guard.setattr(cli.original,'resolve_checkpoint',lambda *a:pytest.fail('resume must use saved checkpoint'))
        without_checkpoint=[*base[:2],*base[4:]]
        assert cli.main(argv=[*without_checkpoint,'--out-dir',str(out),'--resume'])==0
    d=json.loads((out/'waymo_validation.json').read_text())
    assert d['reused_prefix_windows']==3 and d['completed_windows']==17 and len(calls)==17
    assert len(closed)==2 and all((stopped/k).read_bytes()==v for k,v in before.items())
    assert cli.main(argv=[*base,'--out-dir',str(out),'--resume'])==0 and len(calls)==17
    with pytest.raises(RuntimeError,match='contract'):
        cli.main(argv=[*base,'--out-dir',str(out),'--resume','--surface-chunk','512'])
    assert ckpt.read_bytes()==b'FROZEN'


def test_v1_files_unchanged_and_v2_dependencies_are_additions_only():
    from tools.real_motion.eval_p0_f9_joint_surface_waymo_10hz_fast import IMPLEMENTATION_FILES as v1
    from tools.real_motion.eval_p0_f9_joint_surface_waymo_10hz_fast_v2 import IMPLEMENTATION_FILES as v2
    assert set(v1)<set(v2)
    assert 'real_motion/native/waymo_execution.cpp' not in v1


@pytest.mark.skipif(not __import__('torch').cuda.is_available(),reason='real CUDA unavailable')
def test_actual_cuda_v2_graph_and_all_six_bytes(tmp_path,native):
    from tools.real_motion.waymo_fast_execution_v2 import FastV2WaymoProvider,FastV2SurfaceExecution
    from tools.real_motion.joint_surface_long_rollout_common import verify_first_block
    source,pcfg,model=model_fixture(tmp_path,device='cuda',shape=(80,80,4))
    provider=FastV2WaymoProvider(model,pcfg,'cuda',2,geometry_mib=8)
    execution=FastV2SurfaceExecution(provider,workers=2,graphs=True,surface_chunk=256)
    try:
        for window in source.windows[4:7]:
            r,raw=source.prediction_inputs(window); prep=provider.prepare_columns(None,r,include_gt=False,raw_window=raw)
            dense,_,_,score=execution.predict(prep)
            verify_first_block(provider,prep.state['rec'],prep,dense,score,execution)
        assert execution.head.counts['graph_replays']>0
    finally: execution.close(); provider.close()
