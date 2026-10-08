from types import SimpleNamespace
import copy
import numpy as np
import pytest
import torch
from real_motion.prepared import OccupancyGrid
from real_motion.sparse_evidence_repair import SparseRepairHead,FREE
from real_motion.source_repair_evidence import build_evidence,map_evidence,compose_repair,oracle_probabilities
from tools.real_motion.source_repair_pilot_common import sample_pairs,probabilities,fill_generation
from tools.real_motion.source_repair_recovery import payload,restore
from real_motion.causal_column_completion import ColumnPlan,GENERATE,KEEP,ADD


def scene():
    grid=OccupancyGrid(x_min=0,y_min=0,z_min=0,voxel_size=(1.,1.,1.),shape_hwd=(8,8,4))
    occ=[np.full(grid.shape_hwd,FREE,np.uint8) for _ in range(4)]
    for t in range(4):occ[t][2,2,1]=4;occ[t][5,5,0]=11
    occ[0][3,2,1]=4
    ids=np.array([[2,2,1]])
    regs=[[(np.eye(4),np.argwhere(o==4)) for o in occ]]
    points=[[r[1].astype(float)+.5 for r in regs[0]]]
    raw=dict(history_occ=occ,history_poses=np.tile(np.eye(4),(4,1,1)),history_observed=[np.ones(grid.shape_hwd,bool) for _ in range(4)])
    state=dict(current=[dict(class_id=4,centroid_world=np.array([2.5,2.5,1.5]))],current_pose=np.eye(4),
        world_to_future=np.tile(np.eye(4),(6,1,1)))
    baseline=np.stack([occ[-1]]*6)
    prepared=SimpleNamespace(raw=raw,state=state,registrations=regs,aligned_history_points=points,
        baseline=baseline,targets=np.tile([[2.5,2.5,1.5]],(6,1,1)),yaws=np.zeros((6,1)))
    return grid,prepared


def test_real_evidence_preserves_metric_points_and_causality():
    grid,p=scene();e=build_evidence(p,grid,expand=False)
    dynamic=e.memory.keys[:,0]==0
    assert len(e.memory.keys[dynamic])==2
    np.testing.assert_array_equal(e.world_points[dynamic],[[2.5,2.5,1.5],[3.5,2.5,1.5]])
    assert e.memory.presence[dynamic].tolist()==[[True]*4,[True,False,False,False]]
    p.raw['future_gt_occ']='POISON'
    p.raw['future_annotations']='POISON'
    other=build_evidence(p,grid,expand=False)
    np.testing.assert_array_equal(other.memory.keys,e.memory.keys)
    assert other.audit['future_GT_used'] is False


def test_real_domain_halo_unknown_and_empty_source():
    grid,p=scene();p.raw['history_observed'][0][:]=False
    e=build_evidence(p,grid,expand=False)
    assert not np.any((e.memory.keys[:,0]==0)&(e.memory.keys[:,2]==1))
    for o in p.raw['history_occ']:o[:]=FREE
    p.state['current']=[];p.registrations=[]
    p.targets=np.empty((6,0,3));p.yaws=np.empty((6,0))
    e=build_evidence(p,grid)
    assert len(e.memory)==0
    assert map_evidence(e,p,grid).shape==(0,6)


def test_halo_is_not_claimed_observed_and_current_occupied_protected():
    grid,p=scene();e=build_evidence(p,grid)
    assert e.audit['halo_points']>0
    assert (~e.memory.presence.any(1)).sum()==e.audit['halo_points']
    mapped=map_evidence(e,p,grid)
    out=compose_repair(p.baseline[0],e,mapped,np.ones(mapped.shape),0)
    np.testing.assert_array_equal(out[p.baseline[0]!=FREE],p.baseline[0][p.baseline[0]!=FREE])
    assert out[3,2,1]==4
    assert np.all(mapped[e.memory.presence[:,-1]&(e.memory.keys[:,0]>=0)]==-1)


def test_static_class_ambiguity_fails_closed():
    grid,p=scene();p.raw['history_occ'][0][5,5,0]=13
    e=build_evidence(p,grid,expand=False)
    assert e.audit['ambiguous_cells']==1
    assert not np.any((e.memory.keys[:,0]==-2)&np.all(e.memory.keys[:,2:]==[5,5,0],axis=1))


def test_motion_world_height_not_shifted_and_full_ego_rotation():
    grid,p=scene();e=build_evidence(p,grid,expand=False)
    p.targets[:,0]=[3.5,2.5,100.]
    mapped=map_evidence(e,p,grid)
    out=compose_repair(p.baseline[0],e,mapped,np.ones(mapped.shape),0)
    assert out[4,2,1]==4
    p.state['world_to_future'][0,0,3]=-2
    mapped=map_evidence(e,p,grid)
    out=compose_repair(p.baseline[0],e,mapped,np.ones(mapped.shape),0)
    # moved point [4.5,2.5,1.5] -> ego [2.5,2.5,1.5], baseline protected.
    assert out[2,2,1]==4


def test_initial_head_add_is_exactly_baseline_and_chunking_consistent():
    grid,p=scene();e=build_evidence(p,grid);m=map_evidence(e,p,grid)
    head=SparseRepairHead('local_consensus')
    outputs=dict(history_source_context=torch.randn(1,128),future_transport_queries=torch.randn(1,6,128))
    with torch.no_grad():
        a=probabilities(head,e,outputs,torch.device('cpu'),chunk=2)
        b=probabilities(head,e,outputs,torch.device('cpu'),chunk=1000)
    np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(compose_repair(p.baseline[0],e,m,a,0),p.baseline[0])
    target=oracle_probabilities(e,m,p.baseline)
    assert target.shape==m.shape


def test_stratified_sampler_has_unbiased_natural_risk_and_no_gt_input():
    mapped=np.tile(np.arange(100)[:,None],(1,6));actors=np.r_[np.full(90,-2),np.arange(10)]
    pair,w=sample_pairs(mapped,actors,np.random.default_rng(1),per_window=24)
    assert len(pair)==24
    assert w.sum()==pytest.approx(1)
    assert w[actors[pair[:,0]]<0].sum()==pytest.approx(.9)
    assert w[actors[pair[:,0]]>=0].sum()==pytest.approx(.1)
    assert len(np.unique(pair,axis=0))==len(pair)


def test_generation_cannot_overwrite_sparse_additions():
    base=np.array([[[FREE,FREE]]],np.uint8)
    plan=ColumnPlan(np.array([[0,0]],np.int32),np.array([GENERATE],np.uint8),np.array([-3],np.int32),
        np.array([11],np.uint8),np.array([[0,1]]),base.reshape(1,2).copy(),base.reshape(1,2).copy(),
        np.ones((1,2,3),bool),np.zeros((1,12),np.float32),np.array([[0,0]],np.int32))
    repaired=base.copy();repaired[0,0,0]=4
    out=fill_generation(repaired,plan,np.array([[ADD,ADD]]))
    assert out.tolist()==[[[4,11]]]


def test_recovery_exact_optimizer_rng_and_contract():
    model=SparseRepairHead('local_consensus');opt=torch.optim.AdamW(model.parameters(),lr=.01)
    rng=np.random.default_rng(1);contract=dict(schedule_steps=10,teacher='fixed')
    model.score.bias.sum().backward();opt.step()
    saved=copy.deepcopy(payload(model,opt,rng,contract,cursor=2,successful=2,executed=8))
    expected=rng.random(10); expected_torch=torch.rand(4)
    other=SparseRepairHead('local_consensus');other_opt=torch.optim.AdamW(other.parameters(),lr=1.)
    other_rng=np.random.default_rng(3)
    assert restore(saved,other,other_opt,other_rng,contract)==[2,2,8]
    np.testing.assert_array_equal(other_rng.random(10),expected)
    assert torch.equal(torch.rand(4),expected_torch)
    assert other_opt.param_groups[0]['lr']==.01
    for k,v in model.state_dict().items():assert torch.equal(v,other.state_dict()[k])
    with pytest.raises(RuntimeError,match='identical'):restore(saved,other,other_opt,other_rng,{**contract,'teacher':'different'})
    with pytest.raises(RuntimeError,match='identical'):restore({**saved,'protocol':'old'},other,other_opt,other_rng,contract)


def real_training_fixture():
    from test_shared_column_evidence import training_fixture,device_fixture
    from tools.real_motion import causal_column_common as reference
    _,teacher,provider,rows=training_fixture();prep,grid,_,_,_=device_fixture()
    def prepare(source,record,*,include_gt,raw_window,outputs):
        result=copy.copy(prep);result.raw=raw_window;result.outputs=outputs
        # The live renderer exactness preflight deliberately bypasses cached
        # preparation once. The mock must support that production API too.
        cached=raw_window.get('_column_causal_preparation')
        result.state=cached['prepared_state'] if cached is not None else prep.state
        result.baseline,result.owners,result.fallbacks,result.components,result.targets,result.yaws=reference.render_column_layers(
            result.state,record,outputs,grid)
        return result
    provider.prepare_columns=prepare
    head=SparseRepairHead('local_consensus',source_dim=teacher.columns.source_dim)
    return teacher,provider,rows,head


def test_actual_motion_renderer_teacher_selected_KD_and_head_backward():
    from tools.real_motion.source_repair_pilot_common import training_step
    teacher,provider,rows,head=real_training_fixture()
    for p in teacher.parameters():p.requires_grad_(False)
    before=copy.deepcopy(teacher.state_dict());opt=torch.optim.AdamW(head.parameters(),lr=.001)
    rng=np.random.default_rng(9)
    for _ in range(3):
        stat=training_step(provider,rows,teacher,head,opt,rng)
        assert stat['optimizer_updated'] and stat['sampled_pairs']>0 and np.isfinite(stat['loss'])
        assert stat['GT_BCE']>0 and stat['KD_BCE']>0
    assert all(torch.equal(v,teacher.state_dict()[k]) for k,v in before.items())
    assert all(p.grad is None for p in teacher.parameters())
    assert head.point[0].weight.grad.abs().sum()>0


def test_actual_joint_sparse_loss_reaches_live_queries_without_geometry_grad(monkeypatch):
    from tools.real_motion.source_repair_pilot_common import training_step
    from tools.real_motion import joint_column_common
    teacher,provider,rows,head=real_training_fixture()
    torch.nn.init.normal_(head.score.weight,std=.1)
    # Isolate the head-only route: motion loss contributes zero, hard NumPy
    # targets/ownership remain detached but queries keep a live autograd graph.
    monkeypatch.setattr(joint_column_common,'motion_loss',lambda out,*a,**k:(out['residual_xy_m'].sum()*0,{}))
    opt=torch.optim.AdamW([*teacher.transport.parameters(),*head.parameters()],lr=.001)
    stat=training_step(provider,rows,teacher,head,opt,np.random.default_rng(9),kd_weight=0,live_motion=True)
    assert stat['optimizer_updated'] and not stat['transport_frozen']
    assert any('decoder' in k and p.grad is not None and p.grad.abs().sum()>0 for k,p in teacher.transport.named_parameters())
    assert head.future.weight.grad.abs().sum()>0


def test_sparse_steps_release_evidence_without_cyclic_gc(monkeypatch):
    import gc
    import weakref
    from tools.real_motion import source_repair_pilot_common as common
    teacher,provider,rows,head=real_training_fixture()
    for p in teacher.parameters():p.requires_grad_(False)
    opt=torch.optim.AdamW(head.parameters(),lr=.001);rng=np.random.default_rng(9)
    refs=[];original=common.build_evidence
    def tracked(*args,**kwargs):
        evidence=original(*args,**kwargs)
        refs.extend(weakref.ref(v) for v in (evidence,evidence.memory))
        return evidence
    monkeypatch.setattr(common,'build_evidence',tracked)
    enabled=gc.isenabled();gc.disable()
    try:
        for _ in range(10):
            stat=common.training_step(provider,rows,teacher,head,opt,rng,kd_weight=0)
            assert stat['optimizer_updated']
            assert all(ref() is None for ref in refs)
    finally:
        if enabled:gc.enable()


def test_halo_has_no_false_recent_observation_age():
    grid,p=scene();e=build_evidence(p,grid)
    halo=~e.memory.presence.any(1)
    assert np.all(e.memory.base_features[halo,-1]==1)


def test_one_command_real_head_evaluation_and_stop_resume(tmp_path,monkeypatch):
    """CPU emulation: real head/teacher/renderer/GT+KD/metrics/recovery.

    Dataset/checkpoint loading and CUDA timing are stubbed; never claims server
    FPS or real nuScenes quality. Compare uninterrupted vs stopped+resumed.
    """
    import json
    from threading import Event
    from pathlib import Path
    from test_shared_column_evidence import training_fixture,device_fixture
    from test_shared_evidence_pilot import MetricNuScenes
    from tools.real_motion import run_p0_f9_source_repair_pilot as pilot
    from tools.real_motion import source_repair_pilot_common as common
    from tools.real_motion import causal_column_common as reference
    from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import sha256
    from tools.real_motion.v18_source_interaction_common import select_population
    _,teacher,_,rows=training_fixture();record,raw=rows[0]
    prep,grid,_,_,_=device_fixture()
    train=[{**copy.deepcopy(record),'scene_name':f'train{s}','t0_token':f'{s}:{i}'} for s in range(10) for i in range(4)]
    dev=[{**copy.deepcopy(record),'scene_name':'dev','t0_token':f'd{i}'} for i in range(512)]
    keys=lambda records:tuple((r['scene_name'],r['t0_token']) for r in records)
    manifest=dict(parent_keys=keys(dev),manifest_fingerprint='fixed')
    paths={k:tmp_path/k for k in ('checkpoint','train-cache','dev-cache','population-manifest','base-checkpoint','train-info','dev-info')}
    for path in paths.values():path.write_bytes(b'original-input')
    digest={str(path):sha256(path) for path in paths.values()}
    ck=dict(cursor_epoch=19,model_configs={},cache_fingerprints=dict(train=digest[str(paths['train-cache'])],dev=digest[str(paths['dev-cache'])]),
        info_fingerprints=dict(train=digest[str(paths['train-info'])],dev=digest[str(paths['dev-info'])]),
        dev_manifest_fingerprint='fixed',dev_keys=keys(dev),train_keys=keys(train))
    def provider(checkpoint,digest,pcfg,device,workers,model,control):
        result=SimpleNamespace(joint=model,model=model.transport,pcfg=pcfg,device=device,workers=workers)
        def load(source,record,*,include_gt):
            data=copy.deepcopy(raw)
            if not include_gt:data['future_gt_occ']=None
            return data
        def prepare(source,record,*,include_gt,raw_window,outputs):
            value=copy.copy(prep);value.raw=raw_window;value.outputs=outputs
            value.state=raw_window['_column_causal_preparation']['prepared_state']
            value.baseline,value.owners,value.fallbacks,value.components,value.targets,value.yaws=reference.render_column_layers(
                value.state,record,outputs,grid)
            value.window=SimpleNamespace(scene_name=record['scene_name'],t0_token=record['t0_token'],future_tokens=tuple(f'f{h}' for h in range(6)))
            return value
        result.load_raw_columns=load;result.prepare_columns=prepare
        return result
    monkeypatch.setattr(pilot,'require_cuda',lambda name:torch.device('cpu'))
    monkeypatch.setattr(pilot,'TRAIN_WINDOWS',40)
    monkeypatch.setattr(pilot,'load_joint',lambda *a,**k:(ck,copy.deepcopy(teacher)))
    monkeypatch.setattr(pilot,'CLEAN_SHA256',digest[str(paths['base-checkpoint'])])
    monkeypatch.setattr(pilot,'make_prepare_config',lambda cfg:SimpleNamespace(grid=grid))
    monkeypatch.setattr(pilot,'load_manifest',lambda path:(manifest,keys(dev[:64]),None))
    monkeypatch.setattr(pilot,'load_cache',lambda path:({},train if str(path)==str(paths['train-cache']) else dev))
    monkeypatch.setattr(pilot,'select_population',lambda keys,scenes,**kwargs:select_population(keys,scenes,calibration_scenes=2,**kwargs))
    monkeypatch.setattr(pilot,'PilotProvider',provider)
    monkeypatch.setattr(pilot,'NuScenesWindowSource',lambda *a,**k:SimpleNamespace(nusc=MetricNuScenes()))
    monkeypatch.setattr(pilot,'joint_training_speed',lambda *a,**k:dict(trials=[],warning='CPU_TEST_NO_REAL_TIMING'))
    monkeypatch.setattr(pilot,'six_frame_speed',lambda *a,**k:dict(trials=[dict(mode='sparse_joint',six_frame_seconds=.2)],boundary='CPU_TEST',exclusions='NOT_REAL_FPS'))
    def run(out,*,resume=None,stop=None):
        argv=['pilot','--config',str(Path(__file__).resolve().parents[1]/'configs/real_motion_occfm.yaml'),
            '--dataroot',str(tmp_path),'--out-dir',str(out),'--eval-windows','2','--fps-windows','1','--max-updates','2']
        for key,path in paths.items():argv+=['--'+key,str(path)]
        if resume:argv+=['--resume',str(resume)]
        def step(*a,**k):
            value=common.training_step(*a,**k)
            if stop is not None:stop.set()
            return value
        monkeypatch.setattr(pilot,'training_step',step);monkeypatch.setattr('sys.argv',argv)
        return pilot.main(stop)
    whole,stopped,resumed=(tmp_path/name for name in ('whole','stopped','resumed'))
    assert run(whole)==0
    assert run(stopped,stop=Event())==130
    assert run(resumed,resume=stopped/'migration_last.pt')==0
    a,b=(torch.load(path/'migration_last.pt',weights_only=False) for path in (whole,resumed))
    assert a['cursor']==b['cursor']==2 and a['executed']==b['executed']==8
    assert all(torch.equal(v,b['head'][k]) for k,v in a['head'].items())
    assert a['numpy_rng']==b['numpy_rng']
    assert all(sha256(path)==digest[str(path)] for path in paths.values())
    report=json.loads((resumed/'bundle.json').read_text(encoding='utf-8'))
    assert report['status']=='complete' and report['actual_cuda'] is False
    assert 'teacher_static_ADD' in report['final_dev64']['variants']
    assert 'support_oracle' in report['final_dev64']['variants']
    assert report['training']['transport_frozen']
    assert not b['deployable']
