import copy
import gc
import weakref
from dataclasses import fields
import numpy as np
import pytest
import torch
from real_motion.joint_causal_columns import LinkedColumns
from real_motion.shared_column_evidence import SharedEvidenceColumns, SharedEvidenceConfig, SharedHistorySession
from real_motion.sparse_column_readout import token_probe, sparse_token_ids
from real_motion.column_device_geometry import DeviceColumnWindow, DevicePlan, actions, compose, targets, sample_training
from real_motion.causal_column_completion import compose_dense, actions_from_probabilities, action_targets
from tools.real_motion import causal_column_common as reference
from test_causal_columns import scene_fixture


def device_fixture(device='cpu'):
    torch.set_num_threads(1)
    prep,grid,cfg=scene_fixture()
    prep.raw={**prep.raw,'history_occ':prep.raw['history_occ'][-4:],
        'history_observed':prep.raw['history_observed'][-4:],'history_poses':prep.raw['history_poses'][-4:]}
    prep.registrations=[r[-4:] for r in prep.registrations]
    point=np.array([[4.5,6.5,.5]])
    background=prep.baseline[0].copy();background[6,6,0]=13
    prep.state={**prep.state,'source_world_points':[point],'source_z_t0':np.array([.5]),
        'source_rel_xy':[np.array([[0.,0.]])],'column_backgrounds':[background.copy() for _ in range(6)]}
    anchor=torch.tensor([[[6.5,6.5]]*6]);output={'residual_xy_m':torch.zeros(1,6,2),
        'yaw_delta_rad':torch.zeros(1,6),'future_transport_queries':torch.randn(1,6,8,requires_grad=True)}
    window=DeviceColumnWindow(prep,grid,cfg,device).render(anchor,output['residual_xy_m'].to(device),output['yaw_delta_rad'].to(device))
    return prep,grid,cfg,output,window


@pytest.mark.parametrize('h',range(6))
def test_device_live_candidates_and_full_patch_lookup_match_reference(h):
    prep,grid,cfg,output,window=device_fixture()
    a=reference.candidate_plan(prep,h,grid,cfg);b=window.candidates(h)
    for field in fields(DevicePlan):
        av=getattr(a,field.name);bv=getattr(b,field.name).cpu().numpy()
        if field.name=='context':assert np.allclose(av,bv,rtol=0,atol=2e-7)
        else:assert np.array_equal(av,bv),field.name
    ijk,labels,flags=window.lookup(h,b,sparse=False)
    expected=reference.sample_column_features(prep,h,a,grid,cfg)
    assert np.array_equal(labels.reshape(len(b),4,7,7,cfg.z_bins).numpy(),expected['history'])
    assert np.array_equal(flags.reshape(len(b),4,7,7,cfg.z_bins).numpy(),expected['flags'])
    assert np.array_equal(targets(b,torch.as_tensor(prep.raw['future_gt_occ'][h])).numpy(),action_targets(a,prep.raw['future_gt_occ'][h]))


@pytest.mark.parametrize('gates',[(.5,.5,None),(.5,.5,.95),(None,None,None)])
@pytest.mark.parametrize('variant',[(True,True),(True,False),(False,True)])
def test_device_actions_composition_all_branches_match(gates,variant):
    prep,grid,cfg,output,window=device_fixture();plan=window.candidates(2);cpu=plan.cpu()
    rng=np.random.default_rng(81);p=rng.random((*plan.base.shape,3)).astype(np.float32)
    p*=cpu.legal;p/=p.sum(-1,keepdims=True)
    a=actions(plan,torch.as_tensor(p),gates);expected=actions_from_probabilities(cpu,p,gates)
    assert np.array_equal(a.numpy(),expected)
    generated=compose(window.baseline[2],plan,a,generation=variant[0],refine=variant[1])
    assert np.array_equal(generated.numpy(),compose_dense(prep.baseline[2],cpu,expected,
        enable_generation=variant[0],enable_refine=variant[1]))


def test_tensor_collision_remove_restores_source_fallback_not_free():
    from test_causal_columns import plan_fixture
    plan=plan_fixture();gpu=DevicePlan.from_cpu(plan,'cpu')
    base=torch.tensor([[[4,17]]],dtype=torch.uint8);action=torch.zeros_like(gpu.flat)
    action[:,1]=1;action[2,0]=2
    assert compose(base,gpu,action).tolist()==[[[5,4]]]


def model_fixture():
    prep,grid,cfg,output,window=device_fixture();torch.manual_seed(81)
    teacher=LinkedColumns(cfg,8,history_frames=4)
    torch.nn.init.normal_(teacher.generation.weight,std=.05)
    torch.nn.init.normal_(teacher.refinement.weight,std=.05)
    student=SharedEvidenceColumns.from_teacher(teacher,shared_config=SharedEvidenceConfig(tile=4,tile_chunk=3))
    torch.nn.init.normal_(student.native.spatial[-1].weight,std=.03)
    return prep,grid,output,window,teacher,student


def test_abandoned_device_windows_release_tensors_without_cyclic_gc():
    prep,grid,cfg,output,original=device_fixture()
    refs=[];gc.collect();enabled=gc.isenabled();gc.disable()
    try:
        for _ in range(24):
            window=DeviceColumnWindow(prep,grid,cfg,'cpu').render(
                torch.tensor([[[6.5,6.5]]*6]),output['residual_xy_m'],output['yaw_delta_rad'])
            refs.extend(weakref.ref(v) for v in (window,window.labels,window.owners[0]))
            del window
        assert all(ref() is None for ref in refs), 'finished windows must not wait for cyclic GC to release device memory'
    finally:
        if enabled:gc.enable()
        gc.collect()


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_repeated_training_steps_release_all_device_windows(device,monkeypatch):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA allocator plateau check requires server GPU')
    from tools.real_motion import shared_evidence_pilot_common as common
    joint,teacher,provider,rows=training_fixture()
    joint=joint.to(device);teacher=teacher.to(device);provider.device=torch.device(device)
    optimizer=common.make_optimizer(joint,frozen=True);rng=torch.Generator(device=device).manual_seed(71)
    original=common.DeviceColumnWindow;refs=[]
    def tracked(*args,**kwargs):
        window=original(*args,**kwargs);refs.extend(weakref.ref(v) for v in (window,window.labels))
        return window
    monkeypatch.setattr(common,'DeviceColumnWindow',tracked)
    allocations=[];gc.collect();enabled=gc.isenabled();gc.disable()
    try:
        for i in range(20):
            stat=common.training_step(joint,optimizer,provider,rows,rng,frozen=True,teacher=teacher.columns)
            assert stat['optimizer_updated']
            assert all(ref() is None for ref in refs), f'GPU window from earlier steps retained at step {i}'
            if device=='cuda':
                torch.cuda.synchronize();allocations.append(torch.cuda.memory_allocated())
        if allocations:assert max(allocations[4:])-min(allocations[4:])<16*2**20
    finally:
        if enabled:gc.enable()
        gc.collect()


def test_native_dense_tiles_same_full_z_features_and_gradients():
    prep,grid,output,window,teacher,student=model_fixture()
    other=copy.deepcopy(student);plan=window.candidates(2);ijk,_,flags=window.lookup(2,plan)
    args=[ijk,flags,plan.base,plan.fallback,plan.context,plan.kind,plan.classes,window.source_features(2,plan,output)]
    a=SharedHistorySession(student,window.labels,window.visibility,mode='dense')
    b=SharedHistorySession(other,window.labels,window.visibility,mode='tiles')
    x=student.read(a,*args);y=other.read(b,*args)
    assert all(torch.allclose(v,w,atol=2e-6,rtol=1e-6) for v,w in zip(x,y))
    sum(v.sum() for v in x).backward(retain_graph=True);sum(v.sum() for v in y).backward()
    for (name,p),(other_name,q) in zip(student.named_parameters(),other.named_parameters()):
        assert name==other_name
        assert (p.grad is None and q.grad is None) or torch.allclose(p.grad,q.grad,atol=3e-5,rtol=2e-5),name
    assert student.native.spatial[-1].weight.grad.abs().sum()>0
    assert output['future_transport_queries'].grad.abs().sum()>0


def test_session_optimizer_update_rejected_and_no_future_gt_features():
    prep,grid,output,window,teacher,student=model_fixture();plan=window.candidates(1)
    session=SharedHistorySession(student,window.labels,window.visibility)
    ijk,_,_=window.lookup(1,plan);before,_=session.gather(ijk)
    prep.raw['future_gt_occ']=[np.zeros_like(x) for x in prep.raw['future_gt_occ']]
    after,_=session.gather(ijk);assert torch.equal(before,after)
    with torch.no_grad():student.column.weight.add_(1)
    with pytest.raises(RuntimeError,match='optimizer'):session.gather(ijk)


def test_zero_shot_token_probe_restores_original_and_handles_empty_subset():
    prep,grid,output,window,teacher,student=model_fixture();teacher.eval();plan=window.candidates(0).cpu()
    batch={k:torch.as_tensor(v) for k,v in reference.sample_column_features(prep,0,plan,grid,teacher.config).items()}
    batch['source_features']=window.source_features(0,DevicePlan.from_cpu(plan,'cpu'),output)
    before={k:v.clone() for k,v in teacher.state_dict().items()};full=teacher(**batch)
    # Full memory nonempty, sparse selection empty: no all-masked attention NaN.
    with token_probe(teacher):
        g,r=teacher(**batch);assert torch.isfinite(g).all() and torch.isfinite(r).all()
        memory=torch.randn(2,196,teacher.config.width);invalid=torch.ones(2,196,dtype=torch.bool)
        invalid[:,8]=False
        args=[memory,invalid,batch['base'][:2],batch['fallback'][:2],batch['context'][:2],batch['kind'][:2],batch['classes'][:2]]
        assert torch.isfinite(teacher.decode_history(*args)).all()
        assert len(sparse_token_ids(4))==36
    assert all(torch.equal(v,teacher.state_dict()[k]) for k,v in before.items())
    assert all(torch.equal(a,b) for a,b in zip(full,teacher(**batch)))
    teacher.train()
    with pytest.raises(RuntimeError,match='eval'):
        with token_probe(teacher):pass


def test_device_sampler_same_budgets_no_duplicates_correct_importance():
    prep,grid,cfg,output,window=device_fixture();plan=window.candidates(0)
    y=targets(plan,torch.as_tensor(prep.raw['future_gt_occ'][0]))
    ids,w=sample_training(plan,y,24,torch.Generator().manual_seed(41))
    assert len(ids)<=48 and len(ids)==len(torch.unique(ids)) and (w>=1).all()


def training_fixture():
    from dataclasses import asdict
    from types import SimpleNamespace
    from test_joint_causal_columns import fixture
    from real_motion.joint_causal_columns import JointCausalColumns
    from real_motion.local_st_world_model_v17 import LocalSTWMV17Config
    prep,grid,cfg,output,window=device_fixture()
    _,_,original,_,record=fixture()
    motion=LocalSTWMV17Config(**{**asdict(original.transport.config),'history_frames':4})
    joint=JointCausalColumns(motion,cfg)
    for key in ('local_semantic_tube','target_source_mask_tube','frame_motion_features'):
        record[key]=record[key][:,-4:].clone()
    torch.nn.init.normal_(joint.columns.refinement.weight,std=.03)
    torch.nn.init.normal_(joint.columns.generation.weight,std=.03)
    teacher=copy.deepcopy(joint).eval()
    joint.columns=SharedEvidenceColumns.from_teacher(joint.columns,
        shared_config=SharedEvidenceConfig(tile=4,tile_chunk=3))
    raw={**prep.raw,'_column_causal_preparation':dict(prepared_state=prep.state,
        registrations=prep.registrations,aligned_history_points=prep.aligned_history_points,
        fixed_candidate_geometry=prep.fixed_candidate_geometry,memory=prep.memory,footprints=prep.footprints)}
    provider=SimpleNamespace(pcfg=SimpleNamespace(grid=grid),device=torch.device('cpu'))
    return joint,teacher,provider,[(record,raw)]


def test_actual_joint_step_shared_gradients_and_frozen_teacher_migration():
    from tools.real_motion.shared_evidence_pilot_common import training_step,make_optimizer
    joint,teacher,provider,rows=training_fixture()
    opt=make_optimizer(joint);rng=torch.Generator().manual_seed(52)
    stat=training_step(joint,opt,provider,rows,rng,profile=True)
    assert stat['optimizer_updated'] and stat['sampled_columns']>0 and np.isfinite(stat['loss'])
    assert joint.columns.source_projection.weight.grad.abs().sum()>0
    assert joint.columns.native.spatial[-1].weight.grad.abs().sum()>0
    assert any('decoder' in k and p.grad is not None and p.grad.abs().sum()>0 for k,p in joint.transport.named_parameters())
    before=copy.deepcopy(joint.transport.state_dict());teacher_before=copy.deepcopy(teacher.state_dict())
    opt=make_optimizer(joint,frozen=True)
    # Existing motion gradients must not be mistaken for newly linked ones.
    for p in joint.transport.parameters():p.grad=None
    stat=training_step(joint,opt,provider,rows,rng,frozen=True,teacher=teacher.columns,kd_weight=.25)
    assert stat['optimizer_updated'] and stat['frozen_transport'] and stat['kd_loss']>=0
    assert all(p.grad is None for p in joint.transport.parameters())
    assert all(torch.equal(v,joint.transport.state_dict()[k]) for k,v in before.items())
    assert all(torch.equal(v,teacher.state_dict()[k]) for k,v in teacher_before.items())
    assert all(p.grad is None for p in teacher.parameters())


def test_new_migration_roundtrip_restores_optimizer_sampling_and_torch_rng(tmp_path):
    from tools.real_motion.shared_evidence_pilot_common import training_step,make_optimizer
    from tools.real_motion.shared_evidence_recovery import migration_payload,restore_migration
    from tools.real_motion.static_evidence_selector_common import atomic_checkpoint
    joint,teacher,provider,rows=training_fixture();opt=make_optimizer(joint,frozen=True)
    rng=torch.Generator().manual_seed(71)
    training_step(joint,opt,provider,rows,rng,frozen=True,teacher=teacher.columns)
    contract=dict(schedule_steps=3,seed=71,teacher='immutable',window_batch=4,source_budget=128)
    path=tmp_path/'migration.pt'
    atomic_checkpoint(path,migration_payload(joint.columns,opt,rng,contract,role='periodic',cursor=1,successful=1,executed=1))
    training_step(joint,opt,provider,rows,rng,frozen=True,teacher=teacher.columns)
    resumed=copy.deepcopy(joint);other=make_optimizer(resumed,frozen=True);other_rng=torch.Generator().manual_seed(1)
    saved=torch.load(path,weights_only=False)
    assert restore_migration(saved,resumed.columns,other,other_rng,contract)==(1,1,1)
    training_step(resumed,other,provider,rows,other_rng,frozen=True,teacher=teacher.columns)
    assert torch.equal(rng.get_state(),other_rng.get_state())
    assert all(torch.equal(v,resumed.state_dict()[k]) for k,v in joint.state_dict().items())
    for state,key in zip(opt.state.values(),other.state.values()):
        for name,value in state.items():assert torch.equal(value,key[name])
    with pytest.raises(RuntimeError,match='identical'):
        restore_migration(saved,resumed.columns,other,other_rng,{**contract,'source_budget':256})
    with pytest.raises(RuntimeError,match='identical'):
        restore_migration({**saved,'protocol':'old_full'},resumed.columns,other,other_rng,contract)
    with pytest.raises(RuntimeError,match='cursor'):
        restore_migration({**saved,'cursor':4},resumed.columns,other,other_rng,contract)


def test_batched_render_three_colliding_sources_and_empty_scene():
    from types import SimpleNamespace
    from tools.real_motion.causal_column_common import component_layers
    from real_motion.rigid_transport import RasterizedRigidComponent
    prep,grid,cfg,output,_=device_fixture()
    source=prep.state['current'][0]
    prep.state['current']=[{**source,'class_id':c} for c in (4,5,7)]
    prep.state['source_world_points']=[np.array([[4.5,6.5,.5],[4.5,6.5,.5]]) for _ in range(3)]
    prep.state['source_z_t0']=np.full(3,.5)
    prep.registrations=[copy.deepcopy(prep.registrations[0]) for _ in range(3)]
    w=DeviceColumnWindow(prep,grid,cfg,'cpu').render(torch.tensor([[[6.5,6.5]]*6]*3),torch.zeros(3,6,2),torch.zeros(3,6))
    for h in range(6):
        components=[RasterizedRigidComponent(c,np.array([[6,6,0]]),2) for c in (4,5,7)]
        baseline,owner,fallback=component_layers(prep.state['column_backgrounds'][h],components)
        assert np.array_equal(w.baseline[h],baseline)
        assert np.array_equal(w.owners[h],owner)
        assert np.array_equal(w.fallback[h],fallback)
    prep.state.update(current=[],source_world_points=[],source_z_t0=np.empty(0));prep.registrations=[]
    w=DeviceColumnWindow(prep,grid,cfg,'cpu').render(torch.empty(0,6,2),torch.empty(0,6,2),torch.empty(0,6))
    assert all(torch.equal(a,torch.as_tensor(b)) for a,b in zip(w.baseline,prep.state['column_backgrounds']))
    assert all(torch.equal(a,b) for a,b in zip(w.baseline,w.fallback))


@pytest.mark.parametrize('angle',[0.,.11,.27])
def test_rotating_sources_rolled_future_ego_full_z_indices_match_cpu(angle):
    from scipy.spatial.transform import Rotation
    prep,grid,cfg,output,_=device_fixture()
    poses=[]
    for h in range(6):
        pose=np.eye(4);pose[:3,:3]=Rotation.from_euler('xyz',[.02,-.03,angle*(h+1)/6]).as_matrix()
        pose[:3,3]=[.1*h,-.03*h,0.]
        poses.append(pose)
    prep.raw['future_poses']=poses;prep.state['world_to_future']=np.linalg.inv(poses)
    anchors=torch.tensor([[[6.5,6.5]]*6]);output['residual_xy_m'][:]=.1;output['yaw_delta_rad'][:]=angle
    row={'anchors_xy_t0_m':anchors}
    prep.baseline,prep.owners,prep.fallbacks,prep.components,prep.targets,prep.yaws=reference.render_column_layers(prep.state,row,output,grid)
    window=DeviceColumnWindow(prep,grid,cfg,'cpu').render(anchors,output['residual_xy_m'],output['yaw_delta_rad'])
    from tools.real_motion.shared_evidence_pilot_common import check_geometry
    audit=check_geometry(prep,window,features=True)
    assert audit['integer_layers_exact'] and audit['full_Z_labels_membership_exact']


@pytest.mark.skipif(not torch.cuda.is_available(),reason='real CUDA/device equivalence runs on server')
def test_cuda_geometry_lookup_composition_integer_gate():
    prep,grid,cfg,output,window=device_fixture('cuda')
    for h in range(6):
        plan=window.candidates(h);reference_plan=reference.candidate_plan(prep,h,grid,cfg)
        for field in fields(DevicePlan):
            if field.name!='context':assert np.array_equal(getattr(plan,field.name).cpu().numpy(),getattr(reference_plan,field.name))
