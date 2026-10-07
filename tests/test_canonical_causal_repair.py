from types import SimpleNamespace
import numpy as np
import pytest
import torch
from real_motion.geometry import OccupancyGrid
from real_motion.canonical_causal_repair import (build_canonical_evidence,build_compact_canonical_support,map_canonical_evidence,
    map_canonical_reference,materialize_canonical_features,repair_targets,compose_canonical,CanonicalRepairHead,FEATURE_DIM,sampled_tasks,repair_loss)
from real_motion.canonical_repair_context import (build_causal_strata,sample_causal_points,
    sample_compact_causal_points,map_sampled_canonical,map_sampled_compact_canonical,full_static_conflicts)


def scene():
    grid=OccupancyGrid(x_min=0,y_min=0,z_min=0,voxel_size=(1.,1.,1.),shape_hwd=(8,8,4))
    occ=np.full((4,*grid.shape_hwd),17,np.uint8)
    occ[:,2,2,1]=4;occ[:3,3,2,1]=4;occ[:,5,5,0]=11
    reg=[[(np.eye(4),np.argwhere(o==4)) for o in occ]]
    center=np.array([2.5,2.5,1.5]);base=np.stack([occ[-1]]*6)
    owner=np.full_like(base,-1,dtype=np.int32);owner[:,2,2,1]=0
    fallback=base.copy();fallback[:,2,2,1]=11
    raw=dict(history_occ=occ,history_observed=np.ones_like(occ,bool),history_poses=np.tile(np.eye(4),(4,1,1)),
             future_poses=np.tile(np.eye(4),(6,1,1)))
    state=dict(current=[dict(class_id=4,centroid_world=center)],current_pose=np.eye(4),
               world_to_future=np.tile(np.eye(4),(6,1,1)))
    prep=SimpleNamespace(raw=raw,state=state,registrations=reg,baseline=base,owners=owner,fallbacks=fallback,
                         targets=np.tile(center[None,None],(6,1,1)),yaws=np.zeros((6,1)))
    return grid,prep


@pytest.mark.parametrize('halo',[False,True])
def test_dense_and_sparse_entity_indices_are_exactly_equivalent(halo):
    grid,prep=scene()
    a=build_canonical_evidence(prep,grid,halo=halo)
    b=build_canonical_evidence(prep,grid,halo=halo,max_lattice_cells=1)
    for key in ('features','labels','actor','classes','world','presence'):
        np.testing.assert_array_equal(getattr(a,key),getattr(b,key))
    assert a.features.shape[1]==FEATURE_DIM
    assert b.audit['sparse_entities']>0


def test_causal_inputs_ignore_gt_and_keep_metric_observations():
    grid,p=scene();a=build_canonical_evidence(p,grid,halo=False)
    p.raw['future_gt_occ']='POISON';p.raw['future_annotations']='POISON'
    b=build_canonical_evidence(p,grid,halo=False)
    np.testing.assert_array_equal(a.features,b.features)
    np.testing.assert_array_equal(a.world[a.actor==0],[[2.5,2.5,1.5],[3.5,2.5,1.5]])
    assert a.presence[a.actor==0].tolist()==[[True]*4,[True,True,True,False]]
    assert not a.audit['future_GT_used']


def test_t0_source_without_history_is_not_dropped_and_unknown_not_free():
    grid,p=scene();p.registrations[0][:3]=[None]*3
    e=build_canonical_evidence(p,grid)
    assert np.any(e.actor==0)
    assert (e.labels[e.actor==0,:3]==18).all()
    assert (e.features[e.actor==0,7:11][:,:3]==0).all()


def test_keep_exact_and_remove_restores_underlying_background():
    grid,p=scene();e=build_canonical_evidence(p,grid);plan=map_canonical_evidence(e,p,grid)
    z=np.zeros(plan.flat.shape,np.float32)
    result=compose_canonical(p.baseline,e,plan,z,z)
    assert all(np.array_equal(x,y) for x,y in zip(result,p.baseline))
    remove=z.copy();remove[plan.legal[...,1]]=1
    result=compose_canonical(p.baseline,e,plan,z,remove,role='dynamic')
    assert all(x[2,2,1]==11 for x in result)
    assert all(x[5,5,0]==11 for x in result)
    assert not (plan.legal[...,0]&plan.legal[...,1]).any()


def test_other_visible_owner_is_never_removed_and_bad_fallback_is_not_positive():
    grid,p=scene();e=build_canonical_evidence(p,grid)
    p.owners[:,2,2,1]=3
    plan=map_canonical_evidence(e,p,grid)
    assert not plan.legal[e.actor==0,:,1].any()
    p.owners[:,2,2,1]=0;plan=map_canonical_evidence(e,p,grid)
    gt=p.baseline.copy();gt[:,2,2,1]=17
    y,valid=repair_targets(e,plan,gt)
    assert valid[e.actor==0,:,1].any()
    assert not y[e.actor==0,:,1].any() # actual restore is road11, not FREE17
    gt[:,2,2,1]=11;y,_=repair_targets(e,plan,gt)
    assert y[e.actor==0,:,1].any()


def test_unknown_gt_has_no_negative_supervision():
    grid,p=scene();e=build_canonical_evidence(p,grid);plan=map_canonical_evidence(e,p,grid)
    gt=np.full_like(p.baseline,18);y,valid=repair_targets(e,plan,gt)
    assert not y.any() and not valid.any()


def test_temporal_actions_are_not_forced_identical_and_source_gradients_live():
    torch.manual_seed(7)
    head=CanonicalRepairHead(source_dim=8,width=16)
    out={'history_source_context':torch.randn(1,8,requires_grad=True),
         'future_transport_queries':torch.randn(1,6,8,requires_grad=True)}
    actors=torch.tensor([0,-2]);feat=torch.randn(2,FEATURE_DIM);classes=torch.tensor([4,11]);labels=classes[:,None].expand(2,4)
    enc=head.encode(feat,labels,actors,classes,out)
    logits=head.decode(enc,actors,torch.randn(2,6,8),torch.full((2,6),17),torch.full((2,6),17),
                       torch.ones(2,6,2,dtype=torch.bool),out)
    assert logits.shape==(2,6,2)
    assert not torch.equal(logits[0,0],logits[0,5])
    loss=repair_loss(head,logits,actors,torch.ones_like(logits),torch.ones_like(logits))
    loss.backward()
    assert out['history_source_context'].grad.abs().sum()>0
    assert out['future_transport_queries'].grad.abs().sum()>0
    assert all(torch.isfinite(p.grad).all() for p in head.parameters() if p.grad is not None)


def test_sampling_importance_weights_preserve_task_population():
    grid,p=scene();e=build_canonical_evidence(p,grid);plan=map_canonical_evidence(e,p,grid)
    gt=p.baseline.copy();gt[:,3,2,1]=4;y,valid=repair_targets(e,plan,gt)
    ids,w=sampled_tasks(e,y,valid,np.random.default_rng(3),per_group=10000)
    assert len(ids)==len(np.unique(ids))
    np.testing.assert_array_equal(w.sum((0,1)),valid.sum((0,1)))
    assert not np.any(w[~valid[ids]])


def test_vectorized_projection_is_reference_exact_with_se3_yaw_and_owners():
    grid,p=scene();e=build_canonical_evidence(p,grid)
    p.yaws[:,0]=np.linspace(-.3,.3,6);p.targets[:,:,0]+=np.linspace(0,2,6)[:,None]
    p.raw['future_poses'][:,1,3]=np.linspace(0,1,6)
    p.state['world_to_future']=np.linalg.inv(p.raw['future_poses'])
    a=map_canonical_evidence(e,p,grid);b=map_canonical_reference(e,p,grid)
    for key in ('flat','base','fallback','legal','context'):np.testing.assert_array_equal(getattr(a,key),getattr(b,key))


def test_source_projection_sharing_has_identical_logits_and_gradients():
    torch.manual_seed(23);head=CanonicalRepairHead(source_dim=8,width=16)
    actors=torch.tensor([0,0,-2]);classes=torch.tensor([4,4,11]);features=torch.randn(3,FEATURE_DIM);labels=classes[:,None].expand(3,4)
    out={'history_source_context':torch.randn(1,8,requires_grad=True),'future_transport_queries':torch.randn(1,6,8,requires_grad=True)}
    encoded=head.encode(features,labels,actors,classes,out)
    ctx=torch.randn(3,6,8);base=torch.full((3,6),17);legal=torch.ones(3,6,2,dtype=torch.bool)
    a=head.decode(encoded,actors,ctx,base,base,legal,out)
    shared=head.project_sources(out);encoded=head.encode(features,labels,actors,classes,shared)
    b=head.decode(encoded,actors,ctx,base,base,legal,shared)
    assert torch.equal(a,b)
    b.sum().backward();assert out['future_transport_queries'].grad.abs().sum()>0


@pytest.mark.parametrize('limit',[1,4000000])
@pytest.mark.parametrize('halo',[False,True])
def test_deferred_train_features_match_eager_without_population_or_rng_changes(limit,halo):
    grid,p=scene();e=build_canonical_evidence(p,grid,halo=halo,max_lattice_cells=limit)
    deferred=build_canonical_evidence(p,grid,halo=halo,max_lattice_cells=limit,materialize_features=False)
    assert deferred.features is None and deferred.labels is None
    for key in ('actor','classes','world','presence'):np.testing.assert_array_equal(getattr(e,key),getattr(deferred,key))
    plan=map_canonical_evidence(e,p,grid);a,b=repair_targets(e,plan,p.baseline)
    ids,w=sampled_tasks(e,a,b,np.random.default_rng(19),per_group=8)
    other,ow=sampled_tasks(deferred,a,b,np.random.default_rng(19),per_group=8)
    np.testing.assert_array_equal(ids,other);np.testing.assert_array_equal(w,ow)
    small=materialize_canonical_features(deferred,p,grid,ids)
    np.testing.assert_array_equal(small.features,e.features[ids])
    np.testing.assert_array_equal(small.labels,e.labels[ids])


def test_compact_sampled_only_support_is_exact_to_legacy_lazy_population():
    grid,p=scene()
    legacy=build_canonical_evidence(p,grid,materialize_features=False)
    legacy.causal_strata=build_causal_strata(legacy)
    compact=build_compact_canonical_support(p,grid)
    assert len(compact)==len(legacy)
    a=np.random.default_rng(123);b=np.random.default_rng(123)
    ids,w=sample_causal_points(legacy,a,per_role=8)
    cids,cw=sample_compact_causal_points(compact,b,per_role=8)
    np.testing.assert_array_equal(ids,cids);np.testing.assert_array_equal(w,cw)
    conflicts=full_static_conflicts(legacy,p,grid)
    old,op=map_sampled_canonical(legacy,ids,p,grid,conflicts)
    new,np_=map_sampled_compact_canonical(compact,cids,p,grid)
    for name in ('features','labels','actor','classes','world','presence'):
        np.testing.assert_array_equal(getattr(old,name),getattr(new,name))
    for name in ('flat','base','fallback','legal','context'):
        np.testing.assert_array_equal(getattr(op,name),getattr(np_,name))
    oy,ov=repair_targets(old,op,p.baseline);ny,nv=repair_targets(new,np_,p.baseline)
    np.testing.assert_array_equal(oy,ny);np.testing.assert_array_equal(ov,nv)


def test_history_outside_t0_grid_is_retained_and_later_enters_query():
    grid,p=scene();p.raw['history_poses'][0,0,3]=8
    e=build_canonical_evidence(p,grid,halo=False)
    static=e.actor<0
    assert np.any(e.world[static,0]>8)
    p.state['world_to_future'][:,0,3]=-8
    plan=map_canonical_evidence(e,p,grid)
    assert np.any(plan.flat[static]>=0)


def test_no_sources_no_static_empty_keeps_all_six_frames():
    grid,p=scene();p.raw['history_occ'][:]=17;p.state['current']=[];p.registrations=[]
    p.targets=np.empty((6,0,3));p.yaws=np.empty((6,0));p.baseline[:]=17;p.owners[:]=-1
    e=build_canonical_evidence(p,grid);plan=map_canonical_evidence(e,p,grid)
    result=compose_canonical(p.baseline,e,plan,np.empty((0,6)),np.empty((0,6)))
    assert e.features.shape==(0,FEATURE_DIM)
    assert len(result)==6 and all(np.array_equal(x,y) for x,y in zip(result,p.baseline))


@pytest.mark.parametrize('bad',[np.nan,-.1,1.1])
def test_compositor_rejects_invalid_probability(bad):
    grid,p=scene();e=build_canonical_evidence(p,grid);plan=map_canonical_evidence(e,p,grid)
    with pytest.raises(ValueError):compose_canonical(p.baseline,e,plan,np.full(plan.flat.shape,bad),np.zeros(plan.flat.shape))


def test_report_formats_numpy_gate_values_without_rerunning_experiment(tmp_path):
    import json
    from tools.real_motion.pilot_p0_f9_canonical_causal_repair import finish_report
    metrics=dict(mIoU=40.,MovingMicro=30.,per_horizon={h:{'MovingMicro':30.} for h in ('1.0','2.0','3.0')})
    report=dict(GPU='fixture',fit={'passes':64},learned={'joint':metrics},old_joint={'joint':metrics},joint_training_probe={})
    report['speed']=[dict(mode=m,stratum=s,seconds=t) for m,t in [('old_joint',1.),('CCR',.2)]
                     for s in ('representative','high_source_stress')]
    finish_report(report,tmp_path)
    saved=json.loads((tmp_path/'pilot.json').read_text(encoding='utf-8'))
    assert saved['status']=='complete' and saved['gate']['speedup_ge_3']
    assert (tmp_path/'summary.txt').exists()
