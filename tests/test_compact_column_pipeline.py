"""Complete-population two-pass/batch kernels must preserve actual training."""
import copy
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch
from scipy.ndimage import binary_dilation

from test_native_column_cpu import compiled
from test_column_cpu_kernels import candidate_fixture
from test_joint_causal_columns import fixture, optimizers, provider_for
from test_causal_column_sampling import fixture as sampling_fixture
from real_motion import native_column_cpu as native
from real_motion.causal_column_sampling import ColumnFeatureSampler
from tools.real_motion import joint_column_common as common
from tools.real_motion import causal_column_common as col
from tools.real_motion.joint_column_full_common import train_full_batch


@pytest.fixture(autouse=True)
def backend(compiled,monkeypatch):
    monkeypatch.setenv('SWFM_COLUMN_CPU_BACKEND','native')
    monkeypatch.setenv('SWFM_COLUMN_CPU_BUNDLE','1')


def test_scan_has_no_dense_rows_and_only_draws_materialize(monkeypatch,compiled):
    prep,grid,cfg=candidate_fixture(32)
    def forbidden(*args,**kwargs): raise AssertionError('dense candidate function used before selection')
    monkeypatch.setattr(compiled,'rows',forbidden); monkeypatch.setattr(compiled,'targets',forbidden)
    plans=common.build_online_column_candidates(prep,cfg,grid,defer_context=True)
    assert all(labels is None and plan.audit()['materialized_rows'] == 0 for _,plan,labels in plans)
    for _,plan,_ in plans:
        for field in ('flat','base','fallback','legal','context'):
            with pytest.raises(RuntimeError,match='after selection'): getattr(plan,field)
    selected=common.select_online_columns(prep,cfg,grid,np.random.default_rng(42),candidates=plans)
    assert sum(len(row[1]) for row in selected) <= 256
    assert sum(p.audit()['materialized_rows'] for _,p,_ in plans) == sum(len(r[1]) for r in selected)
    assert sum(len(p) for _,p,_ in plans) > 20*sum(len(r[1]) for r in selected)
    assert compiled.info()['calls']['support_many'] > 0


@pytest.mark.parametrize('gtmode',['natural','free','random'])
def test_compact_full_fields_every_stratum_subset_order_and_rng(monkeypatch,gtmode):
    prep,grid,cfg=candidate_fixture(8,legacy=True)
    if gtmode != 'natural':
        prep.raw['future_gt_occ']=[np.full(grid.shape_hwd,17,np.uint8) if gtmode == 'free' else
            np.random.default_rng(h).integers(0,18,grid.shape_hwd,dtype=np.uint8) for h in range(6)]
    monkeypatch.setenv('SWFM_COLUMN_CPU_BUNDLE','0')
    dense=common.build_online_column_candidates(prep,cfg,grid,defer_context=True)
    monkeypatch.setenv('SWFM_COLUMN_CPU_BUNDLE','1')
    compact=common.build_online_column_candidates(prep,cfg,grid,defer_context=True)
    for (_,old,y),(_,new,_) in zip(dense,compact):
        assert np.array_equal(new.positive_rows,(y != 0).any(1))
        for ids in (np.arange(len(old))[::-1],slice(None,None,-3),np.arange(len(old))%3 == 0,
                    [-1,0,-1] if len(old) else [],np.empty(0,np.int64)):
            a=old.subset(ids); b,labels=new.materialize(ids)
            assert all(np.array_equal(v,getattr(b,k)) for k,v in vars(a).items())
            assert np.array_equal(y[ids],labels)
    x,y=np.random.default_rng(193),np.random.default_rng(193)
    # Explicitly original positive/negative population order, no replacement.
    prep.cpu_kernels_optimized=True
    a=common.select_online_columns(prep,cfg,grid,x,candidates=dense)
    b=common.select_online_columns(prep,cfg,grid,y,candidates=compact)
    assert x.bit_generator.state == y.bit_generator.state
    for aa,bb in zip(a,b):
        assert all(np.array_equal(v,getattr(bb[1],k)) for k,v in vars(aa[1]).items())
        assert np.array_equal(aa[2],bb[2]) and np.array_equal(aa[3],bb[3])


def test_future_gt_only_changes_supervision_not_candidates_or_network_inputs():
    prep,grid,cfg=candidate_fixture(8)
    other=copy.deepcopy(prep); other.raw['future_gt_occ']=[np.full(grid.shape_hwd,17,np.uint8)]*6
    a=common.build_online_column_candidates(prep,cfg,grid,defer_context=True)
    b=common.build_online_column_candidates(other,cfg,grid,defer_context=True)
    for (_,p,_),(_,q,_) in zip(a,b):
        ids=np.arange(0,len(p),17)
        pp,py=p.materialize(ids); qq,qy=q.materialize(ids)
        assert all(np.array_equal(v,getattr(qq,k)) for k,v in vars(pp).items())
        for key in ('xy','kind','actor','classes','masks'): assert np.array_equal(getattr(p,key),getattr(q,key))
    bad=copy.deepcopy(prep); bad.raw['future_gt_occ'][0]=bad.raw['future_gt_occ'][0].copy()
    bad.raw['future_gt_occ'][0][-1,-1,-1]=19
    with pytest.raises(ValueError,match='supervision'): common.build_online_column_candidates(bad,cfg,grid,defer_context=True)


@pytest.mark.parametrize('gtmode',['natural','free','random'])
def test_train_prior_complete_unsampled_action_counts_and_weights_exact(monkeypatch,compiled,gtmode):
    prep,grid,cfg=candidate_fixture(32,legacy=True)
    if gtmode != 'natural':
        prep.raw['future_gt_occ']=[np.full(grid.shape_hwd,17,np.uint8) if gtmode == 'free' else
            np.random.default_rng(h).integers(0,18,grid.shape_hwd,dtype=np.uint8) for h in range(6)]
    old={'generation':np.zeros(2),'refine':np.zeros(3)}
    new=copy.deepcopy(old)
    monkeypatch.setenv('SWFM_COLUMN_CPU_BUNDLE','0')
    common.count_proposals(prep,grid,cfg,old)
    monkeypatch.setenv('SWFM_COLUMN_CPU_BUNDLE','1')
    def forbidden(*args,**kwargs): raise AssertionError('TRAIN prior materialized full voxel rows')
    monkeypatch.setattr(compiled,'rows',forbidden); monkeypatch.setattr(compiled,'targets',forbidden)
    common.count_proposals(prep,grid,cfg,new)
    assert all(np.array_equal(v,new[k]) for k,v in old.items())
    assert common.weights_from_counts(old) == common.weights_from_counts(new)


def test_batched_support_cross_order_z_bounds_empty_groups_and_threads(compiled):
    rng=np.random.default_rng(12); shape=(35,31,4)
    groups=[rng.integers(0,np.prod(shape),n,dtype=np.int64) for n in (0,1,25,200,0,30)]
    def check():
        xy,rows,bounds=compiled.support_many(groups,shape)
        assert rows[0] == 0 and rows[-1] == len(xy)
        for i,flat in enumerate(groups):
            if not len(flat):
                assert rows[i] == rows[i+1] and np.array_equal(bounds[i],[4,-1]); continue
            ijk=np.column_stack(np.unravel_index(flat,shape)); mask=np.zeros(shape[:2],bool)
            mask[tuple(ijk[:,:2].T)]=True
            assert np.array_equal(xy[rows[i]:rows[i+1]],np.argwhere(binary_dilation(mask)))
            assert np.array_equal(bounds[i],[ijk[:,2].min(),ijk[:,2].max()])
    with ThreadPoolExecutor(max_workers=4) as pool: list(pool.map(lambda _:check(),range(12)))
    assert compiled.support_many([],shape)[0].shape == (0,2)
    with pytest.raises(ValueError): compiled.support_many([np.array([-1],np.int64)],shape)


@pytest.mark.parametrize('z',[1,16,64])
def test_compact_scan_materialization_prior_random_ownership_all_classes_and_vertical_bits(compiled,z):
    from types import SimpleNamespace
    from tools.real_motion.compact_column_candidates import pack_allowed
    rng=np.random.default_rng(z); shape=(19,13,z); n=300
    xy=rng.integers([0,0],shape[:2],(n,2),dtype=np.int32)
    kinds=rng.integers(0,2,n,dtype=np.uint8)
    actors=np.where(kinds == 0,-3,rng.choice([-2,0,1,3],n)).astype(np.int32)
    classes=rng.integers(0,17,n,dtype=np.uint8); allowed=rng.random((n,z)) > .4
    before=rng.integers(0,18,shape,dtype=np.uint8); fallback=rng.integers(0,18,shape,dtype=np.uint8)
    owners=rng.integers(-1,4,shape,dtype=np.int32); gt=rng.integers(0,18,shape,dtype=np.uint8)
    flat=np.empty((n,z),np.int64); base=np.empty((n,z),np.uint8); fall=np.empty_like(base)
    legal=np.empty((n,z,3),bool); active=np.empty(n,bool)
    for kind,actor in set(zip(kinds,actors)):
        take=(kinds == kind)&(actors == actor)
        flat[take],base[take],fall[take],legal[take],active[take]=compiled.rows(
            xy[take],classes[take],allowed[take],before,owners,fallback,int(kind),int(actor))
    labels=compiled.targets(SimpleNamespace(flat=flat,base=base,fallback=fall,legal=legal,classes=classes),gt)
    args=(xy,kinds,actors,classes,pack_allowed(allowed),before,owners,fallback,gt)
    a,p,*rows=compiled.compact(*args,materialize=True)
    assert np.array_equal(a,active) and np.array_equal(p,(labels != 0).any(1))
    assert all(np.array_equal(old,new) for old,new in zip((flat,base,fall,legal,labels),rows))
    aa,pp,counts=compiled.compact(*args,prior_counts=True)
    gen=(kinds == 0)[:,None]&legal[...,1]; ref=(kinds == 1)[:,None]&(legal[...,1]|legal[...,2])
    expected=np.r_[np.bincount((labels[gen] == 1).astype(int),minlength=2),np.bincount(labels[ref],minlength=3)]
    assert np.array_equal(counts,expected) and np.array_equal(a,aa) and np.array_equal(p,pp)
    empty=compiled.compact(*(a[:0] if i < 5 else a for i,a in enumerate(args)),prior_counts=True)
    assert empty[0].size == 0 and np.array_equal(empty[2],np.zeros(5,np.int64))
    bad=list(args);bad[0]=xy.copy();bad[0][0,0]=-1
    with pytest.raises(ValueError): compiled.compact(*bad)
    if z < 64:
        bad=list(args);bad[4]=args[4].copy();bad[4][0] |= np.uint64(1 << z)
        with pytest.raises(ValueError): compiled.compact(*bad)


@pytest.mark.parametrize('history',[4,6])
@pytest.mark.parametrize('row_size',[1,12])
def test_gather_many_matches_per_frame_order_visibility_and_membership(compiled,history,row_size):
    rng=np.random.default_rng(57); shape=(5,7,4); points=36
    hist=rng.integers(0,18,(history,*shape),dtype=np.uint8)
    obs=rng.integers(0,4,hist.shape,dtype=np.uint8)
    indices=rng.integers(-2,9,(history,points,3),dtype=np.int64)
    members=[None if f%3 == 0 else np.unique(rng.integers(0,np.prod(shape),25,dtype=np.int64)) for f in range(history)]
    tables=[]
    for f,owned in enumerate(members):
        if owned is None or f%2: tables.append(None)
        else:
            lo=int(owned[0]); tables.append((lo,np.isin(np.arange(lo,int(owned[-1])+1),owned)))
    valid=np.arange(history)%4 != 2
    values,flags=compiled.gather_many(indices,hist,obs,members,tables,valid,row_size)
    expected=np.full_like(values,18); bits=np.zeros_like(flags)
    for f in range(history):
        if not valid[f]: continue
        v,b=compiled.gather(indices[f],hist[f],obs[f],members[f],tables[f])
        expected[:,f]=v.reshape(-1,row_size); bits[:,f]=b.reshape(-1,row_size)
    assert np.array_equal(values,expected) and np.array_equal(flags,bits)
    index_list=[indices[f] if valid[f] else None for f in range(history)]
    vv,bb=compiled.gather_many(index_list,hist,obs,members,tables,valid,row_size)
    assert np.array_equal(values,vv) and np.array_equal(flags,bb)
    index_list[0]=None
    with pytest.raises(ValueError,match='missing indices'):
        compiled.gather_many(index_list,hist,obs,members,tables,valid,row_size)
    with pytest.raises(ValueError): compiled.gather_many(indices,hist,obs,members,tables,valid,5)


@pytest.mark.parametrize('history',[4,6])
@pytest.mark.parametrize('limit',[0,64])
def test_feature_batch_and_legacy_native_exact_including_dense_maps(monkeypatch,history,limit):
    prep,grid,cfg,plan=sampling_fixture(4)
    for k in ('history_occ','history_observed','history_poses'): prep.raw[k]=prep.raw[k][-history:]
    prep.registrations=[r[-history:] for r in prep.registrations]
    monkeypatch.setenv('SWFM_COLUMN_CPU_BUNDLE','0')
    old=ColumnFeatureSampler(prep,3,plan,grid,cfg,col.pose_motion,max_cache_mib=limit).sample(plan,None)
    monkeypatch.setenv('SWFM_COLUMN_CPU_BUNDLE','1')
    new=ColumnFeatureSampler(prep,3,plan,grid,cfg,col.pose_motion,max_cache_mib=limit).sample(plan,None)
    assert all(np.array_equal(v,new[k]) for k,v in old.items())


@pytest.mark.parametrize('history',[4,6])
def test_previous_native_and_bundle_adamw_updates_gradients_rng_scalars_exact(monkeypatch,history):
    from dataclasses import replace
    from real_motion.joint_causal_columns import JointCausalColumns
    prep,grid,original,control,rec=fixture()
    joint=JointCausalColumns(replace(original.transport.v17_config,history_frames=history),original.columns.config)
    for k in ('history_occ','history_observed','history_poses'): prep.raw[k]=prep.raw[k][-history:]
    prep.registrations=[r[-history:] for r in prep.registrations]
    torch.nn.init.normal_(joint.columns.refinement.weight,std=.01); ref=copy.deepcopy(joint)
    p,q=provider_for(prep,grid,joint),provider_for(prep,grid,ref)
    opt,_=optimizers(joint,control); ropt,_=optimizers(ref,control)
    x,y=np.random.default_rng(61),np.random.default_rng(61)
    keys=('loss','motion_loss','column_loss','translation_smooth_l1','existence_bce','yaw_periodic_loss',
        'se2_shape_loss','generation_bce','refine_action_ce','grad_norm','column_grad_norm','source_query_gradient_norm')
    with ThreadPoolExecutor(max_workers=4) as pool:
        for update in (1,2,3):
            monkeypatch.setenv('SWFM_COLUMN_CPU_BUNDLE','0')
            a=train_full_batch(ref,ropt,q,None,[(rec,None)]*4,y,update,100,sampling_pool=pool,probe=True)
            monkeypatch.setenv('SWFM_COLUMN_CPU_BUNDLE','1')
            b=train_full_batch(joint,opt,p,None,[(rec,None)]*4,x,update,100,sampling_pool=pool,probe=True)
            assert all(a.get(k) == b.get(k) for k in keys)
            assert x.bit_generator.state == y.bit_generator.state
            assert all(torch.equal(v,ref.state_dict()[k]) for k,v in joint.state_dict().items())
            for k,state in opt.state_dict()['state'].items():
                assert all(torch.equal(v,ropt.state_dict()['state'][k][name]) for name,v in state.items())
            assert b['materialized_candidate_columns'] == b['sampled_columns']
            assert b['full_candidate_columns'] == a['full_candidate_columns']
