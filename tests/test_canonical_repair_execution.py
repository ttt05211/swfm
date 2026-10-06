"""Exact execution, including real edit rules (not just successful imports)."""
import copy
import numpy as np
import pytest
import torch
from concurrent.futures import ThreadPoolExecutor
from real_motion.native_column_cpu import prepare_native, get_prepared_native
from real_motion.canonical_causal_repair import (build_canonical_evidence, map_canonical_evidence,
    repair_targets, compose_canonical, CanonicalRepairHead)
from test_canonical_causal_repair import scene


@pytest.fixture(scope='module')
def kernels():
    prepare_native()
    return get_prepared_native()


def equal(a,b,fields):
    for name in fields:
        x,y=getattr(a,name),getattr(b,name)
        assert x.dtype==y.dtype and x.shape==y.shape
        assert x.tobytes()==y.tobytes(), name


@pytest.mark.parametrize('halo',[False,True])
@pytest.mark.parametrize('budget',[1,4_000_000])
def test_full_support_features_and_legal_actions_bit_exact(kernels,halo,budget):
    grid,p=scene()
    # Include road/sidewalk conflict and OOB projections; no fixed source crop.
    p.raw['history_occ'][0,5,5,0]=13
    p.state['world_to_future'][3,0,3]=-2.3
    p.targets[4,0,:2]+=np.array([.73,1.04]);p.yaws[4,0]=.31
    a=build_canonical_evidence(p,grid,halo=halo,max_lattice_cells=budget)
    b=build_canonical_evidence(p,grid,halo=halo,max_lattice_cells=budget,kernels=kernels)
    equal(a,b,('features','labels','actor','classes','world','presence'))
    x=map_canonical_evidence(a,p,grid);y=map_canonical_evidence(b,p,grid,kernels=kernels)
    equal(x,y,('flat','base','fallback','legal','context'))
    gt=p.baseline.copy();gt[:,2,2,1]=11
    for r,s in zip(repair_targets(a,x,gt),repair_targets(b,y,gt)):assert r.tobytes()==s.tobytes()
    rng=np.random.default_rng(52);pr=rng.random((*x.flat.shape,2)).astype(np.float32)
    for r,s in zip(compose_canonical(p.baseline,a,x,pr[...,0],pr[...,1]),
                   compose_canonical(p.baseline,b,y,pr[...,0],pr[...,1])):assert r.tobytes()==s.tobytes()


def test_empty_and_missing_registration(kernels):
    grid,p=scene();p.registrations[0][0]=None
    for empty in (False,True):
        if empty:
            p.raw['history_occ'][:]=17;p.state['current']=[];p.registrations=[]
        a=build_canonical_evidence(p,grid);b=build_canonical_evidence(p,grid,kernels=kernels)
        equal(a,b,('features','labels','actor','classes','world','presence'))
        equal(map_canonical_evidence(a,p,grid),map_canonical_evidence(b,p,grid,kernels=kernels),
              ('flat','base','fallback','legal','context'))


def test_frozen_execution_does_not_block_source_gradients(kernels):
    grid,p=scene();e=build_canonical_evidence(p,grid,kernels=kernels)
    plan=map_canonical_evidence(e,p,grid,kernels=kernels)
    model=CanonicalRepairHead(source_dim=8,width=16)
    out={'history_source_context':torch.randn(1,8,requires_grad=True),
         'future_transport_queries':torch.randn(1,6,8,requires_grad=True)}
    enc=model.encode(torch.from_numpy(e.features),torch.from_numpy(e.labels),torch.from_numpy(e.actor),
                     torch.from_numpy(e.classes),out)
    y=model.decode(enc,torch.from_numpy(e.actor),torch.from_numpy(plan.context),torch.from_numpy(plan.base),
                   torch.from_numpy(plan.fallback),torch.from_numpy(plan.legal),out)
    y.sum().backward()
    assert out['history_source_context'].grad.abs().sum()>0
    assert out['future_transport_queries'].grad.abs().sum()>0


def test_kernel_rejects_bad_dimensions_before_native_call(kernels):
    with pytest.raises(ValueError):kernels.ccr_lattice(np.array([0],np.int64),np.array([0],np.uint8),(0,3,4),True)
    with pytest.raises(ValueError):kernels.ccr_lattice(np.array([1],np.int64),np.array([4],np.uint8),(3,3,4),True)


def test_native_outputs_reject_copies_or_readonly_buffers(kernels):
    grid,p=scene();e=build_canonical_evidence(p,grid)
    plan=map_canonical_evidence(e,p,grid);n=len(e)
    plan.base=np.asfortranarray(plan.base)
    with pytest.raises(ValueError,match='output arrays'):
        kernels.ccr_plan(np.zeros((n,3),np.int64),e.actor,e.classes,p.baseline[0],p.owners[0],
                         p.fallbacks[0],0,plan,plan.context)
    labels=np.zeros((n,4),np.uint8);labels.flags.writeable=False
    with pytest.raises(ValueError,match='output arrays'):
        kernels.ccr_history(np.zeros((n,3),np.int64),p.raw['history_occ'][0],p.raw['history_observed'][0],
                            0,labels,np.zeros((n,4),bool),np.zeros((n,4),bool))


def test_parallel_evidence_order_and_bounded_warm_cache(kernels):
    from real_motion.canonical_repair_context import FixedCanonicalCache
    grid,p=scene()
    with ThreadPoolExecutor(max_workers=4) as pool:
        a=build_canonical_evidence(p,grid)
        b=build_canonical_evidence(p,grid,kernels=kernels,executor=pool)
        equal(a,b,('features','labels','actor','classes','world','presence'))
        cache=FixedCanonicalCache(1,neighbors=False,kernels=kernels,executor=pool)
        e,_=cache.get(p,grid);again,_=cache.get(p,grid)
        assert e is again and cache.stats()['hits']==1
        # Predicted source motion is NOT a cache input; descriptors stay fixed.
        p.targets[1,0,0]+=4
        assert cache.get(p,grid)[0] is e
        p.raw['history_observed'][0,2,2,1]=False
        assert cache.get(p,grid)[0] is not e
        assert cache.stats()['mib']<=1
        cache.close()


def test_batched_head_preserves_window_objective_and_source_ownership(kernels):
    from real_motion.canonical_repair_batch import batched_repair_losses
    from real_motion.canonical_causal_repair import repair_loss
    grid,p=scene();e=build_canonical_evidence(p,grid,kernels=kernels)
    plan=map_canonical_evidence(e,p,grid,kernels=kernels)
    torch.manual_seed(12);head=CanonicalRepairHead(8,16)
    clone=copy.deepcopy(head)
    out={'history_source_context':torch.randn(2,8,requires_grad=True),
         'future_transport_queries':torch.randn(2,6,8,requires_grad=True)}
    clone_out={k:v.detach().clone().requires_grad_(True) for k,v in out.items()}
    y,w=repair_targets(e,plan,p.baseline)
    target=y.astype(np.float32);weight=w.astype(np.float32)
    individual=[]
    for i in range(2):
        o={k:v[i:i+1] for k,v in out.items()}
        enc=head.encode(torch.from_numpy(e.features),torch.from_numpy(e.labels),torch.from_numpy(e.actor),torch.from_numpy(e.classes),o)
        z=head.decode(enc,torch.from_numpy(e.actor),torch.from_numpy(plan.context),torch.from_numpy(plan.base),torch.from_numpy(plan.fallback),torch.from_numpy(plan.legal),o)
        individual.append(repair_loss(head,z,torch.from_numpy(e.actor),torch.from_numpy(target),torch.from_numpy(weight)))
    packed=batched_repair_losses(clone,[e,e],[plan,plan],clone_out,[1,1],[target,target],[weight,weight],torch.device('cpu'))
    for a,b in zip(individual,packed):torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-6)
    (sum(individual)/2).backward();(sum(packed)/2).backward()
    for a,b in zip(head.parameters(),clone.parameters()):
        if a.grad is not None:torch.testing.assert_close(a.grad,b.grad,atol=3e-6,rtol=3e-6)
    for k in out:torch.testing.assert_close(out[k].grad,clone_out[k].grad,atol=3e-6,rtol=3e-6)
    with pytest.raises(ValueError):
        batched_repair_losses(head,[e],[plan],out,[1],[target],[weight],torch.device('cpu'))
