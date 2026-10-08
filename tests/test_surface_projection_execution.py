from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from real_motion.canonical_causal_repair import CanonicalEvidence, FEATURE_DIM, STATIC, map_canonical_evidence, compose_canonical
from real_motion.surface_projection_execution import map_surface_evidence
from real_motion.surface_canonical_repair import SurfaceAtlas, SurfaceCanonicalRepairHead, augment_evidence, augment_projection
from real_motion.ccr_frozen_b import frozen_b_probabilities
from real_motion.geometry import OccupancyGrid
from test_canonical_causal_repair import scene


@pytest.fixture(scope='module',params=['numpy','native'])
def projection_kernels(request):
    if request.param=='numpy':return None
    from real_motion.native_column_cpu import prepare_native,get_prepared_native
    prepare_native()
    return get_prepared_native()


@pytest.mark.parametrize('n',[0,1,2,3,4,17,4097,16387])
@pytest.mark.parametrize('mixed',[False,True])
def test_fused_projection_phase_and_destinations_byte_exact_near_boundaries(n,mixed,projection_kernels):
    grid, prep=scene(); rng=np.random.default_rng(110+n)
    points=rng.uniform(-2,9,(n,3)); points[:,2]=rng.uniform(-2,5,n)
    if n:
        points[:min(n,8),:]=np.array([0,1,2])+np.arange(min(n,8))[:,None]*np.finfo(float).eps
    actor=np.full(n,STATIC,np.int32)
    if mixed: actor[::5]=0
    classes=np.where(actor<0,11,4).astype(np.uint8)
    e=CanonicalEvidence(rng.normal(size=(n,FEATURE_DIM)).astype(np.float32),
        np.tile(classes[:,None],(1,4)),actor,classes,points,np.ones((n,4),bool),{})
    poses=[]
    for h in range(6):
        angle=.07*(h+1); c,s=np.cos(angle),np.sin(angle); pose=np.eye(4)
        pose[:2,:2]=[[c,-s],[s,c]]; pose[:3,3]=[.031*h,-.051*h,.019*h]; poses.append(pose)
    prep.raw['future_poses']=np.stack(poses); prep.state['world_to_future']=np.linalg.inv(poses)
    prep.yaws[:]=.06
    atlas=SurfaceAtlas(e.world,e.classes,e.presence,e.actor,np.eye(4),grid)
    e=augment_evidence(e,atlas)
    a=map_canonical_evidence(e,prep,grid,kernels=projection_kernels)
    with ThreadPoolExecutor(max_workers=4) as pool:
        b=map_surface_evidence(e,prep,grid,kernels=projection_kernels,executor=pool)
    for key in ('flat','base','fallback','legal','context'):
        np.testing.assert_array_equal(getattr(a,key),getattr(b,key))
    a=augment_projection(e,a,np.eye(4),prep.state['world_to_future'],grid)
    b=augment_projection(e,b,np.eye(4),prep.state['world_to_future'],grid)
    np.testing.assert_array_equal(a.context,b.context)
    p=rng.uniform(size=(n,6))
    for x,y in zip(compose_canonical(prep.baseline,e,a,p,p),compose_canonical(prep.baseline,e,b,p,p)):
        np.testing.assert_array_equal(x,y)


@pytest.mark.parametrize('workers',[2,4])
def test_parallel_kdtree_keeps_ties_stacked_planes_and_all_descriptor_bytes(workers):
    grid=OccupancyGrid(x_min=0,y_min=0,z_min=0,voxel_size=(.4,.4,.4),shape_hwd=(200,200,16))
    xy=np.indices((80,80)).reshape(2,-1).T.astype(np.float64)*.4
    world=np.column_stack((np.tile(xy,(2,1)),np.repeat([.2,2.2],len(xy))))
    cls=np.tile(np.where(xy[:,0]<16,11,13).astype(np.uint8),2)
    presence=np.ones((len(world),4),bool); presence[::7,-1]=False
    evidence=CanonicalEvidence(np.zeros((len(world),FEATURE_DIM),np.float32),
        np.tile(cls[:,None],(1,4)),np.full(len(world),STATIC,np.int32),cls,world,presence,{})
    atlas=SurfaceAtlas(world,cls,presence,evidence.actor,np.eye(4),grid)
    reference=atlas.describe(evidence); atlas.query_workers=workers
    np.testing.assert_array_equal(reference,atlas.describe(evidence))


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_fused_nonzero_surface_head_probability_and_six_dense_bytes(device):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('actual CUDA required')
    grid,prep=scene()
    from real_motion.canonical_causal_repair import build_canonical_evidence
    e=build_canonical_evidence(prep,grid)
    e=augment_evidence(e,SurfaceAtlas(e.world,e.classes,e.presence,e.actor,np.eye(4),grid))
    prep.raw['future_poses'][:,0,3]=np.arange(6)*.013
    prep.state['world_to_future']=np.linalg.inv(prep.raw['future_poses'])
    args=(e,np.eye(4),prep.state['world_to_future'],grid)
    a=augment_projection(args[0],map_canonical_evidence(e,prep,grid),*args[1:])
    b=augment_projection(args[0],map_surface_evidence(e,prep,grid),*args[1:])
    head=SurfaceCanonicalRepairHead(8).to(device).eval().requires_grad_(False)
    with torch.no_grad(): head.surface.weight.normal_(); head.phase.weight.normal_()
    output=dict(history_source_context=torch.randn(1,8,device=device),future_transport_queries=torch.randn(1,6,8,device=device))
    with torch.no_grad():
        left=frozen_b_probabilities(head,e,a,output,torch.device(device))
        right=frozen_b_probabilities(head,e,b,output,torch.device(device))
    np.testing.assert_array_equal(left,right)
    for x,y in zip(compose_canonical(prep.baseline,e,a,left[...,0],left[...,1]),
                   compose_canonical(prep.baseline,e,b,right[...,0],right[...,1])):
        np.testing.assert_array_equal(x,y)
