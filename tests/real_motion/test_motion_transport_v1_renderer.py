import numpy as np,torch
from real_motion.geometry import OccupancyGrid
from real_motion.motion_transport_v1.contracts import CausalInputs,SourceRecord,SourceDecomposition,TrainingTargets,MotionTarget
from real_motion.motion_transport_v1.compositor import render_soft_ordered,compose_hard,soft_argmax_scene,predicted_source_points_world
from real_motion.motion_transport_v1.losses import occupancy_ce_full,motion_loss_sum,calibrated_gradient_ratio
from real_motion.motion_transport_v1.geometry import index_to_metric_center
def fixture():
    g=OccupancyGrid(-2,-2,-.8,(.4,.4,.4),(10,10,4));T=np.eye(4);hist=np.full((6,*g.shape_hwd),17,np.uint8);idx=np.array([[4,4,1],[4,5,1],[5,4,1],[5,5,1],[4,4,2],[5,5,2]]);hist[-1][tuple(idx.T)]=4;hist[-2]=hist[-1];c=CausalInputs('s','s',hist,np.ones_like(hist,bool),tuple([T]*6),tuple([T]*6),np.arange(6)*.5,2.5+np.arange(1,7)*.5);p=index_to_metric_center(idx,g);s=SourceRecord(0,4,idx,p,p.mean(0),np.zeros(3),True,np.array([[-.4,-.4],[.4,.4]]),len(idx));d=SourceDecomposition([s],np.full((6,*g.shape_hwd),17,np.uint8),np.zeros((0,3),int),np.zeros(0,np.uint8),np.zeros((0,3)),np.arange(1,7)*.5);return g,c,d
def test_soft_probability_gradient_and_chunk_equivalence():
    g,c,d=fixture();a=torch.zeros((1,6,3),requires_grad=True);b=a.detach().clone().requires_grad_(True);a.data[:,:,0]=.11;a.data[:,:,2]=.07;b.data.copy_(a.data);sa=render_soft_ordered(c,d,a,[0],grid=g,query_chunk=7);sb=render_soft_ordered(c,d,b,[0],grid=g,query_chunk=99999)
    for x,y in zip(sa.horizons,sb.horizons):assert torch.equal(x.flat_indices,y.flat_indices) and torch.allclose(x.probabilities.sum(1),torch.ones(len(x.probabilities)),atol=1e-5) and torch.allclose(x.probabilities,y.probabilities,atol=1e-7,rtol=1e-6)
    la=sum(x.probabilities.square().sum() for x in sa.horizons);lb=sum(x.probabilities.square().sum() for x in sb.horizons);la.backward();lb.backward();assert torch.allclose(a.grad,b.grad,atol=2e-6,rtol=2e-5)
def test_full_ce_constant_region_and_old_support_cleared():
    g,c,d=fixture();z=torch.zeros((0,6,3),requires_grad=True);scene=render_soft_ordered(c,d,z,[],grid=g);gt=scene.hard_kta.copy();gt[:,0,0,0]=4;loss,_=occupancy_ce_full(scene,TrainingTargets(gt,np.ones_like(gt,bool),{}));assert float(loss)>.01
    delta=torch.zeros((1,6,3),requires_grad=True);delta.data[:,:,0]=1.2;s=render_soft_ordered(c,d,delta,[0],grid=g);old=compose_hard(c,d,torch.zeros_like(delta),[0],grid=g)
    for hi,row in enumerate(s.horizons):assert set(np.flatnonzero(old[hi].reshape(-1)==4)).issubset(set(row.flat_indices.cpu().tolist()))
def test_joint_calibrated_reachability_improves_hard():
    g,c,d=fixture();td=torch.zeros((1,6,3));td[:,:,0]=.35;td[:,:,2]=.08;gt=compose_hard(c,d,td,[0],grid=g);src=d.sources[0];ids=np.arange(len(src.points_world));gxy=np.stack([predicted_source_points_world(src,float(h),td[0,i].numpy(),c.history_ego_to_world[-1])[:,:2] for i,h in enumerate(d.horizons_s)]);targets=TrainingTargets(gt,np.ones_like(gt,bool),{0:MotionTarget(0,np.ones(6,bool),ids,gxy)});delta=torch.zeros((1,6,3),requires_grad=True);scene=render_soft_ordered(c,d,delta,[0],grid=g);occ,_=occupancy_ce_full(scene,targets);mn,n,_=motion_loss_sum(delta,[0],d,targets,c.history_ego_to_world[-1]);mot=mn/n;go=torch.autograd.grad(occ,delta,retain_graph=True)[0];gm=torch.autograd.grad(mot,delta)[0];Go=float(go.norm());Gm=float(gm.norm());cos=float(torch.dot(go.reshape(-1),gm.reshape(-1))/(go.norm()*gm.norm()).clamp_min(1e-12));lam=calibrated_gradient_ratio(Go,Gm,cos,max_ce_antagonistic_fraction_of_motion=.5);assert np.isfinite(lam) and lam>0;opt=torch.optim.Adam([delta],lr=.01);before=np.count_nonzero(compose_hard(c,d,delta,[0],grid=g)!=gt)
    for _ in range(100):
        opt.zero_grad();scene=render_soft_ordered(c,d,delta,[0],grid=g);occ,_=occupancy_ce_full(scene,targets);mn,n,_=motion_loss_sum(delta,[0],d,targets,c.history_ego_to_world[-1]);(occ+lam*mn/n).backward();opt.step()
    assert np.count_nonzero(compose_hard(c,d,delta,[0],grid=g)!=gt)<before;assert soft_argmax_scene(scene).shape==gt.shape
