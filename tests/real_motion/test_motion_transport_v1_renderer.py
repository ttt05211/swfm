import numpy as np,torch
from real_motion.geometry import OccupancyGrid
from real_motion.motion_transport_v1.contracts import CausalInputs,SourceRecord,SourceDecomposition,TrainingTargets,MotionTarget
from real_motion.motion_transport_v1.compositor import render_soft_ordered,compose_hard,soft_argmax_scene,predicted_source_points_world
from real_motion.motion_transport_v1.losses import occupancy_ce_full,motion_loss_sum,calibration_probe_like,calibrated_gradient_ratio,output_gradient_lambda_floor
from real_motion.motion_transport_v1.geometry import index_to_metric_center
def fixture(shape=(10,10,4),origin=(-2,-2,-.8),source_origin=(4,4)):
    g=OccupancyGrid(origin[0],origin[1],origin[2],(.4,.4,.4),shape);T=np.eye(4);hist=np.full((6,*g.shape_hwd),17,np.uint8);x,y=source_origin;idx=np.array([[x,y,1],[x,y+1,1],[x+1,y,1],[x+1,y+1,1],[x,y,2],[x+1,y+1,2]]);hist[-1][tuple(idx.T)]=4;hist[-2]=hist[-1];c=CausalInputs('s','s',hist,np.ones_like(hist,bool),tuple([T]*6),tuple([T]*6),np.arange(6)*.5,2.5+np.arange(1,7)*.5);p=index_to_metric_center(idx,g);s=SourceRecord(0,4,idx,p,p.mean(0),np.zeros(3),True,np.array([[-.4,-.4],[.4,.4]]),len(idx));d=SourceDecomposition([s],np.full((6,*g.shape_hwd),17,np.uint8),np.zeros((0,3),int),np.zeros(0,np.uint8),np.zeros((0,3)),np.arange(1,7)*.5);return g,c,d
def _targets_for_delta(g,c,d,td):
    gt=compose_hard(c,d,td,[0],grid=g);src=d.sources[0];ids=np.arange(len(src.points_world));gxy=np.stack([predicted_source_points_world(src,float(h),td[0,i].numpy(),c.history_ego_to_world[-1])[:,:2] for i,h in enumerate(d.horizons_s)]);return gt,TrainingTargets(gt,np.ones_like(gt,bool),{0:MotionTarget(0,np.ones(6,bool),ids,gxy)})
def _calibrated_lambda(g,c,d,targets):
    base=torch.zeros((1,6,3),requires_grad=True);probe=calibration_probe_like(base,1e-3);scene=render_soft_ordered(c,d,probe,[0],grid=g);occ,_=occupancy_ce_full(scene,targets);mn,n,_=motion_loss_sum(probe,[0],d,targets,c.history_ego_to_world[-1]);mot=mn/n;go=torch.autograd.grad(occ,base,retain_graph=True)[0];gm=torch.autograd.grad(mot,base)[0];Go=float(go.norm());Gm=float(gm.norm());cos=float(torch.dot(go.reshape(-1),gm.reshape(-1))/(go.norm()*gm.norm()).clamp_min(1e-12));head=calibrated_gradient_ratio(Go,Gm,cos,max_ce_antagonistic_fraction_of_motion=.5);floor=output_gradient_lambda_floor(go,gm,max_ce_antagonistic_fraction_of_motion=.5,active_motion_grad_rel=1e-4);lam=max(head if np.isfinite(head) else 0.,floor);return lam,go,gm
def _optimize(g,c,d,targets,lam,steps=100):
    delta=torch.zeros((1,6,3),requires_grad=True);opt=torch.optim.Adam([delta],lr=.01);gt=targets.future_semantics;before=np.count_nonzero(compose_hard(c,d,delta,[0],grid=g)!=gt)
    for _ in range(steps):
        opt.zero_grad();scene=render_soft_ordered(c,d,delta,[0],grid=g);occ,_=occupancy_ce_full(scene,targets);mn,n,_=motion_loss_sum(delta,[0],d,targets,c.history_ego_to_world[-1]);(occ+lam*mn/n).backward();opt.step()
    after=np.count_nonzero(compose_hard(c,d,delta,[0],grid=g)!=gt);return before,after,delta.detach(),scene
def _loss_grads(g,c,d,targets,v):
    x=v.detach().clone().requires_grad_(True);scene=render_soft_ordered(c,d,x,[0],grid=g);occ,_=occupancy_ce_full(scene,targets);mn,n,_=motion_loss_sum(x,[0],d,targets,c.history_ego_to_world[-1]);mot=mn/n;go=torch.autograd.grad(occ,x,retain_graph=True)[0];gm=torch.autograd.grad(mot,x)[0]
    return {'occ':float(occ.detach()),'motion':float(mot.detach()),'g_occ':go.detach(),'g_motion':gm.detach()}
def _grad_row(g,c,d,targets,v):
    x=_loss_grads(g,c,d,targets,v);go=x['g_occ'];gm=x['g_motion'];return {'occ':x['occ'],'motion':x['motion'],'go_norm':float(go.norm()),'gm_norm':float(gm.norm()),'go_h0':go[0,0].tolist(),'gm_h0':gm[0,0].tolist(),'dot':float((go*gm).sum())}
def _one_sided_ce(g,c,d,targets,axis,eps=1e-4):
    z=torch.zeros((1,6,3));f0=_loss_grads(g,c,d,targets,z)['occ'];zp=z.clone();zm=z.clone();zp[0,0,axis]+=eps;zm[0,0,axis]-=eps;fp=_loss_grads(g,c,d,targets,zp)['occ'];fm=_loss_grads(g,c,d,targets,zm)['occ'];return {'backward':(f0-fm)/eps,'forward':(fp-f0)/eps}
def test_soft_probability_gradient_and_chunk_equivalence():
    g,c,d=fixture();a=torch.zeros((1,6,3),requires_grad=True);b=a.detach().clone().requires_grad_(True);a.data[:,:,0]=.11;a.data[:,:,2]=.07;b.data.copy_(a.data);sa=render_soft_ordered(c,d,a,[0],grid=g,query_chunk=7);sb=render_soft_ordered(c,d,b,[0],grid=g,query_chunk=99999)
    for x,y in zip(sa.horizons,sb.horizons):assert torch.equal(x.flat_indices,y.flat_indices) and torch.allclose(x.probabilities.sum(1),torch.ones(len(x.probabilities)),atol=1e-5) and torch.allclose(x.probabilities,y.probabilities,atol=1e-7,rtol=1e-6)
    la=sum(x.probabilities.square().sum() for x in sa.horizons);lb=sum(x.probabilities.square().sum() for x in sb.horizons);la.backward();lb.backward();assert torch.allclose(a.grad,b.grad,atol=2e-6,rtol=2e-5)
def test_full_ce_constant_region_and_old_support_cleared():
    g,c,d=fixture();z=torch.zeros((0,6,3),requires_grad=True);scene=render_soft_ordered(c,d,z,[],grid=g);gt=scene.hard_kta.copy();gt[:,0,0,0]=4;loss,_=occupancy_ce_full(scene,TrainingTargets(gt,np.ones_like(gt,bool),{}));assert float(loss)>.01
    delta=torch.zeros((1,6,3),requires_grad=True);delta.data[:,:,0]=1.2;s=render_soft_ordered(c,d,delta,[0],grid=g);old=compose_hard(c,d,torch.zeros_like(delta),[0],grid=g)
    for hi,row in enumerate(s.horizons):assert set(np.flatnonzero(old[hi].reshape(-1)==4)).issubset(set(row.flat_indices.cpu().tolist()))
def test_large_shift_and_out_of_grid_clear_old_support_and_match_full_grid():
    g,c,d=fixture(shape=(32,32,4),origin=(-6.4,-6.4,-.8),source_origin=(15,15));zero=torch.zeros((1,6,3));old=compose_hard(c,d,zero,[0],grid=g)
    for dx in (4.0,20.0):
        delta=torch.zeros((1,6,3),requires_grad=True);delta.data[:,:,0]=dx;local=render_soft_ordered(c,d,delta,[0],grid=g);full=render_soft_ordered(c,d,delta,[0],grid=g,full_grid_reference=True);la=soft_argmax_scene(local);fa=soft_argmax_scene(full);old_mask=old==4;assert np.count_nonzero(la[old_mask]==4)==0;assert np.array_equal(la,fa)
        for lr,fr in zip(local.horizons,full.horizons):
            fmap={int(f):i for i,f in enumerate(fr.flat_indices.tolist())};take=torch.tensor([fmap[int(f)] for f in lr.flat_indices.tolist()],dtype=torch.long);assert torch.allclose(lr.probabilities,fr.probabilities[take],atol=1e-6,rtol=1e-6)
def test_renderer_finite_difference_dx_and_yaw():
    g,c,d=fixture();target=torch.zeros((1,6,3));target[:,:,0]=.35;target[:,:,2]=.08;_,targets=_targets_for_delta(g,c,d,target);x=torch.zeros((1,6,3),requires_grad=True);x.data[:,:,0]=.137;x.data[:,:,2]=.043
    def f(v):scene=render_soft_ordered(c,d,v,[0],grid=g);loss,_=occupancy_ce_full(scene,targets);return loss
    y=f(x);grad=torch.autograd.grad(y,x)[0];eps=1e-4
    for j in (0,2):
        xp=x.detach().clone();xm=x.detach().clone();xp[0,2,j]+=eps;xm[0,2,j]-=eps;fd=float((f(xp)-f(xm))/(2*eps));assert np.isfinite(fd);assert np.isclose(float(grad[0,2,j]),fd,rtol=.03,atol=3e-4),(j,float(grad[0,2,j]),fd)
def test_joint_calibrated_reachability_improves_hard_original_acceptance_case():
    g,c,d=fixture();td=torch.zeros((1,6,3));td[:,:,0]=.35;td[:,:,2]=.08;gt,targets=_targets_for_delta(g,c,d,td);lam,go,gm=_calibrated_lambda(g,c,d,targets);assert np.isfinite(lam) and lam>0;assert float(((go+lam*gm)*gm).sum())>0;before,after,delta,scene=_optimize(g,c,d,targets,lam,100);assert before==48;assert after<before,(before,after,delta[0,0].tolist(),lam);assert float(delta[:,:,0].mean())>0;assert soft_argmax_scene(scene).shape==gt.shape
def test_eight_direction_reachability_matrix_hard_trend():
    g,c,d=fixture();cases=[(.35,0,.08),(-.35,0,-.08),(0,.35,.08),(0,-.35,-.08),(.35,.35,0),(-.35,.35,0),(.35,-.35,0),(-.35,-.35,0)];rows=[];failed=[]
    for dx,dy,yaw in cases:
        td=torch.zeros((1,6,3));td[:,:,0]=dx;td[:,:,1]=dy;td[:,:,2]=yaw;_,targets=_targets_for_delta(g,c,d,td);lam,_,_=_calibrated_lambda(g,c,d,targets);zero=torch.zeros((1,6,3));probe=calibration_probe_like(zero,1e-3);zero_diag=_grad_row(g,c,d,targets,zero);probe_diag=_grad_row(g,c,d,targets,probe);delta=torch.zeros((1,6,3),requires_grad=True);opt=torch.optim.Adam([delta],lr=.01);gt=targets.future_semantics;before=int(np.count_nonzero(compose_hard(c,d,delta,[0],grid=g)!=gt));snap={}
        for step in range(100):
            opt.zero_grad();scene=render_soft_ordered(c,d,delta,[0],grid=g);occ,_=occupancy_ce_full(scene,targets);mn,n,_=motion_loss_sum(delta,[0],d,targets,c.history_ego_to_world[-1]);(occ+lam*mn/n).backward();opt.step()
            if step in (0,9,24,49,99):snap[str(step+1)]={'delta_h0':delta.detach()[0,0].tolist(),'grad':_grad_row(g,c,d,targets,delta.detach())}
        after=int(np.count_nonzero(compose_hard(c,d,delta,[0],grid=g)!=gt));state=opt.state[delta];row={'target':[dx,dy,yaw],'before':before,'after':after,'lambda':lam,'zero':zero_diag,'probe':probe_diag,'one_sided_dx':_one_sided_ce(g,c,d,targets,0),'one_sided_dy':_one_sided_ce(g,c,d,targets,1),'snapshots':snap,'adam_exp_avg_h0':state['exp_avg'][0,0].tolist(),'adam_exp_avg_sq_h0':state['exp_avg_sq'][0,0].tolist()};rows.append(row)
        if not (before>0 and after<before):failed.append(row)
    assert not failed,{'failed':failed,'all':rows}
