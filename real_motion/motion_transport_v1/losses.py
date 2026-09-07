from __future__ import annotations
import numpy as np,torch,torch.nn.functional as F
from .geometry import f_to_world_matrix

def occupancy_ce_full(scene,targets,*,eps=1e-6):
    if not 0<float(eps)<1/18:raise ValueError('invalid epsilon')
    device=scene.horizons[0].probabilities.device if scene.horizons else torch.device('cpu');vals=[];vvals=[]
    for hi,row in enumerate(scene.horizons):
        gt=torch.as_tensor(targets.future_semantics[hi],device=device,dtype=torch.long).reshape(-1);valid=torch.as_tensor(targets.future_valid[hi],device=device,dtype=torch.bool).reshape(-1);kta=torch.as_tensor(scene.hard_kta[hi],device=device,dtype=torch.long).reshape(-1);base_p=torch.where(gt==kta,torch.full_like(gt,1-17*eps,dtype=torch.float32),torch.full_like(gt,eps,dtype=torch.float32));base=-torch.log(base_p.clamp_min(eps));total=base[valid].sum();variable=torch.zeros((),device=device)
        if row.flat_indices.numel():
            u=row.flat_indices.long();uv=valid[u]
            if bool(uv.any()):
                old=base[u][uv].sum();p=row.probabilities[uv];q=(1-18*eps)*p+eps;new=-torch.log(q.gather(1,gt[u][uv,None]).squeeze(1).clamp_min(eps)).sum();total=total-old+new;variable=new
        den=valid.sum().clamp_min(1);vals.append(total/den);vvals.append(variable/den)
    loss=torch.stack(vals).mean() if vals else torch.zeros((),device=device);var=torch.stack(vvals).mean() if vvals else torch.zeros((),device=device);return loss,{'occ_full':float(loss.detach()),'occ_variable_region':float(var.detach())}
def _pred_points(source,delta,h,t0_pose,ids):
    p=torch.as_tensor(source.points_world[ids],device=delta.device,dtype=torch.float32);c=torch.as_tensor(source.centroid_world,device=delta.device,dtype=torch.float32);v=torch.as_tensor(source.velocity_world,device=delta.device,dtype=torch.float32);yaw=delta[2];cy,sy=torch.cos(yaw),torch.sin(yaw);dx=p[:,0]-c[0];dy=p[:,1]-c[1];rot=torch.stack([cy*dx-sy*dy,sy*dx+cy*dy],1);R=torch.as_tensor(f_to_world_matrix(t0_pose)[:3,:3],device=delta.device,dtype=torch.float32);dw=R@torch.stack([delta[0],delta[1],delta[0]*0]);return c[None,:2]+v[None,:2]*float(h)+rot+dw[None,:2]
def motion_loss_sum(deltas,selected_source_ids,decomp,targets,t0_pose):
    by={int(s.source_id):s for s in decomp.sources};num=deltas.sum()*0.;count=0;rows=[]
    for mi,sid in enumerate(map(int,selected_source_ids)):
        t=targets.motion_targets.get(sid)
        if t is None:continue
        for hi,h in enumerate(decomp.horizons_s):
            if hi>=len(t.valid) or not bool(t.valid[hi]):continue
            pred=_pred_points(by[sid],deltas[mi,hi],float(h),t0_pose,t.point_indices);gt=torch.as_tensor(t.gt_xy_world[hi],device=deltas.device,dtype=torch.float32);one=.5*F.smooth_l1_loss(pred,gt,beta=1.,reduction='none').sum(1).mean();num=num+one;count+=1;rows.append(float(one.detach()))
    return num,count,{'motion_pairs':count,'motion_mean_local':float(np.mean(rows)) if rows else 0.}
def motion_pair_count(selected_source_ids,targets):return int(sum(int(np.asarray(targets.motion_targets[int(s)].valid,bool).sum()) for s in selected_source_ids if int(s) in targets.motion_targets))
def calibration_probe_like(delta,epsilon=1e-3):
    """Deterministic GT-independent sub-voxel probe used only for lambda calibration."""
    eps=float(epsilon)
    if eps<=0:return delta
    if delta.numel()==0:return delta
    m,h,_=delta.shape;mi=torch.arange(m,device=delta.device)[:,None];hi=torch.arange(h,device=delta.device)[None,:];sgn=torch.where(((mi+hi)&1)==0,torch.ones((m,h),device=delta.device),-torch.ones((m,h),device=delta.device));p=torch.zeros_like(delta);p[:,:,0]=sgn*eps;p[:,:,1]=-sgn*eps;p[:,:,2]=sgn*eps;return delta+p
def calibrated_gradient_ratio(g_occ_norm,g_motion_norm,cosine,*,max_ce_antagonistic_fraction_of_motion=.5):
    """Global recovery calibration.

    ``f`` means the CE gradient norm may be at most ``f`` times the weighted
    GT-motion gradient norm during calibration.  A zero CE gradient contributes
    a zero ratio rather than invalidating the batch; a zero motion gradient is
    unusable for calibration.
    """
    go=float(g_occ_norm);gm=float(g_motion_norm);c=float(cosine);f=float(max_ce_antagonistic_fraction_of_motion)
    if not (np.isfinite(go) and np.isfinite(gm) and np.isfinite(c)) or gm<=0:return float('nan')
    if not 0<f<=1:raise ValueError('max_ce_antagonistic_fraction_of_motion must be in (0,1]')
    if go<=0:return 0.0
    return float(go/(gm*f))
def output_gradient_lambda_floor(g_occ,g_motion,*,max_ce_antagonistic_fraction_of_motion=.5,active_motion_grad_rel=1e-4):
    """Minimum lambda that keeps every active conflicting output coordinate motion-directed."""
    go=torch.as_tensor(g_occ).detach().float().reshape(-1);gm=torch.as_tensor(g_motion).detach().float().reshape(-1);f=float(max_ce_antagonistic_fraction_of_motion);rel=float(active_motion_grad_rel)
    if not 0<f<=1:raise ValueError('max_ce_antagonistic_fraction_of_motion must be in (0,1]')
    if not 0<=rel<1:raise ValueError('active_motion_grad_rel must be in [0,1)')
    if gm.numel()==0:return 0.0
    mx=float(gm.abs().max())
    if not np.isfinite(mx) or mx<=0:return 0.0
    active=gm.abs()>=max(1e-12,mx*rel);conflict=active&(go*gm<0)
    if not bool(conflict.any()):return 0.0
    need=(go[conflict].abs()/gm[conflict].abs().clamp_min(1e-12))/f
    return float(need.max())
def gradient_summary(g):
    x=torch.as_tensor(g).detach().float().abs().reshape(-1)
    if x.numel()==0:return {'p50':0.0,'p95':0.0,'p99':0.0,'max':0.0}
    return {'p50':float(x.quantile(.50)),'p95':float(x.quantile(.95)),'p99':float(x.quantile(.99)),'max':float(x.max())}
def lambda_ratio(progress):
    p=float(progress)
    if p<=.1:return 1.
    if p<.3:return 1.-(p-.1)/.2*.75
    return .25
def lambda_at(progress,lambda_ref):return float(lambda_ref)*lambda_ratio(progress)
