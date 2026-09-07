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
def lambda_ratio(progress):
    p=float(progress)
    if p<=.1:return 1.
    if p<.3:return 1.-(p-.1)/.2*.75
    return .25
def lambda_at(progress,lambda_ref):return float(lambda_ref)*lambda_ratio(progress)
