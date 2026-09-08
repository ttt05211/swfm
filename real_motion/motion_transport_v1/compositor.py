from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn.functional as F
from real_motion.geometry import OccupancyGrid
from .contracts import CausalInputs,SourceDecomposition,SourceRecord,SoftHorizon,SoftScene
from .geometry import transform_points,metric_to_floor_index,in_grid,f_to_world_matrix,metric_to_center_index

def _stable_last_write(out,idx,labels):
    if len(idx)==0:return
    _,Y,Z=out.shape;flat=(idx[:,0]*Y+idx[:,1])*Z+idx[:,2];seq=np.arange(len(flat),dtype=np.int64);order=np.lexsort((seq,flat));sf=flat[order];last=np.ones(len(order),bool);last[:-1]=sf[:-1]!=sf[1:];q=idx[order[last]];out[q[:,0],q[:,1],q[:,2]]=labels[order[last]]
def predicted_source_points_world(source,horizon_s,delta,t0_pose):
    d=np.asarray(delta,dtype=np.float64);yaw=float(d[2]);c,s=math.cos(yaw),math.sin(yaw);R=np.asarray([[c,-s,0],[s,c,0],[0,0,1.]],dtype=np.float64);rel=source.points_world-source.centroid_world[None];dW=f_to_world_matrix(t0_pose)[:3,:3]@np.asarray([d[0],d[1],0.]);return source.centroid_world[None]+source.velocity_world[None]*float(horizon_s)+rel@R.T+dW[None]
def compose_hard(causal,decomp,deltas,selected_source_ids,*,grid):
    selected=list(map(int,selected_source_ids));dnp=deltas.detach().float().cpu().numpy() if torch.is_tensor(deltas) else np.asarray(deltas,float)
    if dnp.shape!=(len(selected),len(decomp.horizons_s),3):raise ValueError('delta shape mismatch')
    db={sid:dnp[i] for i,sid in enumerate(selected)};T0=np.asarray(causal.history_ego_to_world[-1],float);outs=[]
    for hi,(h,Th) in enumerate(zip(decomp.horizons_s,causal.future_ego_to_world)):
        out=np.asarray(decomp.background_future[hi]).copy();pts=[];labs=[]
        for src in decomp.sources:
            d=db.get(int(src.source_id),np.zeros((len(decomp.horizons_s),3)))[hi];p=predicted_source_points_world(src,float(h),d,T0);pts.append(p);labs.append(np.full(len(p),int(src.class_id),dtype=out.dtype))
        if len(decomp.rest_points_world):pts.append(np.asarray(decomp.rest_points_world));labs.append(np.asarray(decomp.rest_labels,dtype=out.dtype))
        if pts:
            pf=transform_points(np.linalg.inv(np.asarray(Th,float)),np.concatenate(pts));lab=np.concatenate(labs);idx=metric_to_floor_index(pf,grid);ok=in_grid(idx,grid);_stable_last_write(out,idx[ok],lab[ok])
        outs.append(out)
    return np.stack(outs)
def hard_kta_identity(causal,decomp,*,grid):return compose_hard(causal,decomp,torch.zeros((0,len(decomp.horizons_s),3)),(),grid=grid)
def _source_template(source):
    idx=np.asarray(source.voxel_indices_t0,np.int64);lo=idx.min(0)-1;hi=idx.max(0)+1;arr=np.zeros(tuple((hi-lo+1).astype(int)),np.float32);q=idx-lo;arr[q[:,0],q[:,1],q[:,2]]=1.;return lo,hi,arr
def _sample_grid_antialiased(tmpl,norm_chunk,dims,jitter_voxels):
    """Forward/backward-consistent anti-aliased soft occupancy sample.

    PyTorch 2.6 uses a one-sided derivative at exact trilinear knots. Combined
    with CE near epsilon this can create a spurious first-step gradient. The
    training-only surrogate instead approximates the occupancy averaged over the
    XY voxel footprint using four symmetric quarter-voxel quadrature points.
    Probability and autograd gradient therefore belong to the same smooth
    surrogate. Hard deployment/evaluation remains exact forward-floor transport.
    """
    g=norm_chunk.reshape(1,1,1,-1,3);j=float(jitter_voxels)
    if j<=0:return F.grid_sample(tmpl,g,mode='bilinear',padding_mode='zeros',align_corners=False).reshape(-1)
    offs=torch.tensor([[j,j,0.],[j,-j,0.],[-j,j,0.],[-j,-j,0.]],device=norm_chunk.device,dtype=norm_chunk.dtype);offs=2*offs/dims[None];vals=[]
    for off in offs:
        gj=(norm_chunk+off).reshape(1,1,1,-1,3);vals.append(F.grid_sample(tmpl,gj,mode='bilinear',padding_mode='zeros',align_corners=False).reshape(-1))
    return torch.stack(vals).mean(0)
def _torch_rigid_inverse_sample(source,delta,horizon_s,future_pose,flat_indices,*,grid,template_cache,query_chunk,center_gradient_jitter_voxels=.25):
    device=delta.device;lo,_,arr=template_cache[int(source.source_id)];tmpl=torch.as_tensor(arr,device=device,dtype=torch.float32).permute(2,1,0)[None,None];_,Y,Z=grid.shape_hwd;flat=flat_indices.long();ix=flat//(Y*Z);rem=flat%(Y*Z);iy=rem//Z;iz=rem%Z;step=torch.tensor(grid.voxel_size,device=device,dtype=torch.float32);origin=torch.tensor([grid.x_min,grid.y_min,grid.z_min],device=device,dtype=torch.float32);pf=origin+(torch.stack([ix,iy,iz],1).float()+.5)*step;T=torch.as_tensor(np.asarray(future_pose,np.float32),device=device);pw=pf@T[:3,:3].T+T[:3,3];c=torch.as_tensor(source.centroid_world,device=device,dtype=torch.float32);v=torch.as_tensor(source.velocity_world,device=device,dtype=torch.float32);RWF=torch.as_tensor(template_cache[(int(source.source_id),'R_WF')],device=device);dW=RWF@torch.stack([delta[0],delta[1],delta[0]*0]);yaw=delta[2];cy,sy=torch.cos(yaw),torch.sin(yaw);Rinv=torch.stack([torch.stack([cy,sy,yaw*0]),torch.stack([-sy,cy,yaw*0]),torch.stack([yaw*0,yaw*0,yaw*0+1])]);qw=c+(pw-c-v*float(horizon_s)-dW)@Rinv.T;Ti=torch.as_tensor(template_cache[(int(source.source_id),'world_to_t0')],device=device);qe=qw@Ti[:3,:3].T+Ti[:3,3];u=(qe-origin)/step-.5-torch.as_tensor(lo,device=device,dtype=torch.float32);dims=torch.tensor(arr.shape,device=device,dtype=torch.float32);norm=2*(u+.5)/dims-1;outs=[]
    for start in range(0,len(flat),max(1,int(query_chunk))):
        sl=slice(start,start+query_chunk);outs.append(_sample_grid_antialiased(tmpl,norm[sl],dims,center_gradient_jitter_voxels))
    return torch.cat(outs).clamp(0,1) if outs else torch.zeros((0,),device=device)
def _aabb_flat_indices(source,delta_np,h,future_pose,t0_pose,grid,halo):
    lo,hi,_=_source_template(source);step=np.asarray(grid.voxel_size,float);origin=np.asarray([grid.x_min,grid.y_min,grid.z_min]);mins=origin+lo*step;maxs=origin+(hi+1)*step;corners=np.asarray([[x,y,z] for x in (mins[0],maxs[0]) for y in (mins[1],maxs[1]) for z in (mins[2],maxs[2])]);pw=transform_points(t0_pose,corners);yaw=float(delta_np[2]);c,s=math.cos(yaw),math.sin(yaw);R=np.array([[c,-s,0],[s,c,0],[0,0,1.]]);dW=f_to_world_matrix(t0_pose)[:3,:3]@np.array([delta_np[0],delta_np[1],0]);pred=source.centroid_world+source.velocity_world*float(h)+(pw-source.centroid_world)@R.T+dW;pf=transform_points(np.linalg.inv(np.asarray(future_pose,float)),pred);u=metric_to_center_index(pf,grid);ilo=np.floor(u.min(0)).astype(int)-int(halo);ihi=np.ceil(u.max(0)).astype(int)+int(halo);shape=np.asarray(grid.shape_hwd);ilo=np.maximum(ilo,0);ihi=np.minimum(ihi,shape-1)
    if np.any(ilo>ihi):return np.zeros((0,),np.int64)
    xx,yy,zz=np.meshgrid(np.arange(ilo[0],ihi[0]+1),np.arange(ilo[1],ihi[1]+1),np.arange(ilo[2],ihi[2]+1),indexing='ij');_,Y,Z=grid.shape_hwd;return ((xx.reshape(-1)*Y+yy.reshape(-1))*Z+zz.reshape(-1)).astype(np.int64)
def _hard_source_dest_flats(source,h,future_pose,t0_pose,grid):
    pw=predicted_source_points_world(source,h,(0,0,0),t0_pose);pf=transform_points(np.linalg.inv(np.asarray(future_pose,float)),pw);idx=metric_to_floor_index(pf,grid);idx=idx[in_grid(idx,grid)];_,Y,Z=grid.shape_hwd;return np.unique((idx[:,0]*Y+idx[:,1])*Z+idx[:,2])
def _query_domain_flats(selected,by,pos,deltas,hi,h,Th,T0,grid,halo_voxels,full_grid_reference):
    if full_grid_reference:return np.arange(int(np.prod(grid.shape_hwd)),dtype=np.int64)
    parts=[]
    for sid in selected:
        src=by[sid];parts.append(_hard_source_dest_flats(src,float(h),Th,T0,grid));parts.append(_aabb_flat_indices(src,deltas[pos[sid],hi].detach().float().cpu().numpy(),float(h),Th,T0,grid,halo_voxels))
    parts=[p for p in parts if len(p)]
    return np.unique(np.concatenate(parts)) if parts else np.zeros((0,),np.int64)
def render_soft_ordered(causal,decomp,deltas,selected_source_ids,*,grid,class_count=18,halo_voxels=2,query_chunk=65536,full_grid_reference=False,center_gradient_jitter_voxels=.25):
    selected=list(map(int,selected_source_ids))
    if tuple(deltas.shape)!=(len(selected),len(decomp.horizons_s),3):raise ValueError('delta shape mismatch')
    by={int(s.source_id):s for s in decomp.sources};pos={sid:i for i,sid in enumerate(selected)};T0=np.asarray(causal.history_ego_to_world[-1],float);hard=hard_kta_identity(causal,decomp,grid=grid);cache={}
    for s in decomp.sources:cache[int(s.source_id)]=_source_template(s);cache[(int(s.source_id),'R_WF')]=f_to_world_matrix(T0)[:3,:3].astype(np.float32);cache[(int(s.source_id),'world_to_t0')]=np.linalg.inv(T0).astype(np.float32)
    soft=[];_,Y,Z=grid.shape_hwd
    for hi,(h,Th) in enumerate(zip(decomp.horizons_s,causal.future_ego_to_world)):
        if not selected:
            soft.append(SoftHorizon(torch.zeros(0,dtype=torch.long,device=deltas.device),torch.zeros((0,class_count),device=deltas.device),torch.zeros(0,dtype=torch.long,device=deltas.device)));continue
        U=_query_domain_flats(selected,by,pos,deltas,hi,h,Th,T0,grid,halo_voxels,full_grid_reference);uf=torch.as_tensor(U,device=deltas.device,dtype=torch.long);bg=np.asarray(decomp.background_future[hi]).reshape(-1)[U];P=F.one_hot(torch.as_tensor(bg,device=deltas.device,dtype=torch.long),num_classes=class_count).float()
        for s in decomp.sources:
            sid=int(s.source_id);a=_torch_rigid_inverse_sample(s,deltas[pos[sid],hi],float(h),Th,uf,grid=grid,template_cache=cache,query_chunk=query_chunk,center_gradient_jitter_voxels=center_gradient_jitter_voxels) if sid in pos else torch.as_tensor(np.isin(U,_hard_source_dest_flats(s,float(h),Th,T0,grid)),device=deltas.device,dtype=torch.float32)
            if len(U):cls=F.one_hot(torch.tensor(int(s.class_id),device=deltas.device),num_classes=class_count).float();P=(1-a[:,None])*P+a[:,None]*cls[None]
        if len(decomp.rest_points_world) and len(U):
            pf=transform_points(np.linalg.inv(np.asarray(Th,float)),decomp.rest_points_world);idx=metric_to_floor_index(pf,grid);ok=in_grid(idx,grid);idx=idx[ok];lab=decomp.rest_labels[ok];rf=(idx[:,0]*Y+idx[:,1])*Z+idx[:,2];winner={int(f):int(l) for f,l in zip(rf.tolist(),lab.tolist())};where={int(f):i for i,f in enumerate(U.tolist())}
            for f,l in winner.items():
                j=where.get(f)
                if j is not None:P[j]=F.one_hot(torch.tensor(l,device=P.device),num_classes=class_count).float()
        soft.append(SoftHorizon(uf,P,torch.as_tensor(hard[hi].reshape(-1)[U],device=deltas.device,dtype=torch.long)))
    return SoftScene(soft,hard,tuple(grid.shape_hwd))
def soft_argmax_scene(scene):
    out=np.asarray(scene.hard_kta).copy()
    for hi,row in enumerate(scene.horizons):
        if row.flat_indices.numel():out[hi].reshape(-1)[row.flat_indices.detach().cpu().numpy()]=row.probabilities.argmax(1).detach().cpu().numpy()
    return out
