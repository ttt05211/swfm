from __future__ import annotations
import numpy as np,torch
from real_motion.geometry import OccupancyGrid
from .contracts import CropBatch
from .geometry import f_to_world_matrix,world_to_F,transform_points,metric_to_floor_index,in_grid,index_to_metric_center
from .source_adapter import source_metadata_19

def build_history_crops(causal,sources,selected_source_ids,*,grid:OccupancyGrid,crop_shape_xyz=(64,64,16),xy_resolution_m=.4,mirror=False,device='cpu'):
    selected=[int(x) for x in selected_source_ids];by={int(s.source_id):s for s in sources};M=len(selected);T=causal.history_frames;CX,CY,CZ=map(int,crop_shape_xyz)
    if CZ!=grid.shape_hwd[2]:raise ValueError('V1 crop must keep full occupancy height')
    sem=np.full((M,T,CX,CY,CZ),18,np.int64);valid=np.zeros_like(sem,bool);sm=np.zeros((M,CX,CY),bool);meta=np.zeros((M,19),np.float32);rel=np.asarray(causal.history_timestamps_s,float)-float(causal.history_timestamps_s[-1]);T0=np.asarray(causal.history_ego_to_world[-1],float);F0=f_to_world_matrix(T0)
    gx=(np.arange(CX)-(CX-1)/2.)*float(xy_resolution_m);gy=(np.arange(CY)-(CY-1)/2.)*float(xy_resolution_m);z=np.asarray([grid.z_min+(k+.5)*grid.voxel_size[2] for k in range(CZ)])
    for mi,sid in enumerate(selected):
        s=by[sid]
        if not s.crop_eligible:raise ValueError(f'source {sid} is not crop eligible')
        cF=world_to_F(s.centroid_world[None],T0)[0];vF=s.velocity_world@F0[:3,:3];meta[mi]=source_metadata_19(s,T0)
        for ti,dt in enumerate(rel):
            center=cF[:2]+vF[:2]*float(dt);xx,yy,zz=np.meshgrid(center[0]+gx,center[1]+gy,z,indexing='ij');pF=np.stack([xx,yy,zz],-1).reshape(-1,3);pw=transform_points(F0,pF);ph=transform_points(np.linalg.inv(np.asarray(causal.history_ego_to_world[ti],float)),pw);idx=metric_to_floor_index(ph,grid);ok=in_grid(idx,grid);flat=np.full((len(idx),),18,np.int64);vf=np.zeros((len(idx),),bool);q=idx[ok];flat[ok]=np.asarray(causal.history_semantics[ti])[tuple(q.T)];vf[ok]=np.asarray(causal.history_valid[ti],bool)[tuple(q.T)];sem[mi,ti]=flat.reshape(CX,CY,CZ);valid[mi,ti]=vf.reshape(CX,CY,CZ)
        pF=world_to_F(s.points_world,T0);ix=np.floor((pF[:,0]-cF[0])/xy_resolution_m+CX/2).astype(int);iy=np.floor((pF[:,1]-cF[1])/xy_resolution_m+CY/2).astype(int);ok=(ix>=0)&(ix<CX)&(iy>=0)&(iy<CY);sm[mi,ix[ok],iy[ok]]=True
        if not sm[mi].any():raise RuntimeError(f'empty source mask for eligible source {sid}')
    flags=np.full((M,),bool(mirror),bool)
    return CropBatch(np.asarray(selected,np.int64),torch.as_tensor(sem,device=device),torch.as_tensor(valid,device=device),torch.as_tensor(sm,device=device),torch.as_tensor(np.broadcast_to(rel,(M,T)).copy(),device=device,dtype=torch.float32),torch.as_tensor(meta,device=device),torch.as_tensor(flags,device=device))
