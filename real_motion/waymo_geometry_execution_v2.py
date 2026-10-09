"""Exact tube scatter and bounded native surface fits; no method/cache changes."""
from dataclasses import dataclass
import math

import numpy as np
from .geometry import relative_transform, _occupied_indices_to_xyz, _xyz_to_indices
from .local_st_world_model import (HISTORY_FRAMES, DEFAULT_PATCH_SIZE_M,
    top_surface_semantic, extract_bev_patch, priority_pool2x2)
from .waymo_geometry_execution import FrameGeometry, FrameGeometryCache, readonly, ChunkedSurfaceAtlas
from .surface_canonical_repair import NEIGHBORS, RADIUS, SURFACE_DIM
from .canonical_causal_repair import STATIC
from .source_evidence_audit import transform_points
from .source_evidence_audit import Registration
from scipy.spatial import cKDTree

PROTOCOL = 'waymo_exact_tube_scatter_surface_native_v2'


@dataclass(frozen=True)
class TubeFrameGeometry(FrameGeometry):
    warp_homogeneous: np.ndarray
    warp_labels: np.ndarray
    registration_sorted: tuple

    @property
    def nbytes(self):
        return (super().nbytes+self.warp_homogeneous.nbytes+self.warp_labels.nbytes
                +sum(a.nbytes for a in self.registration_sorted))


class TubeFrameCache(FrameGeometryCache):
    def _build(self,occupancy,visibility,pose):
        f=super()._build(occupancy,visibility,pose)
        indices=np.argwhere(np.asarray(occupancy)!=17)
        xyz=_occupied_indices_to_xyz(indices,self.grid)
        homogeneous=np.concatenate((xyz,np.ones((len(xyz),1),np.float64)),axis=1)
        labels=np.asarray(occupancy)[tuple(indices.T)]
        return TubeFrameGeometry(f.components,f.registration_points,f.canonical_points,
            f.static_indices,f.static_classes,f.static_world,readonly(homogeneous),readonly(labels),
            tuple(readonly(p[np.lexsort(p.T[::-1])]) for p in f.registration_points))


def registration_reference(reference):
    """Ephemeral current-source tree, same sort/subsample; shared across 3 histories."""
    q=np.asarray(reference,np.float64).reshape(-1,3)
    if not np.isfinite(q).all(): raise ValueError('nonfinite registration reference')
    if len(q)<6: return len(q),None,None
    q=q[np.lexsort(q.T[::-1])]
    b=q[np.linspace(0,len(q)-1,min(512,len(q)),dtype=int),:2]
    return len(q),b,cKDTree(b)


def register_presorted(points,sorted_points,prepared_reference,*,allow_yaw=True):
    """Original deterministic trimmed ICP; reuse ONLY immutable sorts/current tree."""
    q_count,b,tree=prepared_reference
    p=np.asarray(points,np.float64).reshape(-1,3)
    if not np.isfinite(p).all(): raise ValueError('nonfinite registration points')
    if len(p)<6 or q_count<6: return Registration(p.copy(),False,0.,float('inf'),0.)
    p=sorted_points; a=p[np.linspace(0,len(p)-1,min(512,len(p)),dtype=int),:2]
    r=np.eye(2); t=b.mean(0)-a.mean(0)
    for _ in range(5):
        d,j=tree.query(a@r.T+t,k=1,workers=1); keep=d<=1.6
        if int(keep.sum())<6:
            return Registration(p.copy(),False,0.,float(np.median(d)),float(keep.mean()))
        ids=np.flatnonzero(keep)
        ids=ids[np.argsort(d[ids],kind='stable')[:max(6,int(len(ids)*.8))]]
        aa,bb=a[ids],b[j[ids]]
        if allow_yaw:
            u,_,vt=np.linalg.svd((aa-aa.mean(0)).T@(bb-bb.mean(0)))
            r_new=vt.T@u.T
            if np.linalg.det(r_new)<0:
                vt[-1]*=-1; r_new=vt.T@u.T
            angle=math.atan2(r_new[1,0],r_new[0,0])
            if abs(angle)>math.pi/4:
                return Registration(p.copy(),False,angle,float(np.median(d)),float(keep.mean()))
            r=r_new
        t=bb.mean(0)-aa.mean(0)@r.T
    d,_=tree.query(a@r.T+t,k=1,workers=1); fraction=float((d<=1.6).mean())
    out=p.copy(); out[:,:2]=p[:,:2]@r.T+t
    return Registration(out,fraction>=.5,math.atan2(r[1,0],r[0,0]),float(np.median(d)),fraction)


def aligned_bev(frame,transform,grid,native):
    """Preserve entire original [4,N] GEMM and distances before integer winner loop."""
    if not len(frame.warp_labels): return np.full(grid.shape_hwd[:2],17,np.uint8)
    xyz=(np.asarray(transform,np.float64)@frame.warp_homogeneous.T).T[:,:3]
    ix,iy,iz,valid=_xyz_to_indices(xyz,grid)
    ix,iy,iz=ix[valid],iy[valid],iz[valid]; xyz=xyz[valid]
    vx,vy,vz=grid.voxel_size
    centers=np.stack((grid.x_min+(ix+.5)*vx,grid.y_min+(iy+.5)*vy,grid.z_min+(iz+.5)*vz),axis=1)
    dist=np.sum((xyz-centers)**2,axis=1)
    flat=(ix*grid.shape_hwd[1]+iy)*grid.shape_hwd[2]+iz
    aligned=native.warp(flat,dist,frame.warp_labels[valid],grid.shape_hwd)
    return top_surface_semantic(aligned,grid=grid)


def build_tubes(history,poses,frames,source_xy,offsets,valid,grid,native,pool,*,defer=False):
    """Four active observations, same six-slot ABI with two provably empty slots."""
    size=int(round(DEFAULT_PATCH_SIZE_M/grid.voxel_size[0]))
    if (size%2 or abs(grid.voxel_size[0]-grid.voxel_size[1])>1e-9
            or int(round(.8/grid.voxel_size[0]))!=2):
        raise ValueError('unchanged square 2x pooled tube grid required')
    n=len(source_xy); out=np.full((n,HISTORY_FRAMES,size//2,size//2),17,np.uint8)
    if not n: return (lambda:out) if defer else out
    def one(t):
        bev=(top_surface_semantic(history[t],grid=grid) if t==3 else
             aligned_bev(frames[t],relative_transform(poses[t],poses[-1]),grid,native))
        rows=np.empty((n,size//2,size//2),np.uint8)
        for i in range(n):
            center=source_xy[i]+offsets[i,t+2] if valid[i,t+2] else source_xy[i]
            rows[i]=priority_pool2x2(extract_bev_patch(bev,center,grid=grid,patch_voxels=size))
        return t+2,rows
    futures=[pool.submit(one,t) for t in range(4)] if pool is not None else None
    def finish():
        for t,rows in (map(one,range(4)) if futures is None else (f.result() for f in futures)):
            out[:,t]=rows
        return out
    return finish if defer else finish()


def describe_native_rows(atlas,cls,query,native):
    distance,index=atlas.trees[cls].query(query*atlas.scale,k=NEIGHBORS,
        distance_upper_bound=RADIUS,workers=1)
    valid=np.isfinite(distance); safe=np.minimum(index,len(atlas.metric[cls])-1)
    delta=(atlas.metric[cls][safe]-query[:,None])/atlas.step
    delta=np.where(valid[...,None],delta,0.)
    weight=np.where(valid,1./(1.+np.where(valid,distance,0.)**2),0.)
    other=atlas.trees[13 if cls==11 else 11]; opposite=np.full(len(query),np.inf)
    if other is not None:
        opposite,_=other.query(query*atlas.scale,k=1,distance_upper_bound=RADIUS,workers=1)
    seen=np.isfinite(opposite); opposite=np.where(seen,opposite/RADIUS,1.)
    return native.fit(delta,weight,distance,atlas.recent[cls][safe],opposite,seen.astype(np.uint8))


class NativeSurfaceAtlas(ChunkedSurfaceAtlas):
    def describe(self,evidence):
        chunk=int(self.chunk_rows)
        if not 256<=chunk<=65536: raise ValueError('bounded surface chunk required')
        result=np.zeros((len(evidence),SURFACE_DIM),np.float32); jobs=[]
        for cls in (11,13):
            ids=np.flatnonzero((evidence.actor==STATIC)&(evidence.classes==cls))
            if not len(ids) or self.trees[cls] is None: continue
            query=transform_points(evidence.world[ids],self.inverse)
            jobs += [(cls,ids[b:b+chunk],query[b:b+chunk]) for b in range(0,len(ids),chunk)]
        def work(job): return job[1],describe_native_rows(self,job[0],job[2],self.native)
        for ids,values in (map(work,jobs) if self.fit_pool is None else self.fit_pool.map(work,jobs)):
            result[ids]=values
        if not np.isfinite(result).all(): raise RuntimeError('nonfinite native surface descriptor')
        return result
