from __future__ import annotations
import numpy as np
from real_motion.geometry import OccupancyGrid,ego_compensate_sequence,relative_transform
from real_motion.prepared import _align_observation_sequence
from real_motion.motion import decompose_masks
from real_motion.kta import KTAConfig,estimate_components
from real_motion.msp import MSPCandidate,candidate_feature,STATE_OBSERVED_MOVING,STATE_DORMANT,CLASS_TO_SLOT
from real_motion.metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS
from real_motion.strong_w2det import StrongW2DetConfig,extract_instances,match_instances,inverse_warp,majority_fill
from .contracts import CausalInputs,SourceRecord,MSPCandidateRecord,SourceDecomposition
from .geometry import index_to_metric_center,transform_points,source_bbox_F,world_to_F

def _validate_times(causal,expected_dt_s,tolerance_s=.05):
    ht=np.asarray(causal.history_timestamps_s,float);ft=np.asarray(causal.future_timestamps_s,float)
    # Match the manifest contract: validate every adjacent keyframe interval,
    # not cumulative distance from t0.  Real nuScenes keyframes can each be a
    # few milliseconds off the nominal 0.5 s cadence, so cumulative drift over
    # six future frames can legitimately exceed the per-step 50 ms tolerance.
    if len(ht)>1 and np.max(np.abs(np.diff(ht)-expected_dt_s))>tolerance_s:raise ValueError('history cadence mismatch')
    if len(ft):
        if len(ht)==0:raise ValueError('future timestamps require a history t0')
        future_steps=np.diff(np.concatenate(([ht[-1]],ft)))
        if np.max(np.abs(future_steps-expected_dt_s))>tolerance_s:raise ValueError('future cadence mismatch')

def decompose_strong_sources(causal,*,grid:OccupancyGrid,cfg:StrongW2DetConfig,frame_dt_s=.5,crop_radius_limit_m=11.2):
    frame_dt_s=float(frame_dt_s);_validate_times(causal,frame_dt_s);hist=np.asarray(causal.history_semantics);sem0,prev=hist[-1],hist[-2];T0=np.asarray(causal.history_ego_to_world[-1],float);Tp=np.asarray(causal.history_ego_to_world[-2],float)
    dyn=np.isin(sem0,np.asarray(DYNAMIC_CLASS_IDS,dtype=sem0.dtype));cur=extract_instances(sem0,T0,grid=grid,cfg=cfg);old=extract_instances(prev,Tp,grid=grid,cfg=cfg);vel=match_instances(old,cur,frame_dt_s,max_speed_mps=cfg.max_match_speed_mps)
    sources=[];covered=np.zeros_like(dyn,bool)
    for j,inst in enumerate(cur):
        idx=np.asarray(inst['voxel_indices'],np.int64);covered[tuple(idx.T)]=True;p0=index_to_metric_center(idx,grid);pw=transform_points(T0,p0);v=np.zeros(3,float);matched=j in vel
        if matched:v[:]=np.asarray(vel[j],float);v[2]=0
        centroid=np.asarray(inst['centroid_world'],float);pF=world_to_F(pw,T0);cF=world_to_F(centroid[None],T0)[0];radius=np.max(np.abs(pF[:,:2]-cF[None,:2]),axis=0);eligible=bool(np.all(radius<=float(crop_radius_limit_m)+1e-12))
        sources.append(SourceRecord(j,int(inst['class_id']),idx,pw,centroid,v,matched,source_bbox_F(pw,T0),len(idx),crop_eligible=eligible,fallback_reason='' if eligible else 'source_xy_radius_exceeds_11p2m'))
    rest_idx=np.argwhere(dyn&~covered).astype(np.int64);rest_labels=sem0[tuple(rest_idx.T)] if len(rest_idx) else np.zeros((0,),dtype=sem0.dtype);rest_world=transform_points(T0,index_to_metric_center(rest_idx,grid)) if len(rest_idx) else np.zeros((0,3),float)
    static=sem0.copy();static[dyn]=int(cfg.free_label);background=[]
    for Th in causal.future_ego_to_world:
        cur_to_future=relative_transform(T0,np.asarray(Th,float));dst,known=inverse_warp(static,cur_to_future,grid,cfg.free_label);background.append(majority_fill(dst,~known,kernel=cfg.fill_kernel,min_fraction=cfg.fill_min_fraction))
    # Strong-W2Det is defined on the frozen 2 Hz protocol: the velocity is a
    # backward difference over frame_dt_s and future propagation uses exactly
    # (i+1)*frame_dt_s.  Real nuScenes timestamps are only used above to validate
    # that a window obeys the cadence tolerance; using their small acquisition
    # jitter here can move boundary points into adjacent voxels and breaks the
    # required bit-exact zero-delta identity with strong_w2det_sequence().
    horizons=np.arange(1,len(causal.future_timestamps_s)+1,dtype=np.float64)*frame_dt_s
    return SourceDecomposition(sources,np.stack(background),rest_idx,np.asarray(rest_labels),rest_world,horizons)

def extract_original_msp_candidates(causal,*,grid,motion_cfg,kta_cfg:KTAConfig):
    aligned=ego_compensate_sequence(causal.history_semantics,causal.history_ego_to_world,-1,grid,17);obs=_align_observation_sequence(causal.history_valid,causal.history_ego_to_world,-1,grid);masks=decompose_masks(aligned,motion_cfg,history_observed=obs);rows=[];enum=0
    for state,mask in ((STATE_OBSERVED_MOVING,masks.moving),(STATE_DORMANT,masks.uncertain)):
        for comp in estimate_components(aligned,mask,grid=grid,cfg=kta_cfg):
            if int(comp.class_id) not in CLASS_TO_SLOT:continue
            cells=np.asarray(comp.bev_cells,np.int64);vx,vy,_=grid.voxel_size;extent=np.asarray([(cells[:,0].max()-cells[:,0].min()+1)*vx,(cells[:,1].max()-cells[:,1].min()+1)*vy],np.float32);cand=MSPCandidate(class_id=int(comp.class_id),state=int(state),centroid_xy_m=np.asarray(comp.centroid_xy_m,np.float32),velocity_xy_mps=np.asarray(comp.velocity_xy_mps,np.float32),extent_xy_m=extent,voxel_count=int(len(comp.voxel_indices)),kta_matched=bool(comp.matched));rows.append((cand,np.asarray(comp.voxel_indices,np.int64),enum));enum+=1
    rows.sort(key=lambda r:(int(r[0].state),int(r[0].class_id),float(r[0].centroid_xy_m[0]),float(r[0].centroid_xy_m[1]),r[2]))
    return [MSPCandidateRecord(i,e,int(c.class_id),int(c.state),vox,candidate_feature(c)) for i,(c,vox,e) in enumerate(rows)]
def _flat(idx,shape):
    if len(idx)==0:return np.zeros((0,),np.int64)
    _,Y,Z=map(int,shape);q=np.asarray(idx,np.int64);return (q[:,0]*Y+q[:,1])*Z+q[:,2]
def map_msp_to_sources(decomp,candidates,*,shape_xyz):
    cf={c.candidate_id:np.unique(_flat(c.voxel_indices_t0,shape_xyz)) for c in candidates};by={}
    for c in candidates:by.setdefault(int(c.class_id),[]).append(c)
    for s in decomp.sources:
        sf=np.unique(_flat(s.voxel_indices_t0,shape_xyz));ids=[];counts=[];states=[]
        for c in by.get(int(s.class_id),[]):
            n=int(np.intersect1d(sf,cf[c.candidate_id],assume_unique=True).size)
            if n:ids.append(c.candidate_id);counts.append(n);states.append(c.state)
        total=sum(counts);s.msp_candidate_indices=tuple(ids);s.overlap_weights=np.asarray(counts,float)/total if total else np.zeros((0,),float);s.mapping_coverage=float(total/max(1,len(sf)));s.observed_moving_fraction=float(sum(n for n,st in zip(counts,states) if st==STATE_OBSERVED_MOVING)/max(1,len(sf)));s.dormant_fraction=float(sum(n for n,st in zip(counts,states) if st==STATE_DORMANT)/max(1,len(sf)))
    return decomp
def source_metadata_19(source,t0_ego_to_world):
    from .geometry import f_to_world_matrix
    cF=world_to_F(source.centroid_world[None],t0_ego_to_world)[0];R=f_to_world_matrix(t0_ego_to_world)[:3,:3];vF=source.velocity_world@R;speed=float(np.linalg.norm(vF[:2]));extent=source.bbox_F[1]-source.bbox_F[0];feat=[cF[0]/40,cF[1]/40,vF[0]/20,vF[1]/20,speed/20,np.log1p(source.voxel_count)/8,extent[0]/10,extent[1]/10,1. if source.kta_matched else 0.,source.observed_moving_fraction,source.dormant_fraction];oh=[0.]*len(DYNAMIC_CLASS_IDS);oh[CLASS_TO_SLOT[int(source.class_id)]]=1.;feat+=oh;out=np.asarray(feat,np.float32)
    if out.shape!=(19,):raise AssertionError(out.shape)
    return out
