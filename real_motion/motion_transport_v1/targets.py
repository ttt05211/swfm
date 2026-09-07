from __future__ import annotations
import math,numpy as np
from real_motion.geometry import quaternion_yaw
from real_motion.nuscenes_adapter import category_to_dynamic_class
from .contracts import SourceDecomposition,TrainingTargets,MotionTarget

def _ann_map(nusc,sample_token):
    sample=nusc.get('sample',sample_token);out={}
    for tok in sample['anns']:
        ann=nusc.get('sample_annotation',tok);out[str(ann['instance_token'])]=ann
    return out
def _inside(points,ann):
    p=np.asarray(points,float);c=np.asarray(ann['translation'],float);yaw=quaternion_yaw(ann['rotation']);cy,sy=math.cos(yaw),math.sin(yaw);d=p-c;lx=cy*d[:,0]+sy*d[:,1];ly=-sy*d[:,0]+cy*d[:,1];w,l,h=map(float,ann['size']);return (np.abs(lx)<=l/2)&(np.abs(ly)<=w/2)&(np.abs(d[:,2])<=h/2)
def _sample_points(p,max_points=64):
    n=len(p)
    if n<=max_points:return np.arange(n,dtype=np.int64)
    p=np.asarray(p,float);chosen=[]
    for i in (np.argmin(p[:,0]),np.argmax(p[:,0]),np.argmin(p[:,1]),np.argmax(p[:,1])):
        if int(i) not in chosen:chosen.append(int(i))
    for i in np.linspace(0,n-1,num=max_points,dtype=np.int64):
        if int(i) not in chosen:chosen.append(int(i))
        if len(chosen)>=max_points:break
    return np.asarray(sorted(chosen[:max_points]),np.int64)
def build_training_targets(source,window,decomp:SourceDecomposition,*,best_coverage_min=.8,second_coverage_max=.2,max_points=64):
    fs=[];fv=[]
    for tok in window.future_tokens:
        s,v=source.load_occ3d(window.scene_name,tok,require_lidar_mask=True);fs.append(np.asarray(s));fv.append(np.asarray(v,bool))
    t0=_ann_map(source.nusc,window.t0_token);fmap=[_ann_map(source.nusc,t) for t in window.future_tokens];dyn=[]
    for token,ann in t0.items():
        cid=category_to_dynamic_class(ann['category_name'])
        if cid is not None:dyn.append((token,int(cid),ann))
    mt={};H=len(window.future_tokens)
    for s in decomp.sources:
        cover=[]
        for token,cid,ann in dyn:
            if cid==int(s.class_id):cover.append((float(_inside(s.points_world,ann).mean()) if len(s.points_world) else 0.,token,ann))
        cover.sort(key=lambda x:(-x[0],x[1]));best=cover[0][0] if cover else 0.;second=cover[1][0] if len(cover)>1 else 0.;valid=np.zeros(H,bool);speed=np.full(H,np.nan,np.float32);pi=_sample_points(s.points_world,max_points);gt=np.zeros((H,len(pi),2),np.float32);token=None
        if cover and best>=best_coverage_min and second<=second_coverage_max:
            token=str(cover[0][1]);ann0=cover[0][2];b0=np.asarray(ann0['translation'],float);yaw0=quaternion_yaw(ann0['rotation']);pts=s.points_world[pi]
            for hi,fm in enumerate(fmap):
                ann=fm.get(token)
                if ann is None:continue
                try:yaw=quaternion_yaw(ann['rotation'])
                except Exception:continue
                bt=np.asarray(ann['translation'],float);dyaw=float(yaw-yaw0);c,ss=math.cos(dyaw),math.sin(dyaw);d=pts[:,:2]-b0[None,:2];rot=np.stack([c*d[:,0]-ss*d[:,1],ss*d[:,0]+c*d[:,1]],1);gt[hi]=(bt[None,:2]+rot).astype(np.float32);dt=float(decomp.horizons_s[hi]);speed[hi]=float(np.linalg.norm(bt[:2]-b0[:2])/dt) if dt>0 else np.nan;valid[hi]=True
        mt[int(s.source_id)]=MotionTarget(int(s.source_id),valid,pi,gt,token,float(best),float(second),speed)
    return TrainingTargets(np.stack(fs),np.stack(fv),mt)
def gt_moving_source_ids(targets,threshold_mps=.5):
    out=[]
    for sid,t in targets.motion_targets.items():
        if t.gt_speed_mps is None:continue
        v=np.asarray(t.valid,bool);s=np.asarray(t.gt_speed_mps,float)
        if np.any(v&np.isfinite(s)&(s>=threshold_mps)):out.append(int(sid))
    return tuple(sorted(out))
