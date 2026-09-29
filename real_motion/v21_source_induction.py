"""V21 Stage-0 causal source-induction contracts.

No learned V21 model is implemented here. Anchor construction is causal.
Future GT identity/trajectory/occupancy is supervision/oracle-only.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib, json, math
from typing import Mapping, Sequence
import numpy as np
from scipy.ndimage import binary_erosion
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull

from .geometry import OccupancyGrid, quaternion_yaw
from .metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS
from .motion_transport import dynamic_annotations, match_sources_to_annotations
from .nuscenes_adapter import category_to_dynamic_class
from .runtime_fastpath import extract_instances_cropped_exact
from .strong_w2det import StrongW2DetConfig
from .v19_innovation_targets import match_future_components_many_to_one
from .v19_scene_memory import build_dynamic_source_memory

PROTOCOL="p0_f9_v21_stage0_causal_source_induction_v1"
PROTOTYPE_PROTOCOL="v21_train_only_binary_iou_kmedoids_v1"
COMPOSITOR_PROTOCOL="v21_v18_free_first_writer_v1"
FREE_LABEL=17
REPORT_INDICES=(1,3,5)
ALL_HORIZONS_S=(0.5,1.0,1.5,2.0,2.5,3.0)
DYNAMIC_IDS=tuple(int(x) for x in DYNAMIC_CLASS_IDS)
AUDIT_ORIGIN=(-80.0,-80.0)
AUDIT_RES=0.4
AUDIT_SHAPE=(400,400)
ANCHOR_RES=1.6
ANCHOR_SHAPE=(100,100)
SHAPE_RES=0.4
F_QUERY=1
F_VIS=2
F_NEW=4

@dataclass(frozen=True)
class V21Target:
    instance_token:str
    class_id:int
    responsibility:str
    existence:tuple[bool,...]
    center_world:tuple
    yaw_world:tuple
    onset_index:int
    eligible_report_horizon:bool
    non_report_horizon_only:bool
    @property
    def key(self): return int(self.class_id),str(self.instance_token)

@dataclass(frozen=True)
class HistoricalAnchor:
    canonical_anchor_id:int
    track_id:int
    class_id:int
    anchor_xyz_t0:tuple[float,float,float]
    last_seen_index:int
    last_seen_age_s:float
    target_token:str|None
    ambiguous:bool
    merge_tokens:tuple[str,...]

@dataclass(frozen=True)
class FrontierAnchor:
    canonical_anchor_id:int
    anchor_xyz_t0:tuple[float,float,float]
    frontier_type_mask:int
    eligible_horizon_mask:int
    first_eligible_horizon:int

@dataclass(frozen=True)
class CoverageMatch:
    target_token:str
    responsibility:str
    anchor_kind:str
    anchor_id:int
    distance_m:float

@dataclass(frozen=True)
class CanonicalShape:
    class_id:int
    cells_ijk:np.ndarray
    observation_key:tuple[str,str]|None=None
    def __post_init__(self):
        x=np.asarray(self.cells_ijk,dtype=np.int32)
        if x.ndim!=2 or x.shape[1]!=3: raise ValueError("cells_ijk must be [N,3]")
        object.__setattr__(self,"cells_ijk",np.unique(x,axis=0) if len(x) else x)
    @property
    def voxel_count(self): return int(len(self.cells_ijk))

@dataclass(frozen=True)
class ShapeAttribution:
    shape:CanonicalShape|None
    ambiguous:bool
    unresolved:bool
    fragment_count:int

@dataclass(frozen=True)
class PrototypeBank:
    protocol:str
    resolution_m:float
    requested_k:int
    medoids_by_class:Mapping[int,tuple[CanonicalShape,...]]
    population_manifest:tuple[tuple[str,str],...]
    fingerprint:str

@dataclass(frozen=True)
class ComposeReport:
    written_voxels:int
    blocked_by_v18_voxels:int
    v21_collision_voxels:int
    historical_frontier_collision_voxels:int
    out_of_bounds_voxels:int

def stable_json_fingerprint(v):
    s=json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=True)
    return hashlib.sha256(s.encode()).hexdigest()

def annotation_map(nusc,token):
    out={}
    for at in nusc.get("sample",str(token))["anns"]:
        a=nusc.get("sample_annotation",at)
        cid=category_to_dynamic_class(a["category_name"])
        if cid is None: continue
        out[str(a["instance_token"])]={
            "instance_token":str(a["instance_token"]),"class_id":int(cid),
            "center_world":np.asarray(a["translation"],dtype=np.float64),
            "yaw_world":float(quaternion_yaw(a["rotation"])),
        }
    return out

def reliable_components_and_tokens(source,scene,token,sem,obs,pose,*,grid,strong_cfg,match_max_distance_m=4.0):
    masked=np.where(np.asarray(obs,bool),np.asarray(sem),int(strong_cfg.free_label)).astype(np.uint8)
    comps=extract_instances_cropped_exact(masked,np.asarray(pose),grid=grid,cfg=strong_cfg)
    anns=dynamic_annotations(source.nusc,str(token))
    matched=match_sources_to_annotations(comps,anns,max_distance_m=float(match_max_distance_m))
    represented={str(x) for x in matched if x is not None}
    ambiguous=set()
    for ann in anns:
        tok=str(ann["instance_token"])
        if tok in represented: continue
        ac=np.asarray(ann["center_world"])
        if any(int(c["class_id"])==int(ann["class_id"]) and
               np.linalg.norm(np.asarray(c["centroid_world"])[:2]-ac[:2])<=1.5*float(match_max_distance_m)
               for c in comps):
            ambiguous.add(tok)
    return comps,matched,ambiguous

def build_v21_targets(source,window,history_occ,history_observed,history_poses,*,grid,strong_cfg,match_max_distance_m=4.0):
    ms,ambs=[],[]
    for i,tok in enumerate(window.history_tokens):
        _,m,a=reliable_components_and_tokens(
            source,window.scene_name,tok,history_occ[i],history_observed[i],history_poses[i],
            grid=grid,strong_cfg=strong_cfg,match_max_distance_m=match_max_distance_m)
        ms.append({str(x) for x in m if x is not None}); ambs.append(a)
    current=ms[-1]; earlier=set().union(*ms[:-1]); ambiguous=set().union(*ambs)
    fm=[annotation_map(source.nusc,t) for t in window.future_tokens]
    toks=sorted(set().union(*(set(x) for x in fm)))
    audit={k:0 for k in ("current_ancestral","dormant_ancestral","birth","ignore_ambiguous","non_report_horizon_only","eligible_targets")}
    out=[]
    for tok in toks:
        ex=tuple(tok in x for x in fm)
        onset=int(np.flatnonzero(np.asarray(ex))[0])
        cid=int(fm[onset][tok]["class_id"])
        report=any(ex[i] for i in REPORT_INDICES)
        nonreport=any(ex) and not report
        if tok in ambiguous: audit["ignore_ambiguous"]+=1; continue
        if tok in current: audit["current_ancestral"]+=1; continue
        resp="DORMANT_ANCESTRAL" if tok in earlier else "BIRTH"
        audit["dormant_ancestral" if resp.startswith("DORMANT") else "birth"]+=1
        if nonreport: audit["non_report_horizon_only"]+=1
        if report:
            out.append(V21Target(
                tok,cid,resp,ex,
                tuple(tuple(float(z) for z in x[tok]["center_world"]) if tok in x else None for x in fm),
                tuple(float(x[tok]["yaw_world"]) if tok in x else None for x in fm),
                onset,True,False))
            audit["eligible_targets"]+=1
    out.sort(key=lambda x:x.key)
    return out,audit

def _world_to_t0(p,T):
    return (np.linalg.inv(np.asarray(T))@np.r_[np.asarray(p,dtype=np.float64),1.0])[:3]

def build_historical_anchors(source,window,history_occ,history_observed,history_poses,*,grid,strong_cfg,frame_dt_s=0.5,match_max_distance_m=4.0):
    masked=np.where(np.asarray(history_observed,bool),np.asarray(history_occ),int(strong_cfg.free_label)).astype(np.uint8)
    tracks,cbf=build_dynamic_source_memory(
        list(masked),list(np.asarray(history_poses)),grid=grid,strong_cfg=strong_cfg,
        frame_dt_s=float(frame_dt_s),max_missing_s=2.5)
    matched=[]; ambiguous=set()
    for i,tok in enumerate(window.history_tokens):
        anns=dynamic_annotations(source.nusc,str(tok))
        m=match_sources_to_annotations(cbf[i],anns,max_distance_m=float(match_max_distance_m))
        matched.append(m); represented={str(x) for x in m if x is not None}
        for a in anns:
            at=str(a["instance_token"])
            if at in represented: continue
            ac=np.asarray(a["center_world"])
            if any(int(c["class_id"])==int(a["class_id"]) and
                   np.linalg.norm(np.asarray(c["centroid_world"])[:2]-ac[:2])<=1.5*float(match_max_distance_m)
                   for c in cbf[i]): ambiguous.add(at)
    t0=np.asarray(history_poses[-1]); out=[]; trackmap={}
    ages={str(x):0 for x in (0.5,1.0,1.5,2.0,2.5)}
    agevox={str(x):0 for x in (0.5,1.0,1.5,2.0,2.5)}
    merges=0
    dormant=[x for x in tracks if not x.observed_at_anchor]
    for n,tr in enumerate(dormant):
        seen=[]
        for ti in np.flatnonzero(tr.valid_history):
            center=np.asarray(tr.centers_world[int(ti)])
            cand=sorted((float(np.linalg.norm(np.asarray(c["centroid_world"])-center)),j)
                        for j,c in enumerate(cbf[int(ti)]) if int(c["class_id"])==int(tr.class_id))
            if cand and cand[0][0]<=1e-6:
                q=matched[int(ti)][cand[0][1]]
                if q is not None: seen.append(str(q))
        uniq=tuple(sorted(set(seen))); merges+=int(len(uniq)>1)
        tok=uniq[0] if len(uniq)==1 else None
        amb=len(uniq)!=1 or any(x in ambiguous for x in uniq)
        age=float(tr.state_age_s(float(frame_dt_s))); k=f"{age:.1f}"
        if k in ages: ages[k]+=1; agevox[k]+=int(tr.last_component_voxel_count)
        aid=-1-n; xyz=_world_to_t0(tr.anchor_center_world(float(frame_dt_s)),t0)
        out.append(HistoricalAnchor(aid,int(tr.track_id),int(tr.class_id),tuple(map(float,xyz)),
                                    int(tr.last_observed_frame),age,tok,amb,uniq))
        trackmap[aid]=tr
    toks=[x.target_token for x in out if x.target_token is not None]
    return out,trackmap,{
        "last_seen_age_histogram":ages,"last_seen_age_voxel_histogram":agevox,
        "association_ambiguous_tokens":len(ambiguous),"track_merge_count":merges,
        "track_collision_count":len(toks)-len(set(toks)),"historical_anchors":len(out)}

def _observed_union(obs,poses,t0,grid):
    out=np.zeros(AUDIT_SHAPE,bool); inv=np.linalg.inv(np.asarray(t0))
    origin=np.asarray([grid.x_min,grid.y_min,grid.z_min]); step=np.asarray(grid.voxel_size)
    ao=np.asarray(AUDIT_ORIGIN)
    for m,p in zip(obs,poses):
        idx=np.argwhere(np.asarray(m,bool))
        if not len(idx): continue
        xyz=origin+(idx+0.5)*step; T=inv@np.asarray(p); xyz=xyz@T[:3,:3].T+T[:3,3]
        ij=np.floor((xyz[:,:2]-ao)/AUDIT_RES).astype(np.int64)
        v=((ij>=0)&(ij<np.asarray(AUDIT_SHAPE))).all(1); q=ij[v]
        out[q[:,0],q[:,1]]=True
    return out

def _poly_mask(points):
    p=np.asarray(points,float)
    if len(p)<3:return np.zeros(AUDIT_SHAPE,bool)
    poly=p[ConvexHull(p).vertices]
    area=np.sum(poly[:,0]*np.roll(poly[:,1],-1)-poly[:,1]*np.roll(poly[:,0],-1))
    if area<0:poly=poly[::-1]
    ox,oy=AUDIT_ORIGIN
    x0=max(0,int(np.floor((poly[:,0].min()-ox)/AUDIT_RES))-1)
    x1=min(AUDIT_SHAPE[0]-1,int(np.floor((poly[:,0].max()-ox)/AUDIT_RES))+1)
    y0=max(0,int(np.floor((poly[:,1].min()-oy)/AUDIT_RES))-1)
    y1=min(AUDIT_SHAPE[1]-1,int(np.floor((poly[:,1].max()-oy)/AUDIT_RES))+1)
    xs=ox+(np.arange(x0,x1+1)+0.5)*AUDIT_RES; ys=oy+(np.arange(y0,y1+1)+0.5)*AUDIT_RES
    gx,gy=np.meshgrid(xs,ys,indexing="ij"); q=np.stack((gx,gy),-1); inside=np.ones(gx.shape,bool)
    for a,b in zip(poly,np.roll(poly,-1,0)):
        e=b-a; r=q-a; inside&=(e[0]*r[...,1]-e[1]*r[...,0])>=-1e-9
    out=np.zeros(AUDIT_SHAPE,bool); out[x0:x1+1,y0:y1+1]=inside; return out

def _future_footprint(pose,t0,grid):
    corners=np.asarray([[x,y,z] for x in (grid.x_min,grid.x_max)
                       for y in (grid.y_min,grid.y_max) for z in (grid.z_min,grid.z_max)],float)
    T=np.linalg.inv(np.asarray(t0))@np.asarray(pose); q=corners@T[:3,:3].T+T[:3,3]
    return _poly_mask(q[:,:2])

def _boundary(m):
    s=np.asarray([[0,1,0],[1,1,1],[0,1,0]],bool); m=np.asarray(m,bool)
    return m&~binary_erosion(m,structure=s,border_value=0) if m.any() else np.zeros_like(m)

def _cell(xy):
    ij=np.floor((np.asarray(xy)-np.asarray(AUDIT_ORIGIN))/ANCHOR_RES).astype(int)
    return None if ((ij<0)|(ij>=np.asarray(ANCHOR_SHAPE))).any() else (int(ij[0]),int(ij[1]))

def canonical_anchor_id(c): return c[0]*ANCHOR_SHAPE[1]+c[1]
def canonical_anchor_center(c): return np.asarray(AUDIT_ORIGIN)+(np.asarray(c)+0.5)*ANCHOR_RES

def build_frontier_anchors(history_observed,history_poses,future_poses,*,grid):
    t0=np.asarray(history_poses[-1]); H=_observed_union(history_observed,history_poses,t0,grid); Hb=_boundary(H)
    cells={}; counts={"query_boundary":0,"visibility_boundary":0,"new_query_boundary":0}
    for hi,p in enumerate(future_poses):
        Q=_future_footprint(p,t0,grid)
        for m,bit,name in ((_boundary(Q),F_QUERY,"query_boundary"),(Hb&Q,F_VIS,"visibility_boundary"),
                           (_boundary(Q&~H),F_NEW,"new_query_boundary")):
            pts=np.argwhere(m); counts[name]+=len(pts)
            for i,j in pts:
                xy=np.asarray(AUDIT_ORIGIN)+(np.asarray([i,j])+0.5)*AUDIT_RES; c=_cell(xy)
                if c is None:continue
                row=cells.setdefault(c,[0,0]); row[0]|=bit; row[1]|=1<<hi
    out=[]
    for c in sorted(cells,key=canonical_anchor_id):
        typ,hm=cells[c]; first=int((hm&-hm).bit_length()-1); xy=canonical_anchor_center(c)
        out.append(FrontierAnchor(canonical_anchor_id(c),(float(xy[0]),float(xy[1]),0.0),typ,hm,first))
    return out,{"observed_union_bev_cells":int(H.sum()),"frontier_native_boundary_samples":counts,
                "frontier_anchor_count":len(out),"deduplicated_anchor_cells":len(out)}

def assign_causal_coverage(targets,historical,frontier,*,t0_pose,coverage_radius_m):
    radius=float(coverage_radius_m); matches=[]; used=set(); by={t.instance_token:t for t in targets}
    for a in sorted(historical,key=lambda x:(x.canonical_anchor_id,x.track_id)):
        tok=a.target_token; t=by.get(tok)
        if tok is None or a.ambiguous or tok in used or t is None or t.responsibility!="DORMANT_ANCESTRAL":continue
        p=_world_to_t0(t.center_world[t.onset_index],t0_pose); d=float(np.linalg.norm(p[:2]-np.asarray(a.anchor_xyz_t0)[:2]))
        matches.append(CoverageMatch(tok,t.responsibility,"historical",a.canonical_anchor_id,d)); used.add(tok)
    births=sorted([t for t in targets if t.responsibility=="BIRTH" and t.instance_token not in used],key=lambda x:x.key)
    front=sorted(frontier,key=lambda x:x.canonical_anchor_id); legal={t.instance_token:0 for t in births}
    if births and front:
        big=1e9; cost=np.full((len(births),len(front)),big)
        for i,t in enumerate(births):
            p=_world_to_t0(t.center_world[t.onset_index],t0_pose)
            for j,a in enumerate(front):
                if a.first_eligible_horizon>t.onset_index:continue
                d=float(np.linalg.norm(p[:2]-np.asarray(a.anchor_xyz_t0)[:2]))
                if d<=radius+1e-9: legal[t.instance_token]+=1; cost[i,j]=d+(i+1)*(j+1)*1e-12
        ri,ci=linear_sum_assignment(cost)
        for i,j in zip(ri,ci):
            if cost[i,j]>=big/2:continue
            t,a=births[i],front[j]; matches.append(CoverageMatch(t.instance_token,t.responsibility,"frontier",a.canonical_anchor_id,float(cost[i,j])))
    histcand=[int(any(a.target_token==t.instance_token and not a.ambiguous for a in historical))
              for t in targets if t.responsibility=="DORMANT_ANCESTRAL"]
    cc=histcand+list(legal.values()); n=len(targets)
    return matches,{"eligible_targets":n,"covered_targets":len(matches),"component_coverage":len(matches)/max(n,1),
        "historical_matches":sum(x.anchor_kind=="historical" for x in matches),
        "frontier_matches":sum(x.anchor_kind=="frontier" for x in matches),
        "mean_legal_candidates_per_positive":float(np.mean(cc)) if cc else 0.0,
        "max_legal_candidates_per_positive":max(cc,default=0),
        "duplicate_target_assignment":len(matches)-len({x.target_token for x in matches}),
        "duplicate_anchor_assignment":len(matches)-len({(x.anchor_kind,x.anchor_id) for x in matches})}

def _component_world_points(c,pose,grid):
    idx=np.asarray(c["voxel_indices"]); origin=np.asarray([grid.x_min,grid.y_min,grid.z_min]); step=np.asarray(grid.voxel_size)
    p=origin+(idx+0.5)*step; T=np.asarray(pose); return p@T[:3,:3].T+T[:3,3]

def _canon(points,center,yaw):
    r=np.asarray(points)-np.asarray(center); c,s=math.cos(yaw),math.sin(yaw)
    q=np.stack((c*r[:,0]+s*r[:,1],-s*r[:,0]+c*r[:,1],r[:,2]),-1)
    return np.unique(np.rint(q/SHAPE_RES).astype(np.int32),axis=0)

def attribute_instance_shape(semantic,pose,anns,token,*,grid,free_label=17,match_max_distance_m=4.0,observation_key=None):
    token=str(token); ann=anns.get(token)
    if ann is None:return ShapeAttribution(None,False,True,0)
    cfg=StrongW2DetConfig(free_label=int(free_label),min_component_voxels=1)
    comps=extract_instances_cropped_exact(np.asarray(semantic,dtype=np.uint8),np.asarray(pose),grid=grid,cfg=cfg)
    mm=match_future_components_many_to_one(comps,anns,max_distance_m=float(match_max_distance_m))
    chosen=[c for c,(t,_) in zip(comps,mm) if t==token]
    if not chosen:return ShapeAttribution(None,False,True,0)
    same=[a for a in anns.values() if int(a["class_id"])==int(ann["class_id"])]
    for comp in chosen:
        cc=np.asarray(comp["centroid_world"])
        ds=sorted(float(np.linalg.norm(cc[:2]-np.asarray(a["center_world"])[:2])) for a in same)
        if len(ds)>1 and ds[0]<=match_max_distance_m and ds[1]-ds[0]<=0.2:
            return ShapeAttribution(None,True,False,len(chosen))
    points=np.concatenate([_component_world_points(c,pose,grid) for c in chosen])
    return ShapeAttribution(CanonicalShape(int(ann["class_id"]),_canon(points,ann["center_world"],ann["yaw_world"]),observation_key),False,False,len(chosen))

def shape_iou(a,b):
    if a.class_id!=b.class_id:return 0.0
    A={tuple(x) for x in a.cells_ijk.tolist()}; B={tuple(x) for x in b.cells_ijk.tolist()}
    return len(A&B)/max(len(A|B),1)

def deterministic_kmedoids(shapes,k):
    n=len(shapes)
    if not n:return ()
    k=min(int(k),n); D=np.zeros((n,n)); keys=[tuple(x.observation_key or ("",str(i))) for i,x in enumerate(shapes)]
    for i in range(n):
        for j in range(i+1,n):D[i,j]=D[j,i]=1-shape_iou(shapes[i],shapes[j])
    med=[min(range(n),key=lambda i:(D[i].sum(),keys[i],i))]
    while len(med)<k:
        cand=[(np.min(D[:,med+[j]],axis=1).sum(),keys[j],j) for j in range(n) if j not in med]
        med.append(int(min(cand)[2]))
    improved=True
    while improved:
        improved=False; cur=np.min(D[:,med],axis=1).sum(); best=None
        for pos in range(len(med)):
            for j in range(n):
                if j in med:continue
                trial=med.copy(); trial[pos]=j; cost=np.min(D[:,trial],axis=1).sum()
                if cost+1e-12<cur:
                    item=(cost,keys[j],pos,j,trial)
                    if best is None or item[:4]<best[:4]:best=item
        if best is not None:med=best[4]; improved=True
    return tuple(sorted(med,key=lambda i:(keys[i],i)))

def build_prototype_bank(shapes_by_class,*,requested_k,population_manifest):
    med={}
    pop=tuple(sorted({(str(a),str(b)) for a,b in population_manifest}))
    for cid in sorted(shapes_by_class):
        rows=sorted(shapes_by_class[cid],key=lambda x:tuple(x.observation_key or ("","")))
        med[int(cid)]=tuple(rows[i] for i in deterministic_kmedoids(rows,requested_k))
    serial={"protocol":PROTOTYPE_PROTOCOL,"resolution_m":SHAPE_RES,"requested_k":requested_k,
            "population":pop,"medoids":{str(k):[x.observation_key for x in v] for k,v in med.items()}}
    return PrototypeBank(PROTOTYPE_PROTOCOL,SHAPE_RES,int(requested_k),med,pop,stable_json_fingerprint(serial))

def oracle_best_prototype(shape,bank):
    rows=bank.medoids_by_class.get(int(shape.class_id),())
    return min(rows,key=lambda x:(-shape_iou(shape,x),tuple(x.observation_key or ("","")))) if rows else None

def rasterize_canonical_shape(shape,center,yaw,future_pose,*,grid):
    local=np.asarray(shape.cells_ijk,float)*SHAPE_RES; c,s=math.cos(yaw),math.sin(yaw)
    rel=np.stack((c*local[:,0]-s*local[:,1],s*local[:,0]+c*local[:,1],local[:,2]),-1)
    world=rel+np.asarray(center); T=np.linalg.inv(np.asarray(future_pose)); ego=world@T[:3,:3].T+T[:3,3]
    origin=np.asarray([grid.x_min,grid.y_min,grid.z_min]); step=np.asarray(grid.voxel_size)
    idx=np.floor((ego-origin)/step).astype(int); valid=((idx>=0)&(idx<np.asarray(grid.shape_hwd))).all(1)
    return (np.unique(idx[valid],axis=0) if valid.any() else np.empty((0,3),int),int((~valid).sum()))

def compose_v21_add_only(base,proposals,*,free_label=17):
    base=np.asarray(base); out=base.copy(); owner=np.full(base.shape,-1,np.int32); kindmap=np.full(base.shape,-1,np.int8)
    order={"historical":0,"frontier":1}; blocked=coll=cross=written=0
    for pi,(kind,aid,cid,indices) in enumerate(sorted(proposals,key=lambda x:(order[x[0]],int(x[1])))):
        idx=np.asarray(indices,dtype=int)
        if not len(idx):continue
        occ=base[idx[:,0],idx[:,1],idx[:,2]]!=free_label; blocked+=int(occ.sum()); q=idx[~occ]
        if not len(q):continue
        prev=owner[q[:,0],q[:,1],q[:,2]]; used=prev>=0; coll+=int(used.sum())
        pk=kindmap[q[:,0],q[:,1],q[:,2]]; cross+=int((used&(pk!=order[kind])).sum()); q=q[~used]
        if len(q):
            out[q[:,0],q[:,1],q[:,2]]=int(cid); owner[q[:,0],q[:,1],q[:,2]]=pi; kindmap[q[:,0],q[:,1],q[:,2]]=order[kind]; written+=len(q)
    return out,ComposeReport(int(written),blocked,coll,cross,0)

def select_scene_balanced_round_robin(keys,n=64):
    keys=[(str(a),str(b)) for a,b in keys]
    if len(keys)!=len(set(keys)):raise ValueError("duplicate population keys")
    order=[]; by={}
    for k in keys:
        if k[0] not in by:order.append(k[0]); by[k[0]]=[]
        by[k[0]].append(k)
    cur={s:0 for s in order}; out=[]; want=min(int(n),len(keys))
    while len(out)<want:
        progress=False
        for s in order:
            i=cur[s]
            if i<len(by[s]):
                out.append(by[s][i]); cur[s]+=1; progress=True
                if len(out)>=want:break
        if not progress:break
    return tuple(out)
