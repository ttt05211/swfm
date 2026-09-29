"""V21 Stage-0 causal source-induction contracts.

No learned V21 model is implemented here. Anchor construction is causal.
Future GT identity/trajectory/occupancy is supervision/oracle-only.
"""
from __future__ import annotations
from dataclasses import dataclass, field
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

PROTOCOL="p0_f9_v21_stage0_causal_source_induction_v2"
POPULATION_PROTOCOL="p0_f9_v21_population_manifest_v2"
PROTOTYPE_PROTOCOL="v21_train_only_binary_iou_kmedoids_v2"
SHAPE_POOL_PROTOCOL="v21_train_only_canonical_shape_pool_v1"
COMPOSITOR_PROTOCOL="v21_v18_free_first_writer_v1"
FREE_LABEL=17
REPORT_INDICES=(1,3,5)
ALL_HORIZONS_S=(0.5,1.0,1.5,2.0,2.5,3.0)
DYNAMIC_IDS=tuple(int(x) for x in DYNAMIC_CLASS_IDS)
ANCHOR_RES=1.6
SHAPE_RES=0.4
KMEDOIDS_EXACT_MAX_N=512
KMEDOIDS_CLARA_SAMPLE_SIZE=256
KMEDOIDS_CLARA_TRIALS=5
F_QUERY=1
F_VIS=2
F_NEW=4

@dataclass(frozen=True)
class AnchorLattice:
    """Frozen t0-canonical BEV lattice inherited from the Stage-1 index."""
    origin_xy_m:tuple[float,float]
    audit_resolution_m:float
    audit_shape_xy:tuple[int,int]
    anchor_resolution_m:float=ANCHOR_RES
    anchor_shape_xy:tuple[int,int]=field(init=False)

    def __post_init__(self):
        origin=tuple(float(x) for x in self.origin_xy_m)
        shape=tuple(int(x) for x in self.audit_shape_xy)
        audit=float(self.audit_resolution_m); anchor=float(self.anchor_resolution_m)
        if len(origin)!=2 or len(shape)!=2 or min(shape)<=0:
            raise ValueError("invalid V21 BEV lattice origin/shape")
        if audit<=0 or anchor<=0:
            raise ValueError("V21 lattice resolutions must be positive")
        ratio=anchor/audit
        if not math.isclose(ratio,round(ratio),rel_tol=0.0,abs_tol=1e-9):
            raise ValueError("anchor resolution must be an integer multiple of audit resolution")
        object.__setattr__(self,"origin_xy_m",origin)
        object.__setattr__(self,"audit_shape_xy",shape)
        object.__setattr__(self,"audit_resolution_m",audit)
        object.__setattr__(self,"anchor_resolution_m",anchor)
        object.__setattr__(self,"anchor_shape_xy",tuple(int(math.ceil(x/round(ratio))) for x in shape))

    @classmethod
    def from_stage1_index(cls,index):
        high=dict(index.get("highres_lattice") or {})
        origin=high.get("origin_xyz_m"); step=high.get("voxel_size_xyz_m"); shape=high.get("shape_xyz")
        if origin is None or step is None or shape is None:
            raise RuntimeError("Stage-1 index lacks frozen highres_lattice")
        if len(origin)!=3 or len(step)!=3 or len(shape)!=3:
            raise RuntimeError("malformed Stage-1 highres_lattice")
        if not math.isclose(float(step[0]),float(step[1]),rel_tol=0.0,abs_tol=1e-9):
            raise RuntimeError("V21 requires isotropic Stage-1 BEV resolution")
        return cls((float(origin[0]),float(origin[1])),float(step[0]),(int(shape[0]),int(shape[1])))

    def to_dict(self):
        return {"origin_xy_m":list(self.origin_xy_m),"audit_resolution_m":self.audit_resolution_m,
                "audit_shape_xy":list(self.audit_shape_xy),"anchor_resolution_m":self.anchor_resolution_m,
                "anchor_shape_xy":list(self.anchor_shape_xy)}

    @classmethod
    def from_dict(cls,value):
        out=cls(tuple(value["origin_xy_m"]),float(value["audit_resolution_m"]),
                tuple(value["audit_shape_xy"]),float(value["anchor_resolution_m"]))
        declared=tuple(int(x) for x in value.get("anchor_shape_xy",out.anchor_shape_xy))
        if declared!=out.anchor_shape_xy:
            raise RuntimeError("V21 manifest anchor-shape mismatch")
        return out

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
    local_xyz_m:np.ndarray|None=None
    def __post_init__(self):
        x=np.asarray(self.cells_ijk,dtype=np.int32)
        if x.ndim!=2 or x.shape[1]!=3: raise ValueError("cells_ijk must be [N,3]")
        x=np.unique(x,axis=0) if len(x) else x.reshape(0,3)
        local=(x.astype(np.float64)*SHAPE_RES if self.local_xyz_m is None
               else np.asarray(self.local_xyz_m,dtype=np.float64))
        if local.ndim!=2 or local.shape[1]!=3:
            raise ValueError("local_xyz_m must be [M,3]")
        if bool(len(x))!=bool(len(local)):
            raise ValueError("quantized and continuous shape representations disagree on emptiness")
        if len(local):
            local=np.unique(local,axis=0)
        else:
            local=local.reshape(0,3)
        object.__setattr__(self,"cells_ijk",x)
        object.__setattr__(self,"local_xyz_m",local)
    @property
    def voxel_count(self): return int(len(self.cells_ijk))

@dataclass(frozen=True)
class ShapeAttribution:
    shape:CanonicalShape|None
    ambiguous:bool
    unresolved:bool
    fragment_count:int
    voxel_indices:np.ndarray|None=None

@dataclass(frozen=True)
class PrototypeBank:
    protocol:str
    resolution_m:float
    requested_k:int
    medoids_by_class:Mapping[int,tuple[CanonicalShape,...]]
    population_manifest:tuple[tuple[str,str],...]
    fingerprint:str
    population_shape_fingerprint:str=""
    algorithm:str=""
    algorithm_config:Mapping[str,object]=field(default_factory=dict)
    source_provenance:Mapping[str,object]=field(default_factory=dict)

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
        w_m,l_m,h_m=(float(x) for x in a["size"])
        out[str(a["instance_token"])]={
            "instance_token":str(a["instance_token"]),"class_id":int(cid),
            "center_world":np.asarray(a["translation"],dtype=np.float64),
            "yaw_world":float(quaternion_yaw(a["rotation"])),
            "size_lwh":np.asarray([l_m,w_m,h_m],dtype=np.float64),
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

def build_v21_targets(source,window,history_occ,history_observed,history_poses,*,grid,strong_cfg,
                      match_max_distance_m=4.0,history_matches=None,history_ambiguous=None):
    if (history_matches is None)!=(history_ambiguous is None):
        raise ValueError("history_matches/history_ambiguous must be supplied together")
    ms,ambs=[],[]
    if history_matches is None:
        for i,tok in enumerate(window.history_tokens):
            _,m,a=reliable_components_and_tokens(
                source,window.scene_name,tok,history_occ[i],history_observed[i],history_poses[i],
                grid=grid,strong_cfg=strong_cfg,match_max_distance_m=match_max_distance_m)
            ms.append({str(x) for x in m if x is not None}); ambs.append(set(a))
    else:
        if len(history_matches)!=6 or len(history_ambiguous)!=6:
            raise ValueError("expected six precomputed history-evidence frames")
        ms=[{str(x) for x in row if x is not None} for row in history_matches]
        ambs=[{str(x) for x in row} for row in history_ambiguous]
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

def build_historical_anchors(source,window,history_occ,history_observed,history_poses,*,grid,strong_cfg,frame_dt_s=0.5,match_max_distance_m=4.0,workers=1):
    masked=np.where(np.asarray(history_observed,bool),np.asarray(history_occ),int(strong_cfg.free_label)).astype(np.uint8)
    tracks,cbf=build_dynamic_source_memory(
        list(masked),list(np.asarray(history_poses)),grid=grid,strong_cfg=strong_cfg,
        frame_dt_s=float(frame_dt_s),max_missing_s=2.5,workers=int(workers))
    matched=[]; matched_sets=[]; ambiguous=set(); ambiguous_sets=[]
    for i,tok in enumerate(window.history_tokens):
        anns=dynamic_annotations(source.nusc,str(tok))
        m=match_sources_to_annotations(cbf[i],anns,max_distance_m=float(match_max_distance_m))
        matched.append(m); represented={str(x) for x in m if x is not None}; matched_sets.append(represented)
        amb_frame=set()
        for a in anns:
            at=str(a["instance_token"])
            if at in represented: continue
            ac=np.asarray(a["center_world"])
            if any(int(c["class_id"])==int(a["class_id"]) and
                   np.linalg.norm(np.asarray(c["centroid_world"])[:2]-ac[:2])<=1.5*float(match_max_distance_m)
                   for c in cbf[i]):
                ambiguous.add(at)
                amb_frame.add(at)
        ambiguous_sets.append(amb_frame)
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
        "track_collision_count":len(toks)-len(set(toks)),"historical_anchors":len(out)}, {
        "matched_sets":tuple(frozenset(x) for x in matched_sets),
        "ambiguous_sets":tuple(frozenset(x) for x in ambiguous_sets),
        "components_by_frame":tuple(tuple(x) for x in cbf),
        "component_matches_by_frame":tuple(tuple(x) for x in matched),
    }

def _observed_union(obs,poses,t0,grid,lattice):
    out=np.zeros(lattice.audit_shape_xy,bool); inv=np.linalg.inv(np.asarray(t0))
    origin=np.asarray([grid.x_min,grid.y_min,grid.z_min]); step=np.asarray(grid.voxel_size)
    ao=np.asarray(lattice.origin_xy_m)
    for m,p in zip(obs,poses):
        idx=np.argwhere(np.asarray(m,bool))
        if not len(idx): continue
        xyz=origin+(idx+0.5)*step; T=inv@np.asarray(p); xyz=xyz@T[:3,:3].T+T[:3,3]
        ij=np.floor((xyz[:,:2]-ao)/lattice.audit_resolution_m).astype(np.int64)
        v=((ij>=0)&(ij<np.asarray(lattice.audit_shape_xy))).all(1); q=ij[v]
        out[q[:,0],q[:,1]]=True
    return out

def _poly_mask(points,lattice):
    p=np.asarray(points,float)
    if len(p)<3:return np.zeros(lattice.audit_shape_xy,bool)
    poly=p[ConvexHull(p).vertices]
    area=np.sum(poly[:,0]*np.roll(poly[:,1],-1)-poly[:,1]*np.roll(poly[:,0],-1))
    if area<0:poly=poly[::-1]
    ox,oy=lattice.origin_xy_m; res=lattice.audit_resolution_m
    x0=max(0,int(np.floor((poly[:,0].min()-ox)/res))-1)
    x1=min(lattice.audit_shape_xy[0]-1,int(np.floor((poly[:,0].max()-ox)/res))+1)
    y0=max(0,int(np.floor((poly[:,1].min()-oy)/res))-1)
    y1=min(lattice.audit_shape_xy[1]-1,int(np.floor((poly[:,1].max()-oy)/res))+1)
    if x0>x1 or y0>y1:return np.zeros(lattice.audit_shape_xy,bool)
    xs=ox+(np.arange(x0,x1+1)+0.5)*res; ys=oy+(np.arange(y0,y1+1)+0.5)*res
    gx,gy=np.meshgrid(xs,ys,indexing="ij"); q=np.stack((gx,gy),-1); inside=np.ones(gx.shape,bool)
    for a,b in zip(poly,np.roll(poly,-1,0)):
        e=b-a; r=q-a; inside&=(e[0]*r[...,1]-e[1]*r[...,0])>=-1e-9
    out=np.zeros(lattice.audit_shape_xy,bool); out[x0:x1+1,y0:y1+1]=inside; return out

def _future_footprint(pose,t0,grid,lattice):
    corners=np.asarray([[x,y,z] for x in (grid.x_min,grid.x_max)
                       for y in (grid.y_min,grid.y_max) for z in (grid.z_min,grid.z_max)],float)
    T=np.linalg.inv(np.asarray(t0))@np.asarray(pose); q=corners@T[:3,:3].T+T[:3,3]
    return _poly_mask(q[:,:2],lattice)

def _boundary(m):
    s=np.asarray([[0,1,0],[1,1,1],[0,1,0]],bool); m=np.asarray(m,bool)
    return m&~binary_erosion(m,structure=s,border_value=0) if m.any() else np.zeros_like(m)

def _cell(xy,lattice):
    ij=np.floor((np.asarray(xy)-np.asarray(lattice.origin_xy_m))/lattice.anchor_resolution_m).astype(int)
    return None if ((ij<0)|(ij>=np.asarray(lattice.anchor_shape_xy))).any() else (int(ij[0]),int(ij[1]))

def canonical_anchor_id(c,lattice): return c[0]*lattice.anchor_shape_xy[1]+c[1]
def canonical_anchor_center(c,lattice): return np.asarray(lattice.origin_xy_m)+(np.asarray(c)+0.5)*lattice.anchor_resolution_m

def build_frontier_anchors(history_observed,history_poses,future_poses,*,grid,lattice):
    if not isinstance(lattice,AnchorLattice):raise TypeError("lattice must be AnchorLattice")
    t0=np.asarray(history_poses[-1]); H=_observed_union(history_observed,history_poses,t0,grid,lattice); Hb=_boundary(H)
    cells={}; counts={"query_boundary":0,"visibility_boundary":0,"new_query_boundary":0}
    for hi,p in enumerate(future_poses):
        Q=_future_footprint(p,t0,grid,lattice)
        for m,bit,name in ((_boundary(Q),F_QUERY,"query_boundary"),(Hb&Q,F_VIS,"visibility_boundary"),
                           (_boundary(Q&~H),F_NEW,"new_query_boundary")):
            pts=np.argwhere(m); counts[name]+=len(pts)
            for i,j in pts:
                xy=np.asarray(lattice.origin_xy_m)+(np.asarray([i,j])+0.5)*lattice.audit_resolution_m; c=_cell(xy,lattice)
                if c is None:continue
                row=cells.setdefault(c,[0,0]); row[0]|=bit; row[1]|=1<<hi
    out=[]
    for c in sorted(cells,key=lambda x:canonical_anchor_id(x,lattice)):
        typ,hm=cells[c]; first=int((hm&-hm).bit_length()-1); xy=canonical_anchor_center(c,lattice)
        out.append(FrontierAnchor(canonical_anchor_id(c,lattice),(float(xy[0]),float(xy[1]),0.0),typ,hm,first))
    return out,{"observed_union_bev_cells":int(H.sum()),"frontier_native_boundary_samples":counts,
                "frontier_anchor_count":len(out),"deduplicated_anchor_cells":len(out),
                "lattice":lattice.to_dict()}

def assign_causal_coverage(targets,historical,frontier,*,t0_pose,coverage_radius_m,
                           onset_index_by_token=None):
    radius=float(coverage_radius_m); matches=[]; used=set(); by={t.instance_token:t for t in targets}
    onset_index_by_token={} if onset_index_by_token is None else {
        str(k):int(v) for k,v in onset_index_by_token.items()}
    def onset(t):
        idx=onset_index_by_token.get(t.instance_token,int(t.onset_index))
        if idx<0 or idx>=len(t.center_world) or t.center_world[idx] is None:
            raise ValueError(f"invalid coverage onset for {t.instance_token}: {idx}")
        return idx
    for a in sorted(historical,key=lambda x:(x.canonical_anchor_id,x.track_id)):
        tok=a.target_token; t=by.get(tok)
        if tok is None or a.ambiguous or tok in used or t is None or t.responsibility!="DORMANT_ANCESTRAL":continue
        oi=onset(t); p=_world_to_t0(t.center_world[oi],t0_pose); d=float(np.linalg.norm(p[:2]-np.asarray(a.anchor_xyz_t0)[:2]))
        matches.append(CoverageMatch(tok,t.responsibility,"historical",a.canonical_anchor_id,d)); used.add(tok)
    births=sorted([t for t in targets if t.responsibility=="BIRTH" and t.instance_token not in used],key=lambda x:x.key)
    front=sorted(frontier,key=lambda x:x.canonical_anchor_id); legal={t.instance_token:0 for t in births}
    if births and front:
        big=1e9; cost=np.full((len(births),len(front)),big); distance=np.full_like(cost,np.inf)
        for i,t in enumerate(births):
            oi=onset(t); p=_world_to_t0(t.center_world[oi],t0_pose)
            for j,a in enumerate(front):
                if a.first_eligible_horizon>oi:continue
                d=float(np.linalg.norm(p[:2]-np.asarray(a.anchor_xyz_t0)[:2]))
                if d<=radius+1e-9:
                    legal[t.instance_token]+=1; distance[i,j]=d
                    cost[i,j]=d+(i+1)*(j+1)*1e-12
        ri,ci=linear_sum_assignment(cost)
        for i,j in zip(ri,ci):
            if cost[i,j]>=big/2:continue
            t,a=births[i],front[j]
            matches.append(CoverageMatch(
                t.instance_token,t.responsibility,"frontier",a.canonical_anchor_id,float(distance[i,j])))
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
    local=np.stack((c*r[:,0]+s*r[:,1],-s*r[:,0]+c*r[:,1],r[:,2]),-1)
    return np.unique(np.rint(local/SHAPE_RES).astype(np.int32),axis=0),local

def _points_inside_oriented_box(points_world,ann,margin_m=0.2):
    pts=np.asarray(points_world,dtype=np.float64)
    rel=pts-np.asarray(ann["center_world"],dtype=np.float64)[None]
    yaw=float(ann["yaw_world"]); c,s=math.cos(yaw),math.sin(yaw)
    local=np.stack((c*rel[:,0]+s*rel[:,1],-s*rel[:,0]+c*rel[:,1],rel[:,2]),axis=-1)
    half=0.5*np.asarray(ann["size_lwh"],dtype=np.float64)+float(margin_m)
    return (np.abs(local)<=half[None]).all(axis=1)

def attribute_instance_shapes(semantic,pose,anns,*,grid,free_label=17,match_max_distance_m=4.0,
                              tokens=None,observation_keys=None,components=None,component_matches=None):
    """Attribute all requested instance shapes with one component extraction.

    A component is accepted only when its oriented-box overlap set is exactly
    the annotation token selected by the frozen nearest-same-class matcher.
    Everything else fails closed for every involved target token.
    """
    requested=sorted(str(x) for x in (anns if tokens is None else tokens))
    observation_keys={} if observation_keys is None else {str(k):v for k,v in observation_keys.items()}
    if (components is None)!=(component_matches is None):
        raise ValueError("components/component_matches must be supplied together")
    if components is None:
        cfg=StrongW2DetConfig(free_label=int(free_label),min_component_voxels=1)
        comps=extract_instances_cropped_exact(
            np.asarray(semantic,dtype=np.uint8),np.asarray(pose),grid=grid,cfg=cfg)
        mm=match_future_components_many_to_one(comps,anns,max_distance_m=float(match_max_distance_m))
    else:
        comps=list(components)
        if len(comps)!=len(component_matches):
            raise ValueError("precomputed component/match lengths differ")
        mm=[((None if x is None else str(x)),float("nan")) for x in component_matches]
    fragments={tok:[] for tok in requested}; indices={tok:[] for tok in requested}
    assigned_count={tok:0 for tok in requested}; ambiguous=set()
    by_class={}
    for ann in anns.values():by_class.setdefault(int(ann["class_id"]),[]).append(ann)
    for comp,(assigned,_) in zip(comps,mm):
        points=_component_world_points(comp,pose,grid)
        overlaps={str(a["instance_token"]) for a in by_class.get(int(comp["class_id"]),())
                  if bool(_points_inside_oriented_box(points,a).any())}
        assigned=None if assigned is None else str(assigned)
        if assigned in assigned_count:assigned_count[assigned]+=1
        if assigned is None or overlaps!={assigned}:
            ambiguous.update(overlaps)
            if assigned is not None:ambiguous.add(assigned)
            continue
        if assigned in fragments:
            fragments[assigned].append(points)
            indices[assigned].append(np.asarray(comp["voxel_indices"],dtype=np.int64))
    out={}
    for token in requested:
        ann=anns.get(token); count=int(assigned_count[token])
        if ann is None:
            out[token]=ShapeAttribution(None,False,True,0,None); continue
        if token in ambiguous:
            out[token]=ShapeAttribution(None,True,False,count,None); continue
        if not fragments[token]:
            out[token]=ShapeAttribution(None,False,True,count,None); continue
        points=np.concatenate(fragments[token],axis=0); cells,local=_canon(
            points,ann["center_world"],ann["yaw_world"])
        vox=np.unique(np.concatenate(indices[token],axis=0),axis=0)
        out[token]=ShapeAttribution(
            CanonicalShape(int(ann["class_id"]),cells,observation_keys.get(token),local),
            False,False,len(fragments[token]),vox)
    return out

def attribute_instance_shape(semantic,pose,anns,token,*,grid,free_label=17,match_max_distance_m=4.0,observation_key=None):
    token=str(token)
    return attribute_instance_shapes(
        semantic,pose,anns,grid=grid,free_label=free_label,
        match_max_distance_m=match_max_distance_m,tokens=(token,),
        observation_keys={token:observation_key})[token]

def shape_iou(a,b):
    if a.class_id!=b.class_id:return 0.0
    A={tuple(x) for x in a.cells_ijk.tolist()}; B={tuple(x) for x in b.cells_ijk.tolist()}
    return len(A&B)/max(len(A|B),1)

def canonical_shape_arrays_fingerprint(class_id,cells_ijk,observation_key,local_xyz_m):
    """Fingerprint serialized shape arrays without reconstructing the shape.

    This is intentionally separate from ``CanonicalShape`` construction:
    legacy v1 pools stored local coordinates as float32, and running those
    arrays through float64 ``np.unique`` can collapse/reorder equal float32
    rows before integrity verification.
    """
    h=hashlib.sha256(); h.update(str(int(class_id)).encode()); h.update(b"\0")
    h.update(json.dumps(list(observation_key or ("","")),separators=(",",":")).encode())
    h.update(np.asarray(cells_ijk,dtype="<i4").tobytes(order="C"))
    h.update(np.asarray(local_xyz_m,dtype="<f4").tobytes(order="C"))
    return h.hexdigest()

def canonical_shape_fingerprint(shape):
    return canonical_shape_arrays_fingerprint(
        shape.class_id,shape.cells_ijk,shape.observation_key,shape.local_xyz_m)

def canonical_shape_population_fingerprint(shapes_by_class):
    rows=[]
    for cid in sorted(shapes_by_class):
        for shape in sorted(shapes_by_class[cid],key=lambda x:tuple(x.observation_key or ("",""))):
            if int(shape.class_id)!=int(cid):
                raise RuntimeError("shape-pool class key/shape mismatch")
            rows.append([list(shape.observation_key or ("","")),canonical_shape_fingerprint(shape)])
    return stable_json_fingerprint(rows)

def shape_pool_metadata_fingerprint(population_manifest,population_shape_fingerprint,source_provenance):
    population=tuple(sorted((str(a),str(b)) for a,b in population_manifest))
    payload={
        "protocol":SHAPE_POOL_PROTOCOL,
        "resolution_m":SHAPE_RES,
        "population_manifest":[list(x) for x in population],
        "population_shape_fingerprint":str(population_shape_fingerprint),
        "source_provenance":dict(source_provenance),
    }
    return stable_json_fingerprint(payload)

def shape_pool_fingerprint(shapes_by_class,population_manifest,source_provenance):
    population=tuple(sorted((str(a),str(b)) for a,b in population_manifest))
    keys=[tuple(x.observation_key or ("",""))
          for cid in sorted(shapes_by_class) for x in shapes_by_class[cid]]
    if len(keys)!=len(set(keys)) or set(keys)!=set(population):
        raise RuntimeError("shape pool and population manifest differ")
    return shape_pool_metadata_fingerprint(
        population,canonical_shape_population_fingerprint(shapes_by_class),source_provenance)

def index_shape_pool(shapes_by_class):
    """Build a sparse inverted index for exact repeated binary-IoU lookup.

    Only candidates sharing at least one occupied canonical cell can beat IoU
    zero, so the postings avoid a full target x train-pool scan.
    """
    out={}
    for cid,rows in shapes_by_class.items():
        parsed=tuple(sorted(rows,key=lambda x:tuple(x.observation_key or ("",""))))
        if any(int(x.class_id)!=int(cid) for x in parsed):
            raise RuntimeError("shape-pool class key/shape mismatch")
        postings={}; sizes=[]
        for i,shape in enumerate(parsed):
            cells=frozenset(tuple(v) for v in shape.cells_ijk.tolist()); sizes.append(len(cells))
            for cell in cells:postings.setdefault(cell,[]).append(i)
        out[int(cid)]={"shapes":parsed,"sizes":tuple(sizes),
                       "postings":{k:tuple(v) for k,v in postings.items()}}
    return out

def oracle_best_indexed_shape(shape,index):
    row=index.get(int(shape.class_id))
    if not row or not row["shapes"]:return None
    target=frozenset(tuple(v) for v in shape.cells_ijk.tolist())
    overlaps={}
    for cell in target:
        for i in row["postings"].get(cell,()):overlaps[i]=overlaps.get(i,0)+1
    if not overlaps:return row["shapes"][0]
    def _score(item):
        i,inter=item; union=len(target)+row["sizes"][i]-inter
        return (-(inter/max(union,1)),tuple(row["shapes"][i].observation_key or ("","")))
    return row["shapes"][min(overlaps.items(),key=_score)[0]]

def _set_iou(A,B):return len(A&B)/max(len(A|B),1)

def _exact_kmedoids(cell_sets,keys,k):
    n=len(cell_sets)
    if not n:return ()
    k=min(int(k),n); D=np.zeros((n,n),dtype=np.float64)
    for i in range(n):
        for j in range(i+1,n):D[i,j]=D[j,i]=1-_set_iou(cell_sets[i],cell_sets[j])
    med=[min(range(n),key=lambda i:(float(D[i].sum()),keys[i],i))]
    while len(med)<k:
        cand=[(float(np.min(D[:,med+[j]],axis=1).sum()),keys[j],j) for j in range(n) if j not in med]
        med.append(int(min(cand)[2]))
    improved=True
    while improved:
        improved=False; assigned=D[:,med]
        nearest=assigned.min(axis=1); owner=assigned.argmin(axis=1)
        second=(np.partition(assigned,1,axis=1)[:,1]
                if len(med)>1 else np.full(n,np.inf,dtype=np.float64))
        cur=float(nearest.sum()); best=None
        for pos in range(len(med)):
            without=np.where(owner==pos,second,nearest)
            for j in range(n):
                if j in med:continue
                cost=float(np.minimum(without,D[:,j]).sum())
                if cost+1e-12<cur:
                    item=(cost,keys[j],pos,j)
                    if best is None or item[:4]<best[:4]:best=item
        if best is not None:med[best[2]]=best[3]; improved=True
    return tuple(sorted(med,key=lambda i:(keys[i],i)))

def deterministic_kmedoids(shapes,k):
    """Exact PAM for small classes; deterministic bounded-memory CLARA otherwise."""
    if int(k)<=0:raise ValueError("k must be positive")
    n=len(shapes)
    if not n:return ()
    k=min(int(k),n); keys=[tuple(x.observation_key or ("",str(i))) for i,x in enumerate(shapes)]
    sets=[frozenset(tuple(v) for v in x.cells_ijk.tolist()) for x in shapes]
    if n<=KMEDOIDS_EXACT_MAX_N:return _exact_kmedoids(sets,keys,k)
    sample_n=min(n,max(KMEDOIDS_CLARA_SAMPLE_SIZE,40+2*k)); fps=[canonical_shape_fingerprint(x) for x in shapes]
    best=None
    for trial in range(KMEDOIDS_CLARA_TRIALS):
        sample=sorted(range(n),key=lambda i:(hashlib.sha256(f"{trial}:{keys[i]}:{fps[i]}".encode()).digest(),keys[i],i))[:sample_n]
        local=_exact_kmedoids([sets[i] for i in sample],[keys[i] for i in sample],k)
        med=tuple(sample[i] for i in local); medsets=[sets[i] for i in med]
        cost=sum(1-max(_set_iou(row,m) for m in medsets) for row in sets)
        item=(float(cost),tuple(keys[i] for i in med),med)
        if best is None or item[:2]<best[:2]:best=item
    return tuple(sorted(best[2],key=lambda i:(keys[i],i)))

def prototype_bank_fingerprint(bank):
    serial={"protocol":bank.protocol,"resolution_m":bank.resolution_m,"requested_k":bank.requested_k,
            "population":bank.population_manifest,"population_shape_fingerprint":bank.population_shape_fingerprint,
            "algorithm":bank.algorithm,"algorithm_config":dict(bank.algorithm_config),
            "source_provenance":dict(bank.source_provenance),
            "medoids":{str(k):[{"key":x.observation_key,"shape":canonical_shape_fingerprint(x)} for x in v]
                       for k,v in sorted(bank.medoids_by_class.items())}}
    return stable_json_fingerprint(serial)

def build_prototype_bank(shapes_by_class,*,requested_k,population_manifest,source_provenance=None):
    med={}
    pop=tuple(sorted({(str(a),str(b)) for a,b in population_manifest}))
    all_shapes=[]
    for cid in sorted(shapes_by_class):
        rows=sorted(shapes_by_class[cid],key=lambda x:tuple(x.observation_key or ("","")))
        if any(int(x.class_id)!=int(cid) for x in rows):
            raise RuntimeError("prototype class key/shape mismatch")
        all_shapes.extend(rows)
        med[int(cid)]=tuple(rows[i] for i in deterministic_kmedoids(rows,requested_k))
    shape_keys=[tuple(x.observation_key or ("","")) for x in all_shapes]
    if len(shape_keys)!=len(set(shape_keys)) or set(shape_keys)!=set(pop):
        raise RuntimeError("prototype population and attributed shapes differ")
    pop_shape_fp=stable_json_fingerprint([
        [list(x.observation_key or ("","")),canonical_shape_fingerprint(x)]
        for x in sorted(all_shapes,key=lambda z:tuple(z.observation_key or ("","")))])
    algorithm="exact_pam_le_512_else_deterministic_clara_v1"
    algorithm_config={"exact_max_n":KMEDOIDS_EXACT_MAX_N,"clara_sample_size":KMEDOIDS_CLARA_SAMPLE_SIZE,
                      "clara_trials":KMEDOIDS_CLARA_TRIALS}
    bank=PrototypeBank(PROTOTYPE_PROTOCOL,SHAPE_RES,int(requested_k),med,pop,"",pop_shape_fp,
                       algorithm,algorithm_config,dict(source_provenance or {}))
    return PrototypeBank(**{**bank.__dict__,"fingerprint":prototype_bank_fingerprint(bank)})

def oracle_best_prototype(shape,bank):
    rows=bank.medoids_by_class.get(int(shape.class_id),())
    return min(rows,key=lambda x:(-shape_iou(shape,x),tuple(x.observation_key or ("","")))) if rows else None

def scale_prototype_to_target_extent(prototype,target):
    """Oracle diagnostic for factorized shape-code x continuous extent.

    Both shapes are already canonicalized by GT center/yaw.  Only the three
    occupied extents are transferred; no target voxel pattern is copied.
    A future deployable model would predict these three scale factors.
    """
    if int(prototype.class_id)!=int(target.class_id):
        raise ValueError("prototype/target classes differ")
    p=np.asarray(prototype.local_xyz_m,dtype=np.float64)
    t=np.asarray(target.local_xyz_m,dtype=np.float64)
    if not len(p) or not len(t):return prototype
    pspan=np.ptp(p,axis=0); tspan=np.ptp(t,axis=0)
    scale=np.divide(tspan,pspan,out=np.ones(3,dtype=np.float64),where=pspan>1e-12)
    local=p*scale[None]
    cells=np.unique(np.rint(local/SHAPE_RES).astype(np.int32),axis=0)
    return CanonicalShape(int(prototype.class_id),cells,prototype.observation_key,local)

def oracle_best_extent_scaled_prototype(target,bank):
    rows=bank.medoids_by_class.get(int(target.class_id),())
    candidates=[scale_prototype_to_target_extent(x,target) for x in rows]
    return min(candidates,key=lambda x:(-shape_iou(target,x),tuple(x.observation_key or ("","")))) if candidates else None

def rasterize_canonical_shape(shape,center,yaw,future_pose,*,grid):
    local=np.asarray(shape.local_xyz_m,float); c,s=math.cos(yaw),math.sin(yaw)
    rel=np.stack((c*local[:,0]-s*local[:,1],s*local[:,0]+c*local[:,1],local[:,2]),-1)
    world=rel+np.asarray(center); T=np.linalg.inv(np.asarray(future_pose)); ego=world@T[:3,:3].T+T[:3,3]
    origin=np.asarray([grid.x_min,grid.y_min,grid.z_min]); step=np.asarray(grid.voxel_size)
    idx=np.floor((ego-origin)/step).astype(int); valid=((idx>=0)&(idx<np.asarray(grid.shape_hwd))).all(1)
    return (np.unique(idx[valid],axis=0) if valid.any() else np.empty((0,3),int),int((~valid).sum()))

def compose_v21_add_only(base,proposals,*,free_label=17):
    base=np.asarray(base)
    if base.ndim!=3:raise ValueError("base occupancy must be [X,Y,Z]")
    out=base.copy(); claimed=np.zeros(base.shape,bool); kindmap=np.full(base.shape,-1,np.int8)
    order={"historical":0,"frontier":1}; blocked=coll=cross=written=oob=0
    for pi,(kind,aid,cid,indices) in enumerate(sorted(proposals,key=lambda x:(order[x[0]],int(x[1])))):
        idx=np.asarray(indices,dtype=int)
        if idx.ndim!=2 or idx.shape[1]!=3:raise ValueError("proposal indices must be [N,3]")
        if not len(idx):continue
        valid=((idx>=0)&(idx<np.asarray(base.shape))).all(1); oob+=int((~valid).sum())
        idx=np.unique(idx[valid],axis=0)
        if not len(idx):continue
        occ=base[idx[:,0],idx[:,1],idx[:,2]]!=free_label; blocked+=int(occ.sum()); q=idx[~occ]
        if not len(q):continue
        used=claimed[q[:,0],q[:,1],q[:,2]]; coll+=int(used.sum())
        pk=kindmap[q[:,0],q[:,1],q[:,2]]; cross+=int((used&(pk!=order[kind])).sum()); q=q[~used]
        if len(q):
            out[q[:,0],q[:,1],q[:,2]]=int(cid); claimed[q[:,0],q[:,1],q[:,2]]=True; kindmap[q[:,0],q[:,1],q[:,2]]=order[kind]; written+=len(q)
    return out,ComposeReport(int(written),blocked,coll,cross,oob)

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
