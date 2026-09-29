#!/usr/bin/env python3
"""V21 Stage-0 upper-bound audit. No V21 network or V20 completion is used."""
from __future__ import annotations
import argparse,hashlib,json,time
from collections import defaultdict
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import numpy as np
import torch

from real_motion.metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS
from real_motion.nuscenes_adapter import gt_moving_support_sequence
from real_motion.prepared import load_nuscenes_window_raw
from real_motion.runtime_config import add_config_args,load_runtime_config,make_prepare_config
from real_motion.strong_w2det import StrongW2DetConfig
from real_motion.v21_source_induction import (
    ALL_HORIZONS_S,COMPOSITOR_PROTOCOL,POPULATION_PROTOCOL,PROTOCOL,PROTOTYPE_PROTOCOL,REPORT_INDICES,
    KMEDOIDS_CLARA_SAMPLE_SIZE,KMEDOIDS_CLARA_TRIALS,KMEDOIDS_EXACT_MAX_N,
    AnchorLattice,CanonicalShape,PrototypeBank,annotation_map,assign_causal_coverage,
    attribute_instance_shapes,build_frontier_anchors,build_historical_anchors,
    build_v21_targets,compose_v21_add_only,oracle_best_prototype,
    prototype_bank_fingerprint,rasterize_canonical_shape,
    select_scene_balanced_round_robin,stable_json_fingerprint,
)
from tools.real_motion import eval_p0_f9_v18_se2 as base
from tools.real_motion import eval_p0_f9_v18_full_validation as full
from tools.real_motion import benchmark_p0_f9_v18_runtime as runtime
from tools.real_motion.diagnose_p0_f9_v19_innovation_decomposition import CachedSource
from tools.real_motion.eval_p0_f9_v17_local_stwm import window_from_record
from tools.real_motion.train_p0_f9_v18_se2_clean import PROTOCOL as CLEAN_PROTOCOL
from tools.real_motion.v20_unified_common import STAGE1_PROTOCOL

EVAL_PROTOCOL="p0_f9_v21_stage0_upper_bound_audit_v3"
REPORT_H=(1.0,2.0,3.0)
VARIANTS=("UB0_EXACT","UB1_CAUSAL_EXACT","UB2_DORMANT_CAUSAL_SHAPE",
          "UB2_BIRTH_PROTOTYPE","UB2_DEPLOYABLE_REPRESENTATION")
SEM=tuple(range(17)); DYN=tuple(int(x) for x in DYNAMIC_CLASS_IDS)

class Metrics:
    def __init__(self):
        self.oi=np.zeros(3,np.int64); self.ou=np.zeros(3,np.int64)
        self.si=np.zeros((3,17),np.int64); self.su=np.zeros((3,17),np.int64)
        self.mi=np.zeros((3,len(DYN)),np.int64); self.mu=np.zeros((3,len(DYN)),np.int64)
    def update(self,i,p,g,m,free):
        p=np.asarray(p); g=np.asarray(g); m=np.asarray(m,bool)
        po=p!=free; go=g!=free; self.oi[i]+=int((po&go).sum()); self.ou[i]+=int((po|go).sum())
        for j,c in enumerate(SEM):
            pp=p==c; gg=g==c; self.si[i,j]+=int((pp&gg).sum()); self.su[i,j]+=int((pp|gg).sum())
        for j,c in enumerate(DYN):
            pp=(p==c)&m; gg=(g==c)&m; self.mi[i,j]+=int((pp&gg).sum()); self.mu[i,j]+=int((pp|gg).sum())
    @staticmethod
    def iou(a,b):
        a=np.asarray(a,float); b=np.asarray(b,float); o=np.full(np.broadcast_shapes(a.shape,b.shape),np.nan)
        np.divide(a,b,out=o,where=b>0); return 100*o
    def compute(self):
        occ=self.iou(self.oi,self.ou); sem=self.iou(self.si,self.su); mov=self.iou(self.mi,self.mu)
        sm=np.nanmean(sem,1); mm=np.nanmean(mov,1); micro=self.iou(self.mi.sum(1),self.mu.sum(1))
        per={}
        for i,h in enumerate(REPORT_H):
            per[str(h)]={"IoU":float(occ[i]),"mIoU":float(sm[i]),"MovingMacro":float(mm[i]),
                "MovingMicro":float(micro[i]),
                "semantic_per_class":{str(c):float(sem[i,j]) for j,c in enumerate(SEM)},
                "moving_per_class":{str(c):float(mov[i,j]) for j,c in enumerate(DYN)}}
        return {"IoU":float(np.nanmean(occ)),"mIoU":float(np.nanmean(sm)),
                "MovingMacro":float(np.nanmean(mm)),"MovingMicro":float(np.nanmean(micro)),
                "per_horizon":per}

def delta(c,b):
    out={k:float(c[k])-float(b[k]) for k in ("IoU","mIoU","MovingMacro","MovingMicro")}
    out["per_horizon"]={}
    for h in REPORT_H:
        hs=str(h); out["per_horizon"][hs]={k:float(c["per_horizon"][hs][k])-float(b["per_horizon"][hs][k])
                                           for k in ("IoU","mIoU","MovingMacro","MovingMicro")}
        for name in ("semantic_per_class","moving_per_class"):
            out["per_horizon"][hs][name]={}
            for cid,v in c["per_horizon"][hs][name].items():
                bv=b["per_horizon"][hs][name][cid]
                out["per_horizon"][hs][name][cid]=float(v)-float(bv) if np.isfinite(v) and np.isfinite(bv) else float("nan")
    return out

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for x in iter(lambda:f.read(1<<20),b""): h.update(x)
    return h.hexdigest()

def _is_sha256(value):
    value=str(value).lower()
    return len(value)==64 and all(c in "0123456789abcdef" for c in value)

def load_manifest(path):
    x=json.loads(Path(path).read_text(encoding="utf-8"))
    if x.get("protocol")!=POPULATION_PROTOCOL: raise RuntimeError("bad population manifest")
    if x.get("v21_protocol")!=PROTOCOL:raise RuntimeError("population manifest V21 protocol mismatch")
    if x.get("parent_stage1_protocol")!=STAGE1_PROTOCOL:raise RuntimeError("population manifest Stage-1 protocol mismatch")
    if not _is_sha256(x.get("parent_stage1_index_sha256","")):
        raise RuntimeError("population manifest lacks a valid Stage-1 index SHA256")
    payload=dict(x); declared=payload.pop("manifest_fingerprint",None)
    if not declared or stable_json_fingerprint(payload)!=declared:raise RuntimeError("manifest content fingerprint mismatch")
    parent=tuple((str(a),str(b)) for a,b in x["parent_keys"])
    keys=tuple((str(a),str(b)) for a,b in x["selected_keys"])
    if len(parent)!=len(set(parent)) or len(keys)!=len(set(keys)):raise RuntimeError("duplicate population manifest keys")
    if int(x.get("parent_num_windows",-1))!=len(parent) or int(x.get("selected_num_windows",-1))!=len(keys):
        raise RuntimeError("population manifest key/count mismatch")
    if stable_json_fingerprint([list(k) for k in parent])!=x["parent_key_fingerprint"]: raise RuntimeError("parent manifest fingerprint mismatch")
    if stable_json_fingerprint([list(k) for k in keys])!=x["selected_key_fingerprint"]: raise RuntimeError("manifest fingerprint mismatch")
    expected=parent if len(keys)==len(parent) else select_scene_balanced_round_robin(parent,len(keys))
    if keys!=expected:raise RuntimeError("selected population is not the frozen scene-balanced selection")
    lattice=AnchorLattice.from_dict(x["frozen_anchor_lattice"])
    if stable_json_fingerprint(lattice.to_dict())!=x["frozen_anchor_lattice_fingerprint"]:
        raise RuntimeError("manifest lattice fingerprint mismatch")
    return x,keys,lattice

def align_records(records,keys):
    m={}; duplicate=[]
    for r in records:
        key=(str(r["scene_name"]),str(r["t0_token"]))
        if key in m:duplicate.append(key)
        else:m[key]=r
    if duplicate:raise RuntimeError(f"V18 cache has duplicate identities: {duplicate[:5]}")
    missing=[k for k in keys if k not in m]
    if missing: raise RuntimeError(f"V18 cache missing keys: {missing[:5]}")
    return [m[k] for k in keys]

def load_bank(path,*,allow_incomplete=False):
    x=torch.load(path,map_location="cpu",weights_only=False)
    if x.get("protocol")!=PROTOTYPE_PROTOCOL: raise RuntimeError("bad prototype bank")
    if not np.isclose(float(x.get("resolution_m",-1)),0.4):raise RuntimeError("prototype resolution mismatch")
    if int(x.get("requested_k",0)) not in (1,4,8,16):raise RuntimeError("prototype K is outside the frozen sweep")
    expected_algorithm="exact_pam_le_512_else_deterministic_clara_v1"
    expected_algorithm_config={"exact_max_n":KMEDOIDS_EXACT_MAX_N,
        "clara_sample_size":KMEDOIDS_CLARA_SAMPLE_SIZE,"clara_trials":KMEDOIDS_CLARA_TRIALS}
    if x.get("algorithm")!=expected_algorithm or x.get("algorithm_config")!=expected_algorithm_config:
        raise RuntimeError("prototype-bank clustering contract mismatch")
    provenance=dict(x.get("source_provenance") or {})
    if not bool(provenance.get("complete_train_population",False)) and not allow_incomplete:
        raise RuntimeError("formal evaluator refuses an incomplete/diagnostic prototype bank")
    if not _is_sha256(provenance.get("train_cache_sha256","")) or not _is_sha256(provenance.get("info_pkl_sha256","")):
        raise RuntimeError("prototype bank lacks valid source SHA256 provenance")
    if provenance.get("cache_version")!=base.SE2_CACHE_VERSION:
        raise RuntimeError("prototype-bank V18 cache version mismatch")
    if provenance.get("se2_target_contract")!=base.SE2_TARGET_CONTRACT:
        raise RuntimeError("prototype-bank SE2 target contract mismatch")
    total=int(provenance.get("cache_records_total",-1)); used=int(provenance.get("cache_records_used",-1))
    if total<=0 or used<=0 or used>total:
        raise RuntimeError("prototype-bank source record counts are invalid")
    if bool(provenance.get("complete_train_population",False))!=(used==total):
        raise RuntimeError("prototype-bank completeness/count contract mismatch")
    med={}
    for cid,rows in x["medoids_by_class"].items():
        class_id=int(cid)
        if len(rows)>int(x["requested_k"]):raise RuntimeError("prototype class exceeds requested K")
        parsed=tuple(CanonicalShape(int(r["class_id"]),r["cells_ijk"].numpy().astype(np.int32),
                                    (str(r["observation_key"][0]),str(r["observation_key"][1])),
                                    r["local_xyz_m"].numpy().astype(np.float64)) for r in rows)
        if any(s.class_id!=class_id for s in parsed):raise RuntimeError("prototype class key/payload mismatch")
        med[class_id]=parsed
    pop=tuple((str(a),str(b)) for a,b in x["population_manifest"])
    if len(pop)!=len(set(pop)):raise RuntimeError("prototype population contains duplicate observations")
    if stable_json_fingerprint([list(z) for z in pop])!=x.get("population_fingerprint"):
        raise RuntimeError("prototype population fingerprint mismatch")
    bank=PrototypeBank(x["protocol"],float(x["resolution_m"]),int(x["requested_k"]),med,pop,
                       str(x["fingerprint"]),str(x.get("population_shape_fingerprint") or ""),
                       str(x.get("algorithm") or ""),dict(x.get("algorithm_config") or {}),provenance)
    if prototype_bank_fingerprint(bank)!=bank.fingerprint:raise RuntimeError("prototype bank content fingerprint mismatch")
    medoid_keys=[s.observation_key for rows in med.values() for s in rows]
    if len(medoid_keys)!=len(set(medoid_keys)) or not set(medoid_keys).issubset(set(pop)):
        raise RuntimeError("prototype medoid keys are invalid")
    return bank

def frozen_reference_forward(model,gi,device):
    f,t,k,fm,sm=gi["features"],gi["tube"],gi["kta"],gi["frame_motion"],gi["source_mask"]; cfg=model.config; B=f.shape[0]
    with torch.inference_mode(),torch.autocast(device_type="cuda",dtype=torch.bfloat16,enabled=device.type=="cuda"):
        if B==0:return {"residual_xy_m":f.new_empty((0,6,2)),"existence_logits":f.new_empty((0,6)),"yaw_delta_rad":f.new_empty((0,6))}
        e=model.semantic_embedding(t.long())+model.source_mask_embedding(sm.long())
        x=e.permute(0,1,4,2,3).reshape(B*6,cfg.semantic_dim,cfg.tube_hw,cfg.tube_hw); x=model.spatial_stem(x)
        hs,ws=x.shape[-2:]; x=x.reshape(B,6,cfg.d_model,hs,ws)
        x=x+model.kinematic_proj(f).view(B,1,cfg.d_model,1,1)+model.frame_motion_proj(fm.to(x.dtype)).view(B,6,cfg.d_model,1,1)+model.time_embedding+model.spatial_embedding
        for block in model.blocks:x=block(x)
        ctx=x.permute(0,1,3,4,2).reshape(B,6*hs*ws,cfg.d_model)
        q=model.future_query.expand(B,-1,-1)+model.future_time_embedding+model.kinematic_proj(f).unsqueeze(1)+model.kta_future_proj(k.to(x.dtype)/20.0)
        for block in model.decoder:q=block(q,ctx)
        return {"residual_xy_m":model.residual_head(q),"existence_logits":model.existence_head(q)[...,0],"yaw_delta_rad":model.yaw_head(q)[...,0]}

def assert_forward_exact(model,state,device):
    a=runtime._model_forward(model,state["gpu"],device); b=frozen_reference_forward(model,state["gpu"],device)
    bad={k:float((a[k].float()-b[k].float()).abs().max().cpu()) for k in a if not torch.equal(a[k],b[k])}
    if bad: raise RuntimeError(f"V18 current forward != frozen ccf7d77 contract: {bad}")

def validate_clean_e14_checkpoint(ck,path,expected_sha256):
    expected=str(expected_sha256).lower(); actual=sha256(path).lower()
    if not _is_sha256(expected):
        raise ValueError("--expected-checkpoint-sha256 must be one lowercase/uppercase SHA256")
    if actual!=expected:raise RuntimeError(f"Clean-E14 checkpoint SHA256 mismatch: {actual} != {expected}")
    if str(ck.get("training_mode"))!="clean_one_stage_from_scratch_v1_tail_continuation":
        raise RuntimeError(f"expected frozen Clean-E14 tail training_mode, got {ck.get('training_mode')!r}")
    if int(ck.get("epoch",-1))!=14:raise RuntimeError(f"expected frozen Clean-E14 epoch=14, got {ck.get('epoch')!r}")
    return actual

def history_shape_cache(source,w,raw,pcfg,gate,matched_sets,ambiguous_sets,
                        components_by_frame,component_matches_by_frame):
    """Attribute historical shapes directly from the already extracted Strong evidence."""
    if len(matched_sets)!=6 or len(ambiguous_sets)!=6:
        raise ValueError("expected six precomputed history-evidence frames")
    if len(components_by_frame)!=6 or len(component_matches_by_frame)!=6:
        raise ValueError("expected six precomputed component frames")
    attrs=[]
    for i,tok in enumerate(w.history_tokens):
        matched_set={str(x) for x in matched_sets[i] if x is not None}
        amb_set={str(x) for x in ambiguous_sets[i]}
        tokens=sorted(matched_set-amb_set); anns=annotation_map(source.nusc,tok)
        masked=np.where(raw["history_observed"][i],raw["history_occ"][i],pcfg.free_label).astype(np.uint8)
        attrs.append(attribute_instance_shapes(
            masked,raw["history_poses"][i],anns,grid=pcfg.grid,free_label=pcfg.free_label,
            match_max_distance_m=gate,tokens=tokens,
            observation_keys={x:(str(tok),x) for x in tokens},
            components=components_by_frame[i],component_matches=component_matches_by_frame[i]))
    return attrs

def future_shape_cache(source,w,raw,targets,pcfg,gate):
    wanted={t.instance_token for t in targets}; out=[]; maps=[]
    for hi,tok in enumerate(w.future_tokens):
        anns=annotation_map(source.nusc,tok); tokens=sorted(wanted&set(anns))
        maps.append(anns)
        out.append(attribute_instance_shapes(
            raw["future_gt_occ"][hi],raw["future_poses"][hi],anns,grid=pcfg.grid,
            free_label=pcfg.free_label,match_max_distance_m=gate,tokens=tokens,
            observation_keys={x:(str(tok),x) for x in tokens}))
    return out,maps

def target_masks_from_attributions(attrs,shape):
    out=[]
    for row in attrs:
        masks={}
        for tok,at in row.items():
            if at.ambiguous or at.voxel_indices is None:continue
            m=np.zeros(shape,bool); idx=np.asarray(at.voxel_indices,dtype=np.int64)
            if len(idx):m[idx[:,0],idx[:,1],idx[:,2]]=True
            masks[tok]=m
        out.append(masks)
    return out

def proposal_key(t,m,rank):
    if m is not None:return m.anchor_kind,m.anchor_id
    return ("historical",-2000000+rank) if t.responsibility=="DORMANT_ANCESTRAL" else ("frontier",2000000+rank)

def add_counts(base,pred,gt,tmask,free):
    added=(base==free)&(pred!=free); ta=tmask&(base==free)
    return {"added":int(added.sum()),"occ_tp":int((added&(gt!=free)).sum()),
            "semantic_tp":int((added&(pred==gt)&(gt!=free)).sum()),"target_addable":int(ta.sum()),
            "target_recovered":int((ta&(pred==gt)&(gt!=free)).sum())}
def merge_counts(a,b):
    for k,v in b.items():a[k]+=int(v)
def quality(x):
    return {"addition_occupancy_precision":x["occ_tp"]/x["added"] if x["added"] else float("nan"),
            "addition_semantic_precision":x["semantic_tp"]/x["added"] if x["added"] else float("nan"),
            "v21_target_semantic_recall":x["target_recovered"]/x["target_addable"] if x["target_addable"] else float("nan"),**x}

def annotation_center_inside_grid(center_world,ego_to_world,grid):
    center=np.asarray(center_world,dtype=np.float64)
    ego=(np.linalg.inv(np.asarray(ego_to_world,dtype=np.float64))@np.r_[center,1.0])[:3]
    lower=np.asarray([grid.x_min,grid.y_min,grid.z_min],dtype=np.float64)
    upper=lower+np.asarray(grid.voxel_size,dtype=np.float64)*np.asarray(grid.shape_hwd,dtype=np.float64)
    return bool(((ego>=lower)&(ego<upper)).all())

def accumulate_coverage_stratum(store,distances,name,subset,all_match_tokens,
                                historical,frontier,t0_pose,radius):
    subset=list(subset); tokens={t.instance_token for t in subset}; row=store[name]
    matches,report=assign_causal_coverage(
        subset,historical,frontier,t0_pose=t0_pose,coverage_radius_m=radius)
    if report["duplicate_target_assignment"] or report["duplicate_anchor_assignment"]:
        raise RuntimeError(f"V21 stratum assignment violated one-to-one contract: {name}")
    eligible=int(report["eligible_targets"]); row["eligible"]+=eligible
    row["covered_under_all_target_assignment"]+=len(tokens&all_match_tokens)
    row["reassigned_covered"]+=int(report["covered_targets"])
    row["candidate_sum"]+=float(report["mean_legal_candidates_per_positive"])*eligible
    row["historical_matches"]+=int(report["historical_matches"])
    row["frontier_matches"]+=int(report["frontier_matches"])
    row["max_candidates"]=max(row["max_candidates"],int(report["max_legal_candidates_per_positive"]))
    distances[name].extend(float(x.distance_m) for x in matches)

def coverage_strata_report(store,distances):
    out={}
    for name,row in store.items():
        eligible=int(row["eligible"]); d=np.asarray(distances[name],dtype=np.float64)
        out[name]={
            "eligible_targets":eligible,
            "covered_under_all_target_assignment":int(row["covered_under_all_target_assignment"]),
            "coverage_under_all_target_assignment":row["covered_under_all_target_assignment"]/max(eligible,1),
            "reassigned_covered_targets":int(row["reassigned_covered"]),
            "reassigned_geometric_coverage":row["reassigned_covered"]/max(eligible,1),
            "historical_matches":int(row["historical_matches"]),
            "frontier_matches":int(row["frontier_matches"]),
            "mean_legal_candidates_per_positive":row["candidate_sum"]/max(eligible,1),
            "max_legal_candidates_per_positive":int(row["max_candidates"]),
            "matched_distance_m":({"p50":float(np.quantile(d,0.50)),"p90":float(np.quantile(d,0.90)),
                                    "p99":float(np.quantile(d,0.99)),"max":float(d.max())}
                                   if len(d) else None),
        }
    return out

def main():
    p=argparse.ArgumentParser(); add_config_args(p)
    for name in ("val-cache","population-manifest","checkpoint","prototype-bank","dataroot","info-pkl","output"):
        p.add_argument("--"+name,required=True)
    p.add_argument("--expected-checkpoint-sha256",required=True)
    p.add_argument("--coverage-radius-m",type=float,choices=(0.8,1.6,3.2),required=True)
    p.add_argument("--match-max-distance-m",type=float,default=4.0); p.add_argument("--device",default="cuda")
    p.add_argument("--exactness-windows",type=int,default=1); p.add_argument("--moving-workers",type=int,default=1)
    p.add_argument("--allow-incomplete-prototype-bank",action="store_true")
    p.add_argument("--enforce-stage0b-gate",action="store_true"); a=p.parse_args()
    pcfg=make_prepare_config(load_runtime_config(a.config,a.override)); manifest,keys,lattice=load_manifest(a.population_manifest)
    _,records0=base.load_cache(a.val_cache); records=align_records(records0,keys)
    bank=load_bank(a.prototype_bank,allow_incomplete=a.allow_incomplete_prototype_bank)
    source=CachedSource(a.dataroot,info_pkl=a.info_pkl,verbose=False); strong=StrongW2DetConfig(free_label=int(pcfg.free_label))
    device=torch.device(a.device if a.device!="cuda" or torch.cuda.is_available() else "cpu")
    ck,model,_=full._load_model(a.checkpoint,CLEAN_PROTOCOL,device)
    checkpoint_sha=validate_clean_e14_checkpoint(ck,a.checkpoint,a.expected_checkpoint_sha256)

    states={"V18_BASE":Metrics(),**{v:Metrics() for v in VARIANTS}}
    ages={str(x):Metrics() for x in (0.5,1.0,1.5,2.0,2.5)}
    q={v:defaultdict(int,{"added":0,"occ_tp":0,"semantic_tp":0,"target_addable":0,"target_recovered":0}) for v in VARIANTS}
    audit=defaultdict(int); aa=defaultdict(int); sa=defaultdict(int); ca={v:defaultdict(int) for v in VARIANTS}; cov=defaultdict(float)
    cov_strata=defaultdict(lambda:defaultdict(float)); cov_distances=defaultdict(list)
    perclass=defaultdict(int); perh=defaultdict(int); totalvox=coveredvox=alltargets=movingtargets=allvox=movingvox=0
    scene=defaultdict(lambda:{"V18_BASE":Metrics(),**{v:Metrics() for v in VARIANTS}})
    checked=0; started=time.perf_counter()

    for wi,rec in enumerate(records,1):
        w=window_from_record(rec); raw=load_nuscenes_window_raw(source,w,pcfg,include_gt=True)
        st=runtime._prepare_record(rec,source,pcfg,strong,device,raw_window=raw); runtime._stage_gpu_inputs(st,device)
        try:
            if checked<a.exactness_windows: assert_forward_exact(model,st,device); runtime._exactness_check(model,st,pcfg,strong,device); checked+=1
            basepred=runtime._forecast_once(model,st,pcfg,strong,device)
        finally: runtime._release_gpu_inputs(st)
        for h in range(6):
            noop,_=compose_v21_add_only(basepred[h],[],free_label=pcfg.free_label)
            if not np.array_equal(noop,basepred[h]):raise RuntimeError("zero-contribution identity failed")

        hist,_,ha,history_evidence=build_historical_anchors(
            source,w,raw["history_occ"],raw["history_observed"],raw["history_poses"],
            grid=pcfg.grid,strong_cfg=strong,frame_dt_s=pcfg.frame_dt_s,
            match_max_distance_m=a.match_max_distance_m)
        hmatches=history_evidence["matched_sets"]
        hambiguous=history_evidence["ambiguous_sets"]
        hattrs=history_shape_cache(
            source,w,raw,pcfg,a.match_max_distance_m,hmatches,hambiguous,
            history_evidence["components_by_frame"],history_evidence["component_matches_by_frame"])
        targets,ta=build_v21_targets(source,w,np.asarray(raw["history_occ"]),np.asarray(raw["history_observed"]),np.asarray(raw["history_poses"]),
                                     grid=pcfg.grid,strong_cfg=strong,match_max_distance_m=a.match_max_distance_m,
                                     history_matches=hmatches,history_ambiguous=hambiguous)
        for k,v in ta.items():audit[k]+=int(v)
        alltargets+=len(targets)
        for t in targets:
            perclass[str(t.class_id)]+=1
            for hi in REPORT_INDICES:
                if t.existence[hi]:perh[str(ALL_HORIZONS_S[hi])]+=1
        front,fa=build_frontier_anchors(raw["history_observed"],raw["history_poses"],raw["future_poses"],grid=pcfg.grid,lattice=lattice)
        for k,v in ha.items():
            if isinstance(v,dict):
                for kk,vv in v.items():aa[f"{k}/{kk}"]+=int(vv)
            else:aa[k]+=int(v)
        aa["frontier_anchor_count"]+=fa["frontier_anchor_count"]
        matches,cr=assign_causal_coverage(targets,hist,front,t0_pose=raw["history_poses"][-1],coverage_radius_m=a.coverage_radius_m)
        if cr["duplicate_target_assignment"] or cr["duplicate_anchor_assignment"]:
            raise RuntimeError("V21 causal assignment violated one-to-one contract")
        mb={m.target_token:m for m in matches}; cov["eligible"]+=cr["eligible_targets"]; cov["covered"]+=cr["covered_targets"]
        cov["candidate_sum"]+=cr["mean_legal_candidates_per_positive"]*cr["eligible_targets"]
        cov["hist"]+=cr["historical_matches"]; cov["front"]+=cr["frontier_matches"]
        cov["dup_t"]+=cr["duplicate_target_assignment"]; cov["dup_a"]+=cr["duplicate_anchor_assignment"]
        cov["maxcand"]=max(cov["maxcand"],cr["max_legal_candidates_per_positive"])

        future_attrs,future_ann_maps=future_shape_cache(source,w,raw,targets,pcfg,a.match_max_distance_m)
        masks=target_masks_from_attributions(future_attrs,pcfg.grid.shape_hwd)
        onset_resolved=set(); any_resolved=set(); report_component=set()
        for t in targets:
            tok=t.instance_token
            resolved=[hi for hi,row in enumerate(future_attrs)
                      if (row.get(tok) is not None and row[tok].shape is not None)]
            if resolved:
                any_resolved.add(tok); first=resolved[0]
                sa[f"first_resolved_horizon/{ALL_HORIZONS_S[first]:.1f}s"]+=1
                if first>t.onset_index:sa["resolved_only_after_annotation_onset"]+=1
            else:sa["no_future_shape_resolved"]+=1
            if t.onset_index in resolved:onset_resolved.add(tok)
            if any(tok in masks[hi] for hi in REPORT_INDICES):report_component.add(tok)
            ann=future_ann_maps[t.onset_index].get(tok)
            if ann is not None:
                inside=annotation_center_inside_grid(
                    ann["center_world"],raw["future_poses"][t.onset_index],pcfg.grid)
                sa["annotation_onset_center_inside_grid" if inside else "annotation_onset_center_outside_grid"]+=1
                if inside and tok not in onset_resolved:sa["onset_inside_grid_but_shape_unresolved"]+=1
        sa["onset_shape_resolved"]+=len(onset_resolved)
        sa["any_future_shape_resolved"]+=len(any_resolved)
        sa["report_horizon_component_resolved"]+=len(report_component)
        all_match_tokens=set(mb)
        strata={"onset_shape_resolved":onset_resolved,
                "any_future_shape_resolved":any_resolved,
                "report_horizon_component_resolved":report_component}
        for name,tokens in strata.items():
            accumulate_coverage_stratum(
                cov_strata,cov_distances,name,
                (t for t in targets if t.instance_token in tokens),all_match_tokens,
                hist,front,raw["history_poses"][-1],a.coverage_radius_m)
        onset={}; hshape={}; proto={}; hage={}; rank={t.instance_token:i for i,t in enumerate(targets)}
        for t in targets:
            tok=t.instance_token; oi=t.onset_index; at=future_attrs[oi].get(tok)
            if at is None or at.shape is None:
                if at is not None and at.ambiguous:sa["shape_ambiguous"]+=1
                sa["shape_unresolved"]+=1
            else:
                ann=future_ann_maps[oi][tok]
                roundtrip,_=rasterize_canonical_shape(
                    at.shape,ann["center_world"],ann["yaw_world"],raw["future_poses"][oi],grid=pcfg.grid)
                expected=np.unique(np.asarray(at.voxel_indices,dtype=np.int64),axis=0)
                if not np.array_equal(roundtrip,expected):
                    raise RuntimeError(f"future-onset exact-shape round-trip failed: {w.scene_name}/{w.t0_token}/{tok}")
                sa["onset_roundtrip_exact"]+=1
                onset[tok]=at.shape; pp=oracle_best_prototype(at.shape,bank)
                if pp is not None:proto[tok]=pp
                elif t.responsibility=="BIRTH":sa["prototype_missing_class"]+=1
            if t.responsibility=="DORMANT_ANCESTRAL":
                found=None
                for idx in range(5,-1,-1):
                    if tok in hmatches[idx] and tok not in hambiguous[idx]:
                        hh=hattrs[idx].get(tok)
                        if hh is not None and not hh.ambiguous and hh.shape is not None:
                            found=(hh.shape,idx); break
                if found is None:sa["dormant_history_shape_unresolved"]+=1
                else:hshape[tok]=found[0]; hage[tok]=(5-found[1])*pcfg.frame_dt_s

        moving_rows=gt_moving_support_sequence(source.nusc,str(w.t0_token),w.future_tokens,ALL_HORIZONS_S,grid=pcfg.grid,workers=a.moving_workers)
        moving=np.stack([x[0] for x in moving_rows]); mt=set()
        for hi in REPORT_INDICES:mt.update(str(r["instance_token"]) for r in moving_rows[hi][1])
        movingtargets+=sum(t.instance_token in mt for t in targets)
        covered=set(mb)
        for hi in REPORT_INDICES:
            for t in targets:
                m=masks[hi].get(t.instance_token)
                if m is None:continue
                n=int(m.sum()); allvox+=n; totalvox+=n
                if t.instance_token in covered:coveredvox+=n
                if t.instance_token in mt:movingvox+=int((m&moving[hi]).sum())

        for pos,hi in enumerate(REPORT_INDICES):
            gt=np.asarray(raw["future_gt_occ"][hi],np.uint8); bh=np.asarray(basepred[hi],np.uint8)
            states["V18_BASE"].update(pos,bh,gt,moving[hi],pcfg.free_label); scene[w.scene_name]["V18_BASE"].update(pos,bh,gt,moving[hi],pcfg.free_label)
            tmask=np.zeros(pcfg.grid.shape_hwd,bool)
            for t in targets:
                if t.instance_token in masks[hi]:tmask|=masks[hi][t.instance_token]
            props={v:[] for v in VARIANTS}; ageprops={k:[] for k in ages}; oob=defaultdict(int)
            for t in targets:
                tok=t.instance_token
                if not t.existence[hi]:continue
                ann=future_ann_maps[hi].get(tok)
                if ann is None:continue
                cm=mb.get(tok); kind,aid=proposal_key(t,cm,rank[tok])
                reps={"UB0_EXACT":onset.get(tok),
                      "UB1_CAUSAL_EXACT":onset.get(tok) if cm else None,
                      "UB2_DORMANT_CAUSAL_SHAPE":(hshape.get(tok) if t.responsibility=="DORMANT_ANCESTRAL" else onset.get(tok)) if cm else None,
                      "UB2_BIRTH_PROTOTYPE":(proto.get(tok) if t.responsibility=="BIRTH" else onset.get(tok)) if cm else None,
                      "UB2_DEPLOYABLE_REPRESENTATION":(hshape.get(tok) if t.responsibility=="DORMANT_ANCESTRAL" else proto.get(tok)) if cm else None}
                for v,s in reps.items():
                    if s is None:continue
                    idx,oo=rasterize_canonical_shape(s,ann["center_world"],ann["yaw_world"],raw["future_poses"][hi],grid=pcfg.grid)
                    props[v].append((kind,aid,t.class_id,idx)); oob[v]+=oo
                if cm and t.responsibility=="DORMANT_ANCESTRAL" and tok in onset:
                    age=f"{hage.get(tok,-1):.1f}"
                    if age in ageprops:
                        idx,_=rasterize_canonical_shape(onset[tok],ann["center_world"],ann["yaw_world"],raw["future_poses"][hi],grid=pcfg.grid)
                        ageprops[age].append((kind,aid,t.class_id,idx))
            for v in VARIANTS:
                pred,r=compose_v21_add_only(bh,props[v],free_label=pcfg.free_label); states[v].update(pos,pred,gt,moving[hi],pcfg.free_label)
                scene[w.scene_name][v].update(pos,pred,gt,moving[hi],pcfg.free_label); merge_counts(q[v],add_counts(bh,pred,gt,tmask,pcfg.free_label))
                ca[v]["v21_collision_voxels"]+=r.v21_collision_voxels; ca[v]["historical_frontier_collision_voxels"]+=r.historical_frontier_collision_voxels
                ca[v]["blocked_by_v18_voxels"]+=r.blocked_by_v18_voxels
                ca[v]["out_of_bounds_voxels"]+=oob[v]+r.out_of_bounds_voxels
            for age,pp in ageprops.items():
                pred,_=compose_v21_add_only(bh,pp,free_label=pcfg.free_label); ages[age].update(pos,pred,gt,moving[hi],pcfg.free_label)
        if wi==1 or wi%16==0 or wi==len(records):print(f"v21_stage0 {wi}/{len(records)}",flush=True)

    br=states["V18_BASE"].compute(); reports={"V18_BASE":br}
    for v in VARIANTS:
        rr=states[v].compute(); reports[v]={"metrics":rr,"delta_vs_v18_pp":delta(rr,br),"addition_quality":quality(q[v])}
    ub1=reports["UB1_CAUSAL_EXACT"]["delta_vs_v18_pp"]["mIoU"]; ub2=reports["UB2_DEPLOYABLE_REPRESENTATION"]["delta_vs_v18_pp"]["mIoU"]
    retention=ub2/ub1 if ub1>0 else float("nan"); eligible=int(cov["eligible"]); covered=int(cov["covered"])
    compcov=covered/max(eligible,1); voxcov=coveredvox/max(totalvox,1); meanc=cov["candidate_sum"]/max(eligible,1)
    sd=[]; scene_rows={}
    for s,x in scene.items():
        b=x["V18_BASE"].compute(); d=x["UB2_DEPLOYABLE_REPRESENTATION"].compute(); z=d["mIoU"]-b["mIoU"]
        if np.isfinite(z):sd.append(float(z)); scene_rows[str(s)]=float(z)
    eps=1e-12
    gate={"delta_mIoU_ge_0_50":ub2>=0.50,"retain_ub1_mIoU_headroom_ge_0_70":np.isfinite(retention) and retention>=0.70,
          "causal_component_coverage_ge_0_70":compcov>=0.70,"mean_candidates_le_10":meanc<=10.0}
    gate["pass"]=all(gate.values())
    result={"protocol":EVAL_PROTOCOL,"v21_protocol":PROTOCOL,
        "scientific_baseline":{"branch":"freeze/v18-main-final-20260918","commit":"ccf7d77e65e9773f441b35083d625b06791bfeaa",
            "checkpoint":str(Path(a.checkpoint).resolve()),"checkpoint_sha256":checkpoint_sha,
            "checkpoint_training_mode":ck.get("training_mode"),"checkpoint_epoch":ck.get("epoch"),
            "forward_exactness":"literal frozen ccf7d77 forward elementwise comparison","renderer_exactness":"frozen runtime exactness check",
            "zero_contribution_identity_windows":len(records),"exactness_windows":checked},
        "population":{"manifest":str(Path(a.population_manifest).resolve()),"num_windows":len(records),
            "selected_key_fingerprint":manifest["selected_key_fingerprint"],"parent_key_fingerprint":manifest["parent_key_fingerprint"],
            "manifest_fingerprint":manifest["manifest_fingerprint"],"parent_stage1_index_sha256":manifest["parent_stage1_index_sha256"],
            "frozen_anchor_lattice":lattice.to_dict()},
        "prototype_bank":{"path":str(Path(a.prototype_bank).resolve()),"requested_k":bank.requested_k,
            "fingerprint":bank.fingerprint,"population_shape_fingerprint":bank.population_shape_fingerprint,
            "algorithm":bank.algorithm,"algorithm_config":dict(bank.algorithm_config),
            "source_provenance":dict(bank.source_provenance)},
        "coverage_radius_m":a.coverage_radius_m,"metrics":reports,
        "headroom":{"ub1_exact_delta_mIoU_pp":ub1,"ub2_deployable_delta_mIoU_pp":ub2,"ub2_retention_of_ub1_mIoU":retention},
        "coverage":{"eligible_targets":eligible,"covered_targets":covered,"component_coverage":compcov,"voxel_coverage":voxcov,
            "historical_matches":int(cov["hist"]),"frontier_matches":int(cov["front"]),"uncovered_targets":eligible-covered,
            "mean_legal_candidates_per_positive":meanc,"max_legal_candidates_per_positive":int(cov["maxcand"]),
            "duplicate_target_assignment":int(cov["dup_t"]),"duplicate_anchor_assignment":int(cov["dup_a"])},
        "coverage_strata_diagnostic":coverage_strata_report(cov_strata,cov_distances),
        "target_mass":{"all_v21_targets":alltargets,"moving_eligible_v21_targets":movingtargets,
            "all_v21_target_voxels_report_horizons":allvox,"moving_eligible_target_voxels_inside_frozen_support":movingvox,
            "per_class_targets":dict(perclass),"per_report_horizon_target_components":dict(perh)},
        "target_audit":dict(audit),"anchor_audit":dict(aa),"shape_audit":dict(sa),
        "collision_audit":{k:dict(v) for k,v in ca.items()},
        "dormant_age_stratum_ub1_exact":{age:{"metrics":m.compute(),"delta_vs_v18_pp":delta(m.compute(),br)} for age,m in ages.items()},
        "scene_delta_mIoU_pp":{"scenes":len(sd),"positive":sum(x>eps for x in sd),
            "zero":sum(abs(x)<=eps for x in sd),"negative":sum(x<-eps for x in sd),
            "mean":float(np.mean(sd)) if sd else float("nan"),"median":float(np.median(sd)) if sd else float("nan"),
            "min":float(np.min(sd)) if sd else float("nan"),"max":float(np.max(sd)) if sd else float("nan"),
            "by_scene":scene_rows},
        "stage0b_gate":gate,
        "contracts":{"report_horizons_s":list(REPORT_H),"all_state_horizons_s":list(ALL_HORIZONS_S),
            "target":"DORMANT_ANCESTRAL + BIRTH, report-horizon eligible","compositor":COMPOSITOR_PROTOCOL,
            "moving_metric":"frozen Moving-mIoU v2; unchanged",
            "coverage_strata":"diagnostic only; formal all-target assignment and gate are unchanged",
            "v20_completion_used":False,"transformer_used":False},
        "elapsed_seconds":time.perf_counter()-started}
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({"output":str(Path(a.output).resolve()),"windows":len(records),"UB0_dmIoU":reports["UB0_EXACT"]["delta_vs_v18_pp"]["mIoU"],
                      "UB1_dmIoU":ub1,"UB2_dmIoU":ub2,"coverage":compcov,"mean_candidates":meanc,"gate_pass":gate["pass"]},indent=2))
    if a.enforce_stage0b_gate and not gate["pass"]:raise SystemExit(2)

if __name__=="__main__":main()
