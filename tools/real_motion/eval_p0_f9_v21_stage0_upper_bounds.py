#!/usr/bin/env python3
"""V21 Stage-0 upper-bound audit. No V21 network or V20 completion is used."""
from __future__ import annotations
import argparse,hashlib,json,os,time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import numpy as np
import torch

from real_motion.metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS
from real_motion.nuscenes_adapter import NuScenesWindowSource,gt_moving_support_sequence
from real_motion.prepared import load_nuscenes_window_raw
from real_motion.runtime_config import add_config_args,load_runtime_config,make_prepare_config
from real_motion.strong_w2det import StrongW2DetConfig
from real_motion.v21_source_induction import (
    ALL_HORIZONS_S,COMPOSITOR_PROTOCOL,POPULATION_PROTOCOL,PROTOCOL,PROTOTYPE_PROTOCOL,REPORT_INDICES,
    KMEDOIDS_CLARA_SAMPLE_SIZE,KMEDOIDS_CLARA_TRIALS,KMEDOIDS_EXACT_MAX_N,
    AnchorLattice,CanonicalShape,PrototypeBank,annotation_map,assign_causal_coverage,
    attribute_instance_shapes,build_frontier_anchors,build_historical_anchors,
    build_v21_targets,compose_v21_add_only,oracle_best_prototype,
    oracle_best_extent_scaled_prototype,
    prototype_bank_fingerprint,rasterize_canonical_shape,
    select_scene_balanced_round_robin,shape_iou,stable_json_fingerprint,
)
from tools.real_motion import eval_p0_f9_v18_se2 as base
from tools.real_motion import eval_p0_f9_v18_full_validation as full
from tools.real_motion import benchmark_p0_f9_v18_runtime as runtime
from tools.real_motion.eval_p0_f9_v17_local_stwm import window_from_record
from tools.real_motion.train_p0_f9_v18_se2_clean import PROTOCOL as CLEAN_PROTOCOL
from tools.real_motion.v20_unified_common import STAGE1_PROTOCOL

EVAL_PROTOCOL="p0_f9_v21_stage0_upper_bound_audit_v6_per_horizon_gt_ceiling"
REPORT_H=(1.0,2.0,3.0)
VARIANTS=("UB_V21_SCOPE_PER_HORIZON_GT","UB0_EXACT","UB1_CAUSAL_EXACT","UB2_DORMANT_CAUSAL_SHAPE",
          "UB2_BIRTH_PROTOTYPE","UB2_DEPLOYABLE_REPRESENTATION",
          "UB2_FACTORIZED_ORACLE_EXTENT")
SEM=tuple(range(17)); DYN=tuple(int(x) for x in DYNAMIC_CLASS_IDS)

class Metrics:
    def __init__(self):
        self.oi=np.zeros(3,np.int64); self.ou=np.zeros(3,np.int64)
        self.si=np.zeros((3,17),np.int64); self.su=np.zeros((3,17),np.int64)
        self.mi=np.zeros((3,len(DYN)),np.int64); self.mu=np.zeros((3,len(DYN)),np.int64)
    @staticmethod
    def counts(p,g,m,free):
        p=np.asarray(p); g=np.asarray(g); m=np.asarray(m,bool)
        if p.shape!=g.shape or p.shape!=m.shape:
            raise ValueError("prediction/GT/moving-support shapes differ")
        n=int(free)+1
        if n!=18 or p.size and (int(p.min())<0 or int(g.min())<0 or
                                int(p.max())>=n or int(g.max())>=n):
            raise ValueError("V21 metric labels violate frozen 0..17 contract")
        code=g.reshape(-1).astype(np.int16,copy=False)*n+p.reshape(-1).astype(np.int16,copy=False)
        conf=np.bincount(code,minlength=n*n).reshape(n,n)
        diag=np.diag(conf); rows=conf.sum(1); cols=conf.sum(0)
        si=diag[:17]; su=rows[:17]+cols[:17]-si
        oi=int(conf[:free,:free].sum()); ou=int(conf.sum()-conf[free,free])
        moving_code=code[m.reshape(-1)]
        moving_conf=np.bincount(moving_code,minlength=n*n).reshape(n,n)
        mdiag=np.diag(moving_conf); mrows=moving_conf.sum(1); mcols=moving_conf.sum(0)
        ids=np.asarray(DYN,dtype=np.int64); mi=mdiag[ids]; mu=mrows[ids]+mcols[ids]-mi
        return oi,ou,si,su,mi,mu
    def update(self,i,p=None,g=None,m=None,free=17,*,counts=None):
        row=self.counts(p,g,m,free) if counts is None else counts
        oi,ou,si,su,mi,mu=row
        self.oi[i]+=oi; self.ou[i]+=ou; self.si[i]+=si; self.su[i]+=su
        self.mi[i]+=mi; self.mu[i]+=mu
        return row
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

def summarize_prototype_fit(rows):
    def one(items):
        iou=np.asarray([x[0] for x in items],dtype=np.float64)
        ratio=np.asarray([x[1] for x in items],dtype=np.float64)
        scaled_iou=np.asarray([x[2] for x in items],dtype=np.float64)
        scaled_ratio=np.asarray([x[3] for x in items],dtype=np.float64)
        if not len(iou):
            return {"targets":0}
        return {
            "targets":int(len(iou)),
            "shape_iou_mean":float(iou.mean()),
            "shape_iou_p10":float(np.percentile(iou,10)),
            "shape_iou_p50":float(np.percentile(iou,50)),
            "shape_iou_p90":float(np.percentile(iou,90)),
            "prototype_to_exact_voxel_ratio_mean":float(ratio.mean()),
            "prototype_to_exact_voxel_ratio_p10":float(np.percentile(ratio,10)),
            "prototype_to_exact_voxel_ratio_p50":float(np.percentile(ratio,50)),
            "prototype_to_exact_voxel_ratio_p90":float(np.percentile(ratio,90)),
            "oversized_targets":int((ratio>1.0).sum()),
            "undersized_targets":int((ratio<1.0).sum()),
            "oracle_extent_shape_iou_mean":float(scaled_iou.mean()),
            "oracle_extent_shape_iou_p50":float(np.percentile(scaled_iou,50)),
            "oracle_extent_prototype_to_exact_voxel_ratio_mean":float(scaled_ratio.mean()),
            "oracle_extent_prototype_to_exact_voxel_ratio_p50":float(np.percentile(scaled_ratio,50)),
        }
    all_rows=[x for values in rows.values() for x in values]
    return {"population":"causally-covered BIRTH targets at query-entry onset",
            "all":one(all_rows),
            "per_class":{str(cid):one(values) for cid,values in sorted(rows.items())}}

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

def future_shape_cache(source,w,raw,targets,pcfg,gate,workers=1):
    wanted={t.instance_token for t in targets}
    def _one(item):
        hi,tok=item
        anns=annotation_map(source.nusc,tok); tokens=sorted(wanted&set(anns))
        attrs=attribute_instance_shapes(
            raw["future_gt_occ"][hi],raw["future_poses"][hi],anns,grid=pcfg.grid,
            free_label=pcfg.free_label,match_max_distance_m=gate,tokens=tokens,
            observation_keys={x:(str(tok),x) for x in tokens})
        return attrs,anns
    items=list(enumerate(w.future_tokens)); nworkers=max(1,min(int(workers),len(items)))
    if nworkers==1:
        rows=[_one(x) for x in items]
    else:
        with ThreadPoolExecutor(max_workers=nworkers) as pool:
            rows=list(pool.map(_one,items))
    return [x[0] for x in rows],[x[1] for x in rows]

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

def per_horizon_gt_component_indices(attribution):
    """Return an exact attributed GT component in its native future grid.

    This deliberately bypasses canonical-shape transport.  It is used only by
    the V21-scope ceiling: target population, semantics and add-only composition
    stay frozen, while each report horizon receives its own attributed GT
    component.  Ambiguous or unresolved attribution fails closed.
    """
    if (attribution is None or bool(attribution.ambiguous) or
            bool(attribution.unresolved) or attribution.voxel_indices is None):
        return None
    indices=np.asarray(attribution.voxel_indices,dtype=np.int64)
    if indices.ndim!=2 or indices.shape[1]!=3:
        raise ValueError("per-horizon GT component indices must be [N,3]")
    return np.unique(indices,axis=0)

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
                                historical,frontier,t0_pose,radius,onset_index_by_token=None):
    subset=list(subset); tokens={t.instance_token for t in subset}; row=store[name]
    matches,report=assign_causal_coverage(
        subset,historical,frontier,t0_pose=t0_pose,coverage_radius_m=radius,
        onset_index_by_token=onset_index_by_token)
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
    p.add_argument("--cpu-workers",type=int,default=0,
                   help="parallel frame I/O/component workers; 0 uses min(8, cpu_count-1)")
    p.add_argument("--allow-incomplete-prototype-bank",action="store_true")
    p.add_argument("--enforce-stage0b-gate",action="store_true"); a=p.parse_args()
    if a.cpu_workers<0 or a.moving_workers<1:
        raise ValueError("worker counts are invalid")
    cpu_workers=(max(1,min(8,(os.cpu_count() or 2)-1))
                 if a.cpu_workers==0 else max(1,a.cpu_workers))
    pcfg=make_prepare_config(load_runtime_config(a.config,a.override)); manifest,keys,lattice=load_manifest(a.population_manifest)
    _,records0=base.load_cache(a.val_cache); records=align_records(records0,keys)
    bank=load_bank(a.prototype_bank,allow_incomplete=a.allow_incomplete_prototype_bank)
    # dev64/dev512 are scene-balanced and offer almost no large-array cache
    # reuse.  Retaining every 3-D occupancy volume creates memory pressure;
    # bounded frame-level concurrency is both faster and much smaller.
    source=NuScenesWindowSource(a.dataroot,info_pkl=a.info_pkl,verbose=False); strong=StrongW2DetConfig(free_label=int(pcfg.free_label))
    device=torch.device(a.device if a.device!="cuda" or torch.cuda.is_available() else "cpu")
    ck,model,_=full._load_model(a.checkpoint,CLEAN_PROTOCOL,device)
    checkpoint_sha=validate_clean_e14_checkpoint(ck,a.checkpoint,a.expected_checkpoint_sha256)

    states={"V18_BASE":Metrics(),**{v:Metrics() for v in VARIANTS}}
    ages={str(x):Metrics() for x in (0.5,1.0,1.5,2.0,2.5)}
    q={v:defaultdict(int,{"added":0,"occ_tp":0,"semantic_tp":0,"target_addable":0,"target_recovered":0}) for v in VARIANTS}
    audit=defaultdict(int); aa=defaultdict(int); sa=defaultdict(int); ca={v:defaultdict(int) for v in VARIANTS}; cov=defaultdict(float)
    identity_cov=defaultdict(float)
    cov_strata=defaultdict(lambda:defaultdict(float)); cov_distances=defaultdict(list)
    prototype_fit=defaultdict(list)
    perclass=defaultdict(int); queryperclass=defaultdict(int); perh=defaultdict(int)
    totalvox=coveredvox=alltargets=querytargets=movingtargets=querymovingtargets=allvox=movingvox=0
    scene=defaultdict(lambda:{"V18_BASE":Metrics(),**{v:Metrics() for v in VARIANTS}})
    checked=0; started=time.perf_counter(); timings=defaultdict(float)

    for wi,rec in enumerate(records,1):
        window_started=time.perf_counter(); w=window_from_record(rec)
        stage_started=time.perf_counter()
        raw=load_nuscenes_window_raw(
            source,w,pcfg,include_gt=True,io_workers=cpu_workers)
        timings["raw_load"]+=time.perf_counter()-stage_started
        stage_started=time.perf_counter()
        st=runtime._prepare_record(rec,source,pcfg,strong,device,raw_window=raw); runtime._stage_gpu_inputs(st,device)
        try:
            if checked<a.exactness_windows: assert_forward_exact(model,st,device); runtime._exactness_check(model,st,pcfg,strong,device); checked+=1
            basepred=runtime._forecast_once(model,st,pcfg,strong,device)
        finally: runtime._release_gpu_inputs(st)
        timings["v18_prepare_forecast"]+=time.perf_counter()-stage_started
        for h in range(6):
            noop,_=compose_v21_add_only(basepred[h],[],free_label=pcfg.free_label)
            if not np.array_equal(noop,basepred[h]):raise RuntimeError("zero-contribution identity failed")

        stage_started=time.perf_counter()
        hist,_,ha,history_evidence=build_historical_anchors(
            source,w,raw["history_occ"],raw["history_observed"],raw["history_poses"],
            grid=pcfg.grid,strong_cfg=strong,frame_dt_s=pcfg.frame_dt_s,
            match_max_distance_m=a.match_max_distance_m,workers=cpu_workers)
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
        identity_matches,identity_report=assign_causal_coverage(
            targets,hist,front,t0_pose=raw["history_poses"][-1],coverage_radius_m=a.coverage_radius_m)
        if identity_report["duplicate_target_assignment"] or identity_report["duplicate_anchor_assignment"]:
            raise RuntimeError("V21 annotation-identity assignment violated one-to-one contract")
        identity_cov["eligible"]+=identity_report["eligible_targets"]
        identity_cov["covered"]+=identity_report["covered_targets"]
        identity_cov["candidate_sum"]+=identity_report["mean_legal_candidates_per_positive"]*identity_report["eligible_targets"]
        identity_cov["hist"]+=identity_report["historical_matches"]
        identity_cov["front"]+=identity_report["frontier_matches"]
        identity_cov["maxcand"]=max(identity_cov["maxcand"],identity_report["max_legal_candidates_per_positive"])
        timings["history_frontier"]+=time.perf_counter()-stage_started

        stage_started=time.perf_counter()
        future_attrs,future_ann_maps=future_shape_cache(
            source,w,raw,targets,pcfg,a.match_max_distance_m,workers=cpu_workers)
        masks=target_masks_from_attributions(future_attrs,pcfg.grid.shape_hwd)
        onset_resolved=set(); any_resolved=set(); report_component=set(); first_resolved={}
        for t in targets:
            tok=t.instance_token
            resolved=[hi for hi,row in enumerate(future_attrs)
                      if (row.get(tok) is not None and row[tok].shape is not None)]
            if resolved:
                any_resolved.add(tok); first=resolved[0]; first_resolved[tok]=first
                sa[f"first_resolved_horizon/{ALL_HORIZONS_S[first]:.1f}s"]+=1
                if first>t.onset_index:sa["resolved_only_after_annotation_onset"]+=1
            else:sa["no_future_shape_resolved"]+=1
            if t.onset_index in resolved:onset_resolved.add(tok)
            if any(tok in masks[hi] for hi in REPORT_INDICES):report_component.add(tok)
            onset_attr=future_attrs[t.onset_index].get(tok)
            if onset_attr is not None and onset_attr.ambiguous:
                sa["annotation_onset_shape_ambiguous"]+=1
            ann=future_ann_maps[t.onset_index].get(tok)
            if ann is not None:
                inside=annotation_center_inside_grid(
                    ann["center_world"],raw["future_poses"][t.onset_index],pcfg.grid)
                sa["annotation_onset_center_inside_grid" if inside else "annotation_onset_center_outside_grid"]+=1
                if inside and tok not in onset_resolved:sa["onset_inside_grid_but_shape_unresolved"]+=1
        sa["onset_shape_resolved"]+=len(onset_resolved)
        sa["any_future_shape_resolved"]+=len(any_resolved)
        sa["report_horizon_component_resolved"]+=len(report_component)
        eval_targets=[t for t in targets if t.instance_token in report_component]
        query_onset={t.instance_token:first_resolved[t.instance_token] for t in eval_targets}
        querytargets+=len(eval_targets)
        for t in eval_targets:queryperclass[str(t.class_id)]+=1
        matches,cr=assign_causal_coverage(
            eval_targets,hist,front,t0_pose=raw["history_poses"][-1],
            coverage_radius_m=a.coverage_radius_m,onset_index_by_token=query_onset)
        if cr["duplicate_target_assignment"] or cr["duplicate_anchor_assignment"]:
            raise RuntimeError("V21 query-entry assignment violated one-to-one contract")
        mb={m.target_token:m for m in matches}; cov["eligible"]+=cr["eligible_targets"]; cov["covered"]+=cr["covered_targets"]
        cov["candidate_sum"]+=cr["mean_legal_candidates_per_positive"]*cr["eligible_targets"]
        cov["hist"]+=cr["historical_matches"]; cov["front"]+=cr["frontier_matches"]
        cov["dup_t"]+=cr["duplicate_target_assignment"]; cov["dup_a"]+=cr["duplicate_anchor_assignment"]
        cov["maxcand"]=max(cov["maxcand"],cr["max_legal_candidates_per_positive"])
        all_match_tokens={m.target_token for m in identity_matches}
        strata={"onset_shape_resolved":onset_resolved,
                "any_future_shape_resolved":any_resolved,
                "report_horizon_component_resolved":report_component}
        for name,tokens in strata.items():
            accumulate_coverage_stratum(
                cov_strata,cov_distances,name,
                (t for t in targets if t.instance_token in tokens),all_match_tokens,
                hist,front,raw["history_poses"][-1],a.coverage_radius_m)
        accumulate_coverage_stratum(
            cov_strata,cov_distances,"report_component_query_entry",eval_targets,set(mb),
            hist,front,raw["history_poses"][-1],a.coverage_radius_m,query_onset)
        timings["future_attribution_assignment"]+=time.perf_counter()-stage_started
        representation_started=time.perf_counter()
        onset={}; hshape={}; proto={}; scaled_proto={}; hage={}; rank={t.instance_token:i for i,t in enumerate(eval_targets)}
        for t in eval_targets:
            tok=t.instance_token; oi=query_onset[tok]; at=future_attrs[oi].get(tok)
            if at is None or at.shape is None:
                if at is not None and at.ambiguous:sa["shape_ambiguous"]+=1
                sa["query_entry_shape_unresolved"]+=1
            else:
                ann=future_ann_maps[oi][tok]
                roundtrip,_=rasterize_canonical_shape(
                    at.shape,ann["center_world"],ann["yaw_world"],raw["future_poses"][oi],grid=pcfg.grid)
                expected=np.unique(np.asarray(at.voxel_indices,dtype=np.int64),axis=0)
                if not np.array_equal(roundtrip,expected):
                    raise RuntimeError(f"query-entry exact-shape round-trip failed: {w.scene_name}/{w.t0_token}/{tok}")
                sa["query_entry_roundtrip_exact"]+=1
                onset[tok]=at.shape; pp=oracle_best_prototype(at.shape,bank)
                if pp is not None:
                    proto[tok]=pp
                    scaled_proto[tok]=oracle_best_extent_scaled_prototype(at.shape,bank)
                    if t.responsibility=="BIRTH" and tok in mb:
                        exact_n=max(len(at.shape.cells_ijk),1)
                        scaled=scaled_proto[tok]
                        prototype_fit[int(t.class_id)].append(
                            (shape_iou(at.shape,pp),len(pp.cells_ijk)/exact_n,
                             shape_iou(at.shape,scaled),len(scaled.cells_ijk)/exact_n))
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
        timings["representation_prepare"]+=time.perf_counter()-representation_started

        stage_started=time.perf_counter()
        moving_rows=gt_moving_support_sequence(
            source.nusc,str(w.t0_token),w.future_tokens,ALL_HORIZONS_S,
            grid=pcfg.grid,workers=max(a.moving_workers,cpu_workers))
        timings["moving_support"]+=time.perf_counter()-stage_started
        moving=np.stack([x[0] for x in moving_rows]); mt=set()
        for hi in REPORT_INDICES:mt.update(str(r["instance_token"]) for r in moving_rows[hi][1])
        movingtargets+=sum(t.instance_token in mt for t in targets)
        querymovingtargets+=sum(t.instance_token in mt for t in eval_targets)
        covered=set(mb)
        for hi in REPORT_INDICES:
            for t in eval_targets:
                m=masks[hi].get(t.instance_token)
                if m is None:continue
                n=int(m.sum()); allvox+=n; totalvox+=n
                if t.instance_token in covered:coveredvox+=n
                if t.instance_token in mt:movingvox+=int((m&moving[hi]).sum())

        stage_started=time.perf_counter()
        for pos,hi in enumerate(REPORT_INDICES):
            gt=np.asarray(raw["future_gt_occ"][hi],np.uint8); bh=np.asarray(basepred[hi],np.uint8)
            metric_counts=states["V18_BASE"].update(pos,bh,gt,moving[hi],pcfg.free_label)
            scene[w.scene_name]["V18_BASE"].update(pos,counts=metric_counts)
            tmask=np.zeros(pcfg.grid.shape_hwd,bool)
            for t in eval_targets:
                if t.instance_token in masks[hi]:tmask|=masks[hi][t.instance_token]
            props={v:[] for v in VARIANTS}; ageprops={k:[] for k in ages}; oob=defaultdict(int)
            for t in eval_targets:
                tok=t.instance_token
                if hi<query_onset[tok] or not t.existence[hi]:continue
                ann=future_ann_maps[hi].get(tok)
                if ann is None:continue
                cm=mb.get(tok); kind,aid=proposal_key(t,cm,rank[tok])
                gt_component=per_horizon_gt_component_indices(future_attrs[hi].get(tok))
                if gt_component is None:
                    sa["report_horizon_gt_component_missing_target_frames"]+=1
                else:
                    props["UB_V21_SCOPE_PER_HORIZON_GT"].append(
                        (kind,aid,t.class_id,gt_component))
                    sa["report_horizon_gt_component_target_frames"]+=1
                reps={"UB0_EXACT":onset.get(tok),
                      "UB1_CAUSAL_EXACT":onset.get(tok) if cm else None,
                      "UB2_DORMANT_CAUSAL_SHAPE":(hshape.get(tok) if t.responsibility=="DORMANT_ANCESTRAL" else onset.get(tok)) if cm else None,
                      "UB2_BIRTH_PROTOTYPE":(proto.get(tok) if t.responsibility=="BIRTH" else onset.get(tok)) if cm else None,
                      "UB2_DEPLOYABLE_REPRESENTATION":(hshape.get(tok) if t.responsibility=="DORMANT_ANCESTRAL" else proto.get(tok)) if cm else None,
                      "UB2_FACTORIZED_ORACLE_EXTENT":(hshape.get(tok) if t.responsibility=="DORMANT_ANCESTRAL" else scaled_proto.get(tok)) if cm else None}
                for v,s in reps.items():
                    if s is None:continue
                    idx,oo=rasterize_canonical_shape(s,ann["center_world"],ann["yaw_world"],raw["future_poses"][hi],grid=pcfg.grid)
                    props[v].append((kind,aid,t.class_id,idx)); oob[v]+=oo
                if cm and t.responsibility=="DORMANT_ANCESTRAL" and tok in onset:
                    age=f"{hage.get(tok,-1):.1f}"
                    if age in ageprops:
                        idx,_=rasterize_canonical_shape(onset[tok],ann["center_world"],ann["yaw_world"],raw["future_poses"][hi],grid=pcfg.grid)
                        ageprops[age].append((kind,aid,t.class_id,idx))
            def _evaluate_variant(v):
                pred,r=compose_v21_add_only(bh,props[v],free_label=pcfg.free_label)
                counts=Metrics.counts(pred,gt,moving[hi],pcfg.free_label)
                return v,r,counts,add_counts(bh,pred,gt,tmask,pcfg.free_label)
            variant_workers=max(1,min(cpu_workers,len(VARIANTS)))
            if variant_workers==1:
                variant_rows=[_evaluate_variant(v) for v in VARIANTS]
            else:
                with ThreadPoolExecutor(max_workers=variant_workers) as pool:
                    variant_rows=list(pool.map(_evaluate_variant,VARIANTS))
            for v,r,metric_counts,addition_counts in variant_rows:
                states[v].update(pos,counts=metric_counts)
                scene[w.scene_name][v].update(pos,counts=metric_counts); merge_counts(q[v],addition_counts)
                ca[v]["v21_collision_voxels"]+=r.v21_collision_voxels; ca[v]["historical_frontier_collision_voxels"]+=r.historical_frontier_collision_voxels
                ca[v]["blocked_by_v18_voxels"]+=r.blocked_by_v18_voxels
                ca[v]["out_of_bounds_voxels"]+=oob[v]+r.out_of_bounds_voxels
            for age,pp in ageprops.items():
                pred,_=compose_v21_add_only(bh,pp,free_label=pcfg.free_label); ages[age].update(pos,pred,gt,moving[hi],pcfg.free_label)
        timings["render_metrics"]+=time.perf_counter()-stage_started
        timings["total_window"]+=time.perf_counter()-window_started
        if wi==1 or wi%16==0 or wi==len(records):
            elapsed=time.perf_counter()-started
            print(f"v21_stage0 {wi}/{len(records)} workers={cpu_workers} "
                  f"seconds_per_window={timings['total_window']/wi:.2f} "
                  f"elapsed_seconds={elapsed:.1f}",flush=True)

    br=states["V18_BASE"].compute(); reports={"V18_BASE":br}
    for v in VARIANTS:
        rr=states[v].compute(); reports[v]={"metrics":rr,"delta_vs_v18_pp":delta(rr,br),"addition_quality":quality(q[v])}
    scope_counts=q["UB_V21_SCOPE_PER_HORIZON_GT"]
    scope_exactness={
        "all_added_voxels_are_occupied_gt":scope_counts["occ_tp"]==scope_counts["added"],
        "all_added_voxels_match_gt_semantics":scope_counts["semantic_tp"]==scope_counts["added"],
        "all_addable_target_voxels_recovered":scope_counts["target_recovered"]==scope_counts["target_addable"],
    }
    if not all(scope_exactness.values()):
        raise RuntimeError(
            "per-horizon GT scope ceiling violated exact add-only recovery: "
            f"{scope_exactness}; counts={dict(scope_counts)}")
    scope_gt=reports["UB_V21_SCOPE_PER_HORIZON_GT"]["delta_vs_v18_pp"]["mIoU"]
    query_entry_rigid=reports["UB0_EXACT"]["delta_vs_v18_pp"]["mIoU"]
    rigid_retention=query_entry_rigid/scope_gt if scope_gt>0 else float("nan")
    ub1=reports["UB1_CAUSAL_EXACT"]["delta_vs_v18_pp"]["mIoU"]; ub2=reports["UB2_DEPLOYABLE_REPRESENTATION"]["delta_vs_v18_pp"]["mIoU"]
    retention=ub2/ub1 if ub1>0 else float("nan"); eligible=int(cov["eligible"]); covered=int(cov["covered"])
    factorized_ub2=reports["UB2_FACTORIZED_ORACLE_EXTENT"]["delta_vs_v18_pp"]["mIoU"]
    factorized_retention=factorized_ub2/ub1 if ub1>0 else float("nan")
    compcov=covered/max(eligible,1); voxcov=coveredvox/max(totalvox,1); meanc=cov["candidate_sum"]/max(eligible,1)
    sd=[]; scene_rows={}
    for s,x in scene.items():
        b=x["V18_BASE"].compute(); d=x["UB2_DEPLOYABLE_REPRESENTATION"].compute(); z=d["mIoU"]-b["mIoU"]
        if np.isfinite(z):sd.append(float(z)); scene_rows[str(s)]=float(z)
    eps=1e-12
    gate={"delta_mIoU_ge_0_50":ub2>=0.50,"retain_ub1_mIoU_headroom_ge_0_70":np.isfinite(retention) and retention>=0.70,
          "causal_component_coverage_ge_0_70":compcov>=0.70,"mean_candidates_le_10":meanc<=10.0}
    gate["pass"]=all(gate.values())
    factorized_gate={"delta_mIoU_ge_0_50":factorized_ub2>=0.50,
        "retain_ub1_mIoU_headroom_ge_0_70":np.isfinite(factorized_retention) and factorized_retention>=0.70,
        "causal_component_coverage_ge_0_70":compcov>=0.70,"mean_candidates_le_10":meanc<=10.0}
    factorized_gate["pass"]=all(factorized_gate.values())
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
        "v21_scope_ceiling":{"per_horizon_gt_component_delta_mIoU_pp":scope_gt,
            "query_entry_rigid_exact_delta_mIoU_pp":query_entry_rigid,
            "query_entry_rigid_retention_of_per_horizon_gt":rigid_retention,
            "rigid_shape_gap_mIoU_pp":scope_gt-query_entry_rigid,
            "exactness":scope_exactness},
        "headroom":{"ub1_exact_delta_mIoU_pp":ub1,"ub2_deployable_delta_mIoU_pp":ub2,"ub2_retention_of_ub1_mIoU":retention},
        "factorized_oracle_extent_headroom":{"ub1_exact_delta_mIoU_pp":ub1,
            "ub2_factorized_delta_mIoU_pp":factorized_ub2,
            "ub2_factorized_retention_of_ub1_mIoU":factorized_retention},
        "coverage":{"population":"report_horizon_component_resolved at query-entry onset",
            "eligible_targets":eligible,"covered_targets":covered,"component_coverage":compcov,"voxel_coverage":voxcov,
            "historical_matches":int(cov["hist"]),"frontier_matches":int(cov["front"]),"uncovered_targets":eligible-covered,
            "mean_legal_candidates_per_positive":meanc,"max_legal_candidates_per_positive":int(cov["maxcand"]),
            "duplicate_target_assignment":int(cov["dup_t"]),"duplicate_anchor_assignment":int(cov["dup_a"])},
        "annotation_identity_coverage_diagnostic":{
            "eligible_targets":int(identity_cov["eligible"]),"covered_targets":int(identity_cov["covered"]),
            "coverage":identity_cov["covered"]/max(identity_cov["eligible"],1),
            "historical_matches":int(identity_cov["hist"]),"frontier_matches":int(identity_cov["front"]),
            "mean_legal_candidates_per_positive":identity_cov["candidate_sum"]/max(identity_cov["eligible"],1),
            "max_legal_candidates_per_positive":int(identity_cov["maxcand"])},
        "coverage_strata_diagnostic":coverage_strata_report(cov_strata,cov_distances),
        "target_mass":{"all_annotation_v21_targets":alltargets,"query_entry_component_targets":querytargets,
            "moving_eligible_annotation_targets":movingtargets,"moving_eligible_query_entry_targets":querymovingtargets,
            "query_entry_target_voxels_report_horizons":allvox,
            "moving_eligible_query_entry_voxels_inside_frozen_support":movingvox,
            "per_class_annotation_targets":dict(perclass),"per_class_query_entry_targets":dict(queryperclass),
            "per_report_horizon_annotation_targets":dict(perh)},
        "target_audit":dict(audit),"anchor_audit":dict(aa),"shape_audit":dict(sa),
        "prototype_fit_diagnostic":summarize_prototype_fit(prototype_fit),
        "collision_audit":{k:dict(v) for k,v in ca.items()},
        "dormant_age_stratum_ub1_exact":{age:{"metrics":m.compute(),"delta_vs_v18_pp":delta(m.compute(),br)} for age,m in ages.items()},
        "scene_delta_mIoU_pp":{"scenes":len(sd),"positive":sum(x>eps for x in sd),
            "zero":sum(abs(x)<=eps for x in sd),"negative":sum(x<-eps for x in sd),
            "mean":float(np.mean(sd)) if sd else float("nan"),"median":float(np.median(sd)) if sd else float("nan"),
            "min":float(np.min(sd)) if sd else float("nan"),"max":float(np.max(sd)) if sd else float("nan"),
            "by_scene":scene_rows},
        "stage0b_gate":gate,
        "factorized_oracle_extent_gate":factorized_gate,
        "performance":{"cpu_workers":cpu_workers,"moving_workers":max(a.moving_workers,cpu_workers),
            "seconds_total":time.perf_counter()-started,
            "seconds_by_stage":{k:float(v) for k,v in timings.items()},
            "mean_seconds_per_window":timings["total_window"]/max(len(records),1)},
        "contracts":{"report_horizons_s":list(REPORT_H),"all_state_horizons_s":list(ALL_HORIZONS_S),
            "target":"DORMANT_ANCESTRAL + BIRTH with report-horizon occupancy component",
            "query_entry_onset":"first unambiguous future occupancy component in frozen Omega_max; annotation existence remains six-frame",
            "per_horizon_gt_scope_ceiling":"each report horizon uses its own unambiguous attributed GT component; V21 target scope and V18-free add-only compositor remain frozen; diagnostic only",
            "ub0_exact_semantics":"query-entry exact canonical shape rigidly transported with GT state; not a per-horizon or whole-scene GT oracle",
            "compositor":COMPOSITOR_PROTOCOL,
            "moving_metric":"frozen Moving-mIoU v2; unchanged",
            "coverage_strata":"annotation-onset strata are diagnostic; formal gate uses report-component query-entry assignment",
            "oracle_extent_diagnostic":"GT query-entry occupied extent only; diagnostic representation ceiling, not a deployable prediction",
            "v20_completion_used":False,"transformer_used":False},
        "elapsed_seconds":time.perf_counter()-started}
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({"output":str(Path(a.output).resolve()),"windows":len(records),
                      "V21_scope_per_horizon_GT_dmIoU":scope_gt,
                      "query_entry_rigid_exact_dmIoU":query_entry_rigid,
                      "query_entry_rigid_retention":rigid_retention,
                      "UB1_dmIoU":ub1,"UB2_dmIoU":ub2,
                      "factorized_oracle_extent_dmIoU":factorized_ub2,
                      "coverage":compcov,"mean_candidates":meanc,
                      "gate_pass":gate["pass"],
                      "factorized_oracle_extent_gate_pass":factorized_gate["pass"]},indent=2))
    if a.enforce_stage0b_gate and not gate["pass"]:raise SystemExit(2)

if __name__=="__main__":main()
