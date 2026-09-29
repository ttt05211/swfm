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
from real_motion.v19_innovation_targets import match_future_components_many_to_one
from real_motion.runtime_fastpath import extract_instances_cropped_exact
from real_motion.v21_source_induction import (
    ALL_HORIZONS_S,COMPOSITOR_PROTOCOL,PROTOCOL,REPORT_INDICES,
    CanonicalShape,PrototypeBank,annotation_map,assign_causal_coverage,
    attribute_instance_shape,build_frontier_anchors,build_historical_anchors,
    build_v21_targets,compose_v21_add_only,oracle_best_prototype,
    rasterize_canonical_shape,reliable_components_and_tokens,stable_json_fingerprint,
)
from tools.real_motion import eval_p0_f9_v18_se2 as base
from tools.real_motion import eval_p0_f9_v18_full_validation as full
from tools.real_motion import benchmark_p0_f9_v18_runtime as runtime
from tools.real_motion.diagnose_p0_f9_v19_innovation_decomposition import CachedSource
from tools.real_motion.eval_p0_f9_v17_local_stwm import window_from_record
from tools.real_motion.train_p0_f9_v18_se2_clean import PROTOCOL as CLEAN_PROTOCOL

EVAL_PROTOCOL="p0_f9_v21_stage0_upper_bound_audit_v1"
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

def load_manifest(path):
    x=json.loads(Path(path).read_text())
    if x.get("protocol")!="p0_f9_v21_population_manifest_v1": raise RuntimeError("bad population manifest")
    keys=tuple((str(a),str(b)) for a,b in x["selected_keys"])
    if stable_json_fingerprint([list(k) for k in keys])!=x["selected_key_fingerprint"]: raise RuntimeError("manifest fingerprint mismatch")
    return x,keys

def align_records(records,keys):
    m={(str(r["scene_name"]),str(r["t0_token"])):r for r in records}
    missing=[k for k in keys if k not in m]
    if missing: raise RuntimeError(f"V18 cache missing keys: {missing[:5]}")
    return [m[k] for k in keys]

def load_bank(path):
    x=torch.load(path,map_location="cpu",weights_only=False)
    if x.get("protocol")!="v21_train_only_binary_iou_kmedoids_v1": raise RuntimeError("bad prototype bank")
    med={}
    for cid,rows in x["medoids_by_class"].items():
        med[int(cid)]=tuple(CanonicalShape(int(r["class_id"]),r["cells_ijk"].numpy().astype(np.int32),
                                           (str(r["observation_key"][0]),str(r["observation_key"][1]))) for r in rows)
    return PrototypeBank(x["protocol"],float(x["resolution_m"]),int(x["requested_k"]),med,
                         tuple((str(a),str(b)) for a,b in x["population_manifest"]),str(x["fingerprint"]))

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

def last_history_shape(source,w,raw,target,pcfg,strong,gate):
    tok=target.instance_token
    for i in range(5,-1,-1):
        _,matched,amb=reliable_components_and_tokens(source,w.scene_name,w.history_tokens[i],raw["history_occ"][i],
            raw["history_observed"][i],raw["history_poses"][i],grid=pcfg.grid,strong_cfg=strong,match_max_distance_m=gate)
        if tok not in {str(x) for x in matched if x is not None} or tok in amb: continue
        anns=annotation_map(source.nusc,w.history_tokens[i]); masked=np.where(raw["history_observed"][i],raw["history_occ"][i],pcfg.free_label).astype(np.uint8)
        return attribute_instance_shape(masked,raw["history_poses"][i],anns,tok,grid=pcfg.grid,free_label=pcfg.free_label,
                                        match_max_distance_m=gate,observation_key=(str(w.history_tokens[i]),tok)),i
    return None,None

def target_masks(source,w,raw,targets,pcfg,gate):
    wanted={t.instance_token for t in targets}; out=[{} for _ in range(6)]
    cfg=StrongW2DetConfig(free_label=int(pcfg.free_label),min_component_voxels=1)
    for hi,tok in enumerate(w.future_tokens):
        anns=annotation_map(source.nusc,tok)
        comps=extract_instances_cropped_exact(np.asarray(raw["future_gt_occ"][hi],np.uint8),np.asarray(raw["future_poses"][hi]),grid=pcfg.grid,cfg=cfg)
        mm=match_future_components_many_to_one(comps,anns,max_distance_m=gate)
        for comp,(inst,_) in zip(comps,mm):
            if inst is None or str(inst) not in wanted:continue
            m=out[hi].setdefault(str(inst),np.zeros(pcfg.grid.shape_hwd,bool)); idx=np.asarray(comp["voxel_indices"])
            m[idx[:,0],idx[:,1],idx[:,2]]=True
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

def main():
    p=argparse.ArgumentParser(); add_config_args(p)
    for name in ("val-cache","population-manifest","checkpoint","prototype-bank","dataroot","info-pkl","output"):
        p.add_argument("--"+name,required=True)
    p.add_argument("--coverage-radius-m",type=float,choices=(0.8,1.6,3.2),required=True)
    p.add_argument("--match-max-distance-m",type=float,default=4.0); p.add_argument("--device",default="cuda")
    p.add_argument("--exactness-windows",type=int,default=1); p.add_argument("--moving-workers",type=int,default=1)
    p.add_argument("--enforce-stage0b-gate",action="store_true"); a=p.parse_args()
    pcfg=make_prepare_config(load_runtime_config(a.config,a.override)); manifest,keys=load_manifest(a.population_manifest)
    _,records0=base.load_cache(a.val_cache); records=align_records(records0,keys); bank=load_bank(a.prototype_bank)
    source=CachedSource(a.dataroot,info_pkl=a.info_pkl,verbose=False); strong=StrongW2DetConfig(free_label=int(pcfg.free_label))
    device=torch.device(a.device if a.device!="cuda" or torch.cuda.is_available() else "cpu")
    _,model,_=full._load_model(a.checkpoint,CLEAN_PROTOCOL,device)

    states={"V18_BASE":Metrics(),**{v:Metrics() for v in VARIANTS}}
    ages={str(x):Metrics() for x in (0.5,1.0,1.5,2.0,2.5)}
    q={v:defaultdict(int,{"added":0,"occ_tp":0,"semantic_tp":0,"target_addable":0,"target_recovered":0}) for v in VARIANTS}
    audit=defaultdict(int); aa=defaultdict(int); sa=defaultdict(int); ca={v:defaultdict(int) for v in VARIANTS}; cov=defaultdict(float)
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

        targets,ta=build_v21_targets(source,w,np.asarray(raw["history_occ"]),np.asarray(raw["history_observed"]),np.asarray(raw["history_poses"]),
                                     grid=pcfg.grid,strong_cfg=strong,match_max_distance_m=a.match_max_distance_m)
        for k,v in ta.items():audit[k]+=int(v)
        alltargets+=len(targets)
        for t in targets:
            perclass[str(t.class_id)]+=1
            for hi in REPORT_INDICES:
                if t.existence[hi]:perh[str(ALL_HORIZONS_S[hi])]+=1
        hist,_,ha=build_historical_anchors(source,w,raw["history_occ"],raw["history_observed"],raw["history_poses"],
                                           grid=pcfg.grid,strong_cfg=strong,frame_dt_s=pcfg.frame_dt_s,match_max_distance_m=a.match_max_distance_m)
        front,fa=build_frontier_anchors(raw["history_observed"],raw["history_poses"],raw["future_poses"],grid=pcfg.grid)
        for k,v in ha.items():
            if isinstance(v,dict):
                for kk,vv in v.items():aa[f"{k}/{kk}"]+=int(vv)
            else:aa[k]+=int(v)
        aa["frontier_anchor_count"]+=fa["frontier_anchor_count"]
        matches,cr=assign_causal_coverage(targets,hist,front,t0_pose=raw["history_poses"][-1],coverage_radius_m=a.coverage_radius_m)
        mb={m.target_token:m for m in matches}; cov["eligible"]+=cr["eligible_targets"]; cov["covered"]+=cr["covered_targets"]
        cov["candidate_sum"]+=cr["mean_legal_candidates_per_positive"]*cr["eligible_targets"]
        cov["hist"]+=cr["historical_matches"]; cov["front"]+=cr["frontier_matches"]
        cov["dup_t"]+=cr["duplicate_target_assignment"]; cov["dup_a"]+=cr["duplicate_anchor_assignment"]
        cov["maxcand"]=max(cov["maxcand"],cr["max_legal_candidates_per_positive"])

        onset={}; hshape={}; proto={}; hage={}; rank={t.instance_token:i for i,t in enumerate(targets)}
        for t in targets:
            tok=t.instance_token; oi=t.onset_index; anns=annotation_map(source.nusc,w.future_tokens[oi])
            at=attribute_instance_shape(raw["future_gt_occ"][oi],raw["future_poses"][oi],anns,tok,grid=pcfg.grid,
                                        free_label=pcfg.free_label,match_max_distance_m=a.match_max_distance_m,
                                        observation_key=(str(w.future_tokens[oi]),tok))
            if at.ambiguous:sa["shape_ambiguous"]+=1
            if at.shape is None:sa["shape_unresolved"]+=1
            else:
                onset[tok]=at.shape; pp=oracle_best_prototype(at.shape,bank)
                if pp is not None:proto[tok]=pp
                elif t.responsibility=="BIRTH":sa["prototype_missing_class"]+=1
            if t.responsibility=="DORMANT_ANCESTRAL":
                hh,idx=last_history_shape(source,w,raw,t,pcfg,strong,a.match_max_distance_m)
                if hh is None or hh.shape is None:sa["dormant_history_shape_unresolved"]+=1
                else:hshape[tok]=hh.shape; hage[tok]=(5-idx)*pcfg.frame_dt_s

        moving_rows=gt_moving_support_sequence(source.nusc,str(w.t0_token),w.future_tokens,ALL_HORIZONS_S,grid=pcfg.grid,workers=a.moving_workers)
        moving=np.stack([x[0] for x in moving_rows]); mt=set()
        for hi in REPORT_INDICES:mt.update(str(r["instance_token"]) for r in moving_rows[hi][1])
        movingtargets+=sum(t.instance_token in mt for t in targets)
        masks=target_masks(source,w,raw,targets,pcfg,a.match_max_distance_m); covered=set(mb)
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
                ann=annotation_map(source.nusc,w.future_tokens[hi]).get(tok)
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
                ca[v]["blocked_by_v18_voxels"]+=r.blocked_by_v18_voxels; ca[v]["out_of_bounds_voxels"]+=oob[v]
            for age,pp in ageprops.items():
                pred,_=compose_v21_add_only(bh,pp,free_label=pcfg.free_label); ages[age].update(pos,pred,gt,moving[hi],pcfg.free_label)
        if wi==1 or wi%16==0 or wi==len(records):print(f"v21_stage0 {wi}/{len(records)}",flush=True)

    br=states["V18_BASE"].compute(); reports={"V18_BASE":br}
    for v in VARIANTS:
        rr=states[v].compute(); reports[v]={"metrics":rr,"delta_vs_v18_pp":delta(rr,br),"addition_quality":quality(q[v])}
    ub1=reports["UB1_CAUSAL_EXACT"]["delta_vs_v18_pp"]["mIoU"]; ub2=reports["UB2_DEPLOYABLE_REPRESENTATION"]["delta_vs_v18_pp"]["mIoU"]
    retention=ub2/ub1 if ub1>0 else float("nan"); eligible=int(cov["eligible"]); covered=int(cov["covered"])
    compcov=covered/max(eligible,1); voxcov=coveredvox/max(totalvox,1); meanc=cov["candidate_sum"]/max(eligible,1)
    sd=[]
    for s,x in scene.items():
        b=x["V18_BASE"].compute(); d=x["UB2_DEPLOYABLE_REPRESENTATION"].compute(); z=d["mIoU"]-b["mIoU"]
        if np.isfinite(z):sd.append(float(z))
    gate={"delta_mIoU_ge_0_50":ub2>=0.50,"retain_ub1_mIoU_headroom_ge_0_70":np.isfinite(retention) and retention>=0.70,
          "causal_component_coverage_ge_0_70":compcov>=0.70,"mean_candidates_le_10":meanc<=10.0}
    gate["pass"]=all(gate.values())
    result={"protocol":EVAL_PROTOCOL,"v21_protocol":PROTOCOL,
        "scientific_baseline":{"branch":"freeze/v18-main-final-20260918","commit":"ccf7d77e65e9773f441b35083d625b06791bfeaa",
            "checkpoint":str(Path(a.checkpoint).resolve()),"checkpoint_sha256":sha256(a.checkpoint),
            "forward_exactness":"literal frozen ccf7d77 forward elementwise comparison","renderer_exactness":"frozen runtime exactness check",
            "zero_contribution_identity_windows":len(records),"exactness_windows":checked},
        "population":{"manifest":str(Path(a.population_manifest).resolve()),"num_windows":len(records),
            "selected_key_fingerprint":manifest["selected_key_fingerprint"],"parent_key_fingerprint":manifest["parent_key_fingerprint"]},
        "prototype_bank":{"path":str(Path(a.prototype_bank).resolve()),"requested_k":bank.requested_k,"fingerprint":bank.fingerprint},
        "coverage_radius_m":a.coverage_radius_m,"metrics":reports,
        "headroom":{"ub1_exact_delta_mIoU_pp":ub1,"ub2_deployable_delta_mIoU_pp":ub2,"ub2_retention_of_ub1_mIoU":retention},
        "coverage":{"eligible_targets":eligible,"covered_targets":covered,"component_coverage":compcov,"voxel_coverage":voxcov,
            "historical_matches":int(cov["hist"]),"frontier_matches":int(cov["front"]),"uncovered_targets":eligible-covered,
            "mean_legal_candidates_per_positive":meanc,"max_legal_candidates_per_positive":int(cov["maxcand"]),
            "duplicate_target_assignment":int(cov["dup_t"]),"duplicate_anchor_assignment":int(cov["dup_a"])},
        "target_mass":{"all_v21_targets":alltargets,"moving_eligible_v21_targets":movingtargets,
            "all_v21_target_voxels_report_horizons":allvox,"moving_eligible_target_voxels_inside_frozen_support":movingvox,
            "per_class_targets":dict(perclass),"per_report_horizon_target_components":dict(perh)},
        "target_audit":dict(audit),"anchor_audit":dict(aa),"shape_audit":dict(sa),
        "collision_audit":{k:dict(v) for k,v in ca.items()},
        "dormant_age_stratum_ub1_exact":{age:{"metrics":m.compute(),"delta_vs_v18_pp":delta(m.compute(),br)} for age,m in ages.items()},
        "scene_delta_mIoU_pp":{"median":float(np.median(sd)) if sd else float("nan"),"min":float(np.min(sd)) if sd else float("nan"),"max":float(np.max(sd)) if sd else float("nan")},
        "stage0b_gate":gate,
        "contracts":{"report_horizons_s":list(REPORT_H),"all_state_horizons_s":list(ALL_HORIZONS_S),
            "target":"DORMANT_ANCESTRAL + BIRTH, report-horizon eligible","compositor":COMPOSITOR_PROTOCOL,
            "moving_metric":"frozen Moving-mIoU v2; unchanged","v20_completion_used":False,"transformer_used":False},
        "elapsed_seconds":time.perf_counter()-started}
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({"output":str(Path(a.output).resolve()),"windows":len(records),"UB0_dmIoU":reports["UB0_EXACT"]["delta_vs_v18_pp"]["mIoU"],
                      "UB1_dmIoU":ub1,"UB2_dmIoU":ub2,"coverage":compcov,"mean_candidates":meanc,"gate_pass":gate["pass"]},indent=2))
    if a.enforce_stage0b_gate and not gate["pass"]:raise SystemExit(2)

if __name__=="__main__":main()
