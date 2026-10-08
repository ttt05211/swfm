#!/usr/bin/env python3
"""One-shot static-surface repair screen: diagnosis + fixed policy ablation.

Run all predeclared static alternatives on the same DEV512 windows in one
pass, with resumable small CPU-only accumulators. No model retraining or
hyperparameter search. Dynamic CCR, V18 transport and original frozen B
remain unchanged. Only GT-independent static ADD decisions are varied.

GT is used only for scoring, never evidence, candidate selection or decisions.
This is a development-set screen; no new independent validation or official
candidate acceptance is claimed. Formal FPS remains the frozen B measurement.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from real_motion.canonical_causal_repair import compose_canonical
from real_motion.ccr_static_frozen_policies import (
    POLICIES, direct_priority_plan, static_policy_score, best_safe_variant,
)
from real_motion.canonical_repair_execution import CanonicalCpuExecution
from real_motion.causal_geometry_cache import CausalGeometryCache
from real_motion.ccr_frozen_b import frozen_b_probabilities
from real_motion.ccr_val_history_cache import (
    namespace as val_history_cache_namespace,
    validate_manifest as validate_val_history_cache_manifest,
)
from real_motion.causal_column_completion import (
    ADD, GENERATE, REFINE, actions_from_probabilities, compose_dense,
)
from real_motion.column_runtime_pipeline import CachedColumnSource, prefetch_raw_columns
from real_motion.nuscenes_adapter import NuScenesWindowSource, gt_moving_support_sequence
from real_motion.runtime_config import add_config_args,load_runtime_config,make_prepare_config
from real_motion.v21_source_induction import stable_json_fingerprint
from tools.real_motion import causal_column_common as columns
from tools.real_motion.ccr_screen_common import build_inputs,map_inputs,old_execution
from tools.real_motion.eval_p0_f9_v18_se2 import load_cache
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import (
    Metrics,align_records,load_manifest,sha256,
)
from tools.real_motion.shared_evidence_pilot_common import moving_support_masks
from tools.real_motion.point_ccr_v18_fps_common import load_point_head
from tools.real_motion.run_p0_f9_shared_evidence_pilot import PilotProvider,require_cuda
from tools.real_motion.static_evidence_selector_common import CLEAN_SHA256,write_json
from tools.real_motion.train_p0_f9_causal_columns import record_keys
from tools.real_motion.train_p0_f9_joint_causal_columns import load_joint
from tools.real_motion.validate_p0_f9_ccr_frozen_b_expanded import (
    _attach_old_local_fixed_geometry,
)

PROTOCOL="p0_f9_ccr_static_one_shot_fixed_policies_v1"
CLASSES=(11,13)
HORIZONS={1:"1.0s",3:"2.0s",5:"3.0s"}
FREE=17
PREDICTIONS=("baseline","B_joint","B_static_only","Old_full_REMOVE_off","Old_static_only")
VARIANTS=("baseline","Old_Local_REMOVE_off",*POLICIES)
COHORTS=("DEV512","DEV64","outside_DEV64_448")


def _make_counts():
    return defaultdict(int)


def _flatten_support(flat,legal,cls_mask,volume):
    valid=cls_mask[:,None] & legal
    ids=np.asarray(flat[valid],np.int64)
    result=np.zeros(volume,bool)
    if len(ids):
        if ids.min()<0 or ids.max()>=volume:
            raise RuntimeError("illegal static support flat index")
        result[ids]=True
    return result


def _old_class_support(plan,cid,kind,volume):
    row=(plan.classes==cid)&(plan.kind==kind)&(
        (plan.actor==-3) if kind==GENERATE else (plan.actor==-2))
    return _flatten_support(plan.flat,plan.legal[...,ADD],row,volume)


def _count_where(out,name,mask):
    out[name]+=int(np.count_nonzero(mask))



def _static_surface_conflicts(evidence,plan,h,baseline):
    """Dense unique-voxel conflict types, GT-independent.

    A source is direct if its canonical lattice cell had occupancy at >=1
    historical time (presence.any); pure face-halo has zero history presence.
    Collision is tested in the PROJECTED future raster, not t0 lattice.
    Only V18-free voxels can be added. Classes with both direct and halo at
    one target are treated as direct (strongest support for that class).
    """
    base=np.asarray(baseline).ravel();volume=base.size
    if volume==0:raise RuntimeError("empty forecast grid")
    flat=np.asarray(plan.flat[:,h],np.int64)
    valid=(flat>=0)&(flat<volume)
    direct=np.asarray(evidence.presence,bool).any(axis=1)
    static=np.asarray(evidence.actor)==-2
    classes=np.asarray(evidence.classes)
    group={}
    for cid in (11,13):
        for flag,label in ((True,"direct"),(False,"halo")):
            mask=static&(classes==cid)&(direct==flag)&valid
            dense=np.zeros(volume,bool)
            dense[flat[mask]]=True
            group[cid,label]=dense
    d11,h11=group[11,"direct"],group[11,"halo"]
    d13,h13=group[13,"direct"],group[13,"halo"]
    conflict=(d11|h11)&(d13|h13)&(base==FREE)
    typed={
        "road_direct_sidewalk_pure_halo":conflict&d11&~d13,
        "sidewalk_direct_road_pure_halo":conflict&d13&~d11,
        "both_direct":conflict&d11&d13,
        "both_pure_halo":conflict&~d11&~d13,
    }
    if not np.array_equal(
        np.logical_or.reduce(tuple(typed.values())),conflict):
        raise RuntimeError("unpartitioned surface-class conflict")
    # In the frozen planner all projected colliding surface rows MUST be
    # illegal for ADD. A failure means this audit is not seeing the same mask.
    colliding=valid&static&conflict[flat.clip(0,volume-1)]
    if np.any(plan.legal[colliding,h,0]):
        raise RuntimeError("surface conflict and planner legality disagree")
    typed["any"]=conflict
    return typed


def _count_surface_conflicts(count,base,gt,b,old_static,cid,conflicts):
    """How many OLD-successful, B-missed GT voxels were conflict blocked?"""
    need=(base==FREE)&(gt==cid)
    win=need&(old_static==cid)&(b!=cid)
    categories=(
        "any","road_direct_sidewalk_pure_halo",
        "sidewalk_direct_road_pure_halo","both_direct","both_pure_halo",
    )
    for name in categories:
        conflict=conflicts[name]
        _count_where(count,"blocked_"+name+"_all",conflict)
        _count_where(count,"blocked_"+name+"_GT",need&conflict)
        _count_where(count,"Old_static_win_blocked_"+name,win&conflict)
    preferred=("road_direct_sidewalk_pure_halo" if cid==11 else
               "sidewalk_direct_road_pure_halo")
    _count_where(count,"own_real_vs_other_halo_GT",need&conflicts[preferred])
    _count_where(count,"Old_static_win_own_real_vs_other_halo",win&conflicts[preferred])


def _static_diagnostic(count,base,gt,b,b_static,old,old_static,ccr,old_gen,old_refine,cid,ccr_projected=None):
    """Unique dense-voxel accounting, not duplicated candidate-query counts."""
    volume=base.size
    if any(x.size!=volume for x in (gt,b,b_static,old,old_static,ccr,old_gen,old_refine)):
        raise RuntimeError("static diagnosis volume mismatch")
    need=(base==FREE)&(gt==cid)
    old_any=old_gen|old_refine
    _count_where(count,"GT_missing_on_V18_free",need)
    _count_where(count,"CCR_support_GT",need&ccr)
    if ccr_projected is not None:
        _count_where(count,"CCR_projected_GT",need&ccr_projected)
        _count_where(count,"CCR_projected_but_illegal_GT",need&ccr_projected&~ccr)
    _count_where(count,"CCR_no_support_GT",need&~ccr)
    _count_where(count,"Old_GEN_support_GT",need&old_gen)
    _count_where(count,"Old_REFINE_support_GT",need&old_refine)
    _count_where(count,"Old_union_support_GT",need&old_any)
    _count_where(count,"Old_only_support_GT",need&old_any&~ccr)
    _count_where(count,"CCR_only_support_GT",need&ccr&~old_any)
    _count_where(count,"Neither_support_GT",need&~ccr&~old_any)

    _count_where(count,"B_static_correct_on_missing",need&(b_static==cid))
    _count_where(count,"B_joint_correct_on_missing",need&(b==cid))
    _count_where(count,"Old_static_correct_on_missing",need&(old_static==cid))
    _count_where(count,"Old_full_correct_on_missing",need&(old==cid))

    # Where the actual Old static branch succeeds but frozen B joint misses:
    # distinguish impossible-under-CCR-support from reachable-but-missed.
    win=need&(old_static==cid)&(b!=cid)
    _count_where(count,"Old_static_win_over_B_joint",win)
    _count_where(count,"Old_static_win_CCR_no_support",win&~ccr)
    _count_where(count,"Old_static_win_CCR_has_support",win&ccr)
    _count_where(count,"Old_static_win_GEN_support",win&old_gen)
    _count_where(count,"Old_static_win_REFINE_support",win&old_refine)
    _count_where(count,"Old_static_win_CCR_wrote_but_dynamic_overrode",win&ccr&(b_static==cid))
    _count_where(count,"Old_static_win_CCR_reachable_not_written",win&ccr&(b_static!=cid))

    # Errors committed by B's STATIC writer itself. GT categories are disjoint.
    b_add=(base==FREE)&(b_static==cid)
    _count_where(count,"B_static_added",b_add)
    _count_where(count,"B_static_added_correct",b_add&(gt==cid))
    _count_where(count,"B_static_FP_free",b_add&(gt==FREE))
    _count_where(count,"B_static_FP_opposite_surface",b_add&(gt==(13 if cid==11 else 11)))
    _count_where(count,"B_static_FP_other_occupied",b_add&(gt!=FREE)&(gt!=cid)&(gt!=(13 if cid==11 else 11)))
    _count_where(count,"B_static_missed_GT_on_support",need&ccr&(b_static!=cid))
    _count_where(count,"B_static_missed_GT_outside_support",need&~ccr)

    # Final IoU counts. These are computed using full scene voxels and the
    # official per-horizon intersection/union definition for each static class.
    for name,pred in (
        ("baseline",base),("B_joint",b),("B_static_only",b_static),
        ("Old_full_REMOVE_off",old),("Old_static_only",old_static),
    ):
        pp=pred==cid;gg=gt==cid
        _count_where(count,name+"_inter",pp&gg)
        _count_where(count,name+"_union",pp|gg)


def _finish(raw):
    out=dict(sorted(raw.items()))
    need=out.get("GT_missing_on_V18_free",0)
    wins=out.get("Old_static_win_over_B_joint",0)
    out["Old_wins_blocked_any_share"]=(
        out.get("Old_static_win_blocked_any",0)/wins if wins else None)
    out["Old_wins_own_real_vs_other_halo_share"]=(
        out.get("Old_static_win_own_real_vs_other_halo",0)/wins if wins else None)
    out["GT_missing_CCR_support_recall"]=(
        out.get("CCR_support_GT",0)/need if need else None)
    out["GT_missing_Old_support_recall"]=(
        out.get("Old_union_support_GT",0)/need if need else None)
    out["Old_static_wins_unreachable_CCR_share"]=(
        out.get("Old_static_win_CCR_no_support",0)/wins if wins else None)
    out["Old_static_wins_reachable_CCR_share"]=(
        out.get("Old_static_win_CCR_has_support",0)/wins if wins else None)
    for name in PREDICTIONS:
        i=out.get(name+"_inter",0);u=out.get(name+"_union",0)
        out[name+"_IoU_pct"]=100*i/u if u else None
    b=out["B_joint_IoU_pct"];old=out["Old_full_REMOVE_off_IoU_pct"]
    out["B_minus_Old_IoU_pp"]=b-old if b is not None and old is not None else None
    return out



def _cohort():
    return dict(windows=0,metrics={name:Metrics() for name in VARIANTS})


def _analyze_unique_fp(count,baseline,gt,b,old,b_static,plan,evidence,score,h,cid):
    """Paired B/Old final semantic FP/TP with direct/halo attribution.

    Only actual final B/Old class predictions are counted for paired errors.
    Provenance is class-specific B static accepted ADD evidence; mixed support
    reports direct if at least one real accepted source contributed.
    """
    base=np.asarray(baseline).ravel()
    g=np.asarray(gt).ravel();bp=np.asarray(b).ravel()
    op=np.asarray(old).ravel();bs=np.asarray(b_static).ravel()
    changed=(base==FREE)
    b_only_fp=(bp==cid)&(g!=cid)&(op!=cid)
    old_only_fp=(op==cid)&(g!=cid)&(bp!=cid)
    b_only_tp=(bp==cid)&(g==cid)&(op!=cid)
    old_only_tp=(op==cid)&(g==cid)&(bp!=cid)
    for name,mask in (
        ("B_only_FP",b_only_fp),("Old_only_FP",old_only_fp),
        ("B_only_TP",b_only_tp),("Old_only_TP",old_only_tp),
        ("B_only_FP_GT_free",b_only_fp&(g==FREE)),
        ("B_only_FP_GT_opposite",b_only_fp&(g==(13 if cid==11 else 11))),
        ("B_only_FP_GT_other_occupied",b_only_fp&(g!=FREE)&(g!=cid)&(g!=(13 if cid==11 else 11))),
    ):
        _count_where(count,name,mask)
    # Since REMOVE is off and V18 occupancy is immutable, a final newly
    # predicted road/sidewalk label must have originated from an ADD.
    if np.any(b_only_fp&~changed):
        raise RuntimeError("B-only static FP changed original V18 occupancy")
    active=(np.asarray(evidence.actor)==-2)&(np.asarray(evidence.classes)==cid)
    active=active&plan.legal[:,h,0]&(score[:,h,0]>=0.5)
    flat=np.asarray(plan.flat[:,h],np.int64)
    direct=np.asarray(evidence.presence,bool).any(1)
    volume=len(base)
    direct_dest=np.zeros(volume,bool)
    halo_dest=np.zeros(volume,bool)
    direct_dest[flat[active&direct]]=True
    halo_dest[flat[active&~direct]]=True
    b_static_fp=b_only_fp&(bs==cid)
    _count_where(count,"B_only_FP_static_actual",b_static_fp)
    _count_where(count,"B_only_FP_static_with_direct",b_static_fp&direct_dest)
    _count_where(count,"B_only_FP_static_halo_only",b_static_fp&~direct_dest&halo_dest)
    if np.any(b_static_fp&~(direct_dest|halo_dest)):
        raise RuntimeError("static B FP cannot be traced to accepted evidence")


def _metric_report(cohorts):
    computed={}
    for name,cohort in cohorts.items():
        if cohort["windows"]==0:continue
        metrics={key:value.compute() for key,value in cohort["metrics"].items()}
        b=metrics["frozen_B"]
        variants={}
        for key,item in metrics.items():
            variants[key]=dict(
                IoU=item["IoU"],mIoU=item["mIoU"],
                MovingMacro=item["MovingMacro"],MovingMicro=item["MovingMicro"],
                delta_mIoU_vs_B=item["mIoU"]-b["mIoU"],
                delta_MovingMicro_vs_B=item["MovingMicro"]-b["MovingMicro"],
                per_horizon={
                    h:dict(
                        mIoU=m["mIoU"],MovingMicro=m["MovingMicro"],
                        road_iou=m["semantic_per_class"]["11"],
                        sidewalk_iou=m["semantic_per_class"]["13"],
                        delta_road_vs_B=(m["semantic_per_class"]["11"]
                                         -b["per_horizon"][h]["semantic_per_class"]["11"]),
                        delta_sidewalk_vs_B=(m["semantic_per_class"]["13"]
                                             -b["per_horizon"][h]["semantic_per_class"]["13"]),
                    ) for h,m in item["per_horizon"].items()
                })
        best=best_safe_variant({k:v for k,v in metrics.items() if k in POLICIES})
        computed[name]=dict(
            windows=cohort["windows"],variants=variants,
            strict_nonregression_policy=best,
            note="Descriptive same-DEV comparison, not independent acceptance.")
    return computed


def _persist_state(path,index,contract,rows,cohorts):
    state=dict(protocol=PROTOCOL,index=index,contract=contract,rows=rows,cohorts=cohorts)
    tmp=path.with_name(path.name+".tmp")
    torch.save(state,tmp)
    os.replace(tmp,path)


def _recover_state(path,contract):
    # Only load a state file created by this local evaluator and owned by
    # the user. torch serialization is not safe for untrusted inputs.
    state=torch.load(path,map_location="cpu",weights_only=False)
    if state.get("protocol")!=PROTOCOL or state.get("contract")!=contract:
        raise RuntimeError("resume state does not match frozen run/keys/checkpoints")
    return state

def _summary(r):
    lines=["===== ONE-SHOT STATIC CCR REPAIR SCREEN =====","status="+r["status"]]
    for cohort,items in r.get("policy_screen",{}).items():
        lines.append("--- "+cohort+f' ({items["windows"]} windows) ---')
        for name,metric in items["variants"].items():
            lines.append(
                f'{name:40s} mIoU={metric["mIoU"]:.4f} '
                f'dmIoU_vs_B={metric["delta_mIoU_vs_B"]:+.4f} '
                f'Moving={metric["MovingMicro"]:.4f} '
                f'dMoving_vs_B={metric["delta_MovingMicro_vs_B"]:+.4f} '
                f'road_123={[round(h["road_iou"],3) for h in metric["per_horizon"].values()]} '
                f'sidewalk_123={[round(h["sidewalk_iou"],3) for h in metric["per_horizon"].values()]}')
        lines.append("strict_nonregression_policy="+items["strict_nonregression_policy"])

    for label,rr in r.get("report",{}).items():
        lines.append("CLASS "+label)
        for h,item in rr.items():
            lines.append(
                f'{h}: B-Old={item["B_minus_Old_IoU_pp"]:+.3f} pp '
                f'missing={item["GT_missing_on_V18_free"]} '
                f'CCR_support={item["GT_missing_CCR_support_recall"]} '
                f'Old_support={item["GT_missing_Old_support_recall"]} '
                f'old_static_wins={item["Old_static_win_over_B_joint"]} '
                f'old_wins_CCR_no_support={item["Old_static_win_CCR_no_support"]} '
                f'old_wins_CCR_reachable_not_written={item["Old_static_win_CCR_reachable_not_written"]} '
                f'B_static_FP_free={item["B_static_FP_free"]} '
                f'B_static_FP_opposite={item["B_static_FP_opposite_surface"]} '
                f'blocked_conflict_GT={item["blocked_any_GT"]} '
                f'blocked_own_real_vs_halo_GT={item["own_real_vs_other_halo_GT"]} '
                f'old_wins_blocked_conflict={item["Old_static_win_blocked_any"]} '
                f'old_wins_own_real_vs_halo={item["Old_static_win_own_real_vs_other_halo"]} '
                f'blocked_both_direct_GT={item["blocked_both_direct_GT"]} '
                f'blocked_both_halo_GT={item["blocked_both_pure_halo_GT"]} '
                f'B_only_TP={item["B_only_TP"]} '
                f'Old_only_TP={item["Old_only_TP"]} '
                f'B_only_FP={item["B_only_FP"]} '
                f'Old_only_FP={item["Old_only_FP"]} '
                f'B_only_FP_GT_free={item["B_only_FP_GT_free"]} '
                f'B_only_FP_direct={item["B_only_FP_static_with_direct"]} '
                f'B_only_FP_halo_only={item["B_only_FP_static_halo_only"]}')
    if "error" in r:lines.append("error="+r["error"])
    if r.get("candidate_interpretation"):
        lines.append("descriptive_candidate="+str(r["candidate_interpretation"]["descriptive_static_candidate"]))
        lines.append("accepted_for_deployment=False")
    lines.append("All policies are static-only and predefined; immutable V18 and dynamic CCR.")
    lines.append("Known DEV population: exploratory diagnostics, NOT new independent test or FPS.")
    return "\n".join(lines)+"\n"


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);add_config_args(p)
    for key in ("checkpoint","ccr-checkpoint","base-checkpoint","dev-cache",
                "population-manifest","dataroot","dev-info","out-dir"):
        p.add_argument("--"+key,required=True)
    p.add_argument("--device",default="cuda")
    p.add_argument("--windows",type=int,default=512,choices=(512,))
    p.add_argument("--resume",action="store_true",help="resume the same immutable DEV512 screen from its checkpoint")
    p.add_argument("--cpu-workers",type=int,default=10)
    p.add_argument("--ccr-cpu-execution",choices=("numpy","native","native_parallel"),default="native_parallel")
    p.add_argument("--ccr-cpu-workers",type=int,default=4)
    p.add_argument("--ccr-val-history-cache",required=True)
    p.add_argument("--ccr-val-history-cache-ram-mib",type=int,default=512)
    p.add_argument("--old-local-batch-size",type=int,default=256)
    a=p.parse_args(argv)

    out=Path(a.out_dir)
    if a.resume:
        if not (out/"progress_state.pt").is_file():
            p.error("--resume requires an existing progress_state.pt in out-dir")
    elif out.exists():
        p.error("fresh output required unless --resume is specified")
    for key in ("config","checkpoint","ccr_checkpoint","base_checkpoint","dev_cache",
                "population_manifest","dev_info"):
        if not Path(getattr(a,key) or "").is_file():p.error("missing "+key)
    if not Path(a.dataroot).is_dir() or not 1<=a.cpu_workers<=16 or not 1<=a.ccr_cpu_workers<=8:
        p.error("invalid paths/worker limits")
    device=require_cuda(a.device);torch.set_num_threads(1)
    out.mkdir(parents=True,exist_ok=a.resume);started=time.perf_counter()
    result=dict(status="running",protocol=PROTOCOL,read_only=True,training=False,
                threshold_search=False,full4369_rerun=False,
                frozen_weight_only=True,immutable_dynamic=True,
                static_policies=list(POLICIES),
                B=dict(ADD="raw weighted sigmoid@0.5",REMOVE=False),
                Old_Local=dict(GEN_ADD=.5,REFINE_ADD=.5,REMOVE=False))
    def persist():
        write_json(out/"static_one_shot.json",result)
        (out/"summary.txt").write_text(_summary(result),encoding="utf-8")
    persist()

    execution=val_cache=None
    try:
        cfg=load_runtime_config(a.config,a.override);config_fp=stable_json_fingerprint(cfg)
        epoch19_sha=sha256(a.checkpoint)
        if sha256(a.base_checkpoint)!=CLEAN_SHA256:
            raise RuntimeError("Clean-E14 checkpoint fingerprint mismatch")
        ck,teacher=load_joint(
            a.checkpoint,device,reference_sha=CLEAN_SHA256,config_sha=config_fp,
            allow_diagnostic=True)
        if (teacher.transport.config.history_frames!=4 or ck.get("cursor_epoch")!=19
                or ck["model_configs"].get("adaptive_context") is not None):
            raise RuntimeError("frozen epoch19 strict-four Local required")
        teacher.eval().requires_grad_(False)
        for path,expected in ((a.dev_cache,ck["cache_fingerprints"]["dev"]),
                              (a.dev_info,ck["info_fingerprints"]["dev"])):
            if sha256(path)!=expected:
                raise RuntimeError("epoch19 dev data fingerprint mismatch")

        saved=torch.load(a.ccr_checkpoint,map_location="cpu",weights_only=False)
        head=load_point_head(
            saved,teacher_sha256=epoch19_sha,config_fingerprint=config_fp,
            source_dim=teacher.columns.source_dim,device=device,
            allow_completed_epoch_boundary=True)
        head.eval().requires_grad_(False)
        result["checkpoints"]=dict(
            epoch19_sha256=epoch19_sha,CCR_sha256=sha256(a.ccr_checkpoint),
            CCR_epoch=int(saved.get("epoch",0)))

        manifest,keys64,_=load_manifest(a.population_manifest)
        dev512=tuple(map(tuple,manifest["parent_keys"]))
        if (len(keys64)!=64 or len(dev512)!=512
                or manifest["manifest_fingerprint"]!=ck["dev_manifest_fingerprint"]
                or dev512!=tuple(map(tuple,ck["dev_keys"]))
                or saved["contract"]["dev_manifest_fingerprint"]!=manifest["manifest_fingerprint"]):
            raise RuntimeError("frozen DEV population mismatch")
        keys=dev512
        _,all_dev=load_cache(a.dev_cache);record_keys(all_dev)
        records=align_records(all_dev,keys);del all_dev
        result["population"]=dict(
            name="DEV512",
            windows=len(records),key_fingerprint=stable_json_fingerprint([list(x) for x in keys]),
            note="Diagnostic only; overlaps previously studied validation population.")

        provider=PilotProvider(
            a.base_checkpoint,CLEAN_SHA256,make_prepare_config(cfg),
            device,a.cpu_workers,teacher,None)
        execution=CanonicalCpuExecution(a.ccr_cpu_execution,a.ccr_cpu_workers)
        provider.ccr_execution=execution
        source=CachedColumnSource(
            NuScenesWindowSource(a.dataroot,info_pkl=a.dev_info,verbose=False),1024)
        source.copy_on_insert=False

        root=Path(__file__).resolve().parents[2]
        val_cache=CausalGeometryCache(
            a.ccr_val_history_cache,val_history_cache_namespace(provider,a,root),
            max_bytes=0,ram_bytes=a.ccr_val_history_cache_ram_mib*2**20,
            reserve_bytes=0,compression_level=6)
        validate_val_history_cache_manifest(val_cache,a)
        provider.ccr_history_cache=val_cache
        provider.ccr_history_cache_mode="require"
        provider.ccr_history_cache_source=source
        provider.ccr_verify_cached_full_evidence_remaining=1

        contract=dict(checkpoints=result["checkpoints"],population=result["population"],
                      policies=list(POLICIES))
        if a.resume:
            state=_recover_state(out/"progress_state.pt",contract)
            offset=int(state["index"])
            if not 0<=offset<len(records):
                raise RuntimeError("resume checkpoint is complete or invalid")
            rows=state["rows"]
            cohorts=state["cohorts"]
            result["resumed_at"]=offset
        else:
            offset=0
            rows={(cid,h):_make_counts() for cid in CLASSES for h in HORIZONS}
            cohorts={k:_cohort() for k in COHORTS}
        dev64_set=set(map(tuple,keys64))
        with old_execution(teacher,provider):
            for wi,(record,raw) in enumerate(
                    prefetch_raw_columns(provider,source,records[offset:]),
                    offset+1):
                output=teacher.motion(record,device)
                cached=raw.get("_column_causal_preparation")
                if cached is not None and not getattr(provider,"columns_checked",False):
                    del raw["_column_causal_preparation"]
                    try:
                        prep=provider.prepare_columns(
                            source,record,include_gt=True,raw_window=raw,outputs=output)
                    finally:
                        raw["_column_causal_preparation"]=cached
                    print("CCR_VAL_CACHE_LIVE_EXACTNESS_PREFLIGHT PASS",flush=True)
                else:
                    prep=provider.prepare_columns(
                        source,record,include_gt=True,raw_window=raw,outputs=output)

                evidence=build_inputs(provider,prep)
                plan=map_inputs(provider,evidence,prep)
                score=frozen_b_probabilities(head,evidence,plan,output,device)
                B=compose_canonical(
                    prep.baseline,evidence,plan,score[...,0],score[...,1],
                    thresholds=(.5,.95),role="all")
                B_static=compose_canonical(
                    prep.baseline,evidence,plan,score[...,0],score[...,1],
                    thresholds=(.5,.95),role="static")

                prioritized_plan,priority_counts=direct_priority_plan(
                    evidence,plan,prep.baseline)
                if np.array_equal(prioritized_plan.legal,plan.legal):
                    prioritized_score=score
                else:
                    prioritized_score=frozen_b_probabilities(
                        head,evidence,prioritized_plan,output,device)
                    dyn=np.asarray(evidence.actor)>=0
                    if not np.array_equal(prioritized_score[dyn],score[dyn]):
                        raise RuntimeError("static legality change modified dynamic scores")
                predictions={"frozen_B":B}
                for policy in POLICIES:
                    if policy=="frozen_B":continue
                    use_priority="priority" in policy
                    use_plan=prioritized_plan if use_priority else plan
                    use_score=prioritized_score if use_priority else score
                    gated=static_policy_score(use_score,evidence,head,policy)
                    predictions[policy]=compose_canonical(
                        prep.baseline,evidence,use_plan,gated[...,0],gated[...,1],
                        thresholds=(.5,.95),role="all")

                # Strong static intervention invariants on all SIX frames:
                # unchanged V18 occupied voxels + exactly identical dynamic
                # ADD outputs on every dynamically written destination.
                actors=np.asarray(evidence.actor)
                for h in range(6):
                    base=np.asarray(prep.baseline[h]).ravel()
                    dyn=(actors>=0)&plan.legal[:,h,0]&(score[:,h,0]>=.5)
                    dyn_dst=np.unique(plan.flat[dyn,h])
                    for pname,frames in predictions.items():
                        pred=np.asarray(frames[h]).ravel()
                        if not np.array_equal(pred[base!=FREE],base[base!=FREE]):
                            raise RuntimeError("static policy overwrote frozen V18 occupancy: "+pname)
                        if not np.array_equal(pred[dyn_dst],np.asarray(B[h]).ravel()[dyn_dst]):
                            raise RuntimeError("static policy modified dynamic ADD output: "+pname)

                # Build old Local's history/ego-only frontier and read out only
                # 1/2/3s. GT is not fed to candidate generation or prediction.
                _attach_old_local_fixed_geometry(prep,provider,teacher)
                moving_support=gt_moving_support_sequence(
                    source.nusc,prep.window.t0_token,prep.window.future_tokens,
                    tuple(.5*(h+1) for h in range(6)),grid=provider.pcfg.grid,
                    workers=provider.workers)
                moving=moving_support_masks(moving_support,provider.pcfg.grid.shape_hwd)
                key=(str(record["scene_name"]),str(record["t0_token"]))
                group_names=["DEV512",("DEV64" if key in dev64_set else "outside_DEV64_448")]
                for group_name in group_names:cohorts[group_name]["windows"]+=1
                for ri,h in enumerate(HORIZONS):
                    baseline=np.asarray(prep.baseline[h]).ravel()
                    gt=np.asarray(raw["future_gt_occ"][h]).ravel()
                    conflicts=_static_surface_conflicts(evidence,plan,h,baseline)
                    old_plan=columns.candidate_plan(
                        prep,h,provider.pcfg.grid,teacher.columns.config)
                    prob=columns.predict_probabilities(
                        teacher.columns,prep,h,old_plan,provider.pcfg.grid,device,
                        a.old_local_batch_size)
                    act=actions_from_probabilities(old_plan,prob,(.5,.5,None))
                    old=np.asarray(compose_dense(prep.baseline[h],old_plan,act)).ravel()
                    static_actions=act.copy()
                    static_actions[old_plan.actor>=0]=0
                    old_static=np.asarray(compose_dense(
                        prep.baseline[h],old_plan,static_actions)).ravel()

                    for group_name in group_names:
                        metric=cohorts[group_name]["metrics"]
                        metric["baseline"].update(ri,prep.baseline[h],raw["future_gt_occ"][h],moving[h])
                        metric["Old_Local_REMOVE_off"].update(
                            ri,old.reshape(raw["future_gt_occ"][h].shape),
                            raw["future_gt_occ"][h],moving[h])
                        for policy,pred_frames in predictions.items():
                            metric[policy].update(
                                ri,pred_frames[h],raw["future_gt_occ"][h],moving[h])

                    volume=gt.size
                    for cid in CLASSES:
                        static_role=(evidence.actor==-2)&(evidence.classes==cid)
                        ccr=_flatten_support(
                            plan.flat[:,h:h+1],plan.legal[:,h:h+1,0],
                            static_role,volume)
                        ccr_projected=_flatten_support(
                            plan.flat[:,h:h+1],plan.flat[:,h:h+1]>=0,
                            static_role,volume)
                        old_gen=_old_class_support(old_plan,cid,GENERATE,volume)
                        old_refine=_old_class_support(old_plan,cid,REFINE,volume)
                        _static_diagnostic(
                            rows[(cid,h)],baseline,gt,
                            np.asarray(B[h]).ravel(),np.asarray(B_static[h]).ravel(),
                            old,old_static,ccr,old_gen,old_refine,cid,
                            ccr_projected=ccr_projected)
                        _count_surface_conflicts(
                            rows[(cid,h)],baseline,gt,np.asarray(B[h]).ravel(),
                            old_static,cid,conflicts)
                        _analyze_unique_fp(
                            rows[(cid,h)],baseline,gt,np.asarray(B[h]).ravel(),
                            old,np.asarray(B_static[h]).ravel(),plan,evidence,score,h,cid)

                if wi==1 or wi%8==0 or wi==len(records):
                    print(f"STATIC_ONE_SHOT {wi}/{len(records)}",flush=True)
                if wi%8==0 or wi==len(records):
                    _persist_state(out/"progress_state.pt",wi,contract,rows,cohorts)
                    result["completed_windows"]=wi
                    persist()

        result["report"]={
            ("driveable_surface" if cid==11 else "sidewalk"):{
                label:_finish(rows[(cid,h)]) for h,label in HORIZONS.items()
            } for cid in CLASSES
        }
        result["policy_screen"]=_metric_report(cohorts)
        candidate=result["policy_screen"]["DEV512"]["strict_nonregression_policy"]
        outside=result["policy_screen"]["outside_DEV64_448"]["strict_nonregression_policy"]
        result["candidate_interpretation"]=dict(
            full_DEV512_nonregression=candidate,
            outside_DEV64_nonregression=outside,
            descriptive_static_candidate=(
                candidate if candidate!="frozen_B" and candidate==outside else "frozen_B"),
            accepted_for_deployment=False,
            note="All policies compared on previously used DEV data; repeat on independent holdout before claiming gain.")
        result.update(status="complete",seconds=time.perf_counter()-started,
                      future_GT_model_input=False)
        persist()
        print(_summary(result),flush=True)
        return 0
    except BaseException as exc:
        result.update(status="failed",error=type(exc).__name__+": "+str(exc),
                      seconds=time.perf_counter()-started)
        persist()
        raise
    finally:
        if val_cache is not None:val_cache.close()
        if execution is not None:execution.close()


if __name__=="__main__":
    raise SystemExit(main())
