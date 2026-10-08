#!/usr/bin/env python3
"""Read-only root-cause diagnosis for static class 11/13 (road/sidewalk).

Keep frozen V18 transport and frozen B (raw weighted ADD@.5, REMOVE off)
unchanged. One forward per head; compare causal candidate support, threshold
misses, and final writes versus matched old Local (ADD .5/.5, REMOVE off).

The only GT use is post-prediction scoring. No threshold search, training,
new model, FPS claim, checkpoint write, or full4369 repetition.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from real_motion.canonical_causal_repair import compose_canonical
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
from real_motion.nuscenes_adapter import NuScenesWindowSource
from real_motion.runtime_config import add_config_args,load_runtime_config,make_prepare_config
from real_motion.v21_source_induction import stable_json_fingerprint
from tools.real_motion import causal_column_common as columns
from tools.real_motion.ccr_screen_common import build_inputs,map_inputs,old_execution
from tools.real_motion.eval_p0_f9_v18_se2 import load_cache
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import (
    align_records,load_manifest,sha256,
)
from tools.real_motion.point_ccr_v18_fps_common import load_point_head
from tools.real_motion.run_p0_f9_shared_evidence_pilot import PilotProvider,require_cuda
from tools.real_motion.static_evidence_selector_common import CLEAN_SHA256,write_json
from tools.real_motion.train_p0_f9_causal_columns import record_keys
from tools.real_motion.train_p0_f9_joint_causal_columns import load_joint
from tools.real_motion.validate_p0_f9_ccr_frozen_b_expanded import (
    _attach_old_local_fixed_geometry,
)

PROTOCOL="p0_f9_ccr_static_11_13_gap_diagnostic_v1"
CLASSES=(11,13)
HORIZONS={1:"1.0s",3:"2.0s",5:"3.0s"}
FREE=17
PREDICTIONS=("baseline","B_joint","B_static_only","Old_full_REMOVE_off","Old_static_only")


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


def _summary(r):
    lines=["===== CCR STATIC SURFACE GAP =====","status="+r["status"]]
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
                f'B_static_FP_opposite={item["B_static_FP_opposite_surface"]}')
    if "error" in r:lines.append("error="+r["error"])
    lines.append("Read-only diagnostic on a known development subset, not new test evidence.")
    return "\n".join(lines)+"\n"


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);add_config_args(p)
    for key in ("checkpoint","ccr-checkpoint","base-checkpoint","dev-cache",
                "population-manifest","dataroot","dev-info","out-dir"):
        p.add_argument("--"+key,required=True)
    p.add_argument("--device",default="cuda")
    p.add_argument("--windows",type=int,default=64,choices=(64,512))
    p.add_argument("--cpu-workers",type=int,default=10)
    p.add_argument("--ccr-cpu-execution",choices=("numpy","native","native_parallel"),default="native_parallel")
    p.add_argument("--ccr-cpu-workers",type=int,default=4)
    p.add_argument("--ccr-val-history-cache",required=True)
    p.add_argument("--ccr-val-history-cache-ram-mib",type=int,default=512)
    p.add_argument("--old-local-batch-size",type=int,default=256)
    a=p.parse_args(argv)

    out=Path(a.out_dir)
    if out.exists():p.error("fresh output required")
    for key in ("config","checkpoint","ccr_checkpoint","base_checkpoint","dev_cache",
                "population_manifest","dev_info"):
        if not Path(getattr(a,key) or "").is_file():p.error("missing "+key)
    if not Path(a.dataroot).is_dir() or not 1<=a.cpu_workers<=16 or not 1<=a.ccr_cpu_workers<=8:
        p.error("invalid paths/worker limits")
    device=require_cuda(a.device);torch.set_num_threads(1)
    out.mkdir(parents=True);started=time.perf_counter()
    result=dict(status="running",protocol=PROTOCOL,read_only=True,training=False,
                threshold_search=False,full4369_rerun=False,
                B=dict(ADD="raw weighted sigmoid@0.5",REMOVE=False),
                Old_Local=dict(GEN_ADD=.5,REFINE_ADD=.5,REMOVE=False))
    def persist():
        write_json(out/"static_gap.json",result)
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
        keys=keys64 if a.windows==64 else dev512
        _,all_dev=load_cache(a.dev_cache);record_keys(all_dev)
        records=align_records(all_dev,keys);del all_dev
        result["population"]=dict(
            name="DEV64" if a.windows==64 else "DEV512",
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

        rows={(cid,h):_make_counts() for cid in CLASSES for h in HORIZONS}
        with old_execution(teacher,provider):
            for wi,(record,raw) in enumerate(prefetch_raw_columns(provider,source,records),1):
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

                # Build old Local's history/ego-only frontier and read out only
                # 1/2/3s. GT is not fed to candidate generation or prediction.
                _attach_old_local_fixed_geometry(prep,provider,teacher)
                for h in HORIZONS:
                    baseline=np.asarray(prep.baseline[h]).ravel()
                    gt=np.asarray(raw["future_gt_occ"][h]).ravel()
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

                if wi==1 or wi%8==0 or wi==len(records):
                    print(f"CCR_STATIC_GAP {wi}/{len(records)}",flush=True)

        result["report"]={
            ("driveable_surface" if cid==11 else "sidewalk"):{
                label:_finish(rows[(cid,h)]) for h,label in HORIZONS.items()
            } for cid in CLASSES
        }
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
