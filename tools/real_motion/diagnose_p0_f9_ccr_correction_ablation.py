#!/usr/bin/env python3
"""Read-only Point-CCR correction ablation on a fixed weighted checkpoint.

A: corrected probabilities from CanonicalRepairHead.probabilities(), ADD@0.5,
   REMOVE disabled.
B: exact "no correction, raw sigmoid@0.5" decision, expressed without changing
   the model as a role-specific threshold on corrected ADD probability:
       p_corrected >= 1 / (1 + positive_weight[role, ADD]).
   REMOVE disabled.

No training, optimizer update, threshold search, oracle, old-Local inference, or
checkpoint write. Future GT is used only for metrics/action-quality accounting.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from real_motion.canonical_causal_repair import compose_canonical, repair_targets
from real_motion.canonical_repair_execution import CanonicalCpuExecution
from real_motion.causal_geometry_cache import CausalGeometryCache
from real_motion.ccr_val_history_cache import (
    namespace as val_history_cache_namespace,
    validate_manifest as validate_val_history_cache_manifest,
)
from real_motion.column_runtime_pipeline import CachedColumnSource, prefetch_raw_columns
from real_motion.nuscenes_adapter import NuScenesWindowSource, gt_moving_support_sequence
from real_motion.runtime_config import add_config_args, load_runtime_config, make_prepare_config
from real_motion.source_evidence_audit import edit_quality
from real_motion.v21_source_induction import stable_json_fingerprint
from tools.real_motion.ccr_screen_common import build_inputs, map_inputs
from tools.real_motion.eval_p0_f9_v18_se2 import load_cache
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import (
    Metrics, align_records, delta, load_manifest, sha256,
)
from tools.real_motion.pilot_p0_f9_canonical_causal_repair import probabilities
from tools.real_motion.point_ccr_v18_fps_common import load_point_head
from tools.real_motion.run_p0_f9_shared_evidence_pilot import PilotProvider, require_cuda
from tools.real_motion.shared_evidence_pilot_common import moving_support_masks
from tools.real_motion.static_evidence_selector_common import CLEAN_SHA256, write_json
from tools.real_motion.train_p0_f9_causal_columns import record_keys
from tools.real_motion.train_p0_f9_joint_causal_columns import load_joint


PROTOCOL="p0_f9_ccr_weight_correction_ablation_v1"
REPORT=(1,3,5)
VARIANTS=("A_corrected_add05","B_raw_add05")


def _safe_div(a,b):
    return None if not b else float(a)/float(b)


def _bucket():
    return dict(valid=0,target_pos=0,pred_pos=0,tp=0,fp=0,fn=0)


def _update(bucket,valid,target,pred):
    valid=np.asarray(valid,bool)
    target=np.asarray(target,bool)&valid
    pred=np.asarray(pred,bool)&valid
    bucket["valid"]+=int(valid.sum())
    bucket["target_pos"]+=int(target.sum())
    bucket["pred_pos"]+=int(pred.sum())
    bucket["tp"]+=int((target&pred).sum())
    bucket["fp"]+=int((~target&pred&valid).sum())
    bucket["fn"]+=int((target&~pred).sum())


def _finish(bucket):
    out=dict(bucket)
    out["precision"]=_safe_div(out["tp"],out["tp"]+out["fp"])
    out["recall"]=_safe_div(out["tp"],out["tp"]+out["fn"])
    p,r=out["precision"],out["recall"]
    out["f1"]=None if p is None or r is None or p+r==0 else 2*p*r/(p+r)
    out["prevalence"]=_safe_div(out["target_pos"],out["valid"])
    out["predicted_positive_rate"]=_safe_div(out["pred_pos"],out["valid"])
    return out


def _effective_thresholds(head):
    w=np.asarray(head.positive_weight.detach().cpu(),np.float64)
    if w.shape!=(2,2) or not np.isfinite(w).all() or np.any(w<1):
        raise RuntimeError(f"invalid Point-CCR positive_weight: shape={w.shape} values={w.tolist()}")
    return dict(
        weights=dict(
            static_ADD=float(w[0,0]),dynamic_ADD=float(w[1,0]),
            static_REMOVE=float(w[0,1]),dynamic_REMOVE=float(w[1,1])),
        B_equivalent_threshold_on_corrected_ADD=dict(
            static=float(1.0/(1.0+w[0,0])),
            dynamic=float(1.0/(1.0+w[1,0]))),
    )


def _summary(result):
    lines=["===== CCR CORRECTION ABLATION =====","status="+result["status"]]
    if "decision_rule" in result:
        d=result["decision_rule"]
        lines.append(
            "weights: "
            f'sADD={d["weights"]["static_ADD"]:.4f} '
            f'dADD={d["weights"]["dynamic_ADD"]:.4f}'
        )
        lines.append(
            "B exact equivalent corrected thresholds: "
            f'static={d["B_equivalent_threshold_on_corrected_ADD"]["static"]:.4f} '
            f'dynamic={d["B_equivalent_threshold_on_corrected_ADD"]["dynamic"]:.4f}'
        )
    m=result.get("metrics",{})
    if m:
        for key,label in (("baseline","BASE"),("A_corrected_add05","A"),("B_raw_add05","B")):
            x=m[key]
            lines.append(f'{label:4s} mIoU={x["mIoU"]:.4f} MovingMicro={x["MovingMicro"]:.4f}')
            if key!="baseline":
                for h in ("1.0","2.0","3.0"):
                    y=x["per_horizon"][h]
                    lines.append(
                        f'  {h}s mIoU={y["mIoU"]:.4f} MovingMicro={y["MovingMicro"]:.4f}')
        if "B_minus_A_pp" in result:
            d=result["B_minus_A_pp"]
            lines.append(
                f'B-A: mIoU={d["mIoU"]:+.4f} MovingMicro={d["MovingMicro"]:+.4f}')
    action=result.get("action_ADD",{})
    for role in ("static","dynamic"):
        a=action.get(f"A/{role}",{});b=action.get(f"B/{role}",{})
        if a and b:
            lines.append(
                f'{role} ADD A P/R={a["precision"]:.4f}/{a["recall"]:.4f} '
                f'B P/R={b["precision"]:.4f}/{b["recall"]:.4f}')
    q=result.get("quality",{})
    for key,label in (("A_corrected_add05","A"),("B_raw_add05","B")):
        if key in q:
            lines.append(
                f'{label} add={q[key].get("added",0)} '
                f'semantic_precision={q[key].get("addition_semantic_precision")}')
    lines.append("No threshold search; B is algebraically fixed by checkpoint positive_weight.")
    lines.append("full details: correction_ablation.json")
    return "\n".join(lines)+"\n"


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);add_config_args(p)
    for key in ("checkpoint","ccr-checkpoint","base-checkpoint","dev-cache",
                "population-manifest","dataroot","dev-info","out-dir"):
        p.add_argument("--"+key,required=True)
    p.add_argument("--device",default="cuda")
    p.add_argument("--windows",type=int,default=512,choices=(64,512))
    p.add_argument("--cpu-workers",type=int,default=10)
    p.add_argument("--ccr-cpu-execution",choices=("numpy","native","native_parallel"),default="native_parallel")
    p.add_argument("--ccr-cpu-workers",type=int,default=4)
    p.add_argument("--ccr-val-history-cache",required=True)
    p.add_argument("--ccr-val-history-cache-ram-mib",type=int,default=512)
    a=p.parse_args(argv)

    out=Path(a.out_dir)
    if out.exists():p.error("fresh output required")
    for key in ("config","checkpoint","ccr_checkpoint","base_checkpoint","dev_cache",
                "population_manifest","dev_info"):
        if not Path(getattr(a,key) or "").is_file():p.error("missing "+key)
    if (not Path(a.dataroot).is_dir() or not 1<=a.cpu_workers<=16
            or not 1<=a.ccr_cpu_workers<=8
            or not 0<=a.ccr_val_history_cache_ram_mib<=16384):
        p.error("invalid paths/workers")

    device=require_cuda(a.device);torch.set_num_threads(1)
    out.mkdir(parents=True);started=time.perf_counter()
    result=dict(status="running",protocol=PROTOCOL,read_only=True,training=False,
                threshold_search=False,remove_enabled=False)
    def persist():
        write_json(out/"correction_ablation.json",result)
        (out/"summary.txt").write_text(_summary(result),encoding="utf-8")
    persist()

    execution=val_cache=None
    try:
        cfg=load_runtime_config(a.config,a.override)
        config_fp=stable_json_fingerprint(cfg)
        epoch19_sha=sha256(a.checkpoint)
        if sha256(a.base_checkpoint)!=CLEAN_SHA256:
            raise RuntimeError("Clean-E14 base checkpoint fingerprint mismatch")
        ck,teacher=load_joint(
            a.checkpoint,device,reference_sha=CLEAN_SHA256,
            config_sha=config_fp,allow_diagnostic=True)
        if (teacher.transport.config.history_frames!=4 or ck.get("cursor_epoch")!=19
                or ck["model_configs"].get("adaptive_context") is not None):
            raise RuntimeError("selected four-history epoch19 Local checkpoint required")
        teacher.eval().requires_grad_(False)
        for path,expected in ((a.dev_cache,ck["cache_fingerprints"]["dev"]),
                              (a.dev_info,ck["info_fingerprints"]["dev"])):
            if sha256(path)!=expected:
                raise RuntimeError("epoch19/data provenance mismatch: "+path)

        saved=torch.load(a.ccr_checkpoint,map_location="cpu",weights_only=False)
        head=load_point_head(
            saved,teacher_sha256=epoch19_sha,config_fingerprint=config_fp,
            source_dim=teacher.columns.source_dim,device=device,
            allow_completed_epoch_boundary=True)
        head.eval().requires_grad_(False)
        result["decision_rule"]=_effective_thresholds(head)
        if max(result["decision_rule"]["weights"]["static_ADD"],
               result["decision_rule"]["weights"]["dynamic_ADD"])<=1.000001:
            raise RuntimeError("checkpoint is not a positive-weighted ADD checkpoint; use the pre-natural-BCE weighted checkpoint")
        result["checkpoint"]=dict(
            path=str(Path(a.ccr_checkpoint).resolve()),sha256=sha256(a.ccr_checkpoint),
            epoch=int(saved.get("epoch",0)),updates=int(saved.get("updates",0)))

        manifest,keys64,_=load_manifest(a.population_manifest)
        parent=tuple(map(tuple,manifest["parent_keys"]))
        if (len(keys64)!=64 or len(parent)!=512
                or manifest["manifest_fingerprint"]!=ck["dev_manifest_fingerprint"]
                or parent!=tuple(map(tuple,ck["dev_keys"]))
                or saved["contract"]["dev_manifest_fingerprint"]!=manifest["manifest_fingerprint"]):
            raise RuntimeError("frozen dev manifest mismatch")
        keys=keys64 if a.windows==64 else parent
        _,all_dev=load_cache(a.dev_cache);record_keys(all_dev)
        records=align_records(all_dev,keys);del all_dev

        provider=PilotProvider(
            a.base_checkpoint,CLEAN_SHA256,make_prepare_config(cfg),
            device,a.cpu_workers,teacher,None)
        execution=CanonicalCpuExecution(a.ccr_cpu_execution,a.ccr_cpu_workers)
        provider.ccr_execution=execution
        source=CachedColumnSource(
            NuScenesWindowSource(a.dataroot,info_pkl=a.dev_info,verbose=False),1024)
        source.copy_on_insert=False

        root=Path(__file__).resolve().parents[2]
        namespace=val_history_cache_namespace(provider,a,root)
        val_cache=CausalGeometryCache(
            a.ccr_val_history_cache,namespace,max_bytes=0,
            ram_bytes=int(a.ccr_val_history_cache_ram_mib)*2**20,
            reserve_bytes=0,compression_level=6)
        manifest_cache=validate_val_history_cache_manifest(val_cache,a)
        provider.ccr_history_cache=val_cache
        provider.ccr_history_cache_mode="require"
        provider.ccr_history_cache_source=source
        provider.ccr_verify_cached_full_evidence_remaining=1
        result["val_history_cache"]=dict(
            root=str(Path(a.ccr_val_history_cache).resolve()),namespace=val_cache.namespace,
            disk_gib=manifest_cache.get("disk_gib"),ram_mib=a.ccr_val_history_cache_ram_mib)

        metrics={"baseline":Metrics(),**{k:Metrics() for k in VARIANTS}}
        quality={k:defaultdict(int) for k in VARIANTS}
        action=defaultdict(_bucket)
        thresholds=result["decision_rule"]["B_equivalent_threshold_on_corrected_ADD"]

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
            p_corr=probabilities(head,evidence,plan,output,device)
            target,valid=repair_targets(evidence,plan,raw["future_gt_occ"])
            zeros=np.zeros_like(p_corr[...,0],np.float32)

            pred_A=(p_corr[...,0]>=.5)&plan.legal[...,0]
            cut=np.where(
                evidence.actor[:,None]>=0,
                thresholds["dynamic"],thresholds["static"])
            pred_B=(p_corr[...,0]>=cut)&plan.legal[...,0]

            dense_A=compose_canonical(
                prep.baseline,evidence,plan,pred_A.astype(np.float32),zeros,
                thresholds=(.5,None),role="all")
            dense_B=compose_canonical(
                prep.baseline,evidence,plan,pred_B.astype(np.float32),zeros,
                thresholds=(.5,None),role="all")
            dense={"A_corrected_add05":dense_A,"B_raw_add05":dense_B}

            roles={"static":evidence.actor<0,"dynamic":evidence.actor>=0}
            for role_name,role_mask in roles.items():
                role6=np.broadcast_to(role_mask[:,None],target[...,0].shape)
                mask=role6&valid[...,0]
                _update(action[f"A/{role_name}"],mask,target[...,0],pred_A)
                _update(action[f"B/{role_name}"],mask,target[...,0],pred_B)
                for h in range(6):
                    hm=role_mask&valid[:,h,0]
                    _update(action[f"A/{role_name}/h{h+1}"],hm,target[:,h,0],pred_A[:,h])
                    _update(action[f"B/{role_name}/h{h+1}"],hm,target[:,h,0],pred_B[:,h])

            support=gt_moving_support_sequence(
                source.nusc,prep.window.t0_token,prep.window.future_tokens,
                tuple(.5*(h+1) for h in range(6)),grid=provider.pcfg.grid,
                workers=provider.workers)
            moving=moving_support_masks(support,provider.pcfg.grid.shape_hwd)
            for ri,h in enumerate(REPORT):
                gt=raw["future_gt_occ"][h];before=prep.baseline[h]
                metrics["baseline"].update(ri,before,gt,moving[h])
                for name in VARIANTS:
                    metrics[name].update(ri,dense[name][h],gt,moving[h])
                    for key,value in edit_quality(before,dense[name][h],gt).items():
                        quality[name][key]+=value
            if wi==1 or wi%16==0 or wi==len(records):
                print(f"CCR_CORRECTION_ABLATION {wi}/{len(records)}",flush=True)

        result["metrics"]={k:v.compute() for k,v in metrics.items()}
        result["vs_baseline_pp"]={
            k:delta(result["metrics"][k],result["metrics"]["baseline"]) for k in VARIANTS}
        result["B_minus_A_pp"]=delta(
            result["metrics"]["B_raw_add05"],result["metrics"]["A_corrected_add05"])
        result["action_ADD"]={k:_finish(v) for k,v in sorted(action.items())}
        result["quality"]={}
        for name in VARIANTS:
            q=dict(quality[name]);added=q.get("added",0)
            q["addition_semantic_precision"]=_safe_div(q.get("added_semantic_tp",0),added)
            result["quality"][name]=q
        result["population"]=dict(
            windows=len(records),mode="dev64" if a.windows==64 else "dev512",
            manifest_fingerprint=manifest["manifest_fingerprint"],
            key_fingerprint=stable_json_fingerprint([list(x) for x in keys]))
        result.update(
            status="complete",elapsed_seconds=time.perf_counter()-started,
            future_GT_model_input=False,
            algebraic_equivalence=(
                "B raw-sigmoid@0.5 is evaluated exactly as corrected-p >= 1/(1+w), "
                "with checkpoint role-specific ADD weights; ranking is unchanged."))
        persist();print(_summary(result),flush=True);return 0
    except BaseException as exc:
        result.update(status="failed",error=type(exc).__name__+": "+str(exc),
                      elapsed_seconds=time.perf_counter()-started)
        persist();raise
    finally:
        if val_cache is not None:
            result["val_history_cache_stats"]=val_cache.stats();val_cache.close()
        if execution is not None:execution.close()
        persist()


if __name__=="__main__":
    raise SystemExit(main())
