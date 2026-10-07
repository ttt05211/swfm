#!/usr/bin/env python3
"""Frozen-B expanded validation on the entire 4369-window validation cache.

One read-only pass reports:
  1) full4369;
  2) windows outside frozen DEV512;
  3) scenes outside frozen DEV512 (if any actually exist).

Candidate B is fixed: weighted Point-CCR checkpoint, raw ADD sigmoid @0.5,
REMOVE disabled.  Old Local is recomputed on the SAME records with its historical
full-validation fixed thresholds (GEN ADD=.5, REFINE ADD=.5, REMOVE off), so no
REMOVE-on DEV512 number is mixed into the comparison.

The pass also records per-class IoU gaps, static/dynamic ADD false-positive /
false-negative counts, semantic precision and per-scene gain/loss summaries.
No training, threshold search, checkpoint mutation or FPS timing occurs here.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

if __package__ in (None,""):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from real_motion.canonical_causal_repair import compose_canonical, repair_targets
from real_motion.canonical_repair_execution import CanonicalCpuExecution
from real_motion.causal_geometry_cache import CausalGeometryCache
from real_motion.ccr_frozen_b import frozen_b_probabilities, effective_corrected_add_thresholds
from real_motion.ccr_val_history_cache import (
    namespace as val_history_cache_namespace,
    validate_manifest as validate_val_history_cache_manifest,
)
from real_motion.causal_column_completion import actions_from_probabilities, compose_dense
from real_motion.column_runtime_pipeline import CachedColumnSource, prefetch_raw_columns
from real_motion.metrics.moving_miou_v2 import NUSCENES_LABELS
from real_motion.nuscenes_adapter import NuScenesWindowSource, gt_moving_support_sequence
from real_motion.runtime_config import add_config_args, load_runtime_config, make_prepare_config
from real_motion.source_evidence_audit import edit_quality
from real_motion.v21_source_induction import stable_json_fingerprint
from tools.real_motion import causal_column_common as columns
from tools.real_motion.ccr_screen_common import build_inputs, map_inputs, old_execution
from tools.real_motion.eval_p0_f9_v18_se2 import load_cache
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import (
    Metrics, align_records, delta, load_manifest, sha256,
)
from tools.real_motion.point_ccr_v18_fps_common import load_point_head
from tools.real_motion.run_p0_f9_shared_evidence_pilot import PilotProvider, require_cuda
from tools.real_motion.shared_evidence_pilot_common import moving_support_masks
from tools.real_motion.static_evidence_selector_common import CLEAN_SHA256, write_json
from tools.real_motion.train_p0_f9_causal_columns import record_keys
from tools.real_motion.train_p0_f9_joint_causal_columns import load_joint


PROTOCOL="p0_f9_ccr_frozen_b_expanded_validation_v1"
REPORT=(1,3,5)
VARIANTS=("baseline","frozen_b","old_local_remove_off")
OLD_LOCAL_GATES=(.5,.5,None)


def _safe_div(a,b):
    return None if not b else float(a)/float(b)


def _bucket():
    return dict(valid=0,target_pos=0,pred_pos=0,tp=0,fp=0,fn=0)


def _update_bucket(bucket,valid,target,pred):
    valid=np.asarray(valid,bool)
    target=np.asarray(target,bool)&valid
    pred=np.asarray(pred,bool)&valid
    bucket["valid"]+=int(valid.sum())
    bucket["target_pos"]+=int(target.sum())
    bucket["pred_pos"]+=int(pred.sum())
    bucket["tp"]+=int((target&pred).sum())
    bucket["fp"]+=int((~target&pred&valid).sum())
    bucket["fn"]+=int((target&~pred).sum())


def _finish_bucket(x):
    out=dict(x)
    out["precision"]=_safe_div(out["tp"],out["tp"]+out["fp"])
    out["recall"]=_safe_div(out["tp"],out["tp"]+out["fn"])
    out["prevalence"]=_safe_div(out["target_pos"],out["valid"])
    out["predicted_positive_rate"]=_safe_div(out["pred_pos"],out["valid"])
    return out


def _new_subset():
    return dict(
        windows=0,scenes=set(),
        metrics={k:Metrics() for k in VARIANTS},
        quality={k:defaultdict(int) for k in ("frozen_b","old_local_remove_off")},
        action=defaultdict(_bucket),
    )


def _attach_old_local_fixed_geometry(prep,provider,teacher):
    """Attach exactly the history/ego geometry needed by old Local columns.

    This reuses the cached Strong/registration/render state and only rebuilds
    old Local's future static memory + footprint/frontier live. No GT enters.
    """
    raw=prep.raw
    footprint=columns.history_grid_footprint_bev_sequence(
        raw["history_poses"],raw["future_poses"],provider.pcfg.grid,
        workers=max(1,min(3,provider.workers)))
    memory=columns.build_future_static_memory_only(
        raw["history_occ"],raw["history_observed"],raw["history_poses"],raw["future_poses"],
        grid=provider.pcfg.grid,dynamic_class_ids=columns.DYN,free_label=columns.FREE,
        workers=max(1,min(3,provider.workers)))
    prep.footprints=footprint
    prep.memory=memory
    prep.fixed_candidate_geometry=columns.fixed_candidate_geometry(
        memory,footprint,provider.pcfg.grid,teacher.columns.config)


def _per_class_gap(metrics):
    rows=[]
    b=metrics["frozen_b"];base=metrics["baseline"];old=metrics["old_local_remove_off"]
    for cid in range(17):
        bvals=[];basevals=[];oldvals=[]
        per_h={}
        for h in ("1.0","2.0","3.0"):
            bv=b["per_horizon"][h]["semantic_per_class"][str(cid)]
            av=base["per_horizon"][h]["semantic_per_class"][str(cid)]
            ov=old["per_horizon"][h]["semantic_per_class"][str(cid)]
            per_h[h]=dict(
                frozen_b=bv,baseline=av,old_local_remove_off=ov,
                b_minus_baseline=(bv-av if np.isfinite(bv) and np.isfinite(av) else None),
                b_minus_old=(bv-ov if np.isfinite(bv) and np.isfinite(ov) else None))
            if np.isfinite(bv):bvals.append(bv)
            if np.isfinite(av):basevals.append(av)
            if np.isfinite(ov):oldvals.append(ov)
        mean=lambda x:None if not x else float(np.mean(x))
        bm,am,om=mean(bvals),mean(basevals),mean(oldvals)
        rows.append(dict(
            class_id=cid,class_name=NUSCENES_LABELS[cid],
            frozen_b_mean_iou=bm,baseline_mean_iou=am,old_local_remove_off_mean_iou=om,
            b_minus_baseline_mean_pp=(None if bm is None or am is None else bm-am),
            b_minus_old_mean_pp=(None if bm is None or om is None else bm-om),
            per_horizon=per_h))
    return rows


def _scene_summary(scene_metrics):
    rows=[]
    for scene,variants in scene_metrics.items():
        m={k:v.compute() for k,v in variants.items()}
        rows.append(dict(
            scene=scene,
            windows=int(variants["_windows"]),
            baseline_mIoU=m["baseline"]["mIoU"],
            frozen_b_mIoU=m["frozen_b"]["mIoU"],
            old_local_mIoU=m["old_local_remove_off"]["mIoU"],
            b_minus_baseline_mIoU=m["frozen_b"]["mIoU"]-m["baseline"]["mIoU"],
            b_minus_old_mIoU=m["frozen_b"]["mIoU"]-m["old_local_remove_off"]["mIoU"],
            baseline_MovingMicro=m["baseline"]["MovingMicro"],
            frozen_b_MovingMicro=m["frozen_b"]["MovingMicro"],
            old_local_MovingMicro=m["old_local_remove_off"]["MovingMicro"],
            b_minus_baseline_MovingMicro=m["frozen_b"]["MovingMicro"]-m["baseline"]["MovingMicro"],
            b_minus_old_MovingMicro=m["frozen_b"]["MovingMicro"]-m["old_local_remove_off"]["MovingMicro"],
        ))
    def summarize(key):
        vals=np.asarray([r[key] for r in rows],np.float64)
        return dict(
            scenes=len(rows),improved=int((vals>0).sum()),worse=int((vals<0).sum()),
            tied=int((vals==0).sum()),mean=float(vals.mean()) if len(vals) else None,
            median=float(np.median(vals)) if len(vals) else None,
            p10=float(np.percentile(vals,10)) if len(vals) else None,
            p90=float(np.percentile(vals,90)) if len(vals) else None)
    return dict(
        counts=dict(
            b_vs_baseline_mIoU=summarize("b_minus_baseline_mIoU"),
            b_vs_old_mIoU=summarize("b_minus_old_mIoU"),
            b_vs_baseline_Moving=summarize("b_minus_baseline_MovingMicro"),
            b_vs_old_Moving=summarize("b_minus_old_MovingMicro")),
        best_vs_old=sorted(rows,key=lambda x:x["b_minus_old_mIoU"],reverse=True)[:10],
        worst_vs_old=sorted(rows,key=lambda x:x["b_minus_old_mIoU"])[:10],
        per_scene=rows)


def _finalize_subset(state):
    if state["windows"]==0:
        return dict(available=False,windows=0,scenes=0)
    metrics={k:v.compute() for k,v in state["metrics"].items()}
    quality={}
    for name,q0 in state["quality"].items():
        q=dict(q0);added=q.get("added",0)
        q["addition_semantic_precision"]=_safe_div(q.get("added_semantic_tp",0),added)
        quality[name]=q
    return dict(
        available=True,windows=int(state["windows"]),scenes=len(state["scenes"]),
        metrics=metrics,
        delta_b_vs_baseline=delta(metrics["frozen_b"],metrics["baseline"]),
        delta_b_vs_old_local_remove_off=delta(metrics["frozen_b"],metrics["old_local_remove_off"]),
        action_ADD={k:_finish_bucket(v) for k,v in sorted(state["action"].items())},
        quality=quality,
        per_class=_per_class_gap(metrics))


def _summary(result):
    lines=["===== FROZEN B EXPANDED VALIDATION =====","status="+result["status"]]
    if "decision_rule" in result:
        lines.append("frozen B="+json.dumps(result["decision_rule"],sort_keys=True))
    for name,row in result.get("subsets",{}).items():
        if not row.get("available"):
            lines.append(f"{name}: unavailable / 0 windows")
            continue
        m=row["metrics"]
        lines.append(f'--- {name}: windows={row["windows"]} scenes={row["scenes"]} ---')
        for k,label in (("baseline","BASE"),("frozen_b","B"),("old_local_remove_off","OLD_OFF")):
            x=m[k]
            lines.append(f'{label:7s} mIoU={x["mIoU"]:.4f} MovingMicro={x["MovingMicro"]:.4f}')
        d=row["delta_b_vs_old_local_remove_off"]
        lines.append(f'B-OLD_OFF mIoU={d["mIoU"]:+.4f} Moving={d["MovingMicro"]:+.4f}')
        for role in ("static","dynamic"):
            q=row["action_ADD"].get(role,{})
            if q:
                lines.append(
                    f'{role} ADD P={q["precision"]:.4f} R={q["recall"]:.4f} '
                    f'FP={q["fp"]} FN={q["fn"]}')
    cov=result.get("population_contract",{})
    if cov:
        lines.append("population="+json.dumps(cov,sort_keys=True))
    lines.append("Old Local comparator is recomputed with fixed (.5,.5,REMOVE-off) on the same records.")
    lines.append("No threshold search, training, checkpoint write or independent-test claim.")
    lines.append("full details: frozen_b_expanded_validation.json")
    return "\n".join(lines)+"\n"


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);add_config_args(p)
    for key in ("checkpoint","ccr-checkpoint","base-checkpoint","dev-cache",
                "population-manifest","dataroot","dev-info","out-dir"):
        p.add_argument("--"+key,required=True)
    p.add_argument("--device",default="cuda")
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
    if (not Path(a.dataroot).is_dir() or not 1<=a.cpu_workers<=16
            or not 1<=a.ccr_cpu_workers<=8 or a.old_local_batch_size<1
            or not 0<=a.ccr_val_history_cache_ram_mib<=16384):
        p.error("invalid paths/budgets")

    device=require_cuda(a.device);torch.set_num_threads(1)
    out.mkdir(parents=True);started=time.perf_counter()
    result=dict(status="running",protocol=PROTOCOL,read_only=True,training=False,
                threshold_search=False,independent_test_claim=False)
    def persist():
        write_json(out/"frozen_b_expanded_validation.json",result)
        (out/"summary.txt").write_text(_summary(result),encoding="utf-8")
    persist()

    execution=val_cache=None
    try:
        cfg=load_runtime_config(a.config,a.override);config_fp=stable_json_fingerprint(cfg)
        epoch19_sha=sha256(a.checkpoint)
        if sha256(a.base_checkpoint)!=CLEAN_SHA256:
            raise RuntimeError("Clean-E14 base checkpoint fingerprint mismatch")
        ck,teacher=load_joint(
            a.checkpoint,device,reference_sha=CLEAN_SHA256,
            config_sha=config_fp,allow_diagnostic=True)
        if (teacher.transport.config.history_frames!=4 or ck.get("cursor_epoch")!=19
                or ck["model_configs"].get("adaptive_context") is not None):
            raise RuntimeError("selected strict-four epoch19 Local checkpoint required")
        teacher.eval().requires_grad_(False)
        for path,expected in ((a.dev_cache,ck["cache_fingerprints"]["dev"]),
                              (a.dev_info,ck["info_fingerprints"]["dev"])):
            if sha256(path)!=expected:raise RuntimeError("epoch19/data provenance mismatch: "+path)

        saved=torch.load(a.ccr_checkpoint,map_location="cpu",weights_only=False)
        head=load_point_head(
            saved,teacher_sha256=epoch19_sha,config_fingerprint=config_fp,
            source_dim=teacher.columns.source_dim,device=device,
            allow_completed_epoch_boundary=True)
        head.eval().requires_grad_(False)
        result["decision_rule"]={
            "name":"frozen_B","ADD_score":"sigmoid(raw_weighted_logit)",
            "ADD_threshold":.5,"REMOVE":False,
            **effective_corrected_add_thresholds(head)}
        result["checkpoint"]={
            "path":str(Path(a.ccr_checkpoint).resolve()),"sha256":sha256(a.ccr_checkpoint),
            "epoch":int(saved.get("epoch",0)),"updates":int(saved.get("updates",0))}

        manifest,keys64,_=load_manifest(a.population_manifest)
        dev512=tuple(map(tuple,manifest["parent_keys"]))
        if (len(dev512)!=512 or len(set(dev512))!=512
                or manifest["manifest_fingerprint"]!=ck["dev_manifest_fingerprint"]
                or dev512!=tuple(map(tuple,ck["dev_keys"]))
                or saved["contract"]["dev_manifest_fingerprint"]!=manifest["manifest_fingerprint"]):
            raise RuntimeError("frozen DEV512 manifest mismatch")
        dev512_set=set(dev512);dev512_scenes={s for s,_ in dev512}

        _,all_dev=load_cache(a.dev_cache)
        full_keys=record_keys(all_dev)
        if len(full_keys)!=4369 or len(set(full_keys))!=4369:
            raise RuntimeError("expanded validation requires exact unique full4369 cache")
        records=align_records(all_dev,full_keys);del all_dev
        outside_window=sum((str(r["scene_name"]),str(r["t0_token"])) not in dev512_set for r in records)
        outside_scene=sum(str(r["scene_name"]) not in dev512_scenes for r in records)
        full_scenes={str(r["scene_name"]) for r in records}
        outside_scene_names=full_scenes-dev512_scenes
        result["population_contract"]=dict(
            full4369_windows=len(records),full4369_scenes=len(full_scenes),
            dev512_windows=512,dev512_scenes=len(dev512_scenes),
            outside_dev512_windows=outside_window,
            outside_dev512_scene_windows=outside_scene,
            outside_dev512_scenes=len(outside_scene_names),
            note=("These are enlarged frozen-rule validation subsets, not newly untouched test sets. "
                  "If DEV512 already covers every validation scene, scene-disjoint remainder is correctly reported as empty."))

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
        cache_manifest=validate_val_history_cache_manifest(val_cache,a)
        provider.ccr_history_cache=val_cache
        provider.ccr_history_cache_mode="require"
        provider.ccr_history_cache_source=source
        provider.ccr_verify_cached_full_evidence_remaining=1
        result["val_history_cache"]=dict(
            root=str(Path(a.ccr_val_history_cache).resolve()),namespace=val_cache.namespace,
            disk_gib=cache_manifest.get("disk_gib"),ram_mib=a.ccr_val_history_cache_ram_mib)

        subsets={
            "full4369":_new_subset(),
            "outside_dev512_windows":_new_subset(),
            "outside_dev512_scenes":_new_subset(),
        }
        scene_metrics=defaultdict(lambda:{
            "baseline":Metrics(),"frozen_b":Metrics(),"old_local_remove_off":Metrics(),"_windows":0})

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
                target,valid=repair_targets(evidence,plan,raw["future_gt_occ"])
                dense_b=compose_canonical(
                    prep.baseline,evidence,plan,score[...,0],score[...,1],
                    thresholds=(.5,.95),role="all")
                pred_add=(score[...,0]>=.5)&plan.legal[...,0]

                # Matched old-Local comparator: same record/transport, four histories,
                # GEN/REFINE ADD=.5 and REMOVE disabled, exactly as historical full eval.
                _attach_old_local_fixed_geometry(prep,provider,teacher)
                old_by_h={}
                for h in REPORT:
                    old_plan=columns.candidate_plan(
                        prep,h,provider.pcfg.grid,teacher.columns.config)
                    old_p=columns.predict_probabilities(
                        teacher.columns,prep,h,old_plan,provider.pcfg.grid,device,
                        a.old_local_batch_size)
                    old_by_h[h]=compose_dense(
                        prep.baseline[h],old_plan,
                        actions_from_probabilities(old_plan,old_p,OLD_LOCAL_GATES))

                support=gt_moving_support_sequence(
                    source.nusc,prep.window.t0_token,prep.window.future_tokens,
                    tuple(.5*(h+1) for h in range(6)),grid=provider.pcfg.grid,
                    workers=provider.workers)
                moving=moving_support_masks(support,provider.pcfg.grid.shape_hwd)

                key=(str(record["scene_name"]),str(record["t0_token"]))
                scene=str(record["scene_name"])
                active=["full4369"]
                if key not in dev512_set:active.append("outside_dev512_windows")
                if scene not in dev512_scenes:active.append("outside_dev512_scenes")

                roles={"static":evidence.actor<0,"dynamic":evidence.actor>=0}
                for name in active:
                    st=subsets[name];st["windows"]+=1;st["scenes"].add(scene)
                    for role_name,role_mask in roles.items():
                        role6=np.broadcast_to(role_mask[:,None],target[...,0].shape)
                        mask=role6&valid[...,0]
                        _update_bucket(st["action"][role_name],mask,target[...,0],pred_add)

                sm=scene_metrics[scene];sm["_windows"]+=1
                for ri,h in enumerate(REPORT):
                    gt=raw["future_gt_occ"][h];base=prep.baseline[h]
                    preds={"baseline":base,"frozen_b":dense_b[h],
                           "old_local_remove_off":old_by_h[h]}
                    for name in active:
                        st=subsets[name]
                        for variant,pred in preds.items():
                            st["metrics"][variant].update(ri,pred,gt,moving[h])
                        for variant in ("frozen_b","old_local_remove_off"):
                            for qk,qv in edit_quality(base,preds[variant],gt).items():
                                st["quality"][variant][qk]+=qv
                    for variant,pred in preds.items():
                        sm[variant].update(ri,pred,gt,moving[h])

                if wi==1 or wi%32==0 or wi==len(records):
                    print(f"FROZEN_B_EXPANDED {wi}/{len(records)}",flush=True)

        result["subsets"]={k:_finalize_subset(v) for k,v in subsets.items()}
        result["scene_diagnostics"]=_scene_summary(scene_metrics)
        result.update(
            status="complete",elapsed_seconds=time.perf_counter()-started,
            old_local_thresholds=list(OLD_LOCAL_GATES),
            old_local_threshold_source="fixed historical full-validation REMOVE-off contract",
            future_GT_model_input=False)
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
