#!/usr/bin/env python3
"""Evaluate oracle/learned/speed/random KTA-V17 selection at equal source budgets.

Accuracy always uses one frozen full-batch V17 forward so every routing strategy
sees exactly the same expert predictions.  Optional latency benchmarking runs
additional selected-source forwards that are *not* used for metric computation;
this prevents batch-size numerical differences from contaminating the paired
accuracy comparison.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from real_motion.kta_v17_selector import (
    FEATURE_CONTRACT,
    SELECTOR_CACHE_VERSION,
    SELECTOR_FEATURE_DIM,
    SELECTOR_PROTOCOL,
    UTILITY_CONTRACT,
    KtaV17Selector,
    random_fraction_mask,
    selector_features,
    speed_score,
    top_fraction_mask,
)
from real_motion.metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS
from real_motion.metrics.occupancy_iou import OccupancyIoUMultiHorizon
from real_motion.msp_wm_cache import MSPWorldModelCacheDataset
from real_motion.nuscenes_adapter import NuScenesWindowSource
from real_motion.rigid_transport import compose_component_replacements_in_input_order, rasterize_rigid_component
from real_motion.runtime_config import add_config_args, load_runtime_config, make_prepare_config
from real_motion.strong_w2det import StrongW2DetConfig, extract_instances, match_instances
from tools.real_motion import eval_p0_f9_frozen_sparse_occfm as safe
from tools.real_motion.eval_p0_f9_v17_local_stwm import load_cache, load_model, t0_xy_to_world_preserve_source_z, window_from_record

PROTOCOL = "p0_f9_kta_v17_selector_eval_v1"


def _parse_budgets(raw):
    out=[]
    for z in str(raw).split(","):
        if not z.strip(): continue
        q=float(z); q=q/100.0 if q>1 else q
        if not 0<=q<=1: raise ValueError("budget outside [0,1]")
        out.append(q)
    out.extend([0.0,1.0]); return tuple(sorted(set(out)))


def _load_utility(path):
    obj=torch.load(path,map_location="cpu",weights_only=False)
    if obj.get("version")!=SELECTOR_CACHE_VERSION: raise RuntimeError("utility cache version mismatch")
    meta=obj.get("metadata") or {}
    if meta.get("feature_contract")!=FEATURE_CONTRACT or meta.get("utility_contract")!=UTILITY_CONTRACT:
        raise RuntimeError("utility cache contract mismatch")
    return meta,{str(r["sample_id"]):r for r in obj.get("records",[])}


def _load_selector(path,device):
    ck=torch.load(path,map_location="cpu",weights_only=False)
    if ck.get("protocol")!=SELECTOR_PROTOCOL: raise RuntimeError("selector checkpoint protocol mismatch")
    if ck.get("feature_contract")!=FEATURE_CONTRACT or ck.get("utility_contract")!=UTILITY_CONTRACT:
        raise RuntimeError("selector checkpoint contract mismatch")
    model=KtaV17Selector(int(ck["input_dim"]),int(ck["hidden_dim"])).to(device)
    model.load_state_dict(ck["state_dict"],strict=True); model.eval()
    return ck,model


def _new_state(free):
    return {"safe":safe._new_metrics(),"occ":OccupancyIoUMultiHorizon(free_label=int(free)),"selected":0,"sources":0}


def _update(st,h,pred,gt,moving,sel,n):
    safe._update(st["safe"],float(h),pred,gt,moving); st["occ"].update(float(h),pred,gt)
    st["selected"]+=int(sel); st["sources"]+=int(n)


def _report(st):
    r=safe._report(st["safe"]); r["occupancy"]=st["occ"].compute()
    r["selection"]={"ratio":float(st["selected"]/max(st["sources"],1)),"selected_source_events":st["selected"],"source_events":st["sources"]}
    return r


def _mix(anchor,base,v17,mask,free,grid):
    repl=[v17[i] if bool(mask[i]) else base[i] for i in range(len(base))]
    return compose_component_replacements_in_input_order(anchor,base,repl,dynamic_class_ids=DYNAMIC_CLASS_IDS,free_label=int(free),grid=grid)


def _positive_capped_mask(scores, q):
    """Select at most the Top-Q sources, but never force a predicted-harmful edit."""
    top = top_fraction_mask(scores, q)
    return top & (np.asarray(scores) > 0.0)


def _timed_forward(model, rec, ids, device, repeats=1):
    if len(ids)==0: return 0.0
    idx=torch.as_tensor(ids,dtype=torch.long,device=device)
    feat=rec["features"].float().to(device)[idx]
    tube=rec["local_semantic_tube"].to(device)[idx]
    kta=rec["kta_displacement_xy_m"].float().to(device)[idx]
    fm=rec["frame_motion_features"].float().to(device)[idx]
    sm=rec["target_source_mask_tube"].to(device)[idx]
    rows=[]
    for _ in range(max(1,int(repeats))):
        if device.type=="cuda": torch.cuda.synchronize(device)
        t=time.perf_counter()
        with torch.no_grad(),torch.autocast(device_type="cuda",dtype=torch.bfloat16,enabled=device.type=="cuda"):
            model(feat,tube,kta,fm,sm)
        if device.type=="cuda": torch.cuda.synchronize(device)
        rows.append((time.perf_counter()-t)*1000.0)
    return float(np.mean(rows))


def main():
    p=argparse.ArgumentParser(); add_config_args(p)
    p.add_argument("--local-stwm-cache",required=True); p.add_argument("--p0f9-cache",required=True)
    p.add_argument("--utility-cache",required=True); p.add_argument("--v17-checkpoint",required=True)
    p.add_argument("--selector-checkpoint",required=True); p.add_argument("--dataroot",required=True); p.add_argument("--info-pkl",required=True)
    p.add_argument("--budgets",default="0,10,20,40,100"); p.add_argument("--random-repeats",type=int,default=5); p.add_argument("--random-seed",type=int,default=20260913)
    p.add_argument("--measure-latency",action="store_true")
    p.add_argument("--latency-warmup-windows",type=int,default=8)
    p.add_argument("--latency-repeats",type=int,default=3)
    p.add_argument("--max-windows",type=int,default=0); p.add_argument("--output",required=True); p.add_argument("--device",default="cuda")
    a=p.parse_args(); budgets=_parse_budgets(a.budgets)
    cfg=load_runtime_config(a.config,a.override); pcfg=make_prepare_config(cfg)
    vmeta,records=load_cache(a.local_stwm_cache); umeta,utility=_load_utility(a.utility_cache)
    ds=MSPWorldModelCacheDataset(a.p0f9_cache); sid2={str(e["sample_id"]):i for i,e in enumerate(ds.entries)}
    records=[r for r in records if str(r["sample_id"]) in sid2 and str(r["sample_id"]) in utility]
    if a.max_windows>0: records=records[:min(len(records),a.max_windows)]
    if not records: raise RuntimeError("no common evaluation records")
    if set(umeta.get("scene_names",[])) != {str(r["scene_name"]) for r in records} and a.max_windows<=0:
        raise RuntimeError("utility cache/evaluation scene set mismatch")
    device=torch.device(a.device if a.device!="cuda" or torch.cuda.is_available() else "cpu")
    vck,v17=load_model(a.v17_checkpoint,device); sck,selector=_load_selector(a.selector_checkpoint,device)
    positive_cap_valid = sck.get("score_semantics") != "pairwise_ranking_logit_not_calibrated_utility"
    if str(Path(sck["val_cache"]).resolve()) != str(Path(a.utility_cache).resolve()):
        raise RuntimeError("selector checkpoint was not diagnosed against this utility cache")
    source=NuScenesWindowSource(a.dataroot,info_pkl=a.info_pkl,verbose=False); strong=StrongW2DetConfig(free_label=int(pcfg.free_label))
    learned={q:_new_state(pcfg.free_label) for q in budgets}
    learned_positive={q:_new_state(pcfg.free_label) for q in budgets}
    oracle={q:_new_state(pcfg.free_label) for q in budgets}; speed={q:_new_state(pcfg.free_label) for q in budgets}
    random={r:{q:_new_state(pcfg.free_label) for q in budgets} for r in range(a.random_repeats)}
    latency_topq={q:{"selector_ms":[],"v17_selected_ms":[],"selected":[]} for q in budgets}
    latency_positive={q:{"selector_ms":[],"v17_selected_ms":[],"selected":[]} for q in budgets}
    full_v17_ms=[]

    for wi,rec in enumerate(records):
        sid=str(rec["sample_id"]); urec=utility[sid]; n=int(rec["features"].shape[0])
        if int(urec["features"].shape[0])!=n: raise RuntimeError(f"{sid}: utility source count mismatch")
        w=window_from_record(rec); hist=[source.load_semantics(w.scene_name,t) for t in w.history_tokens]
        hp=[np.asarray(source.pose(t),dtype=np.float64) for t in w.history_tokens]; fp=[np.asarray(source.pose(t),dtype=np.float64) for t in w.future_tokens]
        cur=extract_instances(hist[-1],hp[-1],grid=pcfg.grid,cfg=strong); prev=extract_instances(hist[-2],hp[-2],grid=pcfg.grid,cfg=strong)
        vel=match_instances(prev,cur,float(pcfg.frame_dt_s),max_speed_mps=strong.max_match_speed_mps)
        if len(cur)!=n: raise RuntimeError(f"{sid}: Strong source count mismatch")
        feat=rec["features"].float().to(device); tube=rec["local_semantic_tube"].to(device); kta=rec["kta_displacement_xy_m"].float().to(device); fm=rec["frame_motion_features"].float().to(device); sm=rec["target_source_mask_tube"].to(device)
        with torch.no_grad(),torch.autocast(device_type="cuda",dtype=torch.bfloat16,enabled=device.type=="cuda"):
            vo=v17(feat,tube,kta,fm,sm)
        residual=vo["residual_xy_m"].float().cpu().numpy()
        sx=torch.as_tensor(urec["features"],dtype=torch.float32,device=device)
        if device.type=="cuda": torch.cuda.synchronize(device)
        ts=time.perf_counter()
        with torch.no_grad(): learned_score=(selector(sx).float().cpu().numpy()*float(sck["target_std"])+float(sck["target_mean"]))
        if device.type=="cuda": torch.cuda.synchronize(device)
        selector_ms=(time.perf_counter()-ts)*1000.0
        oracle_score=torch.as_tensor(urec["gt_utility_pct"]).numpy(); speed_score_arr=speed_score(rec).numpy()
        masks={
            "learned":{q:top_fraction_mask(learned_score,q) for q in budgets},
            "oracle":{q:top_fraction_mask(oracle_score,q) for q in budgets},
            "speed":{q:top_fraction_mask(speed_score_arr,q) for q in budgets},
        }
        if positive_cap_valid:
            masks["learned_positive"]={q:_positive_capped_mask(learned_score,q) for q in budgets}
        rm={r:{q:random_fraction_mask(n,q,a.random_seed+r*100003,sample_id=sid) for q in budgets} for r in range(a.random_repeats)}
        if a.measure_latency and wi >= int(a.latency_warmup_windows):
            reps=max(1,int(a.latency_repeats))
            full_v17_ms.append(_timed_forward(v17,rec,list(range(n)),device,reps))
            for q in budgets:
                ids=np.flatnonzero(masks["learned"][q]).tolist()
                latency_topq[q]["selector_ms"].append(selector_ms)
                latency_topq[q]["selected"].append(len(ids))
                latency_topq[q]["v17_selected_ms"].append(_timed_forward(v17,rec,ids,device,reps))
                if positive_cap_valid:
                    ids2=np.flatnonzero(masks["learned_positive"][q]).tolist()
                    latency_positive[q]["selector_ms"].append(selector_ms)
                    latency_positive[q]["selected"].append(len(ids2))
                    latency_positive[q]["v17_selected_ms"].append(_timed_forward(v17,rec,ids2,device,reps))
        payload=safe._sample_payload(ds[sid2[sid]],torch.device("cpu")); t0=hp[-1]
        for horizon,hi in safe.REPORT.items():
            base=[]; lv=[]; dt=(hi+1)*float(pcfg.frame_dt_s)
            for i,c in enumerate(cur):
                sc=np.asarray(c["centroid_world"],dtype=np.float64); vv=np.asarray(vel.get(i,np.zeros(3)),dtype=np.float64)
                base.append(rasterize_rigid_component(c["voxel_indices"],int(c["class_id"]),t0,fp[hi],source_center_world=sc,target_center_world=sc+vv*dt,yaw_delta_rad=0.0,grid=pcfg.grid))
                xy=rec["anchors_xy_t0_m"][i,hi].numpy()+residual[i,hi]; dst=t0_xy_to_world_preserve_source_z(xy,sc,t0)
                lv.append(rasterize_rigid_component(c["voxel_indices"],int(c["class_id"]),t0,fp[hi],source_center_world=sc,target_center_world=dst,yaw_delta_rad=0.0,grid=pcfg.grid))
            gt=payload["gt"][hi]; moving=payload["moving"][hi]; anchor=payload["anchor"][hi]
            for q in budgets:
                routed=(("learned",learned),("oracle",oracle),("speed",speed))
                if positive_cap_valid:
                    routed=(("learned",learned),("learned_positive",learned_positive),("oracle",oracle),("speed",speed))
                for name,table in routed:
                    m=masks[name][q]; pred=_mix(anchor,base,lv,m,pcfg.free_label,pcfg.grid); _update(table[q],horizon,pred,gt,moving,m.sum(),n)
                for r in range(a.random_repeats):
                    m=rm[r][q]; pred=_mix(anchor,base,lv,m,pcfg.free_label,pcfg.grid); _update(random[r][q],horizon,pred,gt,moving,m.sum(),n)
        if wi==0 or (wi+1)%16==0 or wi+1==len(records): print(f"selector eval {wi+1}/{len(records)} {sid}",flush=True)

    lr={str(q):_report(learned[q]) for q in budgets}
    lpr={str(q):_report(learned_positive[q]) for q in budgets} if positive_cap_valid else {}
    orr={str(q):_report(oracle[q]) for q in budgets}; sr={str(q):_report(speed[q]) for q in budgets}
    rr={}
    for q in budgets:
        rows=[_report(random[r][q]) for r in range(a.random_repeats)]
        rr[str(q)]={"Moving_mean":float(np.mean([x["moving"]["mIoU"] for x in rows])),"Moving_std":float(np.std([x["moving"]["mIoU"] for x in rows])),"mIoU_mean":float(np.mean([x["overall"]["mIoU"] for x in rows])),"IoU_mean":float(np.mean([x["occupancy"]["IoU"] for x in rows])),"repeats":rows}
    def _lat_report(src):
        out={}
        for q in budgets:
            s=src[q]
            out[str(q)]={
                "selector_ms_mean":float(np.mean(s["selector_ms"])) if s["selector_ms"] else float("nan"),
                "v17_selected_ms_mean":float(np.mean(s["v17_selected_ms"])) if s["v17_selected_ms"] else float("nan"),
                "combined_ms_mean":float(np.mean(np.asarray(s["selector_ms"])+np.asarray(s["v17_selected_ms"]))) if s["selector_ms"] else float("nan"),
                "mean_selected_sources":float(np.mean(s["selected"])) if s["selected"] else float("nan"),
            }
        return out
    lat_topq=_lat_report(latency_topq) if a.measure_latency else {}
    lat_positive=_lat_report(latency_positive) if a.measure_latency and positive_cap_valid else {}
    report={
        "protocol":PROTOCOL,"num_windows":len(records),"budgets":list(budgets),
        "v17_checkpoint":str(Path(a.v17_checkpoint).resolve()),
        "selector_checkpoint":str(Path(a.selector_checkpoint).resolve()),
        "utility_cache":str(Path(a.utility_cache).resolve()),
        "learned":lr,
        "learned_positive_cap":lpr,
        "positive_cap_valid":bool(positive_cap_valid),
        "selector_score_semantics":sck.get("score_semantics","calibrated_utility_regression"),
        "oracle_gt_utility_rank":orr,"speed_rule":sr,"random":rr,
        "latency":{
            "measured":bool(a.measure_latency),
            "warmup_windows":int(a.latency_warmup_windows),
            "repeats":int(a.latency_repeats),
            "full_v17_ms_mean":float(np.mean(full_v17_ms)) if full_v17_ms else float("nan"),
            "topq_by_budget":lat_topq,
            "positive_cap_by_budget":lat_positive,
            "scope":"same warmed GPU V17 timing path plus selector forward only; occupancy I/O and compositor excluded",
        },
    }
    op=Path(a.output); op.parent.mkdir(parents=True,exist_ok=True); op.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("=== KTA/V17 SELECTOR EVAL ===")
    if positive_cap_valid:
        print(f"{'Q':>7} {'learned':>10} {'learned+':>10} {'oracle':>10} {'speed':>10} {'random':>10}")
        for q in budgets:
            print(f"{100*q:6.1f}% {lr[str(q)]['moving']['mIoU']:10.4f} {lpr[str(q)]['moving']['mIoU']:10.4f} {orr[str(q)]['moving']['mIoU']:10.4f} {sr[str(q)]['moving']['mIoU']:10.4f} {rr[str(q)]['Moving_mean']:10.4f}")
    else:
        print("ranking selector score is not calibrated utility; learned+ abstention is disabled")
        print(f"{'Q':>7} {'learned':>10} {'oracle':>10} {'speed':>10} {'random':>10}")
        for q in budgets:
            print(f"{100*q:6.1f}% {lr[str(q)]['moving']['mIoU']:10.4f} {orr[str(q)]['moving']['mIoU']:10.4f} {sr[str(q)]['moving']['mIoU']:10.4f} {rr[str(q)]['Moving_mean']:10.4f}")
    if a.measure_latency: print("latency=",json.dumps(report["latency"],indent=2))
    print(f"saved {op}")


if __name__=="__main__": main()
