#!/usr/bin/env python3
"""Cheap no-training ablation: how much does frozen V16 actually use its tube?

Kinematic features and KTA future displacements are held fixed.  Only the local
semantic tube is perturbed:
  original  : untouched six-frame tube;
  repeat_t0 : repeat the current frame six times (remove visual temporal change);
  reverse   : reverse tube order while keeping kinematic/time embeddings fixed;
  background: replace every semantic cell by free/background.

If original beats repeat/reverse, temporal visual information matters.  If
background alone hurts, the model uses spatial context even if visual motion is
weak.  This diagnostic does not read nuScenes or run occupancy rasterization.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset

from real_motion.local_st_world_model import (
    LOCAL_STWM_CACHE_VERSION,LOCAL_TUBE_CONTRACT,MODEL_PROTOCOL,
    LocalSpatialTemporalWorldModel,config_from_mapping,
)
from real_motion.metrics.moving_miou_v2 import SPEED_THRESHOLD_MPS
from real_motion.motion_transport import FUTURE_FRAMES
from real_motion.motion_transport_v2 import TARGET_CONTRACT

MODES=("original","repeat_t0","reverse","background")


def load_cache(path):
    obj=torch.load(path,map_location="cpu",weights_only=False)
    if obj.get("version")!=LOCAL_STWM_CACHE_VERSION: raise RuntimeError("V16 cache version mismatch")
    meta=obj.get("metadata") or {}
    if meta.get("target_contract")!=TARGET_CONTRACT or meta.get("local_tube_contract")!=LOCAL_TUBE_CONTRACT:
        raise RuntimeError("V16 cache contract mismatch")
    return meta,obj.get("records") or []


def load_model(path,device):
    ck=torch.load(path,map_location="cpu",weights_only=False)
    if ck.get("protocol")!=MODEL_PROTOCOL: raise RuntimeError("V16 checkpoint protocol mismatch")
    model=LocalSpatialTemporalWorldModel(config_from_mapping(ck.get("model_config"))).to(device)
    model.load_state_dict(ck["state_dict"],strict=True); model.eval(); return ck,model


def flatten(records):
    fs=[]; tubes=[]; kta=[]; target=[]; target_disp=[]; valid=[]
    for r in records:
        sup=r["supervised_source"].bool()
        if not bool(sup.any()): continue
        fs.append(r["features"][sup].float()); tubes.append(r["local_semantic_tube"][sup].to(torch.uint8))
        kta.append(r["kta_displacement_xy_m"][sup].float()); target.append(r["target_residual_xy_m"][sup].float())
        target_disp.append(r["target_displacement_xy_m"][sup].float()); valid.append(r["target_valid"][sup].bool())
    return tuple(torch.cat(x,dim=0) for x in (fs,tubes,kta,target,target_disp,valid))


def perturb(tube,mode,free_label=17):
    if mode=="original": return tube
    if mode=="repeat_t0": return tube[:,-1:].expand_as(tube)
    if mode=="reverse": return torch.flip(tube,dims=(1,))
    if mode=="background": return torch.full_like(tube,int(free_label))
    raise ValueError(mode)


def true_moving(target_disp,valid):
    dt=torch.arange(1,FUTURE_FRAMES+1,dtype=target_disp.dtype,device=target_disp.device)[None]*0.5
    speed=torch.linalg.vector_norm(target_disp,dim=-1)/dt
    return valid & (speed>=float(SPEED_THRESHOLD_MPS))


def latest_mean(err,mask):
    vals=[]
    for i in range(mask.shape[0]):
        ids=torch.nonzero(mask[i],as_tuple=False).flatten()
        if ids.numel(): vals.append(err[i,int(ids[-1])])
    return float(torch.stack(vals).mean()) if vals else float("nan")


def main():
    p=argparse.ArgumentParser(); p.add_argument("--cache",required=True); p.add_argument("--checkpoint",required=True)
    p.add_argument("--output",required=True); p.add_argument("--batch-size",type=int,default=256); p.add_argument("--device",default="cuda"); a=p.parse_args()
    meta,records=load_cache(a.cache)
    if not records: raise RuntimeError("cache empty")
    device=torch.device(a.device if a.device!="cuda" or torch.cuda.is_available() else "cpu")
    ck,model=load_model(a.checkpoint,device); data=flatten(records); loader=DataLoader(TensorDataset(*data),batch_size=a.batch_size,shuffle=False)
    accum={m:{"all":[],"moving":[],"fde_all":[],"fde_moving":[],"h":{1:[],3:[],5:[]},"pred":[]} for m in MODES}
    with torch.no_grad():
        for raw in loader:
            f,tube,kta,target,target_disp,valid=[x.to(device) for x in raw]
            moving=true_moving(target_disp,valid.bool())
            for mode in MODES:
                pt=perturb(tube,mode)
                with torch.autocast(device_type="cuda",dtype=torch.bfloat16,enabled=device.type=="cuda"):
                    out=model(f,pt,kta)
                pred=out["residual_xy_m"].float(); err=torch.linalg.vector_norm(pred-target.float(),dim=-1)
                if bool(valid.any()): accum[mode]["all"].append(err[valid].cpu())
                if bool(moving.any()): accum[mode]["moving"].append(err[moving].cpu())
                accum[mode]["fde_all"].append(latest_mean(err,valid.bool())); accum[mode]["fde_moving"].append(latest_mean(err,moving))
                for hi in (1,3,5):
                    m=moving[:,hi]
                    if bool(m.any()): accum[mode]["h"][hi].append(err[:,hi][m].cpu())
                accum[mode]["pred"].append(pred.cpu())
    original=torch.cat(accum["original"]["pred"],dim=0)
    report={}
    for mode in MODES:
        d=accum[mode]; pred=torch.cat(d["pred"],dim=0)
        report[mode]={
            "all_valid_ade_m":float(torch.cat(d["all"]).mean()),
            "true_moving_ade_m":float(torch.cat(d["moving"]).mean()),
            "all_valid_fde_m":float(np.nanmean(d["fde_all"])),
            "true_moving_fde_m":float(np.nanmean(d["fde_moving"])),
            "true_moving_1s_ade_m":float(torch.cat(d["h"][1]).mean()),
            "true_moving_2s_ade_m":float(torch.cat(d["h"][3]).mean()),
            "true_moving_3s_ade_m":float(torch.cat(d["h"][5]).mean()),
            "prediction_delta_rms_vs_original_m":float(torch.sqrt(torch.mean((pred-original)**2))),
        }
    print("\n=== V16 FROZEN LOCAL-TUBE ABLATION ===")
    print(f"{'mode':12s} {'ADE':>8s} {'MovingADE':>10s} {'MovingFDE':>10s} {'1s':>8s} {'2s':>8s} {'3s':>8s} {'dPredRMS':>10s}")
    for mode in MODES:
        r=report[mode]
        print(f"{mode:12s} {r['all_valid_ade_m']:8.4f} {r['true_moving_ade_m']:10.4f} {r['true_moving_fde_m']:10.4f} {r['true_moving_1s_ade_m']:8.4f} {r['true_moving_2s_ade_m']:8.4f} {r['true_moving_3s_ade_m']:8.4f} {r['prediction_delta_rms_vs_original_m']:10.4f}")
    result={"protocol":"p0_f9_v16_frozen_local_tube_ablation_v1","checkpoint":str(Path(a.checkpoint).resolve()),"checkpoint_epoch":int(ck.get("epoch",-1)),
            "cache":str(Path(a.cache).resolve()),"num_windows":len(records),"report":report,"cache_metadata":meta}
    op=Path(a.output); op.parent.mkdir(parents=True,exist_ok=True); op.write_text(json.dumps(result,indent=2),encoding="utf-8"); print(f"saved {op}")


if __name__=="__main__": main()
