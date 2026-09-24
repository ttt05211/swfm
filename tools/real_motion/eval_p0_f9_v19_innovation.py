#!/usr/bin/env python3
"""Evaluate trained V19 Innovation on frozen Clean-E14 + Static Memory."""
from __future__ import annotations
import argparse, json, sys, time
from contextlib import nullcontext
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from real_motion.nuscenes_adapter import NuScenesWindowSource, gt_moving_support_for_horizon
from real_motion.metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS
from real_motion.prepared import load_nuscenes_window_raw
from real_motion.runtime_config import add_config_args,load_runtime_config,make_prepare_config
from real_motion.strong_w2det import StrongW2DetConfig
from real_motion.v19_innovation import ResidualInnovationHead,ResidualInnovationIntervalHead,base_explained_bev,build_future_aligned_history_and_static_memory,decode_innovation
from real_motion.v19_innovation_v5 import ResidualInnovationEndpointHead
from real_motion.v19_scene_memory import protected_add_only
from tools.real_motion import eval_p0_f9_v18_se2 as base
from tools.real_motion import eval_p0_f9_v18_full_validation as full
from tools.real_motion.benchmark_p0_f9_v18_runtime import _forecast_once,_prepare_record,_release_gpu_inputs,_stage_gpu_inputs
from tools.real_motion.eval_p0_f9_v17_local_stwm import window_from_record
from tools.real_motion.eval_p0_f9_v19_memory_ablation import _delta,_finalize,_new_raw,_update
from tools.real_motion.train_p0_f9_v18_se2_clean import PROTOCOL as CLEAN_PROTOCOL
from tools.real_motion.train_p0_f9_v19_innovation import PROTOCOL as INNOVATION_TRAIN_PROTOCOL

PROTOCOL="p0_f9_v19_innovation_frozen_base_eval_v1"
HORIZONS=(1.0,2.0,3.0); REPORT={1.0:1,2.0:3,3.0:5}
VARIANTS=("v18","v18_static","v18_static_innovation")

class CachedSource(NuScenesWindowSource):
    from functools import lru_cache
    @lru_cache(maxsize=768)
    def load_semantics(self,scene_name,token): return super().load_semantics(scene_name,token)
    @lru_cache(maxsize=768)
    def load_occ3d(self,scene_name,token,require_lidar_mask=True):
        return super().load_occ3d(scene_name,token,require_lidar_mask=require_lidar_mask)
    @lru_cache(maxsize=4096)
    def pose(self,token): return super().pose(token)

def _autocast(device,enabled):
    return torch.autocast(device_type="cuda",dtype=torch.bfloat16) if enabled and device.type=="cuda" else nullcontext()

def _load_innovation(path,device):
    ck=torch.load(path,map_location="cpu",weights_only=False)
    if ck.get("protocol")!=INNOVATION_TRAIN_PROTOCOL:
        raise RuntimeError(f"unexpected innovation checkpoint protocol: {ck.get('protocol')}")
    head_type=str(ck.get("head_type","voxel_bins"))
    if head_type=="vertical_interval":
        model=ResidualInnovationIntervalHead(**dict(ck["architecture"])).to(device)
    elif head_type=="vertical_endpoints_v5":
        model=ResidualInnovationEndpointHead(**dict(ck["architecture"])).to(device)
    elif head_type=="voxel_bins":
        model=ResidualInnovationHead(**dict(ck["architecture"])).to(device)
    else:
        raise RuntimeError(f"unknown innovation head_type: {head_type}")
    model.load_state_dict(ck["innovation_state_dict"],strict=True); model.eval()
    return ck,model

def main():
    p=argparse.ArgumentParser(); add_config_args(p)
    p.add_argument("--val-cache",required=True); p.add_argument("--base-checkpoint",required=True)
    p.add_argument("--innovation-checkpoint",required=True); p.add_argument("--dataroot",required=True)
    p.add_argument("--info-pkl",required=True); p.add_argument("--output",required=True)
    p.add_argument("--max-windows",type=int,default=0)
    p.add_argument("--alignment-workers",type=int,default=4)
    p.add_argument("--preserve-record-order",action="store_true")
    p.add_argument("--num-shards",type=int,default=1)
    p.add_argument("--shard-index",type=int,default=0)
    p.add_argument("--add-threshold",type=float,default=.5)
    p.add_argument("--vertical-threshold",type=float,default=.5); p.add_argument("--device",default="cuda")
    p.add_argument("--no-amp",action="store_true"); a=p.parse_args()
    if a.alignment_workers<=0: raise ValueError("alignment-workers must be positive")
    pcfg=make_prepare_config(load_runtime_config(a.config,a.override))
    _,records=base.load_cache(a.val_cache)
    if a.max_windows>0: records=records[:min(len(records),a.max_windows)]
    global_num_windows=len(records)
    if not a.preserve_record_order:
        records=sorted(records,key=lambda r:str(window_from_record(r).scene_name))
    if a.num_shards<=0 or not 0<=a.shard_index<a.num_shards:
        raise ValueError("invalid shard specification")
    if a.num_shards>1:
        n=len(records)
        lo=n*a.shard_index//a.num_shards
        hi=n*(a.shard_index+1)//a.num_shards
        records=records[lo:hi]
    if not records: raise RuntimeError("empty validation cache shard")
    device=torch.device(a.device if a.device!="cuda" or torch.cuda.is_available() else "cpu")
    amp=device.type=="cuda" and not a.no_amp
    base_ck,base_model,_=full._load_model(a.base_checkpoint,CLEAN_PROTOCOL,device)
    innov_ck,innovation=_load_innovation(a.innovation_checkpoint,device)
    if int(innov_ck["architecture"]["vertical_bins"])!=int(pcfg.grid.shape_hwd[2]):
        raise RuntimeError("innovation checkpoint vertical bins do not match runtime grid")
    source=CachedSource(a.dataroot,info_pkl=a.info_pkl,verbose=False)
    strong_cfg=StrongW2DetConfig(free_label=int(pcfg.free_label))
    raw_by_variant={v:_new_raw() for v in VARIANTS}
    proposed=added=windows_with_additions=0; started=time.perf_counter()
    for wi,rec in enumerate(records,start=1):
        w=window_from_record(rec); raw=load_nuscenes_window_raw(source,w,pcfg,include_gt=True)
        state=_prepare_record(rec,source,pcfg,strong_cfg,device); _stage_gpu_inputs(state,device)
        try: pred_all=_forecast_once(base_model,state,pcfg,strong_cfg,device)
        finally: _release_gpu_inputs(state)
        sem,geo,_,static_all=build_future_aligned_history_and_static_memory(
            raw["history_occ"],
            raw["history_observed"],
            raw["history_poses"],
            raw["future_poses"],
            grid=pcfg.grid,
            free_label=int(pcfg.free_label),
            dynamic_class_ids=tuple(int(x) for x in DYNAMIC_CLASS_IDS),
            workers=int(a.alignment_workers),
        )
        explained=[]
        for fi in range(len(raw["future_poses"])):
            pred=np.asarray(pred_all[fi],dtype=np.uint8)
            explained.append(
                protected_add_only(
                    pred,
                    static_all[fi],
                    free_label=int(pcfg.free_label),
                )
            )
        explained=np.stack(explained).astype(np.uint8)
        sem_t=torch.from_numpy(sem[None]).to(device); geo_t=torch.from_numpy(geo[None]).to(device)
        base_t=torch.from_numpy(base_explained_bev(explained,free_label=int(pcfg.free_label))[None]).to(device)
        with torch.inference_mode(),_autocast(device,amp): out=innovation(sem_t,geo_t,base_t)
        proposal=decode_innovation(out,free_label=int(pcfg.free_label),add_threshold=a.add_threshold,vertical_threshold=a.vertical_threshold)[0].cpu().numpy().astype(np.uint8)
        proposed+=int((proposal!=int(pcfg.free_label)).sum()); final=[]; window_added=0
        for fi in range(len(explained)):
            f=protected_add_only(explained[fi],proposal[fi],free_label=int(pcfg.free_label))
            n=int(((f!=int(pcfg.free_label))&(explained[fi]==int(pcfg.free_label))).sum())
            added+=n; window_added+=n; final.append(f)
        windows_with_additions+=int(window_added>0)
        for hi,h in enumerate(HORIZONS):
            fi=REPORT[h]; gt=np.asarray(raw["future_gt_occ"][fi],dtype=np.uint8); ftok=str(w.future_tokens[fi])
            moving,_,_=gt_moving_support_for_horizon(source.nusc,str(w.t0_token),ftok,h,grid=pcfg.grid)
            for name,pred in (("v18",np.asarray(pred_all[fi],dtype=np.uint8)),("v18_static",explained[fi]),("v18_static_innovation",final[fi])):
                _update(raw_by_variant[name],hi,pred,gt,moving,int(pcfg.free_label))
        if wi==1 or wi%25==0 or wi==len(records):
            print(f"v19_innovation_eval {wi}/{len(records)} rate={wi/max(time.perf_counter()-started,1e-9):.3f} win/s",flush=True)
    metrics={v:_finalize(r) for v,r in raw_by_variant.items()}
    elapsed=max(time.perf_counter()-started,1e-9)
    result={"protocol":PROTOCOL,"num_windows":len(records),"global_num_windows_before_shard":global_num_windows,"num_shards":int(a.num_shards),"shard_index":int(a.shard_index),"base_checkpoint":str(Path(a.base_checkpoint).resolve()),"base_checkpoint_epoch":int(base_ck.get("epoch",-1)),"innovation_checkpoint":str(Path(a.innovation_checkpoint).resolve()),"innovation_epoch":int(innov_ck.get("epoch",-1)),"future_gt_used_for_prediction":False,"metrics":metrics,"delta_vs_v18":{v:_delta(metrics[v],metrics["v18"]) for v in VARIANTS if v!="v18"},"delta_innovation_vs_static":_delta(metrics["v18_static_innovation"],metrics["v18_static"]),"proposal_audit":{"proposed_voxels":proposed,"added_voxels_after_protection":added,"windows_with_additions":windows_with_additions},"raw_counts":{v:{k:np.asarray(x).tolist() for k,x in raw_by_variant[v].items()} for v in VARIANTS},"timing":{"elapsed_s":elapsed,"windows_per_s":len(records)/elapsed}}
    op=Path(a.output); op.parent.mkdir(parents=True,exist_ok=True); op.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print("\n=== V19 INNOVATION FROZEN-BASE EVAL ===")
    for v in VARIANTS:
        m=metrics[v]; print(f"{v:24s} IoU={m['IoU']:.3f} mIoU={m['mIoU']:.3f} MovMacro={m['MovingMacro']:.3f} MovMicro={m['MovingMicro']:.3f}")
    print("innovation_vs_static",json.dumps(result["delta_innovation_vs_static"])); print("proposal_audit",json.dumps(result["proposal_audit"])); print(f"saved {op}")
if __name__=="__main__": main()
