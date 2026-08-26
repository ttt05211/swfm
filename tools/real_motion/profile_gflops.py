#!/usr/bin/env python3
"""Profile supported PyTorch FLOPs for one online SWFM inference window."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];UP=ROOT/'upstream_occfm';sys.path[:0]=[str(UP),str(ROOT)]
import torch
from torch.profiler import profile,ProfilerActivity
from real_motion.nuscenes_adapter import NuScenesWindowSource
from real_motion.prepared import prepare_nuscenes_window
from real_motion.occfm_io import load_official_vae,OccFMVAEAdapter
from real_motion.windows import WindowPlanner
from real_motion.runtime_config import add_config_args,load_runtime_config,get_cfg,save_resolved_config,make_prepare_config,validate_runtime_config
from tools.real_motion.infer_nuscenes import make_model,infer_one

def main():
    p=argparse.ArgumentParser();add_config_args(p);p.add_argument('--dataroot',required=True);p.add_argument('--info-pkl',required=True);p.add_argument('--vae-ckpt',required=True);p.add_argument('--sparse-ckpt',required=True);p.add_argument('--window-index',type=int,default=0);p.add_argument('--device',default='cuda');p.add_argument('--output',required=True);a=p.parse_args();cfg=load_runtime_config(a.config,a.override);source=NuScenesWindowSource(a.dataroot,info_pkl=a.info_pkl,verbose=False);target=None
    for i,w in enumerate(source.iter_windows(history=6,future=6)):
        if i==a.window_index:target=w;break
    if target is None:raise IndexError('window-index out of range')
    ck=torch.load(a.sparse_ckpt,map_location='cpu',weights_only=False);rcfg=ck.get('resolved_config',cfg);validate_runtime_config(rcfg);wh,ww=map(int,get_cfg(rcfg,'MODEL.WINDOW_HW',[20,20]));maxw=int(get_cfg(rcfg,'MODEL.MAX_WINDOWS',8));extra=int(ck.get('cache_metadata',{}).get('latent_extra_radius',get_cfg(rcfg,'MOTION.LATENT_EXTRA_RADIUS',1)));vae,_=load_official_vae(UP,a.vae_ckpt,a.device);va=OccFMVAEAdapter(vae);model=make_model(wh,rcfg).to(a.device);model.load_state_dict(ck['state_dict'],strict=True);model.eval();empty=ck.get('empty_latent')
    if empty is None:raise KeyError('checkpoint lacks empty_latent')
    planner=WindowPlanner((wh,ww),maxw);sample=prepare_nuscenes_window(source,target,make_prepare_config(rcfg),include_gt=False);activities=[ProfilerActivity.CPU]+([ProfilerActivity.CUDA] if str(a.device).startswith('cuda') else [])
    with torch.no_grad():
        infer_one(sample,va,model,empty.to(a.device),planner,a.device,extra,seed=0)
        if torch.cuda.is_available():torch.cuda.synchronize()
        with profile(activities=activities,with_flops=True,record_shapes=True) as prof:_,meta=infer_one(sample,va,model,empty.to(a.device),planner,a.device,extra,seed=1)
        if torch.cuda.is_available():torch.cuda.synchronize()
    flops=sum(int(getattr(e,'flops',0) or 0) for e in prof.key_averages());report={'supported_operator_flops':flops,'supported_operator_GFLOPs':flops/1e9,'num_windows':int(meta['num_windows']),'slot_compute_ratio':float(meta['slot_compute_ratio']),'window_hw':[wh,ww],'note':'torch.profiler(with_flops=True) counts supported PyTorch operators only; compare all baselines with identical software/hardware protocol and report latency separately.'};op=Path(a.output);op.parent.mkdir(parents=True,exist_ok=True);op.write_text(json.dumps(report,indent=2),encoding='utf-8');save_resolved_config(rcfg,op.with_suffix('.resolved.yaml'));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
