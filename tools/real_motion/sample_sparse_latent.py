#!/usr/bin/env python3
"""Sample sparse future WM-dynamic latents with one global noise canvas."""
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];UP=ROOT/'upstream_occfm';sys.path[:0]=[str(UP),str(ROOT)]
import torch
from torch.utils.data import DataLoader
from real_motion.dataset import RealMotionCacheDataset,collate_real_motion
from real_motion.windows import WindowPlanner,crop_windows,scatter_windows,window_coverage
from real_motion.runtime_config import add_config_args,load_runtime_config,get_cfg,save_resolved_config,validate_runtime_config
from tools.real_motion.infer_nuscenes import make_model

def load_empty(path):
    o=torch.load(path,map_location='cpu',weights_only=False)
    if isinstance(o,dict):o=o.get('empty_latent',o.get('latent'))
    if not torch.is_tensor(o) or o.ndim!=3:raise ValueError('empty latent must be [C,H,W]')
    return o

def main():
    p=argparse.ArgumentParser();add_config_args(p);p.add_argument('--cache',required=True);p.add_argument('--ckpt',required=True);p.add_argument('--output',required=True);p.add_argument('--empty-latent',default=None);p.add_argument('--batch-size',type=int,default=1);p.add_argument('--seed',type=int,default=0);p.add_argument('--allow-low-coverage',action='store_true');a=p.parse_args();fallback=load_runtime_config(a.config,a.override);dev='cuda' if torch.cuda.is_available() else 'cpu';ds=RealMotionCacheDataset(a.cache);dl=DataLoader(ds,batch_size=a.batch_size,shuffle=False,collate_fn=collate_real_motion,drop_last=False);ck=torch.load(a.ckpt,map_location='cpu',weights_only=False);cfg=ck.get('resolved_config',fallback);validate_runtime_config(cfg);wh,ww=map(int,get_cfg(cfg,'MODEL.WINDOW_HW',[20,20]));maxw=int(get_cfg(cfg,'MODEL.MAX_WINDOWS',8));mincov=float(get_cfg(cfg,'MODEL.MIN_WINDOW_COVERAGE',.95));model=make_model(wh,cfg).to(dev);model.load_state_dict(ck['state_dict'],strict=True);model.eval();empty=load_empty(a.empty_latent) if a.empty_latent else ck.get('empty_latent')
    if empty is None:raise KeyError('checkpoint lacks empty_latent')
    planner=WindowPlanner((wh,ww),maxw);outs=[];g=torch.Generator(device=dev);g.manual_seed(a.seed)
    with torch.no_grad():
        for batch in dl:
            batch={k:(v.to(dev) if torch.is_tensor(v) else v) for k,v in batch.items()};plan=planner.plan(batch['generation_support'],context_support=batch.get('planning_support'));mc=float(window_coverage(batch['generation_support'],plan).min())
            if mc<mincov and not a.allow_low_coverage:raise RuntimeError(f'future support coverage {mc:.3f} < {mincov:.3f}')
            hist=crop_windows(batch['moving_history_latent'],plan);sta=crop_windows(batch['static_future_latent'],plan);kta=crop_windows(batch['kta_future_latent'],plan);active=crop_windows(batch['generation_support'].unsqueeze(2),plan);B,K=hist.shape[:2];F=sta.shape[2];C=sta.shape[3];valid=plan.valid.reshape(-1);ef=empty.to(dev,dtype=sta.dtype)[None,None].expand(B,F,-1,-1,-1);ew=crop_windows(ef,plan);nw=crop_windows(torch.randn((B,F,C,50,50),device=dev,dtype=sta.dtype,generator=g),plan)
            if not bool(valid.any()):outs.append(ef.cpu());continue
            def flat(x):return x.reshape(B*K,*x.shape[2:])[valid]
            fh,fs,fk,fa,fe,fn=map(flat,(hist,sta,kta,active,ew,nw));orig=plan.origins.reshape(B*K,2)[valid];traj=batch.get('trajectory')
            if traj is not None:traj=traj[:,None].expand(B,K,*traj.shape[1:]).reshape(B*K,*traj.shape[1:])[valid]
            pred=model.sample(fh,tuple(fs.shape[:2])+(C,wh,ww),torch.cat([fs,fk],2),fa,fe,trajectory=traj,window_origins=orig,initial_noise=fn);pred=torch.where(fa.bool().expand_as(pred),pred,fe);pad=torch.zeros(B*K,F,C,wh,ww,device=dev,dtype=pred.dtype);pad[valid]=pred;outs.append(scatter_windows(pad.reshape(B,K,F,C,wh,ww),plan,base=ef).cpu())
    out=torch.cat(outs,0) if outs else torch.empty(0);op=Path(a.output);op.parent.mkdir(parents=True,exist_ok=True);torch.save({'future_dynamic_latent':out,'seed':a.seed},op);save_resolved_config(cfg,op.with_suffix('.resolved.yaml'));print('saved',a.output,tuple(out.shape))
if __name__=='__main__':main()
