#!/usr/bin/env python3
"""Tiny/small diagnostic trainer. Use train_full.py for final experiments."""
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];UP=ROOT/'upstream_occfm';sys.path[:0]=[str(UP),str(ROOT)]
import torch
from torch.utils.data import DataLoader
from real_motion.dataset import RealMotionCacheDataset,collate_real_motion,ShardShuffleSampler
from real_motion.windows import WindowPlanner,crop_windows,window_coverage
from real_motion.models import MotionWindowFlowMatching,RealMotionWindowCFM
from real_motion.checkpoint import load_shape_safe
from real_motion.runtime_config import add_config_args,load_runtime_config,get_cfg,save_resolved_config

def load_empty(path):
    o=torch.load(path,map_location='cpu',weights_only=False)
    if isinstance(o,dict):o=o.get('empty_latent',o.get('latent'))
    if not torch.is_tensor(o) or o.ndim!=3:raise ValueError('empty latent must be [C,H,W]')
    return o

def main():
    p=argparse.ArgumentParser();add_config_args(p);p.add_argument('--cache',required=True);p.add_argument('--empty-latent',required=True);p.add_argument('--upstream-ckpt',required=True);p.add_argument('--output',required=True);p.add_argument('--window',type=int,default=None);p.add_argument('--max-windows',type=int,default=None);p.add_argument('--batch-size',type=int,default=None);p.add_argument('--steps',type=int,default=None);p.add_argument('--lr',type=float,default=None);p.add_argument('--num-workers',type=int,default=4);p.add_argument('--min-window-coverage',type=float,default=None);p.add_argument('--allow-low-coverage',action='store_true');p.add_argument('--amp',action='store_true');a=p.parse_args();cfg=load_runtime_config(a.config,a.override);device='cuda' if torch.cuda.is_available() else 'cpu';wh=int(a.window or get_cfg(cfg,'MODEL.WINDOW_HW',[20,20])[0]);maxw=int(a.max_windows or get_cfg(cfg,'MODEL.MAX_WINDOWS',8));bs=int(a.batch_size or get_cfg(cfg,'OPTIMIZATION.TINY.BATCH_SIZE',2));steps=int(a.steps or get_cfg(cfg,'OPTIMIZATION.TINY.STEPS',2000));lr=float(a.lr or get_cfg(cfg,'OPTIMIZATION.TINY.BASE_LR',2e-5));mincov=float(a.min_window_coverage or get_cfg(cfg,'MODEL.MIN_WINDOW_COVERAGE',0.95));ds=RealMotionCacheDataset(a.cache)
    if len(ds)==0:raise RuntimeError('empty cache dataset')
    sampler=ShardShuffleSampler(ds,seed=20260826) if ds.sharded else None;dl=DataLoader(ds,batch_size=bs,shuffle=sampler is None,sampler=sampler,num_workers=a.num_workers,collate_fn=collate_real_motion,drop_last=False,pin_memory=True);empty=load_empty(a.empty_latent);tr=MotionWindowFlowMatching(in_channels=16,out_channels=16,model_channels=128,channel_multi=[2,4],input_size=[wh,wh],trajectory_length=6,init_kernel_size=7,init_3d_conv_channels=64,attn_dim=32,temporal_attn_head=8,spatial_attn_head=8,prior_channels=int(get_cfg(cfg,'MODEL.PRIOR_CHANNELS',32)));model=RealMotionWindowCFM(tr,rescale_factor=float(get_cfg(cfg,'MODEL.RESCALE_FACTOR',10)),sample_steps=int(get_cfg(cfg,'MODEL.SAMPLE_STEPS',10)),alpha_shift=float(get_cfg(cfg,'MODEL.ALPHA_SHIFT',3))).to(device);rep=load_shape_safe(model.transition,a.upstream_ckpt,verbose=True);opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=float(get_cfg(cfg,'OPTIMIZATION.TINY.WEIGHT_DECAY',0.01)));scaler=torch.amp.GradScaler('cuda',enabled=a.amp and device=='cuda');planner=WindowPlanner((wh,wh),maxw);step=0;model.train()
    while step<steps:
        before=step
        for batch in dl:
            batch={k:(v.to(device,non_blocking=True) if torch.is_tensor(v) else v) for k,v in batch.items()};plan=planner.plan(batch['generation_support'],context_support=batch.get('planning_support'));cov=window_coverage(batch['generation_support'],plan);mc=float(cov.min())
            if mc<mincov and not a.allow_low_coverage:raise RuntimeError(f'future support coverage {mc:.3f} < {mincov:.3f}')
            hist=crop_windows(batch['moving_history_latent'],plan);fut=crop_windows(batch['future_dynamic_target_latent'],plan);sta=crop_windows(batch['static_future_latent'],plan);kta=crop_windows(batch['kta_future_latent'],plan);mask=crop_windows(batch['generation_support'].unsqueeze(2),plan);B,K=hist.shape[:2];F=fut.shape[2];ef=empty.to(device=device,dtype=fut.dtype)[None,None].expand(B,F,-1,-1,-1);ew=crop_windows(ef,plan);valid=plan.valid.reshape(-1)
            def flat(x):return x.reshape(B*K,*x.shape[2:])[valid]
            hist,fut,sta,kta,mask,ew=map(flat,(hist,fut,sta,kta,mask,ew));orig=plan.origins.reshape(B*K,2)[valid]
            if hist.shape[0]==0:continue
            traj=batch.get('trajectory')
            if traj is not None:traj=traj[:,None].expand(B,K,*traj.shape[1:]).reshape(B*K,*traj.shape[1:])[valid]
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=a.amp):loss,info=model.flow_loss(hist,fut,torch.cat([sta,kta],2),mask,known_future=ew,trajectory=traj,window_origins=orig)
            scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),float(get_cfg(cfg,'OPTIMIZATION.TINY.GRAD_NORM_CLIP',5)));scaler.step(opt);scaler.update();step+=1
            if step%20==0:print(f'step={step} loss={loss.item():.6f} coverage={mc:.4f}')
            if step>=steps:break
        if step==before:raise RuntimeError('entire epoch produced no active future windows')
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);torch.save({'state_dict':model.state_dict(),'args':vars(a),'checkpoint_reuse':rep,'cache_metadata':ds.metadata,'empty_latent':empty.cpu(),'resolved_config':cfg},out);save_resolved_config(cfg,out.with_suffix('.resolved.yaml'));print('saved',out)
if __name__=='__main__':main()
