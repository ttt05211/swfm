#!/usr/bin/env python3
"""Full SWFM trainer: shard-aware DDP + OccFM-style optimization + L40S runtime tuning."""
import argparse, math, os, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; UP=ROOT/'upstream_occfm'
sys.path[:0]=[str(UP),str(ROOT)]

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.swa_utils import AveragedModel, get_ema_avg_fn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from real_motion.dataset import (
    RealMotionCacheDataset, collate_real_motion, ShardShuffleSampler,
    DistributedShardSampler,
)
from real_motion.windows import WindowPlan, WindowPlanner, crop_windows, window_coverage
from real_motion.models import MotionWindowFlowMatching, RealMotionWindowCFM
from real_motion.checkpoint import load_shape_safe
from real_motion.runtime_config import add_config_args, load_runtime_config, get_cfg, save_resolved_config
from real_motion.perf import (
    configure_cuda_runtime, autocast_context, needs_grad_scaler,
    dataloader_kwargs, maybe_compile, cuda_device_summary,
)


class TrainWrapper(nn.Module):
    def __init__(self, cfm):
        super().__init__(); self.cfm=cfm
    def forward(self, history, future, prior, mask, empty_win, trajectory, origins):
        return self.cfm.flow_loss(
            history, future, prior, mask, known_future=empty_win,
            trajectory=trajectory, window_origins=origins,
        )


def make_cfm(cfg):
    wh,ww=map(int,get_cfg(cfg,'MODEL.WINDOW_HW'))
    if wh!=ww: raise ValueError('current OccFM transition expects square windows')
    tr=MotionWindowFlowMatching(
        in_channels=16,out_channels=16,model_channels=128,channel_multi=[2,4],
        input_size=[wh,ww],trajectory_length=6,init_kernel_size=7,
        init_3d_conv_channels=64,attn_dim=32,temporal_attn_head=8,
        spatial_attn_head=8,prior_channels=int(get_cfg(cfg,'MODEL.PRIOR_CHANNELS',32)),
    )
    return RealMotionWindowCFM(
        tr,rescale_factor=float(get_cfg(cfg,'MODEL.RESCALE_FACTOR',10.0)),
        sample_steps=int(get_cfg(cfg,'MODEL.SAMPLE_STEPS',10)),
        alpha_shift=float(get_cfg(cfg,'MODEL.ALPHA_SHIFT',3.0)),
    )


def load_empty(path):
    obj=torch.load(path,map_location='cpu',weights_only=False)
    if isinstance(obj,dict): obj=obj.get('empty_latent',obj.get('latent'))
    if not torch.is_tensor(obj) or obj.ndim!=3: raise ValueError('empty latent must be [C,H,W]')
    return obj


def split_weight(named_params):
    decay=[]; no_decay=[]
    for name,p in named_params:
        if not p.requires_grad: continue
        (no_decay if 'ln' in name.lower() or 'norm' in name.lower() else decay).append(p)
    return decay,no_decay


def make_scheduler(optimizer,warmup,total_steps,min_ratio):
    def fn(step):
        if warmup>0 and step<warmup: return step/max(warmup,1)
        progress=(step-warmup)/max(total_steps-warmup,1)
        progress=min(max(progress,0.0),1.0)
        cosine=0.5*(1+math.cos(math.pi*progress))
        return min_ratio+(1-min_ratio)*cosine
    return torch.optim.lr_scheduler.LambdaLR(optimizer,fn)


def make_optimizer(param_groups, lr, cfg, device):
    kwargs={"lr":lr}
    if torch.device(device).type=='cuda' and bool(get_cfg(cfg,'RUNTIME.FUSED_ADAMW',True)):
        kwargs['fused']=True
    try:
        return torch.optim.AdamW(param_groups,**kwargs)
    except (TypeError,RuntimeError) as exc:
        if kwargs.pop('fused',None) is not None:
            print(f'[WARN] fused AdamW unavailable ({exc}); falling back to eager AdamW')
            return torch.optim.AdamW(param_groups,**kwargs)
        raise


def prepare_batch(batch,planner,empty,device,min_cov):
    required=batch['generation_support']
    if 'window_origins' in batch and 'window_valid' in batch:
        plan_cpu=WindowPlan(
            batch['window_origins'].long(),batch['window_valid'].bool(),
            planner.window_hw,tuple(required.shape[-2:]),
        )
    else:
        plan_cpu=planner.plan(required,context_support=batch.get('planning_support'))
    cov=window_coverage(required,plan_cpu); mc=float(cov.min())
    if mc<min_cov: raise RuntimeError(f'future generation-support coverage {mc:.3f} < {min_cov:.3f}')
    batch={k:(v.to(device,non_blocking=True) if torch.is_tensor(v) else v) for k,v in batch.items()}
    plan=WindowPlan(
        plan_cpu.origins.to(device,non_blocking=True),
        plan_cpu.valid.to(device,non_blocking=True),
        plan_cpu.window_hw,plan_cpu.full_hw,
    )
    hist=crop_windows(batch['moving_history_latent'],plan)
    fut=crop_windows(batch['future_dynamic_target_latent'],plan)
    sta=crop_windows(batch['static_future_latent'],plan)
    kta=crop_windows(batch['kta_future_latent'],plan)
    mask=crop_windows(batch['generation_support'].unsqueeze(2),plan)
    B,K=hist.shape[:2]; F=fut.shape[2]
    ef=empty.to(device=device,dtype=fut.dtype)[None,None].expand(B,F,-1,-1,-1)
    ew=crop_windows(ef,plan); valid=plan.valid.reshape(-1)
    if not bool(valid.any()): raise RuntimeError('full-training cache produced a batch with no active windows')
    def flat(x): return x.reshape(B*K,*x.shape[2:])[valid]
    hist,fut,sta,kta,mask,ew=map(flat,(hist,fut,sta,kta,mask,ew))
    orig=plan.origins.reshape(B*K,2)[valid]; traj=batch.get('trajectory')
    if traj is not None:
        traj=traj[:,None].expand(B,K,*traj.shape[1:]).reshape(B*K,*traj.shape[1:])[valid]
    return hist,fut,torch.cat([sta,kta],2),mask,ew,traj,orig,mc


def make_samplers(train_ds,val_ds,distributed,world,rank,seed=20260826):
    if distributed:
        ts=(DistributedShardSampler(train_ds,world,rank,shuffle=True,seed=seed)
            if train_ds.sharded else DistributedSampler(train_ds,num_replicas=world,rank=rank,shuffle=True,drop_last=False,seed=seed))
        vs=(DistributedShardSampler(val_ds,world,rank,shuffle=False,seed=seed)
            if val_ds.sharded else DistributedSampler(val_ds,num_replicas=world,rank=rank,shuffle=False,drop_last=False))
    else:
        ts=ShardShuffleSampler(train_ds,seed=seed) if train_ds.sharded else None
        vs=None
    return ts,vs


@torch.no_grad()
def validate(model,loader,planner,empty,device,min_cov,epoch,cfg,amp_override):
    model.eval(); total=torch.zeros(2,device=device,dtype=torch.float64); torch.manual_seed(123456+epoch)
    for batch in loader:
        args=prepare_batch(batch,planner,empty,device,min_cov)
        with autocast_context(cfg,device,amp_override): loss,_=model(*args[:7])
        weight=args[3].sum().double()*args[1].shape[2]
        total[0]+=loss.detach().double()*weight; total[1]+=weight
    if dist.is_initialized(): dist.all_reduce(total,op=dist.ReduceOp.SUM)
    model.train(); return float((total[0]/total[1].clamp_min(1)).item())


def main():
    p=argparse.ArgumentParser(); add_config_args(p)
    p.add_argument('--train-cache',required=True); p.add_argument('--val-cache',required=True)
    p.add_argument('--empty-latent',required=True); p.add_argument('--upstream-ckpt',required=True)
    p.add_argument('--output-dir',required=True); p.add_argument('--resume',default=None)
    p.add_argument('--amp',action=argparse.BooleanOptionalAction,default=None,
                   help='override RUNTIME.AMP.ENABLED; L40S default is BF16 enabled')
    a=p.parse_args(); cfg=load_runtime_config(a.config,a.override)

    local_rank=int(os.environ.get('LOCAL_RANK','0')); world=int(os.environ.get('WORLD_SIZE','1')); distributed=world>1
    if distributed: dist.init_process_group('nccl'); torch.cuda.set_device(local_rank)
    device=torch.device('cuda',local_rank) if torch.cuda.is_available() else torch.device('cpu')
    rank=dist.get_rank() if distributed else 0
    hw=configure_cuda_runtime(cfg,device)
    if rank==0: print('runtime device:',hw)

    train_ds=RealMotionCacheDataset(a.train_cache); val_ds=RealMotionCacheDataset(a.val_cache)
    if distributed and not bool(train_ds.metadata.get('filtered_empty_generation_support',False)):
        raise RuntimeError('DDP full training requires cache built with empty-generation-support filtering')
    expected_hw=list(map(int,get_cfg(cfg,'MODEL.WINDOW_HW',[20,20]))); expected_k=int(get_cfg(cfg,'MODEL.MAX_WINDOWS',8))
    for name,ds in [('train',train_ds),('val',val_ds)]:
        if ds.metadata.get('precomputed_window_plan'):
            if list(ds.metadata.get('window_plan_hw') or [])!=expected_hw or int(ds.metadata.get('window_plan_max_windows',-1))!=expected_k:
                raise RuntimeError(f'{name} cache window plan does not match current YAML; rebuild cache instead of silently reusing stale origins')

    bpg=int(get_cfg(cfg,'OPTIMIZATION.FULL.BATCH_SIZE_PER_GPU',4)); workers=int(get_cfg(cfg,'OPTIMIZATION.FULL.NUM_WORKERS',8))
    ts,vs=make_samplers(train_ds,val_ds,distributed,world,rank)
    loader_extra=dataloader_kwargs(cfg,workers)
    tl=DataLoader(train_ds,batch_size=bpg,shuffle=ts is None,sampler=ts,num_workers=workers,
                  collate_fn=collate_real_motion,drop_last=False,**loader_extra)
    vl=DataLoader(val_ds,batch_size=bpg,shuffle=False,sampler=vs,num_workers=workers,
                  collate_fn=collate_real_motion,drop_last=False,**loader_extra)
    if len(tl)==0: raise RuntimeError('empty full-training loader')

    cfm=make_cfm(cfg).to(device); wrapper=TrainWrapper(cfm).to(device)
    decay,no_decay=split_weight(wrapper.named_parameters())
    base_lr=float(get_cfg(cfg,'OPTIMIZATION.FULL.BASE_LR',2e-5)); lr=base_lr*bpg*world
    wd=float(get_cfg(cfg,'OPTIMIZATION.FULL.WEIGHT_DECAY',.01))
    opt=make_optimizer([{'params':decay,'weight_decay':wd},{'params':no_decay,'weight_decay':0.0}],lr,cfg,device)
    epochs=int(get_cfg(cfg,'OPTIMIZATION.FULL.NUM_EPOCHS',200))
    sched=make_scheduler(opt,int(get_cfg(cfg,'OPTIMIZATION.FULL.WARMUP_STEPS',1000)),
                         epochs*len(tl),float(get_cfg(cfg,'OPTIMIZATION.FULL.MIN_LR_RATIO',.2)))
    ema=AveragedModel(wrapper,avg_fn=get_ema_avg_fn(float(get_cfg(cfg,'OPTIMIZATION.FULL.EMA_DECAY',.9999))),use_buffers=True).to(device)
    start=0; step=0; best=float('inf')
    if a.resume:
        ck=torch.load(a.resume,map_location='cpu',weights_only=False)
        wrapper.load_state_dict(ck['wrapper_state_dict'],strict=True); opt.load_state_dict(ck['optimizer'])
        sched.load_state_dict(ck['scheduler']); ema.load_state_dict(ck['ema'])
        start=int(ck['epoch'])+1; step=int(ck['global_step']); best=float(ck.get('best_val',best))
    else:
        rep=load_shape_safe(cfm.transition,a.upstream_ckpt,verbose=rank==0); ema.update_parameters(wrapper)
        if rank==0: print('checkpoint reuse:',rep['loaded'],'/',rep['target_total'])

    if distributed:
        ddp_cfg=get_cfg(cfg,'RUNTIME.DDP',{}) or {}
        ddp=DDP(
            wrapper,device_ids=[local_rank],broadcast_buffers=False,find_unused_parameters=False,
            bucket_cap_mb=int(ddp_cfg.get('BUCKET_CAP_MB',50)),
            gradient_as_bucket_view=bool(ddp_cfg.get('GRADIENT_AS_BUCKET_VIEW',True)),
            static_graph=bool(ddp_cfg.get('STATIC_GRAPH',True)),
        )
    else: ddp=wrapper
    train_exec=maybe_compile(ddp,cfg,'train')

    wh,ww=map(int,get_cfg(cfg,'MODEL.WINDOW_HW')); planner=WindowPlanner((wh,ww),int(get_cfg(cfg,'MODEL.MAX_WINDOWS',8)))
    empty=load_empty(a.empty_latent).to(device,non_blocking=True); mincov=float(get_cfg(cfg,'MODEL.MIN_WINDOW_COVERAGE',.95))
    clip=float(get_cfg(cfg,'OPTIMIZATION.FULL.GRAD_NORM_CLIP',5.0)); save_every=int(get_cfg(cfg,'OPTIMIZATION.FULL.SAVE_EVERY_EPOCHS',5))
    out=Path(a.output_dir); ckdir=out/'ckpt'
    if rank==0: ckdir.mkdir(parents=True,exist_ok=True); save_resolved_config(cfg,out/'resolved_config.yaml')
    use_scaler=needs_grad_scaler(cfg,device,a.amp)
    scaler=torch.amp.GradScaler('cuda',enabled=use_scaler)

    for epoch in range(start,epochs):
        if ts is not None and hasattr(ts,'set_epoch'): ts.set_epoch(epoch)
        if vs is not None and hasattr(vs,'set_epoch'): vs.set_epoch(epoch)
        ddp.train()
        for batch in tl:
            args=prepare_batch(batch,planner,empty,device,mincov); opt.zero_grad(set_to_none=True)
            with autocast_context(cfg,device,a.amp): loss,_=train_exec(*args[:7])
            if use_scaler:
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(wrapper.parameters(),clip); scaler.step(opt); scaler.update()
            else:
                loss.backward(); torch.nn.utils.clip_grad_norm_(wrapper.parameters(),clip); opt.step()
            sched.step(); ema.update_parameters(wrapper); step+=1
            if rank==0 and step%50==0:
                mem=(torch.cuda.max_memory_allocated(device)/(1024**3) if device.type=='cuda' else 0.0)
                print(f'epoch={epoch} step={step} loss={loss.item():.6f} lr={sched.get_last_lr()[0]:.3e} peak_GB={mem:.2f}')
        val=validate(ema.module,vl,planner,empty,device,mincov,epoch,cfg,a.amp)
        if rank==0:
            state={'wrapper_state_dict':wrapper.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'ema':ema.state_dict(),'epoch':epoch,'global_step':step,'best_val':min(best,val),'cache_metadata':train_ds.metadata,'empty_latent':empty.cpu(),'state_dict':ema.module.cfm.state_dict(),'raw_state_dict':cfm.state_dict(),'resolved_config':cfg,'hardware':cuda_device_summary(device)}
            torch.save(state,ckdir/'last.pt')
            if (epoch+1)%save_every==0: torch.save(state,ckdir/f'epoch_{epoch+1:04d}.pt')
            if val<best: best=val; state['best_val']=best; torch.save(state,ckdir/'best.pt')
            print(f'epoch={epoch} val={val:.6f} best={best:.6f}')
        if distributed: dist.barrier()
    if distributed: dist.destroy_process_group()


if __name__=='__main__': main()
