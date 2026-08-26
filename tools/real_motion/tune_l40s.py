#!/usr/bin/env python3
"""Empirically choose an efficient SWFM batch size on one NVIDIA L40S.

Run this on the real latent cache before the final experiment. It measures the
actual train step (window planning/crop + BF16 forward/backward + AdamW + EMA),
reports peak memory and input-sample throughput, and catches OOM safely.
"""
import argparse,gc,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];UP=ROOT/'upstream_occfm';sys.path[:0]=[str(UP),str(ROOT)]
import torch
from torch.optim.swa_utils import AveragedModel,get_ema_avg_fn
from torch.utils.data import DataLoader
from real_motion.dataset import RealMotionCacheDataset,collate_real_motion,ShardShuffleSampler
from real_motion.runtime_config import add_config_args,load_runtime_config,get_cfg,save_resolved_config
from real_motion.perf import configure_cuda_runtime,autocast_context,dataloader_kwargs,maybe_compile,cuda_device_summary
from real_motion.checkpoint import load_shape_safe
from tools.real_motion.train_full import make_cfm,TrainWrapper,load_empty,split_weight,make_optimizer,prepare_batch

def next_batch(iterator,loader):
    try:return next(iterator),iterator
    except StopIteration:
        iterator=iter(loader);return next(iterator),iterator

def main():
    p=argparse.ArgumentParser();add_config_args(p);p.add_argument('--cache',required=True);p.add_argument('--empty-latent',required=True);p.add_argument('--upstream-ckpt',required=True);p.add_argument('--candidates',default=None,help='e.g. 4,6,8,10,12');p.add_argument('--warmup',type=int,default=3);p.add_argument('--steps',type=int,default=10);p.add_argument('--num-workers',type=int,default=None);p.add_argument('--output',required=True);a=p.parse_args();cfg=load_runtime_config(a.config,a.override)
    if not torch.cuda.is_available():raise RuntimeError('tune_l40s.py requires CUDA')
    device=torch.device('cuda');hw=configure_cuda_runtime(cfg,device);print('runtime device:',hw)
    if 'l40s' not in hw.get('name','').lower():print('[WARN] device is not reported as L40S; results still describe this GPU')
    candidates=[int(x) for x in (a.candidates.split(',') if a.candidates else get_cfg(cfg,'RUNTIME.L40S_BATCH_CANDIDATES',[4,6,8,10,12,16,20,24]))];candidates=sorted(set(candidates));workers=int(a.num_workers if a.num_workers is not None else get_cfg(cfg,'OPTIMIZATION.FULL.NUM_WORKERS',8));ds=RealMotionCacheDataset(a.cache);empty=load_empty(a.empty_latent).to(device,non_blocking=True);wh,ww=map(int,get_cfg(cfg,'MODEL.WINDOW_HW'));from real_motion.windows import WindowPlanner;planner=WindowPlanner((wh,ww),int(get_cfg(cfg,'MODEL.MAX_WINDOWS',8)));mincov=float(get_cfg(cfg,'MODEL.MIN_WINDOW_COVERAGE',.95))
    cfm=make_cfm(cfg).to(device);load_shape_safe(cfm.transition,a.upstream_ckpt,verbose=True);wrapper=TrainWrapper(cfm).to(device);decay,no_decay=split_weight(wrapper.named_parameters());wd=float(get_cfg(cfg,'OPTIMIZATION.FULL.WEIGHT_DECAY',.01));opt=make_optimizer([{'params':decay,'weight_decay':wd},{'params':no_decay,'weight_decay':0.0}],1e-4,cfg,device);ema=AveragedModel(wrapper,avg_fn=get_ema_avg_fn(float(get_cfg(cfg,'OPTIMIZATION.FULL.EMA_DECAY',.9999))),use_buffers=True).to(device);ema.update_parameters(wrapper);train_exec=maybe_compile(wrapper,cfg,'train');rows=[]
    for bs in candidates:
        sampler=ShardShuffleSampler(ds,seed=20260826) if ds.sharded else None;loader=DataLoader(ds,batch_size=bs,shuffle=sampler is None,sampler=sampler,num_workers=workers,collate_fn=collate_real_motion,drop_last=False,**dataloader_kwargs(cfg,workers));it=iter(loader);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(device);ok=True;measured_time=0.0;measured_samples=0;measured_windows=0
        try:
            for step in range(a.warmup+a.steps):
                batch,it=next_batch(it,loader);n=int(batch['generation_support'].shape[0]);torch.cuda.synchronize();t0=time.perf_counter();args=prepare_batch(batch,planner,empty,device,mincov);nwin=int(args[0].shape[0]);opt.zero_grad(set_to_none=True)
                with autocast_context(cfg,device):loss,_=train_exec(*args[:7])
                loss.backward();torch.nn.utils.clip_grad_norm_(wrapper.parameters(),float(get_cfg(cfg,'OPTIMIZATION.FULL.GRAD_NORM_CLIP',5)));opt.step();ema.update_parameters(wrapper);torch.cuda.synchronize();dt=time.perf_counter()-t0
                if step>=a.warmup:measured_time+=dt;measured_samples+=n;measured_windows+=nwin
        except torch.cuda.OutOfMemoryError:
            ok=False;print(f'batch={bs}: OOM');opt.zero_grad(set_to_none=True);torch.cuda.empty_cache()
        peak=torch.cuda.max_memory_allocated(device)/(1024**3);reserved=torch.cuda.max_memory_reserved(device)/(1024**3);row={'batch_size_per_gpu':bs,'ok':ok,'peak_memory_gb':peak,'peak_reserved_gb':reserved}
        if ok:
            row.update({'samples_per_s':measured_samples/max(measured_time,1e-12),'transition_windows_per_s':measured_windows/max(measured_time,1e-12),'avg_transition_windows_per_step':measured_windows/max(a.steps,1),'step_ms':1000*measured_time/max(a.steps,1)});print(row)
        rows.append(row);del loader,it;gc.collect()
        if not ok:break
    valid=[r for r in rows if r['ok']]
    best=max(valid,key=lambda r:r['samples_per_s']) if valid else None
    headroom=float(get_cfg(cfg,'RUNTIME.L40S_MEMORY_HEADROOM_GB',3.0));total=float(hw.get('total_memory_gb',0.0))
    stable=[r for r in valid if not total or r['peak_reserved_gb'] <= total-headroom]
    stable_best=max(stable,key=lambda r:r['samples_per_s']) if stable else best
    report={'hardware':cuda_device_summary(device),'runtime_compile_train':bool(get_cfg(cfg,'RUNTIME.COMPILE.TRAIN',False)),'candidates':rows,'raw_best_by_throughput':best,'recommended_with_memory_headroom':stable_best,'memory_headroom_gb':headroom,'note':'Freeze the selected batch size in YAML/--override before the paper run; effective LR scales with batch size exactly as train_full.py. Prefer the headroom recommendation over the last barely-fitting batch.'}
    op=Path(a.output);op.parent.mkdir(parents=True,exist_ok=True);op.write_text(json.dumps(report,indent=2),encoding='utf-8');save_resolved_config(cfg,op.with_suffix('.resolved.yaml'));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
