#!/usr/bin/env python3
from __future__ import annotations
import argparse,contextlib,copy,json,math,subprocess,sys,time
from pathlib import Path
import numpy as np,torch,torch.distributed as dist,yaml
from torch.nn.parallel import DistributedDataParallel as DDP
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from real_motion.nuscenes_adapter import NuScenesWindowSource
from real_motion.motion_transport_v1.config import load_config,get
from real_motion.motion_transport_v1.data import ManifestDataset,load_manifest
from real_motion.motion_transport_v1.model import MotionTransportV1
from real_motion.motion_transport_v1.engine import init_distributed,barrier,all_sum,seed_all,sync_model_from_rank0,optimizer_for,calibrate_lambda,rank_epoch_indices,prepare_scene,forward_losses,save_checkpoint
from real_motion.motion_transport_v1.losses import motion_pair_count,lambda_at
from real_motion.motion_transport_v1.routing import route_sources,msp_checkpoint_provenance,sha256_file
from real_motion.motion_transport_v1.source_adapter import decompose_strong_sources
from real_motion.motion_transport_v1.evaluation import evaluate
from real_motion.motion_transport_v1.ema import WarmupEMA
from real_motion.motion_transport_v1.server_acceptance import formal_server_environment
def _git():
    try:return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    except Exception:return 'unknown'
def _sync(ctx):
    if ctx.device.type=='cuda':torch.cuda.synchronize(ctx.device)
    if ctx.world_size>1:dist.barrier()
def _max(v,ctx):
    t=torch.tensor(float(v),device=ctx.device,dtype=torch.float64)
    if ctx.world_size>1:dist.all_reduce(t,op=dist.ReduceOp.MAX)
    return float(t.item())
def _representative(ds,pipe,ctx,cfg,seed,scan=64):
    ids,_=rank_epoch_indices(len(ds),ctx,0,seed);rows=[]
    for idx in ids[:min(scan,len(ids))]:
        _,c=ds[idx];d=decompose_strong_sources(c,grid=pipe.grid,cfg=pipe.strong_cfg,frame_dt_s=float(get(cfg,'input.dt_seconds',.5)),crop_radius_limit_m=float(get(cfg,'crop.max_source_xy_radius_about_centroid_m',11.2)));rows.append((sum(s.crop_eligible for s in d.sources),idx))
    rows.sort();order=[]
    while rows:
        order.append(rows.pop()[1])
        if rows:order.append(rows.pop(0)[1])
    seen=set(order);return order+[x for x in ids if x not in seen]
def _pass(cfg,source,train_ds,ctx,lambda_ref,checkpointing,warmup,measure,seed):
    seed_all(seed);pipe=MotionTransportV1(cfg,msp_checkpoint=get(cfg,'paths.msp_checkpoint'),device=ctx.device);pipe.source_network.activation_checkpointing=bool(checkpointing);raw=pipe.source_network;sync_model_from_rank0(raw,ctx);ddp=DDP(raw,device_ids=[ctx.local_rank] if ctx.device.type=='cuda' else None,broadcast_buffers=False,find_unused_parameters=False) if ctx.world_size>1 else raw;pipe.source_network=ddp;opt=optimizer_for(raw,cfg);acc=int(get(cfg,'training.accumulation_steps',4));ids=_representative(train_ds,pipe,ctx,cfg,seed);micro=0;times=[];prep_times=[];selected=[];modes=[];seen=0
    def phase(n,measure_it):
        nonlocal micro,seen
        done=0
        if measure_it and ctx.device.type=='cuda':torch.cuda.reset_peak_memory_stats(ctx.device)
        while done<n:
            _sync(ctx);group_t0=time.perf_counter();recs=[];take=min(acc,n-done)
            for _ in range(take):
                idx=ids[micro%len(ids)];w,c=train_ds[idx];r=prepare_scene(pipe,source,w,c,cfg,progress=.5,epoch=9000,seed=seed);elig=tuple(s.source_id for s in r.decomp.sources if s.crop_eligible)
                if micro%5<3:r.selected=elig;r.budget_mode='profile_all'
                else:r.selected=route_sources(r.decomp.sources,int(get(cfg,'routing.train_later_sparse_budget',16)),strategy='uniform',seed=seed+micro);r.budget_mode='profile_uniform_q16'
                recs.append(r);selected.append(len(r.selected));modes.append(r.budget_mode);micro+=1;done+=1
            prep_local=time.perf_counter()-group_t0;gs=all_sum(torch.tensor(float(len(recs)),device=ctx.device),ctx);lp=sum(motion_pair_count(r.selected,r.targets) for r in recs);gp=all_sum(torch.tensor(float(lp),device=ctx.device),ctx);opt.zero_grad(set_to_none=True)
            for i,r in enumerate(recs):
                cm=contextlib.nullcontext() if ctx.world_size==1 or i==len(recs)-1 else ddp.no_sync()
                with cm:
                    occ,mn,_,zero,_,_,_,_=forward_losses(pipe,r,cfg,use_amp=True);loss=occ*ctx.world_size/gs
                    if float(gp)>0:loss=loss+lambda_at(.5,lambda_ref)*mn*ctx.world_size/gp
                    (loss+zero).backward()
            torch.nn.utils.clip_grad_norm_(raw.parameters(),float(get(cfg,'training.gradient_clip_norm',1.)));opt.step();_sync(ctx);dt=_max(time.perf_counter()-group_t0,ctx);prep=_max(prep_local,ctx)
            if measure_it:times.append(dt);prep_times.append(prep);seen+=len(recs)
    phase(warmup,False);phase(measure,True);peak=_max(torch.cuda.max_memory_allocated(ctx.device)/(1024**3) if ctx.device.type=='cuda' else 0.,ctx);wall=sum(times);pipe.source_network=raw
    stats={'checkpointing':bool(checkpointing),'warmup_microsteps':warmup,'measure_microsteps':measure,'measured_groups':len(times),'measured_local_scenes':seen,'measured_wall_s':wall,'mean_group_s':float(np.mean(times)) if times else 0.,'p95_group_s':float(np.quantile(times,.95)) if times else 0.,'mean_prepare_s':float(np.mean(prep_times)) if prep_times else 0.,'p95_prepare_s':float(np.quantile(prep_times,.95)) if prep_times else 0.,'mean_wall_s_per_local_scene':wall/max(1,seen),'global_throughput_scenes_per_s':ctx.world_size*seen/wall if wall>0 else 0.,'peak_memory_gib':peak,'selected_sources':{'mean':float(np.mean(selected)) if selected else 0.,'p95':float(np.quantile(selected,.95)) if selected else 0.,'max':max(selected) if selected else 0},'budget_mode_counts':{m:modes.count(m) for m in sorted(set(modes))}}
    return pipe,raw,opt,stats
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default=None);p.add_argument('--override',action='append',default=[]);p.add_argument('--output-dir',default=None);p.add_argument('--dev-max-windows',type=int,default=None);p.add_argument('--force-checkpointing',choices=('auto','on','off'),default='auto');p.add_argument('--formal-server-gate',action='store_true',help='lock a full-dev CUDA/BF16 profile for the GPU count used by this launch (1 or 2)');a=p.parse_args();cfg=load_config(a.config,a.override);ctx=init_distributed(require_cuda=a.formal_server_gate)
    if ctx.world_size not in (1,2):raise RuntimeError('MT-V1 supports 1 or 2 GPUs')
    formal_env=formal_server_environment(ctx,cfg,require_world_size=False,require_gpu_type=False) if a.formal_server_gate else {'torch':torch.__version__,'cuda_build':torch.version.cuda,'world_size':ctx.world_size,'device':str(ctx.device)}
    if a.formal_server_gate and a.dev_max_windows is not None:raise RuntimeError('formal profile requires full dev; remove --dev-max-windows')
    seed=int(get(cfg,'training.initial_seed',3407));seed_all(seed);dr,info,manifest,msp=[get(cfg,x) for x in ('paths.dataroot','paths.info_pkl','paths.manifest','paths.msp_checkpoint')]
    if not all((dr,info,manifest,msp)):raise RuntimeError('profile runtime paths incomplete')
    src=NuScenesWindowSource(dr,info_pkl=info,verbose=False);payload=load_manifest(manifest);train_ds=ManifestDataset(src,payload,'train');dev_ds=ManifestDataset(src,payload,'dev');calib=MotionTransportV1(cfg,msp_checkpoint=msp,device=ctx.device);sync_model_from_rank0(calib.source_network,ctx);lambda_ref,calrows=calibrate_lambda(calib,src,train_ds,cfg,ctx,batches=int(get(cfg,'loss.lambda_calibration_batches',8)),seed=seed);del calib
    if ctx.device.type=='cuda':torch.cuda.empty_cache()
    warm=int(get(cfg,'training.profile_warmup_microsteps',50));measure=int(get(cfg,'training.profile_measure_microsteps',200))
    if a.formal_server_gate and (warm!=50 or measure!=200):raise RuntimeError(f'formal profile requires exactly 50 warmup + 200 measured microsteps, got {warm}+{measure}')
    first=a.force_checkpointing=='on'
    try:pipe,raw,opt,prof=_pass(cfg,src,train_ds,ctx,lambda_ref,first,warm,measure,seed)
    except torch.cuda.OutOfMemoryError as e:
        raise RuntimeError('profile OOM before memory-threshold decision; rerun the same intended training launch with --force-checkpointing on') from e
    reran=False;threshold=float(get(cfg,'training.checkpoint_max_memory_without_recompute_gib',40))
    if a.force_checkpointing=='auto' and not first and prof['peak_memory_gib']>threshold:
        del pipe,raw,opt
        if ctx.device.type=='cuda':torch.cuda.empty_cache()
        pipe,raw,opt,prof=_pass(cfg,src,train_ds,ctx,lambda_ref,True,warm,measure,seed);reran=True
    barrier(ctx);dev_time=0.;dev_report=None
    if ctx.is_main:
        raw.eval();pipe.source_network=raw
        if ctx.device.type=='cuda':torch.cuda.synchronize(ctx.device)
        t=time.perf_counter();dev_report=evaluate(pipe,src,dev_ds,cfg,budgets=(0,16,'all'),strategy='msp',include_soft_main=True,seed=seed,max_windows=a.dev_max_windows)
        if ctx.device.type=='cuda':torch.cuda.synchronize(ctx.device)
        dev_time=time.perf_counter()-t
    if ctx.world_size>1:
        t=torch.tensor([dev_time],device=ctx.device,dtype=torch.float64);dist.broadcast(t,src=0);dev_time=float(t.item())
    od=Path(a.output_dir or Path(get(cfg,'paths.output_root','outputs/motion_transport_v1'))/'profile');tmp=od/'.full_checkpoint_tmp.pt'
    if ctx.is_main:od.mkdir(parents=True,exist_ok=True)
    barrier(ctx);ema=WarmupEMA(raw,int(get(cfg,'training.ema.half_life_optimizer_steps',100)));ema.start();_sync(ctx);t=time.perf_counter();save_checkpoint(tmp,raw=raw,ema=ema,opt=opt,cfg=cfg,ctx=ctx,epoch=0,next_group=0,global_step=0,lambda_ref=lambda_ref,manifest_path=manifest,msp_path=msp,selection_state={},phase='profile');_sync(ctx);save_time=_max(time.perf_counter()-t,ctx)
    if ctx.is_main:tmp.unlink(missing_ok=True)
    acc=int(get(cfg,'training.accumulation_steps',4));local_len=int(math.ceil(len(train_ds)/ctx.world_size));groups=int(math.ceil(local_len/acc));quarter=max(1,int(math.ceil(groups/4)));quarter_saves=sum(1 for gi in range(groups) if (gi+1)%quarter==0 or gi+1==groups);saves_per_epoch_worst=quarter_saves+3;train_epoch_core=prof['mean_wall_s_per_local_scene']*local_len;epoch_total=train_epoch_core+dev_time+saves_per_epoch_worst*save_time;full=a.dev_max_windows is None;formal_lock=bool(a.formal_server_gate and full);safety=max(60.,2.*prof['p95_group_s']);final_reserve=dev_time+2.*save_time+safety
    budget_hours=get(cfg,'training.max_hours');budget=None if budget_hours is None else float(budget_hours)*3600.;fixed=get(cfg,'training.fixed_epoch_cap')
    if not formal_lock:E=0;lock_source='nonformal_profile'
    elif fixed is not None:E=int(fixed);lock_source='fixed_epoch_cap'
    elif budget is not None:E=int(math.floor(max(0,budget-final_reserve)/epoch_total)) if epoch_total>0 else 0;lock_source='max_hours'
    else:E=int(get(cfg,'training.initial_epoch_estimate',5));lock_source='initial_epoch_estimate'
    locked=copy.deepcopy(cfg);locked['loss']['lambda_reference']=float(lambda_ref);locked['network']['activation_checkpointing']=bool(prof['checkpointing']);locked['training']['gpus']=int(ctx.world_size);locked['training']['epochs_locked']=E if E>0 and formal_lock else None;locked['training']['wall_clock_final_reserve_seconds']=float(final_reserve);locked['training']['wall_clock_next_group_guard_seconds']=float(max(prof['p95_group_s']*1.5,1.));locked.setdefault('runtime',{}).update({'code_commit':_git(),'profile_formal_full_dev':full,'profile_formal_server_gate':bool(a.formal_server_gate),'profile_world_size':ctx.world_size,'profile_group_p95_s':prof['p95_group_s'],'profile_dev_s':float(dev_time),'profile_full_checkpoint_s':save_time,'profile_safety_s':float(safety),'profile_environment':formal_env,'profile_epoch_lock_source':lock_source,'manifest_sha256':sha256_file(manifest),'msp_sha256':sha256_file(msp)});result={'spec_version':'MT-V1-SPEC-2','formal_full_dev':full,'formal_server_gate':bool(a.formal_server_gate),'environment':formal_env,'world_size':ctx.world_size,'train_windows':len(train_ds),'dev_windows':len(dev_ds),'lambda_ref':lambda_ref,'gradient_calibration':{'probe_epsilon':float(get(cfg,'loss.gradient_calibration.probe_epsilon',1e-3)),'max_ce_antagonistic_fraction_of_motion':float(get(cfg,'loss.gradient_calibration.max_ce_antagonistic_fraction_of_motion',.5)),'rows':calrows},'msp':msp_checkpoint_provenance(msp),'profile':prof,'checkpointing_reran_after_memory_threshold':reran,'timing':{'estimated_train_epoch_core_s':train_epoch_core,'measured_dev_s':dev_time,'measured_full_checkpoint_s':save_time,'quarter_saves_per_epoch':quarter_saves,'worst_full_saves_per_epoch':saves_per_epoch_worst,'estimated_epoch_total_s':epoch_total,'wall_clock_final_reserve_s':final_reserve,'wall_clock_next_group_guard_s':locked['training']['wall_clock_next_group_guard_seconds'],'max_budget_s':budget},'epoch_lock_source':lock_source,'epochs_locked':locked['training']['epochs_locked'],'dev_timing_report':dev_report}
    if ctx.is_main:
        (od/'profile.json').write_text(json.dumps(result,indent=2));(od/'resolved_profile_config.yaml').write_text(yaml.safe_dump(locked,sort_keys=False));print(json.dumps({k:result[k] for k in ('formal_server_gate','environment','lambda_ref','profile','timing','epoch_lock_source','epochs_locked')},indent=2))
        if a.formal_server_gate and budget is not None and E<=0:raise RuntimeError(f'configured max_hours={budget_hours} leaves no complete epoch after full data/dev/checkpoint timing; raise/disable max_hours or change the training launch')
    barrier(ctx)
    if ctx.world_size>1:dist.destroy_process_group()
if __name__=='__main__':main()