from __future__ import annotations
import contextlib,copy,hashlib,json,math,os,random,time
from dataclasses import dataclass
from pathlib import Path
import numpy as np,torch,torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from .config import get
from .source_adapter import decompose_strong_sources,extract_original_msp_candidates,map_msp_to_sources
from .routing import training_budget_selection,scene_mirror_flag,sha256_file
from .targets import build_training_targets
from .crop_history import build_history_crops
from .compositor import render_soft_ordered,compose_hard
from .losses import occupancy_ce_full,motion_loss_sum,motion_pair_count,lambda_at,calibration_probe_family_like,calibrated_gradient_ratio,output_gradient_lambda_floor,gradient_summary
from .ema import WarmupEMA

@dataclass
class DistContext:
    rank:int=0;world_size:int=1;local_rank:int=0;device:torch.device=torch.device('cpu')
    @property
    def is_main(self):return self.rank==0
def init_distributed(require_cuda=True):
    if 'RANK' in os.environ:
        rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
        if require_cuda and not torch.cuda.is_available():raise RuntimeError('DDP training requires CUDA')
        device=torch.device('cuda',local) if torch.cuda.is_available() else torch.device('cpu')
        if device.type=='cuda':torch.cuda.set_device(local);backend='nccl'
        else:backend='gloo'
        dist.init_process_group(backend=backend,init_method='env://');return DistContext(rank,world,local,device)
    return DistContext(0,1,0,torch.device('cuda',0) if torch.cuda.is_available() else torch.device('cpu'))
def barrier(ctx):
    if ctx.world_size>1:dist.barrier()
def all_sum(x,ctx):
    y=x.clone()
    if ctx.world_size>1:dist.all_reduce(y,op=dist.ReduceOp.SUM)
    return y
def all_max_float(v,ctx):
    t=torch.tensor(float(v),device=ctx.device,dtype=torch.float64)
    if ctx.world_size>1:dist.all_reduce(t,op=dist.ReduceOp.MAX)
    return float(t.item())
def seed_all(seed):
    random.seed(seed);np.random.seed(seed%(2**32-1));torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
def rng_state():return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
def restore_rng(s):
    random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch'])
    if s.get('cuda') is not None and torch.cuda.is_available():torch.cuda.set_rng_state_all(s['cuda'])
def _stable_config(cfg):
    obj=copy.deepcopy(cfg)
    if isinstance(obj.get('runtime'),dict):obj['runtime'].pop('config_path',None)
    return obj
def _hash_json(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),default=str).encode()).hexdigest()
def sync_model_from_rank0(model,ctx):
    if ctx.world_size<=1:return
    with torch.no_grad():
        for p in model.parameters():dist.broadcast(p.data,src=0)
        for b in model.buffers():dist.broadcast(b.data,src=0)
def optimizer_for(model,cfg):
    wd=float(get(cfg,'training.weight_decay',1e-2));decay=[];nodecay=[]
    for name,p in model.named_parameters():
        if not p.requires_grad:continue
        (nodecay if p.ndim==1 or name.endswith('.bias') or 'norm' in name.lower() else decay).append(p)
    return torch.optim.AdamW([{'params':decay,'weight_decay':wd},{'params':nodecay,'weight_decay':0.}],lr=float(get(cfg,'training.peak_lr',3e-4)))
def lr_for_step(step,total,cfg):
    peak=float(get(cfg,'training.peak_lr',3e-4));end=float(get(cfg,'training.end_lr',3e-6));warm=max(1,int(round(total*float(get(cfg,'training.lr_warmup_fraction',.05)))))
    if step<warm:return peak*float(step+1)/warm
    q=min(1.,max(0.,(step-warm)/max(1,total-warm)));return end+.5*(peak-end)*(1+math.cos(math.pi*q))
def set_lr(opt,lr):
    for g in opt.param_groups:g['lr']=float(lr)
def rank_epoch_indices(n,ctx,epoch,seed):
    gen=torch.Generator().manual_seed(int(seed)+int(epoch)*1009);perm=torch.randperm(n,generator=gen).tolist();target=int(math.ceil(n/ctx.world_size))*ctx.world_size;pad=target-n
    if pad:perm+=perm[:pad]
    rows=[perm[r:target:ctx.world_size] for r in range(ctx.world_size)];return rows[ctx.rank],pad
@dataclass
class PreparedScene:
    window:object;causal:object;decomp:object;targets:object;selected:tuple;mirror:bool;budget_mode:str
def prepare_scene(pipe,source,window,causal,cfg,*,progress,epoch,seed):
    d=decompose_strong_sources(causal,grid=pipe.grid,cfg=pipe.strong_cfg,frame_dt_s=float(get(cfg,'input.dt_seconds',.5)),crop_radius_limit_m=float(get(cfg,'crop.max_source_xy_radius_about_centroid_m',11.2)));cand=extract_original_msp_candidates(causal,grid=pipe.grid,motion_cfg=pipe.motion_cfg,kta_cfg=pipe.msp_kta_cfg);map_msp_to_sources(d,cand,shape_xyz=tuple(pipe.grid.shape_hwd));key=f'{causal.sample_id}:epoch{epoch}';selected,mode=training_budget_selection(d.sources,progress,seed=seed,sample_id=key,warmup_fraction=float(get(cfg,'routing.train_all_source_warmup_fraction',.2)),later_all_probability=float(get(cfg,'routing.train_later_all_source_probability',.5)),sparse_budget=int(get(cfg,'routing.train_later_sparse_budget',16)));mirror=scene_mirror_flag(seed=seed,sample_id=key,probability=float(get(cfg,'training.augmentation.probability',.5)));targets=build_training_targets(source,window,d,best_coverage_min=float(get(cfg,'targets.best_box_source_coverage_min',.8)),second_coverage_max=float(get(cfg,'targets.second_box_source_coverage_max',.2)),max_points=int(get(cfg,'targets.motion_points_per_source_max',64)),class_count=int(get(cfg,'input.class_count',18)));return PreparedScene(window,causal,d,targets,selected,mirror,mode)
def _losses_for_delta(pipe,rec,cfg,delta):
    soft=render_soft_ordered(rec.causal,rec.decomp,delta,rec.selected,grid=pipe.grid,class_count=int(get(cfg,'input.class_count',18)),halo_voxels=int(get(cfg,'renderer.query_bbox_halo_voxels',2)),query_chunk=int(get(cfg,'renderer.query_chunk',65536)));occ,olog=occupancy_ce_full(soft,rec.targets,eps=float(get(cfg,'renderer.probability_epsilon',1e-6)));mnum,mcount,mlog=motion_loss_sum(delta,rec.selected,rec.decomp,rec.targets,np.asarray(rec.causal.history_ego_to_world[-1]));return occ,mnum,mcount,soft,{**olog,**mlog}
def forward_losses(pipe,rec,cfg,*,use_amp=True,need_hard=False):
    crops=build_history_crops(rec.causal,rec.decomp.sources,rec.selected,grid=pipe.grid,crop_shape_xyz=tuple(get(cfg,'crop.shape_xyz',[64,64,16])),xy_resolution_m=float(get(cfg,'crop.xy_resolution_m',.4)),mirror=rec.mirror,device=pipe.device);enabled=bool(use_amp and pipe.device.type=='cuda')
    with torch.autocast(device_type=pipe.device.type,dtype=torch.bfloat16,enabled=enabled):delta,zero=pipe.source_network(crops,source_microbatch=int(get(cfg,'training.source_microbatch',16)))
    delta=delta.float();occ,mn,mcount,soft,ll=_losses_for_delta(pipe,rec,cfg,delta);hard=compose_hard(rec.causal,rec.decomp,delta,rec.selected,grid=pipe.grid) if need_hard else None;return occ,mn,mcount,zero,delta,soft,hard,{**ll,'selected_sources':len(rec.selected),'budget_mode':rec.budget_mode}
def _global_grad(loss,params,ctx):
    gs=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True);flat=torch.cat([torch.zeros_like(p).reshape(-1) if g is None else g.reshape(-1) for p,g in zip(params,gs)]).detach()
    if ctx.world_size>1:dist.all_reduce(flat,op=dist.ReduceOp.SUM);flat/=ctx.world_size
    return flat
def _cosine(a,b):return float(torch.dot(a,b)/(torch.linalg.vector_norm(a)*torch.linalg.vector_norm(b)).clamp_min(1e-12)) if a.numel() and b.numel() else float('nan')
def calibrate_lambda(pipe,source,dataset,cfg,ctx,*,batches=8,seed=3407):
    raw=pipe.source_network;params=list(raw.head2.parameters());head_refs=[];floors=[];rows=[];raw.train();idxs,_=rank_epoch_indices(len(dataset),ctx,0,seed);eps=float(get(cfg,'loss.gradient_calibration.probe_epsilon',1e-3));frac=float(get(cfg,'loss.gradient_calibration.max_ce_antagonistic_fraction_of_motion',.5));active_rel=float(get(cfg,'loss.gradient_calibration.active_motion_grad_rel',1e-4))
    for bi in range(int(batches)):
        w,c=dataset[idxs[bi%len(idxs)]];rec=prepare_scene(pipe,source,w,c,cfg,progress=0.,epoch=0,seed=seed);_,_,_,zero,delta,_,_,_=forward_losses(pipe,rec,cfg,use_amp=False);scenes=all_sum(torch.tensor(1.,device=ctx.device),ctx);candidate_rows=[]
        for pi,probe in enumerate(calibration_probe_family_like(delta,eps)):
            occ,mnum,mcount,_,_=_losses_for_delta(pipe,rec,cfg,probe);pairs=all_sum(torch.tensor(float(mcount),device=ctx.device),ctx);os=occ*ctx.world_size/scenes;ms=mnum*ctx.world_size/pairs.clamp_min(1) if float(pairs)>0 else mnum*0;go=_global_grad(os+zero,params,ctx);gm=_global_grad(ms+zero,params,ctx);Go=float(torch.linalg.vector_norm(go));Gm=float(torch.linalg.vector_norm(gm));cos=_cosine(go,gm);head=calibrated_gradient_ratio(Go,Gm,cos,max_ce_antagonistic_fraction_of_motion=frac);floor_local=0.;go_out=gm_out=None
            if delta.numel() and delta.requires_grad and float(pairs)>0:
                go_out=torch.autograd.grad(os+zero,delta,retain_graph=True,allow_unused=True)[0];gm_out=torch.autograd.grad(ms+zero,delta,retain_graph=True,allow_unused=True)[0]
                if go_out is not None and gm_out is not None:floor_local=output_gradient_lambda_floor(go_out,gm_out,max_ce_antagonistic_fraction_of_motion=frac,active_motion_grad_rel=active_rel)
            floor=all_max_float(floor_local,ctx);ratio=max(head if np.isfinite(head) else 0.,floor);cand={'probe_index':pi,'G_occ_head':Go,'G_mot_head':Gm,'head_grad_cosine':cos,'head_ratio':head,'output_conflict_floor':floor,'ratio':ratio,'motion_pairs_global':float(pairs),'head_occ_abs':gradient_summary(go),'head_motion_abs':gradient_summary(gm)}
            if go_out is not None:cand['output_occ_abs']=gradient_summary(go_out)
            if gm_out is not None:cand['output_motion_abs']=gradient_summary(gm_out)
            candidate_rows.append(cand)
        finite=[x for x in candidate_rows if np.isfinite(x['ratio'])];best=max(finite,key=lambda x:x['ratio']) if finite else None
        if best is None:rows.append({'batch':bi,'probe_epsilon':eps,'probe_family_size':len(candidate_rows),'probe_candidates':candidate_rows,'ratio':float('nan')});continue
        row={'batch':bi,'probe_epsilon':eps,'probe_family_size':len(candidate_rows),'selected_probe_index':best['probe_index'],**best,'probe_candidates':candidate_rows};rows.append(row);head_refs.append(float(max((x['head_ratio'] for x in finite if np.isfinite(x['head_ratio'])),default=0.)));floors.append(float(max((x['output_conflict_floor'] for x in finite),default=0.)))
    if not head_refs and not any(x>0 for x in floors):raise RuntimeError('lambda calibration has no batch with usable motion gradient')
    ref=max(float(np.median(head_refs)) if head_refs else 0.,max(floors) if floors else 0.)
    if not np.isfinite(ref) or ref<=0:raise RuntimeError('lambda calibration produced invalid reference')
    return ref,rows
def save_checkpoint(path,*,raw,ema,opt,cfg,ctx,epoch,next_group,global_step,lambda_ref,manifest_path,msp_path,selection_state=None,phase='train'):
    local=rng_state();states=[None]*ctx.world_size
    if ctx.world_size>1:dist.all_gather_object(states,local)
    else:states=[local]
    if not ctx.is_main:return
    payload={'version':'motion_transport_v1_checkpoint_v2','spec_version':'MT-V1-SPEC-2','model_state_dict':raw.state_dict(),'ema_state_dict':ema.state_dict(),'optimizer_state_dict':opt.state_dict(),'amp_scaler_state_dict':None,'config':cfg,'config_hash':_hash_json(_stable_config(cfg)),'epoch':int(epoch),'next_group':int(next_group),'phase':str(phase),'global_step':int(global_step),'lambda_ref':float(lambda_ref),'world_size':ctx.world_size,'rank_rng_states':states,'manifest_sha256':sha256_file(manifest_path),'msp_sha256':sha256_file(msp_path),'selection_state':selection_state or {}}
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');torch.save(payload,tmp);os.replace(tmp,p)
def load_resume(path,*,raw,ema,opt,cfg,ctx,manifest_path,msp_path):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    if ck.get('spec_version')!='MT-V1-SPEC-2' or int(ck.get('world_size',-1))!=ctx.world_size:raise RuntimeError('resume spec/WORLD_SIZE mismatch')
    if ck.get('manifest_sha256')!=sha256_file(manifest_path) or ck.get('msp_sha256')!=sha256_file(msp_path):raise RuntimeError('resume data/MSP provenance mismatch')
    if ck.get('config_hash')!=_hash_json(_stable_config(cfg)):raise RuntimeError('resume resolved-config contract mismatch')
    raw.load_state_dict(ck['model_state_dict'],strict=True);ema.load_state_dict(ck['ema_state_dict']);opt.load_state_dict(ck['optimizer_state_dict']);restore_rng(ck['rank_rng_states'][ctx.rank]);return ck
def _wall_clock_limit_s(cfg):
    value=get(cfg,'training.max_hours')
    if value is None:return None
    value=float(value)
    return value*3600. if value>0 else None
def _final_reserve_s(cfg):return max(0.,float(get(cfg,'training.wall_clock_final_reserve_seconds',0) or 0))
def _estimated_dev_s(cfg):
    value=get(cfg,'runtime.profile_dev_s')
    return _final_reserve_s(cfg) if value is None else max(0.,float(value))
def _estimated_save_s(cfg):return max(0.,float(get(cfg,'runtime.profile_full_checkpoint_s',0) or 0))
def _phase_budget_should_stop(started,cfg,ctx,*,current_phase_s=0.,post_phase_reserve_s=0.,next_guard_s=0.):
    elapsed=all_max_float(time.monotonic()-started,ctx);required=max(0.,float(current_phase_s))+max(0.,float(post_phase_reserve_s))+max(0.,float(next_guard_s));limit=_wall_clock_limit_s(cfg)
    if limit is None:return False,elapsed,required
    return bool(elapsed+required>=limit),elapsed,required
def _budget_should_stop(started,cfg,ctx):
    stop,elapsed,_=_phase_budget_should_stop(started,cfg,ctx,post_phase_reserve_s=_final_reserve_s(cfg),next_guard_s=float(get(cfg,'training.wall_clock_next_group_guard_seconds',0) or 0));return stop,elapsed
def train(pipe,source,train_ds,dev_ds,cfg,ctx,*,manifest_path,msp_path,output_dir,resume=None):
    epochs=get(cfg,'training.epochs_locked')
    if epochs is None:raise RuntimeError('run formal profile first')
    epochs=int(epochs);acc=int(get(cfg,'training.accumulation_steps',4));seed=int(get(cfg,'training.initial_seed',3407));seed_all(seed+ctx.rank);raw=pipe.source_network;sync_model_from_rank0(raw,ctx);opt=optimizer_for(raw,cfg);ema=WarmupEMA(raw,int(get(cfg,'training.ema.half_life_optimizer_steps',100)))
    if resume:lambda_ref=float(torch.load(resume,map_location='cpu',weights_only=False)['lambda_ref'])
    else:
        locked=get(cfg,'loss.lambda_reference')
        if locked is None:raise RuntimeError('loss.lambda_reference is null; run formal profile first')
        lambda_ref=float(locked)
        if not np.isfinite(lambda_ref) or lambda_ref<=0:raise RuntimeError('locked lambda_reference invalid')
    ddp=DDP(raw,device_ids=[ctx.local_rank] if ctx.device.type=='cuda' else None,broadcast_buffers=False,find_unused_parameters=False) if ctx.world_size>1 else raw;pipe.source_network=ddp;n=len(train_ds);local_len=int(math.ceil(n/ctx.world_size));groups=int(math.ceil(local_len/acc));total=epochs*groups;start_epoch=start_group=global_step=0;selection={'best_moving':-float('inf'),'best_epoch':None};resume_phase='train'
    if resume:
        ck=load_resume(resume,raw=raw,ema=ema,opt=opt,cfg=cfg,ctx=ctx,manifest_path=manifest_path,msp_path=msp_path);start_epoch=int(ck['epoch']);start_group=int(ck['next_group']);global_step=int(ck['global_step']);selection=dict(ck.get('selection_state') or selection);resume_phase=str(ck.get('phase','train'))
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True);warm=max(1,int(round(total*float(get(cfg,'training.lr_warmup_fraction',.05)))));logp=out/'train.jsonl';started=time.monotonic();termination='planned_complete';budget_stop=False;stop_stage=None;stop_epoch=epochs;stop_group=0;stop_phase='finished';final_raw_completed=False
    for epoch in range(start_epoch,epochs):
        indices,padded=rank_epoch_indices(n,ctx,epoch,seed);g0=start_group if epoch==start_epoch else 0
        if epoch==start_epoch and resume_phase=='dev_pending':g0=groups
        for gi in range(g0,groups):
            stop,elapsed=_budget_should_stop(started,cfg,ctx)
            if stop:
                save_checkpoint(out/'latest.pt',raw=raw,ema=ema,opt=opt,cfg=cfg,ctx=ctx,epoch=epoch,next_group=gi,global_step=global_step,lambda_ref=lambda_ref,manifest_path=manifest_path,msp_path=msp_path,selection_state=selection,phase='budget_stop');termination='wall_clock_budget';budget_stop=True;stop_stage='train_group';stop_epoch=epoch;stop_group=gi;stop_phase='budget_stop';break
            chunk=indices[gi*acc:min((gi+1)*acc,len(indices))];progress=min(1.,global_step/max(1,total-1));records=[]
            for idx in chunk:w,c=train_ds[idx];records.append(prepare_scene(pipe,source,w,c,cfg,progress=progress,epoch=epoch,seed=seed))
            global_scenes=all_sum(torch.tensor(float(len(records)),device=ctx.device),ctx);global_pairs=all_sum(torch.tensor(float(sum(motion_pair_count(r.selected,r.targets) for r in records)),device=ctx.device),ctx);opt.zero_grad(set_to_none=True);logs=[]
            for mi,r in enumerate(records):
                cm=contextlib.nullcontext() if ctx.world_size==1 or mi==len(records)-1 else ddp.no_sync()
                with cm:
                    occ,mnum,_,zero,delta,_,_,ll=forward_losses(pipe,r,cfg,use_amp=True);lam=lambda_at(progress,lambda_ref);loss=occ*ctx.world_size/global_scenes+(lam*mnum*ctx.world_size/global_pairs if float(global_pairs)>0 else mnum*0)+zero;loss.backward();absd=delta.detach().abs().reshape(-1,3) if delta.numel() else None;ll.update({'lambda':lam,'delta_abs_p95':float(delta.detach().abs().quantile(.95)) if delta.numel() else 0.,'delta_xyyaw_abs':{k:(gradient_summary(absd[:,j]) if absd is not None else {'p50':0.,'p95':0.,'p99':0.,'max':0.}) for j,k in enumerate(('dx','dy','yaw'))}});logs.append(ll)
            norm=torch.nn.utils.clip_grad_norm_(raw.parameters(),float(get(cfg,'training.gradient_clip_norm',1.)))
            if not torch.isfinite(torch.as_tensor(norm)):raise FloatingPointError('NaN/Inf gradient norm')
            lr=lr_for_step(global_step,total,cfg);set_lr(opt,lr);opt.step();global_step+=1
            if bool(get(cfg,'training.ema.enabled',True)):
                if not ema.started and global_step>=warm:ema.start()
                elif ema.started:ema.update()
            if ctx.is_main:
                row={'epoch':epoch,'group':gi,'global_step':global_step,'lr':lr,'grad_norm_preclip':float(norm),'clip_ratio':min(1.,float(get(cfg,'training.gradient_clip_norm',1.))/max(float(norm),1e-12)),'lambda_ref':lambda_ref,'global_scenes':float(global_scenes),'global_motion_pairs':float(global_pairs),'rank0_logs':logs,'padded_draws_epoch':padded}
                with open(logp,'a') as f:f.write(json.dumps(row)+'\n')
            quarter=max(1,int(math.ceil(groups/4)))
            if (gi+1)%quarter==0 or gi+1==groups:
                if gi+1>=groups:ne,ng,phase=epoch,groups,'dev_pending'
                else:ne,ng,phase=epoch,gi+1,'train'
                save_checkpoint(out/'latest.pt',raw=raw,ema=ema,opt=opt,cfg=cfg,ctx=ctx,epoch=ne,next_group=ng,global_step=global_step,lambda_ref=lambda_ref,manifest_path=manifest_path,msp_path=msp_path,selection_state=selection,phase=phase)
        if budget_stop:break
        stop_dev,elapsed,_=_phase_budget_should_stop(started,cfg,ctx,current_phase_s=_estimated_dev_s(cfg),post_phase_reserve_s=_final_reserve_s(cfg))
        if stop_dev:
            save_checkpoint(out/'latest.pt',raw=raw,ema=ema,opt=opt,cfg=cfg,ctx=ctx,epoch=epoch,next_group=groups,global_step=global_step,lambda_ref=lambda_ref,manifest_path=manifest_path,msp_path=msp_path,selection_state=selection,phase='dev_pending');termination='wall_clock_budget';budget_stop=True;stop_stage='epoch_dev';stop_epoch=epoch;stop_group=groups;stop_phase='dev_pending';break
        barrier(ctx);save_best=False
        if ctx.is_main:
            if not ema.started:raise RuntimeError('EMA unavailable for checkpoint selection')
            from .evaluation import evaluate
            em=copy.deepcopy(raw).to(ctx.device);ema.copy_to(em);em.eval();old=pipe.source_network;pipe.source_network=em;rep=evaluate(pipe,source,dev_ds,cfg,budgets=(0,16,'all'),strategy='msp',include_soft_main=True,seed=seed);pipe.source_network=old;del em;Path(out,f'dev_epoch_{epoch+1:04d}.json').write_text(json.dumps(rep,indent=2));d=rep['delta_vs_kta']['16'];eligible=float(d['overall_pp'])>=float(get(cfg,'evaluation.overall_noninferiority_pp',-.10)) and float(d['stationary_movable_pp'])>=float(get(cfg,'evaluation.stationary_movable_noninferiority_pp',-.20));moving=float(rep['hard']['16']['moving']['mIoU'])
            if eligible and moving>float(selection.get('best_moving',-float('inf'))):selection={'best_moving':moving,'best_epoch':epoch+1};save_best=True
        if ctx.world_size>1:
            obj=[selection,save_best] if ctx.is_main else [None,None];dist.broadcast_object_list(obj,src=0);selection,save_best=obj
        barrier(ctx);save_checkpoint(out/f'epoch_{epoch+1:04d}.pt',raw=raw,ema=ema,opt=opt,cfg=cfg,ctx=ctx,epoch=epoch+1,next_group=0,global_step=global_step,lambda_ref=lambda_ref,manifest_path=manifest_path,msp_path=msp_path,selection_state=selection,phase='epoch_complete')
        if save_best:save_checkpoint(out/'best.pt',raw=raw,ema=ema,opt=opt,cfg=cfg,ctx=ctx,epoch=epoch+1,next_group=0,global_step=global_step,lambda_ref=lambda_ref,manifest_path=manifest_path,msp_path=msp_path,selection_state=selection,phase='epoch_complete')
        save_checkpoint(out/'latest.pt',raw=raw,ema=ema,opt=opt,cfg=cfg,ctx=ctx,epoch=epoch+1,next_group=0,global_step=global_step,lambda_ref=lambda_ref,manifest_path=manifest_path,msp_path=msp_path,selection_state=selection,phase='epoch_complete');barrier(ctx);start_group=0;resume_phase='train'
    barrier(ctx)
    if not budget_stop:
        stop_raw,elapsed,_=_phase_budget_should_stop(started,cfg,ctx,current_phase_s=_estimated_dev_s(cfg),post_phase_reserve_s=_estimated_save_s(cfg))
        if stop_raw:
            termination='wall_clock_budget';stop_stage='final_raw'
        else:
            if ctx.is_main:
                from .evaluation import evaluate
                old=pipe.source_network;pipe.source_network=raw;raw.eval();Path(out,'final_raw_dev.json').write_text(json.dumps(evaluate(pipe,source,dev_ds,cfg,budgets=(0,16,'all'),strategy='msp',include_soft_main=True,seed=seed),indent=2));pipe.source_network=old;raw.train()
            barrier(ctx);final_raw_completed=True;limit=_wall_clock_limit_s(cfg)
            if limit is not None and all_max_float(time.monotonic()-started,ctx)>=limit:termination='wall_clock_budget';stop_stage='final_raw_overrun'
    barrier(ctx)
    if budget_stop:last_epoch,last_group,last_phase=stop_epoch,stop_group,stop_phase
    else:last_epoch,last_group,last_phase=epochs,0,'finished'
    save_checkpoint(out/'last.pt',raw=raw,ema=ema,opt=opt,cfg=cfg,ctx=ctx,epoch=last_epoch,next_group=last_group,global_step=global_step,lambda_ref=lambda_ref,manifest_path=manifest_path,msp_path=msp_path,selection_state=selection,phase=last_phase)
    return {'epochs_locked':epochs,'global_step':global_step,'lambda_ref':lambda_ref,'selection_state':selection,'termination_reason':termination,'stop_stage':stop_stage,'final_raw_completed':final_raw_completed,'resume_cursor':{'epoch':last_epoch,'next_group':last_group,'phase':last_phase},'elapsed_s':all_max_float(time.monotonic()-started,ctx)}