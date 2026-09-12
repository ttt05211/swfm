#!/usr/bin/env python3
from __future__ import annotations

import argparse,contextlib,json,subprocess,sys
from pathlib import Path
import torch,torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from real_motion.nuscenes_adapter import NuScenesWindowSource
from real_motion.motion_transport_v1.config import load_config,get
from real_motion.motion_transport_v1.data import ManifestDataset,load_manifest
from real_motion.motion_transport_v1.model import MotionTransportV1
from real_motion.motion_transport_v1.engine import init_distributed,barrier,all_sum,seed_all,sync_model_from_rank0,optimizer_for,prepare_scene,forward_losses
from real_motion.motion_transport_v1.losses import motion_pair_count,lambda_at
from real_motion.motion_transport_v1.server_acceptance import formal_server_environment


def _git():
    try:return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    except Exception:return 'unknown'


def _clone_model_state(model):return {k:v.detach().clone() for k,v in model.state_dict().items()}

def _grad_state(model):
    out=[]
    for p in model.parameters():out.append(torch.zeros_like(p) if p.grad is None else p.grad.detach().clone())
    return out

def _optimizer_tensor_state(opt,model):
    rows=[]
    for p in model.parameters():
        s=opt.state.get(p,{})
        rows.append({k:(v.detach().clone() if torch.is_tensor(v) else v) for k,v in s.items()})
    return rows

def _states_close(a,b,*,atol,rtol):
    ok=True;max_abs=0.;max_rel=0.
    for x,y in zip(a,b):
        if x.shape!=y.shape:return False,float('inf'),float('inf')
        d=(x-y).abs();ma=float(d.max()) if d.numel() else 0.;den=torch.maximum(x.abs(),y.abs()).clamp_min(1e-12);mr=float((d/den).max()) if d.numel() else 0.;max_abs=max(max_abs,ma);max_rel=max(max_rel,mr);ok=ok and bool(torch.allclose(x,y,atol=atol,rtol=rtol))
    return ok,max_abs,max_rel

def _optimizer_states_close(a,b,*,atol,rtol):
    ok=True;max_abs=0.;max_rel=0.
    if len(a)!=len(b):return False,float('inf'),float('inf')
    for sa,sb in zip(a,b):
        if set(sa)!=set(sb):return False,float('inf'),float('inf')
        for k in sa:
            x,y=sa[k],sb[k]
            if torch.is_tensor(x):
                if not torch.is_tensor(y):return False,float('inf'),float('inf')
                good,ma,mr=_states_close([x.float()],[y.float()],atol=atol,rtol=rtol);ok=ok and good;max_abs=max(max_abs,ma);max_rel=max(max_rel,mr)
            else:ok=ok and x==y
    return ok,max_abs,max_rel

def _scan_cases(pipe,source,ds,cfg,scan_windows,seed):
    rows=[]
    for idx in range(min(int(scan_windows),len(ds))):
        w,c=ds[idx];rec=prepare_scene(pipe,source,w,c,cfg,progress=.5,epoch=777,seed=seed);eligible=tuple(s.source_id for s in rec.decomp.sources if s.crop_eligible);rec.selected=eligible;pairs=motion_pair_count(eligible,rec.targets);rows.append({'index':idx,'sample_id':c.sample_id,'eligible_sources':len(eligible),'motion_pairs':int(pairs)})
    empty=[r for r in rows if r['eligible_sources']==0]
    positive=[r for r in rows if r['eligible_sources']>0 and r['motion_pairs']>0]
    if not empty:raise RuntimeError(f'DDP acceptance scan found no empty-source window in first {len(rows)} windows')
    if len(positive)<2:raise RuntimeError(f'DDP acceptance scan found fewer than two motion-valid nonempty windows in first {len(rows)} windows')
    low=min(positive,key=lambda r:(r['eligible_sources'],r['motion_pairs'],r['index']));high=max(positive,key=lambda r:(r['eligible_sources'],r['motion_pairs'],-r['index']))
    if low['eligible_sources']==high['eligible_sources']:
        raise RuntimeError(f'DDP acceptance scan did not find distinct eligible source counts in first {len(rows)} windows')
    motion0=next((r for r in rows if r['eligible_sources']>0 and r['motion_pairs']==0),None)
    return {'empty':empty[0],'low':low,'high':high,'nonempty_motion0':motion0,'scanned_windows':len(rows)}

def _prepare_recs(pipe,source,ds,cfg,index,count,seed):
    recs=[]
    for j in range(int(count)):
        w,c=ds[int(index)];r=prepare_scene(pipe,source,w,c,cfg,progress=.5,epoch=9000+j,seed=seed);r.selected=tuple(s.source_id for s in r.decomp.sources if s.crop_eligible);r.budget_mode='ddp_acceptance_all_eligible';recs.append(r)
    return recs

def _run_scenario(cfg,source,ds,ctx,case_index,local_microsteps,checkpointing,lambda_ref,seed,grad_atol,grad_rtol,param_atol,param_rtol):
    seed_all(seed);pipe=MotionTransportV1(cfg,msp_checkpoint=get(cfg,'paths.msp_checkpoint'),device=ctx.device);raw=pipe.source_network;raw.activation_checkpointing=bool(checkpointing);sync_model_from_rank0(raw,ctx);initial=_clone_model_state(raw);recs=_prepare_recs(pipe,source,ds,cfg,case_index,local_microsteps,seed)
    local_summary=[{'sample_id':r.causal.sample_id,'selected_sources':len(r.selected),'motion_pairs':motion_pair_count(r.selected,r.targets)} for r in recs];summaries=[None]*ctx.world_size;dist.all_gather_object(summaries,local_summary)
    gs=float(all_sum(torch.tensor(float(len(recs)),device=ctx.device),ctx));lp=sum(motion_pair_count(r.selected,r.targets) for r in recs);gp=float(all_sum(torch.tensor(float(lp),device=ctx.device),ctx));weight=lambda_at(.5,lambda_ref)

    pipe.source_network=raw;raw.train();opt_ref=optimizer_for(raw,cfg);opt_ref.zero_grad(set_to_none=True)
    for r in recs:
        occ,mn,_,zero,_,_,_,_=forward_losses(pipe,r,cfg,use_amp=True);loss=occ/gs
        if gp>0:loss=loss+weight*mn/gp
        (loss+zero).backward()
    for p in raw.parameters():
        if p.grad is None:p.grad=torch.zeros_like(p)
        dist.all_reduce(p.grad,op=dist.ReduceOp.SUM)
    ref_grads=_grad_state(raw);torch.nn.utils.clip_grad_norm_(raw.parameters(),float(get(cfg,'training.gradient_clip_norm',1.)));opt_ref.step();ref_model=_clone_model_state(raw);ref_opt=_optimizer_tensor_state(opt_ref,raw)

    raw.load_state_dict(initial,strict=True);raw.zero_grad(set_to_none=True);opt_ddp=optimizer_for(raw,cfg);ddp=DDP(raw,device_ids=[ctx.local_rank],broadcast_buffers=False,find_unused_parameters=False);pipe.source_network=ddp;raw.train();opt_ddp.zero_grad(set_to_none=True)
    for i,r in enumerate(recs):
        cm=contextlib.nullcontext() if i==len(recs)-1 else ddp.no_sync()
        with cm:
            occ,mn,_,zero,_,_,_,_=forward_losses(pipe,r,cfg,use_amp=True);loss=occ*ctx.world_size/gs
            if gp>0:loss=loss+weight*mn*ctx.world_size/gp
            (loss+zero).backward()
    ddp_grads=_grad_state(raw);grad_ok,grad_abs,grad_rel=_states_close(ref_grads,ddp_grads,atol=grad_atol,rtol=grad_rtol);torch.nn.utils.clip_grad_norm_(raw.parameters(),float(get(cfg,'training.gradient_clip_norm',1.)));opt_ddp.step();ddp_model=_clone_model_state(raw);ddp_opt=_optimizer_tensor_state(opt_ddp,raw);model_ok,param_abs,param_rel=_states_close(list(ref_model.values()),list(ddp_model.values()),atol=param_atol,rtol=param_rtol);opt_ok,opt_abs,opt_rel=_optimizer_states_close(ref_opt,ddp_opt,atol=param_atol,rtol=param_rtol)
    flags=torch.tensor([int(grad_ok),int(model_ok),int(opt_ok)],device=ctx.device,dtype=torch.int64);dist.all_reduce(flags,op=dist.ReduceOp.MIN);metrics=torch.tensor([grad_abs,grad_rel,param_abs,param_rel,opt_abs,opt_rel],device=ctx.device,dtype=torch.float64);dist.all_reduce(metrics,op=dist.ReduceOp.MAX)
    pipe.source_network=raw;del ddp
    return {'local_microsteps':int(local_microsteps),'checkpointing':bool(checkpointing),'global_scenes':gs,'global_motion_pairs':gp,'lambda_ref_for_acceptance':float(lambda_ref),'lambda_at_progress_0_5':float(weight),'rank_summaries':summaries,'gradient_allclose':bool(flags[0]),'model_after_step_allclose':bool(flags[1]),'optimizer_state_allclose':bool(flags[2]),'max_gradient_abs_diff':float(metrics[0]),'max_gradient_rel_diff':float(metrics[1]),'max_model_abs_diff':float(metrics[2]),'max_model_rel_diff':float(metrics[3]),'max_optimizer_abs_diff':float(metrics[4]),'max_optimizer_rel_diff':float(metrics[5]),'pass':bool(flags.min().item())}

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--override',action='append',default=[]);p.add_argument('--output',required=True);p.add_argument('--scan-windows',type=int,default=256);p.add_argument('--grad-atol',type=float,default=2e-5);p.add_argument('--grad-rtol',type=float,default=5e-4);p.add_argument('--param-atol',type=float,default=2e-6);p.add_argument('--param-rtol',type=float,default=2e-5);a=p.parse_args();cfg=load_config(a.config,a.override);ctx=init_distributed(require_cuda=True)
    try:
        env=formal_server_environment(ctx,cfg);seed=int(get(cfg,'training.initial_seed',3407));seed_all(seed);dr,info,manifest,msp=[get(cfg,x) for x in ('paths.dataroot','paths.info_pkl','paths.manifest','paths.msp_checkpoint')]
        if not all((dr,info,manifest,msp)):raise RuntimeError('DDP acceptance runtime paths incomplete')
        source=NuScenesWindowSource(dr,info_pkl=info,verbose=False);payload=load_manifest(manifest);ds=ManifestDataset(source,payload,'train');scan_pipe=MotionTransportV1(cfg,msp_checkpoint=msp,device=ctx.device)
        cases=None
        if ctx.is_main:cases=_scan_cases(scan_pipe,source,ds,cfg,a.scan_windows,seed)
        obj=[cases];dist.broadcast_object_list(obj,src=0);cases=obj[0];del scan_pipe;torch.cuda.empty_cache();acc=int(get(cfg,'training.accumulation_steps',4));locked=get(cfg,'loss.lambda_reference');lambda_ref=float(locked) if locked is not None and float(locked)>0 else 1.0
        specs=[('heterogeneous_full_accum','low','high',acc,False),('empty_source_rank','empty','high',acc,False),('tail_accumulation','low','high',1,False),('activation_checkpointing','low','high',acc,True)]
        if cases.get('nonempty_motion0') is not None:specs.append(('nonempty_zero_motion_pairs','nonempty_motion0','high',max(1,min(acc,2)),False))
        reports=[]
        for si,(name,k0,k1,micro,ckpt) in enumerate(specs):
            key=k0 if ctx.rank==0 else k1;row=cases[key];rep=_run_scenario(cfg,source,ds,ctx,row['index'],micro,ckpt,lambda_ref,seed+100+si,a.grad_atol,a.grad_rtol,a.param_atol,a.param_rtol);rep.update({'name':name,'rank0_case':cases[k0],'rank1_case':cases[k1]});reports.append(rep);barrier(ctx)
        required={'heterogeneous_full_accum','empty_source_rank','tail_accumulation','activation_checkpointing'};ok=required.issubset({r['name'] for r in reports if r['pass']});result={'spec_version':'MT-V1-SPEC-2','code_commit':_git(),'environment':env,'scan':cases,'tolerances':{'grad_atol':a.grad_atol,'grad_rtol':a.grad_rtol,'param_atol':a.param_atol,'param_rtol':a.param_rtol},'scenarios':reports,'pass':bool(ok)}
        if ctx.is_main:Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
        if not ok:raise RuntimeError('MT-V1 true 2-GPU optimizer-step equivalence failed')
    finally:
        barrier(ctx)
        if ctx.world_size>1:dist.destroy_process_group()

if __name__=='__main__':main()
