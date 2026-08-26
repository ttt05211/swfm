#!/usr/bin/env python3
"""End-to-end latency profiler for the actual online SWFM path."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];UP=ROOT/'upstream_occfm';sys.path[:0]=[str(ROOT),str(UP)]
import torch
from real_motion.nuscenes_adapter import NuScenesWindowSource
from real_motion.prepared import prepare_nuscenes_window,load_nuscenes_window_raw
from real_motion.occfm_io import load_official_vae,OccFMVAEAdapter,file_sha256
from real_motion.support import downsample_support
from real_motion.windows import WindowPlan,WindowPlanner,crop_windows,scatter_windows
from real_motion.composition import static_protected_compose
from real_motion.metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS
from real_motion.runtime_config import add_config_args,load_runtime_config,get_cfg,make_prepare_config,save_resolved_config,validate_runtime_config
from real_motion.perf import configure_cuda_runtime,autocast_context,vae_autocast_context,maybe_compile,cuda_device_summary
from tools.real_motion.infer_nuscenes import make_model

def sync(device):
    if torch.device(device).type=='cuda':torch.cuda.synchronize(device)
def main():
    p=argparse.ArgumentParser();add_config_args(p);p.add_argument('--dataroot',required=True);p.add_argument('--info-pkl',required=True);p.add_argument('--vae-ckpt',required=True);p.add_argument('--sparse-ckpt',required=True);p.add_argument('--window-index',type=int,default=0);p.add_argument('--repeats',type=int,default=10);p.add_argument('--warmup',type=int,default=3);p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,default=20260826);p.add_argument('--output',default=None);a=p.parse_args();fallback=load_runtime_config(a.config,a.override);device=torch.device(a.device);print('runtime device:',configure_cuda_runtime(fallback,device));source=NuScenesWindowSource(a.dataroot,info_pkl=a.info_pkl,verbose=False);target=None
    for i,w in enumerate(source.iter_windows(history=6,future=6)):
        if i==a.window_index:target=w;break
    if target is None:raise IndexError('window-index out of range')
    ck=torch.load(a.sparse_ckpt,map_location='cpu',weights_only=False);cfg=ck.get('resolved_config',fallback);validate_runtime_config(cfg);configure_cuda_runtime(cfg,device);cache=ck.get('cache_metadata',{});wh,ww=map(int,get_cfg(cfg,'MODEL.WINDOW_HW',[20,20]));maxw=int(get_cfg(cfg,'MODEL.MAX_WINDOWS',8));latent=int(cache.get('latent_extra_radius',get_cfg(cfg,'MOTION.LATENT_EXTRA_RADIUS',1)));mode=cache.get('latent_mode',get_cfg(cfg,'CACHE.VAE_LATENT_MODE','sample'));expected=cache.get('vae_ckpt_sha256')
    if expected and file_sha256(a.vae_ckpt)!=expected:raise ValueError('VAE checkpoint fingerprint mismatch')
    vae,_=load_official_vae(UP,a.vae_ckpt,device);va=OccFMVAEAdapter(vae);model=make_model(wh,cfg).to(device);model.load_state_dict(ck['state_dict'],strict=True);model.eval();model.transition=maybe_compile(model.transition,cfg,'inference');empty=ck.get('empty_latent')
    if empty is None:raise KeyError('checkpoint lacks empty_latent')
    planner=WindowPlanner((wh,ww),maxw);pcfg=make_prepare_config(cfg);raw=load_nuscenes_window_raw(source,target,pcfg,include_gt=False)
    def run_once():
        times={};sync(device);t=time.perf_counter();s=prepare_nuscenes_window(source,target,pcfg,include_gt=False,raw=raw);sync(device);times['preprocess_motion_se3_kta_ms']=(time.perf_counter()-t)*1000
        sync(device);t=time.perf_counter()
        with vae_autocast_context(cfg,device):
            zh=va.encode(torch.from_numpy(s['moving_history_occ']).unsqueeze(0),mode=mode,seed=a.seed);zs=va.encode(torch.from_numpy(s['static_future_occ']).unsqueeze(0),mode=mode,seed=a.seed+1);zk=va.encode(torch.from_numpy(s['kta_future_occ']).unsqueeze(0),mode=mode,seed=a.seed+2)
        sync(device);times['condition_vae_encode_ms']=(time.perf_counter()-t)*1000
        sync(device);t=time.perf_counter();gen_cpu=downsample_support(torch.from_numpy(s['generation_support_occ']).bool(),(50,50),latent).unsqueeze(0);histctx_cpu=downsample_support(torch.from_numpy(s['history_candidate_support']).bool(),(50,50),latent).unsqueeze(0);plan_cpu=planner.plan(gen_cpu,context_support=torch.cat([histctx_cpu,gen_cpu],1));gen=gen_cpu.to(device,non_blocking=True);plan=WindowPlan(plan_cpu.origins.to(device,non_blocking=True),plan_cpu.valid.to(device,non_blocking=True),plan_cpu.window_hw,plan_cpu.full_hw);hist=crop_windows(zh,plan);sta=crop_windows(zs,plan);kta=crop_windows(zk,plan);active=crop_windows(gen.unsqueeze(2),plan);B,K=hist.shape[:2];F=zs.shape[1];C=zs.shape[2];ef=empty.to(device,dtype=zs.dtype)[None,None].expand(B,F,-1,-1,-1);ew=crop_windows(ef,plan);noise=crop_windows(torch.randn((B,F,C,50,50),device=device,dtype=zs.dtype),plan);valid=plan.valid.reshape(-1);sync(device);times['support_plan_crop_ms']=(time.perf_counter()-t)*1000
        if bool(valid.any()):
            def flat(x):return x.reshape(B*K,*x.shape[2:])[valid]
            fh,fs,fk,fa,fe,fn=map(flat,(hist,sta,kta,active,ew,noise));orig=plan.origins.reshape(B*K,2)[valid];traj=torch.as_tensor(s['trajectory'],device=device,dtype=zs.dtype).unsqueeze(0);traj=traj[:,None].expand(B,K,*traj.shape[1:]).reshape(B*K,*traj.shape[1:])[valid];sync(device);t=time.perf_counter()
            with autocast_context(cfg,device):pred=model.sample(fh,tuple(fs.shape[:2])+(C,wh,ww),torch.cat([fs,fk],2),fa,fe,trajectory=traj,window_origins=orig,initial_noise=fn)
            sync(device);times['sparse_wm_nfe_ms']=(time.perf_counter()-t)*1000;sync(device);t=time.perf_counter();pad=torch.zeros(B*K,F,C,wh,ww,device=device,dtype=pred.dtype);pad[valid]=torch.where(fa.bool().expand_as(pred),pred,fe);full=scatter_windows(pad.reshape(B,K,F,C,wh,ww),plan,base=ef);sync(device);times['scatter_ms']=(time.perf_counter()-t)*1000
        else:times['sparse_wm_nfe_ms']=0.;times['scatter_ms']=0.;full=ef
        sync(device);t=time.perf_counter()
        with vae_autocast_context(cfg,device):wm=va.decode_labels(full)[0].cpu()
        sync(device);times['vae_decode_ms']=(time.perf_counter()-t)*1000;t=time.perf_counter();_=static_protected_compose(torch.from_numpy(s['static_future_occ']),wm,torch.from_numpy(s['confident_static_future_mask']),DYNAMIC_CLASS_IDS,write_support=torch.from_numpy(s['generation_support_occ']));times['composition_ms']=(time.perf_counter()-t)*1000;times['total_ms']=sum(times.values());return times
    with torch.inference_mode():
        for _ in range(a.warmup):run_once()
        rows=[run_once() for _ in range(a.repeats)]
    report={k:float(sum(r[k] for r in rows)/len(rows)) for k in rows[0]};report.update({'fps':1000./report['total_ms'],'repeats':a.repeats,'hardware':cuda_device_summary(device),'compile_inference':bool(get_cfg(cfg,'RUNTIME.COMPILE.INFERENCE',False))});print(json.dumps(report,indent=2))
    if a.output:
        op=Path(a.output);op.parent.mkdir(parents=True,exist_ok=True);op.write_text(json.dumps(report,indent=2),encoding='utf-8');save_resolved_config(cfg,op.with_suffix('.resolved.yaml'))
if __name__=='__main__':main()
