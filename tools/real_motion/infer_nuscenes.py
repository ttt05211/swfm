#!/usr/bin/env python3
"""End-to-end causal SWFM inference under official OccFM-fut-196 ego conditioning."""
import argparse,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];UP=ROOT/'upstream_occfm';sys.path[:0]=[str(ROOT),str(UP)]
import torch
from real_motion.nuscenes_adapter import NuScenesWindowSource
from real_motion.prepared import prepare_nuscenes_window
from real_motion.occfm_io import load_official_vae,OccFMVAEAdapter,file_sha256
from real_motion.support import downsample_support
from real_motion.windows import WindowPlan,WindowPlanner,crop_windows,scatter_windows,window_coverage
from real_motion.models import MotionWindowFlowMatching,RealMotionWindowCFM
from real_motion.composition import static_protected_compose
from real_motion.metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS
from real_motion.runtime_config import add_config_args,load_runtime_config,get_cfg,make_prepare_config,save_resolved_config,validate_runtime_config
from real_motion.perf import configure_cuda_runtime,autocast_context,vae_autocast_context,maybe_compile,cuda_device_summary


def make_model(window,cfg=None):
    cfg=cfg or {};prior=int(get_cfg(cfg,'MODEL.PRIOR_CHANNELS',32));rescale=float(get_cfg(cfg,'MODEL.RESCALE_FACTOR',10));steps=int(get_cfg(cfg,'MODEL.SAMPLE_STEPS',10));alpha=float(get_cfg(cfg,'MODEL.ALPHA_SHIFT',3));traj_len=int(get_cfg(cfg,'MODEL.TRAJECTORY_LENGTH',12))
    if traj_len!=12:raise ValueError('OccFM-fut 196 requires MODEL.TRAJECTORY_LENGTH=12')
    tr=MotionWindowFlowMatching(in_channels=16,out_channels=16,model_channels=128,channel_multi=[2,4],input_size=[window,window],trajectory_length=traj_len,init_kernel_size=7,init_3d_conv_channels=64,attn_dim=32,temporal_attn_head=8,spatial_attn_head=8,prior_channels=prior)
    return RealMotionWindowCFM(tr,rescale_factor=rescale,sample_steps=steps,alpha_shift=alpha)


def infer_one(sample,va,model,empty,planner,device,latent_extra_radius=1,seed=0,min_coverage=.95,vae_mode='sample',cfg=None):
    cfg=cfg or {};traj_len=int(get_cfg(cfg,'MODEL.TRAJECTORY_LENGTH',12));traj_np=sample.get('trajectory')
    if traj_np is None or tuple(traj_np.shape)!=(traj_len,2):raise RuntimeError(f'online OccFM-fut trajectory must be [{traj_len},2], got {None if traj_np is None else traj_np.shape}')
    with vae_autocast_context(cfg,device):
        zh=va.encode(torch.from_numpy(sample['moving_history_occ']).unsqueeze(0),mode=vae_mode,seed=seed)[0].unsqueeze(0)
        zs=va.encode(torch.from_numpy(sample['static_future_occ']).unsqueeze(0),mode=vae_mode,seed=seed+1)[0].unsqueeze(0)
        zk=va.encode(torch.from_numpy(sample['kta_future_occ']).unsqueeze(0),mode=vae_mode,seed=seed+2)[0].unsqueeze(0)
    gen_cpu=downsample_support(torch.from_numpy(sample['generation_support_occ']).bool(),(50,50),latent_extra_radius).unsqueeze(0);hist_cpu=downsample_support(torch.from_numpy(sample['history_candidate_support']).bool(),(50,50),latent_extra_radius).unsqueeze(0);context_cpu=torch.cat([hist_cpu,gen_cpu],1);plan_cpu=planner.plan(gen_cpu,context_support=context_cpu);coverage=float(window_coverage(gen_cpu,plan_cpu).min());context_cov=float(window_coverage(context_cpu,plan_cpu).mean())
    if coverage<min_coverage:raise RuntimeError(f'future latent support coverage {coverage:.3f} < {min_coverage:.3f}')
    gen=gen_cpu.to(device,non_blocking=True);plan=WindowPlan(plan_cpu.origins.to(device,non_blocking=True),plan_cpu.valid.to(device,non_blocking=True),plan_cpu.window_hw,plan_cpu.full_hw);hist=crop_windows(zh,plan);sta=crop_windows(zs,plan);kta=crop_windows(zk,plan);active=crop_windows(gen.unsqueeze(2),plan);B,K=hist.shape[:2];F=zs.shape[1];C=zs.shape[2];valid=plan.valid.reshape(-1);ef=empty[None,None].expand(B,F,-1,-1,-1).to(device=device,dtype=zs.dtype);ew=crop_windows(ef,plan);g=torch.Generator(device=device);g.manual_seed(int(seed));noise=crop_windows(torch.randn((B,F,C,50,50),device=device,dtype=zs.dtype,generator=g),plan)
    if not bool(valid.any()):full=ef
    else:
        def flat(x):return x.reshape(B*K,*x.shape[2:])[valid]
        fh,fs,fk,fa,fe,fn=map(flat,(hist,sta,kta,active,ew,noise));orig=plan.origins.reshape(B*K,2)[valid];traj=torch.as_tensor(traj_np,device=device,dtype=zs.dtype).unsqueeze(0);traj=traj[:,None].expand(B,K,*traj.shape[1:]).reshape(B*K,*traj.shape[1:])[valid]
        with autocast_context(cfg,device):pred=model.sample(fh,tuple(fs.shape[:2])+(C,*planner.window_hw),torch.cat([fs,fk],2),fa,fe,trajectory=traj,window_origins=orig,initial_noise=fn)
        pred=torch.where(fa.bool().expand_as(pred),pred,fe);pad=torch.zeros(B*K,F,C,*planner.window_hw,device=device,dtype=pred.dtype);pad[valid]=pred;full=scatter_windows(pad.reshape(B,K,F,C,*planner.window_hw),plan,base=ef)
    with vae_autocast_context(cfg,device):wm=va.decode_labels(full)[0].cpu()
    final=static_protected_compose(torch.from_numpy(sample['static_future_occ']),wm,torch.from_numpy(sample['confident_static_future_mask']),DYNAMIC_CLASS_IDS,write_support=torch.from_numpy(sample['generation_support_occ']))
    return final,{'window_coverage':coverage,'context_coverage':context_cov,'num_windows':int(plan.valid.sum()),'slot_compute_ratio':int(plan.valid.sum())*planner.window_hw[0]*planner.window_hw[1]/2500.0,'trajectory_protocol':sample.get('trajectory_protocol','occfm_fut_12step_v1')}


def main():
    p=argparse.ArgumentParser();add_config_args(p);p.add_argument('--dataroot',required=True);p.add_argument('--info-pkl',required=True,help='official temporal info containing gt_ego_fut_trajs for every 6+6 frame');p.add_argument('--vae-ckpt',required=True);p.add_argument('--sparse-ckpt',required=True);p.add_argument('--output',required=True);p.add_argument('--max-samples',type=int,default=None);p.add_argument('--seed',type=int,default=0);p.add_argument('--vae-mode',choices=['auto','sample','mean'],default='auto');p.add_argument('--device',default='cuda');a=p.parse_args();fallback=load_runtime_config(a.config,a.override);ck=torch.load(a.sparse_ckpt,map_location='cpu',weights_only=False);cfg=ck.get('resolved_config',fallback);validate_runtime_config(cfg);device=torch.device(a.device);print('runtime device:',configure_cuda_runtime(cfg,device));cache=ck.get('cache_metadata',{});wh,ww=map(int,get_cfg(cfg,'MODEL.WINDOW_HW',[20,20]));maxw=int(get_cfg(cfg,'MODEL.MAX_WINDOWS',8));latent=int(cache.get('latent_extra_radius',get_cfg(cfg,'MOTION.LATENT_EXTRA_RADIUS',1)));mincov=float(get_cfg(cfg,'MODEL.MIN_WINDOW_COVERAGE',.95));expected=cache.get('vae_ckpt_sha256')
    if expected and file_sha256(a.vae_ckpt)!=expected:raise ValueError('VAE checkpoint fingerprint differs from training cache')
    trained_mode=cache.get('latent_mode');mode=trained_mode if a.vae_mode=='auto' and trained_mode else ('sample' if a.vae_mode=='auto' else a.vae_mode)
    if trained_mode and mode!=trained_mode:raise ValueError(f'VAE latent-mode mismatch: trained {trained_mode}, requested {mode}')
    source=NuScenesWindowSource(a.dataroot,info_pkl=a.info_pkl,verbose=False);vae,_=load_official_vae(UP,a.vae_ckpt,device);va=OccFMVAEAdapter(vae);model=make_model(wh,cfg).to(device);model.load_state_dict(ck['state_dict'],strict=True);model.eval();model.transition=maybe_compile(model.transition,cfg,'inference');empty=ck.get('empty_latent')
    if empty is None:raise KeyError('checkpoint lacks training empty_latent')
    planner=WindowPlanner((wh,ww),maxw);pcfg=make_prepare_config(cfg);root=Path(a.output);root.mkdir(parents=True,exist_ok=True);entries=[]
    with torch.inference_mode():
        for i,w in enumerate(source.iter_windows(history=pcfg.history_frames,future=pcfg.future_frames,max_windows=a.max_samples)):
            s=prepare_nuscenes_window(source,w,pcfg,include_gt=False);pred,meta=infer_one(s,va,model,empty.to(device),planner,device,latent,a.seed+i,mincov,mode,cfg);name=f'{i:06d}.pt';torch.save({'pred_occ':pred,'sample_id':s['sample_id'],'meta':meta},root/name);entries.append({'file':name,'sample_id':s['sample_id'],'meta':meta})
            if i%20==0:print('inferred',i,s['sample_id'],meta)
    (root/'index.json').write_text(json.dumps({'version':'swfm_predictions_v4_occfm_fut196','vae_mode':mode,'trajectory_protocol':get_cfg(cfg,'EGO_PROTOCOL.NAME'),'seed':a.seed,'sparse_ckpt':str(Path(a.sparse_ckpt).resolve()),'hardware':cuda_device_summary(device),'entries':entries},indent=2),encoding='utf-8');save_resolved_config(cfg,root/'resolved_config.yaml')


if __name__=='__main__':main()
