#!/usr/bin/env python3
"""Check official OccFM-fut 196 50x50 transition == modified zero-prior transition."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];UP=ROOT/'upstream_occfm';sys.path[:0]=[str(ROOT),str(UP)]
import torch
from forecast.models.modules.transition_models.FlowMatchingV2 import FLOW_MATCHING_DOWN_X4_DiT
from real_motion.models.transition import MotionWindowFlowMatching
from real_motion.checkpoint import load_shape_safe


def kwargs():
    return dict(in_channels=16,out_channels=16,model_channels=128,channel_multi=[2,4],input_size=[50,50],trajectory_length=12,init_kernel_size=7,init_3d_conv_channels=64,attn_dim=32,temporal_attn_head=8,spatial_attn_head=8)


def set_math_attention(m):
    for mod in m.modules():
        if hasattr(mod,'attention_mode'):mod.attention_mode='math'


def main():
    p=argparse.ArgumentParser();p.add_argument('--ckpt',required=True,help='official occfm_fut epoch=000196.ckpt');p.add_argument('--output',default=None);p.add_argument('--batch',type=int,default=1);p.add_argument('--seed',type=int,default=1234);p.add_argument('--rms-tol',type=float,default=2e-5);p.add_argument('--max-tol',type=float,default=2e-4);p.add_argument('--device',default='cuda');a=p.parse_args();torch.manual_seed(a.seed)
    official=FLOW_MATCHING_DOWN_X4_DiT(**kwargs()).to(a.device).eval();modified=MotionWindowFlowMatching(**kwargs(),prior_channels=32).to(a.device).eval();ro=load_shape_safe(official,a.ckpt,prefixes=('transition_model.',''),verbose=False);rm=load_shape_safe(modified,a.ckpt,prefixes=('transition_model.',''),verbose=False)
    for name,rep in [('official',ro),('modified',rm)]:
        if 'traj_encoder.0.weight' not in set(rep.get('loaded_keys',())):raise SystemExit(f'{name} transition did not load 12-step traj_encoder.0.weight; expected OccFM-fut epoch=000196.ckpt')
    set_math_attention(official);set_math_attention(modified);x=torch.randn(a.batch,12,16,50,50,device=a.device);t=torch.rand(a.batch,device=a.device)*1000;traj=torch.randn(a.batch,12,2,device=a.device);traj[:,:2]=0;prior=torch.zeros(a.batch,12,32,50,50,device=a.device);origins=torch.zeros(a.batch,2,dtype=torch.long,device=a.device)
    with torch.no_grad():y0=official.forward_single(x,t,traj);y1=modified.forward_single(x,t,traj,prior_condition=prior,window_origins=origins)
    diff=(y0-y1).float();report={'protocol':'occfm_fut_12step_v1','expected_checkpoint':'epoch=000196.ckpt','trajectory_length':12,'zero_prefix_steps':2,'official_loaded':ro['loaded'],'modified_loaded':rm['loaded'],'rms':float(diff.square().mean().sqrt().cpu()),'max_abs':float(diff.abs().max().cpu()),'rms_tol':a.rms_tol,'max_tol':a.max_tol};report['pass']=report['rms']<=a.rms_tol and report['max_abs']<=a.max_tol;print(json.dumps(report,indent=2))
    if a.output:
        Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(report,indent=2),encoding='utf-8')
    if not report['pass']:raise SystemExit('OccFM-fut 196 transition equivalence FAILED; do not train sparse WM')


if __name__=='__main__':main()
