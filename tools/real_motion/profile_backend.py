#!/usr/bin/env python3
import argparse,time,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];UP=ROOT/'upstream_occfm';sys.path[:0]=[str(ROOT),str(UP)]
import torch
from real_motion.models import MotionWindowFlowMatching


def sync():
    if torch.cuda.is_available():torch.cuda.synchronize()


def main():
    p=argparse.ArgumentParser();p.add_argument('--window',type=int,default=20);p.add_argument('--batch',type=int,default=4);p.add_argument('--warmup',type=int,default=10);p.add_argument('--iters',type=int,default=50);a=p.parse_args();dev='cuda' if torch.cuda.is_available() else 'cpu'
    m=MotionWindowFlowMatching(in_channels=16,out_channels=16,model_channels=128,channel_multi=[2,4],input_size=[a.window,a.window],trajectory_length=12,init_kernel_size=7,init_3d_conv_channels=64,attn_dim=32,temporal_attn_head=8,spatial_attn_head=8,prior_channels=32).to(dev).eval();x=torch.randn(a.batch,12,16,a.window,a.window,device=dev);prior=torch.randn(a.batch,12,32,a.window,a.window,device=dev);t=torch.full((a.batch,),500.,device=dev);traj=torch.randn(a.batch,12,2,device=dev);traj[:,:2]=0
    @torch.no_grad()
    def run():m({'noised_sequence':x,'prior_condition':prior,'timesteps':t,'trajectory':traj})
    for _ in range(a.warmup):run()
    sync();s=time.perf_counter()
    for _ in range(a.iters):run()
    sync();dt=(time.perf_counter()-s)/a.iters;print({'device':dev,'window':a.window,'batch_windows':a.batch,'trajectory_length':12,'latency_ms':dt*1000})


if __name__=='__main__':main()
