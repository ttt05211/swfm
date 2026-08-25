#!/usr/bin/env python3
"""Sample sparse future moving latents and scatter to a full latent canvas."""
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; UP=ROOT/"upstream_occfm"; sys.path[:0]=[str(UP),str(ROOT)]
import torch
from torch.utils.data import DataLoader
from real_motion.dataset import RealMotionCacheDataset,collate_real_motion
from real_motion.windows import WindowPlanner,crop_windows,scatter_windows
from real_motion.models import MotionWindowFlowMatching,RealMotionWindowCFM

def make_model(window):
    tr=MotionWindowFlowMatching(in_channels=16,out_channels=16,model_channels=128,channel_multi=[2,4],input_size=[window,window],trajectory_length=6,init_kernel_size=7,init_3d_conv_channels=64,attn_dim=32,temporal_attn_head=8,spatial_attn_head=8,prior_channels=32); return RealMotionWindowCFM(tr)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--cache",required=True); p.add_argument("--ckpt",required=True); p.add_argument("--output",required=True); p.add_argument("--empty-latent",required=True)
    p.add_argument("--window",type=int,default=20); p.add_argument("--max-windows",type=int,default=8); p.add_argument("--batch-size",type=int,default=1); a=p.parse_args(); dev="cuda" if torch.cuda.is_available() else "cpu"
    ds=RealMotionCacheDataset(a.cache); dl=DataLoader(ds,batch_size=a.batch_size,shuffle=False,collate_fn=collate_real_motion); model=make_model(a.window).to(dev)
    ck=torch.load(a.ckpt,map_location="cpu",weights_only=False); model.load_state_dict(ck["state_dict"],strict=True); model.eval()
    empty=torch.load(a.empty_latent,map_location="cpu",weights_only=False)
    if isinstance(empty,dict): empty=empty.get("empty_latent",empty.get("latent"))
    if not torch.is_tensor(empty) or empty.ndim!=3: raise ValueError("empty latent must be [C,H,W]")
    planner=WindowPlanner((a.window,a.window),a.max_windows); outputs=[]
    with torch.no_grad():
        for batch in dl:
            batch={k:(v.to(dev) if torch.is_tensor(v) else v) for k,v in batch.items()}; plan=planner.plan(batch.get("planning_support",batch["generation_support"]))
            hist=crop_windows(batch["moving_history_latent"],plan); sta=crop_windows(batch["static_future_latent"],plan); kta=crop_windows(batch["kta_future_latent"],plan); B,K=hist.shape[:2]; valid=plan.valid.reshape(-1)
            def flat(x): return x.reshape(B*K,*x.shape[2:])[valid]
            fhist,fsta,fkta=map(flat,(hist,sta,kta)); prior=torch.cat([fsta,fkta],dim=2); traj=batch.get("trajectory")
            if traj is not None: traj=traj[:,None].expand(B,K,*traj.shape[1:]).reshape(B*K,*traj.shape[1:])[valid]
            pred_valid=model.sample(fhist,tuple(fsta.shape[:2])+(16,a.window,a.window),prior,trajectory=traj); F=pred_valid.shape[1]
            pred_pad=torch.zeros(B*K,F,16,a.window,a.window,device=dev,dtype=pred_valid.dtype); pred_pad[valid]=pred_valid; pred_pad=pred_pad.reshape(B,K,F,16,a.window,a.window)
            base=empty.to(dev,dtype=pred_pad.dtype)[None,None].expand(B,F,-1,-1,-1); outputs.append(scatter_windows(pred_pad,plan,base=base).cpu())
    out=torch.cat(outputs,dim=0); Path(a.output).parent.mkdir(parents=True,exist_ok=True); torch.save({"future_moving_latent":out},a.output); print("saved",a.output,out.shape)
if __name__=="__main__": main()
