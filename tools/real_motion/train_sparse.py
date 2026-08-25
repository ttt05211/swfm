#!/usr/bin/env python3
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; UP=ROOT/"upstream_occfm"; sys.path[:0]=[str(UP),str(ROOT)]
import torch
from torch.utils.data import DataLoader
from real_motion.dataset import RealMotionCacheDataset,collate_real_motion
from real_motion.windows import WindowPlanner,crop_windows
from real_motion.models import MotionWindowFlowMatching,RealMotionWindowCFM
from real_motion.checkpoint import load_shape_safe

def parse():
    p=argparse.ArgumentParser(); p.add_argument("--cache",required=True); p.add_argument("--upstream-ckpt",required=True); p.add_argument("--output",required=True); p.add_argument("--window",type=int,default=20); p.add_argument("--max-windows",type=int,default=8); p.add_argument("--batch-size",type=int,default=2); p.add_argument("--steps",type=int,default=2000); p.add_argument("--lr",type=float,default=2e-5); p.add_argument("--num-workers",type=int,default=4); p.add_argument("--amp",action="store_true"); return p.parse_args()
def make_transition(window,prior_channels):
    return MotionWindowFlowMatching(in_channels=16,out_channels=16,model_channels=128,channel_multi=[2,4],input_size=[window,window],trajectory_length=6,init_kernel_size=7,init_3d_conv_channels=64,attn_dim=32,temporal_attn_head=8,spatial_attn_head=8,prior_channels=prior_channels)
def main():
    a=parse(); device="cuda" if torch.cuda.is_available() else "cpu"; ds=RealMotionCacheDataset(a.cache); dl=DataLoader(ds,batch_size=a.batch_size,shuffle=True,num_workers=a.num_workers,collate_fn=collate_real_motion,drop_last=True,pin_memory=True); model=RealMotionWindowCFM(make_transition(a.window,32)).to(device); report=load_shape_safe(model.transition,a.upstream_ckpt,verbose=True); print("checkpoint reuse:",report["loaded"],"/",report["target_total"]); opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=0.01); scaler=torch.amp.GradScaler("cuda",enabled=a.amp and device=="cuda"); planner=WindowPlanner((a.window,a.window),a.max_windows); step=0; model.train()
    while step<a.steps:
        for batch in dl:
            batch={k:(v.to(device,non_blocking=True) if torch.is_tensor(v) else v) for k,v in batch.items()}; plan=planner.plan(batch.get("planning_support",batch["generation_support"])); hist=crop_windows(batch["moving_history_latent"],plan); fut=crop_windows(batch["future_moving_latent"],plan); sta=crop_windows(batch["static_future_latent"],plan); kta=crop_windows(batch["kta_future_latent"],plan); mask=crop_windows(batch["generation_support"].unsqueeze(2),plan); B,K=hist.shape[:2]; valid=plan.valid.reshape(-1)
            def flat(x): return x.reshape(B*K,*x.shape[2:])[valid]
            hist,fut,sta,kta,mask=map(flat,(hist,fut,sta,kta,mask)); origins=plan.origins.reshape(B*K,2)[valid]
            if hist.shape[0]==0: continue
            prior=torch.cat([sta,kta],dim=2); traj=batch.get("trajectory")
            if traj is not None: traj=traj[:,None].expand(B,K,*traj.shape[1:]).reshape(B*K,*traj.shape[1:])[valid]
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=a.amp): loss,info=model.flow_loss(hist,fut,prior,mask,trajectory=traj,window_origins=origins)
            scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); scaler.step(opt); scaler.update(); step+=1
            if step%20==0: print(f"step={step} loss={loss.item():.6f} active={float(info['active_fraction']):.4f}")
            if step>=a.steps: break
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); torch.save({"state_dict":model.state_dict(),"args":vars(a),"checkpoint_reuse":report},out); print("saved",out)
if __name__=="__main__": main()
