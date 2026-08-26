#!/usr/bin/env python3
"""End-to-end causal SWFM inference for nuScenes/Occ3D windows."""
import argparse, sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
UP = ROOT / "upstream_occfm"
sys.path[:0] = [str(UP), str(ROOT)]

import numpy as np
import torch
from real_motion.nuscenes_adapter import NuScenesWindowSource
from real_motion.prepared import PrepareConfig, prepare_nuscenes_window
from real_motion.occfm_io import load_official_vae, OccFMVAEAdapter, file_sha256
from real_motion.support import downsample_support
from real_motion.windows import WindowPlanner, crop_windows, scatter_windows, window_coverage
from real_motion.models import MotionWindowFlowMatching, RealMotionWindowCFM
from real_motion.composition import static_protected_compose
from real_motion.metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS


def make_model(window):
    return RealMotionWindowCFM(MotionWindowFlowMatching(
        in_channels=16, out_channels=16, model_channels=128, channel_multi=[2,4],
        input_size=[window,window], trajectory_length=6, init_kernel_size=7,
        init_3d_conv_channels=64, attn_dim=32, temporal_attn_head=8,
        spatial_attn_head=8, prior_channels=32,
    ))


def infer_one(sample, va, model, empty, planner, device, latent_extra_radius=1,
              seed=0, min_coverage=0.95, vae_mode="sample"):
    # Match the latent representation used by the training cache. Seeded sample
    # mode is deterministic while staying on the official OccFM latent distribution.
    zh = va.encode(torch.from_numpy(sample["moving_history_occ"]).unsqueeze(0),
                   mode=vae_mode, seed=seed)[0].unsqueeze(0)
    zs = va.encode(torch.from_numpy(sample["static_future_occ"]).unsqueeze(0),
                   mode=vae_mode, seed=seed+1)[0].unsqueeze(0)
    zk = va.encode(torch.from_numpy(sample["kta_future_occ"]).unsqueeze(0),
                   mode=vae_mode, seed=seed+2)[0].unsqueeze(0)
    gen = downsample_support(torch.from_numpy(sample["generation_support_occ"]).bool().to(device),
                             (50,50), latent_extra_radius).unsqueeze(0)
    hist_ctx = downsample_support(torch.from_numpy(sample["history_candidate_support"]).bool().to(device),
                                  (50,50), latent_extra_radius).unsqueeze(0)
    context=torch.cat([hist_ctx, gen], dim=1)
    plan = planner.plan(gen, context_support=context)
    coverage = float(window_coverage(gen, plan).min())
    context_coverage = float(window_coverage(context, plan).mean())
    if coverage < min_coverage:
        raise RuntimeError(f"future latent support coverage {coverage:.3f} < {min_coverage:.3f}")

    hist=crop_windows(zh,plan); sta=crop_windows(zs,plan); kta=crop_windows(zk,plan)
    active=crop_windows(gen.unsqueeze(2),plan)
    B,K=hist.shape[:2]; F=zs.shape[1]; C=zs.shape[2]; valid=plan.valid.reshape(-1)
    ef=empty[None,None].expand(B,F,-1,-1,-1).to(device=device,dtype=zs.dtype)
    ew=crop_windows(ef,plan)
    g=torch.Generator(device=device); g.manual_seed(int(seed))
    global_noise=torch.randn((B,F,C,50,50),device=device,dtype=zs.dtype,generator=g)
    nw=crop_windows(global_noise,plan)
    if not bool(valid.any()):
        full=ef
    else:
        def flat(x): return x.reshape(B*K,*x.shape[2:])[valid]
        fh,fs,fk,fa,fe,fn=map(flat,(hist,sta,kta,active,ew,nw))
        origins=plan.origins.reshape(B*K,2)[valid]
        traj=torch.as_tensor(sample["trajectory"],device=device,dtype=zs.dtype).unsqueeze(0)
        traj=traj[:,None].expand(B,K,*traj.shape[1:]).reshape(B*K,*traj.shape[1:])[valid]
        pred=model.sample(fh,tuple(fs.shape[:2])+(C,planner.window_hw[0],planner.window_hw[1]),
                          torch.cat([fs,fk],2),fa,fe,trajectory=traj,
                          window_origins=origins,initial_noise=fn)
        # Redundant safety clamp before scatter.
        pred=torch.where(fa.bool().expand_as(pred),pred,fe)
        pad=torch.zeros(B*K,F,C,*planner.window_hw,device=device,dtype=pred.dtype)
        pad[valid]=pred
        full=scatter_windows(pad.reshape(B,K,F,C,*planner.window_hw),plan,base=ef)

    wm_occ=va.decode_labels(full)[0].cpu()
    final=static_protected_compose(
        torch.from_numpy(sample["static_future_occ"]), wm_occ,
        torch.from_numpy(sample["confident_static_future_mask"]),
        DYNAMIC_CLASS_IDS,
        write_support=torch.from_numpy(sample["generation_support_occ"]),
    )
    return final, {"window_coverage":coverage,"context_coverage":context_coverage,
                   "num_windows":int(plan.valid.sum()),
                   "slot_compute_ratio":int(plan.valid.sum())*planner.window_hw[0]*planner.window_hw[1]/2500.0}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataroot",required=True); p.add_argument("--info-pkl",required=True)
    p.add_argument("--vae-ckpt",required=True); p.add_argument("--sparse-ckpt",required=True)
    p.add_argument("--output",required=True); p.add_argument("--max-samples",type=int,default=None)
    p.add_argument("--window",type=int,default=None); p.add_argument("--max-windows",type=int,default=None)
    p.add_argument("--latent-extra-radius",type=int,default=None); p.add_argument("--allow-support-override",action="store_true")
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--vae-mode",choices=["auto","sample","mean"],default="auto")
    p.add_argument("--device",default="cuda"); a=p.parse_args()
    source=NuScenesWindowSource(a.dataroot,info_pkl=a.info_pkl,verbose=False)
    ck=torch.load(a.sparse_ckpt,map_location="cpu",weights_only=False)
    train_args=ck.get("args",{})
    cache_meta=ck.get("cache_metadata",{})
    trained_window=train_args.get("window")
    window=int(trained_window if a.window is None and trained_window is not None else (a.window or 20))
    if trained_window is not None and int(trained_window)!=window:
        raise ValueError(f"window-size mismatch: checkpoint trained with {trained_window}, requested {window}")
    max_windows=int(a.max_windows if a.max_windows is not None else train_args.get("max_windows",8))
    trained_extra=cache_meta.get("latent_extra_radius")
    latent_extra=int(trained_extra if a.latent_extra_radius is None and trained_extra is not None else (a.latent_extra_radius if a.latent_extra_radius is not None else 1))
    if trained_extra is not None and int(trained_extra)!=latent_extra and not a.allow_support_override:
        raise ValueError(f"latent support radius mismatch: checkpoint cache used {trained_extra}, requested {latent_extra}; pass --allow-support-override only for an explicit ablation")
    expected_vae_hash=cache_meta.get("vae_ckpt_sha256")
    if expected_vae_hash and file_sha256(a.vae_ckpt)!=expected_vae_hash:
        raise ValueError("VAE checkpoint fingerprint differs from the VAE used to build the training cache")
    vae,_=load_official_vae(UP,a.vae_ckpt,a.device); va=OccFMVAEAdapter(vae)
    model=make_model(window).to(a.device)
    model.load_state_dict(ck["state_dict"],strict=True); model.eval()
    trained_mode=cache_meta.get("latent_mode")
    vae_mode=trained_mode if a.vae_mode=="auto" and trained_mode else ("sample" if a.vae_mode=="auto" else a.vae_mode)
    if trained_mode and vae_mode != trained_mode:
        raise ValueError(f"VAE latent-mode mismatch: checkpoint trained with {trained_mode}, requested {vae_mode}")
    empty=ck.get("empty_latent")
    if empty is None:
        raise KeyError("sparse checkpoint lacks the training empty_latent; retrain/re-save with P0-ready code")
    empty=empty.to(a.device)
    planner=WindowPlanner((window,window),max_windows)
    root=Path(a.output); root.mkdir(parents=True,exist_ok=True)
    entries=[]
    with torch.no_grad():
        for i,w in enumerate(source.iter_windows(history=6,future=6,max_windows=a.max_samples)):
            sample=prepare_nuscenes_window(source,w,PrepareConfig(),include_gt=False)
            pred,meta=infer_one(sample,va,model,empty,planner,a.device,latent_extra,a.seed+i,vae_mode=vae_mode)
            name=f"{i:06d}.pt"
            torch.save({"pred_occ":pred,"sample_id":sample["sample_id"],"meta":meta},root/name)
            entries.append({"file":name,"sample_id":sample["sample_id"],"meta":meta})
            if i%20==0: print("inferred",i,sample["sample_id"],meta)
    (root/"index.json").write_text(json.dumps({
        "version":"swfm_predictions_v1","vae_mode":vae_mode,"seed":a.seed,
        "sparse_ckpt":str(Path(a.sparse_ckpt).resolve()),"entries":entries,
    },indent=2),encoding="utf-8")


if __name__=="__main__": main()
