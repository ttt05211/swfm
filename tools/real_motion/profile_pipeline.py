#!/usr/bin/env python3
"""P0-F / final profiling scaffold for the actual online SWFM path."""
import argparse, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
UP = ROOT / "upstream_occfm"
sys.path[:0] = [str(UP), str(ROOT)]

import torch
from real_motion.nuscenes_adapter import NuScenesWindowSource
from real_motion.prepared import PrepareConfig, prepare_nuscenes_window, load_nuscenes_window_raw
from real_motion.occfm_io import load_official_vae, OccFMVAEAdapter, file_sha256
from real_motion.support import downsample_support
from real_motion.windows import WindowPlanner, crop_windows, scatter_windows
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


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataroot", required=True)
    p.add_argument("--info-pkl", required=True)
    p.add_argument("--vae-ckpt", required=True)
    p.add_argument("--sparse-ckpt", required=True)
    p.add_argument("--window-index", type=int, default=0)
    p.add_argument("--window", type=int, default=None)
    p.add_argument("--max-windows", type=int, default=None)
    p.add_argument("--latent-extra-radius", type=int, default=None)
    p.add_argument("--allow-support-override", action="store_true")
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--device", default="cuda")
    p.add_argument("--vae-mode", choices=["auto","sample","mean"], default="auto")
    p.add_argument("--seed", type=int, default=20260826)
    p.add_argument("--output", default=None)
    a = p.parse_args()

    source = NuScenesWindowSource(a.dataroot, info_pkl=a.info_pkl, verbose=False)
    windows = source.iter_windows(history=6, future=6)
    target = None
    for i,w in enumerate(windows):
        if i == a.window_index:
            target = w; break
    if target is None:
        raise IndexError("window-index out of range")

    ck = torch.load(a.sparse_ckpt, map_location="cpu", weights_only=False)
    train_args=ck.get("args",{}); cache_meta=ck.get("cache_metadata",{})
    trained_window=train_args.get("window")
    window=int(trained_window if a.window is None and trained_window is not None else (a.window or 20))
    if trained_window is not None and int(trained_window)!=window:
        raise ValueError(f"window-size mismatch: checkpoint trained with {trained_window}, requested {window}")
    max_windows=int(a.max_windows if a.max_windows is not None else train_args.get("max_windows",8))
    trained_extra=cache_meta.get("latent_extra_radius")
    latent_extra=int(trained_extra if a.latent_extra_radius is None and trained_extra is not None else (a.latent_extra_radius if a.latent_extra_radius is not None else 1))
    if trained_extra is not None and int(trained_extra)!=latent_extra and not a.allow_support_override:
        raise ValueError(f"latent support radius mismatch: checkpoint cache used {trained_extra}, requested {latent_extra}")
    expected_vae_hash=cache_meta.get("vae_ckpt_sha256")
    if expected_vae_hash and file_sha256(a.vae_ckpt)!=expected_vae_hash:
        raise ValueError("VAE checkpoint fingerprint differs from training cache")
    vae, _ = load_official_vae(UP, a.vae_ckpt, a.device)
    va = OccFMVAEAdapter(vae)
    model = make_model(window).to(a.device)
    model.load_state_dict(ck["state_dict"], strict=True)
    trained_mode=cache_meta.get("latent_mode")
    vae_mode=trained_mode if a.vae_mode=="auto" and trained_mode else ("sample" if a.vae_mode=="auto" else a.vae_mode)
    if trained_mode and vae_mode != trained_mode:
        raise ValueError(f"VAE latent-mode mismatch: checkpoint trained with {trained_mode}, requested {vae_mode}")
    empty = ck.get("empty_latent")
    if empty is None:
        raise KeyError("sparse checkpoint lacks the training empty_latent")
    empty = empty.to(a.device).detach()
    model.eval()
    planner = WindowPlanner((window,window), max_windows)
    # Disk I/O / nuScenes table lookup is data loading, not model latency. Preload
    # raw history + benchmark-provided future ego poses once outside timing.
    raw = load_nuscenes_window_raw(source, target, PrepareConfig(), include_gt=False)

    def run_once():
        times = {}
        sync(a.device); t=time.perf_counter()
        s = prepare_nuscenes_window(source, target, PrepareConfig(), include_gt=False, raw=raw)
        sync(a.device); times["preprocess_motion_se3_kta_ms"]=(time.perf_counter()-t)*1000

        sync(a.device); t=time.perf_counter()
        zh = va.encode(torch.from_numpy(s["moving_history_occ"]).unsqueeze(0), mode=vae_mode, seed=a.seed)
        zs = va.encode(torch.from_numpy(s["static_future_occ"]).unsqueeze(0), mode=vae_mode, seed=a.seed+1)
        zk = va.encode(torch.from_numpy(s["kta_future_occ"]).unsqueeze(0), mode=vae_mode, seed=a.seed+2)
        sync(a.device); times["condition_vae_encode_ms"]=(time.perf_counter()-t)*1000

        sync(a.device); t=time.perf_counter()
        gen = downsample_support(torch.from_numpy(s["generation_support_occ"]).bool().to(a.device),
                                 (50,50), latent_extra).unsqueeze(0)
        hist_ctx = downsample_support(torch.from_numpy(s["history_candidate_support"]).bool().to(a.device),
                                      (50,50), latent_extra).unsqueeze(0)
        context = torch.cat([hist_ctx,gen],dim=1)
        plan=planner.plan(gen,context_support=context)
        hist=crop_windows(zh,plan); sta=crop_windows(zs,plan); kta=crop_windows(zk,plan)
        active=crop_windows(gen.unsqueeze(2),plan)
        B,K=hist.shape[:2]; valid=plan.valid.reshape(-1); F=zs.shape[1]; C=zs.shape[2]
        ef=empty[None,None].expand(B,F,-1,-1,-1).to(zs.dtype)
        ew=crop_windows(ef,plan)
        noise=torch.randn((B,F,C,50,50),device=a.device,dtype=zs.dtype)
        nw=crop_windows(noise,plan)
        sync(a.device); times["support_plan_crop_ms"]=(time.perf_counter()-t)*1000

        if bool(valid.any()):
            def flat(x): return x.reshape(B*K,*x.shape[2:])[valid]
            fh,fs,fk,fa,fe,fn=map(flat,(hist,sta,kta,active,ew,nw))
            origins=plan.origins.reshape(B*K,2)[valid]
            traj=torch.as_tensor(s["trajectory"],device=a.device,dtype=zs.dtype).unsqueeze(0)
            traj=traj[:,None].expand(B,K,*traj.shape[1:]).reshape(B*K,*traj.shape[1:])[valid]
            sync(a.device); t=time.perf_counter()
            pred=model.sample(fh,tuple(fs.shape[:2])+(C,window,window),torch.cat([fs,fk],2),
                              fa,fe,trajectory=traj,window_origins=origins,initial_noise=fn)
            sync(a.device); times["sparse_wm_nfe_ms"]=(time.perf_counter()-t)*1000

            sync(a.device); t=time.perf_counter()
            pp=torch.zeros(B*K,F,C,window,window,device=a.device,dtype=pred.dtype); pp[valid]=pred
            full=scatter_windows(pp.reshape(B,K,F,C,window,window),plan,base=ef)
            sync(a.device); times["scatter_ms"]=(time.perf_counter()-t)*1000
        else:
            times["sparse_wm_nfe_ms"]=0.0
            sync(a.device); t=time.perf_counter(); full=ef
            sync(a.device); times["scatter_ms"]=(time.perf_counter()-t)*1000

        sync(a.device); t=time.perf_counter()
        wm_occ=va.decode_labels(full)[0].cpu()
        sync(a.device); times["vae_decode_ms"]=(time.perf_counter()-t)*1000

        t=time.perf_counter()
        static=torch.from_numpy(s["static_future_occ"])
        protected=torch.from_numpy(s["confident_static_future_mask"])
        write=torch.from_numpy(s["generation_support_occ"])
        _=static_protected_compose(static,wm_occ,protected,DYNAMIC_CLASS_IDS,write_support=write)
        times["composition_ms"]=(time.perf_counter()-t)*1000
        times["total_ms"]=sum(times.values())
        return times

    for _ in range(a.warmup): run_once()
    rows=[run_once() for _ in range(a.repeats)]
    report={k:float(sum(r[k] for r in rows)/len(rows)) for k in rows[0]}
    report["fps"]=1000.0/report["total_ms"]
    report["repeats"]=a.repeats
    print(json.dumps(report,indent=2))
    if a.output:
        Path(a.output).parent.mkdir(parents=True,exist_ok=True)
        Path(a.output).write_text(json.dumps(report,indent=2),encoding="utf-8")


if __name__=="__main__": main()
