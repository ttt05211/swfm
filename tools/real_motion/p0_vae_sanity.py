#!/usr/bin/env python3
"""P0-E: frozen VAE moving-only reconstruction and sparse-canvas sanity."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];UP=ROOT/'upstream_occfm';sys.path[:0]=[str(UP),str(ROOT)]
import torch
from real_motion.prepared import PreparedShardDataset
from real_motion.occfm_io import load_official_vae,OccFMVAEAdapter
from real_motion.support import downsample_support
from real_motion.metrics.moving_miou_v2 import MovingMIoUV2MultiHorizon
from real_motion.runtime_config import add_config_args,load_runtime_config,get_cfg,save_resolved_config
REPORT={1.0:1,2.0:3,3.0:5}
def main():
    p=argparse.ArgumentParser();add_config_args(p);p.add_argument('--prepared',required=True);p.add_argument('--vae-ckpt',required=True);p.add_argument('--output',required=True);p.add_argument('--max-windows',type=int,default=None);p.add_argument('--mode',choices=['sample','mean'],default=None);p.add_argument('--seed',type=int,default=20260826);p.add_argument('--latent-extra-radius',type=int,default=None);p.add_argument('--device',default='cuda');a=p.parse_args();cfg=load_runtime_config(a.config,a.override);mode=a.mode or get_cfg(cfg,'CACHE.VAE_LATENT_MODE','sample');extra=int(a.latent_extra_radius if a.latent_extra_radius is not None else get_cfg(cfg,'MOTION.LATENT_EXTRA_RADIUS',1));ds=PreparedShardDataset(a.prepared);n=len(ds) if a.max_windows is None else min(len(ds),a.max_windows);vae,_=load_official_vae(UP,a.vae_ckpt,a.device);ad=OccFMVAEAdapter(vae);empty=ad.empty_latent(mode=mode,seed=a.seed+999);full=MovingMIoUV2MultiHorizon();causal=MovingMIoUV2MultiHorizon();gtm=MovingMIoUV2MultiHorizon();rss=0.;rn=0
    for i in range(n):
        s=ds[i];z=ad.encode(torch.from_numpy(s['future_moving_occ']).unsqueeze(0),mode=mode,seed=a.seed+i)[0];pf=ad.decode_labels(z).cpu().numpy();cm=downsample_support(torch.from_numpy(s['generation_support_occ']).bool(),(50,50),extra_radius=extra).to(z.device);gm=downsample_support(torch.from_numpy(s['gt_moving_support']).any(dim=-1),(50,50),extra_radius=extra).to(z.device);es=empty[None].expand(z.shape[0],-1,-1,-1).to(z.dtype);zc=torch.where(cm[:,None],z,es);zg=torch.where(gm[:,None],z,es);pc=ad.decode_labels(zc).cpu().numpy();pg=ad.decode_labels(zg).cpu().numpy();outside=~cm[:,None].expand_as(z)
        if bool(outside.any()):rss+=float(((zc[outside]-es[outside])**2).sum().cpu());rn+=int(outside.sum().cpu())
        for h,fi in REPORT.items():gt=s['future_gt_occ'][fi];sup=s['gt_moving_support'][fi];full.update(h,pf[fi],gt,sup);causal.update(h,pc[fi],gt,sup);gtm.update(h,pg[fi],gt,sup)
    report={'num_windows':n,'latent_mode':mode,'full_moving_reconstruction':full.compute(),'causal_generation_support_canvas':causal.compute(),'gt_support_canvas_diagnostic':gtm.compute(),'outside_causal_support_empty_latent_rms':(rss/max(rn,1))**.5};op=Path(a.output);op.parent.mkdir(parents=True,exist_ok=True);op.write_text(json.dumps(report,indent=2),encoding='utf-8');save_resolved_config(cfg,op.with_suffix('.resolved.yaml'));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
