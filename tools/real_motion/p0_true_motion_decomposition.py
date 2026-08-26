#!/usr/bin/env python3
"""P0-A: Full vs causal-static-only vs moving/uncertain-only frozen OccFM."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];UP=ROOT/'upstream_occfm';sys.path[:0]=[str(UP),str(ROOT)]
import numpy as np,torch
from real_motion.prepared import PreparedShardDataset
from real_motion.occfm_io import load_official_vae,load_official_wm,OccFMVAEAdapter,run_frozen_occfm_forecast
from real_motion.metrics.moving_miou_v2 import MovingMIoUV2MultiHorizon,SemanticIoUAccumulator,REPORT_HORIZONS_S
from real_motion.nuscenes_adapter import gt_true_static_mask
from real_motion.runtime_config import add_config_args,load_runtime_config,get_cfg,save_resolved_config
REPORT={1.0:1,2.0:3,3.0:5}
def mean_h(acc):
    per={h:a.compute() for h,a in acc.items()};return {'mIoU':float(np.nanmean([per[h]['mIoU'] for h in REPORT_HORIZONS_S])),'per_horizon':per}
def main():
    p=argparse.ArgumentParser();add_config_args(p);p.add_argument('--prepared',required=True);p.add_argument('--vae-ckpt',required=True);p.add_argument('--wm-ckpt',required=True);p.add_argument('--output',required=True);p.add_argument('--max-windows',type=int,default=None);p.add_argument('--vae-mode',choices=['sample','mean'],default=None);p.add_argument('--seed',type=int,default=20260826);p.add_argument('--device',default='cuda');a=p.parse_args();cfg=load_runtime_config(a.config,a.override);mode=a.vae_mode or get_cfg(cfg,'CACHE.VAE_LATENT_MODE','sample');ds=PreparedShardDataset(a.prepared);n=len(ds) if a.max_windows is None else min(len(ds),a.max_windows)
    if n==0:raise RuntimeError('no prepared windows')
    vae,_=load_official_vae(UP,a.vae_ckpt,a.device);wm,_=load_official_wm(UP,a.wm_ckpt,a.device);ad=OccFMVAEAdapter(vae);branches=('full','static','moving');mm={b:MovingMIoUV2MultiHorizon() for b in branches};overall={b:{h:SemanticIoUAccumulator() for h in REPORT_HORIZONS_S} for b in branches};ts={b:{h:SemanticIoUAccumulator() for h in REPORT_HORIZONS_S} for b in branches}
    for i in range(n):
        s=ds[i];seed=a.seed+i*101;hist={'full':ad.encode(torch.from_numpy(s['full_history_occ']).unsqueeze(0),mode=mode,seed=seed)[0].cpu(),'static':ad.encode(torch.from_numpy(s['static_history_occ']).unsqueeze(0),mode=mode,seed=seed)[0].cpu(),'moving':ad.encode(torch.from_numpy(s['moving_history_occ']).unsqueeze(0),mode=mode,seed=seed)[0].cpu()};zf=ad.encode(torch.from_numpy(s['future_gt_occ']).unsqueeze(0),mode=mode,seed=seed+1)[0].cpu();traj=torch.as_tensor(s['trajectory'],dtype=torch.float32);pred={b:run_frozen_occfm_forecast(wm,hist[b],zf,trajectory=traj,seed=a.seed+i*1009,hist_last=4).numpy() for b in branches}
        for h in REPORT_HORIZONS_S:
            fi=REPORT[h];gt=s['future_gt_occ'][fi];sup=s['gt_moving_support'][fi];sm=gt_true_static_mask(gt,sup)
            for b in branches:overall[b][h].update(pred[b][fi],gt);ts[b][h].update(pred[b][fi],gt,sm);mm[b].update(h,pred[b][fi],gt,sup)
    report={'protocol':'P0-A_frozen_causal_real_motion_decomposition','num_windows':n,'vae_mode':mode,'branches':{}}
    for b in branches:report['branches'][b]={'overall_mIoU':mean_h(overall[b]),'true_static_mIoU':mean_h(ts[b]),'Moving-mIoU_v2':mm[b].compute()}
    report['separability']={'static_only_minus_full_true_static_pp':report['branches']['static']['true_static_mIoU']['mIoU']-report['branches']['full']['true_static_mIoU']['mIoU'],'moving_only_minus_full_moving_pp':report['branches']['moving']['Moving-mIoU_v2']['mIoU']-report['branches']['full']['Moving-mIoU_v2']['mIoU']};op=Path(a.output);op.parent.mkdir(parents=True,exist_ok=True);op.write_text(json.dumps(report,indent=2),encoding='utf-8');save_resolved_config(cfg,op.with_suffix('.resolved.yaml'));print(json.dumps(report['separability'],indent=2))
if __name__=='__main__':main()
