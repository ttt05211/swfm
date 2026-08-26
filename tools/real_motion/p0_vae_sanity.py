#!/usr/bin/env python3
"""P0-E: frozen VAE sanity for true-moving and the actual WM target.

The Moving-mIoU target and the WM training target are intentionally different:
``future_moving_occ`` is metric-only, while ``future_dynamic_target_occ`` is
future dynamic-semantic GT inside causal generation support. P0-E therefore
checks both representations and the sparse E(empty) canvas used by training.
"""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];UP=ROOT/'upstream_occfm';sys.path[:0]=[str(UP),str(ROOT)]
import numpy as np
import torch
from real_motion.prepared import PreparedShardDataset
from real_motion.occfm_io import load_official_vae,OccFMVAEAdapter
from real_motion.support import downsample_support
from real_motion.metrics.moving_miou_v2 import (
    MovingMIoUV2MultiHorizon, SemanticIoUAccumulator, DYNAMIC_CLASS_IDS,
)
from real_motion.runtime_config import add_config_args,load_runtime_config,get_cfg,save_resolved_config

REPORT={1.0:1,2.0:3,3.0:5}


def _semantic_horizon_report(acc):
    per={h:acc[h].compute() for h in REPORT}
    vals=[per[h]['mIoU'] for h in REPORT if not np.isnan(per[h]['mIoU'])]
    return {'mIoU':float(np.mean(vals)) if vals else float('nan'),'per_horizon':per}


def _bev_to_voxel_mask(mask_bev, target):
    mask=np.asarray(mask_bev,dtype=bool)
    target=np.asarray(target)
    if mask.shape!=target.shape[:2]:
        raise ValueError(f'BEV support {mask.shape} does not match semantic target {target.shape}')
    return np.broadcast_to(mask[...,None],target.shape)


def main():
    p=argparse.ArgumentParser();add_config_args(p)
    p.add_argument('--prepared',required=True);p.add_argument('--vae-ckpt',required=True)
    p.add_argument('--output',required=True);p.add_argument('--max-windows',type=int,default=None)
    p.add_argument('--mode',choices=['sample','mean'],default=None);p.add_argument('--seed',type=int,default=20260826)
    p.add_argument('--latent-extra-radius',type=int,default=None);p.add_argument('--device',default='cuda')
    a=p.parse_args();cfg=load_runtime_config(a.config,a.override)
    mode=a.mode or get_cfg(cfg,'CACHE.VAE_LATENT_MODE','sample')
    extra=int(a.latent_extra_radius if a.latent_extra_radius is not None else get_cfg(cfg,'MOTION.LATENT_EXTRA_RADIUS',1))
    ds=PreparedShardDataset(a.prepared);n=len(ds) if a.max_windows is None else min(len(ds),a.max_windows)
    vae,_=load_official_vae(UP,a.vae_ckpt,a.device);ad=OccFMVAEAdapter(vae)
    empty=ad.empty_latent(mode=mode,seed=a.seed+999)

    true_moving=MovingMIoUV2MultiHorizon()
    sparse_moving_projection=MovingMIoUV2MultiHorizon()
    gt_support_diag=MovingMIoUV2MultiHorizon()
    wm_target={h:SemanticIoUAccumulator(DYNAMIC_CLASS_IDS) for h in REPORT}
    sparse_wm_target={h:SemanticIoUAccumulator(DYNAMIC_CLASS_IDS) for h in REPORT}
    rss=0.;rn=0

    for i in range(n):
        s=ds[i]
        # Use the same stochastic seed for both branches so differences reflect
        # semantic content, not different VAE epsilon draws.
        seed=a.seed+i
        z_moving=ad.encode(torch.from_numpy(s['future_moving_occ']).unsqueeze(0),mode=mode,seed=seed)[0]
        z_wm=ad.encode(torch.from_numpy(s['future_dynamic_target_occ']).unsqueeze(0),mode=mode,seed=seed)[0]
        p_moving=ad.decode_labels(z_moving).cpu().numpy()
        p_wm=ad.decode_labels(z_wm).cpu().numpy()

        cm=downsample_support(
            torch.from_numpy(s['generation_support_occ']).bool(),(50,50),extra_radius=extra
        ).to(z_wm.device)
        gm=downsample_support(
            torch.from_numpy(s['gt_moving_support']).any(dim=-1),(50,50),extra_radius=extra
        ).to(z_moving.device)
        es=empty[None].expand(z_wm.shape[0],-1,-1,-1).to(z_wm.dtype)

        # Branch C: the actual training target represented on a causal sparse
        # latent canvas, with all M_gen-exterior cells clamped to exact E(empty).
        z_sparse=torch.where(cm[:,None],z_wm,es)
        p_sparse=ad.decode_labels(z_sparse).cpu().numpy()

        # Legacy diagnostic retained: true-moving latent clipped by GT support.
        z_gt=torch.where(gm[:,None],z_moving,es)
        p_gt=ad.decode_labels(z_gt).cpu().numpy()

        outside=~cm[:,None].expand_as(z_sparse)
        if bool(outside.any()):
            rss+=float(((z_sparse[outside]-es[outside])**2).sum().cpu());rn+=int(outside.sum().cpu())

        for h,fi in REPORT.items():
            gt=s['future_gt_occ'][fi]; moving_sup=s['gt_moving_support'][fi]
            target=s['future_dynamic_target_occ'][fi]
            causal_mask=_bev_to_voxel_mask(s['generation_support_occ'][fi],target)

            # A. Can frozen VAE reconstruct Moving-mIoU-eligible content?
            true_moving.update(h,p_moving[fi],gt,moving_sup)
            # B. Can it reconstruct the actual WM supervision contract?
            wm_target[h].update(p_wm[fi],target,mask=causal_mask)
            # C. Does sparse E(empty) clamping preserve that WM target?
            sparse_wm_target[h].update(p_sparse[fi],target,mask=causal_mask)
            # Also project branch C onto the frozen Moving-mIoU v2 for downstream relevance.
            sparse_moving_projection.update(h,p_sparse[fi],gt,moving_sup)
            gt_support_diag.update(h,p_gt[fi],gt,moving_sup)

    true_report=true_moving.compute()
    wm_report=_semantic_horizon_report(wm_target)
    sparse_report=_semantic_horizon_report(sparse_wm_target)
    sparse_projection=sparse_moving_projection.compute()
    report={
        'num_windows':n,
        'latent_mode':mode,
        'true_moving_reconstruction':true_report,
        'wm_target_reconstruction':wm_report,
        'causal_sparse_wm_target_canvas':{
            'wm_target_dynamic_mIoU':sparse_report,
            'moving_mIoU_v2_projection':sparse_projection,
        },
        'gt_support_canvas_diagnostic':gt_support_diag.compute(),
        'outside_causal_support_empty_latent_rms':(rss/max(rn,1))**.5,
        # Backward-compatible alias for old P0-E result readers.
        'full_moving_reconstruction':true_report,
    }
    op=Path(a.output);op.parent.mkdir(parents=True,exist_ok=True)
    op.write_text(json.dumps(report,indent=2),encoding='utf-8')
    save_resolved_config(cfg,op.with_suffix('.resolved.yaml'));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
