#!/usr/bin/env python3
"""Encode prepared branches with frozen OccFM VAE into sharded latent cache.

L40S optimization:
- batch the frozen VAE over several prepared windows;
- keep VAE FP32 by default for representation parity (optional VAE AMP is explicit);
- optionally precompute sparse-window plans so training never runs the CPU planner.
"""
import argparse, sys, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; UP=ROOT/'upstream_occfm'; sys.path[:0]=[str(UP),str(ROOT)]
import numpy as np
import torch
from real_motion.prepared import PreparedShardDataset
from real_motion.cache import save_sharded_cache
from real_motion.occfm_io import load_official_vae, OccFMVAEAdapter, file_sha256
from real_motion.support import downsample_support
from real_motion.windows import WindowPlanner, window_coverage
from real_motion.runtime_config import add_config_args, load_runtime_config, get_cfg, save_resolved_config
from real_motion.perf import configure_cuda_runtime, vae_autocast_context


def main():
    p=argparse.ArgumentParser(); add_config_args(p)
    p.add_argument('--prepared',required=True); p.add_argument('--vae-ckpt',required=True)
    p.add_argument('--output',required=True); p.add_argument('--empty-latent',required=True)
    p.add_argument('--mode',choices=['sample','mean'],default=None); p.add_argument('--seed',type=int,default=20260826)
    p.add_argument('--shard-size',type=int,default=None)
    p.add_argument('--batch-size',type=int,default=None,help='VAE window batch; L40S default comes from YAML')
    p.add_argument('--latent-extra-radius',type=int,default=None); p.add_argument('--keep-empty-support',action='store_true')
    p.add_argument('--no-precompute-window-plan',action='store_true')
    p.add_argument('--device',default='cuda')
    a=p.parse_args(); cfg=load_runtime_config(a.config,a.override); device=torch.device(a.device)
    print('runtime device:',configure_cuda_runtime(cfg,device))
    mode=a.mode or get_cfg(cfg,'CACHE.VAE_LATENT_MODE','sample')
    shard=int(a.shard_size or get_cfg(cfg,'CACHE.LATENT_SHARD_SIZE',256))
    batch_size=int(a.batch_size or get_cfg(cfg,'CACHE.VAE_BATCH_SIZE',4))
    extra=int(a.latent_extra_radius if a.latent_extra_radius is not None else get_cfg(cfg,'MOTION.LATENT_EXTRA_RADIUS',1))
    filter_empty=bool(get_cfg(cfg,'CACHE.FILTER_EMPTY_GENERATION_SUPPORT',True)) and not a.keep_empty_support
    precompute=bool(get_cfg(cfg,'CACHE.PRECOMPUTE_WINDOW_PLAN',True)) and not a.no_precompute_window_plan
    wh,ww=map(int,get_cfg(cfg,'MODEL.WINDOW_HW',[20,20])); maxw=int(get_cfg(cfg,'MODEL.MAX_WINDOWS',8)); mincov=float(get_cfg(cfg,'MODEL.MIN_WINDOW_COVERAGE',.95))
    planner=WindowPlanner((wh,ww),maxw) if precompute else None

    vae,_=load_official_vae(UP,a.vae_ckpt,device); ad=OccFMVAEAdapter(vae); ds=PreparedShardDataset(a.prepared)
    with torch.inference_mode(),vae_autocast_context(cfg,device):
        empty=ad.empty_latent(mode=mode,seed=a.seed+999).detach().cpu()
    Path(a.empty_latent).parent.mkdir(parents=True,exist_ok=True)
    torch.save({'empty_latent':empty,'mode':mode,'seed':a.seed+999},a.empty_latent)
    skipped=0; min_plan_cov=1.0

    def samples():
        nonlocal skipped,min_plan_cov
        for start in range(0,len(ds),batch_size):
            rows=[]
            for i in range(start,min(start+batch_size,len(ds))):
                s=ds[i]
                gen=downsample_support(torch.from_numpy(s['generation_support_occ']).bool(),(50,50),extra_radius=extra).cpu()
                if filter_empty and not bool(gen.any()):
                    skipped+=1; continue
                hist=downsample_support(torch.from_numpy(s['history_candidate_support']).bool(),(50,50),extra_radius=extra).cpu()
                rows.append((i,s,gen,hist))
            if not rows: continue

            moving=torch.from_numpy(np.stack([r[1]['moving_history_occ'] for r in rows]))
            future=torch.from_numpy(np.stack([r[1]['future_dynamic_target_occ'] for r in rows]))
            static=torch.from_numpy(np.stack([r[1]['static_future_occ'] for r in rows]))
            kta=torch.from_numpy(np.stack([r[1]['kta_future_occ'] for r in rows]))
            batch_seed=a.seed+start*101
            with torch.inference_mode(),vae_autocast_context(cfg,device):
                mh=ad.encode(moving,mode=mode,seed=batch_seed).cpu()
                ft=ad.encode(future,mode=mode,seed=batch_seed+1).cpu()
                st=ad.encode(static,mode=mode,seed=batch_seed+2).cpu()
                kt=ad.encode(kta,mode=mode,seed=batch_seed+3).cpu()

            for j,(i,s,gen,hist) in enumerate(rows):
                planning=torch.cat([hist,gen],0)
                sample={
                    'sample_id':s['sample_id'],
                    'moving_history_latent':mh[j],
                    'future_dynamic_target_latent':ft[j],
                    'static_future_latent':st[j],
                    'kta_future_latent':kt[j],
                    'generation_support':gen,
                    'planning_support':planning,
                    'trajectory':torch.as_tensor(s['trajectory'],dtype=torch.float32),
                }
                if planner is not None:
                    plan=planner.plan(gen.unsqueeze(0),context_support=planning.unsqueeze(0))
                    cov=float(window_coverage(gen.unsqueeze(0),plan)[0])
                    min_plan_cov=min(min_plan_cov,cov)
                    if cov<mincov:
                        raise RuntimeError(
                            f"precomputed window coverage {cov:.3f} < {mincov:.3f} for {s['sample_id']}; "
                            "change WINDOW_HW/MAX_WINDOWS before building the formal cache"
                        )
                    sample['window_origins']=plan.origins[0].cpu()
                    sample['window_valid']=plan.valid[0].cpu()
                yield sample
            if start%(batch_size*25)==0:
                print('encoded',min(start+batch_size,len(ds)),'/',len(ds),'vae_batch',len(rows))

    meta={
        'prepared':str(Path(a.prepared).resolve()),'vae_ckpt':str(Path(a.vae_ckpt).resolve()),
        'vae_ckpt_sha256':file_sha256(a.vae_ckpt),'latent_mode':mode,'seed':a.seed,
        'vae_batch_size':batch_size,'vae_amp_enabled':bool(get_cfg(cfg,'RUNTIME.VAE_AMP.ENABLED',False)),
        'latent_extra_radius':extra,'filtered_empty_generation_support':filter_empty,
        'precomputed_window_plan':precompute,'window_plan_hw':[wh,ww] if precompute else None,
        'window_plan_max_windows':maxw if precompute else None,'resolved_config':cfg,
    }
    idx=save_sharded_cache(a.output,samples(),shard,meta)
    ip=Path(a.output)/'index.json'; obj=json.loads(ip.read_text())
    obj['metadata'].update({
        'filtered_empty_generation_support':filter_empty,'skipped_empty_support':skipped,
        'precomputed_window_plan':precompute,'window_plan_min_coverage':min_plan_cov if precompute else None,
    })
    ip.write_text(json.dumps(obj,indent=2)); save_resolved_config(cfg,Path(a.output)/'resolved_config.yaml')
    print('saved',idx['num_samples'],'skipped',skipped,'precomputed_plan',precompute)


if __name__=='__main__': main()
