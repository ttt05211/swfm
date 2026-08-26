#!/usr/bin/env python3
"""Encode prepared branches with frozen OccFM VAE into sharded latent cache."""
import argparse,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];UP=ROOT/'upstream_occfm';sys.path[:0]=[str(UP),str(ROOT)]
import torch
from real_motion.prepared import PreparedShardDataset
from real_motion.cache import save_sharded_cache
from real_motion.occfm_io import load_official_vae,OccFMVAEAdapter,file_sha256
from real_motion.support import downsample_support
from real_motion.runtime_config import add_config_args,load_runtime_config,get_cfg,save_resolved_config

def main():
    p=argparse.ArgumentParser();add_config_args(p);p.add_argument('--prepared',required=True);p.add_argument('--vae-ckpt',required=True);p.add_argument('--output',required=True);p.add_argument('--empty-latent',required=True);p.add_argument('--mode',choices=['sample','mean'],default=None);p.add_argument('--seed',type=int,default=20260826);p.add_argument('--shard-size',type=int,default=None);p.add_argument('--latent-extra-radius',type=int,default=None);p.add_argument('--keep-empty-support',action='store_true');p.add_argument('--device',default='cuda');a=p.parse_args();cfg=load_runtime_config(a.config,a.override);mode=a.mode or get_cfg(cfg,'CACHE.VAE_LATENT_MODE','sample');shard=int(a.shard_size or get_cfg(cfg,'CACHE.LATENT_SHARD_SIZE',256));extra=int(a.latent_extra_radius if a.latent_extra_radius is not None else get_cfg(cfg,'MOTION.LATENT_EXTRA_RADIUS',1));filter_empty=bool(get_cfg(cfg,'CACHE.FILTER_EMPTY_GENERATION_SUPPORT',True)) and not a.keep_empty_support;vae,_=load_official_vae(UP,a.vae_ckpt,a.device);ad=OccFMVAEAdapter(vae);ds=PreparedShardDataset(a.prepared);empty=ad.empty_latent(mode=mode,seed=a.seed+999).detach().cpu();Path(a.empty_latent).parent.mkdir(parents=True,exist_ok=True);torch.save({'empty_latent':empty,'mode':mode,'seed':a.seed+999},a.empty_latent);skipped=0
    def samples():
        nonlocal skipped
        for i in range(len(ds)):
            s=ds[i];gen=downsample_support(torch.from_numpy(s['generation_support_occ']).bool(),(50,50),extra_radius=extra).cpu()
            if filter_empty and not bool(gen.any()):skipped+=1;continue
            base=a.seed+i*17;mh=ad.encode(torch.from_numpy(s['moving_history_occ']).unsqueeze(0),mode=mode,seed=base)[0].cpu();ft=ad.encode(torch.from_numpy(s['future_dynamic_target_occ']).unsqueeze(0),mode=mode,seed=base+1)[0].cpu();st=ad.encode(torch.from_numpy(s['static_future_occ']).unsqueeze(0),mode=mode,seed=base+2)[0].cpu();kt=ad.encode(torch.from_numpy(s['kta_future_occ']).unsqueeze(0),mode=mode,seed=base+3)[0].cpu();hist=downsample_support(torch.from_numpy(s['history_candidate_support']).bool(),(50,50),extra_radius=extra).cpu();yield {'sample_id':s['sample_id'],'moving_history_latent':mh,'future_dynamic_target_latent':ft,'static_future_latent':st,'kta_future_latent':kt,'generation_support':gen,'planning_support':torch.cat([hist,gen],0),'trajectory':torch.as_tensor(s['trajectory'],dtype=torch.float32)}
    meta={'prepared':str(Path(a.prepared).resolve()),'vae_ckpt':str(Path(a.vae_ckpt).resolve()),'vae_ckpt_sha256':file_sha256(a.vae_ckpt),'latent_mode':mode,'seed':a.seed,'latent_extra_radius':extra,'filtered_empty_generation_support':filter_empty,'resolved_config':cfg};idx=save_sharded_cache(a.output,samples(),shard,meta);ip=Path(a.output)/'index.json';obj=json.loads(ip.read_text());obj['metadata'].update({'filtered_empty_generation_support':filter_empty,'skipped_empty_support':skipped});ip.write_text(json.dumps(obj,indent=2));save_resolved_config(cfg,Path(a.output)/'resolved_config.yaml');print('saved',idx['num_samples'],'skipped',skipped)
if __name__=='__main__':main()
