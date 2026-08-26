#!/usr/bin/env python3
"""Build sharded raw prepared windows from nuScenes + Occ3D under OccFM-fut-196."""
import argparse,sys
from pathlib import Path
from dataclasses import asdict
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from real_motion.nuscenes_adapter import NuScenesWindowSource
from real_motion.prepared import prepare_nuscenes_window,save_prepared_shards,PREPARED_VERSION
from real_motion.runtime_config import add_config_args,load_runtime_config,make_prepare_config,get_cfg,save_resolved_config


def main():
    p=argparse.ArgumentParser();add_config_args(p);p.add_argument('--dataroot',required=True);p.add_argument('--info-pkl',required=True,help='trusted temporal info pickle containing gt_ego_fut_trajs for all 12 window frames');p.add_argument('--output',required=True);p.add_argument('--max-windows',type=int,default=None);p.add_argument('--stride',type=int,default=1);p.add_argument('--shard-size',type=int,default=None);a=p.parse_args();cfg=load_runtime_config(a.config,a.override);pcfg=make_prepare_config(cfg);shard=int(a.shard_size or get_cfg(cfg,'CACHE.PREPARED_SHARD_SIZE',16));source=NuScenesWindowSource(a.dataroot,info_pkl=a.info_pkl,verbose=False)
    def gen():
        for i,w in enumerate(source.iter_windows(history=pcfg.history_frames,future=pcfg.future_frames,stride=a.stride,max_windows=a.max_windows)):
            s=prepare_nuscenes_window(source,w,pcfg)
            if i%25==0:print('prepared',i,s['sample_id'],'traj',s['trajectory'].shape)
            yield s
    meta={'prepared_version':PREPARED_VERSION,'dataroot':str(Path(a.dataroot).resolve()),'info_pkl':str(Path(a.info_pkl).resolve()),'history_frames':pcfg.history_frames,'future_frames':pcfg.future_frames,'frame_dt_s':pcfg.frame_dt_s,'trajectory_protocol':pcfg.trajectory_protocol,'trajectory_length':pcfg.trajectory_length,'trajectory_hist_last':pcfg.trajectory_hist_last,'trajectory_zero_prefix':pcfg.trajectory_zero_prefix,'require_temporal_info':pcfg.require_temporal_info,'upstream_wm_variant':'occfm_fut','upstream_init_variant':'fut_traj_196','tube_radii':list(pcfg.tube_radii),'motion_config':asdict(pcfg.motion),'kta_config':asdict(pcfg.kta),'grid':asdict(pcfg.grid),'causal':True,'resolved_config':cfg};idx=save_prepared_shards(a.output,gen(),shard,meta);save_resolved_config(cfg,Path(a.output)/'resolved_config.yaml');print('saved',idx['num_samples'],'protocol',pcfg.trajectory_protocol)


if __name__=='__main__':main()
