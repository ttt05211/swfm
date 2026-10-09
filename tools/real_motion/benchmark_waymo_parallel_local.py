#!/usr/bin/env python3
"""Synthetic full-grid spawn diagnostic; NOT actual Waymo/L40S/FPS/quality."""
import argparse
from pathlib import Path
import pickle
import sys
if __package__ in (None,''): sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from real_motion.geometry import OccupancyGrid
from real_motion.prepared import PrepareConfig
from real_motion.local_st_world_model_v17 import LocalSTWMV17Config
from real_motion.joint_surface_ccr import JointSurfaceCCR
from real_motion.waymo_i2world import file_sha256
from real_motion.waymo_i2world_10hz import WaymoI2World10HzSource
from real_motion.native_column_cpu import prepare_native
from real_motion.waymo_native_execution import prepare_waymo_native
from tools.real_motion.joint_surface_checkpoint_selection import evaluation_payload,AVERAGE_EPOCHS
from tools.real_motion.waymo_parallel_execution import paired_parallel_speed
from tools.real_motion.waymo_zero_shot_common import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--out-dir',required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda'); p.add_argument('--windows',type=int,default=16)
    p.add_argument('--repeats',type=int,default=2); a=p.parse_args()
    out=Path(a.out_dir).resolve()
    if out.exists() or not 4<=a.windows<=64 or not 1<=a.repeats<=4: p.error('new output / bounded workload required')
    if a.device=='cuda' and not torch.cuda.is_available(): p.error('actual CUDA required')
    out.mkdir(parents=True); root=out/'data'; root.mkdir(); prepare_native(); prepare_waymo_native()
    torch.set_num_threads(1); torch.manual_seed(71)
    pcfg=PrepareConfig(grid=OccupancyGrid()); infos=[]; poses={0:{}}
    rng=np.random.default_rng(42); cars=rng.integers(35,150,(16,2))
    for f in range(a.windows+10):
        infos.append(dict(timestamp=f*100000,image=dict(image_idx=1000000+f)))
        pose=np.eye(4); pose[:3,3]=[f*.012,0,f*.002]; poses[0][f]=[dict(ego2global=pose)]
    for name,value in (('waymo_infos_val.pkl',infos),('cam_infos_vali.pkl',poses)):
        with (root/name).open('wb') as handle: pickle.dump(value,handle)
    source=WaymoI2World10HzSource.from_files(root,cache_mib=32)
    for frame in source.frames:
        lab=np.full(source.shape,23,np.uint8); lab[:,:140,1]=13; lab[:,145:180,1]=14
        for x,y in cars: lab[x+frame.frame:x+frame.frame+3,y:y+4,2:5]=1
        frame.path.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(frame.path,voxel_label=lab)
    inventory=source.preflight(source.windows)
    model=JointSurfaceCCR(LocalSTWMV17Config(history_frames=4,d_model=16,semantic_dim=4,blocks=1,
        decoder_blocks=1),width=16,z_bins=16).eval().requires_grad_(False)
    checkpoint=out/'SYNTHETIC_NOT_SCIENTIFIC_MEAN.pt'
    torch.save(evaluation_payload(model.state_dict(),model.configs(),{},
        dict(positive_weights=model.columns.positive_weight.tolist()),
        [dict(epoch=e) for e in AVERAGE_EPOCHS],average=True),checkpoint); del model
    spec=dict(waymo_root=str(root),raw_free_label=23,frame_cache_mib=32,shape=source.shape,
        windows=len(source.windows),manifest_fingerprint=source.manifest_fingerprint,data=source.metadata,
        inventory=inventory,checkpoint=str(checkpoint),checkpoint_sha256=file_sha256(checkpoint),
        pcfg=pcfg,device=a.device,workers=2,backend='v2',graphs=a.device=='cuda',geometry_mib=256,
        surface_chunk=4096,parallel_majority=True,history_prefetch=True,implementation={})
    result=paired_parallel_speed(spec,list(range(4,4+a.windows)),processes=2,chunk=4,repeats=a.repeats)
    result['local_scope']='synthetic 200x200x16 / 16 sources / random tiny weights / actual spawn CUDA; NOT Waymo/L40S/FPS/quality'
    write_json(out/'speed.json',result)
    print('LOCAL_PARALLEL '+str(result['seconds_per_window'])+' speedup='+str(result['speedup'])+
          ' SIX/probability/counts=PASS',flush=True)
    return 0


if __name__=='__main__': sys.exit(main())
