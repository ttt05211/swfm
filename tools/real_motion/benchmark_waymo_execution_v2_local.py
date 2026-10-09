#!/usr/bin/env python3
"""Full-grid synthetic CPU/CUDA execution diagnostic, NOT Waymo/L40S FPS or quality."""
import argparse
from contextlib import nullcontext
from pathlib import Path
import sys
if __package__ in (None,''): sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import time
import numpy as np
import torch
from real_motion.geometry import OccupancyGrid
from real_motion.prepared import PrepareConfig
from real_motion.local_st_world_model_v17 import LocalSTWMV17Config
from real_motion.joint_surface_ccr import JointSurfaceCCR
from real_motion.native_column_cpu import prepare_native
from real_motion.strong_majority_execution import ParallelNativeMajority,strong_majority_execution
from tools.real_motion.waymo_fast_execution import FastWaymoSurfaceProvider,FastSurfaceBlockExecution
from tools.real_motion.waymo_fast_execution_v2 import FastV2WaymoProvider,FastV2SurfaceExecution
from tools.real_motion.waymo_zero_shot_common import write_json


def workload(sources):
    grid=OccupancyGrid(); history=[]; poses=[]
    rng=np.random.default_rng(42); positions=rng.integers(10,185,(sources,2))
    for f in range(4):
        occ=np.full(grid.shape_hwd,17,np.uint8)
        occ[:,:140,1]=11; occ[:,145:180,1]=13
        for i,(x,y) in enumerate(positions): occ[x+f:x+f+3,y:y+4,2:5]=4 if i%3 else 2
        history.append(occ); pose=np.eye(4); pose[:3,3]=[f*.012,0,.005*f]; poses.append(pose)
    future=[]
    for h in range(6):
        pose=poses[-1].copy(); pose[0,3]+=(h+1)*.012; future.append(pose)
    return dict(scene_name='synthetic',t0_token='3',history_tokens=['0','1','2','3'],
        future_tokens=[str(i) for i in range(4,10)],sample_id='synthetic'),dict(
        history_occ=np.stack(history),history_observed=np.ones((4,*grid.shape_hwd),bool),
        history_poses=np.stack(poses),future_poses=np.stack(future),future_gt_occ=None)


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--out',required=True)
    p.add_argument('--repeats',type=int,default=3); p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    a=p.parse_args(); out=Path(a.out)
    if out.exists(): p.error('new diagnostic output required')
    if a.repeats<1 or a.repeats>20: p.error('repeats must be 1..20')
    if a.device=='cuda' and not torch.cuda.is_available(): p.error('actual CUDA required')
    torch.set_num_threads(1); torch.manual_seed(71); prepare_native()
    pcfg=PrepareConfig(grid=OccupancyGrid())
    model=JointSurfaceCCR(LocalSTWMV17Config(history_frames=4,d_model=16,semantic_dim=4,blocks=1,decoder_blocks=1),
                          width=16,z_bins=16).to(a.device).eval().requires_grad_(False)
    providers=dict(v1=FastWaymoSurfaceProvider(model,pcfg,a.device,4),v2=FastV2WaymoProvider(model,pcfg,a.device,4))
    executions=dict(v1=FastSurfaceBlockExecution(providers['v1'],workers=4,graphs=False),
                    v2=FastV2SurfaceExecution(providers['v2'],workers=4,graphs=False))
    majority=ParallelNativeMajority(4); results=[]
    def sync():
        if a.device=='cuda': torch.cuda.synchronize()
    def run(arm,record,raw):
        fresh={**raw}; tick=time.perf_counter()
        with strong_majority_execution(majority):
            prep=providers[arm].prepare_columns(None,record,include_gt=False,raw_window=fresh)
        sync(); prepared=time.perf_counter()-tick; tick=time.perf_counter()
        dense,_,stages,scores=executions[arm].predict(prep); sync()
        return prep,dense,scores,dict(prepare=prepared,evidence=stages['evidence_projection'],
            head_compose=time.perf_counter()-tick-stages['evidence_projection'],
            prepare_detail=providers[arm].fast_prepare_stages)
    try:
        for sources in (16,48):
            record,raw=workload(sources)
            old=run('v1',record,raw); new=run('v2',record,raw)
            for k in ('features','local_semantic_tube','kta_displacement_xy_m','frame_motion_features',
                      'target_source_mask_tube','source_class_id','anchors_xy_t0_m','source_centroid_xy_t0_m'):
                assert old[0].state['rec'][k].numpy().tobytes()==new[0].state['rec'][k].numpy().tobytes(),k
            assert old[2].tobytes()==new[2].tobytes()
            assert all(x.tobytes()==y.tobytes() for x,y in zip(old[1],new[1]))
            del old,new
            durations={arm:[] for arm in providers}; detail={arm:[] for arm in providers}
            for repeat in range(a.repeats):
                for arm in (('v1','v2') if repeat%2==0 else ('v2','v1')):
                    # Explicit warm single-window microbenchmark: NOT sequential throughput.
                    sync(); tick=time.perf_counter(); data=run(arm,record,raw); sync()
                    durations[arm].append(time.perf_counter()-tick); detail[arm].append(data[3]); del data
            means={k:float(np.mean(v)) for k,v in durations.items()}
            result=dict(requested_sources=sources,seconds=durations,mean=means,speedup=means['v1']/means['v2'],
                byte_parity=True,stages=detail)
            results.append(result); print(f'SYNTHETIC_GRID sources={sources} v1={means["v1"]:.4f} v2={means["v2"]:.4f} speedup={result["speedup"]:.3f} bytes=PASS',flush=True)
        write_json(out,dict(scope='synthetic 200x200x16 warm-window full model execution; random tiny weights; NOT actual Waymo/L40S/FPS/quality',
                           device=a.device,repeats=a.repeats,workloads=results))
    finally:
        for e in executions.values(): e.close()
        providers['v2'].close(); majority.close()
    return 0


if __name__=='__main__': sys.exit(main())
