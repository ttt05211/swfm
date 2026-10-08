#!/usr/bin/env python3
"""CPU geometry microbenchmark only; synthetic, NOT quality/FPS or server ETA."""
import sys
from pathlib import Path
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import time
import numpy as np
from real_motion.canonical_causal_repair import CanonicalEvidence,FEATURE_DIM,STATIC,map_canonical_evidence
from real_motion.surface_canonical_repair import SurfaceAtlas,augment_evidence,augment_projection
from real_motion.surface_projection_execution import map_surface_evidence
from real_motion.geometry import OccupancyGrid


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--points',type=int,default=60000)
    p.add_argument('--repeats',type=int,default=5);a=p.parse_args()
    if a.points<1 or a.repeats<1:p.error('positive budgets required')
    rng=np.random.default_rng(81);grid=OccupancyGrid(x_min=-40,y_min=-40,z_min=-1,
        voxel_size=(.4,.4,.4),shape_hwd=(200,200,16))
    at=np.column_stack((rng.integers(0,200,a.points),rng.integers(0,200,a.points),rng.integers(2,4,a.points)))
    world=np.array([-40,-40,-1])+(at+.5)*.4
    actor=np.full(a.points,STATIC,np.int32);actor[::20]=0
    classes=np.where(actor>=0,4,np.where(at[:,0]<100,11,13)).astype(np.uint8)
    presence=np.ones((a.points,4),bool);presence[::6]=False
    e=CanonicalEvidence(rng.normal(size=(a.points,FEATURE_DIM)).astype(np.float32),
        np.tile(classes[:,None],(1,4)),actor,classes,world,presence,{})
    matrices=np.tile(np.eye(4),(6,1,1));matrices[:,0,3]=np.arange(6)*.13
    baseline=np.full((6,*grid.shape_hwd),17,np.uint8)
    prep=SimpleNamespace(raw={'future_poses':np.linalg.inv(matrices)},state=dict(current_pose=np.eye(4),
        current=[dict(centroid_world=np.zeros(3))],world_to_future=matrices),baseline=baseline,
        owners=np.full_like(baseline,-1,dtype=np.int32),fallbacks=baseline,
        targets=np.zeros((6,1,3)),yaws=np.zeros((6,1)))
    def describe(workers):
        atlas=SurfaceAtlas(e.world,e.classes,e.presence,e.actor,np.eye(4),grid);atlas.query_workers=workers
        return augment_evidence(e,atlas)
    enriched=describe(1);np.testing.assert_array_equal(enriched.features,describe(4).features)
    samples={k:[] for k in ('descriptor_reference','descriptor_parallel','projection_reference','projection_fused')}
    with ThreadPoolExecutor(max_workers=4) as pool:
        def projection(fused):
            fn=map_surface_evidence if fused else map_canonical_evidence
            plan=fn(enriched,prep,grid,executor=pool)
            return augment_projection(enriched,plan,np.eye(4),matrices,grid)
        left,right=projection(False),projection(True)
        for key in ('flat','base','fallback','legal','context'):
            np.testing.assert_array_equal(getattr(left,key),getattr(right,key))
        jobs=dict(descriptor_reference=lambda:describe(1),descriptor_parallel=lambda:describe(4),
                  projection_reference=lambda:projection(False),projection_fused=lambda:projection(True))
        for i in range(a.repeats):
            order=list(jobs);order=order[i%len(order):]+order[:i%len(order)]
            for name in order:
                start=time.perf_counter();value=jobs[name]();samples[name].append(time.perf_counter()-start);del value
    print(f'SYNTHETIC CPU ONLY points={a.points} repeats={a.repeats} complete_geometry_bytes_exact=PASS')
    for name,values in samples.items():print(f'{name}: mean_ms={1000*np.mean(values):.3f} median_ms={1000*np.median(values):.3f}')
    print('NOT full-network FPS, real-data quality, actual joint throughput, or an L40S speedup claim.')


if __name__=='__main__':main()
