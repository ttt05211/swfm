"""Virtual-view hidden-point-removal screen; NOT GT lidar visibility.

Only predicted voxel centers and historical ego poses enter the filter. Nominal
virtual height=1.8m; this is NOT a claim to reproduce actual sensor calibration.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from tools.final_semantic_evaluation import metrics
from .bundle import Bundle


def visible_points(points,origin):
    if len(points)<4:return np.arange(len(points))
    relative=points-np.asarray(origin);distance=np.linalg.norm(relative,axis=1)
    if np.any(distance<1e-6):raise ValueError('virtual origin lies on a sampled occupancy center')
    radius=100*distance.max()
    flipped=relative*(2*radius/distance-1)[:,None]
    hull=ConvexHull(np.concatenate((flipped,np.zeros((1,3))),axis=0))
    return hull.vertices[hull.vertices<len(points)]


def filter_frame(sem,poses,grid,*,multiview):
    ids=np.argwhere((sem>=0)&(sem<17))
    points=np.asarray([grid.x_min,grid.y_min,grid.z_min])+(ids+.5)*np.asarray(grid.voxel_size)
    origins=[np.array([0.,0.,1.8])]
    if multiview:
        inverse=np.linalg.inv(poses[-1])
        origins=[(inverse@p@np.array([0.,0.,1.8,1.]))[:3] for p in poses]
    keep=np.zeros(len(points),bool)
    for origin in origins:keep[visible_points(points,origin)]=True
    output=sem.copy();output[tuple(ids[~keep].T)]=17
    return output,dict(occupied_points=len(points),kept=int(keep.sum()))


def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--bundle',default='../stc_overnight_bundle.zip');ap.add_argument('--out',required=True)
    a=ap.parse_args(argv);b=Bundle(a.bundle);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    counts={n:metrics.empty(1) for n in ('raw','virtual_current_view','virtual_historical_views')}
    removed={n:dict(correct_occupied=0,false_occupied=0,total=0) for n in counts if n!='raw'}
    rows=[r for r in b.records if r['role']=='reference64']
    for i,r in enumerate(rows):
        _,raw,_=b.history(r);sem=raw['history_occ'][-1]
        current,audit1=filter_frame(sem,np.asarray(raw['history_poses']),b.pcfg.grid,multiview=False)
        multi,audit2=filter_frame(sem,np.asarray(raw['history_poses']),b.pcfg.grid,multiview=True)
        # Same-frame GT is accessed only AFTER both self-derived filters.
        gt=b.frame('gt',r['scene_name'],r['t0_token'])['semantics']
        for n,pred in [('raw',sem),('virtual_current_view',current),('virtual_historical_views',multi)]:
            metrics.add(counts[n],0,pred,gt)
            if n!='raw':
                changed=(sem!=17)&(pred==17)
                removed[n]['correct_occupied']+=int((changed&(sem==gt)).sum())
                removed[n]['false_occupied']+=int((changed&(gt==17)).sum())
                removed[n]['total']+=int(changed.sum())
        if (i+1)%8==0:print('VISIBILITY_T0',i+1,{n:metrics.report(c,(0.,),moving=False)['average'] for n,c in counts.items()},flush=True)
    result=dict(status='complete',windows=len(rows),removed=removed,
        scope='DIAGNOSTIC t0 input only; virtual geometry NOT GT mask; NOT future quality/full VAL',
        source='predicted occupied voxel centers only, nominal virtual height1.8, HPR radius100*max_range; no tuning',
        metrics={n:metrics.report(c,(0.,),moving=False) for n,c in counts.items()})
    (out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8');b.close()


if __name__=='__main__':main()
