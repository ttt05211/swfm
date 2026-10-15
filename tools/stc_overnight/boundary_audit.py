"""Paired scene/horizon/class audit of already-complete real-subset controls.

No model, occupancy file, GPU, optimizer or new quality selection is invoked.
Resampling after repeated development is descriptive, not independent testing.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from .summarize import load_rows,score,binary,paired


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def audit(baseline,candidate):
    baseline,candidate=Path(baseline),Path(candidate)
    left,right=read(baseline/'results.json'),read(candidate/'results.json')
    if left['status']!='complete' or right['status']!='complete':raise ValueError('only complete comparisons')
    a,b=left['contract'],right['contract']
    for key in ('keys','model_sha256','modality','known','view','bundle_manifest_sha256','group_size'):
        if a.get(key)!=b.get(key):raise ValueError('not paired: '+key)
    keys=[tuple(k) for k in a['keys']]
    if len({k[0] for k in keys})!=len(keys):raise ValueError('one independent scene per window required by this audit')
    lr,rr=load_rows(baseline),load_rows(candidate)
    if set(lr)!=set(keys) or set(rr)!=set(keys):raise ValueError('missing/extra window counts')
    ac=np.stack([lr[k] for k in keys]);bc=np.stack([rr[k] for k in keys])
    report=paired(baseline,candidate)
    by_horizon={}
    for h,t in enumerate((1.,2.,3.)):
        aa,bb=score(ac.sum(0))['per_horizon'][str(t)],score(bc.sum(0))['per_horizon'][str(t)]
        per=[]
        for i in range(len(keys)):
            ai,bi=score(ac[i])['per_horizon'][str(t)],score(bc[i])['per_horizon'][str(t)]
            per.append({k:bi[k]-ai[k] for k in ('mIoU','IoU')})
        by_horizon[str(t)]=dict(baseline={k:aa[k] for k in ('mIoU','IoU')},
            candidate={k:bb[k] for k in ('mIoU','IoU')},delta={k:bb[k]-aa[k] for k in ('mIoU','IoU')},
            positive_scenes={k:sum(r[k]>0 for r in per) for k in ('mIoU','IoU')},
            baseline_binary=binary(ac[:,h:h+1].sum(0)),candidate_binary=binary(bc[:,h:h+1].sum(0)))
    windows=[]
    for i,key in enumerate(keys):
        aa,bb=score(ac[i])['average'],score(bc[i])['average']
        windows.append(dict(key=list(key),delta={k:bb[k]-aa[k] for k in ('mIoU','IoU')}))
    windows.sort(key=lambda r:r['delta']['mIoU'])
    report.update(by_horizon=by_horizon,worst5=windows[:5],best5=windows[-5:],
        observed_population='reference64, one window per scene; repeated development, not pristine holdout',
        correction_types='aggregate confusions do not identify which individual voxels were repaired/damaged',
        full_goal_verified=False,missing_evidence='ALL4369 paired final scores, not an extrapolated subset delta')
    return report


def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--baseline',required=True);ap.add_argument('--candidate',required=True)
    ap.add_argument('--out',required=True);a=ap.parse_args(argv)
    report=audit(a.baseline,a.candidate)
    Path(a.out).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(delta=report['delta'],by_horizon=report['by_horizon'],
        positive_scenes=report['positive_individual_window_counts'],
        leave_one_scene_out_range=report['leave_one_scene_out_range']),indent=2),flush=True)


if __name__=='__main__':main()
