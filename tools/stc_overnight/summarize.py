"""Evidence summary: keep oracle/supervised studies separate from full64 controls."""
import argparse
import json
from pathlib import Path
import numpy as np
from tools.final_semantic_evaluation import metrics


def load_rows(root):
    values={}
    for f in root.glob('window_*.json'):
        r=json.loads(f.read_text(encoding='utf-8'))
        values[tuple(r['key'])]=np.asarray(r['confusion']['semantic_refine'],np.int64)
    return values


def score(c):return metrics.report(dict(confusion=c,moving=np.zeros_like(c)),(1.,2.,3.),moving=False)


def binary(c):
    return dict(TP=int(c[:,:17,:17].sum()),FP=int(c[:,17,:17].sum()),FN=int(c[:,:17,17].sum()))


def paired(base,candidate,draws=1000):
    left,right=load_rows(base),load_rows(candidate)
    if set(left)!=set(right):raise ValueError('nonmatching windows in paired report')
    keys=sorted(left);a=np.stack([left[k] for k in keys]);b=np.stack([right[k] for k in keys])
    aa=score(a.sum(0));bb=score(b.sum(0));rng=np.random.default_rng(21)
    values=[]
    for i in range(draws):
        at=rng.integers(0,len(keys),len(keys));ra,rb=score(a[at].sum(0)),score(b[at].sum(0))
        values.append([rb['average'][k]-ra['average'][k] for k in ('mIoU','IoU')])
    positive={k:0 for k in ('mIoU','IoU')};loo=[]
    for i in range(len(keys)):
        for k in positive:positive[k]+=score(b[i])['average'][k]>score(a[i])['average'][k]
        va,vb=score(a.sum(0)-a[i]),score(b.sum(0)-b[i]);loo.append([vb['average'][k]-va['average'][k] for k in positive])
    delta={k:bb['average'][k]-aa['average'][k] for k in positive}
    class_delta={str(c):float(np.mean([bb['per_horizon'][str(h)]['semantic_per_class'][str(c)]-
        aa['per_horizon'][str(h)]['semantic_per_class'][str(c)] for h in (1.,2.,3.)])) for c in range(17)}
    result=dict(windows=len(keys),baseline=aa['average'],candidate=bb['average'],delta=delta,
        binary_baseline=binary(a.sum(0)),binary_candidate=binary(b.sum(0)),
        positive_individual_window_counts=positive,class_delta=class_delta,
        paired_bootstrap_percentile95={k:np.percentile(np.asarray(values)[:,i],[2.5,97.5]).tolist() for i,k in enumerate(positive)},
        leave_one_scene_out_range={k:[float(np.min(np.asarray(loo)[:,i])),float(np.max(np.asarray(loo)[:,i]))] for i,k in enumerate(positive)},
        limitation='repeated development selection; resampling is conditional diagnosis, NOT independent significance or full VAL proof')
    return result


def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--root',default='outputs/stc_overnight');a=ap.parse_args(argv);root=Path(a.root)
    rows=[]
    for d in sorted(root.iterdir()):
        if not d.is_dir() or not (d/'results.json').is_file():continue
        r=json.loads((d/'results.json').read_text(encoding='utf-8'))
        for name,value in r.get('metrics',{}).items():
            rows.append(dict(experiment=d.name,route=name,status=r['status'],windows=r.get('windows'),**value['average']))
    reports={}
    for base,candidate in [('dense_identity','dense_kta_recency'),('clean_identity','clean_kta_recency')]:
        if (root/candidate/'window_063.json').exists():reports[candidate]=paired(root/base,root/candidate)
    result=dict(scope='LOCAL REAL SUBSET ONLY; full target mIoU>18.97 AND IoU>28.77 NOT VERIFIED',
        rows=rows,paired=reports)
    (root/'research_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    for name,r in reports.items():print(name,json.dumps(r,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
