"""Read-only hard-label diagnostics; never change scores, predictions or weights."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np

CLASSES = ['others','barrier','bicycle','bus','car','construction_vehicle','motorcycle',
    'pedestrian','traffic_cone','trailer','truck','driveable_surface','other_flat',
    'sidewalk','terrain','manmade','vegetation','free']
REGIONS = ('all', 'prior_full', 'prior_partial', 'prior_none')


def confusion(pred, gt, mask):
    return np.bincount((18*gt[mask].astype(np.int64)+pred[mask]).ravel(), minlength=324).reshape(18, 18)


def ratio(n, d): return 100*float(n)/float(d) if d else None


def report(c):
    diagonal = np.diag(c); union = c.sum(0)+c.sum(1)-diagonal
    sem = np.divide(diagonal[:17]*100, union[:17], out=np.full(17, np.nan), where=union[:17] > 0)
    occupied_gt, occupied_pred = c[:17].sum(), c[:, :17].sum()
    intersection = c[:17, :17].sum(); fp = c[17, :17].sum(); fn = c[:17, 17].sum()
    return dict(voxels=int(c.sum()), mIoU=float(np.nanmean(sem)) if np.isfinite(sem).any() else None,
        IoU=ratio(intersection, intersection+fp+fn), FP=int(fp), FN=int(fn),
        occupied_GT=int(occupied_gt), occupied_pred=int(occupied_pred),
        occupancy_precision=ratio(intersection, occupied_pred), occupancy_recall=ratio(intersection, occupied_gt),
        semantic_correct=int(diagonal.sum()), road_precision=ratio(c[11, 11], c[:, 11].sum()),
        road_recall=ratio(c[11, 11], c[11].sum()),
        per_class_IoU={CLASSES[i]:float(v) if np.isfinite(v) else None for i, v in enumerate(sem)})


def arrays(z, name):
    with np.load(io.BytesIO(z.read(name)), allow_pickle=False) as f: return {k:f[k] for k in f.files}


def run(archive, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        if len(z.namelist()) != len(set(z.namelist())): raise ValueError('duplicate archive member')
        evidence = json.loads(z.read('evaluation.json')); contract = json.loads(z.read('contract.json'))
        manifest = json.loads(z.read('visuals/manifest.json'))
        table64 = {}; class_delta = {}
        for kind in ('occ', 'stc'):
            table64[kind] = {}
            for h in ('1.0', '2.0', '3.0'):
                old = evidence['reports']['control/'+kind]['semantic_refine']['per_horizon'][h]
                new = evidence['reports']['candidate/'+kind]['semantic_refine']['per_horizon'][h]
                prior = evidence['reports']['candidate/'+kind]['transport']['per_horizon'][h]
                table64[kind][h] = dict(control={k:old[k] for k in ('mIoU','IoU')},
                    candidate={k:new[k] for k in ('mIoU','IoU')},
                    candidate_minus_control={k:new[k]-old[k] for k in ('mIoU','IoU')},
                    candidate_refiner_minus_own_prior={k:new[k]-prior[k] for k in ('mIoU','IoU')})
            values = []
            for i in range(17):
                ds = [evidence['reports']['candidate/'+kind]['semantic_refine']['per_horizon'][h]['semantic_per_class'][str(i)]-
                    evidence['reports']['control/'+kind]['semantic_refine']['per_horizon'][h]['semantic_per_class'][str(i)]
                    for h in ('1.0','2.0','3.0')]
                values.append(dict(class_id=i, name=CLASSES[i], delta_IoU=float(np.mean(ds))))
            class_delta[kind] = sorted(values, key=lambda v:v['delta_IoU'])
        counts = {}; edits = {}; pairs = []
        for kind in ('occ','stc'):
            for case in manifest['cases']:
                index = case['index']; prefix = f'visuals/{kind}_{index:02d}'
                old = arrays(z, prefix+'_control.npz'); new = arrays(z, prefix+'_candidate.npz')
                if not np.array_equal(old['gt'], new['gt']) or not np.array_equal(old['t0'], new['t0']):
                    raise ValueError('paired inputs/targets differ')
                if new['prior_coverage'].dtype != np.uint8: raise ValueError('expected quantized coverage')
                for hi in range(3):
                    gt = old['gt'][hi]; cov = new['prior_coverage'][hi]
                    masks = dict(all=np.ones_like(gt,bool), prior_full=cov==255,
                        prior_partial=(cov>0)&(cov<255), prior_none=cov==0)
                    variants = dict(control_prior=old['transport'][hi], control_refiner=old['prediction'][hi],
                        candidate_prior=new['transport'][hi], candidate_refiner=new['prediction'][hi])
                    for region, mask in masks.items():
                        key = f'{kind}/{hi+1}s/{region}'
                        bucket = counts.setdefault(key,{n:np.zeros((18,18),np.int64) for n in variants})
                        item = edits.setdefault(key,dict(voxels=0, control_changed=0, candidate_changed=0,
                            control_helped=0, control_harmed=0, candidate_helped=0, candidate_harmed=0,
                            pair_helped=0, pair_harmed=0, candidate_to_free=0, candidate_from_free=0))
                        item['voxels'] += int(mask.sum())
                        for name, pred in variants.items(): bucket[name] += confusion(pred,gt,mask)
                        for arm, values in (('control',old),('candidate',new)):
                            prior, pred = values['transport'][hi], values['prediction'][hi]
                            item[arm+'_changed'] += int(((prior != pred)&mask).sum())
                            item[arm+'_helped'] += int(((prior != gt)&(pred == gt)&mask).sum())
                            item[arm+'_harmed'] += int(((prior == gt)&(pred != gt)&mask).sum())
                        cp, npred = old['prediction'][hi], new['prediction'][hi]
                        item['pair_helped'] += int(((cp != gt)&(npred == gt)&mask).sum())
                        item['pair_harmed'] += int(((cp == gt)&(npred != gt)&mask).sum())
                        item['candidate_to_free'] += int(((new['transport'][hi] != 17)&(npred == 17)&mask).sum())
                        item['candidate_from_free'] += int(((new['transport'][hi] == 17)&(npred != 17)&mask).sum())
                    pairs.append(dict(kind=kind, index=index, scene=case['scene'], horizon=hi+1,
                        prior_uncovered_voxels=int((cov==0).sum()),
                        control_changes=int((old['prediction'][hi] != old['transport'][hi]).sum()),
                        candidate_changes=int((new['prediction'][hi] != new['transport'][hi]).sum())))
        reports = {k:{name:report(c) for name,c in v.items()} for k,v in counts.items()}
        result = dict(archive=str(Path(archive).resolve()), archive_sha256=hashlib.sha256(Path(archive).read_bytes()).hexdigest(),
            full64_per_horizon=table64, full64_class_deltas=class_delta,
            exported8_region_diagnostics=reports, exported8_edits=edits, cases=pairs,
            scope=dict(official='original FULL GRID64 scores unchanged', arrays='fixed first8 per modality ONLY',
                regions='candidate Prior coverage shared for BOTH models; uint8 255/full,0/none,otherwise partial',
                caveats='not Memory coverage; no logits, model weights, or training logs in this archive'),
            training_objective=contract['objective'])
        (out/'diagnostics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        # Keep just two existing preview PNGs, not an extracted dataset.
        for name in ('occ_01_bev.png','stc_01_bev.png'):
            (out/name).write_bytes(z.read('visuals/'+name))
        print('FULL64')
        print(json.dumps(table64,indent=2))
        print('CLASS DELTAS worst6', json.dumps({k:v[:6] for k,v in class_delta.items()}))
        for kind in ('occ','stc'):
            for region in ('all','prior_full','prior_none'):
                for h in range(1,4):
                    key=f'{kind}/{h}s/{region}'; e=edits[key]; r=reports[key]
                    print(key,json.dumps(dict(voxels=e['voxels'],
                        old_edits_pct=ratio(e['control_changed'],e['voxels']), new_edits_pct=ratio(e['candidate_changed'],e['voxels']),
                        old_refiner=r['control_refiner'], new_refiner=r['candidate_refiner'],
                        net_harmed=e['pair_harmed']-e['pair_helped'])))
        print('OUTPUT',out/'diagnostics.json')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.archive,a.out)
