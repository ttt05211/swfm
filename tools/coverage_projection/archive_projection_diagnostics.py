"""Hard-label projection errors from exported8; NOT actual soft training loss."""
import argparse
import io
import json
from pathlib import Path
import zipfile
import numpy as np
import torch
from torch.nn import functional as F
from .projection import VolumeProjection


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path);p.add_argument('--device',default='cuda:0')
    a=p.parse_args();torch.set_num_threads(1);device=torch.device(a.device)
    with zipfile.ZipFile(a.archive) as z:
        manifest=json.loads(z.read('visuals/manifest.json'));g=manifest['grid']
        renderer=VolumeProjection(g['origin'],g['step'],g['shape'])
        def read(name):
            with np.load(io.BytesIO(z.read(name)),allow_pickle=False) as f:return {k:f[k] for k in f.files}
        def projection(volume,view):
            labels=torch.as_tensor(volume.copy(),device=device).long()
            onehot=F.one_hot(labels,18).permute(3,0,1,2).float()
            return renderer.render(onehot,view)
        rows=[]
        with torch.no_grad():
            for kind in ('occ','stc'):
                for case in manifest['cases']:
                    old=read(f'visuals/{kind}_{case["index"]:02d}_control.npz')
                    new=read(f'visuals/{kind}_{case["index"]:02d}_candidate.npz')
                    for hi in range(3):
                        for view in ('top','oblique'):
                            gs,gd,hit=projection(old['gt'][hi],view);valid=hit>1e-6
                            for name,volume in (('control_prior',old['transport'][hi]),('control_refiner',old['prediction'][hi]),
                                    ('candidate_prior',new['transport'][hi]),('candidate_refiner',new['prediction'][hi])):
                                ps,pd,_=projection(volume,view)
                                sem=float((ps-gs).abs().sum(1).mean())
                                depth=float((pd[valid]-gd[valid]).abs().mean()) if bool(valid.any()) else 0.
                                rows.append(dict(kind=kind,case=case['index'],horizon=hi+1,view=view,
                                    model=name,semantic_L1=sem,normalized_depth_L1=depth,total=sem+depth))
        summary={}
        for kind in ('occ','stc'):
            summary[kind]={}
            for view in ('top','oblique'):
                summary[kind][view]={}
                for name in ('control_prior','control_refiner','candidate_prior','candidate_refiner'):
                    selected=[r for r in rows if r['kind']==kind and r['view']==view and r['model']==name]
                    summary[kind][view][name]={k:float(np.mean([r[k] for r in selected]))
                        for k in ('semantic_L1','normalized_depth_L1','total')}
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(json.dumps(dict(scope='fixed8 HARD argmax rendering only; NOT soft training loss or full64 evidence',
            summary=summary,rows=rows),indent=2),encoding='utf-8')
        print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
