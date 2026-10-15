"""Fixed 1-voxel XY morphology as a DIAGNOSTIC of local forecast holes/specks.

Not a claimed learned refiner or automatic deployment change. No kernel sweep.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from tools.final_semantic_evaluation import metrics
from .bundle import Bundle


@torch.no_grad()
def variants(logits):
    confidence,sem=logits[:17].max(0);margin=confidence-logits[17]
    value=margin[None,None]
    dilate=lambda v:F.max_pool3d(v,(3,3,1),stride=1,padding=(1,1,0))
    erode=lambda v:-dilate(-v)
    closing=erode(dilate(value))[0,0];opening=dilate(erode(value))[0,0]
    original=logits.argmax(0)
    fill=(original==17)&(closing>=0);remove=(original!=17)&(opening<0)
    closed=original.clone();closed[fill]=sem[fill]
    opened=original.clone();opened[remove]=17
    class_margin=(logits[:17]-logits[17:])[None]
    class_closed=erode(dilate(class_margin))[0]
    score,label=class_closed.max(0)
    per_class=original.clone();add=(original==17)&(score>=0);per_class[add]=label[add]
    return dict(original=original,closing_margin=closed,opening_margin=opened,closing_per_class=per_class)


def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--bundle',default='../stc_overnight_bundle.zip');ap.add_argument('--device',default='cuda')
    a=ap.parse_args(argv);torch.set_num_threads(1);b=Bundle(a.bundle);root=Path(a.input);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    c=json.loads((root/'contract.json').read_text(encoding='utf-8'));counts={n:metrics.empty(3) for n in ('original','closing_margin','opening_margin','closing_per_class')}
    edits={n:dict(corrected=0,damaged=0,changed=0) for n in ('closing_margin','opening_margin','closing_per_class')}
    for i,key in enumerate(c['keys']):
        with np.load(root/f'logits_{i:03d}.npz',allow_pickle=False) as z:values=z['logits']
        preds={n:[] for n in counts}
        for h in range(3):
            for n,v in variants(torch.as_tensor(values[h],device=a.device)).items():preds[n].append(v.byte().cpu().numpy())
        targets=b.targets(b.by_key[tuple(key)])
        for n in counts:
            for h in range(3):
                metrics.add(counts[n],h,preds[n][h],targets[h])
                if n in edits:
                    before,after=preds['original'][h],preds[n][h];gt=targets[h]
                    edits[n]['corrected']+=int(((before!=gt)&(after==gt)).sum())
                    edits[n]['damaged']+=int(((before==gt)&(after!=gt)).sum())
                    edits[n]['changed']+=int((before!=after).sum())
        if (i+1)%16==0:print('MORPH_DIAG',i+1,{n:metrics.report(v,(1.,2.,3.),moving=False)['average'] for n,v in counts.items()},flush=True)
    result=dict(status='complete',windows=len(c['keys']),source=c,edits=edits,scientific_updates=0,
        scope='DIAGNOSTIC fixed morphology, NOT learned mechanism or full VAL confirmation',
        metrics={n:metrics.report(v,(1.,2.,3.),moving=False) for n,v in counts.items()})
    (out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8');b.close()


if __name__=='__main__':main()
