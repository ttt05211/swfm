"""Declared frozen local probability regularization; no future labels in operator."""
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
    p = logits.softmax(0)
    mean = F.avg_pool3d(p[None], (3,3,1), stride=1, padding=(1,1,0), count_include_pad=False)[0]
    return dict(original=logits.argmax(0), xy_probability_mean=mean.argmax(0),
                xy_centered_mean=(p+mean).argmax(0), xy_potts_step=(logits+mean).argmax(0))


def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--bundle',default='../stc_overnight_bundle.zip');ap.add_argument('--device',default='cuda')
    a=ap.parse_args(argv);torch.set_num_threads(1);root=Path(a.input);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    c=json.loads((root/'contract.json').read_text(encoding='utf-8'));b=Bundle(a.bundle)
    counts={n:metrics.empty(3) for n in ('original','xy_probability_mean','xy_centered_mean','xy_potts_step')}
    for i,key in enumerate(c['keys']):
        with np.load(root/f'logits_{i:03d}.npz',allow_pickle=False) as z: logits=z['logits']
        preds={n:[] for n in counts}
        for h in range(3):
            for n,v in variants(torch.as_tensor(logits[h],device=a.device)).items(): preds[n].append(v.byte().cpu().numpy())
        gt=b.targets(b.by_key[tuple(key)])
        for n in counts:
            for h in range(3):metrics.add(counts[n],h,preds[n][h],gt[h])
        if (i+1)%16==0:print('SPATIAL',i+1,{n:metrics.report(v,(1.,2.,3.),moving=False)['average'] for n,v in counts.items()},flush=True)
    result=dict(status='complete',windows=len(c['keys']),source=c,scientific_updates=0,
        scope='fixed development diagnosis; no coefficient sweep; no full VAL confirmation',
        metrics={n:metrics.report(v,(1.,2.,3.),moving=False) for n,v in counts.items()})
    (out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8');b.close()


if __name__=='__main__':main()
