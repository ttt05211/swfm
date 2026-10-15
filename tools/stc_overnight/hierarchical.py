"""Fixed occupancy-first probability decoder, versus joint semantic argmax."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from tools.final_semantic_evaluation import metrics
from .bundle import Bundle


@torch.no_grad()
def decode(logits):
    semantic=logits[:17].argmax(0)
    occupied=torch.logsumexp(logits[:17],dim=0)>logits[17]
    return torch.where(occupied,semantic,17)


def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--bundle',default='../stc_overnight_bundle.zip');ap.add_argument('--device',default='cuda')
    a=ap.parse_args(argv);torch.set_num_threads(1);b=Bundle(a.bundle);root=Path(a.input);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    c=json.loads((root/'contract.json').read_text(encoding='utf-8'));counts={n:metrics.empty(3) for n in ['argmax','occupancy_first']}
    for i,key in enumerate(c['keys']):
        with np.load(root/f'logits_{i:03d}.npz',allow_pickle=False) as z:values=z['logits']
        preds={n:[] for n in counts}
        for h in range(3):
            x=torch.as_tensor(values[h],device=a.device)
            preds['argmax'].append(x.argmax(0).byte().cpu().numpy())
            preds['occupancy_first'].append(decode(x).byte().cpu().numpy())
        gt=b.targets(b.by_key[tuple(key)])
        for n in counts:
            for h in range(3):metrics.add(counts[n],h,preds[n][h],gt[h])
        if (i+1)%16==0:print('HIERARCHY',i+1,{n:metrics.report(v,(1.,2.,3.),moving=False)['average'] for n,v in counts.items()},flush=True)
    result=dict(status='complete',windows=len(c['keys']),source=c,scientific_updates=0,
        decoder='occupancy posterior sum(classes0..16)>P(free17), then conditional semantic argmax; FIXED 0.5, no tuning',
        scope='development diagnosis; full VAL confirmation required',
        metrics={n:metrics.report(v,(1.,2.,3.),moving=False) for n,v in counts.items()})
    (out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8');b.close()


if __name__=='__main__':main()
