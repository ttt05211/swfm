"""Fixed equal-weight prediction fusion, using only frozen causal forecasts."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from tools.final_semantic_evaluation import metrics
from .bundle import Bundle


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', default='../stc_overnight_bundle.zip')
    ap.add_argument('--inputs', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args(argv); torch.set_num_threads(1)
    dirs = [Path(v) for v in a.inputs]; b = Bundle(a.bundle)
    contracts = [json.loads((d/'contract.json').read_text(encoding='utf-8')) for d in dirs]
    keys = contracts[0]['keys']; same = ('keys', 'modality', 'known', 'bundle_manifest_sha256')
    for c in contracts:
        if any(c[k] != contracts[0][k] for k in same): raise ValueError('ensemble inputs are not paired')
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    contract = dict(inputs=contracts, rule='fixed equal weights; arithmetic probabilities AND arithmetic logits',
        scope='development subset only; no full VAL confirmation; no training')
    path = out/'contract.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != contract: raise ValueError('different output contract')
    path.write_text(json.dumps(contract, indent=2), encoding='utf-8')
    counts = {name: metrics.empty(3) for name in ('probability_mean', 'logit_mean', 'confidence_select')}
    tic = time.monotonic()
    for i, key in enumerate(keys):
        row = b.by_key[tuple(key)]
        # Files contain model predictions only. No labels used until both fused outputs exist.
        arrays = [np.load(d/f'logits_{i:03d}.npz', allow_pickle=False) for d in dirs]
        values = [z['logits'] for z in arrays]
        validities = [z['valid'] if 'valid' in z else None for z in arrays]
        for z in arrays: z.close()
        preds = {name: [] for name in counts}
        for h in range(3):
            prob, logits = None, None;best_confidence,best_class=None,None
            for v, mask in zip(values,validities):
                x = torch.from_numpy(v[h]).to(a.device)
                y = x.softmax(0)
                if mask is not None:
                    valid = torch.as_tensor(mask,device=a.device)
                    x = x*valid[None]; y = y*valid[None]
                confidence,cls=y.max(0)
                if best_confidence is None:best_confidence,best_class=confidence,cls
                else:
                    choose=confidence>best_confidence
                    best_class=torch.where(choose,cls,best_class);best_confidence=torch.maximum(best_confidence,confidence)
                if prob is None: prob = y; logits = x.clone()
                else: prob.add_(y); logits.add_(x)
            preds['probability_mean'].append(prob.argmax(0).byte().cpu().numpy())
            preds['logit_mean'].append(logits.argmax(0).byte().cpu().numpy())
            preds['confidence_select'].append(best_class.byte().cpu().numpy())
            del x, y, prob, logits
        targets = b.targets(row)
        local = {n: metrics.empty(3) for n in counts}
        for n in counts:
            for h in range(3): metrics.add(local[n], h, preds[n][h], targets[h])
            counts[n]['confusion'] += local[n]['confusion']
        (out/f'window_{i:03d}.json').write_text(json.dumps(dict(key=key,
            confusion={n: local[n]['confusion'].tolist() for n in local})), encoding='utf-8')
        np.savez(out/f'predictions_{i:03d}.npz', **{n: np.stack(preds[n]) for n in preds})
        if (i+1)%8 == 0: print('FUSION', i+1, {n: metrics.report(c, (1.,2.,3.), moving=False)['average'] for n,c in counts.items()}, flush=True)
        del values, targets, preds
    result = dict(status='complete', windows=len(keys), contract=contract, seconds=time.monotonic()-tic,
        metrics={n: metrics.report(c, (1.,2.,3.), moving=False) for n,c in counts.items()})
    (out/'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result['metrics'], indent=2), flush=True); b.close()


if __name__ == '__main__': main()
