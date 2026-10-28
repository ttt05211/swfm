"""Target-label-free global posterior matching: source TRAIN32 vs STC history32."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from .cache import write
from .calibrate import load,evaluate


def logit(p):return np.log(np.clip(p,1e-8,1-1e-8)/(1-np.clip(p,1e-8,1-1e-8)))


def fit(cache,out):
    c=json.loads((cache/'contract.json').read_text());sha=c['source_checkpoint_sha256'];rates={};tic=time.monotonic()
    for domain in ('source','stc'):
        posterior=np.zeros(18,np.float64);decision=0.
        for i in range(32):
            with load(cache/'inputs'/f'{domain}_{i:03d}.npz',sha) as z:
                if str(z['split'])!=('fit32' if domain=='stc' else 'source_TRAIN32'):raise ValueError('wrong posterior fit scenes')
                x=z['logits'];p=np.exp(x-x.max(1,keepdims=True));p/=p.sum(1,keepdims=True)
                posterior+=p.mean((0,2),dtype=np.float64);margin=x[:,:17].max(1)-x[:,17]
                decision+=(1/(1+np.exp(-np.clip(margin,-60,60)))).mean(dtype=np.float64)
            if (i+1)%8==0:print('POSTERIOR_FIT',domain,i+1,'/32','seconds',round(time.monotonic()-tic,1),flush=True)
        rates[domain]=dict(posterior=(posterior/32).tolist(),decision=decision/32)
    src=np.asarray(rates['source']['posterior']);tgt=np.asarray(rates['stc']['posterior']);multi=np.log(src+1e-6)-np.log(tgt+1e-6);multi-=multi.mean()
    binary=np.zeros(18);binary[17]=-(logit(1-src[17])-logit(1-tgt[17]))
    dec=np.zeros(18);dec[17]=-(logit(rates['source']['decision'])-logit(rates['stc']['decision']))
    fitted=dict(protocol='latest_forward_input_only_posterior_matching_v1',source_sha256=sha,
        labelled_target_fit_keys=[],target_input_fit_keys=c['reference64_keys'][:32],held_keys=c['reference64_keys'][32:],
        source_TRAIN_keys=c['source_keys'],VAL_future_occupancy_labels_used=False,source_future_occupancy_labels_used=False,
        GT_future_ego_conditioned=True,rates=rates,
        assumption='domain posterior matching; source/target scene composition can differ; not guaranteed label-shift correction',
        methods=dict(raw=dict(bias=[0.]*18,scale=[1.]*18),posterior18=dict(bias=multi.tolist(),scale=[1.]*18),
            posterior_binary=dict(bias=binary.tolist(),scale=[1.]*18),posterior_decision=dict(bias=dec.tolist(),scale=[1.]*18)))
    write(out/'fit.json',fitted);print('POSTERIOR_BIASES',fitted['methods'],flush=True);return fitted


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--cache',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--fit-only',action='store_true');a=ap.parse_args(argv);torch.set_num_threads(1)
    cache=Path(a.cache);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    fitted=json.loads((out/'fit.json').read_text()) if (out/'fit.json').is_file() else fit(cache,out)
    if not a.fit_only:evaluate(cache,out,fitted)


if __name__=='__main__':main()
