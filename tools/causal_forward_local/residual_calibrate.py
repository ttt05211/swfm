"""TWO global readout scalars: residual strength + free bias, fit32 ONLY."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from .cache import AT,write,split_name
from .calibrate import load,confusion,scores

GAINS=(.8,1.,1.2)
BIASES=(-2.,-1.,0.,1.,2.)


def prior_logits(p,epsilon=1e-4):return np.log((p+epsilon)/(p.sum(0,keepdims=True)+18*epsilon))


def predict(x,p,gain,bias):
    z=gain*x+(1-gain)*prior_logits(p)
    winner=z[:17].argmax(0);maximum=np.take_along_axis(z[:17],winner[None],axis=0)[0]
    return np.where(maximum>=z[17]+bias,winner,17).astype(np.uint8)


def fit(cache,out):
    c=json.loads((cache/'contract.json').read_text());sha=c['source_checkpoint_sha256'];counts={f'{g}:{b}':np.zeros((3,18,18),np.int64) for g in GAINS for b in BIASES};tic=time.monotonic()
    for i in range(32):
        with load(cache/'inputs'/f'stc_{i:03d}.npz',sha) as z:
            if str(z['split'])!='fit32':raise ValueError('first32 fit only')
            x=z['logits'];p=z['prior'];labels=z['labels'][list(AT)].reshape(3,-1)
            for h,at in enumerate(AT):
                prior=prior_logits(p[at])
                for g in GAINS:
                    q=g*x[h]+(1-g)*prior;winner=q[:17].argmax(0);maximum=np.take_along_axis(q[:17],winner[None],0)[0]
                    for b in BIASES:counts[f'{g}:{b}'][h]+=confusion(np.where(maximum>=q[17]+b,winner,17),labels[h])
        if (i+1)%8==0:print('RESIDUAL_FIT',i+1,'/32','seconds',round(time.monotonic()-tic,1),flush=True)
    reports={k:scores(v) for k,v in counts.items()};selected=max(reports,key=lambda k:reports[k]['average']['mIoU']+.5*reports[k]['average']['IoU'])
    g,b=map(float,selected.split(':'))
    fitted=dict(protocol='latest_forward_fit32_two_scalar_residual_free_calibration_v1',source_sha256=sha,
        labelled_target_fit_keys=c['reference64_keys'][:32],held_keys=c['reference64_keys'][32:],
        formula='gain * original_logits + (1-gain) * original_log_prior + free_bias; globally same at ALL SIX horizons',
        target_future_GT_used_for_fitting=True,methods=dict(raw=dict(gain=1.,bias=0.),residual_free=dict(gain=g,bias=b)),
        selected=selected,fit_grid=reports,selection='fit32 mIoU+.5IoU only; no held label selection; no masks')
    write(out/'fit.json',fitted);print('RESIDUAL_FINAL',selected,reports[selected]['average'],flush=True);return fitted


def evaluate(cache,out,fit):
    (out/'counts').mkdir(exist_ok=True);totals={};sha=fit['source_sha256'];tic=time.monotonic()
    for domain in ('stc','gt'):
        for i in range(64):
            name=f'{domain}_{i:03d}';path=out/'counts'/(name+'.json')
            if path.exists():item=json.loads(path.read_text())
            else:
                with load(cache/'inputs'/(name+'.npz'),sha) as z:
                    x=z['logits'];prior=z['prior'];pred={n:[] for n in fit['methods']}
                    for h,at in enumerate(AT):
                        for n,m in fit['methods'].items():pred[n].append(predict(x[h],prior[at],m['gain'],m['bias']))
                    labels=z['labels'][list(AT)].reshape(3,-1)
                    item=dict(key=z['key'].tolist(),confusion={n:[confusion(p[h],labels[h]).tolist() for h in range(3)] for n,p in pred.items()})
                write(path,item)
            for part in (split_name(i),'held32' if i>=32 else 'fit_copy','all64'):
                if part=='fit_copy':continue
                key=domain+'/'+part;t=totals.setdefault(key,dict(windows=0,counts={n:np.zeros((3,18,18),np.int64) for n in fit['methods']}));t['windows']+=1
                for n,q in item['confusion'].items():t['counts'][n]+=np.asarray(q,np.int64)
            if (i+1)%8==0:
                report={k:dict(windows=v['windows'],reports={n:scores(q) for n,q in v['counts'].items()}) for k,v in totals.items()}
                write(out/'results.json',dict(status='running',fit=fit,reports=report));print('RESIDUAL_EVAL',name,'seconds',round(time.monotonic()-tic,1),flush=True)
    report={k:dict(windows=v['windows'],reports={n:scores(q) for n,q in v['counts'].items()}) for k,v in totals.items()}
    write(out/'results.json',dict(status='complete',fit=fit,reports=report))


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--cache',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--fit-only',action='store_true');a=ap.parse_args(argv);torch.set_num_threads(1)
    cache=Path(a.cache);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    fitted=json.loads((out/'fit.json').read_text()) if (out/'fit.json').exists() else fit(cache,out)
    if not a.fit_only:evaluate(cache,out,fitted)


if __name__=='__main__':main()
