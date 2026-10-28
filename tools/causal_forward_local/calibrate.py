"""Small global readout fits on FIRST32 STC scenes, never the held32 labels."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from tools.final_semantic_evaluation import metrics
from .cache import AT,write,split_name

FREE_GRID=(-1.5,-1.,-.5,0.,.5,1.,1.5)


def load(path,sha):
    z=np.load(path,allow_pickle=False)
    if str(z['source_sha256'])!=sha:z.close();raise ValueError('only latest-source cache allowed')
    return z


def scores(counts):return metrics.report(dict(confusion=counts,moving=np.zeros_like(counts)),(1.,2.,3.),moving=False)


def confusion(pred,gt):
    return np.bincount(18*gt.astype(np.int64).ravel()+pred.ravel(),minlength=324).reshape(18,18)


def fit(cache,out):
    c=json.loads((cache/'contract.json').read_text(encoding='utf-8'));sha=c['source_checkpoint_sha256']
    fits={str(v):np.zeros((3,18,18),np.int64) for v in FREE_GRID};xs=[];ys=[]
    tic=time.monotonic()
    for i in range(32):
        with load(cache/'inputs'/f'stc_{i:03d}.npz',sha) as z:
            if str(z['split'])!='fit32':raise ValueError('fit must NEVER use held32 scenes')
            x=z['logits'];y=z['labels'][list(AT)].reshape(3,-1);winner=x[:,:17].argmax(1)
            margin=np.take_along_axis(x[:,:17],winner[:,None],1)[:,0]-x[:,17]
            for v in FREE_GRID:
                pred=np.where(margin>=v,winner,17)
                for h in range(3):fits[str(v)][h]+=confusion(pred[h],y[h])
            rng=np.random.default_rng(21000+i)
            for h in range(3):
                ids=rng.choice(x.shape[-1],8192,replace=False)
                xs.append(x[h,:,ids].copy());ys.append(y[h,ids].copy())
        if (i+1)%8==0:print('CAL_FIT_CACHE',i+1,'/32','seconds',round(time.monotonic()-tic,1),flush=True)
    reports={v:scores(k) for v,k in fits.items()}
    # This scalar choice depends ONLY on first32 fit labels. All tested values,
    # including losses, are recorded; held32 are not used to choose it.
    chosen=max(FREE_GRID,key=lambda v:reports[str(v)]['average']['mIoU']+.5*reports[str(v)]['average']['IoU'])
    bias=np.zeros(18);bias[17]=chosen
    methods=dict(raw=dict(bias=[0.]*18,scale=[1.]*18),free_fit=dict(bias=bias.tolist(),scale=[1.]*18))
    x=torch.from_numpy(np.concatenate(xs)).float();y=torch.from_numpy(np.concatenate(ys)).long()
    del xs,ys
    traces={}
    for arm in ('bias18_CE','bias18_IoU','vector36_IoU'):
        torch.manual_seed(21);b=torch.zeros(18,requires_grad=True);s=torch.zeros(18,requires_grad=arm.startswith('vector'))
        params=[b,s] if s.requires_grad else [b];opt=torch.optim.Adam(params,lr=.05)
        trace=[]
        for step in range(240):
            ids=torch.randint(len(y),(32768,));labels=y[ids]
            scale=s.clamp(-.7,.7).exp();q=x[ids]*scale+(b-b.mean());ce=F.cross_entropy(q,labels)
            if arm.endswith('_CE'):loss=ce
            else:
                prob=q.softmax(1);truth=F.one_hot(labels,18).float();count=truth.sum(0)
                inter=(prob*truth).sum(0);union=prob.sum(0)+count-inter
                miou=(inter[:17]/union[:17].clamp_min(1e-8))[count[:17]>0].mean()
                occ=1-prob[:,17];is_occ=(labels!=17).float();occ_i=(occ*is_occ).sum()
                iou=occ_i/(occ.sum()+is_occ.sum()-occ_i).clamp_min(1e-8)
                loss=.25*ce+1-miou+1-iou
            loss=loss+.005*(b*b).mean()+.02*(s*s).mean()
            for g in opt.param_groups:g['lr']=.05*(.05+.95*.5*(1+np.cos(np.pi*step/240)))
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
            if (step+1)%60==0:trace.append(dict(step=step+1,loss=float(loss.detach()),CE=float(ce.detach())))
        methods[arm]=dict(bias=(b-b.mean()).detach().tolist(),scale=s.clamp(-.7,.7).exp().detach().tolist())
        traces[arm]=trace;print('CAL_FIT_DONE',arm,trace[-1],flush=True)
    fitted=dict(protocol='current_forward_epoch2_first32_labelled_global_calibration_v1',source_sha256=sha,
        labelled_target_fit_keys=c['reference64_keys'][:32],held_keys=c['reference64_keys'][32:],
        TRAIN_future_labels_used=False,VAL_future_labels_used=True,raw_GT_model_unchanged=True,
        selection='fit32 only; no held32 label selection; fixed final240 Adam steps',
        free_grid=reports,chosen_free_bias=chosen,methods=methods,traces=traces)
    write(out/'fit.json',fitted);return fitted


def apply_numpy(x,method):
    return (x*np.asarray(method['scale'],np.float32)[None,:,None]+
        np.asarray(method['bias'],np.float32)[None,:,None]).argmax(1).astype(np.uint8)


def evaluate(cache,out,fitted,domains=('stc','gt')):
    sha=fitted['source_sha256'];names=list(fitted['methods']);totals={};done=[];tic=time.monotonic()
    (out/'counts').mkdir(exist_ok=True)
    for domain in domains:
        for i in range(64):
            name=f'{domain}_{i:03d}';path=out/'counts'/(name+'.json')
            if path.exists():v=json.loads(path.read_text(encoding='utf-8'))
            else:
                with load(cache/'inputs'/(name+'.npz'),sha) as z:
                    x=z['logits'];gt=z['labels'][list(AT)].reshape(3,-1);count={}
                    for n,m in fitted['methods'].items():
                        pred=apply_numpy(x,m);count[n]=[confusion(pred[h],gt[h]).tolist() for h in range(3)]
                    v=dict(key=z['key'].tolist(),confusion=count)
                write(path,v)
            for part in (split_name(i),'held32' if i>=32 else 'fit32_copy','all64'):
                if part=='fit32_copy':continue
                key=domain+'/'+part;entry=totals.setdefault(key,dict(windows=0,counts={n:np.zeros((3,18,18),np.int64) for n in names}))
                entry['windows']+=1
                for n,counts in v['confusion'].items():entry['counts'][n]+=np.asarray(counts,np.int64)
            done.append(name)
            if (i+1)%8==0:
                report={k:dict(windows=v['windows'],reports={n:scores(q) for n,q in v['counts'].items()}) for k,v in totals.items()}
                write(out/'results.json',dict(status='running',completed=done,reports=report,fit=fitted))
                print('CAL_EVAL',name,'seconds',round(time.monotonic()-tic,1),flush=True)
    report={k:dict(windows=v['windows'],reports={n:scores(q) for n,q in v['counts'].items()}) for k,v in totals.items()}
    write(out/'results.json',dict(status='complete',completed=done,reports=report,fit=fitted))
    for k,v in report.items():
        if k.endswith(('all64','held32')):print(k,{n:q['average'] for n,q in v['reports'].items()},flush=True)


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--cache',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--fit-only',action='store_true');ap.add_argument('--domains',nargs='+',choices=('stc','gt'),default=['stc','gt'])
    a=ap.parse_args(argv);torch.set_num_threads(1)
    cache=Path(a.cache);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    fitted=json.loads((out/'fit.json').read_text(encoding='utf-8')) if (out/'fit.json').exists() else fit(cache,out)
    if not a.fit_only:evaluate(cache,out,fitted,a.domains)


if __name__=='__main__':main()
