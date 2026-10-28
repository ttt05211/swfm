"""TWO smooth spatial calibration coefficients, not class/range rule tables."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from tools.causal_forward_all.common import Epoch2Predictor
from tools.shared_history_experiments.refine.geometry import centers
from tools.stc_overnight.bundle import Bundle
from .portable import load_weights
from .geometry_screen import position_row
from .cache import AT,save_arrays,write,split_name
from .calibrate import confusion,scores

INTERCEPTS=(-2.,-1.5,-1.,-.5,0.)
SLOPES=(-2.,-1.,0.,1.,2.)


@torch.no_grad()
def radius(grid,current,poses):
    """Distance from last observed ego, using the SAME conditioned future poses."""
    query=centers(grid,poses.device);matrix=torch.linalg.inv(torch.as_tensor(current,device=poses.device).double())[None]@poses
    fields=[]
    for at in matrix:
        point=query@at[:3,:3].T+at[:3,3]
        fields.append((point[...,:2].square().sum(-1).sqrt()/40).clamp(0,1).float().flatten())
    return torch.stack(fields)


def predicted(maximum,free,winner,rho,a,b):
    bias=np.float32(a)+np.float32(b)*rho
    return np.where(maximum>=free+bias,winner,17).astype(np.uint8)


@torch.no_grad()
def build(a,out):
    body,c,info=load_weights(a.weights,a.device);sha=info['manifest']['source_checkpoint_sha256'];b=Bundle(a.bundle,cache_mib=16)
    p=Epoch2Predictor(body,info['prepare_config'],a.device,2);tic=time.monotonic();(out/'inputs').mkdir(exist_ok=True)
    contract=dict(protocol='latest_forward_CV_state_range_calibration_v1',source_sha256=sha,keys=c['eval_keys'],
        position_blend=.5,velocity_blend=.5,range_formula='rho=min(horizontal_range_in_t0_ego_m/40,1)',
        logit_formula='free_logit += intercept + slope*rho; same model/coefficients in GT and STC',
        fit_keys=c['eval_keys'][:32],held_keys=c['eval_keys'][32:],GT_future_ego=True,known='dense_valid',
        full_grid=True,metric_masks=False,Moving_available=False,source_body_weights_unchanged=True)
    if (out/'contract.json').exists() and json.loads((out/'contract.json').read_text())!=contract:raise ValueError('new range cache output required')
    write(out/'contract.json',contract)
    try:
        for i in range(64):
            for domain in ('stc','gt'):
                name=f'{domain}_{i:03d}';path=out/'inputs'/(name+'.npz')
                if path.exists():continue
                r=b.by_key[tuple(c['eval_keys'][i])];rec,raw,times=b.history(r,domain,'dense_valid');row=p.prepare(rec,raw,times=times)
                fw=p.forward(position_row(row,.5,.5),poses=np.asarray(r['future_poses']))
                maximum,winner=fw['logits'][:,:17].max(1)
                rho=radius(p.provider.pcfg.grid,raw['history_poses'][-1],fw['poses'])
                # These are exact sufficient statistics for ANY free-only field.
                # All six model forecasts precede future occupancy access.
                labels=b.targets(r,tuple(range(6))).reshape(6,-1)
                save_arrays(path,maximum=maximum.cpu().numpy(),free=fw['logits'][:,17].cpu().numpy(),winner=winner.byte().cpu().numpy(),
                    rho=rho.cpu().numpy(),labels=labels,source_sha256=np.asarray(sha),key=np.asarray(c['eval_keys'][i]))
                del fw,maximum,winner,rho,row,raw,labels
            if (i+1)%8==0:print('RANGE_CACHE',i+1,'/64 paired','seconds',round(time.monotonic()-tic,1),flush=True)
    finally:p.close();b.close()
    return contract


def load(path,sha):
    z=np.load(path,allow_pickle=False)
    if str(z['source_sha256'])!=sha:z.close();raise ValueError('wrong current range source')
    return z


def fit(out,cache,fields=None,extended=False):
    fields=out if fields is None else fields
    intercepts=(-3.,-2.5,-2.,-1.5,-1.) if extended else INTERCEPTS
    slopes=(-1.,0.,1.,2.,3.) if extended else SLOPES
    c=json.loads((fields/'contract.json').read_text());sha=c['source_sha256'];counts={d:{f'{a}:{b}':np.zeros((3,18,18),np.int64) for a in intercepts for b in slopes} for d in ('stc','gt')};original=np.zeros((3,18,18),np.int64);tic=time.monotonic()
    for i in range(32):
        reference=json.loads((cache/'counts'/f'gt_{i:03d}.json').read_text());original+=np.asarray(reference['confusion']['semantic_refine'],np.int64)
        for domain in counts:
            with load(fields/'inputs'/f'{domain}_{i:03d}.npz',sha) as z:
                maximum,free,winner,rho,labels=[z[k][list(AT)] for k in ('maximum','free','winner','rho','labels')]
                for a in intercepts:
                    for b in slopes:
                        pred=predicted(maximum,free,winner,rho,a,b)
                        for h in range(3):counts[domain][f'{a}:{b}'][h]+=confusion(pred[h],labels[h])
        if (i+1)%8==0:print('RANGE_FIT',i+1,'/32','seconds',round(time.monotonic()-tic,1),flush=True)
    reports={d:{n:scores(q)['average'] for n,q in choices.items()} for d,choices in counts.items()};base=scores(original)['average']
    utility=lambda n:reports['stc'][n]['mIoU']+.5*reports['stc'][n]['IoU']
    best=max(reports['stc'],key=utility)
    safe=[n for n,r in reports['gt'].items() if r['mIoU']>=base['mIoU']-2 and r['IoU']>=base['IoU']-3]
    moderate=max(safe,key=utility) if safe else '0.0:0.0'
    fitted=dict(contract=c,VAL_future_GT_labels_used_for_fitting=True,parameters=2,intercepts=list(intercepts),slopes=list(slopes),
        fit_scope='first32 STC AND same first32 GT inputs; no held32 fitting',grid=reports,
        reference_GT_fit32=base,selection='STC fit32 mIoU+.5IoU; moderate: additionally GT mIoU loss<=2 and IoU loss<=3',
        methods=dict(state_raw=[0.,0.],state_global_free=[-1.5,0.],range_max=list(map(float,best.split(':'))),range_moderate=list(map(float,moderate.split(':')))))
    write(out/'fit.json',fitted);print('RANGE_COEFFICIENTS',fitted['methods'],flush=True);return fitted


def evaluate(out,cache,fitted,fields=None):
    fields=out if fields is None else fields
    totals={};sha=fitted['contract']['source_sha256']
    for domain in ('stc','gt'):
        for i in range(64):
            with load(fields/'inputs'/f'{domain}_{i:03d}.npz',sha) as z:
                maximum,free,winner,rho,labels=[z[k] for k in ('maximum','free','winner','rho','labels')]
                preds={n:predicted(maximum,free,winner,rho,*ab) for n,ab in fitted['methods'].items()}
                local={n:np.stack([confusion(pred[h],labels[h]) for h in AT]) for n,pred in preds.items()}
            ref=json.loads((cache/'counts'/f'{domain}_{i:03d}.json').read_text());local['original']=np.asarray(ref['confusion']['semantic_refine'],np.int64)
            for part in (split_name(i),'held32' if i>=32 else 'fit_copy','all64'):
                if part=='fit_copy':continue
                key=domain+'/'+part;t=totals.setdefault(key,dict(windows=0,counts={n:np.zeros((3,18,18),np.int64) for n in local}));t['windows']+=1
                for n,q in local.items():t['counts'][n]+=q
    reports={k:dict(windows=v['windows'],reports={n:scores(q) for n,q in v['counts'].items()}) for k,v in totals.items()}
    write(out/'results.json',dict(status='complete',fit=fitted,reports=reports))
    for key in ('stc/held32','stc/all64','gt/all64'):print('RANGE_FINAL',key,{n:v['average'] for n,v in reports[key]['reports'].items()},flush=True)


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('phase',choices=('cache','fit','evaluate'))
    ap.add_argument('--weights',default='../_received/forward_adaln_epoch2_local.zip');ap.add_argument('--bundle',default='../stc_overnight_bundle.zip')
    ap.add_argument('--cache',required=True);ap.add_argument('--out',required=True);ap.add_argument('--device',default='cuda')
    ap.add_argument('--fields');ap.add_argument('--extended',action='store_true');a=ap.parse_args(argv)
    torch.set_num_threads(1);torch.manual_seed(21);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if a.phase=='cache':return build(a,out)
    fields=Path(a.fields) if a.fields else out
    fitted=json.loads((out/'fit.json').read_text()) if (out/'fit.json').exists() else fit(out,Path(a.cache),fields,a.extended)
    if a.phase=='evaluate':return evaluate(out,Path(a.cache),fitted,fields)


if __name__=='__main__':main()
