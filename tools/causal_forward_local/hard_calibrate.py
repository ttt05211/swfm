"""Metric-based fit of ONE global18-vector; only first32 target scenes."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .cache import AT,write
from .calibrate import load,evaluate


def measure(pred,labels):
    c=torch.stack([torch.bincount((18*labels[h]+pred[h]).flatten(),minlength=324).reshape(18,18) for h in range(3)]).double()
    tp=c.diagonal(dim1=1,dim2=2);union=c.sum(1)+c.sum(2)-tp
    sem=torch.where(union[:,:17]>0,100*tp[:,:17]/union[:,:17],torch.nan).nanmean(1).mean()
    occ=(100*c[:,:17,:17].sum((1,2))/(c.sum((1,2))-c[:,17,17])).mean()
    return sem,occ


@torch.no_grad()
def fit(cache,out,device):
    c=json.loads((cache/'contract.json').read_text());sha=c['source_checkpoint_sha256']
    scalar=json.loads((cache/'calibration/fit.json').read_text())['chosen_free_bias']
    xs=[];ys=[]
    for i in range(32):
        with load(cache/'inputs'/f'stc_{i:03d}.npz',sha) as z:
            if str(z['split'])!='fit32':raise ValueError('fit scenes only')
            x=z['logits'];y=z['labels'][list(AT)].reshape(3,-1);rng=np.random.default_rng(21000+i)
            ids=rng.choice(x.shape[-1],32768,replace=False)
            xs.append(x[:,:,ids].transpose(0,2,1).copy());ys.append(y[:,ids].copy())
    x=torch.from_numpy(np.concatenate(xs,axis=1)).to(device);labels=torch.from_numpy(np.concatenate(ys,axis=1)).to(device).long()
    del xs,ys
    bias=x.new_zeros(18);bias[17]=scalar;trace=[]
    for sweep in range(3):
        for cls in range(17):
            other=x+bias;other[:,:,cls]=-torch.inf;maximum,winner=other.max(-1);current=float(bias[cls]);best=None
            for delta in (0.,-.5,.5,-1.,1.,-2.,2.):
                value=float(np.clip(current+delta,-3,3));score=x[:,:,cls]+value
                pred=torch.where((score>maximum)|((score==maximum)&(cls<winner)),cls,winner)
                miou,iou=measure(pred,labels);penalty=.01*(bias.square().sum()-bias[cls]**2+value**2)/18
                objective=float(miou+.5*iou-penalty)
                if best is None or objective>best[0]+1e-8:best=(objective,value,float(miou),float(iou))
            bias[cls]=best[1]
        m,i=measure((x+bias).argmax(-1),labels)
        trace.append(dict(pass_index=sweep+1,mIoU=float(m),IoU=float(i),bias=bias.cpu().tolist()))
        print('METRIC_BIAS',trace[-1],flush=True)
    free=np.zeros(18);free[17]=scalar
    fitted=dict(protocol='latest_forward_epoch2_first32_metric_global18_v1',source_sha256=sha,
        labelled_target_fit_keys=c['reference64_keys'][:32],held_keys=c['reference64_keys'][32:],
        VAL_future_labels_used=True,parameters=17,scalar_free_bias_fixed=scalar,
        objective='sampled uniform full-grid fit32 mIoU + .5 IoU - .01 mean(bias^2)',
        budget='fixed3 coordinate passes; global class offsets only; no spatial/horizon/class-specific masks',
        sample='32768 uniform voxels/horizon/window, deterministic; score ALL grid during evaluation',
        trace=trace,methods=dict(raw=dict(bias=[0.]*18,scale=[1.]*18),
            free_fit=dict(bias=free.tolist(),scale=[1.]*18),metric_bias18=dict(bias=bias.cpu().tolist(),scale=[1.]*18)))
    write(out/'fit.json',fitted);return fitted


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--cache',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--device',default='cuda');ap.add_argument('--fit-only',action='store_true');ap.add_argument('--eval-only',action='store_true')
    a=ap.parse_args(argv);torch.set_num_threads(1);cache=Path(a.cache);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if a.eval_only and not (out/'fit.json').is_file():raise ValueError('--eval-only requires finalized fit.json')
    fitted=json.loads((out/'fit.json').read_text()) if (out/'fit.json').exists() else fit(cache,out,a.device)
    if not a.fit_only:evaluate(cache,out,fitted)


if __name__=='__main__':main()
