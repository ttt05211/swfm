"""Relative posterior calibration using TRAIN/source and unlabeled STC inputs.

No future occupancy target is read here. Source and target populations are
fixed by the portable manifest, not selected by their validation scores.
"""
import numpy as np
import torch
from .replay import prepare_forecast,iter_forecasts


def binary_bias(source,target):
    source=np.asarray(source,np.float64);target=np.asarray(target,np.float64)
    if source.shape!=(18,) or target.shape!=(18,):raise ValueError('18 class marginals required')
    if not np.isfinite(source).all() or not np.isfinite(target).all():raise ValueError('nonfinite marginal')
    if (source<0).any() or (target<0).any() or min(source.sum(),target.sum())<=0:raise ValueError('invalid probability mass')
    source=source/source.sum();target=target/target.sum()
    # Same fixed numerical floor as the development distribution study.
    so=np.clip(1-source[17],1e-6,1-1e-6);to=np.clip(1-target[17],1e-6,1-1e-6)
    bias=np.full(18,np.log(so/to),np.float64);bias[17]=np.log((1-so)/(1-to))
    return bias


@torch.no_grad()
def fit(bundle,predictor,source_sha256,event=None):
    groups=[('source','clean_train','gt'),('target','reference64','stc')]
    sums={};keys={}
    for name,role,modality in groups:
        rows=[r for r in bundle.records if r['role']==role]
        if len(rows)!=(32 if name=='source' else 64):raise ValueError('fixed TRAIN32/reference64 required')
        total=np.zeros(18,np.float64);keys[name]=[]
        for i,r in enumerate(rows):
            if event is not None and event.is_set():raise InterruptedError('calibration interrupted before label access')
            rec,raw,times=bundle.history(r,modality,'dense_valid')
            row=predictor.prepare(rec,raw,times=times)
            context=prepare_forecast(predictor,row,r['future_poses'])
            for hs,fw in iter_forecasts(context):
                if hs[0] in (1,3,5):
                    total+=fw['logits'][0].softmax(0).double().sum(-1).cpu().numpy()
                del fw
            keys[name].append([r['scene_name'],r['t0_token']])
            if (i+1)%16==0:print('UNLABELED_POSTERIOR_FIT',name,i+1,'/',len(rows),flush=True)
        sums[name]=(total/total.sum()).tolist()
    bias=binary_bias(sums['source'],sums['target'])
    return dict(protocol='fixed_source_TRAIN32_unlabeled_reference64_binary_posterior_alignment_v1',
        source_sha256=source_sha256,keys=keys,marginals=sums,bias=bias.tolist(),
        target_GT_labels_used_for_fitting=False,optimizer_steps=0,
        scope='transductive target-input calibration; NOT target-domain-information-free; no coefficient search')
