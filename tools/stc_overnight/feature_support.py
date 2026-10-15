"""Learn source feature support from GT TRAIN, without synthetic or VAL pairs."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import numpy as np
import torch
from .bundle import Bundle


@torch.no_grad()
def fit(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--bundle',default='../stc_overnight_bundle.zip')
    ap.add_argument('--out',required=True);ap.add_argument('--model',default='clean1000');ap.add_argument('--device',default='cuda')
    a=ap.parse_args(argv);torch.set_num_threads(1);torch.manual_seed(21);b=Bundle(a.bundle)
    body,saved,policy=b.load_model(a.model,a.device);samples=[];keys=[]
    for r in b.records:
        if r['role']!='clean_train':continue
        _,raw,_=b.history(r,'gt','dense_valid')
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
            field=body.transport.history_encoder(torch.as_tensor(raw['history_occ'],device=a.device),
                                                 torch.as_tensor(raw['history_observed'],device=a.device))
        values=field.float().permute(0,2,3,1).reshape(4,-1,field.shape[1])
        indices=torch.randperm(values.shape[1],device=a.device)[:1024]
        samples.append(values[:,indices].reshape(-1,field.shape[1]).cpu())
        keys.append([r['scene_name'],r['t0_token']])
    source=torch.cat(samples);mean=source.mean(0);std=source.std(0).clamp_min(.1)
    low=torch.quantile(source,.01,dim=0);high=torch.quantile(source,.99,dim=0)
    x=((source-mean)/std).to(a.device);centers=x[torch.randperm(len(x),device=a.device)[:256]].clone()
    for step in range(30):
        distance=x.square().sum(1,keepdim=True)+centers.square().sum(1)[None]-2*x@centers.T
        ids=distance.argmin(1);sums=torch.zeros_like(centers).index_add_(0,ids,x)
        counts=torch.bincount(ids,minlength=len(centers)).float()
        centers=torch.where(counts[:,None]>0,sums/counts[:,None].clamp_min(1),centers)
        if (step+1)%10==0:print('SOURCE_SUPPORT_KMEANS',step+1,flush=True)
    # Radius must use FINAL centroids, not assignments from the penultimate E-step.
    distance_matrix=x.square().sum(1,keepdim=True)+centers.square().sum(1)[None]-2*x@centers.T
    ids=distance_matrix.argmin(1)
    distance=(x-centers[ids]).square().sum(1).cpu().numpy();labels=ids.cpu().numpy()
    radius=[]
    for k in range(len(centers)):
        values=distance[labels==k]
        radius.append(float(np.sqrt(np.percentile(values,95))) if len(values) else 0.)
    contract=dict(protocol='GT_TRAIN32_source_feature_support_v2',source_sha256=saved['source_sha256'],model=a.model,
        keys=keys,seed=21,samples=len(source),clusters=256,iterations=30,
        coordinate_quantiles=[.01,.99],cluster_radius_quantile=.95,
        VAL_samples_labels_or_statistics_used=False,synthetic_noise_used=False,
        future_poses_or_labels_used=False,scope='source-only feature-support projection; no decoder/body optimization')
    torch.save(dict(mean=mean,std=std,low=low,high=high,centers=centers.cpu(),radius=torch.tensor(radius),contract=contract),a.out)
    print('SOURCE_SUPPORT_DONE',len(source),flush=True);b.close()


@torch.no_grad()
def project(field,saved,mode):
    # Keep v1 explicitly reproducible for the recorded negative pilot; new fits
    # compare source/inference distances in FP32 regardless of outer AMP.
    context=nullcontext() if saved['contract']['protocol'].endswith('_v1') else torch.autocast(device_type=field.device.type,enabled=False)
    with context:return _project(field,saved,mode)


def _project(field,saved,mode):
    mean=saved['mean'].to(field.device);std=saved['std'].to(field.device)
    if mode=='coordinate_clip':
        lo=saved['low'].to(field.device)[None,:,None,None];hi=saved['high'].to(field.device)[None,:,None,None]
        return field.float().clamp(lo,hi).to(field.dtype)
    shape=field.shape;flat=field.float().permute(0,2,3,1).reshape(-1,shape[1]);x=(flat-mean)/std
    c=saved['centers'].to(field.device);distance=x.square().sum(1,keepdim=True)+c.square().sum(1)[None]-2*x@c.T
    ids=distance.argmin(1);nearest=c[ids]
    if mode=='manifold_clip':
        delta=x-nearest;norm=delta.square().sum(1).sqrt().clamp_min(1e-8)
        radius=saved['radius'].to(field.device)[ids];scale=(radius/norm).clamp(max=1)
        x=nearest+delta*scale[:,None]
    elif mode=='quantize':x=nearest
    else:raise ValueError('feature support projection mode')
    return (x*std+mean).reshape(shape[0],shape[2],shape[3],shape[1]).permute(0,3,1,2).to(field.dtype)


if __name__=='__main__':fit()
