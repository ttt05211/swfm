"""One shared weight for negative (predicted-free) static memory evidence."""
import numpy as np
import torch
from tools.fpn_unet_stc_age.processing import weight_history
from tools.source_scene_ego_conditioned.geometry import aligned_memory

BASELINE=(1.,0.)
CANDIDATES=(BASELINE,)+tuple((w,b) for w in (1.,.5,0.) for b in (-.25,0.,.25) if (w,b)!=BASELINE)


def name(c):return f'free_evidence{c[0]:g}_bias{c[1]:+g}'


def change(history,raw,free_weight):
    labels=np.asarray(raw['history_occ']);known=np.asarray(raw['history_observed'],bool)
    n,sx,sy,sz=labels.shape
    if n!=4 or sx%2 or sy%2 or not 0<=free_weight<=1:raise ValueError('four even-XY grids and weight0..1 required')
    free=(known&(labels==17)).reshape(4,sx//2,2,sy//2,2,sz).mean((2,4)).astype(np.float32)
    causal=dict(history[0]);data=dict(causal['memory_reads'])
    # Tracked sources never have FREE semantics. Thus subtracting FREE support
    # from the exact original nonsource coverage retains occupied support.
    adjusted=np.asarray(data['static_valid'])-(1-free_weight)*free
    if np.any(adjusted < -1e-6):raise ValueError('static coverage inconsistent with historical FREE support')
    data['static_valid']=np.maximum(adjusted,0.);causal['memory_reads']=data
    return (causal,*history[1:])


@torch.no_grad()
def predict(body,provider,row,fw):
    result={}
    def decode(w,z):
        v=z[[1,3,5]];score,classes=v[:,:17].max(1)
        for bias in (-.25,0.,.25):
            pred=torch.where(score>=v[:,17]+bias,classes,17)
            result[name((w,bias))]=pred.reshape(3,*fw['shape']).byte().cpu().numpy()
    decode(1.,fw['logits']);del fw['logits'],fw['memory']
    _,raw,_,history=row
    for w in (.5,0.):
        h=change(history,raw,w);h,_=weight_history(h,raw['_shared_source_track'].times)
        memory=aligned_memory(body.columns,fw['lifted'],fw['output'],h,raw,fw['targets'],fw['yaws'],fw['poses'],provider.pcfg.grid)
        z=body.columns(memory,fw['prior'],range(6),fw['shape']);decode(w,z);del memory,z
    return result
