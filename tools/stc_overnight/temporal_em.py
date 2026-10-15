"""Causal binary latent-observer fusion; an isolated, assumption-sensitive pilot.

Inspired by Dawid/Skene observer-error EM (1979), doi:10.2307/2346806.
Temporal STC errors are correlated, so conditional independence is NOT assured.
No GT labels, source noise statistics, future poses or validation scores fit EM.
"""
import numpy as np
import torch
from torch.nn import functional as F
from real_motion.metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS
from tools.shared_history_experiments.refine.geometry import centers,inverse_grid


def binary_observer_posterior(counts,steps=50):
    """81 ternary patterns: 0=missing, 1=free, 2=occupied per observer."""
    counts=np.asarray(counts,np.float64).copy()
    if counts.shape!=(81,) or (counts<0).any() or not np.isfinite(counts).all():raise ValueError('81 finite pattern counts required')
    counts[0]=0 # all four missing is not evidence for either state
    digits=(np.arange(81)[:,None]//(3**np.arange(4)[None]))%3
    known=(digits>0).astype(float);positive=(digits==2).astype(float);negative=known-positive
    if counts.sum()==0:return np.zeros(81),dict(empty=True)
    pi=np.clip((counts[:,None]*positive).sum()/max((counts[:,None]*known).sum(),1),1e-4,1-1e-4)
    sensitivity=np.full(4,.75);false_positive=np.full(4,.01)
    def expectation():
        a=np.log(pi)+(positive*np.log(sensitivity)+negative*np.log1p(-sensitivity)).sum(1)
        b=np.log1p(-pi)+(positive*np.log(false_positive)+negative*np.log1p(-false_positive)).sum(1)
        return np.exp(-np.logaddexp(0,b-a))
    for _ in range(steps):
        p=expectation();occupied=counts*p;free=counts*(1-p)
        sensitivity=(occupied@positive+.5)/(occupied@known+1)
        false_positive=(free@positive+.5)/(free@known+1)
        pi=float((occupied.sum()+.5)/(counts.sum()+1))
        sensitivity=np.clip(sensitivity,1e-6,1-1e-6);false_positive=np.clip(false_positive,1e-6,1-1e-6)
        pi=np.clip(pi,1e-6,1-1e-6)
    posterior=expectation();posterior[0]=0
    return posterior,dict(empty=False,iterations=steps,occupied_prior=float(pi),
        sensitivity=sensitivity.tolist(),false_positive=false_positive.tolist(),
        conditional_independence_is_unverified=True)


@torch.no_grad()
def apply(context):
    raw=context['raw'];renderer=context['renderer'];device=context['poses'].device
    sem=torch.as_tensor(raw['history_occ'],device=device).long()
    valid=torch.as_tensor(raw['history_observed'],device=device).bool() & (sem<18)
    for c in DYNAMIC_CLASS_IDS:valid &= sem!=c
    history=torch.as_tensor(np.asarray(raw['history_poses']),device=device,dtype=torch.float64)
    query=centers(renderer.grid,device);code=torch.zeros(renderer.shape,device=device,dtype=torch.long)
    votes=torch.zeros((18,*renderer.shape),device=device)
    for f in range(4):
        sampling=inverse_grid(query,torch.linalg.inv(history[f])@renderer.pose,renderer.grid)[None]
        cls=F.grid_sample(sem[f].float()[None,None],sampling,mode='nearest',align_corners=False)[0,0].long().clamp(0,17)
        seen=F.grid_sample(valid[f].float()[None,None],sampling,mode='nearest',align_corners=False)[0,0]>0
        digit=torch.where(seen,torch.where(cls==17,1,2),0)
        code+=digit*(3**f);votes.scatter_add_(0,cls[None],seen.float()[None])
    counts=torch.bincount(code.flatten(),minlength=81).cpu().numpy()
    posterior,audit=binary_observer_posterior(counts)
    occupied=torch.as_tensor(posterior,device=device,dtype=torch.float32)[code]
    dist=votes[:17]/votes[:17].sum(0,keepdim=True).clamp_min(1)
    # No occupied semantic evidence means no class can be invented.
    occupied=occupied*(votes[:17].sum(0)>0)
    current=renderer.static[0];is_occupied=current.argmax(0)!=17
    dist=torch.where(is_occupied[None],current[:17],dist)
    estimate=torch.cat((dist*occupied[None],(1-occupied)[None]),0)
    # One fixed equal-evidence blend, not a validation-chosen coefficient.
    renderer.static=.5*renderer.static+.5*estimate[None]
    return audit
