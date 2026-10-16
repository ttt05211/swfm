"""Inference-only ordinary Torch: batch duplicate-address reduction, no CUDA code.

FP64 source SE(2)/ego transforms stay in the original per-source loop. Sources
NEVER share a reduction key or change overlay order. All zero-weight/outside
stencil entries are retained, matching the original grouped_splat convention.
"""
import numpy as np
import torch
from torch.nn import functional as F
from .engine import ReusedTransport
from tools.shared_history_experiments.refine.geometry import inverse_grid
from tools.joint_occ_experiments.soft_geometry import move,stencil
from tools.joint_occ_experiments.fast_autograd import compose_sparse


def packed_splats(mapped,grid,horizons):
    """ONE unique call; return ragged ids/alphas in original source order."""
    if not mapped:return [],[]
    count=int(horizons);sizes=[v.shape[1] for v in mapped]
    points=torch.cat([v.reshape(-1,3) for v in mapped])
    # CPU lengths are already known; one small upload, not per-source uploads.
    slots=np.concatenate([np.repeat(np.arange(count,dtype=np.int64)+i*count,n) for i,n in enumerate(sizes)])
    at,weights,volume=stencil(points,grid)
    addresses=at+torch.as_tensor(slots,device=points.device)[:,None]*volume
    ids,inverse=torch.unique(addresses.flatten(),sorted=True,return_inverse=True)
    alpha=weights.new_zeros(len(ids)).scatter_add_(0,inverse,weights.flatten()).clamp_(0,1)
    stride=count*volume
    lengths=torch.bincount(ids//stride,minlength=len(mapped)).cpu().tolist()
    # One explicit counts sync replaces each source's dynamic unique result.
    return (ids%stride).split(lengths),alpha.split(lengths)


class PackedTransport(ReusedTransport):
    def render(self,poses,targets,yaws,horizons=range(6)):
        hs=torch.as_tensor(list(horizons),device=poses.device);count=len(hs)
        grids=inverse_grid(self.centers[None].expand(count,-1,-1,-1,-1),self.inverse_current[None]@poses[hs],self.grid)
        p=F.grid_sample(self.static.expand(count,-1,-1,-1,-1),grids,align_corners=False,padding_mode='zeros').reshape(count,18,self.volume)
        missing=(1-p.sum(1)).clamp(0,1)
        p=p.scatter_add(1,self.backgrounds[hs,None],missing[:,None])
        background=p.permute(1,0,2).reshape(18,count*self.volume)
        mapped=[];inverse=self.inverse[hs]
        for actor,source in enumerate(self.state['current']):
            # EXACT original calls/shapes/dtype; not FP64 transform fusion.
            points=self.points[actor]
            world=move(points[None],self.source_centers[actor],targets[actor,hs,None],yaws[actor,hs,None])
            mapped.append(torch.einsum('hij,hnj->hni',inverse[:,:3,:3],world)+inverse[:,None,:3,3])
        ids,alpha=packed_splats(mapped,self.grid,count)
        operations=[('overlay',at,int(source['class_id']),None) for source,at in zip(self.state['current'],ids)]
        return compose_sparse(background,operations,alpha).reshape(18,count,self.volume).permute(1,0,2)
