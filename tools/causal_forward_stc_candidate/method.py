"""Pure causal state estimation and one continuous calibration formula."""
from dataclasses import replace
import numpy as np
import torch
from real_motion.shared_source_track import kta_and_anchors
from tools.shared_history_experiments.refine.geometry import centers


def state_row(row,position_blend=.5,velocity_blend=.5):
    rec,raw,ego,history=row;track=raw['_shared_source_track']
    velocities={k:np.array(v,copy=True) for k,v in track.pair_velocity.items()}
    offset=np.zeros((len(history[0]['state']['current']),3),np.float64)
    for i,original in velocities.items():
        good=track.track_valid[i]
        if good.sum()<3:continue
        t=track.times[good];p=track.centers[i,good];u=t-t.mean()
        fitted=(u[:,None]*(p-p.mean(0))).sum(0)/(u*u).sum();fitted[2]=0.
        velocities[i]=(1-velocity_blend)*original+velocity_blend*fitted
        # Position uses the full XYZ line fit, then retains WORLD XY only,
        # exactly as in the locally tested position_row implementation.
        slope=(u[:,None]*(p-p.mean(0))).sum(0)/(u*u).sum()
        intercept=p.mean(0)-slope*t.mean();offset[i,:2]=position_blend*(intercept-track.centers[i,-1])[:2]
    updated=replace(track,pair_velocity=velocities)
    causal=dict(history[0]);state=dict(causal['state'])
    state['current']=[dict(s,centroid_world=np.asarray(s['centroid_world'])+offset[i]) for i,s in enumerate(state['current'])]
    state['source_world_points']=[np.asarray(v)+offset[i] for i,v in enumerate(state['source_world_points'])]
    state['source_rel_xy']=[v[:,:2]-state['current'][i]['centroid_world'][None,:2] for i,v in enumerate(state['source_world_points'])]
    native,kta,anchors=kta_and_anchors(state['current'],updated,state['current_pose']);state['source_z_t0']=native[:,2]
    causal['state']=state;fixed=dict(raw['_column_causal_preparation'],prepared_state=state)
    rec=dict(rec,source_centroid_xy_t0_m=native[:,:2].astype(np.float32),kta_displacement_xy_m=kta,anchors_xy_t0_m=anchors)
    # Observed world points AND historical centers in memory_reads are never
    # shifted. Direct transport applies the estimated target correction ONCE.
    return rec,dict(raw,_shared_source_track=updated,_column_causal_preparation=fixed),ego,(causal,*history[1:])


@torch.no_grad()
def radius(grid,current,poses):
    query=centers(grid,poses.device);matrix=torch.linalg.inv(torch.as_tensor(current,device=poses.device).double())[None]@poses
    fields=[]
    for at in matrix:
        point=query@at[:3,:3].T+at[:3,3]
        fields.append((point[...,:2].square().sum(-1).sqrt()/40).clamp(0,1).float().flatten())
    return torch.stack(fields)


def calibrated_logits(logits,rho,intercept=-2.,slope=1.):
    if logits.ndim!=3 or logits.shape[:2]!=(6,18) or rho.shape!=logits[:,17].shape:
        raise ValueError('complete SIX-frame18-class readout and SAME conditioned distance field required')
    result=logits.clone();result[:,17]+=intercept+slope*rho
    return result
