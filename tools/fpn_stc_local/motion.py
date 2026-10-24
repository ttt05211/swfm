"""Small globally shared motion interventions; identities/geometry stay observed."""
from dataclasses import replace
import numpy as np
import torch
from tools.shared_source_track.adapter import backgrounds
from tools.shared_history_experiments.refine.geometry import TransportPrior
from tools.source_scene_ego_conditioned.geometry import aligned_memory
from tools.joint_occ_experiments.soft_geometry import source_targets,effective_yaw
from tools.fpn_unet_stc_age.processing import weight_history

BASELINE=('learned',1.,0.)
CANDIDATES=(BASELINE,)+tuple((mode,scale,bias) for mode,scale in
    (('learned',1.),('damped',.75),('damped',.5),('KTA',1.)) for bias in (-.25,0.,.25)
    if (mode,scale,bias)!=BASELINE)


def name(c):return f'{c[0]}_motion{c[1]:g}_free{c[2]:+g}'


@torch.no_grad()
def render(body,provider,row,fw,mode,scale):
    _,raw,_,history=row;state=fw['state'];output=fw['output'];poses=fw['poses']
    if mode=='KTA':
        output=dict(output,residual_xy_m=torch.zeros_like(output['residual_xy_m']),
            yaw_delta_rad=torch.zeros_like(output['yaw_delta_rad']))
        targets=source_targets(output,row[0],state['current_pose'],state['source_z_t0'])
        yaws=effective_yaw(output,row[0]['source_class_id'])
    elif mode=='damped':
        centers=poses.new_tensor(np.asarray([s['centroid_world'] for s in state['current']]).reshape(-1,3))
        targets=centers[:,None]+scale*(fw['targets']-centers[:,None]);yaws=fw['yaws']*scale
        track=raw['_shared_source_track']
        changed=replace(track,pair_velocity={i:np.asarray(v)*scale for i,v in track.pair_velocity.items()})
        state=dict(state,column_backgrounds=backgrounds(raw,state,poses,provider,changed))
    else:raise ValueError('explicit damped/KTA mode required')
    weighted,_=weight_history(history,raw['_shared_source_track'].times)
    renderer=TransportPrior(raw,state,provider.pcfg.grid,provider.device)
    prior=renderer.render(poses,targets,yaws)
    memory=aligned_memory(body.columns,fw['lifted'],output,weighted,raw,targets,yaws,poses,provider.pcfg.grid)
    return body.columns(memory,prior,range(6),fw['shape'])


@torch.no_grad()
def predict(body,provider,row,fw):
    shape=fw['shape'];result={}
    def decode(mode,scale,z):
        v=z[[1,3,5]];score,classes=v[:,:17].max(1)
        for bias in (-.25,0.,.25):
            pred=torch.where(score>=v[:,17]+bias,classes,17)
            result[name((mode,scale,bias))]=pred.reshape(3,*shape).byte().cpu().numpy()
    decode('learned',1.,fw['logits'])
    # Avoid retaining original six-frame volumes while rendering alternatives.
    del fw['prior'],fw['memory'],fw['logits']
    for mode,scale in (('damped',.75),('damped',.5),('KTA',1.)):
        z=render(body,provider,row,fw,mode,scale);decode(mode,scale,z);del z
    return result
