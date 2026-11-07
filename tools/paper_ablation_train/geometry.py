"""Training forwards only; labels never enter motion, geometry or memory."""
import numpy as np
import torch
from torch.nn import functional as F
from tools.fpn_unet.adapter import forward_window as full_forward
from tools.shared_source_track.adapter import backgrounds
from tools.source_scene_ego_conditioned.model import relative_ego_features
from tools.source_scene_ego_conditioned.geometry import aligned_memory
from tools.shared_history_experiments.refine.geometry import TransportPrior,centers,inverse_grid
from tools.joint_occ_experiments.soft_geometry import source_targets,effective_yaw,grid_arrays,transform


def ego_only_memory(columns,lifted,history,raw,poses,grid):
    """Same feature/importance/coverage pooling; dynamic WORLD points stay put."""
    data=history[0]['memory_reads'];device=lifted.device
    cast=lambda x:torch.as_tensor(np.asarray(x),device=device)
    query=centers(grid,device,coarse=True);shape=tuple(query.shape[:3]);volume=int(np.prod(shape))
    scores=columns.importance(lifted.permute(0,2,3,4,1).float()).squeeze(-1).clamp(-8,8).exp()
    static_weight=scores*cast(data['static_valid']).float()
    packed=torch.cat((lifted.float()*static_weight[:,None],static_weight[:,None]),1)
    cells,frames=(cast(data[k]).long() for k in ('cell','frame'))
    features=lifted.flatten(2).permute(0,2,1)[frames,cells].float()
    points=cast(data['world']).double()
    with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
        importance=columns.importance(features).float().squeeze(-1).clamp(-8,8).exp()*cast(data['coverage']).float()
    history_poses=poses.new_tensor(np.asarray(raw['history_poses']))
    origin,step,_=grid_arrays(grid);coarse_step=step*np.array([2,2,1]);result=[]
    for h in range(6):
        matrix=torch.linalg.inv(history_poses) @ poses[h][None]
        sampling=inverse_grid(query[None].expand(4,-1,-1,-1,-1),matrix,grid,coarse=True)
        warped=F.grid_sample(packed,sampling,align_corners=False,padding_mode='zeros').sum(0).flatten(1)
        numerator,denominator=warped[:-1],warped[-1:]
        if len(points):
            mapped=transform(points,torch.linalg.inv(poses[h]))
            q=(mapped-mapped.new_tensor(origin))/mapped.new_tensor(coarse_step)-.5
            lower=q.detach().floor().long();fraction=(q-lower).float()
            offsets=torch.tensor([[x,y,z] for x in (0,1) for y in (0,1) for z in (0,1)],device=device)
            at=lower[:,None]+offsets
            weight=torch.where(offsets.bool()[None],fraction[:,None],1-fraction[:,None]).prod(-1)
            weight=weight*((at>=0)&(at<at.new_tensor(shape))).all(-1)
            ids=((at[...,0]*shape[1]+at[...,1])*shape[2]+at[...,2]).clamp(0,volume-1)
            mass=weight*importance[:,None]
            denominator=denominator.scatter_add(1,ids.flatten()[None],mass.flatten()[None])
            values=(features[:,None]*mass[...,None]).reshape(-1,columns.memory_width).T
            numerator=numerator.scatter_add(1,ids.flatten()[None].expand(columns.memory_width,-1),values)
        value=torch.where(denominator>0,numerator/denominator.clamp_min(1e-8),columns.empty[:,None])
        result.append(value.reshape(columns.memory_width,*shape))
    return torch.stack(result)


def forward_window(model,provider,row,*,poses=None):
    if model.arm=='s0_pool': return full_forward(model,provider,row,poses=poses,fresh=True)
    rec,raw,ego,history=row;state=history[0]['state'];device=provider.device
    poses=torch.as_tensor(np.asarray(raw['future_poses']) if poses is None else poses,device=device).double()
    ego_features=relative_ego_features(poses.new_tensor(state['current_pose']),poses)
    sem=torch.as_tensor(np.asarray(raw['history_occ']),device=device)
    known=torch.as_tensor(np.asarray(raw['history_observed']),device=device)
    reads={k:torch.as_tensor(v,device=device) for k,v in history[0]['shared_reads'].items()}
    times=torch.as_tensor(ego[0][:,-1],device=device)
    kta=torch.as_tensor(np.asarray(rec['kta_displacement_xy_m']),device=device).float()
    with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
        fields=model.transport.history_encoder(sem,known)
        output=model.transport(fields,reads,times,None,kta,ego_features)
        lifted=model.columns.lift_history(fields[0]) if model.recipe['memory']!='none' else None
    state=dict(state,column_backgrounds=backgrounds(raw,state,poses,provider,raw['_shared_source_track']))
    targets=source_targets(output,rec,state['current_pose'],state['source_z_t0'])
    yaws=effective_yaw(output,rec['source_class_id'])
    renderer=TransportPrior(raw,state,provider.pcfg.grid,device)
    prior=renderer.render(poses,targets,yaws)
    memory=None
    if not model.recipe['refiner']:
        logits=model.columns.prior_logits(prior)
    else:
        if model.recipe['memory']=='none':
            sx,sy,sz=renderer.shape
            memory=prior.new_zeros((6,model.columns.memory_width,sx//2,sy//2,sz))
        elif model.recipe['memory']=='ego_only':
            memory=ego_only_memory(model.columns,lifted,history,raw,poses,provider.pcfg.grid)
        else:
            memory=aligned_memory(model.columns,lifted,output,history,raw,targets,yaws,poses,provider.pcfg.grid)
        logits=model.columns(memory,prior,range(6),renderer.shape)
    return dict(logits=logits,prior=prior,memory=memory,output=output,state=state,poses=poses,
        targets=targets,yaws=yaws,shape=renderer.shape)
