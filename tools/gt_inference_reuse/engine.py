"""Inference-only reuse. No CUDA code, compile, precision or model-rule changes.

Only initial input staging is outside the timer (same reference boundary).
Fresh Strong, static support, inverse transforms, priors, memory and all learned
features are rebuilt INSIDE EVERY call; no forecast/learned feature cache.
"""
import time
import numpy as np
import torch
from torch.nn import functional as F
from tools.stc_frozen_speed import engine as reference
from tools.stc_overnight import verify_all as old
from tools.shared_source_track.adapter import backgrounds
from tools.shared_history_experiments.refine.geometry import TransportPrior,centers,inverse_grid
from tools.joint_occ_experiments.soft_geometry import source_targets,effective_yaw,move,transform,grid_arrays
from tools.joint_occ_experiments.batching.geometry import grouped_splat
from tools.joint_occ_experiments.fast_autograd import compose_sparse
from real_motion.metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS

BACKENDS=('reference','readout_reuse','geometry_reuse','combined','combined_cpu4')


def prepared(p,row,poses,times):
    ready=reference.prepared(p,row,poses,times)
    # CPU copy of the EXISTING pose input, not a precomputed future transform.
    return ready,np.array(poses,dtype=np.float64,copy=True)


def readout(columns,memory,prior,horizons,shape):
    """Identical point sampling/chunk sizes/MLP; remove repeat casts and gathers."""
    shape=tuple(int(n) for n in shape)
    if prior.shape!=(len(horizons),18,int(np.prod(shape))):raise ValueError('prior/full-grid/horizon shape mismatch')
    if tuple(memory.shape[2:])!=(shape[0]//2,shape[1]//2,shape[2]):raise ValueError('memory must retain Z and halve XY only')
    coarse=F.avg_pool3d(prior.float().reshape(-1,18,*shape),(2,2,1))
    with torch.autocast(device_type=memory.device.type,dtype=torch.bfloat16,enabled=memory.device.type=='cuda'):
        x=columns.input(torch.cat((memory,coarse),1))
        t=x.new_tensor([.5*(h+1)/3 for h in horizons])[:,None]
        conditions=columns.time(t).reshape(len(horizons),3,2,columns.width)
        for i,block in enumerate(columns.blocks):
            scale,offset=conditions[:,i].unbind(1)
            x=block(x)*(1+scale[:,:,None,None,None])+offset[:,:,None,None,None]
    # Original repeats this entire-volume conversion for EACH 65k query chunk.
    field=x.float();rows=[]
    extent=field.new_tensor(shape);cell_extent=field.new_tensor([2,2,1])
    axes=torch.tensor([2,1,0],device=x.device)
    for start in range(0,prior.shape[-1],columns.chunk_size):
        stop=min(start+columns.chunk_size,prior.shape[-1])
        ids=torch.arange(start,stop,device=x.device)
        xyz=torch.stack((ids//(shape[1]*shape[2]),ids//shape[2]%shape[1],ids%shape[2]),-1).float()+.5
        normalized=xyz/extent*2-1
        sampling=normalized[:,axes][None,:,None,None].expand(len(horizons),-1,-1,-1,-1)
        values=F.grid_sample(field,sampling,align_corners=False,padding_mode='border')[:,:,:,0,0].transpose(1,2)
        cell=xyz/cell_extent;phase=cell-cell.floor()-.5
        inputs=torch.cat((values,prior[:,:,start:stop].transpose(1,2),phase[None].expand(len(horizons),-1,-1)),-1)
        with torch.autocast(device_type=x.device.type,dtype=torch.bfloat16,enabled=x.device.type=='cuda'):
            rows.append(columns.readout(inputs).float().transpose(1,2))
    return columns.prior_logits(prior)+torch.cat(rows,-1)


class ReusedTransport(TransportPrior):
    def __init__(self,raw,state,grid,device,poses,staged):
        self.grid,self.state=grid,state;self.shape=tuple(int(n) for n in grid_arrays(grid)[2])
        self.volume=int(np.prod(self.shape));self.centers=centers(grid,device)
        self.history_semantic=staged['semantic'].long()
        sem=self.history_semantic[-1];dynamic=torch.zeros_like(sem,dtype=torch.bool)
        for cls in DYNAMIC_CLASS_IDS:dynamic|=sem==cls
        self.static_semantic=torch.where(dynamic|(sem<0)|(sem>17),17,sem)
        self.static=F.one_hot(self.static_semantic,18).permute(3,0,1,2)[None].float()
        self.pose=self.centers.new_tensor(state['current_pose'])
        self.backgrounds=torch.as_tensor(np.stack(state['column_backgrounds']),device=device).long().flatten(1)
        self.history_poses=poses.new_tensor(np.asarray(raw['history_poses']))
        self.inverse=torch.linalg.inv(poses);self.inverse_current=torch.linalg.inv(self.pose)
        lengths=[len(v) for v in state['source_world_points']]
        # One upload instead of one upload for EVERY source. All original point
        # order, source order, FP64 geometry and overlay order are retained.
        packed=np.concatenate(state['source_world_points']) if lengths else np.empty((0,3))
        self.points=torch.as_tensor(np.asarray(packed),device=device,dtype=poses.dtype).split(lengths)
        self.source_centers=poses.new_tensor(np.asarray([s['centroid_world'] for s in state['current']]).reshape(-1,3))

    def static_support(self,observed):
        # Same four-frame votes and mixture. Reuse input uploads/full XYZ grid,
        # and avoid the old unused int(use.sum()) -> CUDA host synchronization.
        sem=self.history_semantic;valid=observed.bool()&(sem<18)
        for cls in DYNAMIC_CLASS_IDS:valid&=sem!=cls
        votes=torch.zeros((18,*self.shape),device=sem.device)
        for f in range(4):
            mapping=torch.linalg.inv(self.history_poses[f])@self.pose
            sampling=inverse_grid(self.centers,mapping,self.grid)[None]
            labels=F.grid_sample(sem[f].float()[None,None],sampling,mode='nearest',align_corners=False,padding_mode='zeros')[0,0].long().clamp(0,17)
            weight=F.grid_sample(valid[f].float()[None,None],sampling,mode='nearest',align_corners=False,padding_mode='zeros')[0,0]
            votes.scatter_add_(0,labels[None],weight[None])
        use=(self.static_semantic==17)&(votes[:17].max(0).values>=2)
        probability=votes/votes.sum(0,keepdim=True).clamp_min(1)
        mixed=.5*self.static[0]+.5*probability
        self.static=torch.where(use[None],mixed,self.static[0])[None]

    def render(self,poses,targets,yaws,horizons=range(6)):
        hs=torch.as_tensor(list(horizons),device=poses.device);count=len(hs)
        grids=inverse_grid(self.centers[None].expand(count,-1,-1,-1,-1),self.inverse_current[None]@poses[hs],self.grid)
        p=F.grid_sample(self.static.expand(count,-1,-1,-1,-1),grids,align_corners=False,padding_mode='zeros').reshape(count,18,self.volume)
        missing=(1-p.sum(1)).clamp(0,1)
        p=p.scatter_add(1,self.backgrounds[hs,None],missing[:,None])
        background=p.permute(1,0,2).reshape(18,count*self.volume);operations=[];alpha=[]
        inverse=self.inverse[hs]
        for actor,source in enumerate(self.state['current']):
            points=self.points[actor]
            world=move(points[None],self.source_centers[actor],targets[actor,hs,None],yaws[actor,hs,None])
            mapped=torch.einsum('hij,hnj->hni',inverse[:,:3,:3],world)+inverse[:,None,:3,3]
            slots=torch.arange(count,device=poses.device)[:,None].expand(-1,len(points)).flatten()
            ids,weights=grouped_splat(mapped.reshape(-1,3),poses.new_ones(count*len(points)).float(),slots,self.grid,'sum')
            operations.append(('overlay',ids,int(source['class_id']),None));alpha.append(weights)
        return compose_sparse(background,operations,alpha).reshape(18,count,self.volume).permute(1,0,2)


def memory(columns,lifted,output,history,raw,targets,yaws,poses,grid,renderer,horizons):
    data,state=history[0]['memory_reads'],history[0]['state'];device=lifted.device
    cast=lambda x:torch.as_tensor(np.asarray(x),device=device)
    query=centers(grid,device,coarse=True);coarse_shape=tuple(query.shape[:3]);volume=int(np.prod(coarse_shape))
    scores=columns.importance(lifted.permute(0,2,3,4,1).float()).squeeze(-1).clamp(-8,8).exp()
    static_weight=scores*cast(data['static_valid']).float()
    packed=torch.cat((lifted.float()*static_weight[:,None],static_weight[:,None]),1)
    cells,frames,actors=(cast(data[k]).long() for k in ('cell','frame','actor'))
    source_feature=lifted.flatten(2).permute(0,2,1)[frames,cells].float();points=cast(data['world']).double()
    centers_world=renderer.source_centers
    inverse_history=torch.linalg.inv(renderer.history_poses)
    origin,step,shape=grid_arrays(grid);coarse_step=step*np.array([2,2,1])
    # These are invariant THROUGHOUT six readouts, but still rebuilt per call.
    coverage=cast(data['coverage']).float()
    offsets=torch.tensor([[x,y,z] for x in (0,1) for y in (0,1) for z in (0,1)],device=device)
    coarse_origin=poses.new_tensor(origin);coarse_step=poses.new_tensor(coarse_step)
    coarse_extent=offsets.new_tensor(coarse_shape);result=[]
    for h in horizons:
        matrix=inverse_history@poses[h][None]
        sampling=inverse_grid(query[None].expand(4,-1,-1,-1,-1),matrix,grid,coarse=True)
        warped=F.grid_sample(packed,sampling,align_corners=False,padding_mode='zeros').sum(0).flatten(1)
        numerator,denominator=warped[:-1],warped[-1:]
        if len(points):
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                features=source_feature+columns.source_query(output['future_transport_queries'][:,h])[actors].float()
                importance=columns.importance(features).float().squeeze(-1).clamp(-8,8).exp()*coverage
            moved=move(points,centers_world[actors],targets[actors,h],yaws[actors,h])
            mapped=transform(moved,renderer.inverse[h])
            q=(mapped-coarse_origin)/coarse_step-.5
            lower=q.detach().floor().long();fraction=(q-lower).float();at=lower[:,None]+offsets
            weight=torch.where(offsets.bool()[None],fraction[:,None],1-fraction[:,None]).prod(-1)
            weight=weight*((at>=0)&(at<coarse_extent)).all(-1)
            ids=((at[...,0]*coarse_shape[1]+at[...,1])*coarse_shape[2]+at[...,2]).clamp(0,volume-1)
            mass=weight*importance[:,None]
            denominator=denominator.scatter_add(1,ids.flatten()[None],mass.flatten()[None])
            values=(features[:,None]*mass[...,None]).reshape(-1,columns.memory_width).T
            numerator=numerator.scatter_add(1,ids.flatten()[None].expand(columns.memory_width,-1),values)
        value=torch.where(denominator>0,numerator/denominator.clamp_min(1e-8),columns.empty[:,None])
        result.append(value.reshape(columns.memory_width,*coarse_shape))
    return torch.stack(result)


@torch.no_grad()
def fresh(p,ready,bias,backend='combined',profile=None):
    if backend not in BACKENDS:raise ValueError('unknown execution backend')
    if backend=='reference':return reference.fresh(p,ready[0],bias,6,profile)
    use_geometry=backend in ('geometry_reuse','combined','combined_cpu4')
    use_readout=backend in ('readout_reuse','combined','combined_cpu4')
    (row,staged,times),cpu_poses=ready;device=p.provider.device;body=p.body
    def stage(name,fn):
        if profile is None:return fn()
        reference.sync(device);tic=time.perf_counter();v=fn();reference.sync(device);profile[name]=time.perf_counter()-tic;return v
    modified,_=stage('live_KTA_and_age_weights',lambda:old.candidate(row,p.provider,times))
    rec,raw,ego,history=modified;poses=staged['poses'];state=history[0]['state']
    def network():
        with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
            field=body.transport.history_encoder(staged['semantic'],staged['observed'])
            output=body.transport(field,staged['reads'],staged['times'],staged['classes'],torch.as_tensor(rec['kta_displacement_xy_m'],device=device).float())
            lifted=body.columns.lift_history(field)
        return output,lifted
    output,lifted=stage('shared_encoder_STWM_lift',network)
    def strong():
        original=getattr(p.provider,'strong_majority',None)
        if backend=='combined_cpu4':p.provider.strong_majority=p.gt_reuse_majority
        try:return backgrounds(raw,state,cpu_poses if use_geometry else poses,p.provider,raw['_shared_source_track'])
        finally:p.provider.strong_majority=original
    bg=stage('fresh_Strong_CLEAR',strong)
    state=dict(state,column_backgrounds=bg)
    targets=source_targets(output,rec,state['current_pose'],state['source_z_t0'])
    yaws=effective_yaw(output,[int(s['class_id']) for s in state['current']])
    def renderer():
        value=ReusedTransport(raw,state,p.provider.pcfg.grid,device,poses,staged) if use_geometry else TransportPrior(raw,state,p.provider.pcfg.grid,device)
        if use_geometry:value.static_support(staged['observed'])
        else:old.temporal_static_prior(dict(renderer=value,raw=raw,poses=poses),support_only=True)
        return value
    transport=stage('prior_setup_and_static_support',renderer)
    prior=stage('transport_H0',lambda:transport.render(poses,targets,yaws,range(6)))
    if use_geometry:
        mem=stage('memory_H0',lambda:memory(body.columns,lifted,output,history,raw,targets,yaws,poses,p.provider.pcfg.grid,transport,range(6)))
    else:
        from tools.shared_history_experiments.refine.geometry import aligned_memory
        mem=stage('memory_H0',lambda:aligned_memory(body.columns,lifted,output,history,raw,targets,yaws,poses,p.provider.pcfg.grid,range(6)))
    logits=stage('refiner_H0',lambda:readout(body.columns,mem,prior,range(6),transport.shape) if use_readout else body.columns(mem,prior,range(6),transport.shape))
    return stage('calibration_argmax_CPU_H0',lambda:(logits+logits.new_tensor(bias)[None,:,None]).argmax(1).reshape(6,*transport.shape).byte().cpu().numpy())


def strong_audit(p,ready):
    """Untimed six-frame BYTE check: existing original CPU pool vs CPU4 pool."""
    (row,staged,times),cpu_poses=ready
    modified,_=old.candidate(row,p.provider,times);_,raw,_,history=modified;state=history[0]['state']
    original=getattr(p.provider,'strong_majority',None)
    try:
        a=backgrounds(raw,state,staged['poses'],p.provider,raw['_shared_source_track'])
        p.provider.strong_majority=p.gt_reuse_majority
        b=backgrounds(raw,state,cpu_poses,p.provider,raw['_shared_source_track'])
    finally:p.provider.strong_majority=original
    changes=[int(np.count_nonzero(x!=y)) for x,y in zip(a,b)]
    return dict(changed_voxels_by_horizon=changes,strict_byte_parity=len(changes)==6 and not any(changes),
        reference_majority_workers=original.workers,candidate_majority_workers=p.gt_reuse_majority.workers,
        existing_native_implementation_unchanged=True)
