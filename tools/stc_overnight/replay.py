"""One-history encoding, horizon-chunked inference; geometry-aware mirror TTA."""
from dataclasses import dataclass
from dataclasses import replace
import numpy as np
import torch
from tools.shared_history_experiments.model import encode_window
from tools.shared_history_experiments.refine.geometry import TransportPrior, aligned_memory
from tools.joint_occ_experiments.soft_geometry import source_targets, effective_yaw
from tools.scene_ego_head.integration import rebuild_pred_backgrounds
from tools.shared_source_track.adapter import backgrounds as track_backgrounds
from tools.shared_history_experiments.refine.geometry import centers, inverse_grid
from real_motion.metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS
from torch.nn import functional as F


@dataclass(frozen=True)
class Mirror:
    x: bool = False
    y: bool = False

    @property
    def name(self): return 'flip_xy' if self.x and self.y else 'flip_x' if self.x else 'flip_y' if self.y else 'identity'

    def apply(self, record, raw, future_poses, grid):
        if not self.x and not self.y: return record, raw, np.asarray(future_poses, np.float64)
        axes = tuple(i+1 for i, active in enumerate((self.x, self.y)) if active)
        shape = np.asarray(grid.shape_hwd); step = np.asarray(grid.voxel_size)
        origin = np.array([grid.x_min, grid.y_min, grid.z_min]); center = origin+shape*step/2
        affine = np.eye(4)
        for i, active in enumerate((self.x, self.y)):
            if active: affine[i, i] = -1; affine[i, 3] = 2*center[i]
        inverse = np.linalg.inv(affine); reference = np.linalg.inv(np.asarray(raw['history_poses'][-1]))
        # Conjugation yields proper rotations even for a reflection, and ends at t0=I.
        convert = lambda poses: affine[None] @ (reference[None] @ np.asarray(poses)) @ inverse[None]
        value = dict(history_occ=np.flip(raw['history_occ'], axes).copy(),
            history_observed=np.flip(raw['history_observed'], axes).copy(), history_poses=convert(raw['history_poses']))
        return record, value, convert(future_poses)

    def restore(self, value, shape):
        # Supports [C,V], [H,C,V] or already-dense [...,X,Y,Z].
        dense = value.reshape(*value.shape[:-1], *shape) if value.shape[-1] == int(np.prod(shape)) else value
        axes = tuple(dense.ndim-3+i for i, active in enumerate((self.x, self.y)) if active)
        if isinstance(dense, torch.Tensor): return dense.flip(axes) if axes else dense
        return np.flip(dense, axes).copy() if axes else dense


VIEWS = {m.name: m for m in (Mirror(), Mirror(True), Mirror(False, True), Mirror(True, True))}


@dataclass(frozen=True)
class GridShift:
    x: int = 0
    y: int = 0

    @property
    def name(self): return f'shift_{self.x}_{self.y}'

    def valid(self, shape):
        valid = np.ones(shape, bool)
        if self.x: valid[-self.x:] = False
        if self.y: valid[:, -self.y:] = False
        return valid

    def apply(self, record, raw, future_poses, grid):
        shape = grid.shape_hwd
        value = dict(history_occ=np.full_like(raw['history_occ'],17),
                     history_observed=np.zeros_like(raw['history_observed'],bool))
        dest = (slice(None),slice(self.x,None),slice(self.y,None),slice(None))
        src = (slice(None),slice(0,shape[0]-self.x),slice(0,shape[1]-self.y),slice(None))
        for key in value: value[key][dest] = raw[key][src]
        affine = np.eye(4); affine[:2,3] = np.asarray((self.x,self.y))*np.asarray(grid.voxel_size)[:2]
        inv = np.linalg.inv(affine); reference = np.linalg.inv(raw['history_poses'][-1])
        convert = lambda poses: affine[None] @ (reference[None] @ np.asarray(poses)) @ inv[None]
        value['history_poses'] = convert(raw['history_poses'])
        return record, value, convert(future_poses)

    def restore(self, value, shape):
        dense = value.reshape(*value.shape[:-1],*shape) if value.shape[-1]==int(np.prod(shape)) else value
        result = torch.zeros_like(dense)
        result[..., :shape[0]-self.x, :shape[1]-self.y,:] = dense[..., self.x:,self.y:,:]
        invalid = torch.as_tensor(~self.valid(shape),device=value.device)
        result[...,invalid] = -1e4
        result[...,17,:,:,:][...,invalid] = 0
        return result


VIEWS.update({s.name:s for s in (GridShift(1,0),GridShift(0,1),GridShift(1,1))})


def blur_semantic_embedding(module,inputs,value):
    """Fixed binomial XY anti-aliasing of EMBEDDINGS, never geometry/labels.

    Each height and embedding channel stays separate. Existing known channels
    are untouched. This is a declared representation variant, not byte parity.
    """
    n,sx,sy,z,e=value.shape
    x=value.permute(0,3,4,1,2).reshape(n,z*e,sx,sy)
    with torch.autocast(device_type=x.device.type,enabled=False):
        k=x.new_tensor([1.,2.,1.],dtype=torch.float32)
        kernel=(k[:,None]*k[None,:]/16)[None,None].expand(z*e,1,3,3)
        result=F.conv2d(F.pad(x.float(),(1,1,1,1),mode='replicate'),kernel,groups=z*e)
    return result.to(value.dtype).reshape(n,z,e,sx,sy).permute(0,3,4,1,2)


@torch.no_grad()
def semantic_memory_consistency(row,grid,device):
    """Soft, history-only agreement weights; never edits the transport prior.

    Register coarse semantic distributions to each frame's native coordinates.
    One self-vote is the Laplace pseudocount, so disagreement cannot completely
    remove a frame. Dynamic memory and all source identities remain unchanged.
    This is an experimental pooling variant, not a production default.
    """
    rec,raw,ego,history=row
    poses=torch.as_tensor(np.asarray(raw['history_poses']),device=device,dtype=torch.float64)
    sem=torch.as_tensor(raw['history_occ'],device=device).long()
    known=torch.as_tensor(raw['history_observed'],device=device).bool() & (sem<18)
    probabilities=[]
    for f in range(4):
        p=F.one_hot(sem[f].clamp(0,17),18).permute(3,0,1,2).float()*known[f][None]
        probabilities.append(F.avg_pool3d(p[None],(2,2,1))[0])
    probabilities=torch.stack(probabilities)
    query=centers(grid,device,coarse=True)
    reliability=[]
    for f in range(4):
        matrices=torch.linalg.inv(poses) @ poses[f][None]
        sampling=inverse_grid(query[None].expand(4,-1,-1,-1,-1),matrices,grid,coarse=True)
        aligned=F.grid_sample(probabilities,sampling,align_corners=False,padding_mode='zeros')
        observed=aligned.sum(1)
        self_probability=probabilities[f]/probabilities[f].sum(0,keepdim=True).clamp_min(1e-8)
        agreements=(aligned*self_probability[None]).sum(1)
        keep=torch.arange(4,device=device)!=f
        reliability.append(((1+agreements[keep].sum(0))/(1+observed[keep].sum(0))).cpu().numpy())
    weights=np.stack(reliability).astype(np.float32)
    causal=dict(history[0]);reads=dict(causal['memory_reads'])
    reads['static_valid']=reads['static_valid']*weights
    causal['memory_reads']=reads
    return (rec,raw,ego,(causal,history[1])),dict(static_only=True,minimum=float(weights.min()),
        mean=float(weights.mean()),maximum=float(weights.max()),GT_or_future_inputs=False)


def multiframe_velocity(row,provider=None,times=None,filter_center=False,velocity_model='half'):
    """Separate diagnostic KTA variant; does NOT change production pair policy.

    Pinned identities/registrations stay fixed. Sources missing the original
    last pair never acquire a velocity from old frames. With >=3 observations,
    shrink the original pair estimate halfway toward a four-frame line fit.
    """
    from real_motion.shared_source_track import kta_and_anchors
    rec,raw,ego,history=row
    track=raw.get('_shared_source_track')
    if track is None:
        if provider is None or times is None:raise ValueError('baseline needs provider and actual history times')
        from real_motion.shared_source_track import from_raw
        _,track=from_raw(raw,provider,times,current=history[0]['state']['current'])
    velocities={k:np.array(v,copy=True) for k,v in track.pair_velocity.items()};changed=[]
    offsets=np.zeros((len(history[0]['state']['current']),3),np.float64)
    for i,v in velocities.items():
        valid=track.track_valid[i]
        if valid.sum()<3:continue
        t=track.times[valid];p=track.centers[i,valid]
        centered=t-t.mean();fit=(centered[:,None]*(p-p.mean(0))).sum(0)/(centered*centered).sum()
        fit[2]=0.
        if velocity_model=='half':velocities[i]=(v+fit)/2
        elif velocity_model=='full':velocities[i]=fit.copy()
        elif velocity_model=='wiener':
            residual=p-(p.mean(0)[None]+centered[:,None]*fit[None])
            sigma2=max(float((residual[:,:2]**2).sum())/(2*max(len(t)-2,1)),.2**2)
            variance_trace=2*sigma2/float((centered*centered).sum())
            shrink=max(0.,1-variance_trace/max(float((fit[:2]**2).sum()),1e-12))
            velocities[i]=fit*shrink
        else:raise ValueError('unknown historical velocity estimator')
        if filter_center:
            intercept=p.mean(0)-fit*t.mean();offsets[i,:2]=.5*(intercept-p[-1])[:2]
        changed.append(float(np.linalg.norm(fit[:2]-v[:2])))
    updated=replace(track,pair_velocity=velocities)
    raw=dict(raw,_shared_source_track=updated)
    if filter_center:
        # Registration/source patches still describe actual historical evidence.
        # Only the estimated future starting geometry is shifted. Point geometry,
        # centers, future anchors, Strong and transported feature points agree.
        causal=dict(history[0]);state=dict(causal['state'])
        state['current']=[dict(s,centroid_world=np.asarray(s['centroid_world'])+offsets[i]) for i,s in enumerate(state['current'])]
        state['source_world_points']=[np.asarray(p)+offsets[i] for i,p in enumerate(state['source_world_points'])]
        state['source_rel_xy']=[p[:,:2]-np.asarray(s['centroid_world'])[None,:2]
            for p,s in zip(state['source_world_points'],state['current'])]
        reads=dict(causal['memory_reads']);reads['world']=reads['world']+offsets[reads['actor']]
        causal.update(state=state,memory_reads=reads);history=(causal,history[1])
    native,kta,anchors=kta_and_anchors(history[0]['state']['current'],updated,raw['history_poses'][-1])
    if filter_center:history[0]['state']['source_z_t0']=native[:,2]
    rec=dict(rec,kta_displacement_xy_m=kta,anchors_xy_t0_m=anchors)
    if filter_center:rec['source_centroid_xy_t0_m']=native[:,:2].astype(np.float32)
    return (rec,raw,ego,history),dict(changed_sources=len(changed),mean_pair_fit_difference_mps=float(np.mean(changed)) if changed else 0.,
        old_registration_and_source_reads_preserved=True,center_filter=filter_center,
        velocity_model=velocity_model,
        mean_center_correction_m=float(np.linalg.norm(offsets[:,:2],axis=1).mean()) if len(offsets) else 0.)


@torch.no_grad()
def temporal_static_prior(context,support_only=False,strict_fill=False):
    """Ego-register four static semantic observations; hard majority with t0 ties.

    Dynamic source geometry and its prediction are untouched. This is a declared
    causal input/prior variant, not a learned refiner or an inference speed patch.
    Nearest sampling avoids synthesizing classes at label boundaries. Out-of-grid
    samples never vote; the t0 vote is always available.
    """
    renderer = context['renderer']; device = context['poses'].device
    sem = torch.as_tensor(context['raw']['history_occ'], device=device).long()
    valid = torch.as_tensor(context['raw']['history_observed'], device=device).bool() & (sem < 18)
    for cls in DYNAMIC_CLASS_IDS: valid &= sem != cls
    history = torch.as_tensor(context['raw']['history_poses'], device=device, dtype=torch.float64)
    query = centers(renderer.grid, device)
    votes = torch.zeros((18, *renderer.shape), device=device)
    for f in range(4):
        mapping = torch.linalg.inv(history[f]) @ renderer.pose
        grid = inverse_grid(query, mapping, renderer.grid)[None]
        labels = F.grid_sample(sem[f].float()[None,None], grid, mode='nearest',
            align_corners=False, padding_mode='zeros')[0,0].long().clamp(0,17)
        weight = F.grid_sample(valid[f].float()[None,None], grid, mode='nearest',
            align_corners=False, padding_mode='zeros')[0,0]
        votes.scatter_add_(0, labels[None], weight[None])
    current = renderer.static[0].argmax(0)
    if strict_fill:
        confidence,winner=votes[:17].max(0)
        use=(current==17)&(confidence>=3)
        chosen=torch.where(use,winner,current)
        renderer.static=F.one_hot(chosen,18).permute(3,0,1,2)[None].float()
        return dict(strict_past3_supported_fill=int(use.sum()),current_occupied_never_deleted=True)
    if support_only:
        # A currently-free prediction can be a missed surface. Inject SOFT
        # evidence only if >=2 past frames agree on an occupied static class.
        # Never erase/relabel an occupied current prediction with old free space.
        supported=votes[:17].max(0).values>=2
        use=(current==17)&supported
        probability=votes/votes.sum(0,keepdim=True).clamp_min(1)
        mixed=.5*renderer.static[0]+.5*probability
        renderer.static=torch.where(use[None],mixed,renderer.static[0])[None]
        return dict(soft_supported_free_voxels=int(use.sum()),mode='half_current_half_four_history_probability_only_supported_free')
    # A strict 3/4-class consensus may replace t0; with weaker support retain it.
    winner = votes.argmax(0); majority = votes.max(0).values >= 3
    winner = torch.where(majority, winner, current)
    renderer.static = F.one_hot(winner,18).permute(3,0,1,2)[None].float()
    return dict(changed_static_voxels=int((winner != current).sum()),
                evidence='historical predicted semantics/known/ego poses only')


@torch.no_grad()
def prepare_forecast(predictor, row, future_poses):
    body, provider = predictor.body, predictor.provider
    rec, raw, ego, history = row; device = provider.device
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        field, reads, times = encode_window(body, raw, history, ego[0], device)
        output = body.transport(field, reads, times,
            torch.as_tensor(rec['source_class_id'], device=device).long(),
            torch.as_tensor(rec['kta_displacement_xy_m'], device=device).float())
        lifted = body.columns.lift_history(field)
    poses = torch.as_tensor(np.array(future_poses, copy=True), device=device, dtype=torch.float64)
    if poses.shape != (6, 4, 4): raise ValueError('SIX explicitly conditioned future poses required')
    state = history[0]['state']
    if '_shared_source_track' in raw:
        backgrounds = track_backgrounds(raw, state, poses, provider, raw['_shared_source_track'])
    else: backgrounds = rebuild_pred_backgrounds(raw, state, poses, provider)
    state = dict(state, column_backgrounds=backgrounds)
    targets = source_targets(output, rec, state['current_pose'], state['source_z_t0'])
    yaws = effective_yaw(output, [int(s['class_id']) for s in state['current']])
    renderer = TransportPrior(raw, state, provider.pcfg.grid, device)
    return dict(body=body, provider=provider, raw=raw, history=history, output=output, lifted=lifted,field=field,
        poses=poses, state=state, targets=targets, yaws=yaws, renderer=renderer, shape=renderer.shape)


@torch.no_grad()
def iter_forecasts(context, *, horizons=range(6), group_size=1):
    if not 1 <= group_size <= 6: raise ValueError('horizon group must be1..6')
    body, p = context['body'], context['provider']; hs = list(horizons)
    for begin in range(0, len(hs), group_size):
        group = hs[begin:begin+group_size]
        prior = context['renderer'].render(context['poses'], context['targets'], context['yaws'], group)
        memory = aligned_memory(body.columns, context['lifted'], context['output'], context['history'],
            context['raw'], context['targets'], context['yaws'], context['poses'], p.pcfg.grid, group)
        logits = body.columns(memory, prior, group, context['shape'])
        yield group, dict(prior=prior, memory=memory, logits=logits, shape=context['shape'])
        del prior, memory, logits
