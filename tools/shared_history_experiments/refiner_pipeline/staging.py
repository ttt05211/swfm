"""Batch immutable input uploads; retain original FP64 geometry arithmetic."""
from collections import OrderedDict
import numpy as np
import torch
from torch.nn import functional as F
from tools.joint_occ_experiments.soft_geometry import grid_arrays, move, source_targets, effective_yaw
from tools.shared_history_experiments.refine.geometry import centers, inverse_grid
from real_motion.metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS


class PackedInputs:
    """One host->device copy per dtype, all uploads INSIDE measured forward.

    Owned pinned host storage remains alive through this window's forward.
    Only history, source associations, explicit future-ego conditions, and
    navigation inputs are packed. NO future occupancy/mask/annotation fields.
    """
    def __init__(self, arrays, device):
        self.values, self.owners = {}, []
        groups = {}
        for name, value in arrays.items():
            a = np.ascontiguousarray(value)
            groups.setdefault(a.dtype.str, []).append((name, a))
        for group in groups.values():
            flat = np.concatenate([a.reshape(-1) for _, a in group])
            host = torch.from_numpy(flat)
            if device.type == 'cuda': host = host.pin_memory()
            self.owners.append(host)
            uploaded = host.to(device, non_blocking=device.type == 'cuda')
            start = 0
            for name, a in group:
                self.values[name] = uploaded[start:start+a.size].reshape(a.shape)
                start += a.size

    def __getitem__(self, key): return self.values[key]


class Constants:
    """Two bounded grid entries, only shape/origin/step, never scene features."""
    def __init__(self): self.rows = OrderedDict()

    def get(self, grid, device):
        origin, step, shape = grid_arrays(grid)
        key = (tuple(origin), tuple(step), tuple(shape), str(device))
        if key in self.rows:
            self.rows.move_to_end(key); return self.rows[key]
        double = lambda a: torch.as_tensor(np.asarray(a), device=device, dtype=torch.float64)
        offsets = torch.tensor([[x,y,z] for x in (0,1) for y in (0,1) for z in (0,1)], device=device)
        value = dict(full=centers(grid, device), coarse=centers(grid, device, coarse=True),
            origin=double(origin), step=double(step), coarse_step=double(step*np.array([2,2,1])),
            dims=torch.as_tensor(shape, device=device), coarse_dims=torch.as_tensor(shape//np.array([2,2,1]), device=device),
            offsets=offsets, offsets_bool=offsets.bool(), classes=torch.arange(18, device=device),
            class_masks=torch.eye(18,device=device,dtype=torch.bool),
            horizons=torch.arange(6, device=device))
        self.rows[key] = value
        if len(self.rows) > 2: self.rows.popitem(last=False)
        return value


class Inverses:
    """Do NOT skip singularity checks; collect and check once per window."""
    def __init__(self, deferred): self.deferred, self.info = deferred, []
    def __call__(self, matrix):
        if not self.deferred: return torch.linalg.inv(matrix)
        inverse, info = torch.linalg.inv_ex(matrix, check_errors=False)
        self.info.append(info.reshape(-1)); return inverse
    def validate(self):
        if self.info and bool(torch.cat(self.info).ne(0).any()):
            raise RuntimeError('singular pose inverse; no optimizer update permitted')


def pack_row(row, device):
    record, raw, ego, history = row
    state = history[0]['state']; reads = history[0]['shared_reads']; memory = history[0]['memory_reads']
    clouds = state['source_world_points']
    lengths = [len(a) for a in clouds]
    sem = np.asarray(raw['history_occ'][-1]).astype(np.int64)
    static = np.where(np.isin(sem, DYNAMIC_CLASS_IDS) | (sem < 0) | (sem > 17), 17, sem)
    arrays = dict(semantics=np.asarray(raw['history_occ']), observed=np.asarray(raw['history_observed']),
        static=static, backgrounds=np.stack(raw['_column_causal_preparation']['prepared_state']['column_backgrounds']).astype(np.int64),
        history_poses=np.asarray(raw['history_poses'], np.float64), future_poses=np.asarray(raw['future_poses'], np.float64),
        current_pose=np.asarray(state['current_pose'], np.float64),
        clouds=np.concatenate(clouds).astype(np.float64) if clouds else np.empty((0,3), np.float64),
        source_centers=np.asarray([s['centroid_world'] for s in state['current']], np.float64).reshape(-1,3),
        source_z=np.asarray(state['source_z_t0'], np.float64),
        anchors=np.asarray(record['anchors_xy_t0_m']), classes=np.asarray(record['source_class_id'], np.int64),
        kta=np.asarray(record['kta_displacement_xy_m'], np.float32),
        ego_history=np.asarray(ego[0])[None], commands=np.asarray(ego[1])[None])
    arrays.update({'reads/'+k: np.asarray(v) for k,v in reads.items()})
    arrays.update({'memory/'+k: np.asarray(memory[k]) for k in ('static_valid','cell','frame','actor','world','coverage')})
    return PackedInputs(arrays, device), lengths


def splat(points, slots, grid, constants):
    """Same stencil/unique/scatter/clamp, no FP64 fusion or reordering."""
    shape = tuple(int(x) for x in grid_arrays(grid)[2]); volume = int(np.prod(shape))
    q = (points-constants['origin'])/constants['step']-.5
    lower = q.detach().floor().long(); fraction = (q-lower).float()
    at = lower[:,None]+constants['offsets'][None]
    weight = torch.where(constants['offsets_bool'][None], fraction[:,None], 1-fraction[:,None]).prod(-1)
    weight = weight*((at >= 0) & (at < constants['dims'])).all(-1)
    flat = ((at[...,0]*shape[1]+at[...,1])*shape[2]+at[...,2]).clamp(0,volume-1)
    flat = flat+slots[:,None]*volume
    ids, inverse = torch.unique(flat.flatten(), sorted=True, return_inverse=True)
    # Preserve original multiplication by one and the exact sum/clamp chain.
    values = (weight*points.new_ones(len(points)).float()[:,None]).flatten()
    return ids, values.new_zeros(len(ids)).scatter_add(0,inverse,values).clamp(0,1)


class Scene:
    def __init__(self, inputs, lengths, state, grid, constants, inverses, compose):
        self.inputs, self.lengths, self.state, self.grid = inputs, lengths, state, grid
        self.c, self.inverse, self.compose = constants, inverses, compose
        self.shape = tuple(int(x) for x in grid_arrays(grid)[2]); self.volume = int(np.prod(self.shape))
        # Avoid an int64 full-grid 18-class one_hot followed by a float copy.
        self.static = torch.zeros((1,18,self.volume), device=inputs['static'].device, dtype=torch.float32).scatter_(
            1, inputs['static'].reshape(1,1,-1), 1.).reshape(1,18,*self.shape)

    def render(self, poses, targets, yaws):
        grids = inverse_grid(self.c['full'][None].expand(6,-1,-1,-1,-1),
                             self.inverse(self.inputs['current_pose'])[None] @ poses, self.grid)
        p = F.grid_sample(self.static.expand(6,-1,-1,-1,-1),grids,
            align_corners=False,padding_mode='zeros').reshape(6,18,self.volume)
        missing = (1-p.sum(1)).clamp(0,1)
        p = p.scatter_add(1,self.inputs['backgrounds'].flatten(1)[:,None],missing[:,None])
        background = p.permute(1,0,2).reshape(18,6*self.volume)
        inverse = self.inverse(poses); operations, alphas, start = [], [], 0
        for actor, source in enumerate(self.state['current']):
            points = self.inputs['clouds'][start:start+self.lengths[actor]]; start += self.lengths[actor]
            # Keep original source-by-source FP64 transforms, including GEMM.
            world = move(points[None],self.inputs['source_centers'][actor],targets[actor,:,None],yaws[actor,:,None])
            mapped = torch.einsum('hij,hnj->hni',inverse[:,:3,:3],world)+inverse[:,None,:3,3]
            slots = self.c['horizons'][:,None].expand(-1,len(points)).flatten()
            ids, alpha = splat(mapped.reshape(-1,3),slots,self.grid,self.c)
            operations.append(('overlay',ids,int(source['class_id']),None)); alphas.append(alpha)
        return self.compose(background,operations,alphas,self.c).reshape(18,6,self.volume).permute(1,0,2)


def memory(columns,lifted,output,history,raw,targets,yaws,poses,grid,inputs,c,inverse):
    device=lifted.device; coarse_shape=tuple(c['coarse'].shape[:3]); volume=int(np.prod(coarse_shape))
    scores=columns.importance(lifted.permute(0,2,3,4,1).float()).squeeze(-1).clamp(-8,8).exp()
    static_weight=scores*inputs['memory/static_valid'].float()
    packed=torch.cat((lifted.float()*static_weight[:,None],static_weight[:,None]),1)
    cells,frames,actors=(inputs['memory/'+k].long() for k in ('cell','frame','actor'))
    source_feature=lifted.flatten(2).permute(0,2,1)[frames,cells].float()
    points=inputs['memory/world'].double(); coverage=inputs['memory/coverage'].float()
    inverse_history=inverse(inputs['history_poses']); inverse_future=inverse(poses)
    result=[]
    for h in range(6):
        sampling=inverse_grid(c['coarse'][None].expand(4,-1,-1,-1,-1),
            inverse_history @ poses[h][None],grid,coarse=True)
        warped=F.grid_sample(packed,sampling,align_corners=False,padding_mode='zeros').sum(0).flatten(1)
        numerator,denominator=warped[:-1],warped[-1:]
        if len(points):
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                features=source_feature+columns.source_query(output['future_transport_queries'][:,h])[actors].float()
                importance=columns.importance(features).float().squeeze(-1).clamp(-8,8).exp()*coverage
            moved=move(points,inputs['source_centers'][actors],targets[actors,h],yaws[actors,h])
            mapped=moved @ inverse_future[h,:3,:3].T+inverse_future[h,:3,3]
            q=(mapped-c['origin'])/c['coarse_step']-.5
            lower=q.detach().floor().long();fraction=(q-lower).float();at=lower[:,None]+c['offsets']
            weight=torch.where(c['offsets_bool'][None],fraction[:,None],1-fraction[:,None]).prod(-1)
            weight=weight*((at>=0)&(at<c['coarse_dims'])).all(-1)
            ids=((at[...,0]*coarse_shape[1]+at[...,1])*coarse_shape[2]+at[...,2]).clamp(0,volume-1)
            mass=weight*importance[:,None]
            denominator=denominator.scatter_add(1,ids.flatten()[None],mass.flatten()[None])
            values=(features[:,None]*mass[...,None]).reshape(-1,columns.memory_width).T
            numerator=numerator.scatter_add(1,ids.flatten()[None].expand(columns.memory_width,-1),values)
        value=torch.where(denominator>0,numerator/denominator.clamp_min(1e-8),columns.empty[:,None])
        result.append(value.reshape(columns.memory_width,*coarse_shape))
    return torch.stack(result)
