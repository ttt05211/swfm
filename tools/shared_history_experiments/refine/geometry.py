"""Continuous Transport prior and owner-free learned spatial evidence pooling."""
import numpy as np
import torch
from torch.nn import functional as F
from tools.joint_occ_experiments.soft_geometry import grid_arrays, transform, move
from tools.joint_occ_experiments.batching.geometry import grouped_splat
from tools.joint_occ_experiments.fast_autograd import compose_sparse
from real_motion.metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS


def centers(grid, device, *, coarse=False):
    origin, step, shape = grid_arrays(grid)
    if coarse: step = step*np.array([2, 2, 1]); shape = shape//np.array([2, 2, 1])
    xyz = torch.stack(torch.meshgrid(*(torch.arange(int(n), device=device, dtype=torch.float64)
                                     for n in shape), indexing='ij'), -1)
    return (xyz+.5)*xyz.new_tensor(step)+xyz.new_tensor(origin)


def inverse_grid(points, matrix, grid, *, coarse=False):
    origin, step, shape = grid_arrays(grid)
    if coarse: step = step*np.array([2, 2, 1]); shape = shape//np.array([2, 2, 1])
    if matrix.ndim == 2:
        mapped = transform(points, matrix)
    else:
        mapped = torch.einsum('bij,b...j->b...i', matrix[:, :3, :3], points)
        mapped = mapped+matrix[:, :3, 3].reshape(len(matrix), *([1]*(points.ndim-2)), 3)
    return (((mapped-mapped.new_tensor(origin))/(mapped.new_tensor(step)*mapped.new_tensor(shape)))*2-1)[..., [2, 1, 0]].float()


class TransportPrior:
    """Same continuous source overlays as edits=False, WITHOUT repair permissions."""
    def __init__(self, raw, state, grid, device):
        self.grid, self.state = grid, state
        self.shape = tuple(int(n) for n in grid_arrays(grid)[2])
        self.volume = int(np.prod(self.shape)); self.centers = centers(grid, device)
        sem = torch.as_tensor(np.array(raw['history_occ'][-1], copy=True), device=device).long()
        dynamic = torch.zeros_like(sem, dtype=torch.bool)
        for cls in DYNAMIC_CLASS_IDS: dynamic |= sem == cls
        sem = torch.where(dynamic | (sem < 0) | (sem > 17), 17, sem)
        self.static = F.one_hot(sem, 18).permute(3, 0, 1, 2)[None].float()
        self.pose = self.centers.new_tensor(state['current_pose'])
        self.backgrounds = torch.as_tensor(np.stack(state['column_backgrounds']), device=device).long().flatten(1)

    def render(self, poses, targets, yaws, horizons=range(6)):
        hs = torch.as_tensor(list(horizons), device=poses.device)
        count = len(hs); selected = poses[hs]
        grids = inverse_grid(self.centers[None].expand(count, -1, -1, -1, -1),
                             torch.linalg.inv(self.pose)[None] @ selected, self.grid)
        p = F.grid_sample(self.static.expand(count, -1, -1, -1, -1), grids,
            align_corners=False, padding_mode='zeros').reshape(count, 18, self.volume)
        missing = (1-p.sum(1)).clamp(0, 1)
        p = p.scatter_add(1, self.backgrounds[hs, None], missing[:, None])
        background = p.permute(1, 0, 2).reshape(18, count*self.volume)
        operations, alpha = [], []
        inverse = torch.linalg.inv(selected)
        for actor, source in enumerate(self.state['current']):
            points = poses.new_tensor(self.state['source_world_points'][actor])
            world = move(points[None], poses.new_tensor(source['centroid_world']),
                         targets[actor, hs, None], yaws[actor, hs, None])
            mapped = torch.einsum('hij,hnj->hni', inverse[:, :3, :3], world)+inverse[:, None, :3, 3]
            slots = torch.arange(count, device=poses.device)[:, None].expand(-1, len(points)).flatten()
            ids, weights = grouped_splat(mapped.reshape(-1, 3), poses.new_ones(count*len(points)).float(),
                                         slots, self.grid, 'sum')
            operations.append(('overlay', ids, int(source['class_id']), None)); alpha.append(weights)
        return compose_sparse(background, operations, alpha).reshape(18, count, self.volume).permute(1, 0, 2)


def aligned_memory(columns, lifted, output, history, raw, targets, yaws, poses, grid, horizons=range(6)):
    """Every observed class/context may contribute; no road/source halo whitelist."""
    causal = history[0]; data, state = causal['memory_reads'], causal['state']
    device = lifted.device; cast = lambda x: torch.as_tensor(np.asarray(x), device=device)
    query = centers(grid, device, coarse=True)
    coarse_shape = tuple(query.shape[:3]); volume = int(np.prod(coarse_shape))
    static_valid = cast(data['static_valid']).float()
    # Scores are shared for static/dynamic, clamped only for exp stability.
    scores = columns.importance(lifted.permute(0, 2, 3, 4, 1).float()).squeeze(-1).clamp(-8, 8).exp()
    static_weight = scores*static_valid
    packed = torch.cat((lifted.float()*static_weight[:, None], static_weight[:, None]), 1)
    cells, frames, actors = (cast(data[k]).long() for k in ('cell', 'frame', 'actor'))
    source_feature = lifted.flatten(2).permute(0, 2, 1)[frames, cells].float()
    points = cast(data['world']).double()
    centers_world = poses.new_tensor(np.asarray([s['centroid_world'] for s in state['current']]).reshape(-1, 3))
    history_poses = poses.new_tensor(raw['history_poses'])
    origin, step, shape = grid_arrays(grid)
    coarse_grid = dict(origin=origin, step=step*np.array([2, 2, 1]), shape=shape//np.array([2, 2, 1]))
    result = []
    for h in horizons:
        matrix = torch.linalg.inv(history_poses) @ poses[h][None]
        sampling = inverse_grid(query[None].expand(4, -1, -1, -1, -1), matrix, grid, coarse=True)
        warped = F.grid_sample(packed, sampling, align_corners=False, padding_mode='zeros').sum(0).flatten(1)
        numerator, denominator = warped[:-1], warped[-1:]
        if len(points):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                features = source_feature+columns.source_query(output['future_transport_queries'][:, h])[actors].float()
                importance = columns.importance(features).float().squeeze(-1).clamp(-8, 8).exp()*cast(data['coverage']).float()
            moved = move(points, centers_world[actors], targets[actors, h], yaws[actors, h])
            mapped = transform(moved, torch.linalg.inv(poses[h]))
            # Same stencil convention; grid_arrays accepts ordinary GridConfig,
            # so the low-resolution lattice is represented explicitly below.
            q = (mapped-mapped.new_tensor(coarse_grid['origin']))/mapped.new_tensor(coarse_grid['step'])-.5
            lower = q.detach().floor().long(); fraction = (q-lower).float()
            offsets = torch.tensor([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], device=device)
            at = lower[:, None]+offsets
            weight = torch.where(offsets.bool()[None], fraction[:, None], 1-fraction[:, None]).prod(-1)
            weight = weight*((at >= 0) & (at < at.new_tensor(coarse_shape))).all(-1)
            ids = ((at[..., 0]*coarse_shape[1]+at[..., 1])*coarse_shape[2]+at[..., 2]).clamp(0, volume-1)
            mass = weight*importance[:, None]
            denominator = denominator.scatter_add(1, ids.flatten()[None], mass.flatten()[None])
            values = (features[:, None]*mass[..., None]).reshape(-1, columns.memory_width).T
            numerator = numerator.scatter_add(1, ids.flatten()[None].expand(columns.memory_width, -1), values)
        memory = torch.where(denominator > 0, numerator/denominator.clamp_min(1e-8), columns.empty[:, None])
        result.append(memory.reshape(columns.memory_width, *coarse_shape))
    return torch.stack(result)
