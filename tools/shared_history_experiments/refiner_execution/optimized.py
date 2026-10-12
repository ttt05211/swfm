"""Same Refiner mathematics; coordinate/cast and projection execution reuse."""
from contextlib import contextmanager
from collections import OrderedDict
import math
import numpy as np
import torch
from torch.nn import functional as F
from ..refine import engine
from ..refine.model import SemanticRefiner
from ..refine.geometry import centers, inverse_grid
from tools.joint_occ_experiments.soft_geometry import grid_arrays, transform, move

BACKENDS = ('reference', 'query_reuse', 'projection_reuse', 'memory_h6')


class _SharedQueryCast(torch.autograd.Function):
    """Share forward bytes, retain ORIGINAL per-chunk dtype conversion in VJP.

    A plain x.float() outside the loop sums FP32 chunk gradients before BF16
    rounding. The reference rounds EACH chunk gradient before accumulating x.
    Each alias here has its own backward node, exactly like the old casts,
    without allocating/copying the same full FP32 feature volume repeatedly.
    """
    @staticmethod
    def forward(ctx, value, cached):
        ctx.dtype = value.dtype
        return cached

    @staticmethod
    def backward(ctx, gradient):
        return gradient.to(ctx.dtype), None


class Coordinates:
    """Two bounded immutable geometry entries. NEVER cache learned tensors."""
    def __init__(self):
        self.rows = OrderedDict()

    def get(self, shape, device, chunk):
        key = (tuple(shape), str(device), int(chunk))
        if key in self.rows:
            self.rows.move_to_end(key)
            return self.rows[key]
        chunks = []
        for start in range(0, math.prod(shape), chunk):
            ids = torch.arange(start, min(start+chunk, math.prod(shape)), device=device)
            xyz = torch.stack((ids//(shape[1]*shape[2]), ids//shape[2] % shape[1], ids % shape[2]), -1).float()+.5
            normalized = xyz/xyz.new_tensor(shape)*2-1
            cell = xyz/xyz.new_tensor([2, 2, 1])
            chunks.append((ids, normalized[:, [2, 1, 0]][None, :, None, None], cell-cell.floor()-.5))
        self.rows[key] = chunks
        if len(self.rows) > 2: self.rows.popitem(last=False)
        return chunks


def query_forward(columns, memory, prior, horizons, shape, coordinates):
    shape = tuple(int(x) for x in shape)
    if prior.shape != (len(horizons), 18, math.prod(shape)):
        raise ValueError('prior/full-grid/horizon shape mismatch')
    if tuple(memory.shape[2:]) != (shape[0]//2, shape[1]//2, shape[2]):
        raise ValueError('memory must retain Z and halve XY only')
    coarse = F.avg_pool3d(prior.float().reshape(-1, 18, *shape), (2, 2, 1))
    with torch.autocast(device_type=memory.device.type, dtype=torch.bfloat16,
                        enabled=memory.device.type == 'cuda'):
        x = columns.input(torch.cat((memory, coarse), 1))
        t = x.new_tensor([.5*(h+1)/3 for h in horizons])[:, None]
        conditions = columns.time(t).reshape(len(horizons), 3, 2, columns.width)
        for i, block in enumerate(columns.blocks):
            scale, offset = conditions[:, i].unbind(1)
            x = block(x)*(1+scale[:, :, None, None, None])+offset[:, :, None, None, None]
    # Original code casts this ENTIRE feature volume separately in every chunk.
    # One forward copy; retain per-chunk BF16 cast-back/accumulation semantics.
    x32 = x.detach().float() if x.dtype != torch.float32 else x
    rows = []
    for ids, sampling, phase in coordinates.get(shape, x.device, columns.chunk_size):
        read_x = (_SharedQueryCast.apply(x, x32) if x.dtype != torch.float32
                  and torch.is_grad_enabled() and x.requires_grad else x32)
        values = F.grid_sample(read_x, sampling.expand(len(horizons), -1, -1, -1, -1),
            align_corners=False, padding_mode='border')[:, :, :, 0, 0].transpose(1, 2)
        inputs = torch.cat((values, prior[:, :, ids].transpose(1, 2),
                            phase[None].expand(len(horizons), -1, -1)), -1)
        with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16,
                            enabled=x.device.type == 'cuda'):
            rows.append(columns.readout(inputs).float().transpose(1, 2))
    return columns.prior_logits(prior)+torch.cat(rows, -1)


def projected_memory(columns, lifted, output, history, raw, targets, yaws, poses, grid,
                     horizons=range(6), *, batch_horizons=False):
    """Keep per-source transport/overlay untouched; batch only memory warps."""
    causal = history[0]; data, state = causal['memory_reads'], causal['state']
    device = lifted.device; cast = lambda x: torch.as_tensor(np.asarray(x), device=device)
    horizons = tuple(horizons)
    query = centers(grid, device, coarse=True)
    coarse_shape = tuple(query.shape[:3]); volume = int(np.prod(coarse_shape))
    static_valid = cast(data['static_valid']).float()
    scores = columns.importance(lifted.permute(0, 2, 3, 4, 1).float()).squeeze(-1).clamp(-8, 8).exp()
    static_weight = scores*static_valid
    packed = torch.cat((lifted.float()*static_weight[:, None], static_weight[:, None]), 1)
    cells, frames, actors = (cast(data[k]).long() for k in ('cell', 'frame', 'actor'))
    source_feature = lifted.flatten(2).permute(0, 2, 1)[frames, cells].float()
    points = cast(data['world']).double()
    centers_world = poses.new_tensor(np.asarray([s['centroid_world'] for s in state['current']]).reshape(-1, 3))
    # Shared within this differentiable forward only; do NOT persist Pred poses.
    inverse_history = torch.linalg.inv(poses.new_tensor(raw['history_poses']))
    inverse_future = torch.linalg.inv(poses)
    origin, step, shape = grid_arrays(grid)
    coarse_origin = poses.new_tensor(origin); coarse_step = poses.new_tensor(step*np.array([2, 2, 1]))
    offsets = torch.tensor([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], device=device)
    shape_tensor = offsets.new_tensor(coarse_shape)
    coverage = cast(data['coverage']).float()
    if batch_horizons:
        count = len(horizons)
        # H x four-history batches; sum the SAME four samples per horizon.
        matrices = (inverse_history[None] @ poses[list(horizons), None]).reshape(-1, 4, 4)
        sampling = inverse_grid(query[None].expand(4*count, -1, -1, -1, -1), matrices, grid, coarse=True)
        warped_all = F.grid_sample(packed[None].expand(count, -1, -1, -1, -1, -1).reshape(
            4*count, *packed.shape[1:]), sampling, align_corners=False, padding_mode='zeros')
        warped_all = warped_all.reshape(count, 4, *packed.shape[1:]).sum(1).flatten(2)
    result = []
    for slot, h in enumerate(horizons):
        if batch_horizons:
            warped = warped_all[slot]
        else:
            sampling = inverse_grid(query[None].expand(4, -1, -1, -1, -1),
                                    inverse_history @ poses[h][None], grid, coarse=True)
            warped = F.grid_sample(packed, sampling, align_corners=False, padding_mode='zeros').sum(0).flatten(1)
        numerator, denominator = warped[:-1], warped[-1:]
        if len(points):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                features = source_feature+columns.source_query(output['future_transport_queries'][:, h])[actors].float()
                importance = columns.importance(features).float().squeeze(-1).clamp(-8, 8).exp()*coverage
            moved = move(points, centers_world[actors], targets[actors, h], yaws[actors, h])
            mapped = transform(moved, inverse_future[h])
            q = (mapped-coarse_origin)/coarse_step-.5
            lower = q.detach().floor().long(); fraction = (q-lower).float()
            at = lower[:, None]+offsets
            weight = torch.where(offsets.bool()[None], fraction[:, None], 1-fraction[:, None]).prod(-1)
            weight = weight*((at >= 0) & (at < shape_tensor)).all(-1)
            ids = ((at[..., 0]*coarse_shape[1]+at[..., 1])*coarse_shape[2]+at[..., 2]).clamp(0, volume-1)
            mass = weight*importance[:, None]
            denominator = denominator.scatter_add(1, ids.flatten()[None], mass.flatten()[None])
            values = (features[:, None]*mass[..., None]).reshape(-1, columns.memory_width).T
            numerator = numerator.scatter_add(1, ids.flatten()[None].expand(columns.memory_width, -1), values)
        memory = torch.where(denominator > 0, numerator/denominator.clamp_min(1e-8), columns.empty[:, None])
        result.append(memory.reshape(columns.memory_width, *coarse_shape))
    return torch.stack(result)


@contextmanager
def execution(backend, coordinates=None):
    if backend not in BACKENDS: raise ValueError('unknown execution backend')
    if backend == 'reference':
        yield
        return
    coordinates = coordinates or Coordinates()
    forward, memory = SemanticRefiner.forward, engine.aligned_memory
    try:
        SemanticRefiner.forward = lambda self, *a, **kw: query_forward(self, *a, **kw, coordinates=coordinates)
        if backend != 'query_reuse':
            engine.aligned_memory = lambda *a, **kw: projected_memory(*a, **kw,
                batch_horizons=backend == 'memory_h6')
        yield
    finally:
        SemanticRefiner.forward, engine.aligned_memory = forward, memory
