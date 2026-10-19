"""Historical content/importance ONLY; future motion changes alignment, not evidence."""
import numpy as np
import torch
from torch.nn import functional as F
from tools.shared_history_experiments.refine.geometry import centers, inverse_grid
from tools.joint_occ_experiments.soft_geometry import grid_arrays, transform, move


def direct_move(points, historical_centers, future_centers, history_yaw, future_yaw):
    # R(theta_h-theta_t) = R(theta_0-theta_t + theta_h-theta_0).
    # Preserve original world-Z convention; ego projection handles full poses.
    return move(points, historical_centers, future_centers, history_yaw+future_yaw)


def aligned_memory(columns, lifted, output, history, raw, targets, yaws, poses, grid, horizons=range(6)):
    data = history[0]['memory_reads']; device = lifted.device
    cast = lambda x: torch.as_tensor(np.asarray(x), device=device)
    query = centers(grid, device, coarse=True); coarse_shape = tuple(query.shape[:3]); volume = int(np.prod(coarse_shape))
    static_valid = cast(data['static_valid']).float()
    scores = columns.importance(lifted.permute(0, 2, 3, 4, 1).float()).squeeze(-1).clamp(-8, 8).exp()
    static_weight = scores*static_valid
    packed = torch.cat((lifted.float()*static_weight[:, None], static_weight[:, None]), 1)
    cells, frames, actors = (cast(data[k]).long() for k in ('cell', 'frame', 'actor'))
    source_feature = lifted.flatten(2).permute(0, 2, 1)[frames, cells].float()
    points = cast(data['world']).double()
    historical_centers = cast(data['centers_world']).double()[actors, frames]
    relative_yaw = output['history_yaw_to_t0_rad'][actors, frames]
    history_poses = poses.new_tensor(np.asarray(raw['history_poses']))
    origin, step, shape = grid_arrays(grid)
    coarse_grid = dict(origin=origin, step=step*np.array([2, 2, 1]), shape=shape//np.array([2, 2, 1]))
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        features = source_feature
        importance = columns.importance(features).float().squeeze(-1).clamp(-8, 8).exp()*cast(data['coverage']).float()
    result = []
    for h in horizons:
        matrix = torch.linalg.inv(history_poses) @ poses[h][None]
        sampling = inverse_grid(query[None].expand(4, -1, -1, -1, -1), matrix, grid, coarse=True)
        warped = F.grid_sample(packed, sampling, align_corners=False, padding_mode='zeros').sum(0).flatten(1)
        numerator, denominator = warped[:-1], warped[-1:]
        if len(points):
            moved = direct_move(points, historical_centers, targets[actors, h], relative_yaw, yaws[actors, h])
            mapped = transform(moved, torch.linalg.inv(poses[h]))
            q = (mapped-mapped.new_tensor(coarse_grid['origin']))/mapped.new_tensor(coarse_grid['step'])-.5
            lower = q.detach().floor().long(); fraction = (q-lower).float()
            offsets = torch.tensor([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], device=device)
            at = lower[:, None]+offsets
            weight = torch.where(offsets.bool()[None], fraction[:, None], 1-fraction[:, None]).prod(-1)
            weight = weight*((at >= 0)&(at < at.new_tensor(coarse_shape))).all(-1)
            ids = ((at[..., 0]*coarse_shape[1]+at[..., 1])*coarse_shape[2]+at[..., 2]).clamp(0, volume-1)
            mass = weight*importance[:, None]
            denominator = denominator.scatter_add(1, ids.flatten()[None], mass.flatten()[None])
            values = (features[:, None]*mass[..., None]).reshape(-1, columns.memory_width).T
            numerator = numerator.scatter_add(1, ids.flatten()[None].expand(columns.memory_width, -1), values)
        value = torch.where(denominator > 0, numerator/denominator.clamp_min(1e-8), columns.empty[:, None])
        result.append(value.reshape(columns.memory_width, *coarse_shape))
    return torch.stack(result)
