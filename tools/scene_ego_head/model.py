"""Causal scene readout -> three navigation branches of SE(2) step residuals.

Only shared HISTORY fields enter this module. No future-conditioned memory,
pose, occupancy, annotation, planner output, or frozen feature bank is accepted.
"""
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from real_motion.canonical_causal_repair import grid_arrays
from real_motion.ego_navigation import checked_poses
from real_motion.ego_trajectory_head import command_indices
from tools.ego_experiments.ego_kinematic import history_state, integrate

PROTOCOL = 'shared_history_scene_ego_increment_nav3_v1'


@dataclass(frozen=True)
class SceneEgoConfig:
    feature_width: int = 64
    width: int = 128
    heads: int = 4
    pool_side: int = 8

    def __post_init__(self):
        if (min(self.feature_width, self.width, self.heads, self.pool_side) < 1
                or self.width % self.heads or self.width % 2):
            raise ValueError('positive dimensions, even width divisible by heads required')


@dataclass(frozen=True)
class HistoricalScene:
    features: torch.Tensor  # [B,4*S*S,C], differentiable into shared encoder
    geometry: torch.Tensor  # [B,4*S*S,5]: t0 XYZ/40, actual time/1.5, known fraction
    valid: torch.Tensor     # same tokens; observation, NOT GT sensor visibility


def prepare_scene(field, *, observed, history_poses, ego_history, grid, pool_side=8):
    """One-window adapter, axis convention [X,Y]; neither warps nor caches F.

    Historical frames retain their individual features. Pool-cell positions are
    expressed in t0 via measured HISTORY poses, not via future trajectories.
    Coverage-weighted pooling prevents completely unobserved cells contributing.
    """
    if field.ndim != 4 or field.shape[0] != 4:
        raise ValueError('FOUR history feature grids [4,C,X,Y] required')
    device = field.device
    known = torch.as_tensor(observed, device=device)
    origin, step, shape = grid_arrays(grid)
    sx, sy = field.shape[-2:]
    if (tuple(known.shape) != (4, *tuple(shape))
            or tuple(shape[:2]) != (2*sx, 2*sy)
            or not 1 <= pool_side <= min(sx, sy)):
        raise ValueError('observed grid/stride2 field/pool shape differs')
    if known.dtype != torch.bool:
        raise ValueError('observed must be a boolean validity field')
    e = torch.as_tensor(ego_history, device=device).float()
    if e.shape != (4, 10) or not bool(torch.isfinite(e).all()):
        raise ValueError('finite FOUR-frame ego history required')
    times = e[:, 9]*1.5
    if bool((times[1:] <= times[:-1]).any()) or abs(float(times[-1])) > 1e-4:
        raise ValueError('chronological history ending at t0 required')
    poses = checked_poses(history_poses, 4)
    relative = np.linalg.inv(poses[-1])[None] @ poses
    # Coarse centres coincide with the shared encoder's even-kernel lattice.
    xy = torch.stack(torch.meshgrid(torch.arange(sx, device=device),
        torch.arange(sy, device=device), indexing='ij'), -1).float()+.5
    xy = xy*xy.new_tensor(step[:2]*2)+xy.new_tensor(origin[:2])
    xyz = torch.cat((xy, torch.zeros_like(xy[..., :1])), -1)
    # Adaptive pooling centres use PyTorch's actual bin boundaries, even when
    # 100 is not divisible by 8; do not pretend these are uniform 12.5-cell bins.
    centers = F.adaptive_avg_pool2d(xyz.permute(2, 0, 1)[None], pool_side)[0]
    centers = centers.permute(1, 2, 0).reshape(-1, 3).double()
    matrix = torch.as_tensor(relative, device=device, dtype=torch.float64)
    t0 = torch.einsum('fij,pj->fpi', matrix[:, :3, :3], centers)+matrix[:, None, :3, 3]
    coverage = F.avg_pool2d(known.float().mean(-1)[:, None], 2)
    fraction = F.adaptive_avg_pool2d(coverage, pool_side)
    numerator = F.adaptive_avg_pool2d(field.float()*coverage, pool_side)
    values = numerator/fraction.clamp_min(1e-8)
    features = values.permute(0, 2, 3, 1).reshape(1, -1, field.shape[1])
    fraction = fraction.flatten(1)
    geometry = torch.cat((t0.float()/40., e[:, 9, None, None].expand(-1, len(centers), 1),
        fraction[..., None]), -1).reshape(1, -1, 5)
    return HistoricalScene(features, geometry, (fraction > 0).reshape(1, -1))


def accumulate_increments(branches, commands):
    """Select BEFORE accumulating: commands may differ across the six steps.

    XY step vectors are in fixed t0 axes, NOT per-step body axes. Yaw values
    are relative increments; their sum may remain unwrapped for geometry loss.
    """
    if branches.ndim != 4 or branches.shape[1:] != (6, 3, 3):
        raise ValueError('branches must be [B,6,right/left/straight,XY/yaw]')
    ids = command_indices(commands, batch=len(branches), device=branches.device)
    selected = branches.gather(2, ids[:, :, None, None].expand(-1, -1, 1, 3)).squeeze(2)
    return selected, selected.cumsum(1)


class SceneEgoHead(nn.Module):
    def __init__(self, config=SceneEgoConfig()):
        super().__init__()
        self.config = config
        width = config.width
        self.history = nn.Sequential(nn.Linear(46, width), nn.SiLU(),
                                     nn.Linear(width, width), nn.LayerNorm(width))
        self.scene = nn.Linear(config.feature_width, width)
        self.position = nn.Linear(5, width)
        self.null = nn.Parameter(torch.zeros(1, 1, width))
        self.attention = nn.MultiheadAttention(width, config.heads, batch_first=True, dropout=0.)
        self.fuse = nn.Sequential(nn.Linear(width, width), nn.ReLU(),
                                  nn.Linear(width, width), nn.LayerNorm(width))
        self.readout = nn.Sequential(nn.Linear(width, width//2), nn.Softplus(),
                                      nn.Linear(width//2, 6*3*3))
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)

    def forward(self, history, commands, scene):
        e = history.float()
        state = history_state(e)
        c = self.config; batch = len(e); tokens = 4*c.pool_side**2
        if (scene.features.shape != (batch, tokens, c.feature_width)
                or scene.geometry.shape != (batch, tokens, 5)
                or scene.valid.shape != (batch, tokens) or scene.valid.dtype != torch.bool):
            raise ValueError('historical scene token contract differs')
        if (scene.features.device != e.device or scene.geometry.device != e.device
                or scene.valid.device != e.device
                or not bool(torch.isfinite(scene.features).all())
                or not bool(torch.isfinite(scene.geometry).all())):
            raise ValueError('finite scene tokens on ego device required')
        physical = torch.cat((state['velocity']/10., state['acceleration'][:, None]/4.,
            state['omega'][:, None], state['speed'][:, None]/10., state['jitter'][:, None]), -1)
        query = self.history(torch.cat((e.flatten(1), physical), -1))[:, None]
        memory = self.scene(scene.features.float())+self.position(scene.geometry.float())
        memory = torch.cat((memory, self.null.expand(batch, -1, -1)), 1)
        valid = torch.cat((scene.valid, torch.ones(batch, 1, dtype=torch.bool, device=e.device)), 1)
        # One null key handles all-unknown scenes without NaN softmax or GT fallback.
        read, _ = self.attention(query, memory, memory, key_padding_mask=~valid, need_weights=False)
        fused = self.fuse(query+read).squeeze(1)
        residual = self.readout(fused).float().reshape(batch, 6, 3, 3)
        prior = integrate(state)
        steps = torch.diff(prior, dim=1, prepend=torch.zeros_like(prior[:, :1]))
        branches = steps[:, :, None]+residual
        selected, se2 = accumulate_increments(branches, commands)
        return dict(se2=se2, increments=selected, branch_increments=branches,
                    residual_increments=residual, prior=prior)

    def specification(self):
        return dict(protocol=PROTOCOL, **asdict(self.config), history_frames=4,
            future_frames=6, dt=.5, command_order=['right', 'left', 'straight'],
            parameters=sum(p.numel() for p in self.parameters()),
            coordinates='incremental translation in fixed t0 XY; cumulative yaw',
            navigation='explicit destination command per future step; NOT navigation-free')
