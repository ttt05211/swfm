"""Small navigation-conditioned readout of FROZEN, history-only WM features."""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

PROTOCOL = 'surface_frozen_history_ego_se2_cmd_v1'


@dataclass(frozen=True)
class EgoHeadConfig:
    object_dim: int = 128
    surface_dim: int = 64
    width: int = 128
    heads: int = 4
    layers: int = 2
    object_slots: int = 64
    surface_side: int = 8
    history_frames: int = 4
    future_frames: int = 6
    dt: float = .5
    yaw_weight: float = 1.

    def __post_init__(self):
        if (self.history_frames != 4 or self.future_frames != 6 or self.dt != .5
                or min(self.width, self.heads, self.layers,self.object_slots,
                       self.surface_side, self.object_dim, self.surface_dim) < 1 or self.width % self.heads
                or not math.isfinite(self.yaw_weight) or self.yaw_weight <= 0):
            raise ValueError('positive dimensions and frozen FOUR -> SIX / 2Hz required')


def command_indices(commands, *, batch=None, device=None):
    """Indices right=0 / left=1 / straight=2, or STRICT one-hot [B,6,3]."""
    c = torch.as_tensor(commands, device=device)
    if c.ndim == 3:
        if c.shape[-1] != 3 or not ((c == 0) | (c == 1)).all() or not (c.sum(-1) == 1).all():
            raise ValueError('commands must be exact one-hot, not soft/unknown navigation')
        c = c.argmax(-1)
    if (c.ndim != 2 or c.shape[1] != 6 or (batch is not None and c.shape[0] != batch)
            or not ((c >= 0) & (c <= 2) & (c == c.long())).all()):
        raise ValueError('six right/left/straight commands required per window')
    return c.long()


class HistoryEgoTrajectoryHead(nn.Module):
    """Ego queries attend to shared historical objects, surface cells and ego motion.

    Absolute XY/yaw in fixed t0 axes: NOT object residual motion, NOT incremental
    trajectory integration. Existing WM/CCR is deliberately not a submodule.
    """
    def __init__(self, config=EgoHeadConfig()):
        super().__init__(); self.config = config; w = config.width
        self.objects = nn.Linear(config.object_dim + 8, w)
        self.surfaces = nn.Linear(config.surface_dim + 3, w)
        self.ego = nn.Linear(10, w)
        self.roles = nn.Embedding(3, w)
        self.history_time = nn.Embedding(4, w)
        self.horizon = nn.Embedding(6, w)
        self.command = nn.Embedding(3, w)
        layer = nn.TransformerDecoderLayer(w, config.heads, w * 2, dropout=0.,
            batch_first=True, activation='gelu', norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, config.layers, norm=nn.LayerNorm(w))
        self.readout = nn.Linear(w, 3)
        nn.init.zeros_(self.readout.weight); nn.init.zeros_(self.readout.bias)

    def forward(self, bank, commands):
        required = {'objects', 'object_geometry', 'object_valid', 'surfaces',
                    'surface_geometry', 'surface_valid', 'ego_history'}
        if set(bank) != required:
            raise ValueError('head accepts ONLY the seven historical feature fields')
        c = self.config; e = bank['ego_history']; b = len(e)
        shapes = dict(objects=(b,c.object_slots,c.object_dim), object_geometry=(b,c.object_slots,8),
            object_valid=(b,c.object_slots), surfaces=(b,c.surface_side**2,c.surface_dim),
            surface_geometry=(b,c.surface_side**2,3), surface_valid=(b,c.surface_side**2),
            ego_history=(b,4,10))
        if any(tuple(bank[k].shape) != s for k,s in shapes.items()):
            raise ValueError('history feature bank shape mismatch')
        if any(not torch.isfinite(bank[k]).all() for k in required if not k.endswith('valid')):
            raise ValueError('nonfinite historical feature')
        commands = command_indices(commands, batch=b, device=e.device)
        ob = self.objects(torch.cat((bank['objects'],bank['object_geometry']),-1).float()) + self.roles.weight[0]
        st = self.surfaces(torch.cat((bank['surfaces'],bank['surface_geometry']),-1).float()) + self.roles.weight[1]
        eg = self.ego(e.float()) + self.roles.weight[2] + self.history_time.weight[None]
        memory = torch.cat((ob,st,eg),1)
        valid = torch.cat((bank['object_valid'].bool(),bank['surface_valid'].bool(),
                           torch.ones((b,4),dtype=torch.bool,device=e.device)),1)
        # All three modes are computed; supervision selects the supplied nav mode.
        q = (self.horizon.weight[:,None] + self.command.weight[None]).reshape(18,-1)
        decoded = self.decoder(q[None].expand(b,-1,-1),memory,memory_key_padding_mask=~valid)
        residual = self.readout(decoded).reshape(b,6,3,3)
        times = torch.arange(1,7,device=e.device,dtype=e.dtype)*c.dt
        # Causal constant-velocity / yaw-rate initialization, never future CAN bus.
        prior_xy = e[:,-1,5:7,None].transpose(1,2)*10.*times[None,:,None]
        prior_yaw = e[:,-1,8,None]*math.pi*times[None]
        branches = residual + torch.cat((prior_xy,prior_yaw[...,None]),-1)[:,:,None]
        selected = branches.gather(2,commands[:,:,None,None].expand(-1,-1,1,3)).squeeze(2)
        return dict(se2=selected, branches=branches)


def ego_supervision(output, target, valid=None, *, yaw_weight=1.):
    pred = output['se2']; target = torch.as_tensor(target,device=pred.device,dtype=pred.dtype)
    if target.shape != pred.shape or not torch.isfinite(target).all():
        raise ValueError('finite absolute six-horizon XY/yaw labels required')
    mask = torch.ones(pred.shape[:2],dtype=torch.bool,device=pred.device) if valid is None else torch.as_tensor(valid,device=pred.device).bool()
    if mask.shape != pred.shape[:2] or not mask.any(): raise ValueError('nonempty horizon mask required')
    xy = F.smooth_l1_loss(pred[...,:2],target[...,:2],reduction='none').mean(-1)[mask].mean()
    yaw = (1-torch.cos(pred[...,2]-target[...,2]))[mask].mean()
    return xy + yaw_weight*yaw, dict(xy=xy.detach(),yaw=yaw.detach())
