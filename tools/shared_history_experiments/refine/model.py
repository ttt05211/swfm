"""Shared STWM + motion-aligned features + compact XYZ semantic refinement."""
from dataclasses import asdict
import math
import torch
from torch import nn
from torch.nn import functional as F
from real_motion.causal_column_completion import ColumnConfig
from tools.shared_history_experiments.model import SharedTransport
from tools.joint_occ_experiments.model import HistoryEgoHead

PROTOCOL = 'p0_f9_shared_semantic_residual_gt_20pct_v1'


class SpatialBlock(nn.Module):
    def __init__(self, width, dilation=1):
        super().__init__()
        # Volumes retain [C,X,Y,Z]. No height pooling or descriptor fitting.
        self.block = nn.Sequential(nn.GroupNorm(1, width),
            nn.Conv3d(width, width, (3, 3, 1), padding=(dilation, dilation, 0),
                      dilation=(dilation, dilation, 1), groups=width),
            nn.Conv3d(width, width, (1, 1, 3), padding=(0, 0, 1), groups=width),
            nn.Conv3d(width, width*2, 1), nn.SiLU(), nn.Conv3d(width*2, width, 1))

    def forward(self, value): return value + self.block(value)


class SemanticRefiner(nn.Module):
    def __init__(self, source_dim=128, feature_width=64, z_bins=16, memory_width=16,
                 width=32, chunk_size=65536, epsilon=1e-4):
        super().__init__()
        self.config = ColumnConfig(z_bins=z_bins)  # provider metadata, NOT a CCR feature ABI
        self.memory_width, self.width = memory_width, width
        self.z_bins, self.chunk_size, self.epsilon = z_bins, chunk_size, epsilon
        self.lift = nn.Conv2d(feature_width, z_bins*memory_width, 1)
        self.source_query = nn.Linear(source_dim, memory_width, bias=False)
        self.importance = nn.Linear(memory_width, 1)
        self.empty = nn.Parameter(torch.zeros(memory_width))
        self.input = nn.Conv3d(memory_width+18, width, 1)
        self.blocks = nn.ModuleList([SpatialBlock(width, d) for d in (1, 2, 3)])
        self.time = nn.Sequential(nn.Linear(1, width), nn.SiLU(), nn.Linear(width, 3*2*width))
        self.readout = nn.Sequential(nn.Linear(width+18+3, 64), nn.SiLU(),
            nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 18))
        nn.init.zeros_(self.readout[-1].weight); nn.init.zeros_(self.readout[-1].bias)

    def lift_history(self, field):
        if field.ndim != 4 or field.shape[0] != 4: raise ValueError('FOUR shared history feature fields required')
        n, _, sx, sy = field.shape
        return self.lift(field).reshape(n, self.memory_width, self.z_bins, sx, sy).permute(0, 1, 3, 4, 2)

    def prior_logits(self, probability):
        # Finite log prior, not log(one_hot). Zero residual preserves argmax.
        p = probability.float()
        return ((p+self.epsilon)/(p.sum(1, keepdim=True)+18*self.epsilon)).log()

    def forward(self, memory, prior, horizons, shape):
        shape = tuple(int(x) for x in shape)
        if prior.shape != (len(horizons), 18, math.prod(shape)):
            raise ValueError('prior/full-grid/horizon shape mismatch')
        if tuple(memory.shape[2:]) != (shape[0]//2, shape[1]//2, shape[2]):
            raise ValueError('memory must retain Z and halve XY only')
        coarse = F.avg_pool3d(prior.float().reshape(-1, 18, *shape), (2, 2, 1))
        with torch.autocast(device_type=memory.device.type, dtype=torch.bfloat16,
                            enabled=memory.device.type == 'cuda'):
            x = self.input(torch.cat((memory, coarse), 1))
            t = x.new_tensor([.5*(h+1)/3 for h in horizons])[:, None]
            conditions = self.time(t).reshape(len(horizons), 3, 2, self.width)
            for i, block in enumerate(self.blocks):
                scale, offset = conditions[:, i].unbind(1)
                x = block(x)*(1+scale[:, :, None, None, None])+offset[:, :, None, None, None]
        # Chunked point queries: do not instantiate six full high-channel volumes.
        volume = prior.shape[-1]
        rows = []
        for start in range(0, volume, self.chunk_size):
            ids = torch.arange(start, min(start+self.chunk_size, volume), device=x.device)
            xyz = torch.stack((ids//(shape[1]*shape[2]), ids//shape[2] % shape[1], ids % shape[2]), -1).float()+.5
            normalized = xyz/xyz.new_tensor(shape)*2-1
            sampling = normalized[:, [2, 1, 0]][None, :, None, None].expand(len(horizons), -1, -1, -1, -1)
            values = F.grid_sample(x.float(), sampling, align_corners=False, padding_mode='border')[:, :, :, 0, 0].transpose(1, 2)
            # Phase is a coordinate fact, not a hand-fitted Surface descriptor.
            cell = xyz/xyz.new_tensor([2, 2, 1])
            phase = cell-cell.floor()-.5
            inputs = torch.cat((values, prior[:, :, ids].transpose(1, 2),
                                phase[None].expand(len(horizons), -1, -1)), -1)
            with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16,
                                enabled=x.device.type == 'cuda'):
                rows.append(self.readout(inputs).float().transpose(1, 2))
        return self.prior_logits(prior)+torch.cat(rows, -1)


class ResidualJoint(nn.Module):
    def __init__(self, motion_config, *, feature_width=64, z_bins=16, memory_width=16,
                 width=32, chunk_size=65536):
        super().__init__()
        self.transport = SharedTransport(motion_config, feature_width, z_bins)
        self.columns = SemanticRefiner(motion_config.d_model, feature_width, z_bins, memory_width, width, chunk_size)
        self.ego = HistoryEgoHead()  # keep the existing auxiliary configuration

    def configs(self):
        return dict(motion=asdict(self.transport.config), shared_history=dict(width=self.transport.history_encoder.width,
            z_bins=self.columns.z_bins, stride=2), repair=dict(type='full_grid_semantic_residual',
            memory_width=self.columns.memory_width, width=self.columns.width, blocks=3,
            chunk_size=self.columns.chunk_size, epsilon=self.columns.epsilon,
            action_heads=False, surface_descriptors=False, ownership_repair=False),
            ego=dict(type='unchanged_history_control', parameters=8710))
