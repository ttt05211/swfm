"""Horizon AdaGN acts ONLY inside the spatial residual branch."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from tools.shared_history_experiments.refine.model import SemanticRefiner


class AdaGNBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.norm = nn.GroupNorm(1, width, affine=False)
        self.affine = nn.Linear(width, 2*width)
        nn.init.zeros_(self.affine.weight); nn.init.zeros_(self.affine.bias)
        self.convs = nn.Sequential(
            nn.Conv3d(width, width, (3, 3, 1), padding=(dilation, dilation, 0),
                      dilation=(dilation, dilation, 1), groups=width),
            nn.Conv3d(width, width, (1, 1, 3), padding=(0, 0, 1), groups=width),
            nn.Conv3d(width, width*2, 1), nn.SiLU(), nn.Conv3d(width*2, width, 1))

    def forward(self, value, embedding):
        scale, offset = self.affine(embedding).chunk(2, -1)
        normalized = self.norm(value)*(1+scale[:, :, None, None, None])+offset[:, :, None, None, None]
        return value+self.convs(normalized)


class AdaGNRefiner(SemanticRefiner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        width = self.width
        self.blocks = nn.ModuleList([AdaGNBlock(width, d) for d in (1, 2, 3)])
        self.time = nn.Sequential(nn.Linear(1, width), nn.SiLU(), nn.Linear(width, width))

    def forward(self, memory, prior, horizons, shape):
        shape = tuple(int(x) for x in shape)
        if prior.shape != (len(horizons), 18, math.prod(shape)):
            raise ValueError('prior/full-grid/horizon shape mismatch')
        if tuple(memory.shape[2:]) != (shape[0]//2, shape[1]//2, shape[2]):
            raise ValueError('memory must retain Z and halve XY only')
        coarse = F.avg_pool3d(prior.float().reshape(-1, 18, *shape), (2, 2, 1))
        with torch.autocast(device_type=memory.device.type, dtype=torch.bfloat16, enabled=memory.device.type == 'cuda'):
            x = self.input(torch.cat((memory, coarse), 1))
            t = x.new_tensor([.5*(h+1)/3 for h in horizons])[:, None]
            embedding = self.time(t)
            for block in self.blocks: x = block(x, embedding)
        # SAME full-resolution queries, phase, MLP and finite log-prior.
        rows = []
        for start in range(0, prior.shape[-1], self.chunk_size):
            ids = torch.arange(start, min(start+self.chunk_size, prior.shape[-1]), device=x.device)
            xyz = torch.stack((ids//(shape[1]*shape[2]), ids//shape[2] % shape[1], ids % shape[2]), -1).float()+.5
            normalized = xyz/xyz.new_tensor(shape)*2-1
            sampling = normalized[:, [2, 1, 0]][None, :, None, None].expand(len(horizons), -1, -1, -1, -1)
            values = F.grid_sample(x.float(), sampling, align_corners=False, padding_mode='border')[:, :, :, 0, 0].transpose(1, 2)
            cell = xyz/xyz.new_tensor([2, 2, 1]); phase = cell-cell.floor()-.5
            inputs = torch.cat((values, prior[:, :, ids].transpose(1, 2), phase[None].expand(len(horizons), -1, -1)), -1)
            with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=x.device.type == 'cuda'):
                rows.append(self.readout(inputs).float().transpose(1, 2))
        return self.prior_logits(prior)+torch.cat(rows, -1)
