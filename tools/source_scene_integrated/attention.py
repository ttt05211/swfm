"""One multi-scale scene cross-attention, then source-local temporal attention."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class MultiScaleSceneAttention(nn.Module):
    def __init__(self, dim, heads, points):
        super().__init__()
        self.heads, self.points, self.dim = heads, points, dim
        self.offsets = nn.Linear(dim, heads*3*points*2)
        self.weights = nn.Linear(dim, heads*3*points)
        self.output = nn.Linear(dim, dim)
        nn.init.zeros_(self.offsets.weight)
        angles = torch.arange(heads)*2*math.pi/heads
        directions = torch.stack((angles.cos(), angles.sin()), -1)
        directions /= directions.abs().amax(-1, keepdim=True).clamp_min(1e-6)
        radial = directions[:, None, None]*torch.arange(1, points+1)[None, None, :, None]
        with torch.no_grad(): self.offsets.bias.copy_(radial.expand(-1, 3, -1, -1).reshape(-1))
        nn.init.zeros_(self.weights.weight); nn.init.zeros_(self.weights.bias)

    def forward(self, q, pyramid, reads, valid):
        # q is ALREADY pre-normalized, [frame, source, channel].
        frames, sources, dim = q.shape
        offsets = self.offsets(q).reshape(frames, sources, self.heads, 3, self.points, 2)
        scores = self.weights(q).reshape(frames, sources, self.heads, 3*self.points)
        reference = reads['reference_grid'].permute(1, 0, 2).float()
        transform = reads['t0_xy_to_native_grid'].float()
        extent = reads['scene_bounds'][2:]-reads['scene_bounds'][:2]
        values, masks = [], []
        for level, field in enumerate(pyramid):
            step = extent/field.new_tensor(field.shape[-2:])
            delta = offsets[:, :, :, level].float()*step
            grid = reference[:, :, None, None]+torch.einsum('fnkpd,fgd->fnkpg', delta, transform)
            inside = (grid.abs() <= 1).all(-1) & valid.T[:, :, None, None]
            sampled = F.grid_sample(field.float(), grid.reshape(frames, sources*self.heads*self.points, 1, 2),
                                    align_corners=False, padding_mode='zeros')
            sampled = sampled[..., 0].transpose(1, 2).reshape(frames, sources, self.heads, self.points,
                                                            self.heads, dim//self.heads)
            index = torch.arange(self.heads, device=q.device)
            sampled = sampled[:, :, index, :, index, :].permute(1, 2, 0, 3, 4)
            values.append(sampled); masks.append(inside)
        values = torch.stack(values, 3).flatten(3, 4)
        mask = torch.stack(masks, 3).flatten(3, 4)
        attention = scores.float().masked_fill(~mask, -1e4).softmax(-1)*mask
        attention = attention/attention.sum(-1, keepdim=True).clamp_min(1e-8)
        return self.output((values*attention[..., None]).sum(-2).flatten(-2))


class SceneTemporalBlock(nn.Module):
    def __init__(self, dim, heads, points):
        super().__init__()
        self.scene_norm = nn.LayerNorm(dim)
        self.scene = MultiScaleSceneAttention(dim, heads, points)
        self.temporal_norm = nn.LayerNorm(dim)
        self.temporal_attention = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=0.)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim))

    def forward(self, tokens, pyramid, reads, valid):
        # [source, FOUR history tokens, dim]; never attend across sources.
        mask = valid[..., None]
        scene = self.scene(self.scene_norm(tokens).transpose(0, 1), pyramid, reads, valid).transpose(0, 1)
        tokens = (tokens+scene)*mask
        z = self.temporal_norm(tokens)
        tokens = (tokens+self.temporal_attention(z, z, z, key_padding_mask=~valid, need_weights=False)[0])*mask
        return (tokens+self.ffn(self.ffn_norm(tokens)))*mask
