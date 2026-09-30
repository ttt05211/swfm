"""Small V18-style history ST encoder -> independent point-bundle queries.

Direct and feedback-refined decoders share the architecture/parameter count;
neither needs an AE, discrete codebook, future semantic input or source shape.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from .local_st_world_model import SpatialTemporalBlock, FutureQueryBlock
from .sparse_emergence import EmergenceConfig


class SparseEmergenceDecoder(nn.Module):
    def __init__(self, config=EmergenceConfig()):
        super().__init__(); config.validate(); self.config = config
        d, c = config.width, config.context_cells//2
        self.semantic_embedding = nn.Embedding(19,8)
        self.spatial_stem = nn.Sequential(nn.Conv2d(18,d,3,stride=2,padding=1), nn.GroupNorm(1,d), nn.GELU())
        self.space = nn.Parameter(torch.randn(1,1,d,c,c)*.02)
        self.time = nn.Parameter(torch.randn(1,6,d,1,1)*.02)
        self.blocks = nn.ModuleList([SpatialTemporalBlock(d,config.heads,mlp_ratio=2,kernel_size=3)])
        self.query_encoder = nn.Linear(24,d)
        self.query = nn.Parameter(torch.randn(1,config.queries,d)*.02)
        self.decoder = nn.ModuleList([FutureQueryBlock(d,config.heads,mlp_ratio=2) for _ in range(config.layers)])
        self.geometry_feedback = nn.Linear(3,d)
        self.xyz_head = nn.Linear(d,config.points_per_query*3)
        self.semantic_head = nn.Linear(d,config.points_per_query*17)
        self.presence_head = nn.Linear(d,1)
        anchor = torch.rand(config.queries,config.points_per_query,3)*1.2-.6
        anchor[...,:2] *= .3
        anchor[...,:2] += torch.tensor([[-.5,-.5],[-.5,.5],[.5,-.5],[.5,.5]])[:,None]
        self.anchor_logits = nn.Parameter(torch.atanh(anchor.clamp(-.9,.9)))
        nn.init.zeros_(self.xyz_head.weight); nn.init.zeros_(self.xyz_head.bias)
        nn.init.zeros_(self.presence_head.weight); nn.init.constant_(self.presence_head.bias, math.log(.1/.9))

    def forward(self, history, query):
        cfg = self.config; n = len(history)
        if history.ndim != 5 or tuple(history.shape[1:4]) != (6,cfg.context_cells,cfg.context_cells) or query.shape != (n,24):
            raise ValueError("emergence history/query shape mismatch")
        if torch.any(history > 18) or torch.any(history < 0): raise ValueError("invalid causal history labels")
        if not n:
            return {"xyz": query.new_empty((0,cfg.queries*cfg.points_per_query,3)),
                    "semantic_logits": query.new_empty((0,cfg.queries*cfg.points_per_query,17)),
                    "presence_logits": query.new_empty((0,)), "stages": []}
        z = history.shape[-1]; labels = history.long()
        e = self.semantic_embedding(labels)
        occupied = (labels < 17).to(e.dtype)
        heights = ((torch.arange(z,device=e.device,dtype=e.dtype)+.5)/z*2-1).view(1,1,1,1,z,1)
        columns = torch.cat((e.mean(-2), (e*heights*occupied[...,None]).mean(-2),
                             occupied.mean(-1,keepdim=True), (labels != 18).to(e.dtype).mean(-1,keepdim=True)), dim=-1)
        x = self.spatial_stem(columns.permute(0,1,4,2,3).flatten(0,1))
        x = x.reshape(n,6,cfg.width,*x.shape[-2:]) + self.time + self.space
        for block in self.blocks: x = block(x)
        context = x.permute(0,1,3,4,2).flatten(1,3)
        q = self.query_encoder(query.float()).unsqueeze(1)+self.query
        pos = self.anchor_logits.unsqueeze(0).expand(n,-1,-1,-1)
        stages = []
        for i, block in enumerate(self.decoder):
            q = block(q,context)
            if cfg.refinement or i == len(self.decoder)-1:
                pos = pos + self.xyz_head(q).float().reshape(n,cfg.queries,cfg.points_per_query,3)
                scale = torch.tensor((1-1/cfg.patch_cells,1-1/cfg.patch_cells,1-1/z),device=q.device)
                xyz = torch.tanh(pos)*scale
                stage = {"xyz": xyz.flatten(1,2),
                         "semantic_logits": self.semantic_head(q).float().reshape(n,-1,17),
                         "presence_logits": self.presence_head(q.mean(1)).float().squeeze(-1)}
                stages.append(stage)
                if cfg.refinement: q = q+self.geometry_feedback(xyz.mean(2))
        return {**stages[-1], "stages": stages}


def set_loss(output, target_xyz, target_labels, target_mask, sampling_weight, extent_voxels, class_weight=None):
    """THREE groups: foreground Chamfer + positive semantics + patch existence.

    No dense free class competes with shape/semantic learning. Empty patches
    affect ONLY existence; inverse sampling weights preserve its natural
    prior. Nearest semantic assignment is detached, geometry is differentiable.
    Refinement stages share the same three objectives (averaged, not six knobs).
    """
    n = len(target_xyz)
    if (target_xyz.ndim != 3 or target_xyz.shape[-1] != 3 or target_labels.shape != target_mask.shape
            or target_labels.shape != target_xyz.shape[:2] or sampling_weight.shape != (n,)
            or torch.any(sampling_weight <= 0)):
        raise ValueError("target/importance weight contract mismatch")
    positive = target_mask.any(1)
    scale = torch.as_tensor(extent_voxels,device=target_xyz.device,dtype=torch.float32)/2
    losses, stats = [], []
    for stage in output["stages"]:
        p = stage["xyz"].float()
        presence = (F.binary_cross_entropy_with_logits(stage["presence_logits"], positive.float(), reduction="none")
                    * sampling_weight).sum()/sampling_weight.sum()
        geometry = p.sum()*0; semantic = stage["semantic_logits"].sum()*0
        if positive.any():
            valid = target_mask[positive]
            dist = ((p[positive,:,None]-target_xyz[positive,None]).abs()*scale).sum(-1)
            dist = dist.masked_fill(~valid[:,None],float("inf"))
            pred_min, nn_index = dist.min(-1)
            gt_min = dist.min(1).values.masked_fill(~valid,0)
            geometry = (pred_min.mean(1)+gt_min.sum(1)/valid.sum(1)).mean()
            semantic_target = target_labels[positive].gather(1,nn_index.detach())
            semantic = F.cross_entropy(stage["semantic_logits"][positive].reshape(-1,17),
                                       semantic_target.reshape(-1), weight=class_weight)
        losses.append(geometry+semantic+presence)
        stats.append(torch.stack((geometry.detach(),semantic.detach(),presence.detach())))
    if not losses: raise ValueError("cannot train an empty output")
    details = torch.stack(stats).mean(0)
    return torch.stack(losses).mean(), {"geometry": details[0], "semantic": details[1], "presence": details[2],
                                      "positive_patches": positive.sum().detach()}
