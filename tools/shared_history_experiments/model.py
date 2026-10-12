"""One height-preserving spatial encoder, source STWM and point CCR readouts."""
from dataclasses import asdict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from real_motion.local_st_world_model import SpatialTemporalBlock, FutureQueryBlock
from real_motion.causal_column_completion import ColumnConfig
from tools.joint_occ_experiments.model import HistoryEgoHead

PROTOCOL = 'p0_f9_shared_history_gt_soft_addonly_20percent_v1'


class LocalResidual(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.block = nn.Sequential(nn.GroupNorm(1, width),
            nn.Conv2d(width, width, 3, padding=1, groups=width),
            nn.Conv2d(width, width*2, 1), nn.SiLU(), nn.Conv2d(width*2, width, 1))

    def forward(self, x): return x+self.block(x)


class SharedEncoder(nn.Module):
    def __init__(self, z_bins=16, width=64):
        super().__init__()
        self.z_bins, self.width = z_bins, width
        self.semantic = nn.Embedding(19, 4)  # unknown distinct from free17
        # Even kernel aligns output centres with the 2x grid_sample lattice.
        self.stem = nn.Sequential(nn.Conv2d(z_bins*5, width, 4, stride=2, padding=1),
            nn.GroupNorm(1, width), nn.SiLU(), LocalResidual(width), LocalResidual(width))

    def forward(self, semantics, observed):
        if semantics.ndim != 4 or semantics.shape[0] != 4 or semantics.shape[-1] != self.z_bins:
            raise ValueError('encoder requires [4,X,Y,Z] with configured height bins')
        if observed.shape != semantics.shape: raise ValueError('semantic/valid grid differs')
        known = observed.bool() & (semantics >= 0) & (semantics < 18)
        labels = torch.where(known, semantics.long(), 18)
        x = torch.cat((self.semantic(labels), known[..., None].float()), -1)
        x = x.permute(0, 3, 4, 1, 2).flatten(1, 2)
        return self.stem(x)


def sample_points(field, grid):
    """[4,C,X,Y], [P,4,2] -> [P,4,C], differentiable into field."""
    n = grid.shape[0]
    if not n: return field.new_empty((0, 4, field.shape[1]))
    values = F.grid_sample(field.float(), grid.permute(1, 0, 2)[:, :, None].float(),
        mode='bilinear', padding_mode='zeros', align_corners=False)
    return values[..., 0].permute(2, 0, 1)


class SharedTransport(nn.Module):
    def __init__(self, config, width=64, z_bins=16):
        super().__init__()
        if config.history_frames != 4: raise ValueError('strict4 source model required')
        self.config = self.v17_config = config
        dim, hw = config.d_model, config.tube_hw//2
        self.history_encoder = SharedEncoder(z_bins, width)
        self.source_adapter = nn.Conv2d(width+1, dim, 1)
        self.motion_condition = nn.Sequential(nn.Linear(6, dim), nn.SiLU(), nn.LayerNorm(dim))
        self.class_embedding = nn.Embedding(18, dim)
        self.time_embedding = nn.Parameter(torch.empty(1, 4, dim, 1, 1))
        self.spatial_embedding = nn.Parameter(torch.empty(1, 1, dim, hw, hw))
        self.blocks = nn.ModuleList([SpatialTemporalBlock(dim, config.heads) for _ in range(config.blocks)])
        self.future_query = nn.Parameter(torch.empty(1, 6, dim))
        self.kta_future_proj = nn.Sequential(nn.Linear(2, dim), nn.SiLU(), nn.LayerNorm(dim))
        self.decoder = nn.ModuleList([FutureQueryBlock(dim, config.heads) for _ in range(config.decoder_blocks)])
        self.residual_head = nn.Linear(dim, 2)
        self.yaw_head = nn.Linear(dim, 1)
        self.existence_head = nn.Linear(dim, 1)
        for p in (self.time_embedding, self.spatial_embedding, self.future_query): nn.init.trunc_normal_(p, std=.02)
        for head in (self.residual_head, self.yaw_head, self.existence_head):
            nn.init.zeros_(head.weight); nn.init.zeros_(head.bias)

    def forward(self, field, reads, times, classes, kta):
        sources = reads['source_grid'].shape[0]
        dim, hw = self.config.d_model, self.config.tube_hw//2
        if not sources:
            empty = field.new_empty
            return dict(residual_xy_m=empty((0, 6, 2)), yaw_delta_rad=empty((0, 6)),
                existence_logits=empty((0, 6)), history_source_context=empty((0, dim)),
                future_transport_queries=empty((0, 6, dim)))
        if reads['source_grid'].shape != (sources, 4, hw, hw, 2):
            raise ValueError('source crop physical/sample contract differs')
        values = F.grid_sample(field.float(), reads['source_grid'].permute(1, 0, 2, 3, 4).reshape(4, sources*hw, hw, 2),
            padding_mode='zeros', align_corners=False).reshape(4, field.shape[1], sources, hw, hw).permute(2, 0, 1, 3, 4)
        valid = reads['source_valid'].float()
        center = reads['source_center']
        offset = (center-center[:, -1:])/20
        dt = (times[1:]-times[:-1]).clamp_min(.001)*1.5
        segment_valid = valid[:, 1:]*valid[:, :-1]
        segment = (center[:, 1:]-center[:, :-1])/dt[None, :, None]/20*segment_valid[..., None]
        velocity = torch.cat((segment[:, :1], segment), 1)
        condition = torch.cat((offset*valid[..., None], velocity*valid[..., None],
            valid[..., None], times[None, :, None].expand(sources, -1, -1)), -1)
        local = torch.cat((values*valid[:, :, None, None, None], reads['source_mask'][:, :, None]), 2)
        x = self.source_adapter(local.reshape(sources*4, -1, hw, hw)).reshape(sources, 4, dim, hw, hw)
        x = x+self.motion_condition(condition)[:, :, :, None, None]+self.class_embedding(classes)[:, None, :, None, None]
        x = x+self.time_embedding+self.spatial_embedding
        for block in self.blocks: x = block(x)
        context = x.permute(0, 1, 3, 4, 2).reshape(sources, -1, dim)
        q = self.future_query.expand(sources, -1, -1)+self.kta_future_proj(kta.to(x.dtype)/20)
        for block in self.decoder: q = block(q, context)
        return dict(residual_xy_m=self.residual_head(q), yaw_delta_rad=self.yaw_head(q)[..., 0],
            existence_logits=self.existence_head(q)[..., 0], history_source_context=context.mean(1),
            future_transport_queries=q)


class SharedRepair(nn.Module):
    def __init__(self, source_dim, width=64, z_bins=16):
        super().__init__()
        self.config = ColumnConfig(z_bins=z_bins)
        self.width, self.source_dim = width, source_dim
        self.point_adapter = nn.Sequential(nn.Linear(width+4, width), nn.SiLU(), nn.Linear(width, width))
        self.time_score = nn.Linear(width, 1)
        self.surface = nn.Sequential(nn.Linear(12, width), nn.SiLU(), nn.Linear(width, width))
        self.semantic = nn.Embedding(19, 8)
        self.role = nn.Embedding(2, width)
        self.class_projection = nn.Linear(8, width)
        self.source = nn.Linear(source_dim, width, bias=False)
        self.future = nn.Linear(source_dim, width, bias=False)
        self.context = nn.Linear(14+8+1, width)  # continuous context/phase, base, ADD legal
        self.readout = nn.Sequential(nn.SiLU(), nn.Linear(width, 32), nn.SiLU(), nn.Linear(32, 1))
        nn.init.normal_(self.readout[-1].weight, std=.01); nn.init.constant_(self.readout[-1].bias, -2.)

    def encode_history(self, field, reads, times, evidence):
        values = sample_points(field, reads['point_grid'])
        valid = reads['point_valid'].bool()
        n = len(values)
        facts = torch.stack((reads['point_z'], times[None].expand(n, -1),
            valid.float(), torch.as_tensor(evidence.presence, device=field.device).float()), -1)
        x = self.point_adapter(torch.cat((values, facts), -1))
        score = self.time_score(x)[..., 0].float().masked_fill(~valid, -1e4)
        weight = score.softmax(1)*valid.float()
        weight = weight/weight.sum(1, keepdim=True).clamp_min(1e-8)
        x = (x*weight[..., None]).sum(1)
        actor = torch.as_tensor(evidence.actor, device=field.device).long()
        classes = torch.as_tensor(evidence.classes, device=field.device).long()
        surface = torch.as_tensor(evidence.features[:, 27:], device=field.device)
        return x+self.surface(surface)*(actor < 0)[:, None]+self.role((actor >= 0).long())+self.class_projection(self.semantic(classes))

    def forward(self, encoded, output, actor, context, base, legal):
        dynamic = actor >= 0
        x = encoded[:, None]
        if len(output['history_source_context']):
            x = x+self.source(output['history_source_context'])[actor.clamp_min(0)][:, None]*dynamic[:, None, None]
            x = x+self.future(output['future_transport_queries'])[actor.clamp_min(0)]*dynamic[:, None, None]
        condition = torch.cat((context, self.semantic(base.long()), legal[..., :1].to(context.dtype)), -1)
        add = self.readout(x+self.context(condition)).float().sigmoid()
        return torch.cat((add, torch.zeros_like(add)), -1)  # SAME REMOVE-off train/inference


class SharedJoint(nn.Module):
    def __init__(self, motion_config, *, width=64, z_bins=16):
        super().__init__()
        self.transport = SharedTransport(motion_config, width, z_bins)
        self.columns = SharedRepair(motion_config.d_model, width, z_bins)
        self.ego = HistoryEgoHead()  # unchanged, no scene-head upgrade

    def configs(self):
        return dict(motion=asdict(self.transport.config), shared_history=dict(width=self.columns.width,
            z_bins=self.transport.history_encoder.z_bins,
            stride=2, source_patch_m=16., point_chunk=8192, legacy_27=False, surface12=True),
            repair=dict(width=self.columns.width, action='soft_ADD_only_train_and_eval'),
            ego=dict(type='unchanged_history_control', parameters=8710))


def encode_window(model, raw, history, ego_history, device):
    semantics = torch.as_tensor(np.asarray(raw['history_occ']), device=device)
    observed = torch.as_tensor(np.asarray(raw['history_observed']), device=device)
    field = model.transport.history_encoder(semantics, observed)
    reads = {k: torch.as_tensor(v, device=device) for k, v in history[0]['shared_reads'].items()}
    times = torch.as_tensor(ego_history[:, -1], device=device)
    return field, reads, times
