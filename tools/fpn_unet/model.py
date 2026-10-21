"""Learned 2D FPN history encoder + full-grid, height-preserving 3D U-Net."""
from dataclasses import asdict
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from real_motion.causal_column_completion import ColumnConfig
from tools.source_scene_stage1.model import ReaderConfig
from tools.source_scene_ego_conditioned.model import EgoTransport

VARIANT = 'history_FPN_100_50_25_plain_UNet3D_ego_query_v1'


class Residual2D(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.path = nn.Sequential(nn.Conv2d(width, width, 3, padding=1, bias=False),
            nn.GroupNorm(4, width), nn.ReLU(), nn.Conv2d(width, width, 3, padding=1, bias=False),
            nn.GroupNorm(4, width))

    def forward(self, x): return F.relu(x+self.path(x))


class FPNEncoder(nn.Module):
    def __init__(self, z_bins=16, width=64):
        super().__init__()
        if width % 8: raise ValueError('FPN width must be divisible by eight')
        self.z_bins, self.width = z_bins, width
        self.semantic = nn.Embedding(19, 4)
        channels = (width, width*3//2, width*2)
        self.stem = nn.Sequential(nn.Conv2d(z_bins*5, width, 4, stride=2, padding=1, bias=False),
            nn.GroupNorm(4, width), nn.ReLU(), Residual2D(width), Residual2D(width))
        # Even 2x2/stride2 kernels preserve the align_corners=False cell-centre
        # lattice at every scale: all levels cover the SAME physical bounds.
        self.down = nn.ModuleList([nn.Sequential(nn.Conv2d(a, b, 2, stride=2, bias=False),
            nn.GroupNorm(4, b), nn.ReLU(), Residual2D(b), Residual2D(b))
            for a, b in zip(channels, channels[1:])])
        self.lateral = nn.ModuleList([nn.Conv2d(c, width, 1) for c in channels])
        self.smooth = nn.ModuleList([nn.Conv2d(width, width, 3, padding=1) for _ in channels])

    def forward(self, semantics, observed):
        if (semantics.ndim != 4 or semantics.shape[0] != 4 or semantics.shape[-1] != self.z_bins or
                semantics.shape != observed.shape or semantics.shape[1] % 8 or semantics.shape[2] % 8):
            raise ValueError('FOUR [X,Y,Z] histories; XY divisible by eight required')
        known = observed.bool() & (semantics >= 0) & (semantics < 18)
        labels = torch.where(known, semantics.long(), 18)
        x = torch.cat((self.semantic(labels), known[..., None].float()), -1)
        x = x.permute(0, 3, 4, 1, 2).flatten(1, 2)
        levels = [self.stem(x)]
        for down in self.down: levels.append(down(levels[-1]))
        fused = [self.lateral[i](v) for i, v in enumerate(levels)]
        for i in (1, 0):
            fused[i] = fused[i]+F.interpolate(fused[i+1], size=fused[i].shape[-2:], mode='nearest')
        return tuple(smooth(v) for smooth, v in zip(self.smooth, fused))


class FPNTransport(EgoTransport):
    def __init__(self, config, width=64, z_bins=16, reader_config=ReaderConfig(global_side=25)):
        super().__init__(config, width, z_bins, reader_config)
        self.history_encoder = FPNEncoder(z_bins, width)

    def forward(self, fields, reads, times, classes, kta, ego_features=None):
        if len(fields) != 3: raise ValueError('three LEARNED FPN fields required')
        field = fields[0]
        valid = reads['reference_valid'].bool(); sources, frames = valid.shape
        if frames != 4 or kta is None or kta.shape != (sources, 6, 2): raise ValueError('four tokens and SIX KTA required')
        if ego_features is None or ego_features.shape != (6, 4) or not bool(torch.isfinite(ego_features).all()):
            raise ValueError('explicit SIX relative future ego conditions required')
        dim = self.config.d_model
        if not sources:
            return dict(residual_xy_m=field.new_empty((0, 6, 2)), yaw_delta_rad=field.new_empty((0, 6)),
                existence_logits=field.new_empty((0, 6)), history_source_context=field.new_empty((0, dim)),
                future_transport_queries=field.new_empty((0, 6, dim)), history_yaw_to_t0_rad=field.new_empty((0, 4)))
        if not bool(valid[:, -1].all()): raise ValueError('real t0 reference required')
        pyramid = [self.feature_projection(v) for v in fields]
        native = pyramid[0]
        grid = reads['reference_grid'].permute(1, 0, 2).float()
        seed = F.grid_sample(native.float(), grid[:, :, None], align_corners=False,
            padding_mode='zeros')[..., 0].transpose(1, 2)
        pos = torch.cat((reads['reference_xy_t0'].float()/40, times[None, :, None].expand(sources, -1, -1)), -1)
        pos = torch.where(valid[..., None], pos, torch.zeros_like(pos))
        global_condition = self.scene_global(pyramid[-1].mean((-2, -1)))[:, None]
        context = ((seed+self.position(pos).permute(1, 0, 2)+global_condition)*valid.T[..., None]).transpose(0, 1)
        for block in self.encoder_blocks: context = block(context, pyramid, reads, valid)
        history_yaw = self.history_yaw(context-context[:, -1:])[..., 0]*valid
        q = self.initial_queries(kta, ego_features, context.dtype)
        for block in self.decoder: q = block(q, context, valid)
        return dict(residual_xy_m=self.residual_head(q), yaw_delta_rad=self.yaw_head(q)[..., 0],
            existence_logits=q.new_zeros((sources, 6)), history_source_context=(context*valid[..., None]).sum(1)/valid.sum(1, keepdim=True),
            future_transport_queries=q, history_yaw_to_t0_rad=history_yaw)


def conv3d(cin, cout):
    return nn.Sequential(nn.Conv3d(cin, cout, 3, padding=1, bias=False), nn.GroupNorm(4, cout), nn.ReLU(),
        nn.Conv3d(cout, cout, 3, padding=1, bias=False), nn.GroupNorm(4, cout), nn.ReLU())


def resize3d(x, shape):
    # Resize only changes XY; height is always retained.
    return F.interpolate(x.float(), size=shape, mode='trilinear', align_corners=False).to(x.dtype)


class UNetRefiner(nn.Module):
    def __init__(self, feature_width=64, memory_width=16, z_bins=16, epsilon=1e-4, horizon_chunk=2):
        super().__init__()
        self.config = ColumnConfig(z_bins=z_bins)
        self.z_bins, self.memory_width, self.epsilon = z_bins, memory_width, epsilon
        self.horizon_chunk = horizon_chunk
        self.lift = nn.Conv2d(feature_width, memory_width*z_bins, 1)
        self.importance = nn.Linear(memory_width, 1)
        self.empty = nn.Parameter(torch.zeros(memory_width))
        self.fine = conv3d(19, 8)
        self.pool = nn.AvgPool3d((2, 2, 1))  # Spatial-only: channels stay E0=8.
        self.middle = conv3d(8+memory_width, 16)  # 8+16=24 -> 16 -> 16.
        self.low = conv3d(16, 32)
        self.decode_middle = conv3d(32+16, 16)
        self.decode_fine = conv3d(16+8, 8)
        self.head = nn.Conv3d(8, 18, 1)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def lift_history(self, field):
        frames, _, sx, sy = field.shape
        if frames != 4: raise ValueError('FOUR shared P2 fields required')
        return self.lift(field).reshape(frames, self.memory_width, self.z_bins, sx, sy).permute(0, 1, 3, 4, 2)

    def _residual(self, memory, probability, clock):
        with torch.autocast(device_type=memory.device.type, dtype=torch.bfloat16, enabled=memory.device.type == 'cuda'):
            fine = self.fine(torch.cat((probability, clock.expand(-1, 1, *probability.shape[2:])), 1))
            middle = self.middle(torch.cat((self.pool(fine), memory), 1))
            low = self.low(self.pool(middle))
            value = self.decode_middle(torch.cat((resize3d(low, middle.shape[2:]), middle), 1))
            value = self.decode_fine(torch.cat((resize3d(value, fine.shape[2:]), fine), 1))
            return self.head(value).float().flatten(2)

    def prior_logits(self, prior):
        p = prior.float()
        return ((p+self.epsilon)/(p.sum(1, keepdim=True)+18*self.epsilon)).log()

    def forward(self, memory, prior, horizons, shape):
        horizons = tuple(horizons); shape = tuple(map(int, shape)); count = len(horizons)
        if (not count or any(h not in range(6) for h in horizons) or len(shape) != 3 or
                shape[0] % 4 or shape[1] % 4 or shape[2] != self.z_bins or
                prior.shape != (count, 18, math.prod(shape)) or
                memory.shape != (count, self.memory_width, shape[0]//2, shape[1]//2, shape[2])):
            raise ValueError('full-grid prior / half-XY aligned memory / SIX-horizon clock mismatch')
        probability = prior.reshape(count, 18, *shape)
        clock = prior.new_tensor([.5*(h+1)/3 for h in horizons]).reshape(count, 1, 1, 1, 1)
        rows = []
        for start in range(0, count, self.horizon_chunk):
            args = (memory[start:start+self.horizon_chunk], probability[start:start+self.horizon_chunk], clock[start:start+self.horizon_chunk])
            # Exact same U-Net in train/eval. Recompute training activations,
            # not features/geometry/targets; all six horizons remain supervised.
            value = (checkpoint(self._residual, *args, use_reentrant=False)
                if self.training and torch.is_grad_enabled() else self._residual(*args))
            rows.append(value)
        return self.prior_logits(prior)+torch.cat(rows)


class FPNUNetBody(nn.Module):
    def __init__(self, config, *, feature_width=64, z_bins=16, memory_width=16,
                 reader_config=ReaderConfig(global_side=25)):
        super().__init__()
        self.transport = FPNTransport(config, feature_width, z_bins, reader_config)
        self.columns = UNetRefiner(feature_width, memory_width, z_bins)
        self.ego = nn.Identity()  # GT-conditioned only; NO predicted ego head.

    def configs(self):
        return dict(variant=VARIANT, motion=asdict(self.transport.config), reader=asdict(self.transport.reader_config),
            source_policy='shared_pinned', registration='none', validity='track_valid', kta_conditioning=True,
            shared_history=dict(type='residual_backbone_FPN', width=self.transport.history_encoder.width,
                z_bins=self.columns.z_bins, strides=[2, 4, 8],
                stage_widths=[self.transport.history_encoder.width*k//2 for k in (2, 3, 4)],
                levels='P2_P3_P4; no adaptive pooling; P2 shared with memory', top_down='nearest_plus_lateral'),
            encoder=dict(blocks=2, sequence='PreNorm_scene_temporal_FFN'),
            future_query=dict(type='six_learned_plus_KTA_plus_relative_GT_ego', decoder_order='cross_self_FFN'),
            memory=dict(content='history_only_P2_lift', importance='history_features_and_coverage', history_yaw='learned_t0_gauge_zero'),
            repair=dict(type='plain_UNet3D_residual_logits', channels=[8, 16, 32], memory_width=self.columns.memory_width,
                middle_input=8+self.columns.memory_width, z_bins=self.columns.z_bins, downsample='XY_only_avgpool',
                time='broadcast_input_channel', epsilon=self.columns.epsilon, horizon_chunk=self.columns.horizon_chunk,
                training_checkpoint=True, phase=False, point_MLP=False, AdaGN=False),
            objective=dict(ce=1., lovasz=1., motion=0., existence=0., ego=0.))


def make_model(config, **kwargs): return FPNUNetBody(config, **kwargs)


def checkpoint_model(saved, device='cpu'):
    from .run import PROTOCOL
    from real_motion.local_st_world_model_v17 import config_from_mapping_v17
    spec = saved['contract']['model']
    if saved.get('protocol') != PROTOCOL or spec.get('variant') != VARIANT:
        raise RuntimeError('only this combined FPN/UNet checkpoint is accepted')
    model = make_model(config_from_mapping_v17(spec['motion']), feature_width=spec['shared_history']['width'],
        z_bins=spec['shared_history']['z_bins'], memory_width=spec['repair']['memory_width'], reader_config=ReaderConfig(**spec['reader']))
    if model.configs() != spec: raise RuntimeError('checkpoint architecture differs')
    model.load_state_dict(saved['models']['gt'], strict=True)
    return model.to(device).eval()
