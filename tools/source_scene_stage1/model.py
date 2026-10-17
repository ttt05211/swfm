"""Reference-guided local/global reading with explicit spacetime coordinates.

Native history fields are sampled directly. Coordinate transforms, NOT a second
resampled feature volume, put reference/scene positions in the t0 ego frame.
Registration validity, KTA conditioning and downstream memory stay unchanged.
"""
from dataclasses import asdict, dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from tools.shared_history_experiments.model import SharedEncoder
from tools.shared_history_experiments.full_gt.model import GTBody
from tools.shared_history_experiments.refine.model import SemanticRefiner
from real_motion.local_st_world_model import FutureQueryBlock


@dataclass(frozen=True)
class ReaderConfig:
    layers: int = 2
    temporal_layers: int = 2
    points: int = 4
    global_side: int = 8
    bands: int = 8


class PositionTime(nn.Module):
    """Standard Fourier XY/time encoding, no derived velocity/class state."""
    def __init__(self, dim, bands):
        super().__init__()
        self.register_buffer('frequencies', math.pi*2.**torch.arange(bands), persistent=False)
        self.projection = nn.Linear(6*bands, dim)

    def forward(self, x):
        phase = x.float()[..., None]*self.frequencies
        return self.projection(torch.cat((phase.sin(), phase.cos()), -1).flatten(-2))


class SceneReader(nn.Module):
    def __init__(self, dim, heads, points):
        super().__init__()
        self.heads, self.points, self.dim = heads, points, dim
        self.norm = nn.LayerNorm(dim)
        self.offsets = nn.Linear(dim, heads*3*points*2)
        self.weights = nn.Linear(dim, heads*3*points)
        self.local_output = nn.Linear(dim, dim)
        self.global_norm = nn.LayerNorm(dim)
        self.global_attention = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=0.)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim))
        nn.init.zeros_(self.offsets.weight)
        angles = torch.arange(heads)*2*math.pi/heads
        directions = torch.stack((angles.cos(), angles.sin()), -1)
        directions /= directions.abs().amax(-1, keepdim=True).clamp_min(1e-6)
        radial = directions[:, None, None]*torch.arange(1, points+1)[None, None, :, None]
        with torch.no_grad(): self.offsets.bias.copy_(radial.expand(-1, 3, -1, -1).reshape(-1))
        nn.init.zeros_(self.weights.weight); nn.init.zeros_(self.weights.bias)

    def forward(self, q, pyramid, global_content, global_position, reads, valid):
        # q [frame, source, channel]. Sampling offsets live in t0 XY metres;
        # each frame's Jacobian projects them into its native grid_sample axes.
        frames, sources, dim = q.shape
        z = self.norm(q)
        offsets = self.offsets(z).reshape(frames, sources, self.heads, 3, self.points, 2)
        scores = self.weights(z).reshape(frames, sources, self.heads, 3*self.points)
        reference = reads['reference_grid'].permute(1, 0, 2).float()
        transform = reads['t0_xy_to_native_grid'].float()
        extent = reads['scene_bounds'][2:]-reads['scene_bounds'][:2]
        values, masks = [], []
        for level, field in enumerate(pyramid):
            # One level pixel sets the initial scale; learned offsets are free
            # to move beyond it. Levels use the SAME metric scene bounds.
            step = extent/field.new_tensor(field.shape[-2:])
            delta = offsets[:, :, :, level].float()*step
            normalized = torch.einsum('fnkpd,fgd->fnkpg', delta, transform)
            grid = reference[:, :, None, None]+normalized
            inside = (grid.abs() <= 1).all(-1) & valid.T[:, :, None, None]
            sampled = F.grid_sample(field.float(), grid.reshape(frames, sources*self.heads*self.points, 1, 2),
                                    align_corners=False, padding_mode='zeros')
            sampled = sampled[..., 0].transpose(1, 2).reshape(frames, sources, self.heads, self.points,
                                                            self.heads, dim//self.heads)
            index = torch.arange(self.heads, device=q.device)
            sampled = sampled[:, :, index, :, index, :].permute(1, 2, 0, 3, 4)
            # Advanced indexing above yields [head,frame,source,point,head_dim].
            values.append(sampled); masks.append(inside)
        values = torch.stack(values, 3).flatten(3, 4)
        mask = torch.stack(masks, 3).flatten(3, 4)
        attention = scores.float().masked_fill(~mask, -1e4).softmax(-1)*mask
        attention = attention/attention.sum(-1, keepdim=True).clamp_min(1e-8)
        local = (values*attention[..., None]).sum(-2).flatten(-2)
        q = q+self.local_output(local)
        z = self.global_norm(q)
        context, _ = self.global_attention(z, global_content+global_position, global_content, need_weights=False)
        q = q+context
        q = q+self.ffn(self.ffn_norm(q))
        return q*valid.T[..., None]


class MaskedFutureBlock(FutureQueryBlock):
    def forward(self, q, context, valid):
        z = self.self_norm(q)
        q = q+self.self_attn(z, z, z, need_weights=False)[0]
        z = self.cross_ctx_norm(context)
        q = q+self.cross_attn(self.cross_q_norm(q), z, z, key_padding_mask=~valid, need_weights=False)[0]
        return q+self.ffn(self.ffn_norm(q))


class SceneTransport(nn.Module):
    def __init__(self, config, width=64, z_bins=16, reader_config=ReaderConfig()):
        super().__init__()
        if config.history_frames != 4 or config.d_model % config.heads: raise ValueError('four histories/head dimensions')
        self.config = self.v17_config = config
        self.reader_config = reader_config
        dim = config.d_model
        self.history_encoder = SharedEncoder(z_bins, width)
        self.feature_projection = nn.Conv2d(width, dim, 1)
        self.position = PositionTime(dim, reader_config.bands)
        self.readers = nn.ModuleList([SceneReader(dim, config.heads, reader_config.points)
                                      for _ in range(reader_config.layers)])
        self.temporal = nn.ModuleList([nn.TransformerEncoderLayer(dim, config.heads, dim*4,
            dropout=0., activation='gelu', batch_first=True, norm_first=True)
                                      for _ in range(reader_config.temporal_layers)])
        self.future_query = nn.Parameter(torch.empty(1, 6, dim))
        nn.init.trunc_normal_(self.future_query, std=.02)
        self.kta_future_proj = nn.Sequential(nn.Linear(2, dim), nn.SiLU(), nn.LayerNorm(dim))
        self.decoder = nn.ModuleList([MaskedFutureBlock(dim, config.heads) for _ in range(config.decoder_blocks)])
        self.residual_head, self.yaw_head = nn.Linear(dim, 2), nn.Linear(dim, 1)
        for head in (self.residual_head, self.yaw_head):
            nn.init.zeros_(head.weight); nn.init.zeros_(head.bias)

    def forward(self, field, reads, times, classes, kta):
        # classes is a compatibility argument ONLY. Renderer still uses actual
        # source semantics; this learned motion model does not consume classes.
        valid = reads['reference_valid'].bool()
        sources, frames = valid.shape
        if frames != 4 or kta.shape != (sources, 6, 2): raise ValueError('source/horizon contract differs')
        dim = self.config.d_model
        if not sources:
            return dict(residual_xy_m=field.new_empty((0, 6, 2)), yaw_delta_rad=field.new_empty((0, 6)),
                existence_logits=field.new_empty((0, 6)), history_source_context=field.new_empty((0, dim)),
                future_transport_queries=field.new_empty((0, 6, dim)))
        if not bool(valid[:, -1].all()): raise ValueError('every source requires its real t0 reference')
        native = self.feature_projection(field)
        sx, sy = native.shape[-2:]
        sizes = ((sx, sy), (max(1, sx//4), max(1, sy//4)),
                 (min(self.reader_config.global_side, sx), min(self.reader_config.global_side, sy)))
        pyramid = [native if size == (sx, sy) else F.adaptive_avg_pool2d(native, size) for size in sizes]
        grid = reads['reference_grid'].permute(1, 0, 2).float()
        seed = F.grid_sample(native.float(), grid[:, :, None], align_corners=False,
                             padding_mode='zeros')[..., 0].transpose(1, 2)
        # Validity-masked coordinates cannot turn missing-history placeholders
        # into fictitious positions, but valid geometric time is always retained.
        pos = torch.cat((reads['reference_xy_t0'].float()/40,
                         times[None, :, None].expand(sources, -1, -1)), -1)
        pos = torch.where(valid[..., None], pos, torch.zeros_like(pos))
        q = (seed+self.position(pos).permute(1, 0, 2))*valid.T[..., None]
        coarse = pyramid[-1]; gx, gy = coarse.shape[-2:]
        bounds = reads['scene_bounds'].float()
        x = bounds[0]+(torch.arange(gx, device=field.device)+.5)/gx*(bounds[2]-bounds[0])
        y = bounds[1]+(torch.arange(gy, device=field.device)+.5)/gy*(bounds[3]-bounds[1])
        xy = torch.stack(torch.meshgrid(x, y, indexing='ij'), -1).reshape(-1, 2)
        xyz = torch.cat((xy, torch.zeros_like(xy[:, :1])), -1)
        matrix = reads['history_to_t0'].float()
        aligned = torch.einsum('fij,pj->fpi', matrix[:, :3, :3], xyz)+matrix[:, None, :3, 3]
        scene_pos = self.position(torch.cat((aligned[..., :2]/40,
            times[:, None, None].expand(-1, len(xy), -1)), -1))
        content = coarse.flatten(2).transpose(1, 2)
        for reader in self.readers: q = reader(q, pyramid, content, scene_pos, reads, valid)
        context = q.transpose(0, 1)
        for block in self.temporal:
            context = block(context, src_key_padding_mask=~valid)*valid[..., None]
        q = self.future_query.expand(sources, -1, -1)+self.kta_future_proj(kta.to(context.dtype)/20)
        for block in self.decoder: q = block(q, context, valid)
        return dict(residual_xy_m=self.residual_head(q), yaw_delta_rad=self.yaw_head(q)[..., 0],
            existence_logits=q.new_zeros((sources, 6)),
            history_source_context=(context*valid[..., None]).sum(1)/valid.sum(1, keepdim=True),
            future_transport_queries=q)


class SceneBody(nn.Module):
    def __init__(self, config, *, feature_width=64, z_bins=16, memory_width=16, width=32,
                 chunk_size=65536, reader_config=ReaderConfig()):
        super().__init__()
        self.transport = SceneTransport(config, feature_width, z_bins, reader_config)
        self.columns = SemanticRefiner(config.d_model, feature_width, z_bins, memory_width, width, chunk_size)
        self.ego = nn.Identity()

    def configs(self):
        return dict(motion=asdict(self.transport.config), reader=asdict(self.transport.reader_config),
            variant='source_scene_stage1', position_time=True, source_class_feature=False,
            source_mask_feature=False, kta_conditioning=True, registration='unchanged',
            shared_history=dict(width=self.transport.history_encoder.width, z_bins=self.columns.z_bins),
            repair=dict(type='unchanged_semantic_residual', memory_width=self.columns.memory_width,
                        width=self.columns.width, chunk_size=self.columns.chunk_size, epsilon=self.columns.epsilon),
            objective=dict(ce=1., lovasz=1., motion=0., existence=0., ego=0.))


def make_model(arm, config, **kwargs):
    """Identical fresh shared weights; NO trained checkpoint is imported."""
    if arm not in ('baseline', 'reader'): raise ValueError('unknown arm')
    baseline = GTBody(config, **kwargs)
    if arm == 'baseline': return baseline
    candidate = SceneBody(config, **kwargs)
    # Share all matching fresh modules, including future heads/decoder/KTA.
    common = baseline.state_dict(); state = candidate.state_dict()
    candidate.load_state_dict({k: common[k] if k in common and common[k].shape == v.shape else v
                               for k, v in state.items()}, strict=True)
    return candidate


def checkpoint_model(saved, device='cpu'):
    """Explicit new-checkpoint loader; never silently use an old model factory."""
    from real_motion.local_st_world_model_v17 import config_from_mapping_v17
    from .run import PROTOCOL
    c = saved['contract']
    if saved.get('protocol') != PROTOCOL or c.get('source_policy') != 'shared_pinned' or c.get('arm') not in ('baseline', 'reader'):
        raise RuntimeError('not a stage1 shared-pinned checkpoint')
    spec = c['model']; shared, repair = spec['shared_history'], spec['repair']
    cfg = config_from_mapping_v17(spec['motion'])
    kwargs = dict(feature_width=shared['width'], z_bins=shared['z_bins'], memory_width=repair['memory_width'],
                  width=repair['width'], chunk_size=repair['chunk_size'])
    model = (GTBody(cfg, **kwargs) if c['arm'] == 'baseline' else
             SceneBody(cfg, reader_config=ReaderConfig(**spec['reader']), **kwargs))
    model.columns.epsilon = repair['epsilon']
    model.load_state_dict(saved['models']['gt'], strict=True)
    return model.to(device).eval()
