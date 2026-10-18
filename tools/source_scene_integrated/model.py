"""Two Scene-Temporal blocks + AdaGN; retain KTA query and no-ICP transport."""
import torch
from torch import nn
from torch.nn import functional as F
from tools.source_scene_stage1.model import ReaderConfig
from tools.source_scene_kta_query.model import QueryTransport, QueryBody, make_model as fresh_control
from .attention import SceneTemporalBlock
from .refiner import AdaGNRefiner


class IntegratedTransport(QueryTransport):
    def __init__(self, config, width=64, z_bins=16, reader_config=ReaderConfig()):
        super().__init__(config, width, z_bins, reader_config)
        if reader_config.layers != 2 or reader_config.temporal_layers != 2: raise ValueError('fixed TWO integrated encoder blocks')
        del self.readers, self.temporal
        dim = config.d_model
        self.scene_global = nn.Linear(dim, dim)
        self.encoder_blocks = nn.ModuleList([SceneTemporalBlock(dim, config.heads, reader_config.points) for _ in range(2)])

    def forward(self, field, reads, times, classes, kta):
        valid = reads['reference_valid'].bool(); sources, frames = valid.shape
        if frames != 4 or kta is None or kta.shape != (sources, 6, 2): raise ValueError('four tokens and causal KTA required')
        dim = self.config.d_model
        if not sources:
            return dict(residual_xy_m=field.new_empty((0, 6, 2)), yaw_delta_rad=field.new_empty((0, 6)),
                existence_logits=field.new_empty((0, 6)), history_source_context=field.new_empty((0, dim)),
                future_transport_queries=field.new_empty((0, 6, dim)), history_yaw_to_t0_rad=field.new_empty((0, 4)))
        if not bool(valid[:, -1].all()): raise ValueError('real t0 reference required')
        native = self.feature_projection(field); sx, sy = native.shape[-2:]
        sizes = ((sx, sy), (max(1, sx//4), max(1, sy//4)),
                 (min(self.reader_config.global_side, sx), min(self.reader_config.global_side, sy)))
        pyramid = [native if size == (sx, sy) else F.adaptive_avg_pool2d(native, size) for size in sizes]
        grid = reads['reference_grid'].permute(1, 0, 2).float()
        seed = F.grid_sample(native.float(), grid[:, :, None], align_corners=False, padding_mode='zeros')[..., 0].transpose(1, 2)
        pos = torch.cat((reads['reference_xy_t0'].float()/40, times[None, :, None].expand(sources, -1, -1)), -1)
        pos = torch.where(valid[..., None], pos, torch.zeros_like(pos))
        global_condition = self.scene_global(pyramid[-1].mean((-2, -1)))[:, None]
        context = ((seed+self.position(pos).permute(1, 0, 2)+global_condition)*valid.T[..., None]).transpose(0, 1)
        for block in self.encoder_blocks: context = block(context, pyramid, reads, valid)
        history_yaw = self.history_yaw(context-context[:, -1:])[..., 0]*valid
        q = self.future_query.expand(sources, -1, -1)+self.kta_future_proj(kta.to(context.dtype)/20)
        for block in self.decoder: q = block(q, context, valid)
        return dict(residual_xy_m=self.residual_head(q), yaw_delta_rad=self.yaw_head(q)[..., 0],
            existence_logits=q.new_zeros((sources, 6)),
            history_source_context=(context*valid[..., None]).sum(1)/valid.sum(1, keepdim=True),
            future_transport_queries=q, history_yaw_to_t0_rad=history_yaw)


class IntegratedBody(QueryBody):
    def __init__(self, config, *, feature_width=64, z_bins=16, memory_width=16, width=32,
                 chunk_size=65536, reader_config=ReaderConfig()):
        super().__init__(config, feature_width=feature_width, z_bins=z_bins, memory_width=memory_width,
                         width=width, chunk_size=chunk_size, reader_config=reader_config)
        self.transport = IntegratedTransport(config, feature_width, z_bins, reader_config)
        self.columns = AdaGNRefiner(config.d_model, feature_width, z_bins, memory_width, width, chunk_size)

    def configs(self):
        c = super().configs()
        return {**c, 'variant': 'source_scene_integrated_AdaGN',
            'encoder': dict(blocks=2, sequence='PreNorm_scene_temporal_FFN', global_context='per_frame_coarse_GAP_initial_only',
                            global_attention=False, cross_source_attention=False),
            'repair': {**c['repair'], 'conditioning': 'residual_branch_AdaGN_zero_projection', 'GN_affine': False}}


def common_key(key):
    """Map only semantically preserved FRESH modules, never a trained state."""
    prefix = 'transport.encoder_blocks.'
    if key.startswith(prefix):
        index, rest = key[len(prefix):].split('.', 1)
        for new, old in (('scene_norm.', 'readers.'+index+'.norm.'), ('scene.', 'readers.'+index+'.'),
                         ('temporal_norm.', 'temporal.'+index+'.norm1.'),
                         ('temporal_attention.', 'temporal.'+index+'.self_attn.'),
                         ('ffn_norm.', 'temporal.'+index+'.norm2.'),
                         ('ffn.0.', 'temporal.'+index+'.linear1.'), ('ffn.2.', 'temporal.'+index+'.linear2.')):
            if rest.startswith(new):
                suffix = rest[len(new):]
                if new == 'scene.' and suffix.startswith('output.'): suffix = 'local_output.'+suffix[len('output.'):]
                return 'transport.'+old+suffix
    prefix = 'columns.blocks.'
    if key.startswith(prefix):
        index, rest = key[len(prefix):].split('.', 1)
        if rest.startswith('convs.'):
            layer, suffix = rest[len('convs.'):].split('.', 1)
            return prefix+index+'.block.'+str(int(layer)+1)+'.'+suffix
    return key


def make_model(config, **kwargs):
    control = fresh_control(config, **kwargs)
    with torch.random.fork_rng(devices=[]): model = IntegratedBody(config, **kwargs)
    previous = control.state_dict(); state = model.state_dict()
    model.load_state_dict({k: previous[common_key(k)] if common_key(k) in previous and previous[common_key(k)].shape == v.shape else v
                           for k, v in state.items()}, strict=True)
    return model


def checkpoint_model(saved, device='cpu'):
    from .run import PROTOCOL
    from real_motion.local_st_world_model_v17 import config_from_mapping_v17
    spec = saved['contract']['model']; sh, rep = spec['shared_history'], spec['repair']
    if saved.get('protocol') != PROTOCOL or spec.get('variant') != 'source_scene_integrated_AdaGN':
        raise RuntimeError('integrated Scene-Temporal/AdaGN checkpoint required')
    model = IntegratedBody(config_from_mapping_v17(spec['motion']), feature_width=sh['width'], z_bins=sh['z_bins'],
        memory_width=rep['memory_width'], width=rep['width'], chunk_size=rep['chunk_size'], reader_config=ReaderConfig(**spec['reader']))
    model.columns.epsilon = rep['epsilon']; model.load_state_dict(saved['models']['gt'], strict=True)
    return model.to(device).eval()
