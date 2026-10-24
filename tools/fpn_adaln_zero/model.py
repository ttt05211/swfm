"""Replace additive ego queries, not FPN, geometry, history memory or U-Net."""
import copy
import hashlib
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from tools.fpn_unet.model import FPNTransport, FPNUNetBody, make_model as fresh_additive
from tools.source_scene_ego_conditioned.model import CrossFirstFutureBlock
from tools.source_scene_stage1.model import ReaderConfig

VARIANT = 'FPN_UNet_ego_AdaLN_Zero_cross_self_FFN_v1'


def modulate(value, shift, scale):
    return value*(1+scale)+shift


class AdaLNZeroFutureBlock(CrossFirstFutureBlock):
    """THREE independently gated branches; historical K/V are never modulated."""
    def __init__(self, dim, heads):
        super().__init__(dim, heads)
        for name in ('cross_q_norm', 'self_norm', 'ffn_norm'):
            old = getattr(self, name)
            setattr(self, name, nn.LayerNorm(dim, eps=old.eps, elementwise_affine=False))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 9*dim))
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, q, context, valid, condition):
        if condition.shape != (1, q.shape[1], q.shape[2]):
            raise ValueError('per-horizon ego condition, shared across sources, required')
        shift_c, scale_c, gate_c, shift_s, scale_s, gate_s, shift_f, scale_f, gate_f = (
            self.adaLN_modulation(condition).chunk(9, dim=-1))
        historical = self.cross_ctx_norm(context)
        z = modulate(self.cross_q_norm(q), shift_c, scale_c)
        q = q+gate_c*self.cross_attn(z, historical, historical,
            key_padding_mask=~valid, need_weights=False)[0]
        # Normalize the cross-UPDATED query, not the old input query.
        z = modulate(self.self_norm(q), shift_s, scale_s)
        q = q+gate_s*self.self_attn(z, z, z, need_weights=False)[0]
        z = modulate(self.ffn_norm(q), shift_f, scale_f)
        return q+gate_f*self.ffn(z)


class AdaLNTransport(FPNTransport):
    def __init__(self, config, width=64, z_bins=16, reader_config=ReaderConfig(global_side=25)):
        super().__init__(config, width, z_bins, reader_config)
        dim = config.d_model
        del self.ego_future_proj  # No additive ego path, including dormant parameters.
        self.ego_condition_encoder = nn.Sequential(nn.Linear(4, dim), nn.SiLU(), nn.Linear(dim, dim))
        # Normal initialization here: zero only the downstream modulation output.
        self.decoder = nn.ModuleList([AdaLNZeroFutureBlock(dim, config.heads) for _ in range(2)])

    def initial_queries(self, kta, ego_features, dtype):
        return self.future_query.expand(len(kta), -1, -1)+self.kta_future_proj(kta.to(dtype)/20)

    def forward(self, fields, reads, times, classes, kta, ego_features=None):
        if len(fields) != 3: raise ValueError('three learned FPN levels required')
        field = fields[0]; valid = reads['reference_valid'].bool(); sources, frames = valid.shape
        if frames != 4 or kta is None or kta.shape != (sources, 6, 2):
            raise ValueError('four historical tokens and six KTA horizons required')
        if ego_features is None or ego_features.shape != (6, 4) or not bool(torch.isfinite(ego_features).all()):
            raise ValueError('explicit SIX relative future ego conditions required')
        dim = self.config.d_model
        if not sources:
            return dict(residual_xy_m=field.new_empty((0, 6, 2)), yaw_delta_rad=field.new_empty((0, 6)),
                existence_logits=field.new_empty((0, 6)), history_source_context=field.new_empty((0, dim)),
                future_transport_queries=field.new_empty((0, 6, dim)), history_yaw_to_t0_rad=field.new_empty((0, 4)))
        if not bool(valid[:, -1].all()): raise ValueError('real t0 reference required')
        pyramid = [self.feature_projection(v) for v in fields]
        native = pyramid[0]; grid = reads['reference_grid'].permute(1, 0, 2).float()
        seed = F.grid_sample(native.float(), grid[:, :, None], align_corners=False,
            padding_mode='zeros')[..., 0].transpose(1, 2)
        pos = torch.cat((reads['reference_xy_t0'].float()/40,
            times[None, :, None].expand(sources, -1, -1)), -1)
        pos = torch.where(valid[..., None], pos, torch.zeros_like(pos))
        global_condition = self.scene_global(pyramid[-1].mean((-2, -1)))[:, None]
        context = ((seed+self.position(pos).permute(1, 0, 2)+global_condition)*valid.T[..., None]).transpose(0, 1)
        for block in self.encoder_blocks: context = block(context, pyramid, reads, valid)
        history_yaw = self.history_yaw(context-context[:, -1:])[..., 0]*valid
        q = self.initial_queries(kta, ego_features, context.dtype)
        scale = ego_features.new_tensor([20., 20., 1., 1.])
        condition = self.ego_condition_encoder((ego_features/scale).to(context.dtype))[None]
        for block in self.decoder: q = block(q, context, valid, condition)
        return dict(residual_xy_m=self.residual_head(q), yaw_delta_rad=self.yaw_head(q)[..., 0],
            existence_logits=q.new_zeros((sources, 6)),
            history_source_context=(context*valid[..., None]).sum(1)/valid.sum(1, keepdim=True),
            future_transport_queries=q, history_yaw_to_t0_rad=history_yaw)


class AdaLNBody(FPNUNetBody):
    def __init__(self, config, *, feature_width=64, z_bins=16, memory_width=16,
                 reader_config=ReaderConfig(global_side=25), experiment=None):
        super().__init__(config, feature_width=feature_width, z_bins=z_bins,
            memory_width=memory_width, reader_config=reader_config)
        self.transport = AdaLNTransport(config, feature_width, z_bins, reader_config)
        self.experiment = copy.deepcopy(experiment)

    def configs(self):
        original = super().configs()
        return {**original, 'variant': VARIANT,
            'future_query': dict(type='six_learned_plus_KTA;relative_ego_via_AdaLN_Zero',
                decoder_order='cross_self_FFN', additive_ego=False, per_horizon=True,
                branches=['cross_query', 'self_attention', 'FFN'], historical_KV_conditioned=False,
                scale_shift_gate=True, modulation_last_zero=True, condition_encoder_zero=False,
                conditioned_norm_affine=False, original_LN_epsilon_preserved=True, ego_XY_scale_m=20.),
            'experiment': copy.deepcopy(self.experiment)}


def make_model(config, *, expected_control_spec=None, experiment=None, **kwargs):
    control = fresh_additive(config, **kwargs)  # FRESH random values, NEVER trained control weights.
    if expected_control_spec is not None and control.configs() != expected_control_spec:
        raise RuntimeError('control FPN/U-Net architecture differs')
    # Preserve the original post-construction CPU RNG and every semantically common tensor.
    with torch.random.fork_rng(devices=[]):
        model = AdaLNBody(config, experiment=experiment, **kwargs)
    old = control.state_dict()
    model.load_state_dict({k:old[k] if k in old and old[k].shape == v.shape else v
        for k,v in model.state_dict().items()}, strict=True)
    return model


def checkpoint_model(saved, device='cpu'):
    from .run import PROTOCOL
    from real_motion.local_st_world_model_v17 import config_from_mapping_v17
    spec = saved['contract']['model']
    if saved.get('protocol') != PROTOCOL or spec.get('variant') != VARIANT:
        raise RuntimeError('only this AdaLN-Zero TRAIN20%x1 checkpoint is accepted')
    root = Path(__file__).resolve().parents[2]
    for name, digest in (spec.get('experiment') or {}).get('candidate_implementation', {}).items():
        path = (root/name).resolve()
        if root not in path.parents or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError('AdaLN-Zero source implementation differs: '+name)
    model = AdaLNBody(config_from_mapping_v17(spec['motion']), feature_width=spec['shared_history']['width'],
        z_bins=spec['shared_history']['z_bins'], memory_width=spec['repair']['memory_width'],
        reader_config=ReaderConfig(**spec['reader']), experiment=spec.get('experiment'))
    if model.configs() != spec: raise RuntimeError('AdaLN-Zero checkpoint architecture differs')
    model.load_state_dict(saved['models']['gt'], strict=True)
    return model.to(device).eval()
