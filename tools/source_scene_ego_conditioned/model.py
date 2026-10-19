"""Latest TWO+TWO with three explicit, jointly tested data-flow changes."""
import torch
from torch import nn
from torch.nn import functional as F
from real_motion.local_st_world_model import FutureQueryBlock
from tools.source_scene_stage1.model import ReaderConfig
from tools.source_scene_integrated.model import IntegratedTransport, IntegratedBody, make_model as fresh_control

VARIANT = 'integrated22_ego_query_history_only_memory_cross_first'


def relative_ego_features(current_pose, future_poses):
    """Differentiable relative t0-ego SE(2), computed from the renderer's SAME poses."""
    if future_poses.shape != (6, 4, 4) or current_pose.shape != (4, 4):
        raise ValueError('one t0 pose and SIX future poses required')
    if not bool(torch.isfinite(future_poses).all()) or not bool(torch.isfinite(current_pose).all()):
        raise ValueError('finite poses required')
    homogeneous = future_poses.new_tensor([0., 0., 0., 1.])
    if not torch.allclose(future_poses[:, 3], homogeneous.expand(6, -1), atol=1e-8, rtol=0.) or not torch.allclose(current_pose[3], homogeneous, atol=1e-8, rtol=0.):
        raise ValueError('homogeneous SE(3) poses required')
    relative = torch.linalg.inv(current_pose) @ future_poses
    yaw = torch.atan2(relative[:, 1, 0], relative[:, 0, 0])
    return torch.stack((relative[:, 0, 3], relative[:, 1, 3], yaw.sin(), yaw.cos()), -1).float()


class CrossFirstFutureBlock(FutureQueryBlock):
    def forward(self, q, context, valid):
        z = self.cross_ctx_norm(context)
        q = q+self.cross_attn(self.cross_q_norm(q), z, z, key_padding_mask=~valid, need_weights=False)[0]
        # MUST normalize the UPDATED q, never reuse the pre-cross query.
        z = self.self_norm(q)
        q = q+self.self_attn(z, z, z, need_weights=False)[0]
        return q+self.ffn(self.ffn_norm(q))


class EgoTransport(IntegratedTransport):
    def __init__(self, config, width=64, z_bins=16, reader_config=ReaderConfig()):
        super().__init__(config, width, z_bins, reader_config)
        if config.decoder_blocks != 2: raise ValueError('fixed TWO encoder / TWO decoder blocks')
        dim = config.d_model
        self.decoder = nn.ModuleList([CrossFirstFutureBlock(dim, config.heads) for _ in range(2)])
        self.ego_future_proj = nn.Sequential(nn.Linear(4, dim), nn.SiLU(), nn.Linear(dim, dim))
        nn.init.zeros_(self.ego_future_proj[-1].weight); nn.init.zeros_(self.ego_future_proj[-1].bias)

    def initial_queries(self, kta, ego_features, dtype):
        if ego_features is None or ego_features.shape != (6, 4) or not bool(torch.isfinite(ego_features).all()):
            raise ValueError('SIX explicit relative ego conditions required')
        scale = ego_features.new_tensor([20., 20., 1., 1.])
        return (self.future_query.expand(len(kta), -1, -1)+self.kta_future_proj(kta.to(dtype)/20)+
                self.ego_future_proj((ego_features/scale).to(dtype))[None])

    def forward(self, field, reads, times, classes, kta, ego_features=None):
        valid = reads['reference_valid'].bool(); sources, frames = valid.shape
        if frames != 4 or kta is None or kta.shape != (sources, 6, 2): raise ValueError('four tokens and causal KTA required')
        if ego_features is None or ego_features.shape != (6, 4) or not bool(torch.isfinite(ego_features).all()):
            raise ValueError('explicit relative ego trajectory required, including empty-source windows')
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
        q = self.initial_queries(kta, ego_features, context.dtype)
        for block in self.decoder: q = block(q, context, valid)
        return dict(residual_xy_m=self.residual_head(q), yaw_delta_rad=self.yaw_head(q)[..., 0],
            existence_logits=q.new_zeros((sources, 6)),
            history_source_context=(context*valid[..., None]).sum(1)/valid.sum(1, keepdim=True),
            future_transport_queries=q, history_yaw_to_t0_rad=history_yaw)


class EgoBody(IntegratedBody):
    def __init__(self, config, *, feature_width=64, z_bins=16, memory_width=16, width=32,
                 chunk_size=65536, reader_config=ReaderConfig()):
        super().__init__(config, feature_width=feature_width, z_bins=z_bins, memory_width=memory_width,
            width=width, chunk_size=chunk_size, reader_config=reader_config)
        self.transport = EgoTransport(config, feature_width, z_bins, reader_config)
        del self.columns.source_query  # No dormant parameters or optimizer entries.

    def configs(self):
        c = super().configs()
        return {**c, 'variant': VARIANT,
            'future_query': dict(type='six_horizon_learned_plus_KTA_plus_relative_ego',
                ego_fields=['dx_t0_m', 'dy_t0_m', 'sin_relative_yaw', 'cos_relative_yaw'],
                ego_XY_scale_m=20., ego_projection_last_zero=True,
                decoder_order='history_cross_future_self_FFN', history_padding_mask_only=True),
            'memory': dict(content='lifted_history_only', importance='history_features_times_coverage_only',
                future_query_content=False, future_query_importance=False,
                alignment='predicted_source_motion_and_SAME_future_ego', history_yaw='learned_t0_gauge_zero'),
            'source_motion_coordinates': 'UNCHANGED_t0_anchor_plus_XY_residual_then_world; unchanged_world_Z_yaw',
            'future_ego_role': 'behavior_condition_AND_output_projection; NO_double_motion_subtraction'}


def make_model(config, **kwargs):
    control = fresh_control(config, **kwargs)  # FRESH control; never load trained weights.
    with torch.random.fork_rng(devices=[]): model = EgoBody(config, **kwargs)
    old = control.state_dict()
    model.load_state_dict({k:old[k] if k in old else v for k,v in model.state_dict().items()}, strict=True)
    return model


def checkpoint_model(saved, device='cpu'):
    from .run import PROTOCOL
    from real_motion.local_st_world_model_v17 import config_from_mapping_v17
    spec = saved['contract']['model']; sh, rep = spec['shared_history'], spec['repair']
    if (saved.get('protocol') != PROTOCOL or spec.get('variant') != VARIANT or
            spec['encoder']['blocks'] != 2 or spec['motion']['decoder_blocks'] != 2 or
            spec.get('registration') != 'none' or spec.get('kta_conditioning') is not True or
            spec.get('future_query', {}).get('decoder_order') != 'history_cross_future_self_FFN' or
            spec.get('memory', {}).get('future_query_content') is not False or
            spec.get('memory', {}).get('future_query_importance') is not False):
        raise RuntimeError('only SAME ego-query/history-only-memory/cross-first checkpoint allowed')
    body = EgoBody(config_from_mapping_v17(spec['motion']), feature_width=sh['width'], z_bins=sh['z_bins'],
        memory_width=rep['memory_width'], width=rep['width'], chunk_size=rep['chunk_size'], reader_config=ReaderConfig(**spec['reader']))
    body.columns.epsilon = rep['epsilon']
    if body.configs() != spec: raise RuntimeError('checkpoint model specification differs')
    body.load_state_dict(saved['models']['gt'], strict=True)
    return body.to(device).eval().requires_grad_(False)
