"""Scene Reader without KTA query features; learn history-to-t0 relative yaw."""
from dataclasses import asdict
import torch
from torch import nn
from torch.nn import functional as F
from tools.source_scene_stage1.model import SceneTransport, ReaderConfig, make_model as fresh_stage1
from tools.shared_history_experiments.refine.model import SemanticRefiner


class DirectTransport(SceneTransport):
    def __init__(self, config, width=64, z_bins=16, reader_config=ReaderConfig()):
        super().__init__(config, width, z_bins, reader_config)
        del self.kta_future_proj
        self.history_yaw = nn.Linear(config.d_model, 1, bias=False)
        nn.init.zeros_(self.history_yaw.weight)

    def forward(self, field, reads, times, classes, kta):
        # classes/KTA are ABI arguments ONLY. Neither is read by the network.
        valid = reads['reference_valid'].bool(); sources, frames = valid.shape
        if frames != 4: raise ValueError('exact four identity tokens required')
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
        seed = F.grid_sample(native.float(), grid[:, :, None], align_corners=False,
                             padding_mode='zeros')[..., 0].transpose(1, 2)
        pos = torch.cat((reads['reference_xy_t0'].float()/40,
                         times[None, :, None].expand(sources, -1, -1)), -1)
        pos = torch.where(valid[..., None], pos, torch.zeros_like(pos))
        q = (seed+self.position(pos).permute(1, 0, 2))*valid.T[..., None]
        coarse = pyramid[-1]; gx, gy = coarse.shape[-2:]; bounds = reads['scene_bounds'].float()
        x = bounds[0]+(torch.arange(gx, device=field.device)+.5)/gx*(bounds[2]-bounds[0])
        y = bounds[1]+(torch.arange(gy, device=field.device)+.5)/gy*(bounds[3]-bounds[1])
        xy = torch.stack(torch.meshgrid(x, y, indexing='ij'), -1).reshape(-1, 2)
        xyz = torch.cat((xy, torch.zeros_like(xy[:, :1])), -1); matrix = reads['history_to_t0'].float()
        aligned = torch.einsum('fij,pj->fpi', matrix[:, :3, :3], xyz)+matrix[:, None, :3, 3]
        scene_pos = self.position(torch.cat((aligned[..., :2]/40,
            times[:, None, None].expand(-1, len(xy), -1)), -1))
        content = coarse.flatten(2).transpose(1, 2)
        for reader in self.readers: q = reader(q, pyramid, content, scene_pos, reads, valid)
        context = q.transpose(0, 1)
        for block in self.temporal: context = block(context, src_key_padding_mask=~valid)*valid[..., None]
        # Difference to the real t0 token fixes the gauge: t0 is exactly zero.
        # No GT yaw, PCA/ICP, class-specific rule, angle threshold or new loss.
        history_yaw = self.history_yaw(context-context[:, -1:])[..., 0]*valid
        q = self.future_query.expand(sources, -1, -1)
        for block in self.decoder: q = block(q, context, valid)
        return dict(residual_xy_m=self.residual_head(q), yaw_delta_rad=self.yaw_head(q)[..., 0],
            existence_logits=q.new_zeros((sources, 6)),
            history_source_context=(context*valid[..., None]).sum(1)/valid.sum(1, keepdim=True),
            future_transport_queries=q, history_yaw_to_t0_rad=history_yaw)


class DirectBody(nn.Module):
    def __init__(self, config, *, feature_width=64, z_bins=16, memory_width=16, width=32,
                 chunk_size=65536, reader_config=ReaderConfig()):
        super().__init__()
        self.transport = DirectTransport(config, feature_width, z_bins, reader_config)
        self.columns = SemanticRefiner(config.d_model, feature_width, z_bins, memory_width, width, chunk_size)
        self.ego = nn.Identity()

    def configs(self):
        return dict(motion=asdict(self.transport.config), reader=asdict(self.transport.reader_config),
            variant='source_scene_stage2', registration='none', validity='track_valid',
            kta_conditioning=False, kta_residual_skip=True, history_yaw='learned_t0_gauge_zero',
            shared_history=dict(width=self.transport.history_encoder.width, z_bins=self.columns.z_bins),
            repair=dict(memory_width=self.columns.memory_width, width=self.columns.width,
                        chunk_size=self.columns.chunk_size, epsilon=self.columns.epsilon),
            objective=dict(ce=1., lovasz=1., motion=0., existence=0., ego=0.))


def make_model(config, **kwargs):
    # Same RANDOM common modules as prior Reader, NOT its trained checkpoint.
    previous = fresh_stage1('reader', config, **kwargs)
    model = DirectBody(config, **kwargs); common = previous.state_dict()
    state = model.state_dict()
    model.load_state_dict({k: common[k] if k in common and common[k].shape == v.shape else v
                           for k, v in state.items()}, strict=True)
    return model


def checkpoint_model(saved, device='cpu'):
    from .run import PROTOCOL
    from real_motion.local_st_world_model_v17 import config_from_mapping_v17
    c = saved['contract']; spec = c['model']; shared, repair = spec['shared_history'], spec['repair']
    if saved.get('protocol') != PROTOCOL or spec.get('variant') != 'source_scene_stage2':
        raise RuntimeError('not a stage2 direct-history checkpoint')
    model = DirectBody(config_from_mapping_v17(spec['motion']), feature_width=shared['width'],
        z_bins=shared['z_bins'], memory_width=repair['memory_width'], width=repair['width'],
        chunk_size=repair['chunk_size'], reader_config=ReaderConfig(**spec['reader']))
    model.columns.epsilon = repair['epsilon']; model.load_state_dict(saved['models']['gt'], strict=True)
    return model.to(device).eval()
