"""Optional refine-only causal spatial lookup and same-source temporal readout.

The atlas is a coarse, fixed descriptor of historical static occupancy, NOT
future GT or a learned full-resolution BEV backbone. No candidate changes.
"""
from dataclasses import dataclass
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .joint_causal_columns import LinkedColumns, CONTRACT
from .causal_column_completion import REFINE

PROTOCOL = 'p0_f9_joint_adaptive_refine_screen_v1'
LINK_PROTOCOL = 'same_source_six_future_queries_refine_no_detach_v1'
TRAINING_CONTRACT = {**CONTRACT, 'source_link': LINK_PROTOCOL,
    'dynamic_context': 'same_source_all_six_history_conditioned_future_queries',
    'static_context': 'history_static_descriptor_atlas_stride4_eight_bounded_offsets',
    'context_scope': 'refine_only_generation_readout_unchanged_zero_initial_residual'}
EXTRA_KEYS = ('source_sequence', 'dynamic_mask', 'static_atlas', 'atlas_index',
              'lookup_xy', 'lookup_scale', 'lookup_upper')


@dataclass(frozen=True)
class AdaptiveContextConfig:
    stride: int = 4
    points: int = 8
    max_offset_m: float = 8.
    initial_radius_m: float = 4.
    gate_bias: float = -2.

    def validate(self):
        if (self.stride != 4 or self.points != 8 or not math.isfinite(self.max_offset_m)
                or not 0 < self.initial_radius_m < self.max_offset_m
                or not math.isfinite(self.gate_bias)):
            raise ValueError('invalid adaptive context contract')


def static_atlas(memory, footprint, stride=4):
    """Eight channels; UNKNOWN and uncovered cells contribute no content.

    Road/sidewalk volume fractions, per-class height, other-static fraction,
    historical-grid coverage, occupied column fraction, ground column fraction.
    Grid coverage is NOT a lidar-visibility claim. Pool only fixed evidence.
    """
    m = np.asarray(memory); covered = np.asarray(footprint, dtype=bool)
    if m.ndim != 3 or covered.shape != m.shape[:2]: raise ValueError('atlas geometry shape mismatch')
    z = m.shape[2]; height = (np.arange(z, dtype=np.float32)+.5)/z
    road, walk = (m == 11)&covered[..., None], (m == 13)&covered[..., None]
    occupied = (m < 17)&covered[..., None]
    def mean_height(mask):
        count = mask.sum(-1)
        return (mask*height).sum(-1)/np.maximum(count, 1)
    a = np.stack((road.mean(-1), walk.mean(-1), mean_height(road), mean_height(walk),
                  (occupied & ~road & ~walk).mean(-1), covered,
                  occupied.any(-1), (road | walk).any(-1)), axis=0).astype(np.float32)
    x, y = m.shape[:2]; nx, ny = (x+stride-1)//stride, (y+stride-1)//stride
    a = np.pad(a, ((0, 0), (0, nx*stride-x), (0, ny*stride-y)))
    return a.reshape(8, nx, stride, ny, stride).mean((2, 4))


class AdaptiveRefineContext(nn.Module):
    def __init__(self, width, source_dim, heads, config):
        super().__init__(); config.validate(); self.config = config
        self.static_encoder = nn.Sequential(nn.Linear(8, width), nn.GELU(), nn.Linear(width, width))
        self.offsets = nn.Linear(width, 2*config.points)
        nn.init.zeros_(self.offsets.weight)
        angle = torch.arange(config.points)*2*math.pi/config.points
        radial = torch.stack((angle.cos(), angle.sin()), -1)*config.initial_radius_m/config.max_offset_m
        with torch.no_grad(): self.offsets.bias.copy_(radial.atanh().flatten())
        self.point_weights = nn.Linear(width, config.points)
        nn.init.zeros_(self.point_weights.weight); nn.init.zeros_(self.point_weights.bias)
        self.temporal_projection = nn.Linear(source_dim, width, bias=False)
        self.time = nn.Parameter(torch.zeros(6, width))
        self.temporal = nn.MultiheadAttention(width, heads, dropout=0., batch_first=True)
        self.static_out, self.dynamic_out = nn.Linear(width, width, bias=False), nn.Linear(width, width, bias=False)
        nn.init.zeros_(self.static_out.weight); nn.init.zeros_(self.dynamic_out.weight)
        self.gate = nn.Linear(2*width, width)
        nn.init.zeros_(self.gate.weight); nn.init.constant_(self.gate.bias, config.gate_bias)

    def spatial_context(self, q, atlas, index, xy, scale, upper):
        """One small atlas per horizon, not a replicated atlas per query.

        Atlas axes are [grid-X, grid-Y]; grid_sample expects [W=Y,H=X].
        Offsets use the SAME Y,X order as xy/scale. Sampling stays float32.
        """
        delta = self.offsets(q).float().reshape(-1, self.config.points, 2).tanh()*self.config.max_offset_m
        coords = xy[:, None].float()+delta*scale[:, None].float()
        # Index-copy below preserves coordinate/encoder gradients and row order.
        result = q.new_zeros(q.shape)
        for ai in range(len(atlas)):
            ids = torch.nonzero(index == ai).flatten()
            if not len(ids): continue
            with torch.autocast(device_type=q.device.type, enabled=False):
                v = F.grid_sample(atlas[ai:ai+1].float(), coords[ids][None],
                    mode='bilinear', padding_mode='zeros', align_corners=False)[0].permute(1, 2, 0)
            inside = ((coords[ids] >= -1) & (coords[ids] < upper[ids, None])).all(-1)
            valid = inside & (v[..., 5] > 0)
            # Renormalize away padding/unknown contribution, exclude coverage channel.
            content = v/v[..., 5:6].clamp_min(1e-6); content[..., 5] = v[..., 5]
            encoded = self.static_encoder(content)*valid[..., None]
            logits = self.point_weights(q[ids]).float().masked_fill(~valid, -1e9)
            weights = logits.softmax(-1)*valid
            weights = weights/weights.sum(-1, keepdim=True).clamp_min(1e-8)
            c = (encoded*weights[..., None]).sum(1)
            result = result.index_copy(0, ids, c.to(result.dtype))
        return self.static_out(result)

    def forward(self, q, kind, *, source_sequence, dynamic_mask, static_atlas, atlas_index,
                lookup_xy, lookup_scale, lookup_upper):
        c = torch.zeros_like(q)
        static = torch.nonzero((kind == REFINE) & ~dynamic_mask).flatten()
        if len(static):
            cs = self.spatial_context(q[static], static_atlas, atlas_index[static],
                lookup_xy[static], lookup_scale[static], lookup_upper[static])
            c = c.index_copy(0, static, cs.to(c.dtype))
        dynamic = torch.nonzero((kind == REFINE) & dynamic_mask).flatten()
        if len(dynamic):
            memory = self.temporal_projection(source_sequence[dynamic])+self.time.to(q.dtype)
            cd = self.temporal(q[dynamic, None], memory, memory, need_weights=False)[0][:, 0]
            c = c.index_copy(0, dynamic, self.dynamic_out(cd).to(c.dtype))
        return q+torch.sigmoid(self.gate(torch.cat((q, c), -1)))*c


class AdaptiveLinkedColumns(LinkedColumns):
    extra_input_keys = ('source_features', *EXTRA_KEYS)

    def __init__(self, config, source_dim, context_config):
        super().__init__(config, source_dim)
        self.context_config = context_config
        # Preserve the baseline's RNG position (and subsequent dropout stream).
        with torch.random.fork_rng(devices=[]):
            self.adaptive_context = AdaptiveRefineContext(config.width, source_dim, config.heads, context_config)

    def extra_inputs_for(self, prep, h, plan, grid, device):
        q = prep.outputs['future_transport_queries']
        actor = torch.as_tensor(plan.actor, device=device, dtype=torch.long); active = actor >= 0
        if active.any() and int(actor[active].max()) >= len(q): raise RuntimeError('actor/source order mismatch')
        sequence = q.new_zeros((len(plan), 6, self.source_dim))
        if active.any(): sequence = sequence.index_copy(0, torch.nonzero(active).flatten(), q[actor[active]])
        cache = getattr(prep, '_adaptive_atlases', None)
        if cache is None: cache = {}; prep._adaptive_atlases = cache
        if h not in cache: cache[h] = static_atlas(prep.memory[h], prep.footprints[h], self.context_config.stride)
        atlas = cache[h]; shape = np.asarray(grid.shape_hwd[:2], dtype=np.float32)
        padded = np.asarray(atlas.shape[1:], dtype=np.float32)*self.context_config.stride
        # Align-corners=False: query is original voxel center in pooled field.
        xy = ((np.asarray(plan.xy, dtype=np.float32)+.5)/padded*2-1)[:, ::-1].copy()
        scale = (2/(padded*np.asarray(grid.voxel_size[:2])))[::-1].copy()
        upper = (2*shape/padded-1)[::-1].copy()
        return {'source_sequence': sequence, 'dynamic_mask': active,
            'static_atlas': torch.as_tensor(atlas[None], device=device),
            'atlas_index': torch.zeros(len(plan), device=device, dtype=torch.long),
            'lookup_xy': torch.as_tensor(xy, device=device),
            'lookup_scale': torch.as_tensor(np.broadcast_to(scale, (len(plan), 2)).copy(), device=device, dtype=torch.float32),
            'lookup_upper': torch.as_tensor(np.broadcast_to(upper, (len(plan), 2)).copy(), device=device)}

    def forward(self, history, flags, base, fallback, context, kind, classes, *, source_features, **extra):
        if source_features.shape != (len(kind), self.source_dim): raise RuntimeError('source feature shape mismatch')
        if set(extra) != set(EXTRA_KEYS): raise RuntimeError('adaptive context input contract mismatch')
        if extra['source_sequence'].shape != (len(kind), 6, self.source_dim): raise RuntimeError('source sequence shape mismatch')
        if not torch.isfinite(source_features).all() or not torch.isfinite(extra['source_sequence']).all():
            raise RuntimeError('nonfinite source context')
        n = len(kind); atlas = extra['static_atlas']; index = extra['atlas_index']
        if (atlas.ndim != 4 or atlas.shape[1] != 8 or min(atlas.shape[2:]) < 1
                or index.shape != (n,) or index.dtype != torch.long
                or extra['dynamic_mask'].shape != (n,) or extra['dynamic_mask'].dtype != torch.bool
                or any(extra[k].shape != (n, 2) for k in ('lookup_xy', 'lookup_scale', 'lookup_upper'))):
            raise RuntimeError('static lookup shape/type contract mismatch')
        if (not torch.isfinite(atlas).all() or any(not torch.isfinite(extra[k]).all()
                for k in ('lookup_xy', 'lookup_scale', 'lookup_upper'))
                or (index < 0).any() or (index >= len(atlas)).any()
                or (extra['lookup_scale'] <= 0).any()):
            raise RuntimeError('invalid static lookup coordinates/atlas index')
        q = self.encode_local(history, flags, base, fallback, context, kind, classes,
            query_extra=self.source_projection(source_features))
        refined = self.adaptive_context(q, kind, **extra)
        # Generation readout receives exactly the original local query.
        return self.generation(q), self.refinement(refined).reshape(len(kind), self.config.z_bins, 3)
