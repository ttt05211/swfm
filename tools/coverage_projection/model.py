"""Keep the current U-Net widths; distinguish unknown from observed free."""
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from tools.fpn_unet.model import UNetRefiner, resize3d
from tools.causal_forward_train.model import ForwardBody, make_model as fresh_control

VARIANT = 'forward_FPN_AdaLN_UNet_coverage_projection_v1'


class CoverageRefiner(UNetRefiner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        old = self.fine[0]
        self.fine[0] = nn.Conv3d(21, 8, 3, padding=1, bias=False)
        with torch.no_grad():
            self.fine[0].weight.zero_()
            self.fine[0].weight[:, :19].copy_(old.weight)

    def _covered_residual(self, memory, probability, clock, prior_coverage, memory_coverage):
        with torch.autocast(device_type=memory.device.type, dtype=torch.bfloat16,
                            enabled=memory.device.type == 'cuda'):
            # A zero probability vector means UNKNOWN, not a one-hot FREE prior.
            inputs = torch.cat((probability * prior_coverage,
                clock.expand(-1, 1, *probability.shape[2:]), prior_coverage, memory_coverage), 1)
            fine = self.fine(inputs)
            middle = self.middle(torch.cat((self.pool(fine), memory), 1))
            low = self.low(self.pool(middle))
            value = self.decode_middle(torch.cat((resize3d(low, middle.shape[2:]), middle), 1))
            value = self.decode_fine(torch.cat((resize3d(value, fine.shape[2:]), fine), 1))
            return self.head(value).float().flatten(2)

    def forward(self, memory, prior, horizons, shape, *, prior_coverage, memory_coverage):
        hs, shape = tuple(horizons), tuple(map(int, shape)); n = len(hs)
        if (not n or any(h not in range(6) for h in hs) or shape[0] % 4 or shape[1] % 4
                or shape[2] != self.z_bins or prior.shape != (n, 18, math.prod(shape))
                or memory.shape != (n, self.memory_width, shape[0]//2, shape[1]//2, shape[2])
                or prior_coverage.shape != (n, 1, math.prod(shape))
                or memory_coverage.shape != (n, 1, shape[0]//2, shape[1]//2, shape[2])):
            raise ValueError('coverage/prior/memory must describe the same complete future grid')
        pc = prior_coverage.detach().float().clamp(0, 1).reshape(n, 1, *shape)
        mc = F.interpolate(memory_coverage.detach().float(), size=shape, mode='nearest')
        p = prior.reshape(n, 18, *shape)
        clock = prior.new_tensor([.5*(h+1)/3 for h in hs]).reshape(n, 1, 1, 1, 1)
        rows = []
        for start in range(0, n, self.horizon_chunk):
            sl = slice(start, start+self.horizon_chunk)
            args = (memory[sl], p[sl], clock[sl], pc[sl], mc[sl])
            rows.append(checkpoint(self._covered_residual, *args, use_reentrant=False)
                if self.training and torch.is_grad_enabled() else self._covered_residual(*args))
        # Gate only the additive prior. Prediction remains possible EVERYWHERE.
        return self.prior_logits(prior)*pc.flatten(2) + torch.cat(rows)


class CoverageBody(ForwardBody):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        old = self.columns
        self.columns = CoverageRefiner(self.transport.history_encoder.width,
            old.memory_width, old.z_bins, old.epsilon, old.horizon_chunk)

    def configs(self):
        spec = copy.deepcopy(super().configs())
        spec.update(variant=VARIANT, coverage=dict(prior='t0_valid_warp_UNION_source_splat',
            memory='positive_fusion_mass', channels=2, unknown_prior='zero_vector',
            fusion='coverage_times_log_prior_plus_unrestricted_network', detached=True))
        return spec


def make_model(config, **kwargs):
    control = fresh_control(config, **kwargs)
    with torch.random.fork_rng(devices=[]):
        body = CoverageBody(config, **kwargs)
    state = body.state_dict(); old = control.state_dict()
    for name in state:
        if name == 'columns.fine.0.weight':
            state[name].zero_(); state[name][:, :19].copy_(old[name])
        else:
            state[name].copy_(old[name])
    body.load_state_dict(state, strict=True)
    return body


def from_spec(spec, *, candidate):
    from real_motion.local_st_world_model_v17 import config_from_mapping_v17
    from tools.source_scene_stage1.model import ReaderConfig
    cls = CoverageBody if candidate else ForwardBody
    return cls(config_from_mapping_v17(spec['motion']),
        feature_width=spec['shared_history']['width'], z_bins=spec['shared_history']['z_bins'],
        memory_width=spec['repair']['memory_width'], reader_config=ReaderConfig(**spec['reader']),
        experiment=spec.get('experiment'))
