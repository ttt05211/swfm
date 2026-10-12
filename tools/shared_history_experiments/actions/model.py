"""Shared history with ownership-constrained, trainable continuous deletion."""
import torch
from torch import nn
from tools.shared_history_experiments.model import SharedRepair, SharedJoint

PROTOCOL = 'p0_f9_shared_history_gt_soft_addremove_20percent_v1'


class SharedRepairBoth(SharedRepair):
    requires_fallback = True

    def __init__(self, source_dim, width=64, z_bins=16):
        super().__init__(source_dim, width, z_bins)
        # Original 23 conditions + REMOVE legality + fallback embedding8.
        self.context = nn.Linear(14 + 8 + 2 + 8, width)
        self.readout[-1] = nn.Linear(32, 2)
        nn.init.normal_(self.readout[-1].weight, std=.01)
        with torch.no_grad():
            self.readout[-1].bias.copy_(torch.tensor([-2., -4.]))

    def forward(self, encoded, output, actor, context, base, legal, *, fallback):
        if fallback.shape != base.shape or legal.shape != (*base.shape, 2):
            raise ValueError('ADD/REMOVE conditions and fallback shape differ')
        # The canonical planner validates disjoint legality on CPU once;
        # avoid a CUDA-to-host fence for every 8192-point readout chunk.
        dynamic = actor >= 0
        x = encoded[:, None]
        if len(output['history_source_context']):
            x = x + self.source(output['history_source_context'])[actor.clamp_min(0)][:, None] * dynamic[:, None, None]
            x = x + self.future(output['future_transport_queries'])[actor.clamp_min(0)] * dynamic[:, None, None]
        conditions = torch.cat((context, self.semantic(base.long()),
                                legal.to(context.dtype), self.semantic(fallback.long())), -1)
        # KEEP is the remaining probability mass. No deployment threshold and
        # no extra action BCE: dense CE/Lovasz supervises the composed result.
        return self.readout(x + self.context(conditions)).float().sigmoid()


class SharedJointBoth(SharedJoint):
    def __init__(self, motion_config, *, width=64, z_bins=16):
        super().__init__(motion_config, width=width, z_bins=z_bins)
        self.columns = SharedRepairBoth(motion_config.d_model, width, z_bins)

    def configs(self):
        result = super().configs()
        result['repair'].update(action='soft_ADD_REMOVE_train_and_eval',
                                fallback_condition=True, remove_initial_bias=-4.)
        return result
