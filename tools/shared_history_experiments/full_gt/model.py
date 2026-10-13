"""GT-conditioned body only: no learnable ego head, no auxiliary objectives."""
from torch import nn
from tools.shared_history_experiments.model import SharedTransport
from tools.shared_history_experiments.refine.model import ResidualJoint, SemanticRefiner

PROTOCOL = 'shared_semantic_refiner_full_gt_ce_lovasz_2epoch_v1'


class GTBody(ResidualJoint):
    def __init__(self, motion_config, *, feature_width=64, z_bins=16, memory_width=16,
                 width=32, chunk_size=65536):
        nn.Module.__init__(self)
        self.transport = SharedTransport(motion_config, feature_width, z_bins)
        self.columns = SemanticRefiner(motion_config.d_model, feature_width, z_bins,
                                       memory_width, width, chunk_size)
        # Empty compatibility slot for generic optimizer construction, never executed.
        self.ego = nn.Identity()

    def configs(self):
        return {**super().configs(), 'ego': dict(type='external_pose_condition_only', parameters=0),
                'objective': dict(ce=1., lovasz=1., motion=0., existence=0., ego=0.)}


ResidualJoint = GTBody
