"""One clean model: original V18 transport + integrated surface-aware CCR."""
from dataclasses import asdict
from torch import nn
from .local_st_world_model_v18_se2 import LocalSpatialTemporalWorldModelV18SE2
from .surface_canonical_repair import SurfaceCanonicalRepairHead, SURFACE_DIM, PHASE_DIM
from .causal_column_completion import ColumnConfig
from .joint_causal_columns import JointCausalColumns

PROTOCOL = 'p0_f9_clean_joint_surface_ccr_history4_v1'


class JointSurfaceCCR(nn.Module):
    # Identical five causal V18 inputs/autocast, with gradients retained.
    motion = JointCausalColumns.motion

    def __init__(self, motion_config, *, width=64, z_bins=16):
        super().__init__()
        if motion_config.history_frames != 4:
            raise ValueError('clean Surface CCR requires FOUR histories')
        self.transport = LocalSpatialTemporalWorldModelV18SE2(motion_config)
        self.columns = SurfaceCanonicalRepairHead(motion_config.d_model, width)
        # Preparation metadata only, not an extra Local-column network.
        self.columns.config = ColumnConfig(z_bins=z_bins)
        self.columns.static_only_training = False
        self.requires_grad_(True)

    def configs(self):
        return dict(motion=asdict(self.transport.v17_config),
                    repair=dict(source_dim=self.columns.source_dim, width=self.columns.width,
                                surface_dim=SURFACE_DIM, phase_dim=PHASE_DIM))
