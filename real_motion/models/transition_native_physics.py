"""P0-F9 native OccFM transition with gated Strong-W2Det physics conditioning."""
from __future__ import annotations

from .physics_prior import GatedPhysicsCrossAttention
from .transition_full_context import MotionWindowFlowMatchingFullContext


class MotionWindowNativePhysicsTransition(MotionWindowFlowMatchingFullContext):
    """Full-history native forecasting transition with a gated physics prior.

    The parent already contains the P0-F4 full-history/context and token-wise
    prior conditioning path. P0-F9 adds one explicit cross-attention interaction
    at the bottleneck. The gate is initialized to zero, so a freshly constructed
    P0-F9 model remains numerically identical to the parent transition before
    training (apart from the native flow source handled outside this module).
    """

    def __init__(
        self,
        *args,
        prior_channels: int = 16,
        physics_mid_channels: int = 256,
        physics_heads: int = 8,
        **kwargs,
    ) -> None:
        super().__init__(*args, prior_channels=prior_channels, **kwargs)
        self.physics_fusion = GatedPhysicsCrossAttention(
            prior_channels=int(prior_channels),
            hidden_size=int(physics_mid_channels),
            num_heads=int(physics_heads),
        )

    def _mid_physics_fusion(self, x, prior_future):
        return self.physics_fusion(x, prior_future)
