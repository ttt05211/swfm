"""MT-V1-SPEC-2: causal occupancy-only KTA + routed STPN raw-shape transport.

The package intentionally avoids importing the full runtime/model stack at import time so
geometry/contracts remain dependency-light and CI tests do not require nuScenes assets.
"""
from .contracts import CausalInputs, TrainingTargets, SourceRecord, PredictionResult

__all__ = [
    "CausalInputs", "TrainingTargets", "SourceRecord", "PredictionResult",
    "MotionTransportV1", "STPNMotionNetwork",
]


def __getattr__(name):
    if name == "MotionTransportV1":
        from .model import MotionTransportV1
        return MotionTransportV1
    if name == "STPNMotionNetwork":
        from .stpn import STPNMotionNetwork
        return STPNMotionNetwork
    raise AttributeError(name)
