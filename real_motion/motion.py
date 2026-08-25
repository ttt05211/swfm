"""Causal real-motion decomposition utilities.

The final method must never use future GT to build these masks.
This module provides an occupancy-only, causal baseline based on temporal
persistence after ego compensation. A stronger tracker/KTA implementation can
replace it as long as it returns the same three-state contract.
"""
from dataclasses import dataclass
import numpy as np

STATIC = np.uint8(0)
MOVING = np.uint8(1)
UNCERTAIN = np.uint8(2)

@dataclass(frozen=True)
class PersistenceMotionConfig:
    free_label: int = 17
    static_min_persistence: float = 0.80
    moving_max_persistence: float = 0.50
    min_observed_frames: int = 2

def decompose_ego_aligned_history(history_semantics: np.ndarray,
                                  cfg: PersistenceMotionConfig = PersistenceMotionConfig()):
    """Split the current occupied voxels into static/moving/uncertain.

    history_semantics is [T,H,W,D] and must already be transformed into the
    current ego grid. Future GT is never used here.
    """
    x = np.asarray(history_semantics)
    if x.ndim != 4 or x.shape[0] < cfg.min_observed_frames:
        raise ValueError("history_semantics must be [T,H,W,D] with enough frames")
    current = x[-1]
    occupied = current != cfg.free_label
    same = x == current[None, ...]
    observed = (x != cfg.free_label) | occupied[None, ...]
    denom = np.maximum(observed.sum(axis=0), 1)
    persistence = (same & observed).sum(axis=0) / denom
    state = np.full(current.shape, UNCERTAIN, dtype=np.uint8)
    state[(persistence >= cfg.static_min_persistence) & occupied] = STATIC
    state[(persistence <= cfg.moving_max_persistence) & occupied] = MOVING
    state[~occupied] = STATIC
    return state, occupied, persistence.astype(np.float32)
