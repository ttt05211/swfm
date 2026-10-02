"""Strict four-observation view of legacy six-slot V18 cache inputs.

The flat feature ABI is retained, but all old-slot and cross-boundary velocity
channels are zero. No padding tokens enter the four-frame encoder. GT targets
and the last-two-frame KTA anchor are unchanged. Legacy E14 remains SIX-frame.
"""
import torch
from .motion_transport import FEATURE_NAMES

HISTORY4_CONTRACT = 'last4_observations_no_old_slots_no_cross_boundary_velocity_v1'
_EXCLUDED = tuple(i for i, name in enumerate(FEATURE_NAMES) if name.startswith(
    ('hist_offset_0_', 'hist_offset_1_', 'hist_vel_0_', 'hist_vel_1_')) or name in ('hist_valid_0', 'hist_valid_1'))
_OFFSET = tuple(FEATURE_NAMES.index(f'hist_offset_{t}_{a}') for t in range(2, 6) for a in ('x', 'y'))
_SEGMENT = tuple(FEATURE_NAMES.index(f'hist_vel_{t}_{a}') for t in range(2, 5) for a in ('x', 'y'))
_VALID = tuple(FEATURE_NAMES.index(f'hist_valid_{t}') for t in range(2, 6))


def four_frame_motion_inputs(features, tube, frame, mask):
    if tube.ndim != 4 or tube.shape[1] not in (4, 6) or mask is None or mask.shape != tube.shape:
        raise ValueError('four-frame path requires four or legacy six-slot tubes')
    features = features.clone()
    features[:, _EXCLUDED] = 0
    # The first retained frame must use its outgoing segment, not a cached
    # average with the excluded preceding frame (nor half that velocity).
    n = len(features)
    offsets = features[:, _OFFSET].reshape(n, 4, 2)
    segments = features[:, _SEGMENT].reshape(n, 3, 2)
    valid = features[:, _VALID].reshape(n, 4, 1)
    velocities = torch.stack((segments[:, 0], .5*(segments[:, 0]+segments[:, 1]),
        .5*(segments[:, 1]+segments[:, 2]), segments[:, 2]), 1)
    frame = torch.cat((offsets*valid, velocities*valid, valid), -1)
    return features, tube[:, -4:], frame, mask[:, -4:]
