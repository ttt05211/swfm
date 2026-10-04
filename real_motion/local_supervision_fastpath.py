"""Per-update CPU label indices; labels NEVER enter a model/feature path."""
import os
import numpy as np
import torch


def enabled():
    value = os.environ.get('SWFM_LOCAL_FAST_SUPERVISION', '0')
    if value not in ('0', '1'): raise ValueError('SWFM_LOCAL_FAST_SUPERVISION must be 0 or 1')
    return value == '1'


def static_roi_enabled():
    value = os.environ.get('SWFM_LOCAL_STATIC_ROI', '0')
    if value not in ('0', '1'): raise ValueError('SWFM_LOCAL_STATIC_ROI must be 0 or 1')
    return value == '1'


def motion_indices(record, device):
    keys = ('supervised_source', 'se2_target_valid', 'yaw_enabled', 'yaw_label_valid', 'target_source_mask_tube')
    if not enabled() or any(isinstance(record[k], torch.Tensor) and record[k].device.type != 'cpu' for k in keys):
        return None
    values = {k: torch.as_tensor(record[k]) for k in keys}
    sup = values['supervised_source'].bool()
    valid = values['se2_target_valid'].bool() & sup[:, None]
    yaw = valid & values['yaw_label_valid'].bool() & values['yaw_enabled'].bool()[:, None]
    footprint = values['target_source_mask_tube'][:, -1].float().flatten(1).sum(1) > 0
    shape = valid & footprint[:, None]
    masks = {'supervised': sup, 'valid': valid, 'yaw': yaw, 'shape': shape}
    return {k: torch.nonzero(v.flatten(), as_tuple=False).flatten().to(device) for k, v in masks.items()}


def column_indices(kind, legal, target, weight, device):
    """Validate original CPU supervision before upload; retain ascending order."""
    kind, legal, target, weight = map(np.asarray, (kind, legal, target, weight))
    if not np.isfinite(weight).all() or np.any(weight <= 0): raise ValueError('invalid sampling weights')
    if target.shape != legal.shape[:2] or weight.shape != kind.shape:
        raise ValueError('loss shape mismatch')
    if (target.dtype.kind not in 'iu' or np.any(target < 0) or np.any(target > 2)
            or not np.take_along_axis(legal, target[..., None], -1).all()):
        raise ValueError('illegal target action')
    rows = {}
    for task in (0, 1):
        ids = np.flatnonzero(kind == task)
        mask = legal[ids, :, 1] if task == 0 else legal[ids, :, 1] | legal[ids, :, 2]
        # Match the original float32 importance cast, including underflow.
        if len(ids) and np.any(mask & (weight[ids].astype(np.float32)[:, None] > 0)):
            rows[task] = torch.as_tensor(ids, device=device)
    return rows
