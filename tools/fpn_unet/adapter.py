"""Unchanged causal geometry; encode FPN once and share P2 with history lift."""
import numpy as np
import torch
from tools.source_scene_ego_conditioned.model import relative_ego_features
from tools.source_scene_ego_conditioned.adapter import render_future
from tools.shared_source_track.adapter import backgrounds


def forward_window(model, provider, row, *, poses=None, fresh=False):
    rec, raw, ego, history = row; state = history[0]['state']; device = provider.device
    fresh = fresh or poses is not None
    poses = torch.as_tensor(np.asarray(raw['future_poses']) if poses is None else poses, device=device).double()
    current = torch.as_tensor(np.asarray(state['current_pose']), device=device).double()
    ego_features = relative_ego_features(current, poses)
    semantics = torch.as_tensor(np.asarray(raw['history_occ']), device=device)
    observed = torch.as_tensor(np.asarray(raw['history_observed']), device=device)
    reads = {k: torch.as_tensor(v, device=device) for k, v in history[0]['shared_reads'].items()}
    times = torch.as_tensor(ego[0][:, -1], device=device)
    kta = torch.as_tensor(np.asarray(rec['kta_displacement_xy_m']), device=device).float()
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        pyramid = model.transport.history_encoder(semantics, observed)
        output = model.transport(pyramid, reads, times, None, kta, ego_features)
        field = pyramid[0]
        lifted = model.columns.lift_history(field)
    bg = (backgrounds(raw, state, poses, provider, raw['_shared_source_track']) if fresh else
        raw['_column_causal_preparation']['prepared_state']['column_backgrounds'])
    state = dict(state, column_backgrounds=bg)
    return render_future(model, provider, row, dict(field=field, output=output, lifted=lifted,
        state=state, poses=poses, ego_features=ego_features))
