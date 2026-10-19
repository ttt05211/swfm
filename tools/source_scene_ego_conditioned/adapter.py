"""ONE future pose tensor conditions the decoder, background, prior and memory."""
import numpy as np
import torch
from tools.shared_history_experiments.model import encode_window
from tools.shared_history_experiments.refine.geometry import TransportPrior
from tools.shared_source_track.adapter import backgrounds
from tools.joint_occ_experiments.soft_geometry import source_targets, effective_yaw
from .geometry import aligned_memory
from .model import relative_ego_features
from tools.source_motion_audit.engine import intervention


def render_future(model, provider, row, prepared, mode='learned'):
    rec, raw, _, history = row; state = prepared['state']; poses = prepared['poses']
    output = intervention(prepared['output'], mode)
    targets = source_targets(output, rec, state['current_pose'], state['source_z_t0'])
    yaws = effective_yaw(output, [int(s['class_id']) for s in state['current']])
    scene = TransportPrior(raw, state, provider.pcfg.grid, provider.device)
    prior = scene.render(poses, targets, yaws)
    memory = aligned_memory(model.columns, prepared['lifted'], output, history, raw, targets, yaws, poses, provider.pcfg.grid)
    logits = model.columns(memory, prior, range(6), scene.shape)
    return dict(prepared, output=output, targets=targets, yaws=yaws, prior=prior, memory=memory, logits=logits, shape=scene.shape)


def forward_window(model, provider, row, *, poses=None, fresh=False):
    rec, raw, ego, history = row; state = history[0]['state']; device = provider.device
    # An explicit trajectory (including future Pred use) NEVER reuses cached GT backgrounds.
    fresh = fresh or poses is not None
    poses = torch.as_tensor(np.asarray(raw['future_poses']) if poses is None else poses, device=device).double()
    current_pose = torch.as_tensor(np.asarray(state['current_pose']), device=device).double()
    ego_features = relative_ego_features(current_pose, poses)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        field, reads, times = encode_window(model, raw, history, ego[0], device)
        kta = torch.as_tensor(np.asarray(rec['kta_displacement_xy_m']), device=device).float()
        output = model.transport(field, reads, times, None, kta, ego_features)
        lifted = model.columns.lift_history(field)
    bg = (backgrounds(raw, state, poses, provider, raw['_shared_source_track']) if fresh else
          raw['_column_causal_preparation']['prepared_state']['column_backgrounds'])
    state = dict(state, column_backgrounds=bg)
    return render_future(model, provider, row, dict(field=field, output=output, lifted=lifted, state=state, poses=poses, ego_features=ego_features))
