"""Current causal-forward motion path, with coverage-aware Refiner readout."""
import numpy as np
import torch
from tools.source_scene_ego_conditioned.model import relative_ego_features
from tools.shared_source_track.adapter import backgrounds
from tools.joint_occ_experiments.soft_geometry import source_targets, effective_yaw
from .geometry import CoveredPrior
from .memory import aligned_memory


def forward_window(model, provider, row, *, poses):
    rec, raw, ego, history = row; state = history[0]['state']; device = provider.device
    poses = torch.as_tensor(np.asarray(poses), device=device).double()
    current = torch.as_tensor(np.asarray(state['current_pose']), device=device).double()
    ego_features = relative_ego_features(current, poses)
    sem = torch.as_tensor(np.asarray(raw['history_occ']), device=device)
    known = torch.as_tensor(np.asarray(raw['history_observed']), device=device)
    reads = {k: torch.as_tensor(v, device=device) for k, v in history[0]['shared_reads'].items()}
    times = torch.as_tensor(ego[0][:, -1], device=device)
    kta = torch.as_tensor(np.asarray(rec['kta_displacement_xy_m']), device=device).float()
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        pyramid = model.transport.history_encoder(sem, known)
        output = model.transport(pyramid, reads, times, None, kta, ego_features)
        lifted = model.columns.lift_history(pyramid[0])
    bg = backgrounds(raw, state, poses, provider, raw['_shared_source_track'])
    state = dict(state, column_backgrounds=bg)
    targets = source_targets(output, rec, state['current_pose'], state['source_z_t0'])
    yaws = effective_yaw(output, [int(s['class_id']) for s in state['current']])
    scene = CoveredPrior(raw, state, provider.pcfg.grid, device)
    prior, pc = scene.render(poses, targets, yaws)
    memory, mc = aligned_memory(model.columns, lifted, output, history, raw, targets, yaws, poses, provider.pcfg.grid)
    logits = model.columns(memory, prior, range(6), scene.shape, prior_coverage=pc, memory_coverage=mc)
    return dict(output=output, targets=targets, yaws=yaws, prior=prior, memory=memory,
        logits=logits, shape=scene.shape, prior_coverage=pc, memory_coverage=mc)
