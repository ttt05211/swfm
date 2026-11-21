"""Preserve clean motion and mask only the static evidence given to Refiner."""
import numpy as np
import torch

from tools.causal_forward_train.data import forward_window as clean_forward
from tools.joint_occ_experiments.soft_geometry import source_targets, effective_yaw
from tools.shared_history_experiments.refine.geometry import TransportPrior
from tools.shared_source_track.adapter import backgrounds
from tools.source_scene_ego_conditioned.geometry import aligned_memory
from tools.source_scene_ego_conditioned.model import relative_ego_features

from .masking import forward_cut, mask_history


def forward_window(model, provider, row, *, poses, depth=None):
    if depth is None:
        return clean_forward(model, provider, row, poses=poses)
    rec, raw, ego, history = row
    device, grid = provider.device, provider.pcfg.grid
    state = history[0]['state']
    future = torch.as_tensor(np.asarray(poses), device=device, dtype=torch.float64)
    current = torch.as_tensor(np.asarray(state['current_pose']), device=device, dtype=torch.float64)
    labels = torch.as_tensor(np.asarray(raw['history_occ']), device=device)
    known = torch.as_tensor(np.asarray(raw['history_observed']), device=device)
    reads = {k: torch.as_tensor(v, device=device) for k, v in history[0]['shared_reads'].items()}
    times = torch.as_tensor(ego[0][:, -1], device=device)
    kta = torch.as_tensor(np.asarray(rec['kta_displacement_xy_m']), device=device).float()
    prior_raw, memory_known, memory_history = mask_history(raw, history, grid, depth)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        fields = model.transport.history_encoder(labels, known)
        output = model.transport(fields, reads, times, None, kta, relative_ego_features(current, future))
        # Re-encode the visible inputs. Dropping only the warped Memory would
        # leave hidden semantic evidence in neighboring FPN feature vectors.
        masked_p2 = model.transport.history_encoder(labels, torch.as_tensor(memory_known, device=device))[0]
        lifted = model.columns.lift_history(masked_p2)

    targets = source_targets(output, rec, state['current_pose'], state['source_z_t0'])
    yaws = effective_yaw(output, [int(source['class_id']) for source in state['current']])
    unmasked_bg = backgrounds(raw, state, future, provider, raw['_shared_source_track'])
    to_t0 = np.linalg.inv(state['current_pose']) @ np.asarray(poses)
    # Strong can fill from other frames outside the t0 lattice. Hide that
    # fallback as well, before the moving-object overlays are composed.
    masked_bg = [np.where(forward_cut(grid, transform, depth), 17, bg)
                 for transform, bg in zip(to_t0, unmasked_bg)]
    scene = TransportPrior(prior_raw, dict(state, column_backgrounds=masked_bg), grid, device)
    prior = scene.render(future, targets, yaws)
    memory = aligned_memory(model.columns, lifted, output, memory_history, raw,
                            targets, yaws, future, grid)
    logits = model.columns(memory, prior, range(6), scene.shape)
    return dict(prior=prior, logits=logits, shape=scene.shape, targets=targets, yaws=yaws)
