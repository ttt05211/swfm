"""Encode once; fixed Q/history yaw, intervene ONLY future XY/yaw outputs."""
import numpy as np
import torch
from tools.source_scene_stage2.adapter import forward_window
from tools.source_scene_stage2.geometry import aligned_memory
from tools.shared_history_experiments.refine.geometry import TransportPrior
from tools.joint_occ_experiments.soft_geometry import source_targets, effective_yaw

MODES = ('learned', 'KTA_future', 'XY_only', 'yaw_only')


def intervention(output, mode):
    if mode not in MODES: raise ValueError('unknown fixed diagnostic mode')
    changed = dict(output)
    if mode in ('KTA_future', 'yaw_only'): changed['residual_xy_m'] = torch.zeros_like(output['residual_xy_m'])
    if mode in ('KTA_future', 'XY_only'): changed['yaw_delta_rad'] = torch.zeros_like(output['yaw_delta_rad'])
    return changed


@torch.no_grad()
def predictions(body, provider, row, poses):
    # The learned route is the ORIGINAL live forward, not a reimplementation.
    fw = forward_window(body, provider, row, poses=poses, fresh=True)
    rec, raw, _, history = row; state = fw['state']; dest = fw['poses']
    result = {}; output = fw['output']
    def decode(mode, prior, logits):
        for name, value in (('transport', prior), ('semantic_refine', logits)):
            result[mode+'/'+name] = value.argmax(1).reshape(6, *fw['shape']).byte().cpu().numpy()
    decode('learned', fw['prior'], fw['logits'])
    # These large tensors are not reused by the counterfactual routes.
    del fw['prior'], fw['memory'], fw['logits']
    # No future labels/boxes read here. Background/field/lift/Q/history yaw are
    # identical; only future source motion is changed for all dependent paths.
    renderer = TransportPrior(raw, state, provider.pcfg.grid, provider.device)
    for mode in MODES[1:]:
        alternative = intervention(output, mode)
        targets = source_targets(alternative, rec, state['current_pose'], state['source_z_t0'])
        yaws = effective_yaw(alternative, rec['source_class_id'])
        prior = renderer.render(dest, targets, yaws)
        memory = aligned_memory(body.columns, fw['lifted'], alternative, history, raw,
                                targets, yaws, dest, provider.pcfg.grid)
        logits = body.columns(memory, prior, range(6), renderer.shape)
        decode(mode, prior, logits); del prior, memory, logits
    current = np.asarray([s['centroid_world'] for s in state['current']]).reshape(-1, 3)
    prior_out = intervention(output, 'KTA_future')
    prior_targets = source_targets(prior_out, rec, state['current_pose'], state['source_z_t0'])
    diagnostic = dict(residual_xy_m=output['residual_xy_m'].float().cpu().numpy(),
        effective_future_yaw_rad=fw['yaws'].float().cpu().numpy(),
        raw_future_yaw_rad=output['yaw_delta_rad'].float().cpu().numpy(),
        history_yaw_rad=output['history_yaw_to_t0_rad'].float().cpu().numpy(),
        history_valid=np.asarray(raw['_shared_source_track'].track_valid),
        learned_world_displacement=fw['targets'].cpu().numpy()-current[:, None],
        KTA_world_displacement=prior_targets.cpu().numpy()-current[:, None])
    repeat = forward_window(body, provider, row, poses=poses, fresh=True)
    diagnostic['reference_repeat_changed_voxels'] = {
        name: int(np.count_nonzero(result['learned/'+name] !=
            repeat[key].argmax(1).reshape(6, *repeat['shape']).byte().cpu().numpy()))
        for name, key in (('transport', 'prior'), ('semantic_refine', 'logits'))}
    return result, diagnostic
