"""Original baseline live forward; fixed Q/ICP, intervene only future XY/yaw."""
import numpy as np
import torch
from tools.shared_source_track.adapter import forward_conditioned
from tools.shared_history_experiments.refine.geometry import TransportPrior, aligned_memory
from tools.joint_occ_experiments.soft_geometry import source_targets, effective_yaw
from tools.source_motion_audit.engine import MODES, intervention


@torch.no_grad()
def predictions(body, provider, row, poses):
    fw = forward_conditioned(body, provider, row, poses)
    rec, raw, _, history = row; state = fw['state']; dest = fw['poses']; output = fw['output']
    result = {}
    def decode(mode, prior, logits):
        for name, value in (('transport', prior), ('semantic_refine', logits)):
            result[mode+'/'+name] = value.argmax(1).reshape(6, *fw['shape']).byte().cpu().numpy()
    decode('learned', fw['prior'], fw['logits'])
    del fw['prior'], fw['memory'], fw['logits']
    renderer = TransportPrior(raw, state, provider.pcfg.grid, provider.device)
    for mode in MODES[1:]:
        changed = intervention(output, mode)
        targets = source_targets(changed, rec, state['current_pose'], state['source_z_t0'])
        yaws = effective_yaw(changed, rec['source_class_id'])
        prior = renderer.render(dest, targets, yaws)
        memory = aligned_memory(body.columns, fw['lifted'], changed, history, raw, targets, yaws, dest, provider.pcfg.grid)
        logits = body.columns(memory, prior, range(6), renderer.shape)
        decode(mode, prior, logits); del prior, memory, logits
    current = np.asarray([s['centroid_world'] for s in state['current']]).reshape(-1, 3)
    kta = source_targets(intervention(output, 'KTA_future'), rec, state['current_pose'], state['source_z_t0'])
    diagnostic = dict(residual_xy_m=output['residual_xy_m'].float().cpu().numpy(),
        effective_future_yaw_rad=fw['yaws'].float().cpu().numpy(),
        learned_world_displacement=fw['targets'].cpu().numpy()-current[:, None],
        KTA_world_displacement=kta.cpu().numpy()-current[:, None])
    repeat = forward_conditioned(body, provider, row, poses)
    diagnostic['reference_repeat_changed_voxels'] = {
        name: int(np.count_nonzero(result['learned/'+name] !=
            repeat[key].argmax(1).reshape(6, *repeat['shape']).byte().cpu().numpy()))
        for name, key in (('transport', 'prior'), ('semantic_refine', 'logits'))}
    return result, diagnostic
