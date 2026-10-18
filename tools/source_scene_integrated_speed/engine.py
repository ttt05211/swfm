"""Latest no-ICP direct-history H6 entrance; no learned-output caching."""
import time
import torch
from real_motion.shared_source_track import kta_and_anchors
from tools.shared_source_track.adapter import backgrounds
from tools.joint_occ_experiments.soft_geometry import source_targets, effective_yaw
from tools.shared_history_experiments.refine.geometry import TransportPrior
from tools.source_scene_stage2.geometry import aligned_memory
from tools.source_scene_speed.engine import sync, parameters, prepare, aggregate


@torch.no_grad()
def fresh(body, provider, ready, profile=None):
    row, x = ready; rec, raw, _, history = row
    device = provider.device; state = history[0]['state']
    def stage(name, fn):
        if profile is None: return fn()
        sync(device); tick = time.perf_counter(); value = fn(); sync(device)
        profile[name] = time.perf_counter()-tick
        return value
    def live_kta():
        _, kta, anchors = kta_and_anchors(state['current'], raw['_shared_source_track'],
                                        state['current_pose'], dt_s=provider.pcfg.frame_dt_s)
        return dict(rec, kta_displacement_xy_m=kta, anchors_xy_t0_m=anchors)
    rec = stage('live_KTA', live_kta)
    def network():
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            field = body.transport.history_encoder(x['sem'], x['known'])
            output = body.transport(field, x['reads'], x['times'], x['classes'],
                                    torch.as_tensor(rec['kta_displacement_xy_m'], device=device).float())
            lifted = body.columns.lift_history(field)
        return output, lifted
    output, lifted = stage('encoder_motion_lift', network)
    bg = stage('fresh_Strong_CLEAR', lambda: backgrounds(raw, state, x['poses'], provider, raw['_shared_source_track']))
    state = dict(state, column_backgrounds=bg)
    def prior():
        targets = source_targets(output, rec, state['current_pose'], state['source_z_t0'])
        yaws = effective_yaw(output, [int(s['class_id']) for s in state['current']])
        scene = TransportPrior(raw, state, provider.pcfg.grid, device)
        return scene, targets, yaws, scene.render(x['poses'], targets, yaws)
    scene, targets, yaws, p = stage('transport', prior)
    memory = stage('memory', lambda: aligned_memory(body.columns, lifted, output, history, raw,
                   targets, yaws, x['poses'], provider.pcfg.grid))
    logits = stage('refiner', lambda: body.columns(memory, p, range(6), scene.shape))
    return stage('argmax_CPU', lambda: logits.argmax(1).reshape(6, *scene.shape).byte().cpu().numpy())
