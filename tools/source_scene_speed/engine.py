"""Same H6 geometry/readout for BOTH architectures; no learned-output caching."""
import time
import numpy as np
import torch
from real_motion.shared_source_track import kta_and_anchors
from tools.shared_source_track.adapter import backgrounds
from tools.joint_occ_experiments.soft_geometry import source_targets, effective_yaw
from tools.shared_history_experiments.refine.geometry import TransportPrior, aligned_memory


def sync(device):
    if torch.device(device).type == 'cuda': torch.cuda.synchronize(device)


def parameters(model):
    encoder = sum(p.numel() for p in model.transport.history_encoder.parameters())
    motion = sum(p.numel() for p in model.transport.parameters())-encoder
    refiner = sum(p.numel() for p in model.columns.parameters())
    total = sum(p.numel() for p in model.parameters())
    if total != encoder+motion+refiner: raise ValueError('unexpected extra learned module')
    return dict(total=total, encoder=encoder, motion_ex_encoder=motion, refiner=refiner)


def prepare(row, poses, device):
    rec, raw, ego, history = row
    if any(k in raw for k in ('future_gt_occ', 'future_observed', 'future_poses')):
        raise ValueError('historical inputs ONLY; future ego supplied separately')
    if 'column_backgrounds' in history[0]['state']: raise ValueError('cached future background forbidden')
    upload = lambda x: torch.as_tensor(np.asarray(x), device=device)
    staged = dict(sem=upload(raw['history_occ']), known=upload(raw['history_observed']),
                  reads={k: upload(v) for k, v in history[0]['shared_reads'].items()},
                  times=upload(ego[0][:, -1]), classes=upload(rec['source_class_id']).long(),
                  poses=upload(poses).double())
    if staged['poses'].shape != (6, 4, 4) or not bool(torch.isfinite(staged['poses']).all()):
        raise ValueError('six finite future ego conditions required')
    sync(device)
    return row, staged


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


def aggregate(rows, expected):
    if len(rows) != expected or not rows: raise ValueError('incomplete FPS samples')
    seconds = np.array([r['seconds'] for r in rows])
    if not np.isfinite(seconds).all() or np.any(seconds <= 0): raise ValueError('invalid FPS elapsed')
    profiles = [r['profile'] for r in rows if r['profile']]
    return dict(calls=len(rows), six_frame_ms=float(seconds.mean()*1000),
        p90_six_frame_ms=float(np.percentile(seconds, 90)*1000), window_fps=float(len(rows)/seconds.sum()),
        output_frame_fps=float(6*len(rows)/seconds.sum()),
        byte_failed_calls=sum(r['changed_voxels'] > 0 for r in rows),
        max_changed_voxels=max(r['changed_voxels'] for r in rows),
        reference_repeat_max_changed_voxels=max(r['reference_repeat_changed_voxels'] for r in rows),
        strict_byte_parity=not any(r['changed_voxels'] or r['reference_repeat_changed_voxels'] for r in rows),
        profile_mean_ms={k: float(np.mean([p[k] for p in profiles])*1000) for k in profiles[0]} if profiles else {},
        peak_allocated_mib=max(r['peak_allocated_mib'] for r in rows),
        peak_reserved_mib=max(r['peak_reserved_mib'] for r in rows))
