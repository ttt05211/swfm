"""Legacy boundary: prepared/uploaded histories -> fresh six CPU predictions."""
import time
import numpy as np
import torch
from real_motion.shared_source_track import kta_and_anchors
from tools.shared_source_track.adapter import backgrounds
from tools.joint_occ_experiments.soft_geometry import source_targets, effective_yaw
from tools.shared_history_experiments.refine.geometry import TransportPrior
from tools.source_scene_ego_conditioned.geometry import aligned_memory
from tools.source_scene_ego_conditioned.model import relative_ego_features
from tools.source_scene_speed.engine import prepare as prepare_base, sync, aggregate
from tools.fpn_unet.adapter import forward_window

BOUNDARY = dict(includes=['live_KTA', 'fresh_Strong_CLEAR', 'FPN_encoder', 'source_WM', 'history_lift',
    'transport_prior', 'aligned_memory', 'UNet_refiner', 'all6_argmax_CPU_output'],
    excludes=['disk_IO', 'historical_extraction_association_representation', 'initial_input_upload',
        'labels_metrics', 'startup_warmup', 'checkpoint_IO'],
    input='prepared FOUR real histories + explicit GT future ego', output='SIX full-grid CPU semantic frames',
    unit='output_frames_per_second = 6 / seconds_per_window', raw_camera_end_to_end=False,
    note='same declared legacy prepared-history boundary; NOT evidence identical kernels/hardware or old numerical FPS')


def prepare(row, poses, device):
    row, values = prepare_base(row, poses, device)
    values['current_pose'] = torch.as_tensor(np.asarray(row[3][0]['state']['current_pose']), device=device).double()
    sync(device)
    return row, values


@torch.no_grad()
def fresh(body, provider, ready):
    row, x = ready; rec, raw, _, history = row; state = history[0]['state']; device = provider.device
    # Associations are already prepared, but the causal anchor calculation,
    # learned encoder, ego condition and ALL future geometry are fresh each call.
    _, kta, anchors = kta_and_anchors(state['current'], raw['_shared_source_track'],
        state['current_pose'], dt_s=provider.pcfg.frame_dt_s)
    rec = dict(rec, kta_displacement_xy_m=kta, anchors_xy_t0_m=anchors)
    ego = relative_ego_features(x['current_pose'], x['poses'])
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        fields = body.transport.history_encoder(x['sem'], x['known'])
        output = body.transport(fields, x['reads'], x['times'], None,
            torch.as_tensor(kta, device=device).float(), ego)
        lifted = body.columns.lift_history(fields[0])
    bg = backgrounds(raw, state, x['poses'], provider, raw['_shared_source_track'])
    state = dict(state, column_backgrounds=bg)
    targets = source_targets(output, rec, state['current_pose'], state['source_z_t0'])
    yaws = effective_yaw(output, [int(s['class_id']) for s in state['current']])
    scene = TransportPrior(raw, state, provider.pcfg.grid, device)
    prior = scene.render(x['poses'], targets, yaws)
    memory = aligned_memory(body.columns, lifted, output, history, raw, targets, yaws, x['poses'], provider.pcfg.grid)
    logits = body.columns(memory, prior, range(6), scene.shape)
    return logits.argmax(1).reshape(6, *scene.shape).byte().cpu().numpy()


def report(samples, windows, repeats):
    results = {}
    for kind in ('occ', 'stc'):
        rows = [s for s in samples if s['modality'] == kind]
        ids = [(s['index'], s['repeat']) for s in rows]
        if set(ids) != {(i, r) for i in range(windows) for r in range(repeats)} or len(set(ids)) != len(ids):
            raise RuntimeError('incomplete/duplicated timing sample identities')
        results[kind] = aggregate(rows, windows*repeats)
    return results


@torch.no_grad()
def measure(body, predictor, inputs, jobs, keys, *, samples, windows=32, repeats=2, save=None, stop_event=None):
    done = {(s['modality'], s['index'], s['repeat']) for s in samples}
    if len(done) != len(samples): raise RuntimeError('duplicate saved timings')
    body.eval(); device = predictor.provider.device
    for index, job in enumerate(jobs[:windows]):
        for kind in (('occ', 'stc') if index % 2 == 0 else ('stc', 'occ')):
            if all((kind, index, r) in done for r in range(repeats)): continue
            if stop_event is not None and stop_event.is_set(): raise InterruptedError('complete timing call saved')
            rec, historical, times = inputs.history(job, kind)
            row = predictor.prepare(rec, historical, times=times); poses = np.asarray(inputs.poses(job))[:6]
            ready = prepare(row, poses, device)
            fw = forward_window(body, predictor.provider, row, poses=poses, fresh=True)
            reference = fw['logits'].argmax(1).reshape(6, *fw['shape']).byte().cpu().numpy(); del fw
            fw = forward_window(body, predictor.provider, row, poses=poses, fresh=True)
            repeat_reference = fw['logits'].argmax(1).reshape(6, *fw['shape']).byte().cpu().numpy(); del fw
            drift = int(np.count_nonzero(reference != repeat_reference))
            for _ in range(3 if index == 0 else 1): fresh(body, predictor.provider, ready)
            for r in range(repeats):
                if (kind, index, r) in done: continue
                if stop_event is not None and stop_event.is_set(): raise InterruptedError('complete timing call saved')
                sync(device)
                if device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device)
                tick = time.perf_counter(); predicted = fresh(body, predictor.provider, ready); sync(device)
                elapsed = time.perf_counter()-tick
                samples.append(dict(modality=kind, index=index, repeat=r, key=keys[index], seconds=elapsed,
                    changed_voxels=int(np.count_nonzero(predicted != reference)), reference_repeat_changed_voxels=drift,
                    peak_allocated_mib=torch.cuda.max_memory_allocated(device)/2**20 if device.type == 'cuda' else 0.,
                    peak_reserved_mib=torch.cuda.max_memory_reserved(device)/2**20 if device.type == 'cuda' else 0., profile={}))
                done.add((kind, index, r))
                if save: save()
            del row, ready, reference, repeat_reference, predicted
        if (index+1) % 8 == 0: print(f'FPN_UNET_SPEED {index+1}/{windows} both modalities', flush=True)
    return report(samples, windows, repeats)
