"""GT-only joint CE/Lovasz pilot; continuous ADD-only in train AND inference."""
from collections import defaultdict
import time
import numpy as np
import torch
from tools.joint_occ_experiments.data import prepare_projection
from tools.joint_occ_experiments.model import source_auxiliary, ego_geometry_loss
from tools.joint_occ_experiments.soft_geometry import source_targets, effective_yaw, continuous_context
from tools.joint_occ_experiments.fast_autograd import RenderGradientBridge
from tools.joint_occ_experiments.batching.geometry import GroupedScene, grouped_loss
from tools.joint_occ_experiments.soft_inference.fast import predict
from tools.joint_occ_experiments.soft_inference.engine import soft_predictions
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import Metrics
from tools.real_motion.eval_p0_f9_v17_local_stwm import window_from_record
from tools.real_motion.shared_evidence_pilot_common import moving_support_masks
from real_motion.nuscenes_adapter import gt_moving_support_sequence
from .model import encode_window
from .model import SharedJoint
from tools.joint_occ_experiments.engine import forward_window as legacy_forward


def forward_window(model, provider, row):
    if not isinstance(model, SharedJoint):
        fw = legacy_forward(model, provider, row, 'gt', soft=True)
        fw['actions'] = torch.stack((fw['actions'][..., 0], torch.zeros_like(fw['actions'][..., 1])), -1)
        return fw
    record, raw, ego_inputs, history = row
    e, commands, target = ego_inputs
    device = provider.device
    cast = lambda x: torch.as_tensor(np.asarray(x), device=device)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        field, reads, times = encode_window(model, raw, history, e, device)
        output = model.transport(field, reads, times, cast(record['source_class_id']).long(),
                                 cast(record['kta_displacement_xy_m']).float())
        ego = model.ego(cast(e[None]), cast(commands[None]))['se2'][0]
        encoded = model.columns.encode_history(field, reads, times, history[1])
    poses = cast(raw['future_poses']).double()
    prep, evidence, plan = prepare_projection(provider, record, raw, output,
        poses.detach().cpu().numpy(), history, use_gt_cache=True)
    targets = source_targets(output, record, prep.state['current_pose'], prep.state['source_z_t0'])
    yaws = effective_yaw(output, [int(c['class_id']) for c in prep.state['current']])
    mapped, context = continuous_context(evidence, plan, prep.state, targets, yaws, poses, provider.pcfg.grid)
    actor = cast(evidence.actor).long()
    # Decode bounded chunks; E/history encoding is shared and never repeated.
    actions = []
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        for first in range(0, len(evidence), 8192):
            at = slice(first, first+8192)
            actions.append(model.columns(encoded[at], output, actor[at], context[at],
                cast(plan.base[at]), cast(plan.legal[at])))
    actions = torch.cat(actions) if actions else field.new_empty((0, 6, 2), dtype=torch.float32)
    return dict(output=output, ego=ego, ego_target=target, poses=poses, targets=targets, yaws=yaws,
        prep=prep, evidence=evidence, plan=plan, mapped=mapped, actions=actions, field=field)


def train_arm(model, optimizer, provider, rows, mode, *, update, total_updates,
              object_weight=1., ego_weight=1., lovasz_weight=1., probe=False, optimized=True):
    if mode != 'gt' or not optimized or not rows: raise ValueError('GT-only fused shared-history pilot required')
    model.train(); optimizer.zero_grad(set_to_none=True)
    factor = .01+.99*.5*(1+np.cos(np.pi*(update-1)/max(1, total_updates)))
    for group in optimizer.param_groups: group['lr'] = group['initial_lr']*factor
    begun = time.perf_counter()
    stages, terms, links = defaultdict(float), defaultdict(list), defaultdict(float)
    sources = canonical = 0
    for row in rows:
        tick = time.perf_counter(); fw = forward_window(model, provider, row)
        scene = GroupedScene(fw['prep'], provider.pcfg.grid, provider.device)
        bridge = RenderGradientBridge(fw)
        stages['shared_encoder_motion_projection_ccr' if isinstance(model, SharedJoint)
               else 'legacy_history_motion_projection_ccr'] += time.perf_counter()-tick
        sources += len(fw['output']['residual_xy_m']); canonical += len(fw['evidence'])
        labels = np.asarray(row[1]['future_gt_occ'])
        for first in (0, 2, 4):
            hs = [first, first+1]; tick = time.perf_counter()
            probability = scene.render_group(hs, **bridge.leaves, evidence=fw['evidence'], plan=fw['plan'])
            losses, ce, lovasz = grouped_loss(probability, labels[hs], lovasz_weight=lovasz_weight)
            if not bool(torch.isfinite(losses).all()): raise RuntimeError('nonfinite full-grid occupancy objective')
            if probe and first == 0 and losses.requires_grad:
                tensors, names = [fw['actions']], ['occupancy_to_actions']
                if 'field' in fw:
                    tensors += [fw['field']]; names += ['occupancy_to_shared_history']
                if len(fw['output']['residual_xy_m']):
                    tensors += [fw['output']['residual_xy_m'], fw['output']['yaw_delta_rad']]
                    names += ['occupancy_to_xy', 'occupancy_to_yaw']
                for name, grad in zip(names, bridge.probe(losses[1], tensors)):
                    links[name] += 0. if grad is None else float(grad.detach().float().norm())
            stages['soft_render_full_ce_lovasz'] += time.perf_counter()-tick
            tick = time.perf_counter()
            if losses.requires_grad: (losses/(len(rows)*6)).sum().backward()
            stages['occupancy_backward'] += time.perf_counter()-tick
            terms['CE'].extend(ce.detach().unbind()); terms['Lovasz'].extend(lovasz.detach().unbind())
            del probability, losses, ce, lovasz
        tick = time.perf_counter()
        geometry, existence = source_auxiliary(fw['output'], row[0], fw['prep'].state)
        ego_loss = ego_geometry_loss(fw['ego'][None], np.asarray(fw['ego_target'])[None])
        auxiliary = object_weight*(geometry+existence)+ego_weight*ego_loss
        if not bool(torch.isfinite(auxiliary)): raise RuntimeError('nonfinite direct geometry objective')
        bridge.backward_shared(auxiliary/len(rows))
        terms['source_geometry'].append(geometry.detach()); terms['existence'].append(existence.detach())
        terms['ego_geometry'].append(ego_loss.detach())
        stages['shared_network_and_direct_geometry_backward'] += time.perf_counter()-tick
        del fw, scene, bridge, geometry, existence, ego_loss, auxiliary
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
    tick = time.perf_counter(); optimizer.step()
    if provider.device.type == 'cuda': torch.cuda.synchronize(provider.device)
    stages['optimizer_finish'] += time.perf_counter()-tick
    result = {key: float(torch.stack(value).mean()) for key, value in terms.items()}
    result.update(mode=mode, windows=len(rows), sources=sources, canonical_points=canonical,
        seconds=time.perf_counter()-begun, stages_seconds=dict(stages), grad_norm=float(norm),
        gradient_links=dict(links), training_backend='shared_history_sequential_two_horizons_ADDonly',
        peak_memory_mib=torch.cuda.max_memory_allocated(provider.device)/2**20 if provider.device.type == 'cuda' else 0.)
    optimizer.zero_grad(set_to_none=True)
    return result


@torch.no_grad()
def evaluate(models, provider, source, batches, *, remove_threshold=.95, stop_event=None,
             progress=None, saved=None, save=None):
    if set(models) != {'gt'}: raise ValueError('GT-only first shared prototype')
    model = models['gt']; model.eval()
    metrics = {name: Metrics() for name in ('soft_transport', 'soft_add_only')}
    windows, errors, verified = 0, [], False
    if saved:
        if saved.get('protocol') != 'shared_history_soft_ADDonly_eval_v1' or set(saved['metrics']) != set(metrics):
            raise RuntimeError('shared evaluation protocol/readouts differ')
        windows, errors, verified = int(saved['windows']), list(saved['errors']), saved['verified']
        if windows < 0 or len(errors) != windows or windows and verified is not True:
            raise RuntimeError('invalid completed evaluation cursor')
        for name, fields in saved['metrics'].items():
            if set(fields) != set(vars(metrics[name])): raise RuntimeError('metric fields differ')
            for key, value in fields.items():
                array = np.asarray(value, np.int64)
                if array.shape != getattr(metrics[name], key).shape or np.any(array < 0) or not np.array_equal(array, value):
                    raise RuntimeError('invalid integer evaluation counts')
                setattr(metrics[name], key, array)
    def snapshot():
        return dict(protocol='shared_history_soft_ADDonly_eval_v1', windows=windows, errors=list(errors), verified=verified,
            metrics={n: {k: v.tolist() for k, v in vars(m).items()} for n, m in metrics.items()})
    begun, offset = time.perf_counter(), 0
    for rows in batches:
        for row in rows:
            offset += 1
            if offset <= windows: continue
            if stop_event is not None and stop_event.is_set():
                if save: save(snapshot())
                raise InterruptedError('saved after complete shared evaluation window')
            fw = forward_window(model, provider, row)
            predictions = {n: predict(provider, fw, n, range(6), group_size=3) for n in ('transport', 'add_only')}
            if not verified:
                reference = soft_predictions(provider, fw, tuple(predictions), range(6))
                if any(not np.array_equal(a, b) for name in predictions for a, b in zip(predictions[name], reference[name])):
                    raise RuntimeError('shared fast soft six-frame byte mismatch')
            # Future labels/support only enter scoring after BOTH predictions.
            raw = row[1]; window = window_from_record(row[0])
            support = gt_moving_support_sequence(source.nusc, window.t0_token, window.future_tokens,
                tuple(.5*(h+1) for h in range(6)), grid=provider.pcfg.grid, workers=provider.workers)
            moving = moving_support_masks(support, raw['future_gt_occ'].shape[1:])
            for name in predictions:
                for i, h in enumerate((1, 3, 5)):
                    metrics['soft_'+name].update(i, predictions[name][h], raw['future_gt_occ'][h], moving[h])
            error = fw['ego'].float().cpu().numpy()-np.asarray(fw['ego_target'])
            errors.append(dict(xy=np.linalg.norm(error[:, :2], axis=-1).tolist(),
                yaw=(np.abs(np.arctan2(np.sin(error[:, 2]), np.cos(error[:, 2])))*180/np.pi).tolist()))
            windows += 1; verified = True
            if windows % 8 == 0 and save: save(snapshot())
            if windows % 16 == 0:
                if progress: progress(dict(event='shared_history_eval', windows=windows))
                print(f'SHARED_HISTORY_EVAL {windows} '+str({n: m.compute()['mIoU'] for n, m in metrics.items()}), flush=True)
    if save: save(snapshot())
    return dict(windows=windows, variants={'gt': {n: m.compute() for n, m in metrics.items()}},
        trajectory={'gt': dict(xy_mean_m=np.mean([e['xy'] for e in errors], 0).tolist(),
            yaw_mean_deg=np.mean([e['yaw'] for e in errors], 0).tolist())} if errors else {},
        seconds=time.perf_counter()-begun, fast_six_frame_byte_parity=verified,
        main_readout='soft_add_only', scoring='full grid / future GT ego conditioned / NO metric mask')
