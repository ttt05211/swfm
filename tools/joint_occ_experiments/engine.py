"""Paired clean training and fixed hard-output evaluation of the new method."""
from collections import defaultdict
import time
import numpy as np
import torch
from real_motion.canonical_causal_repair import compose_canonical
from real_motion.surface_canonical_repair import augment_projection
from real_motion.nuscenes_adapter import gt_moving_support_sequence
from tools.real_motion.shared_evidence_pilot_common import moving_support_masks
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import Metrics
from tools.real_motion.eval_p0_f9_v17_local_stwm import window_from_record
from .model import torch_poses, gradient_scale, ego_geometry_loss, source_auxiliary
from .data import prepare_projection
from .soft_geometry import (source_targets, effective_yaw, continuous_context, SoftScene, occupancy_loss)


def forward_window(model, provider, row, mode, *, ego_gradient=1., soft=True):
    record, raw, ego_inputs, history = row
    e, commands, target = ego_inputs
    device = provider.device
    output = model.motion(record, device)
    ego = model.ego(torch.as_tensor(e[None], device=device), torch.as_tensor(commands[None], device=device))['se2'][0]
    if mode == 'gt':
        poses = torch.as_tensor(np.asarray(raw['future_poses']), device=device, dtype=torch.float64)
    elif mode == 'pred':
        poses = torch_poses(raw['history_poses'][-1], gradient_scale(ego, ego_gradient))
    else:
        raise ValueError('GT or Pred geometry configuration required')
    prep, evidence, plan = prepare_projection(provider, record, raw, output,
        poses.detach().cpu().numpy(), history, use_gt_cache=mode == 'gt')
    targets = source_targets(output, record, prep.state['current_pose'], prep.state['source_z_t0'])
    yaws = effective_yaw(output, [int(c['class_id']) for c in prep.state['current']])
    if soft:
        mapped, context = continuous_context(evidence, plan, prep.state, targets, yaws, poses, provider.pcfg.grid)
    else:
        enriched = augment_projection(evidence, plan, prep.state['current_pose'], prep.state['world_to_future'], provider.pcfg.grid)
        context = torch.as_tensor(enriched.context, device=device)
        mapped = None
        session = getattr(provider, 'paired_eval_sessions', {}).get(id(model.columns))
        if session is not None:
            actions = torch.from_numpy(session(model.columns, evidence, enriched, output, device))
            return dict(output=output, ego=ego, poses=poses, targets=targets, yaws=yaws,
                prep=prep, evidence=evidence, plan=plan, mapped=None, actions=actions, ego_target=target)
    upload = lambda value: torch.as_tensor(np.array(value, copy=True), device=device)
    actor = upload(evidence.actor).long()
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        linked = model.columns.project_sources(output)
        encoded = model.columns.encode(upload(evidence.features), upload(evidence.labels), actor, upload(evidence.classes), linked)
        logits = model.columns.decode(encoded, actor, context, upload(plan.base), upload(plan.fallback), upload(plan.legal), linked)
    actions = model.columns.probabilities(logits, actor)
    return dict(output=output, ego=ego, poses=poses, targets=targets, yaws=yaws,
        prep=prep, evidence=evidence, plan=plan, mapped=mapped, actions=actions,
        ego_target=target)


def train_arm(model, optimizer, provider, rows, mode, *, update, total_updates,
              object_weight=1., ego_weight=1., lovasz_weight=1., probe=False):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    # Whole configured cycle; never shorten the schedule for a smoke/stop.
    factor = .01+.99*.5*(1+np.cos(np.pi*(update-1)/max(1, total_updates)))
    for group in optimizer.param_groups: group['lr'] = group['initial_lr']*factor
    begun = time.perf_counter()
    stages, terms, links = defaultdict(float), defaultdict(list), defaultdict(float)
    sources = canonical = 0
    ramp = .1+.9*min(1., (update-1)/max(1., total_updates/5))
    for row in rows:
        tick = time.perf_counter()
        fw = forward_window(model, provider, row, mode, ego_gradient=ramp)
        scene = SoftScene(fw['prep'], provider.pcfg.grid, provider.device)
        stages['motion_ego_projection_head'] += time.perf_counter()-tick
        sources += len(row[0]['features'])
        canonical += len(fw['evidence'])
        # Labels enter AFTER forward/candidate/owner construction, as targets.
        labels = torch.as_tensor(np.asarray(row[1]['future_gt_occ']), device=provider.device)
        for h in range(6):
            tick = time.perf_counter()
            probability = scene.render(h, fw['poses'][h], fw['targets'], fw['yaws'], fw['mapped'],
                                       fw['evidence'], fw['plan'], fw['actions'])
            classes = [int(c) for c in np.unique(row[1]['future_gt_occ'][h]) if c < 17]
            loss, ce, lovasz = occupancy_loss(probability, labels[h], lovasz_weight=lovasz_weight, class_ids=classes)
            if not bool(torch.isfinite(loss)): raise RuntimeError('nonfinite final occupancy loss')
            if probe and h == 1 and loss.requires_grad:
                tensors = [fw['actions']]
                names = ['occupancy_to_actions']
                if fw['output']['residual_xy_m'].requires_grad:
                    tensors += [fw['output']['residual_xy_m'], fw['output']['yaw_delta_rad']]
                    names += ['occupancy_to_xy', 'occupancy_to_yaw']
                if mode == 'pred': tensors += [fw['ego']]; names += ['occupancy_to_ego']
                gradients = torch.autograd.grad(loss, tensors, retain_graph=True, allow_unused=True)
                for name, gradient in zip(names, gradients):
                    links[name] += 0. if gradient is None else float(gradient.detach().float().norm())
            stages['soft_render_full_ce_lovasz'] += time.perf_counter()-tick
            tick = time.perf_counter()
            if loss.requires_grad: (loss/(len(rows)*6)).backward(retain_graph=True)
            stages['occupancy_backward'] += time.perf_counter()-tick
            terms['CE'].append(ce.detach()); terms['Lovasz'].append(lovasz.detach())
            del probability, loss, ce, lovasz
        tick = time.perf_counter()
        geometry, existence = source_auxiliary(fw['output'], row[0], fw['prep'].state)
        ego_loss = ego_geometry_loss(fw['ego'][None], np.asarray(fw['ego_target'])[None])
        auxiliary = object_weight*(geometry+existence)+ego_weight*ego_loss
        if not bool(torch.isfinite(auxiliary)): raise RuntimeError('nonfinite direct pose loss')
        (auxiliary/len(rows)).backward()
        terms['source_geometry'].append(geometry.detach()); terms['existence'].append(existence.detach())
        terms['ego_geometry'].append(ego_loss.detach())
        stages['direct_geometry_backward'] += time.perf_counter()-tick
        del fw, scene, labels, geometry, existence, ego_loss, auxiliary
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
    tick = time.perf_counter()
    optimizer.step()
    if provider.device.type == 'cuda': torch.cuda.synchronize(provider.device)
    stages['optimizer_finish'] += time.perf_counter()-tick
    result = {key: float(torch.stack(value).mean()) for key, value in terms.items()}
    result.update(mode=mode, windows=len(rows), sources=sources, canonical_points=canonical,
        seconds=time.perf_counter()-begun, stages_seconds=dict(stages), grad_norm=float(norm),
        ego_occupancy_gradient_scale=ramp, gradient_links=dict(links),
        peak_memory_mib=torch.cuda.max_memory_allocated(provider.device)/2**20 if provider.device.type == 'cuda' else 0.)
    optimizer.zero_grad(set_to_none=True)
    return result


@torch.no_grad()
def hard_predictions(fw, *, remove_threshold=.95):
    probability = fw['actions'].float().cpu().numpy()
    common = fw['prep'].baseline, fw['evidence'], fw['plan'], probability[..., 0], probability[..., 1]
    return dict(transport=fw['prep'].baseline,
        add_only=compose_canonical(*common, thresholds=(.5, None)),
        add_remove=compose_canonical(*common, thresholds=(.5, remove_threshold)))


@torch.no_grad()
def evaluate(models, provider, source, batches, *, remove_threshold=.95, stop_event=None, progress=None,
             saved=None, save=None):
    """One shared input pass; fixed ADD/REMOVE settings, no threshold search."""
    metrics = {mode: {variant: Metrics() for variant in ('transport', 'add_only', 'add_remove')} for mode in models}
    gap = {mode: {variant: Metrics() for variant in ('soft_argmax', 'hard_same_windows')} for mode in models}
    gap_windows = 0
    errors, windows = {mode: [] for mode in models}, 0
    if saved:
        if set(saved['metrics']) != set(models): raise RuntimeError('evaluation arm mismatch')
        windows = int(saved['windows'])
        gap_windows = int(saved.get('gap_windows', 0))
        for mode, variants in saved.get('gap', {}).items():
            for variant, values in variants.items():
                for key, value in values.items(): setattr(gap[mode][variant], key, np.asarray(value, np.int64))
        for mode in models:
            errors[mode] = saved['errors'][mode]
            for variant in metrics[mode]:
                for key, value in saved['metrics'][mode][variant].items():
                    setattr(metrics[mode][variant], key, np.asarray(value, np.int64))
    def snapshot():
        return dict(windows=windows, errors=errors, gap_windows=gap_windows,
            gap={mode: {variant: {key: value.tolist() for key, value in vars(m).items()}
                        for variant, m in variants.items()} for mode, variants in gap.items()},
            metrics={mode: {variant: {key: value.tolist() for key, value in vars(m).items()}
                            for variant, m in variants.items()} for mode, variants in metrics.items()})
    for model in models.values(): model.eval()
    start = time.perf_counter()
    offset = 0
    for rows in batches:
        for row in rows:
            offset += 1
            if offset <= windows: continue
            if stop_event is not None and stop_event.is_set():
                if save: save(snapshot())
                raise InterruptedError('evaluation saved after complete window')
            outputs = {}
            soft_outputs = {}
            for mode, model in models.items():
                fw = forward_window(model, provider, row, mode, soft=False)
                outputs[mode] = hard_predictions(fw, remove_threshold=remove_threshold)
                if gap_windows < 8:
                    mapped, _ = continuous_context(fw['evidence'], fw['plan'], fw['prep'].state,
                        fw['targets'], fw['yaws'], fw['poses'], provider.pcfg.grid)
                    scene = SoftScene(fw['prep'], provider.pcfg.grid, provider.device)
                    actions = fw['actions'].to(provider.device)
                    soft_outputs[mode] = [scene.render(h, fw['poses'][h], fw['targets'], fw['yaws'],
                        mapped, fw['evidence'], fw['plan'], actions).argmax(0).reshape(scene.shape).byte().cpu().numpy()
                        for h in (1, 3, 5)]
                    del scene, actions, mapped
                predicted = fw['ego'].float().cpu().numpy()
                target = np.asarray(fw['ego_target'])
                errors[mode].append(dict(xy=np.linalg.norm(predicted[:, :2]-target[:, :2], axis=-1).tolist(),
                    yaw=(np.abs(np.arctan2(np.sin(predicted[:, 2]-target[:, 2]), np.cos(predicted[:, 2]-target[:, 2])))*180/np.pi).tolist()))
            # Future GT/support are metric-only, read after BOTH predictions.
            record, raw = row[:2]
            window = window_from_record(record)
            moving_rows = gt_moving_support_sequence(source.nusc, window.t0_token, window.future_tokens,
                tuple(.5*(h+1) for h in range(6)), grid=provider.pcfg.grid, workers=provider.workers)
            moving = moving_support_masks(moving_rows, raw['future_gt_occ'].shape[1:])
            for mode, variants in outputs.items():
                for name, prediction in variants.items():
                    for i, h in enumerate((1, 3, 5)):
                        metrics[mode][name].update(i, prediction[h], raw['future_gt_occ'][h], moving[h])
                if mode in soft_outputs:
                    for i, h in enumerate((1, 3, 5)):
                        gap[mode]['soft_argmax'].update(i, soft_outputs[mode][i], raw['future_gt_occ'][h], moving[h])
                        gap[mode]['hard_same_windows'].update(i, variants['add_remove'][h], raw['future_gt_occ'][h], moving[h])
            if soft_outputs: gap_windows += 1
            windows += 1
            if windows % 16 == 0:
                if progress: progress(dict(event='paired_eval', windows=windows, seconds=time.perf_counter()-start))
                if save: save(snapshot())
                print(f'JOINT_OCC_EVAL {windows} shared_inputs GT/Pred x transport/ADD/ADD_REMOVE', flush=True)
    if save: save(snapshot())
    trajectory = {mode: dict(xy_mean_m=np.mean([e['xy'] for e in rows], axis=0).tolist(),
                                yaw_mean_deg=np.mean([e['yaw'] for e in rows], axis=0).tolist())
                  for mode, rows in errors.items() if rows}
    return dict(windows=windows, variants={mode: {name: m.compute() for name, m in variants.items()}
        for mode, variants in metrics.items()}, trajectory=trajectory, seconds=time.perf_counter()-start,
        surrogate_gap=dict(windows=gap_windows, scope='FIRST8 only; diagnostic, never route/threshold selection',
            variants={mode: {name: m.compute() for name, m in variants.items()} for mode, variants in gap.items()}),
        thresholds=[.5, remove_threshold], scoring='full grid; no alignment or GT visibility mask')
