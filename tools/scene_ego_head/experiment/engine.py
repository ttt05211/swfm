"""Full6 continuous Pred training; same-model internal/external/GT evaluation."""
from collections import defaultdict
import time
import numpy as np
import torch
from tools.scene_ego_head.integration import forward_pred_window, forecast_conditioned
from tools.shared_history_experiments.refine.engine import semantic_loss, Metrics, window_from_record, gt_moving_support_sequence, moving_support_masks
from tools.joint_occ_experiments.model import ego_geometry_loss
from real_motion.ego_navigation import relative_se2

ROUTES = ('internal', 'external', 'gt_reference')
READOUTS = ('transport', 'semantic_refine')
EVAL_PROTOCOL = 'scene_ego_joint500_same_model_pose_intervention_v1'


def train_arm(model, optimizer, provider, rows, mode, *, update, total_updates,
              object_weight=1., ego_weight=1., lovasz_weight=1., probe=False, optimized=True):
    if mode != 'pred' or not rows or not optimized: raise ValueError('internal Pred full6 required')
    model.train(); optimizer.zero_grad(set_to_none=True)
    factor = .01+.99*.5*(1+np.cos(np.pi*(update-1)/max(1, total_updates)))
    for group in optimizer.param_groups: group['lr'] = group['initial_lr']*factor
    tick = time.perf_counter(); stages, terms, links = defaultdict(float), defaultdict(list), {}
    sources = 0
    for row in rows:
        start = time.perf_counter(); fw = forward_pred_window(model, provider, row)
        stages['shared_encoder_scene_ego_motion_projection_refine'] += time.perf_counter()-start
        for key, value in fw['stages'].items(): stages['detail/'+key] += value
        start = time.perf_counter()
        # Supervision is accessed ONLY after the internal Pred forward returns.
        losses, ce, lovasz = semantic_loss(fw['logits'], row[1]['future_gt_occ'], lovasz_weight)
        ego_loss = ego_geometry_loss(fw['ego']['se2'], np.asarray(row[2][2])[None])
        loss = (losses.mean()+ego_weight*ego_loss)/len(rows)
        if not bool(torch.isfinite(loss)): raise RuntimeError('nonfinite joint scene-ego objective')
        if probe or update == 2:
            names = ['occupancy_to_scene_ego', 'occupancy_to_shared_history']
            tensors = [fw['ego']['se2'], fw['field']]
            if len(fw['targets']):
                names += ['occupancy_to_xy', 'occupancy_to_yaw']
                tensors += [fw['output']['residual_xy_m'], fw['output']['yaw_delta_rad']]
            grads = torch.autograd.grad(losses.mean(), tensors, retain_graph=True, allow_unused=True)
            for name, grad in zip(names, grads): links[name] = links.get(name, 0.)+(0. if grad is None else float(grad.float().norm()))
        stages['full6_ce_lovasz_and_ego'] += time.perf_counter()-start
        start = time.perf_counter(); loss.backward(); stages['full_network_backward'] += time.perf_counter()-start
        terms['CE'].extend(ce.detach().unbind()); terms['Lovasz'].extend(lovasz.detach().unbind())
        terms['ego_geometry'].append(ego_loss.detach()); sources += len(fw['targets'])
        del fw, loss, losses, ce, lovasz, ego_loss
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
    start = time.perf_counter(); optimizer.step()
    if provider.device.type == 'cuda': torch.cuda.synchronize(provider.device)
    stages['optimizer_finish'] += time.perf_counter()-start
    result = {key: float(torch.stack(value).mean()) for key, value in terms.items()}
    result.update(mode=mode, windows=len(rows), sources=sources, seconds=time.perf_counter()-tick,
        stages_seconds=dict(stages), grad_norm=float(norm), gradient_links=links,
        training_backend='shared_full6_scene_ego_joint', source_auxiliary_computed=False,
        peak_memory_mib=torch.cuda.max_memory_allocated(provider.device)/2**20 if provider.device.type == 'cuda' else 0.)
    optimizer.zero_grad(set_to_none=True)
    return result


def restore_metrics(saved, metadata):
    metrics = {route: {name: Metrics() for name in READOUTS} for route in ROUTES}
    if saved is None: return metrics, [], 0
    if saved.get('protocol') != EVAL_PROTOCOL or saved.get('population') != metadata:
        raise RuntimeError('scene ego evaluation provenance changed')
    seen, windows = saved['seen'], saved['windows']
    if type(seen) is not int or not 0 <= seen <= metadata['requested'] or type(windows) is not int:
        raise ValueError('invalid committed evaluation cursor')
    selected = {tuple(k) for k in metadata['selected_keys']}
    expected = sum(tuple(k) in selected for k in saved['requested_keys'][:seen])
    if (windows != expected or len(saved['errors']) != windows or set(saved['metrics']) != set(ROUTES)):
        raise ValueError('invalid committed evaluation cursor')
    for route in metrics:
        if set(saved['metrics'][route]) != set(READOUTS): raise ValueError('evaluation readouts differ')
        for name, meter in metrics[route].items():
            counts = saved['metrics'][route][name]
            if set(counts) != set(vars(meter)): raise ValueError('evaluation count fields differ')
            for key, old in vars(meter).items():
                value = np.asarray(counts[key])
                if value.shape != old.shape or value.dtype.kind not in 'iu' or (value < 0).any():
                    raise ValueError('invalid integer metric counts')
                setattr(meter, key, value.astype(np.int64))
    return metrics, list(saved['errors']), seen


@torch.no_grad()
def evaluate(models, provider, source, batches, *, planner, remove_threshold=None,
             stop_event=None, progress=None, saved=None, save=None):
    if set(models) != {'pred'}: raise ValueError('one new internal-Pred model only')
    if saved and saved.get('requested_keys') != [list(k) for k in planner.requested]:
        raise RuntimeError('requested key order changed')
    metrics, errors, seen = restore_metrics(saved, planner.metadata)
    model = models['pred']; model.eval(); tick = time.perf_counter(); offset = 0
    def snapshot():
        return dict(protocol=EVAL_PROTOCOL, population=planner.metadata, requested_keys=[list(k) for k in planner.requested],
            seen=seen, windows=len(errors), errors=errors,
            metrics={route: {name: {key: value.tolist() for key, value in vars(meter).items()}
                            for name, meter in meters.items()} for route, meters in metrics.items()})
    def persist():
        if save: save(snapshot())
    try:
        for rows in batches:
            for row in rows:
                key = (row[0]['scene_name'], row[0]['t0_token'])
                if offset >= len(planner.requested) or key != planner.requested[offset]: raise RuntimeError('eval key order differs')
                offset += 1
                if offset <= seen: continue
                if stop_event is not None and stop_event.is_set(): raise InterruptedError('saved complete three-route evaluation window')
                if key not in planner.rows:
                    seen = offset; continue  # exclude from ALL routes, never substitute GT
                internal = forward_pred_window(model, provider, row)
                external_poses = planner.poses(row[0], row[1]['history_poses'][-1])
                predicted = {}; ego_values = {'internal': internal['ego']['se2'][0].float().cpu().numpy(),
                    'external': relative_se2(row[1]['history_poses'][-1], external_poses)}
                for route in ROUTES:
                    if route == 'internal': fw = internal
                    else:
                        poses = external_poses if route == 'external' else row[1]['future_poses']
                        fw = forecast_conditioned(model, provider, row, internal, poses)
                    predicted[route] = {name: values.argmax(1).reshape(6, *fw['shape']).byte().cpu().numpy()
                        for name, values in (('transport', fw['prior']), ('semantic_refine', fw['logits']))}
                    if route != 'internal': del fw
                # All routes completed BEFORE future labels/annotation metrics enter scoring.
                raw = row[1]; w = window_from_record(row[0])
                support = gt_moving_support_sequence(source.nusc, w.t0_token, w.future_tokens,
                    tuple(.5*(h+1) for h in range(6)), grid=provider.pcfg.grid, workers=provider.workers)
                moving = moving_support_masks(support, raw['future_gt_occ'].shape[1:])
                delta = {route: {name: Metrics() for name in READOUTS} for route in ROUTES}
                for route in ROUTES:
                    for name in READOUTS:
                        for i, h in enumerate((1, 3, 5)):
                            delta[route][name].update(i, predicted[route][name][h], raw['future_gt_occ'][h], moving[h])
                target = np.asarray(row[2][2]); error = {}
                for route, values in ego_values.items():
                    d = values-target
                    error[route] = dict(xy=np.linalg.norm(d[:, :2], axis=-1).tolist(),
                        yaw=(np.abs(np.arctan2(np.sin(d[:, 2]), np.cos(d[:, 2])))*180/np.pi).tolist())
                # Commit all three routes atomically, only after every metric succeeds.
                for route in ROUTES:
                    for name in READOUTS:
                        for field in vars(metrics[route][name]):
                            getattr(metrics[route][name], field)[:] += getattr(delta[route][name], field)
                errors.append(error); seen = offset
                del internal, predicted
                if seen % 8 == 0: persist()
                if seen % 16 == 0:
                    print(f'SCENE_EGO_EVAL requested={seen}/{len(planner.requested)} covered={len(errors)}', flush=True)
                    if progress: progress(dict(event='scene_ego_eval', requested=seen, windows=len(errors)))
        if offset != len(planner.requested): raise RuntimeError('incomplete evaluation iterator')
    finally: persist()
    variants = {route: {name: meter.compute() for name, meter in meters.items()} for route, meters in metrics.items()}
    trajectory = {route: dict(xy_mean_m=np.mean([e[route]['xy'] for e in errors], 0).tolist(),
        yaw_mean_deg=np.mean([e[route]['yaw'] for e in errors], 0).tolist()) for route in ('internal', 'external')}
    return dict(windows=len(errors), variants=variants, trajectory=trajectory, population=planner.metadata,
        same_model_internal_minus_external={k: variants['internal']['semantic_refine'][k]-variants['external']['semantic_refine'][k]
            for k in ('mIoU', 'IoU', 'MovingMicro')}, seconds=time.perf_counter()-tick)
