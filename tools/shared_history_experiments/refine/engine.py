"""Full-six CE/Lovasz, identical continuous semantic readout in train/eval."""
from collections import defaultdict
import time
import numpy as np
import torch
from torch.nn import functional as F
from tools.shared_history_experiments.model import encode_window
from tools.joint_occ_experiments.soft_geometry import source_targets, effective_yaw
from tools.joint_occ_experiments.model import source_auxiliary, ego_geometry_loss
from tools.joint_occ_experiments.batching.geometry import _GroupedLovasz
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import Metrics
from tools.real_motion.eval_p0_f9_v17_local_stwm import window_from_record
from tools.real_motion.shared_evidence_pilot_common import moving_support_masks
from real_motion.nuscenes_adapter import gt_moving_support_sequence
from .geometry import TransportPrior, aligned_memory

EVAL_PROTOCOL = 'full_grid_semantic_residual_fixed_eval_v1'


def forward_window(model, provider, row):
    record, raw, ego_inputs, history = row
    e, commands, target = ego_inputs
    device = provider.device; cast = lambda x: torch.as_tensor(np.asarray(x), device=device)
    state = {**history[0]['state'], 'column_backgrounds': raw['_column_causal_preparation']['prepared_state']['column_backgrounds']}
    if (len(state['current']) != len(record['source_class_id']) or
            not np.array_equal(record['source_class_id'], [int(s['class_id']) for s in state['current']])):
        raise RuntimeError('source history/record identity differs')
    current = np.asarray(state['current_pose'])
    xyz = np.asarray([s['centroid_world'] for s in state['current']]).reshape(-1, 3)
    native = (xyz-current[:3, 3]) @ current[:3, :3]
    if not np.allclose(native[:, :2], np.asarray(record['source_centroid_xy_t0_m']), atol=2e-4, rtol=0):
        raise RuntimeError('source order/centroid differs')
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        field, reads, times = encode_window(model, raw, history, e, device)
        output = model.transport(field, reads, times, cast(record['source_class_id']).long(),
                                 cast(record['kta_displacement_xy_m']).float())
        ego = model.ego(cast(e[None]), cast(commands[None]))['se2'][0]
        lifted = model.columns.lift_history(field)
    poses = cast(raw['future_poses']).double()  # explicit GT-trajectory configuration only
    if poses.shape != (6, 4, 4) or not bool(torch.isfinite(poses).all()):
        raise ValueError('six finite GT future ego poses required')
    targets = source_targets(output, record, current, state['source_z_t0'])
    yaws = effective_yaw(output, [int(s['class_id']) for s in state['current']])
    tick = time.perf_counter()
    scene = TransportPrior(raw, state, provider.pcfg.grid, device)
    prior = scene.render(poses, targets, yaws)
    transport_seconds = time.perf_counter()-tick
    tick = time.perf_counter()
    memory = aligned_memory(model.columns, lifted, output, history, raw, targets, yaws,
                            poses, provider.pcfg.grid)
    memory_seconds = time.perf_counter()-tick
    tick = time.perf_counter()
    logits = model.columns(memory, prior, range(6), scene.shape)
    return dict(output=output, ego=ego, ego_target=target, state=state, field=field,
        poses=poses, targets=targets, yaws=yaws, prior=prior, memory=memory,
        logits=logits, shape=scene.shape, stages=dict(transport_prior=transport_seconds,
            aligned_feature_memory=memory_seconds, semantic_refiner=time.perf_counter()-tick))


def semantic_loss(logits, cpu_labels, lovasz_weight=1.):
    labels_np = np.asarray(cpu_labels).reshape(len(logits), -1)
    if logits.shape != (len(labels_np), 18, labels_np.shape[1]): raise ValueError('full-grid labels/logits differ')
    labels = torch.as_tensor(np.array(labels_np, copy=True), device=logits.device).long()
    valid = (labels >= 0) & (labels < 18)
    safe = torch.where(valid, labels, 255)
    ce = F.cross_entropy(logits.float(), safe, ignore_index=255, reduction='none').sum(1)/valid.sum(1).clamp_min(1)
    probabilities = logits.float().softmax(1)
    classes = [[int(c) for c in np.unique(row) if 0 <= c < 17] for row in labels_np]
    width = max(map(len, classes), default=0)
    if width:
        # Unknown positions must be removed from each horizon's full stable sort.
        if not bool(valid.all()):
            from tools.joint_occ_experiments.soft_geometry import lovasz_softmax_reference
            lovasz = torch.stack([lovasz_softmax_reference(p, lab, ignore=255, class_ids=ids)
                                 for p, lab, ids in zip(probabilities, safe, classes)])
        else:
            ids = torch.as_tensor([c+[0]*(width-len(c)) for c in classes], device=logits.device)
            active = torch.arange(width, device=logits.device)[None] < ids.new_tensor(list(map(len, classes)))[:, None]
            lovasz = _GroupedLovasz.apply(probabilities, labels, ids, active)
    else: lovasz = logits.sum((1, 2))*0
    return ce+lovasz_weight*lovasz, ce, lovasz


def train_arm(model, optimizer, provider, rows, mode, *, update, total_updates,
              object_weight=1., ego_weight=1., lovasz_weight=1., probe=False, optimized=True):
    if mode != 'gt' or not rows or not optimized: raise ValueError('GT full-grid H6 prototype required')
    model.train(); optimizer.zero_grad(set_to_none=True)
    factor = .01+.99*.5*(1+np.cos(np.pi*(update-1)/max(1, total_updates)))
    for group in optimizer.param_groups: group['lr'] = group['initial_lr']*factor
    begun = time.perf_counter(); stages, terms, links = defaultdict(float), defaultdict(list), defaultdict(float)
    sources = 0
    for row in rows:
        tick = time.perf_counter(); fw = forward_window(model, provider, row)
        stages['shared_encoder_motion_projection_refine'] += time.perf_counter()-tick
        for key, value in fw['stages'].items(): stages['detail/'+key] += value
        tick = time.perf_counter()
        losses, ce, lovasz = semantic_loss(fw['logits'], row[1]['future_gt_occ'], lovasz_weight)
        geometry, existence = source_auxiliary(fw['output'], row[0], fw['state'])
        ego_loss = ego_geometry_loss(fw['ego'][None], np.asarray(fw['ego_target'])[None])
        loss = (losses.mean()+object_weight*(geometry+existence)+ego_weight*ego_loss)/len(rows)
        if not bool(torch.isfinite(loss)): raise RuntimeError('nonfinite semantic residual objective')
        if probe:
            tensors, names = [fw['field'], fw['memory']], ['occupancy_to_shared_history', 'occupancy_to_memory']
            if len(fw['targets']):
                tensors += [fw['output']['residual_xy_m'], fw['output']['yaw_delta_rad']]
                names += ['occupancy_to_xy', 'occupancy_to_yaw']
            gradients = torch.autograd.grad(losses.mean(), tensors, retain_graph=True, allow_unused=True)
            for name, gradient in zip(names, gradients): links[name] += 0. if gradient is None else float(gradient.float().norm())
        stages['full6_ce_lovasz_and_auxiliary'] += time.perf_counter()-tick
        tick = time.perf_counter(); loss.backward(); stages['full_network_backward'] += time.perf_counter()-tick
        terms['CE'].extend(ce.detach().unbind()); terms['Lovasz'].extend(lovasz.detach().unbind())
        for key, value in (('source_geometry', geometry), ('existence', existence), ('ego_geometry', ego_loss)):
            terms[key].append(value.detach())
        sources += len(fw['targets'])
        del fw, losses, loss, geometry, existence, ego_loss
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
    tick = time.perf_counter(); optimizer.step()
    if provider.device.type == 'cuda': torch.cuda.synchronize(provider.device)
    stages['optimizer_finish'] += time.perf_counter()-tick
    result = {k: float(torch.stack(v).mean()) for k, v in terms.items()}
    result.update(mode=mode, windows=len(rows), sources=sources, seconds=time.perf_counter()-begun,
        stages_seconds=dict(stages), grad_norm=float(norm), gradient_links=dict(links),
        training_backend='shared_full6_semantic_residual',
        peak_memory_mib=torch.cuda.max_memory_allocated(provider.device)/2**20 if provider.device.type == 'cuda' else 0.)
    optimizer.zero_grad(set_to_none=True)
    return result


@torch.no_grad()
def evaluate(models, provider, source, batches, *, remove_threshold=None, stop_event=None,
             progress=None, saved=None, save=None):
    if set(models) != {'gt'}: raise ValueError('GT only')
    model = models['gt']; model.eval()
    metrics = {n: Metrics() for n in ('transport', 'semantic_refine')}
    windows, errors = 0, []
    if saved:
        if saved.get('protocol') != EVAL_PROTOCOL or set(saved['metrics']) != set(metrics):
            raise RuntimeError('semantic residual eval protocol differs')
        windows, errors = int(saved['windows']), list(saved['errors'])
        if windows < 0 or len(errors) != windows: raise RuntimeError('invalid completed eval cursor')
        for name, fields in saved['metrics'].items():
            if set(fields) != set(vars(metrics[name])): raise RuntimeError('metric fields differ')
            for key, value in fields.items():
                array = np.asarray(value, np.int64)
                if array.shape != getattr(metrics[name], key).shape or np.any(array < 0) or not np.array_equal(array, value):
                    raise RuntimeError('invalid integer counts')
                setattr(metrics[name], key, array)
    def snapshot():
        return dict(protocol=EVAL_PROTOCOL, windows=windows, errors=list(errors),
            metrics={n: {k: v.tolist() for k, v in vars(m).items()} for n, m in metrics.items()})
    tick, offset = time.perf_counter(), 0
    for rows in batches:
        for row in rows:
            offset += 1
            if offset <= windows: continue
            if stop_event is not None and stop_event.is_set():
                if save: save(snapshot())
                raise InterruptedError('saved complete semantic residual eval windows')
            fw = forward_window(model, provider, row)
            predictions = {name: values.argmax(1).reshape(6, *fw['shape']).byte().cpu().numpy()
                for name, values in (('transport', fw['prior']), ('semantic_refine', fw['logits']))}
            # Predictions precede all future occupancy/annotation metric reads.
            raw = row[1]; window = window_from_record(row[0])
            support = gt_moving_support_sequence(source.nusc, window.t0_token, window.future_tokens,
                tuple(.5*(h+1) for h in range(6)), grid=provider.pcfg.grid, workers=provider.workers)
            moving = moving_support_masks(support, raw['future_gt_occ'].shape[1:])
            for name in predictions:
                for i, h in enumerate((1, 3, 5)): metrics[name].update(i, predictions[name][h], raw['future_gt_occ'][h], moving[h])
            error = fw['ego'].float().cpu().numpy()-np.asarray(fw['ego_target'])
            errors.append(dict(xy=np.linalg.norm(error[:, :2], axis=-1).tolist(),
                yaw=(np.abs(np.arctan2(np.sin(error[:, 2]), np.cos(error[:, 2])))*180/np.pi).tolist()))
            windows += 1
            if windows % 8 == 0 and save: save(snapshot())
            if windows % 16 == 0:
                if progress: progress(dict(event='semantic_refine_eval', windows=windows))
                print(f'SEMANTIC_REFINE_EVAL {windows}', flush=True)
    if save: save(snapshot())
    return dict(windows=windows, variants={'gt': {n: m.compute() for n, m in metrics.items()}},
        trajectory={'gt': dict(xy_mean_m=np.mean([e['xy'] for e in errors], 0).tolist(),
            yaw_mean_deg=np.mean([e['yaw'] for e in errors], 0).tolist())} if errors else {},
        seconds=time.perf_counter()-tick, main_readout='semantic_refine', scoring='full grid, GT ego conditioned')
