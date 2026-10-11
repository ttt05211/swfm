"""Same sequential network/two-horizon objective, with INTERNAL ego poses."""
from collections import defaultdict
import copy
import time
import numpy as np
import torch
from ..engine import forward_window, hard_predictions
from ..fast_autograd import RenderGradientBridge
from ..model import source_auxiliary, ego_geometry_loss
from ..batching.geometry import GroupedScene, grouped_loss
from ..soft_inference.engine import decode_soft, soft_predictions, ROUTES
from ..soft_inference.fast import predict
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import Metrics
from tools.real_motion.eval_p0_f9_v17_local_stwm import window_from_record
from tools.real_motion.shared_evidence_pilot_common import moving_support_masks
from real_motion.nuscenes_adapter import gt_moving_support_sequence


def train_arm(model, optimizer, provider, rows, mode, *, update, total_updates,
              object_weight=1., ego_weight=1., lovasz_weight=1., probe=False, optimized=True):
    if mode != 'pred' or not optimized or not rows:
        raise ValueError('nonempty fused INTERNAL Pred batch required')
    model.train()
    optimizer.zero_grad(set_to_none=True)
    factor = .01+.99*.5*(1+np.cos(np.pi*(update-1)/max(1, total_updates)))
    for group in optimizer.param_groups: group['lr'] = group['initial_lr']*factor
    begun = time.perf_counter()
    stages, terms, links = defaultdict(float), defaultdict(list), defaultdict(float)
    sources = canonical = 0
    ramp = .1+.9*min(1., (update-1)/max(1., total_updates/5))
    for row in rows:
        tick = time.perf_counter()
        # Pred forward NEVER mixes GT poses; only the occupancy->ego gradient
        # ramps. Original single-window forward/backward order is retained.
        fw = forward_window(model, provider, row, 'pred', ego_gradient=ramp)
        scene = GroupedScene(fw['prep'], provider.pcfg.grid, provider.device)
        bridge = RenderGradientBridge(fw)
        stages['motion_ego_projection_head'] += time.perf_counter()-tick
        sources += len(row[0]['features']); canonical += len(fw['evidence'])
        labels = np.asarray(row[1]['future_gt_occ'])
        for first in (0, 2, 4):
            hs = [first, first+1]
            tick = time.perf_counter()
            probability = scene.render_group(hs, **bridge.leaves, evidence=fw['evidence'], plan=fw['plan'])
            losses, ce, lovasz = grouped_loss(probability, labels[hs], lovasz_weight=lovasz_weight)
            if not bool(torch.isfinite(losses).all()): raise RuntimeError('nonfinite occupancy loss')
            if probe and first == 0 and losses.requires_grad:
                tensors, names = [fw['actions'], fw['ego']], ['occupancy_to_actions', 'occupancy_to_ego']
                if fw['output']['residual_xy_m'].requires_grad:
                    tensors += [fw['output']['residual_xy_m'], fw['output']['yaw_delta_rad']]
                    names += ['occupancy_to_xy', 'occupancy_to_yaw']
                for name, gradient in zip(names, bridge.probe(losses[1], tensors)):
                    links[name] += 0. if gradient is None else float(gradient.detach().float().norm())
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
        if not bool(torch.isfinite(auxiliary)): raise RuntimeError('nonfinite direct geometry loss')
        bridge.backward_shared(auxiliary/len(rows))
        terms['source_geometry'].append(geometry.detach()); terms['existence'].append(existence.detach())
        terms['ego_geometry'].append(ego_loss.detach())
        stages['shared_network_and_direct_geometry_backward'] += time.perf_counter()-tick
        del fw, scene, bridge, geometry, existence, ego_loss, auxiliary, labels
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
    tick = time.perf_counter(); optimizer.step()
    if provider.device.type == 'cuda': torch.cuda.synchronize(provider.device)
    stages['optimizer_finish'] += time.perf_counter()-tick
    result = {key: float(torch.stack(value).mean()) for key, value in terms.items()}
    result.update(mode=mode, windows=len(rows), sources=sources, canonical_points=canonical,
        seconds=time.perf_counter()-begun, stages_seconds=dict(stages), grad_norm=float(norm),
        ego_occupancy_gradient_scale=ramp, gradient_links=dict(links),
        training_backend='sequential_network_two_horizons_internal_ego',
        peak_memory_mib=torch.cuda.max_memory_allocated(provider.device)/2**20 if provider.device.type == 'cuda' else 0.)
    optimizer.zero_grad(set_to_none=True)
    return result


@torch.no_grad()
def readouts(model, provider, row, *, remove_threshold=.95, verify=False):
    # One motion/ego/Strong/projection per window; two CCR contexts for the
    # declared hard vs soft readouts. Both geometries use the SAME Pred poses.
    fw = forward_window(model, provider, row, 'pred', soft=False)
    hard = hard_predictions(fw, remove_threshold=remove_threshold)
    soft_fw = decode_soft(model, provider, fw)
    soft = {name: predict(provider, soft_fw, name, range(6), group_size=3)
            for name in ('transport', 'add_only', 'add_remove')}
    if verify:
        reference = soft_predictions(provider, soft_fw, tuple(soft), range(6))
        for name in soft:
            if any(not np.array_equal(a, b) for a, b in zip(soft[name], reference[name])):
                raise RuntimeError('fast INTERNAL Pred soft six-frame byte mismatch: '+name)
    prediction = fw['ego'].float().cpu().numpy(); target = np.asarray(fw['ego_target'])
    error = dict(xy=np.linalg.norm(prediction[:, :2]-target[:, :2], axis=-1).tolist(),
        yaw=(np.abs(np.arctan2(np.sin(prediction[:, 2]-target[:, 2]),
                             np.cos(prediction[:, 2]-target[:, 2])))*180/np.pi).tolist())
    return {**{'hard_'+k: v for k, v in hard.items()}, **{'soft_'+k: v for k, v in soft.items()}}, error


@torch.no_grad()
def evaluate(models, provider, source, batches, *, remove_threshold=.95, stop_event=None,
             progress=None, saved=None, save=None):
    if set(models) != {'pred'}: raise ValueError('independent INTERNAL Pred model required')
    model = models['pred']; model.eval()
    metrics = {name: Metrics() for name in ROUTES}
    windows, errors, verified = 0, [], False
    if saved:
        if saved.get('protocol') != 'internal_pred_six_readouts_v1' or set(saved.get('metrics', {})) != set(ROUTES):
            raise RuntimeError('Pred evaluation readouts/protocol differ')
        windows, errors, verified = int(saved['windows']), copy.deepcopy(saved['errors']), saved['fast_verified']
        if windows < 0 or len(errors) != windows or windows > 0 and verified is not True:
            raise RuntimeError('invalid completed Pred evaluation prefix')
        for name, fields in saved['metrics'].items():
            if set(fields) != set(vars(metrics[name])): raise RuntimeError('metric fields differ')
            for key, value in fields.items():
                original = np.asarray(value)
                array = np.asarray(value, np.int64)
                if (array.shape != getattr(metrics[name], key).shape or np.any(array < 0)
                        or not np.array_equal(original, array)):
                    raise RuntimeError('invalid saved integer counts')
                setattr(metrics[name], key, array)
    def snapshot():
        return dict(protocol='internal_pred_six_readouts_v1', windows=windows, errors=copy.deepcopy(errors),
            fast_verified=verified, metrics={name: {k: v.tolist() for k, v in vars(m).items()}
                                           for name, m in metrics.items()})
    begun, offset = time.perf_counter(), 0
    for rows in batches:
        for row in rows:
            offset += 1
            if offset <= windows: continue
            if stop_event is not None and stop_event.is_set():
                if save: save(snapshot())
                raise InterruptedError('Pred evaluation saved after complete window')
            predictions, error = readouts(model, provider, row, remove_threshold=remove_threshold, verify=not verified)
            # ALL predictions precede future labels/support used for metrics.
            window = window_from_record(row[0]); raw = row[1]
            support = gt_moving_support_sequence(source.nusc, window.t0_token, window.future_tokens,
                tuple(.5*(h+1) for h in range(6)), grid=provider.pcfg.grid, workers=provider.workers)
            moving = moving_support_masks(support, raw['future_gt_occ'].shape[1:])
            for name, values in predictions.items():
                for i, h in enumerate((1, 3, 5)):
                    metrics[name].update(i, values[h], raw['future_gt_occ'][h], moving[h])
            errors.append(error); windows += 1; verified = True
            if windows % 8 == 0 and save: save(snapshot())
            if windows % 16 == 0:
                summary = {name: m.compute()['mIoU'] for name, m in metrics.items()}
                if progress: progress(dict(event='pred_soft_hard_eval', windows=windows, mIoU=summary))
                print(f'PRED_SOFT_HARD {windows} '+str(summary), flush=True)
            del predictions, support, moving
    if save: save(snapshot())
    trajectory = {'pred': dict(xy_mean_m=np.mean([e['xy'] for e in errors], axis=0).tolist(),
        yaw_mean_deg=np.mean([e['yaw'] for e in errors], axis=0).tolist())} if errors else {}
    return dict(windows=windows, variants={'pred': {k: m.compute() for k, m in metrics.items()}},
        trajectory=trajectory, fast_six_frame_byte_parity=verified, seconds=time.perf_counter()-begun,
        thresholds=[.5, remove_threshold], main_readout='soft_add_only',
        scoring='full grid; INTERNAL predicted future ego; no GT pose alignment or visibility mask')
