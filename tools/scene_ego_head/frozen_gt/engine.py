"""Frozen-body INTERNAL ego inference and explicit same-pose entrance audit."""
import copy
import time
import numpy as np
import torch
from real_motion.ego_navigation import navigation_commands, relative_se2, poses_from_se2
from tools.scene_ego_head.model import prepare_scene
from tools.scene_ego_head.integration import forecast_conditioned
from tools.scene_ego_head.diagnostics.engine import external_forward, errors, trajectory_report
from tools.shared_history_experiments.model import encode_window
from tools.shared_history_experiments.full_gt import engine as gt

EVAL_PROTOCOL = 'full_gt_frozen_body_internal_scene_ego_eval_v1'


def metadata(source, record, raw):
    e = gt_history_inputs(source, record, raw)[0]
    return e, navigation_commands(source.nusc, record['future_tokens']), relative_se2(
        raw['history_poses'][-1], raw['future_poses'])


# Captured once rather than recursing when the runner temporarily overrides gt.history_inputs.
gt_history_inputs = gt.history_inputs


@torch.no_grad()
def entrance_audit(body, provider, row):
    a = gt.forward_window(body, provider, row)
    b = external_forward(body, provider, row, np.stack(row[1]['future_poses']))
    cached = row[1]['_column_causal_preparation']['prepared_state']['column_backgrounds']
    changed = sum(int(np.count_nonzero(np.asarray(x) != np.asarray(y)))
                  for x, y in zip(cached, b['state']['column_backgrounds']))
    fields = {}
    for k in ('field', 'prior', 'memory', 'logits'):
        x, y = a[k], b[k]
        fields[k] = dict(pass_gate=bool(torch.allclose(x, y, rtol=2e-4, atol=2e-5)),
                        max_abs=float((x.float()-y.float()).abs().max()))
    return dict(background_changed_voxels=changed, fields=fields,
        pass_gate=changed == 0 and all(v['pass_gate'] for v in fields.values()),
        semantic_argmax_changed_voxels=int((a['logits'].argmax(1) != b['logits'].argmax(1)).sum()),
        diagnostic_GT_poses_only=True, not_a_Pred_score=True)


@torch.no_grad()
def predict(body, head, provider, row):
    record, raw, inputs, history = row
    historical = {k: raw[k] for k in ('history_occ', 'history_observed', 'history_poses')}
    e, commands = inputs[:2]  # targets and future poses deliberately NOT consumed
    state = history[0]['state']; current = np.asarray(state['current_pose'])
    if not np.array_equal(current, np.asarray(historical['history_poses'][-1])):
        raise RuntimeError('history state does not end at t0')
    if not np.array_equal(record['source_class_id'], [int(s['class_id']) for s in state['current']]):
        raise RuntimeError('source identity/order differs')
    xyz = np.asarray([s['centroid_world'] for s in state['current']]).reshape(-1, 3)
    native = (xyz-current[:3, 3]) @ current[:3, :3]
    if not np.allclose(native[:, :2], record['source_centroid_xy_t0_m'], atol=2e-4, rtol=0):
        raise RuntimeError('source centroid/order differs')
    if not np.allclose(record['anchors_xy_t0_m'], np.asarray(record['source_centroid_xy_t0_m'])[:, None]+
                       np.asarray(record['kta_displacement_xy_m']), atol=2e-4, rtol=0):
        raise RuntimeError('KTA anchors must use fixed t0 axes')
    device = provider.device
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        field, reads, times = encode_window(body, historical, history, e, device)
    known = (np.asarray(historical['history_observed'], bool) &
             (np.asarray(historical['history_occ']) >= 0) & (np.asarray(historical['history_occ']) < 18))
    scene = prepare_scene(field, observed=known, history_poses=historical['history_poses'],
                          ego_history=e, grid=provider.pcfg.grid, pool_side=head.config.pool_side)
    ego = head(torch.as_tensor(np.asarray(e)[None], device=device),
               torch.as_tensor(np.asarray(commands)[None], device=device), scene)
    poses = poses_from_se2(historical['history_poses'][-1], ego['se2'][0].cpu().numpy())
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        output = body.transport(field, reads, times, torch.as_tensor(record['source_class_id'], device=device).long(),
                                torch.as_tensor(record['kta_displacement_xy_m'], device=device).float())
        lifted = body.columns.lift_history(field)
    prepared = dict(field=field, output=output, lifted=lifted)
    result = forecast_conditioned(body, provider, row, prepared, poses)
    result['ego'] = ego
    return result


@torch.no_grad()
def evaluate(body, head, provider, source, batches, *, saved=None, save=None,
             stop_event=None, progress=None, remove_threshold=None):
    body.eval(); head.eval()
    meters = {n: gt.Metrics() for n in ('transport', 'semantic_refine')}
    state = dict(protocol=EVAL_PROTOCOL, keys=[], errors=[])
    if saved:
        if saved.get('protocol') != EVAL_PROTOCOL or len(saved['keys']) != len(saved['errors']):
            raise RuntimeError('frozen-head eval protocol/cursor differs')
        state = copy.deepcopy(saved)
        if set(state['counts']) != set(meters): raise RuntimeError('eval readouts differ')
        for name, meter in meters.items():
            if set(state['counts'][name]) != set(vars(meter)): raise RuntimeError('count fields differ')
            for k, old in vars(meter).items():
                v = np.asarray(state['counts'][name][k])
                if v.shape != old.shape or v.dtype.kind not in 'iu' or (v < 0).any():
                    raise RuntimeError('invalid integer counts')
                setattr(meter, k, v.astype(np.int64))
    def persist():
        state['counts'] = {n: {k: v.tolist() for k, v in vars(m).items()} for n, m in meters.items()}
        if save: save(copy.deepcopy(state))
    tick = time.perf_counter(); offset = 0
    try:
        for rows in batches:
            for row in rows:
                key = [row[0]['scene_name'], row[0]['t0_token']]; offset += 1
                if offset <= len(state['keys']):
                    if state['keys'][offset-1] != key: raise RuntimeError('evaluation prefix identity differs')
                    continue
                if stop_event is not None and stop_event.is_set(): raise InterruptedError('internal eval window saved')
                fw = predict(body, head, provider, row)
                predictions = {n: v.argmax(1).reshape(6, *fw['shape']).byte().cpu().numpy()
                    for n, v in (('transport', fw['prior']), ('semantic_refine', fw['logits']))}
                # Metrics/continuous ego GT are read AFTER all predicted-pose forecasts.
                error = {'head': errors(fw['ego']['se2'][0].cpu().numpy(), row[2][2]),
                         'prior': errors(fw['ego']['prior'][0].cpu().numpy(), row[2][2])}
                del fw
                w = gt.window_from_record(row[0]); raw = row[1]
                support = gt.gt_moving_support_sequence(source.nusc, w.t0_token, w.future_tokens,
                    tuple(.5*(h+1) for h in range(6)), grid=provider.pcfg.grid, workers=provider.workers)
                moving = gt.moving_support_masks(support, raw['future_gt_occ'].shape[1:])
                delta = {n: gt.Metrics() for n in meters}
                for n in meters:
                    for i, h in enumerate((1, 3, 5)):
                        delta[n].update(i, predictions[n][h], raw['future_gt_occ'][h], moving[h])
                for n in meters:
                    for k in vars(meters[n]): getattr(meters[n], k)[:] += getattr(delta[n], k)
                state['keys'].append(key); state['errors'].append(error)
                if offset % 8 == 0: persist()
                if offset % 16 == 0:
                    print(f'FROZEN_INTERNAL_EVAL {offset}', flush=True)
                    if progress: progress(dict(event='frozen_internal_eval', windows=offset))
    finally: persist()
    if offset < len(state['keys']): raise RuntimeError('cursor exceeds evaluation population')
    return dict(windows=len(state['keys']), variants={'internal': {n: m.compute() for n, m in meters.items()}},
        trajectory=trajectory_report(state['errors']), seconds=time.perf_counter()-tick,
        planner_used=False, GT_reference_run=False, scoring='full grid, internal predicted ego; no alignment')
