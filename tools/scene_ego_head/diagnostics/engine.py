"""No optimizer steps; causal ego errors, VJPs, and matched frozen forecasts."""
import copy
import time
import numpy as np
import torch
from tools.scene_ego_head.model import prepare_scene
from tools.scene_ego_head.integration import forward_pred_window, forecast_conditioned
from tools.shared_history_experiments.model import encode_window
from tools.shared_history_experiments.refine.engine import semantic_loss, Metrics, window_from_record, gt_moving_support_sequence, moving_support_masks
from tools.joint_occ_experiments.model import ego_geometry_loss
from real_motion.ego_navigation import relative_se2


def errors(predicted, target):
    d = np.asarray(predicted)-np.asarray(target)
    return dict(xy=np.linalg.norm(d[:, :2], axis=-1).tolist(),
                yaw=(np.abs(np.arctan2(np.sin(d[:, 2]), np.cos(d[:, 2])))*180/np.pi).tolist())


def trajectory_report(rows):
    if not rows: return {}
    return {name: dict(ADE_m=float(np.mean([r[name]['xy'] for r in rows])),
        FDE_3s_m=float(np.mean([r[name]['xy'][-1] for r in rows])),
        xy_mean_m=np.mean([r[name]['xy'] for r in rows], 0).tolist(),
        yaw_mean_deg=np.mean([r[name]['yaw'] for r in rows], 0).tolist()) for name in rows[0]}


def similarity(left, right):
    a, b = left.detach().float().flatten(), right.detach().float().flatten()
    if not bool(torch.isfinite(a).all() & torch.isfinite(b).all()): raise RuntimeError('nonfinite diagnostic gradients')
    na, nb = float(a.norm()), float(b.norm())
    return dict(occupancy_norm=na, ego_norm=nb, cosine=float(torch.dot(a, b)/(a.norm()*b.norm())) if na and nb else None,
                occupancy_over_ego=na/nb if nb else None)


def gradient_window(model, provider, row):
    """Same frozen forward for both losses; eval mode, no clip/backward/Adam."""
    flags = [p.requires_grad for p in model.parameters()]; mode = model.training
    try:
        model.eval().requires_grad_(True)
        with torch.enable_grad():
            fw = forward_pred_window(model, provider, row)
            losses, ce, lovasz = semantic_loss(fw['logits'], row[1]['future_gt_occ'])
            pose = fw['ego']['se2']; direct = ego_geometry_loss(pose, np.asarray(row[2][2])[None])
            occupancy = losses.mean()
            if not bool(torch.isfinite(occupancy) & torch.isfinite(direct)): raise RuntimeError('nonfinite diagnostic loss')
            parameters = list(model.ego.parameters()); roots = [pose, fw['field'], *parameters]
            a = torch.autograd.grad(occupancy, roots, retain_graph=True, allow_unused=True)
            b = torch.autograd.grad(direct, roots, allow_unused=True)
            vector = lambda values: torch.cat([(torch.zeros_like(p) if v is None else v).float().flatten()
                for p, v in zip(parameters, values)])
            pa = torch.zeros_like(pose) if a[0] is None else a[0]
            pb = torch.zeros_like(pose) if b[0] is None else b[0]
            # dL/d(R*yaw) = (dL/dyaw)/R, giving translation-equivalent R10 units.
            scale = pose.new_tensor([1., 1., .1])
            fa = torch.zeros_like(fw['field']) if a[1] is None else a[1]
            fb = torch.zeros_like(fw['field']) if b[1] is None else b[1]
            result = dict(key=[row[0]['scene_name'], row[0]['t0_token']], CE=float(ce.mean()),
                Lovasz=float(lovasz.mean()), ego_R10=float(direct),
                gradients=dict(pose_R10=similarity(pa*scale, pb*scale), shared_F=similarity(fa, fb),
                               ego_parameters=similarity(vector(a[2:]), vector(b[2:]))),
                per_horizon_pose_R10=[similarity(pa[:, h]*scale, pb[:, h]*scale) for h in range(6)],
                trajectory={name: errors(value[0].detach().float().cpu().numpy(), row[2][2])
                            for name, value in (('head', pose), ('prior', fw['ego']['prior']))})
        return result
    finally:
        for p, flag in zip(model.parameters(), flags): p.requires_grad_(flag)
        model.train(mode)


@torch.no_grad()
def ego_window(model, provider, row):
    """Only history encoder + scene ego, NOT STWM/Strong/Refiner or dense labels."""
    raw, e, commands = row[1], row[2][0], row[2][1]
    observed = np.asarray(raw['history_observed'], bool)
    known = observed & (np.asarray(raw['history_occ']) >= 0) & (np.asarray(raw['history_occ']) < 18)
    device = provider.device
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        field = model.transport.history_encoder(torch.as_tensor(np.asarray(raw['history_occ']), device=device),
                                               torch.as_tensor(observed, device=device))
        scene = prepare_scene(field, observed=known, history_poses=raw['history_poses'], ego_history=e,
                              grid=provider.pcfg.grid, pool_side=model.ego.config.pool_side)
        result = model.ego(torch.as_tensor(np.asarray(e)[None], device=device),
                           torch.as_tensor(np.asarray(commands)[None], device=device), scene)
    return {name: errors(value[0].float().cpu().numpy(), row[2][2])
            for name, value in (('head', result['se2']), ('prior', result['prior']))}


def ego_checks(model, provider, batches, *, saved=None, save=None, stop_event=None, progress=None):
    protocol = 'frozen_scene_ego_trajectory_only_v1'; model.eval()
    state = dict(protocol=protocol, rows=[]) if saved is None else copy.deepcopy(saved)
    if state.get('protocol') != protocol: raise RuntimeError('ego trajectory state changed')
    done = len(state['rows']); offset = 0; tick = time.perf_counter()
    try:
        for rows in batches:
            for row in rows:
                offset += 1
                if offset <= done:
                    if state['rows'][offset-1]['key'] != [row[0]['scene_name'], row[0]['t0_token']]:
                        raise RuntimeError('saved trajectory population order changed')
                    continue
                if stop_event is not None and stop_event.is_set(): raise InterruptedError('saved complete ego trajectory windows')
                value = ego_window(model, provider, row)
                state['rows'].append(dict(key=[row[0]['scene_name'], row[0]['t0_token']], errors=value))
                if offset % 8 == 0 and save: save(state)
                if offset % 32 == 0:
                    print(f'EGO_ONLY_CHECK {offset}', flush=True)
                    if progress: progress(dict(event='ego_trajectory_check', windows=offset))
    finally:
        if save: save(state)
    if offset < done: raise RuntimeError('saved trajectory prefix exceeds population')
    result = trajectory_report([r['errors'] for r in state['rows']])
    return dict(windows=len(state['rows']), trajectory=result, variants={}, seconds=time.perf_counter()-tick,
        head_minus_prior={k: result['head'][k]-result['prior'][k] for k in ('ADE_m', 'FDE_3s_m')},
        dense_forecasting=False)


@torch.no_grad()
def external_forward(model, provider, row, poses):
    """GT-trained body, but NO GT future pose/background or auxiliary ego head."""
    record, raw, (e, _, _), history = row
    historical = {k: raw[k] for k in ('history_occ', 'history_observed', 'history_poses')}
    state = history[0]['state']; current = np.asarray(state['current_pose'])
    if not np.array_equal(current, np.asarray(raw['history_poses'][-1])): raise RuntimeError('history pose identity differs')
    if not np.array_equal(record['source_class_id'], [int(s['class_id']) for s in state['current']]):
        raise RuntimeError('source identity/order differs')
    xyz = np.asarray([s['centroid_world'] for s in state['current']]).reshape(-1, 3)
    native = (xyz-current[:3, 3]) @ current[:3, :3]
    if not np.allclose(native[:, :2], record['source_centroid_xy_t0_m'], atol=2e-4, rtol=0):
        raise RuntimeError('source centroid/order differs')
    if not np.allclose(record['anchors_xy_t0_m'], np.asarray(record['source_centroid_xy_t0_m'])[:, None]+
                       np.asarray(record['kta_displacement_xy_m']), atol=2e-4, rtol=0):
        raise RuntimeError('KTA must use fixed t0 axes')
    device = provider.device
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        field, reads, times = encode_window(model, historical, history, e, device)
        output = model.transport(field, reads, times, torch.as_tensor(record['source_class_id'], device=device).long(),
                                 torch.as_tensor(record['kta_displacement_xy_m'], device=device).float())
        lifted = model.columns.lift_history(field)
    return forecast_conditioned(model, provider, row, dict(field=field, output=output, lifted=lifted), poses)


@torch.no_grad()
def matched_external(gt_model, internal_model, provider, source, batches, planner, *, saved=None, save=None,
                     stop_event=None, progress=None, remove_threshold=None):
    protocol = 'gt500_external_vs_internal500_matched_v1'
    routes = ('GT500_external', 'Pred500_internal'); readouts = ('transport', 'semantic_refine')
    meters = {r: {n: Metrics() for n in readouts} for r in routes}
    state = dict(protocol=protocol, planner=planner.metadata, seen=0, errors=[])
    if saved:
        if saved.get('protocol') != protocol or saved.get('planner') != planner.metadata: raise RuntimeError('matched eval provenance changed')
        state = copy.deepcopy(saved)
        if type(state['seen']) is not int or not 0 <= state['seen'] <= len(planner.requested): raise ValueError('invalid eval cursor')
        expected = sum(tuple(k) in planner.rows for k in planner.requested[:state['seen']])
        if len(state['errors']) != expected or set(state['counts']) != set(routes): raise ValueError('invalid eval prefix')
        for r in routes:
            if set(state['counts'][r]) != set(readouts): raise ValueError('readouts differ')
            for n, meter in meters[r].items():
                if set(state['counts'][r][n]) != set(vars(meter)): raise ValueError('count fields differ')
                for k, old in vars(meter).items():
                    value = np.asarray(state['counts'][r][n][k])
                    if value.dtype.kind not in 'iu' or value.shape != old.shape or (value < 0).any(): raise ValueError('invalid counts')
                    setattr(meter, k, value.astype(np.int64))
    def persist():
        state['counts'] = {r: {n: {k: v.tolist() for k, v in vars(m).items()} for n, m in values.items()} for r, values in meters.items()}
        if save: save(state)
    offset = 0; tick = time.perf_counter()
    gt_model.eval(); internal_model.eval()
    try:
        for rows in batches:
            for row in rows:
                key = (row[0]['scene_name'], row[0]['t0_token'])
                if offset >= len(planner.requested) or key != planner.requested[offset]: raise RuntimeError('population order differs')
                offset += 1
                if offset <= state['seen']: continue
                if stop_event is not None and stop_event.is_set(): raise InterruptedError('saved matched complete windows')
                if key not in planner.rows: state['seen'] = offset; continue
                poses = planner.poses(row[0], row[1]['history_poses'][-1])
                output = {}; ego = {}
                for route, model in ((routes[0], gt_model), (routes[1], internal_model)):
                    fw = external_forward(model, provider, row, poses) if route == routes[0] else forward_pred_window(model, provider, row)
                    output[route] = {n: v.argmax(1).reshape(6, *fw['shape']).byte().cpu().numpy()
                                    for n, v in (('transport', fw['prior']), ('semantic_refine', fw['logits']))}
                    values = relative_se2(row[1]['history_poses'][-1], poses) if route == routes[0] else fw['ego']['se2'][0].float().cpu().numpy()
                    ego[route] = errors(values, row[2][2]); del fw
                # GT annotations/occupancy enter metrics only after BOTH frozen predictions.
                w = window_from_record(row[0]); raw = row[1]
                support = gt_moving_support_sequence(source.nusc, w.t0_token, w.future_tokens,
                    tuple(.5*(h+1) for h in range(6)), grid=provider.pcfg.grid, workers=provider.workers)
                moving = moving_support_masks(support, raw['future_gt_occ'].shape[1:])
                delta = {r: {n: Metrics() for n in readouts} for r in routes}
                for r in routes:
                    for n in readouts:
                        for i, h in enumerate((1, 3, 5)): delta[r][n].update(i, output[r][n][h], raw['future_gt_occ'][h], moving[h])
                for r in routes:
                    for n in readouts:
                        for k in vars(meters[r][n]): getattr(meters[r][n], k)[:] += getattr(delta[r][n], k)
                state['errors'].append(ego); state['seen'] = offset
                if offset % 8 == 0: persist()
                if offset % 16 == 0:
                    print(f'GT500_EXTERNAL_MATCHED {offset}/{len(planner.requested)} covered={len(state["errors"])}', flush=True)
                    if progress: progress(dict(event='gt500_external', windows=len(state['errors'])))
    finally: persist()
    if offset != len(planner.requested): raise RuntimeError('incomplete requested population')
    reports = {r: {n: m.compute() for n, m in values.items()} for r, values in meters.items()}
    return dict(windows=len(state['errors']), variants=reports, trajectory=trajectory_report(state['errors']),
        population=planner.metadata, seconds=time.perf_counter()-tick,
        external_minus_internal={k: reports[routes[0]]['semantic_refine'][k]-reports[routes[1]]['semantic_refine'][k]
                                 for k in ('mIoU', 'IoU', 'MovingMicro')}, GT_pose_reference_run=False)
