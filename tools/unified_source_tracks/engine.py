"""Frozen matched DEV64 screen. GT annotation targets are evaluation-only."""
import copy
import time
import numpy as np
import torch
from tools.final_semantic_evaluation import metrics
from tools.scene_ego_head.integration import forecast_conditioned
from tools.scene_ego_head.frozen_gt.engine import deterministic_entrance
from real_motion.motion_transport import annotation_map, dynamic_annotations, match_sources_to_annotations
from real_motion.motion_transport_v2 import build_motion_targets
from .geometry import ARMS, build_tracks, track_velocity, make_row, motion_errors


def label_motion(source, job, row):
    # Existing GT label matching convention (4m), NOT a new source association
    # rule and never exposed to preparation, STWM or feature transport.
    nusc = source.source.nusc; current = row[3][0]['state']['current']
    anns = dynamic_annotations(nusc, job.t0_token)
    tokens = match_sources_to_annotations(current, anns)
    target = build_motion_targets(current, {}, tokens, {a['instance_token']: a for a in anns},
        [annotation_map(nusc, t) for t in job.future_tokens], row[1]['history_poses'][-1])
    return target['target_displacement_xy_m'], target['target_valid']


@torch.no_grad()
def unified_forward(predictor, row, first, poses):
    # Same frozen encoder input/weights: reuse live F/lifted memory, NEVER a bank.
    model, p = predictor.body, predictor.provider; record = row[0]
    with torch.autocast(device_type=p.device.type, dtype=torch.bfloat16, enabled=p.device.type == 'cuda'):
        times = torch.as_tensor(row[2][0][:, -1], device=p.device)
        reads = {k: torch.as_tensor(v, device=p.device) for k, v in row[3][0]['shared_reads'].items()}
        output = model.transport(first['field'], reads, times,
            torch.as_tensor(record['source_class_id'], device=p.device).long(),
            torch.as_tensor(record['kta_displacement_xy_m'], device=p.device).float())
    with torch.autocast(device_type=p.device.type, dtype=torch.bfloat16, enabled=p.device.type == 'cuda'):
        lifted = model.columns.lift_history(first['field'])
    prepared = dict(field=first['field'], lifted=lifted, output=output)
    backgrounds = first['state']['column_backgrounds']
    # Explicit fixed Strong/CLEAR CONTROL: isolate track/anchor from an additional
    # hard-background algorithm change. All routes have the same GT future ego.
    return forecast_conditioned(model, p, row, prepared, poses,
                                background_builder=lambda *a: backgrounds)


def error_report(rows, kind, arm):
    result = {}
    for group in ('all', 'one_point', 'two_points', 'three_plus'):
        values = [sum((r['modalities'][kind]['kta_errors'][arm][group][h] for r in rows), []) for h in range(3)]
        result[group] = dict(counts=[len(v) for v in values],
            mean_m=[float(np.mean(v)) if v else None for v in values],
            p90_m=[float(np.quantile(v, .9)) if v else None for v in values])
    return result


@torch.no_grad()
def evaluate(predictor, source, jobs, keys, contract, *, kinds=('occ', 'stc'), saved=None,
             save=None, stop_event=None, checkpoint_every=8, labeler=label_motion):
    names = [f'{kind}/{arm}/{route}' for kind in kinds for arm in ARMS for route in ('transport', 'semantic_refine')]
    state = dict(contract=contract, keys=[], counts={n: metrics.empty(3) for n in names}, rows=[], seconds=0.)
    if saved is not None:
        state = copy.deepcopy(saved); cursor = len(state['keys'])
        if state['contract'] != contract or state['keys'] != keys[:cursor] or cursor > len(keys):
            raise RuntimeError('saved contract/key prefix changed')
        if len(state['rows']) != cursor or [r['key'] for r in state['rows']] != state['keys'] or set(state['counts']) != set(names):
            raise RuntimeError('saved row/routes changed')
        for c in state['counts'].values():
            for name in ('confusion', 'moving'):
                a = np.asarray(c[name])
                if a.shape != (3, 18, 18) or a.dtype.kind not in 'iu' or np.any(a < 0):
                    raise RuntimeError('saved integer counts invalid')
                c[name] = a.astype(np.int64)
            if np.any(c['confusion'].sum((1, 2)) != cursor*np.prod(predictor.provider.pcfg.grid.shape_hwd)) or np.any(c['moving'] > c['confusion']):
                raise RuntimeError('saved complete-window totals invalid')
        if not np.isfinite(state['seconds']) or state['seconds'] < 0: raise RuntimeError('saved time invalid')
    def persist():
        if save:
            value = copy.deepcopy(state)
            value['counts'] = {n: {k: a.tolist() for k, a in c.items()} for n, c in state['counts'].items()}
            save(value)
    start = time.perf_counter()
    try:
        with deterministic_entrance():
            for index, job in enumerate(jobs):
                if index < len(state['keys']): continue
                if stop_event is not None and stop_event.is_set(): raise InterruptedError('saved complete window')
                predictions, pending = {}, dict(key=keys[index], modalities={})
                motion_rows = {}
                poses = source.poses(job)  # explicit GT trajectory condition, not aligned scoring
                for kind in kinds:
                    record, raw, times = source.history(job, kind)
                    base = predictor.prepare(record, raw, times=times)
                    tick = time.perf_counter(); track = build_tracks(base, predictor.provider, times)
                    velocities = {arm: track_velocity(track, mode) for arm, mode in (('unified_two', 'two'), ('unified_fit', 'fit'))}
                    candidates = {arm: make_row(base, track, velocity, predictor.history) for arm, velocity in velocities.items()}
                    preparation_seconds = time.perf_counter()-tick
                    tick = time.perf_counter(); first = predictor.forward(base, poses=poses)
                    for route, dense in predictor.dense(first).items(): predictions[f'{kind}/baseline/{route}'] = dense
                    ktas = dict(baseline=base[0]['kta_displacement_xy_m'])
                    for arm, row in candidates.items():
                        forecast = unified_forward(predictor, row, first, poses)
                        for route, dense in predictor.dense(forecast).items(): predictions[f'{kind}/{arm}/{route}'] = dense
                        ktas[arm] = row[0]['kta_displacement_xy_m']; del forecast
                    motion_rows[kind] = (base, track, ktas)
                    pending['modalities'][kind] = dict(sources=len(track.links),
                        valid_point_counts=np.bincount(track.valid.sum(1), minlength=5).tolist(),
                        association_audit=track.audit, track_rebuild_seconds=preparation_seconds,
                        three_forecasts_seconds=time.perf_counter()-tick,
                        fitted_speed_over_25mps=int((np.linalg.norm(velocities['unified_fit'][:, :2], axis=1) > 25).sum()))
                    del first
                # All modalities/arms finish forecasting before any GT labels,
                # t0/future annotation identities or Moving supports are read.
                targets = source.targets(job, (1, 3, 5))
                moving = source.moving(job, (1, 3, 5), (1., 2., 3.), predictor.provider.pcfg.grid, predictor.provider.workers)
                for kind, (base, track, ktas) in motion_rows.items():
                    target, valid = labeler(source, job, base)
                    pending['modalities'][kind]['kta_errors'] = motion_errors(ktas, track, target, valid)
                delta = {n: metrics.empty(3) for n in names}
                for n in names:
                    for h, at in enumerate((1, 3, 5)): metrics.add(delta[n], h, predictions[n][at], targets[h], moving[h])
                for n in names:
                    for k in ('confusion', 'moving'): state['counts'][n][k] += delta[n][k]
                state['rows'].append(pending); state['keys'].append(keys[index])
                if len(state['keys']) % checkpoint_every == 0: persist()
                if len(state['keys']) % 16 == 0: print(f'UNIFIED_TRACKS {len(state["keys"])}/{len(jobs)}', flush=True)
    finally:
        state['seconds'] += time.perf_counter()-start; persist()
    return dict(status='complete', windows=len(state['keys']), scientific_updates=0,
        reports={n: metrics.report(c, (1., 2., 3.)) for n, c in state['counts'].items()},
        kta_errors={k: {arm: error_report(state['rows'], k, arm) for arm in ARMS} for k in kinds},
        track_audits={k: dict(sources=sum(r['modalities'][k]['sources'] for r in state['rows']),
            valid_point_counts=np.sum([r['modalities'][k]['valid_point_counts'] for r in state['rows']], 0).tolist()) for k in kinds},
        seconds=state['seconds'], strong_background='baseline Strong/CLEAR fixed across all arms',
        persistent_geometry_writes=False, automatic_selection=False,
        execution='deterministic BF16 controlled quality screen, NOT FPS or deployment promotion')
