import copy
import time
import numpy as np
import torch
from real_motion.ego_navigation import relative_se2, poses_from_se2
from tools.final_semantic_evaluation import metrics
from .statistics import edits_and_margins, confusion, input_report, merge_edits, pose_error, trajectory_report, rigid_pose_errors, BINS
from .input_audit import identity, input_confusions

ROUTES = ('occ_gt', 'occ_all_known_DIAGNOSTIC', 'occ_internal', 'stc_gt',
          'stc_internal', 'stc_GTscene_ego_DIAGNOSTIC')
INDICES = (1, 3, 5)


@torch.no_grad()
def snapshot(predictor, fw):
    ids = torch.tensor(INDICES, device=fw['prior'].device)
    prior = fw['prior'].index_select(0, ids); logits = fw['logits'].index_select(0, ids)
    shape = fw['shape']
    result = dict(transport=prior.argmax(1).reshape(3, *shape).byte().cpu().numpy(),
        semantic_refine=logits.argmax(1).reshape(3, *shape).byte().cpu().numpy(),
        prior_logits=predictor.body.columns.prior_logits(prior).float().cpu().numpy(),
        logits=logits.float().cpu().numpy())
    # Only low-dimensional feature summaries persist. Dense outputs never enter checkpoints.
    f = fw['field'].detach().float().cpu().numpy()
    result['field'] = dict(mean=f.mean((0, 2, 3)).tolist(), std=f.std((0, 2, 3)).tolist(),
                           rms=float(np.sqrt(np.square(f).mean())))
    if fw.get('ego') is not None:
        result['ego'] = {k: fw['ego'][k][0].float().cpu().numpy() for k in ('se2', 'prior')}
    return result


@torch.no_grad()
def head_from_history_field(predictor, field, row):
    """Same FP32 head interface, solely for the GT-all-known validity intervention."""
    from tools.scene_ego_head.model import prepare_scene
    raw, e = row[1], row[2][0]
    known = np.asarray(raw['history_observed'], bool) & (raw['history_occ'] >= 0) & (raw['history_occ'] < 18)
    scene = prepare_scene(field, observed=known, history_poses=raw['history_poses'], ego_history=e,
        grid=predictor.provider.pcfg.grid, pool_side=predictor.head.config.pool_side)
    device = field.device
    out = predictor.head(torch.as_tensor(np.asarray(e)[None], device=device),
                         torch.as_tensor(np.asarray(row[2][1])[None], device=device), scene)
    return {k: out[k][0].float().cpu().numpy() for k in ('se2', 'prior')}


def source_geometry(row):
    state = row[1]['_column_causal_preparation']['prepared_state']
    regs = row[1]['_column_causal_preparation']['registrations']
    velocity = list(state['velocities'].values())
    reads = row[3][0]['shared_reads']; centers = np.asarray(reads['source_center'])
    valid = np.asarray(reads['source_valid'], bool); times = np.asarray(row[2][0])[:, -1]*1.5
    deviations = []
    for c, v in zip(centers, valid):
        if v.sum() >= 3:
            design = np.stack((np.ones(v.sum()), times[v]), 1)
            fit = design @ np.linalg.lstsq(design, c[v], rcond=None)[0]
            deviations.append(float(np.sqrt(np.square(c[v]-fit).sum(1).mean())))
    return dict(sources=len(state['current']), class_counts=np.bincount(row[0]['source_class_id'], minlength=18).tolist(),
        tracks_at_least3=sum(sum(r is not None for r in v) >= 3 for v in regs),
        velocity_entries=len(velocity), speed_mps=[float(np.linalg.norm(v[:2])) for v in velocity],
        deviation_from_linear_motion_m=deviations,
        known_fraction=float(np.asarray(row[1]['history_observed']).mean()))


def row_report(predictor, source, job, artifact=None):
    check = identity(source, job)
    rec, raw, times = source.history(job, 'occ')
    camrec, camera, camtimes = source.history(job, 'stc')
    if (rec != camrec or not np.array_equal(times, camtimes)
            or not np.array_equal(np.asarray(raw['history_poses']), np.asarray(camera['history_poses']))):
        raise RuntimeError('paired historical keys/times/ego poses differ')
    if artifact:
        for token, sem in zip(job.history_tokens, camera['history_occ']): artifact.check(job.scene_name, token, sem)
    # Both historical observations are explicitly permitted diagnostic inputs. No future labels here.
    poses, commands = source.poses(job), source.commands(job)
    gt_row = predictor.prepare(rec, raw, times=times)
    camera_row = predictor.prepare(camrec, camera, times=camtimes)
    allknown = {k: raw[k] for k in ('history_occ', 'history_observed', 'history_poses')}
    allknown['history_observed'] = np.ones_like(raw['history_observed'], dtype=bool)
    all_row = predictor.prepare(rec, allknown, times=times)
    snapshots = {}
    all_known_ego = None
    for name, row, condition in (
            ('occ_gt', gt_row, dict(poses=poses)),
            ('occ_all_known_DIAGNOSTIC', all_row, dict(poses=poses)),
            ('occ_internal', gt_row, dict(commands=commands)),
            ('stc_gt', camera_row, dict(poses=poses)),
            ('stc_internal', camera_row, dict(commands=commands))):
        fw = predictor.forward(row, **condition); snapshots[name] = snapshot(predictor, fw)
        if name == 'occ_all_known_DIAGNOSTIC':
            all_known_ego = head_from_history_field(predictor, fw['field'], (row[0], row[1], (row[2][0], commands, None), row[3]))
        del fw
    gt_ego = snapshots['occ_internal']['ego']; stc_ego = snapshots['stc_internal']['ego']
    if not np.array_equal(gt_ego['prior'], stc_ego['prior']) or not np.array_equal(gt_ego['prior'], all_known_ego['prior']):
        raise RuntimeError('history-only ego prior unexpectedly depends on input modality')
    # Oracle HISTORY scene only, not GT future pose alignment; this is NOT a camera method score.
    crossposes = poses_from_se2(raw['history_poses'][-1], gt_ego['se2'])
    fw = predictor.forward(camera_row, poses=crossposes)
    snapshots['stc_GTscene_ego_DIAGNOSTIC'] = snapshot(predictor, fw); del fw
    # All SIX dense forecasts precede any future occupancy/moving annotation reads.
    targets = source.targets(job, INDICES)
    moving = source.moving(job, INDICES, (1., 2., 3.), predictor.provider.pcfg.grid, predictor.provider.workers)
    next_gt = source.targets(job, (0,))[0]  # input-frame sanity only, never a prediction input
    counts, edits = {}, {}
    for name, value in snapshots.items():
        edits[name] = edits_and_margins(value, targets, moving)
        for route in ('transport', 'semantic_refine'):
            c = metrics.empty(3)
            for h in range(3): metrics.add(c, h, value[route][h], targets[h], moving[h])
            counts[name+'/'+route] = {k: v.tolist() for k, v in c.items()}
    target_poses = relative_se2(raw['history_poses'][-1], poses)
    errors = {name: pose_error(value, target_poses) for name, value in
              (('GT_history_head', gt_ego['se2']), ('GT_all_known_head_DIAGNOSTIC', all_known_ego['se2']),
               ('STC_history_head', stc_ego['se2']), ('kinematic_prior', gt_ego['prior']))}
    head_values = dict(GT_history_head=gt_ego['se2'], GT_all_known_head_DIAGNOSTIC=all_known_ego['se2'],
                       STC_history_head=stc_ego['se2'])
    corrections = {name: pose_error(value, gt_ego['prior']) for name, value in head_values.items()}
    rigid = {name: rigid_pose_errors(raw['history_poses'][-1], value, poses) for name, value in head_values.items()}
    # Current input confusion split by HISTORICAL visibility, solely for diagnostic reporting.
    current_gt, current_cam = raw['history_occ'][-1], camera['history_occ'][-1]
    known = np.asarray(raw['history_observed'][-1], bool)
    split = {name: confusion(current_cam[mask], current_gt[mask]).tolist()
             for name, mask in (('GT_history_known', known), ('GT_history_unknown', ~known))}
    return dict(key=[job.scene_name, job.t0_token], identity=check,
        forecast_counts=counts, edits=edits, trajectory_errors=errors, corrections_to_prior=corrections, rigid_pose_errors=rigid,
        input_confusions=input_confusions(current_cam, current_gt, raw['history_occ'][-2], next_gt),
        input_visibility_split=split, commands=np.asarray(commands).tolist(),
        fields={name: snapshots[name]['field'] for name in ('occ_gt', 'occ_all_known_DIAGNOSTIC', 'stc_gt')},
        geometry=dict(occ=source_geometry(gt_row), stc=source_geometry(camera_row)),
        head_pose_change=pose_error(stc_ego['se2'], gt_ego['se2']))


def checked_rows(state, keys, contract, volume):
    if state['contract'] != contract or len(state['rows']) > len(keys): raise RuntimeError('diagnostic contract/cursor changed')
    for i, row in enumerate(state['rows']):
        if row['key'] != keys[i]: raise RuntimeError('diagnostic population prefix changed')
        expected_routes = {r+'/'+v for r in ROUTES for v in ('transport', 'semantic_refine')}
        if set(row['forecast_counts']) != expected_routes: raise RuntimeError('saved diagnostic routes differ')
        for counts in row['forecast_counts'].values():
            if set(counts) != {'confusion', 'moving'}: raise RuntimeError('saved count fields differ')
            c, m = (np.asarray(counts[k]) for k in ('confusion', 'moving'))
            if (c.dtype.kind not in 'iu' or m.dtype.kind not in 'iu' or c.shape != (3, 18, 18) or m.shape != c.shape
                    or (c < 0).any() or (m < 0).any() or (m > c).any() or np.any(c.sum((1, 2)) != volume)):
                raise RuntimeError('saved diagnostic integer counts invalid')
        for name in ('input_confusions', 'input_visibility_split'):
            for v in row[name].values():
                c = np.asarray(v)
                if c.shape != (18, 18) or c.dtype.kind not in 'iu' or (c < 0).any():
                    raise RuntimeError('saved input confusion invalid')
                if name == 'input_confusions' and c.sum() != volume: raise RuntimeError('saved input totals invalid')
        if sum(np.asarray(c).sum() for c in row['input_visibility_split'].values()) != volume:
            raise RuntimeError('saved input visibility totals invalid')
        for v in row['trajectory_errors'].values():
            for k in ('xy', 'yaw_deg'):
                x = np.asarray(v[k])
                if x.shape != (6,) or not np.isfinite(x).all() or (x < 0).any(): raise RuntimeError('saved ego errors invalid')


def aggregate(rows):
    reports = {}
    for name in rows[0]['forecast_counts']:
        c = {k: np.sum([np.asarray(r['forecast_counts'][name][k], np.int64) for r in rows], axis=0)
             for k in ('confusion', 'moving')}
        reports[name] = metrics.report(c, (1., 2., 3.))
    inputs = {name: input_report(np.sum([r['input_confusions'][name] for r in rows], axis=0))
              for name in rows[0]['input_confusions']}
    split = {name: input_report(np.sum([r['input_visibility_split'][name] for r in rows], axis=0))
             for name in rows[0]['input_visibility_split']}
    trajectories = {name: trajectory_report([r['trajectory_errors'][name] for r in rows])
                    for name in rows[0]['trajectory_errors']}
    corrections = {name: trajectory_report([r['corrections_to_prior'][name] for r in rows])
                   for name in rows[0]['corrections_to_prior']}
    rigid = {name: {k: dict(mean=np.mean([r['rigid_pose_errors'][name][k] for r in rows], axis=0).tolist(),
                           p90=np.percentile([r['rigid_pose_errors'][name][k] for r in rows], 90, axis=0).tolist())
                   for k in rows[0]['rigid_pose_errors'][name]} for name in rows[0]['rigid_pose_errors']}
    edit_report = {name: merge_edits([r['edits'][name] for r in rows]) for name in ROUTES}
    avg = lambda name: reports[name+'/semantic_refine']['average']
    def delta(a, b):
        return {k: avg(a)[k]-avg(b)[k] if avg(a)[k] is not None and avg(b)[k] is not None else None
                for k in ('IoU', 'mIoU', 'MovingMicro')}
    geometry = {}
    for name in ('occ', 'stc'):
        g = [r['geometry'][name] for r in rows]; speeds = [v for r in g for v in r['speed_mps']]
        geometry[name] = dict(sources=sum(r['sources'] for r in g), tracks_at_least3=sum(r['tracks_at_least3'] for r in g),
            class_counts=np.sum([r['class_counts'] for r in g], axis=0).tolist(),
            known_fraction_mean=float(np.mean([r['known_fraction'] for r in g])),
            speed_p50_p90_mps=np.percentile(speeds, (50, 90)).tolist() if speeds else None,
            linear_motion_deviation_median=float(np.median([v for r in g for v in r['deviation_from_linear_motion_m']]))
                if any(r['deviation_from_linear_motion_m'] for r in g) else None,
            deviation_note='includes actual acceleration/turns; NOT a pure sensor-noise measure')
    feature_shift = {}
    for name in ('occ_all_known_DIAGNOSTIC', 'stc_gt'):
        a = np.array([r['fields']['occ_gt']['mean'] for r in rows])
        b = np.array([r['fields'][name]['mean'] for r in rows])
        feature_shift[name] = dict(channel_mean_RMS_change=float(np.sqrt(np.square(a-b).mean())),
            field_RMS_mean=float(np.mean([r['fields'][name]['rms'] for r in rows])),
            note='descriptive channel statistics, NOT a formal distribution-shift test')
    evidence = dict(
        repair_gain={name: {k: reports[name+'/semantic_refine']['average'][k]-reports[name+'/transport']['average'][k]
            if reports[name+'/semantic_refine']['average'][k] is not None and reports[name+'/transport']['average'][k] is not None else None
            for k in ('IoU', 'mIoU', 'MovingMicro')} for name in ('occ_gt', 'stc_gt', 'stc_internal')},
        STC_minus_GT_head_FDE_m=trajectories['STC_history_head']['FDE3s_m']-trajectories['GT_history_head']['FDE3s_m'],
        GT_allknown_minus_GT_head_FDE_m=trajectories['GT_all_known_head_DIAGNOSTIC']['FDE3s_m']-trajectories['GT_history_head']['FDE3s_m'],
        STC_head_minus_kinematic_FDE_m=trajectories['STC_history_head']['FDE3s_m']-trajectories['kinematic_prior']['FDE3s_m'],
        t0_known_policy='GT history lidar mask vs STC all-true; unchanged formal policy',
        prior_validity_policy='Transport static prior reads t0 semantics; encoder/memory use historical known',
        causal_verdict='pending review of these paired diagnostics; no automatic root-cause or method selection')
    return dict(status='complete', windows=len(rows), diagnostic_only=True, scientific_updates=0,
        reports=reports, input_quality=inputs, input_visibility_split=split, trajectory=trajectories,
        corrections_to_prior=corrections, rigid_pose_errors=rigid,
        edits_and_prior_competition=edit_report, geometry=geometry, feature_shift=feature_shift, evidence_summary=evidence,
        isolated_differences=dict(GT_known_to_allTrue=delta('occ_all_known_DIAGNOSTIC', 'occ_gt'),
            STC_GTscene_head_minus_STC_head=delta('stc_GTscene_ego_DIAGNOSTIC', 'stc_internal')),
        histogram_edges=['-inf' if v == -np.inf else '+inf' if v == np.inf else float(v) for v in BINS],
        interpretation_limits=[
            'GT-all-known intervention isolates historical validity on GT semantics, NOT a valid STC mask policy.',
            'STC+GT-history-head is an oracle-history diagnostic, NOT deployable Camera performance.',
            'Axis/neighbor scores never change inputs, choose transforms or prove a release bug alone.',
            'Residual exceeding old-winner barrier is necessary but not sufficient to beat other classes.',
            'Fixed DEV64 diagnosis is not independent test or training/epsilon/head selection.',
            'Voxel net corrections do not equal class-balanced mIoU changes; absence of edits is not proof of cause.'])


def evaluate(predictor, source, jobs, keys, contract, *, artifact=None, saved=None, save=None, stop_event=None):
    state = dict(contract=contract, rows=[]) if saved is None else copy.deepcopy(saved)
    checked_rows(state, keys, contract, int(np.prod(predictor.provider.pcfg.grid.shape_hwd)))
    try:
        for i, job in enumerate(jobs):
            if i < len(state['rows']):
                # On resume validate original selected history bytes again (without model recomputation).
                if artifact:
                    _, camera, _ = source.history(job, 'stc')
                    for t, sem in zip(job.history_tokens, camera['history_occ']): artifact.check(job.scene_name, t, sem)
                continue
            if stop_event is not None and stop_event.is_set(): raise InterruptedError('diagnostic complete window saved')
            tick = time.perf_counter(); value = row_report(predictor, source, job, artifact)
            value['seconds'] = time.perf_counter()-tick
            if value['key'] != keys[i]: raise RuntimeError('live diagnostic key differs')
            state['rows'].append(value)
            if save: save(state)
            if (i+1) % 8 == 0: print(f'STC_CAUSE {i+1}/{len(jobs)} seconds/window={value["seconds"]:.3f}', flush=True)
    finally:
        if save: save(state)
    if len(state['rows']) != len(keys): raise RuntimeError('complete diagnostic population required')
    result = aggregate(state['rows']); result['seconds'] = sum(r['seconds'] for r in state['rows'])
    if artifact: result['STC_artifact_audit'] = artifact.verify()
    return result
