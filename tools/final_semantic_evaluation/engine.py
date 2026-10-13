import copy
import time
import numpy as np
from real_motion.waymo_i2world import WaymoMetrics
from real_motion.waymo_i2world_10hz import format_10hz_reports
from tools.real_motion.joint_long_rollout_common import inherited_observation_masks
from . import metrics


def evaluate(predictor, source, jobs, task, keys, contract, *, saved=None, save=None, stop_event=None, checkpoint_every=8):
    long = task == 'long6'; waymo = task.startswith('waymo')
    indices = (1, 3, 5, 7, 9, 11) if long else (1, 3, 5)
    horizons = tuple(float(i) for i in range(1, 7 if long else 4))
    routes = ('stc_gt', 'stc_pred') if task == 'stc' else ('semantic_refine_rollout',) if long else ('gt',)
    names = ['semantic_refine_rollout'] if long else [r+'/'+n for r in routes for n in ('transport', 'semantic_refine')]
    state = dict(contract=contract, keys=[], counts={n: metrics.empty(len(horizons)) for n in names}, ego_errors=[], seconds=0.)
    if saved:
        if saved['contract'] != contract or saved['keys'] != keys[:len(saved['keys'])] or len(saved['keys']) > len(keys):
            raise RuntimeError('saved evaluation identity/prefix differs')
        if set(saved['counts']) != set(names): raise RuntimeError('saved routes differ')
        state = copy.deepcopy(saved)
        for n in names:
            if set(state['counts'][n]) != {'confusion', 'moving'}: raise RuntimeError('saved count fields differ')
            for k in ('confusion', 'moving'):
                v = np.asarray(state['counts'][n][k])
                if v.shape != (len(horizons), 18, 18) or v.dtype.kind not in 'iu' or np.any(v < 0):
                    raise RuntimeError('saved integer counts invalid')
                state['counts'][n][k] = v.astype(np.int64)
        if len(state['ego_errors']) != (len(state['keys']) if task == 'stc' else 0):
            raise RuntimeError('saved ego error cursor differs')
        if not np.isfinite(state['seconds']) or state['seconds'] < 0: raise RuntimeError('saved time invalid')
        expected = len(state['keys'])*int(np.prod(predictor.provider.pcfg.grid.shape_hwd))
        for c in state['counts'].values():
            if np.any(c['confusion'].sum((1, 2)) != expected) or np.any(c['moving'] > c['confusion']):
                raise RuntimeError('saved confusion totals differ from complete-window cursor')
    def persist():
        if save:
            value = copy.deepcopy(state)
            value['counts'] = {n: {k: v.tolist() for k, v in c.items()} for n, c in state['counts'].items()}
            save(value)
    started = time.perf_counter()
    try:
        for index, job in enumerate(jobs):
            if index < len(state['keys']): continue
            if stop_event is not None and stop_event.is_set(): raise InterruptedError('saved complete evaluation window')
            predictions = {}; error = None
            if waymo:
                record, old = source.prediction_inputs(job)
                raw = {k: old[k] for k in ('history_occ', 'history_observed', 'history_poses')}
                row = predictor.prepare(record, raw)  # trained nominal clock, never retime 10Hz
                fw = predictor.forward(row, poses=old['future_poses'])
                predictions = {'gt/'+k: v for k, v in predictor.dense(fw).items()}; del fw
                targets = source.metric_targets(job); moving = [None]*3
            else:
                record, raw, times = source.history(job, 'stc' if task == 'stc' else 'occ')
                row = predictor.prepare(record, raw, times=times)
                poses = source.poses(job)
                fw = predictor.forward(row, poses=poses[:6]); first = predictor.dense(fw); del fw
                if long:
                    raw2 = dict(history_occ=first['semantic_refine'][-4:].copy(), history_poses=poses[2:6],
                        history_observed=inherited_observation_masks(raw, poses[2:6], predictor.provider.pcfg.grid, 1))
                    rec2 = dict(scene_name=job.scene_name, t0_token=job.future_tokens[5],
                        history_tokens=job.future_tokens[2:6], future_tokens=job.future_tokens[6:12])
                    times2 = [source.source.nusc.get('sample', t)['timestamp']/1e6 for t in rec2['history_tokens']]
                    row2 = predictor.prepare(rec2, raw2, times=times2)
                    fw2 = predictor.forward(row2, poses=poses[6:12]); second = predictor.dense(fw2); del fw2
                    predictions['semantic_refine_rollout'] = np.concatenate((first['semantic_refine'], second['semantic_refine']))
                else:
                    prefix = 'stc_gt/' if task == 'stc' else 'gt/'
                    predictions.update({prefix+k: v for k, v in first.items()})
                    if task == 'stc':
                        commands = source.commands(job)
                        fw2 = predictor.forward(row, commands=commands)
                        predictions.update({'stc_pred/'+k: v for k, v in predictor.dense(fw2).items()})
                        from real_motion.ego_navigation import relative_se2
                        target = relative_se2(raw['history_poses'][-1], poses[:6])
                        p = fw2['ego']['se2'][0].float().cpu().numpy(); delta = p-target
                        error = dict(xy=np.linalg.norm(delta[:, :2], axis=1).tolist(),
                            yaw=(np.abs(np.arctan2(np.sin(delta[:, 2]), np.cos(delta[:, 2])))*180/np.pi).tolist())
                        del fw2
                # All routes / all twelve rollout predictions precede future label and moving-annotation reads.
                targets = source.targets(job, indices)
                moving = source.moving(job, indices, horizons, predictor.provider.pcfg.grid, predictor.provider.workers)
            delta_counts = {n: metrics.empty(len(horizons)) for n in names}
            for n in names:
                for h, at in enumerate(indices): metrics.add(delta_counts[n], h, predictions[n][at], targets[h], moving[h])
            for n in names:
                for k in ('confusion', 'moving'): state['counts'][n][k] += delta_counts[n][k]
            state['keys'].append(keys[index])
            if error is not None: state['ego_errors'].append(error)
            if len(state['keys']) % checkpoint_every == 0: persist()
            if len(state['keys']) % 16 == 0: print(f'FINAL_{task.upper()} {len(state["keys"])}/{len(jobs)}', flush=True)
    finally:
        state['seconds'] += time.perf_counter()-started; persist()
    if state['keys'] != keys: raise RuntimeError('complete population required')
    reports = {n: metrics.report(c, horizons, moving=not waymo) for n, c in state['counts'].items()}
    if waymo:
        official = {n: WaymoMetrics(c['confusion'], len(jobs)).report() for n, c in state['counts'].items()}
        if task == 'waymo10': official = format_10hz_reports(official)
        reports = official  # native-index reports MUST NOT carry false 1/2/3s labels.
    result = dict(status='complete', task=task, windows=len(jobs), reports=reports,
        accumulated_evaluation_seconds=state['seconds'], scientific_updates=0,
        no_training=True, no_automatic_selection=True, metric_mask='none', future_GT_pose_alignment=False)
    if state['ego_errors']:
        xy = np.mean([r['xy'] for r in state['ego_errors']], 0); yaw = np.mean([r['yaw'] for r in state['ego_errors']], 0)
        result['trajectory'] = dict(ADE_m=float(xy.mean()), FDE3s_m=float(xy[-1]), xy_mean_m=xy.tolist(), yaw_mean_deg=yaw.tolist())
    if long:
        result['rollout'] = dict(history_frames=4, future_frames_per_block=6, blocks=2,
            history2='last4 SemanticRefiner predictions; inherited initial real visibility',
            source_handoff='redetect from predicted histories; no legacy owner/static carry',
            GT_future_ego_conditioned=True, static_carry=False)
        if contract['population'].get('alignment'):
            from tools.real_motion.geniedrive_eval_alignment import compatibility_metrics
            result['public_code_compatibility'] = {}
            for n, c in state['counts'].items():
                v = c['confusion']; diagonal = np.diagonal(v, axis1=1, axis2=2)
                raw = dict(sem_inter=diagonal[:, :17], sem_union=(v.sum(1)+v.sum(2)-diagonal)[:, :17],
                    occ_inter=v[:, :17, :17].sum((1, 2)), occ_union=v.sum((1, 2))-v[:, 17, 17])
                result['public_code_compatibility'][n] = compatibility_metrics(raw)
    return result
