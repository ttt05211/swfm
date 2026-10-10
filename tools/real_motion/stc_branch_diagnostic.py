"""Read-only attribution of frozen four-setting errors; no corrective adapter."""
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np
import torch

from real_motion.stc_camera_protocol import SETTINGS, PLAN_SHA, semantics
from real_motion.stc_causal_geometry import motion_jitter_audit
from real_motion.waymo_i2world import CLASS_NAMES, WaymoMetrics, fingerprint, file_sha256
from tools.real_motion.eval_p0_f9_stc_causal_geometry import pose_errors

PROTOCOL = 'p0_f9_frozen_stc_branch_error_attribution_v1'
STAGES = ('strong', 'transport', 'joint')
ROUTES = tuple(f'{s}/{stage}' for s in SETTINGS for stage in STAGES)
GROUPS = dict(road_sidewalk=(11, 13), ground=(11, 12, 13, 14),
              building_vegetation=(15, 16), dynamic=(2, 3, 4, 5, 6, 7, 9, 10),
              other_static=(0, 1, 8))
EDIT_FIELDS = ('changed', 'added', 'removed', 'relabeled', 'corrected', 'damaged',
               'added_occ_tp', 'added_semantic_tp')


def confusion(pred, gt, shape):
    p = semantics(pred, shape).ravel().astype(np.int64)
    g = semantics(gt, shape).ravel().astype(np.int64)
    return np.bincount(18*g+p, minlength=324).reshape(18, 18)


def quality(counts, ids):
    """Group membership and semantic accuracy are different quantities."""
    c = np.asarray(counts, dtype=np.int64)
    ids = list(ids)
    tp = int(c[np.ix_(ids, ids)].sum())
    gt = int(c[ids, :].sum()); pred = int(c[:, ids].sum())
    semantic_tp = int(np.diag(c)[ids].sum())
    return dict(gt=gt, predicted=pred, tp=tp, fp=pred-tp, fn=gt-tp,
                semantic_tp=semantic_tp, within_group_wrong_class=tp-semantic_tp,
                IoU=100*tp/(gt+pred-tp) if gt+pred-tp else None,
                precision=tp/pred if pred else None, recall=tp/gt if gt else None)


def grouped(counts):
    c = np.asarray(counts, dtype=np.int64)
    return {group: {str(i+1): quality(c[i], ids) for i in range(3)}
            for group, ids in GROUPS.items()}


def edit_counts(old, new, gt):
    a, b, g = map(np.asarray, (old, new, gt))
    change = a != b; add = (a == 17) & (b != 17)
    return dict(changed=int(change.sum()), added=int(add.sum()),
        removed=int(((a != 17) & (b == 17)).sum()),
        relabeled=int(((a != 17) & (b != 17) & change).sum()),
        corrected=int((change & (b == g)).sum()), damaged=int((change & (a == g)).sum()),
        added_occ_tp=int((add & (g != 17)).sum()),
        added_semantic_tp=int((add & (b == g)).sum()))


def output_digest(outputs):
    h = hashlib.sha256()
    if not outputs: raise RuntimeError('missing motion outputs for paired geometry check')
    for key, value in sorted(outputs.items()):
        if not isinstance(value, torch.Tensor): continue
        tensor = value.detach().cpu().contiguous()
        # NumPy cannot represent BF16. Reinterpret bytes instead of converting
        # values to FP32: preserve dtype, signed zero and NaN payload bits too.
        # Flatten first so scalar tensors can also be viewed as uint8.
        raw = tensor.reshape(-1).view(torch.uint8).numpy()
        h.update(key.encode()); h.update(str((str(tensor.dtype), tuple(tensor.shape))).encode())
        h.update(raw.tobytes())
    return h.hexdigest()


class PlannerOriginAudit:
    """Reconstruct the exact original row-to-t0 mapping; never change a pose."""
    def __init__(self, source, path=None, *, expected_sha=PLAN_SHA):
        candidate = Path(path) if path else Path(source.manifest.get('come_trajectory_json', '__absent__'))
        self.source = source; self.rows = None; self.positions = {}
        self.metadata = dict(status='unavailable_original_json', path=str(candidate),
                             expected_sha256=expected_sha, cached_pose_origin_verified=False)
        if not candidate.is_file():
            if path: raise FileNotFoundError('explicit planner JSON is missing: '+str(candidate))
            return
        digest = file_sha256(candidate)
        if digest != expected_sha: raise RuntimeError('original planner JSON SHA256 mismatch')
        payload = json.loads(candidate.read_text(encoding='utf-8'))
        if file_sha256(candidate) != digest: raise RuntimeError('planner JSON changed while loading')
        trajs = payload.get('trajs')
        if not isinstance(trajs, dict): raise ValueError('original planner trajs mapping required')
        self.rows = {}
        scenes = {w.scene for w in source.windows}
        for scene in sorted(scenes):
            keys = [k for k in trajs if k.startswith(scene+'-')]
            suffixes = []
            for k in keys:
                suffix = k[len(scene)+1:]
                if not re.fullmatch(r'\d+', suffix): raise ValueError('invalid original planner scene key')
                suffixes.append(int(suffix))
            if len(set(suffixes)) != len(suffixes): raise ValueError('duplicate numeric planner scene index')
            ordered = [k for _, k in sorted(zip(suffixes, keys))]
            frames = sorted((f for f in source.catalog.values() if f.scene == scene), key=lambda f:f.timestamp)
            if len(ordered) != len(frames): raise RuntimeError('planner scene/frame population mismatch: '+scene)
            for i, (f, key) in enumerate(zip(frames, ordered)):
                self.positions[f.token] = i
                self.rows[f.token] = (key, trajs[key])
        self.metadata = dict(status='loaded_exact_original_json', path=str(candidate.resolve()),
            expected_sha256=expected_sha, sha256=digest, cached_pose_origin_verified=False,
            mapping='numeric scene suffix order -> chronological full-scene frame ordinal')

    def window(self, w):
        if self.rows is None: return dict(status='unverified_missing_original_json')
        key, value = self.rows[w.t0]
        rows = np.asarray(value, dtype=np.float64)
        if rows.shape != (7, 3) or not np.isfinite(rows).all():
            raise ValueError('original planner must have finite current + six XY/yaw rows')
        current = self.source.catalog[w.t0].pose
        reconstructed = []
        for x, y, yaw in rows[1:]:
            c, s = np.cos(yaw), np.sin(yaw)
            pose = current.copy(); pose[:3, :3] = current[:3, :3] @ np.array([[c,-s,0],[s,c,0],[0,0,1]])
            pose[:2, 3] = (x, y); reconstructed.append(pose)
        cached = np.asarray(self.source.plans[w.key])
        mismatch = float(np.max(np.abs(cached-np.asarray(reconstructed))))
        anchor_error = float(np.linalg.norm(rows[0, :2]-current[:2, 3]))
        return dict(status='verified' if mismatch <= 1e-6 and anchor_error <= .02 else 'mismatch',
            json_key=key, t0_ordinal=self.positions[w.t0], current_xy_error_m=anchor_error,
            cached_pose_max_abs_error=mismatch, row0_tolerance_m=.02,
            note='stationary/repeated XY cannot by itself establish frame identity; ordinal plus cache checked')


def restore(saved, contract, shape):
    state = deepcopy(saved); digest = state.pop('fingerprint', None)
    if digest != fingerprint(state) or state.get('contract_fingerprint') != fingerprint(contract):
        raise RuntimeError('branch diagnostic state/contract changed; cannot reuse other experiment prefix')
    n = state.get('completed_windows'); voxels = int(np.prod(shape))
    if type(n) is not int or not 0 <= n <= contract['windows']:
        raise ValueError('invalid branch diagnostic cursor')
    def check_counts(counts, size, expected):
        a = np.asarray(counts)
        if (a.shape != size or a.dtype.kind not in 'ui' or (a < 0).any()
                or not (a.sum(axis=(-2, -1)) == expected).all()):
            raise ValueError('invalid integer branch confusion counts')
    if set(state['counts']) != set(ROUTES) or set(state['t0_counts']) != {'occ', 'stc'}:
        raise ValueError('incomplete diagnostic routes')
    for c in state['counts'].values(): check_counts(c, (3,18,18), n*voxels)
    for c in state['t0_counts'].values(): check_counts(c, (18,18), n*voxels)
    if set(state['edits']) != set(SETTINGS): raise ValueError('missing edit counts')
    for rows in state['edits'].values():
        if len(rows) != 3: raise ValueError('invalid edit horizon count')
        for row in rows:
            if set(row) != set(EDIT_FIELDS) or any(type(v) is not int or not 0 <= v <= n*voxels for v in row.values()):
                raise ValueError('invalid integer edits')
            if (row['changed'] != row['added']+row['removed']+row['relabeled']
                or row['removed'] or row['relabeled'] or row['added_semantic_tp'] > row['added_occ_tp']
                or row['added_occ_tp'] > row['added'] or row['corrected']+row['damaged'] > row['changed']):
                raise ValueError('invalid ADD-only edit accounting')
    if len(state['audits']) != n or (n and set(state['verified_settings']) != set(SETTINGS)):
        raise ValueError('incomplete atomic diagnostic prefix')
    selected = contract.get('population', {}).get('selected_keys')
    if selected is not None:
        expected = [scene+'__'+token for scene,token in selected[:n]]
        if [a['key'] for a in state['audits']] != expected:
            raise ValueError('diagnostic prefix identity/order differs from frozen population')
    for v in (state['seconds'], *state['stage_seconds'].values()):
        if not np.isfinite(v) or v < 0: raise ValueError('invalid diagnostic timing')
    return state


def evaluate(source, windows, predictor, contract, *, origin_audit=None, saved=None,
             save=None, progress=None, stop_event=None, checkpoint_every=8):
    if len(windows) != contract['windows'] or checkpoint_every < 1:
        raise ValueError('diagnostic population mismatch')
    state = (dict(completed_windows=0, counts={k:WaymoMetrics().counts.tolist() for k in ROUTES},
        t0_counts={k:np.zeros((18,18),dtype=np.int64).tolist() for k in ('occ','stc')},
        edits={s:[dict.fromkeys(EDIT_FIELDS,0) for _ in range(3)] for s in SETTINGS},
        audits=[], verified_settings=[], seconds=0., stage_seconds={})
        if saved is None else restore(saved, contract, source.shape))
    meters = {k:WaymoMetrics(c,state['completed_windows']) for k,c in state['counts'].items()}
    t0 = {k:np.asarray(v,dtype=np.int64) for k,v in state['t0_counts'].items()}
    checked = set()
    def persist():
        state.update(counts={k:v.counts.tolist() for k,v in meters.items()},
            t0_counts={k:v.tolist() for k,v in t0.items()}, contract_fingerprint=fingerprint(contract))
        value = dict(state); value['fingerprint'] = fingerprint(value)
        if save: save(value)
        return value
    persist()
    try:
        for w in windows[state['completed_windows']:]:
            if stop_event is not None and stop_event.is_set(): break
            tick = time.perf_counter(); stage = {}; predictions = {}; motions = {}; histories = {}
            audit = dict(key=w.key, motion={}, history_observed_fraction={}, forecast_calls=4)
            planned = None
            for setting in SETTINGS:
                start = time.perf_counter(); rec, raw = source.prediction_inputs(w, setting)
                if raw.get('future_gt_occ') is not None:
                    raise RuntimeError('future occupancy cannot enter diagnostic forecasting')
                histories.setdefault(setting.split('_')[0], np.asarray(raw['history_occ'][-1]))
                if setting.endswith('pred'): planned = np.asarray(raw['future_poses'])
                prep, pred, _, detail = predictor.full(rec, raw, verify=setting not in checked)
                checked.add(setting)
                routes = dict(strong=prep.state['anchors'], transport=prep.baseline, joint=pred)
                for name, dense in routes.items():
                    if len(dense) != 6: raise RuntimeError('all six stage outputs required')
                    predictions[f'{setting}/{name}'] = tuple(semantics(x, source.shape) for x in dense)
                # A GT/Pred mismatch here is a genuine conditioning/interface discrepancy,
                # not the normal change of final grid caused by the planner.
                modality = setting.split('_')[0]; digest = output_digest(prep.outputs)
                if modality in motions and motions[modality] != digest:
                    raise RuntimeError('GT/Pred changed frozen history-only motion outputs: '+modality)
                motions[modality] = digest
                if setting.endswith('gt'):
                    audit['motion'][modality] = motion_jitter_audit(prep)
                    class_counts = defaultdict(int)
                    for component in prep.state['current']: class_counts[str(int(component['class_id']))] += 1
                    audit['motion'][modality]['source_class_counts'] = dict(class_counts)
                    audit['history_observed_fraction'][modality] = [float(np.mean(m)) for m in raw['history_observed']]
                stage[setting] = time.perf_counter()-start
                for k,v in detail.items():
                    if isinstance(v,(int,float)) and np.isfinite(v) and v >= 0:
                        stage[setting+'.'+k] = float(v)
                del prep
            if set(predictions) != set(ROUTES): raise RuntimeError('incomplete twelve-stage predictions')
            # Future labels/actual poses and T0 STC error comparison are accessed ONLY here.
            start = time.perf_counter(); targets = source.metric_targets(w)
            deltas = {k:WaymoMetrics() for k in ROUTES}
            for k in ROUTES: deltas[k].add(predictions[k], targets)
            true_t0 = source.frame('occ', w.scene, w.t0)[0]
            t0_delta = {k:confusion(v,true_t0,source.shape) for k,v in histories.items()}
            edits = {s:[edit_counts(predictions[s+'/transport'][hi], predictions[s+'/joint'][hi],gt)
                       for hi,gt in zip((1,3,5),targets)] for s in SETTINGS}
            if any(v['removed'] or v['relabeled'] for rows in edits.values() for v in rows):
                raise RuntimeError('frozen ADD-only / REMOVE-off contract violated')
            audit['pose_errors'] = pose_errors(planned,[source.catalog[t].pose for t in w.future])
            audit['planner_origin'] = origin_audit.window(w) if origin_audit else dict(status='unverified_missing_original_json')
            audit['paired_motion_identical'] = True
            stage['labels_metrics_and_audits'] = time.perf_counter()-start
            # Commit an entire window or nothing (including error statistics).
            for k in ROUTES:
                meters[k].counts += deltas[k].counts; meters[k].windows += 1
            for k,v in t0_delta.items(): t0[k] += v
            for s in SETTINGS:
                for row,delta in zip(state['edits'][s],edits[s]):
                    for k,v in delta.items(): row[k] += v
            state['audits'].append(audit); state['completed_windows'] += 1
            state['verified_settings'] = list(SETTINGS)
            elapsed = time.perf_counter()-tick; state['seconds'] += elapsed
            for k,v in stage.items(): state['stage_seconds'][k] = state['stage_seconds'].get(k,0.)+v
            if progress: progress(dict(window=state['completed_windows'],windows=len(windows),seconds=elapsed,stage_seconds=stage))
            if state['completed_windows'] % checkpoint_every == 0: persist()
    finally: persist()
    return dict(protocol=PROTOCOL,status='complete' if state['completed_windows']==len(windows) else 'stopped',
        completed_windows=state['completed_windows'],reports={k:v.report() for k,v in meters.items()},
        counts={k:v.counts.tolist() for k,v in meters.items()},
        groups={k:grouped(v.counts) for k,v in meters.items()},t0_counts={k:v.tolist() for k,v in t0.items()},
        t0_quality={k:dict(occupied=quality(v,range(17)),groups={g:quality(v,ids) for g,ids in GROUPS.items()},
            semantic_correct_occupied=int(np.diag(v)[:17].sum()),
            semantic_precision=float(np.diag(v)[:17].sum()/v[:,:17].sum()) if v[:,:17].sum() else None)
            for k,v in t0.items()},edits=state['edits'],audits=state['audits'],
        seconds=state['seconds'],stage_seconds=state['stage_seconds'])


def summary(result):
    n = result['completed_windows']
    lines = ['===== FROZEN STC / PLANNER BRANCH DIAGNOSTIC =====',
        f'status={result["status"]}; windows={n}; seconds={result["seconds"]:.2f}',
        'Mean5/6/8/12/14; FOUR histories -> SIX futures; ADD0.5 / REMOVEoff.',
        'No training, correction, GT alignment, visibility-mask changes, or automatic selection.',
        'ONE forecast per setting; Strong and Transport read from that SAME forecast.',
        'Full-grid standard metrics, three report horizons; dynamic group is NOT Moving support.',
        'setting    stage       avg mIoU    avg IoU    dmIoU_vs_Strong    dmIoU_vs_Transport']
    for s in SETTINGS:
        baseline = result['reports'][s+'/strong']['average']['standard_mIoU']
        transport = result['reports'][s+'/transport']['average']['standard_mIoU']
        for stage in STAGES:
            v = result['reports'][s+'/'+stage]['average']
            if v['standard_mIoU'] is None: continue
            lines.append(f'{s:10s} {stage:9s} {v["standard_mIoU"]:11.6f} {v["IoU"]:10.6f} '
                f'{v["standard_mIoU"]-baseline:+18.6f} {v["standard_mIoU"]-transport:+21.6f}')
        totals = {k:sum(row[k] for row in result['edits'][s]) for k in EDIT_FIELDS}
        totals['addition_semantic_precision'] = totals['added_semantic_tp']/totals['added'] if totals['added'] else None
        totals['addition_occ_precision'] = totals['added_occ_tp']/totals['added'] if totals['added'] else None
        lines.append('  CCR edits (sum 1/2/3s): '+json.dumps(totals))
    lines.append('===== GROUP ERROR: sums over 1/2/3s; TP counts group membership =====')
    lines.append('Road/sidewalk is a subset of ground; groups are NOT additive mIoU contributions.')
    for s in SETTINGS:
        for group in ('ground','building_vegetation','dynamic','other_static'):
            rows = {}
            for stage in STAGES:
                c = np.asarray(result['counts'][s+'/'+stage],dtype=np.int64).sum(0)
                q = quality(c,GROUPS[group])
                rows[stage] = {k:q[k] for k in ('fp','fn','within_group_wrong_class','IoU')}
            lines.append(s+' '+group+' '+json.dumps(rows))
    lines.append('===== T0 INPUT QUALITY (no metric mask) =====')
    for k,v in result['t0_quality'].items(): lines.append(k+' '+json.dumps(v))
    lines.append('===== POSE / TRACK AUDIT (not used to correct predictions) =====')
    for hi in (1,3,5):
        errors = {k:dict(median=float(np.median([a['pose_errors'][hi][k] for a in result['audits']])),
                        p90=float(np.percentile([a['pose_errors'][hi][k] for a in result['audits']],90)))
                  for k in ('xy_m','yaw_deg','z_m','tilt_deg')} if n else {}
        lines.append(f'Pred ego {(hi+1)*.5:.1f}s '+json.dumps(errors))
    for modality in ('occ','stc'):
        rows=[a['motion'][modality] for a in result['audits']]
        stats={k:sum(r.get(k,0) for r in rows) for k in ('sources','matched_t0_sources','tracks_with_3plus_frames')}
        stats['available_windows'] = sum(r.get('available',False) for r in rows)
        for k in ('speed_p90_mps','centroid_fit_rmse_p90_m'):
            values=[r[k] for r in rows if r.get(k) is not None]
            stats['median_of_window_'+k] = float(np.median(values)) if values else None
        lines.append(modality+' tracks '+json.dumps(stats))
    origins=defaultdict(int)
    for a in result['audits']: origins[a['planner_origin']['status']] += 1
    lines.append('original planner origin audit: '+json.dumps(dict(origins)))
    if any(a['planner_origin']['status']=='mismatch' for a in result['audits']):
        lines.append('WARNING: original planner/cache origin audit MISMATCH; no poses were corrected.')
    if any(a['planner_origin']['status']=='unverified_missing_original_json' for a in result['audits']):
        lines.append('Original planner JSON missing: cache metadata declaration is NOT independent origin verification.')
    lines.append('GT/Pred history-only motion byte-identical windows: '+str(sum(a['paired_motion_identical'] for a in result['audits'])))
    lines.append('===== LARGEST PER-CLASS JOINT DROPS (average of available 1/2/3s class IoUs) =====')
    def class_average(route, name):
        values=[h['per_class_IoU'][name] for h in result['reports'][route]['horizons'].values()]
        valid=[v for v in values if v is not None]
        return float(np.mean(valid)) if valid else None
    for setting,reference in (('occ_pred','occ_gt'),('stc_gt','occ_gt'),('stc_pred','stc_gt')):
        rows=[]
        for name in CLASS_NAMES[:17]:
            ref=class_average(reference+'/joint',name); val=class_average(setting+'/joint',name)
            if ref is not None and val is not None:
                rows.append(dict(class_name=name,reference_IoU=ref,setting_IoU=val,delta_pp=val-ref,
                    strong_IoU=class_average(setting+'/strong',name),
                    transport_IoU=class_average(setting+'/transport',name)))
        lines.append(setting+' vs '+reference+' '+json.dumps(sorted(rows,key=lambda r:r['delta_pp'])[:6]))
    lines.append('Per-class 1/2/3s IoUs + integer confusions + individual audits: evaluation.json')
    lines.append('seconds/window: '+str(result['seconds']/max(n,1)))
    lines.append('Diagnostic only. Old checkpoints, cached inputs and full4219 results unchanged.')
    return '\n'.join(lines)+'\n'
