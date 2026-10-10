#!/usr/bin/env python3
"""One small frozen-weight comparison; no GT alignment or automatic promotion."""
import sys
from pathlib import Path
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from contextlib import ExitStack
import json
import math
import os
import signal
import threading
import time

import numpy as np
import torch

from real_motion.stc_causal_geometry import (PROTOCOL, RULES, stabilize_stc_ground,
    compensate_planned_ground_pose, motion_jitter_audit)
from real_motion.waymo_i2world import WaymoMetrics, fingerprint, file_sha256
from tools.real_motion import eval_p0_f9_joint_surface_stc as base
from tools.real_motion.stc_shared_execution import Predictor
from tools.real_motion.eval_p0_f9_joint_surface_stc_shared import EXTRA_FILES
from tools.real_motion.waymo_zero_shot_common import write_json

# A fixed design, not a hyperparameter sweep. Unaffected predictions are reused.
ROUTES = dict(b_occ_gt=('occ_gt',False,False), b_occ_pred=('occ_pred',False,False),
    pose_occ_pred=('occ_pred',False,True),
    b_stc_gt=('stc_gt',False,False), b_stc_pred=('stc_pred',False,False), pose_stc_pred=('stc_pred',False,True),
    temporal_stc_gt=('stc_gt',True,False), temporal_stc_pred=('stc_pred',True,False),
    combined_stc_pred=('stc_pred',True,True))


def pose_errors(predicted, actual):
    rows = []
    for p,g in zip(predicted,actual):
        p,g = np.asarray(p),np.asarray(g)
        yaw_p,yaw_g = math.atan2(p[1,0],p[0,0]),math.atan2(g[1,0],g[0,0])
        dy = math.atan2(math.sin(yaw_p-yaw_g),math.cos(yaw_p-yaw_g))
        rows.append(dict(xy_m=float(np.linalg.norm(p[:2,3]-g[:2,3])), yaw_deg=abs(math.degrees(dy)),
            z_m=abs(float(p[2,3]-g[2,3])),
            tilt_deg=math.degrees(math.acos(float(np.clip(p[:3,2]@g[:3,2],-1,1))))))
    return rows


def restore(saved, contract, shape):
    value = dict(saved); digest = value.pop('fingerprint',None)
    if digest != fingerprint(value) or value.get('contract_fingerprint') != fingerprint(contract):
        raise RuntimeError('causal screen state/contract changed; no old prefix migration')
    n = value.get('completed_windows'); voxels = int(np.prod(shape))
    if type(n) is not int or not 0 <= n <= contract['windows'] or set(value['counts']) != set(ROUTES):
        raise ValueError('invalid atomic screen population/cursor')
    for counts in value['counts'].values():
        a = np.asarray(counts)
        if a.shape != (3,18,18) or a.dtype.kind not in 'ui' or (a < 0).any() or not (a.sum((1,2)) == n*voxels).all():
            raise ValueError('invalid screen integer counts')
    if len(value['audits']) != n or (n and set(value['verified_routes']) != set(ROUTES)):
        raise ValueError('incomplete screen audit/exactness prefix')
    t0 = np.asarray(value['t0_counts'])
    if t0.shape != (18,18) or t0.dtype.kind not in 'ui' or (t0 < 0).any() or t0.sum() != n*voxels:
        raise ValueError('invalid STC t0 counts')
    if not np.isfinite(value['seconds']) or value['seconds'] < 0: raise ValueError('invalid screen timing')
    return value


def evaluate(source, windows, predictor, grid, contract, *, saved=None, save=None,
             progress=None, stop_event=None, checkpoint_every=8):
    if len(windows) != contract['windows'] or checkpoint_every < 1: raise ValueError('screen contract population mismatch')
    state = (dict(completed_windows=0, counts={k:WaymoMetrics().counts.tolist() for k in ROUTES},
        t0_counts=np.zeros((18,18),np.int64).tolist(), audits=[], verified_routes=[], seconds=0.)
        if saved is None else restore(saved,contract,source.shape))
    meters = {k:WaymoMetrics(v,state['completed_windows']) for k,v in state['counts'].items()}
    checked = set()
    def persist():
        state.update(counts={k:v.counts.tolist() for k,v in meters.items()}, contract_fingerprint=fingerprint(contract))
        value = dict(state); value['fingerprint'] = fingerprint(value)
        if save: save(value)
        return value
    persist()
    try:
        for w in windows[state['completed_windows']:]:
            if stop_event is not None and stop_event.is_set(): break
            tick = time.perf_counter(); dense = {}; audits = {}; inputs = {}
            # Prediction-side code cannot access metric targets/actual future poses.
            for setting in base.SETTINGS:
                inputs[setting] = source.prediction_inputs(w,setting)
            # Use STC history only for STC compensation; never borrow OCC geometry.
            temporals = {}; compensated = {}
            temporals['stc_gt'],audits['temporal'] = stabilize_stc_ground(inputs['stc_gt'][1],grid)
            # The two STC histories are identical; only their future ego route differs.
            if not np.array_equal(inputs['stc_gt'][1]['history_occ'], inputs['stc_pred'][1]['history_occ']):
                raise RuntimeError('STC GT/Pred historical populations differ')
            temporals['stc_pred'] = {**inputs['stc_pred'][1],
                                    'history_occ':temporals['stc_gt']['history_occ']}
            for setting in ('occ_pred','stc_pred'):
                raw = inputs[setting][1]
                compensated[setting],a = compensate_planned_ground_pose(raw,raw['future_poses'],grid)
                audits['pose_'+setting] = a
            for route,(setting,temporal,pose) in ROUTES.items():
                rec,original = inputs[setting]
                raw = dict(temporals[setting] if temporal else original)
                # Fresh dictionaries; provider memo fields never enter another route.
                raw = {k:v for k,v in raw.items() if not k.startswith('_')}
                if pose: raw['future_poses'] = compensated[setting]
                # combined uses the ORIGINAL modality's causal road plane, so the
                # factorial comparison differs only by the two explicit switches.
                prep,pred,_,stages = predictor.full(rec,raw,verify=route not in checked)
                if len(pred) != 6 or any(np.asarray(x).shape != source.shape for x in pred):
                    raise RuntimeError('all nine routes must finish SIX outputs before GT access')
                checked.add(route); dense[route] = pred
                if route in ('b_occ_gt','b_stc_gt'):
                    audits['motion_'+route] = motion_jitter_audit(prep)
            # ONLY scoring/auditing below this boundary may inspect future truth.
            targets = source.metric_targets(w)
            actual = [source.catalog[t].pose for t in w.future]
            for setting in ('occ_pred','stc_pred'):
                audits['error_'+setting] = pose_errors(inputs[setting][1]['future_poses'],actual)
                audits['error_compensated_'+setting] = pose_errors(compensated[setting],actual)
            truth,_ = source.frame('occ',w.scene,w.t0)
            input_stc = inputs['stc_gt'][1]['history_occ'][-1]
            t0 = np.bincount(truth.astype(np.int64).ravel()*18+input_stc.ravel(),minlength=18**2).reshape(18,18)
            deltas = {k:WaymoMetrics() for k in ROUTES}
            for k in ROUTES: deltas[k].add(dense[k],targets)
            # Commit only a complete nine-route window, never a half comparison.
            for k in ROUTES:
                meters[k].counts += deltas[k].counts; meters[k].windows += 1
            state['t0_counts'] = (np.asarray(state['t0_counts'])+t0).tolist()
            state['audits'].append(dict(key=w.key,**audits)); state['completed_windows'] += 1
            state['verified_routes'] = list(ROUTES); elapsed = time.perf_counter()-tick; state['seconds'] += elapsed
            if progress: progress(dict(window=state['completed_windows'],windows=len(windows),seconds=elapsed,audit=audits))
            if state['completed_windows'] % checkpoint_every == 0: persist()
    finally: persist()
    return dict(status='complete' if state['completed_windows'] == len(windows) else 'stopped',
        completed_windows=state['completed_windows'], reports={k:v.report() for k,v in meters.items()},
        t0_counts=state['t0_counts'], audits=state['audits'], seconds=state['seconds'])


def summary(result):
    lines = ['===== FROZEN CAUSAL EGO / STC GEOMETRY SCREEN =====',
        f'status={result["status"]}; windows={result["completed_windows"]}',
        f'seconds={result["seconds"]:.2f}; seconds/window={result["seconds"]/max(1,result["completed_windows"]):.3f}',
        'Same mean5/6/8/12/14, network/weights unchanged; ADD0.5 / REMOVEoff.',
        'No training, future GT alignment, metric masks or automatic selection.',
        'route                       avg mIoU      avg IoU     dmIoU       dIoU']
    for route,(setting,_,_) in ROUTES.items():
        v = result['reports'][route]['average']; b = result['reports']['b_'+setting]['average']
        if v['standard_mIoU'] is None: continue
        lines.append(f'{route:27s} {v["standard_mIoU"]:10.6f} {v["IoU"]:11.6f} '
            f'{v["standard_mIoU"]-b["standard_mIoU"]:+10.6f} {v["IoU"]-b["IoU"]:+10.6f}')
        scores = result['reports'][route]['horizons']
        lines.append('  1/2/3s: '+json.dumps({h:dict(mIoU=x['standard_mIoU'],IoU=x['IoU'])
                                             for h,x in scores.items()}))
    lines.append('===== POSE ERROR: audit only, never used by adapters =====')
    for name in ('error_occ_pred','error_compensated_occ_pred','error_compensated_stc_pred'):
        rows = [a[name] for a in result['audits']]
        for h in (1,3,5):
            values = {k:dict(median=float(np.median([r[h][k] for r in rows])),
                            p90=float(np.percentile([r[h][k] for r in rows],90)))
                      for k in ('xy_m','yaw_deg','z_m','tilt_deg')} if rows else {}
            lines.append(f'{name} {(h+1)*.5:.1f}s '+json.dumps(values))
    counts = np.asarray(result['t0_counts']); gt_occ=counts[:17,:].sum(); pred_occ=counts[:,:17].sum()
    tp=counts[:17,:17].sum(); semantic=np.trace(counts[:17,:17])
    lines.append('STC t0 quality (no mask): '+json.dumps(dict(occupied_gt=int(gt_occ),occupied_pred=int(pred_occ),
        false_occupied=int(counts[17,:17].sum()),missed_occupied=int(counts[:17,17].sum()),
        occupied_wrong_class=int(tp-semantic),semantic_precision=float(semantic/pred_occ) if pred_occ else None)))
    lines.append('===== CAUSAL ADAPTER / DYNAMIC JITTER AUDITS =====')
    for name in ('temporal','pose_occ_pred','pose_stc_pred'):
        rows=[a[name] for a in result['audits']]
        lines.append(name+' totals: '+json.dumps({k:sum(r[k] for r in rows)
                                                  for k in rows[0]} if rows else {}))
    for name in ('motion_b_occ_gt','motion_b_stc_gt'):
        rows=[a[name] for a in result['audits'] if a[name].get('available')]
        v=dict(available_windows=len(rows),sources=sum(r['sources'] for r in rows),
               matched_t0_sources=sum(r['matched_t0_sources'] for r in rows),
               tracks_with_3plus_frames=sum(r['tracks_with_3plus_frames'] for r in rows))
        for key in ('speed_p90_mps','centroid_fit_rmse_p90_m'):
            values=[r[key] for r in rows if r[key] is not None]
            v['median_window_'+key]=float(np.median(values)) if values else None
        v['velocities_modified']=False
        lines.append(name+': '+json.dumps(v))
    lines += ['Full raw counts, per-window audits and rules: evaluation.json / contract.json.',
              'Pred variants are new causal inference protocols, NOT unchanged official planner reproduction.',
              'This screen does not select a method/checkpoint. Keep old evaluation resumable.']
    return '\n'.join(lines)+'\n'


def main(stop_event=None,argv=None):
    parser = base.parser(); parser.description = __doc__; a=parser.parse_args(argv)
    if a.population != 'dev64' or a.audit_only or a.execution != 'native_parallel':
        parser.error('this entry ONLY runs the fixed dev64 screen; no full evaluation or input-only mode')
    if not 1 <= a.cpu_workers <= 8 or not 0 <= a.frame_cache_mib <= 4096 or a.checkpoint_every < 1:
        parser.error('invalid bounded CPU/cache/checkpoint arguments')
    out = Path(a.out_dir).resolve(); previous=None
    roots=[Path(p).resolve() for p in (a.dataroot,a.stc_root,a.plan_cache)]
    if any(out.is_relative_to(p) for p in roots) or any((p/'training.json').is_file() for p in (out,*out.parents)):
        parser.error('distinct output outside data/cache/training required')
    if a.resume:
        previous=json.loads((out/'contract.json').read_text(encoding='utf-8'))
    elif out.exists(): parser.error('new output required; original results cannot be overwritten')
    if not a.population_manifest: parser.error('frozen dev64 parent manifest required')
    source=base.STCFourSettingSource.from_files(a.dataroot,a.stc_root,a.plan_cache,cache_mib=a.frame_cache_mib)
    manifest,_,_=base.load_manifest(a.population_manifest)
    windows,population=source.select('dev64',manifest['parent_keys']); inventory=source.preflight(windows)
    if not torch.cuda.is_available(): parser.error('actual CUDA required for dataset screen')
    cfg=base.load_runtime_config(a.config); pcfg=base.make_prepare_config(cfg)
    if (tuple(pcfg.grid.shape_hwd)!=base.SHAPE or pcfg.future_frames!=6 or pcfg.frame_dt_s!=.5
            or pcfg.free_label!=17 or not np.allclose(pcfg.grid.voxel_size,(.4,)*3,rtol=0,atol=1e-12)
            or not np.allclose((pcfg.grid.x_min,pcfg.grid.y_min,pcfg.grid.z_min),(-40,-40,-1),rtol=0,atol=1e-12)):
        parser.error('unchanged trained Occ3D geometry required')
    if a.checkpoint: checkpoint=Path(a.checkpoint).resolve()
    elif previous: checkpoint=Path(previous['checkpoint']).resolve()
    else:
        if not a.runs_root or not a.run_dir: parser.error('checkpoint or frozen mean discovery required')
        bundle=base.find_frozen_bundle(a.runs_root,a.run_dir,a.source_bundle_dir)
        checkpoint=Path(bundle['candidates'][base.AVERAGE_NAME]['path']).resolve()
    if out.is_relative_to(checkpoint.parent): parser.error('output cannot be inside source mean directory')
    digest=file_sha256(checkpoint); root=Path(__file__).resolve().parents[2]
    files=(*base.IMPLEMENTATION_FILES,*EXTRA_FILES,'real_motion/stc_causal_geometry.py',
        'tools/real_motion/eval_p0_f9_stc_causal_geometry.py','tools/real_motion/run_p0_f9_stc_causal_geometry.sh')
    contract=dict(protocol=PROTOCOL,rules=RULES,routes=ROUTES,windows=len(windows),population=population,
        inventory=inventory,source=source.metadata,checkpoint=str(checkpoint),checkpoint_sha256=digest,
        config_sha256=file_sha256(a.config),population_manifest_sha256=file_sha256(a.population_manifest),
        implementation={f:file_sha256(root/f) for f in dict.fromkeys(files)},cpu_workers=a.cpu_workers,
        frame_cache_mib=a.frame_cache_mib,graphs=not a.no_graphs,parallel_majority=a.parallel_majority,
        runtime_environment={k:v for k,v in sorted(os.environ.items()) if k.startswith('SWFM_')},
        torch_version=str(torch.__version__))
    saved=None
    if previous and fingerprint(previous)!=fingerprint(contract): raise RuntimeError('causal screen resume contract changed')
    if not a.resume: out.mkdir(parents=True)
    with ExitStack() as stack:
        stack.enter_context(base.evaluation_lock(out))
        if a.resume:
            saved=json.loads((out/'state.json').read_text(encoding='utf-8')); restore(saved,contract,source.shape)
        else: write_json(out/'contract.json',contract)
        saved_model,joint=base.load_evaluation_model(checkpoint,device='cuda',z_bins=16)
        if (saved_model.get('source_epochs')!=list(base.AVERAGE_EPOCHS) or not saved_model.get('averaging')
                or joint.transport.config.history_frames!=4 or file_sha256(checkpoint)!=digest):
            raise RuntimeError('unchanged frozen four-history mean required')
        torch.set_num_threads(1)
        predictor=Predictor(joint,pcfg,'cuda',workers=a.cpu_workers,graphs=not a.no_graphs,
                            parallel_majority=a.parallel_majority,geometry_mib=512)
        stack.callback(predictor.close)
        with (out/'progress.jsonl').open('a',encoding='utf-8') as handle:
            def progress(row):
                handle.write(json.dumps(row,allow_nan=False)+'\n'); handle.flush()
                if row['window']%8==0 or row['window']==1: print(f'CAUSAL_SCREEN {row["window"]}/{row["windows"]} seconds={row["seconds"]:.3f}',flush=True)
            result=evaluate(source,windows,predictor,pcfg.grid,contract,saved=saved,
                save=lambda s:write_json(out/'state.json',s),progress=progress,stop_event=stop_event,
                checkpoint_every=a.checkpoint_every)
        if file_sha256(checkpoint)!=digest: raise RuntimeError('source mean changed during screen')
        result['contract']=contract; write_json(out/'evaluation.json',result)
        report=summary(result); (out/'summary.txt').write_text(report,encoding='utf-8')
        print(report,flush=True); print('RESULT: '+str(out/'summary.txt'),flush=True)
    return 0


if __name__=='__main__':
    stopped=threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM): signal.signal(sig,lambda *_:stopped.set())
    sys.exit(main(stopped))
