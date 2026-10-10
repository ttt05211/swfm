#!/usr/bin/env python3
"""Fixed dev64 Occ/STC Pred yaw comparison; no training or automatic selection."""
import sys
from pathlib import Path
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from contextlib import ExitStack
import json
import os
import signal
import threading
import time
import numpy as np
import torch

from real_motion.stc_trajectory_yaw import PROTOCOL, RULES, tangent_yaw, yaw_error_deg
from real_motion.waymo_i2world import WaymoMetrics, fingerprint, file_sha256
from tools.real_motion import eval_p0_f9_joint_surface_stc as base
from tools.real_motion.stc_shared_execution import Predictor
from tools.real_motion.eval_p0_f9_joint_surface_stc_shared import EXTRA_FILES
from tools.real_motion.waymo_zero_shot_common import write_json

ROUTES = dict(b_occ_pred=('occ_pred', False), yaw_occ_pred=('occ_pred', True),
              b_stc_pred=('stc_pred', False), yaw_stc_pred=('stc_pred', True))


def restore(saved, contract, shape):
    state = dict(saved); digest = state.pop('fingerprint', None)
    if digest != fingerprint(state) or state.get('contract_fingerprint') != fingerprint(contract):
        raise RuntimeError('yaw screen state/contract changed; no other experiment prefix migration')
    n = state.get('completed_windows')
    if type(n) is not int or not 0 <= n <= contract['windows'] or set(state['counts']) != set(ROUTES):
        raise ValueError('invalid yaw population/cursor')
    for counts in state['counts'].values():
        a = np.asarray(counts)
        if (a.shape != (3, 18, 18) or a.dtype.kind not in 'ui' or (a < 0).any()
                or not (a.sum((1, 2)) == n*int(np.prod(shape))).all()):
            raise ValueError('invalid integer yaw counts')
    if len(state['audits']) != n or (n and set(state['verified_routes']) != set(ROUTES)):
        raise ValueError('incomplete atomic yaw prefix')
    if (not np.isfinite(state['seconds']) or state['seconds'] < 0
            or any(not np.isfinite(v) or v < 0 for v in state['stage_seconds'].values())):
        raise ValueError('invalid yaw timing')
    return state


def evaluate(source, windows, predictor, contract, *, saved=None, save=None,
             progress=None, stop_event=None, checkpoint_every=8):
    if len(windows) != contract['windows'] or checkpoint_every < 1:
        raise ValueError('yaw population mismatch')
    state = (dict(completed_windows=0, counts={k:WaymoMetrics().counts.tolist() for k in ROUTES},
        audits=[], verified_routes=[], seconds=0., stage_seconds={})
        if saved is None else restore(saved, contract, source.shape))
    meters = {k:WaymoMetrics(v, state['completed_windows']) for k,v in state['counts'].items()}
    checked = set()
    def persist():
        state.update(counts={k:v.counts.tolist() for k,v in meters.items()},
                     contract_fingerprint=fingerprint(contract))
        value = dict(state); value['fingerprint'] = fingerprint(value)
        if save: save(value)
        return value
    persist()
    try:
        for w in windows[state['completed_windows']:]:
            if stop_event is not None and stop_event.is_set(): break
            tick = time.perf_counter(); stage = {}; dense = {}; inputs = {}
            start = time.perf_counter()
            # Only four historical measured poses/timestamps enter the adapter.
            times = [(source.catalog[t].timestamp-source.catalog[w.t0].timestamp)*1e-6 for t in w.history]
            for setting in ('occ_pred', 'stc_pred'):
                inputs[setting] = source.prediction_inputs(w, setting)
            raw0 = inputs['occ_pred'][1]; raw1 = inputs['stc_pred'][1]
            if raw0.get('future_gt_occ') is not None or raw1.get('future_gt_occ') is not None:
                raise RuntimeError('future occupancy cannot enter yaw forecasting')
            if (not np.array_equal(np.asarray(raw0['history_poses']), np.asarray(raw1['history_poses']))
                    or not np.array_equal(np.asarray(raw0['future_poses']), np.asarray(raw1['future_poses']))):
                raise RuntimeError('paired modalities must share exactly the same measured/planned ego poses')
            corrected, audit = tangent_yaw(raw0['history_poses'], times, raw0['future_poses'])
            unchanged = np.array_equal(corrected, np.asarray(raw0['future_poses']))
            stage['inputs_and_yaw_adapter'] = time.perf_counter()-start
            calls = 0; reused = 0
            for route, (setting, change) in ROUTES.items():
                start = time.perf_counter()
                if change and unchanged:
                    dense[route] = dense['b_'+setting]
                    reused += 1
                else:
                    rec, original = inputs[setting]
                    if original.get('future_gt_occ') is not None:
                        raise RuntimeError('future occupancy cannot enter yaw forecasting')
                    raw = {k:v for k,v in original.items() if not k.startswith('_')}
                    if change: raw['future_poses'] = corrected
                    _, pred, _, detail = predictor.full(rec, raw, verify=route not in checked)
                    if len(pred) != 6 or any(np.asarray(x).shape != source.shape for x in pred):
                        raise RuntimeError('six complete predictions required before future truth access')
                    dense[route] = pred; calls += 1
                    # An unchanged reuse is exact, but must not suppress the first
                    # real changed-yaw canonical check in a later window.
                    checked.add(route)
                    for k,v in detail.items():
                        if isinstance(v, (float,int)) and np.isfinite(v) and v >= 0:
                            stage[route+'/detail/'+k] = float(v)
                stage[route+'/total'] = time.perf_counter()-start
            # No future GT pose, occupancy, visibility or annotations above this boundary.
            if set(dense) != set(ROUTES): raise RuntimeError('incomplete four-route comparison')
            start = time.perf_counter()
            targets = source.metric_targets(w)
            actual = [source.catalog[t].pose for t in w.future]
            errors = dict(original=yaw_error_deg(raw0['future_poses'], actual),
                          corrected=yaw_error_deg(corrected, actual))
            deltas = {k:WaymoMetrics() for k in ROUTES}
            for k in ROUTES: deltas[k].add(dense[k], targets)
            stage['metric_and_pose_audit'] = time.perf_counter()-start
            # Publish counts/audits/timing only at a complete window boundary.
            for k in ROUTES:
                meters[k].counts += deltas[k].counts; meters[k].windows += 1
            state['audits'].append(dict(key=w.key, yaw=audit, yaw_errors_deg=errors,
                                       forecast_calls=calls, unchanged_reuses=reused))
            state['completed_windows'] += 1; state['verified_routes'] = list(ROUTES)
            elapsed = time.perf_counter()-tick; state['seconds'] += elapsed
            for k,v in stage.items(): state['stage_seconds'][k] = state['stage_seconds'].get(k,0.)+v
            if progress: progress(dict(window=state['completed_windows'], windows=len(windows),
                seconds=elapsed, stage_seconds=stage, corrected_horizons=audit['corrected'],
                forecast_calls=calls, unchanged_reuses=reused))
            if state['completed_windows'] % checkpoint_every == 0: persist()
    finally: persist()
    return dict(status='complete' if state['completed_windows']==len(windows) else 'stopped',
        completed_windows=state['completed_windows'], reports={k:v.report() for k,v in meters.items()},
        audits=state['audits'], seconds=state['seconds'], stage_seconds=state['stage_seconds'])


def summary(result):
    n = result['completed_windows']
    lines = ['===== FROZEN PLANNER TANGENT YAW / DEV64 =====',
        f'status={result["status"]}; windows={n}; seconds={result["seconds"]:.2f}',
        'Same frozen mean5/6/8/12/14, strict4->6, ADD0.5/REMOVEoff; no training.',
        'Planner XY/z and yaw-free roll/pitch unchanged. No GT alignment or future GT input.',
        'Fixed causal rule, not official unchanged planner reproduction; no automatic full run.',
        'route              avg mIoU     avg IoU     dmIoU       dIoU']
    for route,(setting,_) in ROUTES.items():
        v=result['reports'][route]['average']; b=result['reports']['b_'+setting]['average']
        if v['standard_mIoU'] is None: continue
        lines.append(f'{route:18s} {v["standard_mIoU"]:10.6f} {v["IoU"]:11.6f} '
            f'{v["standard_mIoU"]-b["standard_mIoU"]:+10.6f} {v["IoU"]-b["IoU"]:+10.6f}')
        lines.append('  1/2/3s: '+json.dumps({h:dict(mIoU=v['standard_mIoU'],IoU=v['IoU'])
            for h,v in result['reports'][route]['horizons'].items()}))
    corrected = sum(a['yaw']['corrected'] for a in result['audits'])
    reasons = {}
    for a in result['audits']:
        for k,v in a['yaw']['fallback_reasons'].items(): reasons[k]=reasons.get(k,0)+v
    lines.append('corrected_horizons: '+str(corrected)+' / '+str(n*6))
    lines.append('fallback_horizons: '+json.dumps(reasons))
    for i in (1,3,5):
        for subset in ('all','changed_only'):
            rows=[a for a in result['audits'] if subset=='all' or a['yaw']['correction_deg'][i]!=0]
            stats={key:dict(median=float(np.median([a['yaw_errors_deg'][key][i] for a in rows])),
                p90=float(np.percentile([a['yaw_errors_deg'][key][i] for a in rows],90)))
                for key in ('original','corrected')} if rows else {}
            lines.append(f'yaw_error_deg {(i+1)*.5:.1f}s {subset} n={len(rows)} '+json.dumps(stats))
    lines.append('forecast_calls: '+str(sum(a['forecast_calls'] for a in result['audits'])))
    lines.append('unchanged_candidate_reuses: '+str(sum(a['unchanged_reuses'] for a in result['audits'])))
    lines.append('seconds/window: '+str(result['seconds']/max(1,n)))
    lines.append('stage_seconds (detail overlaps route totals): '+json.dumps(result['stage_seconds']))
    lines += ['Original four-setting evaluator/checkpoint/data remain unchanged.',
              'If no convincing gain, stop this frozen heuristic route; no automatic retry/selection.']
    return '\n'.join(lines)+'\n'


def main(stop_event=None, argv=None):
    parser=base.parser(); parser.description=__doc__; a=parser.parse_args(argv)
    if a.population!='dev64' or a.audit_only or a.execution!='native_parallel':
        parser.error('fixed dev64 four-route yaw screen only; no full/input-only mode')
    if not 1<=a.cpu_workers<=8 or not 0<=a.frame_cache_mib<=4096 or a.checkpoint_every<1:
        parser.error('invalid bounded execution settings')
    out=Path(a.out_dir).resolve(); previous=None
    roots=[Path(p).resolve() for p in (a.dataroot,a.stc_root,a.plan_cache)]
    if any(out.is_relative_to(p) for p in roots) or any((p/'training.json').is_file() for p in (out,*out.parents)):
        parser.error('distinct output outside input/cache/training required')
    if a.resume: previous=json.loads((out/'contract.json').read_text(encoding='utf-8'))
    elif out.exists(): parser.error('new output required; cannot overwrite old results')
    if not a.population_manifest: parser.error('frozen dev64 parent manifest required')
    source=base.STCFourSettingSource.from_files(a.dataroot,a.stc_root,a.plan_cache,cache_mib=a.frame_cache_mib)
    manifest,_,_=base.load_manifest(a.population_manifest)
    windows,population=source.select('dev64',manifest['parent_keys']); inventory=source.preflight(windows)
    if not torch.cuda.is_available(): parser.error('actual CUDA required for dataset evaluation')
    pcfg=base.make_prepare_config(base.load_runtime_config(a.config))
    if (tuple(pcfg.grid.shape_hwd)!=base.SHAPE or pcfg.future_frames!=6 or pcfg.frame_dt_s!=.5
        or pcfg.free_label!=17 or not np.allclose(pcfg.grid.voxel_size,(.4,)*3,rtol=0,atol=1e-12)
        or not np.allclose((pcfg.grid.x_min,pcfg.grid.y_min,pcfg.grid.z_min),(-40,-40,-1),rtol=0,atol=1e-12)):
        parser.error('unchanged trained Occ3D geometry required')
    if a.checkpoint: checkpoint=Path(a.checkpoint).resolve()
    elif previous: checkpoint=Path(previous['checkpoint']).resolve()
    else:
        if not a.runs_root or not a.run_dir: parser.error('checkpoint or fixed mean discovery required')
        bundle=base.find_frozen_bundle(a.runs_root,a.run_dir,a.source_bundle_dir)
        checkpoint=Path(bundle['candidates'][base.AVERAGE_NAME]['path']).resolve()
    if out.is_relative_to(checkpoint.parent): parser.error('output cannot be inside source mean directory')
    digest=file_sha256(checkpoint); root=Path(__file__).resolve().parents[2]
    files=(*base.IMPLEMENTATION_FILES,*EXTRA_FILES,'real_motion/stc_trajectory_yaw.py',
        'tools/real_motion/eval_p0_f9_stc_trajectory_yaw.py','tools/real_motion/run_p0_f9_stc_trajectory_yaw.sh')
    contract=dict(protocol=PROTOCOL,rules=RULES,routes=ROUTES,windows=len(windows),population=population,
        inventory=inventory,source=source.metadata,checkpoint=str(checkpoint),checkpoint_sha256=digest,
        config_sha256=file_sha256(a.config),population_manifest_sha256=file_sha256(a.population_manifest),
        implementation={f:file_sha256(root/f) for f in dict.fromkeys(files)},cpu_workers=a.cpu_workers,
        frame_cache_mib=a.frame_cache_mib,graphs=not a.no_graphs,parallel_majority=a.parallel_majority,
        runtime_environment={k:v for k,v in sorted(os.environ.items()) if k.startswith('SWFM_')},
        torch_version=str(torch.__version__))
    if previous and fingerprint(previous)!=fingerprint(contract): raise RuntimeError('yaw screen resume contract changed')
    if not a.resume: out.mkdir(parents=True)
    with ExitStack() as stack:
        stack.enter_context(base.evaluation_lock(out))
        saved=json.loads((out/'state.json').read_text(encoding='utf-8')) if a.resume else None
        if saved is not None: restore(saved,contract,source.shape)
        else: write_json(out/'contract.json',contract)
        meta,joint=base.load_evaluation_model(checkpoint,device='cuda',z_bins=16)
        if (meta.get('source_epochs')!=list(base.AVERAGE_EPOCHS) or not meta.get('averaging')
            or joint.transport.config.history_frames!=4 or file_sha256(checkpoint)!=digest):
            raise RuntimeError('unchanged frozen four-history mean required')
        torch.set_num_threads(1)
        predictor=Predictor(joint,pcfg,'cuda',workers=a.cpu_workers,graphs=not a.no_graphs,
                            parallel_majority=a.parallel_majority,geometry_mib=512)
        stack.callback(predictor.close)
        with (out/'progress.jsonl').open('a',encoding='utf-8') as handle:
            def progress(row):
                handle.write(json.dumps(row,allow_nan=False)+'\n'); handle.flush()
                if row['window']%8==0 or row['window']==1:
                    print(f'YAW_SCREEN {row["window"]}/{row["windows"]} seconds={row["seconds"]:.3f} corrected={row["corrected_horizons"]}/6 calls={row["forecast_calls"]}',flush=True)
            result=evaluate(source,windows,predictor,contract,saved=saved,
                save=lambda s:write_json(out/'state.json',s),progress=progress,
                stop_event=stop_event,checkpoint_every=a.checkpoint_every)
        if file_sha256(checkpoint)!=digest: raise RuntimeError('source mean changed during screen')
        result['contract']=contract; write_json(out/'evaluation.json',result)
        report=summary(result); (out/'summary.txt').write_text(report,encoding='utf-8')
        print(report,flush=True); print('RESULT: '+str(out/'summary.txt'),flush=True)
    return 0


if __name__=='__main__':
    stopped=threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM): signal.signal(sig,lambda *_:stopped.set())
    sys.exit(main(stopped))
