#!/usr/bin/env python3
"""Read-only 1--6s open-loop rollout of the already frozen Surface CCR mean."""
import sys
from pathlib import Path
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
from collections import defaultdict
import hashlib
import json
import signal
import threading
import time

import numpy as np
import torch

from real_motion.causal_geometry_cache import CausalGeometryCache, PROTOCOL as CACHE_PROTOCOL
from real_motion.ccr_val_history_cache import namespace as val_namespace, validate_manifest
from real_motion.column_runtime_pipeline import CachedColumnSource, prefetch_raw_columns
from real_motion.causal_rollout_handoff import handoff_from_prepared, PROTOCOL as HANDOFF_PROTOCOL
from real_motion.nuscenes_adapter import NuScenesWindowSource, gt_moving_support_sequence
from real_motion.runtime_config import add_config_args, load_runtime_config, make_prepare_config
from real_motion.v21_source_induction import stable_json_fingerprint
from tools.real_motion import joint_long_rollout_common as rollout
from tools.real_motion import geniedrive_eval_alignment as genie
from tools.real_motion import joint_surface_long_rollout_common as surface
from tools.real_motion.eval_p0_f9_joint_surface_mean_full import find_frozen_bundle
from tools.real_motion.compare_p0_f9_joint_surface_checkpoints import verify_sources, VAL_WINDOWS
from tools.real_motion.joint_surface_checkpoint_selection import (
    AVERAGE_NAME, AVERAGE_EPOCHS, load_evaluation_model, training_implementation,
)
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from tools.real_motion.joint_training_recovery import snapshot_checkpoint
from tools.real_motion.eval_p0_f9_v18_se2 import load_cache
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import sha256, load_manifest
from tools.real_motion.train_p0_f9_causal_columns import record_keys
from tools.real_motion.train_p0_f9_v18_xy_trajectory import DEV64_FP
from tools.real_motion.static_evidence_selector_common import CLEAN_SHA256, write_json, finite_json
from tools.real_motion.run_p0_f9_shared_evidence_pilot import require_cuda

FILES = (
    'tools/real_motion/eval_p0_f9_joint_surface_long_rollout.py',
    'tools/real_motion/joint_surface_long_rollout_common.py',
    'tools/real_motion/joint_long_rollout_common.py',
    'tools/real_motion/geniedrive_eval_alignment.py',
    'real_motion/causal_rollout_handoff.py', 'real_motion/surface_ccr_execution.py',
    'real_motion/runtime_fastpath.py', 'real_motion/rigid_transport.py',
    'real_motion/strong_warp_execution.py',
    'real_motion/canonical_causal_repair.py', 'real_motion/canonical_repair_context.py',
    'tools/real_motion/causal_column_common.py',
    'tools/real_motion/benchmark_p0_f9_v18_runtime.py',
    'tools/real_motion/eval_p0_f9_v18_zero_shot_long_rollout.py',
    'real_motion/native/column_cpu.cpp',
)


def new_state(routes):
    return dict(completed_windows=0, first_block_exactness_passed=False,
        second_block_exactness_passed=False,
        counts={r:{k:v.tolist() for k,v in rollout.legacy._new_raw().items()} for r in routes},
        edits={'first': {}, **{r:{} for r in routes}}, stage_seconds={}, handoff_audit={})


def restore_state(saved, contract, total):
    value = dict(saved); digest = value.pop('fingerprint', None)
    if (digest != stable_json_fingerprint(value)
            or value.get('contract_fingerprint') != stable_json_fingerprint(contract)):
        raise RuntimeError('Surface long resume state/contract changed')
    routes = contract['routes']
    if set(value['counts']) != set(routes) or set(value['edits']) != {'first', *routes}:
        raise RuntimeError('Surface long resume routes incomplete')
    for row in value['counts'].values():
        rollout.validate_resume_state(dict(contract_fingerprint=stable_json_fingerprint(contract),
            completed_windows=value['completed_windows'], first_block_exactness_passed=value['first_block_exactness_passed'],
            raw_counts=row), contract, total)
    if type(value['completed_windows']) is not int or (value['completed_windows'] and not value['second_block_exactness_passed']):
        raise RuntimeError('Surface long resume lacks second-block exactness')
    for block in value['edits'].values():
        if any(type(v) is not int or v < 0 for v in block.values()) or block.get('removed', 0):
            raise RuntimeError('invalid saved Surface ADD-only edit counts')
    for row in (value['stage_seconds'], value['handoff_audit']):
        if any(type(v) not in (int,float) or not np.isfinite(v) or v < 0 for v in row.values()):
            raise RuntimeError('invalid saved Surface timing/handoff totals')
    return value


@torch.no_grad()
def evaluate_windows(provider, source, selected, execution, contract, *, saved=None,
                     save=None, progress=None, stop_event=None, checkpoint_every=8):
    """Only complete ALL-route windows enter integer checkpoints.

    Future GT occupancy/annotation is loaded only after BOTH open-loop blocks
    (and the fixed redetect comparison) have finished. Moving keeps original t0.
    """
    routes = contract['routes']
    state = new_state(routes) if saved is None else restore_state(saved, contract, len(selected))
    counts = {r:{k:np.asarray(v,np.int64) for k,v in row.items()} for r,row in state['counts'].items()}
    stages = defaultdict(float, state['stage_seconds']); audit = defaultdict(float, state['handoff_audit'])
    edits = {r:defaultdict(int,row) for r,row in state['edits'].items()}
    checked_first = checked_second = False  # Reverify live execution on EVERY process, even resume.
    def persist():
        state.update(counts={r:{k:v.tolist() for k,v in row.items()} for r,row in counts.items()},
            stage_seconds=dict(stages), handoff_audit=dict(audit), edits={r:dict(row) for r,row in edits.items()},
            contract_fingerprint=stable_json_fingerprint(contract))
        value = dict(state); value['fingerprint'] = stable_json_fingerprint(value)
        if save: save(value)
        return value
    persist()
    chosen = {(str(w.scene_name),str(w.t0_token)):w for w,_ in selected}
    iterator = prefetch_raw_columns(provider, source, [r for _,r in selected[state['completed_windows']:]], include_gt=False)
    try:
        previous_end = time.perf_counter()
        for record,raw in iterator:
            if stop_event is not None and stop_event.is_set(): raise InterruptedError('stopped before next Surface window')
            tick = time.perf_counter(); timing = {'input_wait':tick-previous_end}
            window = chosen[(str(record['scene_name']),str(record['t0_token']))]
            t = time.perf_counter()
            first = provider.prepare_columns(source, record, include_gt=False, raw_window=raw)
            record = first.state['rec']  # includes rebuilt early-start four-history ABI
            timing['first_prepare'] = time.perf_counter()-t; t = time.perf_counter()
            pred1, first_edits, first_stages, probability = execution.predict(first)
            timing['first_forecast'] = time.perf_counter()-t
            if not checked_first:
                t = time.perf_counter()
                surface.verify_first_block(provider,record,first,pred1,probability,execution)
                state['first_block_exactness_passed'] = checked_first = True
                timing['first_block_exactness'] = time.perf_counter()-t
            del probability
            poses = [source.pose(t) for t in window.future_tokens]
            t = time.perf_counter(); redetect = None
            if 'redetect' in routes:
                redetect = surface.synthetic_preparation(pred1,raw,poses,window,provider)
            carry = handoff_from_prepared(first,pred1[-1],dt_s=provider.pcfg.frame_dt_s,
                                         max_speed_mps=provider.strong.max_match_speed_mps)
            second = surface.synthetic_preparation(pred1,raw,poses,window,provider,handoff=carry,
                frames=None if redetect is None else redetect.state['components_by_frame'])
            timing['second_prepare'] = time.perf_counter()-t; t = time.perf_counter()
            pred2, second_edits, second_stages, probability = execution.predict(second)
            timing['second_forecast'] = time.perf_counter()-t
            if not checked_second:
                t = time.perf_counter(); execution.verify(second,pred2,probability)
                state['second_block_exactness_passed'] = checked_second = True
                timing['second_block_exactness'] = time.perf_counter()-t
            del probability
            predictions = {'reconciled':pred1+pred2}; route_edits = {'reconciled':second_edits}
            if redetect is not None:
                t = time.perf_counter(); other, e, _, _ = execution.predict(redetect)
                predictions['redetect'] = pred1+other; route_edits['redetect'] = e
                timing['redetect_second_forecast'] = time.perf_counter()-t
            handoff = second.state['motion_handoff_audit']
            # No source.load_semantics(future)/annotation support before this line.
            t = time.perf_counter()
            indices = (1,3,5,7,9,11); tokens = tuple(window.future_tokens[i] for i in indices)
            moving = gt_moving_support_sequence(source.nusc,window.t0_token,tokens,
                rollout.REPORT_HORIZONS,grid=provider.pcfg.grid,workers=provider.workers)
            for hi,(idx,token) in enumerate(zip(indices,tokens)):
                gt = source.load_semantics(window.scene_name,token)
                for route,dense in predictions.items():
                    rollout.update_metrics(counts[route],hi,dense[idx],gt,moving[hi][0],provider.pcfg.free_label)
            timing['metrics'] = time.perf_counter()-t
            for key,value in first_edits.items(): edits['first'][key] += value
            for route,row in route_edits.items():
                for key,value in row.items(): edits[route][key] += value
            for key,value in handoff.items():
                if type(value) in (int,float): audit[key] += value
            for label,row in (('first_detail',first_stages),('second_detail',second_stages)):
                for key,value in row.items(): timing[label+'/'+key] = value
            timing['total_window'] = time.perf_counter()-tick+timing['input_wait']
            for key,value in timing.items(): stages[key] += value
            state['completed_windows'] += 1; cursor = state['completed_windows']
            stopping = stop_event is not None and stop_event.is_set()
            if cursor%checkpoint_every==0 or cursor==len(selected) or stopping: persist()
            if progress: progress(dict(event='surface_long_window',window=cursor,windows=len(selected),
                scene_name=window.scene_name,t0_token=window.t0_token,seconds=timing,
                first_edits=first_edits,second_edits=route_edits,handoff=handoff,
                second_sources=len(second.state['current'])))
            del first,second,redetect,pred1,pred2,predictions,raw,record,carry
            if stopping: raise InterruptedError('stopped at complete Surface window')
            previous_end = time.perf_counter()
    except InterruptedError:
        persist(); raise
    finally:
        iterator.close()
    return persist()


def summary(result):
    lines = ['===== FROZEN SURFACE CCR MEAN ZERO-SHOT 1--6s =====',
        'protocol: '+surface.PROTOCOL, 'source_epochs: '+str(list(AVERAGE_EPOCHS))+'; one averaged network, not an ensemble',
        f"population: {result['population']['population']} / {result['windows']} windows / {result['population']['scenes']} scenes",
        'strict4 -> six predictions -> last4 predictions -> six predictions; NO training/recalibration',
        'weighted ADD raw sigmoid@0.5 / REMOVEoff; main route=reconciled, NO route selection',
        'GT ego poses through 6s; NO future occupancy/masks/annotation inputs; inherited initial visibility only',
        'Moving support refers to ORIGINAL t0; horizons=nominal contiguous 2Hz keyframe steps',
        'route          horizon      mIoU        IoU MovingMicro MovingMacro']
    for name,entry in result['routes'].items():
        metrics = entry['metrics']
        for h,row in metrics['per_horizon'].items():
            lines.append(f'{name:14s} {h:>6}s '+' '.join('NA'.rjust(11) if row[k] is None else f'{row[k]:11.6f}'
                         for k in ('mIoU','IoU','MovingMicro','MovingMacro')))
        for group in ('average_1s_2s_3s','average_4s_5s_6s'):
            lines.append(name+' '+group+': '+json.dumps(metrics[group]))
    if 'geniedrive_code_compatibility' in result:
        lines += ['===== GENIEDRIVE PUBLIC CODE COMPATIBILITY, NOT STANDARD mIoU =====',
            '4+20 metadata population, forecast ONLY12 frames; exact-zero semantic classes excluded and rounded2.',
            'Public-code alignment only; paper Table-2 population not independently verified.']
        for name,metrics in result['geniedrive_code_compatibility'].items():
            lines.append(name+': '+json.dumps(metrics))
    lines += ['first/second block exactness: '+str((result['first_block_exactness_passed'],result['second_block_exactness_passed'])),
        'stage_seconds: '+json.dumps(result['stage_seconds']),
        f"seconds/window: {result['stage_seconds'].get('total_window',0)/max(result['windows'],1):.4f}",
        'handoff_audit: '+json.dumps(result['handoff_audit']),
        'Actual dataset quality evaluation, NOT formal FPS. No changes to main1--3s table, weights, optimizer or caches.']
    return '\n'.join(lines)+'\n'


def parser():
    p = argparse.ArgumentParser(description=__doc__); add_config_args(p)
    for name in ('run-dir','runs-root','out-dir','dev-cache','dev-info','dataroot','base-checkpoint','population-manifest'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--source-bundle-dir'); p.add_argument('--resume',action='store_true')
    p.add_argument('--population',choices=('dev64','dev512','all'),default='dev64')
    p.add_argument('--population-alignment',choices=('legacy_cache6s','geniedrive_code10s'),default='legacy_cache6s')
    p.add_argument('--geniedrive-info'); p.add_argument('--no-redetect-comparison',action='store_true')
    p.add_argument('--device',default='cuda'); p.add_argument('--cpu-workers',type=int,default=10)
    p.add_argument('--ccr-cpu-execution',choices=('numpy','native','native_parallel'),default='native_parallel')
    p.add_argument('--ccr-cpu-workers',type=int,default=4); p.add_argument('--surface-query-workers',type=int,default=4)
    p.add_argument('--prefetch-workers',type=int,default=4); p.add_argument('--frame-cache-mib',type=int,default=512)
    p.add_argument('--val-history-cache'); p.add_argument('--val-cache-ram-mib',type=int,default=256)
    p.add_argument('--no-graphs',action='store_true'); p.add_argument('--checkpoint-every',type=int,default=8)
    return p


def main(stop_event=None, argv=None):
    p = parser(); a = p.parse_args(argv); out = Path(a.out_dir).resolve(); root = Path(__file__).resolve().parents[2]
    aligned = a.population_alignment == 'geniedrive_code10s'
    if aligned and (a.population != 'all' or not a.geniedrive_info): p.error('GenieDrive requires all and official metadata')
    if not aligned and a.geniedrive_info: p.error('metadata requires GenieDrive alignment')
    if (not 1<=a.cpu_workers<=16 or not 1<=a.ccr_cpu_workers<=a.cpu_workers or not 1<=a.surface_query_workers<=8
            or not 1<=a.prefetch_workers<=min(4,a.cpu_workers) or not 0<=a.frame_cache_mib<=4096
            or not 0<=a.val_cache_ram_mib<=2048 or a.checkpoint_every<1): p.error('invalid bounded resource budget')
    if any((d/'training.json').is_file() for d in (out,*out.parents)): p.error('evaluation outside training directories required')
    if a.resume:
        if not (out/'evaluation_state.json').is_file() or not (out/'bundle.json').is_file(): p.error('resume SAME evaluation output')
        if (out/'evaluation.json').is_file(): p.error('already complete; read summary.txt')
    elif out.exists(): p.error('new output required; never overwrite an experiment')
    for name in ('config','dev_cache','dev_info','base_checkpoint','population_manifest'):
        if not Path(getattr(a,name) or '').is_file(): p.error('missing '+name)
    if not Path(a.dataroot).is_dir(): p.error('missing dataroot')
    # Reuse the saved source bundle on resume, not a newly discovered comparison.
    bundle = (json.loads((out/'bundle.json').read_text(encoding='utf-8')) if a.resume else
              find_frozen_bundle(a.runs_root,a.run_dir,a.source_bundle_dir))
    digest = bundle.pop('fingerprint')
    if (stable_json_fingerprint(bundle)!=digest or bundle.get('selection_frozen') is not True
            or set(bundle['candidates'])!={AVERAGE_NAME}
            or bundle['run_directory']!=str(Path(a.run_dir).resolve())):
        raise RuntimeError('frozen mean bundle changed')
    bundle['fingerprint'] = digest
    if a.source_bundle_dir and str(Path(a.source_bundle_dir).resolve())!=bundle['source_comparison_directory']:
        p.error('source comparison changed')
    if out.is_relative_to(Path(bundle['source_comparison_directory'])): p.error('output outside source comparison required')
    verify_sources(bundle); trained = bundle['audit']['contract']
    for name in ('dev_cache','dev_info','base_checkpoint'):
        if sha256(getattr(a,name))!=trained['data'][name]: raise RuntimeError('original data changed: '+name)
    if sha256(a.base_checkpoint)!=CLEAN_SHA256 or str(Path(a.dataroot).resolve())!=trained['dataroot']:
        raise RuntimeError('original renderer/dataroot required')
    cfg = load_runtime_config(a.config,a.override); pcfg = make_prepare_config(cfg)
    if (stable_json_fingerprint(cfg)!=trained['runtime_config_fingerprint']
            or training_implementation(root)!=trained['implementation'] or str(torch.__version__)!=trained['torch_version']):
        raise RuntimeError('original model/config/Torch contract required')
    if pcfg.future_frames!=6 or pcfg.free_label!=17 or not np.isclose(pcfg.frame_dt_s,.5,rtol=0,atol=1e-12):
        raise RuntimeError('six future / nominal2Hz / nuScenes semantic contract required')
    manifest,keys64,_ = load_manifest(a.population_manifest)
    if (manifest['manifest_fingerprint']!=trained['dev_manifest_fingerprint'] or manifest['selected_key_fingerprint']!=DEV64_FP
            or len(keys64)!=64 or len(manifest['parent_keys'])!=512): raise RuntimeError('frozen development population changed')
    device = require_cuda(a.device); torch.set_num_threads(1)
    if not a.resume: out.mkdir(parents=True)
    with evaluation_lock(out):
        snapshot = out/'checkpoint_snapshot.pt'; row = bundle['candidates'][AVERAGE_NAME]
        if a.resume:
            if sha256(snapshot)!=row['sha256']: raise RuntimeError('immutable Surface snapshot changed')
        else:
            if snapshot_checkpoint(row['path'],snapshot)!=row['sha256']:
                raise RuntimeError('source mean changed before snapshot publication')
            write_json(out/'bundle.json',bundle)
        saved,joint = load_evaluation_model(snapshot,device=device,z_bins=int(pcfg.grid.shape_hwd[2]))
        if (saved['source_epochs']!=list(AVERAGE_EPOCHS) or joint.transport.config.history_frames!=4
                or saved['weight_fingerprint']!=row['weight_fingerprint']
                or stable_json_fingerprint(saved['training_contract'])!=stable_json_fingerprint(trained)):
            raise RuntimeError('frozen mean recipe/four-history contract changed')
        _,records = load_cache(a.dev_cache); record_keys(records)
        if len(records)!=VAL_WINDOWS: raise RuntimeError('complete original VAL4369 required')
        source = CachedColumnSource(NuScenesWindowSource(a.dataroot,info_pkl=a.dev_info,verbose=False),a.frame_cache_mib)
        if aligned:
            genie.validate_grid(pcfg); selected,population = genie.select_population(a.geniedrive_info,source,records)
        else:
            selected,population = rollout.select_long_population(records,source.iter_windows(history=4,future=12),
                                                                manifest['parent_keys'],a.population)
        del records,saved
        if {w.scene_name for w,_ in selected}&{str(s) for s,_ in trained['prior_keys']}:
            raise RuntimeError('TRAIN/development scene overlap')
        timestamps = [rollout.validate_timestamps(source.nusc,w) for w,_ in selected]
        timestamp_audit = rollout.summarize_timestamps(timestamps)
        provider = surface.SurfaceRolloutProvider(a.base_checkpoint,CLEAN_SHA256,pcfg,device,a.cpu_workers,joint,None)
        provider.raw_prefetch_workers = provider.raw_prefetch_depth = a.prefetch_workers
        provider.raw_io_workers = 1; cache = None; execution = None
        try:
            cache_namespace = None
            if a.val_history_cache:
                seed = val_namespace(provider,a,root)
                cache_namespace = hashlib.sha256((CACHE_PROTOCOL+seed).encode()).hexdigest()
                if (cache_namespace!=trained['val_history_namespace']
                        or not (Path(a.val_history_cache)/cache_namespace/'manifest.json').is_file()):
                    raise RuntimeError('existing verified VAL namespace required; no directory/cache builds')
                cache = CausalGeometryCache(a.val_history_cache,seed,max_bytes=0,ram_bytes=a.val_cache_ram_mib*2**20,reserve_bytes=0)
                validate_manifest(cache,a); provider.rollout_val_cache = cache
            from real_motion.strong_warp_execution import selected_backend
            from real_motion.native_column_cpu import backend_name
            contract = dict(protocol=surface.PROTOCOL,bundle_fingerprint=digest,snapshot_sha256=row['sha256'],
                source_epochs=list(AVERAGE_EPOCHS),weight_fingerprint=row['weight_fingerprint'],population=population,
                selected_future_tokens=[list(w.future_tokens) for w,_ in selected],timestamp_audit=timestamp_audit,
                primary_route=surface.PRIMARY_ROUTE,routes=['reconciled']+([] if a.no_redetect_comparison else ['redetect']),
                handoff_protocol=HANDOFF_PROTOCOL,thresholds=list(surface.THRESHOLDS),
                observation_protocol=rollout.OBSERVATION_PROTOCOL,future_GT_prediction_inputs=False,
                history_frames=4,future_frames_per_block=6,rollout_blocks=2,future_ego_pose_source='GT_through_6s',
                geniedrive_info_sha256=genie.INFO_SHA256 if aligned else None,
                runtime_config_fingerprint=stable_json_fingerprint(cfg),data=trained['data'],
                val_history_namespace=cache_namespace,torch_version=str(torch.__version__),
                execution={k:getattr(a,k) for k in ('device','cpu_workers','ccr_cpu_execution','ccr_cpu_workers',
                    'surface_query_workers','prefetch_workers','frame_cache_mib','val_cache_ram_mib','no_graphs')},
                strong_warp_backend=selected_backend(), integer_cpu_backend=backend_name(),
                implementation=stable_json_fingerprint({n:sha256(root/n) for n in FILES}))
            if a.resume:
                previous=json.loads((out/'contract.json').read_text(encoding='utf-8'))
                if previous!=contract: raise RuntimeError('Surface long resume contract changed')
                state=json.loads((out/'evaluation_state.json').read_text(encoding='utf-8'))
            else:
                write_json(out/'contract.json',contract);state=None
            write_json(out/'timestamp_audit.json',dict(summary=timestamp_audit,per_window=timestamps))
            execution = surface.SurfaceBlockExecution(provider,mode=a.ccr_cpu_execution,workers=a.ccr_cpu_workers,
                query_workers=a.surface_query_workers,graphs=not a.no_graphs)
            print(f'FROZEN SURFACE MEAN: epochs={AVERAGE_EPOCHS}; windows={len(selected)}; primary=reconciled; NO GT inputs',flush=True)
            with (out/'progress.jsonl').open('a' if a.resume else 'x',encoding='utf-8') as log:
                def progress(value):
                    log.write(json.dumps(finite_json(value),allow_nan=False)+'\n');log.flush()
                    if value['window']==1 or value['window']%16==0 or value['window']==value['windows']:
                        print(f"surface_long={value['window']}/{value['windows']} seconds={value['seconds']['total_window']:.3f} linked={value['handoff']['matched_sources']}",flush=True)
                try:
                    final = evaluate_windows(provider,source,selected,execution,contract,saved=state,
                        save=lambda v:write_json(out/'evaluation_state.json',v),progress=progress,
                        stop_event=stop_event,checkpoint_every=a.checkpoint_every)
                except InterruptedError:
                    boundary=json.loads((out/'evaluation_state.json').read_text(encoding='utf-8'))['completed_windows']
                    write_json(out/'evaluation_status.json',dict(status='interrupted',completed_windows=boundary))
                    print(f'STOPPED at complete window {boundary}; resume SAME directory with --resume',flush=True);return 130
            result = {**final,'protocol':surface.PROTOCOL,'status':'complete','windows':len(selected),
                'population':population,'primary_route':surface.PRIMARY_ROUTE,'source_epochs':list(AVERAGE_EPOCHS),
                'snapshot_sha256':row['sha256'],'checkpoint_snapshot':str(snapshot),'weight_fingerprint':row['weight_fingerprint'],
                'routes':{r:dict(metrics=rollout.finalize_metrics({k:np.asarray(v,np.int64) for k,v in counts.items()}),
                                raw_counts=counts) for r,counts in final['counts'].items()},
                'timestamp_audit':timestamp_audit,'execution':execution.head.stats(),
                'val_cache':None if cache is None else cache.stats(), 'future_GT_prediction_inputs':False,
                'no_training':True,'no_automatic_route_selection':True,'paper_table_population_verified':False}
            if aligned:
                result['geniedrive_code_compatibility']={r:genie.compatibility_metrics({k:np.asarray(v,np.int64) for k,v in counts.items()})
                    for r,counts in final['counts'].items()}
            verify_sources(bundle)
            if sha256(snapshot)!=row['sha256']:raise RuntimeError('read-only snapshot changed during evaluation')
            result=finite_json(result);write_json(out/'evaluation.json',result)
            (out/'summary.txt').write_text(summary(result),encoding='utf-8')
            write_json(out/'evaluation_status.json',dict(status='complete',completed_windows=len(selected)))
            print(summary(result),flush=True);return 0
        finally:
            if execution is not None:execution.close()
            if cache is not None:cache.close()


if __name__ == '__main__':
    stopped=threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stopped.set())
    sys.exit(main(stopped) or 0)
