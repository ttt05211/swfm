#!/usr/bin/env python3
"""Predeclared epoch6/8/12 + dev64 top-five mean; read-only resumable dev512 comparison."""
import sys
from pathlib import Path
if __package__ in (None,''): sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse
import json
import math
import signal
import threading
import time

import torch

from real_motion.column_runtime_pipeline import CachedColumnSource
from real_motion.nuscenes_adapter import NuScenesWindowSource
from real_motion.runtime_config import add_config_args,load_runtime_config,make_prepare_config
from real_motion.v21_source_induction import stable_json_fingerprint
from tools.real_motion import ccr_screen_common as ccr
from tools.real_motion import surface_ccr_screen_common as surface
from tools.real_motion.joint_surface_checkpoint_selection import (
    PROTOCOL,METRICS,AVERAGE_EPOCHS,SINGLE_EPOCHS,AVERAGE_NAME,discover,build_bundle,
    load_evaluation_model,training_implementation,
)
from tools.real_motion.joint_surface_checkpoint_evaluation import evaluate_group
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from tools.real_motion.run_p0_f9_shared_evidence_pilot import PilotProvider,require_cuda
from tools.real_motion.eval_p0_f9_v18_se2 import load_cache
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import load_manifest,align_records,sha256
from tools.real_motion.train_p0_f9_causal_columns import record_keys
from tools.real_motion.train_p0_f9_v18_xy_trajectory import DEV64_FP
from tools.real_motion.static_evidence_selector_common import CLEAN_SHA256,write_json,finite_json

EVALUATION_PROTOCOL = 'p0_f9_joint_surface_checkpoint_comparison_v1'
FULL_EVALUATION_PROTOCOL = 'p0_f9_joint_surface_frozen_mean_full4369_v1'
DEV64_WINDOWS, DEV512_WINDOWS, VAL_WINDOWS = 64, 512, 4369


def metric_delta(current, reference):
    """Keep absent-class metrics undefined after JSON NaN -> null conversion."""
    def difference(a,b):
        if a is None or b is None or not math.isfinite(a) or not math.isfinite(b):
            return None
        return float(a)-float(b)
    result={k:difference(current[k],reference[k]) for k in METRICS}
    result['per_horizon']={}
    if set(current['per_horizon'])!=set(reference['per_horizon']):
        raise RuntimeError('comparison horizon schemas differ')
    for h,row in current['per_horizon'].items():
        other=reference['per_horizon'][h]
        values={k:difference(row[k],other[k]) for k in METRICS}
        for name in ('semantic_per_class','moving_per_class'):
            if set(row[name])!=set(other[name]):
                raise RuntimeError('comparison class schemas differ: '+name)
            values[name]={cid:difference(v,other[name][cid]) for cid,v in row[name].items()}
        result['per_horizon'][h]=values
    return result


def parser():
    p=argparse.ArgumentParser(description=__doc__);add_config_args(p)
    for name in ('run-dir','runs-root','out-dir','dev-cache','dev-info','dataroot','base-checkpoint','population-manifest'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--resume',action='store_true',help='resume SAME comparison directory, never training')
    p.add_argument('--bundle-only',action='store_true',help='audit/export only; no CUDA, training or evaluation')
    p.add_argument('--device',default='cuda');p.add_argument('--cpu-workers',type=int,default=10)
    p.add_argument('--frame-cache-mib',type=int,default=2048)
    p.add_argument('--checkpoint-every',type=int,default=8)
    surface.add_args(p)
    p.set_defaults(ccr_history_cache_mode='off',ccr_history_cache=None,ccr_prefetch_workers=4,
                   ccr_motion_superbatch_updates=1,descriptor_disk_mib=0,descriptor_ram_mib=0)
    return p


def verify_sources(bundle):
    for row in [bundle['audit']['anchor'],*bundle['audit']['sources']]:
        if sha256(row['path'])!=row['sha256']:
            raise RuntimeError('original checkpoint changed: '+row['path'])
    for row in bundle['candidates'].values():
        if sha256(row['path'])!=row['sha256']:
            raise RuntimeError('evaluation snapshot changed: '+row['path'])
    for row in bundle.get('source_comparison_files',[]):
        if sha256(row['path'])!=row['sha256']:
            raise RuntimeError('source comparison changed: '+row['path'])


def summary(result):
    full=result.get('population')=='full4369'
    lines=['===== CLEAN JOINT SURFACE CCR / '+('FROZEN MEAN FULL4369' if full else 'CHECKPOINT COMPARISON')+' =====',
        ('full4369 validation includes the development subset; NOT an independent test.' if full else
         'dev512 DEVELOPMENT/selection scores, not independent test;')+' fixed weighted ADD raw0.5 / REMOVEoff.',
        'Mean: epoch5/6/8/12/14, dev64 mIoU top5, equal named-parameter weights, fixed buffers unchanged.',
        'Original checkpoints/optimizer/RNG/caches READ ONLY; no training/threshold search/promotion.',
        'population='+result.get('population','dev512')+' windows='+str(result.get('windows','unknown')),
        'candidate                       IoU       mIoU MovingMacro MovingMicro']
    reports=result.get('reports',{})
    for name,report in reports.items():
        m=report['variants']['joint']['metrics']
        lines.append(f'{name:28s} '+' '.join(f'{m[k]:10.6f}' for k in METRICS))
    if reports:
        lines.append('===== PER HORIZON: Joint four metrics =====')
        for horizon in ('1.0','2.0','3.0'):
            lines.append('horizon='+horizon+'s')
            for name,report in reports.items():
                m=report['variants']['joint']['metrics']['per_horizon'][horizon]
                lines.append(f'{name:28s} '+' '.join('NA'.rjust(10) if m[k] is None else f'{m[k]:10.6f}' for k in METRICS))
        lines.append('===== TRANSPORT four metrics (repair contribution isolation) =====')
        for name,report in reports.items():
            m=report['baseline'];lines.append(f'{name:28s} '+' '.join(f'{m[k]:10.6f}' for k in METRICS))
        if full:
            lines.append('===== BRANCHES: four metrics / changes relative to CURRENT Transport =====')
            for report in reports.values():
                lines.append('scenes='+str(report['scenes']))
                for name,row in report['variants'].items():
                    lines.append(f'{name:28s} '+' '.join(f'{row["metrics"][k]:10.6f}' for k in METRICS))
                    lines.append('  delta_pp='+json.dumps({k:row['delta_vs_v18_pp'][k] for k in METRICS}))
    lines += ['status='+result['status'],
              ('ONE frozen mean evaluated; no single-epoch reruns or imported dev512 scores.' if full else
               'Epoch20 reuses its completed same-contract dev512 report, NOT another inference pass.'),
              'Averaged model is ONE evaluation-only model, not a runtime ensemble or training resume.',
              'Performance='+json.dumps(result.get('performance',{}),ensure_ascii=False)]
    return '\n'.join(lines)+'\n'


def main(stop_event=None,argv=None):
    p=parser();a=p.parse_args(argv);out=Path(a.out_dir).resolve()
    if any((directory/'training.json').is_file() for directory in (out,*out.parents)):
        p.error('comparison output cannot be inside an original training run')
    if not 1<=a.cpu_workers<=16 or not 1<=a.surface_query_workers<=8 or not 0<=a.frame_cache_mib<=8192 or a.checkpoint_every<1:
        p.error('invalid CPU/RAM/checkpoint budget')
    if a.warm_start_head or a.descriptor_disk_mib or a.descriptor_cache or a.ccr_add_only_natural_bce or a.ccr_history_cache:
        p.error('evaluation only: no warm start, TRAIN cache, natural loss or descriptor disk writes')
    if a.resume:
        if a.bundle_only or not (out/'bundle.json').is_file(): p.error('resume requires an existing comparison bundle')
        if (out/'comparison.json').is_file(): p.error('comparison already complete; do not rerun')
    elif out.exists(): p.error('new output required; no checkpoint/results overwrite')
    else: out.mkdir(parents=True)
    with evaluation_lock(out):
        if a.resume:
            bundle=json.loads((out/'bundle.json').read_text(encoding='utf-8'))
            declared=bundle.pop('fingerprint')
            if stable_json_fingerprint(bundle)!=declared or bundle['protocol']!=PROTOCOL:
                raise RuntimeError('comparison bundle fingerprint/protocol mismatch')
            bundle['fingerprint']=declared
            if bundle['run_directory']!=str(Path(a.run_dir).resolve()): p.error('anchor run changed on resume')
        else:
            audit=discover(a.run_dir,a.runs_root)
            for source in [audit['anchor'],*audit['sources']]:
                if out.is_relative_to(Path(source['path']).parent):
                    raise RuntimeError('comparison output cannot be inside an original training run')
            bundle=build_bundle(audit,out)
            bundle.update(run_directory=str(Path(a.run_dir).resolve()),runs_root=str(Path(a.runs_root).resolve()))
            bundle['fingerprint']=stable_json_fingerprint(bundle)
            write_json(out/'bundle.json',bundle)
            lines=['===== ORIGINAL DEV64 FOUR-METRIC AUDIT =====','epoch update IoU mIoU MovingMacro MovingMicro']
            for row in audit['dev64']:
                lines.append(f"{row['epoch']:5d} {row['update']:7d} "+' '.join(f'{row[k]:.6f}' for k in METRICS))
            lines+=['average_epochs='+str(AVERAGE_EPOCHS),'single_epochs='+str(SINGLE_EPOCHS)]
            (out/'dev64_audit.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
            print('\n'.join(lines),flush=True)
        verify_sources(bundle)
        if a.bundle_only: print('Evaluation-only weights exported; no model run.',flush=True);return 0
        return evaluate(a,out,bundle,stop_event)


def evaluate(a,out,bundle,stop_event):
    root=Path(__file__).resolve().parents[2];contract=bundle['audit']['contract']
    population=getattr(a,'population','dev512')
    if population not in ('dev512','full4369'):raise RuntimeError('unsupported evaluation population')
    protocol=FULL_EVALUATION_PROTOCOL if population=='full4369' else EVALUATION_PROTOCOL
    if population=='full4369' and (set(bundle['candidates'])!={AVERAGE_NAME} or bundle.get('selection_frozen') is not True):
        raise RuntimeError('full4369 requires ONE explicitly frozen mean from a completed comparison')
    for name in ('config','dev_cache','dev_info','base_checkpoint','population_manifest'):
        if not Path(getattr(a,name) or '').is_file(): raise RuntimeError('missing '+name)
    if not Path(a.dataroot).is_dir() or not a.ccr_val_history_cache or not Path(a.ccr_val_history_cache).is_dir():
        raise RuntimeError('existing DATAROOT and read-only VAL history cache required')
    for name in ('dev_cache','dev_info','base_checkpoint'):
        if sha256(getattr(a,name))!=contract['data'][name]: raise RuntimeError('training data provenance changed: '+name)
    if sha256(a.base_checkpoint)!=CLEAN_SHA256: raise RuntimeError('wrong E14 renderer reference')
    if str(Path(a.dataroot).resolve())!=contract['dataroot']: raise RuntimeError('DATAROOT changed')
    if training_implementation(root)!=contract['implementation']:
        raise RuntimeError('training/model/renderer implementation changed; do not silently score incompatible weights')
    cfg=load_runtime_config(a.config,a.override)
    if stable_json_fingerprint(cfg)!=contract['runtime_config_fingerprint']: raise RuntimeError('runtime config changed')
    if str(torch.__version__)!=contract['torch_version']: raise RuntimeError('use the original training Torch environment')
    manifest,keys64,_=load_manifest(a.population_manifest);chosen=tuple(map(tuple,manifest['parent_keys']))
    if (manifest['manifest_fingerprint']!=contract['dev_manifest_fingerprint']
            or manifest['selected_key_fingerprint']!=DEV64_FP or len(keys64)!=DEV64_WINDOWS or len(chosen)!=DEV512_WINDOWS):
        raise RuntimeError('frozen dev64/dev512 manifest mismatch')
    _,records=load_cache(a.dev_cache);record_keys(records)
    if len(records)!=VAL_WINDOWS: raise RuntimeError('complete original VAL4369 records required')
    if population=='dev512':
        records=align_records(records,chosen)
        reference=finite_json(bundle['audit']['final_dev512'])
        if reference.get('windows')!=len(records): raise RuntimeError('existing epoch20 report population mismatch')
    else:
        chosen=tuple(record_keys(records));reference=None
    device=require_cuda(a.device);torch.set_num_threads(1);pcfg=make_prepare_config(cfg)
    models={}
    for name,row in bundle['candidates'].items():
        saved,model=load_evaluation_model(row['path'],device=device,z_bins=int(pcfg.grid.shape_hwd[2]))
        if stable_json_fingerprint(saved['training_contract'])!=stable_json_fingerprint(contract):
            raise RuntimeError('snapshot training contract changed')
        models[name]=model
    provider=PilotProvider(a.base_checkpoint,CLEAN_SHA256,pcfg,device,a.cpu_workers,next(iter(models.values())),None)
    source=CachedColumnSource(NuScenesWindowSource(a.dataroot,info_pkl=a.dev_info,verbose=False),a.frame_cache_mib)
    provider.ccr_val_history_cache_source=source
    result=dict(status='running',protocol=protocol,population=population,windows=len(records),
                bundle_fingerprint=bundle['fingerprint'],reports={},performance={})
    begun=time.perf_counter();accumulated=0.;cursor=0;resumed=None
    try:
        if not (Path(a.ccr_val_history_cache)/contract['val_history_namespace']/'manifest.json').is_file():
            raise RuntimeError('existing VAL namespace/manifest missing; no cache directory will be created')
        surface.setup(provider,a)
        if provider.ccr_val_history_cache.namespace!=contract['val_history_namespace']:
            raise RuntimeError('VAL history cache namespace changed')
        provider.ccr_batched_motion=False;provider.ccr_motion_streams=1
        provider.raw_prefetch_workers=provider.raw_prefetch_depth=min(4,a.ccr_prefetch_workers,a.cpu_workers)
        provider.raw_io_workers=1
        execution=dict(protocol=protocol,population=population,bundle=bundle['fingerprint'],
            population_key_fingerprint=stable_json_fingerprint(chosen),thresholds=[.5,None],
            val_history_namespace=provider.ccr_val_history_cache.namespace,
            torch_version=str(torch.__version__),cpu_workers=a.cpu_workers,
            surface_query_workers=a.surface_query_workers,reference_execution=a.surface_reference_execution,
            ccr_cpu_execution=a.ccr_cpu_execution,ccr_cpu_workers=a.ccr_cpu_workers,
            frame_cache_mib=a.frame_cache_mib,probability_chunk=8192,
            raw_prefetch_workers=provider.raw_prefetch_workers,raw_io_workers=provider.raw_io_workers,
            implementation=stable_json_fingerprint({n:sha256(root/n) for n in (
                'tools/real_motion/joint_surface_checkpoint_selection.py',
                'tools/real_motion/joint_surface_checkpoint_evaluation.py',
                'tools/real_motion/compare_p0_f9_joint_surface_checkpoints.py',
                'tools/real_motion/eval_p0_f9_v21_stage0_upper_bounds.py')}))
        if population=='full4369':
            execution['full_entry_sha256']=sha256(root/'tools/real_motion/eval_p0_f9_joint_surface_mean_full.py')
        if bool(a.surface_reference_execution)!=contract['reference_execution']:
            raise RuntimeError('use the original training reference/fused execution mode')
        if a.resume and (out/'evaluation_state.json').is_file():
            resumed=json.loads((out/'evaluation_state.json').read_text(encoding='utf-8'))
            declared=resumed.pop('fingerprint')
            if stable_json_fingerprint(resumed)!=declared or resumed['execution']!=execution:
                raise RuntimeError('evaluation resume inputs/implementation changed')
            cursor=resumed['completed_windows'];accumulated=resumed['elapsed_seconds']
            if type(cursor)is not int or not 0<=cursor<=len(records): raise RuntimeError('invalid comparison cursor')
        def save(wi,states,totals):
            state=dict(execution=execution,completed_windows=wi,states=states,
                       elapsed_seconds=accumulated+time.perf_counter()-begun)
            state['fingerprint']=stable_json_fingerprint(state);write_json(out/'evaluation_state.json',state)
        with (out/'progress.jsonl').open('a' if a.resume else 'x',encoding='utf-8') as log:
            def progress(row):
                log.write(json.dumps(finite_json(row),allow_nan=False)+'\n');log.flush()
                if row['event']=='all_model_window' and (row['window']==1 or row['window']%16==0 or row['window']==len(records)):
                    label='SURFACE_MEAN_FULL' if population=='full4369' else 'SURFACE_COMPARE'
                    print(f"{label} {row['window']}/{len(records)} candidates={len(models)} all-model boundary",flush=True)
            try:
                reports,performance=evaluate_group(provider,source,records,models,
                    saved=resumed['states'] if resumed else None,start_window=cursor,
                    save_state=save,progress=progress,stop_event=stop_event,checkpoint_every=a.checkpoint_every)
            except InterruptedError:
                result.update(status='interrupted',resume='same command and output plus --resume')
                write_json(out/'evaluation_status.json',result)
                print('Stopped at saved all-model boundary; --resume continues evaluation, NOT training.',flush=True)
                return 130
        if reference is not None:reports['epoch_0020_existing']=reference
        result.update(status='complete',reports=reports,execution=execution,performance=performance)
        if any(not isinstance(r['variants']['joint']['metrics'].get(k),(int,float))
               or not math.isfinite(r['variants']['joint']['metrics'][k])
               for r in reports.values() for k in METRICS):
            raise RuntimeError('missing/nonfinite aggregate comparison metric')
        if reference is not None:
            result['best_per_metric']={k:max(reports,key=lambda n:reports[n]['variants']['joint']['metrics'][k]) for k in METRICS}
            reference_metrics=reference['variants']['joint']['metrics']
            result['delta_joint_vs_epoch20']={n:metric_delta(r['variants']['joint']['metrics'],reference_metrics) for n,r in reports.items()}
        verify_sources(bundle)
    finally:
        ccr.close(provider,result)
        result['performance'].update(elapsed_seconds_this_invocation=time.perf_counter()-begun,
                                    accumulated_seconds=accumulated+time.perf_counter()-begun,reused_prefix_windows=cursor)
    filename='full_validation.json' if population=='full4369' else 'comparison.json'
    write_json(out/filename,result);(out/'summary.txt').write_text(summary(finite_json(result)),encoding='utf-8')
    write_json(out/'evaluation_status.json',dict(status='complete',no_promotion=True))
    print(summary(finite_json(result)),flush=True)
    return 0


if __name__=='__main__':
    stopped=threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM): signal.signal(sig,lambda *_:stopped.set())
    sys.exit(main(stopped) or 0)
