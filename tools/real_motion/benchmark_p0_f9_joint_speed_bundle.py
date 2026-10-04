#!/usr/bin/env python3
"""One read-only snapshot: paired TRAIN throughput + dev inference exactness/time.

No new prior/calibration, scientific checkpoint, full training or auto-retry.
Warm geometry misses fail closed; no prefill or shared disk-cache writes.
"""
import sys
from pathlib import Path
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse
import copy
import json
import os
import time
import numpy as np
import torch

from real_motion.causal_geometry_cache import CausalGeometryCache
from real_motion.column_runtime_pipeline import CachedColumnSource
from real_motion.column_gpu_sampling import GpuColumnSampler
from real_motion.column_cpu_pipeline import cpu_sampling_pool
from real_motion.nuscenes_adapter import NuScenesWindowSource
from real_motion.runtime_config import load_runtime_config,make_prepare_config
from real_motion.v21_source_induction import stable_json_fingerprint
from real_motion.local_training_profile import trial_summary
from tools.real_motion.joint_column_full_common import (FullJointColumnProvider,EvaluationJointColumnProvider,
    prefetch_column_batches,train_full_batch)
from tools.real_motion.local_warm_cache_common import geometry_namespace
from tools.real_motion.manage_p0_f9_joint_training import model_directory,status
from tools.real_motion.joint_training_recovery import snapshot_checkpoint
from tools.real_motion.train_p0_f9_joint_causal_columns import load_joint
from tools.real_motion.train_p0_f9_causal_columns import record_keys
from tools.real_motion.eval_p0_f9_v18_se2 import load_cache
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import sha256,align_records,load_manifest
from tools.real_motion.static_evidence_selector_common import CLEAN_SHA256,write_json,finite_json
from tools.real_motion.causal_column_common import evaluate_columns,candidate_plan,predict_probabilities,REPORT


def path_args(contract):
    args=dict(contract['arguments']);cwd=Path(contract.get('launch_cwd',Path.cwd()))
    for k in ('config','train_cache','dev_cache','train_info','dev_info','base_checkpoint','dataroot','population_manifest','causal_geometry_cache'):
        if args.get(k):args[k]=str((cwd/Path(args[k])).resolve())
    return args


def switches(optimized):
    os.environ['SWFM_LOCAL_FAST_SUPERVISION']='1' if optimized else '0'
    os.environ['SWFM_LOCAL_STATIC_ROI']='1' if optimized else '0'


def summary_text(result):
    lines=['===== SINGLE GPU TRAIN + EVALUATION SPEED BUNDLE =====',
        'Same immutable checkpoint / TRAIN labels / RNG seed / batch4-source128 / full evaluation candidates.',
        'Paired paths share current common plumbing; previous disables host-supervision/ROI and uses CPU inference features.',
        'Timing only: no saved optimizer updates, new calibration, checkpoint selection or training launch.']
    for name,row in result.get('train',{}).items():
        lines.append(f"TRAIN {name}: seconds/window={row['seconds_per_window']:.6f} windows/s={row['windows_per_second']:.3f} epoch_if_representative={row['epoch_train_hours_if_representative']*60:.2f}min")
        lines.append('  host_stages_seconds='+json.dumps(row['stage_seconds']))
    for name,row in result.get('evaluation',{}).items():
        lines.append(f"EVAL {name}: windows={row['windows']} seconds/window={row['seconds_per_window']:.6f} full4369_if_representative={row['seconds_per_window']*4369/60:.2f}min")
    if 'train_speedup' in result:lines.append(f"train_speedup={result['train_speedup']:.4f}; eval_speedup={result['eval_speedup']:.4f}")
    lines += ['gate='+json.dumps(result.get('gate',{})), 'status='+result['status'],
        'Small-sample estimates only; OS/frame-cache, source/candidate density and order effects remain.',
        'Model/batch/LR/RNG/threshold contracts unchanged; elapsed time is NOT a real training result.']
    if result.get('error'):lines.append('error='+result['error'])
    return '\n'.join(lines)+'\n'


def train_equivalence(previous, optimized):
    """Exact sampling/RNG counts; tolerance for existing CUDA reductions only."""
    if (previous['sampling_rng_fingerprint'] != optimized['sampling_rng_fingerprint'] or
            previous['sampled_columns_by_update'] != optimized['sampled_columns_by_update'] or
            any(previous[k] != optimized[k] for k in ('windows','batches','sources'))):
        raise RuntimeError('training sampling/RNG/population changed')
    if not np.allclose(previous['loss_by_update'],optimized['loss_by_update'],rtol=2e-4,atol=1e-5):
        raise RuntimeError('paired CUDA training losses changed outside numerical tolerance')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',required=True);p.add_argument('--out-dir',required=True)
    p.add_argument('--train-windows',type=int,default=64);p.add_argument('--eval-windows',type=int,default=8)
    a=p.parse_args();out=Path(a.out_dir).resolve();directory=model_directory(a.run_dir)
    if out.exists():p.error('NEW output required')
    if not 4 <= a.train_windows <= 256 or not 1 <= a.eval_windows <= 32:p.error('bounded diagnostic populations required')
    if (directory/'runtime_status.json').is_file() and status(directory)['matching_trainer_running']:
        p.error('safely stop training before this GPU diagnostic')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('CUDA/BF16 required')
    from real_motion.native_column_cpu import prepare_native
    prepare_native();torch.set_num_threads(1)
    device=torch.device('cuda');args=path_args(json.loads((directory/'execution_contract.json').read_text(encoding='utf-8')))
    out.mkdir(parents=True);snapshot=out/'checkpoint_snapshot.pt'
    digest=snapshot_checkpoint(directory/'last.pt',snapshot)
    cfg=load_runtime_config(args['config'],args.get('override',[]));pcfg=make_prepare_config(cfg)
    ck,joint=load_joint(snapshot,device,reference_sha=CLEAN_SHA256,config_sha=stable_json_fingerprint(cfg),allow_diagnostic=True)
    if ck.get('checkpoint_role') != 'resume_last' or not ck.get('prior_completed',True):raise RuntimeError('completed TRAIN prior and resumable last.pt required')
    if ck['window_batch_size'] != 4 or ck['source_budget'] != 128:raise RuntimeError('this comparison preserves the existing batch4/source128')
    if ck.get('continuation') and ck['attempted_updates'] >= ck['target_updates']:
        raise RuntimeError('completed extension has no remaining updates; use the completed original15 run for timing')
    if sha256(args['base_checkpoint']) != CLEAN_SHA256:raise RuntimeError('reference E14 changed')
    for name,path in (('train',args['train_info']),('dev',args['dev_info'])):
        if sha256(path) != ck['info_fingerprints'][name]:raise RuntimeError('info provenance changed')
    for name,path in (('train',args['train_cache']),('dev',args['dev_cache'])):
        if sha256(path) != ck['cache_fingerprints'][name]:raise RuntimeError('cache provenance changed')
    _,records=load_cache(args['train_cache']);keys=record_keys(records)
    if tuple(keys) != tuple(map(tuple,ck['train_keys'])):raise RuntimeError('TRAIN identity/order changed')
    selected=np.random.default_rng(20261004).choice(len(records),a.train_windows,replace=False)
    train=[records[int(i)] for i in selected];del records
    _,records=load_cache(args['dev_cache']);record_keys(records)
    manifest,dev64,_=load_manifest(args['population_manifest'])
    if manifest['manifest_fingerprint'] != ck['dev_manifest_fingerprint']:raise RuntimeError('dev manifest changed')
    dev=align_records(records,dev64[:a.eval_windows]);del records
    initial=copy.deepcopy(joint.state_dict());del joint
    train_source=CachedColumnSource(NuScenesWindowSource(args['dataroot'],info_pkl=args['train_info'],verbose=False),256)
    dev_source=CachedColumnSource(NuScenesWindowSource(args['dataroot'],info_pkl=args['dev_info'],verbose=False),256)
    result=dict(status='in_progress',checkpoint_sha256=digest,train={},evaluation={},gate={})
    def model():
        torch.manual_seed(20261004)
        _,value=load_joint(snapshot,device,reference_sha=CLEAN_SHA256,config_sha=stable_json_fingerprint(cfg),allow_diagnostic=True)
        value.load_state_dict(initial);return value
    def provider(value,cls=FullJointColumnProvider):
        return cls(args['base_checkpoint'],CLEAN_SHA256,pcfg,device,args.get('cpu_workers',8),value,None)
    def training(optimized):
        switches(optimized);value=model();owner=provider(value)
        namespace=geometry_namespace(cfg,owner,ck['info_fingerprints'],ck['cache_fingerprints'],args['dataroot'])
        cache=CausalGeometryCache(args['causal_geometry_cache'],namespace,max_bytes=int(args.get('causal_cache_gib',48)*2**30),ram_bytes=0)
        owner.causal_geometry_cache=cache
        optimizer=torch.optim.AdamW([{'params':value.transport.parameters(),'lr':5e-4,'initial_lr':5e-4,'weight_decay':1e-4},
            {'params':value.columns.parameters(),'lr':3e-4,'initial_lr':3e-4,'weight_decay':.01}])
        optimizer.load_state_dict(copy.deepcopy(ck['optimizer']))
        sampler=GpuColumnSampler(device);rng=np.random.default_rng(20261004)
        rows=[];pool=cpu_sampling_pool(6,horizons=True)
        iterator=None
        try:
            # Warm up on SAME small population and restore the initial state.
            # First exactness / allocator / Adam setup is excluded from timing.
            warm=list(prefetch_column_batches(owner,train_source,train[:4],4,128))
            if any(not raw.get('_causal_geometry_cache_hit') for b in warm for _,raw in b):raise RuntimeError('warm geometry miss: no cache rebuild/write was attempted')
            train_full_batch(value,optimizer,owner,train_source,warm[0],rng,ck['attempted_updates']+1,ck['schedule_steps'],
                sampling_pool=pool,column_feature_sampler=sampler,continuation=ck.get('continuation'))
            value.load_state_dict(initial);optimizer.load_state_dict(copy.deepcopy(ck['optimizer']))
            torch.manual_seed(20261004);rng=np.random.default_rng(20261004)
            torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device)
            iterator=prefetch_column_batches(owner,train_source,train,4,128);tick=time.perf_counter()
            for i,batch in enumerate(iterator,1):
                waited=time.perf_counter()-tick
                if any(not raw.get('_causal_geometry_cache_hit') for _,raw in batch):raise RuntimeError('warm geometry miss; refusing to benchmark/write a new cache')
                started=time.perf_counter()
                row=train_full_batch(value,optimizer,owner,train_source,batch,rng,ck['attempted_updates']+i,ck['schedule_steps'],
                    sampling_pool=pool,column_feature_sampler=sampler,continuation=ck.get('continuation'),profile=i%8==0)
                torch.cuda.synchronize(device)
                row.update(wall_seconds=waited+time.perf_counter()-started,input_wait_seconds=waited,
                    peak_reserved_mib=torch.cuda.max_memory_reserved(device)/2**20)
                rows.append(row);tick=time.perf_counter()
            measurement=trial_summary(rows)
            if cache.stats()['writes']:raise RuntimeError('read-only warm benchmark unexpectedly wrote geometry cache')
            if any(not row['optimizer_updated'] for row in rows):raise RuntimeError('unsuccessful diagnostic update; no valid throughput result')
            measurement.update(sampling_rng_fingerprint=stable_json_fingerprint(rng.bit_generator.state),
                sampled_columns_by_update=[row['sampled_columns'] for row in rows],loss_by_update=[row['loss'] for row in rows])
            return measurement
        finally:
            if iterator is not None:iterator.close()
            pool.shutdown(wait=True,cancel_futures=True);cache.close()
    try:
        for optimized in (False,True):
            name='optimized' if optimized else 'previous'
            print('SPEED_BUNDLE TRAIN '+name,flush=True)
            result['train'][name]=training(optimized);write_json(out/'audit.json',finite_json(result))
        train_equivalence(result['train']['previous'],result['train']['optimized'])
        # Fixed checkpoint, NOT the diagnostic training updates above.
        reports={}
        for optimized in (False,True):
            name='optimized' if optimized else 'previous';switches(optimized)
            value=model();value.eval();owner=provider(value,EvaluationJointColumnProvider if optimized else FullJointColumnProvider)
            owner.reference_enabled=True
            if not optimized:owner.frozen_metric_counts=None
            # Renderer/startup check and direct real CUDA probability parity
            # are excluded from throughput. Keep original batch256 for both.
            prep=owner.prepare_columns(dev_source,dev[0],include_gt=True)
            if optimized:
                for h in REPORT:
                    plan=candidate_plan(prep,h,pcfg.grid,value.columns.config)
                    before=predict_probabilities(value.columns,prep,h,plan,pcfg.grid,device,256)
                    sampler=GpuColumnSampler(device)
                    with sampler.resident_window(prep,pcfg.grid,value.columns.config):
                        after=predict_probabilities(value.columns,prep,h,plan,pcfg.grid,device,256,
                            feature_backend='gpu',gpu_sampler=sampler,verify_features=True)
                    if not np.array_equal(before,after):raise RuntimeError('real full-population inference probabilities changed')
            events=[];started=time.perf_counter()
            reports[name]=evaluate_columns(owner,dev_source,dev,value.columns,(.5,.5,.95),batch_size=256,
                diagnostic_thresholds=None,feature_backend='gpu' if optimized else 'cpu',progress=events.append)
            elapsed=time.perf_counter()-started
            result['evaluation'][name]=dict(windows=len(dev),seconds=elapsed,seconds_per_window=elapsed/len(dev),progress=events)
            write_json(out/'audit.json',finite_json(result))
        equivalent=finite_json(reports['previous']) == finite_json(reports['optimized'])
        if not equivalent:raise RuntimeError('four-way metrics/confidences/edits changed; do not resume optimized run')
        result.update(status='complete',train_speedup=result['train']['previous']['seconds_per_window']/result['train']['optimized']['seconds_per_window'],
            eval_speedup=result['evaluation']['previous']['seconds_per_window']/result['evaluation']['optimized']['seconds_per_window'])
        result['gate']=dict(real_cuda_probability_exact=True,four_way_reports_exact=equivalent,
            training_sampling_rng_exact=True,training_loss_close=True,
            training_loss_tolerance=dict(rtol=2e-4,atol=1e-5),no_geometry_cache_writes=True,no_scientific_updates_saved=True)
        if sha256(snapshot) != digest:raise RuntimeError('immutable snapshot changed')
    except BaseException as error:
        result.update(status='failed',error=type(error).__name__+': '+str(error));raise
    finally:
        switches(True)
        write_json(out/'audit.json',finite_json(result));(out/'summary.txt').write_text(summary_text(result),encoding='utf-8')
        print(summary_text(result),flush=True)
    return 0


if __name__=='__main__':sys.exit(main())
