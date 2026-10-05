#!/usr/bin/env python3
"""ONE bounded bundle: zero-shot probe, GPU guards, joint speed/FPS, migration.

Different protocol from old full training. Never resumes/modifies epoch19's
optimizer. No full4369, automatic retry, threshold search or promotion.
"""
import sys
from pathlib import Path
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse
import copy
import json
import math
import time
import numpy as np
import torch
from real_motion.shared_column_evidence import SharedEvidenceColumns, PROTOCOL
from real_motion.runtime_config import add_config_args, load_runtime_config, make_prepare_config
from real_motion.column_runtime_pipeline import CachedColumnSource, prefetch_raw_columns
from real_motion.nuscenes_adapter import NuScenesWindowSource
from real_motion.causal_geometry_cache import CausalGeometryCache
from real_motion.v21_source_induction import stable_json_fingerprint
from tools.real_motion.joint_column_full_common import FullJointColumnProvider, build_fixed_geometry, prefetch_column_batches
from tools.real_motion.local_warm_cache_common import geometry_namespace
from tools.real_motion.train_p0_f9_joint_causal_columns import load_joint
from tools.real_motion.eval_p0_f9_v18_se2 import load_cache
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import load_manifest, align_records, sha256
from tools.real_motion.train_p0_f9_causal_columns import record_keys
from tools.real_motion.v18_source_interaction_common import select_population
from tools.real_motion.joint_training_recovery import snapshot_checkpoint
from tools.real_motion.static_evidence_selector_common import CLEAN_SHA256,write_json,atomic_checkpoint
from tools.real_motion.shared_evidence_pilot_common import (evaluate,training_speed,fps_speed,training_step,make_optimizer,sync,GATES)
from tools.real_motion.shared_evidence_recovery import migration_payload,restore_migration,validate_migration

TRAIN_WINDOWS=20430


def require_cuda(name):
    device=torch.device(name)
    if device.type!='cuda' or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('real data bundle requires actual CUDA/BF16; CPU unit tests are separate')
    return device


class PilotProvider(FullJointColumnProvider):
    def load_raw_columns(self,source,record,*,include_gt):
        from tools.real_motion.causal_column_common import FrozenColumns
        raw=FrozenColumns.load_raw_columns(self,source,record,include_gt=include_gt)
        def build():return build_fixed_geometry(raw,record,self.pcfg,self.strong,min(3,self.workers),self.joint.columns.config)
        cache=getattr(self,'causal_geometry_cache',None)
        if cache is None:evidence=build();hit=False
        else:evidence,hit=cache.get_or_build((str(record['scene_name']),str(record['t0_token'])),raw,build,defer_write=True)
        raw['_column_causal_preparation']=evidence;raw['_causal_geometry_cache_hit']=hit
        return raw


def epoch_records(records,seed,epochs):
    return [r for epoch in range(epochs) for r in (records[i] for i in
        np.random.default_rng(np.random.SeedSequence([seed,epoch])).permutation(len(records)))]


def group_count(records):
    count=0;windows=sources=0
    for record in records:
        n=len(record['features'])
        if windows and (windows>=4 or sources+n>128):count+=1;windows=sources=0
        windows+=1;sources+=n
    return count+bool(windows)


def brief(result):
    lines=['===== SHARED EVIDENCE ALL-IN-ONE PILOT =====',f"status: {result['status']}",
        f'protocol: {PROTOCOL}','history=4 -> future=6; fixed thresholds=0.5/0.5/0.95',
        'epoch19 READ ONLY. No old optimizer resume, full4369, threshold tuning, automatic retries or promotion.']
    for phase in ('probe','final_dev64','final_dev512'):
        report=result.get(phase)
        if not report:continue
        lines.append(f"\n{phase}: windows={report['windows']} transport_mIoU={report['baseline']['mIoU']:.6f}")
        for name,item in report['variants'].items():
            d=item['delta_vs_v18_pp'];m=item['metrics']
            lines.append(f"{name}: mIoU={m['mIoU']:.6f} vs_transport={d['mIoU']:+.6f} MovingMicro={m['MovingMicro']:.6f} "
                f"dMoving={d['MovingMicro']:+.6f} add={item['quality'].get('added',0)} remove={item['quality'].get('removed',0)}")
    speed=result.get('training_speed',{})
    for mode in sorted({t['mode'] for t in speed.get('trials',[])}):
        rows=[t for t in speed['trials'] if t['mode']==mode];seconds=np.mean([t['seconds_per_window'] for t in rows])
        lines.append(f'JOINT_TRAIN {mode}: seconds/window={seconds:.6f}; actual backward+optimizer, motion NOT frozen')
    fps=result.get('fps',{})
    for mode in sorted({t['mode'] for t in fps.get('trials',[])}):
        rows=[t for t in fps['trials'] if t['mode']==mode];seconds=np.mean([t['six_frame_seconds'] for t in rows])
        lines.append(f'SIX_FRAME {mode}: seconds={seconds:.6f} FPS={6/seconds:.3f}')
    if fps:lines+=['FPS boundary: '+fps['boundary'],'FPS exclusions: '+fps['excludes'],
        'Existing Strong prior rebuild is INCLUDED but not yet fully device-native. Shared speed model is not an accuracy result.']
    if 'training' in result:lines.append('MIGRATION '+json.dumps(result['training'],ensure_ascii=False))
    if result.get('reused_identical_contract_phases'):
        lines.append('RESUME reused completed identical-contract phases: '+','.join(result['reused_identical_contract_phases']))
    if 'gate' in result:lines.append('PILOT_GATE '+json.dumps(result['gate'],ensure_ascii=False))
    lines.append('route: '+result.get('route','incomplete_no_recommendation'))
    if 'error' in result:lines.append('error: '+result['error'])
    lines.append('Detailed timings/guards/populations: bundle.json; progress: progress.jsonl; resumable diagnostic: migration_last.pt')
    return '\n'.join(lines)+'\n'


def main(stop_event=None):
    p=argparse.ArgumentParser(description=__doc__);add_config_args(p)
    for key in ('checkpoint','train-cache','dev-cache','population-manifest','base-checkpoint','dataroot','train-info','dev-info','out-dir'):
        p.add_argument('--'+key,required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--cpu-workers',type=int,default=8)
    p.add_argument('--causal-geometry-cache');p.add_argument('--eval-windows',type=int,default=64)
    p.add_argument('--fps-windows',type=int,default=6);p.add_argument('--speed-train-windows',type=int,default=32)
    p.add_argument('--speed-repeats',type=int,default=2);p.add_argument('--seed',type=int,default=20261005)
    p.add_argument('--train-fraction',type=float,default=.2);p.add_argument('--train-epochs',type=int,default=1)
    p.add_argument('--max-updates',type=int,default=0,help='0=one complete configured population pass; explicit smoke bound otherwise')
    p.add_argument('--kd-weight',type=float,default=.25);p.add_argument('--resume',help='ONLY this new migration protocol into a NEW output')
    p.add_argument('--final-dev512',action='store_true',help='optional one final selection-population evaluation, not independent test')
    p.add_argument('--no-train',action='store_true',help='same bundled probes/speed, no migration updates')
    a=p.parse_args();out=Path(a.out_dir)
    if out.exists():p.error('new output required; refusing overwrite')
    for key in ('config','checkpoint','train_cache','dev_cache','population_manifest','base_checkpoint','train_info','dev_info'):
        if not Path(getattr(a,key) or '').is_file():p.error('missing '+key)
    if a.resume and not Path(a.resume).is_file():p.error('missing migration checkpoint')
    if (not Path(a.dataroot).is_dir() or not 0<a.train_fraction<1 or not 1<=a.eval_windows<=64
            or min(a.cpu_workers,a.fps_windows,a.speed_train_windows,a.speed_repeats,a.train_epochs)<1
            or a.fps_windows>a.eval_windows or a.max_updates<0 or a.kd_weight<0):p.error('invalid budgets')
    device=require_cuda(a.device)
    torch.set_num_threads(1);torch.manual_seed(a.seed)
    from real_motion.native_column_cpu import backend_name,prepare_native
    if backend_name()=='native':prepare_native()
    out.mkdir(parents=True);result=dict(status='running',protocol=PROTOCOL,stage_seconds={},actual_cuda=device.type=='cuda')
    begun=time.perf_counter();optimizer=joint=None;cursor=successful=executed=0
    cache=None;contract=None;generator=None
    def persist():
        write_json(out/'bundle.json',result);(out/'summary.txt').write_text(brief(result),encoding='utf-8')
    persist()
    try:
      with (out/'progress.jsonl').open('x',encoding='utf-8') as log:
        def progress(row):log.write(json.dumps(row,ensure_ascii=False)+'\n');log.flush()
        snapshot=out/'teacher_snapshot.pt';digest=snapshot_checkpoint(a.checkpoint,snapshot)
        cfg=load_runtime_config(a.config,a.override);pcfg=make_prepare_config(cfg)
        ck,teacher=load_joint(snapshot,device,reference_sha=CLEAN_SHA256,config_sha=stable_json_fingerprint(cfg),allow_diagnostic=True)
        if teacher.transport.config.history_frames!=4 or ck['model_configs'].get('adaptive_context') is not None:
            raise RuntimeError('only strict-four Local teacher supported')
        # Complete provenance checks before any expensive experiment.
        for key,fingerprint in ((a.train_cache,ck['cache_fingerprints']['train']),(a.dev_cache,ck['cache_fingerprints']['dev']),
                (a.train_info,ck['info_fingerprints']['train']),(a.dev_info,ck['info_fingerprints']['dev']),(a.base_checkpoint,CLEAN_SHA256)):
            if sha256(key)!=fingerprint:raise RuntimeError('teacher/data provenance changed: '+key)
        manifest,dev64,_=load_manifest(a.population_manifest)
        if (manifest['manifest_fingerprint']!=ck['dev_manifest_fingerprint']
                or tuple(map(tuple,manifest['parent_keys']))!=tuple(map(tuple,ck['dev_keys']))):
            raise RuntimeError('frozen dev identity/order changed')
        _,dev_all=load_cache(a.dev_cache);record_keys(dev_all)
        dev=align_records(dev_all,dev64[:a.eval_windows]);dev512=align_records(dev_all,manifest['parent_keys']);del dev_all
        _,train_all=load_cache(a.train_cache);keys=record_keys(train_all)
        if tuple(keys)!=tuple(map(tuple,ck['train_keys'])) or len(keys)!=TRAIN_WINDOWS:raise RuntimeError('full training identity/order changed')
        chosen,_=select_population(keys,{s for s,_ in manifest['parent_keys']},fraction=a.train_fraction,seed=a.seed)
        train=align_records(train_all,chosen);del train_all
        student=SharedEvidenceColumns.from_teacher(teacher.columns).to(device)
        ordered=epoch_records(train,a.seed,a.train_epochs);total=group_count(ordered)
        target=min(total,a.max_updates) if a.max_updates else total
        repo=Path(__file__).resolve().parents[2]
        implementation=stable_json_fingerprint({name:sha256(repo/name) for name in (
            'real_motion/shared_column_evidence.py','real_motion/column_device_geometry.py',
            'real_motion/sparse_column_readout.py','tools/real_motion/shared_evidence_pilot_common.py',
            'tools/real_motion/shared_evidence_recovery.py')})
        contract=dict(protocol=PROTOCOL,teacher_sha256=digest,implementation_fingerprint=implementation,
            shared_model=student.contract(),runtime_config_fingerprint=stable_json_fingerprint(cfg),
            train_keys=chosen,train_fraction=a.train_fraction,train_epochs=a.train_epochs,ordered_key_fingerprint=stable_json_fingerprint(record_keys(ordered)) if a.train_epochs==1 else stable_json_fingerprint([(r['scene_name'],r['t0_token']) for r in ordered]),
            eval_keys=dev64[:a.eval_windows],dev512_keys=manifest['parent_keys'],thresholds=GATES,
            diagnostic_budgets=dict(fps_windows=a.fps_windows,speed_train_windows=a.speed_train_windows,
                speed_repeats=a.speed_repeats,final_dev512=a.final_dev512),torch_version=torch.__version__,
            seed=a.seed,kd_weight=a.kd_weight,window_batch=4,source_budget=128,schedule_steps=target,
            schedule='whole_pilot_cosine_no_tail',transport='epoch19_frozen_migration_NOT_scratch_joint',
            sampler='torch_generator_GPU_six_strata_importance_NOT_old_numpy_RNG')
        write_json(out/'contract.json',contract)
        saved=None;reused={}
        if a.resume:
            saved=torch.load(a.resume,map_location='cpu',weights_only=False)
            validate_migration(saved,contract)  # Fail BEFORE expensive probes.
            parent=Path(a.resume).resolve().parent
            if (parent/'contract.json').is_file() and (parent/'bundle.json').is_file():
                prior_contract=json.loads((parent/'contract.json').read_text(encoding='utf-8'))
                prior_result=json.loads((parent/'bundle.json').read_text(encoding='utf-8'))
                if stable_json_fingerprint(prior_contract)==stable_json_fingerprint(contract):
                    reused={k:prior_result[k] for k in ('probe','training_speed') if k in prior_result}
                    if 'fps_initial' in prior_result:reused['fps']=prior_result['fps_initial']
                    elif 'fps' in prior_result:reused['fps']=prior_result['fps']
        result.update(teacher_sha256=digest,teacher_snapshot=str(snapshot),teacher_epoch=ck['cursor_epoch'],
            migration=student.migration_audit,training_population=len(train),dev_selection_population=len(dev),
            history_frames=4,future_frames=6,thresholds=GATES,reused_identical_contract_phases=list(reused))
        result.update(reused)
        provider=PilotProvider(a.base_checkpoint,CLEAN_SHA256,pcfg,device,a.cpu_workers,teacher,None)
        provider.raw_prefetch_workers=provider.raw_prefetch_depth=min(2,a.cpu_workers)
        if a.causal_geometry_cache:
            namespace=geometry_namespace(cfg,provider,ck['info_fingerprints'],ck['cache_fingerprints'],a.dataroot)
            cache=CausalGeometryCache(a.causal_geometry_cache,namespace,max_bytes=0,ram_bytes=256*2**20)
            provider.causal_geometry_cache=cache
            result['cache_policy']='reuse existing valid causal entries; 256MiB RAM; ZERO new disk admissions'
        dev_source=CachedColumnSource(NuScenesWindowSource(a.dataroot,info_pkl=a.dev_info,verbose=False),256)
        train_source=CachedColumnSource(NuScenesWindowSource(a.dataroot,info_pkl=a.train_info,verbose=False),256)
        print(f'SHARED BUNDLE: teacher epoch={ck["cursor_epoch"]}, TRAIN={len(train)}, dev={len(dev)}, migration batches={target}; old experiment unchanged',flush=True)
        if 'probe' not in reused:
            tick=time.perf_counter()
            result['probe']=evaluate(provider,dev_source,dev,teacher,student,probe=True,audit=True,progress=progress,stop_event=stop_event)
            result['stage_seconds']['probe_and_device_guards']=time.perf_counter()-tick;persist()
        if 'training_speed' not in reused:
            tick=time.perf_counter()
            speed_rows=list(prefetch_raw_columns(provider,train_source,train[:a.speed_train_windows]))
            result['stage_seconds']['TRAIN_speed_fixed_preparation']=time.perf_counter()-tick
            tick=time.perf_counter();result['training_speed']=training_speed(teacher,student,provider,speed_rows,repeats=a.speed_repeats,seed=a.seed)
            result['stage_seconds']['joint_training_speed']=time.perf_counter()-tick;del speed_rows;persist()
        if 'fps' not in reused:
            tick=time.perf_counter();result['fps']=fps_speed(provider,dev_source,dev[:a.fps_windows],teacher,student,
                repeats=a.speed_repeats,stop_event=stop_event)
            result['stage_seconds']['six_frame_FPS']=time.perf_counter()-tick;persist()
        elif not getattr(provider,'columns_checked',False):
            # Reused timing is not a substitute for this process's original
            # renderer preflight. One fresh read-only check, no training RNG.
            record=dev[0];raw=provider.load_raw_columns(dev_source,record,include_gt=False)
            with torch.inference_mode():provider.prepare_columns(dev_source,record,include_gt=False,raw_window=raw)
        if a.no_train:
            result.update(status='complete',route='probes_speed_only_no_migration_no_promotion');persist();return 0
        joint=copy.deepcopy(teacher);joint.columns=student
        optimizer=make_optimizer(joint,frozen=True)
        for param in teacher.parameters():param.requires_grad_(False)
        teacher.eval();generator=torch.Generator(device=device).manual_seed(a.seed+1)
        if a.resume:
            cursor,successful,executed=restore_migration(saved,student,optimizer,generator,contract)
        def save_migration(role):
            atomic_checkpoint(out/'migration_last.pt',migration_payload(student,optimizer,generator,contract,
                role=role,cursor=cursor,successful=successful,executed=executed))
        tick=time.perf_counter()
        # Skip consumed groups before loading their data. Source/window order,
        # GPU sampling RNG, optimizer and whole pilot cosine restore together.
        offset=0;group_cursor=0
        while offset<len(ordered) and group_cursor<cursor:
            end=offset;sources=0
            while end<len(ordered):
                n=len(ordered[end]['features'])
                if end>offset and (end-offset>=4 or sources+n>128):break
                sources+=n;end+=1
            offset=end;group_cursor+=1
        for rows in prefetch_column_batches(provider,train_source,ordered[offset:],4,128,io_workers=2):
            if cursor>=target or stop_event is not None and stop_event.is_set():break
            scale=.1+.9*.5*(1+math.cos(math.pi*cursor/max(target,1)))
            for group in optimizer.param_groups:group['lr']=group['initial_lr']*scale
            step=time.perf_counter()
            stat=training_step(joint,optimizer,provider,rows,generator,frozen=True,teacher=teacher.columns,kd_weight=a.kd_weight,
                profile=cursor%32==0)
            cursor+=1;successful+=stat['optimizer_updated'];executed+=len(rows)
            progress(dict(event='shared_migration',cursor=cursor,target=target,seconds=time.perf_counter()-step,**stat))
            if cursor==1 or cursor%32==0:print(f'SHARED_MIGRATION {cursor}/{target} loss={stat["loss"]:.6f} GT={stat["column_loss"]:.6f} KD={stat["kd_loss"]:.6f}',flush=True)
            if cursor%32==0:save_migration('periodic_diagnostic')
        save_migration('complete_diagnostic' if cursor>=target else 'stopped_diagnostic')
        if any(not torch.equal(v,teacher.transport.state_dict()[k]) for k,v in joint.transport.state_dict().items()):
            raise RuntimeError('frozen migration changed teacher transport parameters')
        result['stage_seconds']['migration']=time.perf_counter()-tick
        result['training']=dict(cursor=cursor,target=target,successful_updates=successful,executed_windows=executed,
            checkpoint=str(out/'migration_last.pt'),transport_frozen=True,full_joint_speed_measured_separately=True)
        if cursor<target:
            result.update(status='stopped',route='resume_identical_migration_contract_only');persist();return 130
        tick=time.perf_counter()
        result['final_dev64']=evaluate(provider,dev_source,dev,teacher,student,progress=progress,stop_event=stop_event)
        if a.final_dev512:result['final_dev512']=evaluate(provider,dev_source,dev512,teacher,student,audit=True,progress=progress,stop_event=stop_event)
        result['stage_seconds']['final_evaluation']=time.perf_counter()-tick
        # Same invocation also retimes the TRAINED student. Nonzero learned
        # actions can change composition cost; initial-weight FPS is not enough.
        result['fps_initial']=result['fps']
        tick=time.perf_counter()
        result['fps']=fps_speed(provider,dev_source,dev[:a.fps_windows],teacher,student,
            repeats=a.speed_repeats,stop_event=stop_event)
        result['fps']['note']='trained shared student; same prepared-input boundary and old teacher integer guard'
        result['stage_seconds']['trained_six_frame_FPS']=time.perf_counter()-tick
        final=result.get('final_dev512',result['final_dev64']);old=final['variants']['teacher_joint']['metrics'];new=final['variants']['student_joint']['metrics']
        moving_safe=(new['MovingMicro']>=old['MovingMicro'] and all(new['per_horizon'][h]['MovingMicro']>=old['per_horizon'][h]['MovingMicro'] for h in ('1.0','2.0','3.0')))
        speeds=result['training_speed']['trials'];avg=lambda mode:np.mean([r['seconds_per_window'] for r in speeds if r['mode']==mode])
        fps=result['fps']['trials'];latency=np.mean([r['six_frame_seconds'] for r in fps if r['mode']=='shared_auto'])
        gate=dict(mIoU_not_below_teacher=new['mIoU']>=old['mIoU'],aggregate_and_all_horizon_moving_not_below_teacher=moving_safe,
            actual_joint_training_faster=bool(avg('shared_auto_joint')<avg('current_joint')),six_frame_latency_le_250ms=bool(latency<=.25),
            latency_scope='trained_student_same_bundle_prepared_input_six_dense_frames')
        gate['pass']=all(gate[k] for k in ('mIoU_not_below_teacher',
            'aggregate_and_all_horizon_moving_not_below_teacher','actual_joint_training_faster','six_frame_latency_le_250ms'))
        result.update(status='complete',gate=gate,route='pilot_pass_diagnostic_only_no_automatic_expansion' if gate['pass'] else 'pilot_not_passed_no_automatic_retry')
        if sha256(snapshot)!=digest:raise RuntimeError('immutable teacher snapshot changed')
        persist();print(brief(result),flush=True);return 0
    except BaseException as error:
        result.update(status='interrupted' if isinstance(error,(KeyboardInterrupt,InterruptedError)) else 'failed',
            error=type(error).__name__+': '+str(error),route='preserve_old_experiment_no_automatic_retry')
        persist();raise
    finally:
        result['elapsed_seconds']=time.perf_counter()-begun
        if cache is not None:result['cache_stats']=cache.stats();cache.close()
        persist()


if __name__=='__main__':
    import signal
    from threading import Event
    stopped=Event()
    def stop(signum,frame):stopped.set()
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    sys.exit(main(stopped))
