#!/usr/bin/env python3
"""ONE real-data experiment: support/decomposition, 20% migration, FPS, dev64.

Epoch19, old optimizer and geometry artifacts are read-only. This is NOT
converged scratch joint training, automatic promotion or threshold search.
"""
import sys
from pathlib import Path
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse
import json
import math
import signal
import threading
import time
import numpy as np
import torch
from real_motion.source_repair_evidence import PROTOCOL
from real_motion.sparse_evidence_repair import SparseRepairHead
from real_motion.runtime_config import add_config_args,load_runtime_config,make_prepare_config
from real_motion.column_runtime_pipeline import CachedColumnSource,prefetch_raw_columns
from real_motion.nuscenes_adapter import NuScenesWindowSource
from real_motion.causal_geometry_cache import CausalGeometryCache
from real_motion.v21_source_induction import stable_json_fingerprint
from tools.real_motion.run_p0_f9_shared_evidence_pilot import PilotProvider,epoch_records,group_count,require_cuda
from tools.real_motion.local_warm_cache_common import geometry_namespace
from tools.real_motion.train_p0_f9_joint_causal_columns import load_joint
from tools.real_motion.eval_p0_f9_v18_se2 import load_cache
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import load_manifest,align_records,sha256
from tools.real_motion.train_p0_f9_causal_columns import record_keys
from tools.real_motion.v18_source_interaction_common import select_population
from tools.real_motion.joint_training_recovery import snapshot_checkpoint
from tools.real_motion.static_evidence_selector_common import CLEAN_SHA256,write_json,atomic_checkpoint
from tools.real_motion.joint_column_full_common import prefetch_column_batches
from tools.real_motion.source_repair_pilot_common import evaluate,training_step,six_frame_speed,joint_training_speed
from tools.real_motion.source_repair_recovery import payload,restore

TRAIN_WINDOWS=20430


def brief(result):
    lines=['===== SPARSE SOURCE REPAIR REAL-DATA PILOT =====',f"status: {result['status']}",f'protocol: {PROTOCOL}',
        'Four histories -> six futures; frozen epoch19 transport; ADD-only sparse refine; unchanged generation.',
        'Fixed thresholds: old GEN=.5/ADD=.5/REMOVE=.95; new ADD=.5. No threshold tuning or deployment.']
    for phase in ('initial_audit','final_dev64','final_dev512'):
        report=result.get(phase)
        if not report:continue
        lines.append(f"\n{phase}: windows={report['windows']} transport_mIoU={report['baseline']['mIoU']:.6f}")
        for name,row in report['variants'].items():
            m=row['metrics'];d=row['delta_vs_v18_pp'];q=row['quality']
            lines.append(f"{name}: mIoU={m['mIoU']:.6f} dMiOU={d['mIoU']:+.6f} MovingMicro={m['MovingMicro']:.6f} "
                f"dMoving={d['MovingMicro']:+.6f} added={q.get('added',0)} removed={q.get('removed',0)} precision={q['addition_semantic_precision']}")
        lines.append('TEACHER_CORRECT_ADD_COVERAGE '+json.dumps(report['teacher_ADD_domain_coverage']))
    if 'training' in result:lines.append('MIGRATION '+json.dumps(result['training']))
    for trial in result.get('joint_training_speed',{}).get('trials',[]):
        lines.append(f"JOINT_TRAIN_RUNTIME {trial['mode']} repeat={trial['repeat']} seconds/window={trial['seconds_per_window']:.6f}")
    if result.get('joint_training_speed'):lines.append('JOINT_TRAIN_SCOPE '+result['joint_training_speed']['warning'])
    speed=result.get('six_frame_speed',{})
    for mode in ('old_refine','sparse_refine','old_joint','sparse_joint'):
        rows=[r for r in speed.get('trials',[]) if r['mode']==mode]
        if rows:
            sec=np.mean([r['six_frame_seconds'] for r in rows])
            lines.append(f'SIX_FRAME {mode}: seconds={sec:.6f} FPS={6/sec:.3f}')
    if speed:lines+=['FPS boundary: '+speed['boundary'],'FPS exclusions: '+speed['exclusions']]
    if 'gate' in result:lines.append('PILOT_GATE '+json.dumps(result['gate']))
    lines+=['route: '+result.get('route','incomplete'),
        'Migration time is frozen-motion + selected teacher KD + GT + sparse backward, NOT a full-joint training speedup.',
        'Old REMOVE is NOT represented by the ADD-only student. Domain/quality losses are reported, not hidden.']
    if 'error' in result:lines.append('error: '+result['error'])
    return '\n'.join(lines)+'\n'


def main(stop_event=None):
    p=argparse.ArgumentParser(description=__doc__);add_config_args(p)
    for key in ('checkpoint','train-cache','dev-cache','population-manifest','base-checkpoint','dataroot','train-info','dev-info','out-dir'):
        p.add_argument('--'+key,required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--cpu-workers',type=int,default=8)
    p.add_argument('--causal-geometry-cache');p.add_argument('--eval-windows',type=int,default=64)
    p.add_argument('--fps-windows',type=int,default=6);p.add_argument('--speed-repeats',type=int,default=2)
    p.add_argument('--speed-train-windows',type=int,default=16)
    p.add_argument('--train-fraction',type=float,default=.2);p.add_argument('--seed',type=int,default=20261005)
    p.add_argument('--max-updates',type=int,default=0);p.add_argument('--kd-weight',type=float,default=.25)
    p.add_argument('--resume');p.add_argument('--final-dev512',action='store_true')
    a=p.parse_args();out=Path(a.out_dir)
    if out.exists():p.error('new output required; refusing overwrite')
    for key in ('config','checkpoint','train_cache','dev_cache','population_manifest','base_checkpoint','train_info','dev_info'):
        if not Path(getattr(a,key) or '').is_file():p.error('missing '+key)
    if a.resume and not Path(a.resume).is_file():p.error('missing sparse migration checkpoint')
    if (not Path(a.dataroot).is_dir() or min(a.cpu_workers,a.fps_windows,a.speed_repeats,a.speed_train_windows)<1
            or not 1<=a.eval_windows<=64 or a.fps_windows>a.eval_windows or not 0<a.train_fraction<1
            or a.max_updates<0 or not np.isfinite(a.kd_weight) or a.kd_weight<0):p.error('invalid budgets')
    device=require_cuda(a.device);torch.set_num_threads(1);torch.manual_seed(a.seed)
    from real_motion.native_column_cpu import backend_name,prepare_native
    if backend_name()=='native':prepare_native()
    out.mkdir(parents=True);result=dict(status='running',protocol=PROTOCOL,stage_seconds={});cache=None
    begun=time.perf_counter();cursor=successful=executed=0;head=optimizer=contract=rng=None
    def persist():
        write_json(out/'bundle.json',result);(out/'summary.txt').write_text(brief(result),encoding='utf-8')
    def save():
        if head is not None and optimizer is not None and contract is not None:
            atomic_checkpoint(out/'migration_last.pt',payload(head,optimizer,rng,contract,cursor=cursor,successful=successful,executed=executed))
    persist()
    try:
      with (out/'progress.jsonl').open('x',encoding='utf-8') as log:
        def progress(row):log.write(json.dumps(row,ensure_ascii=False)+'\n');log.flush()
        snapshot=out/'teacher_snapshot.pt';digest=snapshot_checkpoint(a.checkpoint,snapshot)
        cfg=load_runtime_config(a.config,a.override);pcfg=make_prepare_config(cfg)
        ck,teacher=load_joint(snapshot,device,reference_sha=CLEAN_SHA256,config_sha=stable_json_fingerprint(cfg),allow_diagnostic=True)
        if teacher.transport.config.history_frames!=4 or ck['model_configs'].get('adaptive_context') is not None:
            raise RuntimeError('strict-four Local teacher required')
        if ck['cursor_epoch']!=19:raise RuntimeError('this fixed pilot expects selected epoch19 teacher')
        for path,expected in ((a.train_cache,ck['cache_fingerprints']['train']),(a.dev_cache,ck['cache_fingerprints']['dev']),
                (a.train_info,ck['info_fingerprints']['train']),(a.dev_info,ck['info_fingerprints']['dev']),(a.base_checkpoint,CLEAN_SHA256)):
            if sha256(path)!=expected:raise RuntimeError('teacher/data provenance mismatch: '+path)
        manifest,keys64,_=load_manifest(a.population_manifest)
        if (manifest['manifest_fingerprint']!=ck['dev_manifest_fingerprint']
                or tuple(map(tuple,manifest['parent_keys']))!=tuple(map(tuple,ck['dev_keys']))):
            raise RuntimeError('frozen dev population changed')
        _,all_dev=load_cache(a.dev_cache);record_keys(all_dev)
        dev=align_records(all_dev,keys64[:a.eval_windows]);dev512=align_records(all_dev,manifest['parent_keys']);del all_dev
        _,all_train=load_cache(a.train_cache);keys=record_keys(all_train)
        if len(keys)!=TRAIN_WINDOWS or tuple(keys)!=tuple(map(tuple,ck['train_keys'])):raise RuntimeError('full TRAIN identity/order changed')
        chosen,_=select_population(keys,{s for s,_ in manifest['parent_keys']},fraction=a.train_fraction,seed=a.seed)
        train=align_records(all_train,chosen);del all_train
        ordered=epoch_records(train,a.seed,1);total=group_count(ordered);target=min(total,a.max_updates) if a.max_updates else total
        head=SparseRepairHead('local_consensus',source_dim=teacher.columns.source_dim).to(device)
        contract=dict(protocol=PROTOCOL,teacher_sha256=digest,config_fingerprint=stable_json_fingerprint(cfg),train_keys=chosen,
            dev_keys=keys64[:a.eval_windows],seed=a.seed,train_fraction=a.train_fraction,schedule_steps=target,kd_weight=a.kd_weight,
            final_dev512=a.final_dev512,fps_windows=a.fps_windows,speed_repeats=a.speed_repeats,
            speed_train_windows=a.speed_train_windows,
            static_classes=[11,13],halo='six face neighbors',head='local_consensus32_cached_future',
            source_dim=head.source_dim,
            optimizer='AdamW_3e-4_0.01',schedule='whole_pilot_cosine_0.1_floor',sampler='static_dynamic_strata_importance_v1',
            torch_version=torch.__version__,window_batch=4,source_budget=128,
            implementation=stable_json_fingerprint({name:sha256(Path(__file__).resolve().parents[2]/name) for name in (
                'real_motion/sparse_evidence_repair.py','real_motion/source_repair_evidence.py',
                'tools/real_motion/source_repair_pilot_common.py','tools/real_motion/source_repair_recovery.py',
                'tools/real_motion/run_p0_f9_source_repair_pilot.py')}))
        write_json(out/'contract.json',contract)
        optimizer=torch.optim.AdamW(head.parameters(),lr=3e-4,weight_decay=.01);rng=np.random.default_rng(a.seed+1)
        saved=None;previous={}
        if a.resume:
            saved=torch.load(a.resume,map_location='cpu',weights_only=False)
            cursor,successful,executed=restore(saved,head,optimizer,rng,contract)
            parent=Path(a.resume).resolve().parent
            if (parent/'contract.json').is_file() and (parent/'bundle.json').is_file():
                old_contract=json.loads((parent/'contract.json').read_text(encoding='utf-8'))
                if stable_json_fingerprint(old_contract)==stable_json_fingerprint(contract):previous=json.loads((parent/'bundle.json').read_text(encoding='utf-8'))
            print(f'SPARSE_RESTORED completed_update={cursor}/{target}; optimizer/RNG/order/cosine preserved',flush=True)
        teacher.eval()
        for parameter in teacher.parameters():parameter.requires_grad_(False)
        frozen_teacher_state={k:v.detach().cpu().clone() for k,v in teacher.state_dict().items()}
        provider=PilotProvider(a.base_checkpoint,CLEAN_SHA256,pcfg,device,a.cpu_workers,teacher,None)
        provider.raw_prefetch_workers=provider.raw_prefetch_depth=min(2,a.cpu_workers)
        if a.causal_geometry_cache:
            namespace=geometry_namespace(cfg,provider,ck['info_fingerprints'],ck['cache_fingerprints'],a.dataroot)
            cache=CausalGeometryCache(a.causal_geometry_cache,namespace,max_bytes=0,ram_bytes=256*2**20)
            provider.causal_geometry_cache=cache
        dev_source=CachedColumnSource(NuScenesWindowSource(a.dataroot,info_pkl=a.dev_info,verbose=False),256)
        train_source=CachedColumnSource(NuScenesWindowSource(a.dataroot,info_pkl=a.train_info,verbose=False),256)
        result.update(teacher_snapshot=str(snapshot),teacher_sha256=digest,train_windows=len(train),updates=target,dev_windows=len(dev),
            cache_policy='existing fixed causal geometry only; zero NEW disk writes; 256MiB geometry RAM',actual_cuda=device.type=='cuda')
        print(f'SPARSE BUNDLE: TRAIN={len(train)} one pass, updates={target}; dev={len(dev)}; epoch19 READ ONLY',flush=True)
        if 'initial_audit' in previous:result['initial_audit']=previous['initial_audit']
        else:
            result['initial_audit']=evaluate(provider,dev_source,dev,teacher,head,progress=progress,stop_event=stop_event);persist()
        if 'joint_training_speed' in previous:result['joint_training_speed']=previous['joint_training_speed']
        else:
            speed_rows=list(prefetch_raw_columns(provider,train_source,train[:a.speed_train_windows]))
            result['joint_training_speed']=joint_training_speed(provider,teacher,head,speed_rows,repeats=a.speed_repeats,seed=a.seed)
            del speed_rows;persist()
        # Restore RNG once more after read-only diagnostics. Evaluation must NOT
        # alter the resumed training random stream, even if Torch internals do.
        if saved is not None:cursor,successful,executed=restore(saved,head,optimizer,rng,contract)
        offset=0
        for _ in range(cursor):
            end=offset;sources=0
            while end<len(ordered):
                n=len(ordered[end]['features'])
                if end>offset and (end-offset>=4 or sources+n>128):break
                sources+=n;end+=1
            offset=end
        tick=time.perf_counter();train_seconds=0.
        for rows in prefetch_column_batches(provider,train_source,ordered[offset:],4,128,io_workers=2):
            if cursor>=target or stop_event is not None and stop_event.is_set():break
            optimizer.param_groups[0]['lr']=3e-4*(.1+.9*.5*(1+math.cos(math.pi*cursor/target)))
            stat=training_step(provider,rows,teacher,head,optimizer,rng,kd_weight=a.kd_weight)
            cursor+=1;successful+=int(stat['optimizer_updated']);executed+=len(rows);train_seconds+=stat['seconds']
            stat.update(cursor=cursor,target=target,event='sparse_migration',
                allocated_after_mib=torch.cuda.memory_allocated(device)/2**20 if device.type=='cuda' else 0.,
                reserved_after_mib=torch.cuda.memory_reserved(device)/2**20 if device.type=='cuda' else 0.)
            progress(stat)
            if cursor==1 or cursor%32==0:
                print(f'SPARSE_MIGRATION {cursor}/{target} loss={stat["loss"]:.6f} GT={stat["GT_BCE"]:.6f} KD={stat["KD_BCE"]:.6f} allocated_after={stat["allocated_after_mib"]:.1f}MiB',flush=True)
                save()
        save();result['stage_seconds']['migration_this_invocation']=time.perf_counter()-tick
        result['training']=dict(cursor=cursor,target=target,successful_updates=successful,executed_windows=executed,
            seconds_training_step_this_invocation=train_seconds,checkpoint=str(out/'migration_last.pt'),transport_frozen=True)
        if cursor<target:
            result.update(status='stopped',route='resume_identical_sparse_migration_only');persist();return 130
        result['final_dev64']=evaluate(provider,dev_source,dev,teacher,head,progress=progress,stop_event=stop_event);persist()
        if a.final_dev512:result['final_dev512']=evaluate(provider,dev_source,dev512,teacher,head,progress=progress,stop_event=stop_event);persist()
        result['six_frame_speed']=six_frame_speed(provider,dev_source,dev[:a.fps_windows],teacher,head,repeats=a.speed_repeats,stop_event=stop_event)
        final=result.get('final_dev512',result['final_dev64']);old=final['variants']['teacher_joint']['metrics'];new=final['variants']['student_joint']['metrics']
        latency=np.mean([t['six_frame_seconds'] for t in result['six_frame_speed']['trials'] if t['mode']=='sparse_joint'])
        gate=dict(mIoU_not_below_teacher=new['mIoU']>=old['mIoU'],
            MovingMicro_not_below_teacher=new['MovingMicro']>=old['MovingMicro'] and all(
                new['per_horizon'][h]['MovingMicro']>=old['per_horizon'][h]['MovingMicro'] for h in ('1.0','2.0','3.0')),
            six_frame_joint_latency_le_150ms=bool(latency<=.15))
        gate['pass']=all(gate.values());result.update(status='complete',gate=gate,
            route='candidate_for_separate_joint_training_NOT_promoted' if gate['pass'] else 'pilot_not_passed_no_automatic_retry')
        if any(not torch.equal(v.detach().cpu(),frozen_teacher_state[k]) for k,v in teacher.state_dict().items()):
            raise RuntimeError('read-only teacher parameters/buffers changed')
        result['teacher_parameters_and_buffers_unchanged']=True
        if sha256(snapshot)!=digest:raise RuntimeError('immutable teacher snapshot changed')
        print(brief(result),flush=True);persist();return 0
    except BaseException as error:
        # Errors during a step recover only the preceding COMPLETED periodic
        # checkpoint; never save half-updated optimizer/RNG as a valid boundary.
        result.update(status='interrupted' if isinstance(error,(KeyboardInterrupt,InterruptedError)) else 'failed',
            error=type(error).__name__+': '+str(error),route='old_experiments_preserved_no_automatic_retry');persist();raise
    finally:
        result['elapsed_seconds_this_invocation']=time.perf_counter()-begun
        if cache is not None:result['geometry_cache']=cache.stats();cache.close()
        persist()


if __name__=='__main__':
    stopped=threading.Event()
    def request_stop(signum,frame):
        stopped.set();print('Stop requested: save after COMPLETED migration update; evaluation stops at window boundary.',flush=True)
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,request_stop)
    sys.exit(main(stopped))
