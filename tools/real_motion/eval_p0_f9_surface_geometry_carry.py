#!/usr/bin/env python3
"""Frozen causal geometry-carry ablations: TRAIN64 + dev64, no retraining."""
import sys
from pathlib import Path
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import hashlib
import json
import signal
import threading
import time

import numpy as np
import torch

from tools.real_motion import eval_p0_f9_joint_surface_long_rollout as old
from tools.real_motion import surface_rollout_geometry_carry as carry
from real_motion.v21_source_induction import select_scene_balanced_round_robin
from real_motion.strong_majority_execution import ParallelNativeMajority, strong_majority_execution

PROTOCOL = 'p0_f9_surface_frozen_geometry_carry_screen_v1'
TRAIN_WINDOWS = 20430
FILES = (*old.FILES, 'tools/real_motion/eval_p0_f9_surface_geometry_carry.py',
         'tools/real_motion/surface_rollout_geometry_carry.py',
         'real_motion/strong_majority_execution.py', 'real_motion/v18_execution_trial.py',
         'real_motion/surface_projection_execution.py')


def implementation(root):
    return old.stable_json_fingerprint({n:old.sha256(root/n) for n in FILES})


def validate_recipe(recipe, bundle_digest, code_digest):
    value=dict(recipe); fingerprint=value.pop('result_fingerprint',None)
    if fingerprint != old.stable_json_fingerprint(value):
        raise RuntimeError('carry recipe content changed')
    selected=value.get('selected_train_route')
    if (value.get('protocol')!=PROTOCOL or value.get('status')!='complete'
            or value.get('bundle_fingerprint')!=bundle_digest or value.get('implementation')!=code_digest
            or selected not in carry.ROUTES[1:]
            or selected!=choose_train_candidate(value['reports']['train64'])):
        raise RuntimeError('completed matching TRAIN-eligible carry recipe required; no DEV route selection')
    return selected


def choose_train_candidate(reports):
    """Frozen rule; accepts TRAIN reports ONLY. DEV never selects a method."""
    base = reports['baseline']['metrics']; eligible = []
    for name in carry.ROUTES[1:]:
        row = reports[name]['metrics']; b = base['average_4s_5s_6s']; v = row['average_4s_5s_6s']
        values = [v[k] for k in ('mIoU','IoU','MovingMicro')] + [b[k] for k in ('mIoU','IoU','MovingMicro')]
        if any(x is None or not np.isfinite(x) for x in values): continue
        if (v['mIoU'] >= b['mIoU']+.05 and v['IoU'] >= b['IoU'] and v['MovingMicro'] >= b['MovingMicro']
                and all(row['per_horizon'][str(h)]['mIoU'] >= base['per_horizon'][str(h)]['mIoU'] for h in (4.,5.,6.))):
            eligible.append(name)
    return max(eligible, key=lambda n: reports[n]['metrics']['average_4s_5s_6s']['mIoU']) if eligible else None


def zero_counts(): return {k:v.tolist() for k,v in old.rollout.legacy._new_raw().items()}


def restore(saved, contract, jobs):
    state = old.new_state(contract['routes']) if saved is None else old.restore_state(saved, contract, len(jobs))
    if saved is None:
        state.update(scene_counts={}, candidate_audit={}, comparison_quality={})
    for name, scenes in state['scene_counts'].items():
        if name not in contract['routes']: raise RuntimeError('saved scene route changed')
        total = {k:np.zeros_like(v) for k,v in old.rollout.legacy._new_raw().items()}
        for row in scenes.values():
            _, restored = old.rollout.validate_resume_state(dict(contract_fingerprint=old.stable_json_fingerprint(contract),
                completed_windows=state['completed_windows'], first_block_exactness_passed=state['first_block_exactness_passed'],
                raw_counts=row),contract,len(jobs))
            for k in total: total[k] += restored[k]
        if any(not np.array_equal(total[k], state['counts'][name][k]) for k in total):
            raise RuntimeError('saved dataset/scene counts disagree')
    for name, row in state['counts'].items():
        if name not in state['scene_counts'] and any(np.any(v) for v in row.values()):
            raise RuntimeError('missing saved scene counts')
    for rows in (state['candidate_audit'],state['comparison_quality']):
        if any(type(v) is not int or v < 0 for row in rows.values() for v in row.values()):
            raise RuntimeError('invalid saved candidate audit')
    return state


@torch.no_grad()
def evaluate(provider, sources, caches, jobs, execution, contract, *, saved=None, save=None,
             progress=None, stop_event=None, checkpoint_every=8):
    """First block + model-dependent second preparation shared, labels last."""
    state = restore(saved, contract, jobs); checked = set()
    counts = {n:{k:np.asarray(v,np.int64) for k,v in row.items()} for n,row in state['counts'].items()}
    def persist():
        state['counts'] = {n:{k:v.tolist() for k,v in row.items()} for n,row in counts.items()}
        state['contract_fingerprint'] = old.stable_json_fingerprint(contract)
        result = dict(state); result['fingerprint'] = old.stable_json_fingerprint(result)
        if save: save(result)
        return result
    persist()
    # Ordered per-split prefetch avoids accidentally looking up TRAIN keys in a
    # VAL cache. At most four raw windows, no learned/future outputs persisted.
    try:
        while state['completed_windows'] < len(jobs):
            start = state['completed_windows']; split = jobs[start][0]
            stop = next((i for i in range(start,len(jobs)) if jobs[i][0] != split),len(jobs))
            subset = jobs[start:stop]; source = sources[split]
            provider.rollout_val_cache = caches.get(split)
            lookup = {(str(w.scene_name),str(w.t0_token)):w for _,w,_ in subset}
            iterator = old.prefetch_raw_columns(provider,source,[r for _,_,r in subset],include_gt=False)
            try:
                for record, raw in iterator:
                    if stop_event is not None and stop_event.is_set(): raise InterruptedError('stopped before window')
                    tick = time.perf_counter(); stages = {}; w = lookup[(str(record['scene_name']),str(record['t0_token']))]
                    t=time.perf_counter(); first=provider.prepare_columns(source,record,include_gt=False,raw_window=raw)
                    pred1,first_edits,_,prob=execution.predict(first); stages['first_block']=time.perf_counter()-t
                    if split not in checked:
                        old.surface.verify_first_block(provider,first.state['rec'],first,pred1,prob,execution)
                        state['first_block_exactness_passed']=True
                    del prob
                    poses=[source.pose(token) for token in w.future_tokens]
                    handoff=old.handoff_from_prepared(first,pred1[-1],dt_s=provider.pcfg.frame_dt_s,
                        max_speed_mps=provider.strong.max_match_speed_mps)
                    t=time.perf_counter(); second=old.surface.synthetic_preparation(pred1,raw,poses,w,provider,handoff=handoff)
                    stages['second_shared_prepare']=time.perf_counter()-t
                    t=time.perf_counter(); candidates,audit=carry.candidates(first,pred1,second,poses[6:],provider)
                    stages['carry_prepare']=time.perf_counter()-t
                    prediction={}; edit_rows={}
                    for route in contract['candidate_routes']:
                        t=time.perf_counter(); dense,edits,_,scores=execution.predict(candidates[route])
                        if split not in checked: execution.verify(candidates[route],dense,scores)
                        stages[route]=time.perf_counter()-t;prediction[route]=pred1+dense;edit_rows[route]=edits
                        del scores
                    state['second_block_exactness_passed']=True;checked.add(split)
                    # GT occupancy + original-t0 GT moving support only NOW,
                    # after all candidate predictions. Nothing below feeds back.
                    t=time.perf_counter(); tokens=tuple(w.future_tokens[i] for i in (1,3,5,7,9,11))
                    moving=old.gt_moving_support_sequence(source.nusc,w.t0_token,tokens,old.rollout.REPORT_HORIZONS,
                        grid=provider.pcfg.grid,workers=provider.workers)
                    for hi,(idx,token) in enumerate(zip((1,3,5,7,9,11),tokens)):
                        gt=source.load_semantics(w.scene_name,token)
                        for route in contract['candidate_routes']:
                            name=split+':'+route
                            old.rollout.update_metrics(counts[name],hi,prediction[route][idx],gt,moving[hi][0],17)
                            scenes=state['scene_counts'].setdefault(name,{})
                            row={k:np.asarray(v,np.int64) for k,v in scenes.get(w.scene_name,zero_counts()).items()}
                            old.rollout.update_metrics(row,hi,prediction[route][idx],gt,moving[hi][0],17)
                            scenes[w.scene_name]={k:v.tolist() for k,v in row.items()}
                            if hi >= 3:
                                before=prediction['baseline'][idx]; after=prediction[route][idx]
                                quality=state['comparison_quality'].setdefault(name,{})
                                changes=dict(changed=int((before!=after).sum()),
                                    corrected=int(((before!=gt)&(after==gt)).sum()),
                                    damaged=int(((before==gt)&(after!=gt)).sum()),
                                    added=int(((before==17)&(after!=17)).sum()),
                                    removed=int(((before!=17)&(after==17)).sum()))
                                for k,v in changes.items():quality[k]=quality.get(k,0)+v
                    stages['metrics']=time.perf_counter()-t;stages['window_total']=time.perf_counter()-tick
                    for k,v in stages.items():state['stage_seconds'][split+'/'+k]=state['stage_seconds'].get(split+'/'+k,0.)+v
                    for tag,row in audit.items():
                        total=state['candidate_audit'].setdefault(split+'/'+tag,{})
                        for k,v in row.items():total[k]=total.get(k,0)+v
                    for route,row in edit_rows.items():
                        total=state['edits'][split+':'+route]
                        for k,v in row.items():total[k]=total.get(k,0)+v
                    for k,v in first_edits.items():state['edits']['first'][k]=state['edits']['first'].get(k,0)+v
                    state['completed_windows']+=1;cursor=state['completed_windows']
                    stopping=stop_event is not None and stop_event.is_set()
                    if cursor%checkpoint_every==0 or cursor==len(jobs) or stopping:persist()
                    if progress:progress(dict(window=cursor,windows=len(jobs),split=split,scene=w.scene_name,
                        t0_token=w.t0_token,seconds=stages,audit=audit))
                    del first,second,candidates,pred1,prediction,raw,record,handoff
                    if stopping:raise InterruptedError('stopped at ALL-candidate window boundary')
            finally:iterator.close()
    except InterruptedError:persist();raise
    return persist()


def reports(state, populations, routes):
    result={}
    for split in populations:
        rows={}
        for route in routes:
            name=split+':'+route;raw={k:np.asarray(v,np.int64) for k,v in state['counts'][name].items()}
            metrics=old.rollout.finalize_metrics(raw);scene_rows={}
            for scene,counts in state['scene_counts'].get(name,{}).items():
                reference=state['scene_counts'][split+':baseline'][scene]
                scene_rows[scene]=(old.rollout.finalize_metrics({k:np.asarray(v,np.int64) for k,v in counts.items()})['average_4s_5s_6s']['mIoU']
                    -old.rollout.finalize_metrics({k:np.asarray(v,np.int64) for k,v in reference.items()})['average_4s_5s_6s']['mIoU'])
            class_iou=old.rollout.legacy._safe_iou(raw['sem_inter'],raw['sem_union'])
            with np.errstate(invalid='ignore'):
                per_class={str(c):float(np.mean(class_iou[3:,i][np.isfinite(class_iou[3:,i])]))
                    if np.isfinite(class_iou[3:,i]).any() else None
                    for i,c in enumerate(old.rollout.legacy.SEMANTIC_CLASSES)}
            rows[route]=dict(metrics=metrics,raw_counts=state['counts'][name],class_mIoU_4_6=per_class,
                change_quality=state['comparison_quality'].get(name,{}),scene_delta=dict(by_scene=scene_rows,
                    positive=sum(v>0 for v in scene_rows.values()),negative=sum(v<0 for v in scene_rows.values()),
                    zero=sum(v==0 for v in scene_rows.values())))
        result[split]=rows
    return old.finite_json(result)


def summary(result):
    lines=['===== FROZEN SURFACE GEOMETRY CARRY =====','protocol: '+PROTOCOL,
        'One frozen mean 5/6/8/12/14; ADD raw0.5 / REMOVEoff; NO training/GT prediction inputs.',
        'Inference algorithm ablation, NOT byte-identical lossless optimization.',
        'First 1--3s shared exactly. Static known background state / past predicted SE(2) only.',
        'Future GT ego poses remain explicit conditions; Moving support refers to original t0.']
    for split,rows in result['reports'].items():
        lines.append(f'===== {split}: {result["populations"][split]["selected_windows"]} windows =====')
        base=rows['baseline']['metrics']['average_4s_5s_6s']
        for route,row in rows.items():
            v=row['metrics']['average_4s_5s_6s']
            micro=(f'{v["MovingMicro"]-base["MovingMicro"]:+.6f}'
                   if v['MovingMicro'] is not None and base['MovingMicro'] is not None else 'NA')
            lines.append(f'{route:13s} avg4--6 mIoU={v["mIoU"]:.6f} IoU={v["IoU"]:.6f} '
                f'dMiOU={v["mIoU"]-base["mIoU"]:+.6f} dIoU={v["IoU"]-base["IoU"]:+.6f} '
                f'dMovingMicro={micro}')
            for h in ('4.0','5.0','6.0'):
                p=row['metrics']['per_horizon'][h]
                mm=f'{p["MovingMicro"]:.6f}' if p['MovingMicro'] is not None else 'NA'
                lines.append(f'  {h}s mIoU={p["mIoU"]:.6f} IoU={p["IoU"]:.6f} MovingMicro={mm}')
            lines.append('  quality='+json.dumps(row['change_quality'])+' scenes='+json.dumps({k:v for k,v in row['scene_delta'].items() if k!='by_scene'}))
    lines += ['TRAIN-only eligible candidate: '+str(result['selected_train_route']),
        'TRAIN gate: avg dMiOU>=0.05pp, avg IoU/MovingMicro nonnegative, 4/5/6s mIoU each nonnegative.',
        'DEV diagnostic ONLY; no automatic promotion/retry, original results untouched.',
        'candidate_audit: '+json.dumps(result['candidate_audit']), 'stage_seconds: '+json.dumps(result['stage_seconds'])]
    if 'geniedrive_code_compatibility' in result:
        lines += ['Public-code compatibility, NOT verified paper Table-2 replication:',
                  json.dumps(result['geniedrive_code_compatibility'])]
    return '\n'.join(lines)+'\n'


def select_screen(records, source, parent, split):
    selected,audit=old.rollout.select_long_population(records,source.iter_windows(history=4,future=12),parent,
        'dev64' if split=='dev64' else 'all')
    if split=='train64':
        if len(selected)<64:raise RuntimeError('fewer than64 TRAIN long windows')
        keys=select_scene_balanced_round_robin([(w.scene_name,w.t0_token) for w,_ in selected],64)
        by={(w.scene_name,w.t0_token):(w,r) for w,r in selected};selected=[by[k] for k in keys]
    audit={**audit,'population':split,'selected_windows':len(selected),'scenes':len({w.scene_name for w,_ in selected}),
        'selected_keys':[[w.scene_name,w.t0_token] for w,_ in selected]}
    return selected,audit


def main(stop_event=None,argv=None):
    p=old.parser();p.description=__doc__
    p.add_argument('--train-cache');p.add_argument('--train-info')
    p.add_argument('--experiment',choices=('screen','all'),default='screen')
    p.add_argument('--selection-from',help='completed screen evaluation.json; TRAIN-selected recipe ONLY')
    p.add_argument('--majority-workers',type=int,default=4)
    a=p.parse_args(argv);root=Path(__file__).resolve().parents[2];out=Path(a.out_dir).resolve()
    if a.experiment=='screen' and (not a.train_cache or not a.train_info or a.selection_from):p.error('screen requires TRAIN inputs, no selection-from')
    if a.experiment=='all' and (not a.selection_from or not a.geniedrive_info):p.error('all requires frozen TRAIN screen recipe + official metadata')
    if a.experiment=='screen' and (a.population!='dev64' or a.population_alignment!='legacy_cache6s' or a.geniedrive_info):
        p.error('screen population is fixed TRAIN64 + dev64')
    if a.experiment=='all' and (a.population!='all' or a.population_alignment!='geniedrive_code10s'):
        p.error('all requires official GenieDrive code population')
    if not 1<=a.cpu_workers<=16 or not 0<=a.majority_workers<=min(8,a.cpu_workers):p.error('invalid CPU budget')
    if not 1<=a.ccr_cpu_workers<=a.cpu_workers or not 1<=a.surface_query_workers<=8 or not 1<=a.prefetch_workers<=min(4,a.cpu_workers):p.error('invalid execution workers')
    if not 0<=a.frame_cache_mib<=4096 or not 0<=a.val_cache_ram_mib<=2048 or a.checkpoint_every<1:p.error('invalid bounded cache/checkpoint budget')
    if a.resume:
        if not (out/'evaluation_state.json').is_file():p.error('resume SAME geometry-carry directory')
        if (out/'evaluation.json').exists():p.error('already completed; read summary')
    elif out.exists():p.error('new output required')
    for name in ('config','dev_cache','dev_info','base_checkpoint','population_manifest',
                 *(('train_cache','train_info') if a.experiment=='screen' else ('selection_from','geniedrive_info'))):
        if not Path(getattr(a,name) or '').is_file():p.error('missing '+name)
    if any((d/'training.json').exists() for d in (out,*out.parents)):p.error('output outside training required')
    bundle=(json.loads((out/'bundle.json').read_text(encoding='utf-8')) if a.resume else
        old.find_frozen_bundle(a.runs_root,a.run_dir,a.source_bundle_dir))
    digest=bundle.pop('fingerprint')
    if (old.stable_json_fingerprint(bundle)!=digest or bundle.get('selection_frozen') is not True
            or set(bundle['candidates'])!={old.AVERAGE_NAME} or bundle['run_directory']!=str(Path(a.run_dir).resolve())):
        raise RuntimeError('frozen Surface bundle changed')
    bundle['fingerprint']=digest;old.verify_sources(bundle);trained=bundle['audit']['contract']
    if out.is_relative_to(Path(bundle['source_comparison_directory'])):p.error('output outside source comparison required')
    if a.source_bundle_dir and str(Path(a.source_bundle_dir).resolve())!=bundle['source_comparison_directory']:p.error('source comparison changed')
    keys=('dev_cache','dev_info','base_checkpoint')+(('train_cache','train_info') if a.experiment=='screen' else ())
    for name in keys:
        if old.sha256(getattr(a,name))!=trained['data'][name]:raise RuntimeError('original data changed: '+name)
    if old.sha256(a.base_checkpoint)!=old.CLEAN_SHA256 or str(Path(a.dataroot).resolve())!=trained['dataroot']:raise RuntimeError('original E14/dataroot required')
    cfg=old.load_runtime_config(a.config,a.override);pcfg=old.make_prepare_config(cfg)
    if (old.stable_json_fingerprint(cfg)!=trained['runtime_config_fingerprint'] or old.training_implementation(root)!=trained['implementation']
            or str(torch.__version__)!=trained['torch_version']):raise RuntimeError('original config/model/Torch required')
    if pcfg.future_frames!=6 or pcfg.free_label!=17 or not np.isclose(pcfg.frame_dt_s,.5,rtol=0,atol=1e-12):raise RuntimeError('six nominal2Hz/free17 required')
    manifest,keys64,_=old.load_manifest(a.population_manifest)
    if manifest['manifest_fingerprint']!=trained['dev_manifest_fingerprint'] or manifest['selected_key_fingerprint']!=old.DEV64_FP or len(keys64)!=64 or len(manifest['parent_keys'])!=512:raise RuntimeError('frozen dev population changed')
    device=old.require_cuda(a.device);torch.set_num_threads(1)
    if not a.resume:out.mkdir(parents=True)
    with old.evaluation_lock(out):
        row=bundle['candidates'][old.AVERAGE_NAME];snapshot=out/'checkpoint_snapshot.pt'
        if a.resume:
            if old.sha256(snapshot)!=row['sha256']:raise RuntimeError('immutable mean snapshot changed')
        else:
            if old.snapshot_checkpoint(row['path'],snapshot)!=row['sha256']:raise RuntimeError('source mean changed during snapshot')
            old.write_json(out/'bundle.json',bundle)
        saved_model,joint=old.load_evaluation_model(snapshot,device=device,z_bins=pcfg.grid.shape_hwd[2])
        if saved_model['source_epochs']!=list(old.AVERAGE_EPOCHS) or joint.transport.config.history_frames!=4 or saved_model['weight_fingerprint']!=row['weight_fingerprint'] or old.stable_json_fingerprint(saved_model['training_contract'])!=old.stable_json_fingerprint(trained):raise RuntimeError('frozen mean recipe changed')
        del saved_model
        routes=list(carry.ROUTES);selection_digest=None;selected_train=None
        if a.experiment=='all':
            recipe=json.loads(Path(a.selection_from).read_text(encoding='utf-8'));selection_digest=old.sha256(a.selection_from)
            selected_train=validate_recipe(recipe,digest,implementation(root))
            routes=['baseline',selected_train]
        sources={};populations={};jobs=[]
        splits=('train64','dev64') if a.experiment=='screen' else ('all',)
        for split in splits:
            training=split=='train64';info=a.train_info if training else a.dev_info
            source=old.CachedColumnSource(old.NuScenesWindowSource(a.dataroot,info_pkl=info,verbose=False),a.frame_cache_mib)
            sources[split]=source;_,records=old.load_cache(a.train_cache if training else a.dev_cache);keys_all=old.record_keys(records)
            if len(records)!=(TRAIN_WINDOWS if training else old.VAL_WINDOWS):raise RuntimeError('complete original cache required')
            if split=='all':old.genie.validate_grid(pcfg);selected,population=old.genie.select_population(a.geniedrive_info,source,records)
            else:selected,population=select_screen(records,source,keys_all if training else manifest['parent_keys'],split)
            populations[split]=population;jobs.extend((split,w,r) for w,r in selected);del records
        if a.experiment=='screen' and sources['train64'].allowed_scenes & sources['dev64'].allowed_scenes:raise RuntimeError('TRAIN/DEV scene overlap')
        if {w.scene_name for split,w,_ in jobs if split!='train64'} & {str(s) for s,_ in trained['prior_keys']}:
            raise RuntimeError('DEV/TRAIN prior scene overlap')
        timestamps=[old.rollout.validate_timestamps(sources[split].nusc,w) for split,w,_ in jobs]
        provider=old.surface.SurfaceRolloutProvider(a.base_checkpoint,old.CLEAN_SHA256,pcfg,device,a.cpu_workers,joint,None)
        provider.raw_prefetch_workers=provider.raw_prefetch_depth=a.prefetch_workers
        provider.raw_io_workers=1
        cache=None;execution=None;majority=None
        try:
            caches={};namespace=None
            if a.val_history_cache:
                seed=old.val_namespace(provider,a,root);namespace=hashlib.sha256((old.CACHE_PROTOCOL+seed).encode()).hexdigest()
                if namespace!=trained['val_history_namespace'] or not (Path(a.val_history_cache)/namespace/'manifest.json').is_file():raise RuntimeError('existing verified VAL cache required')
                cache=old.CausalGeometryCache(a.val_history_cache,seed,max_bytes=0,ram_bytes=a.val_cache_ram_mib*2**20,reserve_bytes=0)
                old.validate_manifest(cache,a);caches['dev64' if a.experiment=='screen' else 'all']=cache
            from real_motion.strong_warp_execution import selected_backend
            from real_motion.native_column_cpu import backend_name
            contract=dict(protocol=PROTOCOL,bundle_fingerprint=digest,source_sha256=row['sha256'],experiment=a.experiment,
                routes=[split+':'+route for split in splits for route in routes],candidate_routes=routes,populations=populations,
                future_tokens=[list(w.future_tokens) for _,w,_ in jobs],timestamp_audit=old.rollout.summarize_timestamps(timestamps),
                thresholds=[.5,None],static_protocol=carry.PROTOCOL,selection_sha256=selection_digest,
                initial_observations=4,future_frames_per_block=6,original_static_state_retained=True,
                future_GT_prediction_inputs=False,future_ego_poses='GT_through6s',val_namespace=namespace,
                strong_warp_backend=selected_backend(),integer_cpu_backend=backend_name(),
                runtime_config_fingerprint=old.stable_json_fingerprint(cfg),torch_version=str(torch.__version__),
                geniedrive_info_sha256=old.genie.INFO_SHA256 if a.experiment=='all' else None,
                data={k:trained['data'][k] for k in keys},execution={k:getattr(a,k) for k in
                    ('device','cpu_workers','ccr_cpu_execution','ccr_cpu_workers','surface_query_workers','prefetch_workers',
                     'frame_cache_mib','val_cache_ram_mib','no_graphs','majority_workers')},
                implementation=implementation(root))
            if a.resume:
                if json.loads((out/'contract.json').read_text(encoding='utf-8'))!=contract:raise RuntimeError('carry resume contract changed')
                saved=json.loads((out/'evaluation_state.json').read_text(encoding='utf-8'))
            else:old.write_json(out/'contract.json',contract);saved=None
            old.write_json(out/'timestamp_audit.json',dict(summary=contract['timestamp_audit'],per_window=timestamps))
            execution=old.surface.SurfaceBlockExecution(provider,mode=a.ccr_cpu_execution,workers=a.ccr_cpu_workers,query_workers=a.surface_query_workers,graphs=not a.no_graphs)
            if a.majority_workers:majority=ParallelNativeMajority(a.majority_workers)
            with strong_majority_execution(majority), (out/'progress.jsonl').open('a' if a.resume else 'x',encoding='utf-8') as log:
                def progress(v):
                    log.write(json.dumps(old.finite_json(v),allow_nan=False)+'\n');log.flush()
                    if v['window']==1 or v['window']%8==0 or v['window']==len(jobs):print(f'carry={v["window"]}/{len(jobs)} split={v["split"]} seconds={v["seconds"]["window_total"]:.3f}',flush=True)
                try:state=evaluate(provider,sources,caches,jobs,execution,contract,saved=saved,save=lambda v:old.write_json(out/'evaluation_state.json',v),progress=progress,stop_event=stop_event,checkpoint_every=a.checkpoint_every)
                except InterruptedError:
                    boundary=json.loads((out/'evaluation_state.json').read_text(encoding='utf-8'))['completed_windows']
                    old.write_json(out/'evaluation_status.json',dict(status='interrupted',completed_windows=boundary))
                    print('STOPPED at complete four-route boundary; resume SAME directory',flush=True);return 130
            result=dict(**state,protocol=PROTOCOL,status='complete',bundle_fingerprint=digest,populations=populations,
                reports=reports(state,populations,routes),no_training=True,future_GT_prediction_inputs=False,
                timestamp_audit=contract['timestamp_audit'],source_epochs=list(old.AVERAGE_EPOCHS),
                snapshot_sha256=row['sha256'],weight_fingerprint=row['weight_fingerprint'],
                implementation=contract['implementation'],candidate_routes=routes,
                majority_execution=None if majority is None else majority.stats())
            result['selected_train_route']=choose_train_candidate(result['reports']['train64']) if a.experiment=='screen' else selected_train
            if a.experiment=='all':result['geniedrive_code_compatibility']={r:old.genie.compatibility_metrics({k:np.asarray(v,np.int64) for k,v in result['counts']['all:'+r].items()}) for r in routes}
            old.verify_sources(bundle)
            if old.sha256(snapshot)!=row['sha256']:raise RuntimeError('snapshot changed during read-only screen')
            result=old.finite_json(result);result['result_fingerprint']=old.stable_json_fingerprint(result)
            old.write_json(out/'evaluation.json',result);(out/'summary.txt').write_text(summary(result),encoding='utf-8')
            print(summary(result),flush=True);return 0
        finally:
            if majority is not None:majority.close()
            if execution is not None:execution.close()
            if cache is not None:cache.close()


if __name__=='__main__':
    event=threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:event.set())
    sys.exit(main(event) or 0)
