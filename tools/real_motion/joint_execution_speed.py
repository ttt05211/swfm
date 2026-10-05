"""One bounded run: exact eval comparison AND actual six-frame generation FPS."""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import hashlib
import json
import statistics
import time

import numpy as np
import torch

from real_motion.column_execution import execution_session
from real_motion.column_runtime_pipeline import CachedColumnSource
from real_motion.causal_column_completion import actions_from_probabilities, compose_sparse, sparse_layout
from real_motion.causal_column_sampling import ColumnHistoryIndex
from real_motion.v21_source_induction import stable_json_fingerprint
from tools.real_motion import causal_column_common as columns
from tools.real_motion import benchmark_p0_f9_v18_runtime as runtime
from tools.real_motion.joint_eval_speed import speed_records, _pass


MODES = {'parallel_raw':(False,False,False),'async_readback':(True,False,False),
         'graph_async':(True,True,False),'reuse_graph_async':(True,True,True)}


def synchronize(device):
    if device.type == 'cuda': torch.cuda.synchronize(device)


def _set(model,name):
    async_output,graphs,reuse = MODES[name]
    model.column_inference_optimized=True; model.column_readback_optimized=False
    model.column_probability_optimized=False; model.column_map_prefetch=False
    model.column_async_readback=async_output; model.column_probability_fingerprint=True
    return execution_session(model,graphs=graphs,reuse=reuse) if graphs or reuse else nullcontext(None)


def _probability_fingerprint(events):
    return stable_json_fingerprint([{h:(p['queries'],p['probability_sha256'])
        for h,p in event['prediction_seconds_by_horizon'].items()} for event in events])


def generation_six(provider,source,record,raw,model,gates,batch_size,*,transport_inputs=None,loaded_history=False):
    """Prepared causal inputs -> six FINISHED joint grids, no GT/metrics/E14.

    Rebuild Strong/KTA future priors INSIDE the timer, matching the old main
    generation boundary. Prior extraction/registration/static memory is separate.
    Cached V18 input features are explicit; this is NOT raw-sensor end-to-end FPS.
    """
    if raw.get('future_gt_occ') is not None:raise RuntimeError('FPS cannot consume future GT')
    if loaded_history:
        # Strict in-memory boundary charges source extraction/association,
        # registration, static evidence AND prior once. No cached causal state.
        current_raw={k:v for k,v in raw.items() if not k.startswith('_')}
    else:
        causal = raw['_column_causal_preparation']; fixed = causal['prepared_state']
        state = {**fixed,'gpu':None}
        state.pop('column_backgrounds',None)
        anchors,baseline = runtime._strong_all_horizons(state['current_sem'],state['current_pose'],state['future_poses'],
            state['current'],state['velocities'],state['source_world_points'],frame_dt_s=float(provider.pcfg.frame_dt_s),
            grid=provider.pcfg.grid,cfg=provider.strong,runtime_device=provider.device)
        state['anchors'],state['baseline_by_hi'] = anchors,baseline
        state['baseline_clear_by_hi'] = [runtime.baseline_clear_mask(x,grid=provider.pcfg.grid) for x in baseline]
        state['baseline_clear_flat_by_hi'] = [runtime.baseline_clear_flat_indices(x,grid=provider.pcfg.grid) for x in baseline]
        current_raw = {**raw,'_column_causal_preparation':{**causal,'prepared_state':state}}
    # Same resident input-tensor starting point as frozen E14, not H2D for
    # Local alone. This stages INPUTS only; live motion is still timed.
    outputs = None if transport_inputs is None else runtime._model_forward(
        provider.joint.transport,transport_inputs,provider.device,return_latents=True)
    prep = provider.prepare_columns(source,record,include_gt=False,raw_window=current_raw,outputs=outputs)
    index = ColumnHistoryIndex(prep,provider.pcfg.grid); result = []; profiles=[]
    # Candidate CPU work is deployment work, not removed from the clock.
    execution=getattr(model,'column_execution_session',None)
    memory_window=execution.patch_window() if execution is not None else nullcontext(None)
    with memory_window, ThreadPoolExecutor(max_workers=min(3,provider.workers)) as pool:
        plans={h:pool.submit(columns.candidate_plan,prep,h,provider.pcfg.grid,model.config) for h in range(6)}
        for h in range(6):
            plan=plans[h].result()
            p=columns.predict_probabilities(model,prep,h,plan,provider.pcfg.grid,provider.device,batch_size,history_index=index)
            profile=dict(model.last_prediction_profile)
            if 'probability_sha256' not in profile:profile['_probabilities']=p
            profiles.append(profile)
            action=actions_from_probabilities(plan,p,gates)
            ids,_,after=compose_sparse(plan,action,layout=sparse_layout(plan))
            dense=prep.baseline[h].copy();dense.reshape(-1)[ids]=after;result.append(dense)
    return result,profiles


def _generation_fingerprint(predictions,profiles):
    sha=hashlib.sha256()
    if len(predictions) != 6 or len(profiles) != 6: raise RuntimeError('six finished frames required for FPS')
    for a in predictions: sha.update(np.ascontiguousarray(a).view(np.uint8))
    for p in profiles:
        if '_probabilities' in p:p['probability_sha256']=hashlib.sha256(p.pop('_probabilities').view(np.uint8)).hexdigest()
    return sha.hexdigest(),stable_json_fingerprint([(p['queries'],p['probability_sha256']) for p in profiles])


def benchmark_execution(provider,source,records,model,gates,out,*,windows=32,repeats=2,batch_size=256,
                        fps_windows=18,stop_event=None):
    rows,keys=speed_records(records,windows)
    if not 1 <= fps_windows <= windows: raise ValueError('FPS budget must be within speed population')
    model_attrs=('column_inference_optimized','column_readback_optimized','column_probability_optimized',
        'column_map_prefetch','column_async_readback','column_probability_fingerprint','column_inference_verify_remaining',
        'column_sampling_workers')
    original={k:getattr(model,k,None) for k in model_attrs}
    provider_attrs=('raw_prefetch_workers','raw_prefetch_depth','raw_io_workers','columns_checked')
    previous={k:getattr(provider,k,None) for k in provider_attrs}
    result=dict(status='running',actual_cuda=provider.device.type == 'cuda',windows=windows,fps_windows=fps_windows,
        keys=keys,population_key_fingerprint=stable_json_fingerprint(keys),repeats=repeats,batch_size=batch_size,
        thresholds=gates,trials=[],fps_trials=[],strict_fps_trials=[],rejected=[],no_weight_or_threshold_selection=True,
        no_persistent_geometry_writes=True)
    expected=expected_prob=None;fps_expected={};fps_probability={};accepted=set(MODES)
    try:
      provider.raw_prefetch_workers=provider.raw_prefetch_depth=min(4,provider.workers)
      provider.raw_io_workers=max(1,min(4,provider.workers//provider.raw_prefetch_workers))
      model.column_sampling_workers=provider.workers
      with (out/'execution_progress.jsonl').open('x',encoding='utf-8') as log:
        for repeat in range(repeats):
          order=list(MODES) if repeat % 2 == 0 else list(reversed(MODES))
          for name in order:
            if name not in accepted:continue
            if stop_event is not None and stop_event.is_set(): raise InterruptedError('execution benchmark stopped')
            fresh=CachedColumnSource(source.source,source.limit/2**20)
            model.column_inference_verify_remaining=3
            with _set(model,name) as execution:
              try:
                _pass(provider,fresh,rows[:min(2,windows)],model,gates,batch_size,None,stop_event)
              except RuntimeError as exc:
                if name == 'parallel_raw' or 'exactness' not in str(exc): raise
                accepted.remove(name);result['rejected'].append(dict(mode=name,stage='eval_warmup',reason=str(exc)))
                print(f'EXECUTION {name} REJECTED: {exc}',flush=True);continue
              provider.frozen_metric_counts.clear(); events=[]
              def progress(e):
                events.append(e);log.write(json.dumps(dict(e,trial=name,repeat=repeat+1))+'\n');log.flush()
              synchronize(provider.device)
              if provider.device.type == 'cuda':torch.cuda.reset_peak_memory_stats(provider.device)
              tick=time.perf_counter()
              fingerprint,report=_pass(provider,fresh,rows,model,gates,batch_size,progress,stop_event)
              synchronize(provider.device);seconds=time.perf_counter()-tick
              prob=_probability_fingerprint(events)
              if expected is None:expected,expected_prob=fingerprint,prob
              if fingerprint != expected or prob != expected_prob:
                if name == 'parallel_raw':raise RuntimeError('baseline not repeatable')
                accepted.remove(name);result['rejected'].append(dict(mode=name,stage='eval_timed',reason='probability bytes or integer counts differ'))
                print(f'EXECUTION {name} REJECTED: timed exactness',flush=True);continue
              stages=defaultdict(float);patch=defaultdict(int)
              for e in events:
                for k in ('input_wait_seconds','compute_seconds','column_probability_seconds','composition_metrics_seconds'):
                  stages[k]+=e.get(k,0.)/windows
                for p in e['prediction_seconds_by_horizon'].values():
                  for k,v in p.get('host_stages_seconds_NOT_cuda_kernel_time',{}).items():stages[k]+=v/windows
                  for k in ('patch_cache_hits','patch_cache_misses','encoded_patches'):patch[k]+=p.get(k,0)
                  patch['peak_patch_cache_bytes']=max(patch['peak_patch_cache_bytes'],p.get('patch_cache_peak_bytes',0))
              trial=dict(name=name,repeat=repeat+1,seconds_per_window=seconds/windows,counts_fingerprint=fingerprint,
                  probability_fingerprint=prob,host_seconds_per_window=dict(stages),patch_audit=dict(patch),
                  execution_audit={} if execution is None else execution.audit(),
                  peak_reserved_mib=torch.cuda.max_memory_reserved(provider.device)/2**20 if provider.device.type == 'cuda' else None)
              result['trials'].append(trial)
              print(f'EXECUTION {name} repeat={repeat+1}/{repeats} seconds/window={seconds/windows:.4f} bytes/counts=PASS',flush=True)
            del fresh,report,events
        # No GT loading in FPS. Each window is prepared once then all modes
        # generate live current forecasts; no learned proposal/probability cache.
        fps_source=CachedColumnSource(source.source,source.limit/2**20)
        for wi,record in enumerate(rows[:fps_windows]):
          if stop_event is not None and stop_event.is_set():raise InterruptedError('FPS stopped')
          io_tick=time.perf_counter();raw=provider.load_raw_columns(fps_source,record,include_gt=False)
          preparation_seconds=time.perf_counter()-io_tick
          if raw.get('future_gt_occ') is not None:raise RuntimeError('FPS path must never load future GT')
          # Same source/history-only input to old V18 strict generation timer.
          state={**raw['_column_causal_preparation']['prepared_state'],'rec':record,'window':columns.window_from_record(record),'gpu':None}
          runtime._stage_gpu_inputs(state,provider.device)
          try:
            runtime._forecast_once_with_prior_rebuild(provider.reference,state,provider.pcfg,provider.strong,provider.device)
            synchronize(provider.device);tick=time.perf_counter()
            runtime._forecast_once_with_prior_rebuild(provider.reference,state,provider.pcfg,provider.strong,provider.device)
            synchronize(provider.device);e14_seconds=time.perf_counter()-tick
            transport_inputs=state['gpu']  # same five input tensors, current Local.forward trims to strict four
          finally:runtime._release_gpu_inputs(state)
          # E14 legacy six history vs Local four is labelled, never a matched
          # observation budget experiment. Same card and same ordered windows.
          for repeat in range(repeats):
            order=list(MODES) if repeat % 2 == 0 else list(reversed(MODES))
            for name in order:
              if name not in accepted:continue
              with _set(model,name) as execution:
                model.column_inference_verify_remaining=0
                model.column_probability_fingerprint=False  # correctness hashes AFTER synchronized timer
                # Warm graph shapes/allocator separately. These costs are
                # reported, not disguised as steady-state generation latency.
                warm_tick=time.perf_counter()
                generation_six(provider,fps_source,record,raw,model,gates,batch_size,transport_inputs=transport_inputs)
                synchronize(provider.device);warm_seconds=time.perf_counter()-warm_tick
                synchronize(provider.device);tick=time.perf_counter()
                pred,profiles=generation_six(provider,fps_source,record,raw,model,gates,batch_size,transport_inputs=transport_inputs)
                synchronize(provider.device);seconds=time.perf_counter()-tick
                fp,pp=_generation_fingerprint(pred,profiles)
                if name == 'parallel_raw':
                  if wi in fps_expected and (fp != fps_expected[wi] or pp != fps_probability[wi]):
                    raise RuntimeError('baseline six-frame FPS not repeatable; no result accepted')
                  fps_expected[wi]=fp;fps_probability[wi]=pp
                elif fp != fps_expected[wi] or pp != fps_probability[wi]:
                  accepted.remove(name);result['rejected'].append(dict(mode=name,stage='six_frame_FPS',window=wi+1,
                      reason='dense output or probability bytes differ'));continue
                row=dict(name=name,window=wi+1,repeat=repeat+1,seconds=seconds,frames=6,
                    dense_fingerprint=fp,probability_fingerprint=pp,warmup_seconds=warm_seconds,
                    raw_and_causal_preparation_seconds=preparation_seconds,e14_generation_seconds=e14_seconds,
                    execution_audit={} if execution is None else execution.audit(),
                    host_profiles=profiles)
                result['fps_trials'].append(row)
                log.write(json.dumps(dict(event='six_frame_generation',**row))+'\n');log.flush()
              print(f'FPS {name} window={wi+1}/{fps_windows} repeat={repeat+1}/{repeats} six_frames_seconds={seconds:.4f} exact=PASS',flush=True)
          del raw,transport_inputs
      aggregates={};fps={}
      for name in accepted:
        trials=[t for t in result['trials'] if t['name']==name]
        if len(trials)!=repeats:continue
        aggregates[name]=statistics.median(t['seconds_per_window'] for t in trials)
        ft=[t for t in result['fps_trials'] if t['name']==name]
        if len(ft)!=fps_windows*repeats:continue
        sec=sum(t['seconds'] for t in ft);prep=sum(t['raw_and_causal_preparation_seconds'] for t in ft)
        fps[name]=dict(six_frame_seconds=sec/len(ft),six_frame_amortized_fps=6*len(ft)/sec,
            causal_input_preparation_including_IO_seconds=prep/len(ft))
      base=[t for t in result['fps_trials'] if t['name']=='parallel_raw']
      e14=sum(t['e14_generation_seconds'] for t in base)/len(base)
      fastest_generation=max(fps,key=lambda n:fps[n]['six_frame_amortized_fps'])
      # Also measure the often hidden NEW causal registration/static-memory
      # cost. Bounded six-window check, only baseline + fastest verified mode.
      # This is an actual fresh run, never a sum of overlapping stage timers.
      for wi,record in enumerate(rows[:min(6,fps_windows)]):
        if stop_event is not None and stop_event.is_set():raise InterruptedError('strict FPS stopped')
        raw=provider.load_raw_columns(fps_source,record,include_gt=False)
        staged=runtime._gpu_inputs(record,provider.device)
        for repeat in range(repeats):
          names=list(dict.fromkeys(('parallel_raw',fastest_generation)))
          if repeat % 2:names.reverse()
          for name in names:
            with _set(model,name):
              model.column_inference_verify_remaining=0;model.column_probability_fingerprint=False
              generation_six(provider,fps_source,record,raw,model,gates,batch_size,transport_inputs=staged,loaded_history=True)
              synchronize(provider.device);tick=time.perf_counter()
              pred,profiles=generation_six(provider,fps_source,record,raw,model,gates,batch_size,transport_inputs=staged,loaded_history=True)
              synchronize(provider.device);seconds=time.perf_counter()-tick
              fp,pp=_generation_fingerprint(pred,profiles)
              if fp != fps_expected[wi] or pp != fps_probability[wi]:
                raise RuntimeError('strict loaded-history FPS differs from prepared causal forecast; no result accepted')
              row=dict(name=name,window=wi+1,repeat=repeat+1,seconds=seconds,frames=6,host_profiles=profiles)
              result['strict_fps_trials'].append(row)
              with (out/'execution_progress.jsonl').open('a',encoding='utf-8') as strict_log:
                strict_log.write(json.dumps(dict(event='strict_six_frame_generation',**row))+'\n')
              print(f'STRICT FPS {name} window={wi+1}/{min(6,fps_windows)} seconds={seconds:.4f} exact=PASS',flush=True)
        del raw,staged
      strict={}
      for name in dict.fromkeys(('parallel_raw',fastest_generation)):
        ts=[t for t in result['strict_fps_trials'] if t['name']==name]
        sec=sum(t['seconds'] for t in ts)
        strict[name]=dict(windows=len(ts),six_frame_seconds=sec/len(ts),six_frame_amortized_fps=6*len(ts)/sec)
      result.update(status='complete',aggregates=aggregates,fps=fps,integer_counts_exact=True,probability_bytes_exact=True,
          strict_loaded_history_fps=strict,
          six_frame_dense_outputs_exact=True,fastest_eval=min(aggregates,key=aggregates.get),
          fastest_generation=fastest_generation,
          legacy_E14_same_card_six_frame_seconds=e14,legacy_E14_same_card_fps=6/e14,
          fps_boundary='resident causal source tensors + registered history + static memory -> Strong/KTA prior rebuild + live transport + ALL SIX column forecasts + SIX dense compositions; excludes disk/GT/metrics/E14 and correctness hashing',
          strict_fps_boundary='loaded FOUR history occupancy/visibility/poses + resident source tensors -> fresh source extraction/matching/registration/static evidence + prior + live model + all SIX dense joint frames; excludes I/O/GT/metrics/hashes; cached source tensor extraction is still explicitly outside',
          preparation_boundary='raw loading + CPU fixed geometry/registration/static memory; includes I/O and old prior prefill (rebuilt in generation). Separately reported, NOT additive with generation or raw-sensor E2E FPS',
          legacy_E14_history_frames=6,local_history_frames=model.history_frames,no_automatic_backend_promotion=True)
    finally:
      for obj,values in ((model,original),(provider,previous)):
        for k,v in values.items():
          if v is None:
            if hasattr(obj,k):delattr(obj,k)
          else:setattr(obj,k,v)
    return result


def summary_text(r):
    lines=['===== LOCAL EXECUTION: EVAL + SIX-FRAME FPS / NO SELECTION =====',
        f"actual_cuda={r['actual_cuda']} eval_windows={r['windows']} fps_windows={r['fps_windows']} repeats={r['repeats']} batch={r['batch_size']}",
        f"thresholds={r['thresholds']} probability_bytes_exact={r['probability_bytes_exact']} counts_exact={r['integer_counts_exact']} six_dense_frames_exact={r['six_frame_dense_outputs_exact']}"]
    for n,s in r['aggregates'].items():
        f=r['fps'].get(n,{})
        lines.append(f"{n}: eval_seconds/window={s:.6f} eval_speedup={r['aggregates']['parallel_raw']/s:.3f} six_frame_seconds={f.get('six_frame_seconds')} FPS={f.get('six_frame_amortized_fps')}")
        trials=[t for t in r['trials'] if t['name']==n]
        hits=sum(t['patch_audit'].get('patch_cache_hits',0) for t in trials)
        misses=sum(t['patch_audit'].get('patch_cache_misses',0) for t in trials)
        lines.append(f"  identical_patch_hit_fraction={hits/max(1,hits+misses):.3%} graph_replays={sum(t['execution_audit'].get('cuda_graph_replays',0) for t in trials)}")
        lines.append('  host_seconds/window='+json.dumps({k:statistics.median(t['host_seconds_per_window'][k] for t in trials)
            for k in trials[0]['host_seconds_per_window']},sort_keys=True))
        lines.append('  graph_capture_failures='+json.dumps([e for t in trials for e in t['execution_audit'].get('cuda_graph_failures',[])],ensure_ascii=False))
    lines += ['fastest_eval='+r['fastest_eval'],'fastest_six_frame_generation='+r['fastest_generation'],
        'strict_loaded_history_FPS='+json.dumps(r['strict_loaded_history_fps'],sort_keys=True),
        f"legacy_E14_same_card_FPS={r['legacy_E14_same_card_fps']:.3f} (six histories vs Local four; NOT matched history budget)",
        'FPS boundary: '+r['fps_boundary'],'Preparation boundary: '+r['preparation_boundary'],
        'Strict in-memory FPS boundary: '+r['strict_fps_boundary'],
        'Graph warmup/capture excluded from steady-state FPS, reported per trial. ALL SIX dense outputs AND probabilities verified outside timing.',
        'Rejected modes: '+json.dumps(r['rejected'],ensure_ascii=False),
        'Eval windows/s is NOT frame FPS. Host stage times are NOT CUDA active utilization.',
        'No weight/threshold changes, training, checkpoint selection, full rerun or automatic backend promotion.']
    return '\n'.join(lines)+'\n'
