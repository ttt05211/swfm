"""Same snapshot, ordered windows, batch and integer counts; speed only."""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import statistics
import time

import torch

from real_motion.column_runtime_pipeline import CachedColumnSource
from real_motion.v21_source_induction import select_scene_balanced_round_robin, stable_json_fingerprint
from tools.real_motion import causal_column_common as columns


def speed_records(records, windows):
    keys = [(str(r['scene_name']), str(r['t0_token'])) for r in records]
    if len(keys) != len(set(keys)): raise ValueError('duplicate speed population keys')
    if not 1 <= windows <= len(keys): raise ValueError('invalid speed population budget')
    chosen = select_scene_balanced_round_robin(keys, windows)
    by_key = dict(zip(keys, records))
    return [by_key[key] for key in chosen], chosen


def _pass(provider, source, records, model, gates, batch_size, progress, stop_event):
    state = None
    with ThreadPoolExecutor(max_workers=min(3,provider.workers)) as pool:
        iterator = columns.evaluation_steps(provider,source,records,model,gates,
            batch_size=batch_size,progress=progress,stop_event=stop_event,
            diagnostic_thresholds=None,candidate_pool=pool,verbose=False)
        try:
            while True:
                _, state = next(iterator)
        except StopIteration as complete: report = complete.value
        finally: iterator.close()
    return stable_json_fingerprint(columns.pack_evaluation_state(state)), report


def benchmark_evaluation(provider, source, records, model, gates, out, *,
                         windows=32, repeats=2, batch_size=256, stop_event=None,column_suite=False):
    rows, keys = speed_records(records, windows)
    workers = min(4,provider.workers)
    # No candidate/patch/network batch change. SAME immutable CPU frame-cache
    # budget per trial. Only raw window concurrency and output readback differ.
    modes = [('serial_raw',1,False),('parallel_raw',workers,False),('parallel_buffered',workers,True)]
    settings = {name:(raw,buffered,False,False) for name,raw,buffered in modes}
    if column_suite:
        # Keep the PROVEN parallel raw prefetch and original chunk readback.
        # No repeats of the known slower packed-upload bundle or serial I/O.
        settings = {'parallel_raw':(workers,False,False,False),
            'parallel_columns':(workers,False,True,False),
            'parallel_columns_prefetch':(workers,False,True,True)}
    result = dict(status='running',windows=windows,scenes=len({k[0] for k in keys}),keys=keys,
        population_key_fingerprint=stable_json_fingerprint(keys),repeats=repeats,
        batch_size=batch_size,actual_cuda=provider.device.type == 'cuda',trials=[],
        warmup_windows_per_trial=min(2,windows),frame_cache_mib=source.limit/2**20,
        no_persistent_geometry_writes=True,no_weight_or_threshold_selection=True,column_probability_suite=column_suite)
    expected = expected_probability = None
    original = {name:getattr(model,name,None) for name in ('column_inference_optimized',
        'column_inference_verify_remaining','column_readback_optimized','column_probability_optimized','column_map_prefetch',
        'column_probability_fingerprint')}
    original_provider = {name:getattr(provider,name,None) for name in
        ('raw_prefetch_workers','raw_prefetch_depth','raw_io_workers')}
    with (out/'progress.jsonl').open('x',encoding='utf-8') as stream:
      try:
        for repeat in range(repeats):
          # Reverse second round to reduce OS-cache/order advantage; do not
          # claim true cold-disk timing or pretend cache/page-cache is flushed.
          order = list(settings) if repeat % 2 == 0 else list(reversed(settings))
          for name in order:
            if stop_event is not None and stop_event.is_set(): raise InterruptedError('speed benchmark stopped')
            raw_workers, buffered, column_fast, map_prefetch = settings[name]
            provider.raw_prefetch_workers=provider.raw_prefetch_depth=raw_workers
            provider.raw_io_workers=max(1,min(4,provider.workers//raw_workers))
            if hasattr(provider,'frozen_metric_counts'):provider.frozen_metric_counts.clear()
            fresh = CachedColumnSource(source.source,source.limit/2**20)
            model.column_inference_optimized=True;model.column_readback_optimized=buffered
            model.column_probability_optimized=column_fast;model.column_map_prefetch=map_prefetch
            model.column_probability_fingerprint=column_suite
            model.column_inference_verify_remaining=3
            # Strong all-six/reference probability verification and first-use
            # kernel setup happen here, OUTSIDE timed end-to-end windows.
            _pass(provider,fresh,rows[:min(2,windows)],model,gates,batch_size,None,stop_event)
            if hasattr(provider,'frozen_metric_counts'):provider.frozen_metric_counts.clear()
            events = []
            def progress(row):
                events.append(row)
                stream.write(json.dumps(dict(row,trial=name,repeat=repeat+1))+'\n');stream.flush()
            if provider.device.type == 'cuda':
                torch.cuda.synchronize(provider.device);torch.cuda.reset_peak_memory_stats(provider.device)
            started = time.perf_counter()
            fingerprint, report = _pass(provider,fresh,rows,model,gates,batch_size,progress,stop_event)
            if provider.device.type == 'cuda': torch.cuda.synchronize(provider.device)
            seconds=time.perf_counter()-started
            if expected is None: expected = fingerprint
            if fingerprint != expected: raise RuntimeError('speed trial changed integer counts/confidence/quality; no speed result accepted')
            probability_fingerprint = None
            if column_suite:
                profiles = [event['prediction_seconds_by_horizon'] for event in events]
                if len(profiles) != windows or any(set(row) != {'1.0','2.0','3.0'} or
                        any(not p.get('probability_sha256') for p in row.values()) for row in profiles):
                    raise RuntimeError('missing complete per-window probability fingerprints')
                probability_fingerprint = stable_json_fingerprint([
                    {h:(p['queries'],p['probability_sha256']) for h,p in row.items()} for row in profiles])
                if expected_probability is None: expected_probability=probability_fingerprint
                if probability_fingerprint != expected_probability:
                    raise RuntimeError('speed trial changed probability BYTES; no speed result accepted')
            totals = defaultdict(float);worker = defaultdict(float);probability_stages=defaultdict(float)
            audits=defaultdict(int);readbacks=0;probability_verified=0
            for event in events:
                for field in ('input_wait_seconds','compute_seconds','reference_seconds','moving_support_seconds',
                    'candidate_wait_seconds','column_probability_seconds','composition_metrics_seconds'):
                    totals[field]+=event.get(field,0.)
                for field,value in event.get('raw_prefetch_worker_seconds',{}).items():
                    if field != 'geometry_workers': worker[field]+=value
                for profile in event.get('prediction_seconds_by_horizon',{}).values():
                    readbacks+=profile.get('probability_readback_transfers',0)
                    for field in ('inverse_map_seconds','sampling_wait_seconds','patch_gather_seconds','inverse_map_worker_seconds'):
                        probability_stages[field]+=profile.get(field,0.)
                    for field,value in profile.get('host_stages_seconds_NOT_cuda_kernel_time',{}).items():
                        probability_stages[field]+=value
                    for field in ('queries','compiled_patch_gather','horizon_inputs_budget_fallback'):
                        audits[field]+=profile.get(field,0)
                    audits['peak_horizon_inputs_bytes']=max(audits['peak_horizon_inputs_bytes'],profile.get('horizon_inputs_bytes',0))
                    audits['peak_horizon_inputs_working_bytes_bound']=max(audits['peak_horizon_inputs_working_bytes_bound'],
                        profile.get('horizon_inputs_working_bytes_bound',0))
                    probability_verified+=int(profile.get('probability_exactness_passed',False))
            trial=dict(name=name,repeat=repeat+1,seconds=seconds,seconds_per_window=seconds/windows,
                windows_per_second=windows/seconds,counts_fingerprint=fingerprint,
                raw_workers=raw_workers,buffered_readback=buffered,packed_upload=buffered and provider.device.type == 'cuda',readback_transfers=readbacks,
                stages_seconds_per_window={k:v/windows for k,v in totals.items()},
                column_probability_stages_seconds_per_window={k:v/windows for k,v in probability_stages.items()},
                column_audit=dict(audits),column_probability_optimized=column_fast,column_map_prefetch=map_prefetch,
                probability_verifications_in_timed_windows=probability_verified,
                probability_fingerprint=probability_fingerprint,
                raw_worker_OVERLAPPING_seconds_per_window={k:v/windows for k,v in worker.items()},
                peak_reserved_mib=torch.cuda.max_memory_reserved(provider.device)/2**20 if provider.device.type == 'cuda' else None)
            result['trials'].append(trial)
            print(f"SPEED {name} repeat={repeat+1}/{repeats} windows={windows} seconds/window={seconds/windows:.4f} counts_exact=PASS",flush=True)
            del fresh,report
      finally:
        for obj,values in ((model,original),(provider,original_provider)):
            for name,value in values.items():
                if value is None:
                    if hasattr(obj,name):delattr(obj,name)
                else:setattr(obj,name,value)
    aggregates = {}
    for name in settings:
        trials=[t for t in result['trials'] if t['name'] == name]
        aggregates[name]=dict(seconds_per_window=statistics.median(t['seconds_per_window'] for t in trials),
            stages_seconds_per_window={k:statistics.median(t['stages_seconds_per_window'][k] for t in trials)
                for k in trials[0]['stages_seconds_per_window']},
            raw_worker_OVERLAPPING_seconds_per_window={k:statistics.median(t['raw_worker_OVERLAPPING_seconds_per_window'][k] for t in trials)
                for k in trials[0]['raw_worker_OVERLAPPING_seconds_per_window']},
            column_probability_stages_seconds_per_window={k:statistics.median(t['column_probability_stages_seconds_per_window'][k] for t in trials)
                for k in trials[0]['column_probability_stages_seconds_per_window']})
    best=min(aggregates,key=lambda name:aggregates[name]['seconds_per_window'])
    result.update(status='complete',integer_counts_exact=True,aggregates=aggregates,fastest_measured=best)
    if column_suite:
        result.update(column_pipeline_speedup=aggregates['parallel_raw']['seconds_per_window']/aggregates['parallel_columns_prefetch']['seconds_per_window'],
            column_probability_speedup=aggregates['parallel_raw']['stages_seconds_per_window']['column_probability_seconds']/
                max(1e-12,aggregates['parallel_columns_prefetch']['stages_seconds_per_window']['column_probability_seconds']),
            no_automatic_backend_promotion=True,probability_bytes_exact=True,probability_fingerprint=expected_probability)
    else:
        result.update(raw_prefetch_speedup=aggregates['serial_raw']['seconds_per_window']/aggregates['parallel_raw']['seconds_per_window'],
            buffered_speedup=aggregates['parallel_raw']['seconds_per_window']/aggregates['parallel_buffered']['seconds_per_window'],
            combined_speedup=aggregates['serial_raw']['seconds_per_window']/aggregates['parallel_buffered']['seconds_per_window'])
    return result


def summary_text(result):
    lines=['===== LOCAL EVAL SPEED ONLY / NO SELECTION =====',
        f"actual_cuda={result['actual_cuda']} windows={result['windows']} scenes={result['scenes']} repeats={result['repeats']} batch={result['batch_size']}",
        f"integer_counts_exact={result['integer_counts_exact']} population_fp={result['population_key_fingerprint']}",
        ('parallel_columns = compiled byte patches + horizon metadata/source hoist; prefetch = CURRENT+NEXT CPU maps; original network batch/readback'
            if result.get('column_probability_suite') else
            'parallel_buffered = bounded probability readback + packed byte upload + horizon-level source finite check; network batch unchanged')]
    for name, row in result['aggregates'].items():
        lines.append(f"{name}: seconds/window={row['seconds_per_window']:.6f} windows/s={1/row['seconds_per_window']:.3f}")
        lines.append('  serial_host_stages='+json.dumps(row['stages_seconds_per_window'],sort_keys=True))
        lines.append('  raw_worker_OVERLAPPING='+json.dumps(row['raw_worker_OVERLAPPING_seconds_per_window'],sort_keys=True))
        if result.get('column_probability_suite'):
            lines.append('  column_probability_breakdown='+json.dumps(row['column_probability_stages_seconds_per_window'],sort_keys=True))
    newest=result['aggregates'][result['fastest_measured']]; stages=newest['stages_seconds_per_window']
    bottleneck=max(('input_wait_seconds','column_probability_seconds','candidate_wait_seconds',
        'composition_metrics_seconds','moving_support_seconds','reference_seconds'),key=lambda k:stages[k])
    lines += [(f"column_pipeline_speedup={result['column_pipeline_speedup']:.3f} column_probability_speedup={result['column_probability_speedup']:.3f}"
        if result.get('column_probability_suite') else
        f"raw_prefetch_speedup={result['raw_prefetch_speedup']:.3f} buffered_speedup={result['buffered_speedup']:.3f} combined_speedup={result['combined_speedup']:.3f}"),
        'fastest_measured='+result['fastest_measured'],
        'largest_measured_host_stage='+bottleneck,
        'OS/frame-cache and short-workload variation remain. No full ETA promise, weight/threshold change, training, promotion or full rerun.',
        'raw_worker_OVERLAPPING timings in speed.json are NOT additive with serial host stages.']
    if result.get('column_probability_suite'):
        breakdown = newest['column_probability_stages_seconds_per_window']
        serial_fields = ('inverse_map_seconds','sampling_wait_seconds','horizon_inputs','input_upload','source_inputs',
            'network_forward_host','calibration_host','probability_readback','finite_check_wait')
        column_bottleneck = max(serial_fields,key=lambda k:breakdown.get(k,0.))
        lines += [f"all_timed_windows_probability_bytes_exact={result['probability_bytes_exact']}",
            'largest_column_host_stage='+column_bottleneck,
            'patch_gather_seconds and inverse_map_worker_seconds are OVERLAPPING; network_forward_host is dispatch time, readback may include GPU wait.',
            'No automatic promotion. To use measured best in interim evaluator: '+
            {'parallel_raw':'FULL_JOINT_EVAL_COLUMN_OPTIMIZED=0 FULL_JOINT_EVAL_IO_OPTIMIZED=0',
             'parallel_columns':'FULL_JOINT_EVAL_COLUMN_OPTIMIZED=1 FULL_JOINT_EVAL_COLUMN_MAP_PREFETCH=0 FULL_JOINT_EVAL_IO_OPTIMIZED=0',
             'parallel_columns_prefetch':'FULL_JOINT_EVAL_COLUMN_OPTIMIZED=1 FULL_JOINT_EVAL_COLUMN_MAP_PREFETCH=1 FULL_JOINT_EVAL_IO_OPTIMIZED=0'}[result['fastest_measured']]]
    return '\n'.join(lines)+'\n'
