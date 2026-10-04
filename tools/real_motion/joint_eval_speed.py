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
                         windows=32, repeats=2, batch_size=256, stop_event=None):
    rows, keys = speed_records(records, windows)
    workers = min(4,provider.workers)
    # No candidate/patch/network batch change. SAME immutable CPU frame-cache
    # budget per trial. Only raw window concurrency and output readback differ.
    modes = [('serial_raw',1,False),('parallel_raw',workers,False),('parallel_buffered',workers,True)]
    settings = {name:(raw,buffered) for name,raw,buffered in modes}
    result = dict(status='running',windows=windows,scenes=len({k[0] for k in keys}),keys=keys,
        population_key_fingerprint=stable_json_fingerprint(keys),repeats=repeats,
        batch_size=batch_size,actual_cuda=provider.device.type == 'cuda',trials=[],
        warmup_windows_per_trial=min(2,windows),frame_cache_mib=source.limit/2**20,
        no_persistent_geometry_writes=True,no_weight_or_threshold_selection=True)
    expected = None
    original = {name:getattr(model,name,None) for name in ('column_inference_optimized',
        'column_inference_verify_remaining','column_readback_optimized')}
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
            raw_workers, buffered = settings[name]
            provider.raw_prefetch_workers=provider.raw_prefetch_depth=raw_workers
            provider.raw_io_workers=max(1,min(4,provider.workers//raw_workers))
            if hasattr(provider,'frozen_metric_counts'):provider.frozen_metric_counts.clear()
            fresh = CachedColumnSource(source.source,source.limit/2**20)
            model.column_inference_optimized=True;model.column_readback_optimized=buffered
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
            totals = defaultdict(float);worker = defaultdict(float);readbacks=0
            for event in events:
                for field in ('input_wait_seconds','compute_seconds','reference_seconds','moving_support_seconds',
                    'candidate_wait_seconds','column_probability_seconds','composition_metrics_seconds'):
                    totals[field]+=event.get(field,0.)
                for field,value in event.get('raw_prefetch_worker_seconds',{}).items():
                    if field != 'geometry_workers': worker[field]+=value
                for profile in event.get('prediction_seconds_by_horizon',{}).values():
                    readbacks+=profile.get('probability_readback_transfers',0)
            trial=dict(name=name,repeat=repeat+1,seconds=seconds,seconds_per_window=seconds/windows,
                windows_per_second=windows/seconds,counts_fingerprint=fingerprint,
                raw_workers=raw_workers,buffered_readback=buffered,packed_upload=buffered and provider.device.type == 'cuda',readback_transfers=readbacks,
                stages_seconds_per_window={k:v/windows for k,v in totals.items()},
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
                for k in trials[0]['raw_worker_OVERLAPPING_seconds_per_window']})
    best=min(aggregates,key=lambda name:aggregates[name]['seconds_per_window'])
    result.update(status='complete',integer_counts_exact=True,aggregates=aggregates,fastest_measured=best,
        raw_prefetch_speedup=aggregates['serial_raw']['seconds_per_window']/aggregates['parallel_raw']['seconds_per_window'],
        buffered_speedup=aggregates['parallel_raw']['seconds_per_window']/aggregates['parallel_buffered']['seconds_per_window'],
        combined_speedup=aggregates['serial_raw']['seconds_per_window']/aggregates['parallel_buffered']['seconds_per_window'])
    return result


def summary_text(result):
    lines=['===== LOCAL EVAL SPEED ONLY / NO SELECTION =====',
        f"actual_cuda={result['actual_cuda']} windows={result['windows']} scenes={result['scenes']} repeats={result['repeats']} batch={result['batch_size']}",
        f"integer_counts_exact={result['integer_counts_exact']} population_fp={result['population_key_fingerprint']}",
        'parallel_buffered = bounded probability readback + packed byte upload + horizon-level source finite check; network batch unchanged']
    for name, row in result['aggregates'].items():
        lines.append(f"{name}: seconds/window={row['seconds_per_window']:.6f} windows/s={1/row['seconds_per_window']:.3f}")
        lines.append('  serial_host_stages='+json.dumps(row['stages_seconds_per_window'],sort_keys=True))
        lines.append('  raw_worker_OVERLAPPING='+json.dumps(row['raw_worker_OVERLAPPING_seconds_per_window'],sort_keys=True))
    newest=result['aggregates']['parallel_buffered']; stages=newest['stages_seconds_per_window']
    bottleneck=max(('input_wait_seconds','column_probability_seconds','candidate_wait_seconds',
        'composition_metrics_seconds','moving_support_seconds','reference_seconds'),key=lambda k:stages[k])
    lines += [f"raw_prefetch_speedup={result['raw_prefetch_speedup']:.3f} buffered_speedup={result['buffered_speedup']:.3f} combined_speedup={result['combined_speedup']:.3f}",
        'fastest_measured='+result['fastest_measured'],
        'largest_measured_host_stage='+bottleneck,
        'OS/frame-cache and short-workload variation remain. No full ETA promise, weight/threshold change, training, promotion or full rerun.',
        'raw_worker_OVERLAPPING timings in speed.json are NOT additive with serial host stages.']
    return '\n'.join(lines)+'\n'
