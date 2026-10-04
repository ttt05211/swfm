"""Window-major, bounded-RAM evaluation. Every live model is evaluated anew."""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import time

from real_motion.column_runtime_pipeline import prefetch_raw_columns
from tools.real_motion import causal_column_common as columns
from tools.real_motion.eval_p0_f9_v17_local_stwm import window_from_record


def evaluate_group(jobs, source, records, *, thresholds=(.5, .5, None), batch_size=256,
                   progress=None, stop_event=None, start_window=0, saved_states=None,
                   save_state=None, checkpoint_every=8, optimized=True, dev64_keys=None):
    """jobs = {name: (provider, columns_model)}; one current + one next raw window.

    The generators use the EXACT single-checkpoint metric/compositor path.
    Sharing is limited to raw history, fixed causal geometry, moving SUPPORT
    (metric-only), and frozen E14 counts. No learned candidate/features reused.
    Checkpointing commits all models at the same completed-window boundary.
    """
    if not jobs or not 0 <= start_window <= len(records) or checkpoint_every < 1:
        raise ValueError('invalid evaluation group/cursor')
    if saved_states is not None and set(saved_states) != set(jobs):
        raise RuntimeError('saved checkpoint group changed')
    if start_window and saved_states is None: raise RuntimeError('missing saved metric states')
    first = next(iter(jobs.values()))[0]
    shared = {}; counts = OrderedDict(); iterators = {}; states = {}
    totals = dict(raw_geometry_windows=0, moving_support_windows=0, model_windows=0,
                  checkpoint_compute_seconds={name: 0. for name in jobs}, shared_prepare_seconds=0.,
                  input_wait_seconds=0., all_model_window_seconds=0.)
    def rows():
        for record in records[start_window:]:
            key = (str(record['scene_name']), str(record['t0_token']))
            if key != shared['key']: raise RuntimeError('shared raw window identity/order mismatch')
            yield record, shared['raw']
    with ThreadPoolExecutor(max_workers=min(3, first.workers)) as pool:
        for name, (provider, model) in jobs.items():
            if hasattr(provider, 'frozen_metric_counts'): provider.frozen_metric_counts = counts
            model.column_inference_optimized = optimized
            model.column_inference_verify_remaining = 3 if optimized else 0
            iterators[name] = columns.evaluation_steps(provider, source, records, model, thresholds,
                batch_size=batch_size, diagnostic_thresholds=None, candidate_pool=pool,
                feature_backend='cpu', stop_event=stop_event, window_iter=rows(),
                resume_state=saved_states[name] if saved_states is not None else None,
                start_window=start_window, verbose=False, dev64_keys=dev64_keys, yield_initial_state=True)
            _, states[name] = next(iterators[name])
        if save_state:
            save_state(start_window, {n: columns.pack_evaluation_state(s) for n, s in states.items()}, totals)
        raw_iterator = prefetch_raw_columns(first, source, records[start_window:])
        try:
            previous_end = time.perf_counter()
            for wi, (record, raw) in enumerate(raw_iterator, start_window+1):
                if stop_event is not None and stop_event.is_set(): raise InterruptedError('group stopped at window boundary')
                if raw is None: raise RuntimeError('shared evaluation requires causal raw look-ahead provider')
                tick = time.perf_counter()
                input_wait = tick-previous_end; totals['input_wait_seconds'] += input_wait
                window = window_from_record(record)
                raw['_evaluation_moving_support'] = columns.gt_moving_support_sequence(source.nusc,
                    window.t0_token, window.future_tokens, tuple(.5*(h+1) for h in range(6)),
                    grid=first.pcfg.grid, workers=first.workers)
                shared.update(raw=raw, key=(str(record['scene_name']), str(record['t0_token'])))
                preparation = time.perf_counter()-tick
                totals['shared_prepare_seconds'] += preparation
                totals['raw_geometry_windows'] += 1; totals['moving_support_windows'] += 1
                for name, iterator in iterators.items():
                    event, state = next(iterator)
                    states[name] = state
                    totals['checkpoint_compute_seconds'][name] += event['compute_seconds']
                    totals['model_windows'] += 1
                    if progress: progress({**event, 'checkpoint': name, 'timing_scope': 'checkpoint_compute_only'})
                wall = time.perf_counter()-tick+input_wait
                totals['all_model_window_seconds'] += wall
                if save_state and (wi % checkpoint_every == 0 or wi == len(records)):
                    save_state(wi, {n: columns.pack_evaluation_state(s) for n, s in states.items()}, totals)
                if progress: progress(dict(event='group_window_complete', window=wi, windows=len(records), models=len(jobs),
                    seconds=wall, input_wait_seconds=input_wait, shared_metric_support_seconds=preparation))
                if stop_event is not None and stop_event.is_set():
                    if save_state: save_state(wi, {n: columns.pack_evaluation_state(s) for n, s in states.items()}, totals)
                    raise InterruptedError('group stopped at completed all-model window')
                previous_end = time.perf_counter()
            reports = {}
            for name, iterator in iterators.items():
                try: next(iterator)
                except StopIteration as done: reports[name] = done.value
                else: raise RuntimeError('incomplete synchronized checkpoint evaluation')
            return reports, totals
        finally:
            raw_iterator.close()
            for iterator in iterators.values(): iterator.close()


def rank_reports(reports):
    rows = []
    for name, report in reports.items():
        row = report['all']; joint = row['variants']['joint']['metrics']
        reference = row['reference_metrics'].get('frozen_E14')
        delta = joint['MovingMicro']-reference['MovingMicro'] if reference else None
        rows.append(dict(checkpoint=name, mIoU=joint['mIoU'], IoU=joint['IoU'], MovingMicro=joint['MovingMicro'],
            delta_mIoU_vs_transport=joint['mIoU']-row['baseline']['mIoU'],
            delta_MovingMicro_vs_E14=delta, moving_nonnegative_vs_E14=delta is not None and delta >= -1e-10))
    import math
    if any(not math.isfinite(r[k]) for r in rows for k in ('mIoU', 'IoU', 'MovingMicro')):
        raise RuntimeError('nonfinite ranking metric; do not select a checkpoint')
    rows.sort(key=lambda r: (-r['mIoU'], -r['MovingMicro'], r['checkpoint']))
    eligible = [r for r in rows if r['moving_nonnegative_vs_E14']]
    return dict(by_mIoU=rows, moving_safe_best=eligible[0]['checkpoint'] if eligible else None,
        overall_mIoU_best=rows[0]['checkpoint'],
        rule='highest joint mIoU among retained dev64-shortlisted candidates with nonnegative aggregate MovingMicro vs E14',
        note='No automatic deployment. E14 is legacy six-history reference; full4369 includes selection dev512.')
