"""Fixed-weight dev calibration: ONE probability pass, exact layered threshold grid."""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import itertools
import time
import numpy as np

from real_motion.column_runtime_pipeline import prefetch_raw_columns
from real_motion.causal_column_sampling import ColumnHistoryIndex
from tools.real_motion import causal_column_common as col

PROTOCOL = 'p0_f9_joint_dev512_threshold_calibration_v1'
LEVELS = (.5, .75, .95, None)
FIXED = (.5, .5, None)


def specifications():
    jobs = [(f'g{i}', (g, None, None), 'generation') for i, g in enumerate(LEVELS)]
    jobs += [(f'r{i}', (None, a, r), 'refine') for i, (a, r) in enumerate(itertools.product(LEVELS, repeat=2))]
    jobs += [(f'j{i}', gates, 'joint') for i, gates in enumerate(itertools.product(LEVELS, repeat=3))]
    return jobs


def proposals(plan, probability, *, verify=False):
    """Factor 64 joint compositions into 4 generation + 16 refinement passes.

    The deployed compositor refines first, then generates only in STILL-free
    cells. No GT enters this composition. Original row/source order is kept.
    """
    layout = col.sparse_layout(plan); before = layout[2]
    generation, refinement, removes = {}, {}, {}
    for i, g in enumerate(LEVELS):
        actions = col.actions_from_probabilities(plan, probability, (g, None, None))
        generation[g] = col.compose_sparse(plan, actions, enable_refine=False, layout=layout)[2]
        yield f'g{i}', layout, generation[g], 0
    for i, (a, r) in enumerate(itertools.product(LEVELS, repeat=2)):
        actions = col.actions_from_probabilities(plan, probability, (None, a, r))
        refinement[a, r] = col.compose_sparse(plan, actions, enable_generation=False, layout=layout)[2]
        removes[a, r] = int(((actions == col.REMOVE) & (plan.actor[:, None] >= 0)).sum())
        yield f'r{i}', layout, refinement[a, r], removes[a, r]
    for i, (g, a, r) in enumerate(itertools.product(LEVELS, repeat=3)):
        after = refinement[a, r].copy()
        take = (after == col.FREE) & (generation[g] != before)
        after[take] = generation[g][take]
        if verify:
            actions = col.actions_from_probabilities(plan, probability, (g, a, r))
            reference = col.compose_sparse(plan, actions, layout=layout)[2]
            if not np.array_equal(after, reference):
                raise RuntimeError('factorized threshold composition differs from deployed renderer')
        yield f'j{i}', layout, after, removes[a, r]


def initial_state():
    names = [name for name, _, _ in specifications()]
    return dict(bases={'all': col.Metrics()}, metrics={'all': {n: col.Metrics() for n in names}},
        quality={'all': {n: defaultdict(int) for n in names}},
        scenes={'all': defaultdict(lambda: {n: col.Metrics() for n in ('baseline', *names)})},
        counts_windows=defaultdict(int), reference_metrics={'all': {}}, audits=defaultdict(int), scores={'all': {}})


def summarize(state):
    reports = col.report_states(state['bases']['all'], state['metrics']['all'], state['quality']['all'], state['scenes']['all'])
    jobs = specifications(); fixed_name = next(n for n, g, kind in jobs if kind == 'joint' and g == FIXED)
    fixed = reports['variants'][fixed_name]['metrics']
    if not all(np.isfinite(fixed[k]) for k in ('mIoU', 'IoU', 'MovingMicro')):
        raise RuntimeError('nonfinite fixed-threshold metrics; calibration cannot select a result')
    candidates = []
    for name, gates, kind in jobs:
        if kind != 'joint': continue
        row = reports['variants'][name]; metrics = row['metrics']; d = col.delta(metrics, fixed)
        moving = [d['MovingMicro'], *(v['MovingMicro'] for v in d['per_horizon'].values())]
        checks = dict(mIoU_nonnegative_vs_fixed=np.isfinite(d['mIoU']) and d['mIoU'] >= -1e-10,
            MovingMicro_aggregate_and_all_horizons_nonnegative_vs_fixed=all(np.isfinite(v) and v >= -1e-10 for v in moving))
        scene_delta = {s: values[name].compute()['mIoU']-values[fixed_name].compute()['mIoU']
                       for s, values in state['scenes']['all'].items()}
        candidates.append(dict(name=name, thresholds=gates, metrics=metrics, delta_vs_fixed_pp=d,
            quality=row['quality'], safety=checks, eligible=all(checks.values()),
            scene_delta_vs_fixed=dict(positive=sum(v > 0 for v in scene_delta.values()),
                negative=sum(v < 0 for v in scene_delta.values()), zero=sum(v == 0 for v in scene_delta.values()), by_scene=scene_delta)))
    def descending(value): return -value if np.isfinite(value) else float('inf')
    candidates.sort(key=lambda r: (descending(r['metrics']['mIoU']), descending(r['metrics']['MovingMicro']), r['name'] != fixed_name,
        r['quality'].get('removed', 0), r['name']))
    eligible = [r for r in candidates if r['eligible']]
    best = eligible[0]
    # Do not advertise a different threshold when it has no actual mIoU gain.
    if best['delta_vs_fixed_pp']['mIoU'] <= 1e-10: best = next(r for r in candidates if r['name'] == fixed_name)
    g, a, r = best['thresholds']
    gn = next(n for n, gates, kind in jobs if kind == 'generation' and gates == (g, None, None))
    rn = next(n for n, gates, kind in jobs if kind == 'refine' and gates == (None, a, r))
    references = {n: v.compute() for n, v in state['reference_metrics']['all'].items()}
    return dict(windows=state['counts_windows']['all'], scenes=len(state['scenes']['all']), baseline=reports['baseline'],
        reference_metrics=references, fixed=next(r for r in candidates if r['name'] == fixed_name),
        selected=best, overall_mIoU_best=candidates[0], candidates=candidates,
        selected_branches={n: reports['variants'][key] for n, key in (('generation', gn), ('refine', rn), ('joint', best['name']))},
        rule='maximize joint mIoU subject to aggregate AND 1/2/3s MovingMicro not below fixed 0.5/0.5/REMOVE-off',
        route='threshold_gain_on_calibration_only' if best['name'] != fixed_name else 'keep_fixed_no_safe_gain',
        note='dev512 already selected checkpoint; calibration/in-sample selection, NOT independent test or promotion')


def sweep(provider, source, records, model, *, batch_size=256, progress=None, stop_event=None,
          start_window=0, saved_state=None, save_state=None, checkpoint_every=8):
    if not 0 <= start_window <= len(records) or checkpoint_every < 1 or batch_size < 1:
        raise ValueError('invalid sweep cursor/budgets')
    if start_window and saved_state is None: raise RuntimeError('missing saved threshold counts')
    state = initial_state()
    if saved_state is not None: col.restore_evaluation_state(state, saved_state)
    if state['counts_windows'].get('all', 0) != start_window: raise RuntimeError('saved threshold cursor mismatch')
    model.column_sampling_workers = provider.workers
    model.column_inference_optimized = True; model.column_inference_verify_remaining = 3
    totals = dict(probability_horizons=0, threshold_settings=64, branch_settings=20,
        prepare_seconds=0., probability_seconds=0., composition_metrics_seconds=0., input_wait_seconds=0.,
        composition_exactness_horizons=0, metric_reuse_hits=0, metric_distinct_predictions=0)
    if save_state: save_state(start_window, col.pack_evaluation_state(state), totals)
    iterator = prefetch_raw_columns(provider, source, records[start_window:])
    try:
      with ThreadPoolExecutor(max_workers=min(3, provider.workers)) as pool:
        previous_end = time.perf_counter()
        for wi, (record, raw) in enumerate(iterator, start_window+1):
            if stop_event is not None and stop_event.is_set(): raise InterruptedError('threshold sweep stopped at window boundary')
            tick = time.perf_counter(); wait = tick-previous_end; totals['input_wait_seconds'] += wait
            prep = provider.prepare_columns(source, record, include_gt=True, raw_window=raw)
            references = provider.reference_predictions(prep, record)
            moving = col.gt_moving_support_sequence(source.nusc, prep.window.t0_token, prep.window.future_tokens,
                tuple(.5*(h+1) for h in range(6)), grid=provider.pcfg.grid, workers=provider.workers)
            prep.column_history_index = ColumnHistoryIndex(prep, provider.pcfg.grid)
            planning = {h: pool.submit(col.candidate_plan, prep, h, provider.pcfg.grid, model.config) for h in col.REPORT}
            totals['prepare_seconds'] += time.perf_counter()-tick
            scene = state['scenes']['all'][prep.window.scene_name]
            for ri, h in enumerate(col.REPORT):
                before_probability = time.perf_counter(); plan = planning[h].result()
                probability = col.predict_probabilities(model, prep, h, plan, provider.pcfg.grid, provider.device, batch_size)
                totals['probability_seconds'] += time.perf_counter()-before_probability; totals['probability_horizons'] += 1
                metrics_tick = time.perf_counter()
                gt = np.asarray(prep.raw['future_gt_occ'][h]); mask = moving[h][0]
                base_counts = col.Metrics.counts(prep.baseline[h], gt, mask, col.FREE)
                state['bases']['all'].update(ri, counts=base_counts); scene['baseline'].update(ri, counts=base_counts)
                for name, predictions in references.items():
                    state['reference_metrics']['all'].setdefault(name, col.Metrics()).update(ri,
                        counts=col.Metrics.counts(predictions[h], gt, mask, col.FREE))
                target = support = None; metric_cache = {}
                for name, layout, after, remove_decisions in proposals(plan, probability, verify=wi == start_window+1):
                    _, ids, before, _ = layout
                    # All settings share the same sparse support: gather GT
                    # only once for metric counting, never as model input.
                    if target is None: target, support = gt.reshape(-1)[ids], mask.reshape(-1)[ids]
                    # Many thresholds produce identical voxels. Cache only
                    # THIS horizon's integer counts; equality guards hash
                    # collisions, and source-layer decisions remain separate.
                    fingerprint = hash(after.tobytes())
                    cached = next((v for v in metric_cache.get(fingerprint, ())
                                   if np.array_equal(v[0], after)), None)
                    if cached is None:
                        counts = col.sparse_counts(base_counts, before, after, target, support, col.DYN)
                        quality = col.edit_quality(before, after, target)
                        metric_cache.setdefault(fingerprint, []).append((after, counts, quality))
                        totals['metric_distinct_predictions'] += 1
                    else:
                        _, counts, quality = cached; totals['metric_reuse_hits'] += 1
                    if wi == start_window+1 and name in ('j0', 'j3', 'j63'):
                        dense = prep.baseline[h].copy(); dense.reshape(-1)[ids] = after
                        exact = col.Metrics.counts(dense, gt, mask, col.FREE)
                        if any(not np.array_equal(a, b) for a, b in zip(counts, exact)):
                            raise RuntimeError('threshold sparse/full metric exactness failed')
                    state['metrics']['all'][name].update(ri, counts=counts); scene[name].update(ri, counts=counts)
                    quality = {**quality, 'source_layer_REMOVE_decisions': remove_decisions}
                    for key, value in quality.items(): state['quality']['all'][name][key] += value
                if wi == start_window+1: totals['composition_exactness_horizons'] += 1
                totals['composition_metrics_seconds'] += time.perf_counter()-metrics_tick
            state['counts_windows']['all'] += 1
            for key, value in prep.source_audit.items(): state['audits'][key] += value
            stopped = stop_event is not None and stop_event.is_set()
            if save_state and (wi % checkpoint_every == 0 or wi == len(records) or stopped):
                save_state(wi, col.pack_evaluation_state(state), totals)
            if progress: progress(dict(event='threshold_window_complete', window=wi, windows=len(records),
                seconds=time.perf_counter()-tick+wait, performance=dict(totals)))
            if stop_event is not None and stop_event.is_set():
                if save_state: save_state(wi, col.pack_evaluation_state(state), totals)
                raise InterruptedError('threshold sweep stopped at complete-window boundary')
            previous_end = time.perf_counter()
    finally: iterator.close()
    return summarize(state), totals
