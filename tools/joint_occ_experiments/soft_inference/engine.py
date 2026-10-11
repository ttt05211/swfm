"""Independent readouts of the SAME frozen model, using the training surrogate.

Quality retains the original GT cache route. Timing rebuilds Strong every call
and checks all six outputs against that route before reporting valid latency.
No future labels, optimizer, loss, or gradients are used by prediction/timing.
"""
import time
from types import SimpleNamespace
import numpy as np
import torch
from real_motion.surface_canonical_repair import augment_projection
from real_motion.canonical_causal_repair import compose_canonical
from ..engine import forward_window, hard_predictions
from ..data import prepare_projection
from ..soft_geometry import source_targets, effective_yaw, continuous_context
from ..batching.geometry import GroupedScene
from real_motion.final_dataflow import build_causal_motion_prior, _stage_history_inputs
from tools.real_motion import benchmark_p0_f9_v18_runtime as runtime
from real_motion.runtime_fastpath import compose_component_replacements_fast_exact
from real_motion.metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS
from real_motion.strong_majority_execution import strong_majority_execution
from real_motion.motion_transport_v2 import SOURCE_XY_FEATURE_SCALE_M

ROUTES = ('hard_transport', 'hard_add_only', 'hard_add_remove',
          'soft_transport', 'soft_add_only', 'soft_add_remove')
SPEED_ROUTES = ('hard_add_remove', 'soft_add_only', 'soft_add_remove')
OPTIMIZED_SPEED_ROUTES = SPEED_ROUTES+('soft_add_only_fast', 'soft_add_remove_fast')
TIMING_BOUNDARY = 'CausalHistoryState -> fresh Strong/KTA + live motion + CCR + projection + SIX dense CPU outputs'


def live_prior(record, history, frame_dt_s):
    kta, anchors = build_causal_motion_prior(history, frame_dt_s)
    if getattr(history, 'kta_encoding', 'physical') == 'legacy_upgrade':
        # Exact upgrade_v1_record_targets arithmetic, not an epsilon waiver.
        # Keep the representation this frozen model actually consumed.
        features = torch.as_tensor(record['features']).detach().cpu().float().numpy()
        source = features[:, :2].astype(np.float64)*SOURCE_XY_FEATURE_SCALE_M
        kta = (anchors.astype(np.float64)-source[:, None]).astype(np.float32)
    return kta, anchors


def recognize_prior_encoding(record, history, frame_dt_s):
    """History-only compatibility check OUTSIDE timing; never use cached KTA live."""
    def array(key): return torch.as_tensor(record[key]).detach().cpu().float().numpy()
    kta, anchors = live_prior(record, history, frame_dt_s)
    if not np.array_equal(anchors, array('anchors_xy_t0_m')):
        raise RuntimeError('live causal anchors differ from frozen quality inputs')
    if np.array_equal(kta, array('kta_displacement_xy_m')): return 'physical'
    history.kta_encoding = 'legacy_upgrade'
    upgraded, _ = live_prior(record, history, frame_dt_s)
    if not np.array_equal(upgraded, array('kta_displacement_xy_m')):
        raise RuntimeError('KTA differs from BOTH exact known causal encodings')
    return 'legacy_upgrade'


def decode_soft(model, provider, fw):
    """The unmodified dense continuous-context head used by the recent audit."""
    evidence, plan = fw['evidence'], fw['plan']
    mapped, context = continuous_context(evidence, plan, fw['prep'].state,
        fw['targets'], fw['yaws'], fw['poses'], provider.pcfg.grid)
    upload = lambda value: torch.as_tensor(np.array(value, copy=True), device=provider.device)
    actor = upload(evidence.actor).long()
    with torch.autocast(device_type=provider.device.type, dtype=torch.bfloat16,
                        enabled=provider.device.type == 'cuda'):
        linked = model.columns.project_sources(fw['output'])
        encoded = model.columns.encode(upload(evidence.features), upload(evidence.labels), actor,
                                       upload(evidence.classes), linked)
        logits = model.columns.decode(encoded, actor, context, upload(plan.base),
                                      upload(plan.fallback), upload(plan.legal), linked)
    return {**fw, 'mapped': mapped, 'actions': model.columns.probabilities(logits, actor)}


def soft_predictions(provider, fw, variants, horizons):
    scene = GroupedScene(fw['prep'], provider.pcfg.grid, provider.device)
    actions = fw['actions'].to(provider.device)
    result = {}
    for name in variants:
        if name not in ('transport', 'add_only', 'add_remove'): raise ValueError('unknown soft variant')
        current = (torch.stack((actions[..., 0], torch.zeros_like(actions[..., 1])), -1)
                   if name == 'add_only' else actions)
        values = []
        for h in horizons:
            p = scene.render_group([h], poses=fw['poses'], targets=fw['targets'], yaws=fw['yaws'],
                mapped=fw['mapped'], actions=current, evidence=fw['evidence'], plan=fw['plan'],
                edits=name != 'transport')
            values.append(p[0].argmax(0).reshape(scene.shape).byte().cpu().numpy())
            del p
        result[name] = values
    return result


@torch.no_grad()
def quality_predictions(model, provider, row, *, remove_threshold=.95, horizons=(1, 3, 5)):
    fw = forward_window(model, provider, row, 'gt', soft=False)
    hard = hard_predictions(fw, remove_threshold=remove_threshold)
    soft = soft_predictions(provider, decode_soft(model, provider, fw),
                            ('transport', 'add_only', 'add_remove'), horizons)
    return {**{'hard_'+name: [values[h] for h in horizons] for name, values in hard.items()},
            **{'soft_'+name: values for name, values in soft.items()}}


def prepare_speed_row(provider, row):
    """History-only extraction/matching/H2D, OUTSIDE the old FPS boundary.

    Keep no Strong backgrounds, KTA future priors, learned outputs or labels.
    Identical causal velocity matcher and history staging to old formal FPS.
    """
    record, raw, inputs, history = row
    tick = time.perf_counter()
    state = history[0]['state']
    previous = runtime.extract_instances_cropped_exact(raw['history_occ'][-2], raw['history_poses'][-2],
        grid=provider.pcfg.grid, cfg=provider.strong)
    velocities = runtime.match_instances(previous, state['current'], float(provider.pcfg.frame_dt_s),
        max_speed_mps=provider.strong.max_match_speed_mps)
    h = SimpleNamespace(current=state['current'], current_pose=state['current_pose'],
        source_count=len(state['current']), velocities=velocities)
    h.kta_encoding = recognize_prior_encoding(record, h, provider.pcfg.frame_dt_s)
    staged = _stage_history_inputs(record, provider.device)
    sync(provider.device)
    raw_view = {**raw, '_speed_history': h, '_speed_gpu_history': staged}
    return (record, raw_view, inputs, history), time.perf_counter()-tick


def profiled(stats, device, name, function):
    if stats is None: return function()
    sync(device); begun = time.perf_counter()
    value = function()
    sync(device); stats[name] = stats.get(name, 0.)+time.perf_counter()-begun
    return value


def fresh_forward(model, provider, row, *, soft, profile=None):
    """Prepared causal history -> FRESH Strong + WM + projection + CCR.

    GT ego is the declared condition. The independently supervised ego head is
    unused in this GT route. Cached Strong/backgrounds and predictions are NOT
    consumed, even though read-only history association/descriptors are reused.
    """
    record, raw, _, history = row
    if '_speed_history' not in raw: raise RuntimeError('prepare causal history OUTSIDE inference timer first')
    # Forecast-dependent KTA is LIVE and included, as in old forecast_six.
    kta, anchors = live_prior(record, raw['_speed_history'], provider.pcfg.frame_dt_s)
    staged = raw['_speed_gpu_history']
    live_record = {**record, 'kta_displacement_xy_m': torch.from_numpy(kta),
        'anchors_xy_t0_m': torch.from_numpy(anchors), 'features': staged['features'],
        'local_semantic_tube': staged['tube'], 'frame_motion_features': staged['frame_motion'],
        'target_source_mask_tube': staged['source_mask']}
    output = profiled(profile, provider.device, 'motion', lambda: model.motion(live_record, provider.device))
    poses = torch.as_tensor(np.asarray(raw['future_poses']), device=provider.device, dtype=torch.float64)
    state, h = history[0]['state'], raw['_speed_history']
    def build_backgrounds():
        with strong_majority_execution(getattr(provider, 'strong_majority', None)):
            strong, components = runtime._strong_all_horizons(raw['history_occ'][-1], state['current_pose'], raw['future_poses'],
                state['current'], h.velocities, state['source_world_points'], frame_dt_s=provider.pcfg.frame_dt_s,
                grid=provider.pcfg.grid, cfg=provider.strong, runtime_device=provider.device,
                majority_backend='native' if getattr(provider, 'strong_majority', None) is not None else 'sparse_cuda')
        return [compose_component_replacements_fast_exact(a, c, [],
            dynamic_class_ids=DYNAMIC_CLASS_IDS, free_label=17, grid=provider.pcfg.grid,
            precomputed_clear_flat_indices=runtime.baseline_clear_flat_indices(c, grid=provider.pcfg.grid))
            for a, c in zip(strong, components)]
    backgrounds = profiled(profile, provider.device, 'fresh_Strong_CLEAR', build_backgrounds)
    # A private transient view passes JUST-computed backgrounds to the common
    # projection helper, avoiding its historical re-extraction. Never read or
    # overwrite the persisted/cached backgrounds from the original raw window.
    view = {**raw, '_column_causal_preparation': {'prepared_state': {'column_backgrounds': backgrounds}}}
    prep, evidence, plan = profiled(profile, provider.device, 'hard_projection_owner_plan', lambda:
        prepare_projection(provider, live_record, view, output, poses.detach().cpu().numpy(), history, use_gt_cache=True))
    fw = dict(output=output, poses=poses, prep=prep, evidence=evidence, plan=plan,
        targets=source_targets(output, live_record, prep.state['current_pose'], prep.state['source_z_t0']),
        yaws=effective_yaw(output, [int(c['class_id']) for c in prep.state['current']]))
    if soft: return profiled(profile, provider.device, 'CCR_decode', lambda: decode_soft(model, provider, fw))
    enriched = augment_projection(evidence, plan, prep.state['current_pose'],
                                  prep.state['world_to_future'], provider.pcfg.grid)
    session = provider.paired_eval_sessions[id(model.columns)]
    fw['actions'] = torch.from_numpy(session(model.columns, evidence, enriched, output, provider.device))
    return fw


@torch.no_grad()
def infer(model, provider, row, route, *, remove_threshold=.95, fast_horizons=3, profile=None):
    if route not in OPTIMIZED_SPEED_ROUTES: raise ValueError('unknown inference timing route')
    fw = fresh_forward(model, provider, row, soft=route.startswith('soft_'), profile=profile)
    if route == 'hard_add_remove':
        actions = fw['actions'].float().cpu().numpy()
        return compose_canonical(fw['prep'].baseline, fw['evidence'], fw['plan'],
                                 actions[..., 0], actions[..., 1], thresholds=(.5, remove_threshold))
    variant = route.removeprefix('soft_').removesuffix('_fast')
    if route.endswith('_fast'):
        from .fast import predict
        return profiled(profile, provider.device, 'soft_render_argmax_transfer',
                        lambda: predict(provider, fw, variant, range(6), group_size=fast_horizons))
    return profiled(profile, provider.device, 'soft_render_argmax_transfer',
                    lambda: soft_predictions(provider, fw, (variant,), range(6))[variant])


def sync(device):
    if device.type == 'cuda': torch.cuda.synchronize(device)


def mismatch_voxels(expected, actual):
    if len(expected) != 6 or len(actual) != 6: raise RuntimeError('timing must produce ALL SIX frames')
    if any(a.shape != b.shape for a, b in zip(expected, actual)): raise RuntimeError('inference volume differs')
    return sum(int(np.count_nonzero(a != b)) for a, b in zip(expected, actual))


@torch.no_grad()
def measure_window(model, provider, row, index, *, repeats=2, remove_threshold=.95,
                   compare_optimized=False, fast_horizons=3):
    if repeats < 2: raise ValueError('at least TWO alternating timing passes required')
    expected = quality_predictions(model, provider, row, remove_threshold=remove_threshold, horizons=range(6))
    row, preparation_seconds = prepare_speed_row(provider, row)
    routes = OPTIMIZED_SPEED_ROUTES if compare_optimized else SPEED_ROUTES
    # Warm EACH route on this window outside measurement, including graph
    # capture/verification and immutable coordinate initialization. Learned
    # projections, Strong, head probabilities and six outputs are rebuilt.
    mismatches, samples = {}, []
    for route in routes:
        actual = infer(model, provider, row, route, remove_threshold=remove_threshold, fast_horizons=fast_horizons)
        sync(provider.device)
        mismatches[route] = mismatch_voxels(expected[route.removesuffix('_fast')], actual)
        del actual
    for repeat in range(repeats):
        order = routes if (index+repeat) % 2 == 0 else tuple(reversed(routes))
        for route in order:
            sync(provider.device)
            if provider.device.type == 'cuda': torch.cuda.reset_peak_memory_stats(provider.device)
            begun = time.perf_counter()
            actual = infer(model, provider, row, route, remove_threshold=remove_threshold, fast_horizons=fast_horizons)
            sync(provider.device)
            seconds = time.perf_counter()-begun
            mismatch = mismatch_voxels(expected[route.removesuffix('_fast')], actual)
            mismatches[route] += mismatch
            samples.append(dict(route=route, repeat=repeat, seconds=seconds,
                mismatched_voxels=mismatch,
                peak_allocated_mib=(torch.cuda.max_memory_allocated(provider.device)/2**20
                                    if provider.device.type == 'cuda' else None),
                peak_reserved_mib=(torch.cuda.max_memory_reserved(provider.device)/2**20
                                   if provider.device.type == 'cuda' else None),
                device_total_mib=(torch.cuda.get_device_properties(provider.device).total_memory/2**20
                                  if provider.device.type == 'cuda' else None)))
            del actual
    details = {}
    if compare_optimized and index == 0:
        # Separate serialized profiling pass AFTER warm timing, not included in
        # throughput. Synchronization makes these stages non-overlapping.
        for route in routes:
            details[route] = {}
            actual = infer(model, provider, row, route, remove_threshold=remove_threshold,
                fast_horizons=fast_horizons, profile=details[route])
            mismatches[route] += mismatch_voxels(expected[route.removesuffix('_fast')], actual)
    return dict(key=[row[0]['scene_name'], row[0]['t0_token']], samples=samples,
                serialized_profile_seconds=details,
                kta_encoding=getattr(row[1].get('_speed_history'), 'kta_encoding', None),
                excluded_history_prepare_seconds=preparation_seconds,
                mismatched_voxels=mismatches, six_frame_byte_parity=not any(mismatches.values()))


def speed_summary(windows, *, actual_cuda, routes=SPEED_ROUTES):
    summary = dict(windows=len(windows), actual_cuda=actual_cuda,
        six_frame_byte_parity=bool(windows) and all(w['six_frame_byte_parity'] for w in windows),
        scope=TIMING_BOUNDARY, timing_boundary_version=2,
        excluded_history_extract_match_h2d_seconds=sum(w.get('excluded_history_prepare_seconds', 0.) for w in windows),
        excludes=['disk I/O', 'history association/representation/descriptors', 'labels/metrics',
                  'warmup/graph capture', 'checkpoint writes'],
        notes='paired WARM diagnostic latency, NOT old-model FPS or raw-input end-to-end FPS; reserved memory includes persistent allocator/graphs',
        routes={})
    for route in routes:
        samples = [s for w in windows for s in w['samples'] if s['route'] == route]
        if not samples: continue
        seconds = np.array([s['seconds'] for s in samples])
        summary['routes'][route] = dict(measurements=len(samples), seconds=float(seconds.sum()),
            seconds_per_window=float(seconds.mean()), six_frame_latency_ms=float(seconds.mean()*1000),
            p90_ms=float(np.percentile(seconds, 90)*1000),
            output_frames_per_second=float(6/seconds.mean()),
            peak_allocated_mib=max((s['peak_allocated_mib'] for s in samples if s['peak_allocated_mib'] is not None), default=None),
            peak_reserved_mib=max((s['peak_reserved_mib'] for s in samples if s['peak_reserved_mib'] is not None), default=None))
    summary['valid_same_quality_route_timing'] = summary['six_frame_byte_parity'] and actual_cuda
    summary['serialized_profile_NOT_latency'] = [w['serialized_profile_seconds'] for w in windows
                                                 if w.get('serialized_profile_seconds')]
    if any(r.endswith('_fast') for r in routes):
        comparison = {}
        for variant in ('add_only', 'add_remove'):
            name = 'soft_'+variant
            old, new = summary['routes'].get(name), summary['routes'].get(name+'_fast')
            if old is None or new is None: continue
            repeats = sorted({s['repeat'] for w in windows for s in w['samples']})
            old_by_repeat = {j: sum(s['seconds'] for w in windows for s in w['samples']
                if s['route'] == name and s['repeat'] == j) for j in repeats}
            new_by_repeat = {j: sum(s['seconds'] for w in windows for s in w['samples']
                if s['route'] == name+'_fast' and s['repeat'] == j) for j in repeats}
            each_pass = bool(repeats) and all(new_by_repeat[j] < old_by_repeat[j] for j in repeats)
            parity = bool(windows) and all(w['mismatched_voxels'][name+'_fast'] == 0 for w in windows)
            speedup = old['seconds_per_window']/new['seconds_per_window']
            candidate_samples = [s for w in windows for s in w['samples'] if s['route'] == name+'_fast']
            memory_safe = bool(candidate_samples) and all(s.get('device_total_mib') is not None
                and s.get('peak_reserved_mib') is not None
                and s['peak_reserved_mib'] <= .9*s['device_total_mib'] for s in candidate_samples)
            comparison[variant] = dict(speedup=speedup, six_frame_byte_parity=parity, faster_each_pass=each_pass,
                memory_headroom_ge_10percent=memory_safe,
                pass_gate=summary['valid_same_quality_route_timing'] and parity and each_pass and memory_safe and speedup >= 1.1)
        summary['optimization_comparison'] = comparison
        summary['automatic_promotion'] = False
    return summary
