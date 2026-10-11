"""Independent readouts of the SAME frozen model, using the training surrogate.

Quality retains the original GT cache route. Timing rebuilds Strong every call
and checks all six outputs against that route before reporting valid latency.
No future labels, optimizer, loss, or gradients are used by prediction/timing.
"""
import time
import numpy as np
import torch
from real_motion.surface_canonical_repair import augment_projection
from real_motion.canonical_causal_repair import compose_canonical
from ..engine import forward_window, hard_predictions
from ..data import prepare_projection
from ..soft_geometry import source_targets, effective_yaw, continuous_context
from ..batching.geometry import GroupedScene

ROUTES = ('hard_transport', 'hard_add_only', 'hard_add_remove',
          'soft_transport', 'soft_add_only', 'soft_add_remove')
SPEED_ROUTES = ('hard_add_remove', 'soft_add_only', 'soft_add_remove')


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


def fresh_forward(model, provider, row, *, soft):
    """Prepared causal history -> FRESH Strong + WM + projection + CCR.

    GT ego is the declared condition. The independently supervised ego head is
    unused in this GT route. Cached Strong/backgrounds and predictions are NOT
    consumed, even though read-only history association/descriptors are reused.
    """
    record, raw, _, history = row
    output = model.motion(record, provider.device)
    poses = torch.as_tensor(np.asarray(raw['future_poses']), device=provider.device, dtype=torch.float64)
    prep, evidence, plan = prepare_projection(provider, record, raw, output,
        poses.detach().cpu().numpy(), history, use_gt_cache=False)
    fw = dict(output=output, poses=poses, prep=prep, evidence=evidence, plan=plan,
        targets=source_targets(output, record, prep.state['current_pose'], prep.state['source_z_t0']),
        yaws=effective_yaw(output, [int(c['class_id']) for c in prep.state['current']]))
    if soft: return decode_soft(model, provider, fw)
    enriched = augment_projection(evidence, plan, prep.state['current_pose'],
                                  prep.state['world_to_future'], provider.pcfg.grid)
    session = provider.paired_eval_sessions[id(model.columns)]
    fw['actions'] = torch.from_numpy(session(model.columns, evidence, enriched, output, provider.device))
    return fw


@torch.no_grad()
def infer(model, provider, row, route, *, remove_threshold=.95):
    if route not in SPEED_ROUTES: raise ValueError('unknown inference timing route')
    fw = fresh_forward(model, provider, row, soft=route.startswith('soft_'))
    if route == 'hard_add_remove':
        actions = fw['actions'].float().cpu().numpy()
        return compose_canonical(fw['prep'].baseline, fw['evidence'], fw['plan'],
                                 actions[..., 0], actions[..., 1], thresholds=(.5, remove_threshold))
    variant = route.removeprefix('soft_')
    return soft_predictions(provider, fw, (variant,), range(6))[variant]


def sync(device):
    if device.type == 'cuda': torch.cuda.synchronize(device)


def mismatch_voxels(expected, actual):
    if len(expected) != 6 or len(actual) != 6: raise RuntimeError('timing must produce ALL SIX frames')
    if any(a.shape != b.shape for a, b in zip(expected, actual)): raise RuntimeError('inference volume differs')
    return sum(int(np.count_nonzero(a != b)) for a, b in zip(expected, actual))


@torch.no_grad()
def measure_window(model, provider, row, index, *, repeats=2, remove_threshold=.95):
    if repeats < 2: raise ValueError('at least TWO alternating timing passes required')
    expected = quality_predictions(model, provider, row, remove_threshold=remove_threshold, horizons=range(6))
    # Warm EACH route on this window outside measurement, including graph
    # capture/verification and immutable coordinate initialization. Learned
    # projections, Strong, head probabilities and six outputs are rebuilt.
    mismatches, samples = {}, []
    for route in SPEED_ROUTES:
        actual = infer(model, provider, row, route, remove_threshold=remove_threshold)
        sync(provider.device)
        mismatches[route] = mismatch_voxels(expected[route], actual)
        del actual
    for repeat in range(repeats):
        order = SPEED_ROUTES if (index+repeat) % 2 == 0 else tuple(reversed(SPEED_ROUTES))
        for route in order:
            sync(provider.device)
            if provider.device.type == 'cuda': torch.cuda.reset_peak_memory_stats(provider.device)
            begun = time.perf_counter()
            actual = infer(model, provider, row, route, remove_threshold=remove_threshold)
            sync(provider.device)
            seconds = time.perf_counter()-begun
            mismatch = mismatch_voxels(expected[route], actual)
            mismatches[route] += mismatch
            samples.append(dict(route=route, repeat=repeat, seconds=seconds,
                mismatched_voxels=mismatch,
                peak_allocated_mib=(torch.cuda.max_memory_allocated(provider.device)/2**20
                                    if provider.device.type == 'cuda' else None),
                peak_reserved_mib=(torch.cuda.max_memory_reserved(provider.device)/2**20
                                   if provider.device.type == 'cuda' else None)))
            del actual
    return dict(key=[row[0]['scene_name'], row[0]['t0_token']], samples=samples,
                mismatched_voxels=mismatches, six_frame_byte_parity=not any(mismatches.values()))


def speed_summary(windows, *, actual_cuda):
    summary = dict(windows=len(windows), actual_cuda=actual_cuda,
        six_frame_byte_parity=bool(windows) and all(w['six_frame_byte_parity'] for w in windows),
        scope='prepared four-history inputs -> fresh Strong/WM/CCR/projection/full six-frame CPU semantics',
        excludes=['disk I/O', 'history association/representation/descriptors', 'labels/metrics',
                  'warmup/graph capture', 'checkpoint writes'],
        notes='paired WARM diagnostic latency, NOT old-model FPS or raw-input end-to-end FPS; reserved memory includes persistent allocator/graphs',
        routes={})
    for route in SPEED_ROUTES:
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
    return summary
