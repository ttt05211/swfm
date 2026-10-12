import copy
import time
from types import SimpleNamespace
import numpy as np
import torch
from real_motion.metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS
from real_motion.strong_majority_execution import strong_majority_execution
from real_motion.runtime_fastpath import compose_component_replacements_fast_exact
from tools.joint_occ_experiments.soft_inference.engine import recognize_prior_encoding, live_prior, sync
from tools.real_motion import benchmark_p0_f9_v18_runtime as runtime
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import Metrics
from tools.real_motion.eval_p0_f9_v17_local_stwm import window_from_record
from tools.real_motion.shared_evidence_pilot_common import moving_support_masks
from real_motion.nuscenes_adapter import gt_moving_support_sequence
from tools.joint_occ_experiments.soft_geometry import source_targets, effective_yaw
from ..refine import engine as quality
from ..refine.geometry import TransportPrior, aligned_memory

PROTOCOL = 'semantic_refiner_frozen_edits_fresh_fps_v2'
TIMING_POLICY = 'complete_population_separate_strict_byte_parity_v2'
REPORT_H = (1, 3, 5)


def edit_counts(before, after, target, mask=None):
    before, after, target = map(np.asarray, (before, after, target))
    if before.shape != after.shape or before.shape != target.shape: raise ValueError('edit shapes differ')
    valid = (target >= 0) & (target < 18)
    if mask is not None:
        if np.shape(mask) != target.shape: raise ValueError('edit mask shape differs')
        valid &= np.asarray(mask, bool)
    changed = (before != after) & valid
    correct, old_correct = after == target, before == target
    routes = dict(added=changed & (before == 17) & (after != 17),
        removed=changed & (before != 17) & (after == 17),
        relabeled=changed & (before != 17) & (after != 17))
    result = dict(valid_voxels=int(valid.sum()), changed=int(changed.sum()))
    for name, at in routes.items():
        result[name] = dict(changed=int(at.sum()), corrected=int((at & correct).sum()),
            damaged=int((at & old_correct).sum()), wrong_to_wrong=int((at & ~correct & ~old_correct).sum()),
            occupancy_corrected=int((at & ((before != 17) != (target != 17)) &
                ((after != 17) == (target != 17))).sum()),
            occupancy_damaged=int((at & ((before != 17) == (target != 17)) &
                ((after != 17) != (target != 17))).sum()))
    return result


def add_counts(total, values):
    for key, value in values.items():
        if isinstance(value, dict): add_counts(total.setdefault(key, {}), value)
        else: total[key] = total.get(key, 0)+int(value)


def prepare_speed(model, provider, row):
    """Only deterministic history and input uploads; NO learned features/prior."""
    record, raw, inputs, history = row
    tick = time.perf_counter(); device = provider.device
    state = history[0]['state']
    previous = runtime.extract_instances_cropped_exact(raw['history_occ'][-2], raw['history_poses'][-2],
        grid=provider.pcfg.grid, cfg=provider.strong)
    velocities = runtime.match_instances(previous, state['current'], float(provider.pcfg.frame_dt_s),
        max_speed_mps=provider.strong.max_match_speed_mps)
    causal = SimpleNamespace(current=state['current'], current_pose=state['current_pose'],
        source_count=len(state['current']), velocities=velocities)
    causal.kta_encoding = recognize_prior_encoding(record, causal, provider.pcfg.frame_dt_s)
    # Neither future labels/annotations nor cached Strong can enter timed input.
    raw_view = {k: raw[k] for k in ('history_occ', 'history_observed', 'history_poses', 'future_poses')}
    record_view = {k: record[k] for k in ('features', 'source_class_id', 'source_centroid_xy_t0_m')}
    upload = lambda a: torch.as_tensor(np.asarray(a), device=device)
    staged = dict(semantic=upload(raw_view['history_occ']), observed=upload(raw_view['history_observed']),
        reads={k: upload(v) for k, v in history[0]['shared_reads'].items()},
        times=upload(inputs[0][:, -1]), classes=upload(record_view['source_class_id']).long(),
        poses=upload(raw_view['future_poses']).double())
    sync(device)
    return (record_view, raw_view, history, causal, staged), time.perf_counter()-tick


@torch.no_grad()
def fresh_predictions(model, provider, prepared, profile=None, *, final_only=False, debug=None):
    record, raw, history, causal, staged = prepared
    state = {**history[0]['state']}; device = provider.device
    def stage(name, function):
        if profile is None: return function()
        sync(device); tick = time.perf_counter(); value = function(); sync(device)
        profile[name] = time.perf_counter()-tick
        return value
    def strong():
        with strong_majority_execution(getattr(provider, 'strong_majority', None)):
            anchors, components = runtime._strong_all_horizons(raw['history_occ'][-1], state['current_pose'],
                raw['future_poses'], state['current'], causal.velocities, state['source_world_points'],
                frame_dt_s=provider.pcfg.frame_dt_s, grid=provider.pcfg.grid, cfg=provider.strong,
                runtime_device=device, majority_backend='native' if getattr(provider, 'strong_majority', None) else 'sparse_cuda')
        return [compose_component_replacements_fast_exact(a, c, [], dynamic_class_ids=DYNAMIC_CLASS_IDS,
            free_label=17, grid=provider.pcfg.grid,
            precomputed_clear_flat_indices=runtime.baseline_clear_flat_indices(c, grid=provider.pcfg.grid))
            for a, c in zip(anchors, components)]
    state['column_backgrounds'] = stage('fresh_Strong_CLEAR', strong)
    def network():
        kta, anchors = live_prior(record, causal, provider.pcfg.frame_dt_s)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            field = model.transport.history_encoder(staged['semantic'], staged['observed'])
            out = model.transport(field, staged['reads'], staged['times'], staged['classes'],
                torch.as_tensor(kta, device=device))
            lifted = model.columns.lift_history(field)
        return out, lifted, anchors
    output, lifted, anchors = stage('live_KTA_encoder_STWM_lift', network)
    targets = source_targets(output, {**record, 'anchors_xy_t0_m': anchors}, state['current_pose'], state['source_z_t0'])
    yaws = effective_yaw(output, [int(s['class_id']) for s in state['current']])
    scene = TransportPrior(raw, state, provider.pcfg.grid, device)
    prior = stage('continuous_Transport', lambda: scene.render(staged['poses'], targets, yaws))
    memory = stage('aligned_memory', lambda: aligned_memory(model.columns, lifted, output, history,
        raw, targets, yaws, staged['poses'], provider.pcfg.grid))
    logits = stage('semantic_refiner', lambda: model.columns(memory, prior, range(6), scene.shape))
    if debug is not None: debug['logits'] = logits  # reference only; inspection AFTER timer
    outputs = (('semantic_refine', logits),) if final_only else (('transport', prior), ('semantic_refine', logits))
    return stage('six_frame_argmax_CPU', lambda: {
        name: values.argmax(1).reshape(6, *scene.shape).byte().cpu().numpy()
        for name, values in outputs})


def comparison_details(actual, expected, logits=None, reference_logits=None):
    mismatches, per_horizon = {}, {}
    for name, values in actual.items():
        target = np.asarray(expected[name]); values = np.asarray(values)
        if values.shape != target.shape or values.shape[0] != 6:
            raise RuntimeError('comparison must retain SAME complete six-frame shape')
        if np.any((values < 0) | (values > 17)): raise RuntimeError('invalid semantic output')
        mask = (values != target).reshape(6, -1)
        per_horizon[name] = mask.sum(1).astype(int).tolist()
        mismatches[name] = int(mask.sum())
    result = dict(mismatches=mismatches, per_horizon=per_horizon)
    if logits is not None:
        lo, hi = torch.aminmax(logits.detach().float())
        if not bool(torch.isfinite(torch.stack((lo, hi))).all()): raise RuntimeError('nonfinite inference logits')
        result['logit_range'] = [float(lo), float(hi)]
    if reference_logits is not None and logits is not None:
        mask = (actual['semantic_refine'] != expected['semantic_refine']).reshape(6, -1)
        h, at = np.nonzero(mask)
        # Bounded evidence, not a full-grid probability dump or acceptance tolerance.
        h, at = h[:16], at[:16]
        if len(h):
            device = logits.device
            fresh = logits[torch.as_tensor(h, device=device), :, torch.as_tensor(at, device=device)].float().cpu().numpy()
            reference = np.asarray(reference_logits)[h, :, at]
            if not np.isfinite(reference).all(): raise RuntimeError('nonfinite reference logits')
            old = expected['semantic_refine'].reshape(6, -1)[h, at].astype(int)
            new = actual['semantic_refine'].reshape(6, -1)[h, at].astype(int)
            sorted_ref, sorted_new = np.sort(reference, 1), np.sort(fresh, 1)
            result['changed_voxel_samples'] = [dict(horizon=.5*(int(h[i])+1), flat=int(at[i]),
                reference_class=int(old[i]), fresh_class=int(new[i]),
                reference_top2_margin=float(sorted_ref[i, -1]-sorted_ref[i, -2]),
                fresh_top2_margin=float(sorted_new[i, -1]-sorted_new[i, -2]),
                max_abs_logit_difference=float(np.max(np.abs(fresh[i]-reference[i])))) for i in range(len(h))]
        else: result['changed_voxel_samples'] = []
    return result


@torch.no_grad()
def measure(model, provider, row, expected, repeats=3, *, reference_logits=None):
    prepared, preparation = prepare_speed(model, provider, row)
    checks, seconds = [], []
    def inspect(actual, debug, phase):
        details = comparison_details(actual, expected, debug.pop('logits'), reference_logits)
        checks.append(dict(phase=phase, **details))
    debug = {}
    first = fresh_predictions(model, provider, prepared, debug=debug)
    inspect(first, debug, 'warmup')  # output differences NEVER filter timing population
    sync(provider.device)
    if provider.device.type == 'cuda': torch.cuda.reset_peak_memory_stats(provider.device)
    for index in range(repeats):
        debug = {}
        sync(provider.device); tick = time.perf_counter()
        actual = fresh_predictions(model, provider, prepared, final_only=True, debug=debug); sync(provider.device)
        seconds.append(time.perf_counter()-tick)
        inspect(actual, debug, f'timed_{index+1}')
    profile, debug = {}, {}
    extra = fresh_predictions(model, provider, prepared, profile, final_only=True, debug=debug)
    inspect(extra, debug, 'serialized_profile')
    mismatches = [c['mismatches'] for c in checks]
    return dict(timing_policy=TIMING_POLICY, timing_complete=len(seconds) == repeats,
        six_frame_byte_parity=not any(v for item in mismatches for v in item.values()),
        mismatches=mismatches, checks=checks, seconds=seconds, history_preparation_seconds=preparation,
        serialized_profile_seconds=profile, peak_reserved_mib=torch.cuda.max_memory_reserved(provider.device)/2**20
            if provider.device.type == 'cuda' else 0.)


def validate_saved(saved):
    if saved.get('protocol') != PROTOCOL or type(saved.get('windows')) is not int or not 0 <= saved['windows'] <= 512:
        raise RuntimeError('invalid diagnostic cursor/protocol')
    if len(saved['errors']) != saved['windows'] or len(saved['speed']) > saved['windows']:
        raise RuntimeError('invalid completed diagnostic window counts')
    def integers(value):
        if isinstance(value, dict): return all(integers(v) for v in value.values())
        return type(value) is int and value >= 0
    if not integers(saved['edits']): raise RuntimeError('invalid integer edit counts')


@torch.no_grad()
def evaluate(models, provider, source, batches, *, speed_keys, speed_repeats=3,
             stop_event=None, progress=None, saved=None, save=None, **_):
    model = models['gt']; model.eval()
    metrics = {n: Metrics() for n in ('transport', 'semantic_refine')}
    windows, edits, errors, speed = 0, {}, [], {}
    if saved:
        validate_saved(saved); windows = saved['windows']
        edits, errors, speed = copy.deepcopy(saved['edits']), copy.deepcopy(saved['errors']), copy.deepcopy(saved['speed'])
        if not set(speed) <= {'|'.join(k) for k in speed_keys}: raise RuntimeError('saved timing population differs')
        if set(saved['metrics']) != set(metrics): raise RuntimeError('metric routes differ')
        for n, values in saved['metrics'].items():
            if set(values) != set(vars(metrics[n])): raise RuntimeError('metric fields differ')
            for k, v in values.items():
                array = np.asarray(v)
                if array.shape != getattr(metrics[n], k).shape or np.any(array < 0) or not np.array_equal(array, array.astype(np.int64)):
                    raise RuntimeError('invalid integer metric state')
                setattr(metrics[n], k, array.astype(np.int64))
    def snapshot():
        return dict(protocol=PROTOCOL, windows=windows, errors=copy.deepcopy(errors), edits=copy.deepcopy(edits),
            speed=copy.deepcopy(speed), metrics={n: {k: v.tolist() for k, v in vars(m).items()} for n, m in metrics.items()})
    started, offset = time.perf_counter(), 0
    for rows in batches:
        for row in rows:
            offset += 1
            if offset <= windows: continue
            if stop_event is not None and stop_event.is_set():
                if save: save(snapshot())
                raise InterruptedError('saved complete semantic diagnostic windows')
            fw = quality.forward_window(model, provider, row)
            predictions = {n: v.argmax(1).reshape(6, *fw['shape']).byte().cpu().numpy()
                for n, v in (('transport', fw['prior']), ('semantic_refine', fw['logits']))}
            error = fw['ego'].float().cpu().numpy()-np.asarray(fw['ego_target'])
            key = (row[0]['scene_name'], row[0]['t0_token'])
            reference_logits = fw['logits'].float().cpu().numpy() if key in speed_keys else None
            del fw
            timing = measure(model, provider, row, predictions, speed_repeats,
                reference_logits=reference_logits) if key in speed_keys else None
            # ALL cached-quality and fresh-timing predictions precede metric GT reads.
            raw = row[1]; window = window_from_record(row[0])
            support = gt_moving_support_sequence(source.nusc, window.t0_token, window.future_tokens,
                tuple(.5*(h+1) for h in range(6)), grid=provider.pcfg.grid, workers=provider.workers)
            moving = moving_support_masks(support, raw['future_gt_occ'].shape[1:])
            delta = {}
            for h in range(6):
                before, after, target = predictions['transport'][h], predictions['semantic_refine'][h], raw['future_gt_occ'][h]
                dynamic_union = np.isin(target, DYNAMIC_CLASS_IDS) | np.isin(before, DYNAMIC_CLASS_IDS) | np.isin(after, DYNAMIC_CLASS_IDS)
                delta[str(.5*(h+1))] = {
                    'full_grid': edit_counts(before, after, target),
                    'GT_dynamic_semantics': edit_counts(before, after, target, np.isin(target, DYNAMIC_CLASS_IDS)),
                    'dynamic_semantic_union_including_FP': edit_counts(before, after, target, dynamic_union),
                    'original_t0_Moving_support': edit_counts(before, after, target, moving[h]),
                    'Moving_support_dynamic_union': edit_counts(before, after, target, moving[h] & dynamic_union)}
            # Finish the entire window before publishing any counters/timing.
            for n in predictions:
                for i, h in enumerate(REPORT_H): metrics[n].update(i, predictions[n][h], raw['future_gt_occ'][h], moving[h])
            add_counts(edits, delta)
            errors.append(dict(xy=np.linalg.norm(error[:, :2], axis=-1).tolist(),
                yaw=(np.abs(np.arctan2(np.sin(error[:, 2]), np.cos(error[:, 2])))*180/np.pi).tolist()))
            if timing is not None: speed['|'.join(key)] = timing
            windows += 1
            if save: save(snapshot())
            if windows % 16 == 0:
                print(f'REFINER_AUDIT {windows} speed_windows={len(speed)}', flush=True)
                if progress: progress(dict(event='semantic_refiner_audit', windows=windows, speed_windows=len(speed)))
    if save: save(snapshot())
    return dict(windows=windows, variants={'gt': {n: m.compute() for n, m in metrics.items()}},
        trajectory={'gt': dict(xy_mean_m=np.mean([e['xy'] for e in errors], 0).tolist(),
            yaw_mean_deg=np.mean([e['yaw'] for e in errors], 0).tolist())} if errors else {},
        edits=edits, speed=speed, seconds=time.perf_counter()-started, main_readout='semantic_refine')


def speed_summary(speed, expected_windows, repeats, actual_cuda):
    rows = list(speed.values()); samples = [t for row in rows for t in row['seconds']]
    parity = len(rows) == expected_windows and all(r['six_frame_byte_parity'] and len(r['seconds']) == repeats for r in rows)
    coverage = len(rows) == expected_windows and all(len(r['seconds']) == repeats for r in rows)
    valid = coverage and actual_cuda and bool(samples) and all(np.isfinite(samples)) and min(samples) > 0
    maxima = {}
    for row in rows:
        items = row.get('mismatches', [])
        if isinstance(items, dict): items = [items]
        for item in items:
            for name, count in item.items(): maxima[name] = max(maxima.get(name, 0), count)
    return dict(windows=len(rows), repeats=repeats, six_frame_byte_parity=parity, actual_cuda=actual_cuda,
        timing_policy=TIMING_POLICY, timing_complete=valid, measured_calls=len(samples),
        expected_calls=expected_windows*repeats, byte_parity_failed_windows=sum(not r['six_frame_byte_parity'] for r in rows),
        max_changed_voxels_per_six_frame_call=maxima,
        valid_same_quality_route_timing=valid and parity,
        fps_label='WARM actual-execution diagnostic; NOT bit-identical-reference FPS' if not parity else 'WARM byte-verified diagnostic',
        six_frame_ms=1000*np.mean(samples) if samples else None,
        window_fps=len(samples)/sum(samples) if valid else None,
        output_frame_fps=6*len(samples)/sum(samples) if valid else None,
        p90_six_frame_ms=1000*np.percentile(samples, 90) if samples else None,
        peak_reserved_mib=max((r.get('peak_reserved_mib', 0.) for r in rows), default=0.),
        profile_mean_ms={k: 1000*np.mean([r['serialized_profile_seconds'][k] for r in rows])
            for k in rows[0].get('serialized_profile_seconds', {})} if coverage else {},
        allocator_peak_includes_untimed_diagnostics=True,
        scope='prepared deterministic history -> FRESH Strong/KTA + Shared Encoder/STWM + continuous Transport + aligned memory + Refiner + SIX CPU semantics',
        excludes='disk/history CC-association/read-coordinate preparation/initial history H2D/ego auxiliary/labels/metrics/warmup/serialized profile',
        projection_internal_tensor_materialization_is_timed=True, formal_FPS=False)
