"""GT-only shared-field screen. No KD, learned-feature cache or proposal cap."""
from collections import defaultdict
from contextlib import nullcontext
import math
import time

import numpy as np
import torch

from real_motion.causal_column_completion import (
    GENERATE, REFINE, actions_from_probabilities, compose_dense,
)
from real_motion.causal_column_model import column_loss
from real_motion.height_causal_field import gather_centres
from real_motion.source_evidence_audit import edit_quality
from real_motion.column_runtime_pipeline import prefetch_raw_columns
from real_motion.nuscenes_adapter import gt_moving_support_sequence
from tools.real_motion import causal_column_common as columns
from tools.real_motion.joint_column_common import (
    build_online_column_candidates, select_online_columns, count_proposals, weights_from_counts,
)
from tools.real_motion.shared_evidence_pilot_common import (
    GATES, fresh_prior, causal_template, moving_support_masks,
)
from tools.real_motion import benchmark_p0_f9_v18_runtime as runtime
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import Metrics

PROTOCOL = 'p0_f9_height_shared_field_gt_screen_v1'


def sync(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def epoch_groups(records, seed, epoch, window_batch=4, source_budget=128):
    """Whole-window batches; each record appears once, even oversized sources."""
    if min(window_batch, source_budget) < 1 or epoch < 0 or not records:
        raise ValueError('positive population, batch/source budget and epoch required')
    ids = np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(len(records))
    groups, group, sources = [], [], 0
    for i in ids:
        row = records[int(i)]
        n = len(row['features'])
        if group and (len(group) >= window_batch or sources+n > source_budget):
            groups.append(group); group = []; sources = 0
        group.append(row); sources += n
    if group:
        groups.append(group)
    return groups


def learning_rate(update, steps, base_lr):
    if not 0 <= update <= steps or steps < 1 or not math.isfinite(base_lr) or base_lr <= 0:
        raise ValueError('invalid fixed cosine budget')
    return base_lr*(.1+.9*.5*(1+math.cos(math.pi*update/steps)))


def rows_for(prep, grid, config, horizons=range(6)):
    return [(h, columns.candidate_plan(prep, h, grid, config), None, None) for h in horizons]


def train_step(provider, rows, teacher, head, optimizer, rng, *, candidate_pool=None):
    """True fresh native-field forward/backward for EVERY window, then one step.

    Equal-window mean of the original two-task loss, as in the local prototype.
    Backward per window releases each full-grid graph before the next window;
    accumulation changes neither window population nor gradient normalization.
    Motion intentionally frozen in this controlled screen. Future GT enters
    only action labels/training sampler, never the encoder/centre addresses.
    """
    device = provider.device
    if not rows:
        raise ValueError('empty training batch')
    if any(p.requires_grad for p in teacher.parameters()):
        raise RuntimeError('this screen requires frozen epoch19 transport/reference')
    head.train(); teacher.eval(); optimizer.zero_grad(set_to_none=True)
    sync(device); begun = time.perf_counter(); stages = defaultdict(float)
    losses, terms, counts = [], defaultdict(list), defaultdict(int)
    for record, raw in rows:
        tick = time.perf_counter()
        with torch.no_grad():
            output = teacher.motion(record, device)
            prep = provider.prepare_columns(None, record, include_gt=True, raw_window=raw, outputs=output)
        stages['live_motion_render'] += time.perf_counter()-tick
        tick = time.perf_counter()
        if candidate_pool is None:
            candidates = build_online_column_candidates(prep, teacher.columns.config, provider.pcfg.grid)
        else:
            from tools.real_motion.joint_column_common import OnlineColumnCandidateBuilder
            builder = OnlineColumnCandidateBuilder(prep, teacher.columns.config, provider.pcfg.grid)
            candidates = list(candidate_pool.map(builder.build, range(6)))
        # Sampling/RNG remains ordered and main-thread owned.
        selected = select_online_columns(prep, teacher.columns.config, provider.pcfg.grid, rng, candidates)
        stages['candidate_and_sample'] += time.perf_counter()-tick
        if not selected or not sum(len(r[1]) for r in selected):
            counts['windows_without_legal_queries'] += 1
            continue
        tick = time.perf_counter()
        samples = gather_centres(prep, selected, provider.pcfg.grid, teacher.columns.config, device, columns.pose_motion)
        target = torch.as_tensor(np.concatenate([r[2] for r in selected]), device=device)
        weight = torch.as_tensor(np.concatenate([r[3] for r in selected]), device=device)
        history = torch.as_tensor(np.asarray(raw['history_occ']), device=device)
        observed = torch.as_tensor(np.asarray(raw['history_observed']), device=device)
        stages['centre_gather_upload'] += time.perf_counter()-tick
        counts['sampled_columns'] += len(samples.kind)
        counts['dynamic_refine_columns'] += sum(int(((p.kind == REFINE) & (p.actor >= 0)).sum()) for _, p, _, _ in selected)
        counts['static_refine_columns'] += sum(int(((p.kind == REFINE) & (p.actor < 0)).sum()) for _, p, _, _ in selected)
        counts['generation_columns'] += sum(int((p.kind == GENERATE).sum()) for _, p, _, _ in selected)
        counts['boundary_cpu_rows'] += samples.boundary_cpu_rows
        tick = time.perf_counter()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            field = head.encode_history(history, observed)
            generation, refinement = head(field, samples, output)
            loss, stats = column_loss(head, generation, refinement, samples.kind, samples.legal,
                                      target, weight, materialize_stats=False)
        (loss/len(rows)).backward()
        stages['fresh_field_head_loss_backward'] += time.perf_counter()-tick
        losses.append(loss.detach())
        for key, value in stats.items():
            terms[key].append(value)
        # Do not retain graph tensors in provider, progress records or caches.
        del loss, stats, field, generation, refinement, samples, output, prep, candidates, selected
        del target, weight, history, observed
    if not losses:
        raise RuntimeError('entire batch has no supervised legal edits; refusing silent no-op')
    norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 5.)
    if not torch.isfinite(norm):
        raise RuntimeError('nonfinite shared-field gradient; previous complete checkpoint preserved')
    tick = time.perf_counter(); optimizer.step(); sync(device)
    stages['optimizer_and_finish'] += time.perf_counter()-tick
    loss_value = float(torch.stack(losses).sum()/len(rows))
    stats = {k: float(torch.stack(v).mean()) for k, v in terms.items()}
    optimizer.zero_grad(set_to_none=True)
    return dict(loss=loss_value, **stats, grad_norm=float(norm), **counts, windows=len(rows),
                seconds=time.perf_counter()-begun, stages_seconds=dict(stages),
                optimizer_updated=True, transport_frozen=True, GT_only=True, KD=False,
                allocated_after_mib=torch.cuda.memory_allocated(device)/2**20 if device.type == 'cuda' else 0.)


@torch.no_grad()
def calibrate_train(provider, source, records, teacher, head, *, progress=None, stop_event=None):
    counts = dict(generation=np.zeros(2, np.int64), refine=np.zeros(3, np.int64))
    for i, (record, raw) in enumerate(prefetch_raw_columns(provider, source, records), 1):
        if stop_event is not None and stop_event.is_set():
            raise InterruptedError('TRAIN calibration stopped; no partially calibrated checkpoint')
        output = teacher.motion(record, provider.device)
        prep = provider.prepare_columns(source, record, include_gt=True, raw_window=raw, outputs=output)
        count_proposals(prep, provider.pcfg.grid, teacher.columns.config, counts)
        if i == 1 or i % 32 == 0 or i == len(records):
            print(f'FIELD_TRAIN_PRIOR {i}/{len(records)} TRAIN_only', flush=True)
    weights = weights_from_counts(counts)
    weights.update(population='fixed epoch19 unsampled proposals on TRAIN subset only', windows=len(records))
    head.generation_pos_weight.fill_(weights['generation_pos_weight'])
    head.refine_class_weights.copy_(torch.tensor(weights['refine_class_weights'], device=provider.device))
    if progress:
        progress(dict(event='train_prior', weights=weights))
    return weights


@torch.no_grad()
def probabilities(head, prep, output, rows, grid, config, device, *, chunk=1024):
    """All original legal queries/heights; encode four maps once per window."""
    if chunk < 1:
        raise ValueError('positive readout chunk required')
    if not sum(len(p) for _, p, _, _ in rows):
        return [np.empty((*p.base.shape, 3), np.float32) for _, p, _, _ in rows]
    samples = gather_centres(prep, rows, grid, config, device, columns.pose_motion)
    history = torch.as_tensor(np.asarray(prep.raw['history_occ']), device=device)
    observed = torch.as_tensor(np.asarray(prep.raw['history_observed']), device=device)
    chunks = []
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        field = head.encode_history(history, observed)
        for begin in range(0, len(samples.kind), chunk):
            small = samples.subset(slice(begin, begin+chunk))
            g, r = head(field, small, output)
            chunks.append(head.probabilities(g, r, small))
    p = torch.cat(chunks).float().cpu().numpy()
    if not np.isfinite(p).all():
        raise RuntimeError('nonfinite shared-field probability')
    result, begin = [], 0
    for _, plan, _, _ in rows:
        result.append(p[begin:begin+len(plan)]); begin += len(plan)
    return result


@torch.no_grad()
def evaluate(provider, source, records, teacher, head, *, include_old=False, progress=None, stop_event=None):
    # The reference must use the already validated fast Local execution, not
    # resurrect the per-query CPU sampler that took seconds per horizon.
    from real_motion.column_execution import execution_session
    if include_old:
        teacher.columns.column_inference_optimized = True
        teacher.columns.column_async_readback = True
        teacher.columns.column_probability_optimized = False
        teacher.columns.column_sampling_workers = provider.workers
        teacher.columns.column_inference_verify_remaining = 3
    with execution_session(teacher.columns, graphs=True, reuse=False) if include_old else nullcontext():
        return _evaluate(provider, source, records, teacher, head, include_old=include_old,
                         progress=progress, stop_event=stop_event)


@torch.no_grad()
def _evaluate(provider, source, records, teacher, head, *, include_old=False, progress=None, stop_event=None):
    teacher.eval(); head.eval()
    names = ('generation', 'refine', 'joint') + (('old_joint',) if include_old else ())
    base = Metrics(); metrics = {n: Metrics() for n in names}
    quality = {n: defaultdict(int) for n in names}
    scenes = defaultdict(lambda: {n: Metrics() for n in ('baseline', *names)})
    started = previous = time.perf_counter(); stages = defaultdict(float)
    for wi, (record, raw) in enumerate(prefetch_raw_columns(provider, source, records), 1):
        tick = time.perf_counter(); window_start = tick; stages['input_wait'] += tick-previous
        if stop_event is not None and stop_event.is_set():
            raise InterruptedError('evaluation stopped at window boundary; training checkpoint remains valid')
        output = teacher.motion(record, provider.device)
        prep = provider.prepare_columns(source, record, include_gt=True, raw_window=raw, outputs=output)
        stages['prepare_and_motion'] += time.perf_counter()-tick; tick = time.perf_counter()
        rows = rows_for(prep, provider.pcfg.grid, teacher.columns.config, columns.REPORT)
        p = probabilities(head, prep, output, rows, provider.pcfg.grid, teacher.columns.config, provider.device)
        stages['field_candidates_probabilities'] += time.perf_counter()-tick
        tick = time.perf_counter()
        support = gt_moving_support_sequence(source.nusc, prep.window.t0_token, prep.window.future_tokens,
                    tuple(.5*(h+1) for h in range(6)), grid=provider.pcfg.grid, workers=provider.workers)
        moving = moving_support_masks(support, provider.pcfg.grid.shape_hwd)
        for ri, ((h, plan, _, _), prob) in enumerate(zip(rows, p)):
            act = actions_from_probabilities(plan, prob, GATES)
            predictions = {n: compose_dense(prep.baseline[h], plan, act,
                enable_generation=n != 'refine', enable_refine=n != 'generation') for n in names if n != 'old_joint'}
            if include_old:
                old_p = columns.predict_probabilities(teacher.columns, prep, h, plan, provider.pcfg.grid, provider.device, 256)
                predictions['old_joint'] = compose_dense(prep.baseline[h], plan, actions_from_probabilities(plan, old_p, GATES))
            gt = raw['future_gt_occ'][h]; scene = scenes[str(record['scene_name'])]
            base.update(ri, prep.baseline[h], gt, moving[h]); scene['baseline'].update(ri, prep.baseline[h], gt, moving[h])
            for name in names:
                metrics[name].update(ri, predictions[name], gt, moving[h])
                scene[name].update(ri, predictions[name], gt, moving[h])
                for key, value in edit_quality(prep.baseline[h], predictions[name], gt).items():
                    quality[name][key] += value
        sync(provider.device); previous = time.perf_counter()
        stages['moving_support_reference_and_metrics'] += previous-tick
        if progress:
            progress(dict(event='field_evaluation', window=wi, windows=len(records), compute_seconds=previous-window_start))
        if wi == 1 or wi % 16 == 0 or wi == len(records):
            print(f'FIELD_EVAL {wi}/{len(records)}', flush=True)
    result = columns.report_states(base, metrics, quality, scenes)
    result.update(windows=len(records), seconds=time.perf_counter()-started, stages_seconds=dict(stages))
    return result


@torch.no_grad()
def six_frame_speed(provider, source, records, teacher, head, *, repeats=2, stop_event=None):
    """Paired old/new FPS: fresh prior + LIVE motion + ALL SIX dense frames.

    Source tensors resident before timing; four-history registration/IO/GT and
    metrics explicitly excluded. Same prepared-input boundary as local pilot.
    """
    from real_motion.column_execution import execution_session
    teacher.eval(); head.eval(); trials = []
    teacher.columns.column_inference_optimized = True
    teacher.columns.column_async_readback = True
    teacher.columns.column_probability_optimized = False
    teacher.columns.column_sampling_workers = provider.workers
    teacher.columns.column_inference_verify_remaining = 0
    with execution_session(teacher.columns, graphs=True, reuse=False):
        for wi, (record, raw) in enumerate(prefetch_raw_columns(provider, source, records, include_gt=False), 1):
            if raw.get('future_gt_occ') is not None:
                raise RuntimeError('FPS may not load future GT')
            template = causal_template(raw, record)
            inputs = runtime._gpu_inputs(record, provider.device)
            def run(old):
                sync(provider.device); tick = time.perf_counter()
                state = fresh_prior(template, provider).state
                fresh_raw = {**raw, '_column_causal_preparation': {**raw['_column_causal_preparation'], 'prepared_state': state}}
                output = runtime._model_forward(teacher.transport, inputs, provider.device, return_latents=True)
                prep = provider.prepare_columns(source, record, include_gt=False, raw_window=fresh_raw, outputs=output)
                rows = rows_for(prep, provider.pcfg.grid, teacher.columns.config)
                if old:
                    probs = [columns.predict_probabilities(teacher.columns, prep, h, p, provider.pcfg.grid, provider.device, 256)
                             for h, p, _, _ in rows]
                else:
                    probs = probabilities(head, prep, output, rows, provider.pcfg.grid, teacher.columns.config, provider.device)
                dense = [compose_dense(prep.baseline[h], p, actions_from_probabilities(p, prob, GATES))
                         for (h, p, _, _), prob in zip(rows, probs)]
                sync(provider.device); seconds = time.perf_counter()-tick
                # Output validation/hash costs are outside latency.
                if len(dense) != 6 or any(d.shape != tuple(provider.pcfg.grid.shape_hwd) for d in dense):
                    raise RuntimeError('FPS requires six complete dense outputs')
                return dense, seconds
            expected = {old: run(old)[0] for old in (True, False)}  # warm-up outside timings
            for repeat in range(repeats):
                for old in ((True, False) if repeat % 2 == 0 else (False, True)):
                    if stop_event is not None and stop_event.is_set():
                        raise InterruptedError('FPS stopped; training checkpoint preserved')
                    dense, seconds = run(old)
                    if any(not np.array_equal(a, b) for a, b in zip(dense, expected[old])):
                        raise RuntimeError('FPS repeated execution changed dense outputs')
                    mode = 'old_joint' if old else 'shared_field'
                    trials.append(dict(mode=mode, window=wi, repeat=repeat+1, six_frame_seconds=seconds,
                                       sources=len(record['features']), key=[str(record['scene_name']), str(record['t0_token'])]))
                    print(f'FIELD_FPS {mode} {wi}/{len(records)} six_seconds={seconds:.4f} FPS={6/seconds:.2f}', flush=True)
            del inputs, expected
    means = {mode: float(np.mean([r['six_frame_seconds'] for r in trials if r['mode'] == mode]))
             for mode in ('old_joint', 'shared_field')}
    return dict(trials=trials, six_frame_mean_seconds=means,
                six_frame_amortized_FPS={k: 6/v for k, v in means.items()},
                speedup=means['old_joint']/means['shared_field'],
                boundary='resident source tensors + registered FOUR histories -> fresh Strong/KTA prior + live motion + all candidates/fields + SIX dense compositions',
                excludes='disk I/O, initial source extraction/registration, GT/metrics, warm-up/graph capture, output equality checks',
                not_raw_sensor_end_to_end=True, all_six_dense_repeat_exact=True)
