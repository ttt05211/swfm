"""Point CCR server screen: causal Monte Carlo, GT-only, frozen transport.

No distillation, learned-feature cache, GT candidate selection or dev tuning.
The old Local frontier GEN support is NOT preserved by this representation.
"""
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
import json
import time

import numpy as np
import torch

from real_motion.canonical_causal_repair import (
    CanonicalRepairHead, build_canonical_evidence, map_canonical_evidence,
    repair_targets, compose_canonical,
)
from real_motion.canonical_repair_context import FixedCanonicalCache
from real_motion.column_runtime_pipeline import prefetch_raw_columns
from real_motion.column_execution import execution_session
from real_motion.nuscenes_adapter import gt_moving_support_sequence
from real_motion.source_evidence_audit import edit_quality
from real_motion.v21_source_induction import stable_json_fingerprint
from tools.real_motion import causal_column_common as columns
from tools.real_motion.pilot_p0_f9_canonical_causal_repair import probabilities, loss_for_causal, timed_full
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import Metrics, sha256
from tools.real_motion.shared_evidence_pilot_common import moving_support_masks, GATES
from tools.real_motion.height_field_screen_common import sync
from real_motion.causal_column_completion import actions_from_probabilities, compose_dense

PROTOCOL = 'p0_f9_point_ccr_gt_screen_v1'
SUPPORT_NOTE = 'canonical historical static/dynamic support; NOT equivalent to old frontier GEN'


def add_args(parser):
    parser.add_argument('--descriptor-cache')
    parser.add_argument('--descriptor-disk-mib', type=int, default=16384)
    parser.add_argument('--descriptor-ram-mib', type=int, default=1024)
    parser.add_argument('--samples-per-role', type=int, default=1024)


def make_head(teacher, device):
    return CanonicalRepairHead(source_dim=teacher.columns.source_dim).to(device)


def model_contract(head):
    return dict(mode='point_CCR', source_dim=head.source_dim, width=head.width)


def contract_extra(args, root):
    if min(args.descriptor_disk_mib, args.descriptor_ram_mib) < 0 or args.samples_per_role < 1:
        raise ValueError('invalid finite CCR cache/sampling budget')
    files = ('real_motion/canonical_causal_repair.py', 'real_motion/canonical_repair_context.py',
             'tools/real_motion/ccr_screen_common.py', 'tools/real_motion/train_p0_f9_point_ccr.py',
             'tools/real_motion/pilot_p0_f9_canonical_causal_repair.py')
    return dict(objective='equal_window_role_action_BCE_causal_MC_GT_only',
                thresholds=dict(CCR_ADD=.5, CCR_REMOVE=.95, old_Local=(.5, .5, .95)),
                samples_per_role=args.samples_per_role, remove_loss_weight=.25,
                support=SUPPORT_NOTE,
                ccr_implementation=stable_json_fingerprint({p: sha256(root/p) for p in files}))


def setup(provider, args):
    # Reclaimable OS file cache is not treated as available tensor memory.
    # Explicit modest RAM bounds avoid duplicating the entire 20k population.
    if args.descriptor_ram_mib > 8192:
        raise ValueError('CCR screen RAM cache is limited to 8GiB; do not use full-population RAM')
    if args.descriptor_cache and args.descriptor_disk_mib:
        import shutil
        parent = Path(args.descriptor_cache).resolve()
        while not parent.exists():
            parent = parent.parent
        if shutil.disk_usage(parent).free < 2*2**30:
            raise RuntimeError('less than 2GiB free: disable descriptor disk writes or free space first')
    provider.ccr_cache = FixedCanonicalCache(args.descriptor_ram_mib, neighbors=False,
        disk_root=args.descriptor_cache, max_disk_mib=args.descriptor_disk_mib, async_writes=True)
    if provider.ccr_cache.disk is not None:
        provider.ccr_cache.disk.reserve = 2*2**30
    provider.ccr_samples_per_role = args.samples_per_role
    print('CCR_FIXED_INPUT_CACHE '+json.dumps(provider.ccr_cache.stats()), flush=True)


def close(provider, result):
    cache = getattr(provider, 'ccr_cache', None)
    if cache is not None:
        if cache.disk is not None:
            cache.disk.flush()
        result['descriptor_cache'] = cache.stats(); cache.close()


@torch.no_grad()
def calibrate_train(provider, source, records, teacher, head, *, progress=None, stop_event=None):
    counts = np.zeros((2, 2, 2), np.int64)
    started = time.perf_counter()
    for wi, (record, raw) in enumerate(prefetch_raw_columns(provider, source, records), 1):
        if stop_event is not None and stop_event.is_set():
            raise InterruptedError('TRAIN-only prior interrupted; restart prior before any update')
        output = teacher.motion(record, provider.device)
        prep = provider.prepare_columns(source, record, include_gt=True, raw_window=raw, outputs=output)
        evidence, _ = provider.ccr_cache.get(prep, provider.pcfg.grid)
        plan = map_canonical_evidence(evidence, prep, provider.pcfg.grid)
        targets, valid = repair_targets(evidence, plan, raw['future_gt_occ'])
        for role in range(2):
            for action in range(2):
                mask = valid[..., action] & ((evidence.actor >= 0) == bool(role))[:, None]
                positive = int(targets[..., action][mask].sum())
                counts[role, action] += (int(mask.sum())-positive, positive)
        if wi == 1 or wi % 32 == 0 or wi == len(records):
            print(f'CCR_TRAIN_PRIOR {wi}/{len(records)} unsampled TRAIN_only', flush=True)
    weights = np.sqrt(counts[..., 0]/np.maximum(counts[..., 1], 1)).clip(1, 32).astype(np.float32)
    head.positive_weight.copy_(torch.as_tensor(weights, device=provider.device))
    report = dict(population='TRAIN-only full unsampled legal CCR action support', windows=len(records),
                  counts=counts.tolist(), positive_weights=weights.tolist(), seconds=time.perf_counter()-started,
                  probability_correction='subtract log(pos_weight); not empirical calibration guarantee')
    if progress:
        progress(dict(event='ccr_train_prior', **report))
    return report


def train_step(provider, rows, teacher, head, optimizer, rng, *, candidate_pool=None):
    if not rows or any(p.requires_grad for p in teacher.parameters()):
        raise RuntimeError('nonempty whole-window batch and frozen epoch19 motion required')
    device = provider.device; sync(device); started = time.perf_counter()
    head.train(); teacher.eval(); optimizer.zero_grad(set_to_none=True)
    stages = defaultdict(float); losses = []; sampled = total = 0
    for record, raw in rows:
        tick = time.perf_counter()
        with torch.no_grad():
            output = teacher.motion(record, device)
            prep = provider.prepare_columns(None, record, include_gt=True, raw_window=raw, outputs=output)
        stages['live_motion_render'] += time.perf_counter()-tick; tick = time.perf_counter()
        evidence, _ = provider.ccr_cache.get(prep, provider.pcfg.grid)
        conflicts = provider.ccr_cache.static_conflicts(evidence, prep, provider.pcfg.grid)
        stages['fixed_input_hash_cache_conflicts'] += time.perf_counter()-tick; tick = time.perf_counter()
        # Sample BEFORE future projection/GT labels. Every class/source/history
        # stratum has positive inclusion probability; N/k restores sums.
        loss, n = loss_for_causal(head, evidence, output, prep, provider.pcfg.grid,
            raw['future_gt_occ'], rng, device, conflicts, per_role=provider.ccr_samples_per_role)
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite CCR loss; previous completed checkpoint preserved')
        stages['sample_live_projection_encoder_loss'] += time.perf_counter()-tick; tick = time.perf_counter()
        (loss/len(rows)).backward()
        stages['backward'] += time.perf_counter()-tick
        losses.append(loss.detach()); sampled += n; total += len(evidence)
        del loss, evidence, prep, output, conflicts
    tick = time.perf_counter(); norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 5.)
    if not torch.isfinite(norm):
        raise RuntimeError('nonfinite CCR gradient; previous completed checkpoint preserved')
    optimizer.step(); sync(device); stages['clip_optimizer_finish'] += time.perf_counter()-tick
    value = float(torch.stack(losses).mean()); optimizer.zero_grad(set_to_none=True)
    return dict(loss=value, grad_norm=float(norm), windows=len(rows), sampled_points=sampled,
                canonical_points=total, seconds=time.perf_counter()-started, stages_seconds=dict(stages),
                transport_frozen=True, GT_only=True, KD=False, optimizer_updated=True,
                allocated_after_mib=torch.cuda.memory_allocated(device)/2**20 if device.type == 'cuda' else 0.)


def old_execution(teacher, provider):
    teacher.columns.column_inference_optimized = True
    teacher.columns.column_async_readback = True
    teacher.columns.column_probability_optimized = False
    teacher.columns.column_sampling_workers = provider.workers
    teacher.columns.column_inference_verify_remaining = 3
    return execution_session(teacher.columns, graphs=True, reuse=False)


@torch.no_grad()
def evaluate(provider, source, records, teacher, head, *, include_old=False, progress=None, stop_event=None):
    teacher.eval(); head.eval(); names = ('static_repair', 'dynamic_repair', 'joint') + (('old_joint',) if include_old else ())
    base = Metrics(); metrics = {name: Metrics() for name in names}
    quality = {name: defaultdict(int) for name in names}
    scenes = defaultdict(lambda: {name: Metrics() for name in ('baseline', *names)})
    started = previous = time.perf_counter(); stages = defaultdict(float)
    with old_execution(teacher, provider) if include_old else nullcontext():
        for wi, (record, raw) in enumerate(prefetch_raw_columns(provider, source, records), 1):
            tick = time.perf_counter(); stages['input_wait'] += tick-previous
            if stop_event is not None and stop_event.is_set():
                raise InterruptedError('CCR evaluation interrupted; resume last completed training checkpoint')
            output = teacher.motion(record, provider.device)
            prep = provider.prepare_columns(source, record, include_gt=True, raw_window=raw, outputs=output)
            stages['prepare_motion_render'] += time.perf_counter()-tick; tick = time.perf_counter()
            # Fresh complete causal domain. Evaluation never samples by GT or
            # reuses TRAIN sampled plans/learned features.
            evidence = build_canonical_evidence(prep, provider.pcfg.grid)
            plan = map_canonical_evidence(evidence, prep, provider.pcfg.grid)
            p = probabilities(head, evidence, plan, output, provider.device)
            predictions = {name: compose_canonical(prep.baseline, evidence, plan, p[..., 0], p[..., 1],
                role={'static_repair': 'static', 'dynamic_repair': 'dynamic', 'joint': 'all'}[name])
                for name in names if name != 'old_joint'}
            stages['fresh_canonical_all_six_probabilities_compose'] += time.perf_counter()-tick; tick = time.perf_counter()
            support = gt_moving_support_sequence(source.nusc, prep.window.t0_token, prep.window.future_tokens,
                tuple(.5*(h+1) for h in range(6)), grid=provider.pcfg.grid, workers=provider.workers)
            moving = moving_support_masks(support, provider.pcfg.grid.shape_hwd)
            for ri, h in enumerate(columns.REPORT):
                gt = raw['future_gt_occ'][h]; scene = scenes[str(record['scene_name'])]
                before = prep.baseline[h]; base.update(ri, before, gt, moving[h]); scene['baseline'].update(ri, before, gt, moving[h])
                if include_old:
                    old_plan = columns.candidate_plan(prep, h, provider.pcfg.grid, teacher.columns.config)
                    old_p = columns.predict_probabilities(teacher.columns, prep, h, old_plan, provider.pcfg.grid, provider.device, 256)
                    predictions['old_joint'] = {h: compose_dense(before, old_plan, actions_from_probabilities(old_plan, old_p, GATES))}
                for name in names:
                    dense = predictions[name][h]
                    metrics[name].update(ri, dense, gt, moving[h]); scene[name].update(ri, dense, gt, moving[h])
                    for key, value in edit_quality(before, dense, gt).items():
                        quality[name][key] += value
            sync(provider.device); previous = time.perf_counter(); stages['moving_old_reference_metrics'] += previous-tick
            if progress:
                progress(dict(event='ccr_evaluation', window=wi, windows=len(records), stages_seconds=dict(stages)))
            if wi == 1 or wi % 16 == 0 or wi == len(records):
                print(f'CCR_EVAL {wi}/{len(records)}', flush=True)
    report = columns.report_states(base, metrics, quality, scenes)
    report.update(windows=len(records), seconds=time.perf_counter()-started, stages_seconds=dict(stages), support=SUPPORT_NOTE)
    return report


@torch.no_grad()
def six_frame_speed(provider, source, records, teacher, head, *, repeats=2, stop_event=None):
    teacher.eval(); head.eval(); trials = []
    with old_execution(teacher, provider):
        for wi, (record, raw) in enumerate(prefetch_raw_columns(provider, source, records, include_gt=False), 1):
            if raw.get('future_gt_occ') is not None:
                raise RuntimeError('FPS cannot load future GT')
            case = dict(record=record, causal=raw)
            expected = {old: timed_full(case, teacher, provider, head, old=old, verify_outputs=True) for old in (True, False)}
            for repeat in range(repeats):
                for old in ((True, False) if repeat % 2 == 0 else (False, True)):
                    if stop_event is not None and stop_event.is_set():
                        raise InterruptedError('CCR FPS stopped; training checkpoint preserved')
                    row = timed_full(case, teacher, provider, head, old=old, verify_outputs=True)
                    if row['dense_sha256'] != expected[old]['dense_sha256']:
                        raise RuntimeError('six-frame execution repeated dense outputs changed')
                    row.update(window=wi, repeat=repeat+1); trials.append(row)
                    print(f"CCR_FPS {row['mode']} {wi}/{len(records)} six_seconds={row['seconds']:.4f} FPS={6/row['seconds']:.2f}", flush=True)
    means = {mode: float(np.mean([r['seconds'] for r in trials if r['mode'] == mode])) for mode in ('old_joint', 'CCR')}
    return dict(trials=trials, six_frame_mean_seconds=means, six_frame_amortized_FPS={k: 6/v for k, v in means.items()},
                speedup=means['old_joint']/means['CCR'],
                boundary='resident source tensors + registered FOUR histories -> fresh prior + live motion + fresh full CCR domain + SIX dense compositions',
                excludes='I/O, initial source extraction/registration, GT/metrics, graph warmup; NOT raw-sensor E2E',
                old_generation_scope_preserved=False, fixed_descriptor_cache_used=False)


def gate(new, old, speed):
    # Diagnostic tolerances, NOT permission to deploy a degraded model.
    return dict(mIoU_within_0_20pp=new['mIoU'] >= old['mIoU']-.2,
                MovingMicro_within_0_20pp=new['MovingMicro'] >= old['MovingMicro']-.2,
                all_horizons_Moving_within_0_20pp=all(new['per_horizon'][h]['MovingMicro'] >= old['per_horizon'][h]['MovingMicro']-.2
                                                    for h in ('1.0', '2.0', '3.0')),
                paired_inference_speedup_ge_3=speed['speedup'] >= 3.)


def brief(result):
    lines = ['===== POINT CCR / GT-ONLY THREE-PASS SCREEN =====', 'status: '+result['status'], 'protocol: '+PROTOCOL,
             '4 histories -> 6 futures; epoch19 motion FROZEN; RANDOM point head; no KD/AE.',
             'Fixed CCR_ADD=0.5 / CCR_REMOVE=0.95; old Local=(0.5,0.5,0.95).',
             'Support: '+SUPPORT_NOTE]
    if 'training_population' in result:
        lines.append('TRAIN '+json.dumps(result['training_population']))
    reports = result.get('reports', {}); initial = reports.get('initial_dev64', {}).get('variants', {}).get('old_joint')
    for row in reports.get('epochs', []):
        m = row['evaluation']['variants']['joint']['metrics']
        lines.append(f"epoch={row['epoch']} update={row['update']} dev64_mIoU={m['mIoU']:.6f} MovingMicro={m['MovingMicro']:.6f}" +
            (f" vs_old_mIoU={m['mIoU']-initial['metrics']['mIoU']:+.6f} vs_old_Moving={m['MovingMicro']-initial['metrics']['MovingMicro']:+.6f}" if initial else ''))
    final = reports.get('final_dev512')
    if final:
        lines.append('===== FINAL DEV512 (development population; NOT independent test) =====')
        old = final['variants']['old_joint']['metrics']
        for name, item in final['variants'].items():
            m = item['metrics']; q = item['quality']
            lines.append(f"{name}: IoU={m['IoU']:.6f} mIoU={m['mIoU']:.6f} MovingMacro={m['MovingMacro']:.6f} MovingMicro={m['MovingMicro']:.6f} "
                f"vs_old_mIoU={m['mIoU']-old['mIoU']:+.6f} vs_old_Moving={m['MovingMicro']-old['MovingMicro']:+.6f} "
                f"add={q.get('added', 0)} remove={q.get('removed', 0)} semantic_precision={q['addition_semantic_precision']}")
    if 'speed' in reports:
        lines += ['SIX_FRAME '+json.dumps(reports['speed']['six_frame_mean_seconds']),
                  'FPS=6/mean_six_frame_latency '+json.dumps(reports['speed']['six_frame_amortized_FPS']),
                  'FPS boundary: '+reports['speed']['boundary'], 'FPS excludes: '+reports['speed']['excludes']]
    for key in ('training', 'gate', 'descriptor_cache'):
        if key in result:
            lines.append(key.upper()+' '+json.dumps(result[key]))
    lines += ['checkpoint: '+result.get('checkpoint', 'not yet saved'), 'route: '+result.get('route', 'in_progress'),
              'No dev-best/threshold search/full4369/automatic extension/promotion; frozen screen speed is NOT joint training speed.']
    if 'error' in result:
        lines.append('error: '+result['error'])
    return '\n'.join(lines)+'\n'
