#!/usr/bin/env python3
"""Read-only epoch19 Local joint evaluation: strict4 history -> two6 -> 6s."""
import sys
from pathlib import Path
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import time
import numpy as np
import torch

from real_motion.joint_causal_columns import FULL4_PROTOCOL, FULL4_EXT_PROTOCOL
from real_motion.column_runtime_pipeline import CachedColumnSource, prefetch_raw_columns
from real_motion.nuscenes_adapter import NuScenesWindowSource, gt_moving_support_sequence
from real_motion.runtime_config import add_config_args, load_runtime_config, make_prepare_config
from real_motion.v21_source_induction import stable_json_fingerprint
from tools.real_motion.train_p0_f9_joint_causal_columns import load_joint
from tools.real_motion.joint_column_full_common import EvaluationJointColumnProvider
from tools.real_motion.joint_training_recovery import snapshot_checkpoint
from tools.real_motion.eval_p0_f9_v18_se2 import load_cache
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import load_manifest, sha256
from tools.real_motion.static_evidence_selector_common import CLEAN_SHA256, write_json, finite_json
from tools.real_motion.train_p0_f9_v18_xy_trajectory import DEV64_FP
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from tools.real_motion import joint_long_rollout_common as rollout
from tools.real_motion import causal_column_common as columns
from tools.real_motion import benchmark_p0_f9_v18_runtime as runtime
from real_motion.causal_rollout_handoff import handoff_from_prepared, PROTOCOL as HANDOFF_PROTOCOL

HANDOFF_MODES = ('redetect', 'reconciled', 'transport_history')


def parse_handoff_modes(value):
    modes = tuple(value.split(','))
    if not modes or modes[0] != 'redetect' or len(set(modes)) != len(modes) or set(modes)-set(HANDOFF_MODES):
        raise argparse.ArgumentTypeError('start with redetect, then optional reconciled,transport_history; no duplicates')
    return modes


def forecast_clean_reference(provider, source, record, first_raw, poses):
    """Matched-population E14, native SIX history, no future sensor/label input.

    Only the two older real input frames are loaded here. The second block uses
    all six E14 predictions, reproducing its original inference history budget.
    """
    tokens = tuple(record['history_tokens'])
    if len(tokens) != 6: raise RuntimeError('native E14 comparison requires six initial history tokens')
    # The joint population checks only its last four input keys. Validate the
    # two extra reference keys independently BEFORE loading their occupancy.
    samples = [source.nusc.get('sample', str(t)) for t in tokens]
    if len(set(tokens)) != 6 or len({s['scene_token'] for s in samples}) != 1:
        raise RuntimeError('native E14 history identity/scene mismatch')
    for a, b, token in zip(samples, samples[1:], tokens[1:]):
        gap = (float(b['timestamp'])-float(a['timestamp']))/1e6
        if str(a['next']) != str(token) or not rollout.INTERVAL_BOUNDS_S[0] <= gap <= rollout.INTERVAL_BOUNDS_S[1]:
            raise RuntimeError('native E14 inputs must be contiguous PAST keyframes')
    older = [source.load_semantics(str(record['scene_name']), str(t)) for t in tokens[:2]]
    hist = older+list(first_raw['history_occ'])
    hist_poses = [source.pose(str(t)) for t in tokens[:2]]+list(first_raw['history_poses'])
    def block(history, history_poses, future_poses):
        state = rollout.legacy._build_block_state(history, history_poses, future_poses,
            provider.pcfg, provider.strong, provider.device)
        runtime._stage_gpu_inputs(state, provider.device)
        try: return runtime._forecast_once(provider.reference, state, provider.pcfg, provider.strong, provider.device)
        finally: runtime._release_gpu_inputs(state)
    pred1 = block(hist, hist_poses, poses[:6])
    if not getattr(provider, 'clean_rollout_checked', False):
        # Match the ORIGINAL cached native-six forward/renderer once, not just
        # an internally self-consistent reconstructed reference.
        cached_raw = dict(history_occ=np.stack(hist), history_poses=hist_poses,
            future_poses=poses[:6], future_gt_occ=None)
        state = runtime._prepare_record(record, None, provider.pcfg, provider.strong,
            provider.device, raw_window=cached_raw)
        runtime._stage_gpu_inputs(state, provider.device)
        try:
            exact = runtime._forecast_once(provider.reference, state, provider.pcfg, provider.strong, provider.device)
        finally: runtime._release_gpu_inputs(state)
        rollout.assert_dense_equal(exact, pred1)
        provider.clean_rollout_checked = True
    return pred1+block(pred1, poses[:6], poses[6:])


def summary_text(result):
    def number(x): return 'NA' if x is None or not np.isfinite(x) else f'{x:.6f}'
    metrics = result['metrics']
    lines = ['===== FROZEN LOCAL JOINT ZERO-SHOT 1--6s =====',
        f"protocol: {rollout.PROTOCOL}",
        f"checkpoint_epoch: {result['checkpoint_epoch']}; update: {result['checkpoint_update']}",
        f"population: {result['population']['population']} / {result['windows']} windows / {result['population']['scenes']} scenes",
        f"long_eligible_in_parent: {result['population']['eligible_windows']} / {result['population']['requested_parent_windows']}",
        'history_frames_per_block: 4; future_frames_per_block: 6; rollout_blocks: 2',
        'thresholds: (0.5,0.5,REMOVE-off); weights frozen; NO training/recalibration',
        'history2: first-block predictions at 1.5/2/2.5/3s; NO future GT occupancy/mask/annotation inputs',
        'future ego poses: GT through 6s (trajectory-conditioned forecasting)',
        'synthetic visibility: inherited initial real-history observation union, NOT future sensor visibility',
        'horizon      mIoU        IoU MovingMicro MovingMacro']
    for h, row in metrics['per_horizon'].items():
        lines.append(f"{h:>6}s "+' '.join(f'{number(row[k]):>11}' for k in ('mIoU', 'IoU', 'MovingMicro', 'MovingMacro')))
    for name in ('average_1s_2s_3s', 'average_4s_5s_6s'):
        lines.append(name+': '+json.dumps(finite_json(metrics[name]), ensure_ascii=False))
    lines += ['horizons: nominal keyframe steps; NO timestamp interpolation/retiming',
        'timestamp_audit: '+json.dumps(result['timestamp_audit'])]
    lines += [f"first_block_exactness_passed: {result['first_block_exactness_passed']}",
        'block_edit_totals: '+json.dumps(result['edits']),
        'stage_seconds: '+json.dumps(result['stage_seconds']),
        f"seconds/window: {result['accumulated_window_seconds']/result['windows']:.4f}",
        'No automatic checkpoint selection, retry, promotion, or changes to the 1--3s main table.']
    if 'handoff_comparison' in result:
        lines += ['', '===== MATCHED POPULATION CAUSAL HANDOFF =====',
            'Shared first joint block, frozen thresholds/weights. E14 (if present) retains SIX histories.',
            'route             horizon      mIoU MovingMicro']
        for mode, entry in result['handoff_comparison'].items():
            for h, row in entry['metrics']['per_horizon'].items():
                lines.append(f"{mode:17} {h:>6}s {number(row['mIoU']):>11} {number(row['MovingMicro']):>11}")
            row = entry['metrics']['average_4s_5s_6s']
            lines.append(f"{mode:17} avg4-6s {number(row['mIoU']):>11} {number(row['MovingMicro']):>11}")
        lines += ['handoff_audit: '+json.dumps(result.get('handoff_audit', {})),
            'NO GT-based source assignment, parameter search or automatic winning-route deployment.',
            'These are achievable frozen-inference candidates, NOT a mathematical prediction upper bound.']
    return '\n'.join(lines)+'\n'


def main(stop_event=None):
    parser = argparse.ArgumentParser(description=__doc__); add_config_args(parser)
    for key in ('checkpoint', 'dev-cache', 'population-manifest', 'base-checkpoint', 'dataroot', 'dev-info', 'out-dir'):
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--population', choices=('dev64', 'dev512', 'all'), default='dev64')
    parser.add_argument('--expected-epoch', type=int, default=19)
    parser.add_argument('--device', default='cuda'); parser.add_argument('--cpu-workers', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--feature-backend', choices=('cpu', 'gpu'), default='cpu')
    parser.add_argument('--reference-inference', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--handoff-modes', type=parse_handoff_modes, default=('redetect',))
    parser.add_argument('--compare-e14', action='store_true', help='same selected keys, original six-history E14 rollout')
    args = parser.parse_args(); out = Path(args.out_dir).resolve(); started = time.perf_counter()
    if min(args.cpu_workers, args.batch_size, args.expected_epoch) < 1 or args.cpu_workers > 16:
        parser.error('positive budgets/epoch and at most 16 CPU workers required')
    for key in ('config', 'checkpoint', 'dev_cache', 'population_manifest', 'base_checkpoint', 'dev_info'):
        if not str(getattr(args, key) or '').strip() or not Path(getattr(args, key)).is_file():
            parser.error('missing file '+key)
    if not Path(args.dataroot).is_dir(): parser.error('missing dataroot')
    if args.resume:
        if not (out/'evaluation_state.json').is_file(): parser.error('resume requires evaluation_state.json')
        if (out/'evaluation.json').exists(): parser.error('evaluation already complete; read summary.txt')
    else:
        if out.exists(): parser.error('NEW evaluation directory required; explicitly use --resume')
        out.mkdir(parents=True)
    device = torch.device(args.device)
    if device.type == 'cuda' and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise RuntimeError('CUDA/BF16 required; no silent CPU fallback')
    if args.feature_backend == 'gpu' and device.type != 'cuda': parser.error('GPU feature backend requires CUDA')
    torch.set_num_threads(1)
    from real_motion.native_column_cpu import backend_name, prepare_native
    if backend_name() == 'native': prepare_native()
    with evaluation_lock(out):
        snapshot = out/'checkpoint_snapshot.pt'
        if args.resume:
            previous = json.loads((out/'contract.json').read_text(encoding='utf-8'))
            digest = sha256(snapshot)
            if digest != previous['snapshot_sha256'] or sha256(args.checkpoint) != digest:
                raise RuntimeError('frozen source/snapshot checkpoint changed')
        else: digest = snapshot_checkpoint(args.checkpoint, snapshot)
        cfg = load_runtime_config(args.config, args.override); pcfg = make_prepare_config(cfg)
        if (pcfg.future_frames != 6 or pcfg.free_label != 17
                or not np.isclose(pcfg.frame_dt_s, .5, rtol=0, atol=1e-12)):
            raise RuntimeError('frozen six-future/2Hz/nuScenes semantic contract required')
        config_sha = stable_json_fingerprint(cfg)
        ck, joint = load_joint(snapshot, device, reference_sha=CLEAN_SHA256, config_sha=config_sha, allow_diagnostic=True)
        if (ck['protocol'] not in (FULL4_PROTOCOL, FULL4_EXT_PROTOCOL)
                or joint.transport.config.history_frames != 4 or joint.columns.history_frames != 4
                or ck['cursor_epoch'] != args.expected_epoch or ck['cursor_batch'] != 0):
            raise RuntimeError('requires completed expected epoch, strict4 FULL Local checkpoint')
        if not ck.get('prior_completed', True) or ck.get('successful_updates', 0) <= 0:
            raise RuntimeError('checkpoint has no completed successful training')
        if sha256(args.base_checkpoint) != CLEAN_SHA256: raise RuntimeError('reference E14 changed')
        cache_sha, info_sha = sha256(args.dev_cache), sha256(args.dev_info)
        if cache_sha != ck['cache_fingerprints']['dev'] or info_sha != ck['info_fingerprints']['dev']:
            raise RuntimeError('dev cache/info provenance differs from frozen checkpoint')
        manifest, _, _ = load_manifest(args.population_manifest)
        if (manifest['selected_key_fingerprint'] != DEV64_FP
                or manifest['manifest_fingerprint'] != ck['dev_manifest_fingerprint']
                or tuple(map(tuple, manifest['parent_keys'])) != tuple(map(tuple, ck['dev_keys']))):
            raise RuntimeError('frozen dev64/dev512 identity/order changed')
        _, records = load_cache(args.dev_cache)
        source = CachedColumnSource(NuScenesWindowSource(args.dataroot, info_pkl=args.dev_info, verbose=False), 256)
        selected, population = rollout.select_long_population(records,
            source.iter_windows(history=4, future=12), manifest['parent_keys'], args.population)
        del records
        if {w.scene_name for w, _ in selected} & {s for s, _ in ck['train_keys']}:
            raise RuntimeError('TRAIN/dev scene overlap')
        timestamp_rows = [rollout.validate_timestamps(source.nusc, window) for window, _ in selected]
        timestamp_audit = rollout.summarize_timestamps(timestamp_rows)
        print('TIMESTAMP AUDIT: '+json.dumps(timestamp_audit), flush=True)
        contract = dict(protocol=rollout.PROTOCOL, snapshot_sha256=digest, checkpoint_epoch=args.expected_epoch,
            checkpoint_update=ck['attempted_updates'], config_fingerprint=config_sha,
            dev_cache_sha256=cache_sha, dev_info_sha256=info_sha, dev_manifest_fingerprint=manifest['manifest_fingerprint'],
            population=population, selected_future_tokens=[list(w.future_tokens) for w, _ in selected],
            thresholds=list(rollout.THRESHOLDS), observation_protocol=rollout.OBSERVATION_PROTOCOL,
            feature_backend=args.feature_backend, inference_batch_size=args.batch_size,
            optimized_inference=not args.reference_inference, active_history_frames=4,
            timestamp_audit=timestamp_audit,
            relative_future_frames=6, future_ego_pose_source='GT_through_6s', future_GT_prediction_inputs=False)
        comparison_enabled = args.handoff_modes != ('redetect',) or args.compare_e14
        modes = list(args.handoff_modes)+(['clean_E14_native6'] if args.compare_e14 else [])
        if comparison_enabled:
            contract.update(handoff_modes=modes, handoff_protocol=HANDOFF_PROTOCOL,
                comparison_protocol='shared_first_block_causal_motion_handoff_comparison_v1')
        cursor = 0; raw_counts = rollout.legacy._new_raw(); gate = False
        stages = defaultdict(float); edit_totals = {'first': defaultdict(int), 'second': defaultdict(int)}
        comparison_counts = {mode: rollout.legacy._new_raw() for mode in modes if mode != 'redetect'}
        handoff_totals = defaultdict(float)
        if args.resume:
            saved = json.loads((out/'evaluation_state.json').read_text(encoding='utf-8'))
            cursor, raw_counts = rollout.validate_resume_state(saved, contract, len(selected))
            gate = bool(saved['first_block_exactness_passed'])
            stages.update(saved['stage_seconds'])
            for block in edit_totals: edit_totals[block].update(saved['edits'][block])
            if comparison_enabled:
                stored = saved.get('comparison_raw_counts', {})
                if set(stored) != set(comparison_counts): raise RuntimeError('incomplete handoff resume routes')
                for mode in comparison_counts:
                    checked = {**saved, 'raw_counts': stored[mode]}
                    _, comparison_counts[mode] = rollout.validate_resume_state(checked, contract, len(selected))
                handoff_totals.update(saved.get('handoff_audit_totals', {}))
        else: write_json(out/'contract.json', contract)
        write_json(out/'timestamp_audit.json', dict(summary=timestamp_audit, per_window=timestamp_rows))
        joint.eval().requires_grad_(False)
        provider = EvaluationJointColumnProvider(args.base_checkpoint, CLEAN_SHA256, pcfg, device,
            args.cpu_workers, joint, None)
        model = joint.columns
        model.column_inference_optimized = not args.reference_inference
        model.column_inference_verify_remaining = 3 if not args.reference_inference else 0
        fingerprint = stable_json_fingerprint(contract)
        def save():
            value = dict(protocol=rollout.PROTOCOL, contract_fingerprint=fingerprint,
                completed_windows=cursor, raw_counts={k: v.tolist() for k, v in raw_counts.items()},
                first_block_exactness_passed=gate, stage_seconds=dict(stages),
                edits={k: dict(v) for k, v in edit_totals.items()})
            if comparison_enabled:
                value.update(comparison_raw_counts={mode: {k: v.tolist() for k, v in counts.items()}
                    for mode, counts in comparison_counts.items()}, handoff_audit_totals=dict(handoff_totals))
            write_json(out/'evaluation_state.json', value)
        if not args.resume: save()
        print(f'FROZEN JOINT LONG: epoch={args.expected_epoch} windows={len(selected)} '
              f'parent_eligible={population["eligible_windows"]}/{population["requested_parent_windows"]} '
              f'completed={cursor}; strict4 -> six + six; thresholds=0.5/0.5/REMOVE-off', flush=True)
        if comparison_enabled: print('FIXED HANDOFF ROUTES: '+','.join(modes)+'; first joint block shared; no GT-based tuning', flush=True)
        long_by_key = {(w.scene_name, w.t0_token): w for w, _ in selected}
        iterator = prefetch_raw_columns(provider, source, [r for _, r in selected[cursor:]], include_gt=False)
        try:
          with (out/'progress.jsonl').open('a', encoding='utf-8') as log, ThreadPoolExecutor(max_workers=min(3, args.cpu_workers)) as pool:
            previous_end = time.perf_counter()
            for record, raw in iterator:
                if stop_event is not None and stop_event.is_set(): raise InterruptedError('stopped before next window')
                tick = time.perf_counter(); input_wait = tick-previous_end
                window = long_by_key[(str(record['scene_name']), str(record['t0_token']))]
                t = time.perf_counter()
                prep1 = provider.prepare_columns(source, record, include_gt=False, raw_window=raw)
                first = prep1.raw
                if len(first['history_occ']) != 4 or first.get('future_gt_occ') is not None:
                    raise RuntimeError('first-block four-history/GT-free contract violated')
                preparation = time.perf_counter()-t
                t = time.perf_counter()
                pred1, edits1 = rollout.predict_joint_block(prep1, model, pcfg.grid, device,
                    args.batch_size, args.feature_backend, pool)
                first_forecast = time.perf_counter()-t
                # Gate BOTH source reconstruction and all six FULL joint outputs
                # against the existing normal three-second deployment path once.
                if not gate:
                    t = time.perf_counter()
                    state = rollout.build_four_history_state(first['history_occ'], first['history_poses'],
                        first['future_poses'], pcfg, provider.strong, device)
                    rollout.assert_four_inputs_equal(record, state['rec'])
                    runtime._stage_gpu_inputs(state, device)
                    try:
                        rebuilt = runtime._forecast_once(joint.transport, state, pcfg, provider.strong, device)
                    finally: runtime._release_gpu_inputs(state)
                    rollout.assert_dense_equal(prep1.baseline, rebuilt)
                    del state, rebuilt
                    optimized = model.column_inference_optimized
                    model.column_inference_optimized = False
                    try:
                        _, reference = columns.forecast_columns(provider, source, record, model, rollout.THRESHOLDS, args.batch_size)
                    finally: model.column_inference_optimized = optimized
                    rollout.assert_dense_equal(reference, pred1)
                    gate = True; stages['first_use_exactness'] += time.perf_counter()-t
                    print('FIRST BLOCK EXACTNESS: strict4 rebuilt motion inputs + all6 transport + all6 full joint PASS', flush=True)
                t = time.perf_counter()
                poses = [source.pose(tok) for tok in window.future_tokens]
                prep2 = rollout.synthetic_preparation(pred1, first, poses, window, provider)
                second_prepare = time.perf_counter()-t
                t = time.perf_counter()
                pred2, edits2 = rollout.predict_joint_block(prep2, model, pcfg.grid, device,
                    args.batch_size, args.feature_backend, pool)
                second_forecast = time.perf_counter()-t
                # Bounded to this window; never cache learned poses or handoff
                # features across windows/checkpoints. Compute all routes BEFORE
                # fetching future labels/annotation supports for metrics.
                comparison_predictions = {}; handoff_audit = None
                for mode in args.handoff_modes[1:]:
                    t = time.perf_counter()
                    if mode == 'reconciled':
                        handoff = handoff_from_prepared(prep1, pred1[-1], dt_s=pcfg.frame_dt_s,
                            max_speed_mps=provider.strong.max_match_speed_mps)
                        alternate = rollout.synthetic_preparation(pred1, first, poses, window, provider,
                            motion_handoff=handoff, component_frames=prep2.state['components_by_frame'])
                        handoff_audit = alternate.state['motion_handoff_audit']
                        for key, value in handoff_audit.items():
                            if isinstance(value, (int, float)): handoff_totals[key] += value
                    else:
                        # Mechanism ablation: exclude ALL column additions from
                        # second-block input. This is not an equivalent full
                        # history method and is explicitly reported separately.
                        alternate = rollout.synthetic_preparation(prep1.baseline, first, poses, window, provider)
                    future, _ = rollout.predict_joint_block(alternate, model, pcfg.grid, device,
                        args.batch_size, args.feature_backend, pool)
                    comparison_predictions[mode] = pred1+future
                    stages[mode+'_second_block'] += time.perf_counter()-t
                    del alternate, future
                if args.compare_e14:
                    t = time.perf_counter()
                    comparison_predictions['clean_E14_native6'] = forecast_clean_reference(provider, source, record, first, poses)
                    stages['clean_E14_native6'] += time.perf_counter()-t
                # Metric-only future labels and instance annotations are requested
                # AFTER the full open-loop forecast. Moving uses ORIGINAL t0.
                t = time.perf_counter()
                report_tokens = tuple(window.future_tokens[i] for i in (1, 3, 5, 7, 9, 11))
                moving = gt_moving_support_sequence(source.nusc, window.t0_token, report_tokens,
                    rollout.REPORT_HORIZONS, grid=pcfg.grid, workers=args.cpu_workers)
                predictions = pred1+pred2
                for hi, (idx, tok) in enumerate(zip((1, 3, 5, 7, 9, 11), report_tokens)):
                    gt = source.load_semantics(window.scene_name, tok)
                    rollout.update_metrics(raw_counts, hi, predictions[idx], gt, moving[hi][0], pcfg.free_label)
                    for mode, value in comparison_predictions.items():
                        rollout.update_metrics(comparison_counts[mode], hi, value[idx], gt, moving[hi][0], pcfg.free_label)
                metric_seconds = time.perf_counter()-t
                for block, edits in (('first', edits1), ('second', edits2)):
                    for key, value in edits.items(): edit_totals[block][key] += value
                times = dict(input_wait=input_wait, first_prepare=preparation, first_forecast=first_forecast,
                    second_prepare=second_prepare, second_forecast=second_forecast, metrics=metric_seconds)
                times['total_window'] = time.perf_counter()-tick+input_wait
                for key, value in times.items(): stages[key] += value
                cursor += 1; save()
                row = dict(event='long_rollout_window_complete', window=cursor, windows=len(selected),
                    scene_name=window.scene_name, t0_token=window.t0_token, seconds=times,
                    second_block_sources=len(prep2.state['current']), first_edits=edits1, second_edits=edits2)
                if comparison_enabled: row.update(handoff_audit=handoff_audit, compared_routes=modes)
                log.write(json.dumps(row)+'\n'); log.flush()
                print(f'joint_long_rollout={cursor}/{len(selected)} seconds={times["total_window"]:.3f} '
                      f'block2_sources={len(prep2.state["current"])}'
                      +(f' linked={handoff_audit["matched_sources"]}' if handoff_audit else ''), flush=True)
                del prep1, prep2, pred1, pred2, predictions, comparison_predictions, first, raw
                if stop_event is not None and stop_event.is_set(): raise InterruptedError('stopped at completed window')
                previous_end = time.perf_counter()
        except InterruptedError:
            save(); write_json(out/'evaluation_status.json', dict(status='interrupted', completed_windows=cursor,
                windows=len(selected), source_checkpoint_unchanged=True))
            print(f'STOPPED at completed window {cursor}; resume with same arguments and --resume', flush=True)
            return 130
        finally: iterator.close()
        if sha256(snapshot) != digest: raise RuntimeError('immutable checkpoint snapshot changed')
        result = dict(status='complete', protocol=rollout.PROTOCOL, checkpoint_epoch=args.expected_epoch,
            checkpoint_update=ck['attempted_updates'], checkpoint_snapshot=str(snapshot), snapshot_sha256=digest,
            source_checkpoint=str(Path(args.checkpoint).resolve()), population=population,
            windows=len(selected), thresholds=list(rollout.THRESHOLDS), first_block_exactness_passed=gate,
            metrics=rollout.finalize_metrics(raw_counts), raw_counts={k: v.tolist() for k, v in raw_counts.items()},
            observation_protocol=rollout.OBSERVATION_PROTOCOL, stage_seconds=dict(stages),
            edits={k: dict(v) for k, v in edit_totals.items()}, accumulated_window_seconds=stages['total_window'],
            seconds_this_invocation=time.perf_counter()-started, contract_fingerprint=fingerprint,
            history_frames=4, future_frames_per_block=6, rollout_blocks=2, future_GT_prediction_inputs=False,
            report_horizons_s=list(rollout.REPORT_HORIZONS),
            timestamp_audit=timestamp_audit,
            future_ego_pose_used_through_s=6, no_training=True, no_dev_threshold_selection=True)
        if comparison_enabled:
            all_counts = {'redetect': raw_counts, **comparison_counts}
            total_matched = handoff_totals.get('matched_sources', 0)
            audit = dict(protocol=HANDOFF_PROTOCOL, **dict(handoff_totals))
            audit.update(matched_fraction=total_matched/max(handoff_totals.get('current_sources', 0), 1),
                mean_velocity_correction_mps=handoff_totals.get('velocity_correction_mps_sum', 0)/max(total_matched, 1),
                mean_shape_origin_offset_m=handoff_totals.get('centroid_shape_offset_m_sum', 0)/max(total_matched, 1))
            result.update(handoff_comparison={mode: dict(metrics=rollout.finalize_metrics(counts),
                raw_counts={k: v.tolist() for k, v in counts.items()},
                history_frames=6 if mode == 'clean_E14_native6' else 4) for mode, counts in all_counts.items()},
                handoff_audit=audit,
                comparison_protocol=contract['comparison_protocol'], no_automatic_route_selection=True)
        result = finite_json(result)
        write_json(out/'evaluation.json', result); write_json(out/'evaluation_status.json', dict(status='complete', completed_windows=cursor))
        (out/'summary.txt').write_text(summary_text(result), encoding='utf-8')
        print(summary_text(result), flush=True); return 0


if __name__ == '__main__':
    import signal
    from threading import Event
    event = Event()
    def stop(signum, frame): event.set()
    signal.signal(signal.SIGINT, stop); signal.signal(signal.SIGTERM, stop)
    sys.exit(main(event))
