"""Frozen GT-pilot DEV512 soft/hard quality and paired six-frame inference latency."""
import json
from pathlib import Path
import signal
import threading
import time
import numpy as np
import torch
from .. import run as legacy
from ..model import JointOccEgo, PROTOCOL
from ..data import HistoryInputs, attach_caches, prefetched_batches
from ..execution import BothActionsExecution
from ..performance.probe import legacy_fingerprint
from ..horizons.probe import execution_fingerprint
from ..diagnostics.run import model_digest, metric_state, restore_metrics
from .engine import ROUTES, SPEED_ROUTES, OPTIMIZED_SPEED_ROUTES, quality_predictions, measure_window, speed_summary
from .receipts import completed_quality
from tools.real_motion.joint_training_recovery import snapshot_checkpoint, preserve_training_rng
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from tools.real_motion.eval_p0_f9_v17_local_stwm import window_from_record
from tools.real_motion.shared_evidence_pilot_common import moving_support_masks
from real_motion.nuscenes_adapter import gt_moving_support_sequence

EVAL_PROTOCOL = 'joint_occ_frozen_soft_hard_dev512_paired_inference_v3'


def validate_state(state, contract):
    n = state['windows']
    if type(n) is not int or not 0 <= n <= len(contract['dev_keys']):
        raise RuntimeError('invalid quality cursor')
    if (n == 0) != (state['metric_counts'] is None):
        raise RuntimeError('quality cursor/count mismatch')
    restore_metrics(state['metric_counts'])
    speed = state['speed_windows']
    routes = contract['speed_routes']
    if len(speed) > len(contract['speed_keys']) or (speed and n != len(contract['dev_keys'])):
        raise RuntimeError('invalid timing cursor')
    for i, row in enumerate(speed):
        if row['key'] != contract['speed_keys'][i] or len(row['samples']) != len(routes)*contract['speed_repeats']:
            raise RuntimeError('timing population/repeats differ')
        pairs = {(s['route'], s['repeat']) for s in row['samples']}
        if pairs != {(r, j) for r in routes for j in range(contract['speed_repeats'])}:
            raise RuntimeError('incomplete paired timing window')
        if any(not np.isfinite(s['seconds']) or s['seconds'] <= 0 for s in row['samples']):
            raise RuntimeError('invalid inference latency')
        if set(row['mismatched_voxels']) != set(routes): raise RuntimeError('missing timing parity')
        if row['six_frame_byte_parity'] != (not any(row['mismatched_voxels'].values())):
            raise RuntimeError('inconsistent timing parity')


def summarize(result):
    lines = ['===== FROZEN GT PILOT / DEV512 SOFT-HARD =====',
        f"status={result['status']}; windows={result['windows']}/512; scientific_updates=0",
        f"source={result['source_checkpoint']}; epoch={result['source_epochs']}; update={result['source_updates']}",
        'FOUR histories -> SIX futures; future GT ego conditioned; no future occupancy/mask/annotation inputs.',
        'Same frozen weights/population/full-grid metrics. Hard ADD0.5 / REMOVE0.95; soft probabilities -> argmax, no threshold tuning.']
    if result['contract'].get('quality_reuse'):
        lines.append('DEV512 quality reused from verified integer counts; NO 512-window re-scoring; old timing NOT reused.')
    if result['windows']:
        lines += ['route                  mIoU        IoU MovingMicro',
            *[f"{name:20s} {m['mIoU']:10.6f} {m['IoU']:10.6f} {m['MovingMicro']:11.6f}"
              for name, m in result['metrics'].items()]]
        for name, m in result['metrics'].items():
            for h in ('1.0', '2.0', '3.0'):
                v = m['per_horizon'][h]
                lines.append(f"{name} {h}s mIoU={v['mIoU']:.6f} IoU={v['IoU']:.6f} MovingMicro={v['MovingMicro']:.6f}")
        for variant in ('transport', 'add_only', 'add_remove'):
            a, b = (result['metrics'][kind+'_'+variant] for kind in ('hard', 'soft'))
            lines.append(f"soft_minus_hard {variant}: dMiOU={b['mIoU']-a['mIoU']:+.6f} dIoU={b['IoU']-a['IoU']:+.6f}")
    speed = result['speed']
    lines += ['===== SAME-WINDOW WARM INFERENCE / ALL SIX FRAMES =====',
        'Timing boundary v2: historical instance extraction/matching/H2D OUTSIDE, fresh Strong/KTA INSIDE (same old FPS boundary).',
        f"windows={speed['windows']}; repeats={result['contract']['speed_repeats']}; "
        f"fresh_Strong_vs_quality_six_frame_byte_parity={speed['six_frame_byte_parity']}; "
        f"valid_same_quality_route_timing={speed['valid_same_quality_route_timing']}"]
    for name, r in speed['routes'].items():
        lines.append(f"{name:20s} six_frame_ms={r['six_frame_latency_ms']:.3f} "
            f"output_frames/s={r['output_frames_per_second']:.3f} p90_ms={r['p90_ms']:.3f} "
            f"peak_reserved_MiB={r['peak_reserved_mib']}")
    if speed.get('optimization_comparison'):
        lines.append('SOFT_OPTIMIZATION '+json.dumps(speed['optimization_comparison']))
        lines.append('SERIALIZED_FIRST_WINDOW_PROFILE_NOT_THROUGHPUT '+json.dumps(speed['serialized_profile_NOT_latency']))
        lines.append('FAST keeps interpolation/source order; three-horizon grouping + one CPU transfer + no backward buffers; NO automatic promotion.')
    lines += [speed['scope'], 'EXCLUDES: '+', '.join(speed['excludes']), speed['notes'],
        'Timing parity failures remain FAIL, never waived. Both soft variants reported; no automatic selection/promotion.',
        'Original weights/Adam/RNG/history caches are read only. Resume SAME output directory.']
    return '\n'.join(lines)+'\n'


def main(argv=None, stop_event=None):
    with preserve_training_rng():
        return _main(argv, stop_event)


def _main(argv, stop_event):
    p = legacy.parser(); p.description = __doc__
    p.add_argument('--resume-evaluation', action='store_true')
    p.add_argument('--reuse-quality-from', help='completed frozen DEV512 evaluation; only remeasure speed in NEW output')
    p.add_argument('--speed-windows', type=int, default=32)
    p.add_argument('--speed-repeats', type=int, default=2)
    p.add_argument('--compare-optimized-speed', action='store_true', help='compare both old/fast soft variants on SAME short population')
    p.add_argument('--fast-horizons', type=int, choices=(1, 2, 3), default=3)
    a = p.parse_args(argv)
    if a.compare_optimized_speed and not (a.reuse_quality_from or a.resume_evaluation):
        raise ValueError('optimization probe REUSES completed quality; no automatic 512-window scoring')
    if a.mode != 'gt' or a.training_backend != 'fused' or a.eval_population != 'dev512':
        raise ValueError('frozen GT-only fused pilot and exact DEV512 required')
    if not 1 <= a.speed_windows <= 64 or not 2 <= a.speed_repeats <= 4:
        raise ValueError('timing requires 1..64 fixed windows and 2..4 alternating repeats')
    if (not 1 <= a.cpu_workers <= 20 or not 1 <= a.prefetch_workers <= 8
            or not 0 <= a.frame_cache_mib <= 8192 or not 0 <= a.geometry_ram_mib <= 4096
            or not 0 <= a.descriptor_ram_mib <= 4096):
        raise ValueError('bounded CPU/RAM budgets required')
    device = legacy.require_cuda(a.device); torch.set_num_threads(1)
    root, out = Path(__file__).resolve().parents[3], Path(a.out_dir).resolve()
    quality_parent = Path(a.reuse_quality_from).resolve() if a.reuse_quality_from else None
    if a.resume_evaluation:
        if not out.is_dir(): raise RuntimeError('existing evaluation directory required')
    else:
        if quality_parent is not None:
            quality_contract, _, _ = completed_quality(quality_parent)
            source = Path(quality_contract['source_checkpoint']).resolve()
            if out == quality_parent or quality_parent in out.parents:
                raise RuntimeError('NEW timing directory OUTSIDE old quality directory required')
        else:
            if not a.resume: raise ValueError('explicit frozen source --resume checkpoint required')
            source = Path(a.resume).resolve()
        if out == source.parent or source.parent in out.parents:
            raise RuntimeError('NEW evaluation directory OUTSIDE original training directory required')
        out.mkdir(parents=True, exist_ok=False)
    with evaluation_lock(out):
        if a.resume_evaluation:
            prior = json.loads((out/'contract.json').read_text(encoding='utf-8'))
            source = Path(prior['source_checkpoint'])
            if quality_parent is None and prior.get('quality_reuse'):
                quality_parent = Path(prior['quality_reuse']['directory'])
        else:
            snapshot_checkpoint(quality_parent/'source_snapshot.pt' if quality_parent else source,
                                out/'source_snapshot.pt')
        digest = legacy.sha256(out/'source_snapshot.pt')
        saved = torch.load(out/'source_snapshot.pt', map_location='cpu', weights_only=False)
        c = saved['contract']
        if (saved.get('protocol') != PROTOCOL or c.get('modes') != ['gt'] or c.get('epochs') != 3
                or saved.get('epoch') != 3 or saved.get('batch') != 0
                or saved.get('updates') != c.get('total_updates') or c.get('training_backend') != 'fused'
                or c.get('implementation') not in (legacy_fingerprint(root), execution_fingerprint(root))):
            raise RuntimeError('completed compatible three-epoch GT-only checkpoint required')
        if c['execution']['torch_version'] != str(torch.__version__) or c['execution']['cuda_version'] != torch.version.cuda:
            raise RuntimeError('use original checkpoint Torch/CUDA environment')
        if c['thresholds'] != [.5, .95]: raise RuntimeError('original fixed ADD0.5/REMOVE0.95 required')
        for key, expected in c['data'].items():
            if legacy.sha256(getattr(a, key)) != expected: raise RuntimeError('source data differs: '+key)
        model = JointOccEgo(legacy.config_from_mapping_v17(c['model']['motion']), width=c['model']['repair']['width']).to(device)
        if model.configs() != c['model']: raise RuntimeError('model architecture differs')
        model.load_state_dict(saved['models']['gt'], strict=True); model.eval().requires_grad_(False)
        initial = model_digest(model)
        pcfg = legacy.make_prepare_config(legacy.load_runtime_config(a.config, a.override))
        provider = legacy.PilotProvider(a.base_checkpoint, legacy.CLEAN_SHA256, pcfg, device, a.cpu_workers, model, None)
        provider.raw_io_workers = 1
        from real_motion.canonical_repair_execution import CanonicalCpuExecution
        from real_motion.surface_projection_execution import SurfaceMapExecution
        from real_motion.strong_majority_execution import ParallelNativeMajority
        provider.ccr_execution = SurfaceMapExecution(CanonicalCpuExecution('native_parallel', min(4, a.cpu_workers)))
        provider.strong_majority = ParallelNativeMajority(min(4, a.cpu_workers))
        sessions = {}
        try:
            train_source = legacy.CachedColumnSource(legacy.NuScenesWindowSource(a.dataroot, info_pkl=a.train_info),
                a.frame_cache_mib, copy_on_insert=False)
            source_dev = legacy.CachedColumnSource(legacy.NuScenesWindowSource(a.dataroot, info_pkl=a.dev_info),
                a.frame_cache_mib, copy_on_insert=False)
            provider.ccr_history_cache_source, provider.ccr_val_history_cache_source = train_source, source_dev
            caches = attach_caches(provider, a, root)
            if {k: caches[k]['namespace'] for k in ('train', 'val')} != c['cache_namespace']:
                raise RuntimeError('original read-only history namespaces required')
            keys = [tuple(k) for k in c['eval_keys']]
            if len(keys) != 512 or len(set(keys)) != 512 or c['eval_population'] != 'dev512':
                raise RuntimeError('source exact DEV512 population required; no intersection or resampling')
            if {k[0] for k in keys} & {k[0] for k in c['train_keys']}: raise RuntimeError('TRAIN/DEV scene overlap')
            _, records = legacy.load_cache(a.dev_cache)
            dev = legacy.align_records(records, keys); del records
            speed_keys = legacy.select_scene_balanced_round_robin(keys, a.speed_windows)
            speed_records = legacy.align_records(dev, speed_keys)
            contract = dict(protocol=EVAL_PROTOCOL, source_checkpoint=str(source), source_sha256=digest,
                source_epochs=saved['epoch'], source_updates=saved['updates'], model_digest=initial,
                dev_keys=keys, speed_keys=speed_keys, speed_repeats=a.speed_repeats,
                thresholds=c['thresholds'], quality_routes=ROUTES,
                speed_routes=OPTIMIZED_SPEED_ROUTES if a.compare_optimized_speed else SPEED_ROUTES,
                compare_optimized_speed=a.compare_optimized_speed, fast_horizons=a.fast_horizons,
                implementation=legacy.stable_json_fingerprint({f.name: legacy.sha256(f) for f in Path(__file__).parent.glob('*.py')}),
                source_implementation=c['implementation'], data=c['data'], cache_namespace=c['cache_namespace'],
                runtime=dict(actual_cuda=device.type == 'cuda', cpu_workers=a.cpu_workers,
                    prefetch_workers=a.prefetch_workers, config_sha256=legacy.sha256(a.config), overrides=a.override))
            contract = json.loads(json.dumps(contract))
            imported = None
            if quality_parent is not None:
                _, imported, receipt = completed_quality(quality_parent, contract)
                contract['quality_reuse'] = receipt
            contract['timing_boundary_version'] = 2
            if a.resume_evaluation:
                if contract != prior: raise RuntimeError('frozen evaluation contract changed; use NEW output')
                state = json.loads((out/'state.json').read_text(encoding='utf-8'))
            else:
                legacy.write_json(out/'contract.json', contract)
                state = imported if imported is not None else dict(windows=0, metric_counts=None,
                    speed_windows=[], quality_seconds=0., input_wait_seconds=0.)
            validate_state(state, contract)
            metrics = restore_metrics(state['metric_counts'])
            descriptors = HistoryInputs(provider, ram_mib=a.descriptor_ram_mib)
            sessions[id(model.columns)] = BothActionsExecution(model.columns, device, graphs=True, capture_full_chunks_only=True)
            provider.paired_eval_sessions = sessions
            begun = time.perf_counter()
            def save():
                state['metric_counts'] = metric_state(metrics) if state['windows'] else None
                legacy.write_json(out/'state.json', state)
            def progress(event, **values):
                with (out/'progress.jsonl').open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(dict(event=event, **values))+'\n')
                print(event+' '+json.dumps(values), flush=True)
            save()
            progress('FROZEN_SOURCE', checkpoint=str(source), epoch=saved['epoch'], update=saved['updates'],
                quality_windows=512, speed_windows=a.speed_windows, resumed_quality=state['windows'], optimizer_steps=0)
            halted = lambda: stop_event is not None and stop_event.is_set()
            for rows in prefetched_batches(provider, source_dev, [[r] for r in dev[state['windows']:]],
                                           descriptors, workers=a.prefetch_workers):
                if halted(): break
                row = rows[0]; tick = time.perf_counter()
                predictions = quality_predictions(model, provider, row, remove_threshold=.95)
                # Label/support access is metric-only, AFTER all six readout routes.
                window = window_from_record(row[0])
                support = gt_moving_support_sequence(source_dev.nusc, window.t0_token, window.future_tokens,
                    tuple(.5*(h+1) for h in range(6)), grid=pcfg.grid, workers=provider.workers)
                moving = moving_support_masks(support, row[1]['future_gt_occ'].shape[1:])
                for name, values in predictions.items():
                    for i, h in enumerate((1, 3, 5)):
                        metrics[name].update(i, values[i], row[1]['future_gt_occ'][h], moving[h])
                state['windows'] += 1; state['quality_seconds'] += time.perf_counter()-tick
                state['input_wait_seconds'] += row[1].get('_joint_prepare_timing', {}).get('input_wait', 0.)
                if state['windows'] % 8 == 0: save()
                if state['windows'] % 32 == 0:
                    progress('SOFT_HARD_DEV512', windows=state['windows'],
                        mIoU={k: m.compute()['mIoU'] for k, m in metrics.items()}, prefix_only=state['windows'] < 512)
                del predictions, moving, support
            save()
            if state['windows'] == 512 and not halted():
                for rows in prefetched_batches(provider, source_dev, [[r] for r in speed_records[len(state['speed_windows']):]],
                                               descriptors, workers=a.prefetch_workers):
                    if halted(): break
                    timing = measure_window(model, provider, rows[0], len(state['speed_windows']), repeats=a.speed_repeats,
                        compare_optimized=a.compare_optimized_speed, fast_horizons=a.fast_horizons)
                    state['speed_windows'].append(timing); save()
                    progress('PAIRED_SIX_FRAME_SPEED', windows=len(state['speed_windows']),
                        total=a.speed_windows, byte_parity=timing['six_frame_byte_parity'])
            save()
            if model_digest(model) != initial: raise RuntimeError('frozen evaluation modified weights')
            if legacy.sha256(out/'source_snapshot.pt') != digest: raise RuntimeError('frozen snapshot changed')
            complete = state['windows'] == 512 and len(state['speed_windows']) == a.speed_windows
            result = dict(status='complete' if complete else 'stopped', contract=contract,
                source_checkpoint=str(source), source_epochs=saved['epoch'], source_updates=saved['updates'],
                source_sha256=digest, windows=state['windows'], scientific_updates=0, source_read_only=True,
                metrics={k: m.compute() for k, m in metrics.items()} if state['windows'] else {},
                speed=speed_summary(state['speed_windows'], actual_cuda=device.type == 'cuda', routes=contract['speed_routes']),
                quality_seconds=state['quality_seconds'], input_wait_seconds=state['input_wait_seconds'],
                seconds_this_invocation=time.perf_counter()-begun,
                route='frozen_readouts_no_automatic_selection_or_promotion')
            legacy.write_json(out/'evaluation.json', result)
            (out/'summary.txt').write_text(summarize(result), encoding='utf-8')
            print(summarize(result), flush=True); print('RESULT: '+str(out/'summary.txt'), flush=True)
        finally:
            for session in sessions.values(): session.close()
            for name in ('ccr_history_cache', 'ccr_val_history_cache', 'ccr_execution', 'strong_majority'):
                resource = getattr(provider, name, None)
                if resource is not None: resource.close()
    return 0


if __name__ == '__main__':
    stopped = threading.Event()
    def stop(*_):
        stopped.set()
        print('Stop requested; save after complete quality/timing window. No training updates.', flush=True)
    signal.signal(signal.SIGINT, stop); signal.signal(signal.SIGTERM, stop)
    main(stop_event=stopped)
