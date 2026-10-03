#!/usr/bin/env python3
"""Random-init ALL-window joint training, whole-run cosine15/20."""
from __future__ import annotations
import sys
from pathlib import Path
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import copy
import json
import os
import random
import subprocess
import time
from datetime import datetime, timezone
import numpy as np
import torch
from dataclasses import asdict, replace
from real_motion.column_cpu_pipeline import horizon_pipeline_enabled, sampling_worker_budget, cpu_sampling_pool
from real_motion.causal_geometry_cache import CausalGeometryCache
from real_motion.joint_causal_columns import JointCausalColumns, FULL_PROTOCOL, FULL_CONTRACT, FULL4_PROTOCOL, FULL4_CONTRACT, LINK_PROTOCOL
from real_motion.causal_column_completion import ColumnConfig
from real_motion.column_runtime_pipeline import CachedColumnSource, prefetch_raw_columns
from real_motion.local_st_world_model_v17 import config_from_mapping_v17
from real_motion.nuscenes_adapter import NuScenesWindowSource
from real_motion.runtime_config import add_config_args, load_runtime_config, make_prepare_config
from real_motion.v21_source_induction import stable_json_fingerprint
from tools.real_motion.joint_column_full_common import FullJointColumnProvider, prefetch_column_batches, epoch_order, train_full_batch
from tools.real_motion.joint_column_common import count_proposals, weights_from_counts
from tools.real_motion.causal_column_common import calibrate_columns, evaluate_columns, safe_metrics
from tools.real_motion.train_p0_f9_joint_causal_columns import load_joint
from tools.real_motion.train_p0_f9_causal_columns import record_keys
from tools.real_motion.v18_source_interaction_common import select_population, validate_records
from tools.real_motion.eval_p0_f9_v18_se2 import load_cache
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import load_manifest, align_records, sha256, validate_clean_e14_checkpoint
from tools.real_motion.static_evidence_selector_common import CLEAN_SHA256, write_json, atomic_checkpoint, finite_json
from tools.real_motion.train_p0_f9_v18_xy_trajectory import DEV64_FP
from tools.real_motion.joint_training_recovery import (
    save_resume_checkpoint, preserve_training_rng, process_start_token, stop_requested, validate_prior_resume,
)

TRAIN_WINDOWS = 20430
PRIOR_WINDOWS = 1024
CALIBRATION_WINDOWS = 64


class SamplingPoolOwner:
    def __init__(self, workers, *, horizons=False): self.pool = cpu_sampling_pool(workers, horizons=horizons)
    def close(self): self.pool.shutdown(wait=True, cancel_futures=True)


def batch_indices(order, source_counts, windows, sources):
    rows = []; n = 0
    for i in order:
        i = int(i); count = source_counts[i]
        if rows and (len(rows) >= windows or n+count > sources):
            yield tuple(rows); rows = []; n = 0
        rows.append(i); n += count
    if rows: yield tuple(rows)


def training_plans(source_counts, epochs, seed, windows, sources):
    return [list(batch_indices(epoch_order(len(source_counts), seed, e), source_counts, windows, sources))
            for e in range(epochs)]


def full_summary(summary):
    lines = ['===== FULL ONE-STAGE JOINT V18 + CAUSAL COLUMNS =====', f"protocol: {summary.get('protocol', FULL_PROTOCOL)}",
        f"active_history_frames: {summary.get('active_history_frames', 6)}; future_frames: 6",
        f"epochs: {summary['epochs_completed']}/{summary['epochs']}", f"train_windows_per_epoch: {summary['train_windows']}",
        f"successful_updates: {summary['successful_updates']}", f"executed_windows: {summary['executed_windows']}",
        f"window_batch_size: {summary['window_batch_size']}", f"source_budget: {summary['source_budget']}",
        f"paired_control: {summary['paired_control']}", 'initialization: RANDOM, E14 reference only',
        'schedule: whole configured training cycle cosine; NO fixed-LR tail',
        f"TRAIN-only in-sample thresholds: {summary['thresholds']}",
        f"gradient_link_observed: {summary['gradient_link_observed']}"]
    for population in ('dev64', 'all'):
        row = summary['evaluation'][population]
        lines += [f"\n===== {population}: {row['windows']} windows =====", f"learned_transport_mIoU: {row['baseline']['mIoU']:.6f}"]
        for name in ('generation', 'refine', 'joint'):
            item = row['variants'][name]; d = item['delta_vs_v18_pp']; q = item['quality']
            lines += [f"{name}: mIoU={item['metrics']['mIoU']:.6f} dMiOU_vs_transport={d['mIoU']:+.6f} "
                f"dMovingMicro={d['MovingMicro']:+.6f} add={q.get('added',0)} remove={q.get('removed',0)} "
                f"semantic_precision={q['addition_semantic_precision']}"]
        for name, d in row['joint_vs_reference_pp'].items():
            lines += [f"joint vs {name}: dMiOU={d['mIoU']:+.6f} dIoU={d['IoU']:+.6f} dMovingMicro={d['MovingMicro']:+.6f}"]
        lines += [f"gate: {summary['gates'][population]}"]
    lines += [f"route: {summary['route']}", f"stage_seconds: {summary['stage_seconds']}",
        f"elapsed_seconds_this_invocation: {summary['elapsed_seconds']:.2f}",
        f"candidate_checkpoint: {summary['candidate_checkpoint']}",
        'Final-epoch selection only; fixed dev64 monitors, no best-by-dev or automatic retries.',
        'No full4369. All train20430 windows optimized; TRAIN64 calibration is IN-SAMPLE, not held out.']
    return '\n'.join(lines)+'\n'


def main(stop_event=None):
    caches = []; runtime_state = {}
    try:
        try: return _main(stop_event, caches, runtime_state)
        finally:
            # Drain bounded cache writes and sampling work before advertising
            # that the process has exited. A drain failure is still reported.
            for cache in caches: cache.close()
    except BaseException as error:
        if 'write' in runtime_state:
            runtime_state['write']('failed', error=f'{type(error).__name__}: {error}',
                                   recovery='resume the last fully published checkpoint; never save a partial optimizer update')
        raise


def _main(stop_event, caches, runtime_state):
    parser = argparse.ArgumentParser(description=__doc__); add_config_args(parser)
    for key in ('train-cache', 'dev-cache', 'population-manifest', 'base-checkpoint', 'dataroot', 'train-info', 'dev-info', 'out-dir'):
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--history-frames', type=int, choices=(4, 6), help='fresh training defaults to strict four; omitted on resume restores checkpoint history count')
    parser.add_argument('--window-batch-size', type=int, default=4)
    parser.add_argument('--source-budget', type=int, default=128)
    parser.add_argument('--device', default='cuda'); parser.add_argument('--cpu-workers', type=int, default=8)
    parser.add_argument('--frame-cache-mib', type=int, default=256)
    parser.add_argument('--causal-geometry-cache', help='bounded compressed immutable geometry; never GT/model state')
    parser.add_argument('--causal-cache-gib', type=float, default=48.)
    parser.add_argument('--causal-cache-ram-mib', type=int, default=4096)
    parser.add_argument('--eval-batch-size', type=int, default=256)
    parser.add_argument('--checkpoint-every', type=int, default=128)
    parser.add_argument('--seed', type=int, default=20261002)
    parser.add_argument('--paired-control', action='store_true', help='also train matched V18-only (off by default)')
    parser.add_argument('--resume', help='full-protocol last.pt into a NEW output directory; epochs/geometry/recipe must match')
    parser.add_argument('--prewarm-causal-cache', action='store_true', help='explicitly persist all TRAIN history geometry before optimization; first build cost is not free')
    parser.add_argument('--profile-every', type=int, default=0, help='opt-in host/CUDA-stream stage clocks every N updates; 0 leaves normal path unchanged')
    parser.add_argument('--sampling-workers', type=int, default=0, help='0 uses 6 combined native horizon workers (legacy 4); horizon path capped at 8')
    parser.add_argument('--persistent-sampling-pool', action='store_true', help='reuse bounded pure-CPU sampler pool across batches; same RNG/order/objective')
    parser.add_argument('--io-workers', type=int, default=2, help='bounded next-batch window loaders')
    parser.add_argument('--reference-cpu-pipeline', action='store_true', help='diagnostic fallback only; disable parallel warm prepare/shared sparse history/batched render readback')
    args = parser.parse_args(); started = time.perf_counter(); out = Path(args.out_dir)
    from real_motion.native_column_cpu import backend_name, prepare_native
    if backend_name() == 'native':
        print('NATIVE_CPU_PREFLIGHT '+json.dumps(prepare_native()), flush=True)
    if out.exists(): parser.error('NEW output directory required')
    if (min(args.epochs, args.window_batch_size, args.source_budget, args.cpu_workers, args.eval_batch_size, args.checkpoint_every) < 1
            or min(args.frame_cache_mib, args.causal_cache_gib, args.causal_cache_ram_mib) < 0 or not np.isfinite(args.causal_cache_gib)):
        parser.error('positive budgets/epochs required')
    if min(args.profile_every, args.sampling_workers) < 0 or args.io_workers < 1 or (args.prewarm_causal_cache and not args.causal_geometry_cache):
        parser.error('nonnegative profile interval and a cache directory for prewarm required')
    for key in ('config', 'train_cache', 'dev_cache', 'population_manifest', 'base_checkpoint', 'train_info', 'dev_info'):
        if not str(getattr(args, key) or '').strip() or not Path(getattr(args, key)).is_file(): parser.error(f'missing {key}')
    if not Path(args.dataroot).is_dir(): parser.error('missing dataroot')
    if args.resume and not Path(args.resume).is_file(): parser.error('missing resume checkpoint')
    if args.history_frames is None:
        if args.resume:
            metadata = torch.load(args.resume, map_location='cpu', weights_only=False)
            args.history_frames = int(metadata['model_configs']['motion']['history_frames']); del metadata
        else: args.history_frames = 4
    device = torch.device(args.device)
    if device.type == 'cuda' and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()): raise RuntimeError('CUDA/BF16 required')
    torch.set_num_threads(1); torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    if device.type == 'cuda': torch.cuda.manual_seed_all(args.seed)
    cfg = load_runtime_config(args.config, args.override); pcfg = make_prepare_config(cfg); config_sha = stable_json_fingerprint(cfg)
    manifest, dev64, _ = load_manifest(args.population_manifest)
    if len(dev64) != 64 or len(manifest['parent_keys']) != 512 or manifest['selected_key_fingerprint'] != DEV64_FP:
        raise RuntimeError('frozen dev64/dev512 identity/order required')
    meta, records = load_cache(args.train_cache); train_keys = record_keys(records)
    if len(records) != TRAIN_WINDOWS: raise RuntimeError(f'ALL{TRAIN_WINDOWS} training population required')
    dev_scenes = {str(s) for s, _ in manifest['parent_keys']}
    if {s for s, _ in train_keys} & dev_scenes: raise RuntimeError('train/dev scene leakage')
    validate_records(records)
    # ALL train windows are optimized. TRAIN64 is explicitly in-sample, never
    # mislabeled "held-out"; prior1024 uses unsampled counts, not balanced draws.
    prior_keys, cal_keys = select_population(train_keys, dev_scenes, fraction=PRIOR_WINDOWS/len(records)+1e-12,
        calibration_scenes=CALIBRATION_WINDOWS//2, seed=args.seed)
    prior, calibration = align_records(records, prior_keys), align_records(records, cal_keys)
    if len(prior) != PRIOR_WINDOWS or len(calibration) != CALIBRATION_WINDOWS: raise RuntimeError('prior/calibration population mismatch')
    _, dev_all = load_cache(args.dev_cache); record_keys(dev_all)
    dev_keys = tuple(tuple(k) for k in manifest['parent_keys']); dev = align_records(dev_all, dev_keys); del dev_all
    base = torch.load(args.base_checkpoint, map_location='cpu', weights_only=False)
    base_sha = validate_clean_e14_checkpoint(base, args.base_checkpoint, CLEAN_SHA256)
    if not np.isclose(base.get('yaw_weight', 19.), 19.): raise RuntimeError('reference yaw loss mismatch')
    motion_config = replace(config_from_mapping_v17(base['model_config']), history_frames=args.history_frames); del base
    joint = JointCausalColumns(motion_config, ColumnConfig(z_bins=int(pcfg.grid.shape_hwd[2]))).to(device)
    control = copy.deepcopy(joint.transport) if args.paired_control else None
    provider = FullJointColumnProvider(args.base_checkpoint, base_sha, pcfg, device, args.cpu_workers, joint, control)
    source = CachedColumnSource(NuScenesWindowSource(args.dataroot, info_pkl=args.train_info, verbose=False), args.frame_cache_mib)
    dev_source = CachedColumnSource(NuScenesWindowSource(args.dataroot, info_pkl=args.dev_info, verbose=False), args.frame_cache_mib)
    optimizer = torch.optim.AdamW([
        {'params': joint.transport.parameters(), 'lr': 5e-4, 'initial_lr': 5e-4, 'weight_decay': 1e-4},
        {'params': joint.columns.parameters(), 'lr': 3e-4, 'initial_lr': 3e-4, 'weight_decay': .01}])
    control_optimizer = torch.optim.AdamW(control.parameters(), lr=5e-4, weight_decay=1e-4) if control is not None else None
    if control_optimizer: control_optimizer.param_groups[0]['initial_lr'] = 5e-4
    plans = training_plans([len(r['features']) for r in records], args.epochs, args.seed+2, args.window_batch_size, args.source_budget)
    target_updates = sum(map(len, plans)); schedule_steps = target_updates
    protocol, contract = (FULL4_PROTOCOL, FULL4_CONTRACT) if args.history_frames == 4 else (FULL_PROTOCOL, FULL_CONTRACT)
    identity = {'protocol': protocol, 'training_contract': contract, 'source_link': LINK_PROTOCOL, 'mode': 'full',
        'reference_checkpoint_sha256': base_sha, 'runtime_config_fingerprint': config_sha, 'model_configs': joint.configs(),
        'train_keys': train_keys, 'prior_keys': prior_keys, 'calibration_keys': cal_keys, 'dev_keys': dev_keys,
        'dev_manifest_fingerprint': manifest['manifest_fingerprint'], 'seed': args.seed, 'epochs': args.epochs,
        'window_batch_size': args.window_batch_size, 'source_budget': args.source_budget, 'paired_control': args.paired_control,
        'target_updates': target_updates, 'schedule_steps': schedule_steps,
        'info_fingerprints': {'train': sha256(args.train_info), 'dev': sha256(args.dev_info)},
        'cache_fingerprints': {'train': sha256(args.train_cache), 'dev': sha256(args.dev_cache)},
        'patch_resolution_m': float(meta.get('patch_resolution_m', .8))}
    if args.causal_geometry_cache:
        from tools.real_motion.local_warm_cache_common import geometry_namespace
        namespace = geometry_namespace(cfg, provider, identity['info_fingerprints'], identity['cache_fingerprints'], args.dataroot)
        provider.causal_geometry_cache = CausalGeometryCache(args.causal_geometry_cache, namespace,
            max_bytes=int(args.causal_cache_gib*2**30), ram_bytes=args.causal_cache_ram_mib*2**20)
        caches.append(provider.causal_geometry_cache)
        print('CAUSAL GEOMETRY CACHE: '+json.dumps({**provider.causal_geometry_cache.stats(),
            'cold_strong_device': str(device), 'persistence': 'bounded_async_after_main_thread_Strong'},
            ensure_ascii=False), flush=True)
    rng = np.random.default_rng(args.seed+1); weights = None
    cursor_epoch = cursor_batch = updates = successes = executed = sampled = 0; link_observed = False
    history = []; counts = {'generation': np.zeros(2, np.float64), 'refine': np.zeros(3, np.float64)}
    prior_completed = False; prior_cursor = 0
    epoch_totals = {}; epoch_seconds = 0.; epoch_stats_complete = True
    if args.resume:
        ck, restored = load_joint(args.resume, device, reference_sha=base_sha, config_sha=config_sha, allow_diagnostic=True)
        # Whole-run cosine depends on the declared total epochs. Resume cannot
        # silently change that horizon, batch geometry or initial LR schedule.
        if (ck['checkpoint_role'] != 'resume_last'
                or any(stable_json_fingerprint(ck.get(k)) != stable_json_fingerprint(v)
                       for k, v in identity.items())):
            raise RuntimeError('full resume contract/population mismatch; screen checkpoint cannot initialize full training')
        joint.load_state_dict(restored.state_dict()); del restored; optimizer.load_state_dict(ck['optimizer'])
        if control is not None:
            control.load_state_dict(ck['control_state_dict']); control_optimizer.load_state_dict(ck['control_optimizer'])
        cursor_epoch, cursor_batch = ck['cursor_epoch'], ck['cursor_batch']; updates, successes = ck['attempted_updates'], ck['successful_updates']
        executed, sampled, link_observed = ck['executed_windows'], ck['sampled_columns'], ck['gradient_link_observed']
        weights = ck['TRAIN_weights']; history = ck['epoch_history']
        prior_completed, prior_cursor, counts = validate_prior_resume(ck, PRIOR_WINDOWS)
        epoch_totals = copy.deepcopy(ck.get('epoch_totals', {})); epoch_seconds = ck.get('epoch_seconds', 0.)
        epoch_stats_complete = ck.get('epoch_stats_complete', cursor_batch == 0)
        if (type(epoch_stats_complete) is not bool or not np.isfinite(epoch_seconds) or epoch_seconds < 0
                or any(set(v) != {'sum', 'count'} or type(v['count']) is not int
                       or not 0 < v['count'] <= cursor_batch or not np.isfinite(v['sum'])
                       for v in epoch_totals.values())):
            raise RuntimeError('invalid accumulated epoch statistics')
        rng.bit_generator.state = ck['sampling_rng_state']; torch.set_rng_state(ck['torch_rng_state'])
        if 'python_rng_state' in ck: random.setstate(ck['python_rng_state'])
        if 'numpy_global_rng_state' in ck: np.random.set_state(ck['numpy_global_rng_state'])
        if device.type == 'cuda': torch.cuda.set_rng_state_all(ck['cuda_rng_states'])
        if not (0 <= cursor_epoch <= args.epochs and 0 <= updates <= target_updates): raise RuntimeError('invalid resume cursor')
        if cursor_epoch < args.epochs and not 0 <= cursor_batch <= len(plans[cursor_epoch]): raise RuntimeError('invalid batch cursor')
        if cursor_epoch == args.epochs and cursor_batch: raise RuntimeError('completed run cursor must be zero')
        expected_updates = sum(map(len, plans[:cursor_epoch]))+cursor_batch
        expected_windows = cursor_epoch*len(records)+(sum(map(len, plans[cursor_epoch][:cursor_batch])) if cursor_epoch < args.epochs else 0)
        if updates != expected_updates or executed != expected_windows or not 0 <= successes <= updates:
            raise RuntimeError('resume window/update counters inconsistent with epoch order')
    out.mkdir(parents=True); write_json(out/'execution_contract.json', {**identity, 'arguments': vars(args), 'launch_cwd': str(Path.cwd())})
    def write_status(phase, **extra):
        runtime_state['phase'] = phase
        write_json(out/'runtime_status.json', {'phase': phase, 'pid': os.getpid(),
            'process_start_token': process_start_token(os.getpid()), 'out_dir': str(out.resolve()),
            'argv': list(sys.argv), 'updated_utc': datetime.now(timezone.utc).isoformat(),
            'checkpoint': str((out/'last.pt').resolve()), 'attempted_updates': updates,
            'successful_updates': successes, 'cursor_epoch': cursor_epoch, 'cursor_batch': cursor_batch,
            'prior_completed': prior_completed, 'prior_cursor': prior_cursor, **extra})
    runtime_state['write'] = write_status
    write_status('initializing')
    print(f'TRAINING_PROCESS pid={os.getpid()} out_dir={out.resolve()} status={out}/runtime_status.json', flush=True)
    sampling_pool = None
    cpu_horizons = not args.reference_cpu_pipeline and horizon_pipeline_enabled()
    cpu_pool_workers = sampling_worker_budget(args.cpu_workers, args.sampling_workers, horizons=cpu_horizons)
    if args.persistent_sampling_pool:
        owner = SamplingPoolOwner(cpu_pool_workers, horizons=cpu_horizons)
        caches.append(owner); sampling_pool = owner.pool
    print('CPU_PIPELINE '+json.dumps({'task_granularity': 'horizon' if cpu_horizons else 'window',
        'combined_workers': cpu_pool_workers, 'io_workers': args.io_workers,
        'candidate_workers': max(1, cpu_pool_workers//2) if cpu_horizons else cpu_pool_workers,
        'feature_workers': max(1, cpu_pool_workers-cpu_pool_workers//2) if cpu_horizons else cpu_pool_workers,
        'sampling_rng': 'caller_window_horizon_order', 'geometry_ram_mib': args.causal_cache_ram_mib,
        'prefetch': 'one_batch_unchanged', 'nested_pools': False}), flush=True)
    stages = {}; tick = time.perf_counter()
    label = 'FULL RESUME' if args.resume else 'FULL RANDOM INIT'
    print(f'{label}: windows={len(records)} epochs={args.epochs} updates={target_updates} batch<={args.window_batch_size} '
          f'sources<={args.source_budget} whole_cosine_steps={schedule_steps} paired_control={args.paired_control}', flush=True)
    print(f'OBSERVATIONS: {args.history_frames} historical occupancy frames; six futures. Frozen E14 reference is legacy six-frame, NOT a matched four-frame baseline.', flush=True)
    if args.resume:
        print(f'RESTORED checkpoint={args.resume} completed_updates={updates} epoch_cursor={cursor_epoch} '
              f'batch_cursor={cursor_batch} next_update={updates+1} optimizer/RNG/cosine_restored '
              f'prior={"complete_skipped" if prior_completed else str(prior_cursor)+"/"+str(PRIOR_WINDOWS)+"_resumed"}', flush=True)
    with (out/'progress.jsonl').open('x', encoding='utf-8') as handle:
        def progress(row):
            handle.write(json.dumps(finite_json(row), ensure_ascii=False, allow_nan=False)+'\n'); handle.flush()
        if weights is None:
            # A zero-update checkpoint can safely resume an unfinished TRAIN
            # prior. These are the ORIGINAL initial buffers, not new weights.
            weights = {'generation_pos_weight': float(joint.columns.generation_pos_weight),
                'refine_class_weights': joint.columns.refine_class_weights.detach().cpu().tolist(),
                'population': 'pending_original_TRAIN1024_prior_no_training_updates'}
        def payload(role):
            return {**identity, 'checkpoint_role': role, 'screen_pass': False, 'cursor_epoch': cursor_epoch, 'cursor_batch': cursor_batch,
                'attempted_updates': updates, 'successful_updates': successes, 'executed_windows': executed,
                'sampled_columns': sampled, 'gradient_link_observed': link_observed, 'TRAIN_weights': weights, 'epoch_history': history,
                'prior_completed': prior_completed, 'prior_cursor': prior_cursor,
                'prior_counts': {k: v.tolist() for k, v in counts.items()},
                'epoch_totals': epoch_totals, 'epoch_seconds': epoch_seconds, 'epoch_stats_complete': epoch_stats_complete,
                'state_dict': {k: v.detach().cpu().clone() for k, v in joint.state_dict().items()}}
        def save_last():
            state = {**payload('resume_last'), 'optimizer': optimizer.state_dict(), 'sampling_rng_state': rng.bit_generator.state,
                'torch_rng_state': torch.get_rng_state(), 'cuda_rng_states': torch.cuda.get_rng_state_all() if device.type == 'cuda' else [],
                'python_rng_state': random.getstate(), 'numpy_global_rng_state': np.random.get_state()}
            if control is not None: state.update(control_state_dict=control.state_dict(), control_optimizer=control_optimizer.state_dict())
            save_resume_checkpoint(out/'last.pt', state)
            write_status(runtime_state['phase'], checkpoint_saved=True)
        def stop_safely(reason):
            save_last()
            progress({'event': 'stopped_safely', 'update': updates, 'reason': reason, 'checkpoint': str(out/'last.pt')})
            write_status('stopped', reason=reason)
            print(f'STOPPED safely {reason}; completed_update={updates}: {out}/last.pt', flush=True)
            return 130
        save_last()
        if stop_requested(stop_event): return stop_safely('before prior/training')
        if args.prewarm_causal_cache:
            from tools.real_motion.local_warm_cache_common import warm_causal_cache
            print('PREWARM: all TRAIN20430 causal disk entries; no GT/optimizer updates, original Strong device.', flush=True)
            write_status('prewarm')
            try:
                warm = warm_causal_cache(provider, source, records, progress=lambda row:
                    (progress(row), print(f"warm_cache={row['windows']}/{len(records)} seconds={row['seconds']:.1f}", flush=True)),
                    stop_event=stop_event)
            except InterruptedError:
                return stop_safely('during prewarm; cache remains reusable')
            write_json(out/'warm_cache.json', warm); stages['prewarm_causal_cache'] = warm['seconds']
        tick = time.perf_counter()
        if not prior_completed:
            joint.eval(); write_status('prior')
            for wi, (record, raw) in enumerate(prefetch_raw_columns(provider, source, prior[prior_cursor:]), prior_cursor+1):
                if stop_requested(stop_event): return stop_safely('during TRAIN prior')
                prep = provider.prepare_columns(source, record, include_gt=True, raw_window=raw)
                count_proposals(prep, pcfg.grid, joint.columns.config, counts)
                prior_cursor = wi
                if wi == 1 or wi % 32 == 0 or wi == len(prior): print(f'prior_audit={wi}/{len(prior)} TRAIN_only', flush=True)
                if stop_requested(stop_event): return stop_safely('during TRAIN prior')
                if wi % args.checkpoint_every == 0: save_last()
            weights = weights_from_counts(counts)
            weights.update(population='deterministic_TRAIN1024_unsampled_proposals_initial_random_transport', records_used=len(prior),
                complete_train_population=False, key_fingerprint=stable_json_fingerprint(prior_keys))
            joint.columns.generation_pos_weight.fill_(weights['generation_pos_weight'])
            joint.columns.refine_class_weights.copy_(torch.tensor(weights['refine_class_weights'], device=device))
            prior_completed = True
        write_json(out/'TRAIN_prior_counts.json', weights); stages['TRAIN1024_prior'] = time.perf_counter()-tick
        save_last(); training_started = time.perf_counter(); monitor_seconds = 0.
        if stop_requested(stop_event): return stop_safely('before training')
        for e in range(cursor_epoch, args.epochs):
            write_status('training')
            epoch_started = time.perf_counter(); done_batch = cursor_batch if e == cursor_epoch else 0
            scheduled = (records[i] for group in plans[e][done_batch:] for i in group)
            previous_end = time.perf_counter(); recent_time = []; recent_windows = []; aggregate = {}
            for bi, rows in enumerate(prefetch_column_batches(provider, source, scheduled, args.window_batch_size, args.source_budget,
                                                            io_workers=args.io_workers), done_batch+1):
                compute_started = time.perf_counter(); wait = compute_started-previous_end
                if stop_requested(stop_event): return stop_safely('before next update')
                if tuple((r['scene_name'], r['t0_token']) for r, _ in rows) != tuple((records[i]['scene_name'], records[i]['t0_token']) for i in plans[e][bi-1]):
                    raise RuntimeError('batch planning/order mismatch')
                stats = train_full_batch(joint, optimizer, provider, source, rows, rng, updates+1, schedule_steps,
                    probe=updates < 2 or (updates+1) % 128 == 0, patch_resolution=identity['patch_resolution_m'],
                    control=control, control_optimizer=control_optimizer,
                    profile=args.profile_every > 0 and (updates+1) % args.profile_every == 0,
                    sampling_pool=sampling_pool, sampling_workers=args.sampling_workers, optimize_cpu=not args.reference_cpu_pipeline)
                updates += 1; successes += int(stats['optimizer_updated']); executed += stats['windows']; sampled += stats['sampled_columns']
                link_observed |= (stats['source_query_gradient_norm'] or 0.) > 0
                cursor_epoch, cursor_batch = e, bi; wall = time.perf_counter()-compute_started+wait
                epoch_seconds += wall
                recent_time.append(wall); recent_windows.append(stats['windows'])
                if len(recent_time) > 128: recent_time.pop(0); recent_windows.pop(0)
                for k in ('loss', 'motion_loss', 'column_loss', 'generation_bce', 'refine_action_ce'):
                    if k in stats:
                        aggregate.setdefault(k, []).append(stats[k])
                        total = epoch_totals.setdefault(k, {'sum': 0., 'count': 0})
                        total['sum'] += stats[k]; total['count'] += 1
                progress({'event': 'train_full', 'epoch': e+1, 'epoch_batch': bi, 'epoch_batches': len(plans[e]),
                    'update': updates, **stats, 'seconds': wall, 'input_wait_seconds': wait,
                    'learning_rates': [g['lr'] for g in optimizer.param_groups]})
                if bi == 1 or bi % 32 == 0 or bi == len(plans[e]):
                    per_window = sum(recent_time)/max(sum(recent_windows), 1)
                    print(f"epoch={e+1}/{args.epochs} batch={bi}/{len(plans[e])} update={updates}/{target_updates} "
                        f"motion={stats['motion_loss']:.5f} columns={stats['column_loss']:.5f} lr={optimizer.param_groups[0]['lr']:.3g} "
                        f"seconds/window={per_window:.3f} remaining_train_hours_if_current_speed={(args.epochs*len(records)-executed)*per_window/3600:.2f}", flush=True)
                    if hasattr(provider, 'causal_geometry_cache'):
                        print('GEOMETRY_CACHE '+json.dumps(provider.causal_geometry_cache.stats()), flush=True)
                stopping = stop_event is not None and stop_event.is_set()
                if updates % args.checkpoint_every == 0 and not stopping: save_last()
                if stopping:
                    return stop_safely('after completed update')
                previous_end = time.perf_counter()
            epoch_train_seconds = time.perf_counter()-epoch_started
            # Keep the completed epoch recoverable before validation begins.
            save_last(); monitor_started = time.perf_counter(); joint.eval()
            if control is not None: control.eval()
            provider.reference_enabled = True
            write_status('epoch_monitor')
            try:
                with preserve_training_rng(rng):
                    report = evaluate_columns(provider, dev_source, align_records(dev, dev64), joint.columns, (.5, .5, None),
                        progress=progress, batch_size=args.eval_batch_size, diagnostic_thresholds=None, stop_event=stop_event)
            except InterruptedError: return stop_safely('during epoch monitor; completed batches will not replay')
            finally: provider.reference_enabled = False
            monitor_seconds += time.perf_counter()-monitor_started
            row = report['all']; record = {'epoch': e+1, 'training_seconds_this_invocation': epoch_train_seconds,
                'training_means_this_invocation': {k: sum(v)/len(v) for k, v in aggregate.items()},
                'training_seconds_accumulated': epoch_seconds,
                'training_means_accumulated': {k: v['sum']/v['count'] for k,v in epoch_totals.items() if v['count']},
                'training_statistics_complete_epoch': epoch_stats_complete,
                'dev64_fixed_gate': row, 'successful_updates': successes, 'attempted_updates': updates}
            history.append(record); write_json(out/f'monitor_epoch_{e+1:04d}.json', report); write_json(out/'epoch_history.json', history)
            progress({'event': 'epoch_complete', **record})
            d = row['joint_vs_reference_pp']['frozen_E14']
            print(f"EPOCH_RESULT {e+1}: transport={row['baseline']['mIoU']:.6f} joint={row['variants']['joint']['metrics']['mIoU']:.6f} "
                f"vs_E14={d['mIoU']:+.6f} MovingMicro={d['MovingMicro']:+.6f}", flush=True)
            cursor_epoch, cursor_batch = e+1, 0
            epoch_totals = {}; epoch_seconds = 0.; epoch_stats_complete = True
            # Weight-only snapshots are NOT resumable. Last.pt has optimizer/RNG.
            atomic_checkpoint(out/f'epoch_{e+1:04d}.pt', payload('epoch_snapshot')); save_last()
            keep = {10, 14, 15, 20, *range(max(1, e-1), e+2)}
            for p in out.glob('epoch_*.pt'):
                if int(p.stem.split('_')[-1]) not in keep: p.unlink()
            if stop_requested(stop_event): return stop_safely('after epoch monitor')
        stages['training'] = time.perf_counter()-training_started-monitor_seconds; stages['dev64_epoch_monitors'] = monitor_seconds
        if executed != args.epochs*len(records) or updates != target_updates: raise RuntimeError('incomplete full training population')
        joint.eval(); tick = time.perf_counter()
        write_status('final_TRAIN_calibration')
        try:
            with preserve_training_rng(rng):
                gates, calibration_report = calibrate_columns(provider, source, calibration, joint.columns,
                    progress=progress, batch_size=args.eval_batch_size, stop_event=stop_event)
        except InterruptedError: return stop_safely('during final calibration; training already complete')
        calibration_report.update(population='TRAIN64_in_sample_ALL_train_windows_optimized', held_out=False)
        write_json(out/'TRAIN_calibration.json', calibration_report); stages['TRAIN64_calibration'] = time.perf_counter()-tick
        candidate = {**payload('calibrated_candidate'), 'thresholds': gates, 'calibration': calibration_report}
        atomic_checkpoint(out/'candidate.pt', candidate)
        # Reconstructing a module consumes initialization RNG even when all its
        # weights are immediately loaded. Keep diagnostic interruption resumable
        # with exactly the completed training RNG, not that extra initialization.
        with preserve_training_rng(rng):
            ck, persisted = load_joint(out/'candidate.pt', device, reference_sha=base_sha, config_sha=config_sha, allow_diagnostic=True)
        provider.joint, provider.model = persisted, persisted.transport; provider.reference_enabled = True; tick = time.perf_counter()
        write_status('final_dev512')
        try:
            with preserve_training_rng(rng):
                evaluation = evaluate_columns(provider, dev_source, dev, persisted.columns, tuple(gates), progress=progress,
                    batch_size=args.eval_batch_size, dev64_keys=dev64, stop_event=stop_event)
        except InterruptedError: return stop_safely('during final dev512; training already complete')
        stages['final_dev512'] = time.perf_counter()-tick; checks = {}
        for population in ('dev64', 'all'):
            row = evaluation[population]; jm = row['variants']['joint']['metrics']
            checks[population] = {'branches_gate': row['gate']['pass'], 'gradient_link_observed': link_observed,
                'joint_nonnegative_vs_E14': safe_metrics(jm, row['reference_metrics']['frozen_E14'])}
            if control is not None: checks[population]['joint_nonnegative_vs_paired_control'] = safe_metrics(jm, row['reference_metrics']['paired_scratch_V18_only'])
            checks[population]['pass'] = all(checks[population].values())
        passed = all(c['pass'] for c in checks.values()); ck['screen_pass'] = passed; atomic_checkpoint(out/'candidate.pt', ck)
        try: commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        except (OSError, subprocess.CalledProcessError): commit = 'unavailable'
        summary = {**identity, 'git_commit': commit, 'epochs_completed': cursor_epoch, 'train_windows': len(records),
            'active_history_frames': args.history_frames,
            'successful_updates': successes, 'attempted_updates': updates, 'executed_windows': executed, 'sampled_columns': sampled,
            'gradient_link_observed': link_observed, 'thresholds': gates, 'evaluation': evaluation, 'gates': checks,
            'stage_seconds': stages, 'elapsed_seconds': time.perf_counter()-started, 'screen_pass': passed,
            'candidate_checkpoint': str((out/'candidate.pt').resolve()), 'checkpoint_sha256': sha256(out/'candidate.pt'),
            'route': 'full_joint_passed_no_automatic_promotion' if passed else 'full_joint_not_passed_no_automatic_retry'}
        write_json(out/'summary.json', summary); (out/'summary.txt').write_text(full_summary(summary), encoding='utf-8')
        print(full_summary(summary), flush=True)
        write_status('finished', summary=str(out/'summary.txt'))


if __name__ == '__main__':
    import signal
    from threading import Event
    stop_event = Event()
    # Signal handlers only set a flag: never serialize while an update or an
    # atomic checkpoint is incomplete. Existing b700519 runs need the switcher.
    def request_stop(signum, frame): stop_event.set()
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    sys.exit(main(stop_event) or 0)
