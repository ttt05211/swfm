#!/usr/bin/env python3
"""Independent RANDOM-init GT, Pred or paired screen; no legacy promotion."""
import sys
from pathlib import Path
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import copy
from collections import defaultdict
from dataclasses import replace
import json
import random
import signal
import threading
import time
import numpy as np
import torch
from real_motion.local_st_world_model_v17 import config_from_mapping_v17
from real_motion.runtime_config import add_config_args, load_runtime_config, make_prepare_config
from real_motion.column_runtime_pipeline import CachedColumnSource
from real_motion.nuscenes_adapter import NuScenesWindowSource
from real_motion.v21_source_induction import select_scene_balanced_round_robin, stable_json_fingerprint
from tools.real_motion.run_p0_f9_shared_evidence_pilot import PilotProvider, require_cuda
from tools.real_motion.eval_p0_f9_v18_se2 import load_cache
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import load_manifest, align_records, sha256
from tools.real_motion.static_evidence_selector_common import CLEAN_SHA256, write_json
from tools.real_motion.train_p0_f9_causal_columns import record_keys
from tools.real_motion.height_field_screen_common import epoch_groups
from tools.real_motion.joint_training_recovery import save_resume_checkpoint, preserve_training_rng
from tools.joint_occ_experiments.model import JointOccEgo, PROTOCOL
from tools.joint_occ_experiments.data import attach_caches, HistoryInputs, prefetched_batches
from tools.joint_occ_experiments.engine import train_arm, evaluate
from tools.joint_occ_experiments.execution import BothActionsExecution


def rng_state():
    return dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                numpy=np.random.get_state(), python=random.getstate())


def restore_rng(state):
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']: torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])
    np.random.set_state(state['numpy']); random.setstate(state['python'])


def paired_update(models, optimizers, provider, rows, **kwargs):
    if len(models) == 1:
        mode = next(iter(models))
        return {mode: train_arm(models[mode], optimizers[mode], provider, rows, mode, **kwargs)}
    initial = rng_state()
    gt = train_arm(models['gt'], optimizers['gt'], provider, rows, 'gt', **kwargs)
    after = rng_state()
    restore_rng(initial)
    try:
        pred = train_arm(models['pred'], optimizers['pred'], provider, rows, 'pred', **kwargs)
    finally:
        restore_rng(after)
    return dict(gt=gt, pred=pred)


def checkpoint(models, optimizers, contract, epoch, batch, updates, reports):
    return dict(protocol=PROTOCOL, contract=contract, epoch=epoch, batch=batch, updates=updates,
        models={k: m.state_dict() for k, m in models.items()},
        optimizers={k: o.state_dict() for k, o in optimizers.items()}, rng=rng_state(), reports=reports)


def restore_checkpoint(saved, models, optimizers, contract):
    if saved.get('protocol') != PROTOCOL or saved.get('contract') != contract:
        raise RuntimeError('resume protocol/data/model/schedule/objective differs; legacy checkpoints are NOT resumable here')
    for key in models:
        models[key].load_state_dict(saved['models'][key], strict=True)
        optimizers[key].load_state_dict(saved['optimizers'][key])
    restore_rng(saved['rng'])
    return int(saved['epoch']), int(saved['batch']), int(saved['updates']), saved['reports']


def summarize(result):
    modes = result.get('contract', {}).get('modes', ['gt', 'pred'])
    lines = ['===== JOINT OCCUPANCY / INTERNAL EGO =====', f"status={result['status']}",
        f"active_modes={modes}; RANDOM init; four histories -> six futures; independent complete models.",
        f"training_backend={result.get('contract', {}).get('training_backend', 'reference')}",
        'GT uses GT future poses; Pred uses INTERNAL predicted poses everywhere (no alignment).',
        'Navigation-conditioned, GT-derived destination commands; NOT navigation-free.',
        'Final full-grid CE+Lovasz + direct source geometry/existence + direct R10 ego geometry.',
        'TRAIN uses a continuous surrogate; INFERENCE uses hard Strong/A1/CCR; soft != hard.',
        'ADD=0.5; REMOVE=0.95; ADD-only and ADD+REMOVE both reported; no threshold search.',
        f"updates_each={result.get('updates',0)}; completed_epochs={result.get('epoch',0)}",
        'No fitted TRAIN prior, new frozen feature bank, legacy weight/optimizer/cache writes or automatic retry.']
    if 'performance' in result: lines.append('PERFORMANCE '+json.dumps(result['performance'], ensure_ascii=False))
    report = result.get('evaluation')
    if report:
        lines.append(f"===== FIXED {result['contract']['eval_population']} / {report['windows']} WINDOWS =====")
        for mode, variants in report['variants'].items():
            for name, m in variants.items():
                lines.append(f"{mode}_{name:12s} mIoU={m['mIoU']:.6f} IoU={m['IoU']:.6f} MovingMicro={m['MovingMicro']:.6f}")
                for h, row in m['per_horizon'].items():
                    lines.append(f"  {h}s mIoU={row['mIoU']:.6f} IoU={row['IoU']:.6f} MovingMicro={row['MovingMicro']:.6f}")
        lines.append('ego errors: '+json.dumps(report['trajectory']))
        if report.get('surrogate_gap'):
            gap = report['surrogate_gap']
            lines.append(f"===== SOFT/HARD GAP / FIRST {gap['windows']} WINDOWS ONLY =====")
            for mode, variants in gap['variants'].items():
                lines.append(f"{mode}: soft_argmax_mIoU={variants['soft_argmax']['mIoU']:.6f} "
                    f"hard_mIoU={variants['hard_same_windows']['mIoU']:.6f}; diagnostic, no route selection")
    lines.append('This is a 3-epoch PILOT, not a converged paper result or verification of legacy FPS.')
    if 'error' in result: lines.append('error: '+result['error'])
    return '\n'.join(lines)+'\n'


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    add_config_args(p)
    for name in ('train-cache', 'dev-cache', 'population-manifest', 'base-checkpoint', 'dataroot',
                 'train-info', 'dev-info', 'ccr-history-cache', 'ccr-val-history-cache', 'out-dir'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--mode', choices=('gt', 'pred', 'paired'), default='paired')
    p.add_argument('--training-backend', choices=('fused', 'reference'), default='fused',
                   help='same objective; reference retained for gradient/speed diagnostics')
    p.add_argument('--train-fraction', type=float, default=.2)
    p.add_argument('--seed', type=int, default=20261011)
    p.add_argument('--window-batch-size', type=int, default=4)
    p.add_argument('--source-budget', type=int, default=128)
    p.add_argument('--device', default='cuda')
    p.add_argument('--cpu-workers', type=int, default=10)
    p.add_argument('--prefetch-workers', type=int, default=4)
    p.add_argument('--frame-cache-mib', type=int, default=2048)
    p.add_argument('--geometry-ram-mib', type=int, default=512)
    p.add_argument('--descriptor-ram-mib', type=int, default=512)
    p.add_argument('--motion-lr', type=float, default=5e-4)
    p.add_argument('--repair-lr', type=float, default=3e-4)
    p.add_argument('--ego-lr', type=float, default=3e-4)
    p.add_argument('--object-weight', type=float, default=1.)
    p.add_argument('--ego-weight', type=float, default=1.)
    p.add_argument('--lovasz-weight', type=float, default=1.)
    p.add_argument('--remove-threshold', type=float, default=.95)
    p.add_argument('--eval-population', choices=('dev64', 'dev512'), default='dev512')
    p.add_argument('--checkpoint-every', type=int, default=64)
    p.add_argument('--max-updates', type=int, default=0, help='safe stop; configured cosine unchanged')
    p.add_argument('--resume', help='this protocol ONLY; SAME output directory allowed')
    p.add_argument('--evaluate-only', action='store_true', help='requires --resume, read-only weights/optimizer')
    return p


def main(stop_event=None, argv=None):
    a = parser().parse_args(argv)
    root, out = Path(__file__).resolve().parents[2], Path(a.out_dir)
    if out.exists() and not a.resume: raise RuntimeError('NEW experiment directory required')
    if a.evaluate_only and not a.resume: raise ValueError('evaluate-only requires this experiment checkpoint')
    if (not 0 < a.train_fraction <= 1 or min(a.epochs, a.window_batch_size, a.source_budget, a.checkpoint_every) < 1
            or not 1 <= a.cpu_workers <= 20 or not 1 <= a.prefetch_workers <= 8 or a.max_updates < 0
            or min(a.geometry_ram_mib, a.frame_cache_mib, a.descriptor_ram_mib) < 0
            or max(a.geometry_ram_mib, a.descriptor_ram_mib) > 4096 or a.frame_cache_mib > 8192
            or not .5 <= a.remove_threshold <= 1
            or any(not np.isfinite(x) or x <= 0 for x in (a.motion_lr, a.repair_lr, a.ego_lr, a.object_weight, a.ego_weight))
            or not np.isfinite(a.lovasz_weight) or a.lovasz_weight < 0):
        raise ValueError('invalid explicit data/CPU/RAM/optimizer/loss budgets')
    for name in ('config', 'train_cache', 'dev_cache', 'population_manifest', 'base_checkpoint', 'train_info', 'dev_info'):
        if not Path(getattr(a, name) or '').is_file(): raise RuntimeError('missing '+name)
    device = require_cuda(a.device)
    torch.set_num_threads(1)
    torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    random.seed(a.seed); np.random.seed(a.seed)
    pcfg = make_prepare_config(load_runtime_config(a.config, a.override))
    metadata = torch.load(a.base_checkpoint, map_location='cpu', weights_only=False)
    config = replace(config_from_mapping_v17(metadata['model_config']), history_frames=4)
    del metadata
    _, all_train = load_cache(a.train_cache)
    _, all_dev = load_cache(a.dev_cache)
    record_keys(all_train); record_keys(all_dev)
    if len(all_train) != 20430 or len(all_dev) != 4369:
        raise RuntimeError('frozen TRAIN20430 / VAL4369 required')
    manifest, dev64, _ = load_manifest(a.population_manifest)
    parent = [tuple(k) for k in manifest['parent_keys']]
    if len(parent) != 512 or len(dev64) != 64: raise RuntimeError('frozen DEV64/512 manifest required')
    train_keys = select_scene_balanced_round_robin(record_keys(all_train), max(1, round(len(all_train)*a.train_fraction)))
    train = align_records(all_train, train_keys)
    evaluation = align_records(all_dev, dev64 if a.eval_population == 'dev64' else parent)
    if {r['scene_name'] for r in train} & {r['scene_name'] for r in evaluation}: raise RuntimeError('TRAIN/DEV scene overlap')
    groups = [epoch_groups(train, a.seed, e, a.window_batch_size, a.source_budget) for e in range(a.epochs)]
    total_updates = sum(len(g) for g in groups)
    modes = ['gt', 'pred'] if a.mode == 'paired' else [a.mode]
    first_model = JointOccEgo(config).to(device)
    models = {modes[0]: first_model}
    if len(modes) == 2: models[modes[1]] = copy.deepcopy(first_model)
    # Base checkpoint supplies architecture + cache contract only. NO weights copied.
    provider = PilotProvider(a.base_checkpoint, CLEAN_SHA256, pcfg, device, a.cpu_workers, first_model, None)
    provider.raw_io_workers = 1
    from real_motion.canonical_repair_execution import CanonicalCpuExecution
    from real_motion.surface_projection_execution import SurfaceMapExecution
    from real_motion.strong_majority_execution import ParallelNativeMajority
    provider.ccr_execution = SurfaceMapExecution(CanonicalCpuExecution('native_parallel', min(4, a.cpu_workers)))
    provider.strong_majority = ParallelNativeMajority(min(4, a.cpu_workers))
    train_source = CachedColumnSource(NuScenesWindowSource(a.dataroot, info_pkl=a.train_info), a.frame_cache_mib, copy_on_insert=False)
    dev_source = CachedColumnSource(NuScenesWindowSource(a.dataroot, info_pkl=a.dev_info), a.frame_cache_mib, copy_on_insert=False)
    provider.ccr_history_cache_source, provider.ccr_val_history_cache_source = train_source, dev_source
    cache_report = attach_caches(provider, a, root)
    descriptors = HistoryInputs(provider, ram_mib=a.descriptor_ram_mib)
    optimizers = {mode: torch.optim.AdamW([
        dict(params=model.transport.parameters(), lr=a.motion_lr, initial_lr=a.motion_lr, weight_decay=1e-4),
        dict(params=model.columns.parameters(), lr=a.repair_lr, initial_lr=a.repair_lr, weight_decay=.01),
        dict(params=model.ego.parameters(), lr=a.ego_lr, initial_lr=a.ego_lr, weight_decay=.01)])
        for mode, model in models.items()}
    files = sorted(Path(__file__).parent.glob('*.py'))
    dependencies = ('real_motion/joint_surface_ccr.py', 'real_motion/joint_causal_columns.py',
        'real_motion/local_st_world_model_v18_se2.py', 'real_motion/local_st_world_model_v17.py',
        'real_motion/surface_canonical_repair.py', 'real_motion/surface_projection_execution.py',
        'real_motion/surface_ccr_execution.py', 'real_motion/ego_navigation.py',
        'tools/ego_experiments/ego_kinematic.py', 'tools/real_motion/joint_training_recovery.py')
    contract = dict(protocol=PROTOCOL, modes=modes, training_backend=a.training_backend,
        model=first_model.configs(), epochs=a.epochs, seed=a.seed,
        train_keys=[list(k) for k in train_keys], eval_keys=[[r['scene_name'], r['t0_token']] for r in evaluation],
        eval_population=a.eval_population, window_batch=a.window_batch_size, source_budget=a.source_budget,
        total_updates=total_updates, lr=[a.motion_lr, a.repair_lr, a.ego_lr],
        weights=[a.object_weight, a.ego_weight, a.lovasz_weight], thresholds=[.5, a.remove_threshold],
        schedule='whole_cycle_cosine_floor_0.01', initialization='identical_RANDOM_no_teacher',
        cache_namespace={k: cache_report[k]['namespace'] for k in ('train', 'val')},
        data={key: sha256(getattr(a, key)) for key in ('train_cache', 'dev_cache', 'train_info', 'dev_info', 'base_checkpoint')},
        execution=dict(torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
            motion_precision='BF16' if device.type == 'cuda' else 'FP32', device_type=device.type),
        implementation=stable_json_fingerprint({**{str(p.relative_to(root)): sha256(p) for p in files},
                                               **{p: sha256(root/p) for p in dependencies}}))
    epoch = batch = updates = 0
    reports = dict(training=[], gradient_links={}, epochs=[])
    if a.resume:
        saved = torch.load(a.resume, map_location='cpu', weights_only=False)
        epoch, batch, updates, reports = restore_checkpoint(saved, models, optimizers, contract)
        if (not 0 <= epoch <= a.epochs or epoch == a.epochs and batch != 0
                or epoch < a.epochs and not 0 <= batch < len(groups[epoch])
                or updates != sum(len(g) for g in groups[:epoch])+batch):
            raise RuntimeError('invalid completed optimizer-update cursor')
        print(f'JOINT_OCC_RESUME modes={modes} completed_update={updates} epoch={epoch} batch={batch}; Adam/RNG/cosine intact', flush=True)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out/'cache_audit.json', cache_report)
    write_json(out/'contract.json', contract)
    print(f'JOINT_OCC RANDOM modes={modes} backend={a.training_backend} train={len(train)} '
          f'epochs={a.epochs} updates_each={total_updates}; full6 CE/Lovasz; no new persistent cache/prior', flush=True)
    result = dict(status='running', contract=contract, epoch=epoch, updates=updates)
    tick = time.perf_counter()
    def progress(row):
        with (out/'progress.jsonl').open('a', encoding='utf-8') as handle: handle.write(json.dumps(row)+'\n')
    def save():
        save_resume_checkpoint(out/'last.pt', checkpoint(models, optimizers, contract, epoch, batch, updates, reports))
    healthy = True
    try:
        if not a.evaluate_only:
            while epoch < a.epochs:
                batches = prefetched_batches(provider, train_source, groups[epoch][batch:], descriptors,
                                               workers=a.prefetch_workers)
                for rows in batches:
                    if stop_event is not None and stop_event.is_set(): break
                    healthy = False
                    pair_tick = time.perf_counter()
                    stats = paired_update(models, optimizers, provider, rows, update=updates+1, total_updates=total_updates,
                        object_weight=a.object_weight, ego_weight=a.ego_weight, lovasz_weight=a.lovasz_weight,
                        optimized=a.training_backend == 'fused',
                        probe=updates == 0)
                    # Publish cursor ONLY after ALL active arms finish Adam updates.
                    updates += 1; batch += 1
                    if batch == len(groups[epoch]): epoch += 1; batch = 0
                    healthy = True
                    preparing = rows[0][1].get('_joint_prepare_timing', {})
                    item = dict(event='paired_train' if len(modes) == 2 else 'single_train',
                        modes=modes, update=updates, completed_epochs=epoch, epoch_batch=batch,
                        paired_compute_seconds=time.perf_counter()-pair_tick,
                        prepare_seconds=preparing, **stats)
                    item['cpu_worker_seconds_sum_overlapping'] = {
                        name: sum(row[1].get('_joint_worker_timing', {}).get(name, 0.) for row in rows)
                        for name in ('io_and_ego_metadata', 'history_descriptors')}
                    progress(item)
                    reports['training'].append(item)
                    if updates == 1: reports['gradient_links'] = {k: s['gradient_links'] for k, s in stats.items()}
                    if updates % 16 == 0 or updates == 1:
                        losses = ' '.join(f"{mode}_CE={stats[mode]['CE']:.4f}" for mode in modes)
                        print(f"JOINT_OCC {updates}/{total_updates} completed_epochs={epoch} {losses} "
                            f"seconds/window_all_active_modes_including_input="
                            f"{(item['paired_compute_seconds']+sum(preparing.values()))/len(rows):.4f}", flush=True)
                    if updates % a.checkpoint_every == 0 or batch == 0: save()
                    if batch == 0 or a.max_updates and updates >= a.max_updates: break
                batches.close()
                if stop_event is not None and stop_event.is_set() or a.max_updates and updates >= a.max_updates: break
            save()
        result.update(epoch=epoch, updates=updates)
        rows = reports['training'][-64:]
        performance = {}
        for mode in models:
            count = sum(row[mode]['windows'] for row in rows)
            stages = defaultdict(float)
            for row in rows:
                for name, duration in row[mode]['stages_seconds'].items(): stages[name] += duration
            performance[mode] = dict(windows=count, seconds_per_window=sum(row[mode]['seconds'] for row in rows)/max(count, 1),
                host_stage_seconds_per_window={k: v/max(count, 1) for k, v in stages.items()},
                stages_are_host_elapsed_not_GPU_utilization=True)
        result['performance'] = performance
        count = sum(row[modes[0]]['windows'] for row in rows)
        performance['paired_pipeline'] = dict(windows=count,
            active_modes=modes,
            seconds_per_window_including_input=sum(row['paired_compute_seconds']+
                sum(row['prepare_seconds'].values()) for row in rows)/max(count, 1),
            input_wait_seconds_per_window=sum(row['prepare_seconds'].get('input_wait', 0.) for row in rows)/max(count, 1),
            descriptors_seconds_per_window=sum(row['prepare_seconds'].get('descriptors', 0.) for row in rows)/max(count, 1),
            excludes='checkpoint writes, startup and quality evaluation')
        reports['cache_audit'] = cache_report
        reports['descriptor_stats'] = descriptors.stats()
        result['descriptor_stats'] = reports['descriptor_stats']
        if (epoch == a.epochs or a.evaluate_only) and not (stop_event is not None and stop_event.is_set()):
            population_fp = stable_json_fingerprint(contract['eval_keys'])
            state_path = out/f'eval_state_update{updates}.json'
            saved_eval = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else None
            if saved_eval and (saved_eval['population'] != population_fp or saved_eval['update'] != updates):
                raise RuntimeError('evaluation state provenance mismatch')
            def save_eval(state):
                write_json(state_path, dict(population=population_fp, update=updates, state=state))
            sessions = {}
            try:
                for model in models.values():
                    model.eval().requires_grad_(False)
                    sessions[id(model.columns)] = BothActionsExecution(model.columns, device, graphs=True,
                                                                         capture_full_chunks_only=True)
                provider.paired_eval_sessions = sessions
                with preserve_training_rng():
                    batches = prefetched_batches(provider, dev_source, [[r] for r in evaluation], descriptors,
                                                   workers=a.prefetch_workers)
                    result['evaluation'] = evaluate(models, provider, dev_source, batches,
                        remove_threshold=a.remove_threshold, stop_event=stop_event, progress=progress,
                        saved=saved_eval['state'] if saved_eval else None, save=save_eval)
                result['evaluation_execution'] = {str(i): s.stats() for i, s in enumerate(sessions.values())}
            finally:
                if 'batches' in locals(): batches.close()
                for session in sessions.values(): session.close()
                provider.paired_eval_sessions = {}
                for model in models.values(): model.requires_grad_(True)
            result['status'] = 'complete' if epoch == a.epochs else 'interim_read_only_evaluation'
        else: result['status'] = 'stopped_at_complete_paired_update' if len(modes) == 2 else 'stopped_at_complete_update'
    except InterruptedError as error:
        result.update(status='stopped_at_complete_eval_window', error=str(error))
    except Exception as error:
        result.update(status='failed_previous_checkpoint_preserved', error=repr(error))
        # Do not save mismatched arms or cursor after a failed half-pair.
        if healthy and not a.evaluate_only: save()
        raise
    finally:
        if 'batches' in locals(): batches.close()
        for name in ('ccr_history_cache', 'ccr_val_history_cache'):
            cache = getattr(provider, name)
            result[name] = cache.stats(); cache.close()
        provider.ccr_execution.close(); provider.strong_majority.close()
        result['seconds_this_invocation'] = time.perf_counter()-tick
        result['gradient_links'] = reports.get('gradient_links', {})
        write_json(out/'results.json', result)
        (out/'summary.txt').write_text(summarize(result), encoding='utf-8')
    print(summarize(result), flush=True)
    return 0


if __name__ == '__main__':
    stopped = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: stopped.set())
    sys.exit(main(stopped))
