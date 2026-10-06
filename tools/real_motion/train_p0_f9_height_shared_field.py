#!/usr/bin/env python3
"""Finite server screen of the fast native shared field; NOT old shared-evidence.

Random new head, frozen selected epoch19 motion, GT-only, 20% TRAIN x 3 passes.
No KD/AE, old optimizer resume, threshold search, full4369 or auto-promotion.
"""
import sys
from pathlib import Path
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import signal
import threading
import time

import numpy as np
import torch

from real_motion.height_causal_field import HeightCausalField
from real_motion.runtime_config import add_config_args, load_runtime_config, make_prepare_config
from real_motion.column_runtime_pipeline import CachedColumnSource
from real_motion.nuscenes_adapter import NuScenesWindowSource
from real_motion.causal_geometry_cache import CausalGeometryCache
from real_motion.v21_source_induction import stable_json_fingerprint
from tools.real_motion.run_p0_f9_shared_evidence_pilot import PilotProvider, require_cuda
from tools.real_motion.local_warm_cache_common import geometry_namespace
from tools.real_motion.train_p0_f9_joint_causal_columns import load_joint
from tools.real_motion.eval_p0_f9_v18_se2 import load_cache
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import load_manifest, align_records, sha256
from tools.real_motion.train_p0_f9_causal_columns import record_keys
from tools.real_motion.v18_source_interaction_common import select_population
from tools.real_motion.joint_training_recovery import snapshot_checkpoint, save_resume_checkpoint, preserve_training_rng
from tools.real_motion.static_evidence_selector_common import CLEAN_SHA256, write_json, finite_json
from tools.real_motion.joint_column_full_common import prefetch_column_batches
from tools.real_motion.height_field_screen_common import (
    PROTOCOL, epoch_groups, learning_rate, train_step, calibrate_train, evaluate, six_frame_speed,
)
from tools.real_motion.height_field_screen_recovery import payload, restore


def brief(result):
    lines = ['===== HEIGHT-AWARE SHARED FIELD / GT-ONLY SCREEN =====',
             'status: '+result['status'], 'protocol: '+PROTOCOL,
             '4 histories -> 6 futures; RANDOM new shared_field head; epoch19 motion FROZEN.',
             'No distillation/AE. Fixed thresholds=(0.5,0.5,0.95). Final pass only, no best-by-dev.']
    if 'training_population' in result:
        lines.append('TRAIN '+json.dumps(result['training_population']))
    reports = result.get('reports', {})
    old = reports.get('initial_dev64', {}).get('variants', {}).get('old_joint')
    for row in reports.get('epochs', []):
        joint = row['evaluation']['variants']['joint']['metrics']
        lines.append(f"epoch={row['epoch']} update={row['update']} dev64_mIoU={joint['mIoU']:.6f} "
                     f"MovingMicro={joint['MovingMicro']:.6f}" +
                     (f" vs_old_mIoU={joint['mIoU']-old['metrics']['mIoU']:+.6f} "
                      f"vs_old_Moving={joint['MovingMicro']-old['metrics']['MovingMicro']:+.6f}" if old else ''))
    final = reports.get('final_dev512')
    if final:
        lines.append('===== FINAL DEV512 (selection/development, not independent test) =====')
        for name, row in final['variants'].items():
            m = row['metrics']; d = row['delta_vs_v18_pp']; q = row['quality']
            lines.append(f"{name}: IoU={m['IoU']:.6f} mIoU={m['mIoU']:.6f} MovingMicro={m['MovingMicro']:.6f} "
                         f"vs_transport_mIoU={d['mIoU']:+.6f} add={q.get('added', 0)} "
                         f"remove={q.get('removed', 0)} precision={q['addition_semantic_precision']}")
    if 'speed' in reports:
        s = reports['speed']
        lines.append('SIX_FRAME '+json.dumps(s['six_frame_mean_seconds']))
        lines.append('FPS=6/mean_six_frame_latency '+json.dumps(s['six_frame_amortized_FPS']))
        lines += ['FPS boundary: '+s['boundary'], 'FPS excludes: '+s['excludes']]
    if 'training' in result:
        lines.append('ACTUAL_SCREEN_TRAINING '+json.dumps(result['training']))
    if 'gate' in result:
        lines.append('DIAGNOSTIC_GATE '+json.dumps(result['gate']))
    lines += ['checkpoint: '+result.get('checkpoint', 'not yet saved'),
              'route: '+result.get('route', 'in_progress'),
              'Frozen-motion screen throughput is NOT full-joint training FPS/speedup. No old checkpoint edits.']
    if 'error' in result:
        lines.append('error: '+result['error'])
    return '\n'.join(lines)+'\n'


def parser():
    p = argparse.ArgumentParser(description=__doc__); add_config_args(p)
    for key in ('checkpoint', 'train-cache', 'dev-cache', 'population-manifest', 'base-checkpoint',
                'dataroot', 'train-info', 'dev-info', 'out-dir'):
        p.add_argument('--'+key, required=True)
    p.add_argument('--resume'); p.add_argument('--device', default='cuda')
    p.add_argument('--cpu-workers', type=int, default=8)
    p.add_argument('--causal-geometry-cache')
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--train-fraction', type=float, default=.2)
    p.add_argument('--seed', type=int, default=20261006)
    p.add_argument('--lr', type=float, default=2e-3)
    p.add_argument('--prior-windows', type=int, default=256)
    p.add_argument('--eval-windows', type=int, default=64)
    p.add_argument('--fps-windows', type=int, default=6)
    p.add_argument('--speed-repeats', type=int, default=2)
    p.add_argument('--max-updates', type=int, default=0,
                   help='optional safe early-stop budget, NOT a different cosine schedule')
    p.add_argument('--skip-dev512', action='store_true')
    return p


def main(stop_event=None, argv=None):
    p = parser(); a = p.parse_args(argv); out = Path(a.out_dir)
    if out.exists():
        p.error('new output required even on resume; refusing overwrite')
    for key in ('config', 'checkpoint', 'train_cache', 'dev_cache', 'population_manifest',
                'base_checkpoint', 'train_info', 'dev_info'):
        if not Path(getattr(a, key) or '').is_file():
            p.error('missing '+key)
    if a.resume and not Path(a.resume).is_file():
        p.error('missing new shared-field screen checkpoint')
    if (not Path(a.dataroot).is_dir() or min(a.epochs, a.cpu_workers, a.prior_windows, a.fps_windows, a.speed_repeats) < 1
            or not 1 <= a.eval_windows <= 64 or a.fps_windows > a.eval_windows or not 0 < a.train_fraction < 1
            or not np.isfinite(a.lr) or a.lr <= 0 or a.max_updates < 0):
        p.error('invalid finite screen budgets')
    device = require_cuda(a.device); torch.set_num_threads(1); torch.manual_seed(a.seed)
    from real_motion.native_column_cpu import backend_name, prepare_native
    if backend_name() == 'native':
        prepare_native()
    out.mkdir(parents=True); begun = time.perf_counter(); cache = None
    result = dict(status='running', reports={}); reports = result['reports']
    epoch = batch = updates = executed = 0
    head = optimizer = rng = contract = None
    def persist():
        write_json(out/'screen.json', result)
        (out/'summary.txt').write_text(brief(result), encoding='utf-8')
    def save():
        save_resume_checkpoint(out/'last.pt', payload(head, optimizer, rng, contract,
            epoch=epoch, batch=batch, updates=updates, executed=executed, reports=reports))
        result['checkpoint'] = str(out/'last.pt')
    persist()
    try:
        with (out/'progress.jsonl').open('x', encoding='utf-8') as log:
            def progress(row):
                log.write(json.dumps(finite_json(row), ensure_ascii=False, allow_nan=False)+'\n'); log.flush()
            snapshot = out/'epoch19_snapshot.pt'; digest = snapshot_checkpoint(a.checkpoint, snapshot)
            cfg = load_runtime_config(a.config, a.override)
            ck, teacher = load_joint(snapshot, device, reference_sha=CLEAN_SHA256,
                config_sha=stable_json_fingerprint(cfg), allow_diagnostic=True)
            if (teacher.transport.config.history_frames != 4 or ck.get('cursor_epoch') != 19
                    or ck['model_configs'].get('adaptive_context') is not None):
                raise RuntimeError('fixed selected four-history epoch19 Local teacher required')
            for path, expected in ((a.train_cache, ck['cache_fingerprints']['train']),
                    (a.dev_cache, ck['cache_fingerprints']['dev']), (a.train_info, ck['info_fingerprints']['train']),
                    (a.dev_info, ck['info_fingerprints']['dev']), (a.base_checkpoint, CLEAN_SHA256)):
                if sha256(path) != expected:
                    raise RuntimeError('epoch19/data provenance mismatch: '+path)
            manifest, keys64, _ = load_manifest(a.population_manifest)
            parent_keys = tuple(map(tuple, manifest['parent_keys']))
            if (len(keys64) != 64 or len(parent_keys) != 512
                    or manifest['manifest_fingerprint'] != ck['dev_manifest_fingerprint']
                    or parent_keys != tuple(map(tuple, ck['dev_keys']))):
                raise RuntimeError('frozen dev identity/order changed')
            _, all_dev = load_cache(a.dev_cache); record_keys(all_dev)
            dev = align_records(all_dev, keys64[:a.eval_windows])
            dev512 = align_records(all_dev, parent_keys); del all_dev
            _, all_train = load_cache(a.train_cache); keys = record_keys(all_train)
            if len(keys) != 20430 or tuple(keys) != tuple(map(tuple, ck['train_keys'])):
                raise RuntimeError('full TRAIN20430 identity/order changed')
            chosen, _ = select_population(keys, {s for s, _ in parent_keys}, fraction=a.train_fraction, seed=a.seed)
            train = align_records(all_train, chosen); del all_train
            groups = [epoch_groups(train, a.seed, i) for i in range(a.epochs)]
            sizes = [[len(g) for g in rows] for rows in groups]
            counts = [len(g) for g in groups]; steps = sum(counts)
            if a.max_updates > steps or a.prior_windows > len(train):
                raise RuntimeError('requested early-stop/prior population exceeds fixed screen budget')
            torch.manual_seed(a.seed)  # teacher construction must not affect head initialization
            head = HeightCausalField('shared_field', z_bins=teacher.columns.config.z_bins,
                                     source_dim=teacher.columns.source_dim).to(device)
            optimizer = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=.01)
            rng = np.random.default_rng(a.seed+1)
            root = Path(__file__).resolve().parents[2]
            implementation = stable_json_fingerprint({name: sha256(root/name) for name in (
                'real_motion/height_causal_field.py', 'real_motion/causal_column_completion.py',
                'real_motion/causal_column_model.py', 'tools/real_motion/causal_column_common.py',
                'tools/real_motion/joint_column_common.py', 'tools/real_motion/height_field_screen_common.py',
                'tools/real_motion/height_field_screen_recovery.py', 'tools/real_motion/train_p0_f9_height_shared_field.py')})
            contract = dict(protocol=PROTOCOL, teacher_sha256=digest, config_fingerprint=stable_json_fingerprint(cfg),
                cache_fingerprints=ck['cache_fingerprints'], info_fingerprints=ck['info_fingerprints'],
                dev_manifest_fingerprint=ck['dev_manifest_fingerprint'], train_keys=chosen, dev_keys=keys64[:a.eval_windows],
                final_dev_keys=parent_keys, seed=a.seed, epochs=a.epochs, train_fraction=a.train_fraction,
                prior_keys=chosen[:a.prior_windows], epoch_batches=counts, epoch_batch_sizes=sizes, schedule_steps=steps,
                window_batch=4, source_budget=128, lr=a.lr, weight_decay=.01,
                objective='equal_window_original_two_task_column_loss_GT_only',
                schedule='whole_screen_cosine_0.1_floor_no_tail', thresholds=(.5, .5, .95),
                model=dict(mode='shared_field', z_bins=head.z_bins, source_dim=head.source_dim, width=head.width, semantic_dim=4),
                transport_frozen=True, final_dev512=not a.skip_dev512,
                fps_windows=a.fps_windows, speed_repeats=a.speed_repeats,
                implementation=implementation, torch_version=str(torch.__version__))
            write_json(out/'contract.json', contract)
            if a.resume:
                saved = torch.load(a.resume, map_location='cpu', weights_only=False)
                (epoch, batch, updates, executed), reports = restore(saved, head, optimizer, rng, contract)
                result['reports'] = reports
                print(f'FIELD_RESUME update={updates}/{steps} next_epoch={epoch+1} batch_cursor={batch}; optimizer/RNG/cosine preserved', flush=True)
            teacher.eval().requires_grad_(False)
            provider = PilotProvider(a.base_checkpoint, CLEAN_SHA256, make_prepare_config(cfg), device, a.cpu_workers, teacher, None)
            provider.raw_prefetch_workers = provider.raw_prefetch_depth = min(2, a.cpu_workers)
            if a.causal_geometry_cache:
                namespace = geometry_namespace(cfg, provider, ck['info_fingerprints'], ck['cache_fingerprints'], a.dataroot)
                cache = CausalGeometryCache(a.causal_geometry_cache, namespace, max_bytes=0, ram_bytes=256*2**20)
                provider.causal_geometry_cache = cache
            train_source = CachedColumnSource(NuScenesWindowSource(a.dataroot, info_pkl=a.train_info, verbose=False), 256)
            dev_source = CachedColumnSource(NuScenesWindowSource(a.dataroot, info_pkl=a.dev_info, verbose=False), 256)
            del ck
            result.update(actual_cuda=device.type == 'cuda', teacher_snapshot=str(snapshot), teacher_sha256=digest,
                training_population=dict(windows=len(train), epochs=a.epochs, total_window_exposures=len(train)*a.epochs,
                                         optimizer_updates=steps, prior_windows=a.prior_windows),
                cache_policy='read existing fixed geometry; zero NEW geometry disk writes; 256MiB geometry RAM')
            print(f'FIELD_SCREEN TRAIN={len(train)} x {a.epochs} passes; {steps} updates; dev={len(dev)}; no KD, old epoch19 READ ONLY', flush=True)
            if 'train_prior' not in reports:
                reports['train_prior'] = calibrate_train(provider, train_source, train[:a.prior_windows], teacher, head,
                                                          progress=progress, stop_event=stop_event)
                save(); persist()
            if 'initial_dev64' not in reports:
                with preserve_training_rng(rng):
                    reports['initial_dev64'] = evaluate(provider, dev_source, dev, teacher, head, include_old=True,
                                                         progress=progress, stop_event=stop_event)
                save(); persist()
            def monitor_pending():
                if epoch and not any(r['epoch'] == epoch for r in reports.get('epochs', [])):
                    with preserve_training_rng(rng):
                        report = evaluate(provider, dev_source, dev, teacher, head, progress=progress, stop_event=stop_event)
                    reports.setdefault('epochs', []).append(dict(epoch=epoch, update=updates, evaluation=report))
                    m = report['variants']['joint']['metrics']; old = reports['initial_dev64']['variants']['old_joint']['metrics']
                    print(f"FIELD_EPOCH {epoch}/{a.epochs} mIoU={m['mIoU']:.6f} vs_old={m['mIoU']-old['mIoU']:+.6f} "
                          f"MovingMicro={m['MovingMicro']:.6f} vs_old_Moving={m['MovingMicro']-old['MovingMicro']:+.6f}", flush=True)
                    save(); persist()
            with ThreadPoolExecutor(max_workers=min(3, a.cpu_workers)) as pool:
                while epoch < a.epochs:
                    monitor_pending()
                    if stop_event is not None and stop_event.is_set():
                        break
                    ordered = [r for g in groups[epoch][batch:] for r in g]
                    previous_end = time.perf_counter()
                    for rows in prefetch_column_batches(provider, train_source, ordered, 4, 128, io_workers=min(2, a.cpu_workers)):
                        waited = time.perf_counter()-previous_end
                        if (stop_event is not None and stop_event.is_set()) or (a.max_updates and updates >= a.max_updates):
                            break
                        lr = learning_rate(updates, steps, a.lr)
                        optimizer.param_groups[0]['lr'] = lr
                        stat = train_step(provider, rows, teacher, head, optimizer, rng, candidate_pool=pool)
                        updates += 1; batch += 1; executed += len(rows)
                        reports['train_seconds'] = reports.get('train_seconds', 0.)+stat['seconds']
                        reports['input_wait_seconds'] = reports.get('input_wait_seconds', 0.)+waited
                        progress(dict(event='height_field_train', epoch=epoch+1, epoch_batch=batch, epoch_batches=counts[epoch],
                                      update=updates, target=steps, lr=lr, input_wait_seconds=waited, **stat))
                        if updates == 1 or updates % 32 == 0:
                            print(f"FIELD_TRAIN epoch={epoch+1}/{a.epochs} batch={batch}/{counts[epoch]} update={updates}/{steps} "
                                  f"loss={stat['loss']:.6f} seconds/window={stat['seconds']/len(rows):.4f} "
                                  f"dynamic_columns={stat.get('dynamic_refine_columns', 0)} allocated_after={stat['allocated_after_mib']:.1f}MiB", flush=True)
                        if batch == counts[epoch]:
                            epoch += 1; batch = 0; save(); persist(); break
                        if updates % 32 == 0 or stop_event is not None and stop_event.is_set():
                            save(); persist()
                        previous_end = time.perf_counter()
                    if (stop_event is not None and stop_event.is_set()) or (a.max_updates and updates >= a.max_updates):
                        break
            save(); persist()
            result['training'] = dict(completed_epochs=epoch, next_batch_cursor=batch, updates=updates,
                executed_windows=executed, step_seconds_per_window=reports.get('train_seconds', 0.)/max(executed, 1),
                input_wait_seconds_per_window=reports.get('input_wait_seconds', 0.)/max(executed, 1),
                transport_frozen=True, training_speedup_NOT_claimed=True)
            if epoch < a.epochs or stop_event is not None and stop_event.is_set():
                result.update(status='stopped', route='resume_identical_shared_field_screen'); persist(); return 0
            monitor_pending()
            if not a.skip_dev512 and 'final_dev512' not in reports:
                with preserve_training_rng(rng):
                    reports['final_dev512'] = evaluate(provider, dev_source, dev512, teacher, head, include_old=True,
                                                       progress=progress, stop_event=stop_event)
                save(); persist()
            if 'speed' not in reports:
                with preserve_training_rng(rng):
                    reports['speed'] = six_frame_speed(provider, dev_source, dev[:a.fps_windows], teacher, head,
                                                        repeats=a.speed_repeats, stop_event=stop_event)
                save(); persist()
            final = reports.get('final_dev512', reports['epochs'][-1]['evaluation'])
            old = final['variants'].get('old_joint', reports['initial_dev64']['variants']['old_joint'])['metrics']
            new = final['variants']['joint']['metrics']
            gate = dict(mIoU_within_0_05pp=new['mIoU'] >= old['mIoU']-.05,
                        MovingMicro_within_0_10pp=new['MovingMicro'] >= old['MovingMicro']-.10,
                        all_horizons_Moving_within_0_10pp=all(new['per_horizon'][h]['MovingMicro'] >= old['per_horizon'][h]['MovingMicro']-.1
                                                           for h in ('1.0', '2.0', '3.0')),
                        paired_inference_speedup_ge_3=reports['speed']['speedup'] >= 3.)
            gate['pass'] = all(gate.values())
            if sha256(snapshot) != digest:
                raise RuntimeError('immutable reference snapshot changed')
            result.update(status='complete', gate=gate, route='candidate_for_further_training_NOT_promoted' if gate['pass']
                          else 'screen_not_passed_no_automatic_retry')
            save(); persist(); print(brief(result), flush=True); return 0
    except InterruptedError as error:
        result.update(status='stopped', error=str(error), route='resume_last_completed_shared_field_checkpoint')
        persist(); return 0
    except BaseException as error:
        # Do not snapshot half-updated tensors/counters after a failed step.
        result.update(status='failed', error=type(error).__name__+': '+str(error),
                      route='recover_last_completed_periodic_checkpoint_only')
        persist(); raise
    finally:
        result['elapsed_seconds_this_invocation'] = time.perf_counter()-begun
        if cache is not None:
            result['geometry_cache'] = cache.stats(); cache.close()
        persist()


if __name__ == '__main__':
    stopped = threading.Event()
    def request_stop(signum, frame):
        stopped.set(); print('Stop requested: finish current update, atomically save head/optimizer/RNG/cursor; no kill -9.', flush=True)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, request_stop)
    sys.exit(main(stopped))
