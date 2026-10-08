#!/usr/bin/env python3
"""One bundle: fixed epoch3 surface CCR paired FPS + full4369 frozen validation.

No training, threshold search, new geometry cache, checkpoint overwrite or
independent-test claim. Ctrl+C saves exact integer counts at a window boundary.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import signal
import sys
import threading
import time

if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from real_motion.canonical_causal_repair import compose_canonical, repair_targets
from real_motion.canonical_repair_execution import CanonicalCpuExecution
from real_motion.causal_geometry_cache import CausalGeometryCache
from real_motion.ccr_frozen_b import frozen_b_probabilities
from real_motion.ccr_val_history_cache import namespace, validate_manifest
from real_motion.causal_column_completion import actions_from_probabilities, compose_dense
from real_motion.column_runtime_pipeline import CachedColumnSource, prefetch_raw_columns
from real_motion.nuscenes_adapter import NuScenesWindowSource, gt_moving_support_sequence
from real_motion.runtime_config import add_config_args, load_runtime_config, make_prepare_config
from real_motion.surface_canonical_repair import SurfaceCanonicalRepairHead, augment_evidence, augment_projection
from real_motion.surface_ccr_execution import SurfaceExecution
from real_motion.v21_source_induction import stable_json_fingerprint
from tools.real_motion import causal_column_common as columns, ccr_screen_common as ccr
from tools.real_motion import surface_ccr_screen_common as surface
from tools.real_motion.eval_p0_f9_v18_se2 import load_cache
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import align_records, load_manifest, sha256
from tools.real_motion.height_field_screen_recovery import validate_cursor
from tools.real_motion.point_ccr_v18_fps_common import load_point_head
from tools.real_motion.run_p0_f9_shared_evidence_pilot import PilotProvider, require_cuda
from tools.real_motion.shared_evidence_pilot_common import moving_support_masks
from tools.real_motion.static_evidence_selector_common import CLEAN_SHA256, write_json
from tools.real_motion.joint_training_recovery import snapshot_checkpoint
from tools.real_motion.train_p0_f9_causal_columns import record_keys
from tools.real_motion.train_p0_f9_joint_causal_columns import load_joint
from tools.real_motion.validate_p0_f9_ccr_frozen_b_expanded import _attach_old_local_fixed_geometry
from tools.real_motion.surface_ccr_validation_common import (
    PROTOCOL, Accumulator, load_progress, save_progress, paired_speed,
)

# Exact Linux/Windows-LF fingerprint of 36714f1, not an arbitrary resume bypass.
# The head and all cache/scientific contracts must still be identical.
LEGACY_EXECUTION_IMPLEMENTATION = '25a932b9dff199d3fcc47eb829d3741304ee5d121b4f996462b39fe176f5d17d'
UNCHANGED_SURFACE_HEAD_SHA256 = '92dde104ff0930c521461bb83d5e6f5aa2bf2410ba44492ffe95b2c3cb6f4933'
# Audited execution-only changes: bounded identical KD queries + reuse of
# float64 phases. Parameter names/shapes, arithmetic outputs, support and
# thresholds remain byte-tested. Never accept arbitrary new head hashes.
SURFACE_EXECUTION_HEAD_SHA256 = '9c858913ab9aba902bbb6ec183ab19dd1b34a60de5539aef4fa7438f3483aa0c'
PREVIOUS_EXECUTION_IMPLEMENTATION = 'c19c01f9560c77df050a9764231c84311e3a24dc5e96207b317a26fd64167adb'


def compatible_execution_implementations(root):
    digest=sha256(Path(root)/'real_motion/surface_canonical_repair.py')
    if digest==UNCHANGED_SURFACE_HEAD_SHA256:
        return (LEGACY_EXECUTION_IMPLEMENTATION,)
    if digest==SURFACE_EXECUTION_HEAD_SHA256:
        return (LEGACY_EXECUTION_IMPLEMENTATION,PREVIOUS_EXECUTION_IMPLEMENTATION)
    return ()


def load_surface(saved, baseline, *, teacher_sha, config_fp, manifest_fp, device):
    c = saved.get('contract', {})
    head = SurfaceCanonicalRepairHead(baseline.source_dim, baseline.width).to(device)
    if (saved.get('protocol') != surface.PROTOCOL or c.get('protocol') != surface.PROTOCOL
            or saved.get('transport_frozen') is not True or saved.get('deployable') is not False
            or c.get('teacher_sha256') != teacher_sha or c.get('config_fingerprint') != config_fp
            or c.get('dev_manifest_fingerprint') != manifest_fp or c.get('model') != surface.model_contract(head)
            or c.get('static_only_training') is not True
            or c.get('thresholds') != dict(CCR_ADD=.5, CCR_REMOVE=None, old_Local=(.5, .5, None))):
        raise RuntimeError('surface checkpoint scientific/model/decision contract mismatch')
    validate_cursor(c, *[saved[k] for k in ('epoch', 'batch', 'updates', 'executed')])
    if saved['epoch'] != 3 or saved['batch'] != 0 or c.get('epochs') != 3 or c.get('train_fraction') != 1.:
        raise RuntimeError('fixed third completed full-TRAIN pass required, not dev-best selection')
    head.load_state_dict(saved['head'], strict=True)
    if not all(bool(torch.isfinite(v).all()) for v in head.state_dict().values()):
        raise RuntimeError('nonfinite surface checkpoint')
    # The comparison must use the true B, not reconstruct B from possibly
    # mutated shared weights in a superficially compatible student checkpoint.
    for name, value in baseline.state_dict().items():
        if not torch.equal(value.cpu(), head.state_dict()[name].cpu()):
            raise RuntimeError('original dynamic/shared CCR changed: ' + name)
    return head.eval().requires_grad_(False)


def summary(result):
    lines = ['===== SURFACE CCR / FROZEN EXPANDED VALIDATION + EXECUTION =====',
             'status=' + result['status'], 'Fixed epoch3, ADD raw0.5 / REMOVE-off; no training/threshold search.']
    speed = result.get('speed')
    if speed:
        lines.append('FORMAL_FPS ' + str(speed['six_frame_amortized_FPS']))
        lines.append('MEAN_SIX_MS ' + str({k: 1000*v for k, v in speed['six_frame_mean_seconds'].items()}))
        lines.append('P90_SIX_MS ' + str(speed['p90_six_ms']))
        lines.append('selected_execution=' + speed['selected_execution'] + ' (same output bytes, latency only)')
        lines.append('HOST_STAGES_MS (nested phase is INCLUDED, do not sum twice) ' + str(speed['stages_mean_ms']))
        chosen = speed['selected_execution']
        top = sorted(((k, v) for k, v in speed['stages_mean_ms'][chosen].items() if 'INCLUDED' not in k),
                     key=lambda row: row[1], reverse=True)[:3]
        lines.append('TOP3_FORECAST_STAGES_MS ' + str(top))
        lines.append('PEAK_MEMORY_MIB ' + str(speed.get('memory_mib')))
        lines.append('FPS boundary: ' + speed['boundary'])
        lines.append('Separate history descriptor preparation ms/window=' + str(1000*speed['surface_descriptor_prepare_seconds_per_window']))
    for name, row in result.get('subsets', {}).items():
        lines.append(f'--- {name}: windows={row["windows"]} scenes={row["scenes"]} ---')
        if not row['available']: continue
        for variant, m in row['metrics'].items():
            lines.append(f'{variant}: IoU={m["IoU"]:.6f} mIoU={m["mIoU"]:.6f} '
                         f'MovingMacro={m["MovingMacro"]:.6f} MovingMicro={m["MovingMicro"]:.6f}')
        for compare in ('surface_vs_B', 'surface_vs_old', 'surface_vs_transport'):
            d = row[compare]
            lines.append(f'{compare}: dMiOU={d["mIoU"]:+.6f} dIoU={d["IoU"]:+.6f} '
                         f'dMovingMicro={d["MovingMicro"]:+.6f}')
        for h in ('1.0', '2.0', '3.0'):
            d = row['surface_vs_B']['per_horizon'][h]
            lines.append(f'{h}s vs_B dMiOU={d["mIoU"]:+.6f} Moving={d["MovingMicro"]:+.6f} '
                         f'road11={d["semantic_per_class"]["11"]:+.6f} sidewalk13={d["semantic_per_class"]["13"]:+.6f}')
            for cls in ('11', '13'):
                errors = row['road_sidewalk_TP_FP_FN'][h][cls]
                difference = {k: errors['surface_CCR'][k] - errors['frozen_B'][k] for k in ('TP', 'FP', 'FN')}
                lines.append(f'{h}s class{cls} TP/FP/FN vs_B: ' + str(difference))
    lines += ['completed_windows=' + str(result.get('completed_windows', 0)),
              'quality_eval_execution=' + str(result.get('quality_eval_execution', 'not started')),
              'performance=' + str(result.get('performance', {})),
              'These subsets were already used in research: enlarged validation, NOT independent tests.',
              'Old Local comparator: same live transport, (.5,.5,REMOVE-off).',
              'Original weights/optimizer/RNG/caches unchanged. No automatic clean-joint training or promotion.']
    if result.get('error'): lines.append('error=' + result['error'])
    if result.get('val_cache'):
        cache = result['val_cache']
        lines.append('VAL_CACHE ' + str({k: cache[k] for k in ('hits', 'misses', 'writes') if k in cache}))
    if result.get('timing_note'): lines.append(result['timing_note'])
    return '\n'.join(lines) + '\n'


def parser():
    p = argparse.ArgumentParser(description=__doc__); add_config_args(p)
    for key in ('checkpoint', 'ccr-checkpoint', 'frozen-b-checkpoint', 'base-checkpoint', 'dev-cache',
                'population-manifest', 'dataroot', 'dev-info', 'out-dir', 'ccr-val-history-cache'):
        p.add_argument('--' + key, required=True)
    p.add_argument('--device', default='cuda'); p.add_argument('--cpu-workers', type=int, default=10)
    p.add_argument('--ccr-cpu-execution', choices=('numpy', 'native', 'native_parallel'), default='native_parallel')
    p.add_argument('--ccr-cpu-workers', type=int, default=4)
    p.add_argument('--ccr-val-history-cache-ram-mib', type=int, default=512)
    p.add_argument('--old-local-batch-size', type=int, default=256)
    p.add_argument('--eval-surface-execution', choices=('eager', 'full_chunk_graph'), default='eager',
                   help='quality eval only; warmed FPS never selects a cold variable-tail graph policy')
    p.add_argument('--resume-eval', help='previous evaluation_progress.pt; NEVER a training checkpoint')
    p.add_argument('--max-windows', type=int, default=0, help='planned safe stop; full population/selection unchanged')
    return p


@torch.no_grad()
def main(argv=None, stop_event=None):
    p = parser(); a = p.parse_args(argv); out = Path(a.out_dir)
    if out.exists(): p.error('fresh output required even on resume; no overwrite')
    for name in ('config', 'checkpoint', 'ccr_checkpoint', 'frozen_b_checkpoint', 'base_checkpoint',
                 'dev_cache', 'population_manifest', 'dev_info'):
        if not Path(getattr(a, name)).is_file(): p.error('missing ' + name)
    if (not Path(a.dataroot).is_dir() or not Path(a.ccr_val_history_cache).is_dir()
            or not 1 <= a.cpu_workers <= 16 or not 1 <= a.ccr_cpu_workers <= 8
            or not 0 <= a.ccr_val_history_cache_ram_mib <= 4096
            or a.old_local_batch_size != 256 or not 0 <= a.max_windows <= 4369):
        p.error('invalid paths/budgets; old comparator batch stays 256')
    device = require_cuda(a.device); torch.set_num_threads(1)
    out.mkdir(parents=True); begun = time.perf_counter()
    result = dict(status='running', protocol=PROTOCOL, training=False, threshold_search=False,
                  independent_test_claim=False)
    accumulator = Accumulator(); speed = None; performance = defaultdict(float)
    execution = val_cache = fast = None; contract = None
    def persist():
        result['completed_windows'] = accumulator.cursor
        result['performance'] = dict(performance)
        if fast is not None: result['speed_execution_full_eval'] = fast.stats()
        if val_cache is not None: result['val_cache'] = val_cache.stats()
        write_json(out/'expanded_validation.json', result)
        (out/'summary.txt').write_text(summary(result), encoding='utf-8')
    def save():
        if contract is not None:
            save_progress(out/'evaluation_progress.pt', contract, accumulator, speed, performance)
        persist()
    persist()
    try:
        # Snapshot inputs once. No dependence on an externally mutable last.pt
        # during a long read-only validation or its later continuation.
        snapshots = {}
        digests = {}
        for name in ('checkpoint', 'ccr_checkpoint', 'frozen_b_checkpoint'):
            target = out/(name + '_snapshot.pt')
            digests[name] = snapshot_checkpoint(getattr(a, name), target)
            snapshots[name] = target
        cfg = load_runtime_config(a.config, a.override); config_fp = stable_json_fingerprint(cfg)
        if sha256(a.base_checkpoint) != CLEAN_SHA256: raise RuntimeError('Clean E14 fingerprint mismatch')
        ck, teacher = load_joint(snapshots['checkpoint'], device, reference_sha=CLEAN_SHA256,
                                config_sha=config_fp, allow_diagnostic=True)
        if (teacher.transport.config.history_frames != 4 or ck.get('cursor_epoch') != 19
                or ck['model_configs'].get('adaptive_context') is not None):
            raise RuntimeError('strict four-history epoch19 teacher required')
        teacher.eval().requires_grad_(False)
        for path, expected in ((a.dev_cache, ck['cache_fingerprints']['dev']), (a.dev_info, ck['info_fingerprints']['dev'])):
            if sha256(path) != expected: raise RuntimeError('teacher/data provenance mismatch: ' + path)
        manifest, keys64, _ = load_manifest(a.population_manifest)
        keys512 = tuple(map(tuple, manifest['parent_keys']))
        if (len(keys64) != 64 or len(keys512) != 512 or len(set(keys512)) != 512
                or manifest['manifest_fingerprint'] != ck['dev_manifest_fingerprint']
                or keys512 != tuple(map(tuple, ck['dev_keys']))):
            raise RuntimeError('frozen DEV512 population changed')
        saved_b = torch.load(snapshots['frozen_b_checkpoint'], map_location='cpu', weights_only=False)
        reference = load_point_head(saved_b, teacher_sha256=digests['checkpoint'], config_fingerprint=config_fp,
            source_dim=teacher.columns.source_dim, device=device, allow_completed_epoch_boundary=True)
        reference.eval().requires_grad_(False); del saved_b
        saved = torch.load(snapshots['ccr_checkpoint'], map_location='cpu', weights_only=False)
        if saved['contract'].get('warm_start_head_sha256') != digests['frozen_b_checkpoint']:
            raise RuntimeError('not the Frozen B used to initialize this surface checkpoint')
        head = load_surface(saved, reference, teacher_sha=digests['checkpoint'], config_fp=config_fp,
                            manifest_fp=manifest['manifest_fingerprint'], device=device)
        result['checkpoint'] = dict(sha256=digests['ccr_checkpoint'], epoch=saved['epoch'], updates=saved['updates'],
                                   snapshot=str(snapshots['ccr_checkpoint']))
        del saved
        _, records = load_cache(a.dev_cache); keys = record_keys(records)
        if len(keys) != 4369 or len(set(keys)) != 4369 or not set(keys512).issubset(keys):
            raise RuntimeError('exact unique full4369 cache with DEV512 included required')
        dev512_set, dev512_scenes = set(keys512), {s for s, _ in keys512}
        monitor = align_records(records, keys64)
        provider = PilotProvider(a.base_checkpoint, CLEAN_SHA256, make_prepare_config(cfg), device, a.cpu_workers, teacher, None)
        execution = CanonicalCpuExecution(a.ccr_cpu_execution, a.ccr_cpu_workers); provider.ccr_execution = execution
        provider.raw_prefetch_workers = provider.raw_prefetch_depth = min(4, a.cpu_workers)
        provider.raw_io_workers = 1
        source = CachedColumnSource(NuScenesWindowSource(a.dataroot, info_pkl=a.dev_info, verbose=False), 1024)
        source.copy_on_insert = False
        root = Path(__file__).resolve().parents[2]
        val_cache = CausalGeometryCache(a.ccr_val_history_cache, namespace(provider, a, root), max_bytes=0,
            ram_bytes=a.ccr_val_history_cache_ram_mib*2**20, reserve_bytes=0, compression_level=6)
        validate_manifest(val_cache, a)
        provider.ccr_history_cache, provider.ccr_history_cache_source = val_cache, source
        provider.ccr_history_cache_mode = 'require'; provider.ccr_verify_cached_full_evidence_remaining = 1
        contract = dict(protocol=PROTOCOL, checkpoints=digests, config=config_fp, population=keys,
            manifest=manifest['manifest_fingerprint'], val_cache_namespace=val_cache.namespace,
            cpu_workers=a.cpu_workers, execution=a.ccr_cpu_execution, execution_workers=a.ccr_cpu_workers,
            old_local_batch=256, thresholds=(.5, None), torch_version=str(torch.__version__),
            implementation=stable_json_fingerprint({f: sha256(root/f) for f in (
                'real_motion/surface_canonical_repair.py', 'real_motion/surface_ccr_execution.py',
                'tools/real_motion/surface_ccr_validation_common.py', 'tools/real_motion/validate_p0_f9_surface_ccr_expanded.py')}))
        if a.resume_eval:
            compatible = compatible_execution_implementations(root)
            speed, performance = load_progress(a.resume_eval, contract, accumulator,
                                                compatible_implementations=compatible)
            result['resumed_from'] = str(Path(a.resume_eval).resolve())
            result['execution_fix_compatible_implementations'] = list(compatible)
            result['timing_note'] = ('performance includes the saved prefix; quality_eval_execution '
                                     'and per-window progress describe this invocation, not a paired speedup')
            if accumulator.cursor > len(records): raise RuntimeError('invalid evaluation cursor')
            print(f'SURFACE_EVAL_RESUME {accumulator.cursor}/{len(records)} exact integer counts restored', flush=True)
        if speed is None:
            speed = paired_speed(provider, source, monitor, teacher, head, reference, stop_event=stop_event)
            result['speed'] = speed; save()
        result['speed'] = speed
        # Hot single-window FPS does not justify recapturing variable shapes
        # in a one-pass evaluation. Reference eager is the conservative default;
        # the opt-in graph policy only captures fixed full chunks (at most two
        # keys: with/without sources), leaving every variable tail unchanged.
        result['quality_eval_execution'] = a.eval_surface_execution
        fast = SurfaceExecution(head, device, graphs=a.eval_surface_execution == 'full_chunk_graph',
                                capture_full_chunks_only=True)
        print('QUALITY_EVAL_EXECUTION '+a.eval_surface_execution+
              ' (separate from warmed FORMAL_FPS; no variable-tail capture)', flush=True)
        before = time.perf_counter()
        with (out/'progress.jsonl').open('x', encoding='utf-8') as log, ccr.old_execution(teacher, provider):
            for record, raw in prefetch_raw_columns(provider, source, records[accumulator.cursor:]):
                tick = time.perf_counter(); input_wait = tick - before
                performance['input_wait_seconds'] += input_wait
                if stop_event is not None and stop_event.is_set(): raise InterruptedError('safe window boundary stop')
                window_stages = {}
                def measure(name, fn):
                    tick = time.perf_counter(); value = fn()
                    elapsed = time.perf_counter() - tick
                    performance[name] += elapsed; window_stages[name] = elapsed
                    return value
                output = measure('motion_seconds', lambda: teacher.motion(record, device))
                cached = raw.get('_column_causal_preparation')
                preflight = cached is not None and not getattr(provider, 'columns_checked', False)
                if preflight: del raw['_column_causal_preparation']
                try:
                    prep = measure('render_seconds', lambda: provider.prepare_columns(source, record, include_gt=True, raw_window=raw, outputs=output))
                finally:
                    if preflight: raw['_column_causal_preparation'] = cached
                plain = measure('history_evidence_seconds', lambda: ccr._build_inputs_base(provider, prep))
                enriched = measure('surface_descriptor_seconds', lambda: augment_evidence(plain, surface._atlas(provider, prep, plain)))
                plan = measure('live_projection_seconds', lambda: ccr.map_inputs(provider, plain, prep))
                surface_plan = measure('live_surface_phase_seconds', lambda: augment_projection(
                    enriched, plan, prep.state['current_pose'], prep.state['world_to_future'], provider.pcfg.grid))
                prior_execution = fast.stats()
                new = measure('surface_probability_seconds', lambda: fast(head, enriched, surface_plan, output, device))
                after_execution = fast.stats()
                execution_seconds = {k: v-prior_execution['host_seconds'].get(k, 0.)
                    for k, v in after_execution['host_seconds'].items()}
                execution_counts = {k: v-prior_execution['counts'].get(k, 0)
                    for k, v in after_execution['counts'].items()}
                old_b = measure('B_probability_seconds', lambda: frozen_b_probabilities(reference, plain, plan, output, device))
                # All validation windows, not just the FPS population: confirm the
                # frozen dynamic scores before any GT-based metric computation.
                if not np.array_equal(new[plain.actor >= 0], old_b[plain.actor >= 0]):
                    raise RuntimeError('dynamic probability bytes changed in full validation')
                predictions = {'baseline': prep.baseline}
                for name, score in (('surface_CCR', new), ('frozen_B', old_b)):
                    predictions[name] = measure(name + '_compose_seconds', lambda score=score: compose_canonical(
                        prep.baseline, plain, plan, score[..., 0], score[..., 1], thresholds=(.5, None), role='all'))
                target, valid = measure('repair_targets_seconds', lambda: repair_targets(plain, plan, raw['future_gt_occ']))
                def old_predictions():
                    _attach_old_local_fixed_geometry(prep, provider, teacher)
                    rows = {}
                    for h in (1, 3, 5):
                        old_plan = columns.candidate_plan(prep, h, provider.pcfg.grid, teacher.columns.config)
                        score = columns.predict_probabilities(teacher.columns, prep, h, old_plan, provider.pcfg.grid, device, 256)
                        rows[h] = compose_dense(prep.baseline[h], old_plan, actions_from_probabilities(old_plan, score, (.5, .5, None)))
                    return rows
                predictions['old_local_remove_off'] = measure('old_local_seconds', old_predictions)
                support = measure('moving_support_seconds', lambda: gt_moving_support_sequence(source.nusc,
                    prep.window.t0_token, prep.window.future_tokens, tuple(.5*(h+1) for h in range(6)),
                    grid=provider.pcfg.grid, workers=provider.workers))
                moving = moving_support_masks(support, provider.pcfg.grid.shape_hwd)
                measure('integer_metrics_seconds', lambda: accumulator.update(record, predictions, raw['future_gt_occ'], moving,
                    plain, target, valid, {'surface_CCR': new, 'frozen_B': old_b}, dev512_keys=dev512_set, dev512_scenes=dev512_scenes))
                if device.type == 'cuda': torch.cuda.synchronize(device)
                before = time.perf_counter()
                log.write(json.dumps(dict(event='surface_expanded', window=accumulator.cursor, windows=len(records),
                    stages_seconds=window_stages, input_wait_seconds=input_wait,
                    ccr_history_cache_hit=bool(raw.get('_ccr_history_cache_hit')),
                    canonical_points=len(plain), surface_execution=a.eval_surface_execution,
                    surface_execution_host_seconds=execution_seconds,
                    surface_execution_counts=execution_counts)) + '\n'); log.flush()
                if accumulator.cursor % 32 == 0: save()
                if accumulator.cursor == 1 or accumulator.cursor % 32 == 0:
                    print(f'SURFACE_EXPANDED {accumulator.cursor}/{len(records)} '
                          f'cache_hit={bool(raw.get("_ccr_history_cache_hit"))} '
                          f'points={len(plain)} readout_ms={1000*window_stages["surface_probability_seconds"]:.2f} '
                          f'capture_ms={1000*execution_seconds.get("capture_and_parity",0.):.2f}', flush=True)
                if a.max_windows and accumulator.cursor >= a.max_windows: break
        result.update(accumulator.report())
        result['speed_execution_full_eval'] = fast.stats()
        result['status'] = 'complete' if accumulator.cursor == len(records) else 'stopped'
        result['route'] = 'frozen_expanded_results_NO_automatic_promotion' if result['status'] == 'complete' else 'resume_evaluation_progress'
        save(); print(summary(result), flush=True); return 0
    except InterruptedError as exc:
        result.update(status='stopped', error=str(exc), route='resume_evaluation_progress'); save(); return 0
    except BaseException as exc:
        # Never overwrite completed integer progress with a half-updated window.
        result.update(status='failed', error=type(exc).__name__ + ': ' + str(exc), route='resume_last_atomic_evaluation_progress')
        persist(); raise
    finally:
        result['elapsed_seconds_this_invocation'] = time.perf_counter() - begun
        if fast is not None:
            result['speed_execution_full_eval'] = fast.stats(); fast.close()
        if val_cache is not None:
            result['val_cache'] = val_cache.stats(); val_cache.close()
        if execution is not None: execution.close()
        persist()


if __name__ == '__main__':
    stopped = threading.Event()
    def request_stop(signum, frame):
        stopped.set(); print('Stop requested: save integer progress after current window; no checkpoint writes.', flush=True)
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, request_stop)
    raise SystemExit(main(stop_event=stopped))
