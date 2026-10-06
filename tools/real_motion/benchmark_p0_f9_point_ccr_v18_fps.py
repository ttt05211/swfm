#!/usr/bin/env python3
"""Same-window, complete-six-frame FPS; no training, GT scoring or promotion."""
import sys
from pathlib import Path
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import json
import os
import signal
import threading
import time

import torch

from real_motion.column_runtime_pipeline import CachedColumnSource
from real_motion.native_column_cpu import prepare_native, get_prepared_native
from real_motion.nuscenes_adapter import NuScenesWindowSource
from real_motion.runtime_config import add_config_args, load_runtime_config, make_prepare_config
from real_motion.v21_source_induction import stable_json_fingerprint
from tools.real_motion.eval_p0_f9_v18_se2 import load_cache
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import load_manifest, sha256
from tools.real_motion.joint_training_recovery import snapshot_checkpoint
from tools.real_motion.run_p0_f9_shared_evidence_pilot import PilotProvider, require_cuda
from tools.real_motion.static_evidence_selector_common import CLEAN_SHA256, write_json, finite_json
from tools.real_motion.train_p0_f9_causal_columns import record_keys
from tools.real_motion.train_p0_f9_joint_causal_columns import load_joint
from tools.real_motion.point_ccr_v18_fps_common import (
    PROTOCOL, ARMS, BOUNDARIES, select_population, load_point_head,
    rebuild_prior, assert_prior_exact, forecast, aggregate, resolve_arm_models,
)
from tools.real_motion import benchmark_p0_f9_v18_runtime as runtime


def brief(result):
    lines = ['===== SAME-WINDOW V18 + POINT CCR / SIX-FRAME FPS ONLY =====',
             'status='+result['status'], 'protocol='+PROTOCOL,
             'Clean-E14=legacy SIX histories; epoch19 V18/point CCR=FOUR histories.',
             'FPS=6/mean SIX-frame latency, not mean of per-window FPS. No GT/metrics/training.',
             'Native arm: exact Strong integer vote + redundant motion projection/LayerNorm reuse; NO motion CUDA Graph.',
             'Same weights, full support, fixed CCR ADD=0.5 / REMOVE=0.95; fresh point features in EVERY arm.']
    for boundary, rows in result.get('aggregate', {}).items():
        lines.append('\n===== '+boundary+' =====')
        for arm in ARMS:
            r = rows[arm]
            lines.append(f'{arm}: six_ms={r["mean_six_ms"]:.3f} FPS={r["FPS"]:.3f} '
                         f'p90_ms={r["p90_ms"]:.3f} windows={r["windows"]} samples={r["samples"]}')
        for family in ('clean_e14_6h', 'epoch19_v18_4h', 'point_ccr_4h'):
            speedup = rows[family+'_original']['mean_six_ms']/rows[family+'_native']['mean_six_ms']
            lines.append(f'{family} paired_speedup={speedup:.3f}')
        stages = rows['point_ccr_4h_native']['stage_mean_ms']
        lines.append('point_ccr_native host_stage_ms_NOT_GPU_utilization='+json.dumps(stages, sort_keys=True))
    if 'population' in result:
        lines.append('population='+json.dumps(result['population'], ensure_ascii=False))
    if 'exactness' in result:
        lines.append('exactness='+json.dumps(result['exactness']))
    if 'arm_models' in result:
        lines.append('actual_arm_models='+json.dumps(result['arm_models'], sort_keys=True))
    prepare = result.get('preparation_seconds_excluded', [])
    if prepare:
        lines.append(f'raw_fixed_preparation_seconds/window={sum(prepare)/len(prepare):.6f} (separate, excluded from FPS)')
    lines += ['fresh_prior: resident source tensors + registered history -> fresh ALL-six Strong/KTA + live model + SIX dense outputs.',
              'cached_prior: SAME boundary except immutable Strong/KTA is precomputed; CCR candidates/features/readouts remain fresh.',
              'Both EXCLUDE disk I/O, initial source tensor extraction/history registration, compilation/warmup, GT, metrics and correctness hashes.',
              'Preparation time reported separately. This is NOT raw-sensor end-to-end FPS.',
              'Cross-history Clean-E14 vs epoch19 is descriptive, NOT a matched scientific accuracy comparison.',
              'Point CCR previously failed the quality gate: fast FPS does NOT approve its accuracy or deployment.',
              'No source checkpoint writes, threshold search, training resume, full4369 evaluation or promotion.']
    if 'error' in result: lines.append('error='+result['error'])
    return '\n'.join(lines)+'\n'


def main(stop_event=None, argv=None):
    p = argparse.ArgumentParser(description=__doc__); add_config_args(p)
    for key in ('checkpoint', 'ccr-checkpoint', 'base-checkpoint', 'dev-cache',
                'population-manifest', 'dataroot', 'dev-info', 'out-dir'):
        p.add_argument('--'+key, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--windows', type=int, default=20)
    p.add_argument('--stress-windows', type=int, default=2)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--cpu-workers', type=int, default=8)
    a = p.parse_args(argv); out = Path(a.out_dir)
    if out.exists(): p.error('fresh output required; no experiment overwrite')
    for key in ('config', 'checkpoint', 'ccr_checkpoint', 'base_checkpoint', 'dev_cache',
                'population_manifest', 'dev_info'):
        if not Path(getattr(a, key) or '').is_file(): p.error('missing '+key)
    if (not Path(a.dataroot).is_dir() or not 1 <= a.windows <= 64 or a.repeats < 2
            or not 0 <= a.stress_windows < a.windows or not 1 <= a.cpu_workers <= 16):
        p.error('invalid finite same-window benchmark budget')
    device = require_cuda(a.device); torch.set_num_threads(1)
    out.mkdir(parents=True); started = time.perf_counter()
    result = dict(status='running', protocol=PROTOCOL, actual_cuda=True, trials=[],
                  GPU=torch.cuda.get_device_name(device), torch_version=str(torch.__version__),
                  cpu_workers=a.cpu_workers, repeats=a.repeats, no_training=True)
    def persist():
        write_json(out/'speed.json', result)
        (out/'summary.txt').write_text(brief(result), encoding='utf-8')
    persist()
    original_backend = os.environ.get('SWFM_COLUMN_CPU_BACKEND')
    try:
        # Only the native arm's Strong voter changes. All CCR domain/feature
        # code follows the same NumPy path in BOTH arms, isolating V18 changes.
        os.environ['SWFM_COLUMN_CPU_BACKEND'] = 'numpy'
        tick = time.perf_counter(); result['native_preflight'] = prepare_native(out/'native_build')
        result['compile_seconds_excluded'] = time.perf_counter()-tick
        sources = dict(epoch19=a.checkpoint, point_ccr=a.ccr_checkpoint, clean_e14=a.base_checkpoint)
        snapshots = {k:out/(k+'_snapshot.pt') for k in sources}
        digests = {k:snapshot_checkpoint(path, snapshots[k]) for k,path in sources.items()}
        result['checkpoint_sha256'] = digests
        cfg = load_runtime_config(a.config, a.override); config_fp = stable_json_fingerprint(cfg)
        ck, teacher = load_joint(snapshots['epoch19'], device, reference_sha=CLEAN_SHA256,
                                 config_sha=config_fp, allow_diagnostic=True)
        if (teacher.transport.config.history_frames != 4 or ck.get('cursor_epoch') != 19
                or ck['model_configs'].get('adaptive_context') is not None):
            raise RuntimeError('selected FOUR-history epoch19 Local transport required')
        teacher.eval().requires_grad_(False)
        for path, expected in ((a.dev_cache, ck['cache_fingerprints']['dev']),
                (a.dev_info, ck['info_fingerprints']['dev']), (snapshots['clean_e14'], CLEAN_SHA256)):
            if sha256(path) != expected: raise RuntimeError('checkpoint/data provenance mismatch: '+str(path))
        saved = torch.load(snapshots['point_ccr'], map_location='cpu', weights_only=False)
        head = load_point_head(saved, teacher_sha256=digests['epoch19'], config_fingerprint=config_fp,
                               source_dim=teacher.columns.source_dim, device=device)
        manifest, keys64, _ = load_manifest(a.population_manifest)
        parent = tuple(map(tuple, manifest['parent_keys']))
        if (len(keys64) != 64 or len(parent) != 512
                or manifest['manifest_fingerprint'] != ck['dev_manifest_fingerprint']
                or tuple(map(tuple, ck['dev_keys'])) != parent
                or saved['contract']['dev_manifest_fingerprint'] != manifest['manifest_fingerprint']
                or tuple(map(tuple, saved['contract']['final_dev_keys'])) != parent):
            raise RuntimeError('frozen dev population identity/order mismatch')
        _, records = load_cache(a.dev_cache); record_keys(records)
        chosen, population = select_population(records, keys64, windows=a.windows, stress_windows=a.stress_windows)
        del records, saved
        result['population'] = dict(windows=len(chosen), scenes=len({r['scene_name'] for r in chosen}),
            manifest_fingerprint=manifest['manifest_fingerprint'],
            key_fingerprint=stable_json_fingerprint([m['key'] for m in population]),
            selection='scene-balanced round-robin + source-count-only stress; no GT/errors/latency selection')
        write_json(out/'fps_manifest.json', dict(protocol=PROTOCOL, **result['population'], keys=population))
        provider = PilotProvider(snapshots['clean_e14'], CLEAN_SHA256, make_prepare_config(cfg),
                                 device, a.cpu_workers, teacher, None)
        models = resolve_arm_models(provider, teacher)
        provider.reference.eval().requires_grad_(False)
        result['arm_models'] = {arm:dict(history_frames=model.config.history_frames,
            checkpoint_sha256=digests['clean_e14' if arm.startswith('clean') else 'epoch19'])
            for arm,model in models.items()}
        print('FPS_MODEL_IDENTITY '+json.dumps(result['arm_models'], sort_keys=True), flush=True)
        persist()
        source = CachedColumnSource(NuScenesWindowSource(a.dataroot, info_pkl=a.dev_info, verbose=False), 128)
        result['preparation_seconds_excluded'] = []
        exact_windows = 0
        with (out/'progress.jsonl').open('x', encoding='utf-8') as log:
            for index, (record, meta) in enumerate(zip(chosen, population)):
                if stop_event is not None and stop_event.is_set(): break
                tick = time.perf_counter()
                raw = provider.load_raw_columns(source, record, include_gt=False)
                case = dict(record=record, raw=raw, gpu=runtime._gpu_inputs(record, device))
                torch.cuda.synchronize(device)
                result['preparation_seconds_excluded'].append(time.perf_counter()-tick)
                dense_prior, _ = rebuild_prior(case, provider, native=False, backgrounds=True)
                native_prior, _ = rebuild_prior(case, provider, native=True, backgrounds=True)
                assert_prior_exact(native_prior, dense_prior)
                assert_prior_exact(dense_prior, raw['_column_causal_preparation']['prepared_state'])
                # Provider's first renderer check must be OUTSIDE every clock.
                output = runtime._model_forward(teacher.transport, case['gpu'], device, return_latents=True)
                provider.prepare_columns(None, record, include_gt=False, raw_window=raw, outputs=output)
                del output, dense_prior, native_prior
                refs = {}
                for boundary in BOUNDARIES:
                    for arm in ARMS:
                        model = models[arm]
                        point = head if arm.startswith('point') else None
                        row = forecast(case, provider, model, point, native=arm.endswith('native'), boundary=boundary)
                        family = arm.rsplit('_', 1)[0]
                        if family in refs and refs[family] != row['signature']:
                            raise RuntimeError('ALL-six motion/probability/dense byte mismatch: '+arm+'/'+boundary)
                        refs[family] = row['signature']
                exact_windows += 1
                for repeat in range(a.repeats):
                    order = [(b, arm) for b in BOUNDARIES for arm in ARMS]
                    if (repeat+index) % 2: order.reverse()
                    for boundary, arm in order:
                        model = models[arm]
                        row = forecast(case, provider, model, head if arm.startswith('point') else None,
                                       native=arm.endswith('native'), boundary=boundary)
                        if refs[arm.rsplit('_', 1)[0]] != row['signature']:
                            raise RuntimeError('repeated forecast byte mismatch: '+arm+'/'+boundary)
                        row.update(boundary=boundary, arm=arm, repeat=repeat+1,
                                   key='/'.join(meta['key']), stratum=meta['stratum'])
                        result['trials'].append(row)
                        log.write(json.dumps(finite_json(row), allow_nan=False)+'\n'); log.flush()
                print(f'PAIRED_FPS windows={index+1}/{len(chosen)} ALL-six bytes PASS; '
                      f'sources={meta["sources"]} stratum={meta["stratum"]}', flush=True)
                persist()
                del case, raw, refs
        for name, path in sources.items():
            if sha256(path) != digests[name]:
                raise RuntimeError('source checkpoint replaced during read-only benchmark: '+name)
        complete = exact_windows == len(chosen)
        result.update(status='complete' if complete else 'stopped',
            aggregate=aggregate(result['trials']) if result['trials'] else {},
            exactness=dict(windows=exact_windows, all_six_strong_sources_and_clear_exact=True,
                all_motion_latents_probability_bytes_dense_frames_exact=True, source_checkpoints_unchanged=True),
            native_audit=get_prepared_native().info(), elapsed_seconds=time.perf_counter()-started,
            route='speed_only_no_accuracy_or_deployment_promotion')
        persist(); print(brief(result), flush=True)
        return 0 if complete else 130
    except Exception as exc:
        result.update(status='failed', error=str(exc), elapsed_seconds=time.perf_counter()-started)
        persist(); raise
    finally:
        if original_backend is None: os.environ.pop('SWFM_COLUMN_CPU_BACKEND', None)
        else: os.environ['SWFM_COLUMN_CPU_BACKEND'] = original_backend


if __name__ == '__main__':
    stopped = threading.Event()
    def request_stop(signum, frame):
        stopped.set(); print('Stop requested: finish this paired window; no training checkpoint touched.', flush=True)
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, request_stop)
    sys.exit(main(stopped))
