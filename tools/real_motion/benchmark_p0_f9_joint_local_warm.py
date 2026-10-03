#!/usr/bin/env python3
"""Warm-disk Local capacity/throughput probe. Never launches full training."""
from __future__ import annotations
import sys
from pathlib import Path
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import hashlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import os
import subprocess
import time
import traceback
from types import SimpleNamespace
import numpy as np
import torch
from real_motion.causal_geometry_cache import CausalGeometryCache
from real_motion.causal_geometry_cache import PROTOCOL as GEOMETRY_PROTOCOL
from real_motion.causal_column_completion import ColumnConfig
from real_motion.column_runtime_pipeline import CachedColumnSource
from real_motion.joint_causal_columns import JointCausalColumns
from real_motion.local_st_world_model_v17 import config_from_mapping_v17
from real_motion.local_training_profile import CpuProfiles, trial_summary, recommend_trials
from real_motion.nuscenes_adapter import NuScenesWindowSource
from real_motion.runtime_config import load_runtime_config, make_prepare_config
from real_motion.strong_w2det import StrongW2DetConfig
from real_motion.v21_source_induction import stable_json_fingerprint
from tools.real_motion.joint_column_common import count_proposals, weights_from_counts
from tools.real_motion.joint_column_full_common import FullJointColumnProvider, prefetch_column_batches, train_full_batch
from tools.real_motion.local_warm_cache_common import geometry_namespace, warm_causal_cache
from tools.real_motion.eval_p0_f9_v18_se2 import load_cache
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import sha256, validate_clean_e14_checkpoint
from tools.real_motion.static_evidence_selector_common import CLEAN_SHA256, write_json, atomic_checkpoint
from tools.real_motion.train_p0_f9_causal_columns import record_keys
from tools.real_motion.train_p0_f9_joint_causal_columns import load_joint
from tools.real_motion.v18_source_interaction_common import validate_records

PROTOCOL = 'local_warm_disk_capacity_timing_v1'


def measured_batches(joint, optimizer, provider, source, records, *, windows, sources,
                     workers, persistent=True, io_workers=2, cpu_profiles=None, max_batches=None, progress=None,
                     optimize_cpu=True, optimize_kernels=True):
    """Actual current-model forward/backward, strict warm hits, no weights saved."""
    rng = np.random.default_rng(20261003); rows = []
    pool = ThreadPoolExecutor(max_workers=workers) if persistent else None
    iterator = prefetch_column_batches(provider, source, records, windows, sources, io_workers=io_workers)
    tick = time.perf_counter()
    try:
        for update, batch in enumerate(iterator, 1):
            waited = time.perf_counter()-tick
            if any(not raw.get('_causal_geometry_cache_hit') for _, raw in batch):
                raise RuntimeError('benchmark requires persisted warm disk hits for EVERY window')
            started = time.perf_counter()
            kwargs = dict(profile=True, sampling_pool=pool, sampling_workers=workers, cpu_profiles=cpu_profiles,
                optimize_cpu=optimize_cpu, optimize_kernels=optimize_kernels)
            kwargs['patch_resolution'] = getattr(provider, 'patch_resolution_m', .8)
            fn = lambda: train_full_batch(joint, optimizer, provider, source, batch, rng, update, 100000, **kwargs)
            stats = fn()
            stats.update(input_wait_seconds=waited, wall_seconds=waited+time.perf_counter()-started,
                peak_reserved_mib=torch.cuda.max_memory_reserved(provider.device)/2**20 if provider.device.type == 'cuda' else 0.)
            rows.append(stats)
            if progress and (update == 1 or update % 8 == 0): progress(update, stats)
            tick = time.perf_counter()
            if max_batches is not None and update >= max_batches: break
    finally:
        iterator.close()
        if pool is not None: pool.shutdown(wait=True, cancel_futures=True)
    return rows


def worker(contract_path, phase, trial):
    c = json.loads(Path(contract_path).read_text(encoding='utf-8')); out = Path(c['out'])
    from real_motion.native_column_cpu import prepare_native, get_native
    recipe = json.loads(trial) if trial else {}
    os.environ['SWFM_COLUMN_CPU_BACKEND'] = recipe.get('backend', 'numpy')
    os.environ['SWFM_COLUMN_CPU_BUNDLE'] = '1' if recipe.get('bundle',True) else '0'
    if recipe.get('backend') == 'native':
        print('NATIVE_CPU_PREFLIGHT '+json.dumps(prepare_native()), flush=True)
    torch.set_num_threads(1); torch.manual_seed(c['seed']); device = torch.device('cuda')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): raise RuntimeError('CUDA/BF16 required')
    torch.cuda.manual_seed_all(c['seed'])
    cfg = load_runtime_config(c['config'], []); pcfg = make_prepare_config(cfg)
    if sha256(out/'records.pt') != c['records_sha']: raise RuntimeError('diagnostic record subset changed')
    data = torch.load(out/'records.pt', map_location='cpu', weights_only=False)
    base = torch.load(c['base_checkpoint'], map_location='cpu', weights_only=False)
    mc = replace(config_from_mapping_v17(base['model_config']), history_frames=c['history_frames']); del base
    joint = JointCausalColumns(mc, ColumnConfig(z_bins=int(pcfg.grid.shape_hwd[2]))).to(device)
    if c.get('snapshot'):
        if sha256(c['snapshot']) != c['snapshot_sha']: raise RuntimeError('diagnostic checkpoint snapshot changed')
        _, joint = load_joint(c['snapshot'], device, reference_sha=c['base_sha'], config_sha=stable_json_fingerprint(cfg), allow_diagnostic=True)
        if joint.transport.config.history_frames != c['history_frames'] or hasattr(joint.columns, 'context_config'):
            raise RuntimeError('checkpoint must match Local/history protocol; no silent six-to-four migration')
    provider = FullJointColumnProvider(c['base_checkpoint'], c['base_sha'], pcfg, device, 8, joint, None)
    provider.patch_resolution_m = c['patch_resolution_m']
    source = CachedColumnSource(NuScenesWindowSource(c['dataroot'], info_pkl=c['train_info'], verbose=False), 256)
    namespace = geometry_namespace(cfg, provider, c['info_fingerprints'], c['cache_fingerprints'], c['dataroot'])
    # Deliberately ZERO geometry RAM cache: small sample cannot make all geometry
    # appear RAM-resident when full20430 must read compressed disk entries.
    cache = CausalGeometryCache(c['geometry_cache'], namespace, max_bytes=int(c['cache_gib']*2**30), ram_bytes=0)
    provider.causal_geometry_cache = cache
    try:
        print(f"diagnostic_phase={phase} history={c['history_frames']} (existing checkpoints read-only)", flush=True)
        if phase == 'warm':
            warm = warm_causal_cache(provider, source, data['typical']+data['stress'], progress=lambda r:
                print(f"warm_disk={r['windows']}/{len(data['typical'])+len(data['stress'])} seconds={r['seconds']:.1f}", flush=True))
            # Diagnostic weights only: use a small TRAIN-only proposal audit, not
            # a new scientific prior definition. Formal full run STILL uses1024.
            counts = {'generation': np.zeros(2), 'refine': np.zeros(3)}
            if c.get('snapshot'):
                weights = {'generation_pos_weight': float(joint.columns.generation_pos_weight),
                    'refine_class_weights': joint.columns.refine_class_weights.cpu().tolist(), 'population': 'snapshot_TRAIN_weights'}
            else:
                joint.eval()
                for record in data['typical'][:16]:
                    with torch.no_grad(): prep = provider.prepare_columns(source, record, include_gt=True)
                    count_proposals(prep, pcfg.grid, joint.columns.config, counts)
                weights = weights_from_counts(counts); weights['population'] = 'diagnostic_TRAIN16_not_formal1024_prior'
            write_json(out/'warm_cache.json', warm); write_json(out/'diagnostic_weights.json', weights)
            return 0
        weights = json.loads((out/'diagnostic_weights.json').read_text(encoding='utf-8'))
        joint.columns.generation_pos_weight.fill_(weights['generation_pos_weight'])
        joint.columns.refine_class_weights.copy_(torch.tensor(weights['refine_class_weights'], device=device))
        optimizer = torch.optim.AdamW([
            {'params': joint.transport.parameters(), 'lr': 5e-4, 'initial_lr': 5e-4, 'weight_decay': 1e-4},
            {'params': joint.columns.parameters(), 'lr': 3e-4, 'initial_lr': 3e-4, 'weight_decay': .01}])
        available, total = torch.cuda.mem_get_info(device)
        t = json.loads(trial); name = t['name']; args = dict(windows=t['window_batch'], sources=t['source_budget'],
            workers=t['workers'], persistent=t['persistent'], io_workers=2, optimize_cpu=t.get('optimize_cpu', True),
            optimize_kernels=t.get('optimize_kernels', True))
        print(f"trial={name} source_budget={t['source_budget']} CUDA warm-up then dense stress (excluded from throughput)", flush=True)
        # Two warm-up minibatches; not included in steady throughput.
        warm_records = data['typical'][:min(len(data['typical']), 2*t['window_batch'])]
        measured_batches(joint, optimizer, provider, source, warm_records, max_batches=2, **args)
        # Source-dense stress batch kept separate from representative throughput.
        stress_records = [data['stress'][i % len(data['stress'])] for i in range(t['window_batch'])]
        stress = measured_batches(joint, optimizer, provider, source, stress_records, **args)
        capacity_peak = max(r['peak_reserved_mib'] for r in stress)
        if phase == 'profile':
            profiles = CpuProfiles()
            # Do not nest main and worker cProfiles: newer Python versions use
            # one monitoring slot. Independent main/worker passes are untimed.
            profiles.run('main_thread', measured_batches, joint, optimizer, provider, source,
                data['typical'][:t['window_batch']], **args)
            prof = measured_batches(joint, optimizer, provider, source, data['typical'][:t['window_batch']], cpu_profiles=profiles, **args)
            (out/'cpu_profile.txt').write_text(profiles.text(), encoding='utf-8')
            write_json(out/'profile_stages.json', prof)
        else:
            print(f'trial={name} measuring warm steady-state', flush=True)
            measured = measured_batches(joint, optimizer, provider, source, data['typical']*c['repeats'],
                progress=lambda u, r: print(f"trial={name} measured_batch={u} windows={r['windows']} sources={r['sources']} seconds={r['wall_seconds']:.3f}", flush=True), **args)
            m = trial_summary(measured); capacity_peak = max(capacity_peak, m['peak_reserved_mib'])
            result = dict(**t, status='ok', available_memory_mib=available/2**20, total_memory_mib=total/2**20,
                capacity_peak_reserved_mib=capacity_peak, stress_max_sources=max(r['sources'] for r in stress),
                stress_max_windows=max(r['windows'] for r in stress), measurement=m, geometry_cache=cache.stats(),
                rows=measured, diagnostic_weights=weights, history_frames=c['history_frames'],
                benchmark_contract_fingerprint=stable_json_fingerprint(c),
                execution=dict(torch_version=torch.__version__, cuda_version=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(device), temporal_attention_batch_limit=65535,
                    temporal_attention_implementation='independent_batch_chunk_v1',
                    model_code_sha256=sha256(Path(__file__).resolve().parents[2]/'real_motion/local_st_world_model.py'),
                    cpu_pipeline=('deferred_train_context_exact_numpy_v2' if args['optimize_kernels'] else 'window_shared_history_parallel_warm_v1')
                        if args['optimize_cpu'] else 'reference_serial_window_per_horizon_jobs',
                    pipeline_code_sha256=sha256(Path(__file__).resolve().with_name('joint_column_full_common.py'))))
            result['execution']['native_cpu'] = get_native().info() if get_native() is not None else None
            result['execution']['cpu_bundle'] = recipe.get('bundle',True) and get_native() is not None
            write_json(out/(name+'.json'), result)
            print(f"{name}: {m['windows_per_second']:.3f} windows/s, mean_batch={m['mean_windows_per_batch']:.2f}, peak_reserved={capacity_peak:.0f}MiB", flush=True)
        return 0
    except torch.cuda.OutOfMemoryError:
        if phase == 'warm': raise
        t = json.loads(trial); write_json(out/((t['name'] if phase == 'trial' else 'profile_oom')+'.json'), {**t, 'status': 'oom'})
        print('CUDA OOM isolated to diagnostic child; existing checkpoint untouched', flush=True); return 42
    except Exception as error:
        # Non-OOM errors are fatal, not a reason to silently try a smaller batch
        # in a possibly poisoned CUDA context. Keep completed trials untouched.
        t = json.loads(trial) if trial else {}
        write_json(out/('failure_'+phase+'_'+t.get('name', 'warm')+'.json'),
            dict(**t, status='error', error_type=type(error).__name__, error=str(error),
                traceback=traceback.format_exc(), benchmark_contract_fingerprint=stable_json_fingerprint(c)))
        raise
    finally: cache.close()


def validate_continuation(c, out):
    """Recover only this immutable diagnostic population, never a training run."""
    if c.get('protocol') != PROTOCOL or Path(c['out']).resolve() != out.resolve():
        raise RuntimeError('continuation contract/output mismatch')
    if sha256(out/'records.pt') != c['records_sha']:
        raise RuntimeError('diagnostic record subset changed')
    data = torch.load(out/'records.pt', map_location='cpu', weights_only=False)
    for kind in ('typical', 'stress'):
        if [list(k) for k in record_keys(data[kind])] != c[kind+'_keys']:
            raise RuntimeError('diagnostic identity/order changed: '+kind)
    if sha256(c['base_checkpoint']) != c['base_sha']:
        raise RuntimeError('base checkpoint changed')
    if sha256(c['train_info']) != c['info_fingerprints']['train']:
        raise RuntimeError('TRAIN info changed')
    if c.get('snapshot') and sha256(c['snapshot']) != c['snapshot_sha']:
        raise RuntimeError('diagnostic checkpoint snapshot changed')
    if c.get('original_checkpoint') and sha256(c['original_checkpoint']) != c['snapshot_sha']:
        raise RuntimeError('source checkpoint changed externally; stop active writer before continuation')
    cfg = load_runtime_config(c['config'], [])
    if c.get('runtime_config_fingerprint') and stable_json_fingerprint(cfg) != c['runtime_config_fingerprint']:
        raise RuntimeError('runtime config changed')
    warm = json.loads((out/'warm_cache.json').read_text(encoding='utf-8'))
    if not warm.get('complete') or warm['windows'] != len(data['typical'])+len(data['stress']):
        raise RuntimeError('completed diagnostic prefill required; do not silently rebuild')
    # Old v1 contracts lack a separate config hash. Their persisted namespace
    # still binds config, Strong, history, column config and source provenance.
    pcfg = make_prepare_config(cfg)
    provider = SimpleNamespace(strong=StrongW2DetConfig(free_label=pcfg.free_label),
        joint=SimpleNamespace(transport=SimpleNamespace(config=SimpleNamespace(history_frames=c['history_frames'])),
            columns=SimpleNamespace(config=ColumnConfig(z_bins=int(pcfg.grid.shape_hwd[2])))))
    namespace = geometry_namespace(cfg, provider, c['info_fingerprints'], c['cache_fingerprints'], c['dataroot'])
    directory = Path(c['geometry_cache'])/hashlib.sha256((GEOMETRY_PROTOCOL+namespace).encode()).hexdigest()
    saved_directory = warm['cache'].get('directory')
    saved_namespace = warm['cache'].get('namespace')
    if ((saved_directory is not None and directory.resolve() != Path(saved_directory).resolve())
            or (saved_namespace is not None and saved_namespace != directory.name)):
        raise RuntimeError('warm geometry namespace changed; cannot mix timing populations')
    if not directory.is_dir() or next(directory.glob('*.cgc'), None) is None:
        raise RuntimeError('warm geometry namespace/cache missing; no silent rebuild during continuation')
    if saved_directory is None:
        # Original stats() recorded only counters. Do NOT fabricate or mutate
        # historical metadata: derive the existing namespace from the contract
        # and require strict per-window hash/provenance-checked disk hits in the
        # child, before any measured update. Counts are not proof of integrity.
        print('legacy_warm_stats_without_directory: existing contract-derived namespace found; '
              'child still requires integrity-checked warm hits for every window', flush=True)
    return c


def load_completed_trial(out, t, c):
    row = json.loads((out/(t['name']+'.json')).read_text(encoding='utf-8'))
    if any(row.get(k) != v for k, v in t.items()) or row.get('status') not in ('ok', 'oom'):
        raise RuntimeError('incompatible saved trial: '+t['name'])
    fingerprint = row.get('benchmark_contract_fingerprint')
    if fingerprint and fingerprint != stable_json_fingerprint(c):
        raise RuntimeError('saved trial contract changed: '+t['name'])
    if row['status'] == 'ok':
        weights = json.loads((out/'diagnostic_weights.json').read_text(encoding='utf-8'))
        if row['history_frames'] != c['history_frames'] or row['diagnostic_weights'] != weights:
            raise RuntimeError('saved trial history/prior mismatch: '+t['name'])
        actual = trial_summary(row['rows'])
        if actual['windows'] != len(c['typical_keys'])*c['repeats'] or actual['cache_hits'] != actual['windows']:
            raise RuntimeError('saved trial population/warm hits mismatch: '+t['name'])
        if actual != row['measurement']:
            raise RuntimeError('saved timing summary differs from rows: '+t['name'])
    return row


def publish_summary(c, trials, *, started, profile_status, status, error=None, reused=(), print_report=False):
    out = Path(c['out'])
    decision = recommend_trials([t for t in trials if t.get('persistent')
        and (not c.get('cpu_comparison') or t.get('optimize_cpu') and t.get('optimize_kernels', True))])
    summary = dict(protocol=PROTOCOL, status=status, error=error, history_frames=c['history_frames'], future_frames=6,
        trials=trials, recommendation=decision, elapsed_seconds=time.perf_counter()-started,
        elapsed_scope='this_invocation_only_not_prior_completed_trials', reused_trials=list(reused),
        initialization='snapshot' if c.get('snapshot') else 'random_short_diagnostic_not_converged',
        cpu_profile_status=profile_status, speed_only=True, scientific_epochs=0,
        existing_checkpoint_unchanged=(True if not c.get('original_checkpoint') or not error
            else False if 'checkpoint changed' in str(error) else None), geometry_ram_cache_mib=0,
        caution='OS/frame-cache and workload variation remain; ETA excludes full cold prefill/eval/checkpoint; larger batch changes optimizer update count; no automatic full training')
    if c.get('cpu_comparison') and not c.get('native_comparison') and not c.get('native_bundle_comparison'):
        reference = next((t for t in trials if t['name'] == 'reference_b4' and t['status'] == 'ok'), None)
        fastest = next((t for t in trials if t['name'] == decision.get('recommended_trial')), None)
        fixed_batch = next((t for t in trials if t['name'] == 'optimized_b4' and t['status'] == 'ok'), None)
        previous = next((t for t in trials if t['name'] == 'previous_b4' and t['status'] == 'ok'), None)
        summary['cpu_comparison'] = dict(parent=c['cpu_comparison'],
            same_batch4_speedup=(fixed_batch['measurement']['windows_per_second']/reference['measurement']['windows_per_second']
                if reference and fixed_batch else None),
            recommended_speedup=(fastest['measurement']['windows_per_second']/reference['measurement']['windows_per_second']
                if reference and fastest else None),
            same_batch4_speedup_vs_previous=(fixed_batch['measurement']['windows_per_second']/previous['measurement']['windows_per_second']
                if previous and fixed_batch else None),
            note='same records/prior/initialization; timed separately without cProfile; no scientific updates saved')
    if c.get('native_comparison'):
        numpy_trial = next((t for t in trials if t['name'] == 'optimized_b4' and t['status'] == 'ok'), None)
        native_trial = next((t for t in trials if t['name'] == 'native_b4' and t['status'] == 'ok'), None)
        summary['native_comparison'] = dict(speedup=(native_trial['measurement']['windows_per_second']/numpy_trial['measurement']['windows_per_second']
            if numpy_trial and native_trial else None), scope='paired batch4/source128, same records/prior/initialization, compilation excluded',
            native_artifact=native_trial['execution']['native_cpu'] if native_trial else None)
    if c.get('native_bundle_comparison'):
        old=next((t for t in trials if t['name'] == 'previous_native_b4' and t['status'] == 'ok'),None)
        new=next((t for t in trials if t['name'] == 'optimized_native_b4' and t['status'] == 'ok'),None)
        summary['native_bundle_comparison']=dict(speedup=(new['measurement']['windows_per_second']/old['measurement']['windows_per_second']
            if old and new else None),scope='same native backend, batch4/source128/workers4, complete population and RNG; no compile/profile time',
            population_audit=({k:sum(r.get(k,0) for r in new['rows']) for k in ('full_candidate_columns','compact_candidate_columns',
                'materialized_candidate_columns','compact_descriptor_bytes')} if new else None),
            native_artifact=new['execution']['native_cpu'] if new else None)
    write_json(out/'summary.json', summary)
    lines = ['===== LOCAL WARM DISK SPEED ONLY =====', 'status='+status,
        f"history_frames={c['history_frames']}, future_frames=6, geometry_RAM=0MiB",
        'Same random/snapshot start per child; TRAIN16 diagnostic prior (formal full still TRAIN1024). No dev scoring.']
    for t in trials:
        if t['status'] != 'ok': lines.append(t['name']+': '+t['status']); continue
        m = t['measurement']
        lines.append(f"{t['name']} source_budget={t['source_budget']} actual_batch={m['mean_windows_per_batch']:.2f} "
            f"windows/s={m['windows_per_second']:.3f} seconds/window={m['seconds_per_window']:.3f} reserved={t['capacity_peak_reserved_mib']:.0f}MiB "
            f"warm_hits={m['cache_hits']}/{m['windows']} train15_if_representative={m['train15_hours_if_representative']:.2f}h")
        lines.append('  serial_host_stages_seconds='+json.dumps(m['stage_seconds'], sort_keys=True))
        lines.append('  CUDA_stream_seconds_NOT_active_utilization='+json.dumps(m['cuda_stream_seconds'], sort_keys=True))
    lines += ['recommendation='+json.dumps(decision), 'CPU profile status: '+profile_status,
        'reused_trials='+json.dumps(list(reused)),
        'CPU function details: cpu_profile.txt if complete (separate serialized-worker cProfile pass, NOT throughput measurement)', summary['caution']]
    if summary.get('cpu_comparison'): lines.append('CPU_OPTIMIZATION_COMPARISON='+json.dumps(summary['cpu_comparison']))
    if summary.get('native_comparison'): lines.append('NATIVE_CPU_COMPARISON='+json.dumps(summary['native_comparison']))
    if summary.get('native_bundle_comparison'): lines.append('NATIVE_BUNDLE_COMPARISON='+json.dumps(summary['native_bundle_comparison']))
    if error: lines.append('ERROR: '+str(error))
    (out/'summary.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    if print_report: print('\n'.join(lines), flush=True)
    return summary


def compare_cpu_paths(source_run, new_out, *, max_window_batch=32, launch=None, native_compare=False, native_bundle_compare=False):
    """Single bounded paired throughput check; no repeated prefill/prior/scan."""
    import shutil
    if native_compare or native_bundle_compare: max_window_batch = 4
    if native_compare and native_bundle_compare: raise ValueError('choose only one native comparison')
    source_run, new_out = Path(source_run).resolve(), Path(new_out).resolve()
    if new_out.exists(): raise RuntimeError('NEW CPU comparison output required; never overwrite the old timing report')
    c = validate_continuation(json.loads((source_run/'contract.json').read_text(encoding='utf-8')), source_run)
    if len(c['typical_keys']) < max_window_batch: raise RuntimeError('comparison batch larger than frozen sample')
    new_out.mkdir(parents=True)
    for name in ('records.pt', 'warm_cache.json', 'diagnostic_weights.json'):
        shutil.copyfile(source_run/name, new_out/name)
    c = {**c, 'out': str(new_out), 'cpu_comparison': str(source_run)}
    for flag in ('native_comparison','native_bundle_comparison'): c.pop(flag,None)
    if native_compare: c['native_comparison'] = True
    if native_bundle_compare: c['native_bundle_comparison'] = True
    if c.get('snapshot'):
        shutil.copyfile(c['snapshot'], new_out/'checkpoint_snapshot.pt')
        c['snapshot'] = str(new_out/'checkpoint_snapshot.pt')
    write_json(new_out/'contract.json', c)
    if launch is None:
        def launch(phase, t):
            cmd = [sys.executable, '-u', str(Path(__file__).resolve()), '--worker-contract', str(new_out/'contract.json'),
                '--phase', phase, '--trial', json.dumps(t)]
            code = subprocess.run(cmd, check=False).returncode
            if code not in (0, 42): raise RuntimeError(f'{phase} {t["name"]} child failed ({code}); partial comparison saved')
            return code
    started = time.perf_counter(); trials = []; status = 'in_progress'; error = None; profile_status = 'not_run'
    try:
        recipes = [dict(name='reference_b4', window_batch=4, source_budget=128, workers=4, persistent=True, optimize_cpu=False, optimize_kernels=False),
            dict(name='previous_b4', window_batch=4, source_budget=128, workers=4, persistent=True, optimize_cpu=True, optimize_kernels=False)]
        recipes += [dict(name=f'optimized_b{w}', window_batch=w, source_budget=32*w, workers=4, persistent=True, optimize_cpu=True, optimize_kernels=True)
            for w in (4, 8, 16, 32) if w <= max_window_batch]
        if native_compare:
            recipes = [dict(name=name, window_batch=4, source_budget=128, workers=4, persistent=True,
                optimize_cpu=True, optimize_kernels=True, backend=backend)
                for name, backend in (('optimized_b4', 'numpy'), ('native_b4', 'native'))]
        if native_bundle_compare:
            recipes=[dict(name=name,window_batch=4,source_budget=128,workers=4,persistent=True,
                optimize_cpu=True,optimize_kernels=True,backend='native',bundle=bundle)
                for name,bundle in (('previous_native_b4',False),('optimized_native_b4',True))]
        for t in recipes:
            launch('trial', t); row = load_completed_trial(new_out, t, c); trials.append(row)
            publish_summary(c, trials, started=started, profile_status=profile_status, status=status)
            if row['status'] == 'oom': break
        decision = recommend_trials([t for t in trials if t.get('optimize_cpu') and t.get('optimize_kernels', True)])
        if decision['recommended']:
            selected = next(t for t in recipes if t['name'] == decision['recommended_trial'])
            profile_status = 'complete' if launch('profile', selected) == 0 else 'oom_no_automatic_retry'
        if c.get('original_checkpoint') and sha256(c['original_checkpoint']) != c['snapshot_sha']:
            raise RuntimeError('source checkpoint changed externally; comparison used immutable snapshot')
        status = 'complete'
    except BaseException as failure:
        status = 'failed_partial'; error = f'{type(failure).__name__}: {failure}'; raise
    finally:
        publish_summary(c, trials, started=started, profile_status=profile_status, status=status, error=error, print_report=True)
    return 0


def run_contract(c, *, max_window_batch=128, continuation=False, finish_existing=False, launch=None):
    """Resume capacity diagnostics, not optimizer training; reuse completed files."""
    out = Path(c['out']); started = time.perf_counter(); trials = []; reused = []
    profile_status = 'not_run'; status = 'in_progress'; error = None
    if launch is None:
        def launch(phase, t=None):
            cmd = [sys.executable, '-u', str(Path(__file__).resolve()), '--worker-contract', str(out/'contract.json'), '--phase', phase]
            if t is not None: cmd += ['--trial', json.dumps(t)]
            code = subprocess.run(cmd, check=False).returncode
            if code != 0 and not (code == 42 and phase != 'warm'):
                raise RuntimeError(f"{phase} {t['name'] if t else 'warm'} child failed ({code}); partial summary saved, no automatic retry/full launch")
            return code
    def trial(t):
        if continuation and (out/(t['name']+'.json')).is_file():
            row = load_completed_trial(out, t, c); reused.append(t['name'])
            print('reuse_completed_trial='+t['name'], flush=True)
        elif finish_existing:
            return None
        else:
            launch('trial', t); row = load_completed_trial(out, t, c)
        trials.append(row)
        publish_summary(c, trials, started=started, profile_status=profile_status, status='in_progress', reused=reused)
        return row
    try:
        if not continuation: launch('warm')
        for workers, persistent, name in ((4, False, 'legacy_b4'), (4, True, 'pool4_b4'), (6, True, 'pool6_b4')):
            trial(dict(name=name, window_batch=4, source_budget=128, workers=workers, persistent=persistent))
        worker_trials = [t for t in trials if t['status'] == 'ok' and t['persistent']]
        if not worker_trials: raise RuntimeError('no successful four-window worker trial; do not launch full')
        workers = max(worker_trials, key=lambda t: t['measurement']['windows_per_second'])['workers']
        for windows in (8, 16, 32, 64, 128):
            if windows > max_window_batch: break
            row = trial(dict(name=f'pool{workers}_b{windows}', window_batch=windows, source_budget=32*windows,
                workers=workers, persistent=True))
            if row and row['status'] == 'oom': break
        decision = recommend_trials([t for t in trials if t.get('persistent')])
        if decision['recommended']:
            selected = next(t for t in trials if t['name'] == decision['recommended_trial'])
            t = {k: selected[k] for k in ('name', 'window_batch', 'source_budget', 'workers', 'persistent')}
            # Profile is tiny and separate; its old output is never silently
            # reused for another batch, nor mixed with steady-state throughput.
            profile_code = launch('profile', t)
            profile_status = 'complete' if profile_code == 0 else 'oom_no_automatic_retry'
        if c.get('original_checkpoint') and sha256(c['original_checkpoint']) != c['snapshot_sha']:
            raise RuntimeError('source checkpoint changed externally; benchmark used immutable snapshot')
        status = 'completed_existing_trials_only' if finish_existing else 'complete'
    except BaseException as failure:
        status = 'interrupted' if isinstance(failure, KeyboardInterrupt) else 'failed_partial'
        error = f'{type(failure).__name__}: {failure}'
        raise
    finally:
        publish_summary(c, trials, started=started, profile_status=profile_status,
            status=status, error=error, reused=reused, print_report=True)
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker-contract'); p.add_argument('--phase', choices=('warm', 'trial', 'profile')); p.add_argument('--trial')
    p.add_argument('--continue-run', help='existing diagnostic directory; reuse completed cache/prior/trials, NOT a training resume')
    p.add_argument('--compare-run', help='reuse existing frozen diagnostic population in a NEW output; compare reference CPU path and all safe optimized batches')
    p.add_argument('--native-compare', action='store_true', help='only paired current NumPy/native batch4; compile outside throughput clock')
    p.add_argument('--native-bundle-compare',action='store_true',help='only previous native vs complete CPU optimization bundle at batch4')
    p.add_argument('--finish-existing', action='store_true', help='only summarize completed trials and run their tiny CPU profile; skip untested larger batches')
    for key in ('config', 'train-cache', 'dev-cache', 'train-info', 'dev-info', 'base-checkpoint', 'dataroot', 'geometry-cache', 'out-dir'):
        p.add_argument('--'+key)
    p.add_argument('--checkpoint', help='optional read-only Local checkpoint; must match --history-frames')
    p.add_argument('--history-frames', type=int, choices=(4, 6), default=4)
    p.add_argument('--sample-windows', type=int, default=128); p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--max-window-batch', type=int, choices=(4, 8, 16, 32, 64, 128), default=128)
    p.add_argument('--cache-gib', type=float, default=48.); p.add_argument('--seed', type=int, default=20261002)
    a = p.parse_args()
    if a.worker_contract: return worker(a.worker_contract, a.phase, a.trial)
    if a.native_compare and a.native_bundle_compare: p.error('choose only one native comparison')
    if a.compare_run:
        if a.continue_run or a.finish_existing or not a.out_dir: p.error('--compare-run requires NEW --out-dir and cannot continue/finish the old report')
        return compare_cpu_paths(a.compare_run,a.out_dir,max_window_batch=min(a.max_window_batch,32),
            native_compare=a.native_compare,native_bundle_compare=a.native_bundle_compare)
    if a.native_compare or a.native_bundle_compare: p.error('native comparison requires --compare-run and NEW --out-dir')
    if a.continue_run:
        out = Path(a.continue_run).resolve()
        c = validate_continuation(json.loads((out/'contract.json').read_text(encoding='utf-8')), out)
        return run_contract(c, max_window_batch=a.max_window_batch, continuation=True, finish_existing=a.finish_existing)
    if a.finish_existing: p.error('--finish-existing requires --continue-run')
    for key in ('config', 'train_cache', 'dev_cache', 'train_info', 'dev_info', 'base_checkpoint'):
        if not getattr(a, key) or not Path(getattr(a, key)).is_file(): p.error('missing '+key)
    if not a.out_dir or not a.geometry_cache or not a.dataroot or not Path(a.dataroot).is_dir(): p.error('missing output/cache/dataroot')
    if a.sample_windows < a.max_window_batch or a.repeats < 1 or not np.isfinite(a.cache_gib) or a.cache_gib <= 0:
        p.error('sample must exercise largest batch; positive replay/storage budgets required')
    out = Path(a.out_dir).resolve()
    if out.exists(): p.error('NEW output directory required')
    cfg = load_runtime_config(a.config, [])
    base = torch.load(a.base_checkpoint, map_location='cpu', weights_only=False)
    base_sha = validate_clean_e14_checkpoint(base, a.base_checkpoint, CLEAN_SHA256); del base
    meta, records = load_cache(a.train_cache); keys = record_keys(records); validate_records(records)
    if len(records) != 20430: raise RuntimeError('full20430 TRAIN cache required')
    if a.sample_windows > len(records)-8: raise ValueError('insufficient distinct stress/typical records')
    indices = np.random.default_rng(a.seed).choice(len(records), a.sample_windows, replace=False).tolist()
    remaining = [i for i in range(len(records)) if i not in set(indices)]
    dense = sorted(remaining, key=lambda i: (-len(records[i]['features']), keys[i]))[:8]
    out.mkdir(parents=True)
    atomic_checkpoint(out/'records.pt', {'typical': [records[i] for i in indices], 'stress': [records[i] for i in dense]}); del records
    if (out/'records.pt').stat().st_size > 2**30: raise RuntimeError('diagnostic record artifact exceeds1GiB limit')
    c = dict(protocol=PROTOCOL, out=str(out), config=str(Path(a.config).resolve()), dataroot=str(Path(a.dataroot).resolve()),
        train_info=str(Path(a.train_info).resolve()), base_checkpoint=str(Path(a.base_checkpoint).resolve()), base_sha=base_sha,
        geometry_cache=str(Path(a.geometry_cache).resolve()), cache_gib=a.cache_gib, seed=a.seed,
        history_frames=a.history_frames, repeats=a.repeats, records_sha=sha256(out/'records.pt'),
        info_fingerprints={'train': sha256(a.train_info), 'dev': sha256(a.dev_info)},
        cache_fingerprints={'train': sha256(a.train_cache), 'dev': sha256(a.dev_cache)},
        typical_keys=[keys[i] for i in indices], stress_keys=[keys[i] for i in dense],
        patch_resolution_m=float(meta.get('patch_resolution_m', .8)), runtime_config_fingerprint=stable_json_fingerprint(cfg))
    if a.checkpoint:
        original = Path(a.checkpoint).resolve(); before = sha256(original)
        snapshot = out/'checkpoint_snapshot.pt'
        # Immutable single copy; never run trials against a concurrently replaced last.pt.
        import shutil
        shutil.copyfile(original, snapshot)
        if sha256(snapshot) != before or sha256(original) != before: raise RuntimeError('checkpoint changed while snapshotting; stop active writer first')
        c.update(snapshot=str(snapshot), snapshot_sha=before, original_checkpoint=str(original))
    write_json(out/'contract.json', c)
    return run_contract(c, max_window_batch=a.max_window_batch)


if __name__ == '__main__': sys.exit(main())
