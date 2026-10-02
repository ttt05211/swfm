#!/usr/bin/env python3
"""Warm-disk Local capacity/throughput probe. Never launches full training."""
from __future__ import annotations
import sys
from pathlib import Path
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import subprocess
import time
import numpy as np
import torch
from real_motion.causal_geometry_cache import CausalGeometryCache
from real_motion.causal_column_completion import ColumnConfig
from real_motion.column_runtime_pipeline import CachedColumnSource
from real_motion.joint_causal_columns import JointCausalColumns
from real_motion.local_st_world_model_v17 import config_from_mapping_v17
from real_motion.local_training_profile import CpuProfiles, trial_summary, recommend_trials
from real_motion.nuscenes_adapter import NuScenesWindowSource
from real_motion.runtime_config import load_runtime_config, make_prepare_config
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
                     workers, persistent=True, io_workers=2, cpu_profiles=None, max_batches=None, progress=None):
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
            kwargs = dict(profile=True, sampling_pool=pool, sampling_workers=workers, cpu_profiles=cpu_profiles)
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
            workers=t['workers'], persistent=t['persistent'], io_workers=2)
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
                rows=measured, diagnostic_weights=weights, history_frames=c['history_frames'])
            write_json(out/(name+'.json'), result)
            print(f"{name}: {m['windows_per_second']:.3f} windows/s, mean_batch={m['mean_windows_per_batch']:.2f}, peak_reserved={capacity_peak:.0f}MiB", flush=True)
        return 0
    except torch.cuda.OutOfMemoryError:
        if phase == 'warm': raise
        t = json.loads(trial); write_json(out/((t['name'] if phase == 'trial' else 'profile_oom')+'.json'), {**t, 'status': 'oom'})
        print('CUDA OOM isolated to diagnostic child; existing checkpoint untouched', flush=True); return 42
    finally: cache.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker-contract'); p.add_argument('--phase', choices=('warm', 'trial', 'profile')); p.add_argument('--trial')
    for key in ('config', 'train-cache', 'dev-cache', 'train-info', 'dev-info', 'base-checkpoint', 'dataroot', 'geometry-cache', 'out-dir'):
        p.add_argument('--'+key)
    p.add_argument('--checkpoint', help='optional read-only Local checkpoint; must match --history-frames')
    p.add_argument('--history-frames', type=int, choices=(4, 6), default=4)
    p.add_argument('--sample-windows', type=int, default=128); p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--max-window-batch', type=int, choices=(4, 8, 16, 32, 64, 128), default=128)
    p.add_argument('--cache-gib', type=float, default=48.); p.add_argument('--seed', type=int, default=20261002)
    a = p.parse_args()
    if a.worker_contract: return worker(a.worker_contract, a.phase, a.trial)
    for key in ('config', 'train_cache', 'dev_cache', 'train_info', 'dev_info', 'base_checkpoint'):
        if not getattr(a, key) or not Path(getattr(a, key)).is_file(): p.error('missing '+key)
    if not a.out_dir or not a.geometry_cache or not a.dataroot or not Path(a.dataroot).is_dir(): p.error('missing output/cache/dataroot')
    if a.sample_windows < a.max_window_batch or a.repeats < 1 or not np.isfinite(a.cache_gib) or a.cache_gib <= 0:
        p.error('sample must exercise largest batch; positive replay/storage budgets required')
    out = Path(a.out_dir).resolve()
    if out.exists(): p.error('NEW output directory required')
    started = time.perf_counter(); cfg = load_runtime_config(a.config, [])
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
        patch_resolution_m=float(meta.get('patch_resolution_m', .8)))
    if a.checkpoint:
        original = Path(a.checkpoint).resolve(); before = sha256(original)
        snapshot = out/'checkpoint_snapshot.pt'
        # Immutable single copy; never run trials against a concurrently replaced last.pt.
        import shutil
        shutil.copyfile(original, snapshot)
        if sha256(snapshot) != before or sha256(original) != before: raise RuntimeError('checkpoint changed while snapshotting; stop active writer first')
        c.update(snapshot=str(snapshot), snapshot_sha=before, original_checkpoint=str(original))
    write_json(out/'contract.json', c)
    def launch(phase, t=None):
        cmd = [sys.executable, '-u', str(Path(__file__).resolve()), '--worker-contract', str(out/'contract.json'), '--phase', phase]
        if t is not None: cmd += ['--trial', json.dumps(t)]
        code = subprocess.run(cmd, check=False).returncode
        if code != 0 and not (code == 42 and phase != 'warm'): raise RuntimeError(f'{phase} child failed ({code}); no automatic retry/full launch')
        return code
    launch('warm'); trials = []
    for workers, persistent, name in ((4, False, 'legacy_b4'), (4, True, 'pool4_b4'), (6, True, 'pool6_b4')):
        t = dict(name=name, window_batch=4, source_budget=128, workers=workers, persistent=persistent)
        launch('trial', t); trials.append(json.loads((out/(name+'.json')).read_text(encoding='utf-8')))
    worker_trials = [t for t in trials if t['status'] == 'ok' and t['persistent']]
    if not worker_trials: raise RuntimeError('no successful four-window worker trial; do not launch full')
    fastest = max(worker_trials, key=lambda t: t['measurement']['windows_per_second']); workers = fastest['workers']
    for windows in (8, 16, 32, 64, 128):
        if windows > a.max_window_batch: break
        name = 'pool'+str(workers)+'_b'+str(windows)
        t = dict(name=name, window_batch=windows, source_budget=32*windows, workers=workers, persistent=True)
        launch('trial', t); row = json.loads((out/(name+'.json')).read_text(encoding='utf-8')); trials.append(row)
        if row['status'] == 'oom': break
    decision = recommend_trials([t for t in trials if t.get('persistent')])
    profile_status = 'not_run'
    if decision['recommended']:
        selected = next(t for t in trials if t['name'] == decision['recommended_trial'])
        if selected['status'] != 'ok': raise RuntimeError('invalid diagnostic recommendation')
        profile_code = launch('profile', {k: selected[k] for k in ('name', 'window_batch', 'source_budget', 'workers', 'persistent')})
        profile_status = 'complete' if profile_code == 0 else 'oom_no_automatic_retry'
    if c.get('original_checkpoint') and sha256(c['original_checkpoint']) != c['snapshot_sha']:
        raise RuntimeError('source checkpoint changed externally; benchmark used immutable snapshot')
    summary = dict(protocol=PROTOCOL, history_frames=a.history_frames, future_frames=6,
        trials=trials, recommendation=decision, elapsed_seconds=time.perf_counter()-started,
        initialization='snapshot' if c.get('snapshot') else 'random_short_diagnostic_not_converged',
        cpu_profile_status=profile_status,
        speed_only=True, scientific_epochs=0, existing_checkpoint_unchanged=True, geometry_ram_cache_mib=0,
        caution='OS/frame-cache and workload variation remain; ETA excludes full cold prefill/eval/checkpoint; larger batch changes optimizer update count; no automatic full training')
    write_json(out/'summary.json', summary)
    lines = ['===== LOCAL WARM DISK SPEED ONLY =====', f'history_frames={a.history_frames}, future_frames=6, geometry_RAM=0MiB',
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
        'CPU function details: cpu_profile.txt if complete (separate serialized-worker cProfile pass, NOT throughput measurement)', summary['caution']]
    (out/'summary.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8'); print('\n'.join(lines), flush=True)
    return 0


if __name__ == '__main__': sys.exit(main())
