"""Read-only checkpoint, same-batch GT training A/B/C execution probe.

Transient model/Adam clones do real backward, but NO scientific checkpoint,
cursor, optimizer or RNG is published to the source training directory.
"""
from contextlib import nullcontext
import copy
import json
from pathlib import Path
import time
import torch
from .. import run as legacy
from ..engine import train_arm
from ..model import JointOccEgo, PROTOCOL
from ..data import HistoryInputs, attach_caches, prefetched_batches
from .loss import prepared_loss_execution


BACKENDS = ('current_fused', 'prepared_combined', 'prepared_sparse')
DEPENDENCIES = ('real_motion/joint_surface_ccr.py', 'real_motion/joint_causal_columns.py',
    'real_motion/local_st_world_model_v18_se2.py', 'real_motion/local_st_world_model_v17.py',
    'real_motion/surface_canonical_repair.py', 'real_motion/surface_projection_execution.py',
    'real_motion/surface_ccr_execution.py', 'real_motion/ego_navigation.py',
    'tools/ego_experiments/ego_kinematic.py', 'tools/real_motion/joint_training_recovery.py')


def legacy_fingerprint(root):
    # IDENTICAL to the old trainer; nested opt-in tools are not in this hash.
    paths = list((root/'tools/joint_occ_experiments').glob('*.py'))
    return legacy.stable_json_fingerprint({**{str(p.relative_to(root)): legacy.sha256(p) for p in paths},
                                         **{p: legacy.sha256(root/p) for p in DEPENDENCIES}})


def assert_close_state(left, right):
    """Numerical, NOT byte, parity: shared BF16 backward may round differently."""
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=2e-4, atol=2e-5)
        return float((left.float()-right.float()).abs().max()) if left.numel() else 0.
    if isinstance(left, dict):
        if left.keys() != right.keys(): raise AssertionError('different state keys')
        return max((assert_close_state(left[k], right[k]) for k in left), default=0.)
    if isinstance(left, (list, tuple)):
        if len(left) != len(right): raise AssertionError('different state lengths')
        return max((assert_close_state(a, b) for a, b in zip(left, right)), default=0.)
    if left != right: raise AssertionError('different scalar state')
    return 0.


def measure(model, optimizer, provider, saved, batches, *, passes=2, progress=print):
    """Reset SAME source weights/Adam/RNG before every pass; AB/BA order."""
    if not batches or passes < 2: raise ValueError('nonempty batches and two alternating passes required')
    weights, adam, contract = saved['models']['gt'], saved['optimizers']['gt'], saved['contract']
    warm, warm_reference, final_reference = {}, None, None
    def reset():
        model.load_state_dict(weights, strict=True); optimizer.load_state_dict(copy.deepcopy(adam))
        legacy.restore_rng(saved['rng']); model.train().requires_grad_(True)
    def step(name, rows, i, probe=False):
        scope = (nullcontext() if name == BACKENDS[0] else
                 prepared_loss_execution(rows, provider.device, sparse=name == 'prepared_sparse'))
        with scope:
            return train_arm(model, optimizer, provider, rows, 'gt',
                update=min(saved['updates']+i+1, contract['total_updates']), total_updates=contract['total_updates'],
                object_weight=contract['weights'][0], ego_weight=contract['weights'][1],
                lovasz_weight=contract['weights'][2], probe=probe)
    # First-use allocation/probe is deliberately OUTSIDE throughput timing.
    for name in BACKENDS:
        reset(); stats = step(name, batches[0], 0, probe=True)
        state = dict(model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                     adam=copy.deepcopy(optimizer.state_dict()))
        if warm_reference is None: warm_reference = state; reference_stats = stats
        difference = assert_close_state(warm_reference, state)
        for key in ('CE', 'Lovasz', 'source_geometry', 'existence', 'ego_geometry', 'grad_norm'):
            torch.testing.assert_close(torch.tensor(stats[key]), torch.tensor(reference_stats[key]), rtol=2e-4, atol=2e-5)
        assert_close_state({k: torch.tensor(v) for k, v in reference_stats['gradient_links'].items()},
                           {k: torch.tensor(v) for k, v in stats['gradient_links'].items()})
        warm[name] = dict(numerical_parity=True, max_state_abs_difference=difference,
                          gradient_links=stats['gradient_links'])
    timed = {name: [] for name in BACKENDS}
    parity = {}
    for repeat in range(passes):
        for name in (BACKENDS if repeat % 2 == 0 else BACKENDS[::-1]):
            reset()
            if provider.device.type == 'cuda':
                torch.cuda.synchronize(provider.device); torch.cuda.reset_peak_memory_stats(provider.device)
            tick = time.perf_counter()
            reports = [step(name, rows, i) for i, rows in enumerate(batches)]
            if provider.device.type == 'cuda': torch.cuda.synchronize(provider.device)
            elapsed = time.perf_counter()-tick
            windows = sum(s['windows'] for s in reports)
            row = dict(pass_index=repeat+1, windows=windows, seconds=elapsed,
                       seconds_per_window=elapsed/windows,
                       peak_allocated_mib=(torch.cuda.max_memory_allocated(provider.device)/2**20
                                           if provider.device.type == 'cuda' else None),
                       peak_reserved_mib=(torch.cuda.max_memory_reserved(provider.device)/2**20
                                          if provider.device.type == 'cuda' else None))
            row['host_stages_seconds_per_window'] = {
                stage: sum(s['stages_seconds'].get(stage, 0.) for s in reports)/windows
                for stage in reports[0]['stages_seconds']}
            row['unattributed_seconds_per_window_including_target_preparation'] = (
                elapsed-sum(s['seconds'] for s in reports))/windows
            timed[name].append(row)
            state = dict(model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                         adam=copy.deepcopy(optimizer.state_dict()))
            if final_reference is None: final_reference = state
            parity[name] = dict(numerical_parity=True, max_state_abs_difference=assert_close_state(final_reference, state))
            progress('GT_SPEED '+json.dumps(dict(backend=name, **row)), flush=True)
    means = {name: sum(r['seconds'] for r in rows)/sum(r['windows'] for r in rows) for name, rows in timed.items()}
    candidates = {}
    for name in BACKENDS[1:]:
        speedup = means[BACKENDS[0]]/means[name]
        candidates[name] = dict(speedup=speedup, numerical_parity=parity[name]['numerical_parity'],
            faster_each_pass=all(a['seconds'] < b['seconds'] for a, b in zip(timed[name], timed[BACKENDS[0]])),
            pass_gate=speedup >= 1.1 and all(a['seconds'] < b['seconds'] for a, b in zip(timed[name], timed[BACKENDS[0]])))
    reset()  # Also leave these transient objects at the unmodified source state.
    return dict(seconds_per_window=means, timed_passes=timed, warmup=warm, final_state_parity=parity,
                candidates=candidates, scope='GT real forward/backward+Adam+target preparation; input I/O excluded',
                source_weights_optimizer_rng_unchanged=True,
                route='probe_only_no_automatic_training_backend_change')


def main(argv=None):
    parser = legacy.parser()
    parser.description = __doc__
    parser.add_argument('--probe-steps', type=int, default=8)
    a = parser.parse_args(argv)
    if not a.resume or not 1 <= a.probe_steps <= 32: raise ValueError('source checkpoint and 1..32 probe steps required')
    if a.mode != 'gt' or a.training_backend != 'fused' or a.evaluate_only:
        raise ValueError('GT fused training probe only; no Pred/evaluation mode')
    if not 1 <= a.cpu_workers <= 20 or not 1 <= a.prefetch_workers <= 8: raise ValueError('bounded workers required')
    if not 0 <= a.geometry_ram_mib <= 4096 or not 0 <= a.frame_cache_mib <= 8192 or not 0 <= a.descriptor_ram_mib <= 4096:
        raise ValueError('bounded memory required')
    out = Path(a.out_dir).resolve(); source_checkpoint = Path(a.resume).resolve()
    if out == source_checkpoint.parent or source_checkpoint.parent in out.parents:
        raise ValueError('probe output must be OUTSIDE the source training directory')
    if out.exists(): raise RuntimeError('NEW probe directory required; never overwrite')
    device = legacy.require_cuda(a.device)
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[3]
    out.mkdir(parents=True)
    from tools.real_motion.joint_training_recovery import snapshot_checkpoint
    digest = snapshot_checkpoint(source_checkpoint, out/'source_snapshot.pt')
    saved = torch.load(out/'source_snapshot.pt', map_location='cpu', weights_only=False)
    contract = saved['contract']
    if (saved.get('protocol') != PROTOCOL or contract.get('modes') != ['gt']
            or contract.get('training_backend') != 'fused' or contract.get('implementation') != legacy_fingerprint(root)):
        raise RuntimeError('exact CURRENT fused GT-only checkpoint/code required')
    if (contract['execution']['torch_version'] != str(torch.__version__)
            or contract['execution']['cuda_version'] != torch.version.cuda):
        raise RuntimeError('original Torch/CUDA precision environment required')
    for key, expected in contract['data'].items():
        if legacy.sha256(getattr(a, key)) != expected: raise RuntimeError('source data differs: '+key)
    config = legacy.config_from_mapping_v17(contract['model']['motion'])
    model = JointOccEgo(config, width=contract['model']['repair']['width']).to(device)
    if model.configs() != contract['model']: raise RuntimeError('model architecture differs')
    # Adam param grouping must match the source; dummy constructor LR is never used.
    optimizer = torch.optim.AdamW([
        dict(params=model.transport.parameters(), lr=contract['lr'][0]),
        dict(params=model.columns.parameters(), lr=contract['lr'][1]),
        dict(params=model.ego.parameters(), lr=contract['lr'][2])])
    pcfg = legacy.make_prepare_config(legacy.load_runtime_config(a.config, a.override))
    provider = legacy.PilotProvider(a.base_checkpoint, legacy.CLEAN_SHA256, pcfg, device, a.cpu_workers, model, None)
    provider.raw_io_workers = 1
    from real_motion.canonical_repair_execution import CanonicalCpuExecution
    from real_motion.surface_projection_execution import SurfaceMapExecution
    from real_motion.strong_majority_execution import ParallelNativeMajority
    provider.ccr_execution = SurfaceMapExecution(CanonicalCpuExecution('native_parallel', min(4, a.cpu_workers)))
    provider.strong_majority = ParallelNativeMajority(min(4, a.cpu_workers))
    train_source = legacy.CachedColumnSource(legacy.NuScenesWindowSource(a.dataroot, info_pkl=a.train_info),
                                            a.frame_cache_mib, copy_on_insert=False)
    provider.ccr_history_cache_source = train_source
    # VAL namespace uses declared metadata only; no dev frames/evaluation are loaded.
    provider.ccr_val_history_cache_source = None
    try:
        caches = attach_caches(provider, a, root)
        if {k: caches[k]['namespace'] for k in ('train', 'val')} != contract['cache_namespace']:
            raise RuntimeError('original history cache namespaces required')
        _, records = legacy.load_cache(a.train_cache)
        legacy.record_keys(records)
        train = legacy.align_records(records, [tuple(k) for k in contract['train_keys']])
        del records
        groups = [legacy.epoch_groups(train, contract['seed'], e, contract['window_batch'], contract['source_budget'])
                  for e in range(contract['epochs'])]
        if sum(len(g) for g in groups) != contract['total_updates']: raise RuntimeError('source schedule/groups differ')
        epoch, cursor = int(saved['epoch']), int(saved['batch'])
        if (not 0 <= epoch <= len(groups) or epoch == len(groups) and cursor != 0
                or epoch < len(groups) and not 0 <= cursor < len(groups[epoch])
                or saved['updates'] != sum(len(g) for g in groups[:epoch])+cursor):
            raise RuntimeError('invalid source cursor')
        jobs = ([g for e, sequence in enumerate(groups) if e >= epoch
                 for j, g in enumerate(sequence) if e > epoch or j >= cursor]+groups[0])[:a.probe_steps]
        descriptors = HistoryInputs(provider, ram_mib=a.descriptor_ram_mib)
        batches = list(prefetched_batches(provider, train_source, jobs, descriptors, workers=a.prefetch_workers))
        report = measure(model, optimizer, provider, saved, batches)
        report.update(status='complete', source_checkpoint=str(source_checkpoint), source_snapshot_sha256=digest,
            source_completed_updates=saved['updates'], scientific_updates_published=0, cache_audit=caches,
            probe_keys=[[r['scene_name'], r['t0_token']] for group in jobs for r in group],
            implementation=legacy.stable_json_fingerprint({p.name: legacy.sha256(p) for p in Path(__file__).parent.glob('*.py')}))
        legacy.write_json(out/'speed.json', report)
        lines = ['===== READ-ONLY GT TRAINING SPEED A/B/C =====',
                 f"source_update={saved['updates']}; scientific_updates_published=0; original Adam/RNG unchanged",
                 'Same checkpoint/init/RNG/batches/Adam/cosine; two alternating passes; warmup excluded.',
                 'Target preparation INCLUDED; shared input I/O excluded. Numerical parity, NOT BF16 byte identity.']
        for name, seconds in report['seconds_per_window'].items():
            lines.append(f'{name}: {seconds:.6f} seconds/window')
        lines += ['CANDIDATES '+json.dumps(report['candidates']), 'No automatic continuation/backend promotion; old GT resume remains compatible.']
        text = '\n'.join(lines)+'\n'
        (out/'summary.txt').write_text(text, encoding='utf-8'); print(text, flush=True)
    except Exception as error:
        legacy.write_json(out/'speed.json', dict(status='failed', error=str(error),
            source_checkpoint=str(source_checkpoint), source_snapshot_sha256=digest,
            scientific_updates_published=0, source_weights_optimizer_rng_unchanged=True))
        raise
    finally:
        for name in ('ccr_history_cache', 'ccr_val_history_cache'):
            cache = getattr(provider, name, None)
            if cache is not None: cache.close()
        provider.ccr_execution.close(); provider.strong_majority.close()
    return 0


if __name__ == '__main__': main()
