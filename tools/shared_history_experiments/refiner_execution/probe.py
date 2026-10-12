"""Frozen Refiner execution A/B; temporary backward/Adam, no published updates."""
import copy
import json
from pathlib import Path
import signal
import sys
import threading
import time
import torch
from tools.joint_occ_experiments import run as legacy
from tools.joint_occ_experiments.performance.probe import state_parity
from tools.shared_history_experiments.speed.probe import _cpu_clone, _is_oom
from tools.real_motion.joint_training_recovery import snapshot_checkpoint
from ..refine import engine, run as training
from ..refine.model import PROTOCOL
from .optimized import BACKENDS, Coordinates, execution


def tensor_check(left, right, *, rtol=3e-5, atol=2e-7):
    """Fixed gradient budgets; failures reported, never relaxed."""
    if left.keys() != right.keys(): raise RuntimeError('different gradient/output keys')
    maximum, mismatched, count = 0., 0, 0
    for name, a in left.items():
        b = right[name]
        if a is None or b is None:
            if a is not b: raise RuntimeError('gradient link differs: '+name)
            continue
        if a.shape != b.shape or a.dtype != b.dtype: raise RuntimeError('shape/dtype differs: '+name)
        close = torch.isclose(a, b, rtol=rtol, atol=atol)
        mismatched += int((~close).sum()); count += a.numel()
        if a.numel(): maximum = max(maximum, float((a-b).abs().max()))
    return dict(pass_gate=mismatched == 0, mismatched_elements=mismatched,
        compared_elements=count, max_abs_difference=maximum, budgets=dict(rtol=rtol, atol=atol))


def measure(model, optimizer, provider, saved, batches, *, passes=2, stop_event=None, progress=print,
            profile_directory=None):
    """Same weights/Adam/RNG/batches, alternating order; snapshots OUTSIDE timer."""
    if not batches or passes < 2: raise ValueError('nonempty batches, two alternating passes required')
    contract = saved['contract']
    caches = {name: Coordinates() for name in BACKENDS}
    initial_mode, initial_grad = model.training, [p.requires_grad for p in model.parameters()]
    def reset():
        model.load_state_dict(saved['models']['gt'], strict=True)
        optimizer.load_state_dict(copy.deepcopy(saved['optimizers']['gt']))
        optimizer.zero_grad(set_to_none=True)
        legacy.restore_rng(saved['rng']); model.train().requires_grad_(True)
    def state(): return _cpu_clone(dict(model=model.state_dict(), adam=optimizer.state_dict()))
    def step(name, rows, index, probe=False):
        with execution(name, caches[name]):
            return engine.train_arm(model, optimizer, provider, rows, 'gt',
                update=min(saved['updates']+index+1, contract['total_updates']),
                total_updates=contract['total_updates'], object_weight=contract['weights'][0],
                ego_weight=contract['weights'][1], lovasz_weight=contract['weights'][2], probe=probe)
    warm, failed, references = {}, {}, None
    timed = {name: [] for name in BACKENDS}
    finals = {name: [] for name in BACKENDS}
    profiles = {}
    try:
        for name in BACKENDS:
            if stop_event is not None and stop_event.is_set(): raise InterruptedError('discarded probe interrupted')
            reset(); gradients, logits = {}, {}
            def capture_gradient(opt, *_):
                gradients.update({k: None if p.grad is None else p.grad.detach().float().cpu().clone()
                                  for k, p in model.named_parameters()})
            def capture_logits(module, inputs, output):
                if not logits: logits['first_window_full6'] = output.detach().float().cpu().clone()
            hooks = [optimizer.register_step_pre_hook(capture_gradient),
                     model.columns.register_forward_hook(capture_logits)]
            try:
                stats = step(name, batches[0], 0, probe=True)
                current = state()
            except RuntimeError as error:
                if name == 'reference' or not _is_oom(error): raise
                failed[name] = dict(stage='warmup', error=str(error)); continue
            finally:
                for hook in hooks: hook.remove()
            if references is None: references = (current, gradients, logits, stats)
            first, g, z, ref_stats = references
            keys = ('CE', 'Lovasz', 'source_geometry', 'existence', 'ego_geometry')
            loss_check = tensor_check({k: torch.tensor(ref_stats[k]) for k in keys},
                {k: torch.tensor(stats[k]) for k in keys}, rtol=2e-6, atol=3e-7)
            forward = tensor_check(z, logits, rtol=2e-4, atol=2e-5)
            changes = int((z['first_window_full6'].argmax(1) != logits['first_window_full6'].argmax(1)).sum())
            warm[name] = dict(loss=loss_check, gradients=tensor_check(g, gradients),
                forward=forward, full6_semantic_changed_voxels=changes,
                state=state_parity(first, current), gradient_links=stats['gradient_links'])
            progress('REFINER_EXEC_WARM '+json.dumps(dict(backend=name, **warm[name])), flush=True)
            del current, gradients, logits
        references = None  # release dense CPU logits before timing
        del first, g, z, ref_stats
        for repeat in range(passes):
            for name in (BACKENDS if repeat % 2 == 0 else BACKENDS[::-1]):
                if name in failed: continue
                if stop_event is not None and stop_event.is_set(): raise InterruptedError('discarded probe interrupted')
                reset()
                if provider.device.type == 'cuda':
                    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(provider.device)
                    torch.cuda.synchronize(provider.device)
                try:
                    begun = time.perf_counter()
                    reports = []
                    for index, rows in enumerate(batches):
                        if stop_event is not None and stop_event.is_set(): raise InterruptedError('discarded probe interrupted')
                        reports.append(step(name, rows, index))
                    if provider.device.type == 'cuda': torch.cuda.synchronize(provider.device)
                    seconds = time.perf_counter()-begun
                    windows = sum(row['windows'] for row in reports)
                    peak = torch.cuda.max_memory_reserved(provider.device)/2**20 if provider.device.type == 'cuda' else None
                    row = dict(pass_index=repeat+1, windows=windows, seconds=seconds,
                        seconds_per_window=seconds/windows, peak_reserved_mib=peak,
                        free_vram_mib=torch.cuda.mem_get_info(provider.device)[0]/2**20 if provider.device.type == 'cuda' else None,
                        stages_seconds_per_window={k: sum(r['stages_seconds'].get(k, 0.) for r in reports)/windows
                            for k in reports[0]['stages_seconds']})
                    timed[name].append(row); finals[name].append(state())
                    progress('REFINER_EXEC '+json.dumps(dict(backend=name, **row)), flush=True)
                except RuntimeError as error:
                    if name == 'reference' or not _is_oom(error): raise
                    failed[name] = dict(stage='timed', pass_index=repeat+1, error=str(error))
                    optimizer.zero_grad(set_to_none=True)
                    if provider.device.type == 'cuda': torch.cuda.empty_cache()
        baseline_repeat = [state_parity(finals['reference'][0], s) for s in finals['reference'][1:]]
        replay_ok = bool(baseline_repeat) and all(r['numerical_parity'] for r in baseline_repeat)
        means = {n: sum(r['seconds'] for r in rows)/sum(r['windows'] for r in rows)
            if len(rows) == passes else None for n, rows in timed.items()}
        capacity = torch.cuda.get_device_properties(provider.device).total_memory/2**20 if provider.device.type == 'cuda' else None
        candidates = {}
        for name in BACKENDS[1:]:
            if name in failed or means[name] is None:
                candidates[name] = dict(pass_gate=False, failure=failed.get(name, 'incomplete')); continue
            comparisons = [state_parity(a, b) for a, b in zip(finals['reference'], finals[name])]
            numeric = all(r['numerical_parity'] for r in comparisons)
            faster = all(a['seconds'] > b['seconds'] for a, b in zip(timed['reference'], timed[name]))
            headroom = capacity is not None and all(r['peak_reserved_mib'] <= .9*capacity
                and r['free_vram_mib'] >= .1*capacity for r in timed[name])
            w = warm[name]
            candidates[name] = dict(speedup=means['reference']/means[name],
                final_state_parity=comparisons, baseline_repeat_parity=replay_ok,
                faster_each_pass=faster, memory_headroom_ge_10percent=headroom,
                pass_gate=bool(numeric and replay_ok and faster and headroom and w['loss']['pass_gate']
                    and w['gradients']['pass_gate'] and w['forward']['pass_gate']
                    and w['full6_semantic_changed_voxels'] == 0))
        if profile_directory is not None:
            names = ['reference']
            completed = [n for n in BACKENDS[1:] if means[n] is not None]
            if completed: names.append(min(completed, key=means.get))
            activities = [torch.profiler.ProfilerActivity.CPU]
            if provider.device.type == 'cuda': activities.append(torch.profiler.ProfilerActivity.CUDA)
            for name in names:
                if stop_event is not None and stop_event.is_set(): raise InterruptedError('discarded profile interrupted')
                reset()
                with torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True) as trace:
                    step(name, batches[0], 0)
                path = Path(profile_directory)/(name+'_trace.json'); trace.export_chrome_trace(str(path))
                events = trace.key_averages()
                rows = [dict(operator=e.key, calls=e.count, cpu_self_us=e.self_cpu_time_total,
                    device_self_us=getattr(e, 'self_device_time_total', 0.),
                    cpu_memory_bytes=e.self_cpu_memory_usage,
                    device_memory_bytes=getattr(e, 'self_device_memory_usage', 0)) for e in events]
                profiles[name] = dict(trace=str(path), top_device=sorted(rows, key=lambda r:r['device_self_us'], reverse=True)[:30],
                    top_cpu=sorted(rows, key=lambda r:r['cpu_self_us'], reverse=True)[:30])
        return dict(status='complete', seconds_per_window=means, timed_passes=timed, warmup=warm,
            candidates=candidates, baseline_repeat_checks=baseline_repeat, failed=failed, profiles=profiles,
            scientific_updates_published=0, automatic_promotion=False, source_weights_Adam_RNG_unchanged=True,
            scope='same source weights/Adam/RNG/B4/source128/full6 CE+Lovasz+motion+existence+ego; target prep INCLUDED; input I/O excluded; NOT FPS')
    finally:
        reset(); model.train(initial_mode)
        for p, flag in zip(model.parameters(), initial_grad): p.requires_grad_(flag)
        for cache in caches.values(): cache.rows.clear()


def main(event=None, argv=None):
    event = event or threading.Event()
    p = legacy.parser(); p.set_defaults(mode='gt', epochs=2, eval_population='dev512')
    p.add_argument('--probe-steps', type=int, default=4)
    p.add_argument('--passes', type=int, default=2)
    p.add_argument('--profile', action='store_true')
    a = p.parse_args(argv)
    if (not a.resume or a.max_updates or a.evaluate_only or a.epochs != 2 or a.mode != 'gt'
            or a.window_batch_size != 4 or a.source_budget != 128
            or not 1 <= a.probe_steps <= 16 or not 2 <= a.passes <= 4):
        raise ValueError('frozen GT epoch2 source, B4/source128; 1..16 steps and 2..4 passes')
    out, source = Path(a.out_dir).resolve(), Path(a.resume).resolve()
    if out == source.parent or source.parent in out.parents: raise ValueError('new probe OUTSIDE source directory')
    out.mkdir(parents=True, exist_ok=False); (out/'frozen').mkdir()
    digest = snapshot_checkpoint(source, out/'frozen/epoch2.pt')
    saved = torch.load(out/'frozen/epoch2.pt', map_location='cpu', weights_only=False)
    if (saved.get('protocol') != PROTOCOL or saved['epoch'] != 2 or saved['batch'] != 0
            or saved['updates'] != saved['contract']['total_updates']):
        raise RuntimeError('completed epoch2 Refiner required; NOT loss500 or old CCR')
    inner = vars(a).copy()
    for key in ('probe_steps', 'passes', 'profile'): inner.pop(key)
    inner.update(resume=str(out/'frozen/epoch2.pt'), evaluate_only=True)
    args = []
    for key, value in inner.items():
        if key == 'override' or value is None: continue
        if isinstance(value, bool):
            if value: args.append('--'+key.replace('_', '-'))
        else: args += ['--'+key.replace('_', '-'), str(value)]
    for value in a.override or []: args += ['--override', value]
    original_eval, original_batches = engine.evaluate, legacy.prefetched_batches
    def batches(provider, source, ignored, descriptors, **kwargs):
        _, records = legacy.load_cache(a.train_cache)
        selected = legacy.align_records(records, [tuple(k) for k in saved['contract']['train_keys']])
        jobs = legacy.epoch_groups(selected, saved['contract']['seed'], 0, 4, 128)[:a.probe_steps]
        legacy.write_json(out/'population.json', [[r['scene_name'], r['t0_token']] for group in jobs for r in group])
        return original_batches(provider, provider.ccr_history_cache_source, jobs, descriptors, **kwargs)
    def evaluate(models, provider, source, rows, **kwargs):
        model = models['gt']; c = saved['contract']
        optimizer = torch.optim.AdamW([
            dict(params=model.transport.parameters(), lr=c['lr'][0], initial_lr=c['lr'][0], weight_decay=1e-4),
            dict(params=model.columns.parameters(), lr=c['lr'][1], initial_lr=c['lr'][1], weight_decay=.01),
            dict(params=model.ego.parameters(), lr=c['lr'][2], initial_lr=c['lr'][2], weight_decay=.01)])
        report = measure(model, optimizer, provider, saved, list(rows), passes=a.passes, stop_event=event,
            profile_directory=out if a.profile else None)
        report.update(source_checkpoint=str(a.resume), source_update=saved['updates'], source_sha256=digest,
            implementation={f.name: legacy.sha256(f) for f in Path(__file__).parent.glob('*.py')})
        legacy.write_json(out/'speed.json', report)
        return dict(windows=0, variants={}, trajectory={}, discarded_training_probe=True)
    try:
        engine.evaluate, legacy.prefetched_batches = evaluate, batches
        status = training.main(event, args)
    finally:
        engine.evaluate, legacy.prefetched_batches = original_eval, original_batches
    if legacy.sha256(source) != digest or legacy.sha256(out/'frozen/epoch2.pt') != digest:
        raise RuntimeError('source/frozen checkpoint changed')
    if not (out/'speed.json').is_file():
        print('Probe interrupted; no temporary updates saved. Use a NEW probe directory to retry.', flush=True)
        return status
    report = json.loads((out/'speed.json').read_text(encoding='utf-8'))
    lines = ['===== FROZEN REFINER EXECUTION A/B =====',
        f'source_update={saved["updates"]}; scientific_updates=0; no promotion; NOT FPS.',
        'Full6 CE+Lovasz+motion+existence+ego, same B4/source128/Adam/RNG. Warmup/profile excluded.']
    for name, seconds in report['seconds_per_window'].items(): lines.append(f'{name}: {seconds} seconds/window')
    for name, row in report['candidates'].items(): lines.append('CANDIDATE '+name+' '+json.dumps(row))
    lines.append('Numerical FAIL remains FAIL; all completed timings retained, no tolerance changes.')
    (out/'summary.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)
    return status


if __name__ == '__main__':
    event = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: event.set())
    sys.exit(main(event) or 0)
