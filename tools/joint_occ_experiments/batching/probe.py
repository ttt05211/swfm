"""Read-only real-GPU A/B/C: sequential / packed networks / packed + two horizons."""
from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import time
import numpy as np
import torch
from .. import run as legacy
from ..performance import probe as base
from ..engine import train_arm as reference_step, forward_window
from ..soft_geometry import SoftScene, occupancy_loss
from .engine import train_arm
from .geometry import GroupedScene, grouped_loss

BACKENDS = ('current_fused', 'packed_h1', 'packed_h2')


def execution_fingerprint(root, group):
    files = sorted(Path(__file__).parent.glob('*.py'))
    return legacy.stable_json_fingerprint(dict(base=base.legacy_fingerprint(root), horizon_group=group,
        sources={p.name: legacy.sha256(p) for p in files}, protocol='packed_GT_execution_v1'))


def render_audit(model, provider, row, group):
    """Actual first-window boundary probabilities + gradients, no optimizer."""
    fw = forward_window(model, provider, row, 'gt')
    names = ('poses', 'targets', 'yaws', 'mapped', 'actions')
    horizons = list(range(group))
    def compute(batched):
        inputs = {key: fw[key].detach().clone().requires_grad_(True) for key in names}
        if batched:
            scene = GroupedScene(fw['prep'], provider.pcfg.grid, provider.device)
            probability = scene.render_group(horizons, **inputs, evidence=fw['evidence'], plan=fw['plan'])
            loss, _, _ = grouped_loss(probability, row[1]['future_gt_occ'][horizons])
        else:
            scene = SoftScene(fw['prep'], provider.pcfg.grid, provider.device)
            probability = torch.stack([scene.render(h, inputs['poses'][h], inputs['targets'], inputs['yaws'], inputs['mapped'],
                fw['evidence'], fw['plan'], inputs['actions']) for h in horizons])
            loss = torch.stack([occupancy_loss(p, torch.as_tensor(row[1]['future_gt_occ'][h], device=provider.device),
                class_ids=[int(c) for c in np.unique(row[1]['future_gt_occ'][h]) if 0 <= c < 17])[0]
                for p, h in zip(probability, horizons)])
        gradients = torch.autograd.grad(loss.sum(), list(inputs.values()), allow_unused=True)
        return dict(probability=probability.detach(), losses=loss.detach(),
                    gradients={name: gradient for name, gradient in zip(names, gradients)})
    expected, actual = compute(False), compute(True)
    diagnostics = {}
    def compare(name, a, b, atol, rtol):
        if a is None or b is None:
            diagnostics[name] = dict(pass_gate=a is b, max_abs_difference=0. if a is b else None)
            return
        close = torch.isclose(a, b, atol=atol, rtol=rtol)
        diagnostics[name] = dict(pass_gate=bool(close.all()), mismatched_elements=int((~close).sum()),
            max_abs_difference=float((a-b).abs().max()) if a.numel() else 0.)
    compare('probability', expected['probability'], actual['probability'], 2e-6, 2e-5)
    compare('losses', expected['losses'], actual['losses'], 3e-7, 2e-6)
    for name in names: compare(name, expected['gradients'][name], actual['gradients'][name], 2e-7, 3e-5)
    return dict(pass_gate=all(row['pass_gate'] for row in diagnostics.values()), details=diagnostics,
        scope='actual first-window original-vs-grouped boundary probabilities/loss/gradients; not BF16 byte identity',
        budgets=dict(probability=[2e-6, 2e-5], loss=[3e-7, 2e-6], gradients=[2e-7, 3e-5]))


def measure(model, optimizer, provider, saved, batches, *, passes=2, progress=print):
    if not batches or passes < 2: raise ValueError('two passes and nonempty batches required')
    contract = saved['contract']
    if contract['window_batch'] != 4 or contract['source_budget'] != 128:
        raise RuntimeError('original batch4/source128 required')
    def reset():
        model.load_state_dict(saved['models']['gt'], strict=True)
        optimizer.load_state_dict(copy.deepcopy(saved['optimizers']['gt']))
        legacy.restore_rng(saved['rng']); model.train().requires_grad_(True)
    def state():
        return dict(model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    adam=copy.deepcopy(optimizer.state_dict()))
    with ThreadPoolExecutor(max_workers=min(4, provider.workers)) as pool:
        def step(name, rows, index, probe=False):
            options = dict(update=min(saved['updates']+index+1, contract['total_updates']),
                total_updates=contract['total_updates'], object_weight=contract['weights'][0],
                ego_weight=contract['weights'][1], lovasz_weight=contract['weights'][2], probe=probe)
            if name == 'current_fused': return reference_step(model, optimizer, provider, rows, 'gt', **options)
            return train_arm(model, optimizer, provider, rows, 'gt', **options,
                horizon_group=int(name[-1]), projection_pool=pool)
        warm = {}; warm_state = warm_stats = None
        for name in BACKENDS:
            reset()
            audit = dict(pass_gate=True, scope='reference') if name == 'current_fused' else render_audit(model, provider, batches[0][0], int(name[-1]))
            # Audits must not alter the warmup RNG sequence.
            reset(); stats = step(name, batches[0], 0, probe=True); current = state()
            if warm_state is None: warm_state, warm_stats = current, stats
            parity = base.state_parity(warm_state, current)
            diagnostic = lambda values: {key: torch.tensor(values[key]) for key in
                ('CE', 'Lovasz', 'source_geometry', 'existence', 'ego_geometry', 'grad_norm')}
            losses = base.state_parity(diagnostic(warm_stats), diagnostic(stats))
            links = base.state_parity({k: torch.tensor(v) for k, v in warm_stats['gradient_links'].items()},
                                     {k: torch.tensor(v) for k, v in stats['gradient_links'].items()})
            warm[name] = dict(numerical_parity=parity['numerical_parity'] and losses['numerical_parity'] and links['numerical_parity'] and audit['pass_gate'],
                state_parity=parity, loss_parity=losses, link_parity=links, loss_gradient_audit=audit,
                gradient_links=stats['gradient_links'])
            progress('GT_SPEED_WARM '+json.dumps(dict(backend=name, **warm[name])), flush=True)
        timed = {name: [] for name in BACKENDS}; parity = {}; reference = None
        for repeat in range(passes):
            for name in (BACKENDS if repeat % 2 == 0 else BACKENDS[::-1]):
                reset()
                if provider.device.type == 'cuda':
                    torch.cuda.synchronize(provider.device); torch.cuda.reset_peak_memory_stats(provider.device)
                tick = time.perf_counter()
                reports = [step(name, rows, i) for i, rows in enumerate(batches)]
                if provider.device.type == 'cuda': torch.cuda.synchronize(provider.device)
                elapsed = time.perf_counter()-tick; windows = sum(s['windows'] for s in reports)
                current = state()
                if reference is None: reference = current
                result = base.state_parity(reference, current)
                row = dict(pass_index=repeat+1, windows=windows, seconds=elapsed, seconds_per_window=elapsed/windows,
                    peak_allocated_mib=torch.cuda.max_memory_allocated(provider.device)/2**20 if provider.device.type == 'cuda' else None,
                    peak_reserved_mib=torch.cuda.max_memory_reserved(provider.device)/2**20 if provider.device.type == 'cuda' else None,
                    final_state_parity=result,
                    host_stages_seconds_per_window={key: sum(s['stages_seconds'][key] for s in reports)/windows for key in reports[0]['stages_seconds']})
                timed[name].append(row)
                previous = parity.setdefault(name, dict(numerical_parity=True, max_state_abs_difference=0.))
                previous['numerical_parity'] &= result['numerical_parity']
                previous['max_state_abs_difference'] = max(previous['max_state_abs_difference'], result['max_state_abs_difference'])
                progress('GT_SPEED '+json.dumps(dict(backend=name, **row)), flush=True)
        means = {name: sum(r['seconds'] for r in rows)/sum(r['windows'] for r in rows) for name, rows in timed.items()}
        capacity = torch.cuda.get_device_properties(provider.device).total_memory/2**20 if provider.device.type == 'cuda' else None
        candidates = {}
        root = Path(__file__).resolve().parents[3]
        for name in BACKENDS[1:]:
            faster = all(a['seconds'] < b['seconds'] for a, b in zip(timed[name], timed['current_fused']))
            numerical = parity[name]['numerical_parity'] and warm[name]['numerical_parity']
            replay = parity['current_fused']['numerical_parity']
            headroom = capacity is not None and max(r['peak_reserved_mib'] for r in timed[name]) <= .9*capacity
            speedup = means['current_fused']/means[name]
            candidates[name] = dict(speedup=speedup, numerical_parity=numerical, baseline_repeat_parity=replay,
                faster_each_pass=faster, memory_headroom_ge_10percent=headroom,
                execution_fingerprint=execution_fingerprint(root, int(name[-1])),
                pass_gate=numerical and replay and faster and headroom and speedup >= 1.1)
        reset()
        return dict(seconds_per_window=means, timed_passes=timed, warmup=warm, final_state_parity=parity,
            candidates=candidates, global_batch=4, source_budget=128, projection_workers=min(4, provider.workers),
            source_weights_optimizer_rng_unchanged=True, route='probe_only_no_automatic_training_backend_change')


def main(argv=None):
    old_measure, old_backends = base.measure, base.BACKENDS
    try:
        base.measure, base.BACKENDS = measure, BACKENDS
        return base.main(argv)
    finally:
        base.measure, base.BACKENDS = old_measure, old_backends


if __name__ == '__main__': main()
