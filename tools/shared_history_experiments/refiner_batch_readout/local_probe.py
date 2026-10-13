"""Local synthetic B4/H6 and tiny complete-network A/B, NOT real-data FPS."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
import numpy as np
import torch
from tools.shared_history_experiments.refine.model import SemanticRefiner
from tools.shared_history_experiments.full_gt.engine import semantic_loss
from tools.shared_history_experiments.refiner_execution.probe import tensor_check
from tools.joint_occ_experiments.performance.probe import state_parity
from .optimized import BACKENDS, batch_forward, replay_update


def sync(device):
    if device.type == 'cuda': torch.cuda.synchronize(device)


def snapshot(model, adam):
    def cpu(value):
        if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
        if isinstance(value, dict): return {k: cpu(v) for k, v in value.items()}
        if isinstance(value, list): return [cpu(v) for v in value]
        if isinstance(value, tuple): return tuple(cpu(v) for v in value)
        return value
    return dict(model=cpu(model.state_dict()), adam=cpu(adam.state_dict()))


def synthetic_step(model, adam, memories, priors, labels, shape, backend, *, audit=False):
    adam.zero_grad(set_to_none=True)
    for x in (*memories, *priors): x.grad = None
    outputs, losses = [], []
    if backend.startswith('packed'):
        zs = batch_forward(model, memories, priors, shape, backend)
        roots = []
        for z, lab in zip(zs, labels):
            roots.append(semantic_loss(z, lab)[0].mean()/len(labels))
            if audit: outputs.append(z.detach().cpu().clone())
        torch.autograd.backward(roots)
        losses = [r.detach() for r in roots]
    else:
        for memory, prior, lab in zip(memories, priors, labels):
            z = batch_forward(model, [memory], [prior], shape, backend)[0]
            loss = semantic_loss(z, lab)[0].mean()/len(labels)
            if audit: outputs.append(z.detach().cpu().clone())
            loss.backward(); losses.append(loss.detach())
            del z, loss
    grads = None
    if audit:
        grads = {k: None if p.grad is None else p.grad.detach().cpu().clone() for k, p in model.named_parameters()}
        grads.update({f'memory{i}': x.grad.detach().cpu().clone() for i, x in enumerate(memories)})
        grads.update({f'prior{i}': x.grad.detach().cpu().clone() for i, x in enumerate(priors)})
    if not bool(torch.isfinite(torch.stack(losses)).all()): raise RuntimeError('nonfinite synthetic objective')
    if not audit:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        adam.step()
    return dict(loss=torch.stack(losses).sum().detach().cpu(), gradients=grads, outputs=outputs)


def parity(a, b):
    forward = tensor_check({str(i): z for i, z in enumerate(a['outputs'])},
                           {str(i): z for i, z in enumerate(b['outputs'])}, rtol=2e-4, atol=2e-5)
    return dict(loss=tensor_check({'loss': a['loss']}, {'loss': b['loss']}, rtol=2e-6, atol=3e-7),
                gradient=tensor_check(a['gradients'], b['gradients']), forward=forward,
                changed_voxels=sum(int((z.argmax(1) != y.argmax(1)).sum())
                                   for z, y in zip(a['outputs'], b['outputs'])))


def measure(model, adam, step, *, warmup, repeats, device, audit=None):
    initial = snapshot(model, adam)
    rows, states = {n: [] for n in BACKENDS}, {}
    audit_rows = {}
    if audit is not None:
        reference = None
        for name in BACKENDS:
            model.load_state_dict(initial['model']); adam.load_state_dict(copy.deepcopy(initial['adam']))
            result = audit(name)
            if reference is None: reference = result
            audit_rows[name] = parity(reference, result)
            print('LOCAL_PARITY '+json.dumps(dict(backend=name, **audit_rows[name])), flush=True)
    for pass_index in range(2):
        for name in (BACKENDS if pass_index == 0 else tuple(reversed(BACKENDS))):
            model.load_state_dict(initial['model']); adam.load_state_dict(copy.deepcopy(initial['adam']))
            try:
                for i in range(warmup): step(name, i)
                sync(device)
                free_before = torch.cuda.mem_get_info(device)[0]/2**20 if device.type == 'cuda' else None
                if device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device)
                durations = []
                for i in range(repeats):
                    sync(device); begun = time.perf_counter()
                    step(name, warmup+i); sync(device)
                    durations.append(time.perf_counter()-begun)
                row = dict(pass_index=pass_index+1, seconds_per_B4=statistics.mean(durations),
                    seconds_per_window=statistics.mean(durations)/4,
                    free_device_mib_before_timer=free_before,
                    peak_allocated_mib=torch.cuda.max_memory_allocated(device)/2**20 if device.type == 'cuda' else 0.)
                final = snapshot(model, adam)
                if 'reference' in states: row['state_parity'] = state_parity(states['reference'], final)
                else: row['state_parity'] = state_parity(final, final)
                states.setdefault(name, final)
                rows[name].append(row)
                print('LOCAL_SPEED '+json.dumps(dict(backend=name, **row)), flush=True)
            except torch.cuda.OutOfMemoryError:
                rows[name].append(dict(pass_index=pass_index+1, error='CUDA OOM; not a passing candidate'))
                adam.zero_grad(set_to_none=True); torch.cuda.empty_cache()
    base = statistics.mean([r['seconds_per_window'] for r in rows['reference']])
    summary = {}
    for name, values in rows.items():
        valid = [r for r in values if 'seconds_per_window' in r]
        sec = statistics.mean([r['seconds_per_window'] for r in valid]) if len(valid) == 2 else None
        summary[name] = dict(seconds_per_window=sec, speedup=base/sec if sec else None,
            final_state_pass=all(r['state_parity']['numerical_parity'] for r in valid) and len(valid) == 2,
            faster_each_pass=bool(sec and all(v['seconds_per_window'] < r['seconds_per_window']
                for v, r in zip(valid, rows['reference']))), passes=values, single_step=audit_rows.get(name))
    return dict(summary=summary, no_automatic_promotion=True, scientific_updates_published=0)


def synthetic(device, shape, chunk, warmup, repeats):
    model = SemanticRefiner(z_bins=shape[2], chunk_size=chunk).to(device)
    with torch.no_grad(): model.readout[-1].weight.normal_(std=.025)  # avoid zero-head hiding differences
    adam = torch.optim.AdamW(model.parameters(), lr=1e-3)
    memories, priors, labels = [], [], []
    for _ in range(4):
        memories.append(torch.randn(6, 16, shape[0]//2, shape[1]//2, shape[2], device=device).requires_grad_())
        ids = torch.randint(18, (6, math.prod(shape)), device=device)
        ids[torch.rand(ids.shape, device=device) < .65] = 17
        prior = torch.nn.functional.one_hot(ids, 18).permute(0, 2, 1).float()*.9+.1/18
        priors.append(prior.requires_grad_())
        labels.append(ids.cpu().numpy().reshape(6, *shape))
    step = lambda name, _: synthetic_step(model, adam, memories, priors, labels, shape, name)
    audit = lambda name: synthetic_step(model, adam, memories, priors, labels, shape, name, audit=True)
    report = measure(model, adam, step, warmup=warmup, repeats=repeats, device=device, audit=audit)
    report.update(scope='synthetic Refiner + full6 CE/Lovasz + backward/Adam; NO source projection/encoder/I/O',
                  shape=list(shape), chunk_size=chunk, refiner_width=32, memory_width=16,
                  weights='random nonzero-head synthetic; NOT a trained checkpoint')
    return report


def tiny_replay(device, warmup, repeats):
    # Deliberately use repository test fixtures, NOT pretend they are nuScenes.
    from test_full_gt_training import fixture
    model, adam, provider, row = fixture()
    model.to(device); provider.device = device
    with torch.no_grad(): model.columns.readout[-1].weight.normal_(std=.025)
    rows = [copy.deepcopy(row) for _ in range(4)]
    report = measure(model, adam,
        lambda name, i: replay_update(model, adam, provider, rows, name, update=i+1, total_updates=warmup+repeats),
        warmup=warmup, repeats=repeats, device=device)
    report.update(scope='tiny 8x8x4 fixture, FULL shared encoder/STWM/geometry/refiner/loss/backward/Adam',
                  quality_evaluation=False, actual_nuscenes=False, current_L40S_performance=False)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    p.add_argument('--xy', type=int, default=64)
    p.add_argument('--z', type=int, default=16)
    p.add_argument('--chunk', type=int, default=8192)
    p.add_argument('--warmup', type=int, default=1)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--device', default='cuda')
    p.add_argument('--tiny-replay', action='store_true')
    a = p.parse_args(argv)
    if a.xy < 4 or a.xy % 2 or min(a.z, a.chunk, a.warmup, a.repeats) < 1:
        raise ValueError('positive budgets, even XY required')
    out = Path(a.out)
    if out.exists(): raise RuntimeError('NEW report file required')
    device = torch.device(a.device)
    torch.set_num_threads(1); torch.manual_seed(20261013)
    if device.type == 'cuda': torch.cuda.manual_seed_all(20261013)
    result = dict(protocol='local_refiner_b4_readout_ab_v1', torch_version=str(torch.__version__),
        cuda_version=torch.version.cuda, device=str(device), seed=20261013,
        code_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in Path(__file__).parent.glob('*.py')},
        GPU=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        device_total_mib=torch.cuda.get_device_properties(device).total_memory/2**20 if device.type == 'cuda' else None)
    result['synthetic'] = synthetic(device, (a.xy, a.xy, a.z), a.chunk, a.warmup, a.repeats)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    if a.tiny_replay:
        result['tiny_replay'] = tiny_replay(device, a.warmup, a.repeats)
        out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('REPORT '+str(out.resolve()), flush=True)
    return result


if __name__ == '__main__': main()
