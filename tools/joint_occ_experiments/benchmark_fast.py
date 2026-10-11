"""Synthetic same-input training-kernel timing; NOT full training or FPS."""
import argparse
import json
import time
import torch
from .fast_autograd import compose_sparse
from .soft_geometry import sparse_layer, occupancy_loss


def reference(background, operations, alphas):
    result = background.clone()
    for (kind, ids, cls, destination), alpha in zip(operations, alphas):
        if kind == 'overlay':
            result = sparse_layer(result, ids, alpha, cls)
        else:
            local = result[:, ids].clone()
            amount = local[cls].clone()*alpha
            local[cls] = local[cls]-amount
            result[:, ids] = local.scatter_add(0, destination[None], amount[None])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--voxels', type=int, default=640000)
    parser.add_argument('--operations', type=int, default=24)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if min(args.voxels, args.operations, args.repeats) < 1: parser.error('positive budgets required')
    device = torch.device(args.device)
    torch.set_num_threads(1); torch.manual_seed(28)
    probability = torch.rand(18, args.voxels, device=device)
    probability = (probability/probability.sum(0)).requires_grad_()
    labels = torch.arange(args.voxels, device=device) % 18
    operations, alphas = [], []
    for i in range(args.operations):
        ids = torch.randperm(args.voxels, device=device)[:min(args.voxels, 1024)]
        kind = 'remove' if i % 3 == 0 else 'overlay'
        destination = torch.full_like(ids, 17) if kind == 'remove' else None
        operations.append((kind, ids, i % 17, destination))
        alphas.append(torch.rand(len(ids), device=device).requires_grad_())
    sync = (lambda: torch.cuda.synchronize(device)) if device.type == 'cuda' else (lambda: None)
    timing = {}
    for name, render in [('reference', reference), ('fused', compose_sparse)]:
        times = []
        for trial in range(args.repeats+1):
            probability.grad = None
            for alpha in alphas: alpha.grad = None
            sync(); start = time.perf_counter()
            loss, _, _ = occupancy_loss(render(probability, operations, alphas), labels,
                                         class_ids=list(range(17)), reference=name == 'reference')
            loss.backward(); sync()
            elapsed = time.perf_counter()-start
            if trial: times.append(elapsed)
        timing[name] = sum(times)/len(times)
    print(json.dumps(dict(scope='synthetic full-grid composition+CE+Lovasz forward/backward ONLY',
        device=str(device), voxels=args.voxels, operations=args.operations, seconds=timing,
        speedup=timing['reference']/timing['fused'],
        excludes='WM/CCR forward/backward, geometry, data, inference; NOT server training ETA/FPS'), indent=2))


if __name__ == '__main__': main()
