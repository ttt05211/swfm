"""Same-input full-grid loss A/B timing; NOT end-to-end training or FPS."""
import argparse
import json
import time
import numpy as np
import torch
from ..soft_geometry import occupancy_loss
from .loss import prepare_target, occupancy_loss_prepared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--voxels', type=int, default=640000)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--distribution', choices=('sparse', 'dense'), default='sparse')
    args = parser.parse_args()
    if min(args.voxels, args.repeats) < 1: parser.error('positive budgets required')
    torch.set_num_threads(1); torch.manual_seed(373)
    device = torch.device(args.device)
    if args.distribution == 'sparse':
        probability = torch.zeros(18, args.voxels, device=device)
        probability.scatter_(0, torch.randint(18, (1, args.voxels), device=device), 1.)
        probability = probability*.65+probability.roll(1, 0)*.35
    else:
        probability = torch.rand(18, args.voxels, device=device)
        probability = probability/probability.sum(0)
    probability.requires_grad_()
    labels_cpu = (np.arange(args.voxels) % 18).astype(np.uint8)
    target = torch.as_tensor(labels_cpu, device=device)
    sync = (lambda: torch.cuda.synchronize(device)) if device.type == 'cuda' else (lambda: None)
    def trial(backend):
        probability.grad = None
        sync(); start = time.perf_counter()
        if backend != 'current_fused':
            # INCLUDE target preparation and upload in timed candidate.
            loss = occupancy_loss_prepared(probability, prepare_target(labels_cpu, device),
                                          sparse=backend == 'prepared_sparse')[0]
        else:
            loss = occupancy_loss(probability, target, class_ids=list(range(17)))[0]
        loss.backward(); sync()
        return time.perf_counter()-start, float(loss.detach()), probability.grad.detach().clone()
    backends = ['current_fused', 'prepared_combined', 'prepared_sparse']
    baseline = trial(backends[0])
    for backend in backends[1:]:
        candidate = trial(backend)
        torch.testing.assert_close(torch.tensor(baseline[1]), torch.tensor(candidate[1]), rtol=2e-6, atol=2e-7)
        torch.testing.assert_close(baseline[2], candidate[2], rtol=3e-5, atol=2e-7)
    timings = {name: [] for name in backends}
    paired = []
    for iteration in range(args.repeats*2):
        measurements = {}
        for name in (backends if iteration % 2 == 0 else backends[::-1]):
            elapsed, _, _ = trial(name)
            timings[name].append(elapsed); measurements[name] = elapsed
        paired.append(measurements)
    means = {k: sum(v)/len(v) for k, v in timings.items()}
    print(json.dumps(dict(device=str(device), voxels=args.voxels, scope='full-grid loss forward/backward ONLY',
        distribution=args.distribution, same_values_and_gradients=True, target_prepare_included=True, paired_AB_BA=paired,
        seconds=means, speedup={name: means['current_fused']/means[name] for name in backends[1:]},
        no_training_checkpoint_or_optimizer_changes=True), indent=2))


if __name__ == '__main__': main()
