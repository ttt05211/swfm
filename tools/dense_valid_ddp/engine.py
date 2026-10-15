"""Global-window mean, gradient SUM before clipping; never average rank means."""
import time
import numpy as np
import torch
from tools.shared_history_experiments.full_gt import engine as gt
from tools.shared_history_experiments.refine.history import ResidualHistoryInputs
from tools.history_validity.data import policy_raw
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import Metrics


class DenseTrainHistory(ResidualHistoryInputs):
    def build(self, raw):
        return super().build(policy_raw(raw, 'dense_valid'))


def dense_rows(rows):
    return [(rec, policy_raw(raw, 'dense_valid'), ego, history) for rec, raw, ego, history in rows]


def train_step(model, optimizer, provider, rows, context, *, global_windows, update, total_updates,
               forward=None, loss_fn=None):
    """Empty rank participates; globally unused parameters retain grad=None."""
    if not 1 <= global_windows <= 4 or not 1 <= update <= total_updates:
        raise ValueError('invalid global batch/update')
    forward, loss_fn = forward or gt.forward_window, loss_fn or gt.semantic_loss
    model.train(); optimizer.zero_grad(set_to_none=True)
    factor = .01+.99*.5*(1+np.cos(np.pi*(update-1)/total_updates))
    for group in optimizer.param_groups: group['lr'] = group['initial_lr']*factor
    begun = time.perf_counter()
    # Keep diagnostics on device until the collective, rather than .item() per horizon.
    totals = torch.zeros(3, device=provider.device, dtype=torch.float64)
    for row in rows:
        fw = forward(model, provider, row)
        losses, ce, lovasz = loss_fn(fw['logits'], row[1]['future_gt_occ'])
        (losses.mean()/global_windows).backward()
        totals += torch.stack((ce.detach().mean(), lovasz.detach().mean(),
                               losses.detach().mean())).double()/global_windows
        del fw, losses, ce, lovasz
    # A failure on any rank must fail ALL ranks before clipping/Adam.
    if context.stop(not bool(torch.isfinite(totals).all())):
        raise RuntimeError('nonfinite full-six objective; no optimizer update published')
    context.synchronize_gradients(model)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    totals = context.sum(totals).cpu().tolist()
    if provider.device.type == 'cuda': torch.cuda.synchronize(provider.device)
    return dict(CE=totals[0], Lovasz=totals[1], loss=totals[2], grad_norm=float(norm),
                local_windows=len(rows), seconds=time.perf_counter()-begun)


def merge_evaluation(states, expected_keys, *, protocol=None):
    """SUM integer counts first, compute IoUs afterwards (not mean rank IoUs)."""
    from tools.perception_robustness.evaluation import ROUTES, PROTOCOL
    protocol = protocol or PROTOCOL
    seen = []
    metrics = {name: Metrics() for name in ROUTES}
    for state in states:
        if state['protocol'] != protocol or state['windows'] != len(state['keys']):
            raise RuntimeError('incomplete/invalid distributed evaluation state')
        seen += list(map(tuple, state['keys']))
        if set(state['metrics']) != set(metrics): raise RuntimeError('metric routes differ')
        for name, fields in state['metrics'].items():
            if set(fields) != set(vars(metrics[name])): raise RuntimeError('metric fields differ')
            for field, value in fields.items():
                counts = np.asarray(value)
                target = getattr(metrics[name], field)
                if counts.dtype.kind not in 'iu' or counts.shape != target.shape or np.any(counts < 0):
                    raise RuntimeError('invalid distributed integer counts')
                target[:] += counts
    expected = list(map(tuple, expected_keys))
    if len(seen) != len(expected) or len(set(seen)) != len(seen) or set(seen) != set(expected):
        raise RuntimeError('distributed evaluation dropped/duplicated windows')
    return dict(windows=len(seen), evaluation_keys=[list(k) for k in expected],
                variants={'gt': {n: m.compute() for n, m in metrics.items()}},
                scoring='full grid; OCC lidar-known / STC dense-valid; GT future ego',
                reduction='integer confusion counts SUM before IoU/mIoU')
