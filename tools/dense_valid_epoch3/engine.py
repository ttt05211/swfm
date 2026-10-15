"""Identical H6 objective/reduction with a non-increasing continuation LR."""
import math
import time
import torch
from tools.shared_history_experiments.full_gt.engine import forward_window, semantic_loss


def tail_lrs(start_lrs, update, steps):
    if not 1 <= update <= steps or steps < 2: raise ValueError('continuation update outside fixed epoch')
    if any(not math.isfinite(v) or v <= 0 for v in start_lrs): raise ValueError('invalid source LR')
    factor = .1+.9*.5*(1+math.cos(math.pi*(update-1)/(steps-1)))
    return [v*factor for v in start_lrs]


def train_step(model, optimizer, provider, rows, context, *, global_windows, update, steps, start_lrs,
               forward=None, loss_fn=None):
    if not 1 <= global_windows <= 4: raise ValueError('invalid global batch')
    model.train(); optimizer.zero_grad(set_to_none=True)
    rates = tail_lrs(start_lrs, update, steps)
    if len(rates) != len(optimizer.param_groups): raise ValueError('optimizer groups changed')
    for group, rate in zip(optimizer.param_groups, rates): group['lr'] = rate
    begun = time.perf_counter(); totals = torch.zeros(3, device=provider.device, dtype=torch.float64)
    forward, loss_fn = forward or forward_window, loss_fn or semantic_loss
    for row in rows:
        fw = forward(model, provider, row)
        losses, ce, lovasz = loss_fn(fw['logits'], row[1]['future_gt_occ'])
        (losses.mean()/global_windows).backward()
        totals += torch.stack((ce.detach().mean(), lovasz.detach().mean(), losses.detach().mean())).double()/global_windows
        del fw, losses, ce, lovasz
    if context.stop(not bool(torch.isfinite(totals).all())): raise RuntimeError('nonfinite full6 objective')
    context.synchronize_gradients(model)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    values = context.sum(totals).cpu().tolist()
    if provider.device.type == 'cuda': torch.cuda.synchronize(provider.device)
    return dict(CE=values[0], Lovasz=values[1], loss=values[2], grad_norm=float(norm),
                lr=rates, local_windows=len(rows), seconds=time.perf_counter()-begun)
