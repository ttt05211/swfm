"""Only U-Net GN affine adaptation, with historical static-consensus targets."""
from contextlib import contextmanager
import math
import numpy as np
import torch
from torch.nn import functional as F
from real_motion.metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS
from tools.shared_history_experiments.refine.geometry import centers, inverse_grid

STEPS = 200
LR = .001
PARAMETERS = 320


def norm_parameters(body):
    values = {}
    for name, layer in body.columns.named_modules():
        if isinstance(layer, torch.nn.GroupNorm):
            for field in ('weight', 'bias'):
                values[f'{name}.{field}'] = getattr(layer, field)
    if len(values) != 20 or sum(v.numel() for v in values.values()) != PARAMETERS:
        raise ValueError('exactly ten U-Net GroupNorm layers / 320 affine scalars required')
    return values


def norm_state(body):
    return {k: v.detach().cpu().clone() for k, v in norm_parameters(body).items()}


@torch.no_grad()
def install(body, state):
    params = norm_parameters(body)
    if set(state) != set(params): raise ValueError('GN patch names differ')
    for name, value in state.items():
        if value.shape != params[name].shape or not torch.isfinite(value).all():
            raise ValueError('invalid GN patch: '+name)
    for name, value in state.items(): params[name].copy_(value)


@contextmanager
def using_norm(body, state):
    original = norm_state(body)
    try:
        install(body, state)
        yield
    finally:
        install(body, original)


def split_keys(keys):
    keys = [list(k) for k in keys]
    if len(keys) != 64 or len({k[0] for k in keys}) != 64:
        raise ValueError('original64 must contain 64 distinct scenes; do not silently resplit')
    return keys[:32], keys[32:]


@torch.no_grad()
def consensus(raw, future_pose, grid, device):
    """No GT semantics/annotations; future_pose is the declared STC+GT input.

    Vote only on static semantics: three agreeing occupied observations, or
    four free observations. Dynamic classes and outside-grid samples abstain.
    """
    sem = torch.as_tensor(np.asarray(raw['history_occ']), device=device).long()
    valid = torch.as_tensor(np.asarray(raw['history_observed']), device=device).bool() & (sem >= 0) & (sem < 18)
    if sem.shape[0] != 4 or valid.shape != sem.shape: raise ValueError('four historical grids required')
    for cls in DYNAMIC_CLASS_IDS: valid &= sem != cls
    poses = torch.as_tensor(np.asarray(raw['history_poses']), device=device, dtype=torch.float64)
    future_pose = torch.as_tensor(future_pose, device=device, dtype=torch.float64)
    query = centers(grid, device)
    if tuple(query.shape[:3]) != tuple(sem.shape[1:]): raise ValueError('grid/history shapes differ')
    votes = torch.zeros(18, *sem.shape[1:], device=device)
    for frame in range(4):
        sampling = inverse_grid(query, torch.linalg.inv(poses[frame]) @ future_pose, grid)[None]
        labels = F.grid_sample(sem[frame][None, None].float(), sampling, mode='nearest',
            align_corners=False, padding_mode='zeros')[0, 0].long().clamp(0, 17)
        weight = F.grid_sample(valid[frame][None, None].float(), sampling, mode='nearest',
            align_corners=False, padding_mode='zeros')[0, 0]
        votes.scatter_add_(0, labels[None], weight[None])
    count, label = votes.max(0)
    good = torch.where(label == 17, count >= 4, count >= 3)
    return torch.where(good, label, 255).flatten()


def balanced_ids(labels, cap=512):
    chosen = []
    for cls in range(18):
        ids = (labels == cls).nonzero().flatten()
        if len(ids) > cap: ids = ids[torch.linspace(0, len(ids)-1, cap, device=ids.device).long()]
        chosen.append(ids)
    result = torch.cat(chosen)
    if not len(result): raise ValueError('no static historical-consensus targets; no automatic fallback')
    return result


def cache_example(fw, labels):
    ids = balanced_ids(labels)
    # Slice BEFORE copying: never retain six-frame storage or upstream graphs.
    return dict(memory=fw['memory'][:1].detach().cpu().clone(),
        prior=fw['prior'][:1].detach().cpu().clone(), shape=tuple(fw['shape']),
        ids=ids.cpu().clone(), labels=labels[ids].cpu().clone())


def fit(body, examples, *, resumed=None, stopped=lambda: False, publish=lambda _: None, steps=STEPS):
    """Bounded optimization; original model is restored even on interruption.

    Full-volume U-Net forward keeps spatial GroupNorm statistics unchanged.
    Only CE is sampled/class-balanced; no arbitrary crop or point-MLP shortcut.
    """
    if not examples: raise ValueError('empty fitting set')
    params = norm_parameters(body)
    original = norm_state(body)
    flags = [(p, p.requires_grad) for p in body.parameters()]
    modes = [(m, m.training) for m in body.modules()]
    device = next(body.parameters()).device
    optimizer = torch.optim.AdamW(list(params.values()), lr=LR, weight_decay=0.)
    update, losses = 0, []
    try:
        body.eval().requires_grad_(False)
        body.columns.train()  # same math, enables existing activation recomputation
        for value in params.values(): value.requires_grad_(True)
        if resumed:
            if resumed['steps'] != steps or not 0 <= resumed['update'] <= steps:
                raise ValueError('resume fixed budget differs')
            install(body, resumed['norm']); optimizer.load_state_dict(resumed['optimizer'])
            update, losses = resumed['update'], list(resumed['losses'])
            if len(losses) != update: raise ValueError('resume update/loss cursor differs')
        def state():
            return dict(update=update, steps=steps, norm=norm_state(body),
                optimizer=optimizer.state_dict(), losses=losses.copy())
        while update < steps and not stopped():
            entry = examples[update % len(examples)]
            lr = LR*(.01+.99*.5*(1+math.cos(math.pi*update/steps)))
            for group in optimizer.param_groups: group['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            logits = body.columns(entry['memory'].to(device), entry['prior'].to(device), [0], entry['shape'])
            loss = F.cross_entropy(logits[0, :, entry['ids'].to(device)].T, entry['labels'].to(device))
            if not torch.isfinite(loss): raise RuntimeError('nonfinite adaptation loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(list(params.values()), 1., error_if_nonfinite=True)
            optimizer.step(); update += 1
            losses.append(dict(update=update, loss=float(loss.detach()), lr=lr, grad_norm=float(norm)))
            del logits, loss
            if update % 20 == 0: publish(state())
        result = state(); publish(result)
        return result
    finally:
        install(body, original)
        for p, enabled in flags: p.requires_grad_(enabled)
        for module, training in modes: module.training = training
        for value in params.values(): value.grad = None
