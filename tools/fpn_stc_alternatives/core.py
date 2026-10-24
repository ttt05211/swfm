"""Small fixed mechanisms; no edits to production weights or old experiments."""
from contextlib import contextmanager
import numpy as np
import torch
from torch.nn import functional as F


def embedding_table(confusion, original, strength):
    counts = torch.as_tensor(confusion, dtype=torch.float64, device=original.device)
    if counts.shape != (18, 18) or bool((counts < 0).any()) or strength not in (.5, 1.):
        raise ValueError('18x18 STC-row/GT-column counts and fixed half/full strength required')
    mass = counts.sum(1, keepdim=True)
    posterior = counts/mass.clamp_min(1)
    expected = posterior.to(original.dtype) @ original[:18]
    expected = torch.where(mass > 0, expected, original[:18])
    return torch.cat((original[:18]+strength*(expected-original[:18]), original[18:]), 0)


@contextmanager
def embedding_override(encoder, table):
    table = table.to(encoder.semantic.weight.device, encoder.semantic.weight.dtype)
    handle = encoder.semantic.register_forward_hook(lambda module, args, output: F.embedding(args[0], table))
    try: yield
    finally: handle.remove()


@contextmanager
def capture_readout(columns):
    values = []
    handle = columns.head.register_forward_pre_hook(lambda module, args: values.append(args[0].detach().float().flatten(2).cpu()))
    try: yield values
    finally: handle.remove()


@contextmanager
def readout_patch(columns, patch):
    weight = torch.as_tensor(patch['weight'], device=columns.head.weight.device, dtype=torch.float32).reshape(18, 8, 1, 1, 1)
    bias = torch.as_tensor(patch['bias'], device=weight.device, dtype=torch.float32)
    def hook(module, args, output):
        with torch.autocast(device_type=weight.device.type, enabled=False):
            return output.float()+F.conv3d(args[0].float(), weight, bias)
    handle = columns.head.register_forward_hook(hook)
    try: yield
    finally: handle.remove()


@torch.no_grad()
def sample_head(logits, activations, targets, *, seed, cap=128):
    z = logits[[1, 3, 5]].cpu(); x = torch.cat(activations)[[1, 3, 5]]
    gt = torch.as_tensor(targets.reshape(3, -1)).long()
    if x.shape != (3, 8, z.shape[-1]): raise ValueError('complete six-frame 8-channel head inputs required')
    rng = np.random.default_rng(seed); rows = []
    for h in range(3):
        for cls in range(18):
            ids = torch.where(gt[h] == cls)[0].numpy()
            if not len(ids): continue
            chosen = rng.choice(ids, min(cap, len(ids)), replace=False)
            rows.append(dict(logits=z[h, :, chosen].T.numpy(), features=x[h, :, chosen].T.numpy(),
                labels=gt[h, chosen].numpy(), weights=np.full(len(chosen), len(ids)/len(chosen), np.float32)))
    return {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}


def fit_head(samples, steps=200, conditional=False):
    """162 additive readout scalars, supervised FIT32 CE+soft-IoU, fixed final."""
    x = torch.as_tensor(samples['features']).float(); z = torch.as_tensor(samples['logits']).float().clone()
    z[:, 17] -= 1.  # The fixed existing calibration, never a newly selected bias.
    labels = torch.as_tensor(samples['labels']).long(); target = F.one_hot(labels, 18).float()
    weights = torch.as_tensor(samples['weights']).float(); weights = weights/weights.sum()
    mean = (x*weights[:, None]).sum(0); std = ((x-mean).square()*weights[:, None]).sum(0).sqrt().clamp_min(1e-6)
    x = (x-mean)/std
    classes = 17 if conditional else 18
    gate = (z[:, :17].max(1).values >= z[:, 17]).float()
    valid = (labels < 17) & gate.bool()
    weight = torch.zeros(classes, 8, requires_grad=True); bias = torch.zeros(classes, requires_grad=True)
    optimizer = torch.optim.Adam([weight, bias], lr=.02); trace = []
    for step in range(steps):
        calibrated = z[:, :classes]+F.linear(x, weight, bias)
        probability = calibrated.softmax(1)*(gate[:, None] if conditional else 1.)
        if conditional:
            ce = (F.cross_entropy(calibrated[valid], labels[valid], reduction='none')*weights[valid]).sum()/weights[valid].sum()
        else: ce = (F.cross_entropy(calibrated, labels, reduction='none')*weights).sum()
        truth = target[:, :classes]
        intersection = (probability*truth*weights[:, None]).sum(0)[:17]
        union = ((probability+truth-probability*truth)*weights[:, None]).sum(0)[:17]
        loss = ce+1-(intersection/union.clamp_min(1e-10)).mean()+.01*(weight.square().mean()+bias.square().mean())
        if not torch.isfinite(loss): raise ValueError('nonfinite readout calibration loss')
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if step in (0, steps-1): trace.append(dict(step=step+1, loss=float(loss.detach())))
    actual = weight.detach()/std; actual_bias = bias.detach()-actual@mean
    if conditional:
        actual = torch.cat((actual, torch.zeros(1, 8))); actual_bias = torch.cat((actual_bias, torch.zeros(1)))
    return dict(weight=actual.tolist(), bias=actual_bias.tolist(), parameters=classes*9, conditional=conditional,
        steps=steps, trace=trace, feature_mean=mean.tolist(), feature_std=std.tolist())


@torch.no_grad()
def tiled_logits(columns, memory, prior, shape, side=128):
    """Four overlapping XY tiles with positive Gaussian blending, Z retained."""
    sx, sy, sz = shape
    if side % 4 or sx % 4 or sy % 4 or side > min(sx, sy): raise ValueError('aligned tile/grid sizes required')
    if side == sx == sy: return columns(memory, prior, range(6), shape)
    if max(sx, sy) > 2*side: raise ValueError('fixed two tiles per XY axis must cover the grid')
    starts_x = [0, sx-side]; starts_y = [0, sy-side]
    if any(at % 2 for at in starts_x+starts_y): raise ValueError('Memory lattice alignment required')
    probability = prior.reshape(6, 18, *shape)
    total = torch.zeros_like(probability); mass = torch.zeros_like(probability[:1, :1])
    axis = torch.arange(side, device=prior.device, dtype=torch.float32)-(side-1)/2
    weight = torch.exp(-.5*((axis[:, None]/(side/8)).square()+(axis[None, :]/(side/8)).square()))
    weight = weight[None, None, :, :, None]
    for x in starts_x:
        for y in starts_y:
            p = probability[:, :, x:x+side, y:y+side].flatten(2)
            m = memory[:, :, x//2:(x+side)//2, y//2:(y+side)//2]
            z = columns(m, p, range(6), (side, side, sz)).reshape(6, 18, side, side, sz)
            total[:, :, x:x+side, y:y+side] += z*weight
            mass[:, :, x:x+side, y:y+side] += weight
    if not bool((mass > 0).all()): raise ValueError('uncovered tiled output')
    return (total/mass).flatten(2)
