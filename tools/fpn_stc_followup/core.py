"""Memory weight temperature and conditional semantic bias, without hard rules."""
from contextlib import contextmanager
import numpy as np
import torch
from torch.nn import functional as F
from tools.fpn_unet_stc_age.processing import weight_history
from tools.source_scene_ego_conditioned.geometry import aligned_memory
from tools.shared_history_experiments.refine.geometry import TransportPrior, inverse_grid
from real_motion.metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS

FREE_BIAS = -1.


@contextmanager
def importance_temperature(columns, temperature):
    if temperature not in (1., 2., float('inf')): raise ValueError('fixed T1/T2/uniform only')
    def hook(module, inputs, value):
        return torch.zeros_like(value) if temperature == float('inf') else value/temperature
    handle = columns.importance.register_forward_hook(hook)
    try: yield
    finally: handle.remove()


@torch.no_grad()
def memory_logits(body, provider, row, fw, temperature):
    _, raw, _, history = row
    history, _ = weight_history(history, raw['_shared_source_track'].times)
    with importance_temperature(body.columns, temperature):
        memory = aligned_memory(body.columns, fw['lifted'], fw['output'], history, raw,
            fw['targets'], fw['yaws'], fw['poses'], provider.pcfg.grid)
    return body.columns(memory, fw['prior'], range(6), fw['shape'])


@torch.no_grad()
def temporal_prior_logits(body, provider, row, fw):
    """Causal age-weighted static probabilities, not hard majority/thresholds."""
    _, raw, _, _ = row
    renderer = TransportPrior(raw, fw['state'], provider.pcfg.grid, provider.device)
    labels = torch.as_tensor(raw['history_occ'], device=provider.device).long()
    known = torch.as_tensor(raw['history_observed'], device=provider.device).bool() & (labels < 18)
    poses = fw['poses'].new_tensor(np.asarray(raw['history_poses']))
    times = np.asarray(raw['_shared_source_track'].times)
    ages = np.exp2((times-times[-1])/.5)
    total = torch.zeros_like(renderer.static); mass = torch.zeros_like(total[:, :1])
    for frame in range(4):
        valid = known[frame] & (labels[frame] >= 0)
        # Same predefined static/source decomposition as TransportPrior. No
        # new class-dependent thresholds or treatment of particular classes.
        for cls in DYNAMIC_CLASS_IDS: valid &= labels[frame] != cls
        probability = F.one_hot(labels[frame].clamp(0, 17), 18).permute(3, 0, 1, 2).float()*valid[None]
        packed = torch.cat((probability, valid[None].float()))[None]
        sampling = inverse_grid(renderer.centers, torch.linalg.inv(poses[frame]) @ renderer.pose, renderer.grid)[None]
        warped = F.grid_sample(packed, sampling, align_corners=False, padding_mode='zeros')
        total += float(ages[frame])*warped[:, :18]; mass += float(ages[frame])*warped[:, 18:]
    renderer.static = torch.where(mass > 0, total/mass.clamp_min(1e-8), renderer.static)
    prior = renderer.render(fw['poses'], fw['targets'], fw['yaws'])
    return body.columns(fw['memory'], prior, range(6), fw['shape'])


@torch.no_grad()
def readouts(logits, shape, semantic_biases=None):
    z = logits[[1, 3, 5]]; score, classes = z[:, :17].max(1)
    occupied = score >= z[:, 17]+FREE_BIAS
    result = dict(original=torch.where(score >= z[:, 17], classes, 17),
        free_minus1=torch.where(occupied, classes, 17))
    for name, bias in (semantic_biases or {}).items():
        bias = torch.as_tensor(bias, device=z.device, dtype=z.dtype)
        if bias.shape != (17,): raise ValueError('17 shared semantic offsets required')
        new_class = (z[:, :17]+bias[None, :, None]).argmax(1)
        # Binary support remains EXACTLY the fixed FREE=-1 prediction. Hence
        # its binary IoU cannot change: only occupied semantic labels change.
        result[name] = torch.where(occupied, new_class, 17)
    return {n: v.reshape(3, *shape).byte().cpu().numpy() for n, v in result.items()}


@torch.no_grad()
def sample_fit(logits, target, *, seed, cap=128):
    """FIT32 ONLY: stratified samples with full-population importance weights."""
    z = logits[[1, 3, 5]].detach().cpu(); gt = torch.as_tensor(target.reshape(3, -1)).long()
    occupied = z[:, :17].max(1).values >= z[:, 17]+FREE_BIAS
    rng = np.random.default_rng(seed); samples = []
    for h in range(3):
        # True-free and predicted-free has zero semantic IoU contribution.
        support = occupied[h] | (gt[h] != 17)
        for cls in range(18):
            ids = torch.where(support & (gt[h] == cls))[0].numpy()
            if not len(ids): continue
            chosen = rng.choice(ids, min(cap, len(ids)), replace=False)
            samples.append(dict(logits=z[h, :17, chosen].T.numpy(), labels=gt[h, chosen].numpy(),
                occupied=occupied[h, chosen].numpy(), weights=np.full(len(chosen), len(ids)/len(chosen), np.float32)))
    if not samples: raise ValueError('empty fitting support')
    return {k: np.concatenate([r[k] for r in samples]) for k in samples[0]}


def fit_bias(samples, objective, *, steps=200):
    """Two fixed objectives, final step only; only 17 centered offsets train."""
    if objective not in ('ce', 'soft_iou'): raise ValueError('fixed calibration objective required')
    z = torch.as_tensor(samples['logits']).float(); labels = torch.as_tensor(samples['labels']).long()
    occupied = torch.as_tensor(samples['occupied']).float(); weights = torch.as_tensor(samples['weights']).float()
    target = F.one_hot(labels, 18)[:, :17].float()
    weights = weights/weights.sum(); valid = (labels < 17) & occupied.bool()
    bias = torch.zeros(17, requires_grad=True); optimizer = torch.optim.Adam([bias], lr=.05)
    trace = []
    for step in range(steps):
        centered = bias-bias.mean(); probability = (z+centered).softmax(1)*occupied[:, None]
        if objective == 'ce':
            loss = (F.cross_entropy((z+centered)[valid], labels[valid], reduction='none')*weights[valid]).sum()/weights[valid].sum()
        else:
            intersection = (probability*target*weights[:, None]).sum(0)
            union = ((probability+target-probability*target)*weights[:, None]).sum(0)
            present = union > 0
            loss = 1-(intersection[present]/union[present]).mean()
        loss = loss+.01*centered.square().mean()
        if not torch.isfinite(loss): raise ValueError('nonfinite calibration loss')
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if step in (0, steps-1): trace.append(dict(step=step+1, loss=float(loss.detach())))
    return (bias-bias.mean()).detach().tolist(), trace
