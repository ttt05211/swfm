"""Fixed B4/H6 refinement and regular-lattice readout candidates.

Weights and objectives are unchanged. Floating point execution/gradients may
differ, especially BF16 accumulation; this is NOT a byte-parity claim.
"""
from contextlib import contextmanager
import math
import types
import torch
from torch.nn import functional as F

BACKENDS = ('reference', 'slice_queries', 'xy_upsample', 'volume_upsample',
            'packed_reference', 'packed_slice', 'packed_upsample', 'packed_volume')
SERVER_BACKENDS = ('reference', 'slice_queries', 'packed_reference', 'packed_slice')


def trunk(model, memory, prior, horizons, shape):
    if prior.shape != (len(horizons), 18, math.prod(shape)):
        raise ValueError('prior/full-grid/horizon shape mismatch')
    if tuple(memory.shape[2:]) != (shape[0]//2, shape[1]//2, shape[2]):
        raise ValueError('memory must halve XY and retain Z')
    coarse = F.avg_pool3d(prior.float().reshape(-1, 18, *shape), (2, 2, 1))
    with torch.autocast(device_type=memory.device.type, dtype=torch.bfloat16,
                        enabled=memory.device.type == 'cuda'):
        x = model.input(torch.cat((memory, coarse), 1))
        t = x.new_tensor([.5*(h+1)/3 for h in horizons])[:, None]
        conditions = model.time(t).reshape(len(horizons), 3, 2, model.width)
        for i, block in enumerate(model.blocks):
            scale, offset = conditions[:, i].unbind(1)
            x = block(x)*(1+scale[:, :, None, None, None])+offset[:, :, None, None, None]
    return x


def upsample_xy(x, shape, *, volume=False):
    """Regular fine voxel centres: bilinear XY, no Z interpolation/pooling."""
    if tuple(x.shape[2:]) != (shape[0]//2, shape[1]//2, shape[2]) or any(s % 2 for s in shape[:2]):
        raise ValueError('exact 2x XY lattice required')
    if volume:
        # Z size unchanged: same centres, no height pooling, avoids layout copies.
        return F.interpolate(x.float(), size=tuple(shape), mode='trilinear', align_corners=False).flatten(2)
    n, c, sx, sy, z = x.shape
    planes = x.float().permute(0, 4, 1, 2, 3).reshape(n*z, c, sx, sy)
    fine = F.interpolate(planes, size=tuple(shape[:2]), mode='bilinear', align_corners=False)
    return fine.reshape(n, z, c, *shape[:2]).permute(0, 2, 3, 4, 1).contiguous().flatten(2)


def regular_forward(model, memory, prior, horizons, shape, *, upsample=False, volume=False):
    """Keep original point MLP weight layouts and original chunk order."""
    shape = tuple(map(int, shape)); horizons = tuple(horizons)
    x = trunk(model, memory, prior, horizons, shape)
    fine = upsample_xy(x, shape, volume=volume) if upsample else None
    rows = []
    for start in range(0, prior.shape[-1], model.chunk_size):
        end = min(start+model.chunk_size, prior.shape[-1])
        ids = torch.arange(start, end, device=x.device)
        xyz = torch.stack((ids//(shape[1]*shape[2]), ids//shape[2] % shape[1], ids % shape[2]), -1).float()+.5
        if fine is None:
            normalized = xyz/xyz.new_tensor(shape)*2-1
            sampling = normalized[:, [2, 1, 0]][None, :, None, None].expand(len(horizons), -1, -1, -1, -1)
            # Per-chunk float casts intentionally retained in the slice-only arm.
            values = F.grid_sample(x.float(), sampling, align_corners=False,
                padding_mode='border')[:, :, :, 0, 0].transpose(1, 2)
        else:
            values = fine[:, :, start:end].transpose(1, 2)
        cell = xyz/xyz.new_tensor([2, 2, 1]); phase = cell-cell.floor()-.5
        inputs = torch.cat((values, prior[:, :, start:end].transpose(1, 2),
                            phase[None].expand(len(horizons), -1, -1)), -1)
        with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16,
                            enabled=x.device.type == 'cuda'):
            rows.append(model.readout(inputs).float().transpose(1, 2))
    return model.prior_logits(prior)+torch.cat(rows, -1)


def batch_forward(model, memories, priors, shape, backend):
    if backend not in BACKENDS or not memories or len(memories) != len(priors) or len(memories) > 4:
        raise ValueError('fixed B<=4 H6 candidate required')
    if any(m.shape[0] != 6 or p.shape[0] != 6 for m, p in zip(memories, priors)):
        raise ValueError('all SIX horizons required for each window')
    if backend.startswith('packed'):
        memory, prior = torch.cat(memories), torch.cat(priors)
        horizons = tuple(range(6))*len(memories)  # repeat times, not horizons 0..23
        if backend == 'packed_reference':
            logits = model(memory, prior, horizons, shape)
        elif backend == 'packed_slice':
            with replace_forward(model, lambda columns, m, p, h, s: regular_forward(columns, m, p, h, s)):
                logits = model(memory, prior, horizons, shape)
        else:
            logits = regular_forward(model, memory, prior, horizons, shape, upsample=True,
                                     volume=backend == 'packed_volume')
        return logits.split(6)
    if backend == 'reference': return tuple(model(m, p, range(6), shape) for m, p in zip(memories, priors))
    return tuple(regular_forward(model, m, p, range(6), shape,
                 upsample=backend in ('xy_upsample', 'volume_upsample'), volume=backend == 'volume_upsample')
                 for m, p in zip(memories, priors))


@contextmanager
def replace_forward(model, replacement):
    """Instance-only temporary hook; restore even when forward/backward fails."""
    old = model.__dict__.get('forward')
    model.forward = types.MethodType(replacement, model)
    try:
        yield
    finally:
        if old is None: del model.__dict__['forward']
        else: model.forward = old


def replay_update(model, adam, provider, rows, backend, *, update=1, total_updates=8, probe=False):
    """Complete GT network/geometry execution; callers discard temporary updates."""
    from tools.shared_history_experiments.full_gt import engine
    from collections import defaultdict
    import time
    import numpy as np
    if backend not in BACKENDS or not 0 < len(rows) <= 4: raise ValueError('fixed B4 required')
    if not backend.startswith('packed'):
        if backend == 'reference':
            return engine.train_arm(model, adam, provider, rows, 'gt', update=update,
                                    total_updates=total_updates, probe=probe)
        def forward(columns, memory, prior, horizons, shape):
            return regular_forward(columns, memory, prior, horizons, shape,
                upsample=backend in ('xy_upsample', 'volume_upsample'), volume=backend == 'volume_upsample')
        with replace_forward(model.columns, forward):
            return engine.train_arm(model, adam, provider, rows, 'gt', update=update,
                                    total_updates=total_updates, probe=probe)
    model.train(); adam.zero_grad(set_to_none=True)
    begun = time.perf_counter(); stages = defaultdict(float); links = defaultdict(float)
    factor = .01+.99*.5*(1+np.cos(np.pi*(update-1)/max(1, total_updates)))
    for group in adam.param_groups: group['lr'] = group['initial_lr']*factor
    # Prepare each window's geometry with its own trainable F/STWM graph.
    tick = time.perf_counter()
    with replace_forward(model.columns, lambda *a, **k: None):
        prepared = [engine.forward_window(model, provider, row) for row in rows]
    stages['shared_encoder_motion_projection_prepare'] = time.perf_counter()-tick
    for fw in prepared:
        for key, value in fw['stages'].items(): stages['detail/'+key] += value
    shape = prepared[0]['shape']
    if any(f['shape'] != shape for f in prepared): raise ValueError('scene grid shapes differ')
    tick = time.perf_counter()
    logits = batch_forward(model.columns, [f['memory'] for f in prepared],
                           [f['prior'] for f in prepared], shape, backend)
    stages['refiner_b4_h6'] = time.perf_counter()-tick
    roots, ce_terms, lovasz_terms = [], [], []
    tick = time.perf_counter()
    for z, row, fw in zip(logits, rows, prepared):
        losses, ce, lovasz = engine.semantic_loss(z, row[1]['future_gt_occ'])
        roots.append(losses.mean()/len(rows))  # original per-window/per-horizon normalization
        ce_terms.append(ce.detach()); lovasz_terms.append(lovasz.detach())
        if probe:
            tensors, names = [fw['field'], fw['memory']], ['occupancy_to_shared_history', 'occupancy_to_memory']
            if len(fw['targets']):
                tensors += [fw['output']['residual_xy_m'], fw['output']['yaw_delta_rad']]
                names += ['occupancy_to_xy', 'occupancy_to_yaw']
            gradients = torch.autograd.grad(losses.mean(), tensors, retain_graph=True, allow_unused=True)
            for name, gradient in zip(names, gradients):
                links[name] += 0. if gradient is None else float(gradient.float().norm())
    if not bool(torch.isfinite(torch.stack(roots)).all()): raise RuntimeError('nonfinite packed objective')
    stages['full6_ce_lovasz'] = time.perf_counter()-tick
    tick = time.perf_counter()
    torch.autograd.backward(roots)
    stages['full_network_backward'] = time.perf_counter()-tick
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
    tick = time.perf_counter()
    adam.step(); adam.zero_grad(set_to_none=True)
    if provider.device.type == 'cuda': torch.cuda.synchronize(provider.device)
    stages['optimizer_finish'] = time.perf_counter()-tick
    return dict(CE=float(torch.cat(ce_terms).mean()), Lovasz=float(torch.cat(lovasz_terms).mean()),
                CE_Lovasz=float(torch.stack(roots).sum().detach()), windows=len(rows),
                sources=sum(len(f['targets']) for f in prepared), seconds=time.perf_counter()-begun,
                grad_norm=float(norm), gradient_links=dict(links), stages_seconds=dict(stages),
                auxiliary_computed=False, training_backend=backend)
