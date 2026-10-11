"""Original per-window forward/shared backward, with TWO renderer horizons."""
from collections import defaultdict
import time
import numpy as np
import torch
from ..engine import forward_window
from ..fast_autograd import RenderGradientBridge
from ..model import source_auxiliary, ego_geometry_loss
from ..batching.geometry import GroupedScene, grouped_loss


def train_arm(model, optimizer, provider, rows, mode, *, update, total_updates,
              object_weight=1., ego_weight=1., lovasz_weight=1., probe=False, optimized=True):
    if mode != 'gt' or not optimized:
        raise ValueError('two-horizon execution supports original fused GT only')
    if not rows:
        raise ValueError('nonempty window batch required')
    model.train()
    optimizer.zero_grad(set_to_none=True)
    factor = .01+.99*.5*(1+np.cos(np.pi*(update-1)/max(1, total_updates)))
    for group in optimizer.param_groups: group['lr'] = group['initial_lr']*factor
    begun = time.perf_counter()
    stages, terms, links = defaultdict(float), defaultdict(list), defaultdict(float)
    sources = canonical = 0
    ramp = .1+.9*min(1., (update-1)/max(1., total_updates/5))
    for row in rows:
        tick = time.perf_counter()
        # SAME single-window WM, ego and CCR shapes/order as the old trainer.
        fw = forward_window(model, provider, row, mode, ego_gradient=ramp)
        scene = GroupedScene(fw['prep'], provider.pcfg.grid, provider.device)
        bridge = RenderGradientBridge(fw)
        stages['motion_ego_projection_head'] += time.perf_counter()-tick
        sources += len(row[0]['features'])
        canonical += len(fw['evidence'])
        # Targets are accessed only after all forward/candidate construction.
        labels = np.asarray(row[1]['future_gt_occ'])
        for first in (0, 2, 4):
            horizons = [first, first+1]
            tick = time.perf_counter()
            probability = scene.render_group(horizons, **bridge.leaves,
                evidence=fw['evidence'], plan=fw['plan'])
            losses, ce, lovasz = grouped_loss(probability, labels[horizons], lovasz_weight=lovasz_weight)
            if not bool(torch.isfinite(losses).all()):
                raise RuntimeError('nonfinite final occupancy loss')
            if probe and first == 0 and losses.requires_grad:
                # Probe the SAME 1s loss, not the sum of 0.5s and 1s.
                tensors, names = [fw['actions']], ['occupancy_to_actions']
                if fw['output']['residual_xy_m'].requires_grad:
                    tensors += [fw['output']['residual_xy_m'], fw['output']['yaw_delta_rad']]
                    names += ['occupancy_to_xy', 'occupancy_to_yaw']
                gradients = bridge.probe(losses[1], tensors)
                for name, gradient in zip(names, gradients):
                    links[name] += 0. if gradient is None else float(gradient.detach().float().norm())
            stages['soft_render_full_ce_lovasz'] += time.perf_counter()-tick
            tick = time.perf_counter()
            # EACH horizon retains its own full-grid/class mean and old scale.
            if losses.requires_grad: (losses/(len(rows)*6)).sum().backward()
            stages['occupancy_backward'] += time.perf_counter()-tick
            terms['CE'].extend(ce.detach().unbind())
            terms['Lovasz'].extend(lovasz.detach().unbind())
            del probability, losses, ce, lovasz
        tick = time.perf_counter()
        geometry, existence = source_auxiliary(fw['output'], row[0], fw['prep'].state)
        ego_loss = ego_geometry_loss(fw['ego'][None], np.asarray(fw['ego_target'])[None])
        auxiliary = object_weight*(geometry+existence)+ego_weight*ego_loss
        if not bool(torch.isfinite(auxiliary)): raise RuntimeError('nonfinite direct pose loss')
        # SAME per-window shared backward, completed BEFORE next forward.
        bridge.backward_shared(auxiliary/len(rows))
        terms['source_geometry'].append(geometry.detach())
        terms['existence'].append(existence.detach())
        terms['ego_geometry'].append(ego_loss.detach())
        stages['shared_network_and_direct_geometry_backward'] += time.perf_counter()-tick
        del fw, scene, labels, geometry, existence, ego_loss, auxiliary, bridge
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
    tick = time.perf_counter()
    optimizer.step()
    if provider.device.type == 'cuda': torch.cuda.synchronize(provider.device)
    stages['optimizer_finish'] += time.perf_counter()-tick
    result = {key: float(torch.stack(value).mean()) for key, value in terms.items()}
    result.update(mode=mode, windows=len(rows), sources=sources, canonical_points=canonical,
        seconds=time.perf_counter()-begun, stages_seconds=dict(stages), grad_norm=float(norm),
        ego_occupancy_gradient_scale=ramp, gradient_links=dict(links),
        training_backend='sequential_network_two_horizons',
        peak_memory_mib=torch.cuda.max_memory_allocated(provider.device)/2**20 if provider.device.type == 'cuda' else 0.)
    optimizer.zero_grad(set_to_none=True)
    return result
