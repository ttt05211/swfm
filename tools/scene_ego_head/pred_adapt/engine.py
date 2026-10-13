from collections import defaultdict
import time
import numpy as np
import torch
from tools.scene_ego_head.integration import forward_pred_window
from tools.shared_history_experiments.refine.engine import semantic_loss
from tools.joint_occ_experiments.model import ego_geometry_loss


def train_arm(model, optimizer, provider, rows, mode, *, update, total_updates,
              object_weight=1., ego_weight=1., lovasz_weight=1., probe=False, optimized=True):
    if mode != 'pred' or not optimized or not rows: raise ValueError('continuous internal Pred required')
    model.train(); optimizer.zero_grad(set_to_none=True)
    factor = .01+.99*.5*(1+np.cos(np.pi*(update-1)/max(1, total_updates)))
    for group in optimizer.param_groups: group['lr'] = group['initial_lr']*factor
    stages = defaultdict(float); terms = defaultdict(list); links = {}; sources = 0; tick = time.perf_counter()
    for row in rows:
        start = time.perf_counter(); fw = forward_pred_window(model, provider, row)
        stages['frozen_history_motion_pred_projection_refine'] += time.perf_counter()-start
        for k, v in fw['stages'].items(): stages['detail/'+k] += v
        start = time.perf_counter()
        losses, ce, lovasz = semantic_loss(fw['logits'], row[1]['future_gt_occ'], lovasz_weight)
        direct = ego_geometry_loss(fw['ego']['se2'], np.asarray(row[2][2])[None])
        loss = (losses.mean()+ego_weight*direct)/len(rows)
        if not bool(torch.isfinite(loss)): raise RuntimeError('nonfinite Pred adaptation loss')
        if probe or update == 2:
            g = torch.autograd.grad(losses.mean(), fw['ego']['se2'], retain_graph=True, allow_unused=True)[0]
            links['occupancy_to_scene_ego'] = links.get('occupancy_to_scene_ego', 0.)+(float(g.float().norm()) if g is not None else 0.)
        stages['full6_ce_lovasz_and_ego'] += time.perf_counter()-start
        start = time.perf_counter(); loss.backward(); stages['refiner_and_ego_backward'] += time.perf_counter()-start
        terms['CE'].extend(ce.detach().unbind()); terms['Lovasz'].extend(lovasz.detach().unbind())
        terms['ego_geometry'].append(direct.detach()); sources += len(fw['targets'])
    if any(p.grad is not None for p in model.transport.parameters()): raise RuntimeError('frozen Encoder/STWM received gradients')
    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5., error_if_nonfinite=True)
    start = time.perf_counter(); optimizer.step()
    if provider.device.type == 'cuda': torch.cuda.synchronize(provider.device)
    stages['optimizer_finish'] += time.perf_counter()-start
    result = {k: float(torch.stack(v).mean()) for k, v in terms.items()}
    result.update(mode=mode, windows=len(rows), sources=sources, seconds=time.perf_counter()-tick,
        stages_seconds=dict(stages), gradient_links=links, grad_norm=float(norm),
        source_auxiliary_computed=False, training_backend='GTbody_RANDOMego_refiner_adapt_full6',
        peak_memory_mib=torch.cuda.max_memory_allocated(provider.device)/2**20 if provider.device.type == 'cuda' else 0.)
    optimizer.zero_grad(set_to_none=True)
    return result
