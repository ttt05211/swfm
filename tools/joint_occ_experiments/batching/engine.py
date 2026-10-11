"""Same global window/source batch and Adam step, genuinely packed networks."""
from collections import defaultdict
import time
import numpy as np
import torch
from ..data import prepare_projection
from ..model import source_auxiliary, ego_geometry_loss
from ..soft_geometry import source_targets, effective_yaw, continuous_context
from ..fast_autograd import RenderGradientBridge
from .geometry import GroupedScene, grouped_loss


def forward_batch(model, provider, rows, *, projection_pool=None):
    if not 1 <= len(rows) <= 4: raise ValueError('keep original global window batch <=4')
    device = provider.device
    keys = ('features', 'local_semantic_tube', 'kta_displacement_xy_m', 'frame_motion_features', 'target_source_mask_tube')
    packed = {key: torch.cat([torch.as_tensor(row[0][key]) for row in rows]) for key in keys}
    lengths = [len(row[0]['features']) for row in rows]
    output = model.motion(packed, device)
    split = {key: value.split(lengths) for key, value in output.items()}
    outputs = [{key: value[i] for key, value in split.items()} for i in range(len(rows))]
    ego = model.ego(torch.as_tensor(np.stack([row[2][0] for row in rows]), device=device),
                    torch.as_tensor(np.stack([row[2][1] for row in rows]), device=device))['se2']
    # ONE readback, only the two fields needed by CPU hard ownership geometry.
    cpu_output = {key: output[key].detach().float().cpu().split(lengths)
                  for key in ('residual_xy_m', 'yaw_delta_rad')}
    def prepare(i):
        record, raw, _, history = rows[i]
        diagnostic = {key: value[i] for key, value in cpu_output.items()}
        return prepare_projection(provider, record, raw, diagnostic, np.asarray(raw['future_poses']), history, use_gt_cache=True)
    prepared = list(projection_pool.map(prepare, range(len(rows)))) if projection_pool else [prepare(i) for i in range(len(rows))]
    forwards, actors, contexts = [], [], []
    offset = 0
    for i, (row, result) in enumerate(zip(rows, prepared)):
        prep, evidence, plan = result
        prep.outputs = outputs[i]
        poses = torch.as_tensor(np.asarray(row[1]['future_poses']), device=device, dtype=torch.float64)
        targets = source_targets(outputs[i], row[0], prep.state['current_pose'], prep.state['source_z_t0'])
        yaws = effective_yaw(outputs[i], [int(c['class_id']) for c in prep.state['current']])
        mapped, context = continuous_context(evidence, plan, prep.state, targets, yaws, poses, provider.pcfg.grid)
        local_actor = np.asarray(evidence.actor)
        actors.append(np.where(local_actor >= 0, local_actor+offset, local_actor))
        contexts.append(context); offset += lengths[i]
        forwards.append(dict(output=outputs[i], ego=ego[i], poses=poses, targets=targets, yaws=yaws,
            prep=prep, evidence=evidence, plan=plan, mapped=mapped, ego_target=row[2][2]))
    upload = lambda arrays: torch.as_tensor(np.concatenate(arrays), device=device)
    actor = upload(actors).long()
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        linked = model.columns.project_sources(output)
        encoded = model.columns.encode(upload([fw['evidence'].features for fw in forwards]),
            upload([fw['evidence'].labels for fw in forwards]), actor,
            upload([fw['evidence'].classes for fw in forwards]), linked)
        logits = model.columns.decode(encoded, actor, torch.cat(contexts),
            upload([fw['plan'].base for fw in forwards]), upload([fw['plan'].fallback for fw in forwards]),
            upload([fw['plan'].legal for fw in forwards]), linked)
    actions = model.columns.probabilities(logits, actor).split([len(fw['evidence']) for fw in forwards])
    for fw, action in zip(forwards, actions): fw['actions'] = action
    return forwards


def train_arm(model, optimizer, provider, rows, mode, *, update, total_updates,
              object_weight=1., ego_weight=1., lovasz_weight=1., probe=False, optimized=True,
              horizon_group=2, projection_pool=None):
    if mode != 'gt' or not optimized: raise ValueError('opt-in GT fused execution only')
    if horizon_group not in (1, 2, 3): raise ValueError('bounded horizon group 1/2/3 required')
    model.train(); optimizer.zero_grad(set_to_none=True)
    factor = .01+.99*.5*(1+np.cos(np.pi*(update-1)/max(1, total_updates)))
    for group in optimizer.param_groups: group['lr'] = group['initial_lr']*factor
    begun = time.perf_counter()
    stages, terms, links = defaultdict(float), defaultdict(list), defaultdict(float)
    tick = time.perf_counter()
    forwards = forward_batch(model, provider, rows, projection_pool=projection_pool)
    stages['motion_ego_projection_head'] += time.perf_counter()-tick
    bridges, auxiliaries = [], []
    for row, fw in zip(rows, forwards):
        scene = GroupedScene(fw['prep'], provider.pcfg.grid, provider.device)
        bridge = RenderGradientBridge(fw); bridges.append(bridge)
        inputs = bridge.leaves
        for first in range(0, 6, horizon_group):
            horizons = list(range(first, min(6, first+horizon_group)))
            tick = time.perf_counter()
            probability = scene.render_group(horizons, inputs['poses'], inputs['targets'], inputs['yaws'], inputs['mapped'],
                fw['evidence'], fw['plan'], inputs['actions'])
            loss, ce, lovasz = grouped_loss(probability, np.asarray(row[1]['future_gt_occ'])[horizons], lovasz_weight=lovasz_weight)
            if not bool(torch.isfinite(loss).all()): raise RuntimeError('nonfinite grouped occupancy loss')
            if probe and 1 in horizons:
                tensors = [fw['actions']]; names = ['occupancy_to_actions']
                if fw['output']['residual_xy_m'].requires_grad:
                    tensors += [fw['output']['residual_xy_m'], fw['output']['yaw_delta_rad']]
                    names += ['occupancy_to_xy', 'occupancy_to_yaw']
                gradients = bridge.probe(loss[horizons.index(1)], tensors)
                for name, gradient in zip(names, gradients):
                    links[name] += 0. if gradient is None else float(gradient.detach().float().norm())
            stages['soft_render_full_ce_lovasz'] += time.perf_counter()-tick
            tick = time.perf_counter()
            if loss.requires_grad: (loss.sum()/(len(rows)*6)).backward()
            stages['occupancy_backward'] += time.perf_counter()-tick
            terms['CE'].extend(ce.detach().unbind()); terms['Lovasz'].extend(lovasz.detach().unbind())
            del probability, loss, ce, lovasz
        tick = time.perf_counter()
        geometry, existence = source_auxiliary(fw['output'], row[0], fw['prep'].state)
        ego_loss = ego_geometry_loss(fw['ego'][None], np.asarray(fw['ego_target'])[None])
        auxiliary = object_weight*(geometry+existence)+ego_weight*ego_loss
        if not bool(torch.isfinite(auxiliary)): raise RuntimeError('nonfinite direct geometry loss')
        auxiliaries.append(auxiliary/len(rows))
        terms['source_geometry'].append(geometry.detach()); terms['existence'].append(existence.detach())
        terms['ego_geometry'].append(ego_loss.detach())
        stages['shared_network_and_direct_geometry_backward'] += time.perf_counter()-tick
    # ONE shared WM/CCR backward for ALL windows, not repeated retained graphs.
    tick = time.perf_counter()
    roots, gradients = [torch.stack(auxiliaries).sum()], [None]
    for bridge in bridges:
        for key, leaf in bridge.leaves.items():
            if leaf.grad is not None: roots.append(bridge.original[key]); gradients.append(leaf.grad)
    torch.autograd.backward(roots, grad_tensors=gradients)
    stages['shared_network_and_direct_geometry_backward'] += time.perf_counter()-tick
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
    tick = time.perf_counter(); optimizer.step()
    if provider.device.type == 'cuda': torch.cuda.synchronize(provider.device)
    stages['optimizer_finish'] += time.perf_counter()-tick
    result = {key: float(torch.stack(values).mean()) for key, values in terms.items()}
    result.update(mode=mode, windows=len(rows), sources=sum(len(row[0]['features']) for row in rows),
        canonical_points=sum(len(fw['evidence']) for fw in forwards), seconds=time.perf_counter()-begun,
        stages_seconds=dict(stages), grad_norm=float(norm), gradient_links=dict(links),
        ego_occupancy_gradient_scale=.1+.9*min(1., (update-1)/max(1., total_updates/5)),
        training_backend='packed_network_horizon_group_'+str(horizon_group),
        peak_memory_mib=torch.cuda.max_memory_allocated(provider.device)/2**20 if provider.device.type == 'cuda' else 0.)
    optimizer.zero_grad(set_to_none=True)
    return result
