"""Training-only continuous rendering. Hard Strong/A1 remains an inference path.

Discrete association/support/owner choice is not differentiated. Continuous
ego transforms, source SE(2), CCR context/phase, splat weights and actions ARE.
The soft inverse warp replaces hard Strong majority voting during training;
outside the current source grid its fill is the freshly pose-conditioned Strong
background. This is a surrogate, not a claim of identical soft/hard outputs.
"""
import math
import numpy as np
import torch
from torch.nn import functional as F
from real_motion.canonical_causal_repair import grid_arrays, FEATURE_DIM
from real_motion.metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS
from real_motion.local_st_world_model_v18_se2 import YAW_ENABLED_CLASS_IDS
from .fast_autograd import compose_sparse, lovasz_sorted

FREE = 17
_CENTER_CACHE = {}


def transform(points, matrix):
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def source_targets(output, record, current_pose, source_z):
    device = output['residual_xy_m'].device
    xy = torch.as_tensor(record['anchors_xy_t0_m'], device=device).double()+output['residual_xy_m'].double()
    z = torch.as_tensor(source_z, device=device, dtype=torch.float64)[:, None, None].expand(-1, 6, 1)
    xyz = torch.cat((xy, z), -1)
    return transform(xyz, torch.as_tensor(current_pose, device=device, dtype=torch.float64))


def effective_yaw(output, classes):
    # Matches renderer_yaw_delta: pedestrians are never shape-rotated.
    enabled = torch.as_tensor(np.isin(classes, YAW_ENABLED_CLASS_IDS), device=output['yaw_delta_rad'].device)
    return output['yaw_delta_rad'].double()*enabled[:, None]


def move(points, center, target, yaw):
    relative = points-center
    c, s = yaw.cos(), yaw.sin()
    xy = torch.stack((c*relative[..., 0]-s*relative[..., 1],
                      s*relative[..., 0]+c*relative[..., 1]), -1)+target[..., :2]
    return torch.cat((xy, points[..., 2:3].expand(*xy.shape[:-1], 1)), -1)


def projected_points(evidence, state, targets, yaws, poses):
    """Vectorized actor gather, six transforms, float64 geometry until readout."""
    device = poses.device
    points = torch.as_tensor(evidence.world, device=device, dtype=torch.float64)
    actors = torch.as_tensor(evidence.actor, device=device, dtype=torch.long)
    dynamic = actors >= 0
    moved = points[:, None].expand(-1, 6, -1)
    if len(state['current']):
        centers = torch.as_tensor(np.array([x['centroid_world'] for x in state['current']]),
                                  device=device, dtype=torch.float64)
        ids = actors.clamp_min(0)
        candidate = move(points[:, None], centers[ids, None], targets[ids], yaws[ids])
        moved = torch.where(dynamic[:, None, None], candidate, moved)
    inverse = torch.linalg.inv(poses)
    mapped = torch.einsum('hij,nhj->nhi', inverse[:, :3, :3], moved)+inverse[None, :, :3, 3]
    return mapped


def continuous_context(evidence, plan, state, targets, yaws, poses, grid):
    mapped = projected_points(evidence, state, targets, yaws, poses)
    origin, step, _ = grid_arrays(grid)
    origin, step = mapped.new_tensor(origin), mapped.new_tensor(step)
    coordinate = (mapped-origin)/step
    # Hard cell addresses are fixed during backward; metric phase is live.
    phase = coordinate-coordinate.detach().floor()-.5
    current = mapped.new_tensor(state['current_pose'])
    relative = torch.linalg.inv(current)[None] @ poses
    rel_yaw = torch.atan2(relative[:, 1, 0], relative[:, 0, 0])
    pose_context = torch.stack((relative[:, 0, 3]/40, relative[:, 1, 3]/40, rel_yaw/math.pi), -1)
    time = mapped.new_tensor([.5, 1, 1.5, 2, 2.5, 3])/3
    actor = torch.as_tensor(evidence.actor, device=poses.device, dtype=torch.long)
    angles = mapped.new_zeros(len(evidence), 6)
    if len(yaws): angles = yaws[actor.clamp_min(0)]*(actor >= 0)[:, None]
    owner = mapped.new_tensor(plan.context[..., 7:8])
    context = torch.cat((mapped[..., :2]/40, time[None, :, None].expand(len(mapped), -1, -1),
        pose_context[None].expand(len(mapped), -1, -1), angles[..., None]/math.pi, owner), -1)
    dz = mapped.new_tensor(evidence.features[:, FEATURE_DIM+1])*step[2]
    z_factor = torch.linalg.inv(poses)[:, 2, :3] @ current[:3, 2]
    height = dz[:, None]*z_factor[None]/step[2]
    surface = torch.cat((phase, mapped[..., 2:3]/40, height[..., None],
                         (height-phase[..., 2])[..., None]), -1)
    surface = surface*(actor < 0)[:, None, None]
    return mapped, torch.cat((context, surface), -1).float()


def stencil(points, grid):
    origin, step, shape = grid_arrays(grid)
    q = (points-points.new_tensor(origin))/points.new_tensor(step)-.5
    lower = q.detach().floor().long()
    fraction = (q-lower).float()
    offsets = torch.tensor([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], device=points.device)
    at = lower[:, None]+offsets[None]
    weights = torch.where(offsets[None].bool(), fraction[:, None], 1-fraction[:, None]).prod(-1)
    dims = torch.as_tensor(shape, device=points.device)
    inside = ((at >= 0) & (at < dims)).all(-1)
    flat = ((at[..., 0]*int(shape[1])+at[..., 1])*int(shape[2])+at[..., 2]).clamp(0, int(np.prod(shape))-1)
    return flat, weights*inside, int(np.prod(shape))


def splat(points, values, grid, *, reduction='amax'):
    at, weights, volume = stencil(points, grid)
    v = weights*values[:, None]
    result = v.new_zeros(volume)
    if reduction == 'amax':
        return result.scatter_reduce(0, at.flatten(), v.flatten(), reduce='amax', include_self=True)
    if reduction == 'sum':
        return result.scatter_add(0, at.flatten(), v.flatten()).clamp(0, 1)
    raise ValueError('unknown splat reduction')


def sparse_splat(points, values, grid, *, reduction='amax'):
    """Identical field on touched voxels, without a full-grid zero per actor."""
    at, weights, _ = stencil(points, grid)
    ids, inverse = torch.unique(at.flatten(), sorted=True, return_inverse=True)
    v = (weights*values[:, None]).flatten()
    result = v.new_zeros(len(ids))
    if reduction == 'amax':
        return ids, result.scatter_reduce(0, inverse, v, reduce='amax', include_self=True)
    if reduction == 'sum':
        return ids, result.scatter_add(0, inverse, v).clamp(0, 1)
    raise ValueError('unknown splat reduction')


def sparse_layer(probability, ids, alpha, cls):
    # Clone ONLY the addressed values needed by alpha's backward. In-place
    # CopySlices has a valid autograd chain and avoids 18*V copies per actor.
    previous = probability[:, ids].clone()
    value = previous*(1-alpha[None])
    value[cls] = value[cls]+alpha
    probability[:, ids] = value
    return probability


class SoftScene:
    """Upload history/static volume and voxel centres ONCE per window."""
    def __init__(self, prep, grid, device):
        self.grid = grid
        origin, step, shape = grid_arrays(grid)
        self.shape = tuple(int(x) for x in shape)
        key = tuple(origin), tuple(step), self.shape, str(device)
        if key not in _CENTER_CACHE:
            # At most two grid/device geometries; only immutable coordinates,
            # never images, poses, descriptors, model features or labels.
            if len(_CENTER_CACHE) >= 2: _CENTER_CACHE.pop(next(iter(_CENTER_CACHE)))
            xyz = torch.stack(torch.meshgrid(*(torch.arange(n, device=device, dtype=torch.float64)
                                               for n in self.shape), indexing='ij'), -1)
            _CENTER_CACHE[key] = (xyz+.5)*xyz.new_tensor(step)+xyz.new_tensor(origin)
        self.centers = _CENTER_CACHE[key]
        sem = torch.as_tensor(np.array(prep.raw['history_occ'][-1], copy=True), device=device).long()
        dynamic = torch.zeros_like(sem, dtype=torch.bool)
        for cls in DYNAMIC_CLASS_IDS: dynamic |= sem == cls
        sem = torch.where(dynamic | (sem > FREE), FREE, sem)
        self.static = torch.zeros(18, sem.numel(), device=device).scatter_(
            0, sem.flatten()[None], torch.ones(1, sem.numel(), device=device)).reshape(1, 18, *self.shape)
        self.current_pose = self.centers.new_tensor(prep.state['current_pose'])
        self.source_points = [self.centers.new_tensor(p) for p in prep.state['source_world_points']]
        self.source_centers = [self.centers.new_tensor(c['centroid_world']) for c in prep.state['current']]
        self.classes = [int(c['class_id']) for c in prep.state['current']]
        self.backgrounds = [torch.as_tensor(np.array(b, copy=True), device=device).long().flatten()
                            for b in prep.state['column_backgrounds']]
        self.baselines = [torch.as_tensor(np.array(b, copy=True), device=device).long().flatten()
                          for b in prep.baseline]
        self.owners = [torch.as_tensor(np.array(b, copy=True), device=device).long().flatten()
                       for b in prep.owners]
        self.fallbacks = [torch.as_tensor(np.array(b, copy=True), device=device).long().flatten()
                          for b in prep.fallbacks]

    def background(self, pose, h):
        # PyTorch 5-D ordering: input [C,X,Y,Z], grid coordinates [Z,Y,X].
        source = transform(self.centers.reshape(-1, 3), torch.linalg.inv(self.current_pose) @ pose)
        origin, step, _ = grid_arrays(self.grid)
        q = (source-source.new_tensor(origin))/source.new_tensor(step)
        dims = source.new_tensor(self.shape)
        sampling = (q/dims*2-1)[:, [2, 1, 0]].float().reshape(1, *self.shape, 3)
        prob = F.grid_sample(self.static, sampling, mode='bilinear', padding_mode='zeros', align_corners=False)[0].reshape(18, -1)
        missing = (1-prob.sum(0)).clamp(0, 1)
        # No 92MB int64 one-hot + 46MB float fill allocation every horizon.
        return prob.scatter_add(0, self.backgrounds[h][None], missing[None])

    def render(self, h, pose, targets, yaws, mapped, evidence, plan, actions, *, edits=True):
        background = self.background(pose, h)
        operations, alphas = [], []
        actor_np = np.asarray(evidence.actor)
        legal = np.asarray(plan.legal)
        # Prepared legality changes with live learned geometry. Cache only
        # THIS window's row selections, never across forwards/checkpoints.
        if not hasattr(self, '_edit_groups'):
            groups = []
            for cls in (11, 13):
                groups.append(('static_remove', cls, np.flatnonzero((actor_np < 0) & (evidence.classes == cls))))
            for i, cls in enumerate(self.classes):
                groups.append(('dynamic_remove', i, np.flatnonzero(actor_np == i)))
            for actor in np.unique(actor_np):
                for cls in np.unique(evidence.classes[actor_np == actor]):
                    groups.append(('add', int(cls), np.flatnonzero((actor_np == actor) & (evidence.classes == cls))))
            self._edit_groups = {kind: [] for kind in ('static_remove', 'dynamic_remove', 'add')}
            for kind, cls, ids in groups:
                action = 0 if kind == 'add' else 1
                rows = [torch.as_tensor(ids[legal[ids, slot, action]], device=pose.device) for slot in range(6)]
                self._edit_groups[kind].append((cls, rows))
        def append(kind, flat, alpha, cls, destination=None):
            operations.append((kind, flat, cls, destination)); alphas.append(alpha)
        if edits:
            for cls, rows in self._edit_groups['static_remove']:
                at = rows[h]
                if not len(at): continue
                flat, alpha = sparse_splat(mapped[at, h], actions[at, h, 1], self.grid)
                alpha = alpha*((self.owners[h][flat] < 0) & (self.baselines[h][flat] == cls))
                append('remove', flat, alpha, cls, torch.full_like(flat, FREE))
        inverse = torch.linalg.inv(pose)
        for i, cls in enumerate(self.classes):
            points = transform(move(self.source_points[i], self.source_centers[i], targets[i, h], yaws[i, h]), inverse)
            flat, alpha = sparse_splat(points, points.new_ones(len(points)).float(), self.grid, reduction='sum')
            append('overlay', flat, alpha, cls)
        if edits:
            for i, rows in self._edit_groups['dynamic_remove']:
                at = rows[h]
                if not len(at): continue
                cls = self.classes[i]
                flat, alpha = sparse_splat(mapped[at, h], actions[at, h, 1], self.grid)
                alpha = alpha*((self.owners[h][flat] == i) & (self.baselines[h][flat] == cls))
                append('remove', flat, alpha, cls, self.fallbacks[h][flat])
            for cls, rows in self._edit_groups['add']:
                at = rows[h]
                if not len(at): continue
                flat, alpha = sparse_splat(mapped[at, h], actions[at, h, 0], self.grid)
                alpha = alpha*(self.baselines[h][flat] == FREE)
                append('overlay', flat, alpha, cls)
        return compose_sparse(background, operations, alphas)

    def render_reference(self, h, pose, targets, yaws, mapped, evidence, plan, actions, *, edits=True):
        """Original autograd chain retained for numerical/gradient audits."""
        prob = self.background(pose, h)
        actor_np = np.asarray(evidence.actor)
        legal = torch.as_tensor(plan.legal[:, h], device=pose.device)
        # Removal is owner-specific. A foreground removal restores lower layers,
        # rather than erasing them. Original free-only ADD legality is retained.
        if edits:
            for cls in (11, 13):
                ids = np.flatnonzero((actor_np < 0) & (evidence.classes == cls))
                ids = ids[np.asarray(plan.legal)[ids, h, 1]]
                if len(ids):
                    rows = torch.as_tensor(ids, device=pose.device)
                    flat, removal = sparse_splat(mapped[rows, h], actions[rows, h, 1], self.grid)
                    removal = removal*((self.owners[h][flat] < 0) & (self.baselines[h][flat] == cls))
                    amount = prob[cls, flat].clone()*removal
                    prob[cls, flat] = prob[cls, flat].clone()-amount
                    prob[FREE, flat] = prob[FREE, flat].clone()+amount
        inverse = torch.linalg.inv(pose)
        for i, cls in enumerate(self.classes):
            points = transform(move(self.source_points[i], self.source_centers[i], targets[i, h], yaws[i, h]), inverse)
            flat, alpha = sparse_splat(points, points.new_ones(len(points)).float(), self.grid, reduction='sum')
            prob = sparse_layer(prob, flat, alpha, cls)
        if edits:
            for i, cls in enumerate(self.classes):
                ids = np.flatnonzero(actor_np == i)
                ids = ids[np.asarray(plan.legal)[ids, h, 1]]
                if len(ids):
                    rows = torch.as_tensor(ids, device=pose.device)
                    flat, removal = sparse_splat(mapped[rows, h], actions[rows, h, 1], self.grid)
                    owned = (self.owners[h][flat] == i) & (self.baselines[h][flat] == cls)
                    local = prob[:, flat].clone()
                    amount = local[cls].clone()*removal*owned
                    local[cls] = local[cls]-amount
                    # Restore the SAME visible lower-owner/background semantic
                    # as the hard A1 compositor. Do not teach REMOVE to expose
                    # the soft current-frame FREE hole instead of that fallback.
                    local = local.scatter_add(0, self.fallbacks[h][flat][None], amount[None])
                    prob[:, flat] = local
        if edits:
            # One scatter per class/actor, not Python loops over every voxel.
            for actor in np.unique(actor_np):
                for cls in np.unique(evidence.classes[actor_np == actor]):
                    ids = np.flatnonzero((actor_np == actor) & (evidence.classes == cls))
                    ids = ids[np.asarray(plan.legal)[ids, h, 0]]
                    if not len(ids): continue
                    rows = torch.as_tensor(ids, device=pose.device)
                    flat, alpha = sparse_splat(mapped[rows, h], actions[rows, h, 0], self.grid)
                    alpha = alpha*(self.baselines[h][flat] == FREE)
                    prob = sparse_layer(prob, flat, alpha, int(cls))
        return prob


def lovasz_softmax_reference(prob, target, ignore=255, class_ids=None):
    """Full-grid semantic Lovasz, present classes 0..16. No voxel sampling."""
    target = target.flatten().long()
    valid = (target >= 0) & (target <= FREE) & (target != ignore)
    labels, prediction = target[valid], prob[:, valid]
    if class_ids is None:
        # Generic/test use; server computes present classes from CPU LABELS
        # once per horizon instead of 17 GPU->host .any() synchronizations.
        class_ids = [int(c) for c in labels.unique().cpu().tolist() if c < FREE]
    if not len(class_ids) or not len(labels): return prob.sum()*0
    ids = torch.as_tensor(class_ids, device=prob.device, dtype=torch.long)
    foreground = (labels[None] == ids[:, None]).float()
    errors = (foreground-prediction.index_select(0, ids)).abs()
    errors, order = torch.sort(errors, dim=1, descending=True, stable=True)
    sorted_fg = foreground.gather(1, order)
    total = sorted_fg.sum(1, keepdim=True)
    intersection = total-sorted_fg.cumsum(1)
    union = total+(1-sorted_fg).cumsum(1)
    grad = 1-intersection/union.clamp_min(1)
    if grad.shape[1] > 1: grad = torch.cat((grad[:, :1], grad[:, 1:]-grad[:, :-1]), 1)
    return (errors*grad).sum(1).mean()


def lovasz_softmax(prob, target, ignore=255, class_ids=None):
    """Identical full-grid stable-sort objective, fused first-order backward."""
    target = target.flatten().long()
    positions = torch.nonzero((target >= 0) & (target <= FREE) & (target != ignore), as_tuple=True)[0]
    labels = target[positions]
    if class_ids is None:
        class_ids = [int(c) for c in labels.unique().cpu().tolist() if c < FREE]
    if not len(class_ids) or not len(labels): return prob.sum()*0
    classes = torch.as_tensor(class_ids, device=prob.device, dtype=torch.long)
    return lovasz_sorted(prob, labels, positions, classes)


def occupancy_loss(prob, target, *, lovasz_weight=1., epsilon=1e-4, class_ids=None, reference=False):
    labels = target.flatten().long()
    valid = (labels >= 0) & (labels <= FREE)
    # Smooth ALL classes rather than clamp gradients to zero at missed voxels.
    probabilities = (prob.float()+epsilon)/(1+18*epsilon)
    ce = (-probabilities[:, valid].gather(0, labels[valid][None]).log().mean()
          if bool(valid.any()) else prob.sum()*0)
    lovasz = (lovasz_softmax_reference if reference else lovasz_softmax)(probabilities, target, class_ids=class_ids)
    return ce+lovasz_weight*lovasz, ce, lovasz
