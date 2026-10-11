"""Bounded horizon batches, with disjoint voxel addresses and original layers."""
import numpy as np
import torch
from torch.nn import functional as F
from torch.autograd.function import once_differentiable
from ..soft_geometry import SoftScene, grid_arrays, move, stencil, occupancy_loss
from ..fast_autograd import compose_sparse


def grouped_splat(points, values, slots, grid, reduction='amax'):
    at, weight, volume = stencil(points, grid)
    at = at + slots[:, None]*volume
    ids, inverse = torch.unique(at.flatten(), sorted=True, return_inverse=True)
    value = (weight*values[:, None]).flatten()
    result = value.new_zeros(len(ids))
    if reduction == 'amax':
        result = result.scatter_reduce(0, inverse, value, reduce='amax', include_self=True)
    elif reduction == 'sum':
        result = result.scatter_add(0, inverse, value).clamp(0, 1)
    else:
        raise ValueError('unknown splat reduction')
    return ids, result


class GroupedScene(SoftScene):
    def render_group(self, horizons, poses, targets, yaws, mapped, evidence, plan, actions, *, edits=True):
        if not horizons or len(horizons) > 3 or len(set(horizons)) != len(horizons):
            raise ValueError('one to three DISTINCT future horizons required')
        if any(h not in range(6) for h in horizons): raise ValueError('future horizon out of range')
        device = poses.device
        hs = torch.as_tensor(horizons, device=device)
        volume = int(np.prod(self.shape)); count = len(horizons)
        selected = poses[hs]
        relative = torch.linalg.inv(self.current_pose)[None] @ selected
        xyz = torch.einsum('gij,nj->gni', relative[:, :3, :3], self.centers.reshape(-1, 3)) + relative[:, None, :3, 3]
        origin, step, _ = grid_arrays(self.grid)
        q = (xyz-xyz.new_tensor(origin))/xyz.new_tensor(step)
        sampling = (q/q.new_tensor(self.shape)*2-1)[..., [2, 1, 0]].float().reshape(count, *self.shape, 3)
        background = F.grid_sample(self.static.expand(count, -1, -1, -1, -1), sampling,
            mode='bilinear', padding_mode='zeros', align_corners=False).reshape(count, 18, volume)
        missing = (1-background.sum(1)).clamp(0, 1)
        background = background.scatter_add(1, torch.stack([self.backgrounds[h] for h in horizons])[:, None], missing[:, None])
        background = background.permute(1, 0, 2).reshape(18, count*volume)
        owners = torch.stack([self.owners[h] for h in horizons]).flatten()
        baseline = torch.stack([self.baselines[h] for h in horizons]).flatten()
        fallback = torch.stack([self.fallbacks[h] for h in horizons]).flatten()
        operations, alphas = [], []
        actor = np.asarray(evidence.actor)
        legal = np.asarray(plan.legal)
        def edit(ids, action):
            # Ragged legal rows are concatenated, not masked into max ties.
            rows, slots, times = [], [], []
            for slot, h in enumerate(horizons):
                at = ids[legal[ids, h, action]]
                if len(at):
                    rows.append(torch.as_tensor(at, device=device))
                    slots.append(torch.full((len(at),), slot, device=device, dtype=torch.long))
                    times.append(torch.full((len(at),), h, device=device, dtype=torch.long))
            if not rows: return None
            rows, slots, times = torch.cat(rows), torch.cat(slots), torch.cat(times)
            return grouped_splat(mapped[rows, times], actions[rows, times, action], slots, self.grid)
        def append(kind, ids, alpha, cls, destination=None):
            operations.append((kind, ids, cls, destination)); alphas.append(alpha)
        if edits:
            for cls in (11, 13):
                result = edit(np.flatnonzero((actor < 0) & (evidence.classes == cls)), 1)
                if result is not None:
                    ids, alpha = result
                    alpha = alpha*((owners[ids] < 0) & (baseline[ids] == cls))
                    append('remove', ids, alpha, cls, torch.full_like(ids, 17))
        inverse = torch.linalg.inv(selected)
        for i, cls in enumerate(self.classes):
            points = move(self.source_points[i][None], self.source_centers[i], targets[i, hs, None], yaws[i, hs, None])
            points = torch.einsum('gij,gnj->gni', inverse[:, :3, :3], points)+inverse[:, None, :3, 3]
            slots = torch.arange(count, device=device)[:, None].expand(-1, len(self.source_points[i])).flatten()
            points = points.reshape(-1, 3)
            ids, alpha = grouped_splat(points, points.new_ones(len(points)).float(), slots, self.grid, 'sum')
            append('overlay', ids, alpha, cls)
        if edits:
            for i, cls in enumerate(self.classes):
                result = edit(np.flatnonzero(actor == i), 1)
                if result is not None:
                    ids, alpha = result
                    alpha = alpha*((owners[ids] == i) & (baseline[ids] == cls))
                    append('remove', ids, alpha, cls, fallback[ids])
            for actor_id in np.unique(actor):
                for cls in np.unique(evidence.classes[actor == actor_id]):
                    result = edit(np.flatnonzero((actor == actor_id) & (evidence.classes == cls)), 0)
                    if result is not None:
                        ids, alpha = result
                        append('overlay', ids, alpha*(baseline[ids] == 17), int(cls))
        return compose_sparse(background, operations, alphas).reshape(18, count, volume).permute(1, 0, 2)


class _GroupedLovasz(torch.autograd.Function):
    """Original full stable sort; EACH horizon has its own class mean."""
    @staticmethod
    def forward(ctx, probability, labels, classes, active):
        selected = probability.gather(1, classes[..., None].expand(-1, -1, labels.shape[1]))
        foreground = (labels[:, None] == classes[..., None]).float()
        delta = selected-foreground
        errors, order = delta.abs().sort(dim=2, descending=True, stable=True)
        sorted_fg = foreground.gather(2, order)
        total = sorted_fg.sum(2, keepdim=True)
        intersection = total-sorted_fg.cumsum(2)
        union = total+(1-sorted_fg).cumsum(2)
        weights = 1-intersection/union.clamp_min(1)
        if weights.shape[2] > 1:
            weights = torch.cat((weights[..., :1], weights[..., 1:]-weights[..., :-1]), 2)
        scale = active.float()/active.sum(1, keepdim=True).clamp_min(1)
        coefficients = weights*delta.sign().gather(2, order)*scale[..., None]
        ctx.shape = probability.shape
        ctx.save_for_backward(classes, active, order, coefficients)
        return ((errors*weights).sum(2)*scale).sum(1)

    @staticmethod
    @once_differentiable
    def backward(ctx, upstream):
        classes, active, order, coefficients = ctx.saved_tensors
        gradient = coefficients.new_zeros(ctx.shape)
        rows, columns = active.nonzero(as_tuple=True)
        # Only real classes are assigned; dummy padded classes never overwrite.
        local = coefficients[rows, columns]*upstream[rows, None]
        gradient[rows, classes[rows, columns]] = torch.zeros_like(local).scatter(1, order[rows, columns], local)
        return gradient, None, None, None


def grouped_loss(probability, cpu_labels, *, lovasz_weight=1., epsilon=1e-4):
    """Return per-horizon terms; never pool class presence or ignore counts."""
    labels_np = np.asarray(cpu_labels).reshape(len(probability), -1)
    if labels_np.shape[1] != probability.shape[2]: raise ValueError('target volume differs')
    labels = torch.as_tensor(np.array(labels_np, copy=True), device=probability.device).long()
    classes = [[int(c) for c in np.unique(row) if 0 <= c < 17] for row in labels_np]
    if not bool(((labels_np >= 0) & (labels_np <= 17)).all()):
        terms = [occupancy_loss(p, t, lovasz_weight=lovasz_weight, epsilon=epsilon, class_ids=c)
                 for p, t, c in zip(probability, labels, classes)]
        return tuple(torch.stack([term[i] for term in terms]) for i in range(3))
    normalized = (probability.float()+epsilon)/(1+18*epsilon)
    ce = -normalized.gather(1, labels[:, None]).log().mean((1, 2))
    width = max(map(len, classes), default=0)
    if width:
        ids = torch.as_tensor([c+[0]*(width-len(c)) for c in classes], device=probability.device)
        active = torch.arange(width, device=probability.device)[None] < torch.as_tensor(list(map(len, classes)), device=probability.device)[:, None]
        lovasz = _GroupedLovasz.apply(normalized, labels, ids, active)
    else:
        lovasz = probability.sum((1, 2))*0
    return ce+lovasz_weight*lovasz, ce, lovasz
