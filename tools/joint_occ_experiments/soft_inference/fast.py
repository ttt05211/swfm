"""Inference-only soft renderer. Never imported by the training renderer.

Same float64 geometry, float32 weights, eight-corner interpolation and ordered
source overlays. No model/pose/plan survives one forward. No GT label access.
"""
import numpy as np
import torch
from torch.nn import functional as F
from ..soft_geometry import SoftScene, grid_arrays, move


def compose_inference(background, operations):
    """Fresh scratch volume, same arithmetic/order; no clone/backward history."""
    if torch.is_grad_enabled(): raise RuntimeError('soft inference compositor requires no_grad')
    for kind, ids, cls, destination, alpha in operations:
        local = background[:, ids]
        if kind == 'overlay':
            value = local*(1-alpha[None])
            value[cls] = value[cls]+alpha
        elif kind == 'remove':
            amount = local[cls].clone()*alpha
            value = local
            value[cls] = value[cls]-amount
            value = value.scatter_add(0, destination[None], amount[None])
        else:
            raise ValueError('unknown soft composition operation')
        background[:, ids] = value
    return background


class InferenceScene(SoftScene):
    def __init__(self, fw, grid, device, *, remove):
        if torch.is_grad_enabled(): raise RuntimeError('soft inference renderer requires no_grad')
        super().__init__(fw['prep'], grid, device)
        self.fw, self.remove = fw, remove
        self.volume = int(np.prod(self.shape))
        origin, step, shape = grid_arrays(grid)
        self.origin = self.centers.new_tensor(origin)
        self.step = self.centers.new_tensor(step)
        self.dims = torch.as_tensor(shape, device=device)
        self.offsets = torch.tensor([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], device=device)
        self.relative_origin = torch.linalg.inv(self.current_pose)
        self.backgrounds = torch.stack(self.backgrounds)
        self.baselines = torch.stack(self.baselines)
        self.owners = torch.stack(self.owners)
        self.fallbacks = torch.stack(self.fallbacks)
        evidence, plan = fw['evidence'], fw['plan']
        actor, classes, legal = np.asarray(evidence.actor), np.asarray(evidence.classes), np.asarray(plan.legal)
        # This forward's legality is live. Select/upload rows ONCE, preserving
        # actor and class order; never cache them across outputs or windows.
        groups = []
        if remove:
            for cls in (11, 13):
                groups.append(('static_remove', cls, np.flatnonzero((actor < 0) & (classes == cls))))
            for i in range(len(self.classes)):
                groups.append(('dynamic_remove', i, np.flatnonzero(actor == i)))
        for actor_id in np.unique(actor):
            for cls in np.unique(classes[actor == actor_id]):
                groups.append(('add', int(cls), np.flatnonzero((actor == actor_id) & (classes == cls))))
        self.groups = {k: [] for k in ('static_remove', 'dynamic_remove', 'add')}
        for kind, cls, ids in groups:
            action = 0 if kind == 'add' else 1
            self.groups[kind].append((cls, [torch.as_tensor(ids[legal[ids, h, action]], device=device)
                                           for h in range(6)]))

    def splat(self, points, values, slots, reduction='amax'):
        # Identical to stencil/grouped_splat; immutable constants are reused.
        q = (points-self.origin)/self.step-.5
        lower = q.floor().long()
        fraction = (q-lower).float()
        at = lower[:, None]+self.offsets[None]
        weights = torch.where(self.offsets[None].bool(), fraction[:, None], 1-fraction[:, None]).prod(-1)
        inside = ((at >= 0) & (at < self.dims)).all(-1)
        flat = ((at[..., 0]*self.shape[1]+at[..., 1])*self.shape[2]+at[..., 2]).clamp(0, self.volume-1)
        flat = flat+slots[:, None]*self.volume
        ids, inverse = torch.unique(flat.flatten(), sorted=True, return_inverse=True)
        value = (weights*inside*values[:, None]).flatten()
        result = value.new_zeros(len(ids))
        if reduction == 'amax':
            result.scatter_reduce_(0, inverse, value, reduce='amax', include_self=True)
        elif reduction == 'sum':
            result.scatter_add_(0, inverse, value); result.clamp_(0, 1)
        else:
            raise ValueError('unknown splat reduction')
        return ids, result

    def render_group(self, horizons, *, edits=True):
        if not horizons or len(horizons) > 3 or len(set(horizons)) != len(horizons):
            raise ValueError('one to three distinct future horizons required')
        if any(h not in range(6) for h in horizons): raise ValueError('future horizon out of range')
        fw, count, device = self.fw, len(horizons), self.current_pose.device
        hs = torch.as_tensor(horizons, device=device)
        selected = fw['poses'][hs]
        relative = self.relative_origin[None] @ selected
        xyz = torch.einsum('gij,nj->gni', relative[:, :3, :3], self.centers.reshape(-1, 3))+relative[:, None, :3, 3]
        q = (xyz-self.origin)/self.step
        sampling = (q/q.new_tensor(self.shape)*2-1)[..., [2, 1, 0]].float().reshape(count, *self.shape, 3)
        background = F.grid_sample(self.static.expand(count, -1, -1, -1, -1), sampling,
            mode='bilinear', padding_mode='zeros', align_corners=False).reshape(count, 18, self.volume)
        missing = (1-background.sum(1)).clamp(0, 1)
        background.scatter_add_(1, self.backgrounds[hs, None], missing[:, None])
        background = background.permute(1, 0, 2).reshape(18, count*self.volume)
        owners, baseline, fallback = (a[hs].flatten() for a in (self.owners, self.baselines, self.fallbacks))
        operations = []
        def edit(rows, action):
            active = [(slot, h, rows[h]) for slot, h in enumerate(horizons) if len(rows[h])]
            if not active: return None
            at = torch.cat([r for _, _, r in active])
            slots = torch.cat([torch.full_like(r, slot) for slot, _, r in active])
            times = torch.cat([torch.full_like(r, h) for _, h, r in active])
            return self.splat(fw['mapped'][at, times], fw['actions'][at, times, action], slots)
        def append(kind, ids, alpha, cls, destination=None):
            operations.append((kind, ids, cls, destination, alpha))
        if edits and self.remove:
            for cls, rows in self.groups['static_remove']:
                result = edit(rows, 1)
                if result is not None:
                    ids, alpha = result
                    append('remove', ids, alpha*((owners[ids] < 0) & (baseline[ids] == cls)), cls, torch.full_like(ids, 17))
        inverse = torch.linalg.inv(selected)
        for i, cls in enumerate(self.classes):
            points = move(self.source_points[i][None], self.source_centers[i], fw['targets'][i, hs, None], fw['yaws'][i, hs, None])
            points = torch.einsum('gij,gnj->gni', inverse[:, :3, :3], points)+inverse[:, None, :3, 3]
            slots = torch.arange(count, device=device)[:, None].expand(-1, len(self.source_points[i])).flatten()
            points = points.reshape(-1, 3)
            ids, alpha = self.splat(points, points.new_ones(len(points)).float(), slots, 'sum')
            append('overlay', ids, alpha, cls)
        if edits and self.remove:
            for i, rows in self.groups['dynamic_remove']:
                result = edit(rows, 1)
                if result is not None:
                    ids, alpha = result; cls = self.classes[i]
                    append('remove', ids, alpha*((owners[ids] == i) & (baseline[ids] == cls)), cls, fallback[ids])
        if edits:
            for cls, rows in self.groups['add']:
                result = edit(rows, 0)
                if result is not None:
                    ids, alpha = result
                    append('overlay', ids, alpha*(baseline[ids] == 17), cls)
        return compose_inference(background, operations).reshape(18, count, self.volume).permute(1, 0, 2)


@torch.no_grad()
def predict(provider, fw, variant, horizons, *, group_size=3):
    if variant not in ('transport', 'add_only', 'add_remove'): raise ValueError('unknown soft variant')
    if group_size not in (1, 2, 3): raise ValueError('bounded future group required')
    horizons = list(horizons)
    scene = InferenceScene(fw, provider.pcfg.grid, provider.device, remove=variant == 'add_remove')
    # Keep SIX small byte grids on GPU; transfer once instead of six fences.
    outputs = []
    for start in range(0, len(horizons), group_size):
        probability = scene.render_group(horizons[start:start+group_size], edits=variant != 'transport')
        outputs.append(probability.argmax(1).reshape(-1, *scene.shape).byte())
        del probability
    if not outputs: return []
    dense = torch.cat(outputs).cpu().numpy()
    return [dense[i] for i in range(len(horizons))]
