"""Differentiable semantic/depth ray projection, NOT Gaussian rasterization.

Inspired by GaussRender's training supervision; no external CUDA extension or
opacity head. Every class probability participates (including class 0).
"""
import math
import torch
from torch.nn import functional as F

PROJECTION = dict(views=['top', 'oblique'], side=64, step_m=.4,
    semantic='mean_ray_L1_of_18_class_distribution',
    depth='GT_hit_only_L1_over_fixed_view_depth_span', weight=1.,
    horizons='(4*zero_based_update+window_slot)%6', renderer='orthographic_volume_rays_v1')


class VolumeProjection:
    def __init__(self, origin, step, shape, *, side=64, step_m=.4):
        self.origin = tuple(map(float, origin)); self.step = tuple(map(float, step))
        self.shape = tuple(map(int, shape)); self.side, self.step_m = side, step_m
        self.cache = {}
        if side < 2 or step_m <= 0 or min(self.shape) < 1:
            raise ValueError('positive volume/ray geometry required')

    def rays(self, view, device):
        key = (view, str(device))
        if key in self.cache: return self.cache[key]
        lo = torch.tensor(self.origin, device=device, dtype=torch.float32)
        extent = lo.new_tensor(self.step)*lo.new_tensor(self.shape); hi = lo+extent
        direction = lo.new_tensor([0., 0., -1.] if view == 'top' else [1., 1., -1.2])
        if view not in ('top', 'oblique'): raise ValueError('fixed top/oblique views only')
        direction = F.normalize(direction, dim=0)
        u = F.normalize(torch.cross(direction, lo.new_tensor([0., 1., 0.]), dim=0), dim=0)
        v = torch.cross(u, direction, dim=0)
        corners = lo + lo.new_tensor([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)])*extent
        uc, vc, dc = corners@u, corners@v, corners@direction
        def centers(a):
            return a.min()+(torch.arange(self.side, device=device)+.5)*(a.max()-a.min())/self.side
        uu, vv = torch.meshgrid(centers(uc), centers(vc), indexing='ij')
        starts = uu.flatten()[:, None]*u + vv.flatten()[:, None]*v + (dc.min()-1)*direction
        parallel = direction.abs() < 1e-7
        safe = torch.where(parallel, torch.ones_like(direction), direction)
        t1, t2 = (lo-starts)/safe, (hi-starts)/safe
        lower, upper = torch.minimum(t1, t2), torch.maximum(t1, t2)
        lower[:, parallel] = -float('inf'); upper[:, parallel] = float('inf')
        inside = ((starts[:, parallel] >= lo[parallel]) & (starts[:, parallel] <= hi[parallel])).all(1)
        near, far = lower.max(1).values.clamp_min(0), upper.min(1).values
        valid = inside & (far > near)
        near, far = near[valid], far[valid]; starts = starts[valid]
        # One constant sample count per view bounds work; step never > step_m.
        samples = max(1, math.ceil(float((far-near).max())/self.step_m))
        f = (torch.arange(samples, device=device)+.5)/samples
        distances = near[:, None]+(far-near)[:, None]*f
        points = starts[:, None]+distances[..., None]*direction
        sampling = (((points-lo)/extent)*2-1)[..., [2, 1, 0]][None, :, :, None]
        scale = (dc.max()-dc.min()).clamp_min(1e-6)
        value = sampling, distances/scale, far/scale
        # Depths are camera-ray distances; the same rays are used for pred/GT.
        self.cache[key] = value
        return value

    def render(self, probability, view):
        if probability.shape != (18, *self.shape): raise ValueError('18 classes on the native future grid required')
        sampling, distance, far = self.rays(view, probability.device)
        p = F.grid_sample(probability[None].float(), sampling, align_corners=False,
            padding_mode='border')[0, :, :, :, 0].permute(1, 2, 0)
        # Occupancy directly from semantic probabilities: no bypass opacity.
        occupied = p[..., :17].sum(-1).clamp(0, 1)
        trans = torch.cat((torch.ones_like(occupied[:, :1]),
            torch.cumprod(1-occupied, 1)), 1)
        weight = trans[:, :-1]*occupied
        conditional = p[..., :17]/occupied[..., None].clamp_min(1e-8)
        semantic = torch.cat(((weight[..., None]*conditional).sum(1), trans[:, -1:]), 1)
        depth = (weight*distance).sum(1)+trans[:, -1]*far
        return semantic, depth, weight.sum(1)

    def loss(self, logits, labels):
        if logits.shape != (18, math.prod(self.shape)):
            raise ValueError('one complete native-grid horizon must be supervised')
        labels = torch.as_tensor(labels, device=logits.device).long().reshape(self.shape)
        if not bool(((labels >= 0) & (labels < 18)).all()):
            raise ValueError('projection requires explicit valid 0..17 GT; no silent remapping')
        probability = logits.float().softmax(0).reshape(18, *self.shape)
        with torch.no_grad():
            target = F.one_hot(labels, 18).permute(3, 0, 1, 2).float()
        semantics, depths = [], []
        for view in ('top', 'oblique'):
            pred_s, pred_d, _ = self.render(probability, view)
            with torch.no_grad(): gt_s, gt_d, hit = self.render(target, view)
            semantics.append((pred_s-gt_s).abs().sum(1).mean())
            valid = hit > 1e-6  # GT-only: missing predictions still incur loss.
            depths.append((pred_d[valid]-gt_d[valid]).abs().mean() if bool(valid.any()) else pred_d.sum()*0)
        semantic, depth = torch.stack(semantics).mean(), torch.stack(depths).mean()
        return semantic+depth, semantic, depth
