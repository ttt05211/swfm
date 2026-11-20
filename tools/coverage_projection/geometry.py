"""Original transport probabilities plus independent causal coverage evidence."""
import numpy as np
import torch
from torch.nn import functional as F
from tools.shared_history_experiments.refine.geometry import TransportPrior as OriginalPrior, inverse_grid
from tools.joint_occ_experiments.soft_geometry import move
from tools.joint_occ_experiments.batching.geometry import grouped_splat
from tools.joint_occ_experiments.fast_autograd import compose_sparse


class CoveredPrior(OriginalPrior):
    def __init__(self, raw, state, grid, device):
        super().__init__(raw, state, grid, device)
        sem = np.asarray(raw['history_occ'][-1])
        known = np.asarray(raw['history_observed'][-1]) & (sem >= 0) & (sem < 18)
        self.valid = torch.as_tensor(known.copy(), device=device)[None, None].float()

    def render(self, poses, targets, yaws, horizons=range(6)):
        hs = torch.as_tensor(tuple(horizons), device=poses.device)
        count = len(hs); selected = poses[hs]
        grids = inverse_grid(self.centers[None].expand(count, -1, -1, -1, -1),
            torch.linalg.inv(self.pose)[None] @ selected, self.grid)
        p = F.grid_sample(self.static.expand(count, -1, -1, -1, -1), grids,
            align_corners=False, padding_mode='zeros').reshape(count, 18, self.volume)
        coverage = F.grid_sample(self.valid.expand(count, -1, -1, -1, -1), grids,
            align_corners=False, padding_mode='zeros').reshape(count, self.volume)
        missing = (1-p.sum(1)).clamp(0, 1)
        p = p.scatter_add(1, self.backgrounds[hs, None], missing[:, None])
        background = p.permute(1, 0, 2).reshape(18, -1)
        operations, alpha = [], []; inverse = torch.linalg.inv(selected)
        for actor, source in enumerate(self.state['current']):
            points = poses.new_tensor(self.state['source_world_points'][actor])
            world = move(points[None], poses.new_tensor(source['centroid_world']),
                targets[actor, hs, None], yaws[actor, hs, None])
            mapped = torch.einsum('hij,hnj->hni', inverse[:, :3, :3], world)+inverse[:, None, :3, 3]
            slots = torch.arange(count, device=poses.device)[:, None].expand(-1, len(points)).flatten()
            ids, weight = grouped_splat(mapped.reshape(-1, 3), poses.new_ones(count*len(points)).float(),
                slots, self.grid, 'sum')
            operations.append(('overlay', ids, int(source['class_id']), None)); alpha.append(weight)
        probability = compose_sparse(background, operations, alpha).reshape(18, count, self.volume).permute(1, 0, 2)
        # Reuse EXACT overlay addresses/weights: a dynamic source moved outside
        # the static t0 footprint is still supported evidence. No future GT mask.
        with torch.no_grad():
            support = coverage.flatten()
            for (_, ids, _, _), weight in zip(operations, alpha):
                support[ids] = support[ids]*(1-weight.detach()) + weight.detach()
        return probability, support.reshape(count, 1, self.volume).clamp(0, 1)
